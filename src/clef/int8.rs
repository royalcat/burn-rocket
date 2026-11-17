//! Host int8 helpers for the NPU expert/attention path: the activation quantizer the
//! NPU path uses and a host emulation of the group-wise W8A8 matmul (`Int8Emu`).
//!
//! The NPU path (`rocket_npu::matmul::I8Context::matmul_gw`) computes, per output
//! element, `sum_g (a_scale[m,g] * b_scale[n,g]) * (A_g . B_g)` where the dot is an exact
//! int32 over one K-group and the sum runs over the groups in order in f32. The activation
//! is quantized per row per `group` with `scale = max(max|x| / 127, 1e-12)` and
//! `q = clamp(round_half_away(x / scale), -127, 127)` (a mirror of
//! `burn_rocket::host::quantize_row_groups`, which is crate-private there). The emulation
//! below does exactly that on the CPU, so the integer part is bit-identical and only the
//! f32 order can differ from the device when the driver splits a group over several
//! K-tiles (`kt < group`).

use super::st::f16_to_f32;
use rayon::prelude::*;

/// One row of symmetric group-wise int8 quantization (see the module docs).
pub fn quantize_row_groups(xrow: &[f32], qrow: &mut [i8], srow: &mut [f32], group: usize) {
    debug_assert_eq!(xrow.len(), qrow.len());
    debug_assert_eq!(srow.len(), xrow.len() / group);
    for (g, s) in srow.iter_mut().enumerate() {
        let seg = &xrow[g * group..(g + 1) * group];
        let amax = seg.iter().fold(0f32, |a, &v| a.max(v.abs()));
        let scale = (amax / 127.0).max(1e-12);
        *s = scale;
        let inv = 1.0 / scale;
        for (qi, &v) in qrow[g * group..(g + 1) * group].iter_mut().zip(seg) {
            *qi = (v * inv).round().clamp(-127.0, 127.0) as i8;
        }
    }
}

/// Quantize `m` rows of width `k` into `q [m, k]` / `s [m, k/group]` (all rows in
/// `x`). `parallel` fans the rows out over the current rayon pool.
pub fn quantize_rows(
    x: &[f32],
    k: usize,
    group: usize,
    q: &mut [i8],
    s: &mut [f32],
    parallel: bool,
) {
    let ng = k / group;
    let one = |((xr, qr), sr): ((&[f32], &mut [i8]), &mut [f32])| {
        quantize_row_groups(xr, qr, sr, group);
    };
    if parallel {
        x.par_chunks(k)
            .zip(q.par_chunks_mut(k))
            .zip(s.par_chunks_mut(ng))
            .for_each(one);
    } else {
        x.chunks(k)
            .zip(q.chunks_mut(k))
            .zip(s.chunks_mut(ng))
            .for_each(one);
    }
}

/// `dst[i] = f16(src[i])`.
pub fn f16_scales_to_f32(src: &[u16], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), dst.len());
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = f16_to_f32(s);
    }
}

/// Row counts of the NPU matmul shape classes. The driver caches one scratch per exact
/// `(M,K,N,group)`, so a call's rows are padded up to a class (and chunked at the largest)
/// to keep the number of distinct shapes small (<= 14 per weight shape).
pub const M_CLASSES: [usize; 14] = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512];
pub const CHUNK_ROWS: usize = 512;

/// Smallest class that holds `m` rows (`m <= CHUNK_ROWS`).
pub fn m_class(m: usize) -> usize {
    assert!((1..=CHUNK_ROWS).contains(&m));
    M_CLASSES.iter().copied().find(|&c| c >= m).unwrap()
}

/// `(first row, rows, padded rows)` of every chunk of an `m`-row call.
pub fn m_chunks(m: usize) -> Vec<(usize, usize, usize)> {
    let mut v = Vec::new();
    let mut start = 0;
    while start < m {
        let rows = (m - start).min(CHUNK_ROWS);
        v.push((start, rows, m_class(rows)));
        start += rows;
    }
    v
}

/// Exact int32 dot of two int8 slices (at most 2^17 elements before i32 could overflow).
#[inline]
pub fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| i32::from(x) * i32::from(y))
        .sum()
}

/// Host emulation of `matmul_gw`: `out[m, n] = sum_g (a_s[m,g] * w_s[n,g]) * dot_g`, groups
/// ascending, f32 accumulate, scale applied as `(a_s * w_s) * (partial as f32)` like the
/// driver's de-tile. `aq [m,k]`, `wq [n,k]` int8, scales `[m,k/group]` / `[n,k/group]`.
#[allow(clippy::too_many_arguments)]
pub fn emu_matmul_gw(
    aq: &[i8],
    a_s: &[f32],
    m: usize,
    k: usize,
    wq: &[i8],
    w_s: &[f32],
    n: usize,
    group: usize,
    out: &mut [f32],
    parallel: bool,
) {
    assert_eq!(aq.len(), m * k);
    assert_eq!(wq.len(), n * k);
    assert_eq!(out.len(), m * n);
    let ng = k / group;
    assert_eq!(a_s.len(), m * ng);
    assert_eq!(w_s.len(), n * ng);
    if m == 0 || n == 0 {
        return;
    }
    // Row blocks keep the activation block hot while the weight rows stream by.
    const MB: usize = 16;
    let block = |bi: usize, o: &mut [f32]| {
        let r0 = bi * MB;
        let rows = o.len() / n;
        for j in 0..n {
            let wr = &wq[j * k..(j + 1) * k];
            let ws = &w_s[j * ng..(j + 1) * ng];
            for r in 0..rows {
                let ar = &aq[(r0 + r) * k..(r0 + r + 1) * k];
                let asr = &a_s[(r0 + r) * ng..(r0 + r + 1) * ng];
                let mut acc = 0f32;
                for g in 0..ng {
                    let p = dot_i8(
                        &ar[g * group..(g + 1) * group],
                        &wr[g * group..(g + 1) * group],
                    );
                    acc += (asr[g] * ws[g]) * (p as f32);
                }
                o[r * n + j] = acc;
            }
        }
    };
    if parallel {
        out.par_chunks_mut(MB * n)
            .enumerate()
            .for_each(|(bi, o)| block(bi, o));
    } else {
        out.chunks_mut(MB * n)
            .enumerate()
            .for_each(|(bi, o)| block(bi, o));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }

    #[test]
    fn row_chunks_cover_every_row_with_few_shape_classes() {
        for m in [1usize, 3, 4, 5, 63, 64, 65, 512, 513, 1100, 4096] {
            let ch = m_chunks(m);
            assert_eq!(ch.iter().map(|c| c.1).sum::<usize>(), m);
            let mut next = 0;
            for &(st, rows, pad) in &ch {
                assert_eq!(st, next);
                assert!(rows <= pad && pad % 4 == 0 && pad <= CHUNK_ROWS);
                next += rows;
            }
        }
        // every M in 1..=4096 lands in the 14 classes
        let classes: std::collections::BTreeSet<usize> =
            (1..=4096).flat_map(m_chunks).map(|c| c.2).collect();
        assert!(classes.len() <= M_CLASSES.len(), "{classes:?}");
    }

    #[test]
    fn quantizer_matches_the_documented_formula() {
        let mut seed = 3;
        let x: Vec<f32> = (0..256).map(|_| lcg(&mut seed) * 5.0).collect();
        let (mut q, mut s) = (vec![0i8; 256], vec![0f32; 2]);
        quantize_row_groups(&x, &mut q, &mut s, 128);
        for g in 0..2 {
            let seg = &x[g * 128..(g + 1) * 128];
            let amax = seg.iter().fold(0f32, |a, &v| a.max(v.abs()));
            assert_eq!(s[g], (amax / 127.0).max(1e-12));
            assert!(q[g * 128..(g + 1) * 128].iter().any(|&v| v.abs() == 127));
            for (i, &v) in seg.iter().enumerate() {
                let want = (v / s[g]).round().clamp(-127.0, 127.0) as i8;
                // `v * (1/s)` and `v / s` agree except at exact .5 boundaries
                assert!((i16::from(q[g * 128 + i]) - i16::from(want)).abs() <= 1);
            }
        }
        // an all-zero group stays finite with zero codes
        let (mut zq, mut zs) = (vec![1i8; 128], vec![0f32; 1]);
        quantize_row_groups(&[0.0; 128], &mut zq, &mut zs, 128);
        assert!(zq.iter().all(|&v| v == 0) && zs[0] > 0.0);
    }

    #[test]
    fn the_emulated_matmul_has_an_exact_integer_part() {
        // unit scales: the result must equal the naive int64 dot per group, summed in
        // f32 (group sums are < 2^24 so every term is exact)
        let (m, k, n, g) = (7usize, 256usize, 5usize, 128usize);
        let ng = k / g;
        let mut seed = 11;
        let aq: Vec<i8> = (0..m * k)
            .map(|_| (lcg(&mut seed) * 127.0).round() as i8)
            .collect();
        let wq: Vec<i8> = (0..n * k)
            .map(|_| (lcg(&mut seed) * 127.0).round() as i8)
            .collect();
        let (a_s, w_s) = (vec![1f32; m * ng], vec![1f32; n * ng]);
        let mut got = vec![0f32; m * n];
        emu_matmul_gw(&aq, &a_s, m, k, &wq, &w_s, n, g, &mut got, false);
        for r in 0..m {
            for j in 0..n {
                let want: i64 = (0..k)
                    .map(|c| i64::from(aq[r * k + c]) * i64::from(wq[j * k + c]))
                    .sum();
                assert_eq!(got[r * n + j] as i64, want, "({r},{j})");
            }
        }
        // the parallel arm is bit-identical to the serial one
        let mut par = vec![0f32; m * n];
        emu_matmul_gw(&aq, &a_s, m, k, &wq, &w_s, n, g, &mut par, true);
        assert_eq!(got, par);
    }

    #[test]
    fn the_emulated_matmul_matches_an_f64_reference_with_real_scales() {
        let (m, k, n, g) = (9usize, 384usize, 6usize, 128usize);
        let ng = k / g;
        let mut seed = 5;
        let a: Vec<f32> = (0..m * k).map(|_| lcg(&mut seed)).collect();
        let (mut aq, mut a_s) = (vec![0i8; m * k], vec![0f32; m * ng]);
        quantize_rows(&a, k, g, &mut aq, &mut a_s, false);
        let wq: Vec<i8> = (0..n * k)
            .map(|_| (lcg(&mut seed) * 100.0).round() as i8)
            .collect();
        let w_s: Vec<f32> = (0..n * ng)
            .map(|_| 0.001 + lcg(&mut seed).abs() * 0.01)
            .collect();
        let mut got = vec![0f32; m * n];
        emu_matmul_gw(&aq, &a_s, m, k, &wq, &w_s, n, g, &mut got, false);
        for r in 0..m {
            for j in 0..n {
                let want: f64 = (0..k)
                    .map(|c| {
                        f64::from(aq[r * k + c])
                            * f64::from(a_s[r * ng + c / g])
                            * f64::from(wq[j * k + c])
                            * f64::from(w_s[j * ng + c / g])
                    })
                    .sum();
                let d = (f64::from(got[r * n + j]) - want).abs();
                assert!(d <= 1e-5 * want.abs().max(1.0), "({r},{j}): {d}");
            }
        }
    }
}
