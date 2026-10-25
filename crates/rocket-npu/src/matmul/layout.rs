//! The NPU cube layouts and their host scatter/gather.
//!
//! fp16 cubes, 1-based in the original code:
//! - feature/output: `(ch-1)/8 * H*8 + 8*(h-1) + (ch-1)%8`
//! - weight: `(c-1)/32 * 32*16 + (k-1)/16 * 16*C + (c-1)%32 + (k-1)%16*32`
//!
//! The loops below move whole 8- and 32-element runs with `copy_from_slice`, which is
//! the same blocked form the C code uses (it was the dominant pack cost).

use half::f16;

/// Feature/output cube element index (1-based row/channel, as in the C helpers).
#[inline]
pub fn feat_idx(h: usize, ch: usize, row: usize) -> usize {
    ((ch - 1) / 8) * h * 8 + 8 * (row - 1) + (ch - 1) % 8
}

/// Weight cube element index (1-based).
#[inline]
pub fn wt_idx(c: usize, k: usize, ch: usize) -> usize {
    ((ch - 1) / 32) * 32 * 16 + ((k - 1) / 16) * 16 * c + ((ch - 1) % 32) + ((k - 1) % 16) * 32
}

/// Scatter one row-major `A[m0.., k0..]` tile into a feature cube slot.
pub fn feat_scatter_tile(
    slot: &mut [f16],
    a: &[f16],
    astride: usize,
    m0: usize,
    k0: usize,
    mtile: usize,
    ktile: usize,
) {
    debug_assert_eq!(slot.len(), mtile.div_ceil(4) * 4 * ktile.div_ceil(32) * 32);
    let h = mtile;
    for row in 0..mtile {
        let arow = &a[(m0 + row) * astride + k0..][..ktile];
        let mut col = 0;
        while col + 8 <= ktile {
            let g = col / 8;
            let dst = &mut slot[g * h * 8 + 8 * row..][..8];
            dst.copy_from_slice(&arow[col..col + 8]);
            col += 8;
        }
        while col < ktile {
            slot[feat_idx(h, col + 1, row + 1)] = arow[col];
            col += 1;
        }
    }
}

/// Scatter one weight row `brow[k0..k0+ktile]` for output channel `kk` (0-based).
pub fn wt_scatter_row(slot: &mut [f16], brow: &[f16], kk: usize, ktile: usize) {
    let c = ktile;
    let kbase = (kk / 16) * 16 * c + (kk % 16) * 32;
    let mut col = 0;
    while col + 32 <= ktile {
        let dst = &mut slot[(col / 32) * 32 * 16 + kbase..][..32];
        dst.copy_from_slice(&brow[col..col + 32]);
        col += 32;
    }
    while col < ktile {
        slot[wt_idx(c, kk + 1, col + 1)] = brow[col];
        col += 1;
    }
}

/// Scatter a contiguous `B[n0.., k0..]` weight tile into a weight cube slot.
pub fn wt_scatter_tile(
    slot: &mut [f16],
    b: &[f16],
    bstride: usize,
    n0: usize,
    k0: usize,
    ntile: usize,
    ktile: usize,
) {
    debug_assert_eq!(
        slot.len(),
        ntile.div_ceil(16) * 16 * ktile.div_ceil(32) * 32
    );
    for kk in 0..ntile {
        let brow = &b[(n0 + kk) * bstride + k0..][..ktile];
        wt_scatter_row(slot, brow, kk, ktile);
    }
}

/// Accumulate one output-cube tile into the row-major fp32 accumulator.
pub fn detile_accum(
    acc: &mut [f32],
    n: usize,
    slot: &[f16],
    m0: usize,
    n0: usize,
    mtile: usize,
    ntile: usize,
) {
    let h = mtile;
    for row in 0..mtile {
        let d = &mut acc[(m0 + row) * n + n0..][..ntile];
        let mut col = 0;
        while col + 8 <= ntile {
            let g = col / 8;
            let src = &slot[g * h * 8 + 8 * row..][..8];
            for (dv, sv) in d[col..col + 8].iter_mut().zip(src) {
                *dv += sv.to_f32();
            }
            col += 8;
        }
        while col < ntile {
            d[col] += slot[feat_idx(h, col + 1, row + 1)].to_f32();
            col += 1;
        }
    }
}

/// Accumulate one **fp32** output-cube tile into the f64 accumulator.
///
/// The fp32 accumulator cube is C2=4 (16 B per 4-channel group), so the de-tile is
/// the `out_idx_i16` index the int32/bf16/tf32 paths read; the fp16-output
/// [`detile_accum`] above is the C2=8 sibling.
pub fn detile_accum_f32(
    acc: &mut [f64],
    n: usize,
    slot: &[f32],
    m0: usize,
    n0: usize,
    mtile: usize,
    ntile: usize,
) {
    let h = mtile;
    for row in 0..mtile {
        let d = &mut acc[(m0 + row) * n + n0..][..ntile];
        let mut col = 0;
        while col + 4 <= ntile {
            let g = col / 4;
            let src = &slot[g * h * 4 + 4 * row..][..4];
            for (j, v) in src.iter().enumerate() {
                d[col + j] += *v as f64;
            }
            col += 4;
        }
        while col < ntile {
            d[col] += slot[out_idx_i16(h, col + 1, row + 1)] as f64;
            col += 1;
        }
    }
}

/// Copy one output-cube tile into the caller's row-major fp16 output.
pub fn detile_store(
    c: &mut [f16],
    n: usize,
    slot: &[f16],
    m0: usize,
    n0: usize,
    mtile: usize,
    ntile: usize,
) {
    let h = mtile;
    for row in 0..mtile {
        let d = &mut c[(m0 + row) * n + n0..][..ntile];
        let mut col = 0;
        while col + 8 <= ntile {
            let g = col / 8;
            let src = &slot[g * h * 8 + 8 * row..][..8];
            d[col..col + 8].copy_from_slice(src);
            col += 8;
        }
        while col < ntile {
            d[col] = slot[feat_idx(h, col + 1, row + 1)];
            col += 1;
        }
    }
}

/// Narrow the fp32 accumulator into the caller's fp16 output.
pub fn narrow_f32_to_f16(c: &mut [f16], acc: &[f32]) {
    debug_assert_eq!(c.len(), acc.len());
    for (d, s) in c.iter_mut().zip(acc) {
        *d = f16::from_f32(*s);
    }
}

/// Narrow the fp32 accumulator into N planes of stride `ldc` (transposed output).
pub fn narrow_f32_to_planes(ct: &mut [f16], ldc: usize, acc: &[f32], m: usize, n: usize) {
    for row in 0..m {
        for col in 0..n {
            ct[col * ldc + row] = f16::from_f32(acc[row * n + col]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn value(i: usize) -> f16 {
        f16::from_f32(i as f32 * 0.25)
    }

    #[test]
    fn feature_round_trip() {
        let (mt, kt): (usize, usize) = (12, 64);
        let a: Vec<f16> = (0..mt * kt).map(value).collect();
        let mut slot = vec![f16::ZERO; mt.div_ceil(4) * 4 * kt];
        feat_scatter_tile(&mut slot, &a, kt, 0, 0, mt, kt);
        let mut acc = vec![0.0f32; mt * kt];
        detile_accum(&mut acc, kt, &slot, 0, 0, mt, kt);
        for (i, v) in acc.iter().enumerate() {
            assert_eq!(*v, value(i).to_f32());
        }
        let mut out = vec![f16::ZERO; mt * kt];
        detile_store(&mut out, kt, &slot, 0, 0, mt, kt);
        assert_eq!(out, a);
    }

    #[test]
    fn ragged_tiles_use_the_live_height() {
        let (mt, kt) = (5, 32);
        let a: Vec<f16> = (0..mt * kt).map(value).collect();
        let mut slot = vec![f16::ZERO; 8 * kt];
        feat_scatter_tile(&mut slot, &a, kt, 0, 0, mt, kt);
        // Row 4 (h=5) lives at the cube height of the live tile, not at 8.
        assert_eq!(slot[feat_idx(5, 1, 5)], a[4 * kt]);
        let mut out = vec![f16::ZERO; mt * kt];
        detile_store(&mut out, kt, &slot, 0, 0, mt, kt);
        assert_eq!(out, a);
    }

    #[test]
    fn weight_round_trip() {
        let (nt, kt) = (20, 96);
        let b: Vec<f16> = (0..nt * kt).map(value).collect();
        let mut slot = vec![f16::ZERO; nt.div_ceil(16) * 16 * kt];
        wt_scatter_tile(&mut slot, &b, kt, 0, 0, nt, kt);
        for k in 0..nt {
            for c in 0..kt {
                assert_eq!(slot[wt_idx(kt, k + 1, c + 1)], b[k * kt + c], "k={k} c={c}");
            }
        }
    }

    #[test]
    fn weight_slice_offsets() {
        let (n, k) = (32, 64);
        let b: Vec<f16> = (0..n * k).map(value).collect();
        let mut slot = vec![f16::ZERO; 16 * 32];
        wt_scatter_tile(&mut slot, &b, k, 16, 32, 16, 32);
        for kk in 0..16 {
            for cc in 0..32 {
                assert_eq!(slot[wt_idx(32, kk + 1, cc + 1)], b[(16 + kk) * k + 32 + cc]);
            }
        }
    }

    #[test]
    fn f32_cube_detiles_at_the_c2_4_index() {
        // A ragged (mtile % 4, ntile % 4) fp32 cube placed at a non-zero origin: the
        // f32 output cube is C2=4, so the element at (row, col) sits at
        // out_idx_i16(mtile, col + 1, row + 1), not at the fp16 C2=8 index.
        let (mt, nt): (usize, usize) = (5, 6);
        let (alloc_m, alloc_n) = (mt.div_ceil(4) * 4, nt.div_ceil(4) * 4);
        let slot: Vec<f32> = (0..alloc_m * alloc_n).map(|i| i as f32 * 0.5).collect();
        let (n, m0, n0) = (20, 4, 4);
        let mut acc = vec![0.0f64; (m0 + mt + 1) * n];
        // A pre-existing partial (a K-tile already accumulated) must be added to.
        acc[(m0 + 2) * n + n0 + 1] = 7.0;
        detile_accum_f32(&mut acc, n, &slot, m0, n0, mt, nt);
        for row in 0..mt {
            for col in 0..nt {
                let want = slot[out_idx_i16(mt, col + 1, row + 1)] as f64;
                let want = if (row, col) == (2, 1) {
                    want + 7.0
                } else {
                    want
                };
                assert_eq!(acc[(m0 + row) * n + n0 + col], want, "row={row} col={col}");
            }
        }
        // Outside the tile nothing moved.
        assert_eq!(acc[0], 0.0);
        assert_eq!(acc[(m0 + mt) * n + n0], 0.0);
        assert_eq!(acc[m0 * n + n0 + nt], 0.0);
    }
}

// ── int8 cubes: feature C2=16, weight (N/32,K/32,32,32), int32 output C2=4 ──────

#[inline]
pub fn feat_idx_i8(h: usize, ch: usize, row: usize) -> usize {
    ((ch - 1) / 16) * h * 16 + 16 * (row - 1) + (ch - 1) % 16
}

#[inline]
pub fn wt_idx_i8(c: usize, k: usize, ch: usize) -> usize {
    ((ch - 1) / 32) * 32 * 32 + ((k - 1) / 32) * 32 * c + ((ch - 1) % 32) + ((k - 1) % 32) * 32
}

#[inline]
pub fn out_idx_i8(h: usize, ch: usize, row: usize) -> usize {
    ((ch - 1) / 4) * h * 4 + 4 * (row - 1) + (ch - 1) % 4
}

pub fn feat_scatter_i8(
    slot: &mut [i8],
    a: &[i8],
    astride: usize,
    m0: usize,
    k0: usize,
    mtile: usize,
    ktile: usize,
) {
    let h = mtile;
    for row in 0..mtile {
        let arow = &a[(m0 + row) * astride + k0..][..ktile];
        let mut col = 0;
        while col + 16 <= ktile {
            let g = col / 16;
            slot[g * h * 16 + 16 * row..][..16].copy_from_slice(&arow[col..col + 16]);
            col += 16;
        }
        while col < ktile {
            slot[feat_idx_i8(h, col + 1, row + 1)] = arow[col];
            col += 1;
        }
    }
}

pub fn wt_scatter_row_i8(slot: &mut [i8], brow: &[i8], kk: usize, ktile: usize) {
    let c = ktile;
    let kbase = (kk / 32) * 32 * c + (kk % 32) * 32;
    let mut col = 0;
    while col + 32 <= ktile {
        let dst = &mut slot[(col / 32) * 32 * 32 + kbase..][..32];
        dst.copy_from_slice(&brow[col..col + 32]);
        col += 32;
    }
    while col < ktile {
        slot[wt_idx_i8(c, kk + 1, col + 1)] = brow[col];
        col += 1;
    }
}

pub fn wt_scatter_tile_i8(
    slot: &mut [i8],
    b: &[i8],
    bstride: usize,
    n0: usize,
    k0: usize,
    ntile: usize,
    ktile: usize,
) {
    for kk in 0..ntile {
        let brow = &b[(n0 + kk) * bstride + k0..][..ktile];
        wt_scatter_row_i8(slot, brow, kk, ktile);
    }
}

/// Accumulate one int32 output cube into the i64 accumulator (the dense int8 path).
///
/// The int32 cube is C2=4 — the same 16 B per 4-channel group as the fp32 cube — so
/// the de-tile index is [`out_idx_i8`]. The dense `rocket_matmul_int8` sums its
/// per-K-tile int32 partials on the host in int64, which is integer-exact; this is the
/// gather half of that.
pub fn detile_accum_i32(
    acc: &mut [i64],
    n: usize,
    slot: &[i32],
    m0: usize,
    n0: usize,
    mtile: usize,
    ntile: usize,
) {
    let h = mtile;
    for row in 0..mtile {
        let d = &mut acc[(m0 + row) * n + n0..][..ntile];
        let mut col = 0;
        while col + 4 <= ntile {
            let g = col / 4;
            let src = &slot[g * h * 4 + 4 * row..][..4];
            for (dv, sv) in d[col..col + 4].iter_mut().zip(src) {
                *dv += *sv as i64;
            }
            col += 4;
        }
        while col < ntile {
            d[col] += slot[out_idx_i8(h, col + 1, row + 1)] as i64;
            col += 1;
        }
    }
}

/// Accumulate one int32 output cube into `acc`, adding `scale * value`.
#[allow(clippy::too_many_arguments)]
pub fn detile_accum_i8_scaled(
    acc: &mut [f32],
    n: usize,
    slot: &[i32],
    m0: usize,
    n0: usize,
    mtile: usize,
    ntile: usize,
    scale: impl Fn(usize, usize) -> f32,
) {
    let h = mtile;
    for row in 0..mtile {
        let d = &mut acc[(m0 + row) * n + n0..][..ntile];
        let mut col = 0;
        while col + 4 <= ntile {
            let g = col / 4;
            let src = &slot[g * h * 4 + 4 * row..][..4];
            for (j, v) in src.iter().enumerate() {
                d[col + j] += scale(m0 + row, n0 + col + j) * (*v as f32);
            }
            col += 4;
        }
        while col < ntile {
            d[col] += scale(m0 + row, n0 + col) * slot[out_idx_i8(h, col + 1, row + 1)] as f32;
            col += 1;
        }
    }
}

// ── int4 cubes: feature C2=32 nibbles, weight (N/64,K/32,64,32) nibbles,
//    int16 output C2=8 ──────────────────────────────────────────────────────────
//
// The int4 operands are NIBBLE-packed, so the indices below count nibbles, not bytes:
// byte `n >> 1`, low nibble when `n` is even and high when odd. Every group base
// carries a factor of 32 (feature/weight K-group, output granule 8 vs 2/4/8) and is
// therefore even, which is what lets the blocked scatters pack 16 whole bytes
// (`i4_pack16`) instead of touching nibbles one at a time.

/// Feature-cube nibble index (1-based row/channel, as in the C helpers).
#[inline]
pub fn feat_idx_i4(h: usize, ch: usize, row: usize) -> usize {
    ((ch - 1) / 32) * h * 32 + 32 * (row - 1) + (ch - 1) % 32
}

/// Weight-cube nibble index (1-based output channel `k` / K `c`).
#[inline]
pub fn wt_idx_i4(c: usize, k: usize, ch: usize) -> usize {
    let n_kgrp = c.div_ceil(32);
    let n_grp = (k - 1) / 64;
    let n_within = (k - 1) % 64;
    let k_grp = (ch - 1) / 32;
    let k_within = (ch - 1) % 32;
    n_grp * n_kgrp * 64 * 32 + k_grp * 64 * 32 + n_within * 32 + k_within
}

/// Output-cube int16 element index (1-based), C2=8.
#[inline]
pub fn out_idx_i4(h: usize, ch: usize, row: usize) -> usize {
    ((ch - 1) / 8) * h * 8 + 8 * (row - 1) + (ch - 1) % 8
}

/// Set nibble `idx` of a packed buffer (even = low half, odd = high half).
#[inline]
fn put_nib(buf: &mut [u8], idx: usize, v: i8) {
    let nib = (v & 0xF) as u8;
    if idx & 1 != 0 {
        buf[idx >> 1] = (buf[idx >> 1] & 0x0F) | (nib << 4);
    } else {
        buf[idx >> 1] = (buf[idx >> 1] & 0xF0) | nib;
    }
}

/// Pack 32 int4 codes (one per `i8`, two per byte) into 16 bytes.
#[inline]
fn pack16_i4(d: &mut [u8], s: &[i8]) {
    for j in 0..16 {
        d[j] = ((s[2 * j] & 0xF) as u8) | (((s[2 * j + 1] & 0xF) as u8) << 4);
    }
}

/// Scatter one `A[m0.., k0..]` tile into the `(K/32, Mt, 32)`-nibble int4 feature cube.
pub fn feat_scatter_i4(
    slot: &mut [u8],
    a: &[i8],
    astride: usize,
    m0: usize,
    k0: usize,
    mtile: usize,
    ktile: usize,
) {
    let h = mtile;
    let ng = ktile / 32;
    for row in 0..mtile {
        let arow = &a[(m0 + row) * astride + k0..][..ktile];
        for g in 0..ng {
            // feat_idx_i4(H, 32g+1, h) = g*H*32 + 32*(h-1), and that base is even.
            let base = g * h * 32 + 32 * row;
            pack16_i4(&mut slot[base >> 1..], &arow[g * 32..]);
        }
        for (col, &v) in arow.iter().enumerate().skip(ng * 32) {
            put_nib(slot, feat_idx_i4(h, col + 1, row + 1), v);
        }
    }
}

/// Scatter one weight row `brow[k0..k0+ktile]` for output channel `kk` (0-based).
pub fn wt_scatter_row_i4(slot: &mut [u8], brow: &[i8], kk: usize, ktile: usize) {
    let n_kgrp = ktile.div_ceil(32);
    let ng = ktile / 32;
    let nbase = (kk / 64) * n_kgrp * 64 * 32 + (kk % 64) * 32;
    for g in 0..ng {
        let base = nbase + g * 64 * 32;
        pack16_i4(&mut slot[base >> 1..], &brow[g * 32..]);
    }
    for (col, &v) in brow.iter().enumerate().skip(ng * 32) {
        put_nib(slot, wt_idx_i4(ktile, kk + 1, col + 1), v);
    }
}

/// Scatter a contiguous `B[n0.., k0..]` weight tile into an int4 weight cube slot.
pub fn wt_scatter_tile_i4(
    slot: &mut [u8],
    b: &[i8],
    bstride: usize,
    n0: usize,
    k0: usize,
    ntile: usize,
    ktile: usize,
) {
    for kk in 0..ntile {
        let brow = &b[(n0 + kk) * bstride + k0..][..ktile];
        wt_scatter_row_i4(slot, brow, kk, ktile);
    }
}

/// Accumulate one int16 output cube into the i64 accumulator.
///
/// int4 x int4 accumulates in the conv's CACC and is written back as **int16**
/// (`out_precision = 1`), so each K-tile's partial saturates above 32767 — the caller
/// must keep `Kt` small enough (or the data low enough) for a bit-exact comparison.
/// The int16 cube is C2=8, so it gathers at [`out_idx_i4`].
pub fn detile_accum_i16(
    acc: &mut [i64],
    n: usize,
    slot: &[i16],
    m0: usize,
    n0: usize,
    mtile: usize,
    ntile: usize,
) {
    let h = mtile;
    for row in 0..mtile {
        let d = &mut acc[(m0 + row) * n + n0..][..ntile];
        let mut col = 0;
        while col + 8 <= ntile {
            let g = col / 8;
            let src = &slot[g * h * 8 + 8 * row..][..8];
            for (dv, sv) in d[col..col + 8].iter_mut().zip(src) {
                *dv += *sv as i64;
            }
            col += 8;
        }
        while col < ntile {
            d[col] += slot[out_idx_i4(h, col + 1, row + 1)] as i64;
            col += 1;
        }
    }
}

/// Accumulate one int16 output cube into `acc`, adding `scale * value`.
///
/// The int4 group-wise readback: every K-tile is one quant group, so its int16 partial
/// takes that group's `a_scale[m,g] * b_scale[n,g]` as it is folded in. The cube is
/// C2=8 ([`out_idx_i4`]), the same one [`detile_accum_i16`] walks.
#[allow(clippy::too_many_arguments)]
pub fn detile_accum_i16_scaled(
    acc: &mut [f32],
    n: usize,
    slot: &[i16],
    m0: usize,
    n0: usize,
    mtile: usize,
    ntile: usize,
    scale: impl Fn(usize, usize) -> f32,
) {
    let h = mtile;
    for row in 0..mtile {
        let d = &mut acc[(m0 + row) * n + n0..][..ntile];
        let mut col = 0;
        while col + 8 <= ntile {
            let g = col / 8;
            let src = &slot[g * h * 8 + 8 * row..][..8];
            for (j, v) in src.iter().enumerate() {
                d[col + j] += scale(m0 + row, n0 + col + j) * (*v as f32);
            }
            col += 8;
        }
        while col < ntile {
            d[col] += scale(m0 + row, n0 + col) * slot[out_idx_i4(h, col + 1, row + 1)] as f32;
            col += 1;
        }
    }
}

// ── bf16 cubes: feature/weight == int16, fp32 output C2=4 ─────────────────────

#[inline]
pub fn feat_idx_i16(h: usize, ch: usize, row: usize) -> usize {
    ((ch - 1) / 8) * h * 8 + 8 * (row - 1) + (ch - 1) % 8
}

#[inline]
pub fn wt_idx_i16(c: usize, k: usize, ch: usize) -> usize {
    ((ch - 1) / 32) * 32 * 16 + ((k - 1) / 16) * 16 * c + ((ch - 1) % 32) + ((k - 1) % 16) * 32
}

#[inline]
pub fn out_idx_i16(h: usize, ch: usize, row: usize) -> usize {
    ((ch - 1) / 4) * h * 4 + 4 * (row - 1) + (ch - 1) % 4
}

/// f32 -> bf16 by truncation of the low 16 bits (the hardware's bf16 MAC input).
#[inline]
pub fn f32_to_bf16_bits(f: f32) -> u16 {
    (f.to_bits() >> 16) as u16
}

pub fn feat_scatter_bf16(
    slot: &mut [u16],
    a: &[f32],
    astride: usize,
    m0: usize,
    k0: usize,
    mtile: usize,
    ktile: usize,
) {
    let h = mtile;
    for row in 0..mtile {
        let arow = &a[(m0 + row) * astride + k0..][..ktile];
        let mut col = 0;
        while col + 8 <= ktile {
            let g = col / 8;
            let dst = &mut slot[g * h * 8 + 8 * row..][..8];
            for (d, s) in dst.iter_mut().zip(&arow[col..col + 8]) {
                *d = f32_to_bf16_bits(*s);
            }
            col += 8;
        }
        while col < ktile {
            slot[feat_idx_i16(h, col + 1, row + 1)] = f32_to_bf16_bits(arow[col]);
            col += 1;
        }
    }
}

pub fn wt_scatter_row_bf16(slot: &mut [u16], brow: &[f32], kk: usize, ktile: usize) {
    let c = ktile;
    let kbase = (kk / 16) * 16 * c + (kk % 16) * 32;
    let mut col = 0;
    while col + 32 <= ktile {
        let dst = &mut slot[(col / 32) * 32 * 16 + kbase..][..32];
        for (d, s) in dst.iter_mut().zip(&brow[col..col + 32]) {
            *d = f32_to_bf16_bits(*s);
        }
        col += 32;
    }
    while col < ktile {
        slot[wt_idx_i16(c, kk + 1, col + 1)] = f32_to_bf16_bits(brow[col]);
        col += 1;
    }
}

pub fn wt_scatter_tile_bf16(
    slot: &mut [u16],
    b: &[f32],
    bstride: usize,
    n0: usize,
    k0: usize,
    ntile: usize,
    ktile: usize,
) {
    for kk in 0..ntile {
        let brow = &b[(n0 + kk) * bstride + k0..][..ktile];
        wt_scatter_row_bf16(slot, brow, kk, ktile);
    }
}

/// Accumulate one fp32 output cube into the f64 accumulator (the bf16 stream's name
/// for the shared C2=4 fp32 de-tile, [`detile_accum_f32`]).
pub fn detile_accum_bf16(
    acc: &mut [f64],
    n: usize,
    slot: &[f32],
    m0: usize,
    n0: usize,
    mtile: usize,
    ntile: usize,
) {
    detile_accum_f32(acc, n, slot, m0, n0, mtile, ntile);
}

#[cfg(test)]
mod int4_tests {
    use super::*;

    /// Read nibble `idx` back as a sign-extended int4 code.
    fn get_nib(buf: &[u8], idx: usize) -> i8 {
        let b = buf[idx >> 1];
        let nib = if idx & 1 != 0 {
            (b >> 4) & 0xF
        } else {
            b & 0xF
        };
        ((nib as i8) << 4) >> 4
    }

    fn code(i: usize) -> i8 {
        (i % 16) as i8 - 8
    }

    #[test]
    fn int4_feature_round_trip() {
        let (mt, kt): (usize, usize) = (12, 64);
        let a: Vec<i8> = (0..mt * kt).map(code).collect();
        let mut slot = vec![0u8; mt.div_ceil(4) * 4 * kt / 2];
        feat_scatter_i4(&mut slot, &a, kt, 0, 0, mt, kt);
        for row in 0..mt {
            for col in 0..kt {
                assert_eq!(
                    get_nib(&slot, feat_idx_i4(mt, col + 1, row + 1)),
                    a[row * kt + col],
                    "row={row} col={col}"
                );
            }
        }
        // A tile with a non-zero origin reads the right window of the source.
        let mut slice = vec![0u8; 8 * 32 / 2];
        feat_scatter_i4(&mut slice, &a, kt, 4, 32, 4, 32);
        for row in 0..4 {
            for col in 0..32 {
                assert_eq!(
                    get_nib(&slice, feat_idx_i4(4, col + 1, row + 1)),
                    a[(4 + row) * kt + 32 + col]
                );
            }
        }
    }

    #[test]
    fn int4_feature_scatter_ignores_the_ktile_tail() {
        // The planner always hands a K%32==0 Ktile, but the C loop keeps a tail branch;
        // it must write the tail nibbles and nothing else (a ragged 40-wide tile).
        let (mt, kt): (usize, usize) = (5, 40);
        let a: Vec<i8> = (0..mt * kt).map(code).collect();
        let mut slot = vec![0u8; 8 * 64 / 2];
        feat_scatter_i4(&mut slot, &a, kt, 0, 0, mt, kt);
        for row in 0..mt {
            for col in 0..kt {
                assert_eq!(
                    get_nib(&slot, feat_idx_i4(mt, col + 1, row + 1)),
                    a[row * kt + col],
                    "row={row} col={col}"
                );
            }
        }
    }

    #[test]
    fn int4_weight_round_trip() {
        let (nt, kt): (usize, usize) = (80, 96);
        let b: Vec<i8> = (0..nt * kt).map(code).collect();
        let mut slot = vec![0u8; nt.div_ceil(64) * 64 * kt / 2];
        wt_scatter_tile_i4(&mut slot, &b, kt, 0, 0, nt, kt);
        for k in 0..nt {
            for c in 0..kt {
                assert_eq!(
                    get_nib(&slot, wt_idx_i4(kt, k + 1, c + 1)),
                    b[k * kt + c],
                    "k={k} c={c}"
                );
            }
        }
    }

    #[test]
    fn int4_weight_slice_offsets() {
        let (n, k) = (128, 64);
        let b: Vec<i8> = (0..n * k).map(code).collect();
        let mut slot = vec![0u8; 64 * 64 / 2];
        wt_scatter_tile_i4(&mut slot, &b, k, 64, 32, 64, 32);
        for kk in 0..64 {
            for cc in 0..32 {
                assert_eq!(
                    get_nib(&slot, wt_idx_i4(32, kk + 1, cc + 1)),
                    b[(64 + kk) * k + 32 + cc]
                );
            }
        }
    }

    #[test]
    fn int4_int16_cube_detiles_in_eight_channel_groups() {
        // Ragged (mtile, ntile) int16 cube at a non-zero origin: the gather walks
        // 8-channel groups (C2=8) at out_idx_i4, and a pre-existing partial in the i64
        // accumulator (an earlier K-tile) must be added to, not replaced.
        let (mt, nt): (usize, usize) = (5, 6);
        let (alloc_m, alloc_n) = (mt.div_ceil(4) * 4, nt.div_ceil(8) * 8);
        let slot: Vec<i16> = (0..alloc_m * alloc_n).map(|i| i as i16 * 7 - 40).collect();
        let (n, m0, n0) = (20, 4, 4);
        let mut acc = vec![0i64; (m0 + mt + 1) * n];
        acc[(m0 + 2) * n + n0 + 1] = -1_000_000;
        detile_accum_i16(&mut acc, n, &slot, m0, n0, mt, nt);
        for row in 0..mt {
            for col in 0..nt {
                let want = slot[out_idx_i4(mt, col + 1, row + 1)] as i64;
                let want = if (row, col) == (2, 1) {
                    want - 1_000_000
                } else {
                    want
                };
                assert_eq!(acc[(m0 + row) * n + n0 + col], want, "row={row} col={col}");
            }
        }
        assert_eq!(acc[0], 0);
        assert_eq!(acc[(m0 + mt) * n + n0], 0);
        assert_eq!(acc[m0 * n + n0 + nt], 0);
    }

    #[test]
    fn int4_group_wise_weight_tiles_round_trip_and_rescatter_identically() {
        // The group-wise resident pack tiles K at `Kt = group`, so one slot is exactly
        // one quant group of `group` nibbles per output channel. Walk every (N-tile,
        // K-tile) slot of a full weight and check (a) the nibbles land at
        // `wt_idx_i4(Kt=group, ·)` and (b) re-scattering a slot is idempotent — the C
        // `rk4_scatter_weights` calls `i4_wt_scatter` `Ntile` times in a row, which is
        // redundant rather than part of the layout.
        let (n, k, group) = (128usize, 256usize, 64usize);
        let (nt, kt) = (128usize, group);
        let b: Vec<i8> = (0..n * k).map(code).collect();
        let slot_bytes = nt.div_ceil(64) * 64 * kt.div_ceil(32) * 32 / 2;
        for ni in 0..n / nt {
            for ki in 0..k / kt {
                let (n0, k0) = (ni * nt, ki * kt);
                let mut slot = vec![0u8; slot_bytes];
                wt_scatter_tile_i4(&mut slot, &b, k, n0, k0, nt, kt);
                let mut again = vec![0u8; slot_bytes];
                for _ in 0..nt {
                    wt_scatter_tile_i4(&mut again, &b, k, n0, k0, nt, kt);
                }
                assert_eq!(again, slot, "re-scatter must be idempotent n0={n0} k0={k0}");
                for kk in 0..nt {
                    for cc in 0..kt {
                        assert_eq!(
                            get_nib(&slot, wt_idx_i4(kt, kk + 1, cc + 1)),
                            b[(n0 + kk) * k + k0 + cc],
                            "n0={n0} k0={k0} kk={kk} cc={cc}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn int4_int16_cube_detiles_scaled_per_group() {
        // The group-wise readback: each tile's int16 cube is one quant group, folded in
        // as `a_scale[row,g] * b_scale[col,g] * partial`, with a pre-existing f32
        // partial (an earlier group) added to.
        let (mt, nt): (usize, usize) = (5, 6);
        let (alloc_m, alloc_n) = (mt.div_ceil(4) * 4, nt.div_ceil(8) * 8);
        let slot: Vec<i16> = (0..alloc_m * alloc_n).map(|i| i as i16 * 3 - 20).collect();
        let (n, m0, n0, g) = (20, 4, 4, 2usize);
        let mut acc = vec![0.0f32; (m0 + mt + 1) * n];
        acc[(m0 + 2) * n + n0 + 1] = 0.5;
        let scale = |row: usize, col: usize| (row as f32 + 1.0) * (col as f32 + 0.5) * g as f32;
        detile_accum_i16_scaled(&mut acc, n, &slot, m0, n0, mt, nt, scale);
        for row in 0..mt {
            for col in 0..nt {
                let v = slot[out_idx_i4(mt, col + 1, row + 1)] as f32;
                let mut want = scale(m0 + row, n0 + col) * v;
                if (row, col) == (2, 1) {
                    want += 0.5;
                }
                assert_eq!(acc[(m0 + row) * n + n0 + col], want, "row={row} col={col}");
            }
        }
        assert_eq!(acc[0], 0.0);
        assert_eq!(acc[(m0 + mt) * n + n0], 0.0);
        assert_eq!(acc[m0 * n + n0 + nt], 0.0);
    }

    #[test]
    fn int4_nibble_layout_matches_the_c_index_math() {
        // wt_idx_i4's shape: (N/64, K/32, 64, 32) nibbles, with the stride of the
        // N-group counting ceil(Kt/32) K-groups.
        assert_eq!(wt_idx_i4(32, 1, 1), 0);
        assert_eq!(wt_idx_i4(32, 2, 1), 32);
        assert_eq!(wt_idx_i4(32, 1, 33), 64 * 32);
        // The next N group starts one K-group plane later, and that plane's stride is
        // 64*32 * ceil(Kt/32) — two K groups double it.
        assert_eq!(wt_idx_i4(32, 65, 1), 64 * 32);
        assert_eq!(wt_idx_i4(64, 65, 1), 2 * 64 * 32);
        // feat_idx_i4 is C2=32: 32 rows of a K-group, then the next K-group.
        assert_eq!(feat_idx_i4(4, 1, 1), 0);
        assert_eq!(feat_idx_i4(4, 2, 1), 1);
        assert_eq!(feat_idx_i4(4, 1, 2), 32);
        assert_eq!(feat_idx_i4(4, 33, 1), 4 * 32);
        // out_idx_i4 is C2=8.
        assert_eq!(out_idx_i4(4, 9, 1), 4 * 8);
        assert_eq!(out_idx_i4(4, 1, 2), 8);
    }
}

#[cfg(test)]
mod int8_bf16_tests {
    use super::*;

    #[test]
    fn int8_feature_round_trip() {
        let (mt, kt): (usize, usize) = (12, 64);
        let a: Vec<i8> = (0..mt * kt).map(|i| (i as i8).wrapping_mul(3)).collect();
        let mut slot = vec![0i8; mt.div_ceil(4) * 4 * kt];
        feat_scatter_i8(&mut slot, &a, kt, 0, 0, mt, kt);
        for row in 0..mt {
            for col in 0..kt {
                assert_eq!(slot[feat_idx_i8(mt, col + 1, row + 1)], a[row * kt + col]);
            }
        }
    }

    #[test]
    fn int8_weight_round_trip() {
        let (nt, kt): (usize, usize) = (40, 96);
        let b: Vec<i8> = (0..nt * kt).map(|i| (i as i8).wrapping_mul(5)).collect();
        let mut slot = vec![0i8; nt.div_ceil(32) * 32 * kt];
        wt_scatter_tile_i8(&mut slot, &b, kt, 0, 0, nt, kt);
        for k in 0..nt {
            for c in 0..kt {
                assert_eq!(slot[wt_idx_i8(kt, k + 1, c + 1)], b[k * kt + c]);
            }
        }
    }

    #[test]
    fn int32_cube_detiles_in_four_channel_groups() {
        // Ragged (mtile % 4, ntile % 4) int32 cube at a non-zero origin: the gather
        // walks 4-channel groups of the int8 output cube (C2=4), so element (row, col)
        // sits at out_idx_i8(mtile, col + 1, row + 1), and a pre-existing partial in
        // the i64 accumulator (an earlier K-tile) must be added to, not replaced.
        let (mt, nt): (usize, usize) = (3, 6);
        let (alloc_m, alloc_n) = (mt.div_ceil(4) * 4, nt.div_ceil(4) * 4);
        let slot: Vec<i32> = (0..alloc_m * alloc_n).map(|i| i as i32 * 7 - 40).collect();
        let (n, m0, n0) = (20, 4, 4);
        let mut acc = vec![0i64; (m0 + mt + 1) * n];
        acc[(m0 + 2) * n + n0 + 1] = -1_000_000;
        detile_accum_i32(&mut acc, n, &slot, m0, n0, mt, nt);
        for row in 0..mt {
            for col in 0..nt {
                let want = slot[out_idx_i8(mt, col + 1, row + 1)] as i64;
                let want = if (row, col) == (2, 1) {
                    want - 1_000_000
                } else {
                    want
                };
                assert_eq!(acc[(m0 + row) * n + n0 + col], want, "row={row} col={col}");
            }
        }
        assert_eq!(acc[0], 0);
        assert_eq!(acc[(m0 + mt) * n + n0], 0);
        assert_eq!(acc[m0 * n + n0 + nt], 0);
    }

    #[test]
    fn bf16_truncation_and_weight_round_trip() {
        assert_eq!(f32_to_bf16_bits(1.0), 0x3f80);
        assert_eq!(f32_to_bf16_bits(-2.0), 0xc000);
        let (nt, kt): (usize, usize) = (20, 64);
        let b: Vec<f32> = (0..nt * kt).map(|i| i as f32 * 0.125 - 1.0).collect();
        let mut slot = vec![0u16; nt.div_ceil(16) * 16 * kt];
        wt_scatter_tile_bf16(&mut slot, &b, kt, 0, 0, nt, kt);
        for k in 0..nt {
            for c in 0..kt {
                assert_eq!(
                    slot[wt_idx_i16(kt, k + 1, c + 1)],
                    f32_to_bf16_bits(b[k * kt + c])
                );
            }
        }
    }
}
