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
