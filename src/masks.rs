//! Additive f16 attention masks (pure host-side math, no FFI).
//!
//! Kept outside the `npu`-gated extension module so the semantics are unit
//! tested on any host: `cargo test -p burn-rocket`.
#![cfg_attr(not(feature = "npu"), allow(dead_code))]

use half::f16;

/// Symmetric band additive mask: `mask[t][j] = 0` for `|t - j| <= window`,
/// `-inf` otherwise.
pub(crate) fn build_window_mask(n: usize, window: usize) -> Vec<f16> {
    let mut mask = vec![f16::ZERO; n * n];
    for t in 0..n {
        let row = &mut mask[t * n..(t + 1) * n];
        let lo = t.saturating_sub(window);
        let hi = (t + window + 1).min(n);
        for m in row[..lo].iter_mut() {
            *m = f16::NEG_INFINITY;
        }
        for m in row[hi..].iter_mut() {
            *m = f16::NEG_INFINITY;
        }
    }
    mask
}
/// Causal sliding-window additive mask: `mask[t][j] = 0` for
/// `t - window < j <= t`, `-inf` otherwise.
pub(crate) fn build_causal_window_mask(n: usize, window: usize) -> Vec<f16> {
    let mut mask = vec![f16::NEG_INFINITY; n * n];
    for t in 0..n {
        let row = &mut mask[t * n..(t + 1) * n];
        let lo = (t + 1).saturating_sub(window);
        for m in row[lo..=t].iter_mut() {
            *m = f16::ZERO;
        }
    }
    mask
}
/// Causal additive mask: `mask[t][j] = 0` for `j <= t`, `-inf` otherwise.
pub(crate) fn build_causal_mask(n: usize) -> Vec<f16> {
    build_causal_mask_block(n, n, 0, 0)
}

/// Causal additive mask for a query chunk: `mask[i][j] = 0` for
/// `k0 + j <= q0 + i`, `-inf` otherwise.
///
/// Lets a causal layer run a query chunk `[q0, q0 + n_q)` against a *subset* of
/// the key axis (`n_kv` keys starting at `k0`) while keeping the mask at
/// `[n_q, n_kv]` instead of materializing the model's full `[n, n]` matrix.
/// `q0 == k0 == 0`, `n_q == n_kv` reproduces [`build_causal_mask`].
///
/// A chunk whose keys all sit above the queries (or a key axis that ends before
/// the chunk) simply masks the rows it cannot see; causal chunk plans never
/// produce a fully masked row, but the builder stays total so an arbitrary
/// `(q0, k0)` pair is well defined.
pub(crate) fn build_causal_mask_block(n_q: usize, n_kv: usize, q0: usize, k0: usize) -> Vec<f16> {
    build_causal_window_mask_block(n_q, n_kv, q0, k0, None)
}

/// Causal sliding-window additive mask for a query chunk:
/// `mask[i][j] = 0` for `q0 + i - window < k0 + j <= q0 + i`, `-inf` otherwise
/// (the same `t - window < j <= t` convention as [`build_causal_window_mask`],
/// i.e. at most `window` keys per row). `None` is pure causal.
pub(crate) fn build_causal_window_mask_block(
    n_q: usize,
    n_kv: usize,
    q0: usize,
    k0: usize,
    window: Option<usize>,
) -> Vec<f16> {
    let mut mask = vec![f16::NEG_INFINITY; n_q * n_kv];
    for i in 0..n_q {
        let q = q0 + i;
        // Visible keys are `[q - window + 1, q]` in absolute terms.
        let lo = match window {
            Some(w) => (q + 1).saturating_sub(w),
            None => 0,
        };
        // Keys are `[k0, k0 + n_kv)`: keep `j` with `lo <= k0 + j <= q`.
        let lo = lo.saturating_sub(k0).min(n_kv);
        let hi = (q + 1).saturating_sub(k0).clamp(lo, n_kv);
        for m in mask[i * n_kv + lo..i * n_kv + hi].iter_mut() {
            *m = f16::ZERO;
        }
    }
    mask
}

/// Bidirectional band mask for a query chunk: `mask[i][j] = 0` for
/// `|(q0 + i) - (k0 + j)| <= window`, `-inf` otherwise.
///
/// Used to run a sliding layer's attention for a chunk of queries against only
/// the keys near it (`k0 = q0 - window` and `n_kv` spanning the band) instead of
/// materializing the full `[n, n]` matrix.
/// Used by the NPU extension; covered by the host tests when `npu` is off.
pub(crate) fn build_window_mask_block(
    n_q: usize,
    n_kv: usize,
    q0: usize,
    k0: usize,
    window: usize,
) -> Vec<f16> {
    let mut mask = vec![f16::NEG_INFINITY; n_q * n_kv];
    for i in 0..n_q {
        let q = q0 + i;
        // Key `k0 + j` is inside the band iff `k0 + j` is in `[q - window, q + window]`.
        let lo = q.saturating_sub(window).saturating_sub(k0).min(n_kv);
        let hi = (q + window + 1).saturating_sub(k0).clamp(lo, n_kv);
        for m in mask[i * n_kv + lo..i * n_kv + hi].iter_mut() {
            *m = f16::ZERO;
        }
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keep(mask: &[f16], n: usize, t: usize, j: usize) -> bool {
        mask[t * n + j] == f16::ZERO
    }

    #[test]
    fn causal_mask_semantics() {
        let n = 6;
        let m = build_causal_mask(n);
        for t in 0..n {
            for j in 0..n {
                assert_eq!(keep(&m, n, t, j), j <= t, "t={t} j={j}");
            }
        }
    }

    #[test]
    fn window_mask_semantics() {
        let n = 8;
        let w = 3;
        let m = build_window_mask(n, w);
        for t in 0..n {
            for j in 0..n {
                assert_eq!(keep(&m, n, t, j), t.abs_diff(j) <= w, "t={t} j={j}");
            }
        }
    }

    #[test]
    fn causal_window_mask_semantics() {
        let n = 8;
        let w = 3;
        let m = build_causal_window_mask(n, w);
        for t in 0..n {
            for j in 0..n {
                let expect = j <= t && t - j < w;
                assert_eq!(keep(&m, n, t, j), expect, "t={t} j={j}");
            }
        }
    }

    #[test]
    fn window_mask_block_semantics() {
        let (n_q, n_kv, q0, k0, w) = (5, 7, 10, 8, 3);
        let m = build_window_mask_block(n_q, n_kv, q0, k0, w);
        for i in 0..n_q {
            for j in 0..n_kv {
                let expect = (q0 + i).abs_diff(k0 + j) <= w;
                assert_eq!(keep(&m, n_kv, i, j), expect, "i={i} j={j}");
            }
        }
    }

    #[test]
    fn window_mask_block_clamped_edges() {
        // First chunk (k0 = q0 = 0) and a chunk whose band runs past the last key.
        let m = build_window_mask_block(4, 6, 0, 0, 2);
        assert!(keep(&m, 6, 0, 0));
        assert!(keep(&m, 6, 0, 2));
        assert!(!keep(&m, 6, 0, 3));
        let m = build_window_mask_block(4, 5, 20, 18, 2);
        // Rows 20..24 vs keys 18..23: row 23 keeps keys 21..23 (22, 23 missing).
        for i in 0..4 {
            for j in 0..5 {
                let expect = (20usize + i).abs_diff(18usize + j) <= 2;
                assert_eq!(keep(&m, 5, i, j), expect, "i={i} j={j}");
            }
        }
    }

    /// The causal full mask and the block builder must agree at the identity offset.
    #[test]
    fn causal_mask_block_matches_the_full_mask() {
        for n in [1usize, 2, 5, 9, 33] {
            assert_eq!(build_causal_mask(n), build_causal_mask_block(n, n, 0, 0));
            // A chunk at the same offset as its keys behaves like a standalone mask.
            for (n_q, n_kv, q) in [(4usize, 4usize, 0usize), (5, 5, 4), (3, 3, 40)] {
                assert_eq!(
                    build_causal_mask_block(n_q, n_kv, q, q),
                    build_causal_mask_block(n_q, n_kv, 0, 0),
                );
            }
        }
        // Row `i` of the identity block keeps exactly `i + 1` keys.
        let n = 9;
        let m = build_causal_mask_block(n, n, 0, 0);
        for i in 0..n {
            assert_eq!(
                m[i * n..(i + 1) * n]
                    .iter()
                    .filter(|v| **v == f16::ZERO)
                    .count(),
                i + 1
            );
        }
    }

    #[test]
    fn causal_mask_block_semantics_and_offsets() {
        let (n_q, n_kv, q0, k0) = (4, 10, 8, 4);
        let m = build_causal_mask_block(n_q, n_kv, q0, k0);
        for i in 0..n_q {
            for j in 0..n_kv {
                let expect = k0 + j <= q0 + i;
                assert_eq!(keep(&m, n_kv, i, j), expect, "i={i} j={j}");
            }
        }
    }

    #[test]
    fn causal_mask_block_ragged_tail() {
        // Last chunk of 30k at a 1024 width: 8 rows left, the key axis is the prefix.
        let (n_q, n_kv, q0) = (8, 29_960, 29_952);
        let m = build_causal_mask_block(n_q, n_kv, q0, 0);
        for i in 0..n_q {
            let probes = [
                0,
                1,
                q0 + i - 1,
                q0 + i,
                (q0 + i + 1).min(n_kv - 1),
                n_kv - 1,
            ];
            for &j in &probes {
                assert_eq!(keep(&m, n_kv, i, j), j <= q0 + i, "i={i} j={j}");
            }
            // The visible run is exactly the row's causal prefix.
            let vis = m[i * n_kv..(i + 1) * n_kv]
                .iter()
                .filter(|v| **v == f16::ZERO)
                .count();
            assert_eq!(vis, q0 + i + 1, "row {i}");
        }
    }

    #[test]
    fn causal_mask_block_key_axis_below_and_above_the_chunk() {
        // Keys entirely below the chunk: every key visible (row spans the whole axis).
        let m = build_causal_mask_block(4, 8, 20, 12);
        for i in 0..4 {
            for j in 0..8 {
                assert!(keep(&m, 8, i, j), "i={i} j={j}");
            }
        }
        // Keys entirely above the chunk: fully masked rows (the builder stays total;
        // a causal chunk plan never asks for this, but the predicate is well defined).
        let m = build_causal_mask_block(4, 8, 20, 24);
        assert!(m.iter().all(|v| *v == f16::NEG_INFINITY));
        // A one-row overlap at the boundary.
        let m = build_causal_mask_block(4, 4, 6, 8); // queries 6..9, keys 8..11
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(keep(&m, 4, i, j), 8 + j <= 6 + i, "i={i} j={j}");
            }
        }
    }

    #[test]
    fn causal_window_mask_block_semantics() {
        let (n_q, n_kv, q0, k0, w) = (5, 12, 16, 8, 4);
        let m = build_causal_window_mask_block(n_q, n_kv, q0, k0, Some(w));
        for i in 0..n_q {
            for j in 0..n_kv {
                let (q, k) = (q0 + i, k0 + j);
                let expect = k <= q && q - k < w;
                assert_eq!(keep(&m, n_kv, i, j), expect, "i={i} j={j}");
            }
        }
        // A dense (un-windowed) block is the `None` case.
        let a = build_causal_window_mask_block(n_q, n_kv, q0, k0, None);
        let b = build_causal_mask_block(n_q, n_kv, q0, k0);
        assert_eq!(a, b);
        // At window 1 only the diagonal is visible; a row's visible run is <= window.
        let m = build_causal_window_mask_block(6, 6, 0, 0, Some(2));
        for i in 0..6 {
            let vis = m[i * 6..(i + 1) * 6]
                .iter()
                .filter(|v| **v == f16::ZERO)
                .count();
            assert_eq!(vis, (i + 1).min(2), "row {i}");
        }
    }
}
