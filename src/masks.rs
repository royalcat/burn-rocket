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
    let mut mask = vec![f16::ZERO; n * n];
    for t in 0..n {
        let row = &mut mask[t * n..(t + 1) * n];
        for m in row[t + 1..].iter_mut() {
            *m = f16::NEG_INFINITY;
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
}
