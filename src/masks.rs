//! Additive f16 attention masks (pure host-side math, no FFI).
//!
//! Kept outside the `npu`-gated extension module so the semantics are unit
//! tested on any host: `cargo test -p burn-rocket`.

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
}
