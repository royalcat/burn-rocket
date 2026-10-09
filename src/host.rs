//! Fused host-side elementwise kernels for the NPU build's CPU glue.
//!
//! `flex` runs elementwise chains as several single-threaded scalar passes
//! (one per op, with a libm call per element for `tanh`/`powf`), which on the
//! board accounts for most of the non-NPU time. These kernels compute the same
//! formulas in one (rayon-parallel) pass. Kept free of `npu`-gated code so the
//! math is unit tested on any host (`cargo test -p burn-rocket`).

/// `sqrt(2/π)` — the constant Burn's `gelu_approximate` uses (f64, as there).
const SQRT_2_OVER_PI: f64 = core::f64::consts::FRAC_2_SQRT_PI * core::f64::consts::FRAC_1_SQRT_2;

/// Burn's `gelu_approximate` (tanh approximation), elementwise:
/// `0.5 x (1 + tanh(sqrt(2/π) (x + 0.044715 x^3)))`.
#[inline]
pub(crate) fn gelu_tanh(x: f32) -> f32 {
    let inner = (x + x * x * x * 0.044715) * SQRT_2_OVER_PI as f32;
    (x * (inner.tanh() + 1.0)) * 0.5
}

/// `out = gelu_approximate(gate) * up`, elementwise.
pub(crate) fn gelu_mul(gate: &[f32], up: &[f32], out: &mut [f32]) {
    debug_assert_eq!(gate.len(), up.len());
    debug_assert_eq!(gate.len(), out.len());
    for ((o, &g), &u) in out.iter_mut().zip(gate).zip(up) {
        *o = gelu_tanh(g) * u;
    }
}

/// `out = silu(gate) * up` with Burn's `silu(x) = x * sigmoid(x)`,
/// `sigmoid(x) = 1 / (1 + exp(-x))`, elementwise (the Qwen MLP gate).
pub(crate) fn silu_mul(gate: &[f32], up: &[f32], out: &mut [f32]) {
    debug_assert_eq!(gate.len(), up.len());
    debug_assert_eq!(gate.len(), out.len());
    for ((o, &g), &u) in out.iter_mut().zip(gate).zip(up) {
        let sigmoid = 1.0 / (1.0 + (-g).exp());
        *o = g * sigmoid * u;
    }
}

/// One row of `x * (mean(x^2) + eps)^-0.5 * weight` in f32. An empty `weight`
/// means no scale (the value-norm form).
pub(crate) fn rms_norm_row(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    debug_assert_eq!(x.len(), out.len());
    debug_assert!(weight.is_empty() || weight.len() == x.len());
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    if weight.is_empty() {
        for (o, &v) in out.iter_mut().zip(x) {
            *o = v * inv;
        }
    } else {
        for ((o, &v), &w) in out.iter_mut().zip(x).zip(weight) {
            *o = v * inv * w;
        }
    }
}

/// Rotate-half RoPE for one head row `x[2 * half]` with `cos`/`sin` `[half]`:
/// `out[j] = x[j] * cos[j] - x[half + j] * sin[j]`,
/// `out[half + j] = x[half + j] * cos[j] + x[j] * sin[j]`.
pub(crate) fn rope_row(x: &[f32], cos: &[f32], sin: &[f32], half: usize, out: &mut [f32]) {
    debug_assert_eq!(x.len(), 2 * half);
    debug_assert_eq!(out.len(), 2 * half);
    debug_assert_eq!(cos.len(), half);
    debug_assert_eq!(sin.len(), half);
    for j in 0..half {
        let (x1, x2) = (x[j], x[half + j]);
        let (c, s) = (cos[j], sin[j]);
        out[j] = x1 * c - x2 * s;
        out[half + j] = x2 * c + x1 * s;
    }
}

/// One row of symmetric group-wise int8 quantization: for each `group`-wide K
/// block, `scale = max|x| / 127` and `q = round(x / scale)` clamped to
/// `[-127, 127]`. `srow` must have `xrow.len() / group` entries.
pub(crate) fn quantize_row_groups(xrow: &[f32], qrow: &mut [i8], srow: &mut [f32], group: usize) {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The composite `burn_tensor::activation::gelu_approximate` formula, as a
    /// f32 reference over the same operations.
    fn gelu_reference(x: f32) -> f32 {
        let inner = x + x.powf(3.0) * 0.044715;
        let inner = inner * SQRT_2_OVER_PI as f32;
        (x * (inner.tanh() + 1.0)) * 0.5
    }

    #[test]
    fn gelu_tanh_matches_the_composite() {
        let xs: Vec<f32> = (-600..=600).map(|i| i as f32 * 0.05).collect();
        for &x in &xs {
            let got = gelu_tanh(x);
            let want = gelu_reference(x);
            assert!(
                (got - want).abs() <= 1e-6 * want.abs().max(1.0),
                "gelu({x}) = {got}, composite {want}"
            );
        }
    }

    #[test]
    fn gelu_mul_is_gelu_times_up() {
        let gate = [0.0f32, 1.0, -1.0, 3.7, -12.5];
        let up = [2.0f32, -1.0, 0.5, 0.25, 8.0];
        let mut out = [0f32; 5];
        gelu_mul(&gate, &up, &mut out);
        for i in 0..5 {
            assert_eq!(out[i], gelu_tanh(gate[i]) * up[i]);
        }
    }

    #[test]
    fn silu_mul_is_silu_times_up() {
        let gate = [0.0f32, 1.0, -1.0, 3.7, -12.5];
        let up = [2.0f32, -1.0, 0.5, 0.25, 8.0];
        let mut out = [0f32; 5];
        silu_mul(&gate, &up, &mut out);
        for i in 0..5 {
            let g = gate[i];
            let sigmoid = 1.0 / (1.0 + (-g).exp());
            let want = g * sigmoid * up[i];
            assert!(
                (out[i] - want).abs() <= 1e-6 * want.abs().max(1.0),
                "silu_mul({g}) = {}, composite {want}",
                out[i]
            );
        }
    }

    #[test]
    fn rms_norm_row_scaled_and_plain() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let w = [1.0f32, 0.5, 2.0, -1.0];
        let eps = 1e-6f32;
        let ms = (1.0 + 4.0 + 9.0 + 16.0) / 4.0;
        let inv = 1.0 / (ms + eps).sqrt();

        let mut out = [0f32; 4];
        rms_norm_row(&x, &w, eps, &mut out);
        for i in 0..4 {
            assert!((out[i] - x[i] * inv * w[i]).abs() < 1e-6);
        }
        rms_norm_row(&x, &[], eps, &mut out);
        for i in 0..4 {
            assert!((out[i] - x[i] * inv).abs() < 1e-6);
        }
    }

    #[test]
    fn rope_row_matches_rotation() {
        let half = 3;
        let x = [1.0f32, 2.0, 3.0, -1.0, 0.5, 4.0];
        let cos = [1.0f32, 0.5, -0.25];
        let sin = [0.0f32, 0.5, 0.75];
        let mut out = [0f32; 6];
        rope_row(&x, &cos, &sin, half, &mut out);
        for j in 0..half {
            let (x1, x2) = (x[j], x[half + j]);
            assert_eq!(out[j], x1 * cos[j] - x2 * sin[j]);
            assert_eq!(out[half + j], x2 * cos[j] + x1 * sin[j]);
        }
    }

    /// Deterministic pseudo-random f32 in [-1, 1] (xorshift, no deps).
    fn rand_unit(seed: &mut u64) -> f32 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        ((*seed >> 40) as f32 / (1u64 << 23) as f32) - 1.0
    }

    #[test]
    fn int8_quantize_roundtrip_error_bound() {
        // Per K-group int8: |x - dequant(q)| <= scale/2, the block max hits ±127,
        // and a wider group with an outlier gets a wider scale (the layout check:
        // one scale per group, in order).
        let (k, group) = (128usize, 32usize);
        let mut seed = 0x1234_5678_9abc_def0u64;
        let mut x = vec![0f32; k];
        for v in x.iter_mut() {
            *v = rand_unit(&mut seed);
        }
        x[70] = 9.5; // outlier in group 2
        let mut q = vec![0i8; k];
        let mut s = vec![0f32; k / group];
        quantize_row_groups(&x, &mut q, &mut s, group);
        for g in 0..k / group {
            let seg = &x[g * group..(g + 1) * group];
            let amax = seg.iter().fold(0f32, |a, &v| a.max(v.abs()));
            assert!((s[g] - amax / 127.0).abs() <= 1e-9 * amax.max(1.0));
            for (i, &v) in seg.iter().enumerate() {
                let deq = q[g * group + i] as f32 * s[g];
                assert!((v - deq).abs() <= s[g] * 0.5 + 1e-6, "g={g} i={i}");
            }
        }
        assert_eq!(q[70], 127, "the group max must quantize to +127");
        assert!(s[2] > s[1], "the outlier group must get the wider scale");

        // An all-zero group stays finite: codes 0, a tiny positive scale.
        let z = [0f32; 32];
        let mut zq = [0i8; 32];
        let mut zs = [0f32; 1];
        quantize_row_groups(&z, &mut zq, &mut zs, 32);
        assert!(zs[0] > 0.0 && zs[0].is_finite());
        assert!(zq.iter().all(|&v| v == 0));
    }

    #[test]
    fn int8_groupwise_matmul_is_close() {
        // The full convention end to end: int8 codes + per-row/per-group scales,
        // int32-exact partials, fp32 dequant (`sum_g a_s[m,g] b_s[n,g] * partial`).
        // Checks the scale application (a transposed/other layout fails loudly).
        let (m, k, n, group) = (8usize, 64usize, 8usize, 32usize);
        let n_g = k / group;
        let mut seed = 7u64;
        let a: Vec<f32> = (0..m * k).map(|_| rand_unit(&mut seed) * 2.0).collect();
        let b: Vec<f32> = (0..n * k).map(|_| rand_unit(&mut seed)).collect();

        let mut aq = vec![0i8; m * k];
        let mut a_s = vec![0f32; m * n_g];
        for r in 0..m {
            quantize_row_groups(
                &a[r * k..(r + 1) * k],
                &mut aq[r * k..(r + 1) * k],
                &mut a_s[r * n_g..(r + 1) * n_g],
                group,
            );
        }
        let mut bq = vec![0i8; n * k];
        let mut b_s = vec![0f32; n * n_g];
        for r in 0..n {
            quantize_row_groups(
                &b[r * k..(r + 1) * k],
                &mut bq[r * k..(r + 1) * k],
                &mut b_s[r * n_g..(r + 1) * n_g],
                group,
            );
        }

        for row in 0..m {
            for col in 0..n {
                let want: f32 = (0..k).map(|i| a[row * k + i] * b[col * k + i]).sum();
                let mut got = 0f32;
                for g in 0..n_g {
                    let partial: i32 = (0..group)
                        .map(|i| {
                            let i = g * group + i;
                            aq[row * k + i] as i32 * bq[col * k + i] as i32
                        })
                        .sum();
                    got += a_s[row * n_g + g] * b_s[col * n_g + g] * partial as f32;
                }
                // The partials are int32-exact; the only error is quantization
                // (per-32-group int8), a few percent for random data. A scale
                // layout mistake would be order-100%.
                let tol = 0.1 * want.abs().max(1.0);
                assert!((got - want).abs() <= tol, "C[{row},{col}]: {got} vs {want}");
            }
        }
    }
}
