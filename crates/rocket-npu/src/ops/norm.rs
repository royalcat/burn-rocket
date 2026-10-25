//! Normalization primitives (RK3588): the per-row broadcast scale, RMSNorm and
//! LayerNorm, plus the fp16-square overflow guard they share.
//!
//! Port of `rocket_norm.c` + `rocket_norm.h` (the `rocket_norm_internal.h` guard
//! included). [`scale_rows`] is the M6b primitive (softmax's tail); [`rmsnorm`] and
//! [`layernorm`] are the M7 transformer norms, each with its exact host reference.
//!
//! # The cost model (read this before "optimizing")
//!
//! RMSNorm/LayerNorm are memory-bound elementwise+reduce ops. Standalone on the NPU they
//! are *submit*-bound (the flat EW path tiles ~1020 rows per submit, so a `[512,3840]`
//! square is ~60 submits), and for an isolated norm the host A76 single memory pass is
//! faster. The win is compositional: when the norm sits between two NPU matmuls (the
//! FFN/attention case), running it on-device lets the activation stay in the NC1HWC2 cube
//! and avoids the de-tile → host → re-pack layout round trip on every op, which is the
//! dominant not-MAC-bound cost. These primitives exist to be **fused into a resident
//! block**, not to beat the host standalone.
//!
//! # The work split
//!
//! The `O(M*H)` work (square, the H-contraction reduce, the final scale) runs on the
//! NPU; the `O(M)` per-row tail (mean, `+eps`, `rsqrt`) is exact fp32 host scalar work.
//! Sending `M` tiny per-row scalars back to the NPU for an *approximate* LUT `rsqrt`
//! would add a round trip **and** hit the rsqrt-LUT domain problem (`ms` spans many
//! decades across rows/layers; a uniform-grid LUT cannot cover that) for no benefit —
//! the DPU rsqrt LUT stays reserved for a large-**tensor** `rsqrt`, not an `M`-vector.
//!
//! # The fp16-square overflow guard
//!
//! Every op here that reduces `x^2` on the NPU hits the same wall: the square is
//! computed in fp16, so `|x|` above ~223 overflows (`223² = 49729` against fp16's
//! 65504) and the reduction returns `inf` for a row the fp64 reference handles fine.
//! [`square_prescale_k`] picks `k` for a power-of-two prescale `p = 2^-k` applied before
//! the square, which is **exact** (no rounding), with the true mean-square recovered on
//! the host as the reduced sum times `4^k`.
//!
//! The **reciprocal** side has no such guard, and cannot: `r[m] = 1/sqrt(ms[m]+eps)` is
//! combined with `weight` into a single fp16 scale `s[m][h] = r[m]*weight[h]` for one
//! EW multiply, so a row whose mean-square is near zero against a small `eps` makes `r`
//! large and can push that product past fp16's 65504 → `inf`, where the fp64 reference
//! stays finite. The input is assumed away from that edge (a normalization input at
//! `|x| ≈ 0` for a whole row is a degenerate activation, not a shape this op refuses);
//! raise `eps` if it is not.

use half::f16;

use crate::Device;
use crate::error::{Error, Result};
use crate::ops::activation::{ew_add, ew_mul};
use crate::ops::reduce::reduce_feature;

/// `rocket_square_prescale_k`: pick `k` so `(|x|max · 2^-k)²` stays under fp16 max.
///
/// `k == 0` — the common case, `|x| <= 223` — means the input is used directly with no
/// prescale and no copy. The C thresholds at 223 (`223² = 49729 < 65504`), takes
/// `ceil(log2(amax / 223))` and applies the scale with `ldexpf`, which is exact.
pub fn square_prescale_k(x: &[f16]) -> i32 {
    let mut amax = 0f32;
    for &v in x {
        let a = v.to_f32().abs();
        if a > amax {
            amax = a;
        }
    }
    if amax > 223.0 {
        (amax / 223.0).log2().ceil() as i32
    } else {
        0
    }
}

/// `2^k` for the small exponents the prescale uses (`k` in `-126..=127`).
fn pow2(k: i32) -> f32 {
    f32::from_bits(((127 + k) as u32) << 23)
}

/// The prescale the C computes as `ldexpf(1.f, -k)` plus the `4^k` recovery factor.
fn prescale(k: i32) -> (f32, f32) {
    (pow2(-k), pow2(2 * k))
}

/// Apply `p` to every element (an exact power-of-two scale), as the C's `xs_buf` pass.
fn prescaled(x: &[f16], p: f32) -> Vec<f16> {
    x.iter().map(|v| f16::from_f32(v.to_f32() * p)).collect()
}

/// `INT_MAX` element guard: the flat EW/LUT entries take an `i32` count, and a
/// truncating product would silently under-process the surface.
fn guard_elems(name: &'static str, elems: usize) -> Result<()> {
    if elems > i32::MAX as usize {
        return Err(Error::shape(name).with_detail(format!("{elems} elements > INT_MAX")));
    }
    Ok(())
}

/// `rocket_scale_rows_ref_fp16`: `out[m][n] = in[m][n] * r[m]` in f32, one RNE narrow.
pub fn scale_rows_ref(m: usize, n: usize, input: &[f16], r: &[f32], out: &mut [f16]) {
    assert!(input.len() >= m * n, "input shorter than M*N");
    assert!(r.len() >= m, "r shorter than M");
    assert!(out.len() >= m * n, "output shorter than M*N");
    for row in 0..m {
        let rm = r[row];
        let ip = &input[row * n..row * n + n];
        let op = &mut out[row * n..row * n + n];
        for (o, &v) in op.iter_mut().zip(ip) {
            *o = f16::from_f32(v.to_f32() * rm);
        }
    }
}

/// `rocket_scale_rows_fp16`: `out[m][n] = in[m][n] * r[m]` on the NPU.
///
/// The per-row scalar `r[m]` is fp32 and is **rounded to fp16** for the multiply (the C
/// header says so explicitly: the broadcast is materialized as an fp16 cube and the DPU
/// EW multiply does the arithmetic, so it inherits that path bit-for-bit). The
/// materialization is a pure host fill — no arithmetic — and then one
/// [`ew_mul`] call.
///
/// `m*n > INT_MAX` is a [`crate::error::Status::Shape`], exactly as in C: the element count goes to the
/// EW entry as an `i32`, and a truncating product would under-process the surface and
/// return a silently partial result.
pub fn scale_rows(
    dev: &Device,
    m: usize,
    n: usize,
    input: &[f16],
    r: &[f32],
    out: &mut [f16],
) -> Result<()> {
    if m < 1 || n < 1 {
        return Err(Error::shape("rocket_scale_rows_fp16"));
    }
    let mn = m * n;
    if input.len() < mn || r.len() < m || out.len() < mn {
        return Err(Error::shape("rocket_scale_rows_fp16/buffers"));
    }
    if mn > i32::MAX as usize {
        return Err(
            Error::shape("rocket_scale_rows_fp16").with_detail(format!("{mn} elements > INT_MAX"))
        );
    }

    let out = &mut out[..mn];
    let mut rb = vec![f16::ZERO; mn];
    for row in 0..m {
        let rv = f16::from_f32(r[row]);
        rb[row * n..row * n + n].fill(rv);
    }
    ew_mul(dev, input, &rb, out)
}

// ── RMSNorm ───────────────────────────────────────────────────────────────────

/// `rocket_rmsnorm_ref_fp16`: the exact fp64 reference of [`rmsnorm`].
///
/// `out[m][h] = x[m][h] / sqrt(mean_h(x[m][h]²) + eps) · weight[h]`, accumulated in
/// f64 over the f16 inputs with one RNE narrow at the end.
#[allow(clippy::too_many_arguments)]
pub fn rmsnorm_ref(m: usize, h: usize, x: &[f16], weight: &[f16], eps: f32, out: &mut [f16]) {
    assert!(x.len() >= m * h, "x shorter than M*H");
    assert!(weight.len() >= h, "weight shorter than H");
    assert!(out.len() >= m * h, "output shorter than M*H");
    let inv_h = 1.0 / h as f64;
    for row in 0..m {
        let xp = &x[row * h..row * h + h];
        let mut ss = 0.0f64;
        for &v in xp {
            let v = v.to_f64();
            ss += v * v;
        }
        let r = 1.0 / (ss * inv_h + f64::from(eps)).sqrt();
        let op = &mut out[row * h..row * h + h];
        for (o, (&v, &w)) in op.iter_mut().zip(xp.iter().zip(weight)) {
            *o = f16::from_f64(v.to_f64() * r * w.to_f64());
        }
    }
}

/// `rocket_rmsnorm_fp16`: per-row RMSNorm over the hidden axis, on the NPU.
///
/// `out[m][h] = x[m][h] / sqrt(mean_h(x[m][h]²) + eps) · weight[h]`
///
/// `weight` is the **effective** per-hidden scale `[H]` (Gemma stores `w` and uses
/// `1 + w`; pass that pre-incremented — this op does not add 1). `eps` is the fp32
/// epsilon (Gemma: 1e-6).
///
/// Chain, exactly as the C: `sq = x ⊙ x` (NPU EW, under the fp16-square prescale) →
/// `ms[m] = mean_h sq` (NPU feature reduce, fp32 accumulation) → `r[m] = 1/sqrt(ms+eps)`
/// (host, exact) → `out = x ⊙ (r ⊙ weight)` (one NPU EW multiply, the per-row `r` and
/// per-column `weight` pre-combined into a single fp16 `s`).
#[allow(clippy::too_many_arguments)]
pub fn rmsnorm(
    dev: &Device,
    m: usize,
    h: usize,
    x: &[f16],
    weight: &[f16],
    eps: f32,
    out: &mut [f16],
) -> Result<()> {
    if m < 1 || h < 1 {
        return Err(Error::shape("rocket_rmsnorm_fp16"));
    }
    let mh = m * h;
    if x.len() < mh || out.len() < mh || weight.len() < h {
        return Err(
            Error::shape("rocket_rmsnorm_fp16/buffers").with_detail(format!(
                "x {} (need {mh}), weight {} (need {h}), out {} (need {mh})",
                x.len(),
                weight.len(),
                out.len()
            )),
        );
    }
    guard_elems("rocket_rmsnorm_fp16", mh)?;

    let x = &x[..mh];
    let weight = &weight[..h];
    let out = &mut out[..mh];

    // 1. fp16-square overflow guard: scale x by p = 2^-k so (|x|max · p)² stays under
    //    fp16 max; k = 0 for the common |x| <= 223 case (no copy, x used directly).
    let k = square_prescale_k(x);
    let (p, four_k) = prescale(k);
    let xs;
    let sq_in: &[f16] = if k > 0 {
        xs = prescaled(x, p);
        &xs
    } else {
        x
    };

    // 2. sq = xs ⊙ xs (NPU), 3. ms = mean_h sq (NPU, fp32 accumulate; true ms = ms·4^k).
    let mut sq = vec![f16::ZERO; mh];
    ew_mul(dev, sq_in, sq_in, &mut sq)?;
    let mut ms = vec![0.0f32; m];
    reduce_feature(dev, m, h, &sq, &mut ms, true)?;

    // 4. per-row r[m] = 1/sqrt(ms·4^k + eps) (host, exact), combined with weight into s.
    let mut s = vec![f16::ZERO; mh];
    for row in 0..m {
        let r = 1.0 / (ms[row] * four_k + eps).sqrt();
        let srow = &mut s[row * h..row * h + h];
        for (cell, &w) in srow.iter_mut().zip(weight) {
            *cell = f16::from_f32(r * w.to_f32());
        }
    }

    // 5. out = x ⊙ s (NPU). One EW multiply: the per-row r and per-column weight are
    //    pre-combined here; the standalone per-row primitive is `scale_rows`.
    ew_mul(dev, x, &s, out)
}

// ── LayerNorm ─────────────────────────────────────────────────────────────────

/// `rocket_layernorm_ref_fp16`: the exact fp64 reference of [`layernorm`].
///
/// `mean = mean_h x`, `var = mean_h x² − mean²` (clamped at 0, as in C), then
/// `out = (x − mean)/sqrt(var + eps) · gamma + beta`, all in f64 over the f16 inputs.
#[allow(clippy::too_many_arguments)]
pub fn layernorm_ref(
    m: usize,
    h: usize,
    x: &[f16],
    gamma: &[f16],
    beta: Option<&[f16]>,
    eps: f32,
    out: &mut [f16],
) {
    assert!(x.len() >= m * h, "x shorter than M*H");
    assert!(gamma.len() >= h, "gamma shorter than H");
    assert!(beta.is_none_or(|b| b.len() >= h), "beta shorter than H");
    assert!(out.len() >= m * h, "output shorter than M*H");
    let inv_h = 1.0 / h as f64;
    for row in 0..m {
        let xp = &x[row * h..row * h + h];
        let (mut sx, mut sx2) = (0.0f64, 0.0f64);
        for &v in xp {
            let v = v.to_f64();
            sx += v;
            sx2 += v * v;
        }
        let mean = sx * inv_h;
        let mut var = sx2 * inv_h - mean * mean;
        if var < 0.0 {
            var = 0.0;
        }
        let r = 1.0 / (var + f64::from(eps)).sqrt();
        let op = &mut out[row * h..row * h + h];
        for (i, o) in op.iter_mut().enumerate() {
            let g = gamma[i].to_f64();
            let b = beta.map_or(0.0, |b| b[i].to_f64());
            *o = f16::from_f64((xp[i].to_f64() - mean) * r * g + b);
        }
    }
}

/// `rocket_layernorm_fp16`: per-row LayerNorm over the hidden axis, on the NPU.
///
/// ```text
/// mean[m]   = mean_h x[m][h]
/// var[m]    = mean_h x[m][h]² − mean[m]²
/// out[m][h] = (x[m][h] − mean[m]) / sqrt(var[m] + eps) · gamma[h] + beta[h]
/// ```
///
/// `gamma`/`beta` are the per-hidden affine `[H]`; `beta` may be `None` for no bias.
/// `eps` is the fp32 epsilon (Whisper: 1e-5).
///
/// **Both reductions share ONE feature-reduce job by STACKING the rows** — `A = [x ; x⊙x]`
/// (2M rows) under the ones weight, so the first `M` outputs are `sum(x)` and the next
/// `M` are `sum(x²)`; that is the C's documented LayerNorm optimization. The `O(M)`
/// per-row tail (mean, var, rsqrt) is exact fp32 host scalar work; the affine is folded
/// to `out = x ⊙ A + B` with `A[m][h] = r[m]·gamma[h]` and
/// `B[m][h] = beta[h] − mean[m]·r[m]·gamma[h]` (one NPU EW multiply + one EW add). The
/// fp16-square prescale matches [`rmsnorm`] and applies to the `x²` branch only.
#[allow(clippy::too_many_arguments)]
pub fn layernorm(
    dev: &Device,
    m: usize,
    h: usize,
    x: &[f16],
    gamma: &[f16],
    beta: Option<&[f16]>,
    eps: f32,
    out: &mut [f16],
) -> Result<()> {
    if m < 1 || h < 1 {
        return Err(Error::shape("rocket_layernorm_fp16"));
    }
    let mh = m * h;
    if x.len() < mh || out.len() < mh || gamma.len() < h || beta.is_some_and(|b| b.len() < h) {
        return Err(
            Error::shape("rocket_layernorm_fp16/buffers").with_detail(format!(
                "x {} / out {} (need {mh}), gamma {} / beta {} (need {h})",
                x.len(),
                out.len(),
                gamma.len(),
                beta.map_or(0, <[f16]>::len)
            )),
        );
    }
    guard_elems("rocket_layernorm_fp16", mh)?;

    let x = &x[..mh];
    let gamma = &gamma[..h];
    let out = &mut out[..mh];

    // fp16-square overflow guard (matches RMSNorm): prescale x by p = 2^-k for the x²
    // branch only — the mean(x) branch uses x directly (no square, no overflow).
    let k = square_prescale_k(x);
    let (p, four_k) = prescale(k);
    let xs;
    let sq_in: &[f16] = if k > 0 {
        xs = prescaled(x, p);
        &xs
    } else {
        x
    };
    let mut sq = vec![f16::ZERO; mh];
    ew_mul(dev, sq_in, sq_in, &mut sq)?;

    // 2. ONE stacked reduce: [x ; sq] (2M rows). Row m -> sum_h x; row M+m -> sum_h sq.
    let mut stack = vec![f16::ZERO; 2 * mh];
    stack[..mh].copy_from_slice(x);
    stack[mh..].copy_from_slice(&sq);
    let mut csum = vec![0.0f32; 2 * m];
    reduce_feature(dev, 2 * m, h, &stack, &mut csum, false)?;

    // 3. per-row mean/var/rsqrt (host, exact). sum(x²) = csum[M+m]·4^k.
    let inv_h = 1.0f32 / h as f32;
    let mut mean = vec![0.0f32; m];
    let mut rrow = vec![0.0f32; m];
    for row in 0..m {
        let mn = csum[row] * inv_h;
        let msq = csum[m + row] * four_k * inv_h;
        let mut var = msq - mn * mn;
        if var < 0.0 {
            var = 0.0;
        }
        mean[row] = mn;
        rrow[row] = 1.0 / (var + eps).sqrt();
    }

    // 4. fold the affine: out = x ⊙ A + B, so (x−mean)·r·gamma+beta == x·A+B.
    let mut a_buf = vec![f16::ZERO; mh];
    let mut b_buf = vec![f16::ZERO; mh];
    for row in 0..m {
        let (rm, mm) = (rrow[row], mean[row]);
        let (arow, brow) = (
            &mut a_buf[row * h..row * h + h],
            &mut b_buf[row * h..row * h + h],
        );
        for i in 0..h {
            let g = gamma[i].to_f32();
            let a = rm * g;
            arow[i] = f16::from_f32(a);
            let b = beta.map_or(0.0, |b| b[i].to_f32());
            brow[i] = f16::from_f32(b - mm * a);
        }
    }

    // 5. tmp = x ⊙ A ; out = tmp + B (two NPU EW passes).
    let mut tmp = vec![f16::ZERO; mh];
    ew_mul(dev, x, &a_buf, &mut tmp)?;
    ew_add(dev, &tmp, &b_buf, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vals(v: &[f32]) -> Vec<f16> {
        v.iter().map(|x| f16::from_f32(*x)).collect()
    }

    fn f64s(v: &[f16]) -> Vec<f64> {
        v.iter().map(|x| x.to_f64()).collect()
    }

    #[test]
    fn reference_scales_each_row_by_its_own_scalar() {
        let x = vals(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let r = [0.5f32, 2.0];
        let mut out = vec![f16::ZERO; 6];
        scale_rows_ref(2, 3, &x, &r, &mut out);
        let got: Vec<f32> = out.iter().map(|v| v.to_f32()).collect();
        assert_eq!(got, vec![0.5, 1.0, 1.5, 8.0, 10.0, 12.0]);

        // The scalar is rounded to fp16 before the multiply, so a value outside the fp16
        // grid lands on the nearest representable one — 1/3 becomes 0.33325...
        let r = [1.0f32 / 3.0];
        let x = vals(&[3.0]);
        let mut out = vec![f16::ZERO; 1];
        scale_rows_ref(1, 1, &x, &r, &mut out);
        assert_eq!(
            out[0],
            f16::from_f32(f16::from_f32(1.0 / 3.0).to_f32() * 3.0)
        );

        // An exact power-of-two scale is exact.
        let r = [0.25f32];
        let x = vals(&[1.0, -8.0, 1024.0]);
        let mut out = vec![f16::ZERO; 3];
        scale_rows_ref(1, 3, &x, &r, &mut out);
        assert_eq!(out, vals(&[0.25, -2.0, 256.0]));
    }

    #[test]
    fn prescale_threshold_is_223() {
        // 223² = 49729 still fits fp16; anything above needs a halving step.
        assert_eq!(square_prescale_k(&vals(&[223.0, -223.0])), 0);
        assert_eq!(square_prescale_k(&vals(&[223.5])), 1);
        assert_eq!(square_prescale_k(&vals(&[446.0])), 1);
        assert_eq!(square_prescale_k(&vals(&[447.0])), 2);
        // 65504 (fp16 max) as a single element: ceil(log2(65504/223)) = ceil(8.2) = 9,
        // so the squared value lands at (65504·2^-9)² = 16376² ... which is still large;
        // the guard is conservative (it keeps (|x|·2^-k) under 223, not just the square
        // under 65504), exactly as the C's `ceilf(log2f(amax / 223.f))` is.
        assert_eq!(square_prescale_k(&vals(&[65504.0])), 9);
        assert_eq!(square_prescale_k(&[]), 0);
        // The prescale itself is exact and the recovery factor undoes it.
        let (p, four_k) = prescale(3);
        assert_eq!(p, 0.125);
        assert_eq!(four_k, 64.0);
        assert_eq!(pow2(0), 1.0);
        assert_eq!(pow2(-1), 0.5);
    }

    #[test]
    fn rmsnorm_reference_is_unit_rms() {
        // gamma = 1: the output row has RMS 1 (up to the eps and the fp16 grid).
        let x = vals(&[3.0, 4.0, 0.0, 0.0, 1.0, -1.0, 2.0, -2.0]);
        let w = vals(&[1.0; 4]);
        let mut out = vec![f16::ZERO; 8];
        rmsnorm_ref(2, 4, &x, &w, 0.0, &mut out);
        // Row 0: [3,4,0,0] -> ms 6.25 -> r 0.4 -> [1.2, 1.6, 0, 0]; row 1: [1,-1,2,-2]
        // -> ms 2.5 -> r 0.6325 -> [0.6325, -0.6325, 1.2649, -1.2649].
        let got = f64s(&out);
        assert!((got[0] - 1.2).abs() < 1e-3);
        assert!((got[1] - 1.6).abs() < 1e-3);
        assert_eq!(got[2], 0.0);
        assert_eq!(got[3], 0.0);
        let r = 1.0 / 2.5f64.sqrt();
        assert!((got[4] - r).abs() < 1e-3);
        assert!((got[5] + r).abs() < 1e-3);
        assert!((got[6] - 2.0 * r).abs() < 1e-3);
        assert!((got[7] + 2.0 * r).abs() < 1e-3);
        // gamma scales per column, not per row.
        let w = vals(&[2.0, 0.5, 1.0, 0.0]);
        rmsnorm_ref(1, 4, &x, &w, 0.0, &mut out);
        let got = f64s(&out);
        assert!((got[0] - 2.4).abs() < 2e-3);
        assert!((got[1] - 0.8).abs() < 2e-3);
        assert_eq!(got[3], 0.0);
    }

    #[test]
    fn rmsnorm_reference_honours_eps() {
        // A zero row with eps > 0 stays finite and zero (the C edge case the header
        // warns about only shows up when eps is ~0 and weight is large).
        let x = vals(&[0.0, 0.0, 0.0, 0.0]);
        let w = vals(&[1.0; 4]);
        let mut out = vec![f16::ZERO; 4];
        rmsnorm_ref(1, 4, &x, &w, 1e-6, &mut out);
        assert!(out.iter().all(|v| v.to_f32() == 0.0));

        // eps shrinks the output by 1/sqrt(ms+eps): [4,4,4,4] ms=16, eps=16 -> r=1/√32.
        let x = vals(&[4.0; 4]);
        rmsnorm_ref(1, 4, &x, &w, 16.0, &mut out);
        let want = (4.0 / 32.0f64.sqrt()) as f32;
        assert!((out[0].to_f32() - want).abs() < 1e-3);
    }

    #[test]
    fn layernorm_reference_zero_mean_unit_var() {
        let x = vals(&[1.0, 2.0, 3.0, 4.0, -2.0, -1.0, 1.0, 2.0]);
        let g = vals(&[1.0; 4]);
        let mut out = vec![f16::ZERO; 8];
        // gamma=1, beta=NULL: the output is the standardized row.
        layernorm_ref(2, 4, &x, &g, None, 0.0, &mut out);
        // Row 0: mean 2.5, var 1.25 -> r = 0.8944 -> [-1.3416, -0.4472, 0.4472, 1.3416].
        let want = [-1.341_641, -0.447_214, 0.447_214, 1.341_641];
        for (o, w) in out.iter().zip(want) {
            assert!((o.to_f64() - w).abs() < 2e-3, "{o} vs {w}");
        }
        // Row 1: mean 0, var 2.5 -> [−1.2649, −0.6325, 0.6325, 1.2649].
        let want = [-1.264_911, -0.632_456, 0.632_456, 1.264_911];
        for (o, w) in out[4..].iter().zip(want) {
            assert!((o.to_f64() - w).abs() < 2e-3, "{o} vs {w}");
        }

        // Non-trivial gamma/beta: out = z·gamma + beta.
        let g = vals(&[2.0, 1.0, 0.5, 0.0]);
        let b = vals(&[0.5, -0.5, 1.0, 2.0]);
        layernorm_ref(1, 4, &x, &g, Some(&b), 0.0, &mut out);
        let z = [-1.341_641, -0.447_214, 0.447_214, 1.341_641];
        for i in 0..4 {
            let want = z[i] * f64::from([2.0f32, 1.0, 0.5, 0.0][i])
                + f64::from([0.5f32, -0.5, 1.0, 2.0][i]);
            assert!(
                (out[i].to_f64() - want).abs() < 2e-3,
                "{i}: {} vs {want}",
                out[i]
            );
        }
    }

    #[test]
    fn layernorm_reference_clamps_negative_variance() {
        // A constant row has var 0; the fp64 path computes it exactly, but a row whose
        // rounding made var < 0 must clamp rather than produce NaN.
        let x = vals(&[3.0; 8]);
        let g = vals(&[1.0; 8]);
        let mut out = vec![f16::ZERO; 8];
        layernorm_ref(1, 8, &x, &g, None, 1e-5, &mut out);
        assert!(out.iter().all(|v| v.to_f32() == 0.0));
        // The device path's variance is computed as msq − mean² in f32 and clamped the
        // same way; this is the shape where that clamp matters.
        assert_eq!(square_prescale_k(&x), 0);
    }

    #[test]
    fn layernorm_and_rmsnorm_agree_when_mean_is_zero_and_gamma_is_unit() {
        // A symmetric row has mean 0, so LayerNorm == RMSNorm up to the eps handling.
        let x = vals(&[1.0, -1.0, 2.0, -2.0]);
        let g = vals(&[1.0; 4]);
        let mut a = vec![f16::ZERO; 4];
        let mut b = vec![f16::ZERO; 4];
        layernorm_ref(1, 4, &x, &g, None, 0.0, &mut a);
        rmsnorm_ref(1, 4, &x, &g, 0.0, &mut b);
        for (u, v) in a.iter().zip(&b) {
            assert!((u.to_f32() - v.to_f32()).abs() < 1e-3);
        }
    }

    #[test]
    fn buffers_are_validated() {
        // The device entry needs a device, but every length/geometry check runs first,
        // so these reach a typed error without one (the C checks the shape before fd).
        let dev = match Device::open() {
            Ok(d) => d,
            Err(_) => return, // no device on this host; the ref path is covered above
        };
        let x = vec![f16::ZERO; 6];
        let w = vec![f16::ONE; 3];
        let mut out = vec![f16::ZERO; 6];
        assert!(rmsnorm(&dev, 0, 3, &x, &w, 1e-6, &mut out).is_err());
        assert!(rmsnorm(&dev, 2, 0, &x, &w, 1e-6, &mut out).is_err());
        assert!(rmsnorm(&dev, 2, 4, &x, &w, 1e-6, &mut out).is_err());
        assert!(rmsnorm(&dev, 3, 3, &x, &w, 1e-6, &mut out).is_err());
        assert!(layernorm(&dev, 0, 3, &x, &w, None, 1e-5, &mut out).is_err());
        assert!(layernorm(&dev, 2, 4, &x, &w, None, 1e-5, &mut out).is_err());
        assert!(layernorm(&dev, 2, 3, &x, &w, Some(&w[..2]), 1e-5, &mut out).is_err());
        assert!(scale_rows(&dev, 0, 3, &x, &[1.0f32; 2], &mut out).is_err());
        assert!(scale_rows(&dev, 2, 4, &x, &[1.0f32; 2], &mut out).is_err());
    }
}
