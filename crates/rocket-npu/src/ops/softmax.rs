//! Row-wise softmax / log-softmax / cross-entropy (RK3588): port of `rocket_softmax.c`.
//!
//! Almost none of this is new hardware work — it is a *composition* of the shipped
//! primitives, which is exactly how the C library builds it:
//!
//! ```text
//!  1. rowmax[m] = max_n x[m][n],  xs = x - rowmax      HOST (O(M*N) memory pass)
//!  2. e = exp(xs)                                      NPU  DPU LUT, Activation::Exp
//!  3. s[m] = sum_n e[m][n]                             NPU  feature-axis ones-matmul (fp32)
//!  4. inv[m] = 1/s[m]                                  HOST (O(M), exact)
//!  5. out = e * inv  (broadcast over n)                NPU  per-row scale = EW multiply
//! ```
//!
//! The `O(M*N)` work (exp, sum, scale) runs on the NPU; the genuinely new piece is the
//! row maximum, and it stays on the host:
//!
//! - the feature reduce can only **sum** — the matmul has no max-reduce datapath;
//! - the only on-NPU max is the PPU max-pool, which would need the sequence axis laid on
//!   the PPU's spatial axis plus telescoping for `N > 16` (the [`crate::ops::reduce`]
//!   machinery with [`crate::regcmd::pool::PoolMethod::Max`]). That is the *resident
//!   fusion* path (keep the scores cube-resident, avoid the readback) and it is a perf
//!   follow-on, not this primitive: the scores are host-resident here, so the host
//!   row-max plus subtract is the same memory pass the packing already does.
//!
//! The row-max subtraction is **mandatory**: `exp` overflows fast. It maps `xs` into
//! `(-inf, 0]`, the `EXP` LUT's default domain, where the deep tail clamps to ~0
//! (correct). The reciprocal is exact host work over `M` scalars — the reciprocal LUT is
//! the resident-fusion alternative (avoids reading `s` back) and is approximate, so it is
//! unnecessary when `s` is already host-side from the reduce.
//!
//! # LogSoftmax
//!
//! Same front half (steps 1-3), but the tail is a subtract of a per-row scalar instead of
//! a divide:
//!
//! ```text
//!  4. ls[m] = log(s[m])                                HOST (O(M), exact)
//!  5. out[m][n] = xs[m][n] - ls[m]                     NPU  per-row EW subtract
//! ```
//!
//! The `log` is on the host, not the DPU `LOG` LUT: `ls` is over the `M` per-row sums
//! only — already host-side as fp32 after the reduce — so the host log is **exact** and
//! adds no round trip (the DPU LOG LUT is for a large-tensor element-wise log). `s >= 1`
//! (the row-max term contributes `exp(0) = 1`), so `log(s) >= 0` and `out <= 0`
//! everywhere. LogSoftmax is better conditioned than softmax — all-additive, no
//! tiny-probability blow-up — so gate it with **absolute** error.
//!
//! # Cross-entropy
//!
//! `CE[m] = logsumexp_n(logits[m][n]) - logits[m][target[m]]`, the numerically stable
//! form: it never materializes softmax (no divide, no tiny-probability blow-up) and reuses
//! the LogSoftmax front half exactly. Step 4 keeps `rowmax` (the C's `rows_sub_max`
//! optionally records it) and step 5 is a **host gather**: there is no hardware gather on
//! the RK3588 NPU (no indexed/scatter read path — the same fact the attention notes
//! record), so it is `M` scalar index lookups. Only the `M` target log-probs are needed,
//! so the broadcast subtract is skipped: CE is strictly cheaper than LogSoftmax. `loss` is
//! fp32 (a sum can exceed fp16 range, and a loss is conventionally fp32).
//!
//! All three keep the C's `s <= 0` guards and its `INT_MAX` element-count guard.

use half::f16;

use crate::Device;
use crate::error::{Error, Result};
use crate::ops::activation::{Activation, activation, ew_sub};
use crate::ops::norm::scale_rows;
use crate::ops::reduce::reduce_feature;

/// The C's `INT_MAX` element-count guard (see [`crate::ops::reduce`] for why).
fn guard_elems(name: &'static str, elems: usize) -> Result<()> {
    if elems > i32::MAX as usize {
        return Err(Error::shape(name).with_detail(format!("{elems} elements > INT_MAX")));
    }
    Ok(())
}

/// `xs[m][n] = x[m][n] - max_n x[m][n]`, the shift into the EXP LUT's domain.
///
/// The max and the subtract happen in **f32** and only the result narrows, so a row
/// spanning more than fp16's dynamic range still shifts exactly. `rowmax`, when wanted,
/// records the pre-shift maximum (the cross-entropy path needs it for the log-sum-exp).
fn rows_sub_max(m: usize, n: usize, x: &[f16], xs: &mut [f16], mut rowmax: Option<&mut [f32]>) {
    for row in 0..m {
        let xp = &x[row * n..row * n + n];
        let sp = &mut xs[row * n..row * n + n];
        let mut mx = f32::NEG_INFINITY;
        for &v in xp {
            let v = v.to_f32();
            if v > mx {
                mx = v;
            }
        }
        if let Some(rm) = rowmax.as_deref_mut() {
            rm[row] = mx;
        }
        for (s, &v) in sp.iter_mut().zip(xp) {
            *s = f16::from_f32(v.to_f32() - mx);
        }
    }
}

// ── Host references ───────────────────────────────────────────────────────────

/// `rocket_softmax_ref_fp16`: the exact fp64 row softmax.
pub fn softmax_ref(m: usize, n: usize, input: &[f16], out: &mut [f16]) {
    assert!(input.len() >= m * n, "input shorter than M*N");
    assert!(out.len() >= m * n, "output shorter than M*N");
    for row in 0..m {
        let xp = &input[row * n..row * n + n];
        let op = &mut out[row * n..row * n + n];
        let mut mx = f64::NEG_INFINITY;
        for &v in xp {
            let v = v.to_f64();
            if v > mx {
                mx = v;
            }
        }
        let mut s = 0.0f64;
        for &v in xp {
            s += (v.to_f64() - mx).exp();
        }
        let inv = 1.0 / s;
        for (o, &v) in op.iter_mut().zip(xp) {
            *o = f16::from_f64((v.to_f64() - mx).exp() * inv);
        }
    }
}

/// `rocket_logsoftmax_ref_fp16`: the exact fp64 row log-softmax (`out = x - logsumexp`).
pub fn logsoftmax_ref(m: usize, n: usize, input: &[f16], out: &mut [f16]) {
    assert!(input.len() >= m * n, "input shorter than M*N");
    assert!(out.len() >= m * n, "output shorter than M*N");
    for row in 0..m {
        let xp = &input[row * n..row * n + n];
        let op = &mut out[row * n..row * n + n];
        let mut mx = f64::NEG_INFINITY;
        for &v in xp {
            let v = v.to_f64();
            if v > mx {
                mx = v;
            }
        }
        let mut s = 0.0f64;
        for &v in xp {
            s += (v.to_f64() - mx).exp();
        }
        let lse = mx + s.ln(); // logsumexp = mx + log(sum exp(x - mx))
        for (o, &v) in op.iter_mut().zip(xp) {
            *o = f16::from_f64(v.to_f64() - lse);
        }
    }
}

/// `rocket_cross_entropy_ref_fp16`: the exact fp64 log-sum-exp minus the target logit.
///
/// An out-of-range target contributes `0` (the C's `(t >= 0 && t < N)` guard), so the
/// loss is the bare log-sum-exp — never a panic.
pub fn cross_entropy_ref(m: usize, n: usize, logits: &[f16], target: &[i32], loss: &mut [f32]) {
    assert!(logits.len() >= m * n, "logits shorter than M*N");
    assert!(target.len() >= m, "target shorter than M");
    assert!(loss.len() >= m, "loss shorter than M");
    for row in 0..m {
        let xp = &logits[row * n..row * n + n];
        let mut mx = f64::NEG_INFINITY;
        for &v in xp {
            let v = v.to_f64();
            if v > mx {
                mx = v;
            }
        }
        let mut s = 0.0f64;
        for &v in xp {
            s += (v.to_f64() - mx).exp();
        }
        let lse = mx + s.ln();
        let t = target[row];
        let g = if t >= 0 && (t as usize) < n {
            xp[t as usize].to_f64()
        } else {
            0.0
        };
        loss[row] = (lse - g) as f32;
    }
}

// ── Device entry points ───────────────────────────────────────────────────────

/// `rocket_softmax_fp16`: `out[m][n] = exp(x - rowmax) / sum exp(x - rowmax)`.
///
/// One row of the input must not be entirely `-inf` in fp16 terms for a meaningful
/// result, but no shape is refused for it (an all-`-inf` row yields `mx = -inf`,
/// `xs = NaN`, and the C's `s <= 0` guard then returns zeros — same here).
pub fn softmax(dev: &Device, m: usize, n: usize, input: &[f16], out: &mut [f16]) -> Result<()> {
    if m < 1 || n < 1 {
        return Err(Error::shape("rocket_softmax_fp16"));
    }
    let mn = m * n;
    if input.len() < mn || out.len() < mn {
        return Err(Error::shape("rocket_softmax_fp16/buffers"));
    }
    guard_elems("rocket_softmax_fp16", mn)?;

    let out = &mut out[..mn];
    let mut xs = vec![f16::ZERO; mn]; // x - rowmax (<= 0)
    let mut e = vec![f16::ZERO; mn]; // exp(xs)
    let mut s = vec![0.0f32; m];
    let mut inv = vec![0.0f32; m];

    // 1. Host row-max + subtract.
    rows_sub_max(m, n, input, &mut xs, None);
    // 2. e = exp(xs) on the NPU (DPU LUT; the deep tail clamps to ~0).
    activation(dev, Activation::Exp, &xs, &mut e)?;
    // 3. s[m] = sum_n e on the NPU (feature-axis ones-matmul, fp32 accumulate).
    reduce_feature(dev, m, n, &e, &mut s, false)?;
    // 4. inv[m] = 1/s on the host (exact, O(M)); s >= 1, so this cannot divide by zero
    //    for a finite row, but the guard is kept as in C.
    for row in 0..m {
        inv[row] = if s[row] > 0.0 { 1.0 / s[row] } else { 0.0 };
    }
    // 5. out = e * inv (per-row broadcast scale = the EW multiply).
    scale_rows(dev, m, n, &e, &inv, out)
}

/// `rocket_logsoftmax_fp16`: `out[m][n] = x[m][n] - logsumexp_n(x[m][n])`.
pub fn logsoftmax(dev: &Device, m: usize, n: usize, input: &[f16], out: &mut [f16]) -> Result<()> {
    if m < 1 || n < 1 {
        return Err(Error::shape("rocket_logsoftmax_fp16"));
    }
    let mn = m * n;
    if input.len() < mn || out.len() < mn {
        return Err(Error::shape("rocket_logsoftmax_fp16/buffers"));
    }
    guard_elems("rocket_logsoftmax_fp16", mn)?;

    let out = &mut out[..mn];
    let mut xs = vec![f16::ZERO; mn]; // x - rowmax (<= 0)
    let mut e = vec![f16::ZERO; mn]; // exp(xs)
    let mut lsb = vec![f16::ZERO; mn]; // log(s[m]) broadcast over n
    let mut s = vec![0.0f32; m];

    rows_sub_max(m, n, input, &mut xs, None);
    activation(dev, Activation::Exp, &xs, &mut e)?;
    reduce_feature(dev, m, n, &e, &mut s, false)?;

    // 4. ls[m] = log(s[m]) on the host (exact, O(M); s >= 1 => ls >= 0), broadcast across n.
    for row in 0..m {
        let ls = if s[row] > 0.0 { s[row].ln() } else { 0.0 };
        let lv = f16::from_f32(ls);
        lsb[row * n..row * n + n].fill(lv);
    }
    // 5. out = xs - ls on the NPU (per-row broadcast subtract = EW subtract).
    ew_sub(dev, &xs, &lsb, out)
}

/// `rocket_cross_entropy_fp16`: the stable per-row NLL against an integer class target.
///
/// Returns `f32` per row (`logsumexp - logits[target]`), always `>= 0` for a finite row.
pub fn cross_entropy(
    dev: &Device,
    m: usize,
    n: usize,
    logits: &[f16],
    target: &[i32],
    loss: &mut [f32],
) -> Result<()> {
    if m < 1 || n < 1 {
        return Err(Error::shape("rocket_cross_entropy_fp16"));
    }
    let mn = m * n;
    if logits.len() < mn || target.len() < m || loss.len() < m {
        return Err(Error::shape("rocket_cross_entropy_fp16/buffers"));
    }
    guard_elems("rocket_cross_entropy_fp16", mn)?;

    let mut xs = vec![f16::ZERO; mn]; // logits - rowmax (<= 0)
    let mut e = vec![f16::ZERO; mn]; // exp(xs)
    let mut s = vec![0.0f32; m]; // row sums of e (NPU)
    let mut rmx = vec![0.0f32; m]; // per-row max (kept for lse)

    // 1. Host row-max + subtract, keeping rowmax for the log-sum-exp.
    rows_sub_max(m, n, logits, &mut xs, Some(&mut rmx));
    // 2. e = exp(xs) on the NPU, 3. s[m] = sum_n e (NPU fp32 reduce).
    activation(dev, Activation::Exp, &xs, &mut e)?;
    reduce_feature(dev, m, n, &e, &mut s, false)?;

    // 4. lse[m] = rowmax + log(s) (host, exact); 5. CE = lse - logits[target] (host gather).
    for row in 0..m {
        let lse = rmx[row] + if s[row] > 0.0 { s[row].ln() } else { 0.0 };
        let t = target[row];
        let g = if t >= 0 && (t as usize) < n {
            logits[row * n + t as usize].to_f32()
        } else {
            0.0
        };
        loss[row] = lse - g;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(values: &[f32]) -> Vec<f16> {
        values.iter().map(|v| f16::from_f32(*v)).collect()
    }

    fn sum_row(v: &[f16]) -> f32 {
        v.iter().map(|x| x.to_f32()).sum()
    }

    #[test]
    fn softmax_reference_on_known_rows() {
        // Uniform row -> uniform distribution.
        let x = row(&[3.0, 3.0, 3.0, 3.0]);
        let mut out = vec![f16::ZERO; 4];
        softmax_ref(1, 4, &x, &mut out);
        for v in &out {
            assert!((v.to_f32() - 0.25).abs() < 1e-3, "{v}");
        }
        assert!((sum_row(&out) - 1.0).abs() < 1e-3);

        // One-hot-ish row: a large gap dominates.
        let x = row(&[10.0, 0.0, 0.0, 0.0]);
        softmax_ref(1, 4, &x, &mut out);
        assert!(out[0].to_f32() > 0.999);
        assert!(out[1].to_f32() < 1e-3);

        // Large offsets must not change the result (the row-max shift is exact in f64).
        let a = row(&[0.0, 1.0, 2.0]);
        let b = row(&[500.0, 501.0, 502.0]);
        let mut oa = vec![f16::ZERO; 3];
        let mut ob = vec![f16::ZERO; 3];
        softmax_ref(1, 3, &a, &mut oa);
        softmax_ref(1, 3, &b, &mut ob);
        for (u, v) in oa.iter().zip(&ob) {
            assert_eq!(u.to_bits(), v.to_bits());
        }

        // Two rows are independent, and every row sums to 1.
        let x = row(&[1.0, 2.0, 3.0, -1.0, -2.0, -3.0]);
        let mut out = vec![f16::ZERO; 6];
        softmax_ref(2, 3, &x, &mut out);
        for r in 0..2 {
            assert!((sum_row(&out[r * 3..r * 3 + 3]) - 1.0).abs() < 2e-3);
        }
        assert!(out[2] > out[1] && out[1] > out[0]);
        assert!(out[3] > out[4] && out[4] > out[5]);
    }

    #[test]
    fn logsoftmax_reference_is_additive_and_normalized() {
        let x = row(&[1.0, 2.0, 3.0]);
        let mut out = vec![f16::ZERO; 3];
        logsoftmax_ref(1, 3, &x, &mut out);
        // out = x - logsumexp, so exp(out) sums to 1 and everything is <= 0.
        assert!(out.iter().all(|v| v.to_f32() <= 0.0));
        let s: f32 = out.iter().map(|v| v.to_f32().exp()).sum();
        assert!((s - 1.0).abs() < 3e-3, "{s}");
        // A uniform row is -ln(N) everywhere.
        let u = row(&[2.5, 2.5, 2.5, 2.5]);
        let mut ou = vec![f16::ZERO; 4];
        logsoftmax_ref(1, 4, &u, &mut ou);
        let want = (4.0f32).ln();
        for v in &ou {
            assert!((v.to_f32() + want).abs() < 1e-3, "{v}");
        }
        // A large offset shifts every output by the same amount.
        let mut big = vec![f16::ZERO; 3];
        let shifted = row(&[101.0, 102.0, 103.0]);
        logsoftmax_ref(1, 3, &shifted, &mut big);
        for (a, b) in out.iter().zip(&big) {
            assert!((a.to_f32() - b.to_f32()).abs() < 1e-2);
        }
    }

    #[test]
    fn cross_entropy_reference_matches_logsoftmax() {
        let x = row(&[0.5, -1.0, 2.0, 2.0]);
        let mut ls = vec![f16::ZERO; 4];
        logsoftmax_ref(1, 4, &x, &mut ls);
        let mut loss = [0.0f32; 1];
        for t in 0..4i32 {
            cross_entropy_ref(1, 4, &x, &[t], &mut loss);
            // CE[t] == -logsoftmax[t], up to the fp16 storage of the reference.
            assert!(
                (loss[0] + ls[t as usize].to_f32()).abs() < 2e-3,
                "t={t}: {} vs {}",
                loss[0],
                -ls[t as usize].to_f32()
            );
            assert!(loss[0] >= 0.0);
        }
        // An out-of-range target contributes no logit, so CE == logsumexp.
        cross_entropy_ref(1, 4, &x, &[-1], &mut loss);
        let lse = {
            let mx = 2.0f64;
            let s: f64 = x.iter().map(|v| (v.to_f64() - mx).exp()).sum();
            (mx + s.ln()) as f32
        };
        assert!((loss[0] - lse).abs() < 1e-5);
        cross_entropy_ref(1, 4, &x, &[4], &mut loss);
        assert!((loss[0] - lse).abs() < 1e-5);

        // Two rows, per-row targets.
        let y = row(&[1.0, 1.0, 10.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
        let mut l2 = [0.0f32; 2];
        cross_entropy_ref(2, 4, &y, &[2, 0], &mut l2);
        assert!(l2[0] < l2[1]);
        assert!(l2.iter().all(|v| *v >= 0.0));
    }

    #[test]
    fn rows_sub_max_is_exact_in_f32() {
        // A row spanning more than fp16's range still shifts exactly: the max and the
        // subtract are f32, only the stored value narrows (and saturates, which is the
        // correct answer for the LUT: exp(-inf) = 0).
        let x = row(&[60000.0, -60000.0, 0.0]);
        let mut xs = vec![f16::ZERO; 3];
        let mut rm = [0.0f32; 1];
        rows_sub_max(1, 3, &x, &mut xs, Some(&mut rm));
        assert_eq!(rm[0], 60000.0);
        assert_eq!(xs[0].to_f32(), 0.0);
        assert_eq!(xs[1].to_f32(), f32::NEG_INFINITY);
        assert_eq!(xs[2].to_f32(), -60000.0);
        // Without a rowmax slot it still shifts.
        let mut only = vec![f16::ZERO; 3];
        rows_sub_max(1, 3, &x, &mut only, None);
        assert_eq!(only, xs);
    }
}
