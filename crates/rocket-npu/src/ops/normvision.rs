//! The vision normalization family (RK3588): BatchNorm, GroupNorm, InstanceNorm and
//! L2-Normalize, built on the same primitives as the transformer norms.
//!
//! Port of `rocket_normvision.c` + `rocket_normvision.h`. Like [`super::norm`] these are
//! memory-bound reduce+elementwise ops: standalone on the NPU they are *submit*-bound and
//! a host single memory pass is faster. They exist to be fused into a resident NCHW-cube
//! block (so the activation never leaves device layout between two NPU convs/matmuls —
//! the dominant not-MAC-bound cost), and as op coverage so a delegate need not spill the
//! node to the CPU.
//!
//! # Layout
//!
//! All take channels-major NCHW-style buffers, `P = H*W` = the spatial count per channel
//! (`P = 1` for a pure `[N,C]` tensor). The reduce axis differs per op (that is the only
//! thing that distinguishes them):
//!
//! | op | reduction |
//! |---|---|
//! | BatchNorm | none — per-channel affine from **precomputed running stats** |
//! | InstanceNorm | `[H,W]` per `(n,c)` == GroupNorm with `G = C` |
//! | GroupNorm | `[C/G, H, W]` per `(n,g)` |
//! | L2-Normalize | `x²` over the last axis per row |
//!
//! # Precision
//!
//! The square branch uses the exact power-of-two prescale from [`super::norm`]
//! (`|x| > ~223` ⇒ `x²` overflows fp16); the reduce accumulates in fp32 (genuine, and
//! what matters for the variance); the per-group mean/var/rsqrt `O(rows)` tail is exact
//! fp32 host scalar work; the affine narrows in fp16. Gate with a **relative** tolerance,
//! not equality (fp16 rounding + reciprocal). The `*_ref` entries are the exact fp64 host
//! references — the same values the C's `_ref_fp16` oracles return.
//!
//! One deliberate hardening over the C: the element count handed to the flat EW entries
//! is guarded against `INT_MAX` in every entry here, where the C only guards some of them
//! (`rocket_batchnorm_fp16`/`_groupnorm_fp16`/`_l2norm_fp16` would truncate silently and
//! under-process the surface). The guard can only turn a silently partial result into a
//! [`crate::error::Status::Shape`] error.

use half::f16;

use crate::Device;
use crate::error::{Error, Result};
use crate::ops::activation::{ew_add, ew_mul};
use crate::ops::norm::{scale_rows, square_prescale_k};
use crate::ops::reduce::reduce_feature;

/// `2^k` for the small exponents the prescale uses.
fn pow2(k: i32) -> f32 {
    f32::from_bits(((127 + k) as u32) << 23)
}

fn guard_elems(name: &'static str, elems: usize) -> Result<()> {
    if elems > i32::MAX as usize {
        return Err(Error::shape(name).with_detail(format!("{elems} elements > INT_MAX")));
    }
    Ok(())
}

// ── BatchNorm (inference) ─────────────────────────────────────────────────────

/// `rocket_batchnorm_ref_fp16`: the exact fp64 reference of [`batchnorm`].
///
/// No reduction happens in inference BN — the per-channel affine is folded from the
/// **stored** running `mean`/`var` (`s[c] = gamma[c]/sqrt(var[c]+eps)`,
/// `b[c] = beta[c] − mean[c]·s[c]`) and applied to every spatial position, in f64.
#[allow(clippy::too_many_arguments)]
pub fn batchnorm_ref(
    n: usize,
    c: usize,
    p: usize,
    x: &[f16],
    gamma: Option<&[f16]>,
    beta: Option<&[f16]>,
    mean: &[f16],
    var: &[f16],
    eps: f32,
    out: &mut [f16],
) {
    let total = n * c * p;
    assert!(x.len() >= total, "x shorter than N*C*P");
    assert!(mean.len() >= c, "mean shorter than C");
    assert!(var.len() >= c, "var shorter than C");
    assert!(gamma.is_none_or(|g| g.len() >= c), "gamma shorter than C");
    assert!(beta.is_none_or(|b| b.len() >= c), "beta shorter than C");
    assert!(out.len() >= total, "output shorter than N*C*P");
    for ch in 0..c {
        let g = gamma.map_or(1.0, |g| g[ch].to_f64());
        let b = beta.map_or(0.0, |b| b[ch].to_f64());
        let s = g / (var[ch].to_f64() + f64::from(eps)).sqrt();
        let bc = b - mean[ch].to_f64() * s;
        for nn in 0..n {
            let off = (nn * c + ch) * p;
            for i in 0..p {
                out[off + i] = f16::from_f64(x[off + i].to_f64() * s + bc);
            }
        }
    }
}

/// `rocket_batchnorm_fp16`: inference BatchNorm over `[N,C,P]` from running statistics.
///
/// ```text
/// out[n][c][p] = (x[n][c][p] − mean[c]) / sqrt(var[c] + eps) · gamma[c] + beta[c]
/// ```
///
/// `mean`/`var` are the per-channel running stats `[C]` (required); `gamma` `[C]` may be
/// `None` (⇒ 1) and `beta` `[C]` may be `None` (⇒ 0). The affine is folded to
/// `out = x·s[c] + b[c]` and run as one NPU EW multiply + one EW add over the broadcast
/// per-channel tensors, exactly as the C does.
#[allow(clippy::too_many_arguments)]
pub fn batchnorm(
    dev: &Device,
    n: usize,
    c: usize,
    p: usize,
    x: &[f16],
    gamma: Option<&[f16]>,
    beta: Option<&[f16]>,
    mean: &[f16],
    var: &[f16],
    eps: f32,
    out: &mut [f16],
) -> Result<()> {
    let total = n * c * p;
    if n < 1 || c < 1 || p < 1 {
        return Err(Error::shape("rocket_batchnorm_fp16"));
    }
    if x.len() < total
        || out.len() < total
        || mean.len() < c
        || var.len() < c
        || gamma.is_some_and(|g| g.len() < c)
        || beta.is_some_and(|b| b.len() < c)
    {
        return Err(Error::shape("rocket_batchnorm_fp16/buffers"));
    }
    guard_elems("rocket_batchnorm_fp16", total)?;

    let x = &x[..total];
    let out = &mut out[..total];
    let mut a_buf = vec![f16::ZERO; total];
    let mut b_buf = vec![f16::ZERO; total];
    for nn in 0..n {
        for ch in 0..c {
            let g = gamma.map_or(1.0, |g| g[ch].to_f32());
            let bt = beta.map_or(0.0, |b| b[ch].to_f32());
            let s = g / (var[ch].to_f32() + eps).sqrt();
            let bc = bt - mean[ch].to_f32() * s;
            let (sv, bv) = (f16::from_f32(s), f16::from_f32(bc));
            let off = (nn * c + ch) * p;
            a_buf[off..off + p].fill(sv);
            b_buf[off..off + p].fill(bv);
        }
    }

    let mut tmp = vec![f16::ZERO; total];
    ew_mul(dev, x, &a_buf, &mut tmp)?;
    ew_add(dev, &tmp, &b_buf, out)
}

// ── GroupNorm ─────────────────────────────────────────────────────────────────

/// `rocket_groupnorm_ref_fp16`: the exact fp64 reference of [`groupnorm`].
#[allow(clippy::too_many_arguments)]
pub fn groupnorm_ref(
    n: usize,
    c: usize,
    g: usize,
    p: usize,
    x: &[f16],
    gamma: Option<&[f16]>,
    beta: Option<&[f16]>,
    eps: f32,
    out: &mut [f16],
) {
    let total = n * c * p;
    assert!(x.len() >= total, "x shorter than N*C*P");
    assert!(out.len() >= total, "output shorter than N*C*P");
    assert!(g > 0 && c.is_multiple_of(g), "G must divide C");
    let cg = c / g;
    let inv_pg = 1.0 / (cg as f64 * p as f64);
    for nn in 0..n {
        for gg in 0..g {
            // mean/var over the group's Cg channels × P spatial positions
            let (mut sx, mut sx2) = (0.0f64, 0.0f64);
            for cc in 0..cg {
                let off = (nn * c + gg * cg + cc) * p;
                for i in 0..p {
                    let v = x[off + i].to_f64();
                    sx += v;
                    sx2 += v * v;
                }
            }
            let mean = sx * inv_pg;
            let mut var = sx2 * inv_pg - mean * mean;
            if var < 0.0 {
                var = 0.0;
            }
            let r = 1.0 / (var + f64::from(eps)).sqrt();
            for cc in 0..cg {
                let ch = gg * cg + cc;
                let gm = gamma.map_or(1.0, |g| g[ch].to_f64());
                let bt = beta.map_or(0.0, |b| b[ch].to_f64());
                let off = (nn * c + ch) * p;
                for i in 0..p {
                    out[off + i] = f16::from_f64((x[off + i].to_f64() - mean) * r * gm + bt);
                }
            }
        }
    }
}

/// `rocket_groupnorm_fp16`: GroupNorm over `[N,C,P]` with `G` groups (`C % G == 0`).
///
/// ```text
/// mean[n][g], var[n][g]  over { x[n][c][p] : c in group g, all p }
/// out[n][c][p] = (x[n][c][p] − mean[n][g]) / sqrt(var[n][g] + eps) · gamma[c] + beta[c]
/// ```
///
/// A group's elements are contiguous in `[N,C,P]` (channels-major), so the per-`(n,g)`
/// reduce is a `[N*G, (C/G)*P]` feature-axis reduce with `[x ; x²]` **stacked in ONE job**,
/// exactly like LayerNorm. The affine is per-**channel**, so it varies within a group's
/// row and is folded to `out = x·A + B` with a full broadcast (one EW multiply + one EW
/// add). `gamma`/`beta` are `[C]` (`None` ⇒ 1 / 0).
#[allow(clippy::too_many_arguments)]
pub fn groupnorm(
    dev: &Device,
    n: usize,
    c: usize,
    g: usize,
    p: usize,
    x: &[f16],
    gamma: Option<&[f16]>,
    beta: Option<&[f16]>,
    eps: f32,
    out: &mut [f16],
) -> Result<()> {
    let total = n * c * p;
    if n < 1 || c < 1 || p < 1 || g < 1 || !c.is_multiple_of(g) {
        return Err(Error::shape("rocket_groupnorm_fp16"));
    }
    if x.len() < total
        || out.len() < total
        || gamma.is_some_and(|g| g.len() < c)
        || beta.is_some_and(|b| b.len() < c)
    {
        return Err(Error::shape("rocket_groupnorm_fp16/buffers"));
    }
    guard_elems("rocket_groupnorm_fp16", total)?;

    let x = &x[..total];
    let out = &mut out[..total];
    let cg = c / g;
    let m = n * g; // one row per (batch, group)
    let pg = cg * p; // elements reduced per group row
    if pg > i32::MAX as usize {
        return Err(Error::shape("rocket_groupnorm_fp16/row"));
    }

    // fp16-square overflow prescale for the x² branch (the mean(x) branch uses x directly)
    let k = square_prescale_k(x);
    let (p_scale, four_k) = (pow2(-k), pow2(2 * k));
    let xs;
    let sq_in: &[f16] = if k > 0 {
        xs = x
            .iter()
            .map(|v| f16::from_f32(v.to_f32() * p_scale))
            .collect::<Vec<_>>();
        &xs
    } else {
        x
    };

    // 1. sq = xs ⊙ xs (NPU). 2. ONE stacked feature-reduce over Pg: [x ; sq] as 2*M rows.
    //    The group's Pg elements are contiguous in [N,C,P], so x viewed as [M][Pg] needs
    //    no reorder.
    let mut sq = vec![f16::ZERO; total];
    ew_mul(dev, sq_in, sq_in, &mut sq)?;
    let mut stack = Vec::with_capacity(2 * total);
    stack.extend_from_slice(x);
    stack.extend_from_slice(&sq);
    let mut csum = vec![0.0f32; 2 * m];
    reduce_feature(dev, 2 * m, pg, &stack, &mut csum, false)?;

    // 3. per-(n,g) mean/var/rsqrt (host, exact). sum(x²) = csum[M+r]·4^k.
    let inv_pg = 1.0f32 / pg as f32;
    let mut meanv = vec![0.0f32; m];
    let mut rinv = vec![0.0f32; m];
    for row in 0..m {
        let mn = csum[row] * inv_pg;
        let msq = csum[m + row] * four_k * inv_pg;
        let mut var = msq - mn * mn;
        if var < 0.0 {
            var = 0.0;
        }
        meanv[row] = mn;
        rinv[row] = 1.0 / (var + eps).sqrt();
    }

    // 4. per-CHANNEL affine fold: out = x ⊙ A + B, A[n,c,*] = rinv[n,g]·gamma[c],
    //    B[n,c,*] = beta[c] − mean[n,g]·A (broadcast across the channel's P positions).
    let mut a_buf = vec![f16::ZERO; total];
    let mut b_buf = vec![f16::ZERO; total];
    for nn in 0..n {
        for gg in 0..g {
            let row = nn * g + gg;
            let (rm, mm) = (rinv[row], meanv[row]);
            for cc in 0..cg {
                let ch = gg * cg + cc;
                let gm = gamma.map_or(1.0, |g| g[ch].to_f32());
                let a = rm * gm;
                let b = beta.map_or(0.0, |b| b[ch].to_f32()) - mm * a;
                let (av, bv) = (f16::from_f32(a), f16::from_f32(b));
                let off = (nn * c + ch) * p;
                a_buf[off..off + p].fill(av);
                b_buf[off..off + p].fill(bv);
            }
        }
    }

    // 5. tmp = x ⊙ A ; out = tmp + B (two NPU EW passes).
    let mut tmp = vec![f16::ZERO; total];
    ew_mul(dev, x, &a_buf, &mut tmp)?;
    ew_add(dev, &tmp, &b_buf, out)
}

// ── InstanceNorm (GroupNorm with G = C) ───────────────────────────────────────

/// `rocket_instancenorm_ref_fp16`: InstanceNorm is GroupNorm with `G = C`.
#[allow(clippy::too_many_arguments)]
pub fn instancenorm_ref(
    n: usize,
    c: usize,
    p: usize,
    x: &[f16],
    gamma: Option<&[f16]>,
    beta: Option<&[f16]>,
    eps: f32,
    out: &mut [f16],
) {
    groupnorm_ref(n, c, c, p, x, gamma, beta, eps, out);
}

/// `rocket_instancenorm_fp16`: normalize each `(n,c)` over its `P` spatial positions.
///
/// Exactly [`groupnorm`] with `G = C` (every channel its own group), as in C.
#[allow(clippy::too_many_arguments)]
pub fn instancenorm(
    dev: &Device,
    n: usize,
    c: usize,
    p: usize,
    x: &[f16],
    gamma: Option<&[f16]>,
    beta: Option<&[f16]>,
    eps: f32,
    out: &mut [f16],
) -> Result<()> {
    groupnorm(dev, n, c, c, p, x, gamma, beta, eps, out)
}

// ── L2-Normalize ──────────────────────────────────────────────────────────────

/// `rocket_l2norm_ref_fp16`: the exact fp64 reference of [`l2norm`].
pub fn l2norm_ref(m: usize, h: usize, x: &[f16], eps: f32, out: &mut [f16]) {
    assert!(x.len() >= m * h, "x shorter than M*H");
    assert!(out.len() >= m * h, "output shorter than M*H");
    for row in 0..m {
        let xp = &x[row * h..row * h + h];
        let mut ss = 0.0f64;
        for &v in xp {
            let v = v.to_f64();
            ss += v * v;
        }
        let r = 1.0 / (ss + f64::from(eps)).sqrt();
        let op = &mut out[row * h..row * h + h];
        for (o, &v) in op.iter_mut().zip(xp) {
            *o = f16::from_f64(v.to_f64() * r);
        }
    }
}

/// `rocket_l2norm_fp16`: L2-normalize each row of an `[M,H]` tensor over its `H` elements.
///
/// ```text
/// out[m][h] = x[m][h] / sqrt(sum_h x[m][h]² + eps)
/// ```
///
/// The TFLite `L2_NORMALIZATION` / ONNX `LpNormalization p=2` form: `eps` is an additive
/// guard **inside** the sqrt (pass a small value, e.g. 1e-12, to avoid a divide by zero
/// on a zero row — the reference divides by `sqrt(eps)` there too).
///
/// Chain, exactly as the C: `sq = x ⊙ x` (NPU, with the fp16 overflow prescale) →
/// `ss[m] = sum_h sq` (NPU fp32 reduce) → `r[m] = 1/sqrt(ss·4^k + eps)` (host) →
/// `out = scale_rows(x, r)` (NPU per-row broadcast).
pub fn l2norm(
    dev: &Device,
    m: usize,
    h: usize,
    x: &[f16],
    eps: f32,
    out: &mut [f16],
) -> Result<()> {
    let mh = m * h;
    if m < 1 || h < 1 {
        return Err(Error::shape("rocket_l2norm_fp16"));
    }
    if x.len() < mh || out.len() < mh {
        return Err(Error::shape("rocket_l2norm_fp16/buffers"));
    }
    guard_elems("rocket_l2norm_fp16", mh)?;

    let x = &x[..mh];
    let out = &mut out[..mh];

    let k = square_prescale_k(x);
    let (p_scale, four_k) = (pow2(-k), pow2(2 * k));
    let xs;
    let sq_in: &[f16] = if k > 0 {
        xs = x
            .iter()
            .map(|v| f16::from_f32(v.to_f32() * p_scale))
            .collect::<Vec<_>>();
        &xs
    } else {
        x
    };

    // sq = xs ⊙ xs (NPU) -> ss[m] = sum_h sq (NPU fp32 reduce); true ss = ss·4^k.
    let mut sq = vec![f16::ZERO; mh];
    ew_mul(dev, sq_in, sq_in, &mut sq)?;
    let mut ss = vec![0.0f32; m];
    reduce_feature(dev, m, h, &sq, &mut ss, false)?;
    let mut r = vec![0.0f32; m];
    for row in 0..m {
        r[row] = 1.0 / (ss[row] * four_k + eps).sqrt();
    }

    // out = x scaled per-row by r (NPU per-row broadcast multiply).
    scale_rows(dev, m, h, x, &r, out)
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
    fn batchnorm_reference_folds_the_running_stats() {
        // N=1 C=2 P=2, x=[1,2 | 3,4], mean=[1,2], var=[1,4], eps=0, gamma=[2,1],
        // beta=[0,1].
        //   c0: s = 2/1 = 2, b = 0 - 1*2 = -2      -> [0, 2]
        //   c1: s = 1/2 = 0.5, b = 1 - 2*0.5 = 0   -> [1.5, 2]
        let x = vals(&[1.0, 2.0, 3.0, 4.0]);
        let mean = vals(&[1.0, 2.0]);
        let var = vals(&[1.0, 4.0]);
        let g = vals(&[2.0, 1.0]);
        let b = vals(&[0.0, 1.0]);
        let mut out = vec![f16::ZERO; 4];
        batchnorm_ref(1, 2, 2, &x, Some(&g), Some(&b), &mean, &var, 0.0, &mut out);
        assert_eq!(out, vals(&[0.0, 2.0, 1.5, 2.0]));

        // gamma/beta NULL => 1/0 (the identity affine form).
        let mut out = vec![f16::ZERO; 4];
        batchnorm_ref(1, 2, 2, &x, None, None, &mean, &var, 0.0, &mut out);
        assert_eq!(out, vals(&[0.0, 1.0, 0.5, 1.0]));

        // The two batch entries share the per-channel stats (N=2).
        let x = vals(&[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]);
        let mut out = vec![f16::ZERO; 8];
        batchnorm_ref(2, 2, 2, &x, None, None, &mean, &var, 0.0, &mut out);
        assert_eq!(out, vals(&[0.0, 1.0, 0.5, 1.0, 0.0, 1.0, 0.5, 1.0]));

        // eps widens the denominator: var=1, eps=3 -> s = 1/2.
        let mut out = vec![f16::ZERO; 4];
        batchnorm_ref(
            1,
            2,
            2,
            &x[..4],
            None,
            None,
            &vals(&[0.0, 0.0]),
            &vals(&[1.0, 1.0]),
            3.0,
            &mut out,
        );
        assert_eq!(out, vals(&[0.5, 1.0, 1.5, 2.0]));
    }

    #[test]
    fn groupnorm_reference_reduces_the_whole_group() {
        // N=1 C=4 G=2 P=2: group 0 = [1,2,3,4] (mean 2.5, var 1.25), group 1 = [1,1,1,1]
        // (mean 1, var 0).
        let x = vals(&[1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 1.0, 1.0]);
        let mut out = vec![f16::ZERO; 8];
        groupnorm_ref(1, 4, 2, 2, &x, None, None, 1e-5, &mut out);
        let r = 1.0 / 1.25f64.sqrt(); // 0.8944
        let want = [
            (1.0 - 2.5) * r,
            (2.0 - 2.5) * r,
            (3.0 - 2.5) * r,
            (4.0 - 2.5) * r,
            0.0,
            0.0,
            0.0,
            0.0,
        ];
        for (o, w) in out.iter().zip(want) {
            assert!((o.to_f64() - w).abs() < 2e-3, "{o} vs {w}");
        }
        assert!(f64s(&out)[4..].iter().all(|v| v.abs() < 1e-6));

        // Per-channel gamma/beta (the affine varies WITHIN a group row). The layout is
        // [C][P], so the channel of flat element i is i / P.
        let g = vals(&[1.0, 2.0, 3.0, 4.0]);
        let b = vals(&[0.0, 1.0, 0.0, -1.0]);
        groupnorm_ref(1, 4, 2, 2, &x, Some(&g), Some(&b), 1e-5, &mut out);
        for i in 0..4 {
            let ch = i / 2;
            let want = want[i] * f64::from([1.0f32, 2.0, 3.0, 4.0][ch])
                + f64::from([0.0f32, 1.0, 0.0, -1.0][ch]);
            assert!(
                (out[i].to_f64() - want).abs() < 2e-2,
                "{i}: {} vs {want}",
                out[i]
            );
        }
        // Group 1 is constant, so it maps to beta alone (plus gamma·0): the variance is
        // 0 and only a non-zero eps keeps 0·(1/sqrt(eps)) finite.
        // Elements 4/5 are channel 2 (beta 0) and 6/7 channel 3 (beta -1).
        assert!((out[4].to_f64() - 0.0).abs() < 1e-6);
        assert!((out[5].to_f64() - 0.0).abs() < 1e-6);
        assert!((out[6].to_f64() + 1.0).abs() < 1e-6);
        assert!((out[7].to_f64() + 1.0).abs() < 1e-6);
    }

    #[test]
    fn instancenorm_reference_is_groupnorm_with_g_equal_c() {
        // N=1 C=2 P=2, x=[1,3 | 2,2]:  c0 mean 2 var 1 -> [-1, 1]; c1 var 0 -> [0, 0].
        let x = vals(&[1.0, 3.0, 2.0, 2.0]);
        let mut a = vec![f16::ZERO; 4];
        let mut b = vec![f16::ZERO; 4];
        instancenorm_ref(1, 2, 2, &x, None, None, 1e-5, &mut a);
        groupnorm_ref(1, 2, 2, 2, &x, None, None, 1e-5, &mut b);
        assert_eq!(a, b);
        assert_eq!(a, vals(&[-1.0, 1.0, 0.0, 0.0]));

        // gamma/beta broadcast per channel.
        let g = vals(&[10.0, 1.0]);
        let bt = vals(&[5.0, -5.0]);
        instancenorm_ref(1, 2, 2, &x, Some(&g), Some(&bt), 1e-5, &mut a);
        assert_eq!(a, vals(&[-5.0, 15.0, -5.0, -5.0]));
    }

    #[test]
    fn l2norm_reference_divides_by_the_row_norm() {
        // [3,4] -> 5; [1,0,-1,0] -> √2; a zero row divides by sqrt(eps).
        let x = vals(&[3.0, 4.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0]);
        let mut out = vec![f16::ZERO; 8];
        l2norm_ref(3, 2, &x, 0.0, &mut out);
        let got = f64s(&out);
        assert!((got[0] - 0.6).abs() < 1e-3);
        assert!((got[1] - 0.8).abs() < 1e-3);
        // [1, 0] and [-1, 0] already have unit norm, so they come back unchanged.
        assert_eq!(got[2], 1.0);
        assert_eq!(got[3], 0.0);
        assert_eq!(got[4], -1.0);
        assert_eq!(got[5], 0.0);
        assert_eq!(got[6], 0.0);
        assert_eq!(got[7], 0.0);

        // eps keeps a zero row finite (and zero).
        l2norm_ref(1, 2, &vals(&[0.0, 0.0]), 1e-12, &mut out);
        assert_eq!(out[0], f16::ZERO);
        assert_eq!(out[1], f16::ZERO);

        // A uniform row normalizes to 1/sqrt(H) each.
        l2norm_ref(1, 4, &vals(&[2.0; 4]), 0.0, &mut out);
        for o in &out[..4] {
            assert!((o.to_f64() - 0.5).abs() < 1e-3);
        }
    }

    #[test]
    fn buffers_are_validated() {
        let dev = match Device::open() {
            Ok(d) => d,
            Err(_) => return, // no device on this host; the refs are covered above
        };
        let x = vec![f16::ZERO; 8];
        let c4 = vec![f16::ONE; 4];
        let c2 = vec![f16::ONE; 2];
        let mut out = vec![f16::ZERO; 8];
        assert!(batchnorm(&dev, 0, 2, 2, &x, None, None, &c2, &c2, 0.0, &mut out).is_err());
        assert!(batchnorm(&dev, 1, 2, 2, &x, None, None, &c2[..1], &c2, 0.0, &mut out).is_err());
        assert!(groupnorm(&dev, 1, 4, 3, 2, &x, None, None, 0.0, &mut out).is_err());
        assert!(groupnorm(&dev, 1, 4, 2, 0, &x, None, None, 0.0, &mut out).is_err());
        assert!(groupnorm(&dev, 1, 4, 2, 2, &x, Some(&c2), None, 0.0, &mut out).is_err());
        assert!(instancenorm(&dev, 0, 2, 2, &x, None, None, 0.0, &mut out).is_err());
        assert!(l2norm(&dev, 0, 4, &x, 0.0, &mut out).is_err());
        assert!(l2norm(&dev, 4, 4, &x, 0.0, &mut out).is_err());
        // Unused when the shapes above fail before touching the device.
        let _ = &c4;
    }
}
