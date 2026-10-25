//! One Whisper/transformer encoder block (RK3588) — the M7 capstone.
//!
//! Port of `rocket_encoder.c` + `rocket_encoder.h`. Whisper's **pre-norm residual**
//! block, assembled from the already-validated primitives:
//!
//! ```text
//! x  = x + MHA( LayerNorm1(x) )                 (attention sublayer)
//! x  = x + MLP( LayerNorm2(x) )                 (feed-forward sublayer)
//! MLP(h) = GELU(h·Wf1^T + bf1)·Wf2^T + bf2      (d -> d_ff -> d)
//! ```
//!
//! The device path runs **fully on the NPU**: both LayerNorms
//! ([`super::norm::layernorm`]), attention ([`crate::attn::mha_self`], the C
//! `rocket_mha_self_fp16` with its own in-sublayer softmax), the residual adds
//! ([`super::activation::ew_add`]), the MLP ([`super::ffn::mlp_fused`]: pinned fc1/fc2,
//! the bias add and the 2-pass GELU over the intermediate) and the second bias add. The
//! only host work is `O(T·d)` glue (row padding, the bias adds, the final residual).
//!
//! All weights are row-major `[out,in]` (PyTorch `nn.Linear`); biases are optional
//! (`None` = none — Whisper's attention has `bq`, `bv`, `bo` and no `bk`).
//!
//! # Order, padding and the host/NPU split
//!
//! Following the C exactly: the row count is padded to `Tp = (T+3) & ~3` for the MLP's
//! matmuls (the matmul `M % 4` requirement), the padded rows are **zero** and are
//! matmul-of-zero — never read out. The LayerNorm and the attention run over the real
//! `T` rows; the MLP runs over `Tp`. The second bias add and the residual run over `T`.
//!
//! `ROCKET_ENCODER_NOFUSE` forces the C's host-handoff MLP (fc1 → host bias → GELU gate +
//! EW multiply → fc2, each op de-tiling in between) instead of the fused form; the C uses
//! that knob for A/B, and the two differ only in the down matmul's K-tiling
//! ([`super::ffn::mlp_fused`] pins it).

use half::f16;

use crate::Device;
use crate::error::{Error, Result};
use crate::knobs;
use crate::matmul::one_shot;
use crate::ops::activation::{Activation, activation, ew_add, ew_mul};
use crate::ops::ffn::mlp_fused;
use crate::ops::norm::{layernorm, layernorm_ref};

/// The block's weight tensors, all row-major fp16 (`[out,in]` for the projections, `[d]`
/// for the LayerNorm affines and the biases).
///
/// The field names follow the C argument order (`ln1_g`, `Wq`, `bq`, …) so a call can be
/// checked against `rocket_encoder_block_fp16` line by line. `ln1_b`/`ln2_b` and every
/// bias are optional (`None` = no bias), matching the C's `NULL` arguments; `ln1_g` and
/// `ln2_g` are required, as in `rocket_layernorm_fp16`.
#[derive(Debug, Clone, Copy)]
pub struct EncoderParams<'a> {
    /// LayerNorm 1 gamma `[d]` (required).
    pub ln1_g: &'a [f16],
    /// LayerNorm 1 beta `[d]` (`None` = no bias).
    pub ln1_b: Option<&'a [f16]>,
    /// Query projection `Wq[d,d]`.
    pub wq: &'a [f16],
    /// Query bias `[d]`.
    pub bq: Option<&'a [f16]>,
    /// Key projection `Wk[d,d]`.
    pub wk: &'a [f16],
    /// Key bias `[d]`.
    pub bk: Option<&'a [f16]>,
    /// Value projection `Wv[d,d]`.
    pub wv: &'a [f16],
    /// Value bias `[d]`.
    pub bv: Option<&'a [f16]>,
    /// Output projection `Wo[d,d]`.
    pub wo: &'a [f16],
    /// Output bias `[d]`.
    pub bo: Option<&'a [f16]>,
    /// LayerNorm 2 gamma `[d]` (required).
    pub ln2_g: &'a [f16],
    /// LayerNorm 2 beta `[d]` (`None` = no bias).
    pub ln2_b: Option<&'a [f16]>,
    /// fc1 `[d_ff,d]`.
    pub wf1: &'a [f16],
    /// fc1 bias `[d_ff]`.
    pub bf1: Option<&'a [f16]>,
    /// fc2 `[d,d_ff]`.
    pub wf2: &'a [f16],
    /// fc2 bias `[d]`.
    pub bf2: Option<&'a [f16]>,
}

/// The shape/alignment contract, shared by [`encoder_block`] and [`encoder_block_ref`].
///
/// `d % n_head == 0` here; the underlying primitives then require `d % 32 == 0`, `d_ff %
/// 32 == 0` and `dh = d/n_head` a multiple of 32 for the device path (the fp16 matmul's
/// K align), which they report as their own planner shape error — the same behaviour as
/// the C, whose `rocket_mha_self_fp16` inherits exactly those refusals.
fn check_shape(t: usize, d: usize, n_head: usize, d_ff: usize) -> Result<()> {
    if t < 1 || d < 1 || n_head < 1 || d_ff < 1 || !d.is_multiple_of(n_head) {
        return Err(Error::shape("rocket_encoder_block_fp16"));
    }
    Ok(())
}

/// Length checks for every tensor of the block (before any allocation or device access).
fn check_buffers(
    t: usize,
    d: usize,
    d_ff: usize,
    x: &[f16],
    p: &EncoderParams,
    out: &[f16],
) -> Result<()> {
    let td = t * d;
    let ok = x.len() >= td
        && out.len() >= td
        && p.ln1_g.len() >= d
        && p.ln2_g.len() >= d
        && p.ln1_b.is_none_or(|b| b.len() >= d)
        && p.ln2_b.is_none_or(|b| b.len() >= d)
        && p.wq.len() >= d * d
        && p.wk.len() >= d * d
        && p.wv.len() >= d * d
        && p.wo.len() >= d * d
        && p.bq.is_none_or(|b| b.len() >= d)
        && p.bk.is_none_or(|b| b.len() >= d)
        && p.bv.is_none_or(|b| b.len() >= d)
        && p.bo.is_none_or(|b| b.len() >= d)
        && p.wf1.len() >= d_ff * d
        && p.wf2.len() >= d * d_ff
        && p.bf1.is_none_or(|b| b.len() >= d_ff)
        && p.bf2.is_none_or(|b| b.len() >= d);
    if !ok {
        return Err(
            Error::shape("rocket_encoder_block_fp16/buffers").with_detail(format!(
                "T={t} d={d} d_ff={d_ff}: x {} / out {} (need {td}), Wq {} / Wf1 {} (need {})",
                x.len(),
                out.len(),
                p.wq.len(),
                p.wf1.len(),
                d_ff * d
            )),
        );
    }
    Ok(())
}

/// `add_bias` over the first `rows` rows of a row-major `[rows, n]` fp16 matrix, in f32
/// with one RNE narrow (the C's `add_bias`).
fn add_bias(c: &mut [f16], b: Option<&[f16]>, rows: usize, n: usize) {
    let Some(b) = b else { return };
    for row in 0..rows {
        let r = &mut c[row * n..row * n + n];
        for (cell, &bv) in r.iter_mut().zip(b) {
            *cell = f16::from_f32(cell.to_f32() + bv.to_f32());
        }
    }
}

/// `rocket_encoder_block_ref_fp16`: the exact fp64 reference of [`encoder_block`].
///
/// Every op is the f64 host form: [`super::norm::layernorm_ref`],
/// [`crate::attn::mha_self_ref`], the residual in f64, the MLP with the bias folded into
/// the f64 first projection and the GELU evaluated with glibc `erf`, and the second
/// residual in f64. Only the two residual results narrow, exactly as the C's reference
/// does (its `f1[o]` intermediate stays f64).
#[allow(clippy::too_many_arguments)]
pub fn encoder_block_ref(
    t: usize,
    d: usize,
    n_head: usize,
    d_ff: usize,
    x: &[f16],
    p: &EncoderParams,
    eps: f32,
    out: &mut [f16],
) -> Result<()> {
    check_shape(t, d, n_head, d_ff)?;
    check_buffers(t, d, d_ff, x, p, out)?;
    let td = t * d;

    // x = x + MHA(LN1(x))
    let mut ln = vec![f16::ZERO; td];
    layernorm_ref(t, d, x, p.ln1_g, p.ln1_b, eps, &mut ln);
    let mut attn = vec![f16::ZERO; td];
    crate::attn::mha_self_ref(
        t, d, n_head, &ln, p.wq, p.bq, p.wk, p.bk, p.wv, p.bv, p.wo, p.bo, &mut attn,
    )?;
    let mut xa = vec![f16::ZERO; td];
    for i in 0..td {
        xa[i] = crate::ops::f16_from_f64(x[i].to_f64() + attn[i].to_f64());
    }

    // x = x + MLP(LN2(x))
    layernorm_ref(t, d, &xa, p.ln2_g, p.ln2_b, eps, &mut ln);
    let mut f1 = vec![0.0f64; d_ff];
    for row in 0..t {
        let hr = &ln[row * d..row * d + d];
        for (o, slot) in f1.iter_mut().enumerate() {
            let w = &p.wf1[o * d..o * d + d];
            let mut a = p.bf1.map_or(0.0, |b| b[o].to_f64());
            for j in 0..d {
                a += hr[j].to_f64() * w[j].to_f64();
            }
            *slot = gelu_d(a);
        }
        let orow = &mut out[row * d..row * d + d];
        for (o, cell) in orow.iter_mut().enumerate() {
            let w = &p.wf2[o * d_ff..o * d_ff + d_ff];
            let mut a = 0.0f64;
            for (j, &fv) in f1.iter().enumerate() {
                a += fv * w[j].to_f64();
            }
            let bias = p.bf2.map_or(0.0, |b| b[o].to_f64());
            *cell = crate::ops::f16_from_f64(xa[row * d + o].to_f64() + a + bias);
        }
    }
    Ok(())
}

/// `5·erf`-based exact GELU (the C's `gelu_d`).
fn gelu_d(x: f64) -> f64 {
    0.5 * x * (1.0 + super::activation::erf64(x * std::f64::consts::FRAC_1_SQRT_2))
}

/// `rocket_encoder_block_fp16`: one pre-norm encoder block, on the NPU.
///
/// `x[T,d]` → `out[T,d]` with the block of [`EncoderParams`]. `eps` is the LayerNorm
/// epsilon (Whisper: 1e-5).
///
/// The MLP is [`mlp_fused`] over `Tp = (T+3) & ~3` rows unless `ROCKET_ENCODER_NOFUSE` is
/// present, in which case it is the host-handoff route the C keeps for A/B: fc1 →
/// host bias add over the real `T` rows → the 2-pass GELU (`GeluGate` then an EW
/// multiply) → fc2. Both routes compute the same function; they differ only in the
/// down matmul's K-tiling.
#[allow(clippy::too_many_arguments)]
pub fn encoder_block(
    dev: &Device,
    t: usize,
    d: usize,
    n_head: usize,
    d_ff: usize,
    x: &[f16],
    p: &EncoderParams,
    eps: f32,
    out: &mut [f16],
) -> Result<()> {
    check_shape(t, d, n_head, d_ff)?;
    check_buffers(t, d, d_ff, x, p, out)?;

    let tp = (t + 3) & !3; // matmul M % 4
    let td = t * d;
    // Tp*d_ff is passed as an i32 to the GELU/EW ops below; reject a truncating product.
    if tp * d_ff > i32::MAX as usize {
        return Err(Error::shape("rocket_encoder_block_fp16")
            .with_detail(format!("{tp}*{d_ff} elements > INT_MAX")));
    }
    let x = &x[..td];
    let out = &mut out[..td];

    // --- attention sublayer: x = x + MHA(LN1(x)) ---
    let mut ln = vec![f16::ZERO; td];
    layernorm(dev, t, d, x, p.ln1_g, p.ln1_b, eps, &mut ln)?;
    let mut attn = vec![f16::ZERO; td];
    crate::attn::mha_self(
        t, d, n_head, &ln, p.wq, p.bq, p.wk, p.bk, p.wv, p.bv, p.wo, p.bo, &mut attn, 1,
    )?;
    let mut xa = vec![f16::ZERO; td];
    ew_add(dev, x, &attn, &mut xa)?;

    // --- feed-forward sublayer: x = x + MLP(LN2(x)) ---
    layernorm(dev, t, d, &xa, p.ln2_g, p.ln2_b, eps, &mut ln)?;
    // LN2 output zero-padded to Tp rows (pad rows are matmul-of-zero, never read out).
    let mut hp = vec![f16::ZERO; tp * d];
    hp[..td].copy_from_slice(&ln);

    // f2 = GELU(LN2(x)·Wf1^T + bf1)·Wf2^T.
    let mut f2 = vec![f16::ZERO; tp * d];
    if !knobs::present("ROCKET_ENCODER_NOFUSE") {
        // The C's default: the cross-op MLP (pinned fc1/fc2, bias and 2-pass GELU over
        // the intermediate, fc2 reading it directly). Shapes it cannot tiling-match
        // fall back inside to the host-handoff route.
        mlp_fused(
            dev,
            tp,
            d,
            d_ff,
            d,
            &hp,
            p.wf1,
            p.bf1,
            p.wf2,
            Activation::Gelu,
            &mut f2,
        )?;
    } else {
        // The C's host-handoff A/B route, op for op: fc1 with the default tiling; the bias
        // add over the REAL T rows (the pad rows keep the zero their matmul produced);
        // the 2-pass GELU spelled out as the gate LUT then an EW multiply (not the
        // combined `activation` call, which multiplies on the host); fc2.
        let mut f1 = vec![f16::ZERO; tp * d_ff];
        one_shot(dev, tp, d, d_ff, &hp, p.wf1, &mut f1)?;
        add_bias(&mut f1, p.bf1, t, d_ff);
        let mut gate = vec![f16::ZERO; tp * d_ff];
        activation(dev, Activation::GeluGate, &f1, &mut gate)?;
        let mut act = vec![f16::ZERO; tp * d_ff];
        ew_mul(dev, &f1, &gate, &mut act)?;
        one_shot(dev, tp, d_ff, d, &act, p.wf2, &mut f2)?;
    }
    add_bias(&mut f2, p.bf2, t, d);

    // out = xa + f2 over the real T rows (the pad rows are dropped, as in C).
    let f2c = &f2[..td];
    ew_add(dev, &xa, f2c, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::ffn::mlp_ref;

    fn vals(v: &[f32]) -> Vec<f16> {
        v.iter().map(|x| f16::from_f32(*x)).collect()
    }

    fn lcg(n: usize, lo: f32, hi: f32, seed: u32) -> Vec<f16> {
        let mut st = 0x9e37_79b9u32 ^ seed;
        (0..n)
            .map(|_| {
                st = st.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let u = ((st >> 8) & 0xffff) as f32 / 65535.0;
                f16::from_f32(lo + (hi - lo) * u)
            })
            .collect()
    }

    /// A tiny block: `T=3, d=8, n_head=2, d_ff=16`. The reference is host-only, so the
    /// matmul alignments (d%32, d_ff%32, dh%32) do not apply — only the device path
    /// inherits them from the fp16 matmul.
    #[allow(clippy::too_many_arguments)]
    fn tiny<'a>(
        ln1_g: &'a [f16],
        ln1_b: &'a [f16],
        w: &'a [f16],
        ln2_g: &'a [f16],
        ln2_b: &'a [f16],
        wf1: &'a [f16],
        wf2: &'a [f16],
        bf1: &'a [f16],
        bf2: &'a [f16],
    ) -> EncoderParams<'a> {
        EncoderParams {
            ln1_g,
            ln1_b: Some(ln1_b),
            wq: w,
            bq: Some(w),
            wk: w,
            bk: None,
            wv: w,
            bv: Some(w),
            wo: w,
            bo: Some(w),
            ln2_g,
            ln2_b: Some(ln2_b),
            wf1,
            bf1: Some(bf1),
            wf2,
            bf2: Some(bf2),
        }
    }

    #[test]
    fn encoder_reference_is_deterministic_and_finite() {
        let (t, d, n_head, d_ff) = (3, 8, 2, 16);
        let x = lcg(t * d, -1.0, 1.0, 1);
        let ln1_g = vals(&[1.0; 8]);
        let ln1_b = vals(&[0.0; 8]);
        let ln2_g = vals(&[1.0; 8]);
        let ln2_b = vals(&[0.0; 8]);
        let wq = lcg(d * d, -0.5, 0.5, 2);
        let wf1 = lcg(d_ff * d, -0.5, 0.5, 3);
        let wf2 = lcg(d * d_ff, -0.5, 0.5, 4);
        let bf1 = lcg(d_ff, -0.1, 0.1, 5);
        let bf2 = lcg(d, -0.1, 0.1, 6);
        let p = tiny(&ln1_g, &ln1_b, &wq, &ln2_g, &ln2_b, &wf1, &wf2, &bf1, &bf2);

        let mut a = vec![f16::ZERO; t * d];
        let mut b = vec![f16::ZERO; t * d];
        encoder_block_ref(t, d, n_head, d_ff, &x, &p, 1e-5, &mut a).unwrap();
        encoder_block_ref(t, d, n_head, d_ff, &x, &p, 1e-5, &mut b).unwrap();
        assert_eq!(a, b, "the reference must be deterministic");
        assert!(
            a.iter().all(|v| v.to_f32().is_finite()),
            "non-finite output: {a:?}"
        );
        assert!(
            a.iter().any(|v| v.to_f32() != 0.0),
            "a zero block means something short-circuited"
        );

        // The residual carries x through: out - x is the block's contribution, and the
        // two LayerNorms + MLP must not have flattened it to the identity.
        let delta: f64 = a
            .iter()
            .zip(&x)
            .map(|(o, i)| (o.to_f64() - i.to_f64()).abs())
            .sum();
        assert!(delta > 1e-3, "the block did nothing (delta {delta})");
    }

    #[test]
    fn encoder_reference_is_the_composition_of_its_parts() {
        // The same block assembled by hand from the public primitives must agree with the
        // reference to within the f64 rounding of the two orders.
        let (t, d, n_head, d_ff) = (3, 8, 2, 16);
        let x = lcg(t * d, -1.0, 1.0, 11);
        let ln1_g = lcg(d, 0.5, 1.5, 12);
        let ln1_b = lcg(d, -0.5, 0.5, 13);
        let ln2_g = lcg(d, 0.5, 1.5, 14);
        let ln2_b = lcg(d, -0.5, 0.5, 15);
        let wq = lcg(d * d, -0.5, 0.5, 16);
        let wf1 = lcg(d_ff * d, -0.5, 0.5, 17);
        let wf2 = lcg(d * d_ff, -0.5, 0.5, 18);
        let bf1 = lcg(d_ff, -0.1, 0.1, 19);
        let bf2 = lcg(d, -0.1, 0.1, 20);
        let p = tiny(&ln1_g, &ln1_b, &wq, &ln2_g, &ln2_b, &wf1, &wf2, &bf1, &bf2);

        let mut got = vec![f16::ZERO; t * d];
        encoder_block_ref(t, d, n_head, d_ff, &x, &p, 1e-5, &mut got).unwrap();

        // Hand assembly through the family primitives.
        let mut ln = vec![f16::ZERO; t * d];
        layernorm_ref(t, d, &x, p.ln1_g, p.ln1_b, 1e-5, &mut ln);
        let mut attn = vec![f16::ZERO; t * d];
        crate::attn::mha_self_ref(
            t, d, n_head, &ln, p.wq, p.bq, p.wk, p.bk, p.wv, p.bv, p.wo, p.bo, &mut attn,
        )
        .unwrap();
        let mut xa = vec![f16::ZERO; t * d];
        for i in 0..t * d {
            xa[i] = crate::ops::f16_from_f64(x[i].to_f64() + attn[i].to_f64());
        }
        layernorm_ref(t, d, &xa, p.ln2_g, p.ln2_b, 1e-5, &mut ln);
        let mut want = vec![f16::ZERO; t * d];
        mlp_ref(
            t,
            d,
            d_ff,
            d,
            &ln,
            p.wf1,
            p.bf1,
            p.wf2,
            Activation::Gelu,
            &mut want,
        );
        // mlp_ref has no bf2, so add it and the residual per the block's formula.
        for row in 0..t {
            for o in 0..d {
                let v = want[row * d + o].to_f64()
                    + f64::from(p.bf2.unwrap()[o].to_f32())
                    + xa[row * d + o].to_f64();
                want[row * d + o] = f16::from_f64(v);
            }
        }
        for (i, (a, b)) in got.iter().zip(&want).enumerate() {
            assert!(
                (a.to_f64() - b.to_f64()).abs() < 2e-3,
                "{i}: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn encoder_shape_contract() {
        let x = vec![f16::ZERO; 8];
        let w = vec![f16::ZERO; 8];
        let g = vec![f16::ONE; 2];
        let p = tiny(&g, &g, &w, &g, &g, &w, &w, &g, &g);
        let mut out = vec![f16::ZERO; 8];
        // d % n_head != 0 and zero dimensions are refused before any work.
        assert!(encoder_block_ref(2, 3, 2, 4, &x, &p, 1e-5, &mut out).is_err());
        assert!(encoder_block_ref(0, 2, 1, 2, &x, &p, 1e-5, &mut out).is_err());
        assert!(encoder_block_ref(2, 2, 0, 2, &x, &p, 1e-5, &mut out).is_err());
        assert!(encoder_block_ref(2, 2, 1, 0, &x, &p, 1e-5, &mut out).is_err());
        // A short weight buffer is caught before the device is touched: with d_ff=4 the
        // fc1 needs 4*2 = 8 elements and this one has 4.
        let short = tiny(&g, &g, &w, &g, &g, &w[..4], &w, &g, &g);
        assert!(encoder_block_ref(4, 2, 1, 4, &x, &short, 1e-5, &mut out).is_err());
        // ... and a short bias.
        let short = tiny(&g, &g, &w, &g, &g, &w, &w, &g[..1], &g);
        assert!(encoder_block_ref(4, 2, 1, 4, &x, &short, 1e-5, &mut out).is_err());
    }
}
