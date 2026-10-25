//! The gated-MLP transformer FFN block (RK3588): the GeGLU/SwiGLU core, the block that
//! wraps it between the three projection matmuls, and the cross-op-fused variants.
//!
//! Port of `rocket_ffn.c` + `rocket_ffn.h`:
//!
//! ```text
//! gate = x · Wg^T        [M,I]      (gate projection)
//! up   = x · Wu^T        [M,I]      (up projection; Wg/Wu share x -> fuse-able)
//! prod = act(gate) ⊙ up  [M,I]      (the gated activation — GeGLU/SwiGLU core)
//! out  = prod · Wd^T     [M,H]      (down projection)
//! ```
//!
//! The only computation here beyond plain matmul is the gated activation
//! `act(gate) ⊙ up`, so that is the reusable primitive ([`geglu`]); the block ([`ffn`])
//! wraps it between the three matmuls, and [`mlp_fused`] is the non-gated encoder MLP
//! (`out = act(x·W1^T + b1)·W2^T`).
//!
//! # Composition, and what "fused" means here
//!
//! The C's fused entries ([`ffn_fused`], [`mlp_fused`]) keep the `[M,I]` intermediates in
//! NPU cube layout across `gate/up → act⊙up → down`, so the host never de-tiles and
//! re-tiles them: the two projections leave their full output cubes in resident BOs, the
//! gated activation runs element-wise over the cube bytes (it commutes with the cube
//! reindex — pad lanes stay 0 because `act(0)·0 = 0`), and the down matmul reads the
//! product cube directly as its input.
//!
//! This port keeps the **tiling contract** that makes those numerics what they are — the
//! pinned tiles ([`crate::matmul::Plan::pinned`]: `Nt(projections)` = `Kt(down)` = 256,
//! the C's `mm_plan_init_pin`), the matched-tiling/precondition gate, KACC-only for the
//! projections with the host-handoff fallback, KACC-then-CPU for the down matmul — and
//! **not** the cube residency: each intermediate comes back to row-major and is packed
//! again. The element-wise ops commute with the cube reindex and de-tiling is lossless,
//! so the result is bit-identical to the C's cube-resident path (the board A/B checks
//! exactly that); what is left on the table is the host round-trip the C's
//! `RESIDENT-FUSION NOTE` describes as the follow-on optimization. The C's documented
//! *numeric* deviation between fused and unfused (the down matmul's K-tiling is pinned to
//! the projections' `Nt` instead of the planner's own choice) is reproduced faithfully,
//! so [`ffn_fused`] and [`ffn`] can differ in the last bits — see the module tests.
//!
//! Shapes the pinned tiling cannot express (or that the KACC EW stride floor refuses)
//! fall back to the host-handoff block internally, exactly as the C does.

use std::f64::consts::FRAC_1_SQRT_2;

use half::f16;

use crate::Device;
use crate::error::{Error, Result};
use crate::matmul::Plan;
use crate::matmul::run::{KaccPolicy, one_shot, one_shot_mt, one_shot_plan};
use crate::ops::activation::{Activation, activation};
use crate::regcmd::desc::BATCH;

/// The C `FFN_BATCH`: one NPU batch's worth of tiles, the single-batch full-cube cap.
const FFN_BATCH: usize = BATCH;

/// The shared cross-op tile: `Nt(gate/up) == Kt(down)`, a multiple of 32, so the
/// projection output cubes alias the down matmul's input feature cube tile for tile.
const TILE: usize = 256;

/// The C `act_exact`: the reference nonlinearity of the gated core (`f64`, glibc math).
///
/// `ROCKET_ACTIVATION_GELU` is the exact `x·Φ(x)` (glibc `erf`, the same function the
/// LUT samples); **every other kind** is SiLU, exactly as the C's `act_exact` defaults —
/// the header documents `SILU`/`GELU` as the two gated-MLP nonlinearities.
fn act_exact(kind: Activation, g: f64) -> f64 {
    if kind == Activation::Gelu {
        0.5 * g * (1.0 + super::activation::erf64(g * FRAC_1_SQRT_2))
    } else {
        g / (1.0 + (-g).exp())
    }
}

fn guard_elems(name: &'static str, elems: usize) -> Result<()> {
    if elems > i32::MAX as usize {
        return Err(Error::shape(name).with_detail(format!("{elems} elements > INT_MAX")));
    }
    Ok(())
}

/// `add_bias`: in-place per-column bias add over the first `rows` rows of a row-major
/// `[rows, n]` fp16 matrix, in f32 with one RNE narrow (the C's `add_bias`/`mlp_add_bias`).
fn add_bias(c: &mut [f16], b: Option<&[f16]>, rows: usize, n: usize) {
    let Some(b) = b else { return };
    for row in 0..rows {
        let r = &mut c[row * n..row * n + n];
        for (cell, &bv) in r.iter_mut().zip(b) {
            *cell = f16::from_f32(cell.to_f32() + bv.to_f32());
        }
    }
}

// ── Gated activation (GeGLU / SwiGLU core) ────────────────────────────────────

/// `rocket_geglu_ref_fp16`: the exact fp64 reference of [`geglu`].
///
/// `prod[i] = act_exact(gate[i]) · up[i]` in f64 with one RNE narrow; the nonlinearity is
/// GELU for [`Activation::Gelu`] and SiLU for anything else (the C's default).
pub fn geglu_ref(gate: &[f16], up: &[f16], kind: Activation, prod: &mut [f16]) {
    assert_eq!(gate.len(), up.len(), "gate/up length mismatch");
    assert!(prod.len() >= gate.len(), "prod shorter than n");
    for i in 0..gate.len() {
        prod[i] = f16::from_f64(act_exact(kind, gate[i].to_f64()) * up[i].to_f64());
    }
}

/// `rocket_geglu_fp16`: `prod[i] = act(gate[i]) · up[i]` elementwise, on the NPU.
///
/// The gated activation: the activation LUT pass (the gated kinds run the C's two-pass
/// `x·gate(x)`) and then one EW multiply. `n` is the flat element count `M*I`.
///
/// The activation LUT domain is bounded (`SiLU [-12,12]`), so gate logits outside it
/// saturate — see `rocket_activation.c`'s findings, which this inherits.
pub fn geglu(
    dev: &Device,
    gate: &[f16],
    up: &[f16],
    kind: Activation,
    prod: &mut [f16],
) -> Result<()> {
    if gate.is_empty() || gate.len() != up.len() || prod.len() < gate.len() {
        return Err(Error::shape("rocket_geglu_fp16"));
    }
    let n = gate.len();
    guard_elems("rocket_geglu_fp16", n)?;
    let mut act_g = vec![f16::ZERO; n];
    activation(dev, kind, gate, &mut act_g)?;
    crate::ops::activation::ew_mul(dev, &act_g, up, &mut prod[..n])
}

// ── The full FFN block ────────────────────────────────────────────────────────

/// `rocket_ffn_ref_fp16`: the exact fp64 reference of [`ffn`].
///
/// `gate`/`up` are accumulated in f64, the gated product stays f64 (no fp16 narrowing of
/// the intermediate — the C allocates one `double[I]`), and only `out` narrows.
///
/// Like the C, the intermediate is one `I`-wide row buffer rather than an `[M,I]` matrix:
/// this is also the `fd < 0` host fallback, so it runs in production on non-NPU hosts.
#[allow(clippy::too_many_arguments)]
pub fn ffn_ref(
    m: usize,
    h: usize,
    i: usize,
    x: &[f16],
    wg: &[f16],
    wu: &[f16],
    wd: &[f16],
    kind: Activation,
    out: &mut [f16],
) {
    assert!(x.len() >= m * h, "x shorter than M*H");
    assert!(wg.len() >= i * h, "Wg shorter than I*H");
    assert!(wu.len() >= i * h, "Wu shorter than I*H");
    assert!(wd.len() >= h * i, "Wd shorter than H*I");
    assert!(out.len() >= m * h, "out shorter than M*H");
    let mut prod = vec![0.0f64; i];
    for row in 0..m {
        let xr = &x[row * h..row * h + h];
        for (o, p) in prod.iter_mut().enumerate() {
            let (wg_row, wu_row) = (&wg[o * h..o * h + h], &wu[o * h..o * h + h]);
            let (mut g, mut u) = (0.0f64, 0.0f64);
            for j in 0..h {
                let xv = xr[j].to_f64();
                g += xv * wg_row[j].to_f64();
                u += xv * wu_row[j].to_f64();
            }
            *p = act_exact(kind, g) * u;
        }
        let orow = &mut out[row * h..row * h + h];
        for (o, cell) in orow.iter_mut().enumerate() {
            let wd_row = &wd[o * i..o * i + i];
            let mut acc = 0.0f64;
            for (p, &w) in prod.iter().zip(wd_row) {
                acc += p * w.to_f64();
            }
            *cell = f16::from_f64(acc);
        }
    }
}

/// Shape/buffer validation shared by the FFN entries (`[M,H]`, `Wg/Wu[I,H]`, `Wd[H,I]`).
#[allow(clippy::too_many_arguments)]
fn check_ffn(
    name: &'static str,
    m: usize,
    h: usize,
    i: usize,
    x: &[f16],
    wg: &[f16],
    wu: &[f16],
    wd: &[f16],
    out: &[f16],
) -> Result<()> {
    if m < 1 || h < 1 || i < 1 {
        return Err(Error::shape(name));
    }
    if x.len() < m * h
        || out.len() < m * h
        || wg.len() < i * h
        || wu.len() < i * h
        || wd.len() < h * i
    {
        return Err(Error::shape(name).with_detail(format!(
            "x {} / out {} (need {}), Wg {} / Wu {} (need {}), Wd {} (need {})",
            x.len(),
            out.len(),
            m * h,
            wg.len(),
            wu.len(),
            i * h,
            wd.len(),
            h * i
        )));
    }
    Ok(())
}

/// `rocket_ffn_fp16`: the full FFN block, host-handoff composition.
///
/// `x[M,H]`, `Wg[I,H]`, `Wu[I,H]`, `Wd[H,I]` (all row-major fp16, the `[N,K]` matmul
/// weight convention), `out[M,H]`. `kind` selects the gated nonlinearity (GELU ⇒ GeGLU,
/// anything else ⇒ SwiGLU, the C's `act_exact` default).
///
/// The projections are the C's `rocket_matmul_fp16_mt` with its `NT = 3` fan-out (three
/// N-slices, one worker fd each — bit-identical to a single fd, split only to use the
/// three NPU cores), then [`geglu`], then the down projection.
///
/// Alignment: `H % 32`, `I % 32`, `M % 4` (the matmul requirements; `I` is both an `N`
/// and the down matmul's `K`). `M == 1` is padded by the one-shot entry, as in C.
#[allow(clippy::too_many_arguments)]
pub fn ffn(
    dev: &Device,
    m: usize,
    h: usize,
    i: usize,
    x: &[f16],
    wg: &[f16],
    wu: &[f16],
    wd: &[f16],
    kind: Activation,
    out: &mut [f16],
) -> Result<()> {
    check_ffn("rocket_ffn_fp16", m, h, i, x, wg, wu, wd, out)?;
    guard_elems("rocket_ffn_fp16", m * i)?;

    const NT: usize = 3; // fan the projections across the 3 NPU cores
    let x = &x[..m * h];
    let wd = &wd[..h * i];
    let out = &mut out[..m * h];

    // gate = x·Wg^T, up = x·Wu^T (Wg/Wu share x — fuse-able; correctness uses two calls)
    let mut gate = vec![f16::ZERO; m * i];
    let mut up = vec![f16::ZERO; m * i];
    one_shot_mt(m, h, i, x, wg, &mut gate, NT)?;
    one_shot_mt(m, h, i, x, wu, &mut up, NT)?;

    // prod = act(gate) ⊙ up (the gated-activation core, on the NPU)
    let mut prod = vec![f16::ZERO; m * i];
    geglu(dev, &gate, &up, kind, &mut prod)?;

    // out = prod·Wd^T
    one_shot_mt(m, i, h, &prod, wd, out, NT)
}

/// `rocket_ffn_fp16_fused`: the cross-op FFN, same math and signature as [`ffn`].
///
/// The pinned tiles (`Nt(gate/up) = Kt(down) = 256`) and the matched-tiling gate are the
/// C's `rocket_ffn_fp16_fused`; see the module docs for what this port keeps (the
/// tiling contract and hence the numerics) and what it does not (cube residency).
///
/// Falls back to [`ffn`] whenever the pinned tiling cannot express the shape — a
/// source-level shape the planner refuses, a tiling that does not match, more
/// projection tiles than one NPU batch, or a KACC EW-stride refusal — exactly as the C
/// does (`goto fb`), including for a genuine device error on the pinned attempt.
#[allow(clippy::too_many_arguments)]
pub fn ffn_fused(
    dev: &Device,
    m: usize,
    h: usize,
    i: usize,
    x: &[f16],
    wg: &[f16],
    wu: &[f16],
    wd: &[f16],
    kind: Activation,
    out: &mut [f16],
) -> Result<()> {
    check_ffn("rocket_ffn_fp16_fused", m, h, i, x, wg, wu, wd, out)?;
    guard_elems("rocket_ffn_fp16_fused", m * i)?;
    let fallback = |dev: &Device, out: &mut [f16]| ffn(dev, m, h, i, x, wg, wu, wd, kind, out);
    let x = &x[..m * h];
    let wd = &wd[..h * i];
    let out = &mut out[..m * h];

    let plans = (
        Plan::pinned(m, h, i, Some(TILE), None), // gate: N=I tiled to TILE
        Plan::pinned(m, h, i, Some(TILE), None), // up:   N=I tiled to TILE
        Plan::pinned(m, i, h, None, Some(TILE)), // down: K=I tiled to TILE
    );
    let (plg, plu, pld) = match plans {
        (Ok(a), Ok(b), Ok(c)) => (a, b, c),
        _ => return fallback(dev, out),
    };

    // Matched-tiling + single-batch preconditions. If any fails, the cube could not alias
    // the consumer's input tile for tile, which is also the condition under which the C
    // gives up and re-plans with the default tiling.
    let cube_elems = plg.cube_elems();
    let matchable = plg.nnt == pld.nkt
        && plg.out_slot == pld.in_slot
        && plg.nmt == pld.nmt
        && plg.mt == pld.mt
        && plg.nmt * plg.nnt <= FFN_BATCH
        && cube_elems <= i32::MAX as usize;
    if !matchable {
        return fallback(dev, out);
    }

    // gate/up left as full output cubes in C (no de-tile); here they come back to
    // row-major, which the elementwise pass cannot tell apart. KACC only: a Tiling
    // refusal means "not cube-resident" and sends the whole block to the fallback.
    let mut gate = vec![f16::ZERO; m * i];
    let mut up = vec![f16::ZERO; m * i];
    if one_shot_plan(dev, &plg, KaccPolicy::Only, x, wg, &mut gate).is_err()
        || one_shot_plan(dev, &plu, KaccPolicy::Only, x, wu, &mut up).is_err()
    {
        return fallback(dev, out);
    }
    let mut prod = vec![f16::ZERO; m * i];
    geglu(dev, &gate, &up, kind, &mut prod)?;

    // out = prod·Wd^T with the pinned K tiling; the C's down matmul is the KACC
    // primitive with a CPU-accumulation fallback on its sentinel.
    one_shot_plan(dev, &pld, KaccPolicy::Forced, &prod, wd, out)
}

// ── The non-gated encoder MLP (fc1 -> +b1 -> act -> fc2) ──────────────────────

/// Host reference + host-handoff NPU path of `out = act(x·W1^T + b1)·W2^T`, the C's
/// `mlp_host_handoff`.
///
/// `x[M,Din]`, `W1[Dff,Din]`, `b1[Dff]` (`None` = no bias), `W2[Dout,Dff]`,
/// `out[M,Dout]`. `kind` is the activation (`GELU` for the standard encoder).
///
/// The reference accumulates the first projection **starting from the bias** in f64 and
/// narrows the activation to fp16 before the second projection — the C's exact wording
/// of "fp16-rounded host reference", which is why the intermediate here is fp16 while
/// [`ffn_ref`]'s stays f64.
#[allow(clippy::too_many_arguments)]
pub fn mlp_ref(
    m: usize,
    din: usize,
    dff: usize,
    dout: usize,
    x: &[f16],
    w1: &[f16],
    b1: Option<&[f16]>,
    w2: &[f16],
    kind: Activation,
    out: &mut [f16],
) {
    assert!(x.len() >= m * din, "x shorter than M*Din");
    assert!(w1.len() >= dff * din, "W1 shorter than Dff*Din");
    assert!(b1.is_none_or(|b| b.len() >= dff), "b1 shorter than Dff");
    assert!(w2.len() >= dout * dff, "W2 shorter than Dout*Dff");
    assert!(out.len() >= m * dout, "out shorter than M*Dout");
    let mut act = vec![f16::ZERO; m * dff];
    for row in 0..m {
        let xr = &x[row * din..row * din + din];
        for o in 0..dff {
            let w = &w1[o * din..o * din + din];
            let mut a = b1.map_or(0.0, |b| b[o].to_f64());
            for j in 0..din {
                a += xr[j].to_f64() * w[j].to_f64();
            }
            act[row * dff + o] = f16::from_f64(act_exact(kind, a));
        }
        let orow = &mut out[row * dout..row * dout + dout];
        for o in 0..dout {
            let w = &w2[o * dff..o * dff + dff];
            let mut a = 0.0f64;
            for j in 0..dff {
                a += act[row * dff + j].to_f64() * w[j].to_f64();
            }
            orow[o] = f16::from_f64(a);
        }
    }
}

/// Shape/buffer validation shared by the MLP entries.
#[allow(clippy::too_many_arguments)]
fn check_mlp(
    name: &'static str,
    m: usize,
    din: usize,
    dff: usize,
    dout: usize,
    x: &[f16],
    w1: &[f16],
    b1: Option<&[f16]>,
    w2: &[f16],
    out: &[f16],
) -> Result<()> {
    if m < 1 || din < 1 || dff < 1 || dout < 1 {
        return Err(Error::shape(name));
    }
    if x.len() < m * din
        || out.len() < m * dout
        || w1.len() < dff * din
        || w2.len() < dout * dff
        || b1.is_some_and(|b| b.len() < dff)
    {
        return Err(Error::shape(name).with_detail(format!(
            "x {} (need {}), W1 {} (need {}), b1 {}, W2 {} (need {}), out {} (need {})",
            x.len(),
            m * din,
            w1.len(),
            dff * din,
            b1.map_or(0, <[f16]>::len),
            w2.len(),
            dout * dff,
            out.len(),
            m * dout
        )));
    }
    Ok(())
}

/// The non-fused MLP: `fc1` (default tiling, single fd) → host bias add → `act` →
/// `fc2`, each op de-tiling to row-major in between.
///
/// This is the C's `mlp_host_handoff` device branch — the reference path behind
/// [`mlp_fused`]'s fallback, exposed because it is also the natural A/B counterpart of
/// the fused form.
#[allow(clippy::too_many_arguments)]
pub fn mlp(
    dev: &Device,
    m: usize,
    din: usize,
    dff: usize,
    dout: usize,
    x: &[f16],
    w1: &[f16],
    b1: Option<&[f16]>,
    w2: &[f16],
    kind: Activation,
    out: &mut [f16],
) -> Result<()> {
    check_mlp("rocket_mlp_fp16", m, din, dff, dout, x, w1, b1, w2, out)?;
    guard_elems("rocket_mlp_fp16", m * dff)?;
    let x = &x[..m * din];
    let out = &mut out[..m * dout];

    let mut f1 = vec![f16::ZERO; m * dff];
    one_shot(dev, m, din, dff, x, w1, &mut f1)?;
    add_bias(&mut f1, b1, m, dff);
    let mut act = vec![f16::ZERO; m * dff];
    activation(dev, kind, &f1, &mut act)?;
    one_shot(dev, m, dff, dout, &act, w2, out)
}

/// `rocket_mlp_fp16_fused`: the cross-op cube-resident non-gated MLP (encoder FFN).
///
/// `out = act(x·W1^T + b1)·W2^T` with the C's pinned tiles (`Nt(fc1) = Kt(fc2) = 256`),
/// the matched-tiling gate, the bias add and the activation over the `[M,Dff]`
/// intermediate, and the KACC/cube sentinel semantics. `b1` may be `None` (no bias).
///
/// Alignment: `Din % 32`, `Dff % 32`, `Dout % 16`, `M % 4`; shapes the pinned tiling
/// cannot match fall back to [`mlp`] (the C's `mlp_host_handoff`), as in
/// [`ffn_fused`]. Shapes whose fc1 output exceeds one NPU batch do the same.
#[allow(clippy::too_many_arguments)]
pub fn mlp_fused(
    dev: &Device,
    m: usize,
    din: usize,
    dff: usize,
    dout: usize,
    x: &[f16],
    w1: &[f16],
    b1: Option<&[f16]>,
    w2: &[f16],
    kind: Activation,
    out: &mut [f16],
) -> Result<()> {
    check_mlp(
        "rocket_mlp_fp16_fused",
        m,
        din,
        dff,
        dout,
        x,
        w1,
        b1,
        w2,
        out,
    )?;
    guard_elems("rocket_mlp_fp16_fused", m * dff)?;
    let fallback =
        |dev: &Device, out: &mut [f16]| mlp(dev, m, din, dff, dout, x, w1, b1, w2, kind, out);
    let x = &x[..m * din];
    let out = &mut out[..m * dout];

    let plans = (
        Plan::pinned(m, din, dff, Some(TILE), None), // fc1: N=Dff tiled to TILE
        Plan::pinned(m, dff, dout, None, Some(TILE)), // fc2: K=Dff tiled to TILE
    );
    let (p1, p2) = match plans {
        (Ok(a), Ok(b)) => (a, b),
        _ => return fallback(dev, out),
    };
    let cube_elems = p1.cube_elems();
    if !(p1.nnt == p2.nkt
        && p1.out_slot == p2.in_slot
        && p1.nmt == p2.nmt
        && p1.mt == p2.mt
        && p1.nmt * p1.nnt <= FFN_BATCH
        && cube_elems <= i32::MAX as usize)
    {
        return fallback(dev, out);
    }

    // fc1, left as a full output cube in C; the bias and the activation then run
    // element-wise over the cube bytes, which is per-element identical to this row-major
    // pass (the C scatters the per-channel bias into the cube, then EW-adds).
    let mut f1 = vec![f16::ZERO; m * dff];
    if one_shot_plan(dev, &p1, KaccPolicy::Only, x, w1, &mut f1).is_err() {
        return fallback(dev, out);
    }
    add_bias(&mut f1, b1, m, dff);
    let mut act = vec![f16::ZERO; m * dff];
    activation(dev, kind, &f1, &mut act)?;

    // out = act·W2^T with the pinned K tiling (KACC, CPU accumulation on its sentinel).
    one_shot_plan(dev, &p2, KaccPolicy::Forced, &act, w2, out)
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
    fn geglu_reference_is_the_gated_product() {
        let gate = vals(&[0.0, 1.0, -1.0, 2.0]);
        let up = vals(&[1.0, 2.0, 3.0, 4.0]);
        let mut prod = vec![f16::ZERO; 4];

        // GELU: gelu(0)=0, gelu(1)=0.841345, gelu(-1)=-0.158655, gelu(2)=1.954500.
        geglu_ref(&gate, &up, Activation::Gelu, &mut prod);
        let got = f64s(&prod);
        assert_eq!(got[0], 0.0);
        assert!((got[1] - 2.0 * 0.841_345).abs() < 2e-3);
        assert!((got[2] + 3.0 * 0.158_655).abs() < 2e-3); // gelu(-1) is negative
        assert!((got[3] - 4.0 * 1.954_500).abs() < 8e-3);

        // SiLU: act(x) = x·sigmoid(x).
        geglu_ref(&gate, &up, Activation::Silu, &mut prod);
        let got = f64s(&prod);
        assert_eq!(got[0], 0.0);
        let silu = |x: f64| x / (1.0 + (-x).exp());
        assert!((got[1] - 2.0 * silu(1.0)).abs() < 2e-3);
        assert!((got[2] - 3.0 * silu(-1.0)).abs() < 2e-3);

        // Any other kind defaults to SiLU, exactly as the C's act_exact does.
        geglu_ref(&gate, &up, Activation::Tanh, &mut prod);
        let got_tanh = f64s(&prod);
        assert_eq!(got_tanh, got);
    }

    #[test]
    fn ffn_reference_matches_a_hand_computed_block() {
        // M=1 H=2 I=2, act = GELU, identity-ish weights so the arithmetic is checkable.
        //   Wg = [[1,0],[0,1]]  Wu = [[2,0],[0,2]]  Wd = [[1,0],[0,1]]
        //   x = [1, 2] -> gate = [1,2], up = [2,4], prod = [gelu(1)·2, gelu(2)·4]
        //   out = prod (Wd identity).
        let x = vals(&[1.0, 2.0]);
        let wg = vals(&[1.0, 0.0, 0.0, 1.0]);
        let wu = vals(&[2.0, 0.0, 0.0, 2.0]);
        let wd = vals(&[1.0, 0.0, 0.0, 1.0]);
        let mut out = vec![f16::ZERO; 2];
        ffn_ref(1, 2, 2, &x, &wg, &wu, &wd, Activation::Gelu, &mut out);
        let got = f64s(&out);
        let phi = |z: f64| 0.5 * (1.0 + crate::ops::activation::erf64(z * FRAC_1_SQRT_2));
        let p0 = phi(1.0) * 1.0 * 2.0; // gelu(gate[0]) · up[0] = 1·Φ(1)·2
        let p1 = phi(2.0) * 2.0 * 4.0; // gelu(gate[1]) · up[1] = 2·Φ(2)·4
        assert!((got[0] - p0).abs() < 2e-3, "{} vs {p0}", got[0]);
        assert!((got[1] - p1).abs() < 1e-2, "{} vs {p1}", got[1]);

        // A second row exercises the per-row loop and the down projection's mixing.
        //   x = [2,0] -> gate = [2,0] -> up = [4,0] -> prod = [gelu(2)·4, 0]
        let x = vals(&[1.0, 2.0, 2.0, 0.0]);
        let mut out = vec![f16::ZERO; 4];
        ffn_ref(2, 2, 2, &x, &wg, &wu, &wd, Activation::Gelu, &mut out);
        let got = f64s(&out);
        assert!((got[0] - p0).abs() < 2e-3);
        assert!((got[1] - p1).abs() < 1e-2);
        assert!((got[2] - p1).abs() < 1e-2);
        assert!(got[3].abs() < 1e-6);
    }

    #[test]
    fn ffn_reference_mixes_through_a_nontrivial_down_projection() {
        // Wd = [[0,1],[1,0]] swaps the two product lanes: the output is the swapped prod.
        let x = vals(&[1.0, 2.0]);
        let wg = vals(&[1.0, 0.0, 0.0, 1.0]);
        let wu = vals(&[2.0, 0.0, 0.0, 2.0]);
        let wd = vals(&[0.0, 1.0, 1.0, 0.0]); // row 0 = [0,1] -> out[0] = prod[1]
        let mut out = vec![f16::ZERO; 2];
        ffn_ref(1, 2, 2, &x, &wg, &wu, &wd, Activation::Silu, &mut out);
        let silu = |z: f64| z / (1.0 + (-z).exp());
        let p0 = silu(1.0) * 2.0; // silu(gate[0])·up[0]
        let p1 = silu(2.0) * 4.0; // silu(gate[1])·up[1]
        assert!((out[0].to_f64() - p1).abs() < 1e-2);
        assert!((out[1].to_f64() - p0).abs() < 2e-3);
    }

    #[test]
    fn mlp_reference_is_act_of_fc1_plus_bias_then_fc2() {
        // M=1 Din=2 Dff=2 Dout=2: W1 = I, b1 = [1, -1], W2 = I, act = GELU.
        //   fc1 = x + b1 = [1+1, 2-1] = [2, 1] -> act = [Φ(2)·2, Φ(1)·1]
        //   out = act (W2 identity)
        let x = vals(&[1.0, 2.0]);
        let w1 = vals(&[1.0, 0.0, 0.0, 1.0]);
        let b1 = vals(&[1.0, -1.0]);
        let w2 = vals(&[1.0, 0.0, 0.0, 1.0]);
        let mut out = vec![f16::ZERO; 2];
        mlp_ref(
            1,
            2,
            2,
            2,
            &x,
            &w1,
            Some(&b1),
            &w2,
            Activation::Gelu,
            &mut out,
        );
        let jitter =
            f16::from_f64(0.5 * 2.0 * (1.0 + crate::ops::activation::erf64(2.0 * FRAC_1_SQRT_2)));
        let got = f64s(&out);
        assert!(
            (got[0] - jitter.to_f64()).abs() < 1e-2,
            "{} vs {jitter}",
            got[0]
        );
        assert!((got[1] - 0.841_345).abs() < 3e-3);

        // No bias: fc1 = x -> act(x).
        let mut out = vec![f16::ZERO; 2];
        mlp_ref(1, 2, 2, 2, &x, &w1, None, &w2, Activation::Gelu, &mut out);
        let got = f64s(&out);
        assert!((got[0] - 0.841_345).abs() < 3e-3);
        assert!((got[1] - 1.954_500).abs() < 6e-3);

        // A non-identity fc2 mixes the activated lanes.
        let w2 = vals(&[0.0, 1.0, 1.0, 0.0]);
        let mut out = vec![f16::ZERO; 2];
        mlp_ref(1, 2, 2, 2, &x, &w1, None, &w2, Activation::Gelu, &mut out);
        let got = f64s(&out);
        assert!((got[0] - 1.954_500).abs() < 6e-3);
        assert!((got[1] - 0.841_345).abs() < 3e-3);
    }

    #[test]
    fn mlp_reference_narrows_the_activation_before_fc2() {
        // The C's reference stores act[] as fp16 (unlike ffn_ref's f64 prod), so the
        // second projection sees the narrowed value. A single wide output makes the
        // difference visible: Dff=1, W2 = [[1000]] with an activation an fp16 ULP away.
        let x = vals(&[0.3]);
        let w1 = vals(&[1.0]);
        let w2 = vals(&[1000.0]);
        let mut out = vec![f16::ZERO; 1];
        mlp_ref(1, 1, 1, 1, &x, &w1, None, &w2, Activation::Gelu, &mut out);
        let act =
            f16::from_f64(0.5 * 0.3 * (1.0 + crate::ops::activation::erf64(0.3 * FRAC_1_SQRT_2)));
        assert_eq!(out[0], f16::from_f32(act.to_f32() * 1000.0));
    }

    #[test]
    fn buffers_are_validated() {
        let dev = match Device::open() {
            Ok(d) => d,
            Err(_) => return, // no device on this host; the refs are covered above
        };
        let x = vec![f16::ZERO; 8];
        let w = vec![f16::ZERO; 8];
        let mut out = vec![f16::ZERO; 8];
        assert!(ffn(&dev, 0, 2, 2, &x, &w, &w, &w, Activation::Gelu, &mut out).is_err());
        assert!(
            ffn(
                &dev,
                2,
                2,
                2,
                &x,
                &w[..3],
                &w,
                &w,
                Activation::Gelu,
                &mut out
            )
            .is_err()
        );
        assert!(geglu(&dev, &x, &x[..4], Activation::Gelu, &mut out).is_err());
        assert!(
            ffn_fused(
                &dev,
                2,
                2,
                2,
                &x,
                &w[..3],
                &w,
                &w,
                Activation::Gelu,
                &mut out
            )
            .is_err()
        );
        assert!(
            mlp(
                &dev,
                2,
                2,
                2,
                2,
                &x,
                &w[..3],
                None,
                &w,
                Activation::Gelu,
                &mut out
            )
            .is_err()
        );
        assert!(
            mlp_fused(
                &dev,
                2,
                2,
                2,
                2,
                &x,
                &w,
                Some(&w[..1]),
                &w,
                Activation::Gelu,
                &mut out
            )
            .is_err()
        );
    }

    #[test]
    fn pinned_tiling_is_the_c_contract() {
        // A shape that matches: M=4, H=64, I=512 -> Nt=Kt=256, nNt=nKt=2.
        let plg = Plan::pinned(4, 64, 512, Some(TILE), None).unwrap();
        let pld = Plan::pinned(4, 512, 64, None, Some(TILE)).unwrap();
        assert_eq!(plg.nt, TILE);
        assert_eq!(pld.kt, TILE);
        assert_eq!(plg.nnt, pld.nkt);
        assert_eq!(plg.out_slot, pld.in_slot);
        assert_eq!(plg.nmt, pld.nmt);
        assert_eq!(plg.mt, pld.mt);
        // The pin must be a multiple of 32 and within the real extent.
        assert!(Plan::pinned(4, 64, 512, Some(100), None).is_err());
        assert!(Plan::pinned(4, 64, 512, Some(1024), None).is_err());
        assert!(Plan::pinned(4, 64, 512, None, Some(1024)).is_err());
        // K/N/M alignment still applies (M==1 is the caller's pad).
        assert!(Plan::pinned(1, 64, 512, Some(TILE), None).is_err());
        assert!(Plan::pinned(4, 48, 512, Some(TILE), None).is_err());
    }
}
