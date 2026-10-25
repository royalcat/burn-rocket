//! On-device self-test for the M8a/M8b convolution, pool and resize port.
//!
//! ```text
//! conv_selftest [--conv] [--dw] [--act] [--conv1d] [--transpose]
//!               [--int8] [--int8-dw] [--pool] [--resize] [--all]
//!               [--reps R]
//!               [--shape IC IH IW OC KH KW [SY SX PT PL DY DX]]
//!               [--dw-shape C IH IW KH KW [SY SX PT PL DY DX]]
//!               [--conv1d-shape IC IT OC KW STRIDE PAD]
//!               [--transpose-shape IC IH IW OC KH KW [SY SX PT PL DY DX OPADY OPADX DW]]
//! ```
//!
//! Default: every family. Each case prints the wall time per rep and
//! cosine / max-abs / max-rel against the device-free reference, then PASS/FAIL.
//! Exit code 1 on a numeric failure, 2 when the NPU cannot be opened.
//!
//! # Gates
//!
//! The direct and depthwise convs are the hardware-validated single job, so the gate is
//! the C header's "relative ~2e-3" on the significant elements plus a cosine floor; the
//! **exact element count is always printed** so a tint of drift is visible. The
//! conv→activation fusion differs from `f(conv)` only through the DPU LUT's fp16
//! interpolation, so its relative gate is `5e-3` (as the C header says, gate the fused f
//! with a tolerance, not equality). Transposed convs gate on the *scatter-add* reference,
//! which is route-independent, so a wrong lowering or a wrong hardware route shows up as
//! a large error rather than last-bit noise.
//!
//! # Raw-file A/B against the C library
//!
//! ```text
//! conv_selftest --io conv        IC IH IW OC KH KW SY SX PT PL DY DX DW  IN W OUT
//! conv_selftest --io conv_act    IC IH IW OC KH KW SY SX PT PL DY DX DW KIND  IN W OUT
//! conv_selftest --io conv1d      IC IT OC KW STRIDE PAD  IN W OUT
//! conv_selftest --io conv_t      IC IH IW OC KH KW SY SX PT PL DY DX OPADY OPADX DW  IN W OUT
//! ```
//!
//! Every tensor is raw little-endian fp16. `W` is `[OC][IC][KH][KW]` for a direct conv
//! (`[C][KH][KW]` depthwise), `[OC][IC][KW]` for conv1d and `[IC][OC][KH][KW]` for a
//! direct transposed conv (`[C][KH][KW]` depthwise). This is the exchange format of
//! `tools/c-oracle/ab_board.c` on the board.

use std::process::ExitCode;
use std::time::Instant;

use half::f16;
use rocket_npu::Device;
use rocket_npu::ops::activation::Activation;
use rocket_npu::ops::conv::{
    Conv2dDesc, ConvContext, ConvInt8Pool, conv1d, conv2d, conv2d_act, conv2d_dw_int8,
    conv2d_dw_int8_model, conv2d_int8, conv2d_oh, conv2d_ow, conv2d_ref, conv2d_ref_int8,
};
use rocket_npu::ops::conv_transpose::{
    ConvTranspose2dDesc, conv_transpose2d, conv_transpose2d_oh, conv_transpose2d_ow,
    conv_transpose2d_plan_code, conv_transpose2d_ref, conv_transpose2d_route,
};
use rocket_npu::ops::pool::{
    PoolDesc, pool_fp16, pool_fp16_plan, pool_int8, pool_ref_fp16, pool_ref_int8, pool_ref_uint8,
    pool_uint8,
};
use rocket_npu::ops::resize::{
    upsample_bilinear, upsample_bilinear_ref, upsample_nearest, upsample_nearest_ref,
};
use rocket_npu::regcmd::pool::PoolMethod;

/// The C header's direct/depthwise fp16 conv gate.
const CONV_REL: f64 = 2e-3;
/// The fused conv→activation gate (the DPU LUT interpolation).
const ACT_REL: f64 = 5e-3;
/// Cosine floor for every case.
const COS_MIN: f64 = 0.9999;

/// cosine, max-abs, max-rel-of-significant-elements, peak and the worst-abs index.
///
/// `max_rel` is taken over the elements whose reference magnitude is at least a tenth of
/// the array's peak: a conv output near zero is a cancellation of O(1) products, where
/// the plain element-wise relative error says nothing and the absolute one is bounded by
/// the fp16 grain.
fn stats(want: &[f64], got: &[f64]) -> (f64, f64, f64, f64, usize) {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    let (mut max_abs, mut worst, mut amax) = (0.0f64, 0usize, 0.0f64);
    for (i, (&w, &g)) in want.iter().zip(got).enumerate() {
        dot += w * g;
        na += w * w;
        nb += g * g;
        amax = amax.max(w.abs());
        let d = (w - g).abs();
        if d > max_abs {
            max_abs = d;
            worst = i;
        }
    }
    let floor = (0.1 * amax).max(1e-6);
    let mut max_rel = 0.0f64;
    for (&w, &g) in want.iter().zip(got) {
        if w.abs() >= floor {
            max_rel = max_rel.max((w - g).abs() / w.abs());
        }
    }
    let cosine = if na == 0.0 && nb == 0.0 {
        1.0
    } else if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    };
    (cosine, max_abs, max_rel, amax, worst)
}

fn as_f64(v: &[f16]) -> Vec<f64> {
    v.iter().map(|x| x.to_f64()).collect()
}

/// A deterministic LCG fill in `[lo, hi]` (the same generator the other self-tests use).
fn fill(n: usize, lo: f32, hi: f32, seed: u32) -> Vec<f16> {
    let mut st = 0x9e37_79b9u32 ^ seed;
    (0..n)
        .map(|_| {
            st = st.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let u = ((st >> 8) & 0xffff) as f32 / 65535.0;
            f16::from_f32(lo + (hi - lo) * u)
        })
        .collect()
}

/// Time `reps` device runs of one case, keeping the last result (milliseconds per rep).
fn timed<T>(reps: usize, mut run: impl FnMut() -> T) -> (T, f64) {
    let t0 = Instant::now();
    let mut last = run();
    for _ in 1..reps {
        last = run();
    }
    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;
    (last, wall_ms)
}

/// Print one case's metrics and gate it.
fn report(name: &str, wall_ms: f64, want: &[f16], got: &[f16], rel_max: f64) -> bool {
    let (w, g) = (as_f64(want), as_f64(got));
    let (cos, abs, rel, peak, worst) = stats(&w, &g);
    let exact = want.iter().zip(got).filter(|(a, b)| a == b).count();
    let pass = cos >= COS_MIN && rel <= rel_max;
    println!(
        "  {name:<40} {wall_ms:9.1} ms  cos={cos:.9}  max_abs={abs:.3e}  \
         max_rel(sig)={rel:.3e}  peak={peak:.3} (worst #{worst})  exact={exact}/{}  -> {}",
        want.len(),
        if pass { "PASS" } else { "FAIL" }
    );
    pass
}

// ── Direct conv ───────────────────────────────────────────────────────────────

/// `(ic, ih, iw, oc, kh, kw, sy, sx, pt, pl, dy, dx)`.
type ConvShape = (
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
);

fn conv_desc(s: ConvShape) -> Conv2dDesc {
    let (ic, ih, iw, oc, kh, kw, sy, sx, pt, pl, dy, dx) = s;
    let mut d = Conv2dDesc::new(ic, ih, iw, oc, kh, kw);
    d.stride_y = sy;
    d.stride_x = sx;
    d.pad_top = pt;
    d.pad_left = pl;
    d.dil_y = dy;
    d.dil_x = dx;
    d
}

fn run_conv(dev: &Device, s: ConvShape, reps: usize, ctx: Option<&mut ConvContext>) -> bool {
    let d = conv_desc(s);
    let (oh, ow) = (conv2d_oh(&d), conv2d_ow(&d));
    let n_in = d.ic * d.ih * d.iw;
    let n_w = d.oc * d.ic * d.kh * d.kw;
    let n_out = d.oc * oh * ow;
    let input = fill(n_in, -1.5, 1.5, 0x11 + (n_in % 9973) as u32);
    // The spread keeps a 32-channel contraction at a peak around 1 for the rel gate.
    let w = fill(n_w, -0.5, 0.5, 0x12 + (n_w % 9973) as u32);
    let mut out = vec![f16::ZERO; n_out];
    let mut want = vec![f16::ZERO; n_out];

    let resident = ctx.is_some();
    let (r, ms) = match ctx {
        Some(ctx) => timed(reps, || ctx.conv2d(&d, &input, &w, &mut out)),
        None => timed(reps, || conv2d(dev, &d, &input, &w, &mut out)),
    };
    if let Err(e) = r {
        println!("  conv {}x{}x{}: {e}", s.0, s.1, s.2);
        return false;
    }
    conv2d_ref(&d, &input, &w, &mut want);
    let tag = if resident { "conv ctx" } else { "conv" };
    let name = format!(
        "{tag} {}x{}x{} k{}x{}s{} p{} d{} oc{}",
        s.0, s.1, s.2, s.4, s.5, s.6, s.8, s.10, s.3
    );
    report(&name, ms, &want, &out, CONV_REL)
}

// ── Depthwise ─────────────────────────────────────────────────────────────────

/// `(c, ih, iw, kh, kw, sy, sx, pt, pl, dy, dx)`.
type DwShape = (
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
);

fn dw_desc(s: DwShape) -> Conv2dDesc {
    let (c, ih, iw, kh, kw, sy, sx, pt, pl, dy, dx) = s;
    let mut d = Conv2dDesc::new(c, ih, iw, c, kh, kw);
    d.stride_y = sy;
    d.stride_x = sx;
    d.pad_top = pt;
    d.pad_left = pl;
    d.dil_y = dy;
    d.dil_x = dx;
    d.depthwise = true;
    d
}

fn run_dw(dev: &Device, s: DwShape, reps: usize) -> bool {
    let d = dw_desc(s);
    let (oh, ow) = (conv2d_oh(&d), conv2d_ow(&d));
    let c = s.0;
    let input = fill(c * d.ih * d.iw, -1.5, 1.5, 0x21 + (c % 9973) as u32);
    let w = fill(c * d.kh * d.kw, -1.0, 1.0, 0x22 + (c % 9973) as u32);
    let mut out = vec![f16::ZERO; c * oh * ow];
    let mut want = vec![f16::ZERO; c * oh * ow];
    let (r, ms) = timed(reps, || conv2d(dev, &d, &input, &w, &mut out));
    if let Err(e) = r {
        println!("  dw {:?}: {e}", s);
        return false;
    }
    conv2d_ref(&d, &input, &w, &mut want);
    let name = format!(
        "dw {}x{}x{} k{}x{}s{} p{}",
        s.0, s.1, s.2, s.3, s.4, s.5, s.7
    );
    report(&name, ms, &want, &out, CONV_REL)
}

// ── conv1d ────────────────────────────────────────────────────────────────────

/// `(ic, it, oc, kw, stride, pad)`.
type Conv1dShape = (usize, usize, usize, usize, usize, usize);

fn run_conv1d(dev: &Device, s: Conv1dShape, reps: usize) -> bool {
    let (ic, it, oc, kw, stride, pad) = s;
    let mut d = Conv2dDesc::new(ic, it, 1, oc, kw, 1);
    d.stride_y = stride;
    d.pad_top = pad;
    let ot = conv2d_oh(&d);
    let input = fill(ic * it, -1.5, 1.5, 0x31 + (ic * 31 + it) as u32);
    let w = fill(oc * ic * kw, -0.5, 0.5, 0x32 + (oc * 31 + kw) as u32);
    let mut out = vec![f16::ZERO; oc * ot];
    let mut want = vec![f16::ZERO; oc * ot];
    let (r, ms) = timed(reps, || {
        conv1d(dev, ic, it, oc, kw, stride, pad, &input, &w, &mut out)
    });
    if let Err(e) = r {
        println!("  conv1d {:?}: {e}", s);
        return false;
    }
    conv2d_ref(&d, &input, &w, &mut want);
    let name = format!("conv1d {ic}x{it} k{kw}s{stride} p{pad} oc{oc}");
    report(&name, ms, &want, &out, CONV_REL)
}

// ── conv → activation ─────────────────────────────────────────────────────────

fn run_act(dev: &Device, s: ConvShape, kind: Activation, reps: usize) -> bool {
    let d = conv_desc(s);
    let (oh, ow) = (conv2d_oh(&d), conv2d_ow(&d));
    let input = fill(
        d.ic * d.ih * d.iw,
        -2.0,
        2.0,
        0x41 + (s.1 * 31 + s.2) as u32,
    );
    let w = fill(
        d.oc * d.ic * d.kh * d.kw,
        -0.5,
        0.5,
        0x42 + (s.3 * 31 + s.4) as u32,
    );
    let n_out = d.oc * oh * ow;
    let mut out = vec![f16::ZERO; n_out];
    let mut conv_out = vec![f16::ZERO; n_out];
    let mut want = vec![f16::ZERO; n_out];
    let (r, ms) = timed(reps, || conv2d_act(dev, &d, kind, &input, &w, &mut out));
    if let Err(e) = r {
        println!("  act {kind:?} {}x{}x{}: {e}", s.0, s.1, s.2);
        return false;
    }
    conv2d_ref(&d, &input, &w, &mut conv_out);

    // Two references, because the fused epilogue is the C's **single-pass** table
    // (`rocket_lut_epilogue_build`), not the accurate two-pass activation:
    //
    //   * `model` = `lut_apply` of that table on the conv output — the host model of the
    //     chain the epilogue programs, so this is what gates the WIRING (a wrong BN scale,
    //     OUT_CVT or table upload shows up here immediately);
    //   * `exact` = `activation_ref` (the true f) — the table's own approximation error,
    //     reported but not gated: the C header calls the fused f "an fp16 LUT
    //     approximation — gate with a tolerance", and its flat negative tail saturates.
    //     The board A/B writes bit-identical values to the C library for all three kinds.
    let mut lut = rocket_npu::ops::activation::LutTable::zeroed();
    let ep = match rocket_npu::ops::activation::lut_epilogue_build(kind, &mut lut) {
        Ok(ep) => ep,
        Err(e) => {
            println!("  act {kind:?}: {e}");
            return false;
        }
    };
    rocket_npu::ops::activation::activation_ref(kind, &conv_out, &mut want);
    let mut model = vec![f16::ZERO; n_out];
    for (m, c) in model.iter_mut().zip(&conv_out) {
        *m = f16::from_f32(rocket_npu::ops::activation::lut_apply(
            &ep,
            &lut,
            c.to_f32(),
        ));
    }

    let (w64, g64) = (as_f64(&model), as_f64(&out));
    let (cos, abs, rel, peak, worst) = stats(&w64, &g64);
    let n_exact = model.iter().zip(&out).filter(|(a, b)| a == b).count();
    let pass = rel <= ACT_REL;
    let name = format!(
        "act {kind:?} {}x{}x{} k{}x{} oc{}",
        s.0, s.1, s.2, s.4, s.5, s.3
    );
    println!(
        "  {name:<40} {ms:9.1} ms  cos={cos:.9}  max_abs={abs:.3e}  max_rel(sig)={rel:.3e}  \
         peak={peak:.3} (worst #{worst})  exact={n_exact}/{}  -> {}",
        model.len(),
        if pass { "PASS" } else { "FAIL" },
    );
    let (cos_e, abs_e, rel_e, _, _) = stats(&as_f64(&want), &as_f64(&model));
    println!(
        "      (device-vs-model max_abs is the C's documented single-pass QUIRK 1: the x≈0 \n\
         \x20      mux spike (~8) and, for gelu, the saturated tail (up to 128). librocketnpu\n\
         \x20      writes those values identically — the board A/B is bit-identical. Table vs\n\
         \x20      exact f: cos={cos_e:.6} max_abs={abs_e:.3e} max_rel(sig)={rel_e:.3e}, not gated)"
    );
    pass
}

// ── Transposed conv ───────────────────────────────────────────────────────────

/// `(ic, ih, iw, oc, kh, kw, sy, sx, pt, pl, opad_y, opad_x, depthwise)`.
type TransposeShape = (
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    bool,
);

fn transpose_desc(s: TransposeShape) -> ConvTranspose2dDesc {
    let (ic, ih, iw, oc, kh, kw, sy, sx, pt, pl, oy, ox, dw) = s;
    let mut d = ConvTranspose2dDesc::new(ic, ih, iw, oc, kh, kw);
    d.stride_y = sy;
    d.stride_x = sx;
    d.pad_top = pt;
    d.pad_left = pl;
    d.opad_y = oy;
    d.opad_x = ox;
    d.depthwise = dw;
    d
}

/// A compact one-line description (a 13-tuple has no `Debug`).
fn tdesc(s: &TransposeShape) -> String {
    format!(
        "{}x{}x{} k{}x{}s{} op{} dw{}",
        s.0, s.1, s.2, s.4, s.5, s.6, s.10, s.12 as u8
    )
}

fn transpose_weights(s: TransposeShape) -> usize {
    let (ic, _ih, _iw, oc, kh, kw, .., dw) = s;
    if dw { ic * kh * kw } else { ic * oc * kh * kw }
}

fn run_transpose(dev: &Device, s: TransposeShape, reps: usize) -> bool {
    let d = transpose_desc(s);
    let code = conv_transpose2d_plan_code(&d);
    if code != 0 {
        println!("  transpose {}: refused (plan {code})", tdesc(&s));
        return false;
    }
    let route = match conv_transpose2d_route(&d) {
        Ok(r) => r,
        Err(e) => {
            println!("  transpose {}: {e}", tdesc(&s));
            return false;
        }
    };
    let oh = conv_transpose2d_oh(&d) as usize;
    let ow = conv_transpose2d_ow(&d) as usize;
    let input = fill(
        d.ic * d.ih * d.iw,
        -1.5,
        1.5,
        0x51 + (s.1 * 31 + s.2) as u32,
    );
    let w = fill(
        transpose_weights(s),
        -0.5,
        0.5,
        0x52 + (s.3 * 31 + s.4) as u32,
    );
    let mut out = vec![f16::ZERO; d.oc * oh * ow];
    let mut want = vec![f16::ZERO; d.oc * oh * ow];
    let (r, ms) = timed(reps, || conv_transpose2d(dev, &d, &input, &w, &mut out));
    if let Err(e) = r {
        println!("  transpose {}: {e}", tdesc(&s));
        return false;
    }
    conv_transpose2d_ref(&d, &input, &w, &mut want);
    let name = format!(
        "transpose {} {}x{}x{} k{}x{}s{} op{}",
        route.name(),
        s.0,
        s.1,
        s.2,
        s.4,
        s.5,
        s.6,
        s.10
    );
    report(&name, ms, &want, &out, CONV_REL)
}

// ── int8 direct conv ──────────────────────────────────────────────────────────

/// A deterministic int8 fill over the full `[-128, 127]` range.
fn fill_i8(n: usize, seed: u32) -> Vec<i8> {
    let mut st = 0x9e37_79b9u32 ^ seed;
    (0..n)
        .map(|_| {
            st = st.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (((st >> 16) & 0xff) as i32 - 128) as i8
        })
        .collect()
}

/// A deterministic int32 fill in `[lo, hi]` (the bias cubes).
fn fill_i32(n: usize, lo: i32, hi: i32, seed: u32) -> Vec<i32> {
    let mut st = 0x9e37_79b9u32 ^ seed;
    (0..n)
        .map(|_| {
            st = st.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let u = ((st >> 8) & 0xffff) as i64;
            (lo as i64 + (hi as i64 - lo as i64) * u / 65535) as i32
        })
        .collect()
}

/// Print one integer case: exact count, the worst absolute difference, PASS/FAIL.
///
/// `tol` is 0 for the exact gates (the int8 direct accumulator and the int8 depthwise
/// requant model) and 1 where only a rounding step can differ.
fn report_int(name: &str, wall_ms: f64, want: &[i64], got: &[i64], tol: i64) -> bool {
    let exact = want.iter().zip(got).filter(|(a, b)| a == b).count();
    let mut worst = 0i64;
    for (a, b) in want.iter().zip(got) {
        worst = worst.max((a - b).abs());
    }
    let pass = worst <= tol;
    println!(
        "  {name:<40} {wall_ms:9.1} ms  max_abs={worst}  exact={exact}/{}  -> {}",
        want.len(),
        if pass { "PASS" } else { "FAIL" }
    );
    pass
}

fn run_int8(dev: &Device, s: ConvShape, reps: usize, ctx: Option<&mut ConvContext>) -> bool {
    let d = conv_desc(s);
    let (oh, ow) = (conv2d_oh(&d), conv2d_ow(&d));
    let n_in = d.ic * d.ih * d.iw;
    let n_w = d.oc * d.ic * d.kh * d.kw;
    let input = fill_i8(n_in, 0x61 + (n_in % 9973) as u32);
    let w = fill_i8(n_w, 0x62 + (n_w % 9973) as u32);
    let mut out = vec![0i32; d.oc * oh * ow];
    let mut want = vec![0i32; d.oc * oh * ow];

    let resident = ctx.is_some();
    let (r, ms) = match ctx {
        Some(ctx) => timed(reps, || ctx.conv2d_int8(&d, &input, &w, &mut out)),
        None => timed(reps, || conv2d_int8(dev, &d, &input, &w, &mut out)),
    };
    if let Err(e) = r {
        println!("  int8 conv {}x{}x{}: {e}", s.0, s.1, s.2);
        return false;
    }
    conv2d_ref_int8(&d, &input, &w, &mut want);
    let name = format!(
        "{}{} {}x{}x{} k{}x{}s{} p{} oc{}",
        if resident {
            "int8 conv ctx"
        } else {
            "int8 conv"
        },
        "",
        s.0,
        s.1,
        s.2,
        s.4,
        s.5,
        s.6,
        s.8,
        s.3
    );
    let w64: Vec<i64> = want.iter().map(|&v| v as i64).collect();
    let g64: Vec<i64> = out.iter().map(|&v| v as i64).collect();
    report_int(&name, ms, &w64, &g64, 0)
}

/// `rocket_conv2d_int8_mt` vs the serial entry: the pool must be **bit-identical**
/// (same tiles, same single jobs; only the scheduling differs).
fn run_int8_mt(dev: &Device, s: ConvShape, workers: usize, reps: usize) -> bool {
    let d = conv_desc(s);
    let (oh, ow) = (conv2d_oh(&d), conv2d_ow(&d));
    let n_in = d.ic * d.ih * d.iw;
    let n_w = d.oc * d.ic * d.kh * d.kw;
    let n_out = d.oc * oh * ow;
    let input = fill_i8(n_in, 0xB1 + (n_in % 9973) as u32);
    let w = fill_i8(n_w, 0xB2 + (n_w % 9973) as u32);
    let mut serial = vec![0i32; n_out];
    if let Err(e) = conv2d_int8(dev, &d, &input, &w, &mut serial) {
        println!("  int8 mt (serial) {}x{}x{}: {e}", s.0, s.1, s.2);
        return false;
    }
    let mut pool = match ConvInt8Pool::new(workers) {
        Ok(p) => p,
        Err(e) => {
            println!("  int8 mt pool: {e}");
            return false;
        }
    };
    let mut par = vec![0i32; n_out];
    let (r, ms) = timed(reps, || pool.conv2d(&d, &input, &w, &mut par));
    if let Err(e) = r {
        println!("  int8 mt {}x{}x{}: {e}", s.0, s.1, s.2);
        return false;
    }
    let mut want = vec![0i32; n_out];
    conv2d_ref_int8(&d, &input, &w, &mut want);
    let exact = par.iter().zip(&serial).filter(|(a, b)| a == b).count();
    let model_exact = par.iter().zip(&want).filter(|(a, b)| a == b).count();
    let same = exact == n_out && model_exact == n_out;
    println!(
        "  {:<40} {ms:9.1} ms  workers={} exact-vs-serial={exact}/{n_out}          exact-vs-ref={model_exact}/{n_out}  -> {}",
        format!("int8 mt {}x{}x{} oc{}", s.0, s.1, s.2, s.3),
        pool.len(),
        if same { "PASS" } else { "FAIL" }
    );
    same
}

// ── int8 depthwise (int8-out, on-chip requant) ────────────────────────────────

/// The per-tensor quant of the C self-test's capture (in/w/out scales), which keeps the
/// OUT_CVT derivation's `truncate_bits` at zero.
const DW_IN_SCALE: f32 = 0.0213;
const DW_W_SCALE: f32 = 0.00347;
const DW_OUT_SCALE: f32 = 0.0925;

fn run_dw_int8(
    dev: &Device,
    s: DwShape,
    in_zp: i32,
    out_zp: i32,
    with_bias: bool,
    reps: usize,
) -> bool {
    let d = dw_desc(s);
    let (oh, ow) = (conv2d_oh(&d), conv2d_ow(&d));
    let c = s.0;
    let input = fill_i8(c * d.ih * d.iw, 0x71 + (c % 9973) as u32);
    let w = fill_i8(c * d.kh * d.kw, 0x72 + (c % 9973) as u32);
    let bias = with_bias.then(|| fill_i32(c, -4000, 4000, 0x73 + c as u32));
    let mut out = vec![0i8; c * oh * ow];
    let mut want = vec![0i8; c * oh * ow];
    let (r, ms) = timed(reps, || {
        conv2d_dw_int8(
            dev,
            &d,
            &input,
            &w,
            bias.as_deref(),
            DW_IN_SCALE,
            DW_W_SCALE,
            DW_OUT_SCALE,
            in_zp,
            0,
            out_zp,
            &mut out,
        )
    });
    if let Err(e) = r {
        println!("  int8 dw {:?}: {e}", s);
        return false;
    }
    conv2d_dw_int8_model(
        &d,
        &input,
        &w,
        bias.as_deref(),
        DW_IN_SCALE,
        DW_W_SCALE,
        DW_OUT_SCALE,
        in_zp,
        out_zp,
        &mut want,
    );
    let name = format!(
        "int8 dw {}x{}x{} k{}x{}s{} zp {}/{} {}",
        s.0,
        s.1,
        s.2,
        s.3,
        s.4,
        s.5,
        in_zp,
        out_zp,
        if with_bias { "bias" } else { "nobias" }
    );
    let w64: Vec<i64> = want.iter().map(|&v| v as i64).collect();
    let g64: Vec<i64> = out.iter().map(|&v| v as i64).collect();
    report_int(&name, ms, &w64, &g64, 0)
}

// ── Pooling ───────────────────────────────────────────────────────────────────

/// One pooling/upsample case.
///
/// `tol` is 0 for the exact gates (max pooling, nearest upsample). An average carries
/// the PPU reciprocal's fp16 quantization and the fp16 output narrowing, so its gate
/// is an **absolute** bound scaled by the array peak: a per-element relative error is
/// meaningless near zero, where the value is a cancellation of large window terms.
fn report_pool(name: &str, wall_ms: f64, want: &[f16], got: &[f16], tol: f64) -> bool {
    let (w, g) = (as_f64(want), as_f64(got));
    let (cos, abs, rel, peak, worst) = stats(&w, &g);
    let exact = want.iter().zip(got).filter(|(a, b)| a == b).count();
    let pass = if tol == 0.0 {
        exact == want.len()
    } else {
        cos >= COS_MIN && abs <= tol * peak.max(1e-6)
    };
    println!(
        "  {name:<40} {wall_ms:9.1} ms  cos={cos:.9}  max_abs={abs:.3e}  \
         max_rel(sig)={rel:.3e}  peak={peak:.3} (worst #{worst})  exact={exact}/{}  -> {}",
        want.len(),
        if pass { "PASS" } else { "FAIL" }
    );
    pass
}

fn run_pool(dev: &Device, d: PoolDesc, reps: usize) -> bool {
    let (oh, ow) = (d.oh() as usize, d.ow() as usize);
    let n_in = d.c * d.ih * d.iw;
    let input = fill(n_in, -3.0, 3.0, 0x81 + (n_in % 9973) as u32);
    let mut out = vec![f16::ZERO; d.c * oh * ow];
    let mut want = vec![f16::ZERO; d.c * oh * ow];
    let (r, ms) = timed(reps, || pool_fp16(dev, &d, &input, &mut out));
    if let Err(e) = r {
        println!("  pool: {e}");
        return false;
    }
    pool_ref_fp16(&d, &input, &mut want);
    let name = format!(
        "pool {:?} {}x{}x{} k{}x{}s{}x{} p{}",
        d.method, d.c, d.ih, d.iw, d.kh, d.kw, d.stride_y, d.stride_x, d.pad_top
    );
    // Max is idempotent with no reciprocal (bit-exact); an average carries the ppu
    // reciprocal's fp16 rounding.
    let tol = if d.method == PoolMethod::Max {
        0.0
    } else {
        2e-3
    };
    report_pool(&name, ms, &want, &out, tol)
}

fn run_pool_int8(dev: &Device, d: PoolDesc, reps: usize, uint8: bool) -> bool {
    let (oh, ow) = (d.oh() as usize, d.ow() as usize);
    let n_in = d.c * d.ih * d.iw;
    let n_out = d.c * oh * ow;
    let input = fill_i8(n_in, 0x91 + (n_in % 9973) as u32);
    let mut out = vec![0i8; n_out];
    let mut want = vec![0i8; n_out];
    let max = d.method == PoolMethod::Max;
    let (r, ms) = timed(reps, || {
        if uint8 {
            // The uint8 entry takes bytes; the same bit patterns are what the int8
            // entry sees, so one conversion pair covers both.
            let src: Vec<u8> = input.iter().map(|&v| v as u8).collect();
            let mut dst = vec![0u8; n_out];
            let r = pool_uint8(dev, &d, &src, &mut dst);
            for (o, v) in out.iter_mut().zip(&dst) {
                *o = *v as i8;
            }
            r
        } else {
            pool_int8(dev, &d, &input, &mut out)
        }
    });
    if let Err(e) = r {
        println!("  pool int8: {e}");
        return false;
    }
    if uint8 {
        let src: Vec<u8> = input.iter().map(|&v| v as u8).collect();
        let mut dst = vec![0u8; n_out];
        pool_ref_uint8(&d, &src, &mut dst);
        for (w, v) in want.iter_mut().zip(&dst) {
            *w = *v as i8;
        }
    } else {
        pool_ref_int8(&d, &input, &mut want);
    }
    let name = format!(
        "pool {} {:?} {}x{}x{} k{}x{}s{}x{} p{}",
        if uint8 { "u8" } else { "i8" },
        d.method,
        d.c,
        d.ih,
        d.iw,
        d.kh,
        d.kw,
        d.stride_y,
        d.stride_x,
        d.pad_top
    );
    let w64: Vec<i64> = want.iter().map(|&v| v as i64).collect();
    let g64: Vec<i64> = out.iter().map(|&v| v as i64).collect();
    // Max is exact integer max; an average can differ by one where the fp16 sum times
    // the reciprocal lands on a rounding boundary (the C gates it the same way).
    report_int(&name, ms, &w64, &g64, if max { 0 } else { 1 })
}

// ── Resize ────────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn run_resize(
    dev: &Device,
    c: usize,
    ih: usize,
    iw: usize,
    sy: usize,
    sx: usize,
    bilinear: bool,
    reps: usize,
) -> bool {
    let input = fill(c * ih * iw, -1.5, 1.5, 0xA1 + (c * 31 + ih) as u32);
    let mut out = vec![f16::ZERO; c * ih * sy * iw * sx];
    let mut want = vec![f16::ZERO; c * ih * sy * iw * sx];
    let (r, ms) = timed(reps, || {
        if bilinear {
            upsample_bilinear(dev, &input, &mut out, c, ih, iw, sy, sx)
        } else {
            upsample_nearest(dev, &input, &mut out, c, ih, iw, sy, sx)
        }
    });
    if let Err(e) = r {
        println!("  resize: {e}");
        return false;
    }
    if bilinear {
        upsample_bilinear_ref(&input, &mut want, c, ih, iw, sy, sx);
    } else {
        upsample_nearest_ref(&input, &mut want, c, ih, iw, sy, sx);
    }
    let name = format!(
        "{} {}x{}x{} s{}x{}",
        if bilinear { "bilinear" } else { "nearest" },
        c,
        ih,
        iw,
        sy,
        sx
    );
    // Nearest is exact block replication; bilinear carries the fp16 triangle kernel's
    // rounding (the C header's own gate).
    let tol = if bilinear { 2e-3 } else { 0.0 };
    report_pool(&name, ms, &want, &out, tol)
}

// ── Raw-file A/B mode ─────────────────────────────────────────────────────────
fn num(args: &[String], i: usize) -> Result<usize, String> {
    args.get(i)
        .ok_or_else(|| format!("missing argument #{i}"))?
        .parse::<usize>()
        .map_err(|_| format!("argument #{i} is not a positive integer"))
}

fn read_f16(path: &str) -> Result<Vec<f16>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    if !bytes.len().is_multiple_of(2) {
        return Err(format!("{path}: odd byte length"));
    }
    Ok(bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| f16::from_bits(u16::from_le_bytes(*c)))
        .collect())
}

fn write_f16(path: &str, data: &[f16]) -> Result<(), String> {
    let mut bytes = Vec::with_capacity(data.len() * 2);
    for v in data {
        bytes.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    std::fs::write(path, bytes).map_err(|e| format!("{path}: {e}"))
}

fn check_len(name: &str, got: &[f16], want: usize) -> Result<(), String> {
    if got.len() != want {
        return Err(format!("{name}: {} elements, expected {want}", got.len()));
    }
    Ok(())
}

/// A signed integer argument (the zero points are negative).
fn sint(args: &[String], i: usize) -> Result<i32, String> {
    args.get(i)
        .ok_or_else(|| format!("missing argument #{i}"))?
        .parse::<i32>()
        .map_err(|_| format!("argument #{i} is not an integer"))
}

/// A raw `f32` bit pattern in hex (the A/B's scale exchange, so both sides read the
/// identical value rather than reparsing a decimal).
fn f32_bits(args: &[String], i: usize) -> Result<f32, String> {
    let s = args
        .get(i)
        .ok_or_else(|| format!("missing argument #{i}"))?;
    let s = s.strip_prefix("0x").unwrap_or(s);
    let bits = u32::from_str_radix(s, 16).map_err(|_| format!("argument #{i} is not hex"))?;
    Ok(f32::from_bits(bits))
}

fn read_raw(path: &str) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|e| format!("{path}: {e}"))
}

fn read_i8(path: &str) -> Result<Vec<i8>, String> {
    Ok(read_raw(path)?.into_iter().map(|b| b as i8).collect())
}

fn read_i32(path: &str) -> Result<Vec<i32>, String> {
    let bytes = read_raw(path)?;
    if !bytes.len().is_multiple_of(4) {
        return Err(format!("{path}: length is not a multiple of 4"));
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| i32::from_le_bytes(*c))
        .collect())
}

fn write_i8(path: &str, data: &[i8]) -> Result<(), String> {
    let bytes: Vec<u8> = data.iter().map(|&v| v as u8).collect();
    std::fs::write(path, bytes).map_err(|e| format!("{path}: {e}"))
}

fn write_i32(path: &str, data: &[i32]) -> Result<(), String> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, bytes).map_err(|e| format!("{path}: {e}"))
}

fn check_len_i8(name: &str, got: &[i8], want: usize) -> Result<(), String> {
    if got.len() != want {
        return Err(format!("{name}: {} elements, expected {want}", got.len()));
    }
    Ok(())
}

fn check_len_i32(name: &str, got: &[i32], want: usize) -> Result<(), String> {
    if got.len() != want {
        return Err(format!("{name}: {} elements, expected {want}", got.len()));
    }
    Ok(())
}

/// The pool geometry arguments shared by the three pool A/B families.
///
/// `FAMILY METHOD C IH IW KH KW SY SX PT PL PB PR IN OUT`
fn io_pool(args: &[String]) -> Result<PoolDesc, String> {
    let method = match num(args, 1)? {
        0 => PoolMethod::Avg,
        1 => PoolMethod::Max,
        other => return Err(format!("bad pooling method {other}")),
    };
    let mut d = PoolDesc::new(
        num(args, 2)?,
        num(args, 3)?,
        num(args, 4)?,
        num(args, 5)?,
        num(args, 6)?,
    );
    d.stride_y = num(args, 7)?;
    d.stride_x = num(args, 8)?;
    d.pad_top = num(args, 9)?;
    d.pad_left = num(args, 10)?;
    d.pad_bottom = num(args, 11)?;
    d.pad_right = num(args, 12)?;
    d.method = method;
    Ok(d)
}

fn io_mode(dev: &Device, args: &[String]) -> Result<(), String> {
    let family = args.first().ok_or("missing --io family")?.as_str();
    let out_path = args.last().ok_or("missing OUT")?.as_str();
    match family {
        "conv" => {
            // conv IC IH IW OC KH KW SY SX PT PL DY DX DW IN W OUT
            let (ic, ih, iw, oc) = (num(args, 1)?, num(args, 2)?, num(args, 3)?, num(args, 4)?);
            let (kh, kw) = (num(args, 5)?, num(args, 6)?);
            let dw = num(args, 13)? != 0;
            let mut d = Conv2dDesc::new(ic, ih, iw, oc, kh, kw);
            d.stride_y = num(args, 7)?;
            d.stride_x = num(args, 8)?;
            d.pad_top = num(args, 9)?;
            d.pad_left = num(args, 10)?;
            d.dil_y = num(args, 11)?;
            d.dil_x = num(args, 12)?;
            d.depthwise = dw;
            let input = read_f16(args.get(14).ok_or("missing IN")?)?;
            let w = read_f16(args.get(15).ok_or("missing W")?)?;
            check_len("IN", &input, ic * ih * iw)?;
            check_len("W", &w, if dw { ic * kh * kw } else { oc * ic * kh * kw })?;
            let (oh, ow) = (conv2d_oh(&d), conv2d_ow(&d));
            let mut out = vec![f16::ZERO; oc * oh * ow];
            conv2d(dev, &d, &input, &w, &mut out).map_err(|e| e.to_string())?;
            write_f16(out_path, &out)
        }
        "conv_act" => {
            // conv_act IC IH IW OC KH KW SY SX PT PL DY DX DW KIND IN W OUT
            let (ic, ih, iw, oc) = (num(args, 1)?, num(args, 2)?, num(args, 3)?, num(args, 4)?);
            let (kh, kw) = (num(args, 5)?, num(args, 6)?);
            let dw = num(args, 13)? != 0;
            let code: i32 = args
                .get(14)
                .ok_or("missing KIND")?
                .parse()
                .map_err(|_| "KIND is not an activation code".to_string())?;
            let kind = Activation::from_code(code).ok_or(format!("unknown activation {code}"))?;
            let mut d = Conv2dDesc::new(ic, ih, iw, oc, kh, kw);
            d.stride_y = num(args, 7)?;
            d.stride_x = num(args, 8)?;
            d.pad_top = num(args, 9)?;
            d.pad_left = num(args, 10)?;
            d.dil_y = num(args, 11)?;
            d.dil_x = num(args, 12)?;
            d.depthwise = dw;
            let input = read_f16(args.get(15).ok_or("missing IN")?)?;
            let w = read_f16(args.get(16).ok_or("missing W")?)?;
            check_len("IN", &input, ic * ih * iw)?;
            check_len("W", &w, if dw { ic * kh * kw } else { oc * ic * kh * kw })?;
            let (oh, ow) = (conv2d_oh(&d), conv2d_ow(&d));
            let mut out = vec![f16::ZERO; oc * oh * ow];
            conv2d_act(dev, &d, kind, &input, &w, &mut out).map_err(|e| e.to_string())?;
            write_f16(out_path, &out)
        }
        "conv1d" => {
            // conv1d IC IT OC KW STRIDE PAD IN W OUT
            let (ic, it, oc, kw) = (num(args, 1)?, num(args, 2)?, num(args, 3)?, num(args, 4)?);
            let (stride, pad) = (num(args, 5)?, num(args, 6)?);
            let input = read_f16(args.get(7).ok_or("missing IN")?)?;
            let w = read_f16(args.get(8).ok_or("missing W")?)?;
            check_len("IN", &input, ic * it)?;
            check_len("W", &w, oc * ic * kw)?;
            let ot = (it + 2 * pad).saturating_sub(kw) / stride + 1;
            let mut out = vec![f16::ZERO; oc * ot];
            conv1d(dev, ic, it, oc, kw, stride, pad, &input, &w, &mut out)
                .map_err(|e| e.to_string())?;
            write_f16(out_path, &out)
        }
        "conv_t" => {
            // conv_t IC IH IW OC KH KW SY SX PT PL DY DX OPADY OPADX DW IN W OUT
            let (ic, ih, iw, oc) = (num(args, 1)?, num(args, 2)?, num(args, 3)?, num(args, 4)?);
            let (kh, kw) = (num(args, 5)?, num(args, 6)?);
            let dw = num(args, 15)? != 0;
            let mut d = ConvTranspose2dDesc::new(ic, ih, iw, oc, kh, kw);
            d.stride_y = num(args, 7)?;
            d.stride_x = num(args, 8)?;
            d.pad_top = num(args, 9)?;
            d.pad_left = num(args, 10)?;
            d.dil_y = num(args, 11)?;
            d.dil_x = num(args, 12)?;
            d.opad_y = num(args, 13)?;
            d.opad_x = num(args, 14)?;
            d.depthwise = dw;
            let input = read_f16(args.get(16).ok_or("missing IN")?)?;
            let w = read_f16(args.get(17).ok_or("missing W")?)?;
            check_len("IN", &input, ic * ih * iw)?;
            check_len("W", &w, if dw { ic * kh * kw } else { ic * oc * kh * kw })?;
            let oh = conv_transpose2d_oh(&d) as usize;
            let ow = conv_transpose2d_ow(&d) as usize;
            let mut out = vec![f16::ZERO; oc * oh * ow];
            conv_transpose2d(dev, &d, &input, &w, &mut out).map_err(|e| e.to_string())?;
            write_f16(out_path, &out)
        }
        "conv_int8" => {
            // conv_int8 IC IH IW OC KH KW SY SX PT PL DY DX IN W OUT
            let (ic, ih, iw, oc) = (num(args, 1)?, num(args, 2)?, num(args, 3)?, num(args, 4)?);
            let (kh, kw) = (num(args, 5)?, num(args, 6)?);
            let mut d = Conv2dDesc::new(ic, ih, iw, oc, kh, kw);
            d.stride_y = num(args, 7)?;
            d.stride_x = num(args, 8)?;
            d.pad_top = num(args, 9)?;
            d.pad_left = num(args, 10)?;
            d.dil_y = num(args, 11)?;
            d.dil_x = num(args, 12)?;
            let input = read_i8(args.get(13).ok_or("missing IN")?)?;
            let w = read_i8(args.get(14).ok_or("missing W")?)?;
            check_len_i8("IN", &input, ic * ih * iw)?;
            check_len_i8("W", &w, oc * ic * kh * kw)?;
            let (oh, ow) = (conv2d_oh(&d), conv2d_ow(&d));
            let mut out = vec![0i32; oc * oh * ow];
            conv2d_int8(dev, &d, &input, &w, &mut out).map_err(|e| e.to_string())?;
            write_i32(out_path, &out)
        }
        "conv_dw_int8" => {
            // conv_dw_int8 IC IH IW KH KW SY SX PT PL DY DX BIAS I_SCALE W_SCALE
            //              O_SCALE IN_ZP W_ZP OUT_ZP IN W OUT
            let (ic, ih, iw) = (num(args, 1)?, num(args, 2)?, num(args, 3)?);
            let (kh, kw) = (num(args, 4)?, num(args, 5)?);
            let mut d = Conv2dDesc::new(ic, ih, iw, ic, kh, kw);
            d.depthwise = true;
            d.stride_y = num(args, 6)?;
            d.stride_x = num(args, 7)?;
            d.pad_top = num(args, 8)?;
            d.pad_left = num(args, 9)?;
            d.dil_y = num(args, 10)?;
            d.dil_x = num(args, 11)?;
            let bias_path = args.get(12).ok_or("missing BIAS")?;
            let bias = if bias_path == "-" {
                None
            } else {
                let b = read_i32(bias_path)?;
                check_len_i32("BIAS", &b, ic)?;
                Some(b)
            };
            let (in_scale, w_scale, out_scale) = (
                f32_bits(args, 13)?,
                f32_bits(args, 14)?,
                f32_bits(args, 15)?,
            );
            let (in_zp, w_zp, out_zp) = (sint(args, 16)?, sint(args, 17)?, sint(args, 18)?);
            let input = read_i8(args.get(19).ok_or("missing IN")?)?;
            let w = read_i8(args.get(20).ok_or("missing W")?)?;
            check_len_i8("IN", &input, ic * ih * iw)?;
            check_len_i8("W", &w, ic * kh * kw)?;
            let (oh, ow) = (conv2d_oh(&d), conv2d_ow(&d));
            let mut out = vec![0i8; ic * oh * ow];
            conv2d_dw_int8(
                dev,
                &d,
                &input,
                &w,
                bias.as_deref(),
                in_scale,
                w_scale,
                out_scale,
                in_zp,
                w_zp,
                out_zp,
                &mut out,
            )
            .map_err(|e| e.to_string())?;
            write_i8(out_path, &out)
        }
        "pool" | "pool_int8" | "pool_uint8" => {
            // FAMILY METHOD C IH IW KH KW SY SX PT PL PB PR IN OUT
            let d = io_pool(args)?;
            pool_fp16_plan(&d).map_err(|e| e.to_string())?;
            let (oh, ow) = (d.oh() as usize, d.ow() as usize);
            let (n_in, n_out) = (d.c * d.ih * d.iw, d.c * oh * ow);
            let in_path = args.get(13).ok_or("missing IN")?;
            match family {
                "pool" => {
                    let input = read_f16(in_path)?;
                    check_len("IN", &input, n_in)?;
                    let mut out = vec![f16::ZERO; n_out];
                    pool_fp16(dev, &d, &input, &mut out).map_err(|e| e.to_string())?;
                    write_f16(out_path, &out)
                }
                "pool_int8" => {
                    let input = read_i8(in_path)?;
                    check_len_i8("IN", &input, n_in)?;
                    let mut out = vec![0i8; n_out];
                    pool_int8(dev, &d, &input, &mut out).map_err(|e| e.to_string())?;
                    write_i8(out_path, &out)
                }
                _ => {
                    let input = read_raw(in_path)?;
                    if input.len() != n_in {
                        return Err(format!("IN: {} bytes, expected {n_in}", input.len()));
                    }
                    let mut out = vec![0u8; n_out];
                    pool_uint8(dev, &d, &input, &mut out).map_err(|e| e.to_string())?;
                    std::fs::write(out_path, &out).map_err(|e| format!("{out_path}: {e}"))
                }
            }
        }
        "resize_nearest" | "resize_bilinear" => {
            // resize_* C IH IW SY SX IN OUT
            let (c, ih, iw) = (num(args, 1)?, num(args, 2)?, num(args, 3)?);
            let (sy, sx) = (num(args, 4)?, num(args, 5)?);
            let input = read_f16(args.get(6).ok_or("missing IN")?)?;
            check_len("IN", &input, c * ih * iw)?;
            let mut out = vec![f16::ZERO; c * ih * sy * iw * sx];
            if family == "resize_bilinear" {
                upsample_bilinear(dev, &input, &mut out, c, ih, iw, sy, sx)
                    .map_err(|e| e.to_string())?;
            } else {
                upsample_nearest(dev, &input, &mut out, c, ih, iw, sy, sx)
                    .map_err(|e| e.to_string())?;
            }
            write_f16(out_path, &out)
        }
        other => Err(format!("unknown --io family '{other}'")),
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

fn usage() {
    eprintln!(
        "usage: conv_selftest [--conv] [--dw] [--act] [--conv1d] [--transpose]\n\
         \x20      [--int8] [--int8-mt] [--int8-dw] [--pool] [--resize] [--all] [--reps R]\n\
         \x20      [--shape IC IH IW OC KH KW [SY SX PT PL DY DX]]\n\
         \x20      [--dw-shape C IH IW KH KW [SY SX PT PL DY DX]]\n\
         \x20      [--conv1d-shape IC IT OC KW STRIDE PAD]\n\
         \x20      [--transpose-shape IC IH IW OC KH KW [SY SX PT PL DY DX OPADY OPADX DW]]\n\
         \n\
         Default: every family.\n\
         Direct: 3x3/5x5/1x1, strides 1/2, pad 0/1/2, dil 1/2, oc 16/32/64, plus\n\
         spatial/OC tiling shapes (64x56x56 k3, 128x28x28 k3 s2, 32x112x112 k3).\n\
         Depthwise: 32x56x56 k3, 64x28x28 k3 s2, 96x14x14 k1.\n\
         Act: SiLU / tanh / GELU on a 3x3 direct conv.\n\
         conv1d: 80x3000 k3 s1/s2 pad1, 384 outputs.\n\
         Transpose: deconv (stride 2/4), lowered (IC % 32, stride 2/4/8, opad),\n\
         depthwise, and opad 0/1.\n\
         int8: 3x3/1x1, IC 3/32/64, OC 16/32/64, stride 1/2, pad 0/1, spatial tiling.\n\
         int8 depthwise: C 32/64/96, 3x3, stride 1/2, zero points at both ends, bias.\n\
         pool: fp16 max/avg at pads 0/1 plus int8/uint8 max/avg.\n\
         resize: nearest and bilinear at scales 2/3/4.\n\
         \n\
         Raw-file A/B against the C library (run one op on IN, write OUT):\n\
         \x20 conv_selftest --io conv     IC IH IW OC KH KW SY SX PT PL DY DX DW IN W OUT\n\
         \x20 conv_selftest --io conv_act IC IH IW OC KH KW SY SX PT PL DY DX DW KIND IN W OUT\n\
         \x20 conv_selftest --io conv1d   IC IT OC KW STRIDE PAD IN W OUT\n\
         \x20 conv_selftest --io conv_t   IC IH IW OC KH KW SY SX PT PL DY DX OPADY OPADX DW IN W OUT\n\
         \x20 conv_selftest --io conv_int8 IC IH IW OC KH KW SY SX PT PL DY DX IN W OUT\n\
         \x20 conv_selftest --io conv_dw_int8 IC IH IW KH KW SY SX PT PL DY DX BIAS \\\n\
         \x20      I_SCALE W_SCALE O_SCALE IN_ZP W_ZP OUT_ZP IN W OUT\n\
         \x20 conv_selftest --io pool|pool_int8|pool_uint8 METHOD C IH IW KH KW SY SX PT PL PB PR IN OUT\n\
         \x20 conv_selftest --io resize_nearest|resize_bilinear C IH IW SY SX IN OUT\n\
         Tensors are raw fp16, except conv_int8 (IN/W int8, OUT int32), conv_dw_int8\n\
         (IN/W/OUT int8, BIAS int32 or -) and the int8/uint8 pools (one byte each).\n\
         The quant scales are f32 bit patterns in hex. W is [OC][IC][KH][KW] (direct),\n\
         [C][KH][KW] (depthwise) or [IC][OC][KH][KW] (direct transpose). This is\n\
         ab_board.c's exchange format."
    );
}

fn parse_shape<const N: usize>(args: &[String], i: usize, optional: usize) -> Option<[usize; N]> {
    let mut v = [0usize; N];
    for (k, slot) in v.iter_mut().enumerate().take(N - optional) {
        *slot = args.get(i + 1 + k)?.parse().ok()?;
    }
    // Defaults for the optional tail: strides/dilation 1, pads/opad 0, direct.
    let defaults = [1usize, 1, 0, 0, 1, 1, 0, 0, 0];
    for (k, slot) in v.iter_mut().enumerate().skip(N - optional).take(optional) {
        *slot = args
            .get(i + 1 + k)
            .and_then(|a| a.parse().ok())
            .unwrap_or(defaults[k - (N - optional)]);
    }
    Some(v)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut conv, mut dw, mut act, mut c1d, mut tr) = (false, false, false, false, false);
    let (mut i8, mut i8dw, mut pool, mut resize) = (false, false, false, false);
    let mut i8mt = false;
    let mut all = args.is_empty();
    let mut reps = 1usize;
    let mut conv_shape: Option<ConvShape> = None;
    let mut dw_shape: Option<DwShape> = None;
    let mut c1d_shape: Option<Conv1dShape> = None;
    let mut tr_shape: Option<TransposeShape> = None;
    let mut io: Option<Vec<String>> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--conv" => conv = true,
            "--dw" => dw = true,
            "--act" => act = true,
            "--conv1d" => c1d = true,
            "--transpose" => tr = true,
            "--int8" => i8 = true,
            "--int8-dw" => i8dw = true,
            "--int8-mt" => i8mt = true,
            "--pool" => pool = true,
            "--resize" => resize = true,
            "--all" => all = true,
            "--reps" => {
                i += 1;
                reps = args.get(i).and_then(|v| v.parse().ok()).unwrap_or(1).max(1);
            }
            "--io" => {
                io = Some(args[i + 1..].to_vec());
                break;
            }
            "--shape" => {
                let Some(v) = parse_shape::<12>(&args, i, 6) else {
                    eprintln!("--shape needs IC IH IW OC KH KW [SY SX PT PL DY DX]");
                    return ExitCode::FAILURE;
                };
                conv_shape = Some((
                    v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11],
                ));
                conv = true;
                i += 6;
            }
            "--dw-shape" => {
                let Some(v) = parse_shape::<11>(&args, i, 6) else {
                    eprintln!("--dw-shape needs C IH IW KH KW [SY SX PT PL DY DX]");
                    return ExitCode::FAILURE;
                };
                dw_shape = Some((
                    v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10],
                ));
                dw = true;
                i += 5;
            }
            "--conv1d-shape" => {
                let Some(v) = parse_shape::<6>(&args, i, 0) else {
                    eprintln!("--conv1d-shape needs IC IT OC KW STRIDE PAD");
                    return ExitCode::FAILURE;
                };
                c1d_shape = Some((v[0], v[1], v[2], v[3], v[4], v[5]));
                c1d = true;
                i += 6;
            }
            "--transpose-shape" => {
                let Some(v) = parse_shape::<13>(&args, i, 7) else {
                    eprintln!(
                        "--transpose-shape needs IC IH IW OC KH KW [SY SX PT PL DY DX OPADY OPADX DW]"
                    );
                    return ExitCode::FAILURE;
                };
                tr_shape = Some((
                    v[0],
                    v[1],
                    v[2],
                    v[3],
                    v[4],
                    v[5],
                    v[6],
                    v[7],
                    v[8],
                    v[9],
                    v[10],
                    v[11],
                    v[12] != 0,
                ));
                tr = true;
                i += 7;
            }
            "-h" | "--help" => {
                usage();
                return ExitCode::SUCCESS;
            }
            other => {
                eprintln!("unknown argument '{other}'");
                usage();
                return ExitCode::FAILURE;
            }
        }
        i += 1;
    }
    if io.is_none() && all {
        (conv, dw, act, c1d, tr) = (true, true, true, true, true);
        (i8, i8dw, pool, resize) = (true, true, true, true);
        i8mt = true;
    }
    if io.is_none() && !(conv || dw || act || c1d || tr || i8 || i8dw || pool || resize || i8mt) {
        usage();
        return ExitCode::FAILURE;
    }

    let dev = match Device::open() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("conv_selftest: cannot open the NPU: {e}");
            return ExitCode::from(2);
        }
    };

    if let Some(args) = io {
        return match io_mode(&dev, &args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("conv_selftest --io: {e}");
                ExitCode::FAILURE
            }
        };
    }

    let mut pass = true;
    if conv {
        println!("direct conv (fp16, CNA native):");
        let shapes = match conv_shape {
            Some(s) => vec![s],
            None => vec![
                // Kernel / stride / pad / dilation sweep.
                (32, 16, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1),
                (32, 16, 16, 32, 3, 3, 2, 2, 0, 0, 1, 1),
                (32, 16, 16, 16, 5, 5, 1, 1, 2, 2, 1, 1),
                (32, 16, 16, 16, 5, 5, 2, 2, 0, 0, 2, 2),
                (64, 8, 8, 16, 1, 1, 1, 1, 0, 0, 1, 1),
                (3, 32, 32, 16, 1, 1, 1, 1, 0, 0, 1, 1),
                (3, 56, 56, 64, 3, 3, 1, 1, 1, 1, 1, 1),
                // Tiling: OC-only, and OC + spatial bands.
                (64, 56, 56, 64, 3, 3, 1, 1, 1, 1, 1, 1),
                (128, 28, 28, 128, 3, 3, 2, 2, 1, 1, 1, 1),
                (32, 112, 112, 32, 3, 3, 1, 1, 1, 1, 1, 1),
            ],
        };
        // The first case goes through the alloc-per-call entry (the C's legacy path);
        // the rest share ONE resident context, which serves different shapes from
        // grow-only BOs and skips the per-call BO churn.
        let mut ctx = ConvContext::new(&dev);
        for (i, s) in shapes.into_iter().enumerate() {
            pass &= if i == 0 {
                run_conv(&dev, s, reps, None)
            } else {
                run_conv(&dev, s, reps, Some(&mut ctx))
            };
        }
        println!();
    }
    if dw {
        println!("depthwise conv (CONV_MODE 3 + DW_EN):");
        let shapes = match dw_shape {
            Some(s) => vec![s],
            None => vec![
                (32, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1),
                (64, 28, 28, 3, 3, 2, 2, 1, 1, 1, 1),
                (96, 14, 14, 1, 1, 1, 1, 0, 0, 1, 1),
            ],
        };
        for s in shapes {
            pass &= run_dw(&dev, s, reps);
        }
        println!();
    }
    if act {
        println!("conv -> activation fusion (DPU LUT epilogue):");
        let s = conv_shape.unwrap_or((32, 16, 16, 32, 3, 3, 1, 1, 1, 1, 1, 1));
        for kind in [Activation::Silu, Activation::Tanh, Activation::Gelu] {
            pass &= run_act(&dev, s, kind, reps);
        }
        println!();
    }
    if c1d {
        println!("conv1d (height-1 mapping):");
        let shapes = match c1d_shape {
            Some(s) => vec![s],
            None => vec![(80, 3000, 384, 3, 1, 1), (80, 1500, 384, 3, 2, 1)],
        };
        for s in shapes {
            pass &= run_conv1d(&dev, s, reps);
        }
        println!();
    }
    if tr {
        println!("transposed conv (hardware deconv route + the lowering):");
        let shapes: Vec<TransposeShape> = match tr_shape {
            Some(s) => vec![s],
            None => vec![
                // Deconv route: direct, stride 2, IC % 32 == 0.
                (32, 16, 16, 32, 3, 3, 2, 2, 1, 1, 1, 1, false),
                (32, 16, 16, 16, 3, 3, 2, 2, 1, 1, 0, 0, false),
                (64, 8, 8, 32, 4, 4, 4, 4, 2, 2, 0, 0, false),
                // Lowered: IC % 32 != 0, a non-power-of-two stride, and depthwise.
                (24, 16, 16, 24, 3, 3, 2, 2, 1, 1, 1, 1, false),
                (32, 16, 16, 16, 3, 3, 3, 3, 1, 1, 1, 1, false),
                (32, 16, 16, 32, 2, 2, 2, 2, 0, 0, 0, 0, true),
                // Depthwise with a trailing output pad: still the lowering (the
                // hardware route has no depthwise form).
                (32, 16, 16, 32, 3, 3, 2, 2, 1, 1, 1, 1, true),
                (32, 16, 16, 32, 3, 3, 1, 1, 1, 1, 0, 0, false),
            ],
        };
        for s in shapes {
            pass &= run_transpose(&dev, s, reps);
        }
        println!();
    }

    if i8 {
        println!("int8 direct conv (int32-raw accumulator):");
        let shapes = match conv_shape {
            Some(s) => vec![s],
            None => vec![
                // The RGB stem (IC=3 padded to 32), 1x1, and the plain 3x3.
                (3, 32, 32, 16, 1, 1, 1, 1, 0, 0, 1, 1),
                (32, 16, 16, 32, 3, 3, 1, 1, 1, 1, 1, 1),
                (32, 16, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1),
                (64, 8, 8, 16, 1, 1, 1, 1, 0, 0, 1, 1),
                // Stride 2, pad 0, and the OC%32 pad (an OC=16 head is padded to 32).
                (32, 16, 16, 32, 3, 3, 2, 2, 0, 0, 1, 1),
                (64, 14, 14, 16, 3, 3, 1, 1, 0, 0, 1, 1),
                // Tiling: OC-only, then OC + row bands (112x112 k3).
                (64, 56, 56, 64, 3, 3, 1, 1, 1, 1, 1, 1),
                (128, 28, 28, 128, 3, 3, 2, 2, 1, 1, 1, 1),
                (32, 112, 112, 32, 3, 3, 1, 1, 1, 1, 1, 1),
            ],
        };
        // The first case exercises the alloc-per-call entry; the rest share one
        // resident context (grow-only BOs, no per-call churn).
        let mut ctx = ConvContext::new(&dev);
        for (idx, s) in shapes.into_iter().enumerate() {
            pass &= if idx == 0 {
                run_int8(&dev, s, reps, None)
            } else {
                run_int8(&dev, s, reps, Some(&mut ctx))
            };
        }
        println!();
    }
    if i8mt {
        println!("int8 direct conv, multicore pool vs serial:");
        let shapes = match conv_shape {
            Some(s) => vec![s],
            None => vec![
                (64, 56, 56, 64, 3, 3, 1, 1, 1, 1, 1, 1),
                (128, 28, 28, 128, 3, 3, 2, 2, 1, 1, 1, 1),
                (32, 112, 112, 32, 3, 3, 1, 1, 1, 1, 1, 1),
                // One OC group: the pool falls back to the serial path by design.
                (64, 14, 14, 32, 3, 3, 1, 1, 0, 0, 1, 1),
            ],
        };
        for s in shapes {
            pass &= run_int8_mt(&dev, s, 3, reps);
        }
        println!();
    }
    if i8dw {
        println!("int8 depthwise conv (int8 out, on-chip requant):");
        let shapes: Vec<(DwShape, i32, i32, bool)> = match dw_shape {
            Some(s) => vec![(s, -2, 5, true)],
            None => vec![
                // C is the int8 depthwise *group* count in effect: the CNA walks
                // channels in G = 64 wide groups, so a C that is not a multiple of 64
                // runs outside the writer's validated envelope (both libraries then
                // emit identical bytes for a wrong surface — see the module docs).
                ((64, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1), -2, 5, true),
                ((64, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1), 37, -20, true),
                ((64, 16, 16, 3, 3, 2, 2, 1, 1, 1, 1), -128, 127, true),
                ((64, 8, 11, 5, 5, 1, 1, 1, 1, 1, 1), 127, -128, false),
                ((128, 7, 5, 3, 3, 1, 2, 1, 1, 1, 1), 0, 0, true),
            ],
        };
        for (s, izp, ozp, bias) in shapes {
            pass &= run_dw_int8(&dev, s, izp, ozp, bias, reps);
        }
        println!();
    }
    if pool {
        println!("PPU pooling (fp16 / int8 / uint8):");
        let mut cases: Vec<PoolDesc> = Vec::new();
        // fp16: global max/avg, a padded max and a padded, overlapping average.
        let mut d = PoolDesc::new(8, 7, 7, 7, 7);
        d.stride_y = 7;
        d.stride_x = 7;
        d.method = PoolMethod::Max;
        cases.push(d);
        let mut d = PoolDesc::new(8, 7, 7, 7, 7);
        d.stride_y = 7;
        d.stride_x = 7;
        cases.push(d); // avg (the default method is Max; set below)
        // Two windows of a 7x7 kernel (the PPU's validated stepping envelope keeps the
        // kernel at 8 or below for a multi-window pool; see the pool module docs).
        let mut d = PoolDesc::new(16, 14, 14, 7, 7);
        d.stride_y = 7;
        d.stride_x = 7;
        d.method = PoolMethod::Avg;
        cases.push(d);
        let mut d = PoolDesc::new(8, 5, 5, 3, 3);
        d.pad_top = 1;
        d.pad_left = 1;
        d.pad_bottom = 1;
        d.pad_right = 1;
        d.method = PoolMethod::Max;
        cases.push(d);
        let mut d = PoolDesc::new(8, 5, 5, 3, 3);
        d.pad_top = 1;
        d.pad_left = 1;
        d.pad_bottom = 1;
        d.pad_right = 1;
        d.method = PoolMethod::Avg;
        cases.push(d);
        // The second entry's method is Avg (built here so the list stays declarative).
        cases[1].method = PoolMethod::Avg;
        for d in &cases {
            pass &= run_pool(&dev, *d, reps);
        }
        // int8 / uint8: a 2x2 stride-2 max (exact integer max) and a padded 3x3 average.
        let mut d = PoolDesc::new(32, 8, 8, 2, 2);
        d.stride_y = 2;
        d.stride_x = 2;
        d.method = PoolMethod::Max;
        pass &= run_pool_int8(&dev, d, reps, false);
        pass &= run_pool_int8(&dev, d, reps, true);
        let mut d = PoolDesc::new(64, 7, 7, 3, 3);
        d.pad_top = 1;
        d.pad_left = 1;
        d.pad_bottom = 1;
        d.pad_right = 1;
        d.method = PoolMethod::Avg;
        pass &= run_pool_int8(&dev, d, reps, false);
        pass &= run_pool_int8(&dev, d, reps, true);
        println!();
    }
    if resize {
        println!("upsample (depthwise transposed conv):");
        for bilinear in [false, true] {
            for (c, ih, iw, sy, sx) in [
                (32, 8, 8, 2, 2),
                (32, 5, 7, 3, 3),
                (32, 4, 4, 4, 4),
                (64, 6, 6, 2, 4),
            ] {
                pass &= run_resize(&dev, c, ih, iw, sy, sx, bilinear, reps);
            }
        }
        println!();
    }

    println!("conv_selftest: {}", if pass { "PASS" } else { "FAIL" });
    if pass {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
