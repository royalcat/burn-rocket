//! On-device self-test for the M8a fp16 convolution port.
//!
//! ```text
//! conv_selftest [--conv] [--dw] [--act] [--conv1d] [--transpose] [--all]
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
    Conv2dDesc, ConvContext, conv1d, conv2d, conv2d_act, conv2d_oh, conv2d_ow, conv2d_ref,
};
use rocket_npu::ops::conv_transpose::{
    ConvTranspose2dDesc, conv_transpose2d, conv_transpose2d_oh, conv_transpose2d_ow,
    conv_transpose2d_plan_code, conv_transpose2d_ref, conv_transpose2d_route,
};

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
        other => Err(format!("unknown --io family '{other}'")),
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

fn usage() {
    eprintln!(
        "usage: conv_selftest [--conv] [--dw] [--act] [--conv1d] [--transpose] [--all] [--reps R]\n\
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
         \n\
         Raw-file A/B against the C library (run one op on IN, write OUT):\n\
         \x20 conv_selftest --io conv     IC IH IW OC KH KW SY SX PT PL DY DX DW IN W OUT\n\
         \x20 conv_selftest --io conv_act IC IH IW OC KH KW SY SX PT PL DY DX DW KIND IN W OUT\n\
         \x20 conv_selftest --io conv1d   IC IT OC KW STRIDE PAD IN W OUT\n\
         \x20 conv_selftest --io conv_t   IC IH IW OC KH KW SY SX PT PL DY DX OPADY OPADX DW IN W OUT\n\
         Tensors are raw fp16; W is [OC][IC][KH][KW] (direct), [C][KH][KW] (depthwise)\n\
         or [IC][OC][KH][KW] (direct transpose). This is ab_board.c's exchange format."
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
    }
    if io.is_none() && !(conv || dw || act || c1d || tr) {
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

    println!("conv_selftest: {}", if pass { "PASS" } else { "FAIL" });
    if pass {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
