//! On-device activation / elementwise self-test: the port of `rocket_activation.c`
//! against its host model and references.
//!
//! ```text
//! activation_selftest [--kind K] [--n N] [--shape M K N] [--reps R] [--all]
//!                     [--ew] [--dump-lut] [--dump-gen]
//!                     [--write-in KIND N PATH] [--run-in|--model-in KIND IN OUT]
//!                     [--cross KIND C_OUT]
//! ```
//!
//! Default: every activation kind at n = 1, 31, 4096 and 262144, plus the EW ops
//! (mul/add/sub/max/min/clip/prelu/div) at two sizes. Each case prints the wall time,
//! the cosine / max-abs / max-rel against the **LUT model** (the same table
//! interpolated in f32, [`rocket_npu::ops::activation::lut_apply`]) and the max-abs
//! against the exact math reference, then a PASS/FAIL line. Exit code 1 on a numeric
//! failure.
//!
//! # The numerics gate
//!
//! The DPU LUT is an fp16 approximation by design, so the device is compared against
//! the *same table's interpolation*, not against an exact math function:
//!
//! - LUT kinds: the device must land within **4 fp16 ULPs** of the model, measured at
//!   the case's largest output magnitude (the model interpolates in f32 while the
//!   hardware interpolates in fixed point, so a fraction of an output quantum is
//!   expected; a misplaced register or table would be orders of magnitude off — see
//!   the `vs exact` column, which is the LUT's own design error), and `cosine >=
//!   0.9999`. Kinds whose codomain fits inside `[-1,1]` (sigmoid/hardsigmoid/tanh/
//!   the gates/exp) are additionally held to an absolute `2^-10`.
//! - EW add/mul/sub/max/min and clip: **bit-exact** (max_abs == 0), as the C library
//!   documents; PReLU's `alpha in [0,1]` path likewise.
//! - div: within 4 ULPs of the reciprocal-LUT-then-multiply composition.
//!
//! One genuine C quirk is excluded explicitly: tables that straddle 0 with the
//! sign-based LE/LO split (tanh's default affine path) glitch in a razor-thin band
//! around `x = 0` (an output spike, the documented LE/LO mux quirk). The C library
//! has the same behaviour — its own gates never sample exactly 0, and the port is
//! bit-identical there — so those elements are counted and reported as `glitch=` and
//! left out of the gate. `ROCKET_LUT_SHIFT=1` (or the repaired LeakyReLU/ELU paths)
//! avoid the mux entirely.
//!
//! The exact-reference column is informational: it measures the LUT design error
//! (domain-bounded for sqrt/rsqrt/reciprocal/log, ~1e-5..5e-3, and the tail
//! saturation of tanh/hardswish).
//!
//! `--dump-lut` / `--dump-gen` print the `rocket_lut_epilogue_build` dump and the
//! generated program (no device needed) for the byte comparison with the C builder;
//! `--write-in`/`--run-in`/`--model-in`/`--cross` exchange raw fp16 files with a C
//! binary on the board for the C-vs-Rust A/B.

use std::fs;
use std::process::ExitCode;
use std::time::Instant;

use half::f16;
use rocket_npu::Device;
use rocket_npu::ops::activation as act;
use rocket_npu::ops::activation::{Activation, EwOp};

/// The absolute bound for the kinds whose output stays inside [-1,1] (2^-10).
const TOL: f32 = 1.0 / 1024.0;
/// The device-vs-model bound as a fraction of the case's output scale: the model
/// interpolates the table in f32 while the hardware interpolates in fixed point, so the
/// agreement bound is a fraction of the output scale (1/256 = ~4 fp16 ULPs at that
/// scale). The bit-exact statement is the C-vs-Rust A/B (`--cross`): both run the same
/// program and come out identical.
const SCALE_FRACTION: f32 = 1.0 / 256.0;
/// How many output ULPs the `x≈0` glitch detection allows before it calls a spike.
const MAX_ULP: f32 = 4.0;
/// The `x≈0` LE/LO mux glitch band, as a fraction of the table's half-width
/// (the C's own repair guards use 1/1024).
const GLITCH_BAND: f32 = 1.0 / 1024.0;

#[derive(Debug, Clone, Copy)]
struct Stats {
    cosine: f64,
    max_abs: f32,
    max_rel: f32,
    worst: usize,
}

/// The fp16 grid spacing at `|v|`.
fn f16_ulp(v: f32) -> f32 {
    let a = v.abs().max(6.1e-5); // the smallest normal f16
    let e = a.log2().floor() as i32;
    2f32.powi(e - 10)
}

fn stats(want: &[f16], got: &[f16], skip: impl Fn(usize, f32) -> bool) -> Stats {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    let (mut max_abs, mut max_rel, mut worst) = (0.0f32, 0.0f32, 0usize);
    for (i, (&w, &g)) in want.iter().zip(got).enumerate() {
        if skip(i, w.to_f32()) {
            continue;
        }
        let (w, g) = (w.to_f32() as f64, g.to_f32() as f64);
        dot += w * g;
        na += w * w;
        nb += g * g;
        let d = (w - g).abs() as f32;
        if d > max_abs {
            max_abs = d;
            worst = i;
        }
        let denom = w.abs().max(1e-6) as f32;
        max_rel = max_rel.max(d / denom);
    }
    let cosine = if na == 0.0 && nb == 0.0 {
        1.0
    } else if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    };
    Stats {
        cosine,
        max_abs,
        max_rel,
        worst,
    }
}

fn ramp(n: usize, x0: f32, x1: f32) -> Vec<f16> {
    if n == 1 {
        return vec![f16::from_f32(0.5 * (x0 + x1))];
    }
    (0..n)
        .map(|i| {
            let t = i as f32 / (n - 1) as f32;
            f16::from_f32(x0 + (x1 - x0) * t)
        })
        .collect()
}

/// The sweep range per kind: the C gates' ranges for the bounded kinds, and the
/// table's own domain (with a little spill past both ends to exercise the clamps)
/// for the shifted ones.
fn sweep_range(kind: Activation) -> (f32, f32) {
    match kind {
        Activation::Sigmoid | Activation::HardSigmoid => (-8.0, 8.0),
        Activation::Tanh => (-10.0, 10.0),
        Activation::Silu | Activation::Gelu => (-8.0, 8.0),
        Activation::GeluGate | Activation::MishGate => (-8.0, 8.0),
        Activation::HardSwish => (-4.0, 4.0),
        Activation::Mish => (-8.0, 8.0),
        Activation::LeakyRelu => (-16.0, 16.0),
        Activation::Elu | Activation::Selu => (-8.0, 8.0),
        Activation::Sqrt
        | Activation::Rsqrt
        | Activation::Reciprocal
        | Activation::Log
        | Activation::Exp
        | Activation::Softplus
        | Activation::Abs => match act::table_domain(kind) {
            Ok((lo, hi)) => (lo * 0.8, hi * 1.1),
            Err(_) => (-8.0, 8.0),
        },
    }
}

/// Run one kind through its public entry point (the parametric kinds have dedicated
/// entries, as in C).
fn device_apply(
    dev: &Device,
    kind: Activation,
    input: &[f16],
    out: &mut [f16],
) -> rocket_npu::Result<()> {
    match kind {
        Activation::LeakyRelu => act::leaky_relu(dev, 0.1, input, out),
        Activation::Elu => act::elu(dev, 1.0, input, out),
        Activation::Selu => act::selu(dev, input, out),
        _ => act::activation(dev, kind, input, out),
    }
}

/// The host model of the same entry point.
fn model_apply(kind: Activation, input: &[f16], out: &mut [f16]) -> rocket_npu::Result<()> {
    match kind {
        Activation::LeakyRelu => act::leaky_relu_model(0.1, input, out),
        Activation::Elu => act::elu_model_alpha(1.0, input, out),
        Activation::Selu => act::selu_model(input, out),
        _ => act::activation_model(kind, input, out),
    }
}

fn parse_kind(s: &str) -> Option<Activation> {
    if let Ok(code) = s.parse::<i32>() {
        return Activation::from_code(code);
    }
    Activation::ALL.into_iter().find(|k| k.name() == s)
}

fn run_kind(dev: &Device, kind: Activation, n: usize, reps: usize) -> bool {
    let (x0, x1) = sweep_range(kind);
    let input = ramp(n, x0, x1);
    let mut got = vec![f16::ZERO; n];
    let mut model = vec![f16::ZERO; n];
    let mut exact = vec![f16::ZERO; n];
    act::activation_ref(kind, &input, &mut exact);

    let t = Instant::now();
    for _ in 0..reps.max(1) {
        if let Err(e) = device_apply(dev, kind, &input, &mut got) {
            println!("  {:<11} n={:<7} device error: {e} -> FAIL", kind.name(), n);
            return false;
        }
    }
    let ms = t.elapsed().as_secs_f64() * 1e3 / reps.max(1) as f64;

    if let Err(e) = model_apply(kind, &input, &mut model) {
        println!("  {:<11} n={:<7} model error: {e} -> FAIL", kind.name(), n);
        return false;
    }

    // The LE/LO mux glitch band: only tables that straddle 0 with the sign-based split
    // (tanh's default affine path) and only where the device spikes away from the model.
    let band = if kind == Activation::Tanh && !rocket_npu::knobs::present("ROCKET_LUT_SHIFT") {
        act::table_domain(kind)
            .map(|(lo, hi)| GLITCH_BAND * lo.abs().max(hi.abs()))
            .unwrap_or(0.0)
    } else {
        0.0
    };
    let scale = model.iter().fold(0.0f32, |m, v| m.max(v.to_f32().abs()));
    let ulp = f16_ulp(scale);
    let mut glitch_mask = vec![false; n];
    let mut glitch = 0usize;
    for i in 0..n {
        let spike = (got[i].to_f32() - model[i].to_f32()).abs() > MAX_ULP * ulp;
        if band > 0.0 && input[i].to_f32().abs() <= band && spike {
            glitch_mask[i] = true;
            glitch += 1;
        }
    }

    let s = stats(&model, &got, |i, _| glitch_mask[i]);
    let ex = stats(&exact, &got, |i, _| glitch_mask[i]);
    let exact_ref = match kind {
        // The parametric kinds have dedicated device entries; the enum ref returns 0.
        Activation::LeakyRelu | Activation::Elu | Activation::Selu => f32::NAN,
        _ => ex.max_abs,
    };
    let unit_codomain = model.iter().all(|v| v.to_f32().abs() <= 1.0);
    let abs_ok = !unit_codomain || s.max_abs <= TOL;
    let scale_tol = (scale * SCALE_FRACTION).max(TOL);
    let pass = s.max_abs <= scale_tol && s.cosine >= 0.9999 && abs_ok;
    println!(
        "  {:<11} n={:<7} {:>8.2} ms  cos={:.9} max_abs={:.3e} (tol {:.1e}, {:.2} ulp) \
         max_rel={:.3e} glitch={glitch} | vs exact {exact_ref:.3e}  x[{}]={:+.4} \
         got={:+.5} model={:+.5} -> {}",
        kind.name(),
        n,
        ms,
        s.cosine,
        s.max_abs,
        scale_tol,
        s.max_abs / ulp,
        s.max_rel,
        s.worst,
        input.get(s.worst).map(|v| v.to_f32()).unwrap_or(0.0),
        got.get(s.worst).map(|v| v.to_f32()).unwrap_or(0.0),
        model.get(s.worst).map(|v| v.to_f32()).unwrap_or(0.0),
        if pass { "PASS" } else { "FAIL" }
    );
    pass
}

fn run_ew(dev: &Device, n: usize, reps: usize) -> bool {
    let a = ramp(n, -4.0, 4.0);
    let b = ramp(n, 1.0, 3.0);
    let mut pass = true;
    let ops = [
        (EwOp::Mul, "mul"),
        (EwOp::Add, "add"),
        (EwOp::Sub, "sub"),
        (EwOp::Max, "max"),
        (EwOp::Min, "min"),
    ];
    for (op, name) in ops {
        let mut got = vec![f16::ZERO; n];
        let mut want = vec![f16::ZERO; n];
        act::ew_ref(op, &a, &b, &mut want);
        let t = Instant::now();
        let mut r = Ok(());
        for _ in 0..reps.max(1) {
            r = match op {
                EwOp::Mul => act::ew_mul(dev, &a, &b, &mut got),
                EwOp::Add => act::ew_add(dev, &a, &b, &mut got),
                EwOp::Sub => act::ew_sub(dev, &a, &b, &mut got),
                EwOp::Max => act::ew_max(dev, &a, &b, &mut got),
                EwOp::Min => act::ew_min(dev, &a, &b, &mut got),
            };
            if r.is_err() {
                break;
            }
        }
        let ms = t.elapsed().as_secs_f64() * 1e3 / reps.max(1) as f64;
        if let Err(e) = r {
            println!("  ew {name:<10} n={n:<7} device error: {e} -> FAIL");
            pass = false;
            continue;
        }
        let s = stats(&want, &got, |_, _| false);
        let ok = s.max_abs == 0.0;
        pass &= ok;
        println!(
            "  ew {name:<10} n={n:<7} {ms:>8.2} ms  cos={:.9} max_abs={:.3e} -> {}",
            s.cosine,
            s.max_abs,
            if ok { "PASS" } else { "FAIL" }
        );
    }

    // clip: constant-operand MAX then MIN.
    {
        let (lo, hi) = (-0.5f32, 1.5f32);
        let x = ramp(n, -4.0, 4.0);
        let mut got = vec![f16::ZERO; n];
        let mut want = vec![f16::ZERO; n];
        act::clip_ref(lo, hi, &x, &mut want);
        match act::clip(dev, lo, hi, &x, &mut got) {
            Ok(()) => {
                let s = stats(&want, &got, |_, _| false);
                let ok = s.max_abs == 0.0;
                pass &= ok;
                println!(
                    "  clip        n={n:<7}          cos={:.9} max_abs={:.3e} -> {}",
                    s.cosine,
                    s.max_abs,
                    if ok { "PASS" } else { "FAIL" }
                );
            }
            Err(e) => {
                println!("  clip        n={n:<7} device error: {e} -> FAIL");
                pass = false;
            }
        }
    }

    // PReLU: the alpha in [0,1] path (bit-exact) and the general path (one extra
    // fp16 product rounding).
    for (alpha, label, exact) in [
        (vec![0.25f32, 0.5, 0.125, 1.0], "prelu/0..1", true),
        (vec![1.5f32, -0.5, 2.0, 0.0], "prelu/gen", false),
    ] {
        let c = alpha.len();
        let s_len = n.div_ceil(c).max(1);
        let nn = c * s_len;
        let x = ramp(nn, -3.0, 3.0);
        let mut got = vec![f16::ZERO; nn];
        let mut want = vec![f16::ZERO; nn];
        act::prelu_ref(c, s_len, &x, &alpha, &mut want).unwrap();
        match act::prelu(dev, c, s_len, &x, &alpha, &mut got) {
            Ok(()) => {
                let s = stats(&want, &got, |_, _| false);
                let scale = want.iter().fold(0.0f32, |m, v| m.max(v.to_f32().abs()));
                let tol = (scale * SCALE_FRACTION).max(TOL);
                let ok = if exact {
                    s.max_abs == 0.0
                } else {
                    s.max_abs <= tol
                };
                pass &= ok;
                println!(
                    "  {label:<11} n={nn:<7}          cos={:.9} max_abs={:.3e} -> {}",
                    s.cosine,
                    s.max_abs,
                    if ok { "PASS" } else { "FAIL" }
                );
            }
            Err(e) => {
                println!("  {label:<11} n={nn:<7} device error: {e} -> FAIL");
                pass = false;
            }
        }
    }

    // div: reciprocal LUT then EW multiply; compare against the same composition.
    {
        let a = ramp(n, -2.0, 2.0);
        let b = ramp(n, 0.5, 31.0);
        let mut recip = vec![f16::ZERO; n];
        let mut want = vec![f16::ZERO; n];
        let mut got = vec![f16::ZERO; n];
        if let Err(e) = act::activation_model(Activation::Reciprocal, &b, &mut recip) {
            println!("  div         n={n:<7} model error: {e} -> FAIL");
            pass = false;
        } else {
            for i in 0..n {
                want[i] = f16::from_f32(a[i].to_f32() * recip[i].to_f32());
            }
            match act::div(dev, &a, &b, &mut got) {
                Ok(()) => {
                    let s = stats(&want, &got, |_, _| false);
                    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.to_f32().abs()));
                    let ulp = f16_ulp(scale);
                    let ok = s.max_abs <= (scale * SCALE_FRACTION).max(TOL);
                    pass &= ok;
                    println!(
                        "  div         n={n:<7}          cos={:.9} max_abs={:.3e} ({:.2} ulp) -> {}",
                        s.cosine,
                        s.max_abs,
                        s.max_abs / ulp,
                        if ok { "PASS" } else { "FAIL" }
                    );
                }
                Err(e) => {
                    println!("  div         n={n:<7} device error: {e} -> FAIL");
                    pass = false;
                }
            }
        }
    }
    pass
}

fn fail(msg: String, code: u8) -> ExitCode {
    eprintln!("{msg}");
    ExitCode::from(code)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let mut kinds: Vec<Activation> = Vec::new();
    let mut sizes = vec![1usize, 31, 4096, 262144];
    let mut reps = 3usize;
    let mut ew = false;
    let mut dump_lut = false;
    let mut dump_gen = false;
    let mut write_in: Option<(Activation, usize, String)> = None;
    let mut file_run: Option<(String, Activation, String, String)> = None;
    let mut cross: Option<(Activation, String)> = None;
    let mut ew_in: Option<(String, String, String, String)> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--kind" | "-k" => {
                i += 1;
                let s = args.get(i).map(String::as_str).unwrap_or("");
                match parse_kind(s) {
                    Some(k) => kinds = vec![k],
                    None => return fail(format!("unknown kind '{s}'"), 2),
                }
            }
            "--n" => {
                i += 1;
                let n: usize = args.get(i).and_then(|v| v.parse().ok()).unwrap_or(4096);
                sizes = vec![n];
            }
            "--shape" => {
                let m: usize = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(64);
                let k: usize = args.get(i + 2).and_then(|v| v.parse().ok()).unwrap_or(64);
                let n: usize = args.get(i + 3).and_then(|v| v.parse().ok()).unwrap_or(64);
                sizes = vec![m * k * n];
                i += 3;
            }
            "--reps" => {
                i += 1;
                reps = args.get(i).and_then(|v| v.parse().ok()).unwrap_or(3).max(1);
            }
            "--ew" => ew = true,
            "--all" => kinds.clear(),
            "--dump-lut" => dump_lut = true,
            "--dump-gen" => dump_gen = true,
            "--write-in" => {
                let kind = args
                    .get(i + 1)
                    .and_then(|s| parse_kind(s))
                    .unwrap_or(Activation::Sigmoid);
                let n = args
                    .get(i + 2)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(4096usize);
                let path = args.get(i + 3).cloned().unwrap_or_default();
                write_in = Some((kind, n, path));
                i += 3;
            }
            "--run-in" | "--model-in" => {
                let mode = args[i].clone();
                let kind = args
                    .get(i + 1)
                    .and_then(|s| parse_kind(s))
                    .unwrap_or(Activation::Sigmoid);
                let inp = args.get(i + 2).cloned().unwrap_or_default();
                let outp = args.get(i + 3).cloned().unwrap_or_default();
                file_run = Some((mode, kind, inp, outp));
                i += 3;
            }
            "--ew-in" => {
                let op = args.get(i + 1).cloned().unwrap_or_default();
                let a = args.get(i + 2).cloned().unwrap_or_default();
                let b = args.get(i + 3).cloned().unwrap_or_default();
                let outp = args.get(i + 4).cloned().unwrap_or_default();
                ew_in = Some((op, a, b, outp));
                i += 4;
            }
            "--cross" => {
                let kind = args
                    .get(i + 1)
                    .and_then(|s| parse_kind(s))
                    .unwrap_or(Activation::Sigmoid);
                let path = args.get(i + 2).cloned().unwrap_or_default();
                cross = Some((kind, path));
                i += 2;
            }
            other => return fail(format!("unknown argument '{other}'"), 2),
        }
        i += 1;
    }

    if dump_lut {
        print!("{}", act::dump_epilogues());
        return ExitCode::SUCCESS;
    }
    if dump_gen {
        // The same fixed geometry the C `lut_dump` oracle prints: the default TANH
        // epilogue, n=1024 (128 cube columns), input 0x1000, output 0x2000.
        let mut lut = act::LutTable::zeroed();
        let ep = act::lut_epilogue_build(Activation::Tanh, &mut lut).unwrap();
        let mut ops = Vec::new();
        let p = rocket_npu::regcmd::activation::LutActParams {
            input_dma: 0x1000,
            output_dma: 0x2000,
            n: 1024,
            lut: lut.words(),
            bn_mul_operand: ep.bn_mul_operand,
            bn_alu_cfg: ep.bn_alu_cfg,
            out_cvt_offset: ep.out_cvt_offset,
            out_cvt_scale: ep.out_cvt_scale,
            out_cvt_minus_exp: ep.out_cvt_minus_exp,
            out_cvt_cvt_type: ep.out_cvt_cvt_type,
            lut_le_start: ep.lut_le_start,
            lut_lo_end: ep.lut_lo_end,
            le_slope_scale: ep.le_slope_scale,
            le_slope_shift: ep.le_slope_shift,
            lo_slope_scale: ep.lo_slope_scale,
            lo_slope_shift: ep.lo_slope_shift,
            le_index_select: ep.le_index_select,
            lo_index_select: ep.lo_index_select,
        };
        let count = rocket_npu::regcmd::activation::emit(&p, &mut ops).unwrap();
        println!("gen ret 0");
        println!("gen 0 count {count}");
        for op in &ops {
            println!("{:016x}", op.0);
        }
        let mut p2 = p.clone();
        p2.n = 1023;
        println!(
            "gen-odd {}",
            match rocket_npu::regcmd::activation::emit(&p2, &mut Vec::new()) {
                Ok(_) => 0,
                Err(e) => e.status().code(),
            }
        );
        p2.n = 0;
        println!(
            "gen-zero {}",
            match rocket_npu::regcmd::activation::emit(&p2, &mut Vec::new()) {
                Ok(_) => 0,
                Err(e) => e.status().code(),
            }
        );
        return ExitCode::SUCCESS;
    }

    let dev = match Device::open() {
        Ok(d) => d,
        Err(e) => return fail(format!("activation_selftest: cannot open the NPU: {e}"), 2),
    };

    if let Some((op, ap, bp, outp)) = ew_in {
        let read = |path: &str| -> Result<Vec<f16>, String> {
            let raw = fs::read(path).map_err(|e| format!("read {path}: {e}"))?;
            Ok(raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| f16::from_bits(u16::from_le_bytes(*c)))
                .collect())
        };
        let a = match read(&ap) {
            Ok(v) => v,
            Err(e) => return fail(e, 1),
        };
        let b = match read(&bp) {
            Ok(v) => v,
            Err(e) => return fail(e, 1),
        };
        let mut out = vec![f16::ZERO; a.len()];
        let r = match op.as_str() {
            "mul" => act::ew_mul(&dev, &a, &b, &mut out),
            "add" => act::ew_add(&dev, &a, &b, &mut out),
            "sub" => act::ew_sub(&dev, &a, &b, &mut out),
            "max" => act::ew_max(&dev, &a, &b, &mut out),
            "min" => act::ew_min(&dev, &a, &b, &mut out),
            "div" => act::div(&dev, &a, &b, &mut out),
            other => return fail(format!("unknown ew op '{other}'"), 2),
        };
        if let Err(e) = r {
            return fail(format!("ew {op}: {e}"), 1);
        }
        let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
        if let Err(e) = fs::write(&outp, bytes) {
            return fail(format!("write {outp}: {e}"), 1);
        }
        println!("ew {op} n={} -> {outp}", a.len());
        return ExitCode::SUCCESS;
    }

    if let Some((mode, kind, inp, outp)) = file_run {
        let raw = match fs::read(&inp) {
            Ok(r) => r,
            Err(e) => return fail(format!("read {inp}: {e}"), 1),
        };
        let input: Vec<f16> = raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f16::from_bits(u16::from_le_bytes(*c)))
            .collect();
        let mut out = vec![f16::ZERO; input.len()];
        let r = if mode == "--run-in" {
            device_apply(&dev, kind, &input, &mut out)
        } else {
            model_apply(kind, &input, &mut out)
        };
        if let Err(e) = r {
            return fail(format!("{mode} {kind}: {e}"), 1);
        }
        let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
        if let Err(e) = fs::write(&outp, bytes) {
            return fail(format!("write {outp}: {e}"), 1);
        }
        println!("{mode} {kind} n={} -> {outp}", input.len());
        return ExitCode::SUCCESS;
    }

    if let Some((kind, n, path)) = write_in {
        let (x0, x1) = sweep_range(kind);
        let input = ramp(n, x0, x1);
        let bytes: Vec<u8> = input
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect();
        if let Err(e) = fs::write(&path, bytes) {
            return fail(format!("write {path}: {e}"), 1);
        }
        println!("wrote {n} f16 inputs ({kind}, {x0}..{x1}) to {path}");
        return ExitCode::SUCCESS;
    }

    if let Some((kind, path)) = cross {
        let (x0, x1) = sweep_range(kind);
        let raw = match fs::read(&path) {
            Ok(r) => r,
            Err(e) => return fail(format!("read {path}: {e}"), 1),
        };
        let c_out: Vec<f16> = raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f16::from_bits(u16::from_le_bytes(*c)))
            .collect();
        let n = c_out.len();
        let input = ramp(n, x0, x1);
        let mut got = vec![f16::ZERO; n];
        if let Err(e) = device_apply(&dev, kind, &input, &mut got) {
            return fail(format!("device: {e}"), 1);
        }
        let mut model = vec![f16::ZERO; n];
        model_apply(kind, &input, &mut model).unwrap();
        let sc = stats(&c_out, &got, |_, _| false);
        let sm = stats(&c_out, &model, |_, _| false);
        let identical = c_out == got;
        println!(
            "  cross {:<11} n={:<7} identical={identical} C-vs-Rust cos={:.9} max_abs={:.3e} \
             | C-vs-model cos={:.9} max_abs={:.3e}",
            kind.name(),
            n,
            sc.cosine,
            sc.max_abs,
            sm.cosine,
            sm.max_abs
        );
        return if identical {
            ExitCode::SUCCESS
        } else {
            ExitCode::FAILURE
        };
    }

    let kinds = if kinds.is_empty() {
        Activation::ALL.to_vec()
    } else {
        kinds
    };
    let mut pass = true;
    println!(
        "activation_selftest: {} kind(s), sizes {sizes:?}",
        kinds.len()
    );
    for kind in &kinds {
        for &n in &sizes {
            pass &= run_kind(&dev, *kind, n, reps);
        }
    }
    if ew || kinds.len() > 1 {
        let ew_sizes: Vec<usize> = if sizes.len() == 1 {
            vec![sizes[0]]
        } else {
            vec![4096, 262144]
        };
        for n in ew_sizes {
            pass &= run_ew(&dev, n, reps);
        }
    }
    println!(
        "activation_selftest: {}",
        if pass { "PASS" } else { "FAIL" }
    );
    if pass {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
