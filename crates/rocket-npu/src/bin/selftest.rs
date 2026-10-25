//! On-device numeric self-test for the one-shot matmul surface.
//!
//!   selftest [M K N] [--context] [--f32out] [--bf16] [--mt N]
//!
//! The fp16 one-shot check (the original CLI) always runs: it is the single-fd
//! baseline every other mode is compared against. In addition:
//!
//! - `--context` exercises the resident-weight path at a canonical pack M (256) and
//!   multiple call Ms;
//! - `--mt N` runs the multi-fd `_mt` fan-out (N column slices);
//! - `--f32out` runs the fp32-output path, so the fp16 reference shows its per-tile
//!   rounding and this one the true fp16-input dot product;
//! - `--bf16` runs the bf16 one-shot (`_mt` when `--mt` is given) and reports its
//!   fidelity against the exact f32 product of the same f32 operands.
//!
//! Every mode compares against a reference accumulated in f64 from the same inputs and
//! prints the cosine / max error plus wall time.

use std::time::Instant;

use half::f16;
use rocket_npu::Device;
use rocket_npu::matmul::{
    Context, one_shot, one_shot_bf16, one_shot_bf16_mt, one_shot_f32out, one_shot_mt,
};

fn rand_f32(seed: &mut u64) -> f32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    ((*seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
}

fn rand_f16(seed: &mut u64) -> f16 {
    f16::from_f32(rand_f32(seed))
}

/// The exact `A[M,K] * B[N,K]^T` product in f64 (f16 and f32 operands both widen).
fn reference<T: Copy + Into<f64>>(m: usize, k: usize, n: usize, a: &[T], b: &[T]) -> Vec<f64> {
    let mut out = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f64;
            for p in 0..k {
                sum += a[i * k + p].into() * b[j * k + p].into();
            }
            out[i * n + j] = sum;
        }
    }
    out
}

fn report(name: &str, got: &[f64], want: &[f64], ms: f64) {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    let mut max = 0.0f64;
    for (g, w) in got.iter().zip(want) {
        let g = *g;
        dot += g * w;
        na += g * g;
        nb += w * w;
        max = max.max((g - w).abs());
    }
    let cos = if na > 0.0 && nb > 0.0 {
        dot / (na.sqrt() * nb.sqrt())
    } else {
        1.0
    };
    println!("{name}: cosine {cos:.8} max_abs {max:.6} {ms:.2} ms");
}

fn report_f16(name: &str, got: &[f16], want: &[f64], ms: f64) {
    let got: Vec<f64> = got.iter().map(|v| v.to_f64()).collect();
    report(name, &got, want, ms);
}

fn report_f32(name: &str, got: &[f32], want: &[f64], ms: f64) {
    let got: Vec<f64> = got.iter().map(|v| *v as f64).collect();
    report(name, &got, want, ms);
}

struct Args {
    dims: Vec<usize>,
    context: bool,
    f32out: bool,
    bf16: bool,
    mt: Option<usize>,
}

fn parse_args() -> Result<Args, Box<dyn std::error::Error>> {
    let mut args = Args {
        dims: Vec::new(),
        context: false,
        f32out: false,
        bf16: false,
        mt: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--context" => args.context = true,
            "--f32out" => args.f32out = true,
            "--bf16" => args.bf16 = true,
            "--mt" => {
                let v = it.next().ok_or("--mt needs a thread count")?;
                args.mt = Some(v.parse()?);
            }
            _ => args.dims.push(arg.parse()?),
        }
    }
    if args.dims.len() > 3 {
        return Err("at most three positional dimensions (M K N)".into());
    }
    Ok(args)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    let m = args.dims.first().copied().unwrap_or(64);
    let k = args.dims.get(1).copied().unwrap_or(256);
    let n = args.dims.get(2).copied().unwrap_or(128);
    println!("shape M={m} K={k} N={n}");

    let mut seed = 0x1234_5678_9abc_def1u64;
    let a: Vec<f16> = (0..m * k).map(|_| rand_f16(&mut seed)).collect();
    let b: Vec<f16> = (0..n * k).map(|_| rand_f16(&mut seed)).collect();
    let want = reference(m, k, n, &a, &b);

    let dev = Device::open()?;
    println!("driver {} caps {:?}", dev.driver_name(), dev.caps());
    let mut c = vec![f16::ZERO; m * n];
    let t = Instant::now();
    one_shot(&dev, m, k, n, &a, &b, &mut c)?;
    report_f16("one_shot", &c, &want, t.elapsed().as_secs_f64() * 1e3);

    if let Some(threads) = args.mt {
        let t = Instant::now();
        one_shot_mt(m, k, n, &a, &b, &mut c, threads)?;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        report_f16(&format!("one_shot_mt({threads})"), &c, &want, ms);
    }

    if args.f32out {
        let mut cf = vec![0.0f32; m * n];
        let t = Instant::now();
        one_shot_f32out(&dev, m, k, n, &a, &b, &mut cf)?;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        report_f32("f32out", &cf, &want, ms);
        if m == 1 {
            println!("f32out: M=1 went through the height-4 pad (row 0 returned)");
        }
    }

    if args.bf16 {
        let a32: Vec<f32> = (0..m * k).map(|_| rand_f32(&mut seed)).collect();
        let b32: Vec<f32> = (0..n * k).map(|_| rand_f32(&mut seed)).collect();
        let want32 = reference(m, k, n, &a32, &b32);
        let mut c32 = vec![0.0f32; m * n];
        let mut name = "bf16".to_string();
        let t = Instant::now();
        match args.mt {
            Some(threads) => {
                one_shot_bf16_mt(m, k, n, &a32, &b32, &mut c32, threads)?;
                name = format!("bf16_mt({threads})");
            }
            None => one_shot_bf16(m, k, n, &a32, &b32, &mut c32)?,
        }
        let ms = t.elapsed().as_secs_f64() * 1e3;
        // The reference is the exact f32 product, so this line reports the bf16
        // truncation error as well as any engine error.
        report_f32(&format!("{name} vs f32"), &c32, &want32, ms);
    }

    if args.context {
        let threads = std::env::var("ROCKET_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4);
        let mut ctx = Context::new(threads)?;
        let t = Instant::now();
        let w = ctx.pack_weight(256, k, n, &b)?;
        println!(
            "pack_weight(256): {:.2} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
        ctx.matmul_prepacked(m, k, n, &a, &mut c, &w)?;
        rocket_npu::device::submit::reset_counters();
        let t = Instant::now();
        ctx.matmul_prepacked(m, k, n, &a, &mut c, &w)?;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        report_f16("prepacked", &c, &want, ms);
        println!(
            "submits {} tasks {}",
            rocket_npu::device::submit::ioctl_submit_count(),
            rocket_npu::device::submit::submitted_task_count()
        );
        if m != 4 {
            let a4 = &a[..4 * k];
            let want4 = reference(4, k, n, a4, &b);
            let mut c4 = vec![f16::ZERO; 4 * n];
            ctx.matmul_prepacked(4, k, n, a4, &mut c4, &w)?;
            let t = Instant::now();
            ctx.matmul_prepacked(4, k, n, a4, &mut c4, &w)?;
            report_f16("prepacked@4", &c4, &want4, t.elapsed().as_secs_f64() * 1e3);
        }
    }
    Ok(())
}
