//! On-device numeric self-test for the fp16 path.
//!
//!   selftest [M K N] [--context]
//!
//! Compares the NPU result with an f64 reference computed from the same f16 inputs
//! and prints the cosine / max error plus wall time. `--context` also exercises the
//! resident-weight path at a canonical pack M (256) and multiple call Ms.

use std::time::Instant;

use half::f16;
use rocket_npu::Device;
use rocket_npu::matmul::{Context, one_shot};

fn rand_f16(seed: &mut u64) -> f16 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    let v = ((*seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5;
    f16::from_f32(v)
}

fn reference(m: usize, k: usize, n: usize, a: &[f16], b: &[f16]) -> Vec<f64> {
    let mut out = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f64;
            for p in 0..k {
                sum += a[i * k + p].to_f64() * b[j * k + p].to_f64();
            }
            out[i * n + j] = sum;
        }
    }
    out
}

fn report(name: &str, got: &[f16], want: &[f64], ms: f64) {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    let mut max = 0.0f64;
    for (g, w) in got.iter().zip(want) {
        let g = g.to_f64();
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1).filter(|a| a != "--context");
    let mut dims = args.by_ref().take(3).map(|s| s.parse::<usize>());
    let m = dims.next().transpose()?.unwrap_or(64);
    let k = dims.next().transpose()?.unwrap_or(256);
    let n = dims.next().transpose()?.unwrap_or(128);
    let context = std::env::args().any(|a| a == "--context");
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
    report("one_shot", &c, &want, t.elapsed().as_secs_f64() * 1e3);

    if context {
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
        report("prepacked", &c, &want, ms);
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
            report("prepacked@4", &c4, &want4, t.elapsed().as_secs_f64() * 1e3);
        }
    }
    Ok(())
}
