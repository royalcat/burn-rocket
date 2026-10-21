//! On-device native-bf16 stream self-test against an f64 reference.
//!
//!   bf16_selftest [M K N]

use std::time::Instant;

use rocket_npu::matmul::Bf16Stream;

fn rnd(seed: &mut u64) -> f32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    ((*seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
}

fn bf16(v: f32) -> f32 {
    f32::from_bits(v.to_bits() & 0xffff_0000)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let m: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(64);
    let k: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(1024);
    let n: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(768);
    println!("bf16 M={m} K={k} N={n}");

    let mut seed = 0x1234_5678_9abc_def1u64;
    let a: Vec<f32> = (0..m * k).map(|_| rnd(&mut seed)).collect();
    let b: Vec<f32> = (0..n * k).map(|_| rnd(&mut seed)).collect();
    let mut want = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f64;
            for p in 0..k {
                sum += bf16(a[i * k + p]) as f64 * bf16(b[j * k + p]) as f64;
            }
            want[i * n + j] = sum;
        }
    }

    let mut s = Bf16Stream::new(5)?;
    let mut got = vec![0f32; m * n];
    s.matmul(m, k, n, &a, &b, &mut got)?;
    let t = Instant::now();
    s.matmul(m, k, n, &a, &b, &mut got)?;
    let ms = t.elapsed().as_secs_f64() * 1e3;

    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    let mut max = 0f64;
    for (g, w) in got.iter().zip(&want) {
        let g = *g as f64;
        dot += g * w;
        na += g * g;
        nb += w * w;
        max = max.max((g - w).abs());
    }
    println!(
        "cosine {:.8} max_abs {max:.6} {ms:.2} ms",
        dot / (na.sqrt() * nb.sqrt())
    );
    Ok(())
}
