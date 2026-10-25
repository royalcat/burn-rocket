//! On-device int8 self-tests: the resident group-wise path (`--npu-int8`), the dense
//! (per-channel) one-shot and resident paths, and the one-shot group-wise twin.
//!
//!   int8_selftest [MODE] [M K N GROUP]
//!
//! `MODE` defaults to `--prepacked-gw`.
//!
//! - `--prepacked-gw` — [`I8Context::pack_weight_gw`] + [`I8Context::matmul_gw`], the
//!   deployed `--npu-int8` path; compared with an f64 dequantized reference.
//! - `--gw-one-shot` — [`one_shot_int8_groupwise`], the same reference; also compared
//!   against the resident path (they must agree bit-for-bit).
//! - `--prepacked-dense` — [`I8Context::pack_weight`] + [`I8Context::matmul`], raw
//!   int32; compared exactly with an i64 CPU reference.
//! - `--dense` — [`one_shot_int8`], the same exact comparison.

use std::time::Instant;

use rocket_npu::Device;
use rocket_npu::matmul::{I8Context, one_shot_int8, one_shot_int8_groupwise};

fn rnd(seed: &mut u64) -> f32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    ((*seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
}

fn quant_rows(x: &[f32], rows: usize, k: usize, group: usize) -> (Vec<i8>, Vec<f32>) {
    let ng = k / group;
    let mut q = vec![0i8; rows * k];
    let mut s = vec![0f32; rows * ng];
    for r in 0..rows {
        for g in 0..ng {
            let base = r * k + g * group;
            let mx = x[base..base + group]
                .iter()
                .fold(0f32, |a, v| a.max(v.abs()));
            let sc = if mx > 0.0 { mx / 127.0 } else { 1.0 };
            s[r * ng + g] = sc;
            for j in 0..group {
                let v = (x[base + j] / sc).round().clamp(-127.0, 127.0);
                q[base + j] = v as i8;
            }
        }
    }
    (q, s)
}

/// Cosine, max |difference| and exact equality of two f64 vectors.
fn compare(got: &[f64], want: &[f64]) -> (f64, f64, bool) {
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    let mut max = 0f64;
    let mut exact = true;
    for (g, w) in got.iter().zip(want) {
        dot += g * w;
        na += g * g;
        nb += w * w;
        max = max.max((g - w).abs());
        exact &= g == w;
    }
    (dot / (na.sqrt() * nb.sqrt()), max, exact)
}

fn dequant_reference(
    a: &[i8],
    b: &[i8],
    a_scale: &[f32],
    b_scale: &[f32],
    dims: (usize, usize, usize, usize),
) -> Vec<f64> {
    let (m, k, n, group) = dims;
    let ng = k / group;
    let mut want = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f64;
            for p in 0..k {
                let g = p / group;
                sum += (a[i * k + p] as f64)
                    * (b[j * k + p] as f64)
                    * (a_scale[i * ng + g] as f64)
                    * (b_scale[j * ng + g] as f64);
            }
            want[i * n + j] = sum;
        }
    }
    want
}

fn exact_reference(a: &[i8], b: &[i8], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut want = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0i64;
            for p in 0..k {
                sum += a[i * k + p] as i64 * b[j * k + p] as i64;
            }
            want[i * n + j] = sum as f64;
        }
    }
    want
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut mode = "prepacked-gw";
    let mut dims: Vec<String> = Vec::new();
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--dense" => mode = "dense",
            "--gw-one-shot" => mode = "gw-one-shot",
            "--prepacked-dense" => mode = "prepacked-dense",
            "--prepacked-gw" => mode = "prepacked-gw",
            other => dims.push(other.to_string()),
        }
    }
    let num = |i: usize, d: usize| dims.get(i).and_then(|v| v.parse().ok()).unwrap_or(d);
    let m: usize = num(0, 64);
    let k: usize = num(1, 1024);
    let n: usize = num(2, 768);
    let group: usize = num(3, 32);
    println!("int8 {mode} M={m} K={k} N={n} group={group}");
    assert_eq!(m % 4, 0, "M must be a multiple of 4");
    assert_eq!(k % 32, 0, "K must be a multiple of 32");
    assert_eq!(n % 32, 0, "N must be a multiple of 32");

    let mut seed = 0x1234_5678_9abc_def1u64;
    let af: Vec<f32> = (0..m * k).map(|_| rnd(&mut seed)).collect();
    let bf: Vec<f32> = (0..n * k).map(|_| rnd(&mut seed)).collect();

    match mode {
        "prepacked-dense" | "dense" => {
            // One scale per row/channel: uniform int8 over the whole K.
            let (a, _) = quant_rows(&af, m, k, k);
            let (b, _) = quant_rows(&bf, n, k, k);
            let want = exact_reference(&a, &b, m, k, n);

            let t = Instant::now();
            let got: Vec<i32> = if mode == "dense" {
                let dev = Device::open()?;
                let mut got = vec![0i32; m * n];
                one_shot_int8(&dev, m, k, n, &a, &b, &mut got)?;
                got
            } else {
                let mut ctx = I8Context::new(5)?;
                let w = ctx.pack_weight(256, k, n, &b)?;
                let mut got = vec![0i32; m * n];
                ctx.matmul(m, k, n, &a, &mut got, &w)?;
                got
            };
            let ms = t.elapsed().as_secs_f64() * 1e3;
            let got: Vec<f64> = got.iter().map(|v| *v as f64).collect();
            let (cos, max, exact) = compare(&got, &want);
            println!("cosine {cos:.8} max_abs {max:.6} exact {exact} {ms:.2} ms",);
            assert!(exact, "int8 dense must be int32-exact");
        }
        _ => {
            assert_eq!(group % 32, 0, "group must be a multiple of 32");
            assert_eq!(k % group, 0, "K must be a multiple of group");
            let (a, a_scale) = quant_rows(&af, m, k, group);
            let (b, b_scale) = quant_rows(&bf, n, k, group);
            let want = dequant_reference(&a, &b, &a_scale, &b_scale, (m, k, n, group));

            let mut ctx = I8Context::new(5)?;
            let w = ctx.pack_weight_gw(256, k, n, &b, group)?;
            let t = Instant::now();
            let mut resident = vec![0f32; m * n];
            ctx.matmul_gw(m, k, n, &a, &a_scale, &b_scale, &mut resident, &w)?;
            let ms = t.elapsed().as_secs_f64() * 1e3;

            let got: Vec<f32> = if mode == "gw-one-shot" {
                let dev = Device::open()?;
                let mut one = vec![0f32; m * n];
                one_shot_int8_groupwise(
                    &dev, m, k, n, &a, &b, &a_scale, &b_scale, &mut one, group,
                )?;
                let same = one == resident;
                let max = one
                    .iter()
                    .zip(&resident)
                    .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
                println!("one-shot vs resident: bit-identical {same} max_abs {max:e}");
                one
            } else {
                resident
            };

            let got64: Vec<f64> = got.iter().map(|v| *v as f64).collect();
            let (cos, max, _) = compare(&got64, &want);
            println!("cosine {cos:.8} max_abs {max:.6} {ms:.2} ms");
        }
    }
    Ok(())
}
