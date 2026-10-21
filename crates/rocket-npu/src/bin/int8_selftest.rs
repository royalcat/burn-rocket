//! On-device group-wise int8 self-test against an f64 dequantized reference.
//!
//!   int8_selftest [M K N GROUP]

use std::time::Instant;

use rocket_npu::matmul::I8Context;

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let m: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(64);
    let k: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(1024);
    let n: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(768);
    let group: usize = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(32);
    println!("int8 M={m} K={k} N={n} group={group}");
    assert_eq!(m % 4, 0, "M must be a multiple of 4");
    assert_eq!(k % group, 0);

    let mut seed = 0x1234_5678_9abc_def1u64;
    let af: Vec<f32> = (0..m * k).map(|_| rnd(&mut seed)).collect();
    let bf: Vec<f32> = (0..n * k).map(|_| rnd(&mut seed)).collect();
    let (a, a_scale) = quant_rows(&af, m, k, group);
    let (b, b_scale) = quant_rows(&bf, n, k, group);

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

    let mut ctx = I8Context::new(5)?;
    let w = ctx.pack_weight_gw(256, k, n, &b, group)?;
    let mut got = vec![0f32; m * n];
    ctx.matmul_gw(m, k, n, &a, &a_scale, &b_scale, &mut got, &w)?;
    let t = Instant::now();
    ctx.matmul_gw(m, k, n, &a, &a_scale, &b_scale, &mut got, &w)?;
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
