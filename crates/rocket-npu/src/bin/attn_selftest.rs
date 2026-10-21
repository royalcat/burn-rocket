//! On-device attention self-test against an f64 reference.

use std::time::Instant;

use half::f16;
use rocket_npu::attn::Attention;

fn rnd(seed: &mut u64) -> f16 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    f16::from_f32(((*seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5)
}

#[allow(clippy::too_many_arguments)]
fn ref_attn(
    t: usize,
    kv: usize,
    dk: usize,
    dv: usize,
    h: usize,
    hkv: usize,
    scale: f32,
    q: &[f16],
    k: &[f16],
    v: &[f16],
    mask: Option<&[f16]>,
) -> Vec<f64> {
    let gqa = h / hkv;
    let mut out = vec![0.0f64; h * t * dv];
    for head in 0..h {
        let hk = head / gqa;
        for row in 0..t {
            let mut sc = vec![0.0f64; kv];
            let mut mx = f64::NEG_INFINITY;
            for j in 0..kv {
                let mut s = 0.0f64;
                for c in 0..dk {
                    s += q[(head * t + row) * dk + c].to_f64() * k[(hk * kv + j) * dk + c].to_f64();
                }
                let mut s = (s as f32) * scale;
                if let Some(m) = mask {
                    s += m[row * kv + j].to_f32();
                }
                if s < -30000.0 {
                    s = -30000.0;
                }
                sc[j] = s as f64;
                mx = mx.max(sc[j]);
            }
            let mut sum = 0.0f64;
            for x in &mut sc {
                *x = (*x - mx).exp();
                sum += *x;
            }
            for c in 0..dv {
                let mut a = 0.0f64;
                for j in 0..kv {
                    a += sc[j] * v[(hk * dv + c) * kv + j].to_f64();
                }
                out[(head * t + row) * dv + c] = a / sum;
            }
        }
    }
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let t: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(82);
    let kv: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(t);
    let heads: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(8);
    let hkv: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(heads);
    let dk = 128;
    let dv = 128;
    let scale = 1.0 / (dk as f32).sqrt();
    println!("attn T={t} kv={kv} heads={heads} kv_heads={hkv} dk={dk}");

    let mut seed = 0xdead_beef_1234_5678u64;
    let q: Vec<f16> = (0..heads * t * dk).map(|_| rnd(&mut seed)).collect();
    let k: Vec<f16> = (0..hkv * kv * dk).map(|_| rnd(&mut seed)).collect();
    let v: Vec<f16> = (0..hkv * dv * kv).map(|_| rnd(&mut seed)).collect();
    let mut mask = vec![f16::ZERO; t * kv];
    for i in 0..t {
        for j in 0..kv {
            if j > i {
                mask[i * kv + j] = f16::NEG_INFINITY;
            }
        }
    }
    let want = ref_attn(t, kv, dk, dv, heads, hkv, scale, &q, &k, &v, Some(&mask));

    let threads = std::env::var("ROCKET_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let mut attn = Attention::new(threads)?;
    let mut out = vec![f16::ZERO; heads * t * dv];
    attn.flash(
        t,
        kv,
        dk,
        dv,
        heads,
        hkv,
        scale,
        0.0,
        &q,
        &k,
        &v,
        Some(&mask),
        &mut out,
    )?;
    let t0 = Instant::now();
    attn.flash(
        t,
        kv,
        dk,
        dv,
        heads,
        hkv,
        scale,
        0.0,
        &q,
        &k,
        &v,
        Some(&mask),
        &mut out,
    )?;
    let ms = t0.elapsed().as_secs_f64() * 1e3;

    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    let mut max: f64 = 0.0;
    for (g, w) in out.iter().zip(&want) {
        let g = g.to_f64();
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
