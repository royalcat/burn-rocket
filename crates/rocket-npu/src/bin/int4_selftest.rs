//! On-device int4 self-test: the dense one-shot against a CPU int64 reference.
//!
//!   int4_selftest [--kt-cap N] [--extreme] [M K N]
//!
//! Shapes default to `M=64 K=256 N=128`; `K%32==0`, `N%64==0`, `M%4==0` (or `M==1`,
//! which the one-shot pads). `--kt-cap` maps to [`one_shot_int4_ex`]'s int16-saturation
//! cap: `0` (the default) is the C `rocket_matmul_int4`, `480` the C's in-model value,
//! which keeps `|partial| <= 64*Kt < 32767` for `[-8,7]` operands. `--extreme` fills both
//! operands with `-8` (worst case: every K-tile partial is `64*Kt`), which forces the
//! saturation regime at `kt_cap=0` and shows the tiled reference reproducing it.
//!
//! Two references are checked. The plain int64 sum is only reachable when no K-tile
//! partial overflows int16 — the conv writes int16, so a bigger `Kt` saturates and the
//! host cannot recover it. The second reference reproduces the tiling (the plan's `Kt`)
//! and clamps each tile's partial to int16 first, which is exact in *both* regimes and
//! therefore also validates that the tiling the one-shot used is the planned one.

use std::time::Instant;

use rocket_npu::Device;
use rocket_npu::matmul::{one_shot_int4_ex, plan_int4};

/// A xorshift64 -> uniform f32 in [-0.5, 0.5), the int8_selftest generator.
fn rnd(seed: &mut u64) -> f32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    ((*seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
}

/// Symmetric int4 codes, one per `i8`, in `[-8,7]`.
fn int4_codes(x: &[f32]) -> Vec<i8> {
    x.iter()
        .map(|v| ((v * 16.0).round() as i32).clamp(-8, 7) as i8)
        .collect()
}

/// `C = A * B^T` summed exactly in int64 over the whole K.
fn exact_reference(a: &[i8], b: &[i8], m: usize, k: usize, n: usize) -> Vec<i64> {
    let mut want = vec![0i64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0i64;
            for p in 0..k {
                sum += a[i * k + p] as i64 * b[j * k + p] as i64;
            }
            want[i * n + j] = sum;
        }
    }
    want
}

/// The same sum with each K-tile's partial saturated to int16 — what the hardware's
/// int16 output writer produces, per `kt`.
fn tiled_reference(a: &[i8], b: &[i8], m: usize, k: usize, n: usize, kt: usize) -> Vec<i64> {
    let mut want = vec![0i64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0i64;
            let mut k0 = 0;
            while k0 < k {
                let k1 = (k0 + kt).min(k);
                let mut tile = 0i64;
                for p in k0..k1 {
                    tile += a[i * k + p] as i64 * b[j * k + p] as i64;
                }
                sum += tile.clamp(i16::MIN as i64, i16::MAX as i64);
                k0 = k1;
            }
            want[i * n + j] = sum;
        }
    }
    want
}

fn compare(got: &[i64], want: &[i64]) -> (i64, bool) {
    let max = got
        .iter()
        .zip(want)
        .fold(0i64, |a, (g, w)| a.max((g - w).abs()));
    let exact = got.iter().zip(want).all(|(g, w)| g == w);
    (max, exact)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut kt_cap = 0usize;
    let mut extreme = false;
    let mut dims: Vec<String> = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--kt-cap" => {
                kt_cap = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--kt-cap needs a value")?
            }
            "--extreme" => extreme = true,
            other => dims.push(other.to_string()),
        }
    }
    let num = |i: usize, d: usize| dims.get(i).and_then(|v| v.parse().ok()).unwrap_or(d);
    let m: usize = num(0, 64);
    let k: usize = num(1, 256);
    let n: usize = num(2, 128);
    println!("int4 one-shot M={m} K={k} N={n} kt_cap={kt_cap} extreme={extreme}");
    assert_eq!(k % 32, 0, "K must be a multiple of 32");
    assert_eq!(n % 64, 0, "N must be a multiple of 64");
    assert!(
        m == 1 || m.is_multiple_of(4),
        "M must be 1 or a multiple of 4"
    );

    let (mt, kt, nt) = plan_int4(m.max(4), k, n)?;
    let capped = if kt_cap >= 32 && kt > kt_cap {
        ((kt_cap / 32) * 32).max(32)
    } else {
        kt
    };
    println!(
        "plan: Mt={mt} Kt={kt} Nt={nt} jobs={} | capped Kt={capped} nKt={}",
        m.div_ceil(mt) * n.div_ceil(nt) * k.div_ceil(kt),
        k.div_ceil(capped)
    );

    let (a, b): (Vec<i8>, Vec<i8>) = if extreme {
        (vec![-8i8; m * k], vec![-8i8; n * k])
    } else {
        let mut seed = 0x1234_5678_9abc_def1u64;
        let af: Vec<f32> = (0..m * k).map(|_| rnd(&mut seed)).collect();
        let bf: Vec<f32> = (0..n * k).map(|_| rnd(&mut seed)).collect();
        (int4_codes(&af), int4_codes(&bf))
    };

    let dev = Device::open()?;
    let mut got = vec![0i32; m * n];
    let t = Instant::now();
    one_shot_int4_ex(&dev, m, k, n, &a, &b, &mut got, kt_cap)?;
    let ms = t.elapsed().as_secs_f64() * 1e3;
    let got: Vec<i64> = got.iter().map(|v| *v as i64).collect();

    let want = exact_reference(&a, &b, m, k, n);
    let (max, exact) = compare(&got, &want);
    let tiled = tiled_reference(&a, &b, m, k, n, capped);
    let (tmax, texact) = compare(&got, &tiled);
    println!("exact {exact} max_abs {max} | tiled-exact {texact} max_abs {tmax} | {ms:.2} ms");
    if exact {
        println!("OK: int32-exact vs the int64 reference");
    } else if texact {
        println!(
            "OK: int32-exact vs the int16-clamped tiled reference; the plain int64 sum \
             differs because Kt={capped} lets a tile partial saturate int16 \
             (pass --kt-cap 480 for full-range [-8,7] data)"
        );
    } else {
        return Err(format!(
            "int4 mismatch: neither reference matches (max_abs {tmax} vs the tiled one)"
        )
        .into());
    }
    Ok(())
}
