//! On-device tf32 one-shot self-test against a tf32-rounded reference.
//!
//!   tf32_selftest [M K N]
//!
//! tf32 is fp32 range + a 10-bit mantissa in a 4-byte container: the caller feeds raw
//! fp32 and the MAC rounds each operand to tf32. The reference is therefore an f64 dot
//! product of the **tf32-rounded** inputs — and the hardware's rounding is a
//! **truncation** (zero the low 13 mantissa bits, toward zero), which is exactly what
//! the C library's own `tf32rt` gate does. A round-to-nearest-even reference is printed
//! next to it to show the difference (it sits ~2^-11 away — a whole rounding step, not
//! an accumulation error), and the plain-f32 f64 dot product gives the rounding budget
//! in context.
//!
//! Without arguments the built-in shape list runs: a tiny tile, a medium tile, ragged
//! M/N tails, a Kt-straddling deep K (16384 > the tf32 Kt ceiling of 8192) and the
//! padded `M == 1` GEMV shim.

use std::time::Instant;

use rocket_npu::Device;
use rocket_npu::matmul::one_shot_tf32;

const SHAPES: [(usize, usize, usize); 6] = [
    (4, 16, 16),      // one tiny tile
    (64, 256, 128),   // medium
    (264, 320, 320),  // ragged M (nMt > 1) and non-power-of-two N
    (512, 1024, 512), // M, K and N all tile
    (64, 16384, 128), // K > Kt: two K-tiles summed in f64 on the host
    (1, 64, 32),      // the M == 1 shim: pad to 4 rows, return row 0
];

fn rnd(seed: &mut u64) -> f32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    ((*seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
}

/// The hardware's tf32 input rounding: zero the low 13 mantissa bits (round toward
/// zero). The C library's gates use the same `bits & 0xFFFFE000` (`tf32rt`), and the
/// only HW-vs-reference difference left is the accumulation order.
fn tf32(v: f32) -> f32 {
    f32::from_bits(v.to_bits() & 0xFFFF_E000)
}

/// Round-to-nearest-even, for the diagnostic column: what a "properly rounded" tf32
/// would give, i.e. one rounding step (up to 2^-11 relative) away from the hardware.
fn tf32_rne(v: f32) -> f32 {
    let bits = v.to_bits();
    f32::from_bits((bits.wrapping_add(0x0FFF + ((bits >> 13) & 1))) & 0xFFFF_E000)
}

/// `want` is the reference the hardware should match; `f32_ref` is the plain-f32 dot
/// product of the same data, i.e. what is lost to the 10-bit mantissa.
fn references(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let trunc = |v: f32| tf32(v) as f64;
    let rne = |v: f32| tf32_rne(v) as f64;
    let at: Vec<f64> = a.iter().map(|v| trunc(*v)).collect();
    let bt: Vec<f64> = b.iter().map(|v| trunc(*v)).collect();
    let ar: Vec<f64> = a.iter().map(|v| rne(*v)).collect();
    let br: Vec<f64> = b.iter().map(|v| rne(*v)).collect();
    let af: Vec<f64> = a.iter().map(|v| *v as f64).collect();
    let bf: Vec<f64> = b.iter().map(|v| *v as f64).collect();
    let mut want = vec![0.0f64; m * n];
    let mut want_rne = vec![0.0f64; m * n];
    let mut want_f32 = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let (mut s, mut r, mut f) = (0.0f64, 0.0f64, 0.0f64);
            for p in 0..k {
                let (x, y) = (i * k + p, j * k + p);
                s += at[x] * bt[y];
                r += ar[x] * br[y];
                f += af[x] * bf[y];
            }
            want[i * n + j] = s;
            want_rne[i * n + j] = r;
            want_f32[i * n + j] = f;
        }
    }
    (want, want_rne, want_f32)
}

/// Cosine, max |difference| and the max difference relative to the reference's peak
/// (the C gates normalize to the global output scale for the same reason: nearly
/// cancelling random dot products make per-element relative error meaningless).
fn compare(got: &[f32], want: &[f64]) -> (f64, f64, f64) {
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    let mut max = 0f64;
    let mut peak = 0f64;
    for (g, w) in got.iter().zip(want) {
        let g = *g as f64;
        dot += g * w;
        na += g * g;
        nb += w * w;
        max = max.max((g - w).abs());
        peak = peak.max(w.abs());
    }
    (
        dot / (na.sqrt() * nb.sqrt()),
        max,
        if peak > 0.0 { max / peak } else { 0.0 },
    )
}

fn run(dev: &Device, m: usize, k: usize, n: usize, seed: &mut u64) -> (f64, f64, f64, f64) {
    let a: Vec<f32> = (0..m * k).map(|_| rnd(seed)).collect();
    let b: Vec<f32> = (0..n * k).map(|_| rnd(seed)).collect();
    let (want, want_rne, want_f32) = references(&a, &b, m, k, n);

    let mut got = vec![0.0f32; m * n];
    let t = Instant::now();
    one_shot_tf32(dev, m, k, n, &a, &b, &mut got).expect("tf32 one-shot");
    let ms = t.elapsed().as_secs_f64() * 1e3;

    let (cos, max, rel) = compare(&got, &want);
    let (cos_rne, _, rel_rne) = compare(&got, &want_rne);
    let (cos_f32, _, rel_f32) = compare(&got, &want_f32);
    println!(
        "tf32 M={m} K={k} N={n}: cosine {cos:.9} max_abs {max:.3e} max_rel {rel:.3e} \
         | vs RNE ref {cos_rne:.9}/{rel_rne:.2e} | vs f32 ref {cos_f32:.9}/{rel_f32:.2e} \
         | {ms:.2} ms"
    );
    (cos, max, rel, cos_f32)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut seed = 0x1234_5678_9abc_def1u64;
    let dev = Device::open()?;

    let shapes: Vec<(usize, usize, usize)> = if args.len() >= 3 {
        let m: usize = args[0].parse()?;
        let k: usize = args[1].parse()?;
        let n: usize = args[2].parse()?;
        vec![(m, k, n)]
    } else {
        SHAPES.to_vec()
    };

    let mut worst_cos = 1.0f64;
    let mut worst_rel = 0.0f64;
    let mut worst_f32 = 0.0f64;
    for (m, k, n) in shapes {
        assert!(m == 1 || m % 4 == 0, "M must be a multiple of 4 (or 1)");
        assert_eq!(k % 16, 0, "K must be a multiple of 16");
        assert_eq!(n % 16, 0, "N must be a multiple of 16");
        let (cos, max, rel, _cos_f32) = run(&dev, m, k, n, &mut seed);
        worst_cos = worst_cos.min(cos);
        worst_rel = worst_rel.max(rel);
        worst_f32 = worst_f32.max(max);
    }
    println!(
        "worst: cosine {worst_cos:.9} max_rel {worst_rel:.3e} (f32-ref max_abs {worst_f32:.3e})"
    );
    // The tf32-rounded (truncated) reference is the contract: the only difference left
    // is the device's fp32 K-accumulation order. The C's tiled gate sits at 1e-5 of the
    // global scale ("one dropped K term reads ~3e-3"), which a correct path clears by
    // two orders of magnitude.
    assert!(
        worst_cos >= 0.999999,
        "tf32 must match the tf32-rounded reference: cosine {worst_cos}"
    );
    assert!(
        worst_rel <= 1e-5,
        "tf32 max_rel {worst_rel} exceeds the fp32-accumulation envelope"
    );
    Ok(())
}
