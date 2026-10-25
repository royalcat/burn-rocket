//! On-device exact int16 self-test: `int16 x int16 -> int64` by the four int8 products.
//!
//!   int16_selftest [M K N]
//!
//! Inputs cover the whole i16 range, so the `[-32768, 32639]` clamp is exercised; the
//! CPU reference clamps identically and the comparison is exact.

use std::time::Instant;

use rocket_npu::Device;
use rocket_npu::matmul::one_shot_int16_exact;

fn rnd(seed: &mut u64) -> i16 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 33) as u16) as i16
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let m: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(64);
    let k: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(256);
    let n: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(128);
    println!("int16 M={m} K={k} N={n}");
    assert_eq!(m % 4, 0, "M must be a multiple of 4");
    assert_eq!(k % 32, 0, "K must be a multiple of 32");
    assert_eq!(n % 32, 0, "N must be a multiple of 32");

    let mut seed = 0x1234_5678_9abc_def1u64;
    let a: Vec<i16> = (0..m * k).map(|_| rnd(&mut seed)).collect();
    let b: Vec<i16> = (0..n * k).map(|_| rnd(&mut seed)).collect();

    let mut want = vec![0i64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0i64;
            for p in 0..k {
                let x = (a[i * k + p] as i32).clamp(-32768, 32639) as i64;
                let y = (b[j * k + p] as i32).clamp(-32768, 32639) as i64;
                sum += x * y;
            }
            want[i * n + j] = sum;
        }
    }

    let dev = Device::open()?;
    let mut got = vec![0i64; m * n];
    one_shot_int16_exact(&dev, m, k, n, &a, &b, &mut got)?;
    let t = Instant::now();
    one_shot_int16_exact(&dev, m, k, n, &a, &b, &mut got)?;
    let ms = t.elapsed().as_secs_f64() * 1e3;

    let max = got
        .iter()
        .zip(&want)
        .fold(0i64, |a, (g, w)| a.max((g - w).abs()));
    let exact = got == want;
    println!("exact {exact} max_abs {max} {ms:.2} ms");
    assert!(exact, "int16_exact must be i64-exact");
    Ok(())
}
