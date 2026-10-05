//! Probe: open the NPU, pack a resident fp16 weight, run one matmul, verify
//! against a CPU reference and print timing/counters.
//!
//! Run on the board from the repo root:
//! `ROCKETNPU_DIR=/root/npu-poc/rocket-userspace/build ./target/release/examples/probe`

use burn_rocket::{RocketCtx, f32_to_f16};
use half::f16;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "driver: {:?}, big cores: {}",
        burn_rocket::driver_name(),
        burn_rocket::num_big_cores()
    );

    let args: Vec<String> = std::env::args().collect();
    let m: usize = args.get(1).map(|s| s.parse()).transpose()?.unwrap_or(3636);
    let k: usize = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(1024);
    let n: usize = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(3072);

    let ctx = RocketCtx::new(5)?;

    // Deterministic pseudo-random inputs in [-0.5, 0.5).
    let mut state = 0x1234_5678_9abc_def0u64;
    let mut rnd = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
    };
    let a32: Vec<f32> = (0..m * k).map(|_| rnd()).collect();
    let b32: Vec<f32> = (0..n * k).map(|_| rnd()).collect();
    let a = f32_to_f16(&a32);
    let b = f32_to_f16(&b32);

    burn_rocket::reset_submit_counters();
    let w = ctx.pack_weight(512, k, n, &b)?;
    println!("packed resident weight [{n} x {k}]");

    let mut c = vec![f16::ZERO; m * n];
    let t0 = std::time::Instant::now();
    ctx.matmul_prepacked(m, k, n, &a, &mut c, &w)?;
    let dt = t0.elapsed().as_secs_f64();
    let gflop = 2.0 * (m as f64) * (k as f64) * (n as f64) / 1e9;
    println!(
        "matmul [{m},{k}] x [{n},{k}]^T: {:.2} ms  {:.1} GFLOP/s",
        dt * 1e3,
        gflop / dt
    );

    // Verify four sampled rows against a CPU reference.
    let rows = [0usize, 1, m / 2, m - 1];
    let mut worst = 0f32;
    for &r in &rows {
        for j in 0..n {
            let mut acc = 0f32;
            for i in 0..k {
                acc += a[r * k + i].to_f32() * b[j * k + i].to_f32();
            }
            worst = worst.max((acc - c[r * n + j].to_f32()).abs());
        }
    }
    println!("worst abs error on 4 rows x {n} cols: {worst:.5}");

    let (ioctls, tasks) = burn_rocket::submit_counters();
    println!("submits since reset: {ioctls} ioctls, {tasks} tasks");
    Ok(())
}
