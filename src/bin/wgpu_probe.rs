//! Phase-1 probe for the Vulkan (CubeCL/wgpu → panvk) GPU backend on the Rock 5B+.
//!
//! Build (dev host, cross): use `cargo zigbuild` — the plain GNU cross toolchain
//! links against glibc 2.44, newer than the board's 2.41:
//!   cargo zigbuild --release --target aarch64-unknown-linux-gnu.2.41 \
//!       --no-default-features --features gpu-wgsl --bin wgpu_probe
//!
//! Run on the board (from /root/embeddings-fast):
//!   ./wgpu_probe --seq 3633 --reps 2
//!
//! Prints matmul GFLOPS (f16/f32), causal attention seconds/GFLOPS at the model's
//! shapes, per-call times (the panthor job watchdog is ~1 s), and a cosine check
//! of the GPU attention against the Flex (CPU) backend at a short sequence.

use std::time::Instant;

use burn::prelude::*;
use burn::tensor::module::attention;
use burn::tensor::ops::AttentionModuleOptions;
use burn::tensor::{DType, DeviceKind, TensorData};

fn main() {
    let seq = arg_usize("--seq", 3633);
    let reps = arg_usize("--reps", 2);
    let check_seq = arg_usize("--check-seq", 256);
    let m = arg_usize("--m", 3636);
    let n = arg_usize("--n", 5120);
    let k = arg_usize("--k", 1024);
    let gemm_dtypes = parse_dtypes(&arg_value("--gemm-dtype", "both"));
    let attn_dtypes = parse_dtypes(&arg_value("--attn-dtype", "f16"));
    let skip_gemm = has_flag("--skip-gemm");
    let skip_attn = has_flag("--skip-attn");
    let skip_check = has_flag("--skip-check");
    let kind = match arg_value("--device", "igpu").as_str() {
        "igpu" => DeviceKind::IntegratedGpu(0),
        "cpu" => DeviceKind::Cpu,
        other => panic!("--device must be igpu|cpu, got '{other}'"),
    };

    // IntegratedGpu(0) only matches wgpu's `DeviceType::IntegratedGpu` adapters, so
    // it cannot silently land on llvmpipe (which wgpu reports as `Cpu`).
    let kind_label = format!("{kind:?}");
    #[cfg(feature = "gpu-spirv")]
    let gpu = Device::vulkan(kind);
    #[cfg(all(feature = "gpu-wgsl", not(feature = "gpu-spirv")))]
    let gpu = Device::wgpu(kind);
    gpu.sync().expect("initial sync");
    println!("[probe] wgpu device up: {kind_label}");

    if !skip_gemm {
        smoke(&gpu);
        for dtype in &gemm_dtypes {
            gemm(&gpu, "main", m, k, n, *dtype, reps);
        }
    }

    if !skip_attn {
        for dtype in &attn_dtypes {
            attention_bench(&gpu, seq, 16, 128, *dtype, reps);
        }
    }

    if !skip_check {
        attention_check(check_seq, 16, 128, &gpu);
    }

    println!("[probe] done");
}

fn sync(device: &Device) {
    device.sync().expect("device sync");
}

/// Step-by-step GPU smoke test: the first op that trips the panthor job watchdog
/// (or crashes) shows up by name.
fn smoke(device: &Device) {
    let t0 = Instant::now();
    let a = Tensor::<2>::from_data(TensorData::zeros::<f32, _>([256, 256]), device);
    sync(device);
    println!("[smoke] from_data 256x256: {:.3}s", t0.elapsed().as_secs_f64());

    let t0 = Instant::now();
    let b = a.clone().cast(DType::F16).cast(DType::F32);
    let _ = b.to_data();
    sync(device);
    println!("[smoke] cast f32->f16->f32: {:.3}s", t0.elapsed().as_secs_f64());

    let t0 = Instant::now();
    let c = a.clone() + b.clone();
    let _ = c.to_data();
    sync(device);
    println!("[smoke] add: {:.3}s", t0.elapsed().as_secs_f64());

    let t0 = Instant::now();
    let d = a.clone().matmul(a.clone());
    let _ = d.to_data();
    sync(device);
    println!("[smoke] matmul 256^3: {:.3}s", t0.elapsed().as_secs_f64());

    let t0 = Instant::now();
    let e = a.clone().cast(DType::F16).matmul(b.cast(DType::F16)).sum();
    let _ = e.to_data();
    sync(device);
    println!("[smoke] matmul f16 256^3 + sum: {:.3}s", t0.elapsed().as_secs_f64());
}

fn gemm(device: &Device, label: &str, m: usize, k: usize, n: usize, dtype: DType, reps: usize) {
    let a = Tensor::<2>::from_data(TensorData::zeros::<f32, _>([m, k]), device).cast(dtype);
    let b = Tensor::<2>::from_data(TensorData::zeros::<f32, _>([k, n]), device).cast(dtype);
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    let mut best = f64::MAX;
    for i in 0..=reps {
        let t0 = Instant::now();
        let out = a.clone().matmul(b.clone());
        let _ = out.to_data();
        sync(device);
        let dt = t0.elapsed().as_secs_f64();
        best = best.min(dt);
        let tag = if i == 0 { " (warmup)" } else { "" };
        println!(
            "[gemm] {label} m={m} k={k} n={n} {dtype:?}: {dt:.3}s => {:.1} GFLOPS{tag}",
            flops / dt / 1e9
        );
    }
    println!(
        "[gemm] {label} {dtype:?}: best {best:.3}s => {:.1} GFLOPS",
        flops / best / 1e9
    );
}

fn attention_bench(
    device: &Device,
    seq: usize,
    heads: usize,
    head_dim: usize,
    dtype: DType,
    reps: usize,
) {
    let shape = [1, heads, seq, head_dim];
    let q = tensor4(device, shape, dtype, 11);
    let k = tensor4(device, shape, dtype, 22);
    let v = tensor4(device, shape, dtype, 33);
    let opts = AttentionModuleOptions {
        scale: Some(1.0 / (head_dim as f64).sqrt()),
        softcap: None,
        is_causal: true,
    };
    // QK^T + PV = 4 * heads * seq^2 * head_dim FLOP per batch (batch = 1).
    let flops = 4.0 * heads as f64 * (seq as f64).powi(2) * head_dim as f64;
    let elem = if dtype == DType::F16 { 2 } else { 4 };
    println!(
        "[attn] seq={seq} heads={heads} d={head_dim} {dtype:?}: q/k/v {:.1} MiB each",
        (heads * seq * head_dim * elem) as f64 / (1024.0 * 1024.0)
    );
    let mut best = f64::MAX;
    for i in 0..=reps {
        let t0 = Instant::now();
        let out = attention(q.clone(), k.clone(), v.clone(), None, None, opts);
        let _ = out.to_data();
        sync(device);
        let dt = t0.elapsed().as_secs_f64();
        best = best.min(dt);
        let tag = if i == 0 { " (warmup, includes tuning)" } else { "" };
        println!(
            "[attn] {dtype:?}: {dt:.3}s => {:.1} GFLOPS{tag}",
            flops / dt / 1e9
        );
    }
    println!(
        "[attn] {dtype:?}: best {best:.3}s => {:.1} GFLOPS (panthor job watchdog ~1 s)",
        flops / best / 1e9
    );
}

/// Cosine between the Flex (CPU, f32) attention and the GPU attention at a short
/// sequence, for both f16 and f32 GPU dtypes.
fn attention_check(seq: usize, heads: usize, head_dim: usize, gpu: &Device) {
    let flex = Device::flex();
    let shape = [1, heads, seq, head_dim];
    let n = heads * seq * head_dim;
    let qd = host_data(n, 101);
    let kd = host_data(n, 202);
    let vd = host_data(n, 303);
    let opts = AttentionModuleOptions {
        scale: Some(1.0 / (head_dim as f64).sqrt()),
        softcap: None,
        is_causal: true,
    };

    let qf = Tensor::<4>::from_data(TensorData::new(qd.clone(), shape), &flex);
    let kf = Tensor::<4>::from_data(TensorData::new(kd.clone(), shape), &flex);
    let vf = Tensor::<4>::from_data(TensorData::new(vd.clone(), shape), &flex);
    let of: Vec<f32> = attention(qf, kf, vf, None, None, opts)
        .cast(DType::F32)
        .to_data()
        .try_to_vec()
        .expect("flex out");
    sync(&flex);

    for dtype in [DType::F16, DType::F32] {
        let qg = Tensor::<4>::from_data(TensorData::new(qd.clone(), shape), gpu).cast(dtype);
        let kg = Tensor::<4>::from_data(TensorData::new(kd.clone(), shape), gpu).cast(dtype);
        let vg = Tensor::<4>::from_data(TensorData::new(vd.clone(), shape), gpu).cast(dtype);
        let og: Vec<f32> = attention(qg, kg, vg, None, None, opts)
            .cast(DType::F32)
            .to_data()
            .try_to_vec()
            .expect("gpu out");
        sync(gpu);
        println!(
            "[check] seq={seq} gpu {dtype:?} vs flex f32: cosine = {:.6}",
            cosine(&of, &og)
        );
    }
}

fn tensor4(device: &Device, shape: [usize; 4], dtype: DType, seed: u32) -> Tensor<4> {
    let n: usize = shape.iter().product();
    Tensor::<4>::from_data(TensorData::new(host_data(n, seed), shape), device).cast(dtype)
}

/// Deterministic pseudo-random values in `[-0.5, 0.5)`.
fn host_data(n: usize, seed: u32) -> Vec<f32> {
    let mut x = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((x >> 8) as f32) / 16_777_216.0 - 0.5
        })
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (*x as f64, *y as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn parse_dtypes(s: &str) -> Vec<DType> {
    match s {
        "f16" => vec![DType::F16],
        "f32" => vec![DType::F32],
        "both" => vec![DType::F16, DType::F32],
        other => panic!("--dtype must be f16|f32|both, got '{other}'"),
    }
}

fn arg_value(name: &str, default: &str) -> String {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| default.to_string())
}

fn arg_usize(name: &str, default: usize) -> usize {
    let v = arg_value(name, "");
    if v.is_empty() {
        default
    } else {
        v.parse().unwrap_or_else(|_| panic!("bad value for {name}: '{v}'"))
    }
}

fn has_flag(name: &str) -> bool {
    std::env::args().any(|a| a == name)
}
