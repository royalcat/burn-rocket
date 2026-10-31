//! `gemm`: flex matmul microbenchmark (no model).

use std::time::Instant;

use anyhow::{Result, bail};
use burn::prelude::*;
use burn::tensor::DType;

use crate::cli::FlagArgs;
use crate::util::device;

pub fn run(it: impl Iterator<Item = String>) -> Result<()> {
    let mut f = FlagArgs::new(it);
    let mut backend = None;
    let mut dtype = DType::F32;
    let mut quant_q8 = false;
    let (mut m, mut n, mut k) = (3632, 3072, 1024);
    let mut reps = 3;
    if let Some(v) = f.take_choice("--cpu-backend", &["flex", "cpu"])? {
        backend = Some(v);
    }
    if let Some(v) = f.take_choice("--dtype", &["auto", "f32", "f16", "bf16"])? {
        dtype = match v.as_str() {
            "f16" => DType::F16,
            "bf16" => DType::BF16,
            _ => DType::F32,
        };
    }
    if let Some(v) = f.take_choice("--quant", &["none", "q8"])? {
        quant_q8 = v == "q8";
    }
    if let Some(v) = f.take_parsed("--m")? {
        m = v;
    }
    if let Some(v) = f.take_parsed("--n")? {
        n = v;
    }
    if let Some(v) = f.take_parsed("--k")? {
        k = v;
    }
    let transb = f.take_bool("--transb");
    if let Some(v) = f.take_parsed("--reps")? {
        reps = v;
    }
    f.finish("gemm")?;
    if quant_q8 && dtype != DType::F32 {
        bail!("--quant q8 is only defined for f32 operands");
    }

    let backend = backend.unwrap_or_else(|| {
        if cfg!(feature = "cpu") {
            "cpu".to_string()
        } else {
            "flex".to_string()
        }
    });
    let device = device(&backend)?;
    let gflops = 2.0 * m as f64 * n as f64 * k as f64;
    let a = Tensor::<2>::random([m, k], burn::tensor::Distribution::Default, &device).cast(dtype);
    let b = if transb {
        Tensor::<2>::random([n, k], burn::tensor::Distribution::Default, &device)
    } else {
        Tensor::<2>::random([k, n], burn::tensor::Distribution::Default, &device)
    }
    .cast(dtype);
    let b = if quant_q8 {
        use burn::tensor::quantization::{QuantScheme, QuantValue, ScaleDtype};
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .per_block([32], ScaleDtype::F16);
        b.quantize_dynamic(&scheme)
    } else {
        b
    };

    let mut best = f64::MAX;
    for i in 0..=reps.max(1) {
        let t0 = Instant::now();
        let out = if transb {
            a.clone().matmul(b.clone().swap_dims(0, 1))
        } else {
            a.clone().matmul(b.clone())
        };
        let data = out.to_data();
        device.sync()?;
        let dt = t0.elapsed().as_secs_f64();
        let sum: f32 = data
            .try_to_vec::<f32>()
            .map(|v| v.iter().sum())
            .unwrap_or(f32::NAN);
        if i == 0 {
            println!(
                "warmup: {dt:.3}s out_shape={:?} sum={sum}",
                out.shape().dims::<2>()
            );
        } else {
            println!("run {i}: {dt:.3}s => {:.1} GFLOPS", gflops / dt / 1e9);
            best = best.min(dt);
        }
    }
    println!(
        "m={m} n={n} k={k} transb={transb} dtype={dtype:?} best={best:.3}s => {:.1} GFLOPS",
        gflops / best / 1e9
    );
    Ok(())
}
