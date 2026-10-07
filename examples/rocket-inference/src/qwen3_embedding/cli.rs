//! `qwen3` subcommands: `bench`, `embed`, `gemm`, `tokenize`
//! (Qwen3-Embedding-0.6B).

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use burn::prelude::*;
use burn::tensor::{DType, Int, TensorData};
use tokenizers::Tokenizer;

use crate::cli::FlagArgs;
use crate::qwen3_embedding::load::load_model;
use crate::qwen3_embedding::model::{stage_stats, stage_stats_reset};
use crate::util::device;
use crate::util::rope::RopeCache;

pub fn run(it: impl Iterator<Item = String>) -> Result<()> {
    let mut it = it;
    let cmd = it.next().unwrap_or_default();
    match cmd.as_str() {
        "bench" | "embed" | "gemm" | "tokenize" => {}
        other => bail!("unknown qwen3 command '{other}' (expected bench|embed|gemm|tokenize)"),
    }
    let args = Args::parse(&cmd, FlagArgs::new(it))?;
    match cmd.as_str() {
        "bench" => run_bench(&args),
        "embed" => run_embed(&args),
        "gemm" => run_gemm(&args),
        "tokenize" => run_tokenize(&args),
        _ => unreachable!(),
    }
}

struct Args {
    model_dir: PathBuf,
    dtype: DType,
    backend: String,
    chunk: usize,
    reps: usize,
    tokens: Option<usize>,
    text: Option<String>,
    text_file: Option<PathBuf>,
    out: Option<PathBuf>,
    m: usize,
    n: usize,
    k: usize,
    transb: bool,
    quant_q8: bool,
    npu: bool,
    npu_threads: usize,
    npu_attn: bool,
    attn_fused: bool,
    key_block: usize,
}

impl Args {
    fn parse(cmd: &str, mut f: FlagArgs) -> Result<Self> {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        let mut args = Args {
            model_dir: PathBuf::from(format!("{home}/models/qwen3-embedding-0.6b")),
            dtype: DType::F32,
            backend: if cfg!(feature = "cpu") { "cpu" } else { "flex" }.to_string(),
            chunk: 256,
            reps: 3,
            tokens: None,
            text: None,
            text_file: None,
            out: None,
            m: 3632,
            n: 3072,
            k: 1024,
            transb: false,
            quant_q8: false,
            npu: false,
            npu_threads: 5,
            npu_attn: true,
            attn_fused: true,
            key_block: 256,
        };
        if let Some(v) = f.take("--model-dir")? {
            args.model_dir = PathBuf::from(v);
        }
        if let Some(v) = f.take("--backend")? {
            args.backend = v;
        }
        if let Some(v) = f.take("--dtype")? {
            args.dtype = match v.as_str() {
                "f32" => DType::F32,
                "f16" => DType::F16,
                "bf16" => DType::BF16,
                other => bail!("unsupported dtype '{other}'"),
            };
        }
        if let Some(v) = f.take_parsed("--chunk")? {
            args.chunk = v;
        }
        if let Some(v) = f.take_parsed("--reps")? {
            args.reps = v;
        }
        if let Some(v) = f.take_parsed("--tokens")? {
            args.tokens = Some(v);
        }
        if let Some(v) = f.take("--text")? {
            args.text = Some(v);
        }
        if let Some(v) = f.take("--text-file")? {
            args.text_file = Some(PathBuf::from(v));
        }
        if let Some(v) = f.take("--out")? {
            args.out = Some(PathBuf::from(v));
        }
        if let Some(v) = f.take_parsed("--m")? {
            args.m = v;
        }
        if let Some(v) = f.take_parsed("--n")? {
            args.n = v;
        }
        if let Some(v) = f.take_parsed("--k")? {
            args.k = v;
        }
        args.transb = f.take_bool("--transb");
        if let Some(v) = f.take_choice("--quant", &["none", "q8"])? {
            args.quant_q8 = v == "q8";
        }
        args.npu = f.take_bool("--npu");
        if let Some(v) = f.take_parsed("--npu-threads")? {
            args.npu_threads = v;
        }
        if let Some(v) = f.take_choice("--npu-attn", &["npu", "cpu"])? {
            args.npu_attn = v == "npu";
        }
        if let Some(v) = f.take_choice("--attn", &["fused", "blocked"])? {
            args.attn_fused = v == "fused";
        }
        if let Some(v) = f.take_parsed("--key-block")? {
            args.key_block = v;
        }
        f.finish(&format!("qwen3 {cmd}"))?;
        Ok(args)
    }

    fn input_text(&self) -> Result<String> {
        if let Some(t) = &self.text {
            return Ok(t.clone());
        }
        let path = self
            .text_file
            .clone()
            .unwrap_or_else(|| PathBuf::from("data/bench_text.txt"));
        std::fs::read_to_string(&path).with_context(|| format!("read text from {}", path.display()))
    }
}

fn tokenize(args: &Args) -> Result<Vec<u32>> {
    let tokenizer = Tokenizer::from_file(args.model_dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    let text = args.input_text()?;
    let enc = tokenizer
        .encode(text.as_str(), true)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
    let mut ids = enc.get_ids().to_vec();
    if let Some(n) = args.tokens {
        ids.truncate(n);
    }
    Ok(ids)
}

fn run_bench(args: &Args) -> Result<()> {
    let device = device(&args.backend)?;
    let (model, cfg) = load_model(
        &args.model_dir,
        args.dtype,
        args.quant_q8,
        args.npu,
        args.npu_threads,
        args.npu_attn,
        &device,
    )?;
    let ids = tokenize(args)?;
    let n = ids.len();
    if n == 0 {
        bail!("empty input");
    }
    let ids_i64: Vec<i64> = ids.iter().map(|&t| t as i64).collect();
    let input = Tensor::<2, Int>::from_data(TensorData::new(ids_i64, [1, n]), &device);
    let rope = RopeCache::new(n, cfg.head_dim, cfg.rope_theta, args.dtype, &device);

    stage_stats_reset();
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    if args.npu {
        burn_rocket::stats_reset();
    }
    let mut times = Vec::new();
    for i in 0..=args.reps {
        let t0 = Instant::now();
        let out = model.forward(
            input.clone(),
            &rope,
            args.chunk,
            args.key_block,
            args.attn_fused,
        );
        let _ = out.to_data();
        device.sync()?;
        let dt = t0.elapsed().as_secs_f64();
        times.push(dt);
        if i == 0 {
            println!("warmup: {dt:.3}s ({:.1} tok/s)", n as f64 / dt);
        } else {
            println!("run {i}: {dt:.3}s ({:.1} tok/s)", n as f64 / dt);
        }
    }
    let best = times.iter().cloned().fold(f64::MAX, f64::min);
    let attn = if args.attn_fused { "fused" } else { "blocked" };
    println!(
        "tokens={n} attn={attn} chunk={} key_block={} dtype={:?} best={best:.3}s => {:.1} tok/s",
        args.chunk,
        args.key_block,
        args.dtype,
        n as f64 / best
    );
    {
        let (attn, mlp, norms) = stage_stats();
        println!("stages over all runs: attention {attn:.2}s, mlp {mlp:.2}s, norms {norms:.2}s");
    }
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    if args.npu {
        let s = burn_rocket::stats();
        println!(
            "npu breakdown over all runs: {} calls, convert {:.2}s, npu {:.2}s, \
             flex+overhead {:.2}s",
            s.calls,
            s.convert_s,
            s.npu_s,
            times.iter().sum::<f64>() - s.convert_s - s.npu_s
        );
    }
    Ok(())
}

fn run_gemm(args: &Args) -> Result<()> {
    let device = device(&args.backend)?;
    let (m, n, k) = (args.m, args.n, args.k);
    let gflops = 2.0 * m as f64 * n as f64 * k as f64;
    let a =
        Tensor::<2>::random([m, k], burn::tensor::Distribution::Default, &device).cast(args.dtype);
    let b = if args.transb {
        Tensor::<2>::random([n, k], burn::tensor::Distribution::Default, &device)
    } else {
        Tensor::<2>::random([k, n], burn::tensor::Distribution::Default, &device)
    }
    .cast(args.dtype);
    let b = if args.quant_q8 {
        use burn::tensor::quantization::{QuantScheme, QuantValue, ScaleDtype};
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .per_block([32], ScaleDtype::F16);
        b.quantize_dynamic(&scheme)
    } else {
        b
    };

    let mut best = f64::MAX;
    for i in 0..=args.reps {
        let t0 = Instant::now();
        let out = if args.transb {
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
        "m={m} n={n} k={k} transb={} dtype={:?} best={best:.3}s => {:.1} GFLOPS",
        args.transb,
        args.dtype,
        gflops / best / 1e9
    );
    Ok(())
}

fn run_tokenize(args: &Args) -> Result<()> {
    let ids = tokenize(args)?;
    println!("{} tokens", ids.len());
    println!(
        "{}",
        ids.iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    );
    Ok(())
}

fn run_embed(args: &Args) -> Result<()> {
    let device = device(&args.backend)?;
    let (model, cfg) = load_model(
        &args.model_dir,
        args.dtype,
        args.quant_q8,
        args.npu,
        args.npu_threads,
        args.npu_attn,
        &device,
    )?;
    let ids = tokenize(args)?;
    let n = ids.len();
    let ids_i64: Vec<i64> = ids.iter().map(|&t| t as i64).collect();
    let input = Tensor::<2, Int>::from_data(TensorData::new(ids_i64, [1, n]), &device);
    let rope = RopeCache::new(n, cfg.head_dim, cfg.rope_theta, args.dtype, &device);

    let t0 = Instant::now();
    let out = model.forward(input, &rope, args.chunk, args.key_block, args.attn_fused);
    let vec: Vec<f32> = out
        .cast(DType::F32)
        .to_data()
        .try_to_vec()
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let dt = t0.elapsed().as_secs_f64();
    println!("tokens={n} time={dt:.3}s ({:.1} tok/s)", n as f64 / dt);

    let json = serde_json::to_string(&vec)?;
    match &args.out {
        Some(path) => {
            std::fs::write(path, &json)?;
            println!("wrote embedding to {}", path.display());
        }
        None => println!("{json}"),
    }
    Ok(())
}
