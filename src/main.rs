mod model;
mod npu;
mod server;

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use burn::prelude::*;
use burn::tensor::{DType, Int, TensorData};
use burn_store::{FloatCastAdapter, ModuleAdapter, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};
use tokenizers::Tokenizer;

use model::{Qwen3Config, Qwen3Embedding, RopeCache};
use server::ServeOptions;

fn main() -> Result<()> {
    let args = Args::parse()?;
    match args.cmd.as_str() {
        "bench" => run_bench(&args),
        "embed" => run_embed(&args),
        "gemm" => run_gemm(&args),
        "serve" => run_serve(&args),
        "tokenize" => run_tokenize(&args),
        other => bail!("unknown command '{other}' (expected bench|embed|gemm|serve|tokenize)"),
    }
}

struct Args {
    cmd: String,
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
    attn_fused: bool,
    key_block: usize,
    port: u16,
    max_tokens: usize,
    model_name: String,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut it = std::env::args().skip(1);
        let cmd = it.next().unwrap_or_else(|| "bench".to_string());
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        let mut args = Args {
            cmd,
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
            attn_fused: true,
            key_block: 256,
            port: 8383,
            max_tokens: 30000,
            model_name: "qwen3-embedding-0.6b".to_string(),
        };
        while let Some(flag) = it.next() {
            let mut value = || it.next().with_context(|| format!("missing value for {flag}"));
            match flag.as_str() {
                "--model-dir" => args.model_dir = PathBuf::from(value()?),
                "--backend" => args.backend = value()?,
                "--dtype" => {
                    args.dtype = match value()?.as_str() {
                        "f32" => DType::F32,
                        "f16" => DType::F16,
                        "bf16" => DType::BF16,
                        other => bail!("unsupported dtype '{other}'"),
                    }
                }
                "--chunk" => args.chunk = value()?.parse()?,
                "--reps" => args.reps = value()?.parse()?,
                "--tokens" => args.tokens = Some(value()?.parse()?),
                "--text" => args.text = Some(value()?),
                "--text-file" => args.text_file = Some(PathBuf::from(value()?)),
                "--out" => args.out = Some(PathBuf::from(value()?)),
                "--m" => args.m = value()?.parse()?,
                "--n" => args.n = value()?.parse()?,
                "--k" => args.k = value()?.parse()?,
                "--transb" => args.transb = true,
                "--quant" => match value()?.as_str() {
                    "none" => args.quant_q8 = false,
                    "q8" => args.quant_q8 = true,
                    other => bail!("unsupported quant '{other}' (expected none|q8)"),
                },
                "--npu" => args.npu = true,
                "--npu-threads" => args.npu_threads = value()?.parse()?,
                "--port" => args.port = value()?.parse()?,
                "--key-block" => args.key_block = value()?.parse()?,
                "--attn" => {
                    args.attn_fused = match value()?.as_str() {
                        "fused" => true,
                        "blocked" => false,
                        other => bail!("--attn must be fused|blocked, got {other}"),
                    }
                }
                "--max-tokens" => args.max_tokens = value()?.parse()?,
                "--model-name" => args.model_name = value()?,
                other => bail!("unknown flag '{other}'"),
            }
        }
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

fn device(args: &Args) -> Result<Device> {
    match args.backend.as_str() {
        #[cfg(feature = "cpu")]
        "cpu" => Ok(Device::cpu()),
        "flex" => Ok(Device::flex()),
        #[cfg(not(feature = "cpu"))]
        "cpu" => bail!("this build has no 'cpu' backend (built without the 'cpu' feature); use flex"),
        other => bail!("unknown backend '{other}' (expected cpu|flex)"),
    }
}

fn rss_mib() -> f64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in s.lines() {
        if let Some(v) = line.strip_prefix("RssAnon:") {
            return v.trim().trim_end_matches(" kB").parse::<f64>().unwrap_or(0.0) / 1024.0;
        }
    }
    0.0
}

fn load_model(
    model_dir: &Path,
    dtype: DType,
    quant_q8: bool,
    npu: bool,
    npu_threads: usize,
    device: &Device,
) -> Result<(Qwen3Embedding, Qwen3Config)> {
    let cfg = Qwen3Config::from_file(&model_dir.join("config.json"))?;
    let _ = npu_threads; // only used by the aarch64+npu build
    if dtype == DType::BF16 {
        bail!("--dtype bf16 is broken in burn-flex 0.22.0-pre.4 (bf16 embedding gather panics); use f32 (or f16 for a smaller model)");
    }
    let mut model = Qwen3Embedding::new(&cfg, device);
    let t0 = Instant::now();
    let mut store = SafetensorsStore::from_file(model_dir.join("model.safetensors"))
        .with_from_adapter(PyTorchToBurnAdapter.chain(FloatCastAdapter::to(dtype)));
    let result = model
        .load_from(&mut store)
        .with_context(|| format!("load {}", model_dir.display()))?;
    if !result.errors.is_empty() {
        bail!("load errors: {:?}", result.errors);
    }
    if !result.missing.is_empty() {
        bail!("{} model parameters missing from file", result.missing.len());
    }
    if !result.unused.is_empty() {
        println!(
            "warning: {} file tensors unused by the model (first: {:?})",
            result.unused.len(),
            result.unused.first()
        );
    }
    println!(
        "loaded {} tensors as {:?} in {:.2}s",
        result.applied.len(),
        dtype,
        t0.elapsed().as_secs_f64()
    );
    drop(store);
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    if npu {
        if dtype != DType::F32 {
            bail!("--npu needs --dtype f32 (NPU activations are f32 on the CPU side)");
        }
        if quant_q8 {
            bail!("--npu packs its own fp16 weights; --quant q8 is not applicable");
        }
        return load_npu_projections(model, cfg, model_dir, npu_threads, device, t0);
    }
    #[cfg(not(all(feature = "npu", target_arch = "aarch64")))]
    if npu {
        bail!("--npu requires an aarch64 build with --features npu");
    }
    if quant_q8 {
        if dtype != DType::F32 {
            bail!("--quant q8 needs --dtype f32 (Q8-resident weights are dequantized to f32 on the fly)");
        }
        use burn::module::{ModuleMapper, Param, ParamGroup};
        use burn::tensor::quantization::{
            Calibration, QuantScheme, QuantValue, ScaleDtype, compute_q_params, compute_range,
        };
        let t0 = Instant::now();
        // Low-RAM mode. Projection weights stay Q8_0-quantized (symmetric int8,
        // 32-value blocks, f16 block scales = llama.cpp's Q8_0); the forward dequantizes
        // each weight on the fly, so only the current layer's f32 weights are
        // materialized (~62 MB) instead of all 1.75 GB. The token-embedding table is
        // kept in f16 (exact for bf16-sourced values in range) and gathered rows are
        // cast back to f32 per forward. flex's bf16 gather is broken (dtype panic).
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .per_block([32], ScaleDtype::F16);
        let proj_group = ParamGroup::from_regex(r"\.(q|k|v|o|gate|up|down)_proj\.weight$")
            .map_err(|e| anyhow::anyhow!("bad param group regex: {e:?}"))?;
        let embed_group = ParamGroup::from_regex(r"embed_tokens\.weight$")
            .map_err(|e| anyhow::anyhow!("bad param group regex: {e:?}"))?;

        struct LowRam {
            scheme: QuantScheme,
            proj_group: ParamGroup,
            embed_group: ParamGroup,
            path: Vec<String>,
        }
        impl ModuleMapper for LowRam {
            fn enter_module(&mut self, name: &str, _container_type: &str) {
                self.path.push(name.to_string());
            }
            fn exit_module(&mut self, _name: &str, _container_type: &str) {
                self.path.pop();
            }
            fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
                let path = self.path.join(".");
                if self.proj_group.matches(&param.id, Some(&path)) {
                    param.map(|tensor| {
                        let range = compute_range(&self.scheme, &tensor, &Calibration::MinMax);
                        let qparams = compute_q_params(&self.scheme, range);
                        tensor.quantize(&self.scheme, qparams)
                    })
                } else if self.embed_group.matches(&param.id, Some(&path)) {
                    param.map(|tensor| tensor.cast(DType::F16))
                } else {
                    param
                }
            }
        }
        let mut mapper = LowRam {
            scheme,
            proj_group,
            embed_group,
            path: Vec::new(),
        };
        model = model.map(&mut mapper);
        model.set_quantized(true);
        // Return freed f32 weight pages to the OS: glibc keeps them in its arenas,
        // which would hide the memory saving (measured: ~1.3 GB retained).
        #[cfg(all(target_os = "linux", target_env = "gnu"))]
        unsafe {
            libc::malloc_trim(0);
        }
        println!(
            "Q8-resident projections + f16 embedding table in {:.2}s (resident {:.0} MiB anon)",
            t0.elapsed().as_secs_f64(),
            rss_mib()
        );
    }
    Ok((model, cfg))
}

/// Pack every projection weight straight into resident NPU buffers (fp16, HF
/// `[out, in]` = `[N, K]` layout, no transpose) and keep only an f16 embedding
/// table on the CPU. Pack-and-drop: the CPU never holds projection weights.
#[cfg(all(feature = "npu", target_arch = "aarch64"))]
fn load_npu_projections(
    mut model: Qwen3Embedding,
    cfg: Qwen3Config,
    model_dir: &Path,
    npu_threads: usize,
    _device: &Device,
    t0: Instant,
) -> Result<(Qwen3Embedding, Qwen3Config)> {
    use crate::model::Proj;
    use crate::npu::{NpuModel, NpuRef, ProjKind};
    use burn_store::ModuleStore;
    use std::sync::Arc;

    let mut store = SafetensorsStore::from_file(model_dir.join("model.safetensors"));
    let t_npu = Instant::now();
    let mut npu = Arc::new(NpuModel::new(cfg.num_hidden_layers, npu_threads)?);
    let mut bytes = 0usize;
    let mut packed = Vec::new();
    for layer in 0..cfg.num_hidden_layers {
        for kind in ProjKind::ALL {
            let key = format!("layers.{layer}.{}", kind.key());
            let tensor = store
                .get_tensor(&key)?
                .ok_or_else(|| anyhow::anyhow!("missing tensor {key}"))?;
            let data = burn_store::bridge::to_data(tensor)?; // [out, in] = [N, K]
            let [n, k] = data.shape.dims::<2>();
            let values: Vec<f32> = data.convert_dtype(DType::F32).try_to_vec()?;
            let b = burn_rocket::f32_to_f16(&values);
            bytes += values.len() * 2;
            Arc::get_mut(&mut npu)
                .expect("sole owner while packing")
                .pack(layer, kind, k, n, &b)?;
            packed.push((layer, kind));
        }
    }
    for (layer, kind) in packed {
        *model.projection_mut(layer, kind) =
            Proj::Npu(NpuRef::new(Arc::clone(&npu), layer, kind));
    }
    model.embed_table_to_f16();
    println!(
        "NPU: packed {} projections ({:.2} GiB fp16, {} threads) in {:.2}s; \
         model ready in {:.2}s (resident {:.0} MiB anon)",
        cfg.num_hidden_layers * ProjKind::ALL.len(),
        bytes as f64 / (1u64 << 30) as f64,
        npu_threads,
        t_npu.elapsed().as_secs_f64(),
        t0.elapsed().as_secs_f64(),
        rss_mib()
    );
    Ok((model, cfg))
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
    let device = device(args)?;
    let (model, cfg) = load_model(
        &args.model_dir,
        args.dtype,
        args.quant_q8,
        args.npu,
        args.npu_threads,
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

    let mut times = Vec::new();
    for i in 0..=args.reps {
        let t0 = Instant::now();
        let out = model.forward(input.clone(), &rope, args.chunk, args.key_block, args.attn_fused);
        let _ = out.to_data();
        device.sync()?;
        let dt = t0.elapsed().as_secs_f64();
        times.push(dt);
        if i == 0 {
            println!("warmup: {dt:.3}s ({:.1} tok/s)", n as f64 / dt);
        } else {
            println!(
                "run {i}: {dt:.3}s ({:.1} tok/s)",
                n as f64 / dt
            );
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
    Ok(())
}

fn run_gemm(args: &Args) -> Result<()> {
    let device = device(args)?;
    let (m, n, k) = (args.m, args.n, args.k);
    let gflops = 2.0 * m as f64 * n as f64 * k as f64;
    let a = Tensor::<2>::random([m, k], burn::tensor::Distribution::Default, &device).cast(args.dtype);
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
        let sum: f32 = data.try_to_vec::<f32>().map(|v| v.iter().sum()).unwrap_or(f32::NAN);
        if i == 0 {
            println!("warmup: {dt:.3}s out_shape={:?} sum={sum}", out.shape().dims::<2>());
        } else {            println!("run {i}: {dt:.3}s => {:.1} GFLOPS", gflops / dt / 1e9);
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
        ids.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(" ")
    );
    Ok(())
}

fn run_serve(args: &Args) -> Result<()> {
    let device = device(args)?;
    let (model, cfg) = load_model(
        &args.model_dir,
        args.dtype,
        args.quant_q8,
        args.npu,
        args.npu_threads,
        &device,
    )?;
    let tokenizer = Tokenizer::from_file(args.model_dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    server::serve(
        model,
        cfg,
        tokenizer,
        device,
        ServeOptions {
            addr: format!("0.0.0.0:{}", args.port),
            max_tokens: args.max_tokens,
            model_name: args.model_name.clone(),
            chunk: args.chunk,
            key_block: args.key_block,
            attn_fused: args.attn_fused,
        },
    )
}

fn run_embed(args: &Args) -> Result<()> {
    let device = device(args)?;
    let (model, cfg) = load_model(
        &args.model_dir,
        args.dtype,
        args.quant_q8,
        args.npu,
        args.npu_threads,
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
