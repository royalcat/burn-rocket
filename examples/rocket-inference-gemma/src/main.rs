//! rocket-inference-gemma: EmbeddingGemma 2 inference example (Burn + flex).
//!
//! P1 covers the text/code path (`embed`, `bench`, `tokenize`); the vision,
//! audio and video paths are added by later modules.

mod audio;
mod audio_frontend;
mod chat;
mod config;
mod gen_loader;
mod gen_model;
mod gen_server;
mod inputs;
mod layers;
mod media;
mod model;
mod server;
mod vision;

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use burn::prelude::*;
use burn::tensor::{DType, Int};
use burn_store::{
    FloatCastAdapter, ModuleAdapter, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore,
};
use tokenizers::Tokenizer;

use config::Emb2Config;
use model::{Emb2Model, stage_stats, stage_stats_reset};

fn main() -> Result<()> {
    let args = Args::parse()?;
    match args.cmd.as_str() {
        "embed" => run_embed(&args),
        "bench" => run_bench(&args),
        "tokenize" => run_tokenize(&args),
        "serve" => run_serve(&args),
        "gen" => run_gen(&args),
        "serve-chat" => run_serve_chat(&args),
        other => bail!(
            "unknown command '{other}' (expected embed|bench|tokenize|serve|gen|serve-chat)"
        ),
    }
}

struct Args {
    cmd: String,
    model_dir: PathBuf,
    dtype: DType,
    backend: String,
    text: Option<String>,
    text_file: Option<PathBuf>,
    image: Option<PathBuf>,
    video: Option<PathBuf>,
    audio: Option<PathBuf>,
    max_soft_tokens: Option<usize>,
    video_soft_tokens: Option<usize>,
    video_fps: f64,
    video_max_frames: usize,
    prompt: Option<String>,
    dim: Option<usize>,
    normalize: bool,
    tokens: Option<usize>,
    reps: usize,
    out: Option<PathBuf>,
    attn_chunk: usize,
    dump_pixels: Option<PathBuf>,
    dump_ids: Option<PathBuf>,
    dump_audio: Option<PathBuf>,
    dump_audio_layers: Option<PathBuf>,
    dump_image: Option<PathBuf>,
    dump_vision: Option<PathBuf>,
    dump_vision_layers: Option<PathBuf>,
    port: u16,
    model_name: String,
    max_tokens: usize,
    quant_q8: bool,
    npu: bool,
    npu_threads: usize,
    npu_attn: bool,
    gen_model_dir: PathBuf,
    messages: Option<String>,
    max_new_tokens: usize,
    greedy: bool,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    enable_thinking: bool,
    gen_f16: bool,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut it = std::env::args().skip(1);
        let cmd = it.next().unwrap_or_else(|| "embed".to_string());
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        let mut args = Args {
            cmd,
            model_dir: PathBuf::from(format!("{home}/models/embeddinggemma-2")),
            dtype: DType::F32,
            backend: if cfg!(feature = "cpu") { "cpu" } else { "flex" }.to_string(),
            text: None,
            text_file: None,
            image: None,
            video: None,
            audio: None,
            max_soft_tokens: None,
            video_soft_tokens: None,
            video_fps: 1.0,
            video_max_frames: 32,
            prompt: None,
            dim: None,
            normalize: true,
            tokens: None,
            reps: 3,
            out: None,
            attn_chunk: 1024,
            dump_pixels: None,
            dump_ids: None,
            dump_audio: None,
            dump_audio_layers: None,
            dump_image: None,
            dump_vision: None,
            dump_vision_layers: None,
            port: 8390,
            model_name: "embeddinggemma-2".to_string(),
            max_tokens: 8192,
            quant_q8: false,
            npu: false,
            npu_threads: 5,
            npu_attn: true,
            gen_model_dir: PathBuf::from("/mnt/hub/models/gemma-4-E2B-it"),
            messages: None,
            max_new_tokens: 64,
            greedy: true,
            temperature: 1.0,
            top_k: 64,
            top_p: 0.95,
            enable_thinking: false,
            gen_f16: false,
        };
        while let Some(flag) = it.next() {
            let mut value = || {
                it.next()
                    .with_context(|| format!("missing value for {flag}"))
            };
            match flag.as_str() {
                "--model-dir" => args.model_dir = PathBuf::from(value()?),
                "--backend" => args.backend = value()?,
                "--dtype" => {
                    args.dtype = match value()?.as_str() {
                        "f32" => DType::F32,
                        "f16" => DType::F16,
                        other => bail!("unsupported dtype '{other}' (expected f32|f16)"),
                    }
                }
                "--text" => args.text = Some(value()?),
                "--text-file" => args.text_file = Some(PathBuf::from(value()?)),
                "--image" => args.image = Some(PathBuf::from(value()?)),
                "--video" => args.video = Some(PathBuf::from(value()?)),
                "--video-fps" => args.video_fps = value()?.parse()?,
                "--video-max-frames" => args.video_max_frames = value()?.parse()?,
                "--audio" => args.audio = Some(PathBuf::from(value()?)),
                "--max-soft-tokens" => args.max_soft_tokens = Some(value()?.parse()?),
                "--video-soft-tokens" => args.video_soft_tokens = Some(value()?.parse()?),
                "--prompt" => args.prompt = Some(value()?),
                "--dim" => args.dim = Some(value()?.parse()?),
                "--no-normalize" => args.normalize = false,
                "--tokens" => args.tokens = Some(value()?.parse()?),
                "--reps" => args.reps = value()?.parse()?,
                "--out" => args.out = Some(PathBuf::from(value()?)),
                "--attn-chunk" => args.attn_chunk = value()?.parse()?,
                "--dump-pixels" => args.dump_pixels = Some(PathBuf::from(value()?)),
                "--dump-ids" => args.dump_ids = Some(PathBuf::from(value()?)),
                "--port" => args.port = value()?.parse()?,
                "--model-name" => args.model_name = value()?,
                "--max-tokens" => args.max_tokens = value()?.parse()?,
                "--gen-model-dir" => args.gen_model_dir = PathBuf::from(value()?),
                "--messages" => args.messages = Some(value()?),
                "--max-new-tokens" => args.max_new_tokens = value()?.parse()?,
                "--sample" => args.greedy = false,
                "--temperature" => args.temperature = value()?.parse()?,
                "--top-k" => args.top_k = value()?.parse()?,
                "--top-p" => args.top_p = value()?.parse()?,
                "--enable-thinking" => args.enable_thinking = true,
                "--f16" => args.gen_f16 = true,
                "--npu" => args.npu = true,
                "--npu-threads" => args.npu_threads = value()?.parse()?,
                "--npu-attn" => {
                    args.npu_attn = match value()?.as_str() {
                        "npu" => true,
                        "cpu" => false,
                        other => bail!("--npu-attn must be npu|cpu, got {other}"),
                    }
                }
                "--quant" => match value()?.as_str() {
                    "none" => args.quant_q8 = false,
                    "q8" => args.quant_q8 = true,
                    other => bail!("unsupported quant '{other}' (expected none|q8)"),
                },
                "--dump-audio" => args.dump_audio = Some(PathBuf::from(value()?)),
                "--dump-audio-layers" => args.dump_audio_layers = Some(PathBuf::from(value()?)),
                "--dump-image" => args.dump_image = Some(PathBuf::from(value()?)),
                "--dump-vision" => args.dump_vision = Some(PathBuf::from(value()?)),
                "--dump-vision-layers" => args.dump_vision_layers = Some(PathBuf::from(value()?)),
                other => bail!("unknown flag '{other}'"),
            }
        }
        Ok(args)
    }

    fn input_text(&self) -> Result<String> {
        if let Some(t) = &self.text {
            return Ok(t.clone());
        }
        if let Some(path) = &self.text_file {
            return std::fs::read_to_string(path)
                .with_context(|| format!("read text from {}", path.display()));
        }
        bail!("provide --text or --text-file")
    }

    fn tokenizer(&self) -> Result<Tokenizer> {
        Tokenizer::from_file(self.model_dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))
    }

    /// Resolve the task prompt prefix from `config_sentence_transformers.json`
    /// (case-insensitive key match; `none`/absent means no prefix).
    fn prompt_prefix(&self) -> Result<String> {
        let Some(name) = &self.prompt else {
            return Ok(String::new());
        };
        if name.eq_ignore_ascii_case("none") || name.eq_ignore_ascii_case("raw") {
            return Ok(String::new());
        }
        let path = self.model_dir.join("config_sentence_transformers.json");
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("read {}", path.display()))?;
        let json: serde_json::Value = serde_json::from_str(&text)?;
        let prompts = json
            .get("prompts")
            .and_then(|p| p.as_object())
            .context("config_sentence_transformers.json has no 'prompts' object")?;
        for (key, v) in prompts {
            if key.eq_ignore_ascii_case(name) {
                return Ok(v.as_str().unwrap_or_default().to_string());
            }
        }
        let mut keys: Vec<&String> = prompts.keys().collect();
        keys.sort();
        bail!("unknown prompt '{name}' (available: {keys:?})")
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

/// Anonymous resident set size (`RssAnon`), MiB.
fn rss_mib() -> f64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in s.lines() {
        if let Some(v) = line.strip_prefix("RssAnon:") {
            return v.trim().trim_end_matches(" kB").parse::<f64>().unwrap_or(0.0) / 1024.0;
        }
    }
    0.0
}

#[allow(clippy::too_many_arguments)]
fn load_model(
    model_dir: &Path,
    dtype: DType,
    quant_q8: bool,
    npu: bool,
    npu_threads: usize,
    npu_attn: bool,
    device: &Device,
) -> Result<(Emb2Model, Emb2Config)> {
    let cfg = Emb2Config::from_file(&model_dir.join("config.json"))?;
    if dtype != DType::F32 {
        bail!(
            "--dtype f16 is not numerically supported (RMSNorm/softmax/PLE precision): \
             cosine drops to 0.98 (text) / 0.70 (image) vs the f32 reference; use f32"
        );
    }
    let vision_cfg = cfg
        .vision_config
        .as_ref()
        .context("checkpoint has no vision_config (text-only checkpoints are not supported yet)")?;
    let audio_cfg = cfg
        .audio_config
        .as_ref()
        .context("checkpoint has no audio_config (text-only checkpoints are not supported yet)")?;
    let t0 = Instant::now();
    let audio_bounds = read_audio_clip_bounds(model_dir, audio_cfg)?;
    let mut model = Emb2Model::new(
        &cfg.text_config,
        vision_cfg,
        audio_cfg,
        &audio_bounds,
        device,
    );
    let mut store = SafetensorsStore::from_file(model_dir.join("model.safetensors"))
        .allow_partial(true)
        .with_from_adapter(PyTorchToBurnAdapter.chain(FloatCastAdapter::to(dtype)));
    let result = model
        .load_from(&mut store)
        .with_context(|| format!("load {}", model_dir.display()))?;
    if !result.errors.is_empty() {
        bail!("load errors: {:?}", result.errors);
    }
    if !result.missing.is_empty() {
        bail!(
            "{} model parameters missing from file (first: {:?})",
            result.missing.len(),
            result.missing.first()
        );
    }
    if !result.unused.is_empty() {
        let sample: Vec<&String> = result.unused.iter().take(8).collect();
        println!(
            "note: {} file tensors unused by the model tree (e.g. {sample:?})",
            result.unused.len()
        );
    }
    println!(
        "loaded {} tensors as {:?} in {:.2}s (resident {:.0} MiB anon)",
        result.applied.len(),
        dtype,
        t0.elapsed().as_secs_f64(),
        rss_mib()
    );
    drop(store);
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    if npu {
        if dtype != DType::F32 {
            bail!("--npu needs --dtype f32 (NPU activations are f32 on the CPU side)");
        }
        // With both flags the projections are quantized first and the text ones
        // are then packed from their dequantized values (lowest-memory mode).
        if quant_q8 {
            model = quantize_low_ram(model)?;
        }
        return load_npu_text(model, cfg, npu_threads, npu_attn, device);
    }
    #[cfg(not(all(feature = "npu", target_arch = "aarch64")))]
    if npu {
        bail!("--npu requires an aarch64 build with --features npu");
    }
    if quant_q8 {
        model = quantize_low_ram(model)?;
    }
    Ok((model, cfg))
}

/// Pack the text backbone's projections into resident fp16 NPU buffers
/// (HF `[N, K]` layout, one pack per weight) and drop the f32 copies.
#[cfg(all(feature = "npu", target_arch = "aarch64"))]
fn load_npu_text(
    mut model: Emb2Model,
    cfg: Emb2Config,
    threads: usize,
    npu_attn: bool,
    device: &Device,
) -> Result<(Emb2Model, Emb2Config)> {

    burn_rocket::init(threads)
        .map_err(|e| anyhow::anyhow!("NPU context creation failed: {e}"))?;
    let t0 = Instant::now();
    let mut count = 0usize;
    let mut bytes = 0usize;
    model.text_mut().for_each_projection_mut(|lin| {
        let w = lin.weight.val();
        let w = if layers::is_quantized() {
            w.dequantize()
        } else {
            w
        };
        let [k, n] = w.shape().dims::<2>();
        let values: Vec<f32> = w
            .cast(DType::F32)
            .into_data()
            .try_to_vec()
            .expect("f32 projection weights");
        // Burn `Linear` is `[in, out]`; the NPU packs HF `[out, in]` = `[N, K]`.
        let mut t = vec![0f32; n * k];
        for i in 0..k {
            let src = &values[i * n..(i + 1) * n];
            for j in 0..n {
                t[j * k + i] = src[j];
            }
        }
        let tensor = Tensor::<2>::from_data(TensorData::new(t, [n, k]), device);
        let id = burn_rocket::pack(tensor);
        layers::register_npu_weight(&lin.weight, id);
        // Shrink the (now redundant) f32 copy in place: `Param::map` keeps the
        // parameter id, so the registry lookup in `lin` still hits.
        let dev = device.clone();
        lin.weight = lin.weight.clone().map(|_| Tensor::zeros([1, 1], &dev));
        count += 1;
        bytes += n * k * 2;
    });
    if npu_attn {
        model.text_mut().set_npu_attn(true);
    }
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    unsafe {
        libc::malloc_trim(0);
    }
    println!(
        "NPU: packed {count} text projections into resident fp16 weights ({:.2} GiB, {threads} threads) \
         in {:.2}s; attention on the {} (resident {:.0} MiB anon)",
        bytes as f64 / (1u64 << 30) as f64,
        t0.elapsed().as_secs_f64(),
        if npu_attn { "NPU" } else { "CPU" },
        rss_mib()
    );
    Ok((model, cfg))
}

/// Low-RAM mode (`--quant q8`): projection weights stay Q8_0-quantized
/// (symmetric int8, 32-value blocks, f16 block scales = llama.cpp's Q8_0) and
/// are dequantized per call, so only the current layer's f32 weights are
/// materialized. The embedding table, norms, scalars and position tables stay
/// f32.
fn quantize_low_ram(model: Emb2Model) -> Result<Emb2Model> {
    use burn::module::{ModuleMapper, Param, ParamGroup};
    use burn::tensor::quantization::{
        Calibration, QuantScheme, QuantValue, ScaleDtype, compute_q_params, compute_range,
    };

    let t0 = Instant::now();
    let scheme = QuantScheme::default()
        .with_value(QuantValue::Q8S)
        .per_block([32], ScaleDtype::F16);
    let group = ParamGroup::from_regex(
        r"(q_proj|k_proj|v_proj|o_proj|post|relative_k_proj|gate_proj|up_proj|down_proj|per_layer_model_projection|per_layer_input_gate|per_layer_projection|embedding_projection|input_proj|input_proj_linear|ffw_layer_1|ffw_layer_2|linear_start|linear_end|output_proj)\.(linear\.)?weight$",
    )
    .map_err(|e| anyhow::anyhow!("bad param group regex: {e:?}"))?;

    struct LowRam {
        scheme: QuantScheme,
        group: ParamGroup,
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
            if self.group.matches(&param.id, Some(&path)) {
                param.map(|tensor| {
                    let range = compute_range(&self.scheme, &tensor, &Calibration::MinMax);
                    let qparams = compute_q_params(&self.scheme, range);
                    tensor.quantize(&self.scheme, qparams)
                })
            } else {
                param
            }
        }
    }
    let mut mapper = LowRam {
        scheme,
        group,
        path: Vec::new(),
    };
    let model = model.map(&mut mapper);
    layers::set_quantized(true);
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    unsafe {
        libc::malloc_trim(0);
    }
    println!(
        "Q8-resident projections in {:.2}s (resident {:.0} MiB anon)",
        t0.elapsed().as_secs_f64(),
        rss_mib()
    );
    Ok(model)
}

/// Read the audio tower's `Gemma4ClippableLinear` clip bounds (`input_min` etc.
/// scalars) straight from the checkpoint, keyed by module path relative to
/// `audio_tower` (e.g. `layers.0.feed_forward1.ffw_layer_1`).
fn read_audio_clip_bounds(
    model_dir: &Path,
    cfg: &config::AudioConfig,
) -> Result<std::collections::HashMap<String, layers::ClipBounds>> {
    use burn_store::ModuleStore;
    let mut store = SafetensorsStore::from_file(model_dir.join("model.safetensors"));
    let mut read = |key: &str| -> Result<f32> {
        let tensor = store
            .get_tensor(key)?
            .ok_or_else(|| anyhow::anyhow!("missing clip scalar {key}"))?;
        let data = burn_store::bridge::to_data(tensor)?.convert_dtype(DType::F32);
        let v: Vec<f32> = data.try_to_vec()?;
        v.first()
            .copied()
            .ok_or_else(|| anyhow::anyhow!("empty clip scalar {key}"))
    };
    let names = [
        "feed_forward1.ffw_layer_1",
        "feed_forward1.ffw_layer_2",
        "feed_forward2.ffw_layer_1",
        "feed_forward2.ffw_layer_2",
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.post",
        "lconv1d.linear_start",
        "lconv1d.linear_end",
    ];
    let mut bounds = std::collections::HashMap::new();
    for layer in 0..cfg.num_hidden_layers {
        for name in names {
            let base = format!("audio_tower.layers.{layer}.{name}");
            let clip = layers::ClipBounds {
                input_min: read(&format!("{base}.input_min"))?,
                input_max: read(&format!("{base}.input_max"))?,
                output_min: read(&format!("{base}.output_min"))?,
                output_max: read(&format!("{base}.output_max"))?,
            };
            bounds.insert(format!("layers.{layer}.{name}"), clip);
        }
    }
    Ok(bounds)
}

/// Pack the generation model's text projections into resident fp16 NPU weights
/// for the prefill pass. `keep_cpu` retains the f32 copies (decode runs on the
/// CPU); with `keep_cpu == false` the CPU copies are dropped.
#[cfg(all(feature = "npu", target_arch = "aarch64"))]
fn pack_gen_text(
    model: &mut gen_model::GenRoot,
    threads: usize,
    device: &Device,
    keep_cpu: bool,
) -> Result<()> {
    burn_rocket::init(threads)
        .map_err(|e| anyhow::anyhow!("NPU context creation failed: {e}"))?;
    layers::set_npu_prefill_only(keep_cpu);
    let t0 = Instant::now();
    let mut count = 0usize;
    let mut bytes = 0usize;
    model.text_mut().for_each_projection_mut(|lin| {
        let (n, k) = layers::pack_linear_into_npu(lin, device, keep_cpu);
        count += 1;
        bytes += n * k * 2;
    });
    if !keep_cpu {
        #[cfg(all(target_os = "linux", target_env = "gnu"))]
        unsafe {
            libc::malloc_trim(0);
        }
    }
    println!(
        "NPU: packed {count} text projections ({:.2} GiB f16) in {:.2}s; \
         prefill on the NPU, decode on the CPU{} (resident {:.0} MiB anon)",
        bytes as f64 / (1u64 << 30) as f64,
        t0.elapsed().as_secs_f64(),
        if keep_cpu { " (CPU copies kept)" } else { "" },
        rss_mib()
    );
    Ok(())
}

/// Build `(input_ids, soft tokens, seq_len)` through the shared input assembly.
fn build_inputs(
    args: &Args,
    model: &Emb2Model,
    device: &Device,
) -> Result<(Tensor<2, Int>, Option<(Vec<usize>, Tensor<2>)>, usize)> {
    let tokenizer = args.tokenizer()?;
    let prefix = args.prompt_prefix()?;
    let (img_soft_default, video_soft_default) = inputs::media_soft_tokens(&args.model_dir);
    let text = if args.text.is_some() || args.text_file.is_some() {
        Some(args.input_text()?)
    } else {
        None
    };
    let req = inputs::MediaInputs {
        text,
        image: args.image.clone(),
        video: args.video.clone(),
        audio: args.audio.clone(),
        prompt_prefix: prefix,
        max_tokens: args.tokens,
        image_soft_tokens: args.max_soft_tokens.unwrap_or(img_soft_default),
        video_soft_tokens: args.video_soft_tokens.unwrap_or(video_soft_default),
        video_fps: args.video_fps,
        video_max_frames: args.video_max_frames,
        attn_chunk: args.attn_chunk,
    };
    let debug = inputs::DebugPaths {
        dump_image: args.dump_image.clone(),
        dump_pixels: args.dump_pixels.clone(),
        dump_ids: args.dump_ids.clone(),
        dump_vision: args.dump_vision.clone(),
        dump_vision_layers: args.dump_vision_layers.clone(),
        dump_audio: args.dump_audio.clone(),
        dump_audio_layers: args.dump_audio_layers.clone(),
    };
    let prepared = inputs::prepare(model, &tokenizer, &req, &debug)?;
    let seq = prepared.ids.len();
    Ok((
        inputs::make_input(&prepared.ids, device),
        prepared.soft,
        seq,
    ))
}


fn run_embed(args: &Args) -> Result<()> {
    let device = device(args)?;
    let (model, cfg) = load_model(
        &args.model_dir,
        args.dtype,
        args.quant_q8,
        args.npu,
        args.npu_threads,
        args.npu_attn,
        &device,
    )?;
    let (input, soft, n) = build_inputs(args, &model, &device)?;
    let rope = model.text().rope_tables(n, args.dtype, &device);

    stage_stats_reset();
    let t0 = Instant::now();
    let embedded = model.embed_ids(input, &rope, args.attn_chunk, soft);
    let mut v: Vec<f32> = embedded.cast(DType::F32).into_data().try_to_vec()?;
    let elapsed = t0.elapsed().as_secs_f64();
    if args.normalize {
        if let Some(dim) = args.dim {
            v.truncate(dim);
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for x in &mut v {
            *x /= norm;
        }
    }
    let (attn, mlp, norms, ple) = stage_stats();
    println!(
        "tokens={n} dim={} forward={elapsed:.2}s (attn {attn:.2}s, mlp {mlp:.2}s, norms {norms:.2}s, ple {ple:.2}s) model layers={} hidden={}",
        v.len(),
        cfg.text_config.num_hidden_layers,
        cfg.text_config.hidden_size,
    );
    if let Ok(path) = std::env::var("DUMP_VEC").map(PathBuf::from) {
        std::fs::write(&path, serde_json::to_string(&v)?)?;
        println!("wrote {}", path.display());
    }
    match &args.out {
        Some(path) => {
            std::fs::write(path, serde_json::to_string(&v)?)
                .with_context(|| format!("write {}", path.display()))?;
            println!("wrote {}", path.display());
        }
        None => {
            let head: Vec<String> = v.iter().take(8).map(|x| format!("{x:.6}")).collect();
            println!("[{}, ...] len={}", head.join(", "), v.len());
        }
    }
    Ok(())
}

fn run_bench(args: &Args) -> Result<()> {
    let device = device(args)?;
    let (model, _cfg) = load_model(
        &args.model_dir,
        args.dtype,
        args.quant_q8,
        args.npu,
        args.npu_threads,
        args.npu_attn,
        &device,
    )?;
    let (input, soft, n) = build_inputs(args, &model, &device)?;
    let rope = model.text().rope_tables(n, args.dtype, &device);

    for rep in 0..args.reps.max(1) {
        stage_stats_reset();
        let t0 = Instant::now();
        let out = model.embed_ids(input.clone(), &rope, args.attn_chunk, soft.clone());
        let _ = out.into_data();
        let wall = t0.elapsed().as_secs_f64();
        let (attn, mlp, norms, ple) = stage_stats();
        println!(
            "rep {rep}: tokens={n} wall={wall:.2}s ({:.1} tok/s) attn={attn:.2}s mlp={mlp:.2}s norms={norms:.2}s ple={ple:.2}s",
            n as f64 / wall
        );
        #[cfg(all(feature = "npu", target_arch = "aarch64"))]
        if args.npu {
            let st = burn_rocket::stats();
            println!(
                "  npu: calls={} convert={:.2}s npu={:.2}s flex+overhead={:.2}s",
                st.calls,
                st.convert_s,
                st.npu_s,
                wall - st.convert_s - st.npu_s
            );
        }
    }
    Ok(())
}

fn run_tokenize(args: &Args) -> Result<()> {
    let tokenizer = args.tokenizer()?;
    let prefix = args.prompt_prefix()?;
    let text = if args.text.is_some() || args.text_file.is_some() {
        args.input_text()?
    } else if args.image.is_some() {
        "<|image|>".to_string()
    } else {
        bail!("provide --text, --text-file or --image")
    };
    let enc = tokenizer
        .encode(format!("{prefix}{text}").as_str(), true)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
    let mut ids = enc.get_ids().to_vec();
    if let Some(n) = args.tokens {
        ids.truncate(n);
    }
    println!("{} tokens", ids.len());
    println!("{ids:?}");
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
        args.npu_attn,
        &device,
    )?;
    let tokenizer = args.tokenizer()?;
    server::serve(
        model,
        cfg,
        tokenizer,
        device,
        args.dtype,
        server::ServeOptions {
            addr: format!("0.0.0.0:{}", args.port),
            model_name: args.model_name.clone(),
            model_dir: args.model_dir.clone(),
            attn_chunk: args.attn_chunk,
            max_tokens: args.max_tokens,
            video_fps: args.video_fps,
            video_max_frames: args.video_max_frames,
        },
    )
}

#[derive(serde::Deserialize)]
struct MessageJson {
    role: String,
    content: String,
}

/// EOS ids from the checkpoint's `generation_config.json` (fallback: 1, 106, 50).
pub(crate) fn gen_eos(model_dir: &Path) -> Vec<u32> {
    let mut eos = vec![1u32, 106, 50];
    if let Ok(text) = std::fs::read_to_string(model_dir.join("generation_config.json")) {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
            match json.get("eos_token_id") {
                Some(serde_json::Value::Number(n)) => {
                    if let Some(v) = n.as_u64() {
                        eos = vec![v as u32];
                    }
                }
                Some(serde_json::Value::Array(a)) => {
                    eos = a.iter().filter_map(|v| v.as_u64()).map(|v| v as u32).collect();
                }
                _ => {}
            }
        }
    }
    eos
}

fn run_gen(args: &Args) -> Result<()> {
    let device = device(args)?;
    let dtype = if args.quant_q8 {
        gen_loader::LoadDtype::Q8
    } else if args.gen_f16 {
        gen_loader::LoadDtype::F16
    } else {
        gen_loader::LoadDtype::F32
    };
    let (mut model, cfg) = gen_loader::load_gen_model(&args.gen_model_dir, dtype, &device)?;
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    if args.npu {
        pack_gen_text(&mut model, args.npu_threads, &device, true)?;
    }
    #[cfg(not(all(feature = "npu", target_arch = "aarch64")))]
    if args.npu {
        bail!("--npu requires an aarch64 build with --features npu");
    }
    let lm_head =
        gen_loader::build_lm_head(model.text(), dtype != gen_loader::LoadDtype::F32, &device);
    let tokenizer = Tokenizer::from_file(args.gen_model_dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;

    let messages: Vec<chat::Message> = match &args.messages {
        Some(json) => serde_json::from_str::<Vec<MessageJson>>(json)
            .context("parse --messages JSON")?
            .into_iter()
            .map(|m| chat::Message {
                role: m.role,
                content: m.content,
            })
            .collect(),
        None => vec![chat::Message::user(args.input_text()?)],
    };
    let rendered = chat::render(&messages, args.enable_thinking);
    // Media inputs expand their `<|image|>` / `<|audio|>` placeholders in the
    // rendered conversation and are encoded with the checkpoint's own towers.
    let (ids, soft) = if args.image.is_some() || args.audio.is_some() {
        let (img_soft, video_soft) = inputs::media_soft_tokens(&args.gen_model_dir);
        let req = inputs::MediaInputs {
            text: Some(rendered.clone()),
            image: args.image.clone(),
            video: None,
            audio: args.audio.clone(),
            prompt_prefix: String::new(),
            max_tokens: None,
            image_soft_tokens: args.max_soft_tokens.unwrap_or(img_soft),
            video_soft_tokens: video_soft,
            video_fps: args.video_fps,
            video_max_frames: args.video_max_frames,
            attn_chunk: args.attn_chunk,
        };
        let prepared = inputs::prepare(&model, &tokenizer, &req, &inputs::DebugPaths::default())?;
        (prepared.ids, prepared.soft)
    } else {
        (chat::encode(&tokenizer, &rendered)?, None)
    };
    if std::env::var("DUMP_RENDER").is_ok() {
        println!("rendered: {rendered:?}");
        println!("prompt ids: {ids:?}");
    }
    let opts = chat::GenOptions {
        max_new_tokens: args.max_new_tokens,
        greedy: args.greedy,
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        eos: gen_eos(&args.gen_model_dir),
    };
    println!(
        "model: {} layers, hidden {}, ple_dim {}, eos {:?}",
        cfg.text_config.num_hidden_layers,
        cfg.text_config.hidden_size,
        cfg.text_config.hidden_size_per_layer_input,
        opts.eos
    );
    gen_model::gen_stage_stats_reset();
    let pad_id = cfg.text_config.pad_token_id.unwrap_or(0);
    let (out, stats) = chat::generate_with_media(
        &model,
        &lm_head,
        &ids,
        soft,
        pad_id,
        &opts,
        args.attn_chunk,
        &device,
    )?;
    let (attn, mlp, ple, norms) = gen_model::gen_stage_stats();
    println!(
        "stages: attn {attn:.2}s, mlp {mlp:.2}s, ple {ple:.2}s, norms {norms:.2}s ({} tokens)",
        stats.prompt_tokens + stats.generated
    );
    let text = tokenizer
        .decode(&out, false)
        .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
    println!("prompt {} tokens, generated {} in {:.2}s (prefill {:.2}s, decode {:.2}s = {:.2} tok/s)",
        stats.prompt_tokens, stats.generated, stats.prefill_s + stats.decode_s,
        stats.prefill_s, stats.decode_s,
        stats.generated as f64 / stats.decode_s.max(1e-9));
    println!("ids: {out:?}");
    println!("text: {text}");
    if let Some(path) = &args.out {
        std::fs::write(path, serde_json::to_string(&out)?)
            .with_context(|| format!("write {}", path.display()))?;
        println!("wrote {}", path.display());
    }
    Ok(())
}

fn run_serve_chat(args: &Args) -> Result<()> {
    let device = device(args)?;
    let dtype = if args.quant_q8 {
        gen_loader::LoadDtype::Q8
    } else if args.gen_f16 {
        gen_loader::LoadDtype::F16
    } else {
        gen_loader::LoadDtype::F32
    };
    let (mut model, _cfg) = gen_loader::load_gen_model(&args.gen_model_dir, dtype, &device)?;
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    if args.npu {
        pack_gen_text(&mut model, args.npu_threads, &device, true)?;
    }
    #[cfg(not(all(feature = "npu", target_arch = "aarch64")))]
    if args.npu {
        bail!("--npu requires an aarch64 build with --features npu");
    }
    let lm_head =
        gen_loader::build_lm_head(model.text(), dtype != gen_loader::LoadDtype::F32, &device);
    let tokenizer = Tokenizer::from_file(args.gen_model_dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    let model_name = if args.model_name == "embeddinggemma-2" {
        // The chat server's default name should match the generation model.
        args.gen_model_dir
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "gemma-4-E2B-it".to_string())
    } else {
        args.model_name.clone()
    };
    gen_server::serve(
        model,
        lm_head,
        tokenizer,
        device,
        gen_server::ChatServeOptions {
            addr: format!("0.0.0.0:{}", args.port),
            model_name,
            model_dir: args.gen_model_dir.clone(),
            attn_chunk: args.attn_chunk,
            max_new_tokens: args.max_new_tokens,
        },
    )
}
