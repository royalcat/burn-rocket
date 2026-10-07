//! `gemma embed|bench|tokenize` subcommands: the EmbeddingGemma 2
//! multimodal embedding model (serving lives in the top-level `serve` command).

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use burn::prelude::*;
use burn::tensor::{DType, Int};
use tokenizers::Tokenizer;

use crate::cli::FlagArgs;
use crate::gemma::embeddinggemma::load::load_model;
use crate::gemma::embeddinggemma::model::{stage_stats, stage_stats_reset};
use crate::gemma::inputs;
use crate::util::device;

/// `(input ids, optional media soft tokens, sequence length)`.
type PreparedInput = (Tensor<2, Int>, Option<(Vec<usize>, Tensor<2>)>, usize);

pub fn run(cmd: &str, it: impl Iterator<Item = String>) -> Result<()> {
    let args = Args::parse(cmd, FlagArgs::new(it))?;
    match cmd {
        "bench" => run_bench(&args),
        "embed" => run_embed(&args),
        "tokenize" => run_tokenize(&args),
        _ => bail!("unknown gemma command '{cmd}'"),
    }
}

struct Args {
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
    quant_q8: bool,
    npu: bool,
    npu_threads: usize,
    npu_attn: bool,
}

impl Args {
    fn parse(cmd: &str, mut f: FlagArgs) -> Result<Self> {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        let mut args = Args {
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
            quant_q8: false,
            npu: false,
            npu_threads: 5,
            npu_attn: true,
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
                other => bail!("unsupported dtype '{other}' (expected f32|f16)"),
            };
        }
        if let Some(v) = f.take("--text")? {
            args.text = Some(v);
        }
        if let Some(v) = f.take("--text-file")? {
            args.text_file = Some(PathBuf::from(v));
        }
        if let Some(v) = f.take("--image")? {
            args.image = Some(PathBuf::from(v));
        }
        if let Some(v) = f.take("--video")? {
            args.video = Some(PathBuf::from(v));
        }
        if let Some(v) = f.take_parsed("--video-fps")? {
            args.video_fps = v;
        }
        if let Some(v) = f.take_parsed("--video-max-frames")? {
            args.video_max_frames = v;
        }
        if let Some(v) = f.take("--audio")? {
            args.audio = Some(PathBuf::from(v));
        }
        if let Some(v) = f.take_parsed("--max-soft-tokens")? {
            args.max_soft_tokens = Some(v);
        }
        if let Some(v) = f.take_parsed("--video-soft-tokens")? {
            args.video_soft_tokens = Some(v);
        }
        if let Some(v) = f.take("--prompt")? {
            args.prompt = Some(v);
        }
        if let Some(v) = f.take_parsed("--dim")? {
            args.dim = Some(v);
        }
        args.normalize = !f.take_bool("--no-normalize");
        if let Some(v) = f.take_parsed("--tokens")? {
            args.tokens = Some(v);
        }
        if let Some(v) = f.take_parsed("--reps")? {
            args.reps = v;
        }
        if let Some(v) = f.take("--out")? {
            args.out = Some(PathBuf::from(v));
        }
        if let Some(v) = f.take_parsed("--attn-chunk")? {
            args.attn_chunk = v;
        }
        if let Some(v) = f.take("--dump-pixels")? {
            args.dump_pixels = Some(PathBuf::from(v));
        }
        if let Some(v) = f.take("--dump-ids")? {
            args.dump_ids = Some(PathBuf::from(v));
        }
        if let Some(v) = f.take("--dump-audio")? {
            args.dump_audio = Some(PathBuf::from(v));
        }
        if let Some(v) = f.take("--dump-audio-layers")? {
            args.dump_audio_layers = Some(PathBuf::from(v));
        }
        if let Some(v) = f.take("--dump-image")? {
            args.dump_image = Some(PathBuf::from(v));
        }
        if let Some(v) = f.take("--dump-vision")? {
            args.dump_vision = Some(PathBuf::from(v));
        }
        if let Some(v) = f.take("--dump-vision-layers")? {
            args.dump_vision_layers = Some(PathBuf::from(v));
        }
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
        f.finish(&format!("gemma {cmd}"))?;
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

/// Build `(input_ids, soft tokens, seq_len)` through the shared input assembly.
fn build_inputs(
    args: &Args,
    model: &crate::gemma::embeddinggemma::model::Emb2Model,
    device: &Device,
) -> Result<PreparedInput> {
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
    let device = device(&args.backend)?;
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
