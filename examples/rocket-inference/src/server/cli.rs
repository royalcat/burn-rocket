//! `serve`: load one model (auto-detected from the checkpoint's `config.json`
//! or selected with `--family`) and serve every endpoint compatible with it.

use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use burn::tensor::DType;
use tokenizers::Tokenizer;

use crate::cli::FlagArgs;
use crate::gemma::embeddinggemma::load as emb2_load;
use crate::gemma::gemma4::loader::{self as gen_loader, LoadDtype};
use crate::gemma::inputs;
use crate::qwen3_embedding::load as qwen3_load;
use crate::qwen35_intent::loader as intent_loader;
use crate::qwen35_intent::loader::IntentLoadOptions;
use crate::util::device;

use super::Settings;
use super::engine::{
    Capabilities, Emb2Engine, Engine, Family, Gemma4Engine, IntentEngine, Qwen3Engine,
};

pub fn run(it: impl Iterator<Item = String>) -> Result<()> {
    let args = Args::parse(FlagArgs::new(it))?;
    let model_dir = args.model_dir();
    let family = match args.family {
        Some(family) => family,
        None => super::engine::detect_family(&model_dir)?,
    };
    args.validate(family)?;

    let device = device(&args.backend())?;
    let tokenizer = || -> Result<Tokenizer> {
        let path = model_dir.join("tokenizer.json");
        Tokenizer::from_file(&path)
            .map_err(|e| anyhow::anyhow!("tokenizer {}: {e}", path.display()))
    };

    // Per-family defaults for the server-side knobs.
    let (default_max_tokens, default_max_new) = match family {
        Family::Qwen3Embed => (30000, 64),
        Family::Emb2 => (8192, 64),
        Family::Intent => (30000, 256),
        Family::Gemma4 => (30000, 64),
    };
    let max_tokens = args.max_tokens.unwrap_or(default_max_tokens);
    let max_new_tokens = args.max_new_tokens.unwrap_or(default_max_new);
    let temperature = args.temperature.unwrap_or(0.0);

    let (engine, capabilities) = match family {
        Family::Qwen3Embed => {
            let (model, cfg) = qwen3_load::load_model(
                &model_dir,
                args.dtype.unwrap_or(DType::F32),
                args.quant.unwrap_or(false),
                args.npu.unwrap_or(false),
                args.npu_threads.unwrap_or(5),
                args.npu_attn.unwrap_or(true),
                &device,
            )?;
            let tokenizer = tokenizer()?;
            (
                Engine::Qwen3(Qwen3Engine {
                    model,
                    cfg,
                    tokenizer,
                    device,
                    chunk: args.chunk.unwrap_or(256),
                    key_block: args.key_block.unwrap_or(256),
                    attn_fused: args.attn.unwrap_or(true),
                    max_tokens,
                }),
                Capabilities {
                    embeddings: true,
                    multimodal: false,
                    chat: false,
                },
            )
        }
        Family::Emb2 => {
            let (model, _cfg) = emb2_load::load_model(
                &model_dir,
                args.dtype.unwrap_or(DType::F32),
                args.quant.unwrap_or(false),
                args.npu.unwrap_or(false).then_some(emb2_load::NpuOpts {
                    threads: args.npu_threads.unwrap_or(5),
                    attn: args.npu_attn.unwrap_or(true),
                    int8: false,
                }),
                &device,
            )?;
            let tokenizer = tokenizer()?;
            let (image_soft_tokens, video_soft_tokens) = inputs::media_soft_tokens(&model_dir);
            (
                Engine::Emb2(Emb2Engine {
                    model,
                    tokenizer,
                    device,
                    dtype: DType::F32,
                    max_tokens,
                    attn_chunk: args.attn_chunk.unwrap_or(1024),
                    image_soft_tokens,
                    video_soft_tokens,
                    video_fps: args.video_fps.unwrap_or(1.0),
                    video_max_frames: args.video_max_frames.unwrap_or(32),
                    prompts: inputs::task_prompts(&model_dir),
                }),
                Capabilities {
                    embeddings: true,
                    multimodal: true,
                    chat: false,
                },
            )
        }
        Family::Intent => {
            let opts = IntentLoadOptions {
                npu: args.npu.unwrap_or(false),
                npu_threads: args.npu_threads.unwrap_or(5),
                embed_f16: args.embed_f16.unwrap_or(false),
                pure_npu: args.pure_npu.unwrap_or(false),
            };
            let loaded = intent_loader::load_for_serving(
                &model_dir,
                &device,
                &opts,
                args.delta_chunk.unwrap_or(64),
                opts.pure_npu,
                args.npu_decode.unwrap_or(false),
            )?;
            (
                Engine::Intent(IntentEngine {
                    model: loaded.model,
                    tokenizer: loaded.tokenizer,
                    stop_ids: loaded.stop_ids,
                    max_tokens,
                }),
                Capabilities {
                    embeddings: false,
                    multimodal: false,
                    chat: true,
                },
            )
        }
        Family::Gemma4 => {
            let dtype = if args.quant.unwrap_or(false) {
                LoadDtype::Q8
            } else if args.gen_f16.unwrap_or(false) {
                LoadDtype::F16
            } else {
                LoadDtype::F32
            };
            #[allow(unused_mut)]
            let (mut model, cfg) = gen_loader::load_gen_model(&model_dir, dtype, &device)?;
            #[cfg(all(feature = "npu", target_arch = "aarch64"))]
            if args.npu.unwrap_or(false) {
                gen_loader::pack_text_for_prefill(
                    &mut model,
                    args.npu_threads.unwrap_or(5),
                    &device,
                    true,
                )?;
            }
            #[cfg(not(all(feature = "npu", target_arch = "aarch64")))]
            if args.npu.unwrap_or(false) {
                bail!("--npu requires an aarch64 build with --features npu");
            }
            let lm_head = gen_loader::build_lm_head(model.text(), dtype != LoadDtype::F32, &device);
            let tokenizer = tokenizer()?;
            let (image_soft_tokens, video_soft_tokens) = inputs::media_soft_tokens(&model_dir);
            (
                Engine::Gemma4(Gemma4Engine {
                    model,
                    lm_head,
                    tokenizer,
                    device,
                    pad_id: cfg.text_config.pad_token_id.unwrap_or(0),
                    eos: crate::gemma::gemma4::chat::gen_eos(&model_dir),
                    attn_chunk: args.attn_chunk.unwrap_or(1024),
                    image_soft_tokens,
                    video_soft_tokens,
                }),
                Capabilities {
                    embeddings: false,
                    multimodal: false,
                    chat: true,
                },
            )
        }
    };

    let model_name = args
        .model_name
        .clone()
        .unwrap_or_else(|| dir_name(&model_dir));
    let settings = Settings {
        model_name,
        family,
        capabilities,
        max_tokens,
        max_new_tokens,
        temperature,
    };
    super::serve(
        &format!("0.0.0.0:{}", args.port.unwrap_or(8383)),
        engine,
        settings,
    )
}

fn dir_name(dir: &Path) -> String {
    dir.file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "model".to_string())
}

/// Flags as given (unset = use the family default).
#[derive(Default)]
struct Args {
    model_dir: Option<PathBuf>,
    family: Option<Family>,
    backend: Option<String>,
    port: Option<u16>,
    model_name: Option<String>,
    max_tokens: Option<usize>,
    max_new_tokens: Option<usize>,
    temperature: Option<f32>,
    dtype: Option<DType>,
    quant: Option<bool>,
    npu: Option<bool>,
    npu_threads: Option<usize>,
    npu_attn: Option<bool>,
    chunk: Option<usize>,
    key_block: Option<usize>,
    attn: Option<bool>,
    attn_chunk: Option<usize>,
    video_fps: Option<f64>,
    video_max_frames: Option<usize>,
    delta_chunk: Option<usize>,
    embed_f16: Option<bool>,
    pure_npu: Option<bool>,
    npu_decode: Option<bool>,
    gen_f16: Option<bool>,
}

impl Args {
    fn parse(mut f: FlagArgs) -> Result<Self> {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        let mut args = Args::default();
        if let Some(v) = f.take("--model-dir")? {
            args.model_dir = Some(PathBuf::from(v));
        }
        if let Some(v) = f.take("--family")? {
            args.family = match v.as_str() {
                "auto" => None,
                other => Some(Family::parse(other)?),
            };
        }
        if let Some(v) = f.take("--backend")? {
            args.backend = Some(v);
        }
        if let Some(v) = f.take_parsed("--port")? {
            args.port = Some(v);
        }
        if let Some(v) = f.take("--model-name")? {
            args.model_name = Some(v);
        }
        if let Some(v) = f.take_parsed("--max-tokens")? {
            args.max_tokens = Some(v);
        }
        if let Some(v) = f.take_parsed("--max-new-tokens")? {
            args.max_new_tokens = Some(v);
        }
        if let Some(v) = f.take_parsed("--temperature")? {
            args.temperature = Some(v);
        }
        if let Some(v) = f.take("--dtype")? {
            args.dtype = Some(match v.as_str() {
                "f32" => DType::F32,
                "f16" => DType::F16,
                other => bail!("unsupported dtype '{other}' (expected f32|f16)"),
            });
        }
        if let Some(v) = f.take_choice("--quant", &["none", "q8"])? {
            args.quant = Some(v == "q8");
        }
        args.npu = f.take_flag("--npu");
        if let Some(v) = f.take_parsed("--npu-threads")? {
            args.npu_threads = Some(v);
        }
        if let Some(v) = f.take_choice("--npu-attn", &["npu", "cpu"])? {
            args.npu_attn = Some(v == "npu");
        }
        if let Some(v) = f.take_parsed("--chunk")? {
            args.chunk = Some(v);
        }
        if let Some(v) = f.take_parsed("--key-block")? {
            args.key_block = Some(v);
        }
        if let Some(v) = f.take_choice("--attn", &["fused", "blocked"])? {
            args.attn = Some(v == "fused");
        }
        if let Some(v) = f.take_parsed("--attn-chunk")? {
            args.attn_chunk = Some(v);
        }
        if let Some(v) = f.take_parsed("--video-fps")? {
            args.video_fps = Some(v);
        }
        if let Some(v) = f.take_parsed("--video-max-frames")? {
            args.video_max_frames = Some(v);
        }
        if let Some(v) = f.take_parsed("--delta-chunk")? {
            args.delta_chunk = Some(v);
        }
        args.embed_f16 = f.take_flag("--embed-f16");
        args.pure_npu = f.take_flag("--pure-npu");
        args.npu_decode = f.take_flag("--npu-decode");
        args.gen_f16 = f.take_flag("--f16");
        f.finish("serve")?;
        // Defaults that are not family-specific.
        if args.model_dir.is_none() {
            args.model_dir = Some(PathBuf::from(format!("{home}/models/qwen3-embedding-0.6b")));
        }
        Ok(args)
    }

    fn model_dir(&self) -> PathBuf {
        self.model_dir
            .clone()
            .expect("model-dir defaulted in parse")
    }

    fn backend(&self) -> String {
        self.backend
            .clone()
            .unwrap_or_else(|| if cfg!(feature = "cpu") { "cpu" } else { "flex" }.to_string())
    }

    /// Reject flags that do not apply to the detected model.
    fn validate(&self, family: Family) -> Result<()> {
        let allowed: &[&str] = match family {
            Family::Qwen3Embed => &[
                "model-dir",
                "family",
                "backend",
                "port",
                "model-name",
                "max-tokens",
                "max-new-tokens",
                "temperature",
                "dtype",
                "quant",
                "npu",
                "npu-threads",
                "npu-attn",
                "chunk",
                "key-block",
                "attn",
            ],
            Family::Emb2 => &[
                "model-dir",
                "family",
                "backend",
                "port",
                "model-name",
                "max-tokens",
                "max-new-tokens",
                "temperature",
                "dtype",
                "quant",
                "npu",
                "npu-threads",
                "npu-attn",
                "attn-chunk",
                "video-fps",
                "video-max-frames",
            ],
            Family::Intent => &[
                "model-dir",
                "family",
                "backend",
                "port",
                "model-name",
                "max-tokens",
                "max-new-tokens",
                "temperature",
                "npu",
                "npu-threads",
                "delta-chunk",
                "embed-f16",
                "pure-npu",
                "npu-decode",
            ],
            Family::Gemma4 => &[
                "model-dir",
                "family",
                "backend",
                "port",
                "model-name",
                "max-tokens",
                "max-new-tokens",
                "temperature",
                "quant",
                "npu",
                "npu-threads",
                "f16",
                "attn-chunk",
            ],
        };
        let set: [(&str, bool); 23] = [
            ("model-dir", self.model_dir.is_some()),
            ("family", self.family.is_some()),
            ("backend", self.backend.is_some()),
            ("port", self.port.is_some()),
            ("model-name", self.model_name.is_some()),
            ("max-tokens", self.max_tokens.is_some()),
            ("max-new-tokens", self.max_new_tokens.is_some()),
            ("temperature", self.temperature.is_some()),
            ("dtype", self.dtype.is_some()),
            ("quant", self.quant.is_some()),
            ("npu", self.npu.is_some()),
            ("npu-threads", self.npu_threads.is_some()),
            ("npu-attn", self.npu_attn.is_some()),
            ("chunk", self.chunk.is_some()),
            ("key-block", self.key_block.is_some()),
            ("attn", self.attn.is_some()),
            ("attn-chunk", self.attn_chunk.is_some()),
            ("video-fps", self.video_fps.is_some()),
            ("video-max-frames", self.video_max_frames.is_some()),
            ("delta-chunk", self.delta_chunk.is_some()),
            ("embed-f16", self.embed_f16.is_some()),
            ("pure-npu", self.pure_npu.is_some()),
            ("npu-decode", self.npu_decode.is_some()),
        ];
        for (name, present) in set {
            if present && !allowed.contains(&name) {
                bail!("--{name} does not apply to the {} model", family.name());
            }
        }
        if self.gen_f16.is_some() && !allowed.contains(&"f16") {
            bail!("--f16 does not apply to the {} model", family.name());
        }
        Ok(())
    }
}
