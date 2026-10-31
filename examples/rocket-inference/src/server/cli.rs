//! `serve`: load one model for the detected/overridden family and serve every
//! endpoint compatible with it.

use std::path::Path;

use anyhow::{Result, bail};
use tokenizers::Tokenizer;

use crate::cli::{FlagArgs, LoadArgs, ResolvedLoad};
use crate::commands::{gemma4_load_dtype, resolve};
use crate::gemma::embeddinggemma::load as emb2_load;
use crate::gemma::gemma4::loader as gen_loader;
use crate::gemma::inputs;
use crate::qwen3_embedding::load as qwen3_load;
use crate::qwen35_intent::loader as intent_loader;
use crate::qwen35_intent::loader::IntentLoadOptions;
use crate::util::checkpoint::Checkpoint;
use crate::util::device;

use super::Settings;
use super::engine::{
    Capabilities, Emb2Engine, Engine, Family, Gemma4Engine, IntentEngine, Qwen3Engine,
};

pub fn run(it: impl Iterator<Item = String>) -> Result<()> {
    let args = Args::parse(FlagArgs::new(it))?;
    let Args {
        load,
        port,
        model_name,
        max_tokens,
        max_new_tokens,
        temperature,
        video_fps,
        video_max_frames,
        delta_chunk,
    } = args;
    let load = resolve(load)?;
    validate(&load, video_fps, video_max_frames, delta_chunk)?;
    let model_dir = load.model_dir.clone();
    let family = load.family;

    let device = device(&load.cpu_backend)?;
    let tokenizer = || -> Result<Tokenizer> {
        crate::util::checkpoint::open_in_dir(&model_dir)?
            .tokenizer()
            .map_err(|e| anyhow::anyhow!("tokenizer: {e:#}"))
    };

    // Per-family defaults for the server-side knobs.
    let (default_max_tokens, default_max_new) = match family {
        Family::Qwen3Embed => (30000, 64),
        Family::Emb2 => (8192, 64),
        Family::Intent => (30000, 256),
        Family::Gemma4 => (30000, 64),
    };
    let max_tokens = max_tokens.unwrap_or(default_max_tokens);
    let max_new_tokens = max_new_tokens.unwrap_or(default_max_new);
    let temperature = temperature.unwrap_or(0.0);

    let (engine, capabilities) = match family {
        Family::Qwen3Embed => {
            let mut ckpt = crate::util::checkpoint::open_in_dir(&model_dir)?;
            let npu = load.placement.any_npu().then_some(qwen3_load::NpuOpts {
                threads: load.npu_threads,
                text: load.placement.text,
                attn: load.placement.text_attn,
            });
            let (model, cfg) =
                qwen3_load::load_model(&mut ckpt, load.dtype, load.quant_q8, npu, &device)?;
            let tokenizer = ckpt.tokenizer()?;
            (
                Engine::Qwen3(Box::new(Qwen3Engine {
                    model,
                    cfg,
                    tokenizer,
                    device,
                    dtype: load.dtype,
                    chunk: load.chunk,
                    key_block: load.key_block,
                    attn_fused: crate::cli::AttnKernel::Fused == load.attn_kernel,
                    max_tokens,
                })),
                Capabilities {
                    embeddings: true,
                    multimodal: false,
                    chat: false,
                },
            )
        }
        Family::Emb2 => {
            let npu = load.placement.any_npu().then_some(emb2_load::NpuOpts {
                threads: load.npu_threads,
                text: load.placement.text,
                attn: load.placement.text_attn,
                vision: load.placement.vision,
                vision_attn: load.placement.vision_attn,
                int8: load.npu_dtype == crate::cli::NpuDtype::Int8,
                i8_group: load.int8_group,
            });
            let (model, _cfg) = emb2_load::load_model(
                &model_dir,
                load.dtype,
                load.quant_q8,
                npu,
                load.defer_towers,
                &device,
            )?;
            let tokenizer = tokenizer()?;
            let (image_soft_tokens, video_soft_tokens) = inputs::media_soft_tokens(&model_dir);
            (
                Engine::Emb2(Box::new(Emb2Engine {
                    model,
                    tokenizer,
                    device,
                    dtype: load.dtype,
                    model_dir: model_dir.clone(),
                    quant_q8: load.quant_q8,
                    npu_pack: load.placement.vision,
                    max_tokens,
                    attn_chunk: load.attn_chunk,
                    vision_attn: load.placement.vision_attn(load.vision_attn_kernel),
                    image_soft_tokens,
                    video_soft_tokens,
                    video_fps: video_fps.unwrap_or(1.0),
                    video_max_frames: video_max_frames.unwrap_or(32),
                    prompts: inputs::task_prompts(&model_dir),
                })),
                Capabilities {
                    embeddings: true,
                    multimodal: true,
                    chat: false,
                },
            )
        }
        Family::Intent => {
            let opts = IntentLoadOptions {
                npu: load.placement.text,
                text: load.placement.text,
                decode: load.placement.decode,
                keep_cpu_weights: load.keep_cpu_weights,
                npu_threads: load.npu_threads,
                embed_f16: load.embed_f16,
            };
            let loaded = intent_loader::load_for_serving(
                &model_dir,
                &device,
                &opts,
                delta_chunk.unwrap_or(64),
            )?;
            (
                Engine::Intent(Box::new(IntentEngine {
                    model: loaded.model,
                    tokenizer: loaded.tokenizer,
                    stop_ids: loaded.stop_ids,
                    max_tokens,
                })),
                Capabilities {
                    embeddings: false,
                    multimodal: false,
                    chat: true,
                },
            )
        }
        Family::Gemma4 => {
            let dtype = gemma4_load_dtype(&load);
            #[allow(unused_mut)]
            let (mut model, cfg) =
                gen_loader::load_gen_model(&model_dir, dtype, &device, load.defer_towers)?;
            #[cfg(all(feature = "npu", target_arch = "aarch64"))]
            if load.placement.text {
                gen_loader::pack_text_for_prefill(
                    &mut model,
                    load.npu_threads,
                    &device,
                    load.keep_cpu_weights,
                    load.placement.decode,
                )?;
            }
            #[cfg(not(all(feature = "npu", target_arch = "aarch64")))]
            if load.placement.any_npu() {
                bail!("an NPU placement requires an aarch64 build with `--features npu`");
            }
            let lm_head = gen_loader::build_lm_head(model.text(), dtype.tensor_dtype(), &device);
            let tokenizer = tokenizer()?;
            let (image_soft_tokens, video_soft_tokens) = inputs::media_soft_tokens(&model_dir);
            (
                Engine::Gemma4(Box::new(Gemma4Engine {
                    model,
                    lm_head,
                    tokenizer,
                    device,
                    model_dir: model_dir.clone(),
                    pad_id: cfg.text_config.pad_token_id.unwrap_or(0),
                    eos: crate::gemma::gemma4::chat::gen_eos(&model_dir),
                    attn_chunk: load.attn_chunk,
                    image_soft_tokens,
                    video_soft_tokens,
                })),
                Capabilities {
                    embeddings: false,
                    multimodal: false,
                    chat: true,
                },
            )
        }
    };

    let model_name = model_name.unwrap_or_else(|| dir_name(&model_dir));
    let settings = Settings {
        model_name,
        family,
        capabilities,
        max_tokens,
        max_new_tokens,
        temperature,
    };
    super::serve(
        &format!("0.0.0.0:{}", port.unwrap_or(8383)),
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
    load: LoadArgs,
    port: Option<u16>,
    model_name: Option<String>,
    max_tokens: Option<usize>,
    max_new_tokens: Option<usize>,
    temperature: Option<f32>,
    video_fps: Option<f64>,
    video_max_frames: Option<usize>,
    delta_chunk: Option<usize>,
}

impl Args {
    fn parse(mut f: FlagArgs) -> Result<Self> {
        let load = LoadArgs::parse(&mut f)?;
        let mut args = Args {
            load,
            ..Default::default()
        };
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
        if let Some(v) = f.take_parsed("--video-fps")? {
            args.video_fps = Some(v);
        }
        if let Some(v) = f.take_parsed("--video-max-frames")? {
            args.video_max_frames = Some(v);
        }
        if let Some(v) = f.take_parsed("--delta-chunk")? {
            args.delta_chunk = Some(v);
        }
        f.finish("serve")?;
        Ok(args)
    }
}

/// Serve-only flags that depend on the detected family.
fn validate(
    load: &ResolvedLoad,
    video_fps: Option<f64>,
    video_max_frames: Option<usize>,
    delta_chunk: Option<usize>,
) -> Result<()> {
    if video_fps.is_some() && load.family != Family::Emb2 {
        bail!(
            "--video-fps only applies to the embeddinggemma video input \
             (the {} model has no video path)",
            load.family.name()
        );
    }
    if video_max_frames.is_some() && load.family != Family::Emb2 {
        bail!(
            "--video-max-frames only applies to the embeddinggemma video input \
             (the {} model has no video path)",
            load.family.name()
        );
    }
    if delta_chunk.is_some() && load.family != Family::Intent {
        bail!("--delta-chunk only applies to the intent model");
    }
    Ok(())
}
