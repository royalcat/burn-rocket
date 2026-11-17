//! Shared input flags and assembly for the embedding commands (`embed`,
//! `bench`): text plus the EmbeddingGemma 2 media inputs, the task prompt and
//! the debug dumps.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use burn::prelude::*;
use burn::tensor::Int;

use crate::cli::{FlagArgs, NpuDtype, ResolvedLoad};
use crate::commands::{input_text, prompt_prefix};
use crate::family::Family;
use crate::gemma::embeddinggemma::load::NpuOpts as Emb2NpuOpts;
use crate::gemma::inputs;
use crate::gemma::vision::{vision_stats, vision_stats_reset};
use crate::util::checkpoint::Checkpoint;

/// `(input ids, optional media soft tokens, sequence length)`.
pub type PreparedInput = (Tensor<2, Int>, Option<(Vec<usize>, Tensor<2>)>, usize);

/// The input half of `embed`/`bench` (everything except the output and reps).
#[derive(Debug, Clone)]
pub struct Inputs {
    pub text: Option<String>,
    pub text_file: Option<PathBuf>,
    pub image: Option<PathBuf>,
    pub video: Option<PathBuf>,
    pub audio: Option<PathBuf>,
    pub video_fps: f64,
    pub video_max_frames: usize,
    pub max_soft_tokens: Option<usize>,
    pub video_soft_tokens: Option<usize>,
    pub prompt: Option<String>,
    pub dim: Option<usize>,
    pub normalize: bool,
    pub tokens: Option<usize>,
}

impl Inputs {
    pub fn parse(f: &mut FlagArgs) -> Result<Self> {
        let mut a = Self {
            text: None,
            text_file: None,
            image: None,
            video: None,
            audio: None,
            video_fps: 1.0,
            video_max_frames: 32,
            max_soft_tokens: None,
            video_soft_tokens: None,
            prompt: None,
            dim: None,
            normalize: true,
            tokens: None,
        };
        if let Some(v) = f.take("--text")? {
            a.text = Some(v);
        }
        if let Some(v) = f.take("--text-file")? {
            a.text_file = Some(PathBuf::from(v));
        }
        if let Some(v) = f.take("--image")? {
            a.image = Some(PathBuf::from(v));
        }
        if let Some(v) = f.take("--video")? {
            a.video = Some(PathBuf::from(v));
        }
        if let Some(v) = f.take("--audio")? {
            a.audio = Some(PathBuf::from(v));
        }
        if let Some(v) = f.take_parsed("--video-fps")? {
            a.video_fps = v;
        }
        if let Some(v) = f.take_parsed("--video-max-frames")? {
            a.video_max_frames = v;
        }
        if let Some(v) = f.take_parsed("--max-soft-tokens")? {
            a.max_soft_tokens = Some(v);
        }
        if let Some(v) = f.take_parsed("--video-soft-tokens")? {
            a.video_soft_tokens = Some(v);
        }
        if let Some(v) = f.take("--prompt")? {
            a.prompt = Some(v);
        }
        if let Some(v) = f.take_parsed("--dim")? {
            a.dim = Some(v);
        }
        a.normalize = !f.take_bool("--no-normalize");
        if let Some(v) = f.take_parsed("--tokens")? {
            a.tokens = Some(v);
        }
        Ok(a)
    }

    pub fn has_input(&self) -> bool {
        self.text.is_some()
            || self.text_file.is_some()
            || self.image.is_some()
            || self.video.is_some()
            || self.audio.is_some()
    }

    pub fn text(&self) -> Result<String> {
        input_text(&self.text, &self.text_file)
    }

    /// Reject flags the detected family cannot honour instead of ignoring them.
    pub fn validate(&self, family: Family) -> Result<()> {
        if !self.has_input() {
            bail!("provide --text/--text-file, or a media input (--image/--video/--audio)");
        }
        if family == Family::Qwen3Embed
            && (self.image.is_some()
                || self.video.is_some()
                || self.audio.is_some()
                || self.max_soft_tokens.is_some()
                || self.video_soft_tokens.is_some()
                || self.prompt.is_some()
                || self.dim.is_some())
        {
            bail!(
                "media inputs, --prompt and --dim are EmbeddingGemma 2 only \
                 (this checkpoint is qwen3)"
            );
        }
        Ok(())
    }
}

/// The `--dump-*` reference hooks (EmbeddingGemma 2).
#[derive(Debug, Default, Clone)]
pub struct DebugPaths {
    pub dump_pixels: Option<PathBuf>,
    pub dump_ids: Option<PathBuf>,
    pub dump_audio: Option<PathBuf>,
    pub dump_audio_layers: Option<PathBuf>,
    pub dump_image: Option<PathBuf>,
    pub dump_vision: Option<PathBuf>,
    pub dump_vision_layers: Option<PathBuf>,
}

impl DebugPaths {
    pub fn parse(f: &mut FlagArgs) -> Result<Self> {
        let mut d = Self::default();
        if let Some(v) = f.take("--dump-pixels")? {
            d.dump_pixels = Some(PathBuf::from(v));
        }
        if let Some(v) = f.take("--dump-ids")? {
            d.dump_ids = Some(PathBuf::from(v));
        }
        if let Some(v) = f.take("--dump-audio")? {
            d.dump_audio = Some(PathBuf::from(v));
        }
        if let Some(v) = f.take("--dump-audio-layers")? {
            d.dump_audio_layers = Some(PathBuf::from(v));
        }
        if let Some(v) = f.take("--dump-image")? {
            d.dump_image = Some(PathBuf::from(v));
        }
        if let Some(v) = f.take("--dump-vision")? {
            d.dump_vision = Some(PathBuf::from(v));
        }
        if let Some(v) = f.take("--dump-vision-layers")? {
            d.dump_vision_layers = Some(PathBuf::from(v));
        }
        Ok(d)
    }

    pub fn is_empty(&self) -> bool {
        self.dump_pixels.is_none()
            && self.dump_ids.is_none()
            && self.dump_audio.is_none()
            && self.dump_audio_layers.is_none()
            && self.dump_image.is_none()
            && self.dump_vision.is_none()
            && self.dump_vision_layers.is_none()
    }

    fn to_inputs(&self) -> inputs::DebugPaths {
        inputs::DebugPaths {
            dump_image: self.dump_image.clone(),
            dump_pixels: self.dump_pixels.clone(),
            dump_ids: self.dump_ids.clone(),
            dump_vision: self.dump_vision.clone(),
            dump_vision_layers: self.dump_vision_layers.clone(),
            dump_audio: self.dump_audio.clone(),
            dump_audio_layers: self.dump_audio_layers.clone(),
        }
    }
}

/// The EmbeddingGemma 2 NPU options for a resolved placement.
pub fn npu_opts(load: &ResolvedLoad) -> Option<Emb2NpuOpts> {
    load.placement.any_npu().then_some(Emb2NpuOpts {
        threads: load.npu_threads,
        text: load.placement.text,
        attn: load.placement.text_attn,
        vision: load.placement.vision,
        vision_attn: load.placement.vision_attn,
        int8: load.npu_dtype == NpuDtype::Int8,
        i8_group: load.int8_group,
    })
}

/// `--defer-towers`: a run that includes media must load the halves it uses
/// before input assembly; text-only runs never touch them.
pub fn maybe_load_towers(
    load: &ResolvedLoad,
    inputs: &Inputs,
    model: &mut crate::gemma::embeddinggemma::model::Emb2Model,
    device: &Device,
) -> Result<()> {
    let need = crate::gemma::TowerSet {
        vision: inputs.image.is_some() || inputs.video.is_some(),
        audio: inputs.audio.is_some(),
    };
    if load.defer_towers && !need.is_empty() {
        // The same precision policy as an eager load: `--quant q8` applies
        // unless `--npu-dtype int8` replaces it.
        crate::gemma::embeddinggemma::load::load_towers(
            model,
            &load.model_dir,
            load.dtype,
            load.quant_q8 && load.npu_dtype != NpuDtype::Int8,
            load.placement.vision,
            need,
            device,
        )?;
    }
    Ok(())
}

/// Assemble the input ids and media soft tokens through the shared processor.
pub fn build_inputs(
    load: &ResolvedLoad,
    inputs: &Inputs,
    debug: &DebugPaths,
    model: &crate::gemma::embeddinggemma::model::Emb2Model,
    device: &Device,
) -> Result<PreparedInput> {
    let tokenizer = crate::util::checkpoint::open_in_dir(&load.model_dir)?.tokenizer()?;
    let prefix = prompt_prefix(&load.model_dir, inputs.prompt.as_deref())?;
    let (img_soft_default, video_soft_default) = inputs::media_soft_tokens(&load.model_dir);
    let text = if inputs.text.is_some() || inputs.text_file.is_some() {
        Some(inputs.text()?)
    } else {
        None
    };
    let vision_attn = load.placement.vision_attn(load.vision_attn_kernel);
    let req = inputs::MediaInputs {
        text,
        images: inputs.image.clone().into_iter().collect(),
        video: inputs.video.clone(),
        audio: inputs.audio.clone(),
        prompt_prefix: prefix,
        max_tokens: inputs.tokens,
        image_soft_tokens: inputs.max_soft_tokens.unwrap_or(img_soft_default),
        video_soft_tokens: inputs.video_soft_tokens.unwrap_or(video_soft_default),
        video_fps: inputs.video_fps,
        video_max_frames: inputs.video_max_frames,
        attn_chunk: load.attn_chunk,
        vision_attn,
    };
    let prepared = inputs::prepare(model, &tokenizer, &req, &debug.to_inputs())?;
    let seq = prepared.ids.len();
    Ok((
        inputs::make_input(&prepared.ids, device),
        prepared.soft,
        seq,
    ))
}

/// Normalize (and MRL-truncate) the embedding and write it as JSON.
pub fn write_vector(
    out: Option<&Path>,
    normalize: bool,
    dim: Option<usize>,
    mut v: Vec<f32>,
) -> Result<()> {
    if normalize {
        if let Some(dim) = dim {
            v.truncate(dim);
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for x in &mut v {
            *x /= norm;
        }
    }
    let json = serde_json::to_string(&v)?;
    match out {
        Some(path) => {
            std::fs::write(path, &json).with_context(|| format!("write {}", path.display()))?;
            println!("wrote {}", path.display());
        }
        None => println!("{json}"),
    }
    Ok(())
}

/// Reset the vision timers and run input assembly, printing the vision line
/// when a tower forward happened.
pub fn prepare_measured(
    load: &ResolvedLoad,
    inputs: &Inputs,
    debug: &DebugPaths,
    model: &crate::gemma::embeddinggemma::model::Emb2Model,
    device: &Device,
) -> Result<(PreparedInput, f64)> {
    vision_stats_reset();
    let t0 = Instant::now();
    let prepared = build_inputs(load, inputs, debug, model, device)?;
    let encode_s = t0.elapsed().as_secs_f64();
    let vision = vision_stats();
    if !vision.is_empty() {
        println!("{}", vision.line());
    }
    Ok((prepared, encode_s))
}
