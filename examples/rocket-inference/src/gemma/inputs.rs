//! Shared input assembly for the CLI and the server: expands media placeholders,
//! encodes media with the towers and returns token ids plus the soft tokens to
//! scatter into the placeholder positions.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use burn::prelude::*;
use burn::tensor::{DType, Int, TensorData};
use tokenizers::Tokenizer;

use crate::gemma::audio_frontend::{self, AudioFeatures};
use crate::gemma::media::{self, PreparedImage};
use crate::gemma::vision::VisionSpec;

/// Soft-token budgets from `processor_config.json` (image and video processor
/// sections), matching the saved checkpoint configuration.
pub fn media_soft_tokens(model_dir: &Path) -> (usize, usize) {
    let mut image = 280;
    let mut video = 140;
    if let Ok(text) = std::fs::read_to_string(model_dir.join("processor_config.json"))
        && let Ok(json) = serde_json::from_str::<serde_json::Value>(&text)
    {
        if let Some(v) = json["image_processor"]["max_soft_tokens"].as_u64() {
            image = v as usize;
        }
        if let Some(v) = json["video_processor"]["max_soft_tokens"].as_u64() {
            video = v as usize;
        }
    }
    (image, video)
}

/// Task prompts from `config_sentence_transformers.json` (name -> prefix).
pub fn task_prompts(model_dir: &Path) -> HashMap<String, String> {
    let mut out = HashMap::new();
    if let Ok(text) = std::fs::read_to_string(model_dir.join("config_sentence_transformers.json"))
        && let Ok(json) = serde_json::from_str::<serde_json::Value>(&text)
        && let Some(prompts) = json["prompts"].as_object()
    {
        for (k, v) in prompts {
            out.insert(k.to_lowercase(), v.as_str().unwrap_or_default().to_string());
        }
    }
    out
}

/// A model that can encode media into text-space soft tokens: implemented by the
/// embedding model and by the generation model (both own the Gemma 4 towers).
pub trait MediaModel {
    fn vision_spec(&self) -> &VisionSpec;
    /// Image soft tokens `[num_soft_tokens, text_hidden]`; `debug_dir` dumps
    /// pooled features (and per-layer outputs when `layers` is set).
    fn encode_image(
        &self,
        img: &PreparedImage,
        chunk: usize,
        debug_dir: Option<&Path>,
        layers: Option<&Path>,
    ) -> Tensor<2>;
    /// Audio soft tokens (valid frames only).
    fn encode_audio(&self, feats: &AudioFeatures, debug_dir: Option<&Path>) -> Tensor<2>;
}

/// Everything needed to build one embedding input.
#[derive(Default, Clone)]
pub struct MediaInputs {
    pub text: Option<String>,
    pub image: Option<PathBuf>,
    pub video: Option<PathBuf>,
    pub audio: Option<PathBuf>,
    pub prompt_prefix: String,
    /// Truncate the token sequence (text-only inputs).
    pub max_tokens: Option<usize>,
    pub image_soft_tokens: usize,
    pub video_soft_tokens: usize,
    pub video_fps: f64,
    pub video_max_frames: usize,
    pub attn_chunk: usize,
}

/// Debug dumps (CLI only).
#[derive(Default, Clone)]
pub struct DebugPaths {
    pub dump_image: Option<PathBuf>,
    pub dump_pixels: Option<PathBuf>,
    pub dump_ids: Option<PathBuf>,
    pub dump_vision: Option<PathBuf>,
    pub dump_vision_layers: Option<PathBuf>,
    pub dump_audio: Option<PathBuf>,
    pub dump_audio_layers: Option<PathBuf>,
}

pub struct Prepared {
    pub ids: Vec<u32>,
    pub soft: Option<(Vec<usize>, Tensor<2>)>,
}

pub fn make_input(ids: &[u32], device: &Device) -> Tensor<2, Int> {
    let ids_i64: Vec<i64> = ids.iter().map(|&t| t as i64).collect();
    let n = ids.len();
    Tensor::<2, Int>::from_data(TensorData::new(ids_i64, [1, n]), device)
}

fn token_id(tokenizer: &Tokenizer, s: &str) -> Result<u32> {
    tokenizer
        .token_to_id(s)
        .with_context(|| format!("tokenizer has no id for '{s}'"))
}

fn dump_f32(path: &Path, values: &[f32]) -> Result<()> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, &bytes).with_context(|| format!("write {}", path.display()))
}

/// Build `(ids, soft tokens)` for the request: expands `<|image|>` / `<|video|>`
/// / `<|audio|>` placeholders to `BOI + token*n + EOI` (n = soft tokens from the
/// towers), tokenizes `prompt + text`, and encodes the media. Placeholders must
/// appear in the text in image-then-video-then-audio order (the default text
/// does).
pub fn prepare<M: MediaModel>(
    model: &M,
    tokenizer: &Tokenizer,
    req: &MediaInputs,
    debug: &DebugPaths,
) -> Result<Prepared> {
    let has_media = req.image.is_some() || req.video.is_some() || req.audio.is_some();
    let mut text = match &req.text {
        Some(t) => t.clone(),
        None if has_media => {
            let mut t = String::new();
            if req.image.is_some() {
                t.push_str("<|image|>");
            }
            if req.video.is_some() {
                t.push_str("<|video|>");
            }
            if req.audio.is_some() {
                t.push_str("<|audio|>");
            }
            t
        }
        None => bail!("provide text, image, video or audio"),
    };
    if has_media && req.max_tokens.is_some() {
        bail!("token truncation cannot be combined with media inputs");
    }

    let mut positions: Vec<usize> = Vec::new();
    let mut token_rows: Vec<Tensor<2>> = Vec::new();
    let mut n_image = 0usize;
    let mut n_video = 0usize;
    let mut n_audio = 0usize;

    if let Some(path) = &req.image {
        let vspec = model.vision_spec();
        let decoded = media::load_image(path)?;
        if let Some(dump) = &debug.dump_image {
            media::dump_rgb(&dump.with_extension("raw"), &decoded)?;
        }
        let resized = media::resize_rgb(
            &decoded,
            vspec.patch_size,
            req.image_soft_tokens,
            vspec.pooling,
        )?;
        if let Some(dump) = &debug.dump_image {
            media::dump_rgb(&dump.with_extension("resized.raw"), &resized)?;
        }
        let img = media::patchify(&resized, vspec.patch_size, vspec.pooling)?;
        n_image = img.num_soft_tokens();
        if let Some(path) = &debug.dump_pixels {
            dump_f32(path, &img.patches)?;
            println!(
                "dumped pixels: {} patches x {} ({}x{} grid) to {}",
                img.num_patches(),
                img.patches.len() / img.num_patches().max(1),
                img.patch_w,
                img.patch_h,
                path.display()
            );
        }
        text = text.replace(
            "<|image|>",
            &format!("<|image>{}<image|>", "<|image|>".repeat(n_image)),
        );
        println!(
            "image {}x{} patches -> {n_image} soft tokens",
            img.patch_w, img.patch_h
        );
        let tokens = model.encode_image(
            &img,
            req.attn_chunk,
            debug.dump_vision.as_deref(),
            debug.dump_vision_layers.as_deref(),
        );
        if let Some(path) = &debug.dump_vision {
            let values: Vec<f32> = tokens.clone().cast(DType::F32).into_data().try_to_vec()?;
            dump_f32(path, &values)?;
            println!(
                "dumped vision soft tokens [{}] to {}",
                values.len() / tokens.dims()[1],
                path.display()
            );
        }
        token_rows.push(tokens);
    }

    if let Some(path) = &req.video {
        let vspec = model.vision_spec();
        let info = media::video_info(path)?;
        let indices = media::sample_frame_indices(&info, req.video_fps, req.video_max_frames);
        let frames = media::extract_frames(path, &info, &indices)?;
        println!(
            "video: {:.2}s at {:.2} fps, {} frames -> sampled {} at {} fps",
            info.duration,
            info.fps,
            info.total_frames,
            frames.len(),
            req.video_fps
        );
        let mut frame_tokens = Vec::with_capacity(frames.len());
        let mut per_frame = 0usize;
        for frame in frames.iter() {
            let resized = media::resize_rgb(
                frame,
                vspec.patch_size,
                req.video_soft_tokens,
                vspec.pooling,
            )?;
            let img = media::patchify(&resized, vspec.patch_size, vspec.pooling)?;
            per_frame = img.num_soft_tokens();
            frame_tokens.push(model.encode_image(&img, req.attn_chunk, None, None));
        }
        n_video = per_frame * frames.len();
        println!("video: {per_frame} soft tokens per frame, {n_video} total");
        let frame_str = format!("<|image>{}<image|>", "<|video|>".repeat(per_frame));
        text = text.replace("<|video|>", &frame_str.repeat(frames.len()));
        token_rows.push(Tensor::cat(frame_tokens, 0));
    }

    if let Some(path) = &req.audio {
        let waveform = audio_frontend::load_waveform(path)?;
        let feats = audio_frontend::mel_features(&waveform);
        n_audio = feats.valid_soft_tokens();
        println!(
            "audio: {} samples, {} mel frames -> {n_audio} soft tokens",
            feats.samples, feats.frames
        );
        if let Some(dump) = &debug.dump_audio {
            dump_f32(dump, &feats.mel)?;
            let mut mb = Vec::with_capacity(feats.mask.len());
            for v in &feats.mask {
                mb.push(*v as u8);
            }
            std::fs::write(dump.with_extension("mask"), &mb)
                .with_context(|| format!("write {}", dump.display()))?;
            println!(
                "dumped audio features [{}x{}] to {}",
                feats.frames,
                audio_frontend::MEL_BINS,
                dump.display()
            );
        }
        text = text.replace(
            "<|audio|>",
            &format!("<|audio>{}<audio|>", "<|audio|>".repeat(n_audio)),
        );
        let tokens = model.encode_audio(&feats, debug.dump_audio_layers.as_deref());
        token_rows.push(tokens);
    }

    let mut ids = {
        let enc = tokenizer
            .encode(format!("{}{text}", req.prompt_prefix).as_str(), true)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
        enc.get_ids().to_vec()
    };
    if let Some(n) = req.max_tokens {
        ids.truncate(n);
    }
    if ids.is_empty() {
        bail!("empty input");
    }
    if let Some(path) = &debug.dump_ids {
        std::fs::write(path, serde_json::to_string(&ids)?)
            .with_context(|| format!("write {}", path.display()))?;
        println!("dumped {} input ids to {}", ids.len(), path.display());
    }

    let check = |n: usize, token: &str, what: &str| -> Result<Vec<usize>> {
        if n == 0 {
            return Ok(Vec::new());
        }
        let tok = token_id(tokenizer, token)?;
        let p: Vec<usize> = ids
            .iter()
            .enumerate()
            .filter(|(_, id)| **id == tok)
            .map(|(i, _)| i)
            .collect();
        if p.len() != n {
            bail!(
                "{what} placeholder mismatch: {} {what} tokens in the input, {n} soft tokens",
                p.len()
            );
        }
        Ok(p)
    };
    positions.extend(check(n_image, "<|image|>", "image")?);
    positions.extend(check(n_video, "<|video|>", "video")?);
    positions.extend(check(n_audio, "<|audio|>", "audio")?);

    let soft = if token_rows.is_empty() {
        None
    } else {
        Some((positions, Tensor::cat(token_rows, 0)))
    };
    Ok(Prepared { ids, soft })
}
