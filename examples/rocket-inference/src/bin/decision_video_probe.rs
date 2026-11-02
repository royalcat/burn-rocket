//! d3 video probe (Phase 3): the Qwen3-VL video processor, the d3 sampling and
//! the multi-frame vision tower against the official runtime's dump.
//!
//! Loads **only** `config.json` and the `visual.*` tensors of a d3 checkpoint (no
//! text backbone, no tokenizer, no readout), decodes the given videos exactly as
//! the app does (ffprobe/ffmpeg, 2 fps, 4-32 frames, temporal pairs), preprocesses
//! them as `Qwen3VLVideoProcessor` does and runs the tower in f32 (the weights are
//! read in the checkpoint's own dtype and widened per linear, exactly like the
//! app's loader). It writes the raw tensors `tools/vision_parity.py` scores:
//!
//! * `frames.{i}.raw` — every sampled frame of video `i`, as
//!   `u32 frames | u32 height | u32 width | rgb bytes` (the decoder's own RGB8,
//!   to compare against the reference's OpenCV decode);
//! * `pixel_values_videos.f32` — `[T*H*W, 1536]` f32, every video's patches in
//!   order (the processor's `pixel_values_videos`);
//! * `pooled_videos.f32` — `[tokens, out_hidden]` f32, the merged tokens of every
//!   video in order (the tower's `pooler_output` for videos);
//! * `meta.json` — per video the source/resized size, the `[T, H/16, W/16]` grid,
//!   its sampled source indices, fps, total frames, frame/pair/token counts and
//!   row offsets, plus the tower geometry and the wall time.
//!
//! The reference is `decision_reference.py --dump-vision DIR` on the same model
//! and videos; `tools/vision_parity.py --ref DIR --probe DIR --stem NAME` compares
//! them (frames exact, pixel `max|Δ| <= 1e-6`, pooled cosine `>= 0.999999`,
//! grid/token/index exact).
//!
//! Usage:
//!   decision_video_probe --model-dir DIR --video PATH [--video PATH ...] --out DIR

// The media/video spans (`#[telemetry::instrument]`) are no-ops here: the probe
// has no tracing subscriber.
mod server {
    pub mod telemetry {
        pub use tracing::instrument;
    }
}

#[path = "../gemma/media.rs"]
#[allow(dead_code)] // the image/audio helpers are not on the video path
pub mod media_impl;

/// `crate::gemma::media`, where the video decoder finds ffmpeg and the temp files.
mod gemma {
    pub use super::media_impl as media;
}

#[path = "../decision/vision/mod.rs"]
mod vision;

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use burn::prelude::*;
use burn::tensor::{DType, TensorData};
use burn_store::{ModuleStore, SafetensorsStore};
use serde_json::{Map, Value, json};
use vision::load::TensorSource;
use vision::processor::prepare_video;
use vision::{VisionConfig, VisionTower, video};

/// The value of a flag (`args[i]` is the flag itself).
fn flag_value(args: &[String], i: &mut usize, flag: &str) -> Result<String> {
    *i += 1;
    args.get(*i)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("{flag} needs a value"))
}

fn main() -> Result<()> {
    let mut model_dir: Option<PathBuf> = None;
    let mut out_dir: Option<PathBuf> = None;
    let mut videos: Vec<PathBuf> = Vec::new();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model-dir" => {
                model_dir = Some(PathBuf::from(flag_value(&args, &mut i, "--model-dir")?))
            }
            "--out" => out_dir = Some(PathBuf::from(flag_value(&args, &mut i, "--out")?)),
            "--video" => videos.push(PathBuf::from(flag_value(&args, &mut i, "--video")?)),
            other => bail!("unexpected argument '{other}' (see the module docs for the usage)"),
        }
        i += 1;
    }
    let model_dir = model_dir.ok_or_else(|| anyhow::anyhow!("--model-dir is required"))?;
    let out_dir = out_dir.ok_or_else(|| anyhow::anyhow!("--out is required"))?;
    if videos.is_empty() {
        bail!("at least one --video is required");
    }
    let device = Device::default();

    let config_path = model_dir.join("config.json");
    let config_text = std::fs::read_to_string(&config_path)
        .with_context(|| format!("read {}", config_path.display()))?;
    let config_json: Value = serde_json::from_str(&config_text)
        .with_context(|| format!("parse {}", config_path.display()))?;
    let config = VisionConfig::from_value(config_json)
        .with_context(|| format!("read the vision config of {}", model_dir.display()))?;

    let started = Instant::now();
    let mut source = Shard::open(&model_dir)?;
    let tower = VisionTower::load(&mut source, &config, &device)
        .with_context(|| format!("load the visual.* tower of {}", model_dir.display()))?;
    let load_s = started.elapsed().as_secs_f64();

    let decode_started = Instant::now();
    let decoded = videos
        .iter()
        .map(|path| {
            video::decode(path, false)
                .map_err(|error| anyhow::anyhow!("{error}"))
                .with_context(|| format!("decode {}", path.display()))
        })
        .collect::<Result<Vec<_>>>()?;
    let decode_s = decode_started.elapsed().as_secs_f64();

    std::fs::create_dir_all(&out_dir).with_context(|| format!("create {}", out_dir.display()))?;
    let encode_started = Instant::now();
    let mut patches = Vec::new();
    let mut pooled = Vec::new();
    let mut entries = Vec::with_capacity(decoded.len());
    let mut lines = Vec::with_capacity(decoded.len());
    let mut patch_offset = 0usize;
    let mut token_offset = 0usize;
    for (index, (path, video)) in videos.iter().zip(&decoded).enumerate() {
        dump_frames(&out_dir.join(format!("frames.{index}.raw")), video)?;
        let prepared = prepare_video(&video.frames, &config, &device)
            .with_context(|| format!("preprocess {}", path.display()))?;
        let features = tower
            .encode_video(&prepared)
            .with_context(|| format!("encode {}", path.display()))?;
        let [rows, _] = features.dims();
        if rows != prepared.tokens {
            bail!(
                "the tower produced {rows} rows for the {} tokens of {}",
                prepared.tokens,
                path.display()
            );
        }
        let stamps: Vec<String> = video
            .timestamps(config.temporal_patch_size)
            .iter()
            .map(|value| format!("{value:.1}"))
            .collect();
        entries.push(json!({
            "path": path.display().to_string(),
            "grid": prepared.grid,
            "patches": prepared.patches(),
            "tokens": prepared.tokens,
            "frame_tokens": prepared.frame_tokens,
            "frames": video.frame_count(),
            "fps": video.fps,
            "total_frames": video.total,
            "indices": video.indices,
            "source": [video.width, video.height],
            "resized": [prepared.resized.1, prepared.resized.0],
            "timestamps": stamps,
            "patch_offset": patch_offset,
            "token_offset": token_offset,
        }));
        lines.push(format!(
            "  {}: {}x{}, {} frames at {:.1} fps of {} -> {}x{}, grid [{}, {}, {}], \
             {} patches, {} tokens ({}/pair), timestamps {:?}",
            path.display(),
            video.width,
            video.height,
            video.frame_count(),
            video.fps,
            video.total,
            prepared.resized.1,
            prepared.resized.0,
            prepared.grid[0],
            prepared.grid[1],
            prepared.grid[2],
            prepared.patches(),
            prepared.tokens,
            prepared.frame_tokens,
            stamps
        ));
        patch_offset += prepared.patches();
        token_offset += prepared.tokens;
        patches.push(prepared.pixel_values);
        pooled.push(features);
    }
    let encode_s = encode_started.elapsed().as_secs_f64();

    let pixels = Tensor::cat(patches, 0);
    let rows = Tensor::cat(pooled, 0);
    let [patch_rows, patch_dim] = pixels.dims();
    let [token_rows, out_hidden] = rows.dims();
    write_f32(&out_dir.join("pixel_values_videos.f32"), &pixels)?;
    write_f32(&out_dir.join("pooled_videos.f32"), &rows)?;

    let mut geometry = Map::new();
    geometry.insert("depth".into(), json!(config.depth));
    geometry.insert("hidden_size".into(), json!(config.hidden_size));
    geometry.insert("num_heads".into(), json!(config.num_heads));
    geometry.insert("intermediate_size".into(), json!(config.intermediate_size));
    geometry.insert("out_hidden_size".into(), json!(config.out_hidden_size));
    geometry.insert("patch_size".into(), json!(config.patch_size));
    geometry.insert(
        "temporal_patch_size".into(),
        json!(config.temporal_patch_size),
    );
    geometry.insert(
        "spatial_merge_size".into(),
        json!(config.spatial_merge_size),
    );
    geometry.insert(
        "num_position_embeddings".into(),
        json!(config.num_position_embeddings),
    );
    geometry.insert("rope_theta".into(), json!(config.rope_theta));
    geometry.insert("hidden_act".into(), json!(config.hidden_act));
    let meta = json!({
        "schema": "d3-video-probe-v1",
        "model_dir": model_dir.display().to_string(),
        "dtype": "f32",
        "vision": Value::Object(geometry),
        "videos": entries,
        "pixel_shape": [patch_rows, patch_dim],
        "pooled_shape": [token_rows, out_hidden],
        "load_s": load_s,
        "decode_s": decode_s,
        "encode_s": encode_s,
    });
    let meta_path = out_dir.join("meta.json");
    std::fs::write(&meta_path, serde_json::to_string_pretty(&meta)? + "\n")
        .with_context(|| format!("write {}", meta_path.display()))?;

    println!(
        "video: {} clip(s), {} patches, {} tokens in {:.2}s (load {:.2}s, decode {:.2}s); \
         pixel_values_videos [{patch_rows}, {patch_dim}], pooled_videos [{token_rows}, {out_hidden}]",
        decoded.len(),
        patch_rows,
        token_rows,
        encode_s,
        load_s,
        decode_s
    );
    for line in &lines {
        println!("{line}");
    }
    println!("wrote {}", out_dir.display());
    Ok(())
}

/// `u32 frames | u32 height | u32 width | rgb bytes`, every sampled frame.
fn dump_frames(path: &Path, video: &video::DecodedVideo) -> Result<()> {
    let (height, width) = (video.height as u32, video.width as u32);
    let mut bytes =
        Vec::with_capacity(12 + video.frames.len() * height as usize * width as usize * 3);
    bytes.extend_from_slice(&(video.frames.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&height.to_le_bytes());
    bytes.extend_from_slice(&width.to_le_bytes());
    for frame in &video.frames {
        if frame.dimensions() != (width, height) {
            bail!(
                "a frame is {:?}, the video is {width}x{height}",
                frame.dimensions()
            );
        }
        bytes.extend_from_slice(frame.as_raw());
    }
    std::fs::write(path, bytes).with_context(|| format!("write {}", path.display()))
}

/// Write a 2-D f32 tensor as little-endian bytes.
fn write_f32(path: &Path, tensor: &Tensor<2>) -> Result<()> {
    let data = tensor.to_data();
    let values = data
        .as_slice::<f32>()
        .map_err(|e| anyhow::anyhow!("read a f32 tensor: {e}"))?;
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(path, bytes).with_context(|| format!("write {}", path.display()))
}

/// The checkpoint's `visual.*` tensors from its safetensors file.
struct Shard {
    store: SafetensorsStore,
}

impl Shard {
    fn open(model_dir: &Path) -> Result<Self> {
        let path = find_safetensors(model_dir)?;
        Ok(Self {
            store: SafetensorsStore::from_file(&path),
        })
    }
}

impl TensorSource for Shard {
    fn vision_tensor(&mut self, key: &str) -> Result<Option<(DType, TensorData)>> {
        let Some(tensor) = self.store.get_tensor(key)? else {
            return Ok(None);
        };
        let data = burn_store::bridge::to_data(tensor)?;
        // The stored dtype, unchanged (see `decision_vision_probe`).
        let dtype = data.dtype();
        Ok(Some((dtype, data)))
    }
}

/// `model.safetensors`, else the single `model-*.safetensors` shard.
fn find_safetensors(model_dir: &Path) -> Result<PathBuf> {
    let direct = model_dir.join("model.safetensors");
    if direct.is_file() {
        return Ok(direct);
    }
    let mut shards: Vec<PathBuf> = std::fs::read_dir(model_dir)
        .with_context(|| format!("read_dir {}", model_dir.display()))?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| {
            path.file_name().is_some_and(|name| {
                let name = name.to_string_lossy();
                name.starts_with("model-") && name.ends_with(".safetensors")
            })
        })
        .collect();
    shards.sort();
    match shards.len() {
        0 => bail!(
            "no model.safetensors or model-*.safetensors in {}",
            model_dir.display()
        ),
        1 => Ok(shards.remove(0)),
        n => bail!(
            "{} holds {n} safetensors shards; merge them into one file first",
            model_dir.display()
        ),
    }
}
