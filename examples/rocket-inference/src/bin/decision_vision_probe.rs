//! d3 vision probe (Phase 2a): the Qwen3.5 vision tower and the Qwen2-VL image
//! preprocessing against the official runtime's dump.
//!
//! Loads **only** `config.json` and the `visual.*` tensors of a d3 checkpoint
//! (no text backbone, no tokenizer, no readout), preprocesses one or more images
//! exactly as `Qwen2VLImageProcessorFast` does, runs the tower in f32 (the weights
//! are read in the checkpoint's own dtype and widened per linear, exactly like the
//! app's loader), and writes the raw tensors `tools/vision_parity.py` scores:
//!
//! * `pixel_values.f32` — `[patches, 1536]` f32, patches of every `--image` in
//!   order (the processor's `pixel_values`);
//! * `pooled.f32` — `[tokens, out_hidden]` f32, the merged tokens of every image
//!   in order (the tower's `pooler_output`);
//! * `meta.json` — per image the source/resized size, the `[1, H/16, W/16]` grid,
//!   its patch and token counts and where its rows start, plus the tower geometry
//!   and the wall time.
//!
//! The reference is `decision_reference.py --dump-vision DIR` on the same model
//! and image; `tools/vision_parity.py --ref DIR --probe DIR --stem NAME` compares
//! them (pixel `max|Δ| <= 1e-6`, pooled cosine `>= 0.9999`, grid/token exact).
//!
//! Usage:
//!   decision_vision_probe --model-dir DIR --image PATH [--image PATH ...] \
//!                         --out DIR [--threads N]
//!
//! The probe includes the app's vision sources by path (`#[path]`), so it runs
//! without the server/CLI stack; the image decoder is the app's own
//! (`gemma::media::load_image`, mozjpeg for JPEGs like PIL) so a parity run
//! compares the tower and the processor, not two decoders.

// The image loader's spans (`#[telemetry::instrument]`) are no-ops here: the probe
// has no tracing subscriber, and `resize_bicubic_aa`/`load_image` are unchanged.
mod server {
    pub mod telemetry {
        pub use tracing::instrument;
    }
}

#[path = "../gemma/media.rs"]
#[allow(dead_code)] // the video/audio helpers are not on the vision path
pub mod media_impl;

/// `crate::gemma::media`, where the vision processor finds the image loader.
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
use vision::{VisionConfig, VisionTower, prepare_image};

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
    let mut images: Vec<PathBuf> = Vec::new();
    let mut threads: Option<usize> = None;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model-dir" => {
                model_dir = Some(PathBuf::from(flag_value(&args, &mut i, "--model-dir")?))
            }
            "--out" => out_dir = Some(PathBuf::from(flag_value(&args, &mut i, "--out")?)),
            "--image" => images.push(PathBuf::from(flag_value(&args, &mut i, "--image")?)),
            "--threads" => {
                threads = Some(
                    flag_value(&args, &mut i, "--threads")?
                        .parse()
                        .context("--threads N")?,
                )
            }
            other => bail!("unexpected argument '{other}' (see the module docs for the usage)"),
        }
        i += 1;
    }
    let model_dir = model_dir.ok_or_else(|| anyhow::anyhow!("--model-dir is required"))?;
    let out_dir = out_dir.ok_or_else(|| anyhow::anyhow!("--out is required"))?;
    if images.is_empty() {
        bail!("at least one --image is required");
    }
    if let Some(threads) = threads {
        // The flex backend parallelizes on rayon, not torch-style thread counts;
        // keep the flag honest by reporting it rather than silently ignoring it.
        println!("note: --threads {threads} is a no-op for the flex backend (rayon decides)");
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

    let encode_started = Instant::now();
    let prepared = images
        .iter()
        .map(|path| prepare_image(path, &config, &device))
        .collect::<Result<Vec<_>>>()?;
    let pooled = tower.encode_all(&prepared)?;
    let encode_s = encode_started.elapsed().as_secs_f64();

    std::fs::create_dir_all(&out_dir).with_context(|| format!("create {}", out_dir.display()))?;
    let pixels = Tensor::cat(prepared.iter().map(|p| p.pixel_values.clone()).collect(), 0);
    let [patch_rows, patch_dim] = pixels.dims();
    let [token_rows, out_hidden] = pooled.dims();
    write_f32(&out_dir.join("pixel_values.f32"), &pixels)?;
    write_f32(&out_dir.join("pooled.f32"), &pooled)?;

    let mut entries = Vec::with_capacity(prepared.len());
    let mut lines = Vec::with_capacity(prepared.len());
    let mut patch_offset = 0usize;
    let mut token_offset = 0usize;
    for (path, image) in images.iter().zip(&prepared) {
        let grid = image.grid;
        entries.push(json!({
            "path": path.display().to_string(),
            "source": [image.source.0, image.source.1],
            "resized": [image.resized.0, image.resized.1],
            "grid": grid,
            "patches": image.patches(),
            "tokens": image.tokens,
            "patch_offset": patch_offset,
            "token_offset": token_offset,
        }));
        lines.push(format!(
            "  {}: {}x{} -> {}x{}, grid [{}, {}, {}], {} patches, {} tokens",
            path.display(),
            image.source.1,
            image.source.0,
            image.resized.1,
            image.resized.0,
            grid[0],
            grid[1],
            grid[2],
            image.patches(),
            image.tokens
        ));
        patch_offset += image.patches();
        token_offset += image.tokens;
    }
    let mut vision = Map::new();
    vision.insert("depth".into(), json!(config.depth));
    vision.insert("hidden_size".into(), json!(config.hidden_size));
    vision.insert("num_heads".into(), json!(config.num_heads));
    vision.insert("intermediate_size".into(), json!(config.intermediate_size));
    vision.insert("out_hidden_size".into(), json!(config.out_hidden_size));
    vision.insert("patch_size".into(), json!(config.patch_size));
    vision.insert(
        "temporal_patch_size".into(),
        json!(config.temporal_patch_size),
    );
    vision.insert(
        "spatial_merge_size".into(),
        json!(config.spatial_merge_size),
    );
    vision.insert(
        "num_position_embeddings".into(),
        json!(config.num_position_embeddings),
    );
    vision.insert("rope_theta".into(), json!(config.rope_theta));
    vision.insert("hidden_act".into(), json!(config.hidden_act));
    let meta = json!({
        "schema": "d3-vision-probe-v1",
        "model_dir": model_dir.display().to_string(),
        "dtype": "f32",
        "vision": Value::Object(vision),
        "images": entries,
        "pixel_shape": [patch_rows, patch_dim],
        "pooled_shape": [token_rows, out_hidden],
        "load_s": load_s,
        "encode_s": encode_s,
    });
    let meta_path = out_dir.join("meta.json");
    std::fs::write(&meta_path, serde_json::to_string_pretty(&meta)? + "\n")
        .with_context(|| format!("write {}", meta_path.display()))?;

    println!(
        "vision: {} image(s), {} patches, {} tokens in {:.2}s (load {:.2}s); \
         pixel_values [{patch_rows}, {patch_dim}], pooled [{token_rows}, {out_hidden}]",
        prepared.len(),
        patch_rows,
        token_rows,
        encode_s,
        load_s
    );
    for line in &lines {
        println!("{line}");
    }
    println!("wrote {}", out_dir.display());
    Ok(())
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
    std::fs::write(path, bytes).with_context(|| format!("write {}", path.display()))?;
    Ok(())
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
        // The stored dtype, unchanged: the tower keeps a bf16 weight bf16 and
        // widens it per linear (exact), exactly like the app's loader.
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
