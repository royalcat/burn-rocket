//! Loading a d3 decision checkpoint: the Qwen3.5 text backbone (reused from the
//! intent family), the 255-code readout head and the tokenizer.

use std::path::Path;

use anyhow::{Context, Result, bail};
use burn::prelude::*;
use burn::tensor::{DType, TensorData};
use tokenizers::Tokenizer;

use super::config::{self, DecisionConfig};
use super::format;
use super::model::DecisionModel;
use super::vision::{TensorSource, load_tower};
use crate::qwen35_intent::loader::{IntentLoadOptions, load_intent_model};
use crate::qwen35_intent::model::IntentModel;
use crate::util::checkpoint::{self, AnyCheckpoint, Checkpoint};

/// The text-backbone key prefix of a d3 checkpoint (`language_model.layers.0...`,
/// `language_model.embed_tokens.weight`).
pub const DECISION_PREFIX: &str = "language_model.";

/// The model name the package declares (`MODEL_MANIFEST.json` `model_name`), if any.
pub fn manifest_model_name(model_dir: &Path) -> Option<String> {
    let text = std::fs::read_to_string(model_dir.join("MODEL_MANIFEST.json")).ok()?;
    let manifest: serde_json::Value = serde_json::from_str(&text).ok()?;
    manifest
        .get("model_name")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
}

/// The vision tower's `visual.*` tensors, in the dtype the checkpoint stores.
///
/// `super::vision::VisionTower::load` only needs these reads, so the tower stays
/// independent of the app's checkpoint abstraction (the standalone vision probes
/// read the same keys from a safetensors file directly). The dtype is passed
/// through instead of widening everything to f32: a d3 package stores bf16 and the
/// tower keeps its matmul weights bf16 (half the resident bytes, exact).
impl TensorSource for AnyCheckpoint {
    fn vision_tensor(&mut self, key: &str) -> Result<Option<(DType, TensorData)>> {
        match self {
            AnyCheckpoint::Safetensors(checkpoint) => {
                Ok(checkpoint.get_stored(key)?.map(|data| (data.dtype(), data)))
            }
            // The decision loader refuses a GGUF checkpoint before the tower is
            // read; a GGUF dequantizes to its native float dtype anyway.
            AnyCheckpoint::Gguf(checkpoint) => {
                let dtype = checkpoint.native_dtype();
                Ok(checkpoint.get(key, dtype)?.map(|data| (data.dtype(), data)))
            }
        }
    }
}

/// Load a d3 checkpoint directory.
///
/// `opts` carries the placement (`npu`/`text`/`keep_cpu_weights`/`embed_f16`);
/// the prefix, the label and the rotary-table cap are set here, the full-attention
/// layers are switched to the checkpoint's bidirectional mode, and the placement
/// invariants are applied exactly as the intent server applies them. An
/// image-capable package (`config.json` `vision_config`) also has its `visual.*`
/// vision tower — loaded here, unless `defer_towers` leaves it to the first media
/// request (`DecisionModel::ensure_vision`).
pub fn load_decision_model(
    model_dir: &Path,
    device: &Device,
    opts: &IntentLoadOptions,
    max_tokens: usize,
    defer_towers: bool,
) -> Result<DecisionModel> {
    let config = DecisionConfig::load(model_dir)?;
    let config_json = read_config_json(model_dir)?;
    let mut ckpt = checkpoint::open_in_dir(model_dir)
        .with_context(|| format!("open the checkpoint in {}", model_dir.display()))?;
    if ckpt.format() == "gguf" {
        bail!(
            "the decision family needs a safetensors checkpoint: a d3 package stores its \
             readout head in readout.safetensors next to the backbone, which a GGUF file \
             cannot carry"
        );
    }
    // The answer-code list is part of the contract (readout row order): recompute
    // it from the tokenizer before spending time on the weights.
    let tokenizer: Tokenizer = ckpt.tokenizer()?;
    check_codes(&tokenizer, &config)?;
    let (image_token_id, video_token_id) = check_media_tokens(&tokenizer, &config_json)?;
    let opts = IntentLoadOptions {
        prefix: DECISION_PREFIX,
        label: "decision",
        rope_max_seq: Some(max_tokens),
        ..opts.clone()
    };
    let mut backbone: IntentModel = load_intent_model(&mut ckpt, device, &opts)?;
    // `attention_mode: noncausal_full_attention`: the full-attention layers read
    // the whole prompt; the Gated DeltaNet layers keep their causal scan.
    backbone.full_attn_causal = false;
    apply_placement(&mut backbone, &opts);
    let hidden = backbone.cfg.hidden_size;
    let readout = config::load_readout(model_dir, hidden, device)?;
    // `vision_config` makes the package image-capable (`visual.*` beside the
    // backbone); a text-only package has no tower and refuses images.
    let vision_declared = config_json.get("vision_config").is_some();
    let vision = if vision_declared && !defer_towers {
        Some(
            load_tower(&mut ckpt, &config_json, device)
                .context("load the vision tower of the d3 checkpoint")?,
        )
    } else {
        None
    };
    // Both branches (and a text-only package) leave the checkpoint's per-tensor
    // copies and freed pages in glibc's arenas: return them, or the tower's
    // 16-bit storage — and, without a tower, the body load's own churn — stays
    // resident behind the freed pages.
    crate::util::trim_heap();
    // The intent loader above already printed `<label> model ready in ...`; this
    // line only reports what the vision tower did.
    match (&vision, vision_declared) {
        (Some(tower), _) => println!(
            "decision: vision tower {} blocks ({} weights)",
            tower.config.depth,
            storage_dtype(ckpt.native_dtype())
        ),
        (None, true) => println!("decision: vision tower deferred to the first media request"),
        (None, false) => println!("decision: text-only package, no vision tower"),
    }
    Ok(DecisionModel {
        backbone,
        readout,
        config,
        tokenizer,
        vision,
        model_dir: model_dir.to_path_buf(),
        defer_towers,
        vision_declared,
        image_token_id,
        video_token_id,
    })
}

/// `<dir>/config.json` as a value (the checkpoint's own config, beside
/// `decision_config.json`).
pub(crate) fn read_config_json(model_dir: &Path) -> Result<serde_json::Value> {
    let path = model_dir.join("config.json");
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))
}

/// A checkpoint's float dtype as the load-report label (`bf16`/`f16`/`f32`).
///
/// The vision tower keeps its matmul weights in this dtype when it is 16-bit.
pub(crate) fn storage_dtype(dtype: DType) -> &'static str {
    match dtype {
        DType::BF16 => "bf16",
        DType::F16 => "f16",
        _ => "f32",
    }
}

/// `config.json` `image_token_id`/`video_token_id`, checked against the
/// tokenizer's `<|image_pad|>`/`<|video_pad|>` tokens: the prompt expansion counts
/// those ids, so a mismatch would splice the wrong rows.
///
/// Only an **image-capable** package (`vision_config`) has to declare them: a
/// text-only d3 package (no tower, no placeholders) loads without them and refuses
/// an image request with [`crate::decision::model::NO_VISION_TOWER`] instead. Both ids come back as
/// `u32::MAX` (never a real token id) when the package has no vision.
fn check_media_tokens(tokenizer: &Tokenizer, config: &serde_json::Value) -> Result<(u32, u32)> {
    let id = |name: &str| -> Option<u32> {
        config
            .get(name)
            .and_then(serde_json::Value::as_u64)
            .map(|v| v as u32)
    };
    if config.get("vision_config").is_none() {
        // Text-only: nothing to expand, nothing to check.
        return Ok((u32::MAX, u32::MAX));
    }
    let image_token_id = id("image_token_id").ok_or_else(|| {
        anyhow::anyhow!(
            "config.json declares a vision_config but no image_token_id; the d3 image \
             path needs the checkpoint's own <|image_pad|> id"
        )
    })?;
    let video_token_id = id("video_token_id").unwrap_or(u32::MAX);
    if tokenizer.token_to_id("<|vision_start|>").is_none() {
        bail!("the tokenizer has no <|vision_start|> token; this is not a d3 image checkpoint");
    }
    for (name, token, expected) in [
        ("image", format::IMAGE_TOKEN, Some(image_token_id)),
        (
            "video",
            format::VIDEO_TOKEN,
            (video_token_id != u32::MAX).then_some(video_token_id),
        ),
    ] {
        let Some(expected) = expected else {
            continue;
        };
        let found = tokenizer.token_to_id(token);
        if found != Some(expected) {
            bail!(
                "the tokenizer maps '{token}' to {found:?}, but config.json declares \
                 {name}_token_id {expected}"
            );
        }
    }
    Ok((image_token_id, video_token_id))
}

/// The placement invariants `load_for_serving` applies for the intent family: a
/// pack-and-drop load has no CPU projection copies, so every projection must go
/// through the NPU; `--decode-device npu` is the explicit single-token choice.
fn apply_placement(backbone: &mut IntentModel, opts: &IntentLoadOptions) {
    backbone.npu_only = opts.text && !opts.keep_cpu_weights;
    backbone.npu_decode = opts.decode;
}

/// The config's codes and token ids must be the canonical list this tokenizer
/// produces (`d3_format.answer_codes`), and every code must still be one token of
/// its own.
fn check_codes(tokenizer: &Tokenizer, config: &DecisionConfig) -> Result<()> {
    let (codes, token_ids) = config::answer_codes(tokenizer)?;
    if config.codes != codes || config.token_ids != token_ids {
        bail!(
            "the answer codes or token ids in decision_config.json differ from the ones \
             this tokenizer gives"
        );
    }
    for (code, expected) in config.codes.iter().zip(&config.token_ids) {
        let encoding = tokenizer
            .encode(code.as_str(), false)
            .map_err(|e| anyhow::anyhow!("tokenize the answer code '{code}': {e}"))?;
        let ids = encoding.get_ids();
        if ids != [*expected] {
            bail!(
                "the answer code '{code}' tokenizes to {ids:?}, but decision_config.json \
                 records token id {expected}; the checkpoint's tokenizer does not match its codes"
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen35_intent::model::IntentTextConfig;

    fn tiny_config() -> IntentTextConfig {
        IntentTextConfig::from_value(serde_json::json!({
            "text_config": {
                "vocab_size": 64,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 16,
                "rms_norm_eps": 1e-6,
                "max_position_embeddings": 64,
                "layer_types": ["full_attention"],
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 8,
                "linear_value_head_dim": 8,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 2,
                "partial_rotary_factor": 0.5,
                "rope_parameters": {"rope_theta": 10000.0},
            }
        }))
        .unwrap()
    }

    #[test]
    fn a_pack_and_drop_placement_has_no_cpu_copies() {
        let device = Device::default();
        let mut backbone = IntentModel::new_stub(&tiny_config(), 64, &device);
        assert!(!backbone.npu_only);
        apply_placement(
            &mut backbone,
            &IntentLoadOptions {
                npu: true,
                text: true,
                keep_cpu_weights: false,
                ..IntentLoadOptions::default()
            },
        );
        assert!(
            backbone.npu_only,
            "npu + keep_cpu_weights=false must force every projection through the NPU"
        );
        assert!(!backbone.npu_decode);

        // Keeping the CPU copies (the default) leaves the mixed path intact.
        apply_placement(
            &mut backbone,
            &IntentLoadOptions {
                npu: true,
                text: true,
                keep_cpu_weights: true,
                ..IntentLoadOptions::default()
            },
        );
        assert!(!backbone.npu_only);

        // `--decode-device npu` reaches the decode flag.
        apply_placement(
            &mut backbone,
            &IntentLoadOptions {
                npu: true,
                text: true,
                decode: true,
                keep_cpu_weights: true,
                ..IntentLoadOptions::default()
            },
        );
        assert!(backbone.npu_decode);
    }

    #[test]
    fn the_manifest_declares_the_served_model_name() {
        let source = Path::new("/tmp/opencode/d3/models/d3-lite");
        if source.join("MODEL_MANIFEST.json").exists() {
            assert_eq!(manifest_model_name(source).as_deref(), Some("d3-lite"));
        }
        let dir = std::env::temp_dir().join(format!("rocket-decision-name-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(manifest_model_name(&dir), None);
        std::fs::write(
            dir.join("MODEL_MANIFEST.json"),
            r#"{"model_name": "from-manifest"}"#,
        )
        .unwrap();
        assert_eq!(manifest_model_name(&dir).as_deref(), Some("from-manifest"));
        // A broken manifest is ignored rather than fatal.
        std::fs::write(dir.join("MODEL_MANIFEST.json"), "{not json").unwrap();
        assert_eq!(manifest_model_name(&dir), None);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The load-time media-token checks only apply to an image-capable package: a
    /// text-only d3 package (no `vision_config`) has no placeholders to expand and
    /// must still load, while one that *does* declare a tower must name both ids
    /// and keep them consistent with the tokenizer.
    #[test]
    fn media_tokens_are_required_only_for_an_image_capable_package() {
        let source = Path::new("/tmp/opencode/d3/models/d3-lite/tokenizer.json");
        let Ok(tokenizer) = Tokenizer::from_file(source) else {
            eprintln!("skipping: {} is not available", source.display());
            return;
        };
        let text_only = serde_json::json!({"text_config": {"model_type": "qwen3_5_text"}});
        assert_eq!(
            check_media_tokens(&tokenizer, &text_only).unwrap(),
            (u32::MAX, u32::MAX)
        );

        // An image package must declare the image id ...
        let no_id = serde_json::json!({"vision_config": {"depth": 12}});
        let message = format!("{:#}", check_media_tokens(&tokenizer, &no_id).unwrap_err());
        assert!(message.contains("no image_token_id"), "{message}");

        // ... and it must match the tokenizer's own <|image_pad|>.
        let real = crate::decision::format::IMAGE_TOKEN;
        let actual = tokenizer.token_to_id(real).expect("<|image_pad|>");
        let good = serde_json::json!({
            "vision_config": {"depth": 12},
            "image_token_id": actual,
            "video_token_id": actual + 1,
        });
        assert_eq!(
            check_media_tokens(&tokenizer, &good).unwrap(),
            (actual, actual + 1)
        );

        let wrong = serde_json::json!({
            "vision_config": {"depth": 12},
            "image_token_id": actual + 3,
        });
        let message = format!("{:#}", check_media_tokens(&tokenizer, &wrong).unwrap_err());
        assert!(message.contains("image_token_id"), "{message}");
    }

    /// A d3 checkpoint whose `decision_config.json` lists the codes in another
    /// order must fail to load before any weight is read.
    #[cfg(unix)]
    #[test]
    fn shuffled_codes_are_refused() {
        let source = Path::new("/tmp/opencode/d3/models/d3-lite");
        if !source.join("decision_config.json").exists() {
            eprintln!("skipping: {} is not available", source.display());
            return;
        }
        let dir =
            std::env::temp_dir().join(format!("rocket-decision-codes-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for name in [
            "config.json",
            "tokenizer.json",
            "readout.safetensors",
            "model-00001-of-00001.safetensors",
        ] {
            std::os::unix::fs::symlink(source.join(name), dir.join(name)).unwrap();
        }
        let mut config: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(source.join("decision_config.json")).unwrap(),
        )
        .unwrap();
        let codes = config["codes"].as_array_mut().unwrap();
        codes.swap(0, 1);
        std::fs::write(
            dir.join("decision_config.json"),
            serde_json::to_string(&config).unwrap(),
        )
        .unwrap();

        let err = load_decision_model(
            &dir,
            &Device::default(),
            &IntentLoadOptions::default(),
            8192,
            false,
        )
        .unwrap_err();
        let message = format!("{err:#}");
        assert!(
            message.contains("differ from the ones this tokenizer gives"),
            "{message}"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
