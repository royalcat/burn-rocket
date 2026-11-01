//! Loading a d3 decision checkpoint: the Qwen3.5 text backbone (reused from the
//! intent family), the 255-code readout head and the tokenizer.

use std::path::Path;

use anyhow::{Context, Result, bail};
use burn::prelude::*;
use burn::tensor::{DType, TensorData};
use tokenizers::Tokenizer;

use super::config::{self, DecisionConfig};
use super::model::DecisionModel;
use super::vision::TensorSource;
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

/// The vision tower's `visual.*` tensors, f32, from a d3 checkpoint.
///
/// `super::vision::VisionTower::load` only needs these reads, so the tower stays
/// independent of the app's checkpoint abstraction (the standalone vision probe
/// reads the same keys from a safetensors file directly).
impl TensorSource for AnyCheckpoint {
    fn vision_tensor(&mut self, key: &str) -> Result<Option<TensorData>> {
        self.get(key, DType::F32)
    }
}

/// Load a d3 checkpoint directory.
///
/// `opts` carries the placement (`npu`/`text`/`keep_cpu_weights`/`embed_f16`);
/// the prefix, the label and the rotary-table cap are set here, the full-attention
/// layers are switched to the checkpoint's bidirectional mode, and the placement
/// invariants are applied exactly as the intent server applies them.
pub fn load_decision_model(
    model_dir: &Path,
    device: &Device,
    opts: &IntentLoadOptions,
    max_tokens: usize,
) -> Result<DecisionModel> {
    let config = DecisionConfig::load(model_dir)?;
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
    Ok(DecisionModel {
        backbone,
        readout,
        config,
        tokenizer,
    })
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
