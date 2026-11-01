//! Model families and checkpoint detection.
//!
//! The CLI is family-agnostic: every model command detects the checkpoint's
//! family from `config.json` (`model_type`) or, for a GGUF directory, from
//! `general.architecture`, and `--family` overrides that detection.

use std::path::Path;

use anyhow::{Context, Result, bail};

/// The model family a checkpoint belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    Qwen3Embed,
    Emb2,
    Intent,
    Gemma4,
    /// vllm-sr Decision 3.0 (`d3-code-readout-v1`): a Qwen3.5 text backbone plus a
    /// 255-code readout head; text-only, no generation.
    Decision,
}

impl Family {
    /// Name used by `--family`, the `/v1/models` listing and the placement line.
    pub fn name(self) -> &'static str {
        match self {
            Self::Qwen3Embed => "qwen3",
            Self::Emb2 => "embeddinggemma",
            Self::Intent => "intent",
            Self::Gemma4 => "gemma4",
            Self::Decision => "decision",
        }
    }

    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "qwen3" => Ok(Self::Qwen3Embed),
            "embeddinggemma" => Ok(Self::Emb2),
            "intent" => Ok(Self::Intent),
            "gemma4" => Ok(Self::Gemma4),
            "decision" => Ok(Self::Decision),
            other => bail!(
                "unknown family '{other}' \
                 (expected auto|qwen3|embeddinggemma|intent|gemma4|decision)"
            ),
        }
    }

    /// Ollama's `general.architecture` for this family (chat families only).
    pub fn ollama_architecture(self) -> &'static str {
        match self {
            Self::Qwen3Embed => "qwen3",
            Self::Emb2 => "embedding_gemma2",
            Self::Intent => "qwen3_5",
            Self::Gemma4 => "gemma4",
            // d3 is a Qwen3.5 backbone; it has no chat surface of its own.
            Self::Decision => "qwen3_5",
        }
    }

    /// Models this command can embed with.
    pub fn is_embedding(self) -> bool {
        matches!(self, Self::Qwen3Embed | Self::Emb2)
    }

    /// Models this command can generate text with.
    pub fn is_chat(self) -> bool {
        matches!(self, Self::Intent | Self::Gemma4)
    }

    // ---- Stage support (the per-stage device flags) ----

    /// A separately switchable text-backbone attention stage (qwen3/emb2).
    /// Gemma 4 prefill attention follows the text projections; the intent
    /// model's attention always runs on the CPU inside its mixer.
    pub fn supports_text_attn(self) -> bool {
        matches!(self, Self::Qwen3Embed | Self::Emb2)
    }

    /// A vision tower that can be packed and run on the NPU (emb2 only).
    pub fn supports_vision(self) -> bool {
        matches!(self, Self::Emb2)
    }

    /// A single-token decode stage that can be placed independently.
    pub fn supports_decode(self) -> bool {
        matches!(self, Self::Intent | Self::Gemma4)
    }

    // ---- Precision / memory flags ----

    pub fn supports_dtype(self) -> bool {
        matches!(self, Self::Qwen3Embed | Self::Emb2 | Self::Gemma4)
    }

    pub fn supports_quant(self) -> bool {
        matches!(self, Self::Qwen3Embed | Self::Emb2 | Self::Gemma4)
    }

    /// Resident group-wise int8 NPU weights (emb2 text backbone).
    pub fn supports_int8(self) -> bool {
        matches!(self, Self::Emb2)
    }

    /// Keeping CPU copies of NPU-packed projections is meaningful for the two
    /// chat families (their decode can stay on the CPU).
    pub fn supports_keep_cpu_weights(self) -> bool {
        self.is_chat()
    }

    pub fn supports_defer_towers(self) -> bool {
        matches!(self, Self::Emb2 | Self::Gemma4)
    }

    pub fn supports_attn_kernel(self) -> bool {
        matches!(self, Self::Qwen3Embed | Self::Emb2)
    }

    /// The blocked-attention tile flags belong to the qwen3 CPU path.
    pub fn supports_attn_tiles(self) -> bool {
        matches!(self, Self::Qwen3Embed)
    }

    pub fn supports_vision_attn_kernel(self) -> bool {
        matches!(self, Self::Emb2)
    }

    /// The chunk width of the chunked (sliding/full) attention path.
    pub fn supports_attn_chunk(self) -> bool {
        matches!(self, Self::Emb2 | Self::Gemma4)
    }

    /// The f16 embedding/LM-head table (intent and decision keep a large tied table).
    pub fn supports_embed_f16(self) -> bool {
        matches!(self, Self::Intent | Self::Decision)
    }
}

/// Detect the model family from a checkpoint's `config.json` (`model_type`), or
/// from a GGUF file's `general.architecture`.
///
/// A `decision_config.json` wins over `config.json`: d3 checkpoints report the
/// backbone's `model_type` (`qwen3_5`), and the readout head is what makes them a
/// decision model.
pub fn detect_family(model_dir: &Path) -> Result<Family> {
    if model_dir.join("decision_config.json").exists() {
        return crate::decision::config::DecisionConfig::load(model_dir)
            .map(|_| Family::Decision)
            .with_context(|| format!("detect the model family in {}", model_dir.display()));
    }
    let path = model_dir.join("config.json");
    if path.exists() {
        let text =
            std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        let json: serde_json::Value =
            serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
        return detect_from_value(&json)
            .with_context(|| format!("detect the model family in {}", model_dir.display()));
    }
    // No config.json: assume a GGUF directory and read the architecture.
    let ck = crate::util::checkpoint::open_in_dir(model_dir)?;
    detect_from_arch(ck.arch()).with_context(|| {
        format!(
            "detect the model family in {} (GGUF architecture)",
            model_dir.display()
        )
    })
}

fn detect_from_arch(arch: Option<&str>) -> Result<Family> {
    match arch {
        Some("qwen3") => Ok(Family::Qwen3Embed),
        Some("gemma-embedding2") => Ok(Family::Emb2),
        Some("qwen35") => Ok(Family::Intent),
        Some("gemma4") => Ok(Family::Gemma4),
        Some(other) => bail!(
            "unsupported GGUF architecture '{other}' \
             (pass --family qwen3|embeddinggemma|intent|gemma4|decision)"
        ),
        None => bail!("checkpoint is neither config.json nor GGUF"),
    }
}

pub(crate) fn detect_from_value(json: &serde_json::Value) -> Result<Family> {
    match json.get("model_type").and_then(|v| v.as_str()) {
        Some("qwen3") => Ok(Family::Qwen3Embed),
        Some("embedding_gemma2") => Ok(Family::Emb2),
        Some("qwen3_5") => Ok(Family::Intent),
        Some("gemma4") => Ok(Family::Gemma4),
        Some(other) => bail!(
            "unsupported model_type '{other}' \
             (pass --family qwen3|embeddinggemma|intent|gemma4|decision)"
        ),
        None => bail!(
            "config.json has no model_type \
             (pass --family qwen3|embeddinggemma|intent|gemma4|decision)"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn family_names_round_trip() {
        for family in [
            Family::Qwen3Embed,
            Family::Emb2,
            Family::Intent,
            Family::Gemma4,
            Family::Decision,
        ] {
            assert_eq!(Family::parse(family.name()).unwrap(), family);
        }
        assert!(Family::parse("d3").is_err());
    }

    #[test]
    fn decision_is_text_only_with_no_optional_stages() {
        let d = Family::Decision;
        assert!(!d.is_embedding());
        assert!(!d.is_chat());
        assert!(d.supports_embed_f16());
        assert!(!d.supports_text_attn());
        assert!(!d.supports_vision());
        assert!(!d.supports_decode());
        assert!(!d.supports_dtype());
        assert!(!d.supports_quant());
        assert!(!d.supports_int8());
        assert!(!d.supports_keep_cpu_weights());
        assert!(!d.supports_defer_towers());
        assert!(!d.supports_attn_kernel());
        assert!(!d.supports_attn_tiles());
        assert!(!d.supports_vision_attn_kernel());
        assert!(!d.supports_attn_chunk());
        assert_eq!(d.ollama_architecture(), "qwen3_5");
    }

    /// A throwaway model directory (no `tempfile` dependency in this crate).
    fn temp_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "rocket-decision-family-{name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn decision_config(format_id: &str) -> String {
        serde_json::json!({
            "format_version": 1,
            "format_id": format_id,
            "prompt": "d3",
            "codes": (0..255).map(|i| format!("C{i}")).collect::<Vec<_>>(),
            "token_ids": (0..255).collect::<Vec<_>>(),
            "temperature": 1.0,
            "attention_mode": "noncausal_full_attention",
            "pooling": "last",
            "max_length": null,
            "readout_dtype": "float32",
        })
        .to_string()
    }

    #[test]
    fn decision_config_wins_over_the_backbone_model_type() {
        let dir = temp_dir("detect");
        std::fs::write(
            dir.join("config.json"),
            r#"{"model_type": "qwen3_5", "text_config": {}}"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("decision_config.json"),
            decision_config("d3-code-readout-v1"),
        )
        .unwrap();
        assert_eq!(detect_family(&dir).unwrap(), Family::Decision);
        // Without the decision config the same directory is a plain qwen3.5.
        std::fs::remove_file(dir.join("decision_config.json")).unwrap();
        assert_eq!(detect_family(&dir).unwrap(), Family::Intent);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_decision_config_with_the_wrong_format_is_refused() {
        let dir = temp_dir("bad-format");
        std::fs::write(
            dir.join("config.json"),
            r#"{"model_type": "qwen3_5", "text_config": {}}"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("decision_config.json"),
            decision_config("other-format"),
        )
        .unwrap();
        let err = format!("{:#}", detect_family(&dir).unwrap_err());
        assert!(err.contains("d3-code-readout-v1"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
