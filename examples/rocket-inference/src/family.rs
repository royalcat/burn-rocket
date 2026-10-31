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
}

impl Family {
    /// Name used by `--family`, the `/v1/models` listing and the placement line.
    pub fn name(self) -> &'static str {
        match self {
            Self::Qwen3Embed => "qwen3",
            Self::Emb2 => "embeddinggemma",
            Self::Intent => "intent",
            Self::Gemma4 => "gemma4",
        }
    }

    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "qwen3" => Ok(Self::Qwen3Embed),
            "embeddinggemma" => Ok(Self::Emb2),
            "intent" => Ok(Self::Intent),
            "gemma4" => Ok(Self::Gemma4),
            other => {
                bail!("unknown family '{other}' (expected auto|qwen3|embeddinggemma|intent|gemma4)")
            }
        }
    }

    /// Ollama's `general.architecture` for this family (chat families only).
    pub fn ollama_architecture(self) -> &'static str {
        match self {
            Self::Qwen3Embed => "qwen3",
            Self::Emb2 => "embedding_gemma2",
            Self::Intent => "qwen3_5",
            Self::Gemma4 => "gemma4",
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
        !self.is_chat() || self == Self::Gemma4
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

    /// The f16 embedding/LM-head table (intent keeps a large tied table).
    pub fn supports_embed_f16(self) -> bool {
        matches!(self, Self::Intent)
    }
}

/// Detect the model family from a checkpoint's `config.json` (`model_type`), or
/// from a GGUF file's `general.architecture`.
pub fn detect_family(model_dir: &Path) -> Result<Family> {
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
             (pass --family qwen3|embeddinggemma|intent|gemma4)"
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
             (pass --family qwen3|embeddinggemma|intent|gemma4)"
        ),
        None => bail!(
            "config.json has no model_type (pass --family qwen3|embeddinggemma|intent|gemma4)"
        ),
    }
}
