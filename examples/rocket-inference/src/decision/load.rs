//! Loading a d3 decision checkpoint: the Qwen3.5 text backbone (reused from the
//! intent family), the 255-code readout head and the tokenizer.

use std::path::Path;

use anyhow::{Context, Result, bail};
use burn::prelude::*;
use tokenizers::Tokenizer;

use super::config::DecisionConfig;
use super::model::DecisionModel;
use crate::qwen35_intent::loader::{IntentLoadOptions, load_intent_model};
use crate::qwen35_intent::model::IntentModel;
use crate::util::checkpoint::{self, Checkpoint};

/// The text-backbone key prefix of a d3 checkpoint (`language_model.layers.0...`,
/// `language_model.embed_tokens.weight`).
pub const DECISION_PREFIX: &str = "language_model.";

/// Load a d3 checkpoint directory.
///
/// `opts` carries the placement (`npu`/`text`); the prefix, the f16 token table
/// and the labeled ready line are set here, and the full-attention layers are
/// switched to the checkpoint's bidirectional mode. `max_tokens` sizes the rotary
/// tables: d3 prompts are never longer than the request cap.
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
    let opts = IntentLoadOptions {
        prefix: DECISION_PREFIX,
        label: "decision",
        rope_max_seq: Some(max_tokens),
        embed_f16: true,
        ..opts.clone()
    };
    let mut backbone: IntentModel = load_intent_model(&mut ckpt, device, &opts)?;
    // `attention_mode: noncausal_full_attention`: the full-attention layers read
    // the whole prompt; the Gated DeltaNet layers keep their causal scan.
    backbone.full_attn_causal = false;
    let hidden = backbone.cfg.hidden_size;
    let readout = super::config::load_readout(model_dir, hidden, device)?;
    let tokenizer: Tokenizer = ckpt.tokenizer()?;
    check_answer_codes(&tokenizer, &config)?;
    Ok(DecisionModel {
        backbone,
        readout,
        config,
        tokenizer,
    })
}

/// The readout scores code tokens, so every code must be one token of its own and
/// that token must be the id the checkpoint recorded. A tokenizer/checkpoint
/// mismatch would silently answer with the wrong options.
fn check_answer_codes(tokenizer: &Tokenizer, config: &DecisionConfig) -> Result<()> {
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
