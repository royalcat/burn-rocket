//! Build a `tokenizers::Tokenizer` from GGUF metadata.
//!
//! GGUF carries the llama.cpp vocabulary, not a `tokenizers`-format file. Two
//! shapes appear:
//!
//! * **BPE** (`tokenizer.ggml.model = "gpt2"`, e.g. Qwen): `tokens` + `merges`,
//!   byte-level.
//! * **unigram/SPM** (`tokenizer.ggml.model = "gemma4"`, e.g. Gemma):
//!   `tokens` + `scores` + `token_type` (no merges).
//!
//! The model is assembled from the `tokenizers` crate's `BPE`/`Unigram` models.
//! The pre-tokenizer is architecture-specific, so callers validate the result
//! against the reference tokenizer (see the `tokenize` CLI and the module tests).

use anyhow::{Context, Result, bail};
use serde_json::Value;
use std::collections::BTreeMap;
use tokenizers::models::bpe::{BPE, Vocab};
use tokenizers::models::unigram::Unigram;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::processors::template::TemplateProcessing;
use tokenizers::{AddedToken, Tokenizer};

use crate::util::checkpoint::GgufCheckpoint;

/// GGUF token_type codes (llama.cpp): 1=normal, 2=unknown, 3=control, 4=user.
const T_NORMAL: i64 = 1;
const T_UNKNOWN: i64 = 2;

fn str_array(meta: &BTreeMap<String, Value>, key: &str) -> Option<Vec<String>> {
    meta.get(key).and_then(Value::as_array).map(|a| {
        a.iter()
            .map(|v| v.as_str().unwrap_or_default().to_string())
            .collect()
    })
}

fn f32_array(meta: &BTreeMap<String, Value>, key: &str) -> Option<Vec<f32>> {
    meta.get(key)
        .and_then(Value::as_array)
        .map(|a| a.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect())
}

fn i64_array(meta: &BTreeMap<String, Value>, key: &str) -> Option<Vec<i64>> {
    meta.get(key)
        .and_then(Value::as_array)
        .map(|a| a.iter().map(|v| v.as_i64().unwrap_or(0)).collect())
}

/// Build a tokenizer from a GGUF checkpoint's metadata.
pub fn build(ckpt: &GgufCheckpoint) -> Result<Tokenizer> {
    let meta = ckpt.metadata();
    let model = meta
        .get("tokenizer.ggml.model")
        .and_then(Value::as_str)
        .unwrap_or("gpt2");
    let tokens =
        str_array(meta, "tokenizer.ggml.tokens").context("GGUF has no tokenizer.ggml.tokens")?;

    let mut tok = match model {
        "gpt2" | "qwen2" | "qwen35" => build_bpe(meta, &tokens)?,
        "gemma4" | "gemma" | "llama" | "spm" => build_unigram(meta, &tokens)?,
        other => bail!("unsupported GGUF tokenizer model '{other}'"),
    };

    // Register the control/special tokens so encode/decode round-trips them.
    let token_type = i64_array(meta, "tokenizer.ggml.token_type").unwrap_or_default();
    let mut specials: Vec<AddedToken> = tokens
        .iter()
        .enumerate()
        .filter_map(|(id, t)| {
            let ty = token_type.get(id).copied().unwrap_or(T_NORMAL);
            (ty != T_NORMAL && ty != T_UNKNOWN && !t.is_empty())
                .then(|| AddedToken::from(t.clone(), true))
        })
        .collect();
    // BOS/EOS/PAD are referenced by the post-processor, so make sure they are
    // special even when the vocab marks them as ordinary tokens.
    for key in [
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.eot_token_id",
        "tokenizer.ggml.padding_token_id",
    ] {
        if let Some(id) = meta.get(key).and_then(Value::as_u64)
            && let Some(t) = tokens.get(id as usize)
            && !t.is_empty()
        {
            specials.push(AddedToken::from(t.clone(), true));
        }
    }
    specials.dedup_by(|a, b| a.content == b.content);
    if !specials.is_empty() {
        tok.add_special_tokens(&specials);
    }
    if let Some(pp) = post_processor(meta, &tokens) {
        tok.with_post_processor(Some(pp));
    }
    Ok(tok)
}

/// A template post-processor that adds the BOS/EOS tokens the GGUF metadata
/// declares (`tokenizer.ggml.add_bos_token` / `add_eos_token`).
fn post_processor(meta: &BTreeMap<String, Value>, tokens: &[String]) -> Option<TemplateProcessing> {
    let add = |k: &str| meta.get(k).and_then(Value::as_bool).unwrap_or(false);
    let (add_bos, add_eos) = (
        add("tokenizer.ggml.add_bos_token"),
        add("tokenizer.ggml.add_eos_token"),
    );
    if !add_bos && !add_eos {
        return None;
    }
    let id = |k: &str| {
        meta.get(k)
            .and_then(Value::as_u64)
            .map(|v| v as u32)
            .filter(|&v| (v as usize) < tokens.len())
    };
    let mut template = String::new();
    let mut specials: Vec<(String, u32)> = Vec::new();
    let tok_str = |id: u32| tokens.get(id as usize).cloned().unwrap_or_default();
    if add_bos && let Some(id) = id("tokenizer.ggml.bos_token_id") {
        let s = tok_str(id);
        template.push_str(&s);
        template.push(' ');
        specials.push((s, id));
    }
    template.push_str("$A");
    if add_eos && let Some(id) = id("tokenizer.ggml.eos_token_id") {
        let s = tok_str(id);
        template.push(' ');
        template.push_str(&s);
        specials.push((s, id));
    }
    TemplateProcessing::builder()
        .try_single(template)
        .ok()?
        .special_tokens(specials)
        .build()
        .ok()
}

fn build_bpe(meta: &BTreeMap<String, Value>, tokens: &[String]) -> Result<Tokenizer> {
    let mut vocab = Vocab::default();
    for (i, t) in tokens.iter().enumerate() {
        vocab.insert(t.clone(), i as u32);
    }
    let merges: Vec<(String, String)> = str_array(meta, "tokenizer.ggml.merges")
        .unwrap_or_default()
        .iter()
        .filter_map(|m| {
            let (a, b) = m.split_once(' ')?;
            Some((a.to_string(), b.to_string()))
        })
        .collect();

    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .build()
        .map_err(|e| anyhow::anyhow!("build BPE from GGUF vocab/merges: {e}"))?;
    let mut tok = Tokenizer::new(bpe);
    // Qwen's byte-level pre-tokenizer does not add a prefix space (the
    // `ByteLevel::default()` would).
    let bl = ByteLevel::new(false, false, true);
    tok.with_pre_tokenizer(Some(bl));
    tok.with_decoder(Some(ByteLevel::new(false, false, true)));
    Ok(tok)
}

fn build_unigram(meta: &BTreeMap<String, Value>, tokens: &[String]) -> Result<Tokenizer> {
    use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};

    let scores = f32_array(meta, "tokenizer.ggml.scores")
        .context("GGUF SPM vocab has no tokenizer.ggml.scores")?;
    let vocab: Vec<(String, f64)> = tokens
        .iter()
        .zip(scores.iter())
        .map(|(t, &s)| (t.clone(), s as f64))
        .collect();
    // The unknown token is the one marked token_type 2 (`<unk>`).
    let token_type = i64_array(meta, "tokenizer.ggml.token_type").unwrap_or_default();
    let unk_id = token_type.iter().position(|&t| t == T_UNKNOWN).or(Some(0));
    let unigram = Unigram::from(vocab, unk_id, true)
        .map_err(|e| anyhow::anyhow!("build Unigram from GGUF vocab/scores: {e}"))?;
    let mut tok = Tokenizer::new(unigram);
    // SentencePiece-style: replace spaces with the meta symbol and split on it,
    // without a prefix space on the first segment (matching the HF tokenizer,
    // whose pre-tokenizer is `Split(" ", MergedWithPrevious)` after a
    // space-to-▁ normalizer).
    tok.with_pre_tokenizer(Some(Metaspace::new('▁', PrependScheme::Never, false)));
    tok.with_decoder(Some(Metaspace::default()));
    Ok(tok)
}
