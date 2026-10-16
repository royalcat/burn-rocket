//! Build a `tokenizers::Tokenizer` from GGUF metadata.
//!
//! GGUF carries the llama.cpp vocabulary, not a `tokenizers`-format file. Three
//! shapes appear:
//!
//! * **byte-level BPE** (`tokenizer.ggml.model = "gpt2"`, e.g. Qwen):
//!   `tokens` + `merges`, byte-level.
//! * **merge-based BPE** (`tokenizer.ggml.model = "gemma4"` with `merges`,
//!   e.g. Gemma 4): `tokens` + `merges`; the `scores` array is a placeholder
//!   (constant) and llama.cpp segments by merge rank over the space-to-`▁`
//!   text, not by score Viterbi — feeding those scores to `Unigram` picks the
//!   fewest pieces and mis-segments words (`Dokumenten` -> `Do|kum|enten`).
//! * **unigram/SPM** (`tokenizer.ggml.model = "llama"`, e.g. real SPM):
//!   `tokens` + `scores` + `token_type` (no merges).
//!
//! The model is assembled from the `tokenizers` crate's `BPE`/`Unigram` models.
//! The pre-tokenizer is architecture-specific, so callers validate the result
//! against the reference tokenizer (see the `tokenize` CLI and the module tests,
//! plus `tools/tokenizer_parity.py` for the dev-host A/B).

use anyhow::{Context, Result, bail};
use serde_json::Value;
use std::collections::BTreeMap;
use tokenizers::models::bpe::{BPE, Vocab};
use tokenizers::models::unigram::Unigram;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::processors::template::TemplateProcessing;
use tokenizers::{AddedToken, Tokenizer};

use crate::util::checkpoint::GgufCheckpoint;

/// GGUF token_type codes (llama.cpp): 1=normal, 2=unknown, 3=control, 4=user,
/// 6=byte (the `<0xXX>` fallback tokens).
const T_NORMAL: i64 = 1;
const T_UNKNOWN: i64 = 2;
const T_BYTE: i64 = 6;

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
    build_from_meta(ckpt.metadata())
}

fn build_from_meta(meta: &BTreeMap<String, Value>) -> Result<Tokenizer> {
    let model = meta
        .get("tokenizer.ggml.model")
        .and_then(Value::as_str)
        .unwrap_or("gpt2");
    let tokens =
        str_array(meta, "tokenizer.ggml.tokens").context("GGUF has no tokenizer.ggml.tokens")?;

    let has_merges = !str_array(meta, "tokenizer.ggml.merges")
        .unwrap_or_default()
        .is_empty();
    let mut tok = match model {
        "gpt2" | "qwen2" | "qwen35" => build_bpe(meta, &tokens)?,
        // A `gemma4` vocab with merges is merge-based: the scores are dead
        // placeholders (all equal) and Unigram would mis-segment words.
        "gemma4" if has_merges => build_merged_bpe(meta, &tokens)?,
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
            // Control (3) and user-defined (4) tokens are the vocab specials,
            // while the byte-fallback tokens (6, `<0xXX>`) are emitted by the
            // models themselves and stay out of the added-token set (the
            // reference `tokenizer.json`'s `added_tokens` excludes them too).
            (ty != T_NORMAL && ty != T_UNKNOWN && ty != T_BYTE && !t.is_empty())
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

/// Merge-based BPE for Gemma 4 GGUF vocabularies.
///
/// The GGUF carries the same `tokens` + `merges` as the reference
/// `tokenizer.json` (a `BPE` model); the `scores` array is a placeholder
/// (constant) and llama.cpp segments by merge rank over the `▁`-encoded text.
fn build_merged_bpe(meta: &BTreeMap<String, Value>, tokens: &[String]) -> Result<Tokenizer> {
    use tokenizers::decoders::DecoderWrapper;
    use tokenizers::decoders::byte_fallback::ByteFallback;
    use tokenizers::decoders::fuse::Fuse;
    use tokenizers::decoders::sequence::Sequence as DecoderSequence;
    use tokenizers::normalizers::NormalizerWrapper;
    use tokenizers::normalizers::Sequence as NormalizerSequence;
    use tokenizers::normalizers::prepend::Prepend;
    use tokenizers::normalizers::replace::Replace;

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
        .unk_token(unknown_token(meta, tokens))
        // The reference model config (see the checkpoint's `tokenizer.json`).
        .byte_fallback(true)
        .fuse_unk(true)
        .build()
        .map_err(|e| anyhow::anyhow!("build BPE from GGUF vocab/merges: {e}"))?;
    let mut tok = Tokenizer::new(bpe);

    // The reference normalizer maps spaces to the SPM meta symbol; the same
    // mark is prepended when the metadata asks for it (`add_space_prefix`).
    let mark = "\u{2581}".to_string();
    let replace =
        Replace::new(" ", mark.clone()).map_err(|e| anyhow::anyhow!("space normalizer: {e}"))?;
    let add_space_prefix = meta
        .get("tokenizer.ggml.add_space_prefix")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    tok.with_normalizer(Some(if add_space_prefix {
        NormalizerWrapper::Sequence(NormalizerSequence::new(vec![
            NormalizerWrapper::Replace(replace),
            NormalizerWrapper::Prepend(Prepend::new(mark.clone())),
        ]))
    } else {
        NormalizerWrapper::Replace(replace)
    }));

    // The decode chain of the reference tokenizer: `▁` -> " ", byte fallback,
    // fuse.
    let unmark = Replace::new(mark, " ").map_err(|e| anyhow::anyhow!("meta decoder: {e}"))?;
    tok.with_decoder(Some(DecoderWrapper::Sequence(DecoderSequence::new(vec![
        DecoderWrapper::Replace(unmark),
        DecoderWrapper::ByteFallback(ByteFallback::new()),
        DecoderWrapper::Fuse(Fuse::new()),
    ]))));
    Ok(tok)
}

/// Name of the unknown token: the metadata id, else the `token_type` entry
/// marked unknown, else `<unk>`.
fn unknown_token(meta: &BTreeMap<String, Value>, tokens: &[String]) -> String {
    let id = meta
        .get("tokenizer.ggml.unknown_token_id")
        .and_then(Value::as_u64)
        .map(|v| v as usize)
        .or_else(|| {
            i64_array(meta, "tokenizer.ggml.token_type")
                .unwrap_or_default()
                .iter()
                .position(|&t| t == T_UNKNOWN)
        });
    id.and_then(|i| tokens.get(i))
        .filter(|t| !t.is_empty())
        .cloned()
        .unwrap_or_else(|| "<unk>".to_string())
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Synthetic metadata: `(token, token_type)` rows plus optional merges.
    fn meta(model: &str, tokens: &[(&str, i64)], merges: &[&str]) -> BTreeMap<String, Value> {
        let mut m = BTreeMap::new();
        m.insert("tokenizer.ggml.model".to_string(), json!(model));
        m.insert(
            "tokenizer.ggml.tokens".to_string(),
            json!(tokens.iter().map(|(t, _)| *t).collect::<Vec<_>>()),
        );
        m.insert(
            "tokenizer.ggml.token_type".to_string(),
            json!(tokens.iter().map(|(_, ty)| *ty).collect::<Vec<_>>()),
        );
        m.insert(
            "tokenizer.ggml.scores".to_string(),
            json!(vec![-1000.0f32; tokens.len()]),
        );
        if !merges.is_empty() {
            m.insert("tokenizer.ggml.merges".to_string(), json!(merges));
        }
        m
    }

    fn encode(tok: &Tokenizer, text: &str) -> Vec<u32> {
        tok.encode(text, false).unwrap().get_ids().to_vec()
    }

    /// A `gemma4` vocab with merges segments by merge rank (llama.cpp
    /// semantics), not by the placeholder scores.
    #[test]
    fn gemma4_merges_take_the_bpe_branch() {
        // Ranks: (b c) = 0, (a b) = 1 -> "abc" -> [a, bc], never [ab, c].
        let tok = build_from_meta(&meta(
            "gemma4",
            &[
                ("<unk>", 2),
                ("a", 1),
                ("b", 1),
                ("c", 1),
                ("ab", 1),
                ("bc", 1),
            ],
            &["b c", "a b"],
        ))
        .unwrap();
        assert_eq!(encode(&tok, "abc"), vec![1, 5]);
    }

    /// Spaces become the meta symbol and merge across it; decode restores them.
    #[test]
    fn gemma4_spaces_and_decode() {
        let tok = build_from_meta(&meta(
            "gemma4",
            &[
                ("<unk>", 2),
                ("a", 1),
                ("b", 1),
                ("\u{2581}", 1),
                ("ab", 1),
                ("\u{2581}ab", 1),
            ],
            &["a b", "\u{2581} ab"],
        ))
        .unwrap();
        assert_eq!(encode(&tok, " ab"), vec![5]);
        assert_eq!(encode(&tok, "ab"), vec![4]);
        assert_eq!(tok.decode(&[5], false).unwrap(), " ab");
    }

    /// Without merges, `gemma4` keeps the score-based Unigram path (real SPM).
    #[test]
    fn gemma4_without_merges_stays_unigram() {
        let tokens = &[("<unk>", 2), ("x", 1), ("y", 1), ("xy", 1), ("yx", 1)];
        // Equal scores: Unigram favors the one-piece path.
        let uni = build_from_meta(&meta("gemma4", tokens, &[])).unwrap();
        assert_eq!(encode(&uni, "xy"), vec![3]);
        // With only the "y x" merge listed, "xy" cannot merge.
        let bpe = build_from_meta(&meta("gemma4", tokens, &["y x"])).unwrap();
        assert_eq!(encode(&bpe, "xy"), vec![1, 2]);
    }

    /// Byte-fallback tokens are emitted by the model, not registered as added
    /// special tokens; control tokens still are.
    #[test]
    fn byte_tokens_are_not_added_specials() {
        let tok = build_from_meta(&meta(
            "gemma4",
            &[
                ("<unk>", 2),
                ("a", 1),
                ("b", 1),
                ("ab", 1),
                ("<0x78>", 6),
                ("<|ctl|>", 3),
            ],
            &["a b"],
        ))
        .unwrap();
        let added = tok.get_added_tokens_decoder();
        assert!(added.contains_key(&5), "<|ctl|> must be an added token");
        assert!(!added.contains_key(&4), "<0x78> must not be an added token");
        // The byte token is still reachable through byte fallback.
        assert_eq!(encode(&tok, "x"), vec![4]);
    }
}
