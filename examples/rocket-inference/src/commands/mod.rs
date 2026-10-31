//! The family-agnostic commands. The model family is detected from the
//! checkpoint, so `embed`/`bench`/`gen`/`tokenize` dispatch to the family's
//! loader and forward path instead of being nested under a family command.

pub mod bench;
pub mod embed;
pub mod embed_input;
pub mod gemm;
pub mod generate;
pub mod tokenize;

use std::path::Path;

use anyhow::{Context, Result};
use burn::tensor::DType;

use crate::cli::{LoadArgs, ResolvedLoad};
use crate::family::Family;
use crate::gemma::gemma4::loader::LoadDtype;

/// Resolve `--model-dir`/`--family`, validate the placement against the family
/// and print the resolved `placement:` line once, before loading.
pub fn resolve(load: LoadArgs) -> Result<ResolvedLoad> {
    resolve_with(load, true)
}

/// [`resolve`] without the placement line (for commands that load no model).
pub fn resolve_quiet(load: LoadArgs) -> Result<ResolvedLoad> {
    resolve_with(load, false)
}

fn resolve_with(load: LoadArgs, announce: bool) -> Result<ResolvedLoad> {
    let model_dir = load.model_dir()?;
    let family = load.family(&model_dir)?;
    let resolved = load.resolve(model_dir, family)?;
    if announce {
        println!("{}", resolved.placement_line());
    }
    Ok(resolved)
}

/// `embed`/`bench` need an embedding model.
pub fn require_embedding(family: Family, command: &str) -> Result<()> {
    if family.is_embedding() {
        return Ok(());
    }
    anyhow::bail!(
        "`{command}` needs an embedding model, but the checkpoint is the {} \
         ({}) model; `gen` is the chat command",
        family.name(),
        if family.is_chat() { "chat" } else { "unknown" }
    )
}

/// `gen` needs a chat model.
pub fn require_chat(family: Family) -> Result<()> {
    if family.is_chat() {
        return Ok(());
    }
    anyhow::bail!(
        "`gen` needs a chat model, but the checkpoint is the {} embedding model; \
         `embed` is the embedding command",
        family.name()
    )
}

/// Resolve the task prompt prefix from `config_sentence_transformers.json`
/// (case-insensitive key match; `none`/`raw`/absent means no prefix).
pub fn prompt_prefix(model_dir: &Path, name: Option<&str>) -> Result<String> {
    let Some(name) = name else {
        return Ok(String::new());
    };
    if name.eq_ignore_ascii_case("none") || name.eq_ignore_ascii_case("raw") {
        return Ok(String::new());
    }
    let path = model_dir.join("config_sentence_transformers.json");
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    let json: serde_json::Value = serde_json::from_str(&text)?;
    let prompts = json
        .get("prompts")
        .and_then(|p| p.as_object())
        .context("config_sentence_transformers.json has no 'prompts' object")?;
    for (key, v) in prompts {
        if key.eq_ignore_ascii_case(name) {
            return Ok(v.as_str().unwrap_or_default().to_string());
        }
    }
    let mut keys: Vec<&String> = prompts.keys().collect();
    keys.sort();
    anyhow::bail!("unknown prompt '{name}' (available: {keys:?})")
}

/// Read `--text`/`--text-file`, requiring one of them.
pub fn input_text(text: &Option<String>, text_file: &Option<std::path::PathBuf>) -> Result<String> {
    if let Some(t) = text {
        return Ok(t.clone());
    }
    if let Some(path) = text_file {
        return std::fs::read_to_string(path)
            .with_context(|| format!("read text from {}", path.display()));
    }
    anyhow::bail!("provide --text <s> or --text-file <path>")
}

/// The Gemma 4 loader dtype from the resolved `--dtype`/`--quant` flags.
pub fn gemma4_load_dtype(load: &ResolvedLoad) -> LoadDtype {
    if load.quant_q8 {
        return LoadDtype::Q8;
    }
    match load.dtype_explicit {
        None => LoadDtype::from_native(&load.model_dir),
        Some(DType::F32) => LoadDtype::F32,
        Some(DType::F16) => LoadDtype::F16,
        Some(DType::BF16) => LoadDtype::Bf16,
        Some(other) => panic!("unsupported gemma4 dtype {other:?}"),
    }
}
