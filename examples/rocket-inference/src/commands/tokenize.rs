//! `tokenize`: tokenize text with the checkpoint's tokenizer.

use std::path::PathBuf;

use anyhow::{Result, bail};

use crate::cli::FlagArgs;
use crate::commands::{prompt_prefix, resolve_quiet};
use crate::util::checkpoint::Checkpoint;

pub fn run(it: impl Iterator<Item = String>) -> Result<()> {
    let mut f = FlagArgs::new(it);
    let load = crate::cli::LoadArgs::parse(&mut f)?;
    let mut text = None;
    let mut text_file: Option<PathBuf> = None;
    let mut prompt = None;
    let mut tokens = None;
    if let Some(v) = f.take("--text")? {
        text = Some(v);
    }
    if let Some(v) = f.take("--text-file")? {
        text_file = Some(PathBuf::from(v));
    }
    if let Some(v) = f.take("--prompt")? {
        prompt = Some(v);
    }
    if let Some(v) = f.take_parsed("--tokens")? {
        tokens = Some(v);
    }
    f.finish("tokenize")?;

    let load = resolve_quiet(load)?;
    if prompt.is_some() && load.family != crate::family::Family::Emb2 {
        bail!("--prompt only applies to the embeddinggemma task prompts");
    }
    let text = crate::commands::input_text(&text, &text_file)?;
    let prefix = prompt_prefix(&load.model_dir, prompt.as_deref())?;
    let ckpt = crate::util::checkpoint::open_in_dir(&load.model_dir)?;
    let tokenizer = ckpt.tokenizer()?;
    let enc = tokenizer
        .encode(format!("{prefix}{text}").as_str(), true)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
    let mut ids = enc.get_ids().to_vec();
    if let Some(n) = tokens {
        ids.truncate(n);
    }
    println!("{} tokens", ids.len());
    println!("{ids:?}");
    Ok(())
}
