//! `gemma` subcommands: the EmbeddingGemma 2 embedding model and the Gemma 4
//! E2B-it generation model.

use anyhow::{Result, bail};

pub fn run(it: impl Iterator<Item = String>) -> Result<()> {
    let mut it = it;
    let cmd = it.next().unwrap_or_default();
    match cmd.as_str() {
        "embed" | "bench" | "tokenize" => super::embeddinggemma::cli::run(&cmd, it),
        "gen" => super::gemma4::cli::run(&cmd, it),
        other => bail!("unknown gemma command '{other}' (expected embed|bench|tokenize|gen)"),
    }
}
