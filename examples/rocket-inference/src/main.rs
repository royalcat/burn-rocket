//! Burn inference for the RK3588 (flex CPU backend, optional NPU offload).
//!
//! One binary, four model families:
//!
//! | family | models |
//! |---|---|
//! | `qwen3` | Qwen3-Embedding-0.6B (OpenAI-compatible embedding server) |
//! | `intent` | Qwen3.5-0.8B intent/query planner (Ollama-compatible server) |
//! | `gemma` | EmbeddingGemma 2 (multimodal embeddings) and Gemma 4 E2B-it (chat) |
//!
//! See `README.md` for commands and flags.

mod cli;
mod gemma;
mod qwen35_intent;
mod qwen3_embedding;
mod util;

use anyhow::{Result, bail};

fn main() -> Result<()> {
    let mut it = std::env::args().skip(1);
    let Some(family) = it.next() else {
        eprintln!("{}", cli::USAGE);
        std::process::exit(2);
    };
    match family.as_str() {
        "qwen3" => qwen3_embedding::cli::run(it),
        "intent" => qwen35_intent::cli::run(it),
        "gemma" => gemma::cli::run(it),
        "help" | "--help" | "-h" => {
            println!("{}", cli::USAGE);
            Ok(())
        }
        other => bail!(
            "unknown family '{other}' (expected qwen3|intent|gemma)\n\n{}",
            cli::USAGE
        ),
    }
}
