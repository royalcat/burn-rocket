//! Burn inference for the RK3588 (flex CPU backend, per-stage NPU offload).
//!
//! One binary, family-agnostic commands: the model family is detected from the
//! checkpoint (`config.json` `model_type` or the GGUF architecture) and
//! `--family` overrides it.
//!
//! | command | models |
//! |---|---|
//! | `serve` | any detected family (embedding and/or chat routes) |
//! | `embed`, `bench` | qwen3 (Qwen3-Embedding-0.6B), embeddinggemma |
//! | `gen` | intent (Qwen3.5-0.8B), gemma4 (Gemma 4 E2B-it) |
//! | `tokenize` | any detected family |
//! | `gemm` | no model (flex matmul microbenchmark) |
//!
//! See `README.md` and `docs/recipes/` for commands and flags.

mod cli;
mod commands;
mod family;
mod gemma;
mod qwen35_intent;
mod qwen3_embedding;
mod server;
mod util;

use anyhow::{Result, bail};

fn main() -> Result<()> {
    let mut it = std::env::args().skip(1);
    let Some(cmd) = it.next() else {
        eprintln!("{}", cli::USAGE);
        std::process::exit(2);
    };
    if matches!(cmd.as_str(), "help" | "--help" | "-h") {
        let sub = it.next();
        match sub {
            None => println!("{}", cli::USAGE),
            Some(name) => match cli::command_usage(&name) {
                Some(text) => println!("{text}"),
                None => {
                    bail!("no help for '{name}' (commands: serve|embed|bench|gen|tokenize|gemm)")
                }
            },
        }
        return Ok(());
    }
    if !matches!(
        cmd.as_str(),
        "serve" | "embed" | "bench" | "gen" | "tokenize" | "gemm"
    ) {
        bail!(
            "unknown command '{cmd}' (expected serve|embed|bench|gen|tokenize|gemm)\n\n{}",
            cli::USAGE
        );
    }
    // `<command> --help` prints the command's flags without resolving `-hf`
    // or touching the model. Only the leading argument is treated as help so a
    // flag value that happens to be `--help` is not swallowed.
    let rest: Vec<String> = it.collect();
    if rest.first().is_some_and(|a| a == "--help" || a == "-h") {
        println!(
            "{}",
            cli::command_usage(&cmd).expect("known command has usage text")
        );
        return Ok(());
    }
    let rest = util::hub::resolve_args(rest.into_iter())?;
    let it = rest.into_iter();
    match cmd.as_str() {
        "serve" => server::cli::run(it),
        "embed" => commands::embed::run(it),
        "bench" => commands::bench::run(it),
        "gen" => commands::generate::run(it),
        "tokenize" => commands::tokenize::run(it),
        "gemm" => commands::gemm::run(it),
        _ => unreachable!("validated above"),
    }
}
