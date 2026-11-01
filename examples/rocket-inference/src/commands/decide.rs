//! `decide`: typed System One decisions (vllm-sr d3, `d3-code-readout-v1`).
//!
//! One process loads the model once and answers every case of a `--requests`
//! array (or the single `--request` object) with the official response shape
//! `{"model", "answers", "usage"}`. `--requests` wraps them as
//! `{"schema": "d3-decision-v1", "model", "cases": [{"name", "response"}]}`,
//! which is what `tools/decision_parity.py` reads.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use serde_json::Value;

use crate::cli::{FlagArgs, LoadArgs, ResolvedLoad};
use crate::commands::{require_decision, resolve};
use crate::decision::request::{CaseInput, DecisionRequest};
use crate::decision::{DecisionModel, load_decision_model};
use crate::qwen35_intent::loader::IntentLoadOptions;
use crate::util::device;
use crate::util::rss_mib;

pub fn run(it: impl Iterator<Item = String>) -> Result<()> {
    let mut args = Args::parse(FlagArgs::new(it))?;
    let load = resolve(std::mem::take(&mut args.load))?;
    require_decision(load.family)?;
    let mut model = load_model(&load, &args)?;
    let model_name = args
        .model_name
        .clone()
        .unwrap_or_else(|| dir_name(&load.model_dir));
    let max_tokens = args.max_state_tokens.unwrap_or(8192);

    let started = Instant::now();
    let output = match (&args.requests, &args.request) {
        (Some(path), None) => {
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("read requests from {}", path.display()))?;
            let cases: Vec<CaseInput> = serde_json::from_str(&text).with_context(|| {
                format!("parse {} (expected a JSON array of cases)", path.display())
            })?;
            if cases.is_empty() {
                bail!("{} holds no cases", path.display());
            }
            let total = cases.len();
            let mut out = Vec::with_capacity(total);
            let mut tokens = 0usize;
            let mut answered = 0usize;
            for (index, case) in cases.into_iter().enumerate() {
                let name = case.name.unwrap_or_else(|| format!("case-{}", index + 1));
                let (response, case_tokens, case_answered) =
                    model.system_one(&model_name, case.request, max_tokens)?;
                tokens += case_tokens;
                answered += case_answered;
                println!(
                    "[{}/{total}] {name}: {case_answered} question(s), {case_tokens} tokens",
                    index + 1
                );
                out.push(serde_json::json!({"name": name, "response": response}));
            }
            println!(
                "decided {answered} question(s) over {total} case(s), {tokens} prompt tokens in {:.2}s ({:.0} MiB anon)",
                started.elapsed().as_secs_f64(),
                rss_mib()
            );
            serde_json::json!({
                "schema": "d3-decision-v1",
                "model": model_name,
                "cases": out,
            })
        }
        (None, Some(json)) => {
            let request: DecisionRequest =
                serde_json::from_str(json).context("parse --request (expected a JSON object)")?;
            let (response, tokens, answered) =
                model.system_one(&model_name, request, max_tokens)?;
            println!(
                "decided {answered} question(s), {tokens} prompt tokens in {:.2}s ({:.0} MiB anon)",
                started.elapsed().as_secs_f64(),
                rss_mib()
            );
            response
        }
        (None, None) => bail!("provide --requests <file> or --request <json object>"),
        (Some(_), Some(_)) => bail!("--requests and --request are mutually exclusive"),
    };
    write_output(&output, args.out.as_deref())
}

fn load_model(load: &ResolvedLoad, args: &Args) -> Result<DecisionModel> {
    let device = device(&load.cpu_backend)?;
    let opts = IntentLoadOptions {
        npu: load.placement.text,
        text: load.placement.text,
        decode: false,
        // A d3 prompt is always a prefill: no decode stage needs CPU copies.
        keep_cpu_weights: false,
        npu_threads: load.npu_threads,
        ..IntentLoadOptions::default()
    };
    let model = load_decision_model(
        &load.model_dir,
        &device,
        &opts,
        args.max_state_tokens.unwrap_or(8192),
    )?;
    println!("decision model resident {:.0} MiB anon", rss_mib());
    Ok(model)
}

fn write_output(output: &Value, path: Option<&std::path::Path>) -> Result<()> {
    match path {
        Some(path) => {
            let mut text = serde_json::to_string(output)?;
            text.push('\n');
            std::fs::write(path, text).with_context(|| format!("write {}", path.display()))?;
            println!("wrote {}", path.display());
        }
        None => println!("{}", serde_json::to_string_pretty(output)?),
    }
    Ok(())
}

fn dir_name(dir: &std::path::Path) -> String {
    dir.file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "model".to_string())
}

struct Args {
    load: LoadArgs,
    requests: Option<PathBuf>,
    request: Option<String>,
    out: Option<PathBuf>,
    max_state_tokens: Option<usize>,
    model_name: Option<String>,
}

impl Args {
    fn parse(mut f: FlagArgs) -> Result<Self> {
        let load = LoadArgs::parse(&mut f)?;
        let mut args = Self {
            load,
            requests: None,
            request: None,
            out: None,
            max_state_tokens: None,
            model_name: None,
        };
        if let Some(v) = f.take("--requests")? {
            args.requests = Some(PathBuf::from(v));
        }
        if let Some(v) = f.take("--request")? {
            args.request = Some(v);
        }
        if let Some(v) = f.take("--out")? {
            args.out = Some(PathBuf::from(v));
        }
        if let Some(v) = f.take_parsed("--max-state-tokens")? {
            if v == 0 {
                bail!("--max-state-tokens must be positive");
            }
            args.max_state_tokens = Some(v);
        }
        args.model_name = f.take("--model-name")?;
        f.finish("decide")?;
        Ok(args)
    }
}
