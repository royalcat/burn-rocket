//! `decide`: typed System One decisions (vllm-sr d3, `d3-code-readout-v1`).
//!
//! One process loads the model once and answers every case of a `--requests`
//! array (or the single `--request` object) with the official response shape
//! `{"model", "answers", "usage"}`. `--requests` wraps them as
//! `{"schema": "d3-decision-v1", "model", "cases": [{"name", "response"}]}`,
//! which is what `tools/decision_parity.py` reads.
//!
//! Unlike `/v1/systemone`, the CLI keeps the runtime's per-question errors in the
//! answer (a `--requests` batch reports every question); the HTTP server refuses
//! a request that carries an `invalid_question` with a 422.
//!
//! `images` entries are local paths here (the parity fixtures point at the
//! checkpoint's own assets): the HTTP server takes data URLs only, like the
//! reference `d3_server.py`, so the fixture files could not be sent over the wire
//! unchanged.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use serde_json::Value;

use crate::cli::{FlagArgs, LoadArgs, ResolvedLoad};
use crate::commands::{require_decision, resolve};
use crate::decision::load::manifest_model_name;
use crate::decision::request::{CaseInput, DecisionRequest, MediaMode};
use crate::decision::{DecisionModel, load_decision_model};
use crate::qwen35_intent::loader::IntentLoadOptions;
use crate::util::device;
use crate::util::rss_mib;

/// The default per-question prompt cap (`--max-state-tokens`).
///
/// 32768 rather than the intent server's 8192: a d3 package declares its own
/// `max_length` (the shipped one is `null`, i.e. the model's 262144), an image
/// question spends one token per merged image token (868 for a receipt page), and
/// the RoPE table this cap sizes is only ~16 MiB. Multi-image requests that the
/// reference answers would otherwise be refused. A deployment with a small memory
/// budget passes the flag (the board recipe uses 8192).
pub const MAX_STATE_TOKENS: usize = 32768;

pub fn run(it: impl Iterator<Item = String>) -> Result<()> {
    let mut args = Args::parse(FlagArgs::new(it))?;
    let load = resolve(std::mem::take(&mut args.load))?;
    require_decision(load.family)?;
    let mut model = load_model(&load, &args)?;
    let model_name = args
        .model_name
        .clone()
        .or_else(|| manifest_model_name(&load.model_dir))
        .unwrap_or_else(|| dir_name(&load.model_dir));
    let max_tokens = args.max_state_tokens.unwrap_or(MAX_STATE_TOKENS);

    let started = Instant::now();
    let output = match (&args.requests, &args.request) {
        (Some(path), None) => {
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("read requests from {}", path.display()))?;
            let values: Vec<Value> = serde_json::from_str(&text).with_context(|| {
                format!("parse {} (expected a JSON array of cases)", path.display())
            })?;
            if values.is_empty() {
                bail!("{} holds no cases", path.display());
            }
            let total = values.len();
            let mut out = Vec::with_capacity(total);
            let mut tokens = 0usize;
            let mut answered = 0usize;
            for (index, value) in values.into_iter().enumerate() {
                let case = CaseInput::from_value(value)
                    .map_err(|message| anyhow::anyhow!("case {}: {message}", index + 1))?;
                let name = case.name.unwrap_or_else(|| format!("case-{}", index + 1));
                if let Some(dir) = &args.dump_prompts {
                    dump_prompts(&mut model, dir, &name, &case.request, max_tokens)?;
                }
                let (response, case_tokens, case_answered) =
                    model.system_one(&model_name, case.request, max_tokens, MediaMode::Paths)?;
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
            let value: Value =
                serde_json::from_str(json).context("parse --request (expected a JSON object)")?;
            let request = DecisionRequest::from_value(value).map_err(|m| anyhow::anyhow!("{m}"))?;
            if let Some(dir) = &args.dump_prompts {
                dump_prompts(&mut model, dir, "case-1", &request, max_tokens)?;
            }
            let (response, tokens, answered) =
                model.system_one(&model_name, request, max_tokens, MediaMode::Paths)?;
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
        embed_f16: load.embed_f16,
        ..IntentLoadOptions::default()
    };
    let model = load_decision_model(
        &load.model_dir,
        &device,
        &opts,
        args.max_state_tokens.unwrap_or(MAX_STATE_TOKENS),
    )?;
    println!("decision model resident {:.0} MiB anon", rss_mib());
    Ok(model)
}

/// `--dump-prompts`: one `<case>__<qid>.prompt.txt` per question, the layout the
/// reference harness writes (byte comparisons against the official runtime).
fn dump_prompts(
    model: &mut DecisionModel,
    dir: &std::path::Path,
    case: &str,
    request: &DecisionRequest,
    max_tokens: usize,
) -> Result<()> {
    std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    let images = request
        .decode_images(MediaMode::Paths)
        .map_err(|message| anyhow::anyhow!("{message}"))?;
    let prepared = model.prepare(request, &images, max_tokens)?;
    for sequence in &prepared.sequences {
        let (id, ids) = (&sequence.id, &sequence.ids);
        let question = prepared
            .questions
            .get(id)
            .ok_or_else(|| anyhow::anyhow!("internal error: '{id}' has no validated question"))?;
        let user =
            crate::decision::format::user_prompt(&request.state, question, &model.config.codes)
                .map_err(|message| anyhow::anyhow!("{message}"))?;
        let prompt = crate::decision::format::chatml_images(&user, prepared.images.len());
        let path = dir.join(format!("{case}__{id}.prompt.txt"));
        std::fs::write(&path, &prompt).with_context(|| format!("write {}", path.display()))?;
        println!("wrote {} ({} tokens)", path.display(), ids.len());
    }
    Ok(())
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
    dump_prompts: Option<PathBuf>,
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
            dump_prompts: None,
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
        if let Some(v) = f.take("--dump-prompts")? {
            args.dump_prompts = Some(PathBuf::from(v));
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
