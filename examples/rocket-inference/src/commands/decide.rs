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
//! `images`/`videos` entries are local paths here (the parity fixtures point at
//! the checkpoint's own assets): the HTTP server takes data URLs only, like the
//! reference `d3_server.py`, so the fixture files could not be sent over the wire
//! unchanged. A video is decoded by `ffmpeg` at 2 frames per second (4-32 frames,
//! temporal pairs), the same sampling the server applies to an upload.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use serde_json::Value;

use crate::clef::{ClefModel, ClefRequest, Invalid};
use crate::cli::{FlagArgs, LoadArgs, ResolvedLoad};
use crate::commands::{require_decision, resolve};
use crate::decision::load::manifest_model_name;
use crate::decision::request::{CaseInput, DecisionRequest, MediaMode};
use crate::decision::{DecisionModel, load_decision_model};
use crate::family::Family;
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

/// The default per-record prompt cap of Clef-Omni (`--max-state-tokens` / `serve
/// --max-tokens`): the release's own default is 64000, but the board recipe is 8192 and
/// the layer-streamed forward is linear-to-quadratic in the record length.
pub const CLEF_MAX_TOKENS: usize = 8192;

pub fn run(it: impl Iterator<Item = String>) -> Result<()> {
    let mut args = Args::parse(FlagArgs::new(it))?;
    let load = resolve(std::mem::take(&mut args.load))?;
    require_decision(load.family)?;
    if load.family == Family::Clef {
        return run_clef(args, &load);
    }
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

/// `decide` for Clef-Omni: the same `--requests`/`--request` inputs, one record per case
/// (all questions answered jointly in one forward). A case the model refuses (invalid
/// request, media, a schema over the cap) is reported as `{"name", "error"}` and the
/// run fails after writing the output; a model failure stops the run.
fn run_clef(args: Args, load: &ResolvedLoad) -> Result<()> {
    if args.dump_prompts.is_some() {
        bail!("--dump-prompts does not apply to the clef model");
    }
    let started = Instant::now();
    let device = if load.placement.text {
        crate::clef::ClefDevice::Npu
    } else {
        crate::clef::ClefDevice::Cpu
    };
    let model = ClefModel::load_placed(
        &load.model_dir,
        crate::clef::pipeline::default_threads(),
        device,
    )?;
    println!(
        "clef model opened in {:.1}s ({:.0} MiB anon)",
        started.elapsed().as_secs_f64(),
        rss_mib()
    );
    let model_name = args
        .model_name
        .clone()
        .unwrap_or_else(|| dir_name(&load.model_dir));
    let max_length = args.max_state_tokens.unwrap_or(CLEF_MAX_TOKENS);
    // One case -> (response or the refusal, prompt tokens).
    let run_case = |mut value: Value| -> Result<std::result::Result<(Value, usize), Invalid>> {
        if let Value::Object(object) = &mut value {
            object.remove("name");
            object
                .entry("model")
                .or_insert_with(|| Value::String(model_name.clone()));
        }
        let request = match ClefRequest::from_value(value) {
            Ok(request) => request,
            Err(refusal) => return Ok(Err(refusal)),
        };
        match model.answer(&request, max_length) {
            Ok(done) => Ok(Ok((done.response, done.tokens))),
            Err(error) => match error.downcast::<Invalid>() {
                Ok(refusal) => Ok(Err(refusal)),
                Err(error) => Err(error),
            },
        }
    };
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
            let (mut out, mut refused, mut tokens) = (Vec::with_capacity(total), 0usize, 0usize);
            for (index, value) in values.into_iter().enumerate() {
                let name = value
                    .get("name")
                    .or_else(|| value.get("id"))
                    .and_then(Value::as_str)
                    .map_or_else(|| format!("case-{}", index + 1), str::to_string);
                let t0 = Instant::now();
                match run_case(value)? {
                    Ok((response, case_tokens)) => {
                        tokens += case_tokens;
                        println!(
                            "[{}/{total}] {name}: {} question(s), {case_tokens} tokens in {:.1}s",
                            index + 1,
                            response["answers"].as_object().map_or(0, |a| a.len()),
                            t0.elapsed().as_secs_f64()
                        );
                        out.push(serde_json::json!({"name": name, "response": response}));
                    }
                    Err(refusal) => {
                        refused += 1;
                        println!("[{}/{total}] {name}: refused: {refusal}", index + 1);
                        out.push(serde_json::json!({"name": name, "error": refusal.0}));
                    }
                }
            }
            println!(
                "decided {} case(s), {tokens} prompt tokens in {:.1}s ({:.0} MiB anon){}",
                total - refused,
                started.elapsed().as_secs_f64(),
                rss_mib(),
                if refused > 0 {
                    format!(", {refused} refused")
                } else {
                    String::new()
                }
            );
            let output = serde_json::json!({
                "schema": "clef-decision-v1",
                "model": model_name,
                "cases": out,
            });
            write_output(&output, args.out.as_deref())?;
            if refused > 0 {
                bail!("{refused} case(s) were refused (see the `error` entries)");
            }
            return Ok(());
        }
        (None, Some(json)) => {
            let value: Value =
                serde_json::from_str(json).context("parse --request (expected a JSON object)")?;
            match run_case(value)? {
                Ok((response, tokens)) => {
                    println!(
                        "decided {} question(s), {tokens} prompt tokens in {:.1}s ({:.0} MiB anon)",
                        response["answers"].as_object().map_or(0, |a| a.len()),
                        started.elapsed().as_secs_f64(),
                        rss_mib()
                    );
                    response
                }
                Err(refusal) => bail!("request refused: {refusal}"),
            }
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
        embed_int8: load.embed_int8,
        npu_int8: load.npu_int8(),
        ..IntentLoadOptions::default()
    };
    let model = load_decision_model(
        &load.model_dir,
        &device,
        &opts,
        args.max_state_tokens.unwrap_or(MAX_STATE_TOKENS),
        load.defer_towers,
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
    let videos = request
        .decode_videos(MediaMode::Paths)
        .map_err(|message| anyhow::anyhow!("{message}"))?;
    let prepared = model.prepare(request, &images, &videos, max_tokens)?;
    for sequence in &prepared.sequences {
        let (id, ids) = (&sequence.id, &sequence.ids);
        let question = prepared
            .questions
            .get(id)
            .ok_or_else(|| anyhow::anyhow!("internal error: '{id}' has no validated question"))?;
        let user =
            crate::decision::format::user_prompt(&request.state, question, &model.config.codes)
                .map_err(|message| anyhow::anyhow!("{message}"))?;
        // The expanded prompt, i.e. the reference dump's `expanded_text`: the
        // image groups plus each video's timestamped groups.
        let prompt = crate::decision::model::dump_prompt(
            &user,
            &prepared.image_token_counts(),
            &prepared.videos,
        );
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
