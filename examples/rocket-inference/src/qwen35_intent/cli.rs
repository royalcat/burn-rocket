//! `intent gen`: one-shot generation with the Qwen3.5-0.8B intent model
//! (serving lives in the top-level `serve` command).

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use tokenizers::Tokenizer;

use crate::cli::FlagArgs;
use crate::qwen35_intent::loader::{self, IntentLoadOptions};
use crate::qwen35_intent::model::IntentModel;
use crate::util::device;
use crate::util::rss_mib;

pub fn run(it: impl Iterator<Item = String>) -> Result<()> {
    let mut it = it;
    let cmd = it.next().unwrap_or_default();
    if cmd != "gen" {
        bail!("unknown intent command '{cmd}' (expected gen)");
    }
    let args = Args::parse(&cmd, FlagArgs::new(it))?;
    run_gen(&args)
}

struct Args {
    model_dir: PathBuf,
    backend: String,
    text: Option<String>,
    text_file: Option<PathBuf>,
    npu: bool,
    npu_threads: usize,
    embed_f16: bool,
    pure_npu: bool,
    npu_decode: bool,
    max_tokens: usize,
    max_new_tokens: usize,
    temperature: f32,
    delta_chunk: usize,
    raw: bool,
    dump_hidden: Option<PathBuf>,
    dump_steps: Option<PathBuf>,
}

impl Args {
    fn parse(cmd: &str, mut f: FlagArgs) -> Result<Self> {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        let mut args = Args {
            model_dir: PathBuf::from(format!("{home}/models/ov-intent-analysis-sft")),
            backend: if cfg!(feature = "cpu") { "cpu" } else { "flex" }.to_string(),
            text: None,
            text_file: None,
            npu: false,
            npu_threads: 5,
            embed_f16: false,
            pure_npu: false,
            npu_decode: false,
            max_tokens: 30000,
            max_new_tokens: 256,
            temperature: 0.0,
            delta_chunk: 64,
            raw: false,
            dump_hidden: None,
            dump_steps: None,
        };
        if let Some(v) = f.take("--model-dir")? {
            args.model_dir = PathBuf::from(v);
        }
        if let Some(v) = f.take("--backend")? {
            args.backend = v;
        }
        if let Some(v) = f.take("--text")? {
            args.text = Some(v);
        }
        if let Some(v) = f.take("--text-file")? {
            args.text_file = Some(PathBuf::from(v));
        }
        args.npu = f.take_bool("--npu");
        if let Some(v) = f.take_parsed("--npu-threads")? {
            args.npu_threads = v;
        }
        args.embed_f16 = f.take_bool("--embed-f16");
        args.pure_npu = f.take_bool("--pure-npu");
        args.npu_decode = f.take_bool("--npu-decode");
        if let Some(v) = f.take_parsed("--max-tokens")? {
            args.max_tokens = v;
        }
        if let Some(v) = f.take_parsed("--max-new-tokens")? {
            args.max_new_tokens = v;
        }
        if let Some(v) = f.take_parsed("--temperature")? {
            args.temperature = v;
        }
        if let Some(v) = f.take_parsed("--delta-chunk")? {
            args.delta_chunk = v;
        }
        args.raw = f.take_bool("--raw");
        if let Some(v) = f.take("--dump-hidden")? {
            args.dump_hidden = Some(PathBuf::from(v));
        }
        if let Some(v) = f.take("--dump-steps")? {
            args.dump_steps = Some(PathBuf::from(v));
        }
        f.finish(&format!("intent {cmd}"))?;
        Ok(args)
    }

    fn input_text(&self) -> Result<String> {
        if let Some(t) = &self.text {
            return Ok(t.clone());
        }
        let path = self
            .text_file
            .clone()
            .unwrap_or_else(|| PathBuf::from("data/bench_text.txt"));
        std::fs::read_to_string(&path).with_context(|| format!("read text from {}", path.display()))
    }
}

/// Build the intent model + tokenizer for `gen`.
fn load_intent(args: &Args) -> Result<(IntentModel, Tokenizer)> {
    let device = device(&args.backend)?;
    let opts = IntentLoadOptions {
        npu: args.npu,
        npu_threads: args.npu_threads,
        embed_f16: args.embed_f16,
        pure_npu: args.pure_npu,
    };
    let loaded = loader::load_for_serving(
        &args.model_dir,
        &device,
        &opts,
        args.delta_chunk,
        args.pure_npu,
        args.npu_decode,
    )?;
    println!("intent model resident {:.0} MiB anon", rss_mib());
    Ok((loaded.model, loaded.tokenizer))
}

/// ChatML wrapper identical to the model's Ollama template.
fn chat_wrap(text: &str) -> String {
    format!("<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n")
}

fn run_gen(args: &Args) -> Result<()> {
    let (mut model, tokenizer) = load_intent(args)?;
    let text = args.input_text()?;
    let prompt = if args.raw { text } else { chat_wrap(&text) };
    let enc = tokenizer
        .encode(prompt.as_str(), false)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
    let ids: Vec<u32> = enc.get_ids().to_vec();
    if ids.len() > args.max_tokens {
        bail!(
            "prompt has {} tokens, over --max-tokens {}",
            ids.len(),
            args.max_tokens
        );
    }
    let stops = loader::stop_ids(&args.model_dir, &tokenizer, &model.cfg);
    if let Some(path) = &args.dump_hidden {
        let hidden = model.forward_hidden(&ids);
        let layers: Vec<Vec<f32>> = hidden
            .iter()
            .map(|h| {
                let [_, s, _] = h.dims();
                h.clone()
                    .slice(burn::tensor::s![.., s - 1..s, ..])
                    .reshape([h.dims()[2]])
                    .to_data()
                    .try_to_vec()
                    .unwrap_or_default()
            })
            .collect();
        std::fs::write(
            path,
            serde_json::to_string(&serde_json::json!({ "last_pos": layers }))?,
        )?;
        println!("wrote {} hidden states to {}", layers.len(), path.display());
    }
    if let Some(path) = &args.dump_steps {
        // Debug: greedy decode recording top-3 logits per step.
        model.cache.reset();
        let mut logits = model.forward_logits(&ids);
        let mut out = Vec::new();
        let mut steps = Vec::new();
        for _ in 0..args.max_new_tokens {
            let v: Vec<f32> = logits.to_data().try_to_vec().unwrap_or_default();
            let mut order: Vec<usize> = (0..v.len()).collect();
            order.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap());
            steps.push(serde_json::json!({
                "top": order[..3].iter().map(|&i| serde_json::json!({"id": i, "logit": v[i]})).collect::<Vec<_>>()
            }));
            let next = order[0] as u32;
            if stops.contains(&next) {
                break;
            }
            out.push(next);
            if out.len() == args.max_new_tokens {
                break;
            }
            logits = model.forward_logits(&[next]);
        }
        std::fs::write(
            path,
            serde_json::to_string(&serde_json::json!({"gen_ids": out, "steps": steps}))?,
        )?;
        println!("wrote {} steps to {}", steps.len(), path.display());
        let _ = &stops;
        return Ok(());
    }
    let t0 = Instant::now();
    let (out, stats) = model.generate(&ids, args.max_new_tokens, &stops, args.temperature, 42);
    let total = t0.elapsed().as_secs_f64();
    let decoded = tokenizer
        .decode(&out, false)
        .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
    println!(
        "prompt={} tokens generated={} done={:?} prefill={:.2}s decode={:.2}s ({:.1} tok/s) total={:.2}s",
        ids.len(),
        out.len(),
        stats.stopped,
        stats.prefill_s,
        stats.decode_s,
        if stats.decode_s > 0.0 {
            out.len() as f64 / stats.decode_s
        } else {
            0.0
        },
        total
    );
    println!(
        "ids: {}",
        out.iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    );
    println!("text: {decoded}");
    println!("resident {:.0} MiB anon", rss_mib());
    Ok(())
}
