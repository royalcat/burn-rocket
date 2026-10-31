//! `gen`: one chat completion (intent or Gemma 4).

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use tokenizers::Tokenizer;

use crate::cli::{FlagArgs, LoadArgs, ResolvedLoad};
use crate::commands::{gemma4_load_dtype, input_text, require_chat, resolve};
use crate::family::Family;
use crate::gemma::gemma4::chat::{self, gen_eos};
use crate::gemma::gemma4::loader as gen_loader;
use crate::gemma::gemma4::model::{gen_stage_stats, gen_stage_stats_reset};
use crate::gemma::inputs;
use crate::qwen35_intent::loader::{self as intent_loader, IntentLoadOptions};
use crate::qwen35_intent::model::IntentModel;
use crate::util::checkpoint::Checkpoint;
use crate::util::device;
use crate::util::rss_mib;

pub fn run(it: impl Iterator<Item = String>) -> Result<()> {
    let mut args = Args::parse(FlagArgs::new(it))?;
    let load = resolve(std::mem::take(&mut args.load))?;
    validate(&load, &args)?;
    match load.family {
        Family::Intent => run_intent(&load, &args),
        Family::Gemma4 => run_gemma4(&load, &args),
        other => bail!("`gen` does not support the {} model", other.name()),
    }
}

struct Args {
    load: LoadArgs,
    text: Option<String>,
    text_file: Option<PathBuf>,
    messages: Option<String>,
    image: Option<PathBuf>,
    audio: Option<PathBuf>,
    max_soft_tokens: Option<usize>,
    max_tokens: Option<usize>,
    max_new_tokens: Option<usize>,
    temperature: Option<f32>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    sample: Option<bool>,
    enable_thinking: Option<bool>,
    raw: Option<bool>,
    delta_chunk: Option<usize>,
    dump_hidden: Option<PathBuf>,
    dump_steps: Option<PathBuf>,
    dump_logits: Option<PathBuf>,
    out: Option<PathBuf>,
}

impl Args {
    fn parse(mut f: FlagArgs) -> Result<Self> {
        let load = LoadArgs::parse(&mut f)?;
        let mut a = Self {
            load,
            text: None,
            text_file: None,
            messages: None,
            image: None,
            audio: None,
            max_soft_tokens: None,
            max_tokens: None,
            max_new_tokens: None,
            temperature: None,
            top_k: None,
            top_p: None,
            sample: None,
            enable_thinking: None,
            raw: None,
            delta_chunk: None,
            dump_hidden: None,
            dump_steps: None,
            dump_logits: None,
            out: None,
        };
        if let Some(v) = f.take("--text")? {
            a.text = Some(v);
        }
        if let Some(v) = f.take("--text-file")? {
            a.text_file = Some(PathBuf::from(v));
        }
        if let Some(v) = f.take("--messages")? {
            a.messages = Some(v);
        }
        if let Some(v) = f.take("--image")? {
            a.image = Some(PathBuf::from(v));
        }
        if let Some(v) = f.take("--audio")? {
            a.audio = Some(PathBuf::from(v));
        }
        if let Some(v) = f.take_parsed("--max-soft-tokens")? {
            a.max_soft_tokens = Some(v);
        }
        if let Some(v) = f.take_parsed("--max-tokens")? {
            a.max_tokens = Some(v);
        }
        if let Some(v) = f.take_parsed("--max-new-tokens")? {
            a.max_new_tokens = Some(v);
        }
        if let Some(v) = f.take_parsed("--temperature")? {
            a.temperature = Some(v);
        }
        if let Some(v) = f.take_parsed("--top-k")? {
            a.top_k = Some(v);
        }
        if let Some(v) = f.take_parsed("--top-p")? {
            a.top_p = Some(v);
        }
        a.sample = f.take_flag("--sample");
        a.enable_thinking = f.take_flag("--enable-thinking");
        a.raw = f.take_flag("--raw");
        if let Some(v) = f.take_parsed("--delta-chunk")? {
            a.delta_chunk = Some(v);
        }
        if let Some(v) = f.take("--dump-hidden")? {
            a.dump_hidden = Some(PathBuf::from(v));
        }
        if let Some(v) = f.take("--dump-steps")? {
            a.dump_steps = Some(PathBuf::from(v));
        }
        if let Some(v) = f.take("--dump-logits")? {
            a.dump_logits = Some(PathBuf::from(v));
        }
        if let Some(v) = f.take("--out")? {
            a.out = Some(PathBuf::from(v));
        }
        f.finish("gen")?;
        Ok(a)
    }

    fn input_text(&self) -> Result<String> {
        input_text(&self.text, &self.text_file)
    }
}

fn validate(load: &ResolvedLoad, args: &Args) -> Result<()> {
    require_chat(load.family)?;
    match load.family {
        Family::Intent => {
            if args.messages.is_some()
                || args.image.is_some()
                || args.audio.is_some()
                || args.max_soft_tokens.is_some()
                || args.sample.is_some()
                || args.top_k.is_some()
                || args.top_p.is_some()
                || args.enable_thinking.is_some()
                || args.dump_logits.is_some()
            {
                bail!(
                    "--messages, media inputs, sampling flags and --dump-logits are \
                     Gemma 4 only (this checkpoint is intent)"
                );
            }
        }
        Family::Gemma4 => {
            if args.input_text().is_err() && args.messages.is_none() {
                bail!("provide --text/--text-file or --messages <json>");
            }
            if args.raw.is_some()
                || args.max_tokens.is_some()
                || args.delta_chunk.is_some()
                || args.dump_hidden.is_some()
                || args.dump_steps.is_some()
            {
                bail!(
                    "--raw, --max-tokens, --delta-chunk and the intent dumps are \
                     intent only (this checkpoint is gemma4)"
                );
            }
        }
        _ => {}
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// intent
// ---------------------------------------------------------------------------

/// ChatML wrapper identical to the model's Ollama template.
fn chat_wrap(text: &str) -> String {
    format!("<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n")
}

fn load_intent(load: &ResolvedLoad, args: &Args) -> Result<(IntentModel, Tokenizer, Vec<u32>)> {
    let device = device(&load.cpu_backend)?;
    let opts = IntentLoadOptions {
        npu: load.placement.text,
        text: load.placement.text,
        decode: load.placement.decode,
        keep_cpu_weights: load.keep_cpu_weights,
        npu_threads: load.npu_threads,
        embed_f16: load.embed_f16,
    };
    let loaded = intent_loader::load_for_serving(
        &load.model_dir,
        &device,
        &opts,
        args.delta_chunk.unwrap_or(64),
    )?;
    println!("intent model resident {:.0} MiB anon", rss_mib());
    Ok((loaded.model, loaded.tokenizer, loaded.stop_ids))
}

fn run_intent(load: &ResolvedLoad, args: &Args) -> Result<()> {
    let (mut model, tokenizer, stops) = load_intent(load, args)?;
    let text = args.input_text()?;
    let prompt = if args.raw.unwrap_or(false) {
        text
    } else {
        chat_wrap(&text)
    };
    let enc = tokenizer
        .encode(prompt.as_str(), false)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
    let ids: Vec<u32> = enc.get_ids().to_vec();
    let max_tokens = args.max_tokens.unwrap_or(30000);
    if ids.len() > max_tokens {
        bail!(
            "prompt has {} tokens, over --max-tokens {}",
            ids.len(),
            max_tokens
        );
    }
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
        let max_new = args.max_new_tokens.unwrap_or(256);
        model.cache.reset();
        let mut logits = model.forward_logits(&ids);
        let mut out = Vec::new();
        let mut steps = Vec::new();
        for _ in 0..max_new {
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
            if out.len() == max_new {
                break;
            }
            logits = model.forward_logits(&[next]);
        }
        std::fs::write(
            path,
            serde_json::to_string(&serde_json::json!({"gen_ids": out, "steps": steps}))?,
        )?;
        println!("wrote {} steps to {}", steps.len(), path.display());
        return Ok(());
    }
    let t0 = Instant::now();
    let (out, stats) = model.generate(
        &ids,
        args.max_new_tokens.unwrap_or(256),
        &stops,
        args.temperature.unwrap_or(0.0),
        42,
    );
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
    if let Some(path) = &args.out {
        std::fs::write(path, serde_json::to_string(&out)?)
            .with_context(|| format!("write {}", path.display()))?;
        println!("wrote {}", path.display());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// gemma4
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
struct MessageJson {
    role: String,
    content: String,
}

fn run_gemma4(load: &ResolvedLoad, args: &Args) -> Result<()> {
    let device = device(&load.cpu_backend)?;
    let dtype = gemma4_load_dtype(load);
    #[allow(unused_mut)]
    let (mut model, cfg) =
        gen_loader::load_gen_model(&load.model_dir, dtype, &device, load.defer_towers)?;
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    if load.placement.text {
        gen_loader::pack_text_for_prefill(
            &mut model,
            load.npu_threads,
            &device,
            load.keep_cpu_weights,
            load.placement.decode,
        )?;
    }
    #[cfg(not(all(feature = "npu", target_arch = "aarch64")))]
    if load.placement.any_npu() {
        bail!("an NPU placement requires an aarch64 build with `--features npu`");
    }
    let lm_head = gen_loader::build_lm_head(model.text(), dtype.tensor_dtype(), &device);
    let tokenizer = crate::util::checkpoint::open_in_dir(&load.model_dir)?.tokenizer()?;

    let messages: Vec<chat::Message> = match &args.messages {
        Some(json) => serde_json::from_str::<Vec<MessageJson>>(json)
            .context("parse --messages JSON")?
            .into_iter()
            .map(|m| chat::Message {
                role: m.role,
                content: m.content,
            })
            .collect(),
        None => vec![chat::Message::user(args.input_text()?)],
    };
    let enable_thinking = args.enable_thinking.unwrap_or(false);
    let rendered = chat::render(&messages, enable_thinking);
    // Media inputs expand their `<|image|>` / `<|audio|>` placeholders in the
    // rendered conversation and are encoded with the checkpoint's own towers.
    let (ids, soft) = if args.image.is_some() || args.audio.is_some() {
        gen_loader::load_towers(
            &mut model,
            &load.model_dir,
            &device,
            crate::gemma::TowerSet {
                vision: args.image.is_some(),
                audio: args.audio.is_some(),
            },
        )?;
        let (img_soft, video_soft) = inputs::media_soft_tokens(&load.model_dir);
        let req = inputs::MediaInputs {
            text: Some(rendered.clone()),
            images: args.image.clone().into_iter().collect(),
            video: None,
            audio: args.audio.clone(),
            prompt_prefix: String::new(),
            max_tokens: None,
            image_soft_tokens: args.max_soft_tokens.unwrap_or(img_soft),
            video_soft_tokens: video_soft,
            video_fps: 1.0,
            video_max_frames: 32,
            attn_chunk: load.attn_chunk,
            // The `--vision-attn` A/B is scoped to EmbeddingGemma 2; generation
            // keeps the tower's blocked default.
            vision_attn: crate::gemma::vision::VisionAttn::Blocked,
        };
        let prepared = inputs::prepare(&model, &tokenizer, &req, &inputs::DebugPaths::default())?;
        (prepared.ids, prepared.soft)
    } else {
        (chat::encode(&tokenizer, &rendered)?, None)
    };
    let opts = chat::GenOptions {
        max_new_tokens: args.max_new_tokens.unwrap_or(64),
        greedy: args.sample != Some(true),
        temperature: args.temperature.unwrap_or(1.0),
        top_k: args.top_k.unwrap_or(64),
        top_p: args.top_p.unwrap_or(0.95),
        eos: gen_eos(&load.model_dir),
        collect_top8: args.dump_logits.is_some(),
    };
    println!(
        "model: {} layers, hidden {}, ple_dim {}, eos {:?}",
        cfg.text_config.num_hidden_layers,
        cfg.text_config.hidden_size,
        cfg.text_config.hidden_size_per_layer_input,
        opts.eos
    );
    gen_stage_stats_reset();
    let pad_id = cfg.text_config.pad_token_id.unwrap_or(0);
    let (out, stats) = chat::generate_with_media(
        &model,
        &lm_head,
        &ids,
        soft,
        pad_id,
        &opts,
        load.attn_chunk,
        &device,
    )?;
    if let Some(path) = &args.dump_logits {
        let dump: Vec<serde_json::Value> = stats
            .top8
            .iter()
            .map(|step| {
                serde_json::json!(
                    step.iter()
                        .map(|(id, v)| serde_json::json!([id, v]))
                        .collect::<Vec<_>>()
                )
            })
            .collect();
        std::fs::write(path, serde_json::to_string(&dump)?)
            .with_context(|| format!("write {}", path.display()))?;
        println!(
            "dumped {} steps of top-8 logits to {}",
            stats.top8.len(),
            path.display()
        );
    }
    let (attn, mlp, ple, norms) = gen_stage_stats();
    println!(
        "stages: attn {attn:.2}s, mlp {mlp:.2}s, ple {ple:.2}s, norms {norms:.2}s ({} tokens)",
        stats.prompt_tokens + stats.generated
    );
    let text = tokenizer
        .decode(&out, false)
        .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
    println!(
        "prompt {} tokens, generated {} in {:.2}s (prefill {:.2}s, decode {:.2}s = {:.2} tok/s)",
        stats.prompt_tokens,
        stats.generated,
        stats.prefill_s + stats.decode_s,
        stats.prefill_s,
        stats.decode_s,
        stats.generated as f64 / stats.decode_s.max(1e-9)
    );
    println!("ids: {out:?}");
    println!("text: {text}");
    if let Some(path) = &args.out {
        std::fs::write(path, serde_json::to_string(&out)?)
            .with_context(|| format!("write {}", path.display()))?;
        println!("wrote {}", path.display());
    }
    Ok(())
}
