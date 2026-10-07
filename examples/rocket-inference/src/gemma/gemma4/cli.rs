//! `gemma gen|serve-chat` subcommands: the Gemma 4 E2B-it generation model.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
#[cfg(all(feature = "npu", target_arch = "aarch64"))]
use burn::prelude::Device;
use tokenizers::Tokenizer;

use crate::cli::FlagArgs;
use crate::gemma::gemma4::chat::{self, gen_eos};
use crate::gemma::gemma4::loader::{self, LoadDtype};
#[cfg(all(feature = "npu", target_arch = "aarch64"))]
use crate::gemma::gemma4::model::GenRoot;
use crate::gemma::gemma4::model::{gen_stage_stats, gen_stage_stats_reset};
use crate::gemma::gemma4::server::{self, ChatServeOptions};
use crate::gemma::inputs;
use crate::util::device;

pub fn run(cmd: &str, it: impl Iterator<Item = String>) -> Result<()> {
    let args = Args::parse(cmd, FlagArgs::new(it))?;
    match cmd {
        "gen" => run_gen(&args),
        "serve-chat" => run_serve_chat(&args),
        _ => bail!("unknown gemma4 command '{cmd}'"),
    }
}

struct Args {
    gen_model_dir: PathBuf,
    backend: String,
    text: Option<String>,
    text_file: Option<PathBuf>,
    image: Option<PathBuf>,
    audio: Option<PathBuf>,
    max_soft_tokens: Option<usize>,
    video_fps: f64,
    video_max_frames: usize,
    attn_chunk: usize,
    dump_logits: Option<PathBuf>,
    out: Option<PathBuf>,
    port: u16,
    model_name: String,
    quant_q8: bool,
    npu: bool,
    npu_threads: usize,
    messages: Option<String>,
    max_new_tokens: usize,
    greedy: bool,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    enable_thinking: bool,
    gen_f16: bool,
}

impl Args {
    fn parse(cmd: &str, mut f: FlagArgs) -> Result<Self> {
        let mut args = Args {
            gen_model_dir: PathBuf::from("/mnt/hub/models/gemma-4-E2B-it"),
            backend: if cfg!(feature = "cpu") { "cpu" } else { "flex" }.to_string(),
            text: None,
            text_file: None,
            image: None,
            audio: None,
            max_soft_tokens: None,
            video_fps: 1.0,
            video_max_frames: 32,
            attn_chunk: 1024,
            dump_logits: None,
            out: None,
            port: 8391,
            model_name: "embeddinggemma-2".to_string(),
            quant_q8: false,
            npu: false,
            npu_threads: 5,
            messages: None,
            max_new_tokens: 64,
            greedy: true,
            temperature: 1.0,
            top_k: 64,
            top_p: 0.95,
            enable_thinking: false,
            gen_f16: false,
        };
        if let Some(v) = f.take("--gen-model-dir")? {
            args.gen_model_dir = PathBuf::from(v);
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
        if let Some(v) = f.take("--image")? {
            args.image = Some(PathBuf::from(v));
        }
        if let Some(v) = f.take("--audio")? {
            args.audio = Some(PathBuf::from(v));
        }
        if let Some(v) = f.take_parsed("--max-soft-tokens")? {
            args.max_soft_tokens = Some(v);
        }
        if let Some(v) = f.take_parsed("--video-fps")? {
            args.video_fps = v;
        }
        if let Some(v) = f.take_parsed("--video-max-frames")? {
            args.video_max_frames = v;
        }
        if let Some(v) = f.take_parsed("--attn-chunk")? {
            args.attn_chunk = v;
        }
        if let Some(v) = f.take("--dump-logits")? {
            args.dump_logits = Some(PathBuf::from(v));
        }
        if let Some(v) = f.take("--out")? {
            args.out = Some(PathBuf::from(v));
        }
        if let Some(v) = f.take_parsed("--port")? {
            args.port = v;
        }
        if let Some(v) = f.take("--model-name")? {
            args.model_name = v;
        }
        if let Some(v) = f.take_choice("--quant", &["none", "q8"])? {
            args.quant_q8 = v == "q8";
        }
        args.npu = f.take_bool("--npu");
        if let Some(v) = f.take_parsed("--npu-threads")? {
            args.npu_threads = v;
        }
        if let Some(v) = f.take("--messages")? {
            args.messages = Some(v);
        }
        if let Some(v) = f.take_parsed("--max-new-tokens")? {
            args.max_new_tokens = v;
        }
        args.greedy = !f.take_bool("--sample");
        if let Some(v) = f.take_parsed("--temperature")? {
            args.temperature = v;
        }
        if let Some(v) = f.take_parsed("--top-k")? {
            args.top_k = v;
        }
        if let Some(v) = f.take_parsed("--top-p")? {
            args.top_p = v;
        }
        args.enable_thinking = f.take_bool("--enable-thinking");
        args.gen_f16 = f.take_bool("--f16");
        f.finish(&format!("gemma {cmd}"))?;
        Ok(args)
    }

    fn input_text(&self) -> Result<String> {
        if let Some(t) = &self.text {
            return Ok(t.clone());
        }
        if let Some(path) = &self.text_file {
            return std::fs::read_to_string(path)
                .with_context(|| format!("read text from {}", path.display()));
        }
        bail!("provide --text or --text-file")
    }
}

/// Pack the generation model's text projections into resident fp16 NPU weights
/// for the prefill pass. `keep_cpu` retains the f32 copies (decode runs on the
/// CPU); with `keep_cpu == false` the CPU copies are dropped.
#[cfg(all(feature = "npu", target_arch = "aarch64"))]
fn pack_gen_text(
    model: &mut GenRoot,
    threads: usize,
    device: &Device,
    keep_cpu: bool,
) -> Result<()> {
    use crate::gemma::layers;
    burn_rocket::init(threads)
        .map_err(|e| anyhow::anyhow!("NPU context creation failed: {e}"))?;
    layers::set_npu_prefill_only(keep_cpu);
    let t0 = std::time::Instant::now();
    let mut count = 0usize;
    let mut bytes = 0usize;
    model.text_mut().for_each_projection_mut(|lin| {
        let (n, k) = layers::pack_linear_into_npu(lin, device, keep_cpu);
        count += 1;
        bytes += n * k * 2;
    });
    if !keep_cpu {
        #[cfg(all(target_os = "linux", target_env = "gnu"))]
        unsafe {
            libc::malloc_trim(0);
        }
    }
    println!(
        "NPU: packed {count} text projections ({:.2} GiB f16) in {:.2}s; \
         prefill on the NPU, decode on the CPU{} (resident {:.0} MiB anon)",
        bytes as f64 / (1u64 << 30) as f64,
        t0.elapsed().as_secs_f64(),
        if keep_cpu { " (CPU copies kept)" } else { "" },
        crate::util::rss_mib()
    );
    Ok(())
}

#[derive(serde::Deserialize)]
struct MessageJson {
    role: String,
    content: String,
}

fn run_gen(args: &Args) -> Result<()> {
    let device = device(&args.backend)?;
    let dtype = if args.quant_q8 {
        LoadDtype::Q8
    } else if args.gen_f16 {
        LoadDtype::F16
    } else {
        LoadDtype::F32
    };
    #[allow(unused_mut)]
    let (mut model, cfg) = loader::load_gen_model(&args.gen_model_dir, dtype, &device)?;
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    if args.npu {
        pack_gen_text(&mut model, args.npu_threads, &device, true)?;
    }
    #[cfg(not(all(feature = "npu", target_arch = "aarch64")))]
    if args.npu {
        bail!("--npu requires an aarch64 build with --features npu");
    }
    let lm_head = loader::build_lm_head(model.text(), dtype != LoadDtype::F32, &device);
    let tokenizer = Tokenizer::from_file(args.gen_model_dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;

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
    let rendered = chat::render(&messages, args.enable_thinking);
    // Media inputs expand their `<|image|>` / `<|audio|>` placeholders in the
    // rendered conversation and are encoded with the checkpoint's own towers.
    let (ids, soft) = if args.image.is_some() || args.audio.is_some() {
        let (img_soft, video_soft) = inputs::media_soft_tokens(&args.gen_model_dir);
        let req = inputs::MediaInputs {
            text: Some(rendered.clone()),
            image: args.image.clone(),
            video: None,
            audio: args.audio.clone(),
            prompt_prefix: String::new(),
            max_tokens: None,
            image_soft_tokens: args.max_soft_tokens.unwrap_or(img_soft),
            video_soft_tokens: video_soft,
            video_fps: args.video_fps,
            video_max_frames: args.video_max_frames,
            attn_chunk: args.attn_chunk,
        };
        let prepared =
            inputs::prepare(&model, &tokenizer, &req, &inputs::DebugPaths::default())?;
        (prepared.ids, prepared.soft)
    } else {
        (chat::encode(&tokenizer, &rendered)?, None)
    };
    if std::env::var("DUMP_RENDER").is_ok() {
        println!("rendered: {rendered:?}");
        println!("prompt ids: {ids:?}");
    }
    let opts = chat::GenOptions {
        max_new_tokens: args.max_new_tokens,
        greedy: args.greedy,
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        eos: gen_eos(&args.gen_model_dir),
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
        args.attn_chunk,
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
        println!("dumped {} steps of top-8 logits to {}", stats.top8.len(), path.display());
    }
    let (attn, mlp, ple, norms) = gen_stage_stats();
    println!(
        "stages: attn {attn:.2}s, mlp {mlp:.2}s, ple {ple:.2}s, norms {norms:.2}s ({} tokens)",
        stats.prompt_tokens + stats.generated
    );
    let text = tokenizer
        .decode(&out, false)
        .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
    println!("prompt {} tokens, generated {} in {:.2}s (prefill {:.2}s, decode {:.2}s = {:.2} tok/s)",
        stats.prompt_tokens, stats.generated, stats.prefill_s + stats.decode_s,
        stats.prefill_s, stats.decode_s,
        stats.generated as f64 / stats.decode_s.max(1e-9));
    println!("ids: {out:?}");
    println!("text: {text}");
    if let Some(path) = &args.out {
        std::fs::write(path, serde_json::to_string(&out)?)
            .with_context(|| format!("write {}", path.display()))?;
        println!("wrote {}", path.display());
    }
    Ok(())
}

fn run_serve_chat(args: &Args) -> Result<()> {
    let device = device(&args.backend)?;
    let dtype = if args.quant_q8 {
        LoadDtype::Q8
    } else if args.gen_f16 {
        LoadDtype::F16
    } else {
        LoadDtype::F32
    };
    #[allow(unused_mut)]
    let (mut model, _cfg) = loader::load_gen_model(&args.gen_model_dir, dtype, &device)?;
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    if args.npu {
        pack_gen_text(&mut model, args.npu_threads, &device, true)?;
    }
    #[cfg(not(all(feature = "npu", target_arch = "aarch64")))]
    if args.npu {
        bail!("--npu requires an aarch64 build with --features npu");
    }
    let lm_head = loader::build_lm_head(model.text(), dtype != LoadDtype::F32, &device);
    let tokenizer = Tokenizer::from_file(args.gen_model_dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    let model_name = if args.model_name == "embeddinggemma-2" {
        // The chat server's default name should match the generation model.
        args.gen_model_dir
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "gemma-4-E2B-it".to_string())
    } else {
        args.model_name.clone()
    };
    server::serve(
        model,
        lm_head,
        tokenizer,
        device,
        ChatServeOptions {
            addr: format!("0.0.0.0:{}", args.port),
            model_name,
            model_dir: args.gen_model_dir.clone(),
            attn_chunk: args.attn_chunk,
            max_new_tokens: args.max_new_tokens,
        },
    )
}
