//! `embed`: one embedding for text and/or media.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Result, bail};
use burn::prelude::*;
use burn::tensor::{DType, Int, TensorData};

use crate::cli::{AttnKernel, FlagArgs, LoadArgs, ResolvedLoad};
use crate::commands::embed_input::{
    self, DebugPaths, Inputs, npu_opts, prepare_measured, write_vector,
};
use crate::commands::resolve;
use crate::family::Family;
use crate::gemma::embeddinggemma::load::load_model as emb2_load_model;
use crate::gemma::embeddinggemma::model::{stage_stats, stage_stats_reset};
use crate::qwen3_embedding::load::{NpuOpts as Qwen3NpuOpts, load_model as qwen3_load_model};
use crate::util::checkpoint::Checkpoint;
use crate::util::device;
use crate::util::rope::RopeCache;

pub fn run(it: impl Iterator<Item = String>) -> Result<()> {
    let mut args = Args::parse(FlagArgs::new(it))?;
    let load = resolve(std::mem::take(&mut args.load))?;
    validate(&load, &args)?;
    match load.family {
        Family::Qwen3Embed => run_qwen3(&load, &args),
        Family::Emb2 => run_emb2(&load, &args),
        other => bail!("`embed` does not support the {} model", other.name()),
    }
}

struct Args {
    load: LoadArgs,
    inputs: Inputs,
    out: Option<PathBuf>,
    debug: DebugPaths,
}

impl Args {
    fn parse(mut f: FlagArgs) -> Result<Self> {
        let load = LoadArgs::parse(&mut f)?;
        let inputs = Inputs::parse(&mut f)?;
        let debug = DebugPaths::parse(&mut f)?;
        let mut out = None;
        if let Some(v) = f.take("--out")? {
            out = Some(PathBuf::from(v));
        }
        f.finish("embed")?;
        Ok(Self {
            load,
            inputs,
            out,
            debug,
        })
    }
}

/// Reject command flags the detected family cannot honour instead of silently
/// ignoring them.
fn validate(load: &ResolvedLoad, args: &Args) -> Result<()> {
    crate::commands::require_embedding(load.family, "embed")?;
    args.inputs.validate(load.family)?;
    if load.family == Family::Qwen3Embed && !args.debug.is_empty() {
        bail!("the --dump-* hooks are EmbeddingGemma 2 only (this checkpoint is qwen3)");
    }
    Ok(())
}

fn run_qwen3(load: &ResolvedLoad, args: &Args) -> Result<()> {
    let device = device(&load.cpu_backend)?;
    let mut ckpt = crate::util::checkpoint::open_in_dir(&load.model_dir)?;
    let npu = load.placement.any_npu().then_some(Qwen3NpuOpts {
        threads: load.npu_threads,
        text: load.placement.text,
        attn: load.placement.text_attn,
    });
    let (model, cfg) = qwen3_load_model(&mut ckpt, load.dtype, load.quant_q8, npu, &device)?;
    let tokenizer = ckpt.tokenizer()?;
    let mut ids = tokenizer
        .encode(args.inputs.text()?.as_str(), true)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?
        .get_ids()
        .to_vec();
    if let Some(n) = args.inputs.tokens {
        ids.truncate(n);
    }
    let n = ids.len();
    if n == 0 {
        bail!("empty input");
    }
    let ids_i64: Vec<i64> = ids.iter().map(|&t| t as i64).collect();
    let input = Tensor::<2, Int>::from_data(TensorData::new(ids_i64, [1, n]), &device);
    let rope = RopeCache::new(n, cfg.head_dim, cfg.rope_theta, load.dtype, &device);

    let t0 = Instant::now();
    let out = model.forward(
        input,
        &rope,
        load.chunk,
        load.key_block,
        load.attn_kernel == AttnKernel::Fused,
    );
    let vec: Vec<f32> = out
        .cast(DType::F32)
        .to_data()
        .try_to_vec()
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let dt = t0.elapsed().as_secs_f64();
    println!("tokens={n} time={dt:.3}s ({:.1} tok/s)", n as f64 / dt);
    // qwen3 embeddings are reported unnormalized; --no-normalize is the default.
    write_vector(args.out.as_deref(), false, None, vec)
}

fn run_emb2(load: &ResolvedLoad, args: &Args) -> Result<()> {
    let device = device(&load.cpu_backend)?;
    let (mut model, cfg) = emb2_load_model(
        &load.model_dir,
        load.dtype,
        load.quant_q8,
        npu_opts(load),
        load.defer_towers,
        &device,
    )?;
    embed_input::maybe_load_towers(load, &args.inputs, &mut model, &device)?;
    let ((input, soft, n), encode_s) =
        prepare_measured(load, &args.inputs, &args.debug, &model, &device)?;
    let rope = model.text().rope_tables(n, load.dtype, &device);

    stage_stats_reset();
    let t0 = Instant::now();
    let embedded = model.embed_ids(input, &rope, load.attn_chunk, soft);
    let v: Vec<f32> = embedded.cast(DType::F32).into_data().try_to_vec()?;
    let elapsed = t0.elapsed().as_secs_f64();
    let (attn, mlp, norms, ple) = stage_stats();
    println!(
        "tokens={n} dim={} encode={encode_s:.2}s forward={elapsed:.2}s (attn {attn:.2}s, mlp {mlp:.2}s, norms {norms:.2}s, ple {ple:.2}s) model layers={} hidden={}",
        args.inputs.dim.unwrap_or(v.len()),
        cfg.text_config.num_hidden_layers,
        cfg.text_config.hidden_size,
    );
    write_vector(
        args.out.as_deref(),
        args.inputs.normalize,
        args.inputs.dim,
        v,
    )
}
