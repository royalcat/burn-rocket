//! `bench`: embedding throughput over repeated forwards.

use std::time::Instant;

use anyhow::{Result, bail};
use burn::prelude::*;
use burn::tensor::{Int, TensorData};

use crate::cli::{AttnKernel, FlagArgs, LoadArgs, ResolvedLoad};
use crate::commands::embed_input::{self, DebugPaths, Inputs, npu_opts, prepare_measured};
use crate::commands::resolve;
use crate::family::Family;
use crate::gemma::embeddinggemma::load::load_model as emb2_load_model;
use crate::gemma::embeddinggemma::model::{stage_stats, stage_stats_reset};
use crate::qwen3_embedding::load::{NpuOpts as Qwen3NpuOpts, load_model as qwen3_load_model};
use crate::qwen3_embedding::model::{
    stage_stats as qwen3_stage_stats, stage_stats_reset as qwen3_stage_reset,
};
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
        other => bail!("`bench` does not support the {} model", other.name()),
    }
}

struct Args {
    load: LoadArgs,
    inputs: Inputs,
    reps: usize,
    debug: DebugPaths,
}

impl Args {
    fn parse(mut f: FlagArgs) -> Result<Self> {
        let load = LoadArgs::parse(&mut f)?;
        let inputs = Inputs::parse(&mut f)?;
        let debug = DebugPaths::parse(&mut f)?;
        let mut reps = 3;
        if let Some(v) = f.take_parsed("--reps")? {
            reps = v;
        }
        f.finish("bench")?;
        Ok(Self {
            load,
            inputs,
            reps,
            debug,
        })
    }
}

fn validate(load: &ResolvedLoad, args: &Args) -> Result<()> {
    crate::commands::require_embedding(load.family, "bench")?;
    args.inputs.validate(load.family)?;
    if args.inputs.dim.is_some() || !args.inputs.normalize {
        bail!("--dim/--no-normalize control the output vector and only apply to `embed`");
    }
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
    let attn_fused = load.attn_kernel == AttnKernel::Fused;

    qwen3_stage_reset();
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    if load.placement.any_npu() {
        burn_rocket::stats_reset();
    }
    let mut times = Vec::new();
    for i in 0..=args.reps.max(1) {
        let t0 = Instant::now();
        let out = model.forward(input.clone(), &rope, load.chunk, load.key_block, attn_fused);
        let _ = out.to_data();
        device.sync()?;
        let dt = t0.elapsed().as_secs_f64();
        times.push(dt);
        if i == 0 {
            println!("warmup: {dt:.3}s ({:.1} tok/s)", n as f64 / dt);
        } else {
            println!("run {i}: {dt:.3}s ({:.1} tok/s)", n as f64 / dt);
        }
    }
    let best = times.iter().cloned().fold(f64::MAX, f64::min);
    println!(
        "tokens={n} attn={} chunk={} key_block={} dtype={:?} best={best:.3}s => {:.1} tok/s",
        load.attn_kernel.name(),
        load.chunk,
        load.key_block,
        load.dtype,
        n as f64 / best
    );
    let (attn, mlp, norms) = qwen3_stage_stats();
    println!("stages over all runs: attention {attn:.2}s, mlp {mlp:.2}s, norms {norms:.2}s");
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    if load.placement.any_npu() {
        let s = burn_rocket::stats();
        println!(
            "npu breakdown over all runs: {} calls, convert {:.2}s, npu {:.2}s, \
             flex+overhead {:.2}s",
            s.calls,
            s.convert_s,
            s.npu_s,
            times.iter().sum::<f64>() - s.convert_s - s.npu_s
        );
    }
    Ok(())
}

fn run_emb2(load: &ResolvedLoad, args: &Args) -> Result<()> {
    let device = device(&load.cpu_backend)?;
    let (mut model, _cfg) = emb2_load_model(
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

    for rep in 0..args.reps.max(1) {
        stage_stats_reset();
        let t0 = Instant::now();
        let out = model.embed_ids(input.clone(), &rope, load.attn_chunk, soft.clone());
        let _ = out.into_data();
        let wall = t0.elapsed().as_secs_f64();
        let (attn, mlp, norms, ple) = stage_stats();
        println!(
            "rep {rep}: tokens={n} encode={encode_s:.2}s wall={wall:.2}s ({:.1} tok/s) attn={attn:.2}s mlp={mlp:.2}s norms={norms:.2}s ple={ple:.2}s",
            n as f64 / wall
        );
        #[cfg(all(feature = "npu", target_arch = "aarch64"))]
        if load.placement.any_npu() {
            let st = burn_rocket::stats();
            println!(
                "  npu: calls={} convert={:.2}s npu={:.2}s flex+overhead={:.2}s",
                st.calls,
                st.convert_s,
                st.npu_s,
                wall - st.convert_s - st.npu_s
            );
        }
    }
    Ok(())
}
