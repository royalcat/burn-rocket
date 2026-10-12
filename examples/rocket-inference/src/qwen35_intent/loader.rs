//! Weight loading for the Qwen3.5-0.8B intent model: reads the HF keys under
//! `model.language_model.*` (ignoring the vision tower and any MTP head) from an
//! [`AnyCheckpoint`] (safetensors or GGUF) and, in NPU mode, packs the projection
//! groups straight into resident NPU memory (`pack`/`pack2`/`pack3`, pack-and-drop).

use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use burn::module::Param;
use burn::prelude::*;
use burn::tensor::{DType, TensorData};

use crate::qwen35_intent::model::{DeltaNet, GatedAttention, IntentModel, IntentTextConfig, Mixer};
use crate::util::checkpoint::{AnyCheckpoint, Checkpoint};
use crate::util::proj::Proj;
use tokenizers::Tokenizer;

pub struct IntentLoadOptions {
    /// Pack projections into resident NPU weights (aarch64 + `npu` feature only).
    pub npu: bool,
    /// NPU worker threads; read by the aarch64 + `npu` build only.
    #[cfg_attr(not(all(feature = "npu", target_arch = "aarch64")), allow(dead_code))]
    pub npu_threads: usize,
    /// Keep only an f16 copy of the tied embedding/LM-head table (saves ~0.5 GB).
    pub embed_f16: bool,
    /// With `npu`: do not retain f16 CPU projection copies. Every projection then
    /// runs on the NPU (single-token decode pays the `M >= 256` padding).
    pub pure_npu: bool,
}

impl Default for IntentLoadOptions {
    fn default() -> Self {
        Self {
            npu: false,
            npu_threads: 5,
            embed_f16: false,
            pure_npu: false,
        }
    }
}

const PREFIX: &str = "model.language_model.";

fn read_tensor(ckpt: &mut AnyCheckpoint, key: &str) -> Result<TensorData> {
    ckpt.get(key, DType::F32)?
        .ok_or_else(|| anyhow::anyhow!("missing tensor {key}"))
}

fn read2(ckpt: &mut AnyCheckpoint, key: &str, device: &Device) -> Result<Tensor<2>> {
    let data = read_tensor(ckpt, key)?;
    if data.shape().num_dims() != 2 {
        bail!("{key}: expected a 2-D tensor, got {:?}", data.shape());
    }
    Ok(Tensor::<2>::from_data(data, device))
}

fn read1(ckpt: &mut AnyCheckpoint, key: &str, device: &Device) -> Result<Tensor<1>> {
    let data = read_tensor(ckpt, key)?;
    if data.shape().num_dims() != 1 {
        bail!("{key}: expected a 1-D tensor, got {:?}", data.shape());
    }
    Ok(Tensor::<1>::from_data(data, device))
}

/// An RMSNorm weight: the GGUF converter stores Qwen3.5's transformer norms with
/// a `+1` unit offset (llama.cpp applies `x * (1 + w)`), so undo it.
fn read_norm1(ckpt: &mut AnyCheckpoint, key: &str, device: &Device) -> Result<Tensor<1>> {
    let t = read1(ckpt, key, device)?;
    Ok(if ckpt.format() == "gguf" { t - 1.0 } else { t })
}

/// Read `[N, K]` HF weights and install them as a CPU `Proj`
/// (`Linear.weight` is `[in, out]`, so the checkpoint layout is transposed).
fn cpu_proj(ckpt: &mut AnyCheckpoint, key: &str, device: &Device) -> Result<Proj> {
    let w = read2(ckpt, key, device)?;
    Ok(Proj::Cpu(burn::nn::Linear {
        weight: Param::from_tensor(w.swap_dims(0, 1)),
        bias: None,
    }))
}

/// The f16 flavour: the decode-time CPU copy in `--npu` builds.
fn cpu_proj_f16(ckpt: &mut AnyCheckpoint, key: &str, device: &Device) -> Result<Proj> {
    let w = read2(ckpt, key, device)?;
    Ok(Proj::Cpu(burn::nn::Linear {
        weight: Param::from_tensor(w.swap_dims(0, 1).cast(DType::F16)),
        bias: None,
    }))
}

#[cfg(all(feature = "npu", target_arch = "aarch64"))]
fn npu_pack(ckpt: &mut AnyCheckpoint, key: &str, device: &Device) -> Result<burn_rocket::WeightId> {
    Ok(burn_rocket::pack(read2(ckpt, key, device)?))
}

#[cfg(all(feature = "npu", target_arch = "aarch64"))]
fn npu_pack2(
    ckpt: &mut AnyCheckpoint,
    a: &str,
    b: &str,
    device: &Device,
) -> Result<burn_rocket::WeightId> {
    let wa = read2(ckpt, a, device)?;
    let wb = read2(ckpt, b, device)?;
    Ok(burn_rocket::pack2(wa, wb))
}

#[cfg(all(feature = "npu", target_arch = "aarch64"))]
fn npu_pack3(
    ckpt: &mut AnyCheckpoint,
    a: &str,
    b: &str,
    c: &str,
    device: &Device,
) -> Result<burn_rocket::WeightId> {
    let wa = read2(ckpt, a, device)?;
    let wb = read2(ckpt, b, device)?;
    let wc = read2(ckpt, c, device)?;
    Ok(burn_rocket::pack3(wa, wb, wc))
}

/// Load the intent model from a checkpoint, with its config.
pub fn load_intent_model(
    ckpt: &mut AnyCheckpoint,
    device: &Device,
    opts: &IntentLoadOptions,
) -> Result<IntentModel> {
    let cfg = IntentTextConfig::from_value(ckpt.config_json()?)
        .context("parse intent config from checkpoint")?;
    if cfg.layer_types.len() != cfg.num_hidden_layers {
        bail!(
            "layer_types has {} entries, expected {}",
            cfg.layer_types.len(),
            cfg.num_hidden_layers
        );
    }

    let t0 = Instant::now();
    let mut model = IntentModel::new_stub(&cfg, cfg.max_position_embeddings, device);

    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    if opts.npu {
        burn_rocket::init(opts.npu_threads)
            .map_err(|e| anyhow::anyhow!("NPU context creation failed: {e}"))?;
    }
    #[cfg(not(all(feature = "npu", target_arch = "aarch64")))]
    if opts.npu {
        bail!("--npu requires an aarch64 build with --npu (the burn-rocket NPU feature)");
    }

    let mut packed = 0usize;

    // Global tensors.
    let embed = read2(ckpt, &format!("{PREFIX}embed_tokens.weight"), device)?;
    model.embed = if opts.embed_f16 {
        embed.cast(DType::F16)
    } else {
        embed
    };
    model.norm_w = read_norm1(ckpt, &format!("{PREFIX}norm.weight"), device)?;

    for (i, layer) in model.layers.iter_mut().enumerate() {
        let p = |s: &str| format!("{PREFIX}layers.{i}.{s}");
        layer.input_norm_w = read_norm1(ckpt, &p("input_layernorm.weight"), device)?;
        layer.post_norm_w = read_norm1(ckpt, &p("post_attention_layernorm.weight"), device)?;

        // MLP (every layer). In NPU mode gate|up is one resident weight.
        #[cfg(all(feature = "npu", target_arch = "aarch64"))]
        if opts.npu {
            let gateup = npu_pack2(
                ckpt,
                &p("mlp.gate_proj.weight"),
                &p("mlp.up_proj.weight"),
                device,
            )?;
            let down = npu_pack(ckpt, &p("mlp.down_proj.weight"), device)?;
            layer.mlp.gateup_fused = Some(gateup);
            layer.mlp.down_fused = Some(down);
            packed += 2;
        }
        if opts.npu {
            if !opts.pure_npu {
                layer.mlp.gate_proj = cpu_proj_f16(ckpt, &p("mlp.gate_proj.weight"), device)?;
                layer.mlp.up_proj = cpu_proj_f16(ckpt, &p("mlp.up_proj.weight"), device)?;
                layer.mlp.down_proj = cpu_proj_f16(ckpt, &p("mlp.down_proj.weight"), device)?;
            }
        } else {
            layer.mlp.gate_proj = cpu_proj(ckpt, &p("mlp.gate_proj.weight"), device)?;
            layer.mlp.up_proj = cpu_proj(ckpt, &p("mlp.up_proj.weight"), device)?;
            layer.mlp.down_proj = cpu_proj(ckpt, &p("mlp.down_proj.weight"), device)?;
        }

        match &mut layer.mixer {
            Mixer::Linear(dn) => load_linear(dn, ckpt, &p, device, opts, &mut packed)?,
            Mixer::Full(fa) => load_full(fa, ckpt, &p, device, opts, &mut packed)?,
        }
    }

    println!(
        "intent model ready in {:.2}s ({} layers, embed {:?}{})",
        t0.elapsed().as_secs_f64(),
        cfg.num_hidden_layers,
        model.embed.dtype(),
        if opts.npu {
            format!(", {packed} resident NPU weights")
        } else {
            String::new()
        }
    );
    Ok(model)
}

#[allow(clippy::too_many_arguments)]
fn load_linear(
    dn: &mut DeltaNet,
    ckpt: &mut AnyCheckpoint,
    p: &impl Fn(&str) -> String,
    device: &Device,
    opts: &IntentLoadOptions,
    packed: &mut usize,
) -> Result<()> {
    let _ = &packed;
    dn.conv_w = {
        let data = read_tensor(ckpt, &p("linear_attn.conv1d.weight"))?;
        // Safetensors stores [conv_dim, 1, kernel]; GGUF (reversed) gives the
        // equivalent 2-D [conv_dim, kernel].
        match data.shape().num_dims() {
            2 => Tensor::<2>::from_data(data, device),
            3 => {
                let [c, _, k] = data.shape().dims::<3>();
                Tensor::<3>::from_data(data, device).reshape([c, k])
            }
            n => bail!("conv1d.weight: expected 2-D or 3-D, got {n}-D"),
        }
    };
    dn.dt_bias = read1(ckpt, &p("linear_attn.dt_bias"), device)?;
    dn.a_log = read1(ckpt, &p("linear_attn.A_log"), device)?;
    // GGUF's `ssm_a` stores `-exp(A_log)` (llama.cpp applies it directly); the
    // model expects `A_log`, so recover it.
    if ckpt.format() == "gguf" {
        dn.a_log = dn.a_log.clone().neg().log();
    }
    dn.norm_w = read1(ckpt, &p("linear_attn.norm.weight"), device)?;
    // a/b are tiny; they always run on the CPU.
    dn.in_proj_a = cpu_proj(ckpt, &p("linear_attn.in_proj_a.weight"), device)?;
    dn.in_proj_b = cpu_proj(ckpt, &p("linear_attn.in_proj_b.weight"), device)?;

    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    if opts.npu {
        // in_proj_qkv|z share one input: one resident weight, one matmul.
        let qkvz = npu_pack2(
            ckpt,
            &p("linear_attn.in_proj_qkv.weight"),
            &p("linear_attn.in_proj_z.weight"),
            device,
        )?;
        let out_id = npu_pack(ckpt, &p("linear_attn.out_proj.weight"), device)?;
        dn.qkvz_fused = Some(qkvz);
        dn.out_fused = Some(out_id);
        *packed += 2;
    }
    if opts.npu {
        if !opts.pure_npu {
            dn.in_proj_qkv = cpu_proj_f16(ckpt, &p("linear_attn.in_proj_qkv.weight"), device)?;
            dn.in_proj_z = cpu_proj_f16(ckpt, &p("linear_attn.in_proj_z.weight"), device)?;
            dn.out_proj = cpu_proj_f16(ckpt, &p("linear_attn.out_proj.weight"), device)?;
        }
    } else {
        dn.in_proj_qkv = cpu_proj(ckpt, &p("linear_attn.in_proj_qkv.weight"), device)?;
        dn.in_proj_z = cpu_proj(ckpt, &p("linear_attn.in_proj_z.weight"), device)?;
        dn.out_proj = cpu_proj(ckpt, &p("linear_attn.out_proj.weight"), device)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn load_full(
    fa: &mut GatedAttention,
    ckpt: &mut AnyCheckpoint,
    p: &impl Fn(&str) -> String,
    device: &Device,
    opts: &IntentLoadOptions,
    packed: &mut usize,
) -> Result<()> {
    let _ = &packed;
    fa.q_norm_w = read_norm1(ckpt, &p("self_attn.q_norm.weight"), device)?;
    fa.k_norm_w = read_norm1(ckpt, &p("self_attn.k_norm.weight"), device)?;

    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    if opts.npu {
        let qkv = npu_pack3(
            ckpt,
            &p("self_attn.q_proj.weight"),
            &p("self_attn.k_proj.weight"),
            &p("self_attn.v_proj.weight"),
            device,
        )?;
        let o_id = npu_pack(ckpt, &p("self_attn.o_proj.weight"), device)?;
        fa.qkv_fused = Some(qkv);
        fa.o_fused = Some(o_id);
        *packed += 2;
    }
    if opts.npu {
        if !opts.pure_npu {
            fa.q_proj = cpu_proj_f16(ckpt, &p("self_attn.q_proj.weight"), device)?;
            fa.k_proj = cpu_proj_f16(ckpt, &p("self_attn.k_proj.weight"), device)?;
            fa.v_proj = cpu_proj_f16(ckpt, &p("self_attn.v_proj.weight"), device)?;
            fa.o_proj = cpu_proj_f16(ckpt, &p("self_attn.o_proj.weight"), device)?;
        }
    } else {
        fa.q_proj = cpu_proj(ckpt, &p("self_attn.q_proj.weight"), device)?;
        fa.k_proj = cpu_proj(ckpt, &p("self_attn.k_proj.weight"), device)?;
        fa.v_proj = cpu_proj(ckpt, &p("self_attn.v_proj.weight"), device)?;
        fa.o_proj = cpu_proj(ckpt, &p("self_attn.o_proj.weight"), device)?;
    }
    Ok(())
}

/// EOS ids: the checkpoint's special ids, falling back to `text_config.eos_token_id`.
pub fn eos_ids(ckpt: &AnyCheckpoint, cfg: &IntentTextConfig) -> Vec<u32> {
    let fallback = || {
        cfg.eos_token_id
            .as_ref()
            .and_then(serde_json::Value::as_u64)
            .map(|v| v as u32)
            .into_iter()
            .collect::<Vec<u32>>()
    };
    let ids = Checkpoint::special_ids(ckpt).eos;
    if ids.is_empty() { fallback() } else { ids }
}

/// Stop ids for generation: EOS plus the ChatML stop specials.
pub fn stop_ids(ckpt: &AnyCheckpoint, tokenizer: &Tokenizer, cfg: &IntentTextConfig) -> Vec<u32> {
    let mut ids = eos_ids(ckpt, cfg);
    for tok in ["<|im_end|>", "<|im_start|>", "<|endoftext|>"] {
        if let Some(id) = tokenizer.token_to_id(tok) {
            ids.push(id);
        }
    }
    ids.sort_unstable();
    ids.dedup();
    ids
}

/// A loaded intent model ready to serve.
pub struct LoadedIntent {
    pub model: IntentModel,
    pub tokenizer: Tokenizer,
    pub stop_ids: Vec<u32>,
}

/// Load the intent model for serving: same options as one-shot `gen`, plus the
/// decode-side settings and the stop-id set. Opens its checkpoint from `model_dir`.
pub fn load_for_serving(
    model_dir: &Path,
    device: &Device,
    opts: &IntentLoadOptions,
    delta_chunk: usize,
    pure_npu: bool,
    npu_decode: bool,
) -> Result<LoadedIntent> {
    let mut ckpt = crate::util::checkpoint::open_in_dir(model_dir)?;
    let cfg = IntentTextConfig::from_value(ckpt.config_json()?)?;
    let mut model = load_intent_model(&mut ckpt, device, opts)?;
    model.chunk = delta_chunk.max(1);
    model.npu_only = pure_npu;
    model.npu_decode = npu_decode;
    let tokenizer = ckpt.tokenizer()?;
    let stop_ids = stop_ids(&ckpt, &tokenizer, &cfg);
    Ok(LoadedIntent {
        model,
        tokenizer,
        stop_ids,
    })
}
