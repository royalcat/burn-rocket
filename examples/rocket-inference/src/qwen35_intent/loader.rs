//! Weight loading for the Qwen3.5-0.8B intent model: reads the HF keys under
//! `model.language_model.*` (ignoring the vision tower and any MTP head) from an
//! [`AnyCheckpoint`] (safetensors or GGUF) and, in NPU mode, packs the projection
//! groups straight into resident NPU memory (`pack`/`pack2`/`pack3`, pack-and-drop).
//! [`IntentLoadOptions::npu_int8`] packs the same groups as group-wise int8 (W8A8)
//! instead of fp16; the fused groups concatenate along N, which is the row order
//! `pack2`/`pack3` use for fp16.
//!
//! The same loader serves the `decision` family (a d3 checkpoint whose text
//! backbone sits at the unprefixed `language_model.*` keys) through
//! [`IntentLoadOptions::prefix`].

use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use burn::module::Param;
use burn::prelude::*;
use burn::tensor::{DType, TensorData};

use crate::qwen35_intent::model::{
    DeltaNet, Embed, GatedAttention, IntentModel, IntentTextConfig, Mixer,
};
use crate::util::checkpoint::{AnyCheckpoint, Checkpoint};
use crate::util::proj::Proj;
use tokenizers::Tokenizer;

#[derive(Debug, Clone)]
pub struct IntentLoadOptions {
    /// Any stage runs on the NPU (creates the engine).
    pub npu: bool,
    /// Pack the prefill projections into resident NPU weights
    /// (aarch64 + `npu` feature only).
    pub text: bool,
    /// Run single-token decode matmuls on the NPU too.
    pub decode: bool,
    /// Keep the f16 CPU projection copies (decode on the CPU needs them; with
    /// decode offloaded they are only a debug/fallback copy).
    pub keep_cpu_weights: bool,
    /// NPU worker threads; read by the aarch64 + `npu` build only.
    #[cfg_attr(not(all(feature = "npu", target_arch = "aarch64")), allow(dead_code))]
    pub npu_threads: usize,
    /// Keep only an f16 copy of the tied embedding/LM-head table (saves ~0.5 GB).
    pub embed_f16: bool,
    /// Store the tied table as per-row symmetric int8 (halves f16 again; the
    /// decision family's default). Mutually exclusive with [`Self::embed_f16`].
    pub embed_int8: bool,
    /// Pack the text projections as group-wise int8 resident NPU weights (the
    /// W8A8 route: one scale per `group`-wide K block, per-row activation
    /// quantization; the group must be a multiple of 32 and divide K). `None` =
    /// fp16, the default everywhere. Independent of [`Self::embed_int8`].
    pub npu_int8: Option<usize>,
    /// Checkpoint key prefix of the text backbone (`model.language_model.` for the
    /// intent SFT, `language_model.` for a d3 decision checkpoint).
    pub prefix: &'static str,
    /// Family label for the ready line.
    pub label: &'static str,
    /// Rotary-table length cap (`None` = the checkpoint's `max_position_embeddings`).
    pub rope_max_seq: Option<usize>,
}

impl Default for IntentLoadOptions {
    fn default() -> Self {
        Self {
            npu: false,
            text: false,
            decode: false,
            keep_cpu_weights: true,
            npu_threads: 5,
            embed_f16: false,
            embed_int8: false,
            npu_int8: None,
            prefix: "model.language_model.",
            label: "intent",
            rope_max_seq: None,
        }
    }
}

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

/// Resident NPU weights packed by one load: the ready line reports their count
/// and packed bytes.
#[derive(Debug, Default)]
struct Packed {
    weights: usize,
    /// `n*k` bytes per int8 weight, `2*n*k` per fp16 weight.
    bytes: usize,
}

#[cfg_attr(not(all(feature = "npu", target_arch = "aarch64")), allow(dead_code))]
impl Packed {
    fn add(&mut self, n: usize, k: usize, int8: Option<usize>) {
        self.weights += 1;
        self.bytes += n * k * if int8.is_some() { 1 } else { 2 };
    }
}

/// Validate a group-wise int8 resident pack: the W8A8 driver stores one scale per
/// `group`-wide K block and its cube scatter pads N to 32, so the group must be a
/// positive multiple of 32, divide K, and N must be 32-aligned.
#[cfg_attr(not(all(feature = "npu", target_arch = "aarch64")), allow(dead_code))]
fn check_int8_pack(key: &str, n: usize, k: usize, group: usize) -> Result<()> {
    if group == 0 || !group.is_multiple_of(32) {
        bail!("--npu-dtype int8: group {group} must be a positive multiple of 32 ({key})");
    }
    if !k.is_multiple_of(group) {
        bail!(
            "--npu-dtype int8: group {group} does not divide {key}'s K={k} \
             (a group must be a multiple of 32 that divides K)"
        );
    }
    if !n.is_multiple_of(32) {
        bail!("--npu-dtype int8: {key} has N={n}, which is not a multiple of 32");
    }
    Ok(())
}

/// A fused projection group's row order: the `[N_i, K]` blocks concatenated along
/// N. `pack2`/`pack3` pack the same order for fp16 (the N blocks are
/// independent), so an int8 fused weight keeps every consumer's slice offsets.
#[cfg_attr(not(all(feature = "npu", target_arch = "aarch64")), allow(dead_code))]
fn cat_group(parts: Vec<Tensor<2>>) -> Tensor<2> {
    Tensor::cat(parts, 0)
}

/// Pack one `[N, K]` (HF convention) weight: group-wise int8 under
/// `--npu-dtype int8`, else fp16.
#[cfg(all(feature = "npu", target_arch = "aarch64"))]
fn npu_pack(
    ckpt: &mut AnyCheckpoint,
    key: &str,
    device: &Device,
    int8: Option<usize>,
    packed: &mut Packed,
) -> Result<burn_rocket::WeightId> {
    let w = read2(ckpt, key, device)?;
    let [n, k] = w.shape().dims::<2>();
    let id = match int8 {
        Some(group) => {
            check_int8_pack(key, n, k, group)?;
            burn_rocket::pack_i8(w, group)
        }
        None => burn_rocket::pack(w),
    };
    packed.add(n, k, int8);
    Ok(id)
}

/// Pack a two-block fused group (`qkv|z`, `gate|up`): one resident weight whose
/// N is `N_a + N_b`.
#[cfg(all(feature = "npu", target_arch = "aarch64"))]
fn npu_pack2(
    ckpt: &mut AnyCheckpoint,
    a: &str,
    b: &str,
    device: &Device,
    int8: Option<usize>,
    packed: &mut Packed,
) -> Result<burn_rocket::WeightId> {
    let wa = read2(ckpt, a, device)?;
    let wb = read2(ckpt, b, device)?;
    let [na, k] = wa.shape().dims::<2>();
    let [nb, kb] = wb.shape().dims::<2>();
    if k != kb {
        bail!("fused group {a} + {b}: K differs ({k} vs {kb})");
    }
    let n = na + nb;
    let id = match int8 {
        Some(group) => {
            check_int8_pack(&format!("{a} + {b}"), n, k, group)?;
            burn_rocket::pack_i8(cat_group(vec![wa, wb]), group)
        }
        None => burn_rocket::pack2(wa, wb),
    };
    packed.add(n, k, int8);
    Ok(id)
}

/// Pack a three-block fused group (`q|k|v`): one resident weight whose N is
/// `N_a + N_b + N_c`.
#[cfg(all(feature = "npu", target_arch = "aarch64"))]
fn npu_pack3(
    ckpt: &mut AnyCheckpoint,
    a: &str,
    b: &str,
    c: &str,
    device: &Device,
    int8: Option<usize>,
    packed: &mut Packed,
) -> Result<burn_rocket::WeightId> {
    let wa = read2(ckpt, a, device)?;
    let wb = read2(ckpt, b, device)?;
    let wc = read2(ckpt, c, device)?;
    let [na, k] = wa.shape().dims::<2>();
    let [nb, kb] = wb.shape().dims::<2>();
    let [nc, kc] = wc.shape().dims::<2>();
    if k != kb || k != kc {
        bail!("fused group {a} + {b} + {c}: K differs ({k} vs {kb} vs {kc})");
    }
    let n = na + nb + nc;
    let id = match int8 {
        Some(group) => {
            check_int8_pack(&format!("{a} + {b} + {c}"), n, k, group)?;
            burn_rocket::pack_i8(cat_group(vec![wa, wb, wc]), group)
        }
        None => burn_rocket::pack3(wa, wb, wc),
    };
    packed.add(n, k, int8);
    Ok(id)
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
    let max_seq = opts.rope_max_seq.unwrap_or(cfg.max_position_embeddings);
    let mut model = IntentModel::new_stub(&cfg, max_seq, device);

    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    if opts.npu {
        burn_rocket::init(opts.npu_threads)
            .map_err(|e| anyhow::anyhow!("NPU context creation failed: {e}"))?;
    }
    #[cfg(not(all(feature = "npu", target_arch = "aarch64")))]
    if opts.npu {
        bail!("--npu requires an aarch64 build with --npu (the burn-rocket NPU feature)");
    }

    let mut packed = Packed::default();
    let prefix = opts.prefix;

    // Global tensors.
    if opts.embed_f16 && opts.embed_int8 {
        bail!("the token table cannot be both f16 and int8 (pick one)");
    }
    if opts.npu_int8.is_some() && !opts.text {
        bail!(
            "npu_int8 is set but the text projections are not packed \
             (`--npu-dtype int8` needs the text stage on the NPU)"
        );
    }
    let embed_key = format!("{prefix}embed_tokens.weight");
    model.embed = if opts.embed_int8 {
        // Per-row symmetric int8. The f32 table is consumed in place (no tensor
        // copy), so the load transient is the table plus the codes — less than
        // the f16 cast's — and only the codes stay resident.
        let data = read_tensor(ckpt, &embed_key)?;
        if data.shape().num_dims() != 2 {
            bail!("{embed_key}: expected a 2-D tensor, got {:?}", data.shape());
        }
        let [rows, cols] = data.shape().dims::<2>();
        let table: Vec<f32> = data
            .try_into_vec()
            .map_err(|e| anyhow::anyhow!("read {embed_key} as f32: {e}"))?;
        Embed::quantize(&table, rows, cols)
    } else {
        let embed = read2(ckpt, &embed_key, device)?;
        if opts.embed_f16 {
            Embed::Dense(embed.cast(DType::F16))
        } else {
            Embed::Dense(embed)
        }
    };
    model.norm_w = read_norm1(ckpt, &format!("{prefix}norm.weight"), device)?;

    for (i, layer) in model.layers.iter_mut().enumerate() {
        let p = |s: &str| format!("{prefix}layers.{i}.{s}");
        layer.input_norm_w = read_norm1(ckpt, &p("input_layernorm.weight"), device)?;
        layer.post_norm_w = read_norm1(ckpt, &p("post_attention_layernorm.weight"), device)?;

        // MLP (every layer). In NPU mode gate|up is one resident weight.
        #[cfg(all(feature = "npu", target_arch = "aarch64"))]
        if opts.text {
            let gateup = npu_pack2(
                ckpt,
                &p("mlp.gate_proj.weight"),
                &p("mlp.up_proj.weight"),
                device,
                opts.npu_int8,
                &mut packed,
            )?;
            let down = npu_pack(
                ckpt,
                &p("mlp.down_proj.weight"),
                device,
                opts.npu_int8,
                &mut packed,
            )?;
            layer.mlp.gateup_fused = Some(gateup);
            layer.mlp.down_fused = Some(down);
        }
        if opts.npu {
            if opts.keep_cpu_weights {
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

    let resident = if opts.npu {
        let kind = match opts.npu_int8 {
            Some(group) => format!("int8 g{group}"),
            None => "fp16".to_string(),
        };
        format!(
            ", {} resident NPU weights ({kind}, {:.1} MiB)",
            packed.weights,
            packed.bytes as f64 / (1u64 << 20) as f64
        )
    } else {
        String::new()
    };
    println!(
        "{} model ready in {:.2}s ({} layers, embed {}{resident})",
        opts.label,
        t0.elapsed().as_secs_f64(),
        cfg.num_hidden_layers,
        model.embed.mode(),
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
    packed: &mut Packed,
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
    if opts.text {
        // in_proj_qkv|z share one input: one resident weight, one matmul.
        let qkvz = npu_pack2(
            ckpt,
            &p("linear_attn.in_proj_qkv.weight"),
            &p("linear_attn.in_proj_z.weight"),
            device,
            opts.npu_int8,
            packed,
        )?;
        let out_id = npu_pack(
            ckpt,
            &p("linear_attn.out_proj.weight"),
            device,
            opts.npu_int8,
            packed,
        )?;
        dn.qkvz_fused = Some(qkvz);
        dn.out_fused = Some(out_id);
    }
    if opts.npu {
        if opts.keep_cpu_weights {
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
    packed: &mut Packed,
) -> Result<()> {
    let _ = &packed;
    fa.q_norm_w = read_norm1(ckpt, &p("self_attn.q_norm.weight"), device)?;
    fa.k_norm_w = read_norm1(ckpt, &p("self_attn.k_norm.weight"), device)?;

    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    if opts.text {
        let qkv = npu_pack3(
            ckpt,
            &p("self_attn.q_proj.weight"),
            &p("self_attn.k_proj.weight"),
            &p("self_attn.v_proj.weight"),
            device,
            opts.npu_int8,
            packed,
        )?;
        let o_id = npu_pack(
            ckpt,
            &p("self_attn.o_proj.weight"),
            device,
            opts.npu_int8,
            packed,
        )?;
        fa.qkv_fused = Some(qkv);
        fa.o_fused = Some(o_id);
    }
    if opts.npu {
        if opts.keep_cpu_weights {
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
) -> Result<LoadedIntent> {
    let mut ckpt = crate::util::checkpoint::open_in_dir(model_dir)?;
    let cfg = IntentTextConfig::from_value(ckpt.config_json()?)?;
    let mut model = load_intent_model(&mut ckpt, device, opts)?;
    model.chunk = delta_chunk.max(1);
    model.npu_only = opts.text && !opts.keep_cpu_weights;
    model.npu_decode = opts.decode;
    let tokenizer = ckpt.tokenizer()?;
    let stop_ids = stop_ids(&ckpt, &tokenizer, &cfg);
    Ok(LoadedIntent {
        model,
        tokenizer,
        stop_ids,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen35_intent::model::Embed;

    /// The int8 branch on the real d3-lite table: the codes and scales are a
    /// finite table of the checkpoint's shape, and every sampled row dequantizes
    /// to finite f32. Skips when the checkpoint is not on this host.
    #[test]
    fn the_int8_table_loads_from_a_real_checkpoint() {
        let dir = Path::new("/tmp/opencode/d3/models/d3-lite");
        if !dir.join("config.json").exists() {
            eprintln!("skipping: {} is not on this host", dir.display());
            return;
        }
        let device = Device::default();
        let mut ckpt = crate::util::checkpoint::open_in_dir(dir).unwrap();
        let cfg = IntentTextConfig::from_value(ckpt.config_json().unwrap()).unwrap();
        let opts = IntentLoadOptions {
            // The d3 package's backbone (`language_model.*`).
            prefix: "language_model.",
            label: "decision-test",
            embed_int8: true,
            ..IntentLoadOptions::default()
        };
        let model = load_intent_model(&mut ckpt, &device, &opts).unwrap();
        assert_eq!(model.embed.mode(), "I8", "the ready line reports the mode");
        match &model.embed {
            Embed::Int8 { codes, scales } => {
                assert_eq!(scales.len(), cfg.vocab_size, "one scale per row");
                assert_eq!(codes.len(), cfg.vocab_size * cfg.hidden_size);
                assert!(
                    scales.iter().all(|s| s.is_finite() && *s >= 0.0),
                    "scales are finite and non-negative"
                );
                assert!(
                    scales.iter().filter(|s| **s > 0.0).count() > cfg.vocab_size / 2,
                    "nearly every row is non-zero"
                );
            }
            Embed::Dense(_) => panic!("expected an int8 table"),
        }
        let ids = [0u32, 1, 2, 100, (cfg.vocab_size - 1) as u32];
        let rows: Vec<f32> = model
            .embed_ids(&ids)
            .into_data()
            .try_to_vec()
            .expect("f32 rows");
        assert_eq!(rows.len(), ids.len() * cfg.hidden_size);
        assert!(rows.iter().all(|v| v.is_finite()), "rows are finite");
        assert!(rows.iter().any(|v| *v != 0.0), "rows are not all zero");
    }

    /// The W8A8 driver takes one scale per `group`-wide K block, stores the codes
    /// as `[N, K/group]` and pads N to 32, so a group that does not divide K (or a
    /// 16-aligned N) must be refused before the pack, with the tensor named.
    #[test]
    fn the_int8_group_must_fit_the_driver() {
        for group in [32, 64, 128, 512, 2048] {
            assert!(
                check_int8_pack("mlp.down_proj", 1024, 2048, group).is_ok(),
                "g{group} divides K=2048 and N=1024 is 32-aligned"
            );
        }
        let message = |n: usize, k: usize, g: usize| {
            format!(
                "{:#}",
                check_int8_pack("mlp.down_proj", n, k, g).unwrap_err()
            )
        };
        assert!(message(1024, 2048, 0).contains("positive multiple of 32"));
        assert!(message(1024, 2048, 48).contains("positive multiple of 32"));
        // K must be a whole number of groups (a ragged last block is not packed).
        let ragged = message(1024, 1000, 64);
        assert!(ragged.contains("does not divide"), "{ragged}");
        assert!(ragged.contains("K=1000"), "{ragged}");
        // N is scattered in 32-row cubes.
        assert!(message(16, 2048, 32).contains("not a multiple of 32"));
    }

    /// A fused group is one resident weight: the checkpoint's `[N_i, K]` blocks
    /// concatenated along N, in the checkpoint's order, which is what `pack2`/
    /// `pack3` do for fp16 and what the consumers' slice offsets assume.
    #[test]
    fn a_fused_group_concatenates_along_n_in_checkpoint_order() {
        let device = Device::default();
        let block = |rows: usize, k: usize, base: f32| {
            let data: Vec<f32> = (0..rows * k).map(|i| base + i as f32).collect();
            Tensor::<2>::from_data(TensorData::new(data, [rows, k]), (&device, DType::F32))
        };
        let a = block(2, 3, 0.0);
        let b = block(1, 3, 100.0);
        let c = block(3, 3, 200.0);
        let fused = cat_group(vec![a.clone(), b.clone(), c.clone()]);
        assert_eq!(fused.shape().dims::<2>(), [6, 3], "N sums, K is shared");
        let values: Vec<f32> = fused.into_data().try_to_vec().expect("f32 rows");
        let mut expected: Vec<f32> = Vec::new();
        for part in [a, b, c] {
            expected.extend(
                part.into_data()
                    .try_to_vec::<f32>()
                    .expect("f32 block rows"),
            );
        }
        assert_eq!(values, expected, "the rows keep the [a; b; c] block order");
    }

    /// The ready line's byte count: `n*k` bytes per int8 weight, `2*n*k` per fp16
    /// weight, summed over the packed groups (96 for d3-lite).
    #[test]
    fn resident_weight_bytes_halve_with_int8() {
        let mut fp16 = Packed::default();
        fp16.add(3584, 1024, None);
        fp16.add(1024, 3584, None);
        assert_eq!(fp16.weights, 2);
        assert_eq!(fp16.bytes, 2 * 2 * 3584 * 1024);

        let mut int8 = Packed::default();
        int8.add(3584, 1024, Some(32));
        int8.add(1024, 3584, Some(32));
        assert_eq!(int8.weights, fp16.weights);
        assert_eq!(int8.bytes, fp16.bytes / 2, "int8 codes are one byte");
    }
}
