//! Qwen3.5-0.8B text-only inference (the base of `guoxuter/ov_intent_analysis_sft`).
//!
//! Architecture (from the HF `config.json` `text_config`):
//! 24 decoder layers mixing **Gated DeltaNet** linear attention (18 layers) with
//! **gated full attention** (6 layers, `full_attention_interval: 4`), hidden 1024,
//! intermediate 3584, vocab 248320 (tied embeddings), head_dim 256 for full
//! attention (8 q / 2 kv heads, partial RoPE 64/256) and 16 heads x 128 for the
//! delta rule. Norms are zero-centered (`(1 + weight)`); the full-attention layer
//! gates its output with `sigmoid(gate)`; the linear layer uses a depthwise
//! causal conv (kernel 4) and a gated RMSNorm.
//!
//! The port follows `transformers.models.qwen3_5.modeling_qwen3_5` (v5.2) exactly:
//! chunked gated delta rule (chunk 64) for prefill and the recurrent form for
//! single-token decode, all delta-rule math in f32.

use std::time::Instant;

use anyhow::Context;
use burn::prelude::*;
use burn::tensor::activation::{sigmoid, silu, softmax, softplus};
use burn::tensor::module::attention;
use burn::tensor::ops::{AttentionModuleOptions, PadMode};
use burn::tensor::{Bool, DType, Int, TensorData, s};

use crate::server::telemetry;
use crate::util::proj::Proj;
use crate::util::rope::RopeCache;

/// Model hyper-parameters, deserialized from the HF `config.json` `text_config`.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct IntentTextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    #[serde(default = "default_hidden_act")]
    #[allow(dead_code)]
    pub hidden_act: String,
    pub max_position_embeddings: usize,
    pub layer_types: Vec<String>,
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    #[serde(default = "one")]
    pub partial_rotary_factor: f64,
    pub rope_parameters: RopeParams,
    #[serde(default)]
    #[allow(dead_code)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
}

fn default_hidden_act() -> String {
    "silu".to_string()
}

fn one() -> f64 {
    1.0
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct RopeParams {
    pub rope_theta: f64,
    #[serde(default)]
    pub partial_rotary_factor: Option<f64>,
    #[serde(default)]
    #[allow(dead_code)]
    pub mrope_section: Option<Vec<usize>>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct IntentConfigFile {
    pub text_config: IntentTextConfig,
}

impl IntentTextConfig {
    /// Parse the text config from an HF-style JSON value: the full checkpoint
    /// config with a nested `text_config` (the same shape the GGUF backend
    /// fetches/synthesizes).
    pub fn from_value(value: serde_json::Value) -> anyhow::Result<Self> {
        let file: IntentConfigFile =
            serde_json::from_value(value).context("parse intent text_config")?;
        Ok(file.text_config)
    }

    /// Number of rotary dims: `head_dim * partial_rotary_factor` (64 for this model).
    pub fn rotary_dim(&self) -> usize {
        let f = self
            .rope_parameters
            .partial_rotary_factor
            .unwrap_or(self.partial_rotary_factor);
        (self.head_dim as f64 * f).round() as usize
    }

    pub fn key_dim(&self) -> usize {
        self.linear_num_key_heads * self.linear_key_head_dim
    }

    pub fn value_dim(&self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }

    pub fn conv_dim(&self) -> usize {
        self.key_dim() * 2 + self.value_dim()
    }
}

/// Partial RoPE: applies the rotation to the first `rotary` dims and passes the
/// rest through, using the shared [`RopeCache`] tables.
#[derive(Debug)]
pub struct PartialRope {
    inner: RopeCache,
    rotary: usize,
}

impl PartialRope {
    pub fn new(max_seq: usize, rotary: usize, theta: f64, dtype: DType, device: &Device) -> Self {
        Self {
            inner: RopeCache::new(max_seq, rotary, theta, dtype, device),
            rotary,
        }
    }

    /// `x` is `[batch, seq, heads, head_dim]`; `start` is the absolute position of
    /// the first row.
    pub fn apply(&self, x: Tensor<4>, start: usize) -> Tensor<4> {
        let d = x.dims()[3];
        if self.rotary >= d {
            return self.inner.apply(x, start);
        }
        let xr = x.clone().slice(s![.., .., .., 0..self.rotary]);
        let xp = x.slice(s![.., .., .., self.rotary..d]);
        Tensor::cat(vec![self.inner.apply(xr, start), xp], 3)
    }
}

/// Explicit rotary tables for one request: `cos`/`sin` of shape `[seq, rotary]`.
///
/// The d3 image prompt needs positions that are not a plain `0..seq` ramp (the
/// multimodal RoPE mixes a temporal/height/width index per token), so the caller
/// builds the tables for the whole prompt and hands them to the forward instead
/// of the [`PartialRope`] cache. `rotary` is the model's partial-rotation width
/// (`head_dim * partial_rotary_factor`, 64 for d3): the table covers the first
/// `rotary` dims of the head and the tail passes through untouched, exactly like
/// the reference's `apply_rotary_pos_emb` with a partial cos/sin.
#[derive(Debug)]
pub struct RopeTables {
    cos: Tensor<2>,
    sin: Tensor<2>,
}

impl RopeTables {
    /// Wrap a `cos`/`sin` pair of the same `[seq, rotary]` shape (f32).
    pub fn new(cos: Tensor<2>, sin: Tensor<2>) -> Self {
        assert_eq!(
            cos.dims(),
            sin.dims(),
            "the rope cos/sin tables must share one shape"
        );
        let [seq, rotary] = cos.dims();
        assert!(
            seq > 0 && rotary > 0 && rotary.is_multiple_of(2),
            "bad rope table shape {seq}x{rotary}"
        );
        Self { cos, sin }
    }

    /// Table rows (prompt length it was built for).
    pub fn len(&self) -> usize {
        self.cos.dims()[0]
    }

    /// Rotated head dims (`head_dim * partial_rotary_factor`).
    pub fn rotary(&self) -> usize {
        self.cos.dims()[1]
    }

    /// Whether the table covers no tokens (an empty prompt can never be run).
    #[allow(dead_code)] // the `len`/`is_empty` pair clippy wants; nothing builds an empty table
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// `x * cos + rotate_half(x) * sin` over the first `rotary` dims of a
    /// `[batch, seq, heads, head_dim]` tensor; the rest of the head passes
    /// through.
    pub fn apply(&self, x: Tensor<4>) -> Tensor<4> {
        let [_, s, _, d] = x.dims();
        let rotary = self.rotary();
        assert_eq!(
            s,
            self.len(),
            "the rope table has {} rows for {s} tokens",
            self.len()
        );
        assert!(
            rotary <= d,
            "the rope table covers {rotary} of {d} head dims"
        );
        if rotary == d {
            return self.rotate(x);
        }
        let head = self.rotate(x.clone().slice(s![.., .., .., 0..rotary]));
        let tail = x.slice(s![.., .., .., rotary..d]);
        Tensor::cat(vec![head, tail], 3)
    }

    /// The rotation on a tensor whose last dim is exactly `rotary`.
    fn rotate(&self, x: Tensor<4>) -> Tensor<4> {
        let [_, s, _, rotary] = x.dims();
        let half = rotary / 2;
        let cos = self.cos.clone().reshape([1, s, 1, rotary]).cast(x.dtype());
        let sin = self.sin.clone().reshape([1, s, 1, rotary]).cast(x.dtype());
        let x1 = x.clone().slice(s![.., .., .., 0..half]);
        let x2 = x.clone().slice(s![.., .., .., half..rotary]);
        let rotated = Tensor::cat(vec![x2.neg(), x1], 3);
        x * cos + rotated * sin
    }
}

// ---------------------------------------------------------------------------
// Caches
// ---------------------------------------------------------------------------

/// Per-request recurrent state: conv history and delta-rule state for the linear
/// layers, K/V for the full-attention layers.
#[derive(Debug, Default)]
pub struct IntentCache {
    /// Number of tokens already processed.
    pub len: usize,
    conv: Vec<Option<Tensor<3>>>,            // [1, conv_dim, kernel-1]
    rec: Vec<Option<Tensor<4>>>,             // [1, n_v_heads, k_head_dim, v_head_dim]
    kv: Vec<Option<(Tensor<4>, Tensor<4>)>>, // ([1, kv, S, D], [1, kv, S, D])
}

impl IntentCache {
    pub fn new(n_layers: usize) -> Self {
        Self {
            len: 0,
            conv: (0..n_layers).map(|_| None).collect(),
            rec: (0..n_layers).map(|_| None).collect(),
            kv: (0..n_layers).map(|_| None).collect(),
        }
    }

    pub fn reset(&mut self) {
        self.len = 0;
        for c in &mut self.conv {
            *c = None;
        }
        for r in &mut self.rec {
            *r = None;
        }
        for k in &mut self.kv {
            *k = None;
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Zero-centered RMSNorm over the last dim: `x * rsqrt(mean(x^2) + eps) * (1 + w)`.
fn zc_norm3(x: Tensor<3>, w: &Tensor<1>, eps: f64) -> Tensor<3> {
    let dt = x.dtype();
    let xf = x.cast(DType::F32);
    let d = xf.dims()[2];
    let ms = (xf.clone() * xf.clone()).mean_dim(2);
    let inv = (ms + eps).sqrt().recip();
    let w = w
        .clone()
        .cast(DType::F32)
        .reshape([1, 1, d])
        .add_scalar(1.0);
    ((xf * inv) * w).cast(dt)
}

/// Zero-centered RMSNorm over a `[b, s, heads, head_dim]` tensor.
fn zc_norm4(x: Tensor<4>, w: &Tensor<1>, eps: f64) -> Tensor<4> {
    let dt = x.dtype();
    let xf = x.cast(DType::F32);
    let d = xf.dims()[3];
    let ms = (xf.clone() * xf.clone()).mean_dim(3);
    let inv = (ms + eps).sqrt().recip();
    let w = w
        .clone()
        .cast(DType::F32)
        .reshape([1, 1, 1, d])
        .add_scalar(1.0);
    ((xf * inv) * w).cast(dt)
}

/// A resident NPU weight handle when the NPU feature is on; `Option<()>`
/// otherwise, so the same struct fields compile on every target.
#[cfg(all(feature = "npu", target_arch = "aarch64"))]
pub type NpuId = Option<burn_rocket::WeightId>;
#[cfg(not(all(feature = "npu", target_arch = "aarch64")))]
pub type NpuId = Option<()>;

/// Select a projection's backend: its resident NPU weight when `npu` is set and
/// a handle exists, else the CPU copy.
fn proj_sel(cpu: &Proj, id: &NpuId, x: Tensor<3>, npu: bool) -> Tensor<3> {
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    if npu && let Some(id) = id {
        return burn_rocket::matmul(x, id);
    }
    #[cfg(not(all(feature = "npu", target_arch = "aarch64")))]
    let _ = (id, npu);
    proj3(cpu, x, false)
}

/// Run one projection: a resident NPU weight (prefill) or a CPU copy (f16 or
/// f32). The CPU copy exists so single-token decode never pays the NPU's
/// `M >= 256` padding.
fn proj3(p: &Proj, x: Tensor<3>, npu: bool) -> Tensor<3> {
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    if npu && let Proj::Npu(id) = p {
        return burn_rocket::matmul(x, id);
    }
    #[cfg(not(all(feature = "npu", target_arch = "aarch64")))]
    let _ = npu;
    match p {
        Proj::Cpu(lin) => {
            let w = lin.weight.val();
            if w.dtype() == DType::F16 {
                burn::tensor::module::linear(x.cast(DType::F16), w, None).cast(DType::F32)
            } else {
                burn::tensor::module::linear(x, w, lin.bias.as_ref().map(|b| b.val()))
            }
        }
        #[cfg(all(feature = "npu", target_arch = "aarch64"))]
        Proj::Npu(id) => {
            // No CPU copy was retained and the caller asked for NPU: run it on
            // the NPU even for a tiny M (padded to 256 rows by the extension).
            burn_rocket::matmul(x, id)
        }
    }
}

/// L2 normalization over the last dim (FLA-aligned, eps 1e-6).
fn l2norm<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    let inv = ((x.clone() * x.clone()).sum_dim(D - 1) + 1e-6)
        .sqrt()
        .recip();
    x * inv
}

/// Identity matrix as a float tensor `[n, n]`.
fn eye(n: usize, device: &Device) -> Tensor<2> {
    let mut data = vec![0f32; n * n];
    for i in 0..n {
        data[i * n + i] = 1.0;
    }
    Tensor::<2>::from_data(TensorData::new(data, [n, n]), device)
}

// ---------------------------------------------------------------------------
// Gated DeltaNet (linear attention layers)
// ---------------------------------------------------------------------------

/// Chunked gated delta rule, prefill path. All layouts follow the HF reference:
/// `q/k` are `[B, H, S, Dk]`, `v` is `[B, H, S, Dv]`, `g`/`beta` are `[B, H, S]`,
/// the state is `[B, H, Dk, Dv]`.
#[allow(clippy::too_many_arguments)]
pub fn chunk_gated_delta_rule(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    g: Tensor<3>,
    beta: Tensor<3>,
    init: Option<Tensor<4>>,
    chunk: usize,
    device: &Device,
) -> (Tensor<4>, Tensor<4>) {
    let [b, h, s, dk] = q.dims();
    let dv = v.dims()[3];
    let c = chunk;
    let pad = (c - s % c) % c;
    let sp = s + pad;
    let nc = sp / c;

    let q = l2norm(q) * (dk as f64).powf(-0.5);
    let k = l2norm(k);

    let (q, k, v, g, beta) = if pad > 0 {
        let q = q.pad([(0, 0), (0, 0), (0, pad), (0, 0)], PadMode::Constant(0.0));
        let k = k.pad([(0, 0), (0, 0), (0, pad), (0, 0)], PadMode::Constant(0.0));
        let v = v.pad([(0, 0), (0, 0), (0, pad), (0, 0)], PadMode::Constant(0.0));
        let g = g.pad([(0, 0), (0, 0), (0, pad)], PadMode::Constant(0.0));
        let beta = beta.pad([(0, 0), (0, 0), (0, pad)], PadMode::Constant(0.0));
        (q, k, v, g, beta)
    } else {
        (q, k, v, g, beta)
    };

    let q = q.reshape([b, h, nc, c, dk]);
    let k = k.reshape([b, h, nc, c, dk]);
    let v = v.reshape([b, h, nc, c, dv]);
    let g = g.reshape([b, h, nc, c]);
    let beta = beta.reshape([b, h, nc, c]);

    let beta_u = beta.clone().unsqueeze_dim::<5>(4); // [., c, 1]
    let v_beta = v * beta_u.clone();
    let k_beta = k.clone() * beta_u;

    let cum = g.cumsum(3); // [b,h,nc,c]
    let up = cum.clone().unsqueeze_dim::<5>(4);
    let lo = cum.clone().unsqueeze_dim::<5>(3);
    let upper = Tensor::<2, Bool>::tril_mask([c, c], 0, device)
        .reshape([1, 1, 1, c, c])
        .expand([b, h, nc, c, c]);
    let pair = (up - lo).mask_fill(upper, f32::NEG_INFINITY).exp(); // [b,h,nc,c,c]

    let kt = k.clone().swap_dims(3, 4); // [b,h,nc,dk,c]
    let ut = k_beta.clone().matmul(kt.clone()) * pair.clone();
    let intra = q.clone().matmul(kt) * pair;
    let decayed_k_beta = k_beta * cum.clone().exp().unsqueeze_dim::<5>(4);

    // Forward substitution for (I + L)^-1, the reference's export path: with
    // U0 = -strict_lower(ut), U' = (I - U0)^-1 - I = U0 + U0 @ U'.
    let mut u = ut.neg().tril(-1);
    for i in 1..c {
        let row = u
            .clone()
            .slice(s![.., .., .., i..i + 1, ..i])
            .reshape([b, h, nc, i]);
        let sub = u.clone().slice(s![.., .., .., ..i, ..i]);
        let corr = (row.clone().unsqueeze_dim::<5>(4) * sub).sum_dim(3); // [.,1,i]
        let new = row.reshape([b, h, nc, 1, i]) + corr;
        u = u.slice_assign(s![.., .., .., i..i + 1, ..i], new);
    }
    let inv = u + eye(c, device)
        .reshape([1, 1, 1, c, c])
        .expand([b, h, nc, c, c]);
    let new_values = inv.clone().matmul(v_beta); // [b,h,nc,c,dv]
    let k_cumdecay = inv.matmul(decayed_k_beta); // [b,h,nc,c,dk]

    let q_scaled = q * cum.clone().exp().unsqueeze_dim::<5>(4);
    let last = cum.clone().slice(s![.., .., .., c - 1..c]); // [b,h,nc,1]
    let key_scaled = k * (last.clone() - cum).exp().unsqueeze_dim::<5>(4);
    let chunk_decay = last.exp().reshape([b, h, nc, 1, 1]);

    let mut state =
        init.unwrap_or_else(|| Tensor::<4>::zeros([b, h, dk, dv], (device, DType::F32)));
    let mut outs: Vec<Tensor<4>> = Vec::with_capacity(nc);
    for i in 0..nc {
        let nv = new_values
            .clone()
            .slice(s![.., .., i..i + 1, .., ..])
            .reshape([b, h, c, dv]);
        let kcd = k_cumdecay
            .clone()
            .slice(s![.., .., i..i + 1, .., ..])
            .reshape([b, h, c, dk]);
        let qs = q_scaled
            .clone()
            .slice(s![.., .., i..i + 1, .., ..])
            .reshape([b, h, c, dk]);
        let ks = key_scaled
            .clone()
            .slice(s![.., .., i..i + 1, .., ..])
            .reshape([b, h, c, dk]);
        let intra_i = intra
            .clone()
            .slice(s![.., .., i..i + 1, .., ..])
            .reshape([b, h, c, c]);
        let cd = chunk_decay
            .clone()
            .slice(s![.., .., i..i + 1, .., ..])
            .reshape([b, h, 1, 1]);

        let inter = qs.matmul(state.clone());
        let v_new = nv - kcd.matmul(state.clone());
        outs.push(inter + intra_i.matmul(v_new.clone()));
        state = state * cd + ks.swap_dims(2, 3).matmul(v_new);
    }

    let out = Tensor::cat(outs, 2).slice(s![.., .., 0..s, ..]);
    (out, state)
}

/// Recurrent gated delta rule, single-token decode (the reference's
/// `torch_recurrent_gated_delta_rule`).
fn recurrent_gated_delta_rule(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    g: Tensor<3>,
    beta: Tensor<3>,
    init: Option<Tensor<4>>,
    device: &Device,
) -> (Tensor<4>, Tensor<4>) {
    let [b, h, _, dk] = q.dims();
    let dv = v.dims()[3];
    let q = l2norm(q.reshape([b, h, dk])) * (dk as f64).powf(-0.5);
    let k = l2norm(k.reshape([b, h, dk]));
    let v = v.reshape([b, h, dv]);
    let g = g.reshape([b, h]);
    let beta = beta.reshape([b, h]);

    let mut state =
        init.unwrap_or_else(|| Tensor::<4>::zeros([b, h, dk, dv], (device, DType::F32)));
    let decay = g.exp().reshape([b, h, 1, 1]);
    state = state * decay;
    let kv_mem = (state.clone() * k.clone().unsqueeze_dim::<4>(3))
        .sum_dim(2)
        .reshape([b, h, dv]);
    let delta = (v - kv_mem) * beta.reshape([b, h, 1]);
    state = state + k.unsqueeze_dim::<4>(3) * delta.unsqueeze_dim::<4>(2);
    let out = (state.clone() * q.unsqueeze_dim::<4>(3))
        .sum_dim(2)
        .reshape([b, h, 1, dv]);
    (out, state)
}

/// Gated RMSNorm: `x_norm * w * silu(gate)` on `[N, D]`.
fn rms_norm_gated(x: Tensor<2>, gate: Tensor<2>, w: &Tensor<1>, eps: f64) -> Tensor<2> {
    let xf = x.cast(DType::F32);
    let ms = (xf.clone() * xf.clone()).mean_dim(1);
    let inv = (ms + eps).sqrt().recip();
    let g = silu(gate.cast(DType::F32));
    ((xf * inv) * w.clone().cast(DType::F32).reshape([1, w.dims()[0]])) * g
}

/// One Gated DeltaNet layer.
#[derive(Debug)]
pub struct DeltaNet {
    pub in_proj_qkv: Proj, // [hidden, conv_dim]
    pub in_proj_z: Proj,   // [hidden, value_dim]
    pub in_proj_b: Proj,   // [hidden, n_v_heads]
    pub in_proj_a: Proj,   // [hidden, n_v_heads]
    pub out_proj: Proj,    // [value_dim, hidden]
    pub out_fused: NpuId,
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    pub qkvz_fused: Option<burn_rocket::WeightId>,
    pub conv_w: Tensor<2>, // [conv_dim, kernel]
    pub dt_bias: Tensor<1>,
    pub a_log: Tensor<1>,
    pub norm_w: Tensor<1>, // [v_head_dim]
    n_v_heads: usize,
    k_head_dim: usize,
    v_head_dim: usize,
    conv_dim: usize,
    key_dim: usize,
    value_dim: usize,
    kernel: usize,
    eps: f64,
}

impl DeltaNet {
    fn new_stub(cfg: &IntentTextConfig, device: &Device) -> Self {
        let hidden = cfg.hidden_size;
        Self {
            in_proj_qkv: Proj::stub(hidden, cfg.conv_dim(), device),
            in_proj_z: Proj::stub(hidden, cfg.value_dim(), device),
            in_proj_b: Proj::stub(hidden, cfg.linear_num_value_heads, device),
            in_proj_a: Proj::stub(hidden, cfg.linear_num_value_heads, device),
            out_proj: Proj::stub(cfg.value_dim(), hidden, device),
            out_fused: None,
            #[cfg(all(feature = "npu", target_arch = "aarch64"))]
            qkvz_fused: None,
            conv_w: Tensor::zeros([cfg.conv_dim(), cfg.linear_conv_kernel_dim], device),
            dt_bias: Tensor::zeros([cfg.linear_num_value_heads], device),
            a_log: Tensor::zeros([cfg.linear_num_value_heads], device),
            norm_w: Tensor::ones([cfg.linear_value_head_dim], device),
            n_v_heads: cfg.linear_num_value_heads,
            k_head_dim: cfg.linear_key_head_dim,
            v_head_dim: cfg.linear_value_head_dim,
            conv_dim: cfg.conv_dim(),
            key_dim: cfg.key_dim(),
            value_dim: cfg.value_dim(),
            kernel: cfg.linear_conv_kernel_dim,
            eps: cfg.rms_norm_eps,
        }
    }

    /// Causal depthwise conv over `x` `[B, S, conv_dim]`; updates the conv state.
    fn conv(&self, x: Tensor<3>, cache: &mut IntentCache, idx: usize) -> Tensor<3> {
        let [b, s, _] = x.dims();
        let c = self.conv_dim;
        let k = self.kernel;
        let device = x.device();
        let x = x.swap_dims(1, 2); // [b, c, s]

        let (hist, new_state) = if let Some(state) = &cache.conv[idx] {
            // Decode: prepend the (k-1)-token history.
            debug_assert_eq!(s, 1, "conv history is only for single-token decode");
            let joined = Tensor::cat(vec![state.clone(), x.clone()], 2); // [b,c,k]
            let last = joined.clone().slice(s![.., .., 1..k]);
            (joined, last)
        } else {
            // Prefill: left-pad k-1 zeros (causal convolution).
            let padded = x.pad([(0, 0), (0, 0), (k - 1, 0)], PadMode::Constant(0.0));
            let last = padded.clone().slice(s![.., .., s..s + k - 1]);
            (padded, last)
        };
        cache.conv[idx] = Some(new_state);

        let mut out = Tensor::<3>::zeros([b, c, s], (&device, DType::F32));
        for j in 0..k {
            let wj = self
                .conv_w
                .clone()
                .slice(s![.., j..j + 1])
                .reshape([1, c, 1]);
            out = out + hist.clone().slice(s![.., .., j..j + s]) * wj;
        }
        silu(out).swap_dims(1, 2) // [b, s, c]
    }

    fn forward(
        &self,
        x: Tensor<3>,
        cache: &mut IntentCache,
        idx: usize,
        chunk: usize,
        npu: bool,
    ) -> Tensor<3> {
        let [b, s, _] = x.dims();
        let device = x.device();

        #[cfg(all(feature = "npu", target_arch = "aarch64"))]
        let (mixed, z) = if npu {
            if let Some(id) = &self.qkvz_fused {
                let fused = burn_rocket::matmul(x.clone(), id); // [b,s,conv_dim+value_dim]
                let qkv = fused.clone().slice(s![.., .., 0..self.conv_dim]);
                let z = fused.slice(s![.., .., self.conv_dim..self.conv_dim + self.value_dim]);
                (qkv, z)
            } else {
                (
                    proj3(&self.in_proj_qkv, x.clone(), true),
                    proj3(&self.in_proj_z, x.clone(), true),
                )
            }
        } else {
            (
                proj3(&self.in_proj_qkv, x.clone(), false),
                proj3(&self.in_proj_z, x.clone(), false),
            )
        };
        #[cfg(not(all(feature = "npu", target_arch = "aarch64")))]
        let (mixed, z) = (
            proj3(&self.in_proj_qkv, x.clone(), false),
            proj3(&self.in_proj_z, x.clone(), false),
        );

        let bb = proj3(&self.in_proj_b, x.clone(), false); // [b,s,hv]
        let aa = proj3(&self.in_proj_a, x, false);

        let a16 = self.a_log.clone().reshape([1, 1, self.n_v_heads]);
        let dt = self.dt_bias.clone().reshape([1, 1, self.n_v_heads]);
        let g = (softplus(aa + dt, 1.0) * a16.exp().neg()).cast(DType::F32); // [b,s,hv]
        let beta = sigmoid(bb).cast(DType::F32);

        let conv = self.conv(mixed, cache, idx); // [b,s,conv_dim]
        let (kdim, vdim) = (self.key_dim, self.value_dim);
        let q = conv
            .clone()
            .slice(s![.., .., 0..kdim])
            .reshape([b, s, self.n_v_heads, self.k_head_dim])
            .swap_dims(1, 2);
        let k = conv
            .clone()
            .slice(s![.., .., kdim..2 * kdim])
            .reshape([b, s, self.n_v_heads, self.k_head_dim])
            .swap_dims(1, 2);
        let v = conv
            .slice(s![.., .., 2 * kdim..2 * kdim + vdim])
            .reshape([b, s, self.n_v_heads, self.v_head_dim])
            .swap_dims(1, 2);
        let g = g.swap_dims(1, 2); // [b,hv,s]
        let beta = beta.swap_dims(1, 2);

        let (out, state) = if s > 1 {
            chunk_gated_delta_rule(q, k, v, g, beta, cache.rec[idx].clone(), chunk, &device)
        } else {
            recurrent_gated_delta_rule(q, k, v, g, beta, cache.rec[idx].clone(), &device)
        };
        cache.rec[idx] = Some(state);

        // [b,hv,s,dv] -> [b*s*hv, dv] gated norm with z -> [b,s,value_dim]
        let out = out
            .swap_dims(1, 2)
            .reshape([s * self.n_v_heads, self.v_head_dim]);
        let z = z.reshape([s * self.n_v_heads, self.v_head_dim]);
        let normed = rms_norm_gated(out, z, &self.norm_w, self.eps);
        proj_sel(
            &self.out_proj,
            &self.out_fused,
            normed.reshape([b, s, self.value_dim]),
            npu,
        )
    }
}

// ---------------------------------------------------------------------------
// Gated full attention
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct GatedAttention {
    pub q_proj: Proj, // [hidden, n_heads*head_dim*2]
    pub k_proj: Proj,
    pub v_proj: Proj,
    pub o_proj: Proj, // [n_heads*head_dim, hidden]
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    pub qkv_fused: Option<burn_rocket::WeightId>,
    pub o_fused: NpuId,
    pub q_norm_w: Tensor<1>,
    pub k_norm_w: Tensor<1>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    eps: f64,
}

impl GatedAttention {
    fn new_stub(cfg: &IntentTextConfig, device: &Device) -> Self {
        let hidden = cfg.hidden_size;
        let nh = cfg.num_attention_heads;
        let kv = cfg.num_key_value_heads;
        let d = cfg.head_dim;
        Self {
            q_proj: Proj::stub(hidden, nh * d * 2, device),
            k_proj: Proj::stub(hidden, kv * d, device),
            v_proj: Proj::stub(hidden, kv * d, device),
            o_proj: Proj::stub(nh * d, hidden, device),
            #[cfg(all(feature = "npu", target_arch = "aarch64"))]
            qkv_fused: None,
            o_fused: None,
            q_norm_w: Tensor::zeros([d], device),
            k_norm_w: Tensor::zeros([d], device),
            n_heads: nh,
            n_kv_heads: kv,
            head_dim: d,
            eps: cfg.rms_norm_eps,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: Tensor<3>,
        cache: &mut IntentCache,
        idx: usize,
        rope: &PartialRope,
        tables: Option<&RopeTables>,
        start: usize,
        npu: bool,
        causal_full: bool,
    ) -> Tensor<3> {
        let [b, s, _] = x.dims();
        let (nh, kv, d) = (self.n_heads, self.n_kv_heads, self.head_dim);

        #[cfg(all(feature = "npu", target_arch = "aarch64"))]
        let (q_raw, k_raw, v_raw) = if npu {
            if let Some(id) = &self.qkv_fused {
                let fused = burn_rocket::matmul(x.clone(), id); // [b,s,nh*d*2 + 2*kv*d]
                let q = fused.clone().slice(s![.., .., 0..nh * d * 2]);
                let k = fused
                    .clone()
                    .slice(s![.., .., nh * d * 2..nh * d * 2 + kv * d]);
                let v = fused.slice(s![.., .., nh * d * 2 + kv * d..nh * d * 2 + 2 * kv * d]);
                (q, k, v)
            } else {
                (
                    proj3(&self.q_proj, x.clone(), true),
                    proj3(&self.k_proj, x.clone(), true),
                    proj3(&self.v_proj, x.clone(), true),
                )
            }
        } else {
            (
                proj3(&self.q_proj, x.clone(), false),
                proj3(&self.k_proj, x.clone(), false),
                proj3(&self.v_proj, x.clone(), false),
            )
        };
        #[cfg(not(all(feature = "npu", target_arch = "aarch64")))]
        let (q_raw, k_raw, v_raw) = (
            proj3(&self.q_proj, x.clone(), false),
            proj3(&self.k_proj, x.clone(), false),
            proj3(&self.v_proj, x, false),
        );

        // q_proj emits [query (head_dim), gate (head_dim)] per head.
        let qg = q_raw.reshape([b, s, nh, 2 * d]);
        let q = qg.clone().slice(s![.., .., .., 0..d]);
        let gate = qg.slice(s![.., .., .., d..2 * d]).reshape([b, s, nh * d]);
        let k = k_raw.reshape([b, s, kv, d]);
        let v = v_raw.reshape([b, s, kv, d]);

        let q = zc_norm4(q, &self.q_norm_w, self.eps);
        let k = zc_norm4(k, &self.k_norm_w, self.eps);
        // An explicit per-request table replaces the position ramp; every other
        // caller passes `None` and keeps the shared cache (bit-identical).
        let (q, k) = match tables {
            Some(tables) => (tables.apply(q), tables.apply(k)),
            None => (rope.apply(q, start), rope.apply(k, start)),
        };

        // Append to the KV cache ([b, kv, S, d] layout).
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);
        let (k_all, v_all) = match &cache.kv[idx] {
            Some((kc, vc)) => (
                Tensor::cat(vec![kc.clone(), k], 2),
                Tensor::cat(vec![vc.clone(), v], 2),
            ),
            None => (k, v),
        };
        cache.kv[idx] = Some((k_all.clone(), v_all.clone()));

        let o = if s > 1 {
            // Prefill: backend fused attention (GQA native). Causal for generation,
            // bidirectional for a d3 decision prompt (`noncausal_full_attention`).
            attention(
                q.swap_dims(1, 2),
                k_all,
                v_all,
                None,
                None,
                AttentionModuleOptions {
                    scale: Some(1.0 / (d as f64).sqrt()),
                    softcap: None,
                    is_causal: causal_full,
                },
            )
        } else {
            // Decode: one query against the whole cache; no mask needed.
            let n_kv = k_all.dims()[2];
            let rep = nh / kv;
            let kk = k_all
                .clone()
                .unsqueeze_dim::<5>(2)
                .repeat_dim(2, rep)
                .reshape([b, nh, n_kv, d]);
            let vv = v_all
                .clone()
                .unsqueeze_dim::<5>(2)
                .repeat_dim(2, rep)
                .reshape([b, nh, n_kv, d]);
            let q = q.swap_dims(1, 2); // [b, nh, 1, d]
            let scores = q.matmul(kk.swap_dims(2, 3)) * (1.0 / (d as f64).sqrt());
            let p = softmax(scores, 3);
            p.matmul(vv)
        };

        let o = o.swap_dims(1, 2).reshape([b, s, nh * d]);
        proj_sel(&self.o_proj, &self.o_fused, o * sigmoid(gate), npu)
    }
}

// ---------------------------------------------------------------------------
// MLP and layers
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct Mlp {
    pub gate_proj: Proj,
    pub up_proj: Proj,
    pub down_proj: Proj,
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    pub gateup_fused: Option<burn_rocket::WeightId>,
    pub down_fused: NpuId,
    intermediate: usize,
}

impl Mlp {
    fn new_stub(cfg: &IntentTextConfig, device: &Device) -> Self {
        Self {
            gate_proj: Proj::stub(cfg.hidden_size, cfg.intermediate_size, device),
            up_proj: Proj::stub(cfg.hidden_size, cfg.intermediate_size, device),
            down_proj: Proj::stub(cfg.intermediate_size, cfg.hidden_size, device),
            #[cfg(all(feature = "npu", target_arch = "aarch64"))]
            gateup_fused: None,
            down_fused: None,
            intermediate: cfg.intermediate_size,
        }
    }

    fn forward(&self, x: Tensor<3>, npu: bool) -> Tensor<3> {
        let n = self.intermediate;
        #[cfg(all(feature = "npu", target_arch = "aarch64"))]
        if npu && let Some(id) = &self.gateup_fused {
            let gu = burn_rocket::matmul(x, id);
            let gate = gu.clone().slice(s![.., .., 0..n]);
            let up = gu.slice(s![.., .., n..2 * n]);
            return proj_sel(&self.down_proj, &self.down_fused, silu(gate) * up, npu);
        }
        let _ = n;
        let gate = proj3(&self.gate_proj, x.clone(), npu);
        let up = proj3(&self.up_proj, x, npu);
        proj_sel(&self.down_proj, &self.down_fused, silu(gate) * up, npu)
    }
}

// The per-layer mixer is stored by value in `Vec<Layer>`; boxing the large
// variant would add indirection in the forward path for no memory win.
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub enum Mixer {
    Linear(DeltaNet),
    Full(GatedAttention),
}

#[derive(Debug)]
pub struct Layer {
    pub mixer: Mixer,
    pub input_norm_w: Tensor<1>,
    pub post_norm_w: Tensor<1>,
    pub mlp: Mlp,
    pub eps: f64,
}

impl Layer {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: Tensor<3>,
        cache: &mut IntentCache,
        idx: usize,
        rope: &PartialRope,
        tables: Option<&RopeTables>,
        start: usize,
        chunk: usize,
        npu: bool,
        causal_full: bool,
    ) -> Tensor<3> {
        let residual = x.clone();
        let h = zc_norm3(x, &self.input_norm_w, self.eps);
        let h = match &self.mixer {
            Mixer::Linear(dn) => dn.forward(h, cache, idx, chunk, npu),
            Mixer::Full(fa) => fa.forward(h, cache, idx, rope, tables, start, npu, causal_full),
        } + residual;

        let residual = h.clone();
        let h = zc_norm3(h, &self.post_norm_w, self.eps);
        self.mlp.forward(h, npu) + residual
    }
}

// ---------------------------------------------------------------------------
// Token table
// ---------------------------------------------------------------------------

/// The tied token-embedding table (`[vocab, hidden]`, also the LM head).
///
/// It is CPU resident in every deployment (the NPU never holds it), and a forward
/// only reads the rows of the current prompt, so the decision family stores it as
/// per-row symmetric int8: half of f16, and one rayon dequant pass per prompt.
// The dense tensor is larger than the two int8 vectors; boxing it would only add
// indirection to the row gather for the dense path.
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub enum Embed {
    /// f32 or f16 `[vocab, hidden]` (the intent family's path).
    Dense(Tensor<2>),
    /// Per-row symmetric int8: `codes[row * hidden + i] = round(v / scale[row])`
    /// clamped to `[-127, 127]`, `scales[row] = max|row| / 127`. A value is at
    /// most half a step (`max|row| / 254`) off; an all-zero row has scale 0.
    Int8 { codes: Vec<i8>, scales: Vec<f32> },
}

impl Embed {
    /// The ready line's label: `F32`, `F16` or `I8`.
    pub fn mode(&self) -> &'static str {
        match self {
            Embed::Dense(t) if t.dtype() == DType::F16 => "F16",
            Embed::Dense(_) => "F32",
            Embed::Int8 { .. } => "I8",
        }
    }

    /// The table as a dense tensor: a clone for [`Embed::Dense`], the dequantized
    /// whole table for [`Embed::Int8`]. Only the tied LM head needs it (the
    /// intent/gemma generation paths, which never enable int8); the forwards use
    /// [`Embed::rows`].
    pub fn dense(&self, device: &Device) -> Tensor<2> {
        match self {
            Embed::Dense(t) => t.clone(),
            Embed::Int8 { codes, scales } => {
                let rows = scales.len();
                let cols = codes.len().checked_div(rows).unwrap_or(0);
                let ids: Vec<u32> = (0..rows as u32).collect();
                let data = dequant_rows(codes, scales, &ids, cols);
                Tensor::<2>::from_data(TensorData::new(data, [rows, cols]), device)
            }
        }
    }

    /// The `[1, s, hidden]` f32 rows of `ids`, in the given order. A dense f16
    /// table yields f16 rows, cast back because the body is f32.
    pub fn rows(&self, ids: &[u32], hidden: usize, device: &Device) -> Tensor<3> {
        let s = ids.len();
        match self {
            Embed::Dense(t) => {
                let idx: Vec<i64> = ids.iter().map(|&t| t as i64).collect();
                let idx = Tensor::<1, Int>::from_data(TensorData::new(idx, [s]), device);
                let rows = t.clone().select(0, idx).reshape([1, s, hidden]);
                if t.dtype() == DType::F16 {
                    rows.cast(DType::F32)
                } else {
                    rows
                }
            }
            Embed::Int8 { codes, scales } => {
                let data = dequant_rows(codes, scales, ids, hidden);
                Tensor::<3>::from_data(TensorData::new(data, [1, s, hidden]), device)
            }
        }
    }

    /// Quantize a `[rows, cols]` f32 table row-wise. Values are rounded half away
    /// from zero (`f32::round`), then clamped to `[-127, 127]` (a no-op by
    /// construction: `|v| <= max|row| = 127 * scale`).
    pub fn quantize(table: &[f32], rows: usize, cols: usize) -> Self {
        assert_eq!(table.len(), rows * cols, "table shape mismatch");
        let (codes, scales) = quantize_rows(table, rows, cols);
        Embed::Int8 { codes, scales }
    }
}

/// Per-row symmetric int8 quantization of a `[rows, cols]` f32 table (rayon over
/// the rows): `scales[row] = max|row| / 127`, `codes = round(v / scale)`.
pub fn quantize_rows(table: &[f32], rows: usize, cols: usize) -> (Vec<i8>, Vec<f32>) {
    use rayon::prelude::*;
    let mut scales = vec![0f32; rows];
    scales.par_iter_mut().enumerate().for_each(|(r, scale)| {
        let row = &table[r * cols..(r + 1) * cols];
        let max = row.iter().fold(0f32, |m, v| m.max(v.abs()));
        // An all-zero row keeps scale 0 and quantizes to all zeros.
        *scale = if max > 0.0 { max / 127.0 } else { 0.0 };
    });
    let mut codes = vec![0i8; rows * cols];
    codes.par_chunks_mut(cols).enumerate().for_each(|(r, out)| {
        let scale = scales[r];
        if scale == 0.0 {
            return;
        }
        let inv = 1.0 / scale;
        let row = &table[r * cols..(r + 1) * cols];
        for (o, &v) in out.iter_mut().zip(row) {
            *o = (v * inv).round().clamp(-127.0, 127.0) as i8;
        }
    });
    (codes, scales)
}

/// Dequantize the rows `ids` of a per-row int8 table into `ids.len() * cols` f32
/// values, in `ids` order (rayon over the rows).
pub fn dequant_rows(codes: &[i8], scales: &[f32], ids: &[u32], cols: usize) -> Vec<f32> {
    use rayon::prelude::*;
    let rows = scales.len();
    let mut out = vec![0f32; ids.len() * cols];
    out.par_chunks_mut(cols)
        .zip(ids.par_iter())
        .for_each(|(dst, &id)| {
            let row = id as usize;
            assert!(row < rows, "token id {id} is outside the {rows}-row table");
            let scale = scales[row];
            let src = &codes[row * cols..(row + 1) * cols];
            for (d, &c) in dst.iter_mut().zip(src) {
                *d = c as f32 * scale;
            }
        });
    out
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct IntentModel {
    pub cfg: IntentTextConfig,
    pub embed: Embed,
    pub layers: Vec<Layer>,
    pub norm_w: Tensor<1>,
    pub rope: PartialRope,
    pub cache: IntentCache,
    pub device: Device,
    pub chunk: usize,
    /// Use the NPU for single-token decode too (padded `M >= 256`; for A/B only).
    pub npu_decode: bool,
    /// No CPU projection copies exist: every projection must go through the NPU.
    pub npu_only: bool,
    /// The full-attention layers are causal (a decoder, the default) or
    /// bidirectional (`noncausal_full_attention`, the d3 decision prompt). The
    /// Gated DeltaNet layers are always causal.
    pub full_attn_causal: bool,
    pub stage: StageStats,
}

/// Per-run generation timings.
#[derive(Debug, Default, Clone, Copy)]
pub struct StageStats {
    pub prefill_s: f64,
    pub decode_s: f64,
    pub steps: usize,
    /// True when generation ended on a stop token (not the length cap).
    pub stopped: bool,
}

impl IntentModel {
    /// Build the module skeleton with lazily-initialized projection weights; the
    /// loader fills every tensor in place.
    pub fn new_stub(cfg: &IntentTextConfig, max_seq: usize, device: &Device) -> Self {
        let layers = cfg
            .layer_types
            .iter()
            .map(|t| Layer {
                mixer: if t == "full_attention" {
                    Mixer::Full(GatedAttention::new_stub(cfg, device))
                } else {
                    Mixer::Linear(DeltaNet::new_stub(cfg, device))
                },
                input_norm_w: Tensor::zeros([cfg.hidden_size], device),
                post_norm_w: Tensor::zeros([cfg.hidden_size], device),
                mlp: Mlp::new_stub(cfg, device),
                eps: cfg.rms_norm_eps,
            })
            .collect();
        Self {
            cfg: cfg.clone(),
            embed: Embed::Dense(Tensor::zeros([cfg.vocab_size, cfg.hidden_size], device)),
            layers,
            norm_w: Tensor::zeros([cfg.hidden_size], device),
            rope: PartialRope::new(
                max_seq.min(cfg.max_position_embeddings),
                cfg.rotary_dim(),
                cfg.rope_parameters.rope_theta,
                DType::F32,
                device,
            ),
            cache: IntentCache::new(cfg.num_hidden_layers),
            device: device.clone(),
            chunk: 64,
            npu_decode: false,
            npu_only: false,
            full_attn_causal: true,
            stage: StageStats::default(),
        }
    }

    /// Gather `[1, s, hidden]` embeddings for `ids` (no EOS/BOS handling) as f32,
    /// in the given order. A dense f16 table yields f16 rows; the model body is
    /// f32, so they are cast back. An int8 table dequantizes only these rows.
    pub fn embed_ids(&self, ids: &[u32]) -> Tensor<3> {
        self.embed.rows(ids, self.cfg.hidden_size, &self.device)
    }

    /// Forward `ids`; appends to the cache. Returns the final-normed hidden state
    /// at the last position `[1, hidden]`. Multi-token input is only allowed for a
    /// fresh cache (prefill).
    pub fn forward_last_normed(&mut self, ids: &[u32]) -> Tensor<2> {
        let embeds = self.embed_ids(ids);
        self.forward_last_normed_embeds(embeds, None)
    }

    /// The same forward from caller-supplied embeddings `[1, seq, hidden]`
    /// (a d3 image prompt splices the vision features into the text rows) with an
    /// optional explicit RoPE table; `None` uses the shared position ramp exactly
    /// like [`IntentModel::forward_last_normed`].
    pub fn forward_last_normed_embeds(
        &mut self,
        embeds: Tensor<3>,
        tables: Option<&RopeTables>,
    ) -> Tensor<2> {
        let s = embeds.dims()[1];
        assert!(
            self.cache.len == 0 || s == 1,
            "only single-token steps are supported after a prefill (cache len {})",
            self.cache.len
        );
        let start = self.cache.len;
        let chunk = self.chunk;
        let eps = self.cfg.rms_norm_eps;
        let npu = self.npu_only || self.npu_decode || s > 1;
        let causal_full = self.full_attn_causal;
        let mut h = embeds;
        let IntentModel {
            layers,
            cache,
            rope,
            ..
        } = self;
        for (i, layer) in layers.iter().enumerate() {
            h = layer.forward(h, cache, i, rope, tables, start, chunk, npu, causal_full);
        }
        let h = zc_norm3(h, &self.norm_w, eps);
        self.cache.len += s;
        let hidden = self.cfg.hidden_size;
        h.slice(s![.., s - 1..s, ..]).reshape([1, hidden])
    }

    /// Forward `ids`; appends to the cache. Returns logits for the last position
    /// `[1, vocab]`. Multi-token input is only allowed for a fresh cache (prefill).
    pub fn forward_logits(&mut self, ids: &[u32]) -> Tensor<2> {
        let last = self.forward_last_normed(ids);
        let table = self.dense_embed();
        if table.dtype() == DType::F16 {
            last.cast(DType::F16)
                .matmul(table.transpose())
                .cast(DType::F32)
        } else {
            last.matmul(table.transpose())
        }
    }

    /// The tied table as a dense tensor (the LM head). Generation only: a d3
    /// decision prompt never reads the head, so it never pays the dequantization.
    fn dense_embed(&self) -> Tensor<2> {
        self.embed.dense(&self.device)
    }

    /// Debug helper: run a fresh forward and return the hidden state after the
    /// embedding and after every layer (25 entries for this model), each
    /// `[1, s, hidden]`. Does not touch generation state beyond resetting it.
    pub fn forward_hidden(&mut self, ids: &[u32]) -> Vec<Tensor<3>> {
        self.cache.reset();
        let chunk = self.chunk;
        let npu = self.npu_only || self.npu_decode || ids.len() > 1;
        let causal_full = self.full_attn_causal;
        let mut h = self.embed_ids(ids);
        let mut out = vec![h.clone()];
        let IntentModel {
            layers,
            cache,
            rope,
            ..
        } = self;
        for (i, layer) in layers.iter().enumerate() {
            h = layer.forward(h, cache, i, rope, None, 0, chunk, npu, causal_full);
            out.push(h.clone());
        }
        out
    }

    /// Greedy/temperature generation. `stop_ids` terminate generation and are not
    /// returned. Returns the generated ids and timings.
    pub fn generate(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        stop_ids: &[u32],
        temperature: f32,
        rng_seed: u64,
    ) -> (Vec<u32>, StageStats) {
        self.cache.reset();
        let t0 = Instant::now();
        let logits = self.prefill(prompt);
        let prefill_s = t0.elapsed().as_secs_f64();
        let mut rng = rng_seed | 1;
        let t1 = Instant::now();
        let (out, stopped) = self.decode(logits, max_new, stop_ids, temperature, &mut rng);
        let decode_s = t1.elapsed().as_secs_f64();
        let stats = StageStats {
            prefill_s,
            decode_s,
            steps: out.len(),
            stopped,
        };
        self.stage = stats;
        (out, stats)
    }

    /// Prefill the prompt and return the last position's logits `[1, vocab]`.
    #[telemetry::instrument(name = "prefill", skip_all, fields(prompt_tokens = prompt.len()))]
    fn prefill(&mut self, prompt: &[u32]) -> Tensor<2> {
        self.forward_logits(prompt)
    }

    /// Decode up to `max_new` tokens from the prefill logits. Returns the
    /// generated ids (stop ids excluded) and whether a stop id ended generation.
    #[telemetry::instrument(
        name = "decode",
        skip_all,
        fields(gen_tokens = tracing::field::Empty, tok_s = tracing::field::Empty)
    )]
    fn decode(
        &mut self,
        mut logits: Tensor<2>,
        max_new: usize,
        stop_ids: &[u32],
        temperature: f32,
        rng: &mut u64,
    ) -> (Vec<u32>, bool) {
        let started = Instant::now();
        let mut out = Vec::with_capacity(max_new);
        let mut stopped = false;
        for _ in 0..max_new {
            let logits_vec: Vec<f32> = logits.clone().to_data().try_to_vec().unwrap_or_default();
            let next = pick_token(&logits_vec, temperature, rng);
            if stop_ids.contains(&next) {
                stopped = true;
                break;
            }
            out.push(next);
            if out.len() == max_new {
                break;
            }
            logits = self.forward_logits(&[next]);
        }
        let decode_s = started.elapsed().as_secs_f64();
        let span = tracing::Span::current();
        span.record("gen_tokens", out.len());
        span.record("tok_s", out.len() as f64 / decode_s.max(1e-9));
        (out, stopped)
    }
}

/// Token selection: greedy below temperature 0.01, else softmax sampling with a
/// small xorshift RNG.
fn pick_token(logits: &[f32], temperature: f32, rng: &mut u64) -> u32 {
    if temperature <= 0.01 {
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best = i;
            }
        }
        return best as u32;
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f64;
    let probs: Vec<f32> = logits
        .iter()
        .map(|&v| {
            let p = ((v - max) as f64 / temperature as f64).exp();
            sum += p;
            p as f32
        })
        .collect();
    let mut u = {
        *rng ^= *rng << 13;
        *rng ^= *rng >> 7;
        *rng ^= *rng << 17;
        (*rng >> 11) as f64 / (1u64 << 53) as f64
    } * sum;
    for (i, &p) in probs.iter().enumerate() {
        u -= p as f64;
        if u <= 0.0 {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::proj::Proj;
    use burn::module::Param;
    use burn::nn::Linear;

    /// A tiny two-layer hybrid (a full-attention layer feeding a delta-rule one)
    /// that exercises every weight without loading a checkpoint. The full layer
    /// goes first so a bidirectional mask reaches the last position through the
    /// (causal) delta layer
    fn tiny_cfg() -> IntentTextConfig {
        IntentTextConfig::from_value(serde_json::json!({
            "text_config": {
                "vocab_size": 64,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 16,
                "rms_norm_eps": 1e-6,
                "max_position_embeddings": 64,
                "layer_types": ["full_attention", "linear_attention"],
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 8,
                "linear_value_head_dim": 8,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 2,
                "partial_rotary_factor": 0.5,
                "rope_parameters": {"rope_theta": 10000.0},
            }
        }))
        .unwrap()
    }

    /// Small deterministic pseudo-random values (a fixed LCG, `u` in `[-0.1, 0.1)`).
    fn values(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = (state >> 40) as f32 / (1u64 << 24) as f32;
                (u - 0.5) * 0.2
            })
            .collect()
    }

    fn tensor2(rows: usize, cols: usize, seed: u64, device: &Device) -> Tensor<2> {
        Tensor::<2>::from_data(
            TensorData::new(values(rows * cols, seed), [rows, cols]),
            device,
        )
    }

    fn tensor1(n: usize, seed: u64, device: &Device) -> Tensor<1> {
        Tensor::<1>::from_data(TensorData::new(values(n, seed), [n]), device)
    }

    fn proj(input: usize, output: usize, seed: u64, device: &Device) -> Proj {
        // `Linear.weight` is `[in, out]`; the values themselves are arbitrary.
        Proj::Cpu(Linear {
            weight: Param::from_tensor(tensor2(input, output, seed, device)),
            bias: None,
        })
    }

    fn filled_model(cfg: &IntentTextConfig, device: &Device) -> IntentModel {
        let mut model = IntentModel::new_stub(cfg, 64, device);
        model.embed = Embed::Dense(tensor2(cfg.vocab_size, cfg.hidden_size, 1, device));
        model.norm_w = tensor1(cfg.hidden_size, 2, device);
        for (i, layer) in model.layers.iter_mut().enumerate() {
            let seed = 10 + i as u64 * 10;
            layer.input_norm_w = tensor1(cfg.hidden_size, seed, device);
            layer.post_norm_w = tensor1(cfg.hidden_size, seed + 1, device);
            layer.mlp.gate_proj = proj(cfg.hidden_size, cfg.intermediate_size, seed + 2, device);
            layer.mlp.up_proj = proj(cfg.hidden_size, cfg.intermediate_size, seed + 3, device);
            layer.mlp.down_proj = proj(cfg.intermediate_size, cfg.hidden_size, seed + 4, device);
            match &mut layer.mixer {
                Mixer::Linear(dn) => {
                    dn.in_proj_qkv = proj(cfg.hidden_size, cfg.conv_dim(), seed + 5, device);
                    dn.in_proj_z = proj(cfg.hidden_size, cfg.value_dim(), seed + 6, device);
                    dn.in_proj_b = proj(
                        cfg.hidden_size,
                        cfg.linear_num_value_heads,
                        seed + 7,
                        device,
                    );
                    dn.in_proj_a = proj(
                        cfg.hidden_size,
                        cfg.linear_num_value_heads,
                        seed + 8,
                        device,
                    );
                    dn.out_proj = proj(cfg.value_dim(), cfg.hidden_size, seed + 9, device);
                    dn.conv_w = tensor2(
                        cfg.conv_dim(),
                        cfg.linear_conv_kernel_dim,
                        seed + 10,
                        device,
                    );
                    dn.dt_bias = tensor1(cfg.linear_num_value_heads, seed + 11, device);
                    dn.a_log = tensor1(cfg.linear_num_value_heads, seed + 12, device);
                    dn.norm_w = tensor1(cfg.linear_value_head_dim, seed + 13, device);
                }
                Mixer::Full(fa) => {
                    let d = cfg.head_dim;
                    fa.q_proj = proj(
                        cfg.hidden_size,
                        cfg.num_attention_heads * d * 2,
                        seed + 5,
                        device,
                    );
                    fa.k_proj = proj(
                        cfg.hidden_size,
                        cfg.num_key_value_heads * d,
                        seed + 6,
                        device,
                    );
                    fa.v_proj = proj(
                        cfg.hidden_size,
                        cfg.num_key_value_heads * d,
                        seed + 7,
                        device,
                    );
                    fa.o_proj = proj(
                        cfg.num_attention_heads * d,
                        cfg.hidden_size,
                        seed + 8,
                        device,
                    );
                    fa.q_norm_w = tensor1(d, seed + 9, device);
                    fa.k_norm_w = tensor1(d, seed + 10, device);
                }
            }
        }
        model
    }

    fn rows(tensor: Tensor<2>) -> Vec<f32> {
        tensor.to_data().try_to_vec().unwrap()
    }

    #[test]
    fn forward_logits_is_the_normed_last_hidden_through_the_tied_table() {
        let device = Device::default();
        let cfg = tiny_cfg();
        let ids = [1u32, 4, 7, 9, 3];
        let mut logits_model = filled_model(&cfg, &device);
        let logits = rows(logits_model.forward_logits(&ids));
        let mut hidden_model = filled_model(&cfg, &device);
        let hidden = hidden_model.forward_last_normed(&ids);
        let expected = rows(hidden.matmul(hidden_model.dense_embed().transpose()));
        assert_eq!(logits.len(), cfg.vocab_size);
        assert_eq!(
            logits, expected,
            "forward_logits must be the readout of the hidden"
        );
    }

    // ---- int8 token table ----

    /// A `[rows, cols]` table whose rows differ in magnitude (row 0 is all zeros,
    /// the last row is the widest) so the per-row scales are all different.
    fn table_with_row_scales(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
        let mut out = values(rows * cols, seed);
        for (r, row) in out.chunks_mut(cols).enumerate() {
            let gain = if r == 0 { 0.0 } else { r as f32 * 0.5 };
            for (i, v) in row.iter_mut().enumerate() {
                *v = *v * gain
                    + if i % 3 == 0 {
                        gain * 0.001
                    } else {
                        -gain * 0.002
                    };
            }
        }
        out
    }

    #[test]
    fn the_int8_table_is_within_half_a_step_of_every_row() {
        let (rows, cols) = (24usize, 17);
        let table = table_with_row_scales(rows, cols, 7);
        let (codes, scales) = quantize_rows(&table, rows, cols);
        for r in 0..rows {
            let row = &table[r * cols..(r + 1) * cols];
            let max = row.iter().fold(0f32, |m, v| m.max(v.abs()));
            if r == 0 {
                assert_eq!(scales[r], 0.0, "an all-zero row keeps scale 0");
                assert!(
                    codes[r * cols..(r + 1) * cols].iter().all(|&c| c == 0),
                    "an all-zero row quantizes to zeros"
                );
                continue;
            }
            assert!(scales[r] > 0.0);
            let step = max / 127.0;
            assert!((scales[r] - step).abs() <= step * 1e-5, "scale = max/127");
            for (i, &v) in row.iter().enumerate() {
                let q = codes[r * cols + i] as f32 * scales[r];
                assert!(
                    (q - v).abs() <= step / 2.0 * 1.0001,
                    "row {r} col {i}: {v} quantized to {q}, step {step}"
                );
                assert!(
                    (-127..=127).contains(&(codes[r * cols + i] as i16)),
                    "codes stay in the symmetric range"
                );
            }
        }
        // The dequantized gather reproduces the rows in id order.
        let ids = [23u32, 0, 5];
        let gathered = dequant_rows(&codes, &scales, &ids, cols);
        assert_eq!(gathered.len(), ids.len() * cols);
        for (k, &id) in ids.iter().enumerate() {
            let r = id as usize;
            for i in 0..cols {
                let want = table[r * cols + i];
                let got = gathered[k * cols + i];
                assert!(
                    (got - want).abs() <= scales[r] / 2.0 * 1.0001,
                    "gather order: id {id} slot {k} col {i}"
                );
            }
        }
    }

    #[test]
    fn an_int8_table_gathers_the_prompt_rows_in_id_order() {
        let device = Device::default();
        let cfg = tiny_cfg();
        let (vocab, hidden) = (cfg.vocab_size, cfg.hidden_size);
        let table = table_with_row_scales(vocab, hidden, 11);
        let quantized = Embed::quantize(&table, vocab, hidden);
        let mut dense = filled_model(&cfg, &device);
        dense.embed = Embed::Dense(Tensor::<2>::from_data(
            TensorData::new(table.clone(), [vocab, hidden]),
            &device,
        ));
        let mut int8 = filled_model(&cfg, &device);
        int8.embed = quantized;

        // Out of order and repeated: the rows must follow `ids` exactly.
        let ids = [7u32, 2, 5, 2];
        let want = rows2(dense.embed_ids(&ids));
        let got = rows2(int8.embed_ids(&ids));
        assert_eq!(got.len(), ids.len() * hidden);
        let mut max_abs = 0f32;
        for (k, &id) in ids.iter().enumerate() {
            let row = &table[id as usize * hidden..(id as usize + 1) * hidden];
            let max = row.iter().fold(0f32, |m, v| m.max(v.abs()));
            for i in 0..hidden {
                let d = (got[k * hidden + i] - want[k * hidden + i]).abs();
                max_abs = max_abs.max(d);
                assert!(
                    d <= max / 254.0 * 1.0001,
                    "id {id} slot {k} col {i}: dense {} vs int8 {}",
                    want[k * hidden + i],
                    got[k * hidden + i]
                );
            }
        }
        // Relative to the row's own magnitude the error is tiny (<= half a step).
        assert!(max_abs > 0.0, "the fixture must exercise the quantizer");
        println!("int8 table vs dense rows: max |delta| {max_abs:.3e}");

        // The whole model still runs from the int8 table, to the same tolerance.
        let h_dense = rows2_row(dense.forward_last_normed(&ids));
        let h_int8 = rows2_row(int8.forward_last_normed(&ids));
        let scale = h_dense.iter().fold(0f32, |m, v| m.max(v.abs()));
        let worst = h_dense
            .iter()
            .zip(&h_int8)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(
            worst <= scale * 2e-2,
            "hidden state moved by {worst} (magnitude {scale})"
        );
    }

    #[test]
    fn the_tied_head_dequantizes_an_int8_table() {
        let device = Device::default();
        let cfg = tiny_cfg();
        let (vocab, hidden) = (cfg.vocab_size, cfg.hidden_size);
        let table = table_with_row_scales(vocab, hidden, 13);
        let (codes, scales) = quantize_rows(&table, vocab, hidden);
        let ids = [1u32, 4, 7];
        let mut int8 = filled_model(&cfg, &device);
        int8.embed = Embed::Int8 {
            codes: codes.clone(),
            scales: scales.clone(),
        };
        // The dense table the int8 one stands for.
        let dequant: Vec<f32> = dequant_rows(
            &codes,
            &scales,
            &(0..vocab as u32).collect::<Vec<_>>(),
            hidden,
        );
        let mut dense = filled_model(&cfg, &device);
        dense.embed = Embed::Dense(Tensor::<2>::from_data(
            TensorData::new(dequant, [vocab, hidden]),
            &device,
        ));
        assert_eq!(
            rows2_row(int8.forward_logits(&ids)),
            rows2_row(dense.forward_logits(&ids)),
            "forward_logits over an int8 table is the readout of its dequantized form"
        );
        // And it still tracks the unquantized table closely (`forward_logits`
        // appends to the cache, so reset before the second prefill).
        int8.cache.reset();
        let mut original = filled_model(&cfg, &device);
        original.embed = Embed::Dense(Tensor::<2>::from_data(
            TensorData::new(table, [vocab, hidden]),
            &device,
        ));
        let a = rows2_row(original.forward_logits(&ids));
        let b = rows2_row(int8.forward_logits(&ids));
        let scale = a.iter().fold(0f32, |m, v| m.max(v.abs()));
        let worst = a
            .iter()
            .zip(&b)
            .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
        assert!(worst <= scale * 2e-2, "logits moved by {worst} of {scale}");
    }

    /// `[1, s, hidden]` -> a flat vector.
    fn rows2(tensor: Tensor<3>) -> Vec<f32> {
        tensor.to_data().try_to_vec().unwrap()
    }

    /// A `[1, hidden]` tensor -> a flat vector.
    fn rows2_row(tensor: Tensor<2>) -> Vec<f32> {
        tensor.to_data().try_to_vec().unwrap()
    }

    #[test]
    fn the_full_attention_mask_defaults_to_causal_and_is_wired() {
        let device = Device::default();
        let cfg = tiny_cfg();
        let ids = [1u32, 4, 7, 9, 3];
        let mut causal = filled_model(&cfg, &device);
        assert!(causal.full_attn_causal, "decoders stay causal by default");
        let causal = rows(causal.forward_last_normed(&ids));
        let mut noncausal = filled_model(&cfg, &device);
        noncausal.full_attn_causal = false;
        let noncausal = rows(noncausal.forward_last_normed(&ids));
        assert!(
            causal.iter().zip(&noncausal).any(|(a, b)| a != b),
            "noncausal_full_attention must change the full-attention layer's mask"
        );
    }
}
