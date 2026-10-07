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

use burn::prelude::*;
use burn::tensor::activation::{sigmoid, silu, softmax, softplus};
use burn::tensor::module::attention;
use burn::tensor::ops::{AttentionModuleOptions, PadMode};
use burn::tensor::{Bool, DType, Int, TensorData, s};

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
    pub fn from_file(path: &std::path::Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let file: IntentConfigFile = serde_json::from_str(&text)?;
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
    if npu {
        if let Some(id) = id {
            return burn_rocket::matmul(x, id);
        }
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
    if npu {
        if let Proj::Npu(id) = p {
            return burn_rocket::matmul(x, id);
        }
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

    fn forward(
        &self,
        x: Tensor<3>,
        cache: &mut IntentCache,
        idx: usize,
        rope: &PartialRope,
        start: usize,
        npu: bool,
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
        let q = rope.apply(q, start);
        let k = rope.apply(k, start);

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
            // Prefill: backend fused causal attention (GQA native).
            attention(
                q.swap_dims(1, 2),
                k_all,
                v_all,
                None,
                None,
                AttentionModuleOptions {
                    scale: Some(1.0 / (d as f64).sqrt()),
                    softcap: None,
                    is_causal: true,
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
        if npu {
            if let Some(id) = &self.gateup_fused {
                let gu = burn_rocket::matmul(x, id);
                let gate = gu.clone().slice(s![.., .., 0..n]);
                let up = gu.slice(s![.., .., n..2 * n]);
                return proj_sel(&self.down_proj, &self.down_fused, silu(gate) * up, npu);
            }
        }
        let _ = n;
        let gate = proj3(&self.gate_proj, x.clone(), npu);
        let up = proj3(&self.up_proj, x, npu);
        proj_sel(&self.down_proj, &self.down_fused, silu(gate) * up, npu)
    }
}

pub enum Mixer {
    Linear(DeltaNet),
    Full(GatedAttention),
}

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
        start: usize,
        chunk: usize,
        npu: bool,
    ) -> Tensor<3> {
        let residual = x.clone();
        let h = zc_norm3(x, &self.input_norm_w, self.eps);
        let h = match &self.mixer {
            Mixer::Linear(dn) => dn.forward(h, cache, idx, chunk, npu),
            Mixer::Full(fa) => fa.forward(h, cache, idx, rope, start, npu),
        } + residual;

        let residual = h.clone();
        let h = zc_norm3(h, &self.post_norm_w, self.eps);
        self.mlp.forward(h, npu) + residual
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

pub struct IntentModel {
    pub cfg: IntentTextConfig,
    pub embed: Tensor<2>, // [vocab, hidden]
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
            embed: Tensor::zeros([cfg.vocab_size, cfg.hidden_size], device),
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
            stage: StageStats::default(),
        }
    }

    /// Gather `[1, s, hidden]` embeddings for `ids` (no EOS/BOS handling). An f16
    /// table yields f16 rows; the model body is f32, so cast them back.
    pub fn embed_ids(&self, ids: &[u32]) -> Tensor<3> {
        let s = ids.len();
        let ids_i64: Vec<i64> = ids.iter().map(|&t| t as i64).collect();
        let idx = Tensor::<1, Int>::from_data(TensorData::new(ids_i64, [s]), &self.device);
        let rows = self
            .embed
            .clone()
            .select(0, idx)
            .reshape([1, s, self.cfg.hidden_size]);
        if self.embed.dtype() == DType::F16 {
            rows.cast(DType::F32)
        } else {
            rows
        }
    }

    /// Forward `ids`; appends to the cache. Returns logits for the last position
    /// `[1, vocab]`. Multi-token input is only allowed for a fresh cache (prefill).
    pub fn forward_logits(&mut self, ids: &[u32]) -> Tensor<2> {
        let s = ids.len();
        assert!(
            self.cache.len == 0 || s == 1,
            "only single-token steps are supported after a prefill (cache len {})",
            self.cache.len
        );
        let start = self.cache.len;
        let chunk = self.chunk;
        let eps = self.cfg.rms_norm_eps;
        let npu = self.npu_only || self.npu_decode || s > 1;
        let mut h = self.embed_ids(ids);
        let IntentModel {
            layers,
            cache,
            rope,
            ..
        } = self;
        for (i, layer) in layers.iter().enumerate() {
            h = layer.forward(h, cache, i, rope, start, chunk, npu);
        }
        let h = zc_norm3(h, &self.norm_w, eps);
        self.cache.len += s;
        let hidden = self.cfg.hidden_size;
        let last = h.slice(s![.., s - 1..s, ..]).reshape([1, hidden]);
        let table = self.embed.clone();
        if table.dtype() == DType::F16 {
            last.cast(DType::F16)
                .matmul(table.transpose())
                .cast(DType::F32)
        } else {
            last.matmul(table.transpose())
        }
    }

    /// Debug helper: run a fresh forward and return the hidden state after the
    /// embedding and after every layer (25 entries for this model), each
    /// `[1, s, hidden]`. Does not touch generation state beyond resetting it.
    pub fn forward_hidden(&mut self, ids: &[u32]) -> Vec<Tensor<3>> {
        self.cache.reset();
        let chunk = self.chunk;
        let npu = self.npu_only || self.npu_decode || ids.len() > 1;
        let mut h = self.embed_ids(ids);
        let mut out = vec![h.clone()];
        let IntentModel {
            layers,
            cache,
            rope,
            ..
        } = self;
        for (i, layer) in layers.iter().enumerate() {
            h = layer.forward(h, cache, i, rope, 0, chunk, npu);
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
        let mut logits = self.forward_logits(prompt);
        let prefill_s = t0.elapsed().as_secs_f64();
        let mut rng = rng_seed | 1;
        let mut out = Vec::with_capacity(max_new);
        let mut decode_s = 0.0;
        let mut stopped = false;
        for _ in 0..max_new {
            let t = Instant::now();
            let logits_vec: Vec<f32> = logits.clone().to_data().try_to_vec().unwrap_or_default();
            let next = pick_token(&logits_vec, temperature, &mut rng);
            if stop_ids.contains(&next) {
                stopped = true;
                break;
            }
            out.push(next);
            if out.len() == max_new {
                break;
            }
            logits = self.forward_logits(&[next]);
            decode_s += t.elapsed().as_secs_f64();
        }
        let stats = StageStats {
            prefill_s,
            decode_s,
            steps: out.len(),
            stopped,
        };
        self.stage = stats;
        (out, stats)
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
