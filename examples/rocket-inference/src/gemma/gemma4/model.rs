//! Gemma 4 text decoder for generation (E2B/E4B/12B/26B family), inference only.
//!
//! Reference: HF `transformers/models/gemma4/modeling_gemma4.py` (5.19). The
//! architecture is the causal sibling of the EmbeddingGemma 2 text backbone
//! implemented in `model.rs`, with three generation-specific pieces:
//!
//! - **KV sharing**: the last `num_kv_shared_layers` layers reuse the K/V of the
//!   last non-shared layer of the same attention type (they keep q/o
//!   projections and use a double-wide MLP);
//! - **PLE with a token table**: `embed_tokens_per_layer` contributes a
//!   token-identity term blended with the context projection (`(ctx + token) / sqrt(2)`);
//! - **proportional p-RoPE** on the full-attention layers (only the first
//!   `partial_rotary_factor * head_dim / 2` frequency pairs rotate; the tail
//!   frequencies are zero, i.e. identity).
//!
//! The output logits are soft-capped (`tanh(logits / c) * c`) and the LM head is
//! tied to `embed_tokens` (the checkpoint has no `lm_head` tensor).

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use burn::module::Param;
use burn::nn::{Embedding, EmbeddingConfig, Linear, RmsNorm, RmsNormConfig};
use burn::prelude::*;
use burn::tensor::activation::{gelu_approximate, softmax};
use burn::tensor::{DType, Int, s};

use std::collections::HashMap;

use crate::gemma::audio::AudioTower;
use crate::gemma::audio_frontend::{AudioFeatures, subsample_mask};
use crate::gemma::config::{AudioConfig, TextConfig, VisionConfig};
use crate::gemma::layers::ClipBounds;
use crate::gemma::layers::lin;
use crate::gemma::media::PreparedImage;
use crate::gemma::vision::{MultimodalEmbedder, VisionSpec, VisionTower};
use crate::gemma::{linear_cfg, repeat_kv, rms_norm_noscale};

// Stage wall-time instrumentation (microseconds) for the generation path.
static T_ATTN_US: AtomicU64 = AtomicU64::new(0);
static T_MLP_US: AtomicU64 = AtomicU64::new(0);
static T_PLE_US: AtomicU64 = AtomicU64::new(0);
static T_NORM_US: AtomicU64 = AtomicU64::new(0);

/// `(attention_s, mlp_s, ple_s, norms_s)` accumulated since the last reset.
pub fn gen_stage_stats() -> (f64, f64, f64, f64) {
    (
        T_ATTN_US.load(Ordering::Relaxed) as f64 / 1e6,
        T_MLP_US.load(Ordering::Relaxed) as f64 / 1e6,
        T_PLE_US.load(Ordering::Relaxed) as f64 / 1e6,
        T_NORM_US.load(Ordering::Relaxed) as f64 / 1e6,
    )
}

pub fn gen_stage_stats_reset() {
    T_ATTN_US.store(0, Ordering::Relaxed);
    T_MLP_US.store(0, Ordering::Relaxed);
    T_PLE_US.store(0, Ordering::Relaxed);
    T_NORM_US.store(0, Ordering::Relaxed);
}

/// Per-layer geometry for generation.
#[derive(Debug, Clone)]
pub struct GenLayerSpec {
    pub head_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub sliding: bool,
    /// Layer type index: 0 = sliding, 1 = full.
    pub type_idx: usize,
    /// Reuses another layer's K/V.
    pub is_kv_shared: bool,
    /// Non-shared layer that stores the full-length K/V for sharing.
    pub stores_shared_kv: bool,
    /// Intermediate size (double-wide on shared layers).
    pub intermediate: usize,
    /// `(theta, partial_rotary_factor)` for this layer type.
    pub rope_theta: f64,
    pub rope_partial: f64,
    /// Full layers of `attention_k_eq_v` models derive V from the raw k_proj.
    pub k_eq_v: bool,
}

/// Plain-data generation spec.
#[derive(Debug, Clone)]
pub struct GenSpec {
    pub hidden: usize,
    pub num_layers: usize,
    pub eps: f64,
    pub sliding_window: usize,
    pub ple_dim: usize,
    pub logit_softcap: Option<f64>,
    pub layers: Vec<GenLayerSpec>,
}

impl GenSpec {
    pub fn from_config(cfg: &TextConfig) -> Self {
        let num_layers = cfg.num_hidden_layers;
        let types = cfg.layer_types();
        let first_shared = num_layers.saturating_sub(cfg.num_kv_shared_layers);
        let mut shared_source = [None, None];
        for (i, t) in types.iter().take(first_shared).enumerate() {
            let idx = if t == "sliding_attention" { 0 } else { 1 };
            shared_source[idx] = Some(i);
        }
        let sliding_theta = cfg
            .rope_parameters
            .sliding_attention
            .as_ref()
            .and_then(|r| r.rope_theta)
            .unwrap_or(10_000.0);
        let full = cfg.rope_parameters.full_attention.as_ref();
        let full_theta = full.and_then(|r| r.rope_theta).unwrap_or(1_000_000.0);
        let full_partial = full.and_then(|r| r.partial_rotary_factor).unwrap_or(1.0);

        let layers = (0..num_layers)
            .map(|i| {
                let sliding = types[i] == "sliding_attention";
                let type_idx = usize::from(!sliding);
                let is_kv_shared = first_shared > 0 && i >= first_shared;
                let head_dim = if sliding {
                    cfg.head_dim()
                } else {
                    cfg.global_head_dim.unwrap_or_else(|| cfg.head_dim())
                };
                let n_kv_heads = if !sliding && cfg.attention_k_eq_v {
                    cfg.num_global_key_value_heads
                        .unwrap_or(cfg.num_key_value_heads)
                } else {
                    cfg.num_key_value_heads
                };
                let double_wide = is_kv_shared && cfg.use_double_wide_mlp;
                GenLayerSpec {
                    head_dim,
                    n_heads: cfg.num_attention_heads,
                    n_kv_heads,
                    sliding,
                    type_idx,
                    is_kv_shared,
                    stores_shared_kv: !is_kv_shared && shared_source[type_idx] == Some(i),
                    intermediate: cfg.intermediate_size * if double_wide { 2 } else { 1 },
                    rope_theta: if sliding { sliding_theta } else { full_theta },
                    rope_partial: if sliding { 1.0 } else { full_partial },
                    k_eq_v: cfg.attention_k_eq_v && !sliding,
                }
            })
            .collect();
        Self {
            hidden: cfg.hidden_size,
            num_layers,
            eps: cfg.rms_norm_eps,
            sliding_window: cfg.sliding_window,
            ple_dim: cfg.hidden_size_per_layer_input,
            logit_softcap: cfg.final_logit_softcapping,
            layers,
        }
    }

    /// `(theta, partial)` per layer type (0 = sliding, 1 = full).
    pub fn rope_of_type(&self, type_idx: usize) -> (f64, f64, usize) {
        let l = self
            .layers
            .iter()
            .find(|l| l.type_idx == type_idx)
            .expect("layer type present");
        (l.rope_theta, l.rope_partial, l.head_dim)
    }
}

/// RoPE cos/sin table `[max_seq, head_dim / 2]`. With `partial < 1` only the
/// first `int(partial * head_dim) / 2` frequencies are non-zero (proportional
/// p-RoPE); the zero tail makes the rotation an identity there.
pub struct GenRopeTable {
    cos: Tensor<2>,
    sin: Tensor<2>,
    half: usize,
}

impl GenRopeTable {
    pub fn new(
        max_seq: usize,
        head_dim: usize,
        theta: f64,
        partial: f64,
        dtype: DType,
        device: &Device,
    ) -> Self {
        let half = head_dim / 2;
        let rope_angles = if partial >= 1.0 {
            half
        } else {
            ((partial * head_dim as f64) as usize) / 2
        };
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| {
                if i < rope_angles {
                    (1.0 / theta.powf(2.0 * i as f64 / head_dim as f64)) as f32
                } else {
                    0.0
                }
            })
            .collect();
        let mut cos = Vec::with_capacity(max_seq * half);
        let mut sin = Vec::with_capacity(max_seq * half);
        for pos in 0..max_seq {
            for f in &inv_freq {
                let angle = pos as f32 * *f;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
        let cos = Tensor::<2>::from_data(TensorData::new(cos, [max_seq, half]), device).cast(dtype);
        let sin = Tensor::<2>::from_data(TensorData::new(sin, [max_seq, half]), device).cast(dtype);
        Self { cos, sin, half }
    }

    /// Applies RoPE to `[batch, seq, heads, head_dim]` starting at `seq_start`.
    pub fn apply(&self, x: Tensor<4>, seq_start: usize) -> Tensor<4> {
        let [_, s, _, d] = x.dims();
        let half = self.half;
        let cos = self
            .cos
            .clone()
            .slice(s![seq_start..seq_start + s, ..])
            .reshape([1, s, 1, half])
            .cast(x.dtype());
        let sin = self
            .sin
            .clone()
            .slice(s![seq_start..seq_start + s, ..])
            .reshape([1, s, 1, half])
            .cast(x.dtype());
        let x1 = x.clone().slice(s![.., .., .., 0..half]);
        let x2 = x.slice(s![.., .., .., half..d]);
        let o1 = x1.clone() * cos.clone() - x2.clone() * sin.clone();
        let o2 = x2 * cos + x1 * sin;
        Tensor::cat(vec![o1, o2], 3)
    }
}

/// Both RoPE tables (sliding and full layers).
pub struct GenRopes {
    tables: [GenRopeTable; 2],
}

impl GenRopes {
    pub fn new(spec: &GenSpec, max_seq: usize, dtype: DType, device: &Device) -> Self {
        let mut tables = Vec::new();
        for type_idx in 0..2 {
            let (theta, partial, head_dim) = spec.rope_of_type(type_idx);
            tables.push(GenRopeTable::new(
                max_seq, head_dim, theta, partial, dtype, device,
            ));
        }
        let mut it = tables.into_iter();
        Self {
            tables: [it.next().unwrap(), it.next().unwrap()],
        }
    }

    pub fn of(&self, spec: &GenLayerSpec) -> &GenRopeTable {
        &self.tables[spec.type_idx]
    }
}

/// KV cache: per-layer K/V for the non-shared layers plus the full-length
/// shared states stored by the last non-shared layer of each type.
pub struct GenKv {
    k: Vec<Option<Tensor<4>>>,
    v: Vec<Option<Tensor<4>>>,
    pub shared: [Option<(Tensor<4>, Tensor<4>)>; 2],
    pub len: usize,
}

impl GenKv {
    pub fn new(num_layers: usize) -> Self {
        Self {
            k: vec![None; num_layers],
            v: vec![None; num_layers],
            shared: [None, None],
            len: 0,
        }
    }

    /// Appends the new K/V of `layer` and returns the full cached tensors.
    fn append(&mut self, layer: usize, k: Tensor<4>, v: Tensor<4>) -> (Tensor<4>, Tensor<4>) {
        let kk = match &self.k[layer] {
            Some(prev) => Tensor::cat(vec![prev.clone(), k], 2),
            None => k,
        };
        let vv = match &self.v[layer] {
            Some(prev) => Tensor::cat(vec![prev.clone(), v], 2),
            None => v,
        };
        self.k[layer] = Some(kk.clone());
        self.v[layer] = Some(vv.clone());
        (kk, vv)
    }
}

/// Causal attention (optionally windowed: `pos - j < window`), chunked over
/// queries. `q` is `[B, H, S, D]`, `k`/`v` are `[B, H, Kn, D]` (GQA already
/// expanded); query `i` sits at absolute position `q_offset + i`.
pub(crate) fn causal_attention(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    q_offset: usize,
    window: Option<usize>,
    chunk: usize,
) -> Tensor<4> {
    let [b, h, s, _d] = q.dims();
    let kn = k.dims()[2];
    let device = q.device();
    let dt = q.dtype();
    let chunk = chunk.max(1);
    let mut outs: Vec<Tensor<4>> = Vec::new();
    let mut q0 = 0;
    while q0 < s {
        let q1 = (q0 + chunk).min(s);
        let pos_first = q_offset + q0;
        let pos_last = q_offset + q1 - 1;
        // Key range: everything up to the last query (causal) and, with a
        // window, at least `pos_first - (window - 1)`.
        let k0 = match window {
            Some(w) => pos_first.saturating_sub(w - 1),
            None => 0,
        };
        let k1 = (pos_last + 1).min(kn);
        let qc = q.clone().slice(s![.., .., q0..q1, ..]);
        let kc = k.clone().slice(s![.., .., k0..k1, ..]);
        let vc = v.clone().slice(s![.., .., k0..k1, ..]);

        let mut sc = qc.matmul(kc.swap_dims(2, 3));
        if sc.dtype() != DType::F32 {
            sc = sc.cast(DType::F32);
        }
        let rows = Tensor::arange(pos_first as i64..(pos_last as i64 + 1), &device).reshape([
            1,
            1,
            q1 - q0,
            1,
        ]);
        let cols = Tensor::arange(k0 as i64..k1 as i64, &device).reshape([1, 1, 1, k1 - k0]);
        let mut keep = cols.clone().lower_equal(rows.clone());
        if let Some(w) = window {
            let w = w as i64;
            keep = keep.bool_and(cols.greater_equal(rows - (w - 1)));
        }
        let fill = keep.bool_not().expand([b, h, q1 - q0, k1 - k0]);
        sc = sc.mask_fill(fill, f32::NEG_INFINITY);
        let p = softmax(sc, 3);
        let p = if dt == DType::F32 { p } else { p.cast(dt) };
        outs.push(p.matmul(vc));
        q0 = q1;
    }
    Tensor::cat(outs, 2)
}

#[derive(Module, Debug)]
pub struct GenAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
}

impl GenAttention {
    fn new(cfg: &TextConfig, spec: &GenLayerSpec, device: &Device) -> Self {
        let h = spec.n_heads;
        let kv = spec.n_kv_heads;
        let d = spec.head_dim;
        let eps = cfg.rms_norm_eps;
        Self {
            q_proj: linear_cfg(cfg.hidden_size, h * d).init(device),
            k_proj: linear_cfg(cfg.hidden_size, kv * d).init(device),
            v_proj: linear_cfg(cfg.hidden_size, kv * d).init(device),
            o_proj: linear_cfg(h * d, cfg.hidden_size).init(device),
            q_norm: RmsNormConfig::new(d).with_epsilon(eps).init(device),
            k_norm: RmsNormConfig::new(d).with_epsilon(eps).init(device),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: Tensor<3>,
        rope: &GenRopeTable,
        spec: &GenLayerSpec,
        gspec: &GenSpec,
        kv: &mut GenKv,
        layer: usize,
        seq_start: usize,
        chunk: usize,
    ) -> Tensor<3> {
        let [b, s, _] = x.dims();
        let (h, kvh, d) = (spec.n_heads, spec.n_kv_heads, spec.head_dim);
        let q = lin(&self.q_proj, x.clone()).reshape([b, s, h, d]);
        let q = rope.apply(self.q_norm.forward(q), seq_start);

        let (k, v) = if spec.is_kv_shared {
            // The K/V of the last non-shared layer of the same type (full
            // length, sliding masks are applied below).
            kv.shared[spec.type_idx]
                .clone()
                .expect("shared K/V stored by the source layer")
        } else {
            let kk = lin(&self.k_proj, x.clone()).reshape([b, s, kvh, d]);
            let raw_v = if spec.k_eq_v {
                // 12B/26B: V is the raw k_proj output through `v_norm`.
                kk.clone().reshape([b, s, kvh, d])
            } else {
                lin(&self.v_proj, x).reshape([b, s, kvh, d])
            };
            let kk = rope.apply(self.k_norm.forward(kk), seq_start);
            let vv = rms_norm_noscale(raw_v, gspec.eps);
            let (kk, vv) = kv.append(layer, kk.swap_dims(1, 2), vv.swap_dims(1, 2));
            if spec.stores_shared_kv {
                kv.shared[spec.type_idx] = Some((kk.clone(), vv.clone()));
            }
            (kk, vv)
        };

        #[cfg(all(feature = "npu", target_arch = "aarch64"))]
        if s > 1 && crate::gemma::layers::prefill_mode() {
            // Generation prefill on the NPU: causal (full layers) or causal +
            // sliding window (sliding layers); decode stays on the CPU.
            let window = if spec.sliding {
                gspec.sliding_window as i64
            } else {
                -1
            };
            let o = burn_rocket::attention_causal_window(
                q.reshape([b, s, h * d]),
                k.reshape([b, s, kvh * d]),
                v.reshape([b, s, kvh * d]),
                h,
                kvh,
                d,
                1.0,
                None,
                window,
            );
            return lin(&self.o_proj, o);
        }

        let q = q.swap_dims(1, 2);
        let k = repeat_kv(k, h / kvh);
        let v = repeat_kv(v, h / kvh);
        let window = spec.sliding.then_some(gspec.sliding_window);
        let o = causal_attention(q, k, v, seq_start, window, chunk);
        let o = o.swap_dims(1, 2).reshape([b, s, h * d]);
        lin(&self.o_proj, o)
    }
}

#[derive(Module, Debug)]
pub struct GenMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl GenMlp {
    fn new(cfg: &TextConfig, spec: &GenLayerSpec, device: &Device) -> Self {
        Self {
            gate_proj: linear_cfg(cfg.hidden_size, spec.intermediate).init(device),
            up_proj: linear_cfg(cfg.hidden_size, spec.intermediate).init(device),
            down_proj: linear_cfg(spec.intermediate, cfg.hidden_size).init(device),
        }
    }

    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let gate = gelu_approximate(lin(&self.gate_proj, x.clone()));
        let up = lin(&self.up_proj, x);
        lin(&self.down_proj, gate * up)
    }
}

#[derive(Module, Debug)]
pub struct GenLayer {
    input_layernorm: RmsNorm,
    self_attn: GenAttention,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    mlp: GenMlp,
    post_feedforward_layernorm: RmsNorm,
    per_layer_input_gate: Linear,
    per_layer_projection: Linear,
    post_per_layer_input_norm: RmsNorm,
    layer_scalar: Param<Tensor<1>>,
}

impl GenLayer {
    fn new(cfg: &TextConfig, spec: &GenLayerSpec, device: &Device) -> Self {
        let eps = cfg.rms_norm_eps;
        Self {
            input_layernorm: RmsNormConfig::new(cfg.hidden_size)
                .with_epsilon(eps)
                .init(device),
            self_attn: GenAttention::new(cfg, spec, device),
            post_attention_layernorm: RmsNormConfig::new(cfg.hidden_size)
                .with_epsilon(eps)
                .init(device),
            pre_feedforward_layernorm: RmsNormConfig::new(cfg.hidden_size)
                .with_epsilon(eps)
                .init(device),
            mlp: GenMlp::new(cfg, spec, device),
            post_feedforward_layernorm: RmsNormConfig::new(cfg.hidden_size)
                .with_epsilon(eps)
                .init(device),
            per_layer_input_gate: linear_cfg(cfg.hidden_size, cfg.hidden_size_per_layer_input)
                .init(device),
            per_layer_projection: linear_cfg(cfg.hidden_size_per_layer_input, cfg.hidden_size)
                .init(device),
            post_per_layer_input_norm: RmsNormConfig::new(cfg.hidden_size)
                .with_epsilon(eps)
                .init(device),
            layer_scalar: Param::from_tensor(Tensor::ones([1], device)),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: Tensor<3>,
        ple_i: Tensor<3>,
        rope: &GenRopeTable,
        spec: &GenLayerSpec,
        gspec: &GenSpec,
        kv: &mut GenKv,
        layer: usize,
        seq_start: usize,
        chunk: usize,
    ) -> Tensor<3> {
        let residual = x.clone();
        let t = Instant::now();
        let h = self.input_layernorm.forward(x);
        T_NORM_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);

        let t = Instant::now();
        let h = self
            .self_attn
            .forward(h, rope, spec, gspec, kv, layer, seq_start, chunk);
        T_ATTN_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);

        let t = Instant::now();
        let h = self.post_attention_layernorm.forward(h);
        T_NORM_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
        let x = residual + h;

        let residual = x.clone();
        let t = Instant::now();
        let h = self.pre_feedforward_layernorm.forward(x);
        T_NORM_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);

        let t = Instant::now();
        let h = self.mlp.forward(h);
        T_MLP_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);

        let t = Instant::now();
        let h = self.post_feedforward_layernorm.forward(h);
        T_NORM_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
        let x = residual + h;

        let t = Instant::now();
        let residual = x.clone();
        let h = gelu_approximate(lin(&self.per_layer_input_gate, x));
        let h = h * ple_i;
        let h = lin(&self.per_layer_projection, h);
        let h = self.post_per_layer_input_norm.forward(h);
        T_PLE_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
        let x = residual + h;
        x * self.layer_scalar.val().reshape([1, 1, 1])
    }
}

#[derive(Module, Debug)]
pub struct GenTextModel {
    embed_tokens: Embedding,
    embed_tokens_per_layer: Embedding,
    /// QAT checkpoints: the PLE table stays packed and rows are dequantized on
    /// lookup (bit-identical values, a fraction of the memory).
    #[module(skip)]
    packed_ple: Option<crate::gemma::qat::PackedTable>,
    per_layer_model_projection: Linear,
    per_layer_projection_norm: RmsNorm,
    layers: Vec<GenLayer>,
    norm: RmsNorm,
    #[module(skip)]
    spec: GenSpec,
}

impl GenTextModel {
    pub fn new(cfg: &TextConfig, device: &Device) -> Self {
        let spec = GenSpec::from_config(cfg);
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| GenLayer::new(cfg, &spec.layers[i], device))
            .collect();
        Self {
            embed_tokens: EmbeddingConfig::new(cfg.vocab_size, cfg.hidden_size).init(device),
            packed_ple: None,
            embed_tokens_per_layer: EmbeddingConfig::new(
                cfg.vocab_size_per_layer_input,
                cfg.num_hidden_layers * cfg.hidden_size_per_layer_input,
            )
            .init(device),
            per_layer_model_projection: linear_cfg(
                cfg.hidden_size,
                cfg.num_hidden_layers * cfg.hidden_size_per_layer_input,
            )
            .init(device),
            per_layer_projection_norm: RmsNormConfig::new(cfg.hidden_size_per_layer_input)
                .with_epsilon(cfg.rms_norm_eps)
                .init(device),
            layers,
            norm: RmsNormConfig::new(cfg.hidden_size)
                .with_epsilon(cfg.rms_norm_eps)
                .init(device),
            spec,
        }
    }

    /// Install a packed PLE table (QAT checkpoints) and drop the f16/f32 copy.
    pub fn set_packed_ple(&mut self, table: crate::gemma::qat::PackedTable) {
        self.packed_ple = Some(table);
    }

    /// Replace the PLE table parameter with a 1-element stub *before* loading:
    /// the packed table is read separately, so the dequantized copy is never
    /// materialized (4.7 GiB in f16, 9.4 GiB in f32).
    pub fn shrink_ple_table(&mut self) {
        let dev = self.embed_tokens.weight.val().device();
        self.embed_tokens_per_layer.weight = self
            .embed_tokens_per_layer
            .weight
            .clone()
            .map(|_| Tensor::zeros([1, 1], &dev));
    }

    pub fn spec(&self) -> &GenSpec {
        &self.spec
    }

    /// Visit every *used* text projection (q/k/v/o, MLP, PLE); the k/v of
    /// KV-shared layers are never called and are skipped.
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    pub fn for_each_projection_mut(&mut self, mut f: impl FnMut(&mut Linear)) {
        for (i, layer) in self.layers.iter_mut().enumerate() {
            if !self.spec.layers[i].is_kv_shared {
                f(&mut layer.self_attn.k_proj);
                f(&mut layer.self_attn.v_proj);
            }
            f(&mut layer.self_attn.q_proj);
            f(&mut layer.self_attn.o_proj);
            f(&mut layer.mlp.gate_proj);
            f(&mut layer.mlp.up_proj);
            f(&mut layer.mlp.down_proj);
            f(&mut layer.per_layer_input_gate);
            f(&mut layer.per_layer_projection);
        }
        f(&mut self.per_layer_model_projection);
    }

    /// Per-layer inputs `[B, S, L, ple_dim]`:
    /// `(norm(projection(x) / sqrt(hidden)) + token_table(ids) * sqrt(ple_dim)) / sqrt(2)`.
    fn per_layer_inputs(&self, ids: Tensor<2, Int>, x: Tensor<3>) -> Tensor<4> {
        let spec = &self.spec;
        let [b, s, _] = x.dims();
        let token = match &self.packed_ple {
            Some(table) => {
                let row_ids: Vec<i32> = ids.clone().into_data().try_to_vec().expect("token ids");
                let row_ids: Vec<u32> = row_ids.iter().map(|&v| v as u32).collect();
                let values = table.gather_f32(&row_ids);
                let device = ids.device();
                Tensor::<3>::from_data(
                    TensorData::new(values, [b, s, spec.num_layers * spec.ple_dim]),
                    &device,
                )
                .mul_scalar((spec.ple_dim as f64).sqrt())
                .reshape([b, s, spec.num_layers, spec.ple_dim])
            }
            None => self
                .embed_tokens_per_layer
                .forward(ids)
                .cast(DType::F32)
                .mul_scalar((spec.ple_dim as f64).sqrt())
                .reshape([b, s, spec.num_layers, spec.ple_dim]),
        };
        let ctx = lin(&self.per_layer_model_projection, x)
            .mul_scalar((spec.hidden as f64).powf(-0.5))
            .reshape([b, s, spec.num_layers, spec.ple_dim]);
        let ctx = self.per_layer_projection_norm.forward(ctx);
        (ctx + token).mul_scalar(std::f64::consts::FRAC_1_SQRT_2)
    }

    /// Runs the decoder over `input_ids` (prefill when `seq_start == 0`, one
    /// token per call otherwise), updating `kv`.
    pub fn forward(
        &self,
        input_ids: Tensor<2, Int>,
        ropes: &GenRopes,
        kv: &mut GenKv,
        seq_start: usize,
        chunk: usize,
    ) -> Tensor<3> {
        let spec = &self.spec;
        let x = self
            .embed_tokens
            .forward(input_ids.clone())
            .cast(DType::F32)
            .mul_scalar((spec.hidden as f64).sqrt());
        self.forward_embeds(input_ids, x, ropes, kv, seq_start, chunk)
    }

    /// The decoder stack over already-embedded inputs; `input_ids` are still the
    /// (media-pad) ids, used for the PLE token-table lookup like the reference.
    pub fn forward_embeds(
        &self,
        input_ids: Tensor<2, Int>,
        x: Tensor<3>,
        ropes: &GenRopes,
        kv: &mut GenKv,
        seq_start: usize,
        chunk: usize,
    ) -> Tensor<3> {
        let spec = &self.spec;
        let [b, s] = input_ids.dims();
        let ple = self.per_layer_inputs(input_ids, x.clone());
        let mut h = x;
        for (i, layer) in self.layers.iter().enumerate() {
            let ls = &spec.layers[i];
            let ple_i = ple
                .clone()
                .slice(s![.., .., i..i + 1, ..])
                .reshape([b, s, spec.ple_dim]);
            h = layer.forward(h, ple_i, ropes.of(ls), ls, spec, kv, i, seq_start, chunk);
        }
        kv.len += s;
        self.norm.forward(h)
    }

    /// Soft-capped logits for the last position of `h`, given the transposed
    /// (tied) LM head `[hidden, vocab]`.
    pub fn logits(&self, h: Tensor<3>, lm_head: &Tensor<2>) -> Tensor<2> {
        let [b, s, _] = h.dims();
        let last = h.slice(s![.., s - 1..s, ..]).reshape([b, self.spec.hidden]);
        // The tied head may be f16 (q8 mode): compute the logits in its dtype and
        // apply the soft cap in f32.
        let last = if last.dtype() != lm_head.dtype() {
            last.cast(lm_head.dtype())
        } else {
            last
        };
        let mut logits = burn::tensor::module::linear(last, lm_head.clone(), None).cast(DType::F32);
        if let Some(c) = self.spec.logit_softcap {
            logits = (logits / c).tanh() * c;
        }
        logits
    }
}

/// Checkpoint root for the generation model: all keys are prefixed `model.`.
#[derive(Module, Debug)]
pub struct GenRoot {
    model: GenInner,
}

#[derive(Module, Debug)]
pub struct GenInner {
    language_model: GenTextModel,
    vision_tower: VisionTower,
    audio_tower: AudioTower,
    embed_vision: MultimodalEmbedder,
    embed_audio: MultimodalEmbedder,
}

impl GenRoot {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        text_cfg: &TextConfig,
        vision_cfg: &VisionConfig,
        audio_cfg: &AudioConfig,
        vision_bounds: &HashMap<String, ClipBounds>,
        audio_bounds: &HashMap<String, ClipBounds>,
        device: &Device,
    ) -> Self {
        Self {
            model: GenInner {
                language_model: GenTextModel::new(text_cfg, device),
                vision_tower: VisionTower::new_with_clip(vision_cfg, vision_bounds, device),
                audio_tower: AudioTower::new(audio_cfg, audio_bounds, device),
                embed_vision: MultimodalEmbedder::new(
                    vision_cfg.hidden_size,
                    text_cfg.hidden_size,
                    device,
                ),
                embed_audio: MultimodalEmbedder::new(
                    audio_cfg.output_proj_dims,
                    text_cfg.hidden_size,
                    device,
                ),
            },
        }
    }

    pub fn text(&self) -> &GenTextModel {
        &self.model.language_model
    }

    pub fn text_mut(&mut self) -> &mut GenTextModel {
        &mut self.model.language_model
    }

    /// Soft tokens for one prepared image: `[num_soft_tokens, text_hidden]`.
    pub fn image_soft_tokens(&self, img: &PreparedImage, chunk: usize) -> Tensor<2> {
        let pooled = self.model.vision_tower.forward(img, chunk, None);
        self.model
            .embed_vision
            .forward(pooled, self.model.vision_tower.spec().eps)
    }

    /// Soft tokens for one audio clip: valid frames only, matching the reference.
    pub fn audio_soft_tokens(
        &self,
        feats: &AudioFeatures,
        debug_dir: Option<&std::path::Path>,
    ) -> Tensor<2> {
        let h = self.model.audio_tower.forward_debug(feats, debug_dir);
        let device = h.device();
        let (m1, _) = subsample_mask(&feats.mask);
        let (m2, _) = subsample_mask(&m1);
        let keep: Vec<i64> = m2
            .iter()
            .enumerate()
            .filter(|(_, v)| **v)
            .map(|(i, _)| i as i64)
            .collect();
        let n = keep.len();
        let idx = Tensor::<1, Int>::from_data(TensorData::new(keep, [n]), &device);
        let h = h.select(0, idx);
        self.model
            .embed_audio
            .forward(h, self.model.audio_tower.spec().eps)
    }

    /// Prefill over `ids` with optional media soft tokens scattered into the
    /// placeholder positions; returns the soft-capped logits of the last position.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill(
        &self,
        ids: &[u32],
        soft: Option<(Vec<usize>, Tensor<2>)>,
        ropes: &GenRopes,
        kv: &mut GenKv,
        lm_head: &Tensor<2>,
        chunk: usize,
        pad_id: u32,
    ) -> Tensor<2> {
        let device = self.model.language_model.embed_table().val().device();
        let mut llm_ids = ids.to_vec();
        if let Some((positions, _)) = &soft {
            for &p in positions {
                llm_ids[p] = pad_id;
            }
        }
        let n = llm_ids.len();
        let ids_i64: Vec<i64> = llm_ids.iter().map(|&t| t as i64).collect();
        let ids_t = Tensor::<2, Int>::from_data(TensorData::new(ids_i64, [1, n]), &device);
        let text = self.text();
        let mut x = text
            .embed_tokens
            .forward(ids_t.clone())
            .cast(DType::F32)
            .mul_scalar((text.spec().hidden as f64).sqrt());
        if let Some((positions, tokens)) = soft {
            let dtype = x.dtype();
            let mut data: Vec<f32> = x.cast(DType::F32).into_data().try_to_vec().expect("embeds");
            let values: Vec<f32> = tokens
                .cast(DType::F32)
                .into_data()
                .try_to_vec()
                .expect("soft tokens");
            let d = text.spec().hidden;
            let dd = values.len() / positions.len().max(1);
            for (i, &pos) in positions.iter().enumerate() {
                data[pos * d..pos * d + dd].copy_from_slice(&values[i * dd..(i + 1) * dd]);
            }
            x = Tensor::<3>::from_data(TensorData::new(data, [1, n, d]), &device).cast(dtype);
        }
        let h = text.forward_embeds(ids_t, x, ropes, kv, 0, chunk);
        text.logits(h, lm_head)
    }
}

impl crate::gemma::inputs::MediaModel for GenRoot {
    fn vision_spec(&self) -> &VisionSpec {
        self.model.vision_tower.spec()
    }

    fn encode_image(
        &self,
        img: &PreparedImage,
        chunk: usize,
        debug_dir: Option<&std::path::Path>,
        layers: Option<&std::path::Path>,
    ) -> Tensor<2> {
        if debug_dir.is_some() {
            let _ = self.model.vision_tower.forward(img, chunk, layers);
        }
        self.image_soft_tokens(img, chunk)
    }

    fn encode_audio(
        &self,
        feats: &AudioFeatures,
        debug_dir: Option<&std::path::Path>,
    ) -> Tensor<2> {
        self.audio_soft_tokens(feats, debug_dir)
    }
}

impl GenTextModel {
    /// The (tied) token-embedding table.
    pub fn embed_table(&self) -> &Param<Tensor<2>> {
        &self.embed_tokens.weight
    }
}
