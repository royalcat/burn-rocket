//! EmbeddingGemma 2 text backbone: an adapted, bidirectional Gemma 4 decoder
//! (Burn, inference only), plus the model assembly that pairs it with the
//! Gemma 4 vision and audio towers (see `crate::gemma::{vision, audio}`).
//!
//! Reference: HF `transformers/models/embedding_gemma2/modeling_embedding_gemma2.py`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use burn::module::Param;
use burn::nn::{Embedding, EmbeddingConfig, Linear, RmsNorm, RmsNormConfig};
use burn::prelude::*;
use burn::tensor::{DType, Int, s};

use crate::gemma::audio::AudioTower;
use crate::gemma::audio_frontend::{AudioFeatures, subsample_mask};
use crate::gemma::config::{AudioConfig, TextConfig, VisionConfig};
use crate::gemma::layers::{ClipBounds, lin};
use crate::gemma::media::PreparedImage;
use crate::gemma::vision::{MultimodalEmbedder, VisionTower};
use crate::gemma::{chunked_attention, linear_cfg, repeat_kv};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Stage timers (attention / mlp / norms / per-layer embeddings), microseconds.
// ---------------------------------------------------------------------------

static T_ATTN_US: AtomicU64 = AtomicU64::new(0);
static T_MLP_US: AtomicU64 = AtomicU64::new(0);
static T_NORM_US: AtomicU64 = AtomicU64::new(0);
static T_PLE_US: AtomicU64 = AtomicU64::new(0);

/// `(attention_s, mlp_s, norms_s, ple_s)` accumulated since the last reset.
pub fn stage_stats() -> (f64, f64, f64, f64) {
    (
        T_ATTN_US.load(Ordering::Relaxed) as f64 / 1e6,
        T_MLP_US.load(Ordering::Relaxed) as f64 / 1e6,
        T_NORM_US.load(Ordering::Relaxed) as f64 / 1e6,
        T_PLE_US.load(Ordering::Relaxed) as f64 / 1e6,
    )
}

pub fn stage_stats_reset() {
    T_ATTN_US.store(0, Ordering::Relaxed);
    T_MLP_US.store(0, Ordering::Relaxed);
    T_NORM_US.store(0, Ordering::Relaxed);
    T_PLE_US.store(0, Ordering::Relaxed);
}

/// Per-layer attention geometry, resolved from `layer_types` + `per_layer_config`.
#[derive(Debug, Clone)]
pub struct LayerSpec {
    pub head_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub sliding: bool,
}

/// Plain-data model spec (kept out of the module tree).
#[derive(Debug, Clone)]
pub struct TextSpec {
    pub hidden_size: usize,
    pub embedding_dim: usize,
    pub ple_dim: usize,
    pub num_layers: usize,
    pub eps: f64,
    pub sliding_window: usize,
    pub sliding_theta: f64,
    pub full_theta: f64,
    pub layers: Vec<LayerSpec>,
}

impl TextSpec {
    pub fn from_config(cfg: &TextConfig) -> Self {
        let types = cfg.layer_types();
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| {
                let ov = cfg.layer_override(i);
                LayerSpec {
                    head_dim: ov.head_dim.unwrap_or_else(|| cfg.head_dim()),
                    n_heads: cfg.num_attention_heads,
                    n_kv_heads: ov.num_key_value_heads.unwrap_or(cfg.num_key_value_heads),
                    sliding: types[i] == "sliding_attention",
                }
            })
            .collect();
        let sliding_theta = cfg
            .rope_parameters
            .sliding_attention
            .as_ref()
            .and_then(|r| r.rope_theta)
            .unwrap_or(10_000.0);
        let full_theta = cfg
            .rope_parameters
            .full_attention
            .as_ref()
            .and_then(|r| r.rope_theta)
            .unwrap_or(1_000_000.0);
        Self {
            hidden_size: cfg.hidden_size,
            embedding_dim: cfg.embedding_dim,
            ple_dim: cfg.hidden_size_per_layer_input,
            num_layers: cfg.num_hidden_layers,
            eps: cfg.rms_norm_eps,
            sliding_window: cfg.sliding_window,
            sliding_theta,
            full_theta,
            layers,
        }
    }
}

/// Precomputed RoPE cos/sin table `[max_seq, head_dim / 2]` (GPT-NeoX rotate-half
/// layout: `cos = cat([f, f])`, pairs are the two halves of the head).
pub struct RopeTable {
    cos: Tensor<2>,
    sin: Tensor<2>,
}

impl RopeTable {
    pub fn new(max_seq: usize, head_dim: usize, theta: f64, dtype: DType, device: &Device) -> Self {
        let half = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| (1.0 / theta.powf(2.0 * i as f64 / head_dim as f64)) as f32)
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
        Self { cos, sin }
    }

    /// Applies RoPE to `x` of shape `[batch, seq, heads, head_dim]`.
    pub fn apply(&self, x: Tensor<4>) -> Tensor<4> {
        let [_, s, _, _] = x.dims();
        let cos = self.cos.clone().slice(s![0..s, ..]);
        let sin = self.sin.clone().slice(s![0..s, ..]);
        crate::gemma::layers::rope_apply(x, cos, sin)
    }
}

#[derive(Module, Debug)]
pub struct TextAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    /// Set in `--npu --npu-attn npu` mode: attention runs on the RK3588 NPU.
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    #[module(skip)]
    npu_attn: bool,
}

impl TextAttention {
    fn forward(
        &self,
        x: Tensor<3>,
        rope: &RopeTable,
        spec: &LayerSpec,
        sliding_window: usize,
        eps: f64,
        chunk: usize,
    ) -> Tensor<3> {
        let [b, s, _] = x.dims();
        let (h, kv, d) = (spec.n_heads, spec.n_kv_heads, spec.head_dim);
        let q = lin(&self.q_proj, x.clone()).reshape([b, s, h, d]);
        let k = lin(&self.k_proj, x.clone()).reshape([b, s, kv, d]);
        let v = lin(&self.v_proj, x).reshape([b, s, kv, d]);

        let q = rope.apply(crate::gemma::layers::rms_norm(&self.q_norm, q));
        let k = rope.apply(crate::gemma::layers::rms_norm(&self.k_norm, k));
        let v = crate::gemma::layers::rms_norm_noscale(v, eps);

        #[cfg(all(feature = "npu", target_arch = "aarch64"))]
        if self.npu_attn {
            // NPU attention takes the [1, S, H*D] layout; sliding layers get the
            // symmetric band mask, full layers no mask.
            let qf = q.reshape([b, s, h * d]);
            let kf = k.reshape([b, s, kv * d]);
            let vf = v.reshape([b, s, kv * d]);
            if spec.sliding && s > chunk {
                // Chunked band attention: a query chunk only attends to the keys
                // within `sliding_window` of it, so the score matrix, the host
                // softmax and the mask stay O(s * (chunk + 2*window)) instead of
                // O(s^2) (which OOMs the board around 30k).
                let w = sliding_window;
                let chunk = chunk.max(1);
                let mut outs: Vec<Tensor<3>> = Vec::new();
                let mut q0 = 0;
                while q0 < s {
                    let q1 = (q0 + chunk).min(s);
                    let k0 = q0.saturating_sub(w);
                    let k1 = (q1 + w).min(s);
                    let qc = qf.clone().slice(s![.., q0..q1, ..]);
                    let kc = kf.clone().slice(s![.., k0..k1, ..]);
                    let vc = vf.clone().slice(s![.., k0..k1, ..]);
                    outs.push(burn_rocket::attention_window_block(
                        qc, kc, vc, h, kv, d, 1.0, None, w as i64, q0 as i64, k0 as i64,
                    ));
                    q0 = q1;
                }
                return lin(&self.o_proj, Tensor::cat(outs, 1));
            }
            let window = if spec.sliding {
                sliding_window as i64
            } else {
                -1
            };
            let o = burn_rocket::attention_window(qf, kf, vf, h, kv, d, 1.0, None, window);
            return lin(&self.o_proj, o);
        }

        let q = q.swap_dims(1, 2);
        let k = repeat_kv(k.swap_dims(1, 2), h / kv);
        let v = repeat_kv(v.swap_dims(1, 2), h / kv);
        let window = spec.sliding.then_some(sliding_window);
        let o = chunked_attention(q, k, v, window, chunk);
        let o = o.swap_dims(1, 2).reshape([b, s, h * d]);
        lin(&self.o_proj, o)
    }
}

#[derive(Module, Debug)]
pub struct TextMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl TextMlp {
    fn new(cfg: &TextConfig, device: &Device) -> Self {
        Self {
            gate_proj: linear_cfg(cfg.hidden_size, cfg.intermediate_size).init(device),
            up_proj: linear_cfg(cfg.hidden_size, cfg.intermediate_size).init(device),
            down_proj: linear_cfg(cfg.intermediate_size, cfg.hidden_size).init(device),
        }
    }

    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let gate = lin(&self.gate_proj, x.clone());
        let up = lin(&self.up_proj, x);
        lin(&self.down_proj, crate::gemma::layers::gelu_mul(gate, up))
    }
}

/// The third residual sub-block of a decoder layer: gate the per-layer embedding
/// slice into the residual stream.
#[derive(Module, Debug)]
pub struct TextPleBlock {
    per_layer_input_gate: Linear,
    per_layer_projection: Linear,
    post_per_layer_input_norm: RmsNorm,
}

impl TextPleBlock {
    fn new(cfg: &TextConfig, device: &Device) -> Self {
        Self {
            per_layer_input_gate: linear_cfg(cfg.hidden_size, cfg.hidden_size_per_layer_input)
                .init(device),
            per_layer_projection: linear_cfg(cfg.hidden_size_per_layer_input, cfg.hidden_size)
                .init(device),
            post_per_layer_input_norm: RmsNormConfig::new(cfg.hidden_size)
                .with_epsilon(cfg.rms_norm_eps)
                .init(device),
        }
    }

    /// Returns the PLE contribution (residual add and `layer_scalar` are applied
    /// by the caller, matching the reference).
    fn forward(&self, x: Tensor<3>, per_layer_input: Tensor<3>) -> Tensor<3> {
        let gate = lin(&self.per_layer_input_gate, x);
        let h = crate::gemma::layers::gelu_mul(gate, per_layer_input);
        let h = lin(&self.per_layer_projection, h);
        crate::gemma::layers::rms_norm(&self.post_per_layer_input_norm, h)
    }
}

/// Projection-only per-layer embeddings (no token-identity table in
/// EmbeddingGemma 2).
#[derive(Module, Debug)]
pub struct TextPle {
    per_layer_model_projection: Linear,
    per_layer_projection_norm: RmsNorm,
}

impl TextPle {
    fn new(cfg: &TextConfig, device: &Device) -> Self {
        Self {
            per_layer_model_projection: linear_cfg(
                cfg.hidden_size,
                cfg.num_hidden_layers * cfg.hidden_size_per_layer_input,
            )
            .init(device),
            per_layer_projection_norm: RmsNormConfig::new(cfg.hidden_size_per_layer_input)
                .with_epsilon(cfg.rms_norm_eps)
                .init(device),
        }
    }

    /// `[B, S, hidden] -> [B, S, num_layers, ple_dim]`.
    fn forward(&self, x: Tensor<3>, spec: &TextSpec) -> Tensor<4> {
        let [b, s, _] = x.dims();
        let scale = (spec.hidden_size as f64).powf(-0.5);
        let p = lin(&self.per_layer_model_projection, x).mul_scalar(scale);
        let p = p.reshape([b, s, spec.num_layers, spec.ple_dim]);
        crate::gemma::layers::rms_norm(&self.per_layer_projection_norm, p)
    }
}

#[derive(Module, Debug)]
pub struct TextLayer {
    input_layernorm: RmsNorm,
    self_attn: TextAttention,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    mlp: TextMlp,
    post_feedforward_layernorm: RmsNorm,
    ple_block: TextPleBlock,
    layer_scalar: Param<Tensor<1>>,
}

impl TextLayer {
    /// `cfg` must be the *per-layer resolved* geometry (head_dim / kv heads of
    /// this layer), as reported by `TextSpec::layers[i]`.
    fn new(cfg: &TextConfig, spec: &LayerSpec, device: &Device) -> Self {
        let h = spec.n_heads;
        let kv = spec.n_kv_heads;
        let d = spec.head_dim;
        let eps = cfg.rms_norm_eps;
        Self {
            input_layernorm: RmsNormConfig::new(cfg.hidden_size)
                .with_epsilon(eps)
                .init(device),
            self_attn: TextAttention {
                q_proj: linear_cfg(cfg.hidden_size, h * d).init(device),
                k_proj: linear_cfg(cfg.hidden_size, kv * d).init(device),
                v_proj: linear_cfg(cfg.hidden_size, kv * d).init(device),
                o_proj: linear_cfg(h * d, cfg.hidden_size).init(device),
                q_norm: RmsNormConfig::new(d).with_epsilon(eps).init(device),
                k_norm: RmsNormConfig::new(d).with_epsilon(eps).init(device),
                #[cfg(all(feature = "npu", target_arch = "aarch64"))]
                npu_attn: false,
            },
            post_attention_layernorm: RmsNormConfig::new(cfg.hidden_size)
                .with_epsilon(eps)
                .init(device),
            pre_feedforward_layernorm: RmsNormConfig::new(cfg.hidden_size)
                .with_epsilon(eps)
                .init(device),
            mlp: TextMlp::new(cfg, device),
            post_feedforward_layernorm: RmsNormConfig::new(cfg.hidden_size)
                .with_epsilon(eps)
                .init(device),
            ple_block: TextPleBlock::new(cfg, device),
            layer_scalar: Param::from_tensor(Tensor::ones([1], device)),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: Tensor<3>,
        ple_i: Tensor<3>,
        rope: &RopeTable,
        spec: &LayerSpec,
        sliding_window: usize,
        eps: f64,
        chunk: usize,
    ) -> Tensor<3> {
        let residual = x.clone();
        let t = Instant::now();
        let h = crate::gemma::layers::rms_norm(&self.input_layernorm, x);
        T_NORM_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);

        let t = Instant::now();
        let h = self
            .self_attn
            .forward(h, rope, spec, sliding_window, eps, chunk);
        T_ATTN_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);

        let t = Instant::now();
        let h = crate::gemma::layers::rms_norm(&self.post_attention_layernorm, h);
        T_NORM_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
        let x = residual + h;

        let residual = x.clone();
        let t = Instant::now();
        let h = crate::gemma::layers::rms_norm(&self.pre_feedforward_layernorm, x);
        T_NORM_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);

        let t = Instant::now();
        let h = self.mlp.forward(h);
        T_MLP_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);

        let t = Instant::now();
        let h = crate::gemma::layers::rms_norm(&self.post_feedforward_layernorm, h);
        T_NORM_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
        let x = residual + h;

        let t = Instant::now();
        let residual = x.clone();
        let h = self.ple_block.forward(x, ple_i);
        T_PLE_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
        let x = residual + h;
        x * self.layer_scalar.val().reshape([1, 1, 1])
    }
}

#[derive(Module, Debug)]
pub struct TextModel {
    embed_tokens: Embedding,
    ple: TextPle,
    layers: Vec<TextLayer>,
    norm: RmsNorm,
    embedding_projection: Linear,
    #[module(skip)]
    spec: TextSpec,
}

/// Root of the checkpoint module tree: keys are prefixed `language_model.*`
/// (the vision tower is added by `vision.rs`).
#[derive(Module, Debug)]
pub struct Emb2Model {
    language_model: TextModel,
    vision_tower: VisionTower,
    embed_vision: MultimodalEmbedder,
    audio_tower: AudioTower,
    embed_audio: MultimodalEmbedder,
}

impl Emb2Model {
    pub fn new(
        text_cfg: &TextConfig,
        vision_cfg: &VisionConfig,
        audio_cfg: &AudioConfig,
        audio_bounds: &HashMap<String, ClipBounds>,
        device: &Device,
    ) -> Self {
        let text_hidden = text_cfg.hidden_size;
        let vision_hidden = vision_cfg.hidden_size;
        let audio_dims = audio_cfg.output_proj_dims;
        Self {
            language_model: TextModel::new(text_cfg, device),
            vision_tower: VisionTower::new(vision_cfg, device),
            embed_vision: MultimodalEmbedder::new(vision_hidden, text_hidden, device),
            audio_tower: AudioTower::new(audio_cfg, audio_bounds, device),
            embed_audio: MultimodalEmbedder::new(audio_dims, text_hidden, device),
        }
    }

    pub fn text(&self) -> &TextModel {
        &self.language_model
    }

    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    pub fn text_mut(&mut self) -> &mut TextModel {
        &mut self.language_model
    }

    /// Soft tokens for one prepared image: `[num_soft_tokens, text_hidden]`.
    pub fn image_soft_tokens(&self, img: &PreparedImage, chunk: usize) -> Tensor<2> {
        let pooled = self.vision_tower.forward(img, chunk, None);
        self.embed_vision
            .forward(pooled, self.vision_tower.spec().eps)
    }

    pub fn audio_soft_tokens_debug(
        &self,
        feats: &AudioFeatures,
        debug_dir: Option<&std::path::Path>,
    ) -> Tensor<2> {
        let h = self.audio_tower.forward_debug(feats, debug_dir);
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
        self.embed_audio.forward(h, self.audio_tower.spec().eps)
    }

    /// Mean-pooled embedding of `input_ids`, with optional image soft tokens
    /// scattered into the positions listed in `soft`.
    pub fn embed_ids(
        &self,
        input_ids: Tensor<2, Int>,
        rope: &TextRope,
        chunk: usize,
        soft: Option<(Vec<usize>, Tensor<2>)>,
    ) -> Tensor<2> {
        let [b, s] = input_ids.dims();
        let d = self.language_model.spec().hidden_size;
        let dim = self.language_model.embedding_dim();
        let device = input_ids.device();
        let mut x = self.language_model.embed_tokens_scaled(input_ids);
        if let Some((positions, tokens)) = soft {
            let dtype = x.dtype();
            let mut data: Vec<f32> = x
                .cast(DType::F32)
                .into_data()
                .try_to_vec()
                .expect("embeddings f32");
            let values: Vec<f32> = tokens
                .cast(DType::F32)
                .into_data()
                .try_to_vec()
                .expect("soft tokens f32");
            let dd = values.len() / positions.len().max(1);
            for (i, &pos) in positions.iter().enumerate() {
                data[pos * d..pos * d + dd].copy_from_slice(&values[i * dd..(i + 1) * dd]);
            }
            x = Tensor::<3>::from_data(TensorData::new(data, [b, s, d]), &device).cast(dtype);
        }
        let h = self.language_model.forward_embeds(x, rope, chunk);
        h.mean_dim(1).reshape([b, dim])
    }
}

impl crate::gemma::inputs::MediaModel for Emb2Model {
    fn vision_spec(&self) -> &crate::gemma::vision::VisionSpec {
        self.vision_tower.spec()
    }

    fn encode_image(
        &self,
        img: &PreparedImage,
        chunk: usize,
        debug_dir: Option<&std::path::Path>,
        layers: Option<&std::path::Path>,
    ) -> Tensor<2> {
        if debug_dir.is_some() {
            let _ = self.vision_tower.forward(img, chunk, layers);
        }
        self.image_soft_tokens(img, chunk)
    }

    fn encode_audio(
        &self,
        feats: &AudioFeatures,
        debug_dir: Option<&std::path::Path>,
    ) -> Tensor<2> {
        self.audio_soft_tokens_debug(feats, debug_dir)
    }
}

impl TextModel {
    pub fn new(cfg: &TextConfig, device: &Device) -> Self {
        let spec = TextSpec::from_config(cfg);
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| TextLayer::new(cfg, &spec.layers[i], device))
            .collect();
        Self {
            embed_tokens: EmbeddingConfig::new(cfg.vocab_size, cfg.hidden_size).init(device),
            ple: TextPle::new(cfg, device),
            layers,
            norm: RmsNormConfig::new(cfg.hidden_size)
                .with_epsilon(cfg.rms_norm_eps)
                .init(device),
            embedding_projection: linear_cfg(cfg.hidden_size, cfg.embedding_dim).init(device),
            spec,
        }
    }

    pub fn spec(&self) -> &TextSpec {
        &self.spec
    }

    /// Visit every text projection weight (q/k/v/o, MLP, PLE, output projection).
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    pub fn for_each_projection_mut(&mut self, mut f: impl FnMut(&mut Linear)) {
        for layer in &mut self.layers {
            f(&mut layer.self_attn.q_proj);
            f(&mut layer.self_attn.k_proj);
            f(&mut layer.self_attn.v_proj);
            f(&mut layer.self_attn.o_proj);
            f(&mut layer.mlp.gate_proj);
            f(&mut layer.mlp.up_proj);
            f(&mut layer.mlp.down_proj);
            f(&mut layer.ple_block.per_layer_input_gate);
            f(&mut layer.ple_block.per_layer_projection);
        }
        f(&mut self.ple.per_layer_model_projection);
        f(&mut self.embedding_projection);
    }

    /// Enable/disable NPU attention on every layer (`--npu-attn`).
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    pub fn set_npu_attn(&mut self, on: bool) {
        for layer in &mut self.layers {
            layer.self_attn.npu_attn = on;
        }
    }

    /// RoPE tables for the sequence length `max_seq`: sliding and full layers
    /// use different head dims and thetas.
    pub fn rope_tables(&self, max_seq: usize, dtype: DType, device: &Device) -> TextRope {
        let mut sliding = None;
        let mut full = None;
        for ls in &self.spec.layers {
            if ls.sliding {
                if sliding.is_none() {
                    sliding = Some(RopeTable::new(
                        max_seq,
                        ls.head_dim,
                        self.spec.sliding_theta,
                        dtype,
                        device,
                    ));
                }
            } else if full.is_none() {
                full = Some(RopeTable::new(
                    max_seq,
                    ls.head_dim,
                    self.spec.full_theta,
                    dtype,
                    device,
                ));
            }
        }
        TextRope { sliding, full }
    }

    /// The scaled token embedding (`x * sqrt(hidden)`).
    pub fn embed_tokens_scaled(&self, input_ids: Tensor<2, Int>) -> Tensor<3> {
        let scale = (self.spec.hidden_size as f64).sqrt();
        self.embed_tokens.forward(input_ids).mul_scalar(scale)
    }

    pub fn embedding_dim(&self) -> usize {
        self.spec.embedding_dim
    }

    /// The decoder stack over already-embedded inputs `[B, S, hidden]`.
    pub fn forward_embeds(&self, x: Tensor<3>, rope: &TextRope, chunk: usize) -> Tensor<3> {
        let [b, s, _] = x.dims();
        let spec = &self.spec;
        let ple = self.ple.forward(x.clone(), spec);
        let mut h = x;
        for (i, layer) in self.layers.iter().enumerate() {
            let ls = &spec.layers[i];
            let ple_i = ple
                .clone()
                .slice(s![.., .., i..i + 1, ..])
                .reshape([b, s, spec.ple_dim]);
            let rope = if ls.sliding {
                rope.sliding.as_ref().expect("sliding rope table")
            } else {
                rope.full.as_ref().expect("full rope table")
            };
            h = layer.forward(h, ple_i, rope, ls, spec.sliding_window, spec.eps, chunk);
        }
        lin(
            &self.embedding_projection,
            crate::gemma::layers::rms_norm(&self.norm, h),
        )
    }
}

/// The two RoPE tables used by a text backbone (only the layer types present).
pub struct TextRope {
    sliding: Option<RopeTable>,
    full: Option<RopeTable>,
}
