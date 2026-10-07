//! Gemma 4 audio tower (USM-style conformer, shared by EmbeddingGemma 2),
//! inference only. Reference: `Gemma4AudioModel` and friends in
//! `transformers/models/gemma4/modeling_gemma4.py`.
//!
//! Batch 1 only. `gradient_clipping` is 1e10 in the published configs, far above
//! any activation magnitude, so those clamps are no-ops and skipped.

use std::collections::HashMap;

use burn::module::Param;
use burn::nn::conv::{Conv1d, Conv1dConfig, Conv2d, Conv2dConfig};
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::nn::PaddingConfig2d;
use burn::prelude::*;
use burn::tensor::DType;

use crate::gemma::layers::lin;
use burn::tensor::activation::{relu, sigmoid, silu, softplus};

use crate::gemma::audio_frontend::{AudioFeatures, MEL_BINS, subsample_mask};
use crate::gemma::config::AudioConfig;
use crate::gemma::layers::{ClipBounds, ClippableLinear};

/// Plain-data audio geometry.
#[derive(Debug, Clone)]
pub struct AudioSpec {
    pub hidden: usize,
    pub inter: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub layers: usize,
    pub eps: f64,
    pub chunk: usize,
    pub past: usize,
    pub future: usize,
    pub cap: f64,
    pub invalid: f32,
    pub residual: f64,
    pub out_dims: usize,
    pub conv_kernel: usize,
    pub subsample_channels: [usize; 2],
}

impl AudioSpec {
    pub fn from_config(cfg: &AudioConfig) -> Self {
        Self {
            hidden: cfg.hidden_size,
            inter: cfg.hidden_size * 4,
            heads: cfg.num_attention_heads,
            head_dim: cfg.hidden_size / cfg.num_attention_heads,
            layers: cfg.num_hidden_layers,
            eps: cfg.rms_norm_eps,
            chunk: cfg.attention_chunk_size,
            past: cfg.attention_context_left - 1,
            future: cfg.attention_context_right,
            cap: cfg.attention_logit_cap,
            invalid: -1e9,
            residual: cfg.residual_weight,
            out_dims: cfg.output_proj_dims,
            conv_kernel: cfg.conv_kernel_size,
            subsample_channels: cfg.subsampling_conv_channels,
        }
    }

    /// `context_size // 2 + 1` relative-position rows.
    pub fn context_size(&self) -> usize {
        self.chunk + self.past + self.future
    }
}

fn conv_cfg(in_channels: usize, out_channels: usize) -> Conv2dConfig {
    Conv2dConfig::new([in_channels, out_channels], [3, 3])
        .with_stride([2, 2])
        .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
        .with_bias(false)
}

#[derive(Module, Debug)]
pub struct ConvLayer {
    conv: Conv2d,
    norm: LayerNorm,
}

impl ConvLayer {
    fn new(in_channels: usize, out_channels: usize, eps: f64, device: &Device) -> Self {
        Self {
            conv: conv_cfg(in_channels, out_channels).init(device),
            norm: LayerNormConfig::new(out_channels)
                .with_epsilon(eps)
                .with_bias(false)
                .init(device),
        }
    }

    /// `[1, C_in, T, W] -> [1, C_out, T/2, W/2]`. Invalid input frames are
    /// already zeroed by the feature extractor, so the reference's pre-conv mask
    /// multiply is a no-op.
    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let h = self.conv.forward(x);
        let h = h.permute([0, 2, 3, 1]);
        let h = relu(self.norm.forward(h));
        h.permute([0, 3, 1, 2])
    }
}

#[derive(Module, Debug)]
pub struct SubSampleConv {
    layer0: ConvLayer,
    layer1: ConvLayer,
    input_proj_linear: Linear,
}

impl SubSampleConv {
    fn new(spec: &AudioSpec, device: &Device) -> Self {
        let [c0, c1] = spec.subsample_channels;
        Self {
            layer0: ConvLayer::new(1, c0, spec.eps, device),
            layer1: ConvLayer::new(c0, c1, spec.eps, device),
            input_proj_linear: LinearConfig::new((c0 / 4) * c1, spec.hidden)
                .with_bias(false)
                .init(device),
        }
    }

    /// `[T, MEL_BINS] -> [T/4, hidden]`.
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let [t, m] = x.dims();
        let h = x.reshape([1, 1, t, m]);
        let h = self.layer0.forward(h);
        let h = self.layer1.forward(h);
        let [_, c, t2, w] = h.dims();
        let h = h.permute([0, 2, 3, 1]).reshape([t2, c * w]);
        self.input_proj_linear.forward(h)
    }
}

#[derive(Module, Debug)]
pub struct AudioAttention {
    q_proj: ClippableLinear,
    k_proj: ClippableLinear,
    v_proj: ClippableLinear,
    post: ClippableLinear,
    relative_k_proj: Linear,
    per_dim_scale: Param<Tensor<1>>,
}

impl AudioAttention {
    fn new(cfg: &AudioConfig, bounds: &HashMap<String, ClipBounds>, path: &str, device: &Device) -> Self {
        let spec = AudioSpec::from_config(cfg);
        let clip = |name: &str| -> Option<ClipBounds> {
            // Checkpoints without clipped linears (the QAT export) carry no
            // bound scalars; those towers run unclipped.
            if !cfg.use_clipped_linears {
                return None;
            }
            Some(
                bounds
                    .get(&format!("{path}.{name}"))
                    .copied()
                    .unwrap_or_else(|| panic!("missing clip bounds for {path}.{name}")),
            )
        };
        Self {
            q_proj: ClippableLinear::with_clip(spec.hidden, spec.heads * spec.head_dim, clip("q_proj"), device),
            k_proj: ClippableLinear::with_clip(spec.hidden, spec.heads * spec.head_dim, clip("k_proj"), device),
            v_proj: ClippableLinear::with_clip(spec.hidden, spec.heads * spec.head_dim, clip("v_proj"), device),
            post: ClippableLinear::with_clip(spec.hidden, spec.hidden, clip("post"), device),
            relative_k_proj: LinearConfig::new(spec.hidden, spec.heads * spec.head_dim)
                .with_bias(false)
                .init(device),
            per_dim_scale: Param::from_tensor(Tensor::zeros([spec.head_dim], device)),
        }
    }

    /// Chunked local attention with relative position bias. `x` is `[T, hidden]`,
    /// `pos` is `[context//2 + 1, hidden]`, `mask` is `[1, blocks, chunk, context]`
    /// (True = valid).
    fn forward(
        &self,
        x: Tensor<2>,
        pos: &Tensor<2>,
        mask: &Tensor<4, Bool>,
        spec: &AudioSpec,
    ) -> Tensor<2> {
        let [t, _] = x.dims();
        let (h, d, c) = (spec.heads, spec.head_dim, spec.chunk);
        let device = x.device();

        let q = self.q_proj.forward(x.clone()).reshape([t, h, d]);
        let k = self.k_proj.forward(x.clone()).reshape([t, h, d]);
        let v = self.v_proj.forward(x).reshape([t, h, d]);

        let q_scale = (d as f64).powf(-0.5) / std::f64::consts::LN_2;
        let k_scale = (1.0 + std::f64::consts::E).ln() / std::f64::consts::LN_2;
        let q = q * (softplus(self.per_dim_scale.val().reshape([1, 1, d]), 1.0).mul_scalar(q_scale));
        let k = k.mul_scalar(k_scale);

        // Queries: pad to a whole number of blocks, then [blocks, chunk, H, D].
        let blocks = t.div_ceil(c);
        let tp = blocks * c;
        let q_pad = if tp > t {
            Tensor::cat(
                vec![q, Tensor::zeros([tp - t, h, d], &device)],
                0,
            )
        } else {
            q
        };
        let q_blocks = q_pad.reshape([blocks, c, h, d]).permute([2, 0, 1, 3]);

        // Keys/values: overlapping context windows of `context_size`, strided by
        // `chunk`, with zero padding at both ends of the sequence.
        let pad_l = spec.past;
        let pad_r = spec.future + c - 1;
        let ctx = spec.context_size();
        let kv_windows = |x: Tensor<3>| -> Tensor<4> {
            let x_pad = Tensor::cat(
                vec![
                    Tensor::zeros([pad_l, h, d], &device),
                    x,
                    Tensor::zeros([pad_r, h, d], &device),
                ],
                0,
            );
            let idx: Vec<i64> = (0..blocks * ctx)
                .map(|i| ((i / ctx) * c + (i % ctx)) as i64)
                .collect();
            let idx = Tensor::<1, Int>::from_data(TensorData::new(idx, [blocks * ctx]), &device);
            x_pad
                .select(0, idx)
                .reshape([blocks, ctx, h, d])
                .permute([2, 0, 1, 3])
        };
        let k_ctx = kv_windows(k);
        let v_ctx = kv_windows(v);

        // Content-based scores `[H, blocks, chunk, ctx]`.
        let ac = q_blocks.clone().matmul(k_ctx.swap_dims(2, 3));

        // Relative-position scores via the relative key projection and rel-shift.
        let relk = lin(&self.relative_k_proj, pos.clone())
            .reshape([ctx / 2 + 1, h, d])
            .permute([1, 2, 0]);        let bd = q_blocks
            .reshape([h, blocks * c, d])
            .matmul(relk)
            .reshape([h, blocks, c, ctx / 2 + 1]);
        let rows = ctx / 2 + 1;
        let bd = Tensor::cat(
            vec![bd, Tensor::zeros([h, blocks, c, ctx + 1 - rows], &device)],
            3,
        )
        .reshape([h, blocks, c * (ctx + 1)])
        .slice(s![.., .., 0..c * ctx])
        .reshape([h, blocks, c, ctx]);

        let mut attn = (ac + bd).mul_scalar(1.0 / spec.cap);
        attn = attn.tanh().mul_scalar(spec.cap);
        let fill = mask.clone().expand([h, blocks, c, ctx]).bool_not();
        attn = attn.mask_fill(fill, spec.invalid);

        let p = burn::tensor::activation::softmax(attn, 3);
        let o = p.matmul(v_ctx);
        let o = o
            .permute([1, 2, 0, 3])
            .reshape([tp, h * d])
            .slice(s![0..t, ..]);
        self.post.forward(o)
    }
}

#[derive(Module, Debug)]
pub struct AudioFeedForward {
    ffw_layer_1: ClippableLinear,
    ffw_layer_2: ClippableLinear,
    pre_layer_norm: RmsNorm,
    post_layer_norm: RmsNorm,
}

impl AudioFeedForward {
    fn new(cfg: &AudioConfig, bounds: &HashMap<String, ClipBounds>, path: &str, device: &Device) -> Self {
        let spec = AudioSpec::from_config(cfg);
        let clip = |name: &str| -> Option<ClipBounds> {
            // Checkpoints without clipped linears (the QAT export) carry no
            // bound scalars; those towers run unclipped.
            if !cfg.use_clipped_linears {
                return None;
            }
            Some(
                bounds
                    .get(&format!("{path}.{name}"))
                    .copied()
                    .unwrap_or_else(|| panic!("missing clip bounds for {path}.{name}")),
            )
        };
        Self {
            ffw_layer_1: ClippableLinear::with_clip(spec.hidden, spec.inter, clip("ffw_layer_1"), device),
            ffw_layer_2: ClippableLinear::with_clip(spec.inter, spec.hidden, clip("ffw_layer_2"), device),
            pre_layer_norm: RmsNormConfig::new(spec.hidden).with_epsilon(spec.eps).init(device),
            post_layer_norm: RmsNormConfig::new(spec.hidden).with_epsilon(spec.eps).init(device),
        }
    }

    fn forward(&self, x: Tensor<2>, spec: &AudioSpec) -> Tensor<2> {
        let residual = x.clone();
        let h = self.pre_layer_norm.forward(x);
        let h = self.ffw_layer_1.forward(h);
        let h = silu(h);
        let h = self.ffw_layer_2.forward(h);
        let h = self.post_layer_norm.forward(h);
        h.mul_scalar(spec.residual) + residual
    }
}

#[derive(Module, Debug)]
pub struct AudioLightConv1d {
    linear_start: ClippableLinear,
    linear_end: ClippableLinear,
    depthwise_conv1d: Conv1d,
    pre_layer_norm: RmsNorm,
    conv_norm: RmsNorm,
}

impl AudioLightConv1d {
    fn new(cfg: &AudioConfig, bounds: &HashMap<String, ClipBounds>, path: &str, device: &Device) -> Self {
        let spec = AudioSpec::from_config(cfg);
        let clip = |name: &str| -> Option<ClipBounds> {
            // Checkpoints without clipped linears (the QAT export) carry no
            // bound scalars; those towers run unclipped.
            if !cfg.use_clipped_linears {
                return None;
            }
            Some(
                bounds
                    .get(&format!("{path}.{name}"))
                    .copied()
                    .unwrap_or_else(|| panic!("missing clip bounds for {path}.{name}")),
            )
        };
        Self {
            linear_start: ClippableLinear::with_clip(spec.hidden, spec.hidden * 2, clip("linear_start"), device),
            linear_end: ClippableLinear::with_clip(spec.hidden, spec.hidden, clip("linear_end"), device),
            depthwise_conv1d: Conv1dConfig::new(spec.hidden, spec.hidden, spec.conv_kernel)
                .with_groups(spec.hidden)
                .with_bias(false)
                .init(device),
            pre_layer_norm: RmsNormConfig::new(spec.hidden).with_epsilon(spec.eps).init(device),
            conv_norm: RmsNormConfig::new(spec.hidden).with_epsilon(spec.eps).init(device),
        }
    }

    fn forward(&self, x: Tensor<2>, spec: &AudioSpec) -> Tensor<2> {
        let residual = x.clone();
        let h = self.pre_layer_norm.forward(x);
        let h = self.linear_start.forward(h);
        let [t, d2] = h.dims();
        // GLU over the last dim: first half gated by sigmoid(second half).
        let a = h.clone().slice(s![.., 0..d2 / 2]);
        let b = h.slice(s![.., d2 / 2..d2]);
        let h = a * sigmoid(b);

        // Causal depthwise conv1d (left pad = kernel - 1).
        let device = h.device();
        let hc = h.swap_dims(0, 1).unsqueeze_dim::<3>(0);
        let pad = spec.conv_kernel - 1;
        let hc = Tensor::cat(vec![Tensor::zeros([1, spec.hidden, pad], &device), hc], 2);
        let hc = self.depthwise_conv1d.forward(hc);
        let h = hc.reshape([spec.hidden, t]).swap_dims(0, 1);

        let h = self.conv_norm.forward(h);
        let h = silu(h);
        let h = self.linear_end.forward(h);
        h + residual
    }
}

#[derive(Module, Debug)]
pub struct AudioLayer {
    feed_forward1: AudioFeedForward,
    feed_forward2: AudioFeedForward,
    self_attn: AudioAttention,
    lconv1d: AudioLightConv1d,
    norm_pre_attn: RmsNorm,
    norm_post_attn: RmsNorm,
    norm_out: RmsNorm,
}

impl AudioLayer {
    fn new(cfg: &AudioConfig, bounds: &HashMap<String, ClipBounds>, index: usize, device: &Device) -> Self {
        let spec = AudioSpec::from_config(cfg);
        let base = format!("layers.{index}");
        Self {
            feed_forward1: AudioFeedForward::new(cfg, bounds, &format!("{base}.feed_forward1"), device),
            feed_forward2: AudioFeedForward::new(cfg, bounds, &format!("{base}.feed_forward2"), device),
            self_attn: AudioAttention::new(cfg, bounds, &format!("{base}.self_attn"), device),
            lconv1d: AudioLightConv1d::new(cfg, bounds, &format!("{base}.lconv1d"), device),
            norm_pre_attn: RmsNormConfig::new(spec.hidden).with_epsilon(spec.eps).init(device),
            norm_post_attn: RmsNormConfig::new(spec.hidden).with_epsilon(spec.eps).init(device),
            norm_out: RmsNormConfig::new(spec.hidden).with_epsilon(spec.eps).init(device),
        }
    }

    fn forward(
        &self,
        x: Tensor<2>,
        pos: &Tensor<2>,
        mask: &Tensor<4, Bool>,
        spec: &AudioSpec,
    ) -> Tensor<2> {
        self.forward_debug(x, pos, mask, spec, None, 0)
    }

    fn forward_debug(
        &self,
        x: Tensor<2>,
        pos: &Tensor<2>,
        mask: &Tensor<4, Bool>,
        spec: &AudioSpec,
        debug_dir: Option<&std::path::Path>,
        index: usize,
    ) -> Tensor<2> {
        let dump = |name: &str, t: &Tensor<2>| {
            if let Some(dir) = debug_dir {
                if index == 0 {
                    dump_tensor(dir, name, t);
                }
            }
        };
        let h = self.feed_forward1.forward(x, spec);
        dump("audio_l0_ff1", &h);
        let residual = h.clone();
        let h = self.norm_pre_attn.forward(h);
        dump("audio_l0_pre_attn", &h);
        let h = self.self_attn.forward(h, pos, mask, spec);
        dump("audio_l0_attn", &h);
        let h = self.norm_post_attn.forward(h);
        dump("audio_l0_post_attn", &h);
        let h = h + residual;
        let h = self.lconv1d.forward(h, spec);
        dump("audio_l0_lconv", &h);
        let h = self.feed_forward2.forward(h, spec);
        dump("audio_l0_ff2", &h);
        let h = self.norm_out.forward(h);
        dump("audio_l0_out", &h);
        h
    }
}

#[derive(Module, Debug)]
pub struct AudioTower {
    subsample_conv_projection: SubSampleConv,
    layers: Vec<AudioLayer>,
    output_proj: Linear,
    #[module(skip)]
    spec: AudioSpec,
}

impl AudioTower {
    pub fn new(cfg: &AudioConfig, bounds: &HashMap<String, ClipBounds>, device: &Device) -> Self {
        let spec = AudioSpec::from_config(cfg);
        let layers = (0..spec.layers)
            .map(|i| AudioLayer::new(cfg, bounds, i, device))
            .collect();
        Self {
            subsample_conv_projection: SubSampleConv::new(&spec, device),
            layers,
            output_proj: LinearConfig::new(spec.hidden, spec.out_dims).init(device),
            spec,
        }
    }

    pub fn spec(&self) -> &AudioSpec {
        &self.spec
    }

    /// Sinusoidal relative position embeddings `[context//2 + 1, hidden]`
    /// (concatenated `[sin..., cos...]` layout, positions `context//2 .. 0`).
    pub fn pos_embeddings(&self, device: &Device) -> Tensor<2> {
        let spec = &self.spec;
        let half = spec.hidden / 2;
        let rows = spec.context_size() / 2 + 1;
        let log_inc = (10_000.0f64).ln() / (half as f64 - 1.0).max(1.0);
        let inv: Vec<f32> = (0..half)
            .map(|i| (-(i as f64) * log_inc).exp() as f32)
            .collect();
        let mut data = vec![0.0f32; rows * spec.hidden];
        for (r, pos) in (0..=spec.context_size() / 2).rev().enumerate() {
            for c in 0..half {
                let a = pos as f32 * inv[c];
                data[r * spec.hidden + c] = a.sin();
                data[r * spec.hidden + half + c] = a.cos();
            }
        }
        let dtype = crate::gemma::layers::weight_dtype(&self.output_proj.weight);
        Tensor::<2>::from_data(TensorData::new(data, [rows, spec.hidden]), device).cast(dtype)
    }

    /// `[frames, MEL_BINS]` log-mel features -> `[T/4, out_dims]` soft tokens
    /// (all frames; callers keep only the valid ones).
    pub fn forward(&self, feats: &AudioFeatures) -> Tensor<2> {
        self.forward_debug(feats, None)
    }

    pub fn forward_debug(&self, feats: &AudioFeatures, debug_dir: Option<&std::path::Path>) -> Tensor<2> {
        let device = self.output_proj.weight.val().device();
        let t = feats.frames;
        let dtype = crate::gemma::layers::weight_dtype(&self.output_proj.weight);
        let x = Tensor::<2>::from_data(TensorData::new(feats.mel.clone(), [t, MEL_BINS]), &device)
            .cast(dtype);
        let mut h = self.subsample_conv_projection.forward(x);
        let t2 = h.dims()[0];
        if let Some(dir) = debug_dir {
            dump_tensor(dir, "audio_subsample", &h);
        }

        // Output mask after the two stride-2 convs (see `replace_audio_token`).
        let (m1, _) = subsample_mask(&feats.mask);
        let (m2, _) = subsample_mask(&m1);

        let spec = &self.spec;
        let c = spec.chunk;
        let ctx = spec.context_size();
        let blocks = t2.div_ceil(c);
        let mut mask_data = vec![false; blocks * c * ctx];
        for b in 0..blocks {
            for i in 0..c {
                let qi = b * c + i;
                for j in 0..ctx {
                    let kj = b * c + j - spec.past;
                    let valid = qi < t2
                        && (kj as usize) < t2
                        && m2.get(kj as usize).copied().unwrap_or(false)
                        && qi >= kj as usize
                        && qi - (kj as usize) < spec.past;
                    mask_data[(b * c + i) * ctx + j] = valid;
                }
            }
        }
        let mask = Tensor::<4, Bool>::from_data(TensorData::new(mask_data, [1, blocks, c, ctx]), &device);

        let pos = self.pos_embeddings(&device);
        if let Some(dir) = debug_dir {
            dump_tensor(dir, "audio_pos", &pos);
        }
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward_debug(h, &pos, &mask, spec, debug_dir, i);
            if let Some(dir) = debug_dir {
                dump_tensor(dir, &format!("audio_layer{i}"), &h);
            }
        }
        let h = lin(&self.output_proj, h);
        if let Some(dir) = debug_dir {
            dump_tensor(dir, "audio_out", &h);
        }
        h
    }
}

/// Write a 2-D f32 tensor as raw little-endian data (debug).
fn dump_tensor(dir: &std::path::Path, name: &str, t: &Tensor<2>) {
    let dims = t.dims();
    let values: Vec<f32> = t
        .clone()
        .cast(DType::F32)
        .into_data()
        .try_to_vec()
        .expect("f32 dump");
    let path = dir.join(format!("{name}.bin"));
    let mut bytes = Vec::with_capacity(values.len() * 4 + 16);
    bytes.extend_from_slice(&(dims[0] as u32).to_le_bytes());
    bytes.extend_from_slice(&(dims[1] as u32).to_le_bytes());
    for v in &values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(&path, &bytes).expect("write debug dump");
}
