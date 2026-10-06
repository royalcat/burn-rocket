//! Gemma 4 vision tower (shared by EmbeddingGemma 2 and Gemma 4), inference only.
//!
//! Reference: `transformers/models/gemma4/modeling_gemma4.py` (Gemma4VisionModel
//! and friends). Only the batch-1, no-padding path is implemented: padding
//! patches are masked out of attention in the reference, so processing the real
//! patches alone is numerically equivalent.

use burn::module::Param;
use burn::nn::{Linear, RmsNorm, RmsNormConfig};
use burn::prelude::*;
use burn::tensor::DType;
use burn::tensor::activation::gelu_approximate;

use crate::config::VisionConfig;
use crate::layers::ClippableLinear;
use crate::media::PreparedImage;
use crate::layers::lin;
use crate::model::{chunked_attention, linear_cfg, repeat_kv, rms_norm_noscale};

/// Plain-data vision geometry.
#[derive(Debug, Clone)]
pub struct VisionSpec {
    pub hidden: usize,
    pub inter: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub eps: f64,
    pub patch_size: usize,
    pub pooling: usize,
    pub position_embedding_size: usize,
    pub rope_theta: f64,
}

impl VisionSpec {
    pub fn from_config(cfg: &VisionConfig) -> Self {
        Self {
            hidden: cfg.hidden_size,
            inter: cfg.intermediate_size,
            layers: cfg.num_hidden_layers,
            heads: cfg.num_attention_heads,
            kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim(),
            eps: cfg.rms_norm_eps,
            patch_size: cfg.patch_size,
            pooling: cfg.pooling_kernel_size,
            position_embedding_size: cfg.position_embedding_size,
            rope_theta: cfg
                .rope_parameters
                .as_ref()
                .and_then(|r| r.rope_theta)
                .unwrap_or(100.0),
        }
    }

    pub fn patch_pixels(&self) -> usize {
        3 * self.patch_size * self.patch_size
    }
}

/// Axial 2-D RoPE: 16 frequencies per axis (head_dim/4), physical layout
/// `H-H-W-W` over the head (see `recomposition_frequencies`).
pub struct AxialRope {
    cos_x: Tensor<2>,
    sin_x: Tensor<2>,
    cos_y: Tensor<2>,
    sin_y: Tensor<2>,
    head_dim: usize,
}

impl AxialRope {
    pub fn new(
        img: &PreparedImage,
        head_dim: usize,
        theta: f64,
        dtype: DType,
        device: &Device,
    ) -> Self {
        let spatial = head_dim / 2;
        let n = spatial / 2;
        // inv_freq = 1 / theta^(arange(0, spatial, 2) / spatial)
        let inv_freq: Vec<f32> = (0..n)
            .map(|i| (1.0 / theta.powf((2.0 * i as f64) / spatial as f64)) as f32)
            .collect();
        let p = img.num_patches();
        let (mut cx, mut sx, mut cy, mut sy) = (
            Vec::with_capacity(p * n),
            Vec::with_capacity(p * n),
            Vec::with_capacity(p * n),
            Vec::with_capacity(p * n),
        );
        for i in 0..p {
            for f in &inv_freq {
                let ax = img.xs[i] as f32 * *f;
                let ay = img.ys[i] as f32 * *f;
                cx.push(ax.cos());
                sx.push(ax.sin());
                cy.push(ay.cos());
                sy.push(ay.sin());
            }
        }
        let t2 = |v: Vec<f32>| {
            Tensor::<2>::from_data(TensorData::new(v, [p, n]), device).cast(dtype)
        };
        Self {
            cos_x: t2(cx),
            sin_x: t2(sx),
            cos_y: t2(cy),
            sin_y: t2(sy),
            head_dim,
        }
    }

    /// Applies axial RoPE to `x` of shape `[patches, heads, head_dim]`.
    pub fn apply(&self, x: Tensor<3>) -> Tensor<3> {
        let d = self.head_dim;
        let half = d / 2;
        let quarter = half / 2;
        let xa = x.clone().slice(s![.., .., 0..quarter]);
        let xb = x.clone().slice(s![.., .., quarter..half]);
        let xc = x.clone().slice(s![.., .., half..half + quarter]);
        let xd = x.slice(s![.., .., half + quarter..d]);
        let cx = self.cos_x.clone().unsqueeze_dim::<3>(1);
        let sx = self.sin_x.clone().unsqueeze_dim::<3>(1);
        let cy = self.cos_y.clone().unsqueeze_dim::<3>(1);
        let sy = self.sin_y.clone().unsqueeze_dim::<3>(1);
        let y0 = xa.clone() * cx.clone() - xb.clone() * sx.clone();
        let y1 = xb * cx + xa * sx;
        let y2 = xc.clone() * cy.clone() - xd.clone() * sy.clone();
        let y3 = xd * cy + xc * sy;
        Tensor::cat(vec![y0, y1, y2, y3], 2)
    }
}

#[derive(Module, Debug)]
pub struct PatchEmbedder {
    input_proj: Linear,
    position_embedding_table: Param<Tensor<3>>,
}

#[derive(Module, Debug)]
pub struct VisionAttention {
    q_proj: ClippableLinear,
    k_proj: ClippableLinear,
    v_proj: ClippableLinear,
    o_proj: ClippableLinear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
}

impl VisionAttention {
    fn new(cfg: &VisionConfig, spec: &VisionSpec, device: &Device) -> Self {
        let (h, kv, d) = (spec.heads, spec.kv_heads, spec.head_dim);
        Self {
            q_proj: ClippableLinear::new(spec.hidden, h * d, device),
            k_proj: ClippableLinear::new(spec.hidden, kv * d, device),
            v_proj: ClippableLinear::new(spec.hidden, kv * d, device),
            o_proj: ClippableLinear::new(h * d, spec.hidden, device),
            q_norm: RmsNormConfig::new(d).with_epsilon(cfg.rms_norm_eps).init(device),
            k_norm: RmsNormConfig::new(d).with_epsilon(cfg.rms_norm_eps).init(device),
        }
    }

    fn forward(&self, x: Tensor<2>, rope: &AxialRope, spec: &VisionSpec, chunk: usize) -> Tensor<2> {
        let p = x.dims()[0];
        let (h, kv, d) = (spec.heads, spec.kv_heads, spec.head_dim);
        let q = self.q_proj.forward(x.clone()).reshape([p, h, d]);
        let k = self.k_proj.forward(x.clone()).reshape([p, kv, d]);
        let v = self.v_proj.forward(x).reshape([p, kv, d]);

        let q = rope.apply(self.q_norm.forward(q));
        let k = rope.apply(self.k_norm.forward(k));
        let v = rms_norm_noscale(v, spec.eps);

        let q = q.swap_dims(0, 1).unsqueeze_dim::<4>(0);
        let k = repeat_kv(k.swap_dims(0, 1).unsqueeze_dim::<4>(0), h / kv);
        let v = repeat_kv(v.swap_dims(0, 1).unsqueeze_dim::<4>(0), h / kv);
        let o = chunked_attention(q, k, v, None, chunk);
        let o = o.reshape([h, p, d]).swap_dims(0, 1).reshape([p, h * d]);
        self.o_proj.forward(o)
    }
}

#[derive(Module, Debug)]
pub struct VisionMlp {
    gate_proj: ClippableLinear,
    up_proj: ClippableLinear,
    down_proj: ClippableLinear,
}

impl VisionMlp {
    fn new(spec: &VisionSpec, device: &Device) -> Self {
        Self {
            gate_proj: ClippableLinear::new(spec.hidden, spec.inter, device),
            up_proj: ClippableLinear::new(spec.hidden, spec.inter, device),
            down_proj: ClippableLinear::new(spec.inter, spec.hidden, device),
        }
    }

    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let gate = gelu_approximate(self.gate_proj.forward(x.clone()));
        let up = self.up_proj.forward(x);
        self.down_proj.forward(gate * up)
    }
}

#[derive(Module, Debug)]
pub struct VisionLayer {
    input_layernorm: RmsNorm,
    self_attn: VisionAttention,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    mlp: VisionMlp,
    post_feedforward_layernorm: RmsNorm,
}

impl VisionLayer {
    fn new(cfg: &VisionConfig, spec: &VisionSpec, device: &Device) -> Self {
        let eps = cfg.rms_norm_eps;
        Self {
            input_layernorm: RmsNormConfig::new(spec.hidden).with_epsilon(eps).init(device),
            self_attn: VisionAttention::new(cfg, spec, device),
            post_attention_layernorm: RmsNormConfig::new(spec.hidden).with_epsilon(eps).init(device),
            pre_feedforward_layernorm: RmsNormConfig::new(spec.hidden)
                .with_epsilon(eps)
                .init(device),
            mlp: VisionMlp::new(spec, device),
            post_feedforward_layernorm: RmsNormConfig::new(spec.hidden)
                .with_epsilon(eps)
                .init(device),
        }
    }

    fn forward(
        &self,
        x: Tensor<2>,
        rope: &AxialRope,
        spec: &VisionSpec,
        chunk: usize,
        debug: Option<(&std::path::Path, usize)>,
    ) -> Tensor<2> {
        let dump = |t: &Tensor<2>, name: &str| {
            if let Some((dir, i)) = debug {
                let values: Vec<f32> = t
                    .clone()
                    .cast(DType::F32)
                    .into_data()
                    .try_to_vec()
                    .expect("f32");
                let mut bytes = Vec::with_capacity(values.len() * 4);
                for v in &values {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                std::fs::write(dir.join(format!("vision_{name}{i}.bin")), &bytes)
                    .expect("write debug tensor");
            }
        };
        let residual = x.clone();
        let h = self.input_layernorm.forward(x);
        dump(&h, "norm");
        let h = self.self_attn.forward(h, rope, spec, chunk);
        dump(&h, "attn");
        let h = self.post_attention_layernorm.forward(h);
        let x = residual + h;

        let residual = x.clone();
        let h = self.pre_feedforward_layernorm.forward(x);
        let h = self.mlp.forward(h);
        dump(&h, "mlp");
        let h = self.post_feedforward_layernorm.forward(h);
        residual + h
    }
}

#[derive(Module, Debug)]
pub struct VisionEncoder {
    layers: Vec<VisionLayer>,
}

#[derive(Module, Debug)]
pub struct VisionTower {
    patch_embedder: PatchEmbedder,
    encoder: VisionEncoder,
    #[module(skip)]
    spec: VisionSpec,
}

impl VisionTower {
    pub fn new(cfg: &VisionConfig, device: &Device) -> Self {
        let spec = VisionSpec::from_config(cfg);
        let layers = (0..spec.layers)
            .map(|_| VisionLayer::new(cfg, &spec, device))
            .collect();
        Self {
            patch_embedder: PatchEmbedder {
                input_proj: linear_cfg(spec.patch_pixels(), spec.hidden).init(device),
                position_embedding_table: Param::from_tensor(Tensor::zeros(
                    [2, spec.position_embedding_size, spec.hidden],
                    device,
                )),
            },
            encoder: VisionEncoder { layers },
            spec,
        }
    }

    pub fn spec(&self) -> &VisionSpec {
        &self.spec
    }

    /// Position embeddings for a prepared image (host-side lookup: the table is
    /// `[2, size, hidden]` and each patch adds its x and y rows).
    fn position_embeddings(&self, img: &PreparedImage, device: &Device) -> Tensor<2> {
        let spec = &self.spec;
        let d = spec.hidden;
        let table: Vec<f32> = self
            .patch_embedder
            .position_embedding_table
            .val()
            .cast(DType::F32)
            .into_data()
            .try_to_vec()
            .expect("position embedding table f32");
        let p = img.num_patches();
        let size = spec.position_embedding_size;
        let mut pos = vec![0.0f32; p * d];
        for i in 0..p {
            let xr = img.xs[i] as usize * d;
            let yr = (size + img.ys[i] as usize) * d;
            for c in 0..d {
                pos[i * d + c] = table[xr + c] + table[yr + c];
            }
        }
        let dtype = crate::layers::weight_dtype(&self.patch_embedder.input_proj.weight);
        Tensor::<2>::from_data(TensorData::new(pos, [p, d]), device).cast(dtype)
    }

    /// Pixel patches `[P, 3 * patch^2]` -> pooled soft tokens `[S, hidden]`
    /// (f32, scaled by `sqrt(hidden)`). With `debug_dir` set, the patch
    /// embeddings and every layer output are dumped as raw f32 for comparison.
    pub fn forward(
        &self,
        img: &PreparedImage,
        chunk: usize,
        debug_dir: Option<&std::path::Path>,
    ) -> Tensor<2> {
        let spec = &self.spec;
        let p = img.num_patches();
        let d = spec.hidden;
        let device = self.patch_embedder.position_embedding_table.val().device();

        let dtype = crate::layers::weight_dtype(&self.patch_embedder.input_proj.weight);
        let pixels = Tensor::<2>::from_data(
            TensorData::new(img.patches.clone(), [p, spec.patch_pixels()]),
            &device,
        )
        .cast(dtype);
        // `2 * (pixel - 0.5)`
        let pixels = pixels.mul_scalar(2.0).sub_scalar(1.0);
        let mut h = self.patch_embedder.input_proj.forward(pixels);
        h = h + self.position_embeddings(img, &device);

        let dump = |t: &Tensor<2>, name: &str| {
            if let Some(dir) = debug_dir {
                let values: Vec<f32> = t
                    .clone()
                    .cast(DType::F32)
                    .into_data()
                    .try_to_vec()
                    .expect("f32");
                let mut bytes = Vec::with_capacity(values.len() * 4);
                for v in &values {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                std::fs::write(dir.join(name), &bytes).expect("write debug tensor");
            }
        };
        dump(&h, "vision_embed.bin");

        let rope = AxialRope::new(img, spec.head_dim, spec.rope_theta, dtype, &device);
        for (i, layer) in self.encoder.layers.iter().enumerate() {
            h = layer.forward(h, &rope, spec, chunk, debug_dir.map(|d| (d, i)));
            dump(&h, &format!("vision_layer{i}.bin"));
        }

        // Spatial average pooling over k x k patch blocks (row-major), then
        // scale by sqrt(hidden) in f32.
        let (gh, gw, k) = (img.soft_h, img.soft_w, spec.pooling);
        let pooled = h
            .reshape([gh, k, gw, k, d])
            .mean_dim(3)
            .mean_dim(1)
            .reshape([gh * gw, d]);
        pooled.mul_scalar((d as f64).sqrt())
    }
}

/// `embed_vision` / `embed_audio`: scale-free RMSNorm then a projection into the
/// text hidden size.
#[derive(Module, Debug)]
pub struct MultimodalEmbedder {
    embedding_projection: Linear,
}

impl MultimodalEmbedder {
    pub fn new(input_dim: usize, text_hidden: usize, device: &Device) -> Self {
        Self {
            embedding_projection: linear_cfg(input_dim, text_hidden).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<2>, eps: f64) -> Tensor<2> {
        lin(&self.embedding_projection, rms_norm_noscale(x, eps))
    }
}
