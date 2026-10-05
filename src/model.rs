//! Qwen3-Embedding-0.6B implemented in Burn (inference only, embedding output).
//!
//! Architecture (from HF `Qwen/Qwen3-Embedding-0.6B/config.json`):
//! decoder-only transformer, pre-norm, GQA (16 q heads / 8 kv heads), head_dim 128,
//! QK-RMSNorm, RoPE theta 1e6, SwiGLU MLP, final RMSNorm, last-token pooling.

use burn::module::Module;
use burn::nn::{
    Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig,
};
use burn::prelude::*;
use burn::tensor::activation::silu;
use burn::tensor::module::attention;
use burn::tensor::ops::AttentionModuleOptions;
use burn::tensor::{DType, Int, TensorData, s};

/// Model hyper-parameters, deserialized from the HF `config.json`.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Qwen3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
}

impl Qwen3Config {
    pub fn from_file(path: &std::path::Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&text)?)
    }
}

/// Precomputed RoPE cos/sin tables, `[max_seq, head_dim / 2]`.
pub struct RopeCache {
    cos: Tensor<2>,
    sin: Tensor<2>,
    half: usize,
}

impl RopeCache {
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
        Self { cos, sin, half }
    }

    /// Applies RoPE to `x` of shape `[batch, seq, heads, head_dim]`, starting at position
    /// `seq_start`. Both q and k use the same table (no partial rotation, no freq scaling).
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

#[derive(Module, Debug)]
pub struct Qwen3Embedding {
    embed_tokens: Embedding,
    layers: Vec<Qwen3Layer>,
    norm: RmsNorm,
    /// True when projection weights are stored Q8-quantized and dequantized per
    /// forward (low-RAM mode, `--quant q8`).
    quantized: bool,
}

impl Qwen3Embedding {
    pub fn new(cfg: &Qwen3Config, device: &Device) -> Self {
        let embed_tokens = EmbeddingConfig::new(cfg.vocab_size, cfg.hidden_size).init(device);
        let layers = (0..cfg.num_hidden_layers)
            .map(|_| Qwen3Layer::new(cfg, device))
            .collect();
        let norm = RmsNormConfig::new(cfg.hidden_size)
            .with_epsilon(cfg.rms_norm_eps)
            .init(device);
        Self {
            embed_tokens,
            layers,
            norm,
            quantized: false,
        }
    }

    /// Mark projection weights as Q8-resident; their forward path then dequantizes
    /// each weight on the fly instead of keeping an f32 copy.
    pub fn set_quantized(&mut self, quantized: bool) {
        self.quantized = quantized;
    }

    pub fn hidden_size(&self) -> usize {
        let [_, d] = self.embed_tokens.weight.shape().dims();
        d
    }

    /// `input_ids`: `[batch, seq]`. Returns last-token pooled embeddings `[batch, hidden]`
    /// (not L2-normalized). With `fused` the backend attention kernel is used; otherwise
    /// queries/keys are processed in `attn_chunk`/`key_block`-sized blocks with a
    /// tensor-op online softmax.
    pub fn forward(
        &self,
        input_ids: Tensor<2, Int>,
        rope: &RopeCache,
        attn_chunk: usize,
        key_block: usize,
        fused: bool,
    ) -> Tensor<2> {
        let [b, s] = input_ids.dims();
        let d = self.hidden_size();
        // The embedding table may be kept in f16 in low-RAM mode while the model body
        // stays f32; the cast is exact (the source safetensors are bf16) and costs one
        // [B, S, D] pass.
        let mut x = self.embed_tokens.forward(input_ids);
        if self.quantized {
            x = x.cast(DType::F32);
        }
        for layer in &self.layers {
            x = layer.forward(x, rope, attn_chunk, key_block, fused, self.quantized);
        }
        let x = self.norm.forward(x);
        // Last-token pooling.
        x.slice(s![.., s - 1..s, ..]).reshape([b, d])
    }
}

/// Linear forward that also supports Q8-resident weights: the weight is dequantized
/// to f32 on the fly and used through the normal float linear path. Only the current
/// layer's weights are materialized, so the resident model stays ~0.6 GB smaller.
fn linear_forward(lin: &Linear, x: Tensor<3>, quantized: bool) -> Tensor<3> {
    if !quantized {
        return lin.forward(x);
    }
    let weight = lin.weight.val().dequantize(); // [in, out] f32
    burn::tensor::module::linear(x, weight, lin.bias.as_ref().map(|b| b.val()))
}

#[derive(Module, Debug)]
struct Qwen3Layer {
    self_attn: Qwen3Attention,
    mlp: Qwen3Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl Qwen3Layer {
    fn new(cfg: &Qwen3Config, device: &Device) -> Self {
        Self {
            self_attn: Qwen3Attention::new(cfg, device),
            mlp: Qwen3Mlp::new(cfg, device),
            input_layernorm: RmsNormConfig::new(cfg.hidden_size)
                .with_epsilon(cfg.rms_norm_eps)
                .init(device),
            post_attention_layernorm: RmsNormConfig::new(cfg.hidden_size)
                .with_epsilon(cfg.rms_norm_eps)
                .init(device),
        }
    }

    fn forward(
        &self,
        x: Tensor<3>,
        rope: &RopeCache,
        attn_chunk: usize,
        key_block: usize,
        fused: bool,
        quantized: bool,
    ) -> Tensor<3> {
        let residual = x.clone();
        let h = self.input_layernorm.forward(x);
        let h = self
            .self_attn
            .forward(h, rope, attn_chunk, key_block, fused, quantized)
            + residual;
        let residual = h.clone();
        let h = self.post_attention_layernorm.forward(h);
        self.mlp.forward(h, quantized) + residual
    }
}

#[derive(Module, Debug)]
struct Qwen3Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

impl Qwen3Attention {
    fn new(cfg: &Qwen3Config, device: &Device) -> Self {
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        Self {
            q_proj: LinearConfig::new(cfg.hidden_size, q_dim)
                .with_bias(false)
                .init(device),
            k_proj: LinearConfig::new(cfg.hidden_size, kv_dim)
                .with_bias(false)
                .init(device),
            v_proj: LinearConfig::new(cfg.hidden_size, kv_dim)
                .with_bias(false)
                .init(device),
            o_proj: LinearConfig::new(q_dim, cfg.hidden_size)
                .with_bias(false)
                .init(device),
            q_norm: RmsNormConfig::new(cfg.head_dim)
                .with_epsilon(cfg.rms_norm_eps)
                .init(device),
            k_norm: RmsNormConfig::new(cfg.head_dim)
                .with_epsilon(cfg.rms_norm_eps)
                .init(device),
            n_heads: cfg.num_attention_heads,
            n_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
        }
    }

    /// Projections + QK-norm + RoPE. Returns q `[B, H, S, D]`, k/v `[B, KV, S, D]`.
    fn project(
        &self,
        x: Tensor<3>,
        rope: &RopeCache,
        quantized: bool,
    ) -> (Tensor<4>, Tensor<4>, Tensor<4>) {
        let [b, s, _] = x.dims();
        let (h, kv, d) = (self.n_heads, self.n_kv_heads, self.head_dim);

        let q = linear_forward(&self.q_proj, x.clone(), quantized).reshape([b, s, h, d]);
        let k = linear_forward(&self.k_proj, x.clone(), quantized).reshape([b, s, kv, d]);
        let v = linear_forward(&self.v_proj, x, quantized).reshape([b, s, kv, d]);

        let q = rope.apply(self.q_norm.forward(q), 0);
        let k = rope.apply(self.k_norm.forward(k), 0);

        (
            q.swap_dims(1, 2), // [B, H, S, D]
            k.swap_dims(1, 2), // [B, KV, S, D]
            v.swap_dims(1, 2),
        )
    }

    /// `fused`: use the backend's fused attention kernel (`burn::tensor::module::attention`;
    /// flex selects a tiled flash-attention path for long sequences). Otherwise use the
    /// tensor-op blocked online-softmax fallback.
    fn forward(
        &self,
        x: Tensor<3>,
        rope: &RopeCache,
        chunk: usize,
        key_block: usize,
        fused: bool,
        quantized: bool,
    ) -> Tensor<3> {
        let [b, s, _] = x.dims();
        let (h, d) = (self.n_heads, self.head_dim);
        let (q, k, v) = self.project(x, rope, quantized);

        let o = if fused {
            // GQA is handled natively (16 query heads, 8 kv heads); causal masking is
            // applied inside the kernel.
            attention(
                q,
                k,
                v,
                None,
                None,
                AttentionModuleOptions {
                    scale: Some(1.0 / (d as f64).sqrt()),
                    softcap: None,
                    is_causal: true,
                },
            )
        } else {
            self.forward_blocked(q, k, v, chunk, key_block)
        };

        let o = o.swap_dims(1, 2).reshape([b, s, h * d]);
        linear_forward(&self.o_proj, o, quantized)
    }

    /// Blocked causal attention with an online softmax, expressed with plain tensor ops.
    fn forward_blocked(
        &self,
        q: Tensor<4>,
        k: Tensor<4>,
        v: Tensor<4>,
        chunk: usize,
        key_block: usize,
    ) -> Tensor<4> {
        let [b, h, s, d] = q.dims();
        let kv = k.dims()[1];
        // GQA: repeat each kv head `n_heads / n_kv_heads` times.
        let k = k
            .unsqueeze_dim::<5>(2)
            .repeat_dim(2, h / kv)
            .reshape([b, h, s, d]);
        let v = v
            .unsqueeze_dim::<5>(2)
            .repeat_dim(2, h / kv)
            .reshape([b, h, s, d]);

        let device = q.device();
        let scale = 1.0 / (d as f64).sqrt();
        let dt = q.dtype();
        let kb = key_block.max(1);

        // Pre-slice keys/values into fixed-size blocks once per layer, so later query
        // blocks reuse them without re-copying growing key prefixes.
        let mut k_blocks: Vec<Tensor<4>> = Vec::new();
        let mut v_blocks: Vec<Tensor<4>> = Vec::new();
        let mut b0 = 0;
        while b0 < s {
            let b1 = (b0 + kb).min(s);
            k_blocks.push(k.clone().slice(s![.., .., b0..b1, ..]));
            v_blocks.push(v.clone().slice(s![.., .., b0..b1, ..]));
            b0 = b1;
        }
        drop(k);
        drop(v);

        // Blocked causal attention with an online softmax (flash-attention style): each
        // query block iterates key blocks up to the diagonal (the only block that needs a
        // causal mask), maintaining running max `m`, normalizer `l` and output `acc`.
        // The result is identical to a full-row softmax; the score matrix never exceeds
        // [B, H, chunk, key_block], which keeps it inside cache for long inputs.
        let mut outs = Vec::new();
        let mut q0 = 0;
        while q0 < s {
            let q1 = (q0 + chunk).min(s);
            let c = q1 - q0;
            let qc = q.clone().slice(s![.., .., q0..q1, ..]); // [B, H, C, D]

            let mut m = Tensor::<4>::full([b, h, c, 1], f32::NEG_INFINITY, (&device, DType::F32));
            let mut l = Tensor::<4>::zeros([b, h, c, 1], (&device, DType::F32));
            let mut acc = Tensor::<4>::zeros([b, h, c, d], (&device, DType::F32));

            let n_blocks = (q1 + kb - 1) / kb;
            for i in 0..n_blocks {
                let k0 = i * kb;
                let k1 = (k0 + kb).min(q1);
                let kc = k_blocks[i].clone().slice(s![.., .., 0..k1 - k0, ..]); // [B, H, K, D]
                let vc = v_blocks[i].clone().slice(s![.., .., 0..k1 - k0, ..]);

                let mut sc = qc.clone().matmul(kc.swap_dims(2, 3)) * scale; // [B, H, C, K]
                if sc.dtype() != DType::F32 {
                    sc = sc.cast(DType::F32);
                }
                if k1 > q0 {
                    // keep where col <= q0 + row
                    let rows = Tensor::arange(0..c as i64, &device).reshape([1, 1, c, 1]);
                    let cols =
                        Tensor::arange(k0 as i64..k1 as i64, &device).reshape([1, 1, 1, k1 - k0]);
                    let keep = cols.lower_equal(rows + q0 as i64);
                    let fill = keep.bool_not().expand([b, h, c, k1 - k0]);
                    sc = sc.mask_fill(fill, f32::NEG_INFINITY);
                }

                let m_new = m.clone().max_pair(sc.clone().max_dim(3)); // [B, H, C, 1]
                let alpha = (m - m_new.clone()).exp(); // accumulator rescale factor
                let p = (sc - m_new.clone()).exp(); // masked entries are exp(-inf) = 0
                l = l * alpha.clone() + p.clone().sum_dim(3);
                let p_dt = if dt == DType::F32 { p } else { p.cast(dt) };
                acc = acc * alpha + p_dt.matmul(vc);
                m = m_new;
            }

            outs.push((acc / l).cast(dt)); // [B, H, C, D]
            q0 = q1;
        }

        Tensor::cat(outs, 2) // [B, H, S, D]
    }
}

#[derive(Module, Debug)]
struct Qwen3Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Qwen3Mlp {
    fn new(cfg: &Qwen3Config, device: &Device) -> Self {
        Self {
            gate_proj: LinearConfig::new(cfg.hidden_size, cfg.intermediate_size)
                .with_bias(false)
                .init(device),
            up_proj: LinearConfig::new(cfg.hidden_size, cfg.intermediate_size)
                .with_bias(false)
                .init(device),
            down_proj: LinearConfig::new(cfg.intermediate_size, cfg.hidden_size)
                .with_bias(false)
                .init(device),
        }
    }

    fn forward(&self, x: Tensor<3>, quantized: bool) -> Tensor<3> {
        let gate = linear_forward(&self.gate_proj, x.clone(), quantized);
        let up = linear_forward(&self.up_proj, x, quantized);
        linear_forward(&self.down_proj, silu(gate) * up, quantized)
    }
}
