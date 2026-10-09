//! Qwen3-Embedding-0.6B implemented in Burn (inference only, embedding output).
//!
//! Architecture (from HF `Qwen/Qwen3-Embedding-0.6B/config.json`):
//! decoder-only transformer, pre-norm, GQA (16 q heads / 8 kv heads), head_dim 128,
//! QK-RMSNorm, RoPE theta 1e6, SwiGLU MLP, final RMSNorm, last-token pooling.

use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig, RmsNorm, RmsNormConfig};
use burn::prelude::*;
use burn::tensor::activation::silu;
use burn::tensor::module::attention;
use burn::tensor::ops::AttentionModuleOptions;
use burn::tensor::{DType, Int, s};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use crate::util::proj::Proj;
#[cfg(all(feature = "npu", target_arch = "aarch64"))]
use crate::util::proj::{FusedGroup, ProjKind};
use crate::util::rope::RopeCache;

// Stage wall-time instrumentation (microseconds): attention (projections + attention
// kernel), MLP (projections + activations) and the norms. Three atomic adds per layer.
static T_ATTN_US: AtomicU64 = AtomicU64::new(0);
static T_MLP_US: AtomicU64 = AtomicU64::new(0);
static T_NORM_US: AtomicU64 = AtomicU64::new(0);

/// `(attention_s, mlp_s, norms_s)` accumulated since the last reset.
pub fn stage_stats() -> (f64, f64, f64) {
    (
        T_ATTN_US.load(Ordering::Relaxed) as f64 / 1e6,
        T_MLP_US.load(Ordering::Relaxed) as f64 / 1e6,
        T_NORM_US.load(Ordering::Relaxed) as f64 / 1e6,
    )
}

/// Reset the stage timers.
pub fn stage_stats_reset() {
    T_ATTN_US.store(0, Ordering::Relaxed);
    T_MLP_US.store(0, Ordering::Relaxed);
    T_NORM_US.store(0, Ordering::Relaxed);
}

/// Model hyper-parameters, deserialized from the HF `config.json`.
#[allow(dead_code)] // full config.json schema; not every field is consumed
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

#[derive(Module, Debug)]
pub struct Qwen3Embedding {
    embed_tokens: Embedding,
    layers: Vec<Qwen3Layer>,
    norm: RmsNorm,
    /// True when projection weights are stored Q8-quantized (low-RAM mode,
    /// `--quant q8`) or packed off the CPU entirely (NPU mode); those forwards
    /// then dequantize per call rather than using a resident copy.
    quantized: bool,
    /// The body dtype quantized/offloaded modes gather into (a 16-bit table is
    /// cast to this after the embedding lookup). Meaningless when `quantized`
    /// is false, where the body follows the loaded tensor dtype.
    #[module(skip)]
    body_dtype: DType,
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
            body_dtype: DType::F32,
        }
    }

    /// Mark projection weights as Q8-resident or NPU-resident; their forward path
    /// then dequantizes each weight on the fly instead of keeping an f32 copy.
    pub fn set_quantized(&mut self, quantized: bool) {
        self.quantized = quantized;
    }

    /// The dtype the model body computes in when `quantized` is set (the table is
    /// cast to it after the embedding lookup).
    pub fn set_body_dtype(&mut self, dtype: DType) {
        self.body_dtype = dtype;
    }

    /// Keep only an f16 copy of the token-embedding table (f32 NPU mode; the
    /// gathered rows are cast back per forward). Native bf16 models keep their
    /// table dtype. Releases the freed pages to the OS.
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    pub fn embed_table_to_f16(&mut self) {
        self.embed_tokens.weight = self
            .embed_tokens
            .weight
            .clone()
            .map(|tensor| tensor.cast(DType::F16));
        self.set_quantized(true);
        #[cfg(all(target_os = "linux", target_env = "gnu"))]
        unsafe {
            libc::malloc_trim(0);
        }
    }

    /// Enable/disable NPU attention on every layer (`--npu` mode).
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    pub fn set_npu_attn(&mut self, on: bool) {
        for layer in &mut self.layers {
            layer.self_attn.npu_attn = on;
        }
    }

    /// Install a fused projection-group weight on one layer (`--npu` mode).
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    pub fn set_fused_handle(&mut self, layer: usize, group: FusedGroup, id: burn_rocket::WeightId) {
        let l = &mut self.layers[layer];
        match group {
            FusedGroup::Qkv => l.self_attn.qkv_fused = Some(id),
            FusedGroup::GateUp => l.mlp.gateup_fused = Some(id),
        }
    }

    /// Mutable access to one projection (used by the loader).
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    pub fn projection_mut(&mut self, layer: usize, kind: ProjKind) -> &mut Proj {
        let l = &mut self.layers[layer];
        match kind {
            ProjKind::Q => &mut l.self_attn.q_proj,
            ProjKind::K => &mut l.self_attn.k_proj,
            ProjKind::V => &mut l.self_attn.v_proj,
            ProjKind::O => &mut l.self_attn.o_proj,
            ProjKind::Gate => &mut l.mlp.gate_proj,
            ProjKind::Up => &mut l.mlp.up_proj,
            ProjKind::Down => &mut l.mlp.down_proj,
        }
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
        // The embedding table may be kept in a 16-bit dtype while the model body
        // computes in another; the cast is exact (the source safetensors are bf16)
        // and costs one [B, S, D] pass.
        let mut x = self.embed_tokens.forward(input_ids);
        if self.quantized {
            x = x.cast(self.body_dtype);
        }
        for layer in &self.layers {
            x = layer.forward(x, rope, attn_chunk, key_block, fused, self.quantized);
        }
        let x = self.norm.forward(x);
        // Last-token pooling.
        x.slice(s![.., s - 1..s, ..]).reshape([b, d])
    }
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
        let t = Instant::now();
        let residual = x.clone();
        let h = self.input_layernorm.forward(x);
        T_NORM_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);

        let t = Instant::now();
        let h = self
            .self_attn
            .forward(h, rope, attn_chunk, key_block, fused, quantized)
            + residual;
        T_ATTN_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);

        let t = Instant::now();
        let residual = h.clone();
        let h = self.post_attention_layernorm.forward(h);
        T_NORM_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);

        let t = Instant::now();
        let out = self.mlp.forward(h, quantized) + residual;
        T_MLP_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
        out
    }
}

#[derive(Module, Debug)]
struct Qwen3Attention {
    #[cfg_attr(all(feature = "npu", target_arch = "aarch64"), module(skip))]
    q_proj: Proj,
    #[cfg_attr(all(feature = "npu", target_arch = "aarch64"), module(skip))]
    k_proj: Proj,
    #[cfg_attr(all(feature = "npu", target_arch = "aarch64"), module(skip))]
    v_proj: Proj,
    #[cfg_attr(all(feature = "npu", target_arch = "aarch64"), module(skip))]
    o_proj: Proj,
    /// Set in `--npu` mode: attention runs on the RK3588 NPU.
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    #[module(skip)]
    npu_attn: bool,
    /// Set in `--npu` mode: q|k|v packed as one resident weight (one matmul).
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    #[module(skip)]
    qkv_fused: Option<burn_rocket::WeightId>,
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
            q_proj: Proj::stub(cfg.hidden_size, q_dim, device),
            k_proj: Proj::stub(cfg.hidden_size, kv_dim, device),
            v_proj: Proj::stub(cfg.hidden_size, kv_dim, device),
            o_proj: Proj::stub(q_dim, cfg.hidden_size, device),
            #[cfg(all(feature = "npu", target_arch = "aarch64"))]
            npu_attn: false,
            #[cfg(all(feature = "npu", target_arch = "aarch64"))]
            qkv_fused: None,
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

    /// Projections + QK-norm + RoPE. Returns q `[B, S, H, D]`, k/v `[B, S, KV, D]`.
    fn project(
        &self,
        x: Tensor<3>,
        rope: &RopeCache,
        quantized: bool,
    ) -> (Tensor<4>, Tensor<4>, Tensor<4>) {
        let [b, s, _] = x.dims();
        let (h, kv, d) = (self.n_heads, self.n_kv_heads, self.head_dim);

        #[cfg(all(feature = "npu", target_arch = "aarch64"))]
        if let Some(id) = &self.qkv_fused {
            // One NPU matmul for q|k|v; the output is split along N (Q, K, V order).
            let qw = h * d;
            let kw = kv * d;
            let vw = kv * d;
            let qkv = burn_rocket::matmul(x, id);
            let q = qkv.clone().slice(s![.., .., 0..qw]).reshape([b, s, h, d]);
            let k = qkv
                .clone()
                .slice(s![.., .., qw..qw + kw])
                .reshape([b, s, kv, d]);
            let v = qkv
                .slice(s![.., .., qw + kw..qw + kw + vw])
                .reshape([b, s, kv, d]);
            let q = rope.apply(self.q_norm.forward(q), 0);
            let k = rope.apply(self.k_norm.forward(k), 0);
            return (q, k, v);
        }

        let q = self
            .q_proj
            .forward(x.clone(), quantized)
            .reshape([b, s, h, d]);
        let k = self
            .k_proj
            .forward(x.clone(), quantized)
            .reshape([b, s, kv, d]);
        let v = self.v_proj.forward(x, quantized).reshape([b, s, kv, d]);

        let q = rope.apply(self.q_norm.forward(q), 0);
        let k = rope.apply(self.k_norm.forward(k), 0);

        (q, k, v)
    }

    /// Attention: `npu_attn` (RK3588 NPU, `--npu`) when set; otherwise `fused` selects
    /// the backend's fused kernel (`burn::tensor::module::attention`; flex uses a tiled
    /// flash-attention path for long sequences) and `blocked` the tensor-op online-softmax
    /// fallback.
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

        #[cfg(all(feature = "npu", target_arch = "aarch64"))]
        if self.npu_attn {
            // NPU attention takes the [1, S, heads*D] layout directly (the
            // head-major f16 buffers and the causal mask are prepared inside).
            let kv = self.n_kv_heads;
            let o = burn_rocket::attention(
                q.reshape([b, s, h * d]),
                k.reshape([b, s, kv * d]),
                v.reshape([b, s, kv * d]),
                h,
                kv,
                d,
                1.0 / (d as f64).sqrt(),
                None,
                true,
            );
            return self.o_proj.forward(o, quantized);
        }

        let (q, k, v) = (q.swap_dims(1, 2), k.swap_dims(1, 2), v.swap_dims(1, 2));
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
        self.o_proj.forward(o, quantized)
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

            let n_blocks = q1.div_ceil(kb);
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
    #[cfg_attr(all(feature = "npu", target_arch = "aarch64"), module(skip))]
    gate_proj: Proj,
    #[cfg_attr(all(feature = "npu", target_arch = "aarch64"), module(skip))]
    up_proj: Proj,
    #[cfg_attr(all(feature = "npu", target_arch = "aarch64"), module(skip))]
    down_proj: Proj,
    /// Set in `--npu` mode: gate|up packed as one resident weight.
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    #[module(skip)]
    gateup_fused: Option<burn_rocket::WeightId>,
    intermediate_size: usize,
}

impl Qwen3Mlp {
    fn new(cfg: &Qwen3Config, device: &Device) -> Self {
        Self {
            gate_proj: Proj::stub(cfg.hidden_size, cfg.intermediate_size, device),
            up_proj: Proj::stub(cfg.hidden_size, cfg.intermediate_size, device),
            down_proj: Proj::stub(cfg.intermediate_size, cfg.hidden_size, device),
            #[cfg(all(feature = "npu", target_arch = "aarch64"))]
            gateup_fused: None,
            intermediate_size: cfg.intermediate_size,
        }
    }

    fn forward(&self, x: Tensor<3>, quantized: bool) -> Tensor<3> {
        #[cfg(all(feature = "npu", target_arch = "aarch64"))]
        if let Some(id) = &self.gateup_fused {
            // One NPU matmul for gate|up; the output is split in half along N.
            let n = self.intermediate_size;
            let gu = burn_rocket::matmul(x, id);
            let gate = gu.clone().slice(s![.., .., 0..n]);
            let up = gu.slice(s![.., .., n..2 * n]);
            return self.down_proj.forward(silu(gate) * up, quantized);
        }
        let gate = self.gate_proj.forward(x.clone(), quantized);
        let up = self.up_proj.forward(x, quantized);
        self.down_proj.forward(silu(gate) * up, quantized)
    }
}
