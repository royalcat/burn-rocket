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
use burn::tensor::module::attention;
use burn::tensor::ops::AttentionModuleOptions;
use burn::tensor::s;

use std::collections::HashMap;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use crate::gemma::config::VisionConfig;
use crate::gemma::layers::ClipBounds;
use crate::gemma::layers::ClippableLinear;
use crate::gemma::layers::gelu_mul;
use crate::gemma::layers::lin;
use crate::gemma::media::PreparedImage;
use crate::gemma::{chunked_attention, linear_cfg, repeat_kv, rms_norm_noscale};
use crate::server::telemetry;

/// How the tower computes its bidirectional self-attention.
///
/// * [`VisionAttn::Npu`] — `burn_rocket::attention_window` (no mask, whole patch
///   sequence) on the RK3588. Since the S1 fast host softmax (log-gemma §19) it
///   is 1.99× the fused CPU kernel at the image size and 1.07× at the video-frame
///   size, so it is the default for `--npu` on an aarch64 `npu` build (S2).
/// * [`VisionAttn::Fused`] — the backend's flash-attention kernel
///   (`burn::tensor::module::attention`, `is_causal: false`): tiled online
///   softmax, no full score matrix, GQA handled inside the kernel. The CPU
///   default since log §18.3 (a strict win in every measured shape).
/// * [`VisionAttn::Blocked`] — the tensor-op path ([`chunked_attention`]): the
///   full `[heads, chunk, patches]` score block is materialized per query chunk.
///   The parity fallback; Gemma 4 keeps it (its tower was not re-measured).
///
/// All three compute the same function; the scale is exactly 1.0 because the
/// tower normalizes q and k (QK-RMSNorm replaces the usual `1/sqrt(head_dim)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VisionAttn {
    Npu,
    #[default]
    Fused,
    Blocked,
}

impl VisionAttn {
    pub fn name(self) -> &'static str {
        match self {
            Self::Npu => "npu",
            Self::Fused => "fused",
            Self::Blocked => "blocked",
        }
    }

    /// Parse `npu|fused|blocked` (the old `--vision-attn` flag value; the CLI
    /// now selects the device and the CPU kernel separately).
    #[allow(dead_code)]
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "npu" => Some(Self::Npu),
            "fused" => Some(Self::Fused),
            "blocked" => Some(Self::Blocked),
            _ => None,
        }
    }

    fn code(self) -> u64 {
        match self {
            Self::Blocked => 0,
            Self::Fused => 1,
            Self::Npu => 2,
        }
    }

    fn from_code(code: u64) -> Self {
        match code {
            0 => Self::Blocked,
            2 => Self::Npu,
            _ => Self::Fused,
        }
    }
}

// ---------------------------------------------------------------------------
// Tower timers (microseconds). One forward is split into patch embedding
// (patches + position embeddings), attention (input norm, q/k/v, RoPE,
// attention, output projection, post-attention norm/residual) and MLP (both
// feed-forward norms, gate/up/down, residual), plus pooling. Timing never
// changes the math; the counters are process-global like the text stages.
// ---------------------------------------------------------------------------

static V_FORWARDS: AtomicU64 = AtomicU64::new(0);
static V_PATCHES: AtomicU64 = AtomicU64::new(0);
static V_SOFT_TOKENS: AtomicU64 = AtomicU64::new(0);
static V_TOTAL_US: AtomicU64 = AtomicU64::new(0);
static V_PATCH_US: AtomicU64 = AtomicU64::new(0);
static V_ATTN_US: AtomicU64 = AtomicU64::new(0);
/// The attention *op* inside `V_ATTN_US` (the fused/blocked call only; the
/// projection/RoPE/norm work of the attention phase is the difference).
static V_KERNEL_US: AtomicU64 = AtomicU64::new(0);
static V_MLP_US: AtomicU64 = AtomicU64::new(0);
static V_POOL_US: AtomicU64 = AtomicU64::new(0);
/// Attention mode of the last recorded forward ([`VisionAttn::code`]).
static V_ATTN_MODE: AtomicU64 = AtomicU64::new(0);

/// Accumulated vision-tower work since the last [`vision_stats_reset`].
#[derive(Debug, Clone, Copy, Default)]
pub struct VisionStats {
    pub forwards: u64,
    pub patches: u64,
    pub soft_tokens: u64,
    pub total_s: f64,
    pub patch_s: f64,
    pub attn_s: f64,
    /// The attention op alone (subset of `attn_s`).
    pub kernel_s: f64,
    pub mlp_s: f64,
    pub pool_s: f64,
    /// Attention mode of the recorded forwards (see [`VisionAttn`]).
    pub attn_mode: VisionAttn,
}

impl VisionStats {
    pub fn is_empty(&self) -> bool {
        self.forwards == 0
    }

    /// `vision: 2394 patches -> 266 soft tokens, vision-attn fused, tower 5.41s
    /// (patch 0.12, attn 1.46, kernel 0.61, mlp 3.71, pool 0.02)`; several
    /// forwards (e.g. video frames) are aggregated and prefixed with the count.
    pub fn line(&self) -> String {
        let head = if self.forwards > 1 {
            format!("vision: {} forwards, ", self.forwards)
        } else {
            "vision: ".to_string()
        };
        format!(
            "{head}{} patches -> {} soft tokens, vision-attn {}, tower {:.2}s (patch {:.2}, attn {:.2}, kernel {:.2}, mlp {:.2}, pool {:.2})",
            self.patches,
            self.soft_tokens,
            self.attn_mode.name(),
            self.total_s,
            self.patch_s,
            self.attn_s,
            self.kernel_s,
            self.mlp_s,
            self.pool_s
        )
    }
}

/// Tower work accumulated since the last reset.
pub fn vision_stats() -> VisionStats {
    let us = |v: &AtomicU64| v.load(Ordering::Relaxed) as f64 / 1e6;
    VisionStats {
        forwards: V_FORWARDS.load(Ordering::Relaxed),
        patches: V_PATCHES.load(Ordering::Relaxed),
        soft_tokens: V_SOFT_TOKENS.load(Ordering::Relaxed),
        total_s: us(&V_TOTAL_US),
        patch_s: us(&V_PATCH_US),
        attn_s: us(&V_ATTN_US),
        kernel_s: us(&V_KERNEL_US),
        mlp_s: us(&V_MLP_US),
        pool_s: us(&V_POOL_US),
        attn_mode: VisionAttn::from_code(V_ATTN_MODE.load(Ordering::Relaxed)),
    }
}

pub fn vision_stats_reset() {
    for counter in [
        &V_FORWARDS,
        &V_PATCHES,
        &V_SOFT_TOKENS,
        &V_TOTAL_US,
        &V_PATCH_US,
        &V_ATTN_US,
        &V_KERNEL_US,
        &V_MLP_US,
        &V_POOL_US,
    ] {
        counter.store(0, Ordering::Relaxed);
    }
}

/// `ROCKET_VISION_LOG=1`: print one line per tower forward (stderr).
fn vision_log() -> bool {
    std::env::var_os("ROCKET_VISION_LOG").is_some_and(|v| v == "1")
}

fn add_us(counter: &AtomicU64, elapsed: std::time::Duration) {
    counter.fetch_add(elapsed.as_micros() as u64, Ordering::Relaxed);
}

/// Per-forward numbers, recorded into the global counters and (with
/// `ROCKET_VISION_LOG=1`) printed as one line.
struct ForwardTiming {
    patches: usize,
    soft_tokens: usize,
    attn_mode: VisionAttn,
    total: std::time::Duration,
    patch: std::time::Duration,
    attn: std::time::Duration,
    kernel: std::time::Duration,
    mlp: std::time::Duration,
    pool: std::time::Duration,
    /// NPU work of the transformer layers: `(calls, convert_s, npu_s)`. Zero in
    /// the CPU modes; only printed under `ROCKET_VISION_LOG=1` (the `embed_vision`
    /// projection runs outside it).
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    npu: (u64, f64, f64),
}

impl ForwardTiming {
    fn record(&self) {
        // The per-forward numbers become the `vision` span's attributes; with
        // tracing off the current span is the `none` span and this is a no-op.
        let span = tracing::Span::current();
        span.record("patches", self.patches);
        span.record("soft_tokens", self.soft_tokens);
        span.record("patch_s", self.patch.as_secs_f64());
        span.record("attn_s", self.attn.as_secs_f64());
        span.record("kernel_s", self.kernel.as_secs_f64());
        span.record("mlp_s", self.mlp.as_secs_f64());
        span.record("pool_s", self.pool.as_secs_f64());
        #[cfg(all(feature = "npu", target_arch = "aarch64"))]
        if self.npu.0 > 0 {
            span.record("npu_s", self.npu.2);
            span.record("convert_s", self.npu.1);
        }
        V_FORWARDS.fetch_add(1, Ordering::Relaxed);
        V_PATCHES.fetch_add(self.patches as u64, Ordering::Relaxed);
        V_SOFT_TOKENS.fetch_add(self.soft_tokens as u64, Ordering::Relaxed);
        V_ATTN_MODE.store(self.attn_mode.code(), Ordering::Relaxed);
        add_us(&V_TOTAL_US, self.total);
        add_us(&V_PATCH_US, self.patch);
        add_us(&V_ATTN_US, self.attn);
        add_us(&V_KERNEL_US, self.kernel);
        add_us(&V_MLP_US, self.mlp);
        add_us(&V_POOL_US, self.pool);
        if vision_log() {
            let one = VisionStats {
                forwards: 1,
                patches: self.patches as u64,
                soft_tokens: self.soft_tokens as u64,
                total_s: self.total.as_secs_f64(),
                patch_s: self.patch.as_secs_f64(),
                attn_s: self.attn.as_secs_f64(),
                kernel_s: self.kernel.as_secs_f64(),
                mlp_s: self.mlp.as_secs_f64(),
                pool_s: self.pool.as_secs_f64(),
                attn_mode: self.attn_mode,
            };
            #[allow(unused_mut)]
            let mut line = one.line();
            #[cfg(all(feature = "npu", target_arch = "aarch64"))]
            if self.npu.0 > 0 {
                line.push_str(&format!(
                    ", npu {} calls convert {:.2}s npu {:.2}s",
                    self.npu.0, self.npu.1, self.npu.2
                ));
            }
            eprintln!("{line}");
        }
    }
}

/// Attention/MLP split of one layer (both include their norms and residuals).
#[derive(Default, Clone, Copy)]
struct LayerTiming {
    attn: std::time::Duration,
    /// The bare attention op (subset of `attn`).
    kernel: std::time::Duration,
    mlp: std::time::Duration,
}

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
        let t2 =
            |v: Vec<f32>| Tensor::<2>::from_data(TensorData::new(v, [p, n]), device).cast(dtype);
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
    /// `--npu`: q|k|v packed as one resident weight (`pack3`, one matmul then a
    /// split along N), or `None` for the per-projection path (clipped linears,
    /// `ROCKET_NPU_FUSED=0`, non-NPU builds).
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    #[module(skip)]
    qkv_fused: Option<burn_rocket::WeightId>,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
}

impl VisionAttention {
    fn new(
        cfg: &VisionConfig,
        spec: &VisionSpec,
        bounds: &HashMap<String, ClipBounds>,
        path: &str,
        device: &Device,
    ) -> Self {
        let (h, kv, d) = (spec.heads, spec.kv_heads, spec.head_dim);
        // E2B ships clipped vision linears (`use_clipped_linears`); EmbeddingGemma 2
        // has none, so missing bounds are only an error when the config asks for them.
        let clip = |name: &str| -> Option<ClipBounds> {
            match bounds.get(&format!("{path}.{name}")) {
                Some(c) => Some(*c),
                None if cfg.use_clipped_linears => {
                    panic!("missing vision clip bounds for {path}.{name}")
                }
                None => None,
            }
        };
        Self {
            q_proj: ClippableLinear::with_clip(spec.hidden, h * d, clip("q_proj"), device),
            k_proj: ClippableLinear::with_clip(spec.hidden, kv * d, clip("k_proj"), device),
            v_proj: ClippableLinear::with_clip(spec.hidden, kv * d, clip("v_proj"), device),
            o_proj: ClippableLinear::with_clip(h * d, spec.hidden, clip("o_proj"), device),
            #[cfg(all(feature = "npu", target_arch = "aarch64"))]
            qkv_fused: None,
            q_norm: RmsNormConfig::new(d)
                .with_epsilon(cfg.rms_norm_eps)
                .init(device),
            k_norm: RmsNormConfig::new(d)
                .with_epsilon(cfg.rms_norm_eps)
                .init(device),
        }
    }

    /// q/k/v for every patch: `[patches, heads, head_dim]` and
    /// `[patches, kv_heads, head_dim]` twice.
    ///
    /// With a fused group handle (`--npu`) this is one NPU matmul over the
    /// concatenated q|k|v columns split along N — rows are independent, so the
    /// result is the per-projection path's, in the same column order.
    fn project(&self, x: Tensor<2>, spec: &VisionSpec) -> (Tensor<3>, Tensor<3>, Tensor<3>) {
        let p = x.dims()[0];
        let (h, kv, d) = (spec.heads, spec.kv_heads, spec.head_dim);
        #[cfg(all(feature = "npu", target_arch = "aarch64"))]
        if let Some(id) = &self.qkv_fused {
            let (qw, kw) = (h * d, kv * d);
            let qkv = burn_rocket::matmul(x, id);
            let q = qkv.clone().slice(s![.., 0..qw]).reshape([p, h, d]);
            let k = qkv.clone().slice(s![.., qw..qw + kw]).reshape([p, kv, d]);
            let v = qkv.slice(s![.., qw + kw..qw + 2 * kw]).reshape([p, kv, d]);
            return (q, k, v);
        }
        (
            self.q_proj.forward(x.clone()).reshape([p, h, d]),
            self.k_proj.forward(x.clone()).reshape([p, kv, d]),
            self.v_proj.forward(x).reshape([p, kv, d]),
        )
    }

    fn forward(
        &self,
        x: Tensor<2>,
        rope: &AxialRope,
        spec: &VisionSpec,
        attn: VisionAttn,
        chunk: usize,
    ) -> (Tensor<2>, std::time::Duration) {
        let p = x.dims()[0];
        let (h, kv, d) = (spec.heads, spec.kv_heads, spec.head_dim);
        let (q, k, v) = self.project(x, spec);

        let q = rope.apply(self.q_norm.forward(q));
        let k = rope.apply(self.k_norm.forward(k));
        let v = rms_norm_noscale(v, spec.eps);

        let t0 = Instant::now();
        // Every arm returns the output in the `[patches, heads * head_dim]`
        // layout the packed `o_proj` expects.
        let o: Tensor<2> = match attn {
            VisionAttn::Npu => npu_attention(q, k, v, spec),
            // The blocked path keeps every head's own K/V copy (its per-chunk
            // score tensor is `[heads, ..]`); the fused kernel maps each query
            // head to its shared K/V head, so k/v stay unrepeated there.
            VisionAttn::Blocked => {
                let q = q.swap_dims(0, 1).unsqueeze_dim::<4>(0);
                let k = repeat_kv(k.swap_dims(0, 1).unsqueeze_dim::<4>(0), h / kv);
                let v = repeat_kv(v.swap_dims(0, 1).unsqueeze_dim::<4>(0), h / kv);
                chunked_attention(q, k, v, None, chunk)
                    .reshape([h, p, d])
                    .swap_dims(0, 1)
                    .reshape([p, h * d])
            }
            VisionAttn::Fused => {
                let q = q.swap_dims(0, 1).unsqueeze_dim::<4>(0);
                let k = k.swap_dims(0, 1).unsqueeze_dim::<4>(0);
                let v = v.swap_dims(0, 1).unsqueeze_dim::<4>(0);
                fused_attention(q, k, v)
                    .reshape([h, p, d])
                    .swap_dims(0, 1)
                    .reshape([p, h * d])
            }
        };
        let kernel = t0.elapsed();
        (self.o_proj.forward(o), kernel)
    }

    /// Pack this layer's attention projections into resident NPU memory and
    /// drop the CPU copies (called by [`VisionTower::pack_npu`]).
    ///
    /// `fused` packs q|k|v as one segmented weight (`pack3`, one matmul) when
    /// none of the three carries clip bounds; otherwise each projection packs
    /// alone. Returns `(weights, resident bytes)`.
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    fn pack_npu(&mut self, device: &Device, fused: bool) -> (usize, usize) {
        let mut weights = 0;
        let mut bytes = 0;
        if fused {
            let parts = [
                self.q_proj.group_weight_nk(device),
                self.k_proj.group_weight_nk(device),
                self.v_proj.group_weight_nk(device),
            ];
            if let [Some(q), Some(k), Some(v)] = parts {
                let [nq, kk] = q.shape().dims::<2>();
                let nk = k.shape().dims::<2>()[0];
                let nv = v.shape().dims::<2>()[0];
                assert_eq!(kk, k.shape().dims::<2>()[1], "q|k|v must share K");
                assert_eq!(kk, v.shape().dims::<2>()[1], "q|k|v must share K");
                bytes += (nq + nk + nv) * kk * 2;
                self.qkv_fused = Some(burn_rocket::pack3(q, k, v));
                weights += 1;
                self.q_proj.drop_weight(device);
                self.k_proj.drop_weight(device);
                self.v_proj.drop_weight(device);
            }
        }
        if self.qkv_fused.is_none() {
            for lin in [&mut self.q_proj, &mut self.k_proj, &mut self.v_proj] {
                if let Some((w, b)) = lin.pack_npu(device) {
                    weights += w;
                    bytes += b;
                }
            }
        }
        if let Some((w, b)) = self.o_proj.pack_npu(device) {
            weights += w;
            bytes += b;
        }
        (weights, bytes)
    }
}

/// NPU attention (`--vision-attn-device npu`): bidirectional, no mask, whole patch
/// sequence through `burn_rocket::attention_window` (the S1 fast host softmax,
/// log-gemma §19).
///
/// `q`/`k`/`v` arrive as `[patches, heads, head_dim]` / `[patches, kv_heads,
/// head_dim]`; the op takes the batch-1 token-major `[1, patches, H*D]` layout
/// and returns `[1, patches, H*D]`, which is exactly what the packed `o_proj`
/// consumes. The scale is 1.0 (QK-RMSNorm) and `window = -1` means no mask. Not
/// chunked: since S1 the full sequence is the fastest arm at both the image and
/// the video-frame size (§19.3).
#[cfg(all(feature = "npu", target_arch = "aarch64"))]
fn npu_attention(q: Tensor<3>, k: Tensor<3>, v: Tensor<3>, spec: &VisionSpec) -> Tensor<2> {
    let p = q.dims()[0];
    let (h, kv, d) = (spec.heads, spec.kv_heads, spec.head_dim);
    let o = burn_rocket::attention_window(
        q.reshape([1, p, h * d]),
        k.reshape([1, p, kv * d]),
        v.reshape([1, p, kv * d]),
        h,
        kv,
        d,
        1.0,
        None,
        -1,
    );
    o.reshape([p, h * d])
}

/// Off-device `--vision-attn-device npu` is rejected before a forward
/// runs (`requires an aarch64 build with --features npu and --npu`).
#[cfg(not(all(feature = "npu", target_arch = "aarch64")))]
fn npu_attention(_q: Tensor<3>, _k: Tensor<3>, _v: Tensor<3>, _spec: &VisionSpec) -> Tensor<2> {
    unreachable!("--vision-attn-device npu requires an aarch64 build with --features npu")
}

/// Bidirectional non-causal attention through the backend's fused kernel
/// (`q`, `k`, `v` are `[1, heads, patches, head_dim]`, k/v not GQA-repeated).
///
/// The tower's softmax runs in f32 on both paths: the blocked path casts the
/// score block before `softmax`, so the fused path feeds the kernel f32 operands
/// and casts the result back to the activation dtype. In an f32 model the casts
/// are no-ops.
///
/// The call covers the whole patch sequence: chunking the queries over the
/// `--attn-chunk` value was measured and is slower (dev f32 tower kernel 0.73 s
/// whole vs 0.83 s at chunk 1024 and 1.01 s at 256), because flex already runs
/// the flash path per query row and the per-call scheduling dominates.
fn fused_attention(q: Tensor<4>, k: Tensor<4>, v: Tensor<4>) -> Tensor<4> {
    let dt = q.dtype();
    let (q, k, v) = if dt == DType::F32 {
        (q, k, v)
    } else {
        (q.cast(DType::F32), k.cast(DType::F32), v.cast(DType::F32))
    };
    let o = attention(
        q,
        k,
        v,
        None,
        None,
        AttentionModuleOptions {
            // QK-RMSNorm replaces the usual 1/sqrt(head_dim).
            scale: Some(1.0),
            softcap: None,
            is_causal: false,
        },
    );
    if dt == DType::F32 { o } else { o.cast(dt) }
}

#[derive(Module, Debug)]
pub struct VisionMlp {
    gate_proj: ClippableLinear,
    up_proj: ClippableLinear,
    down_proj: ClippableLinear,
    /// `--npu`: gate|up packed as one resident weight (`pack2`, one matmul then a
    /// split along N), or `None` for the per-projection path.
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    #[module(skip)]
    gateup_fused: Option<burn_rocket::WeightId>,
    /// `gate`/`up` output width (the fused output's column half).
    inter: usize,
}

impl VisionMlp {
    fn new(
        cfg: &VisionConfig,
        spec: &VisionSpec,
        bounds: &HashMap<String, ClipBounds>,
        path: &str,
        device: &Device,
    ) -> Self {
        let clip = |name: &str| -> Option<ClipBounds> {
            match bounds.get(&format!("{path}.{name}")) {
                Some(c) => Some(*c),
                None if cfg.use_clipped_linears => {
                    panic!("missing vision clip bounds for {path}.{name}")
                }
                None => None,
            }
        };
        Self {
            gate_proj: ClippableLinear::with_clip(
                spec.hidden,
                spec.inter,
                clip("gate_proj"),
                device,
            ),
            up_proj: ClippableLinear::with_clip(spec.hidden, spec.inter, clip("up_proj"), device),
            down_proj: ClippableLinear::with_clip(
                spec.inter,
                spec.hidden,
                clip("down_proj"),
                device,
            ),
            #[cfg(all(feature = "npu", target_arch = "aarch64"))]
            gateup_fused: None,
            inter: spec.inter,
        }
    }

    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        #[cfg(all(feature = "npu", target_arch = "aarch64"))]
        if let Some(id) = &self.gateup_fused {
            // One matmul for gate|up (concatenated along N), split back apart.
            let gateup = burn_rocket::matmul(x, id);
            let gate = gateup.clone().slice(s![.., 0..self.inter]);
            let up = gateup.slice(s![.., self.inter..2 * self.inter]);
            return self.down_proj.forward(gelu_mul(gate, up));
        }
        let gate = self.gate_proj.forward(x.clone());
        let up = self.up_proj.forward(x);
        self.down_proj.forward(gelu_mul(gate, up))
    }

    /// Pack this layer's MLP projections into resident NPU memory and drop the
    /// CPU copies (called by [`VisionTower::pack_npu`]).
    ///
    /// `fused` packs gate|up as one segmented weight (`pack2`, one matmul) when
    /// neither carries clip bounds; otherwise each projection packs alone.
    /// Returns `(weights, resident bytes)`.
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    fn pack_npu(&mut self, device: &Device, fused: bool) -> (usize, usize) {
        let mut weights = 0;
        let mut bytes = 0;
        if fused {
            let parts = [
                self.gate_proj.group_weight_nk(device),
                self.up_proj.group_weight_nk(device),
            ];
            if let [Some(gate), Some(up)] = parts {
                let [ng, kg] = gate.shape().dims::<2>();
                let [nu, ku] = up.shape().dims::<2>();
                assert_eq!((ng, kg), (nu, ku), "gate|up must share [N, K]");
                bytes += 2 * ng * kg * 2;
                self.gateup_fused = Some(burn_rocket::pack2(gate, up));
                weights += 1;
                self.gate_proj.drop_weight(device);
                self.up_proj.drop_weight(device);
            }
        }
        if self.gateup_fused.is_none() {
            for lin in [&mut self.gate_proj, &mut self.up_proj] {
                if let Some((w, b)) = lin.pack_npu(device) {
                    weights += w;
                    bytes += b;
                }
            }
        }
        if let Some((w, b)) = self.down_proj.pack_npu(device) {
            weights += w;
            bytes += b;
        }
        (weights, bytes)
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
    fn new(
        cfg: &VisionConfig,
        spec: &VisionSpec,
        bounds: &HashMap<String, ClipBounds>,
        index: usize,
        device: &Device,
    ) -> Self {
        let eps = cfg.rms_norm_eps;
        let base = format!("layers.{index}");
        Self {
            input_layernorm: RmsNormConfig::new(spec.hidden)
                .with_epsilon(eps)
                .init(device),
            self_attn: VisionAttention::new(
                cfg,
                spec,
                bounds,
                &format!("{base}.self_attn"),
                device,
            ),
            post_attention_layernorm: RmsNormConfig::new(spec.hidden)
                .with_epsilon(eps)
                .init(device),
            pre_feedforward_layernorm: RmsNormConfig::new(spec.hidden)
                .with_epsilon(eps)
                .init(device),
            mlp: VisionMlp::new(cfg, spec, bounds, &format!("{base}.mlp"), device),
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
        attn: VisionAttn,
        chunk: usize,
        debug: Option<(&std::path::Path, usize)>,
    ) -> (Tensor<2>, LayerTiming) {
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
        let t0 = Instant::now();
        let residual = x.clone();
        let h = self.input_layernorm.forward(x);
        dump(&h, "norm");
        let (h, kernel) = self.self_attn.forward(h, rope, spec, attn, chunk);
        dump(&h, "attn");
        let h = self.post_attention_layernorm.forward(h);
        let x = residual + h;
        let attn = t0.elapsed();

        let t1 = Instant::now();
        let residual = x.clone();
        let h = self.pre_feedforward_layernorm.forward(x);
        let h = self.mlp.forward(h);
        dump(&h, "mlp");
        let h = self.post_feedforward_layernorm.forward(h);
        let x = residual + h;
        let mlp = t1.elapsed();
        (x, LayerTiming { attn, kernel, mlp })
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
    /// Host-side f32 copy of `position_embedding_table`, built on the first
    /// forward and reused for every later image: the table is `[2, 10240,
    /// hidden]` (63 MiB for EmbeddingGemma 2) and re-reading it per image was a
    /// full f32 copy of the parameter.
    #[module(skip)]
    position_table_f32: OnceLock<Vec<f32>>,
}

impl VisionTower {
    pub fn new(cfg: &VisionConfig, device: &Device) -> Self {
        Self::new_with_clip(cfg, &HashMap::new(), device)
    }

    /// `bounds` are keyed like `layers.{i}.self_attn.q_proj` (the checkpoint's
    /// `Gemma4ClippableLinear` scalars); E2B has them, EmbeddingGemma 2 does not.
    pub fn new_with_clip(
        cfg: &VisionConfig,
        bounds: &HashMap<String, ClipBounds>,
        device: &Device,
    ) -> Self {
        let spec = VisionSpec::from_config(cfg);
        let layers = (0..spec.layers)
            .map(|i| VisionLayer::new(cfg, &spec, bounds, i, device))
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
            position_table_f32: OnceLock::new(),
        }
    }

    pub fn spec(&self) -> &VisionSpec {
        &self.spec
    }

    /// Pack every layer's projection into resident NPU memory and drop the CPU
    /// copies. Returns `(weights, resident bytes)`.
    ///
    /// With `ROCKET_NPU_FUSED` unset (the default) q|k|v pack as one `pack3`
    /// weight and gate|up as one `pack2` weight, so a layer needs 4 matmuls
    /// instead of 7 and shares each group's input conversion; `ROCKET_NPU_FUSED=0`
    /// restores seven per-projection packs (the A/B switch and the fallback for
    /// clipped linears). Clipped linears are skipped by [`ClippableLinear::pack_npu`]
    /// and never join a group, so Gemma 4 E2B's tower stays on the per-projection
    /// path (and is not wired here at all). The patch embedder and the norms stay
    /// on the CPU.
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    pub fn pack_npu(&mut self, device: &Device) -> (usize, usize) {
        let fused = !std::env::var("ROCKET_NPU_FUSED").is_ok_and(|v| v == "0");
        let mut weights = 0;
        let mut bytes = 0;
        for layer in &mut self.encoder.layers {
            let (w, b) = layer.self_attn.pack_npu(device, fused);
            weights += w;
            bytes += b;
            let (w, b) = layer.mlp.pack_npu(device, fused);
            weights += w;
            bytes += b;
        }
        (weights, bytes)
    }

    /// Position embeddings for a prepared image (host-side lookup: the table is
    /// `[2, size, hidden]` and each patch adds its x and y rows).
    fn position_embeddings(&self, img: &PreparedImage, device: &Device) -> Tensor<2> {
        let spec = &self.spec;
        let d = spec.hidden;
        let table = self.position_table_f32.get_or_init(|| {
            self.patch_embedder
                .position_embedding_table
                .val()
                .cast(DType::F32)
                .into_data()
                .try_to_vec()
                .expect("position embedding table f32")
        });
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
        let dtype = crate::gemma::layers::weight_dtype(&self.patch_embedder.input_proj.weight);
        Tensor::<2>::from_data(TensorData::new(pos, [p, d]), device).cast(dtype)
    }

    /// Pixel patches `[P, 3 * patch^2]` -> pooled soft tokens `[S, hidden]`
    /// (f32, scaled by `sqrt(hidden)`). With `debug_dir` set, the patch
    /// embeddings and every layer output are dumped as raw f32 for comparison.
    #[telemetry::instrument(
        name = "vision",
        skip_all,
        fields(
            patches = tracing::field::Empty,
            soft_tokens = tracing::field::Empty,
            patch_s = tracing::field::Empty,
            attn_s = tracing::field::Empty,
            kernel_s = tracing::field::Empty,
            mlp_s = tracing::field::Empty,
            pool_s = tracing::field::Empty,
            npu_s = tracing::field::Empty,
            convert_s = tracing::field::Empty,
        )
    )]
    pub fn forward(
        &self,
        img: &PreparedImage,
        chunk: usize,
        attn: VisionAttn,
        debug_dir: Option<&std::path::Path>,
    ) -> Tensor<2> {
        let spec = &self.spec;
        let p = img.num_patches();
        let d = spec.hidden;
        let device = self.patch_embedder.position_embedding_table.val().device();
        let forward_start = Instant::now();

        let dtype = crate::gemma::layers::weight_dtype(&self.patch_embedder.input_proj.weight);
        let pixels = Tensor::<2>::from_data(
            TensorData::new(img.patches.clone(), [p, spec.patch_pixels()]),
            &device,
        )
        .cast(dtype);
        // `2 * (pixel - 0.5)`
        let pixels = pixels.mul_scalar(2.0).sub_scalar(1.0);
        let mut h = self.patch_embedder.input_proj.forward(pixels);
        h = h + self.position_embeddings(img, &device);
        let patch = forward_start.elapsed();

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
        #[cfg(all(feature = "npu", target_arch = "aarch64"))]
        let npu_before = burn_rocket::stats();
        let (mut attn_s, mut kernel_s, mut mlp) = (
            std::time::Duration::ZERO,
            std::time::Duration::ZERO,
            std::time::Duration::ZERO,
        );
        for (i, layer) in self.encoder.layers.iter().enumerate() {
            let (out, timing) =
                layer.forward(h, &rope, spec, attn, chunk, debug_dir.map(|d| (d, i)));
            h = out;
            attn_s += timing.attn;
            kernel_s += timing.kernel;
            mlp += timing.mlp;
            dump(&h, &format!("vision_layer{i}.bin"));
        }
        #[cfg(all(feature = "npu", target_arch = "aarch64"))]
        let npu = {
            let after = burn_rocket::stats();
            (
                after.calls - npu_before.calls,
                after.convert_s - npu_before.convert_s,
                after.npu_s - npu_before.npu_s,
            )
        };

        // Spatial average pooling over k x k patch blocks (row-major), then
        // scale by sqrt(hidden) in f32.
        let pool_start = Instant::now();
        let (gh, gw, k) = (img.soft_h, img.soft_w, spec.pooling);
        let pooled = h
            .reshape([gh, k, gw, k, d])
            .mean_dim(3)
            .mean_dim(1)
            .reshape([gh * gw, d]);
        let pooled = pooled.mul_scalar((d as f64).sqrt());
        let pool = pool_start.elapsed();

        ForwardTiming {
            patches: p,
            soft_tokens: gh * gw,
            attn_mode: attn,
            total: forward_start.elapsed(),
            patch,
            attn: attn_s,
            kernel: kernel_s,
            mlp,
            pool,
            #[cfg(all(feature = "npu", target_arch = "aarch64"))]
            npu,
        }
        .record();
        pooled
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

    /// Pack `embedding_projection` into resident NPU memory (pack-and-drop,
    /// NPU build only). Returns `(1, resident bytes)`.
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    pub fn pack_npu(&mut self, device: &Device) -> (usize, usize) {
        let (n, k) = crate::gemma::layers::pack_linear_into_npu(
            &mut self.embedding_projection,
            device,
            false,
        );
        (1, n * k * 2)
    }
}

#[cfg(test)]
mod tests {
    use super::VisionAttn;

    #[test]
    fn attn_mode_parses_and_roundtrips() {
        assert_eq!(VisionAttn::parse("npu"), Some(VisionAttn::Npu));
        assert_eq!(VisionAttn::parse("fused"), Some(VisionAttn::Fused));
        assert_eq!(VisionAttn::parse("blocked"), Some(VisionAttn::Blocked));
        assert_eq!(VisionAttn::parse("flash"), None);
        for mode in [VisionAttn::Npu, VisionAttn::Fused, VisionAttn::Blocked] {
            assert_eq!(VisionAttn::parse(mode.name()), Some(mode));
            assert_eq!(VisionAttn::from_code(mode.code()), mode);
        }
        // The enum default is fused (non-NPU builds); `--vision-attn-device npu`
        // defaults on separately, from the CLI's resolved `--npu`.
        assert_eq!(VisionAttn::default(), VisionAttn::Fused);
    }
}
