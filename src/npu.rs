//! RK3588 NPU offload for the model's projections (`--npu`).
//!
//! Projection weights are packed once into resident NPU buffers (fp16, through
//! `librocketnpu`); the CPU keeps no copies (pack-and-drop). Every projection
//! forward is converted to fp16, run on the NPU and converted back to f32 for
//! the CPU-side ops (attention, norms, RoPE, activations).
//!
//! aarch64-only, behind the `npu` cargo feature.

#![cfg(all(feature = "npu", target_arch = "aarch64"))]

use burn::prelude::*;
use burn::tensor::TensorData;
use burn_rocket::half::f16;
use burn_rocket::{RocketCtx, RocketWeight};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

// Wall-time instrumentation (microseconds) for the NPU path: our f32<->f16
// conversion work vs the library call (host pack + NPU wait).
static T_CONVERT_US: AtomicU64 = AtomicU64::new(0);
static T_NPU_US: AtomicU64 = AtomicU64::new(0);
static NPU_CALLS: AtomicU64 = AtomicU64::new(0);

/// `(convert_s, npu_s, calls)` accumulated since the last reset.
pub fn npu_stats() -> (f64, f64, u64) {
    (
        T_CONVERT_US.load(Ordering::Relaxed) as f64 / 1e6,
        T_NPU_US.load(Ordering::Relaxed) as f64 / 1e6,
        NPU_CALLS.load(Ordering::Relaxed),
    )
}

/// Reset the NPU wall-time counters.
pub fn npu_stats_reset() {
    T_CONVERT_US.store(0, Ordering::Relaxed);
    T_NPU_US.store(0, Ordering::Relaxed);
    NPU_CALLS.store(0, Ordering::Relaxed);
}

/// Which projection of a transformer layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjKind {
    Q = 0,
    K = 1,
    V = 2,
    O = 3,
    Gate = 4,
    Up = 5,
    Down = 6,
}

impl ProjKind {
    pub const ALL: [ProjKind; 7] = [
        Self::Q,
        Self::K,
        Self::V,
        Self::O,
        Self::Gate,
        Self::Up,
        Self::Down,
    ];

    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }

    /// Safetensors key suffix within `model.layers.{i}`.
    pub fn key(self) -> &'static str {
        match self {
            Self::Q => "self_attn.q_proj.weight",
            Self::K => "self_attn.k_proj.weight",
            Self::V => "self_attn.v_proj.weight",
            Self::O => "self_attn.o_proj.weight",
            Self::Gate => "mlp.gate_proj.weight",
            Self::Up => "mlp.up_proj.weight",
            Self::Down => "mlp.down_proj.weight",
        }
    }
}

struct Slot {
    weight: RocketWeight,
    k: usize,
    n: usize,
}

// SAFETY: `librocketnpu` contexts are not thread-safe, but this model is only used
// from one thread at a time (the CLI main thread, or the server's request handler
// behind a mutex). Burn's `Module` trait requires its containers to be `Send + Sync`.
unsafe impl Send for NpuModel {}
unsafe impl Sync for NpuModel {}

/// One NPU context holding the resident weights of the whole model.
///
/// Not `Send`/`Sync` (`librocketnpu` contexts are single-threaded); the server
/// serializes requests, so one instance per process is enough.
pub struct NpuModel {
    ctx: RocketCtx,
    slots: Vec<Option<Slot>>,
    n_layers: usize,
}

impl NpuModel {
    pub fn new(n_layers: usize, nthreads: usize) -> anyhow::Result<Self> {        let ctx = RocketCtx::new(nthreads)
            .map_err(|e| anyhow::anyhow!("NPU context creation failed: {e}"))?;
        Ok(Self {
            ctx,
            slots: (0..n_layers * ProjKind::ALL.len()).map(|_| None).collect(),
            n_layers,
        })
    }

    /// Pack a `[n, k]` row-major fp16 weight (HF layout `[out, in]`, no transpose)
    /// into a resident NPU buffer.
    pub fn pack(
        &mut self,
        layer: usize,
        kind: ProjKind,
        k: usize,
        n: usize,
        b: &[f16],
    ) -> anyhow::Result<()> {
        assert!(layer < self.n_layers);
        // M is only a warm-up row count for tiling; the handle is reused for any
        // M >= 256 with the same K/N.
        let weight = self
            .ctx
            .pack_weight(512, k, n, b)
            .map_err(|e| anyhow::anyhow!("NPU weight pack failed (layer {layer}, {kind:?}): {e}"))?;
        self.slots[layer * ProjKind::ALL.len() + kind.index()] = Some(Slot { weight, k, n });
        Ok(())
    }

    /// `x` is `[1, m, k]` f32; returns `[1, m, n]` f32.
    pub fn forward(&self, layer: usize, kind: ProjKind, x: Tensor<3>) -> Tensor<3> {
        let [b, m, k] = x.dims();
        assert_eq!(b, 1, "the NPU path supports batch size 1");
        let slot = self.slots[layer * ProjKind::ALL.len() + kind.index()]
            .as_ref()
            .unwrap_or_else(|| panic!("NPU weight not packed (layer {layer}, {kind:?})"));
        assert_eq!(k, slot.k, "activation K mismatch for {kind:?}");
        let n = slot.n;

        let t_conv = Instant::now();
        let device = x.device();
        let data = x.into_data();
        let v: Vec<f32> = data.try_to_vec().expect("f32 activations");
        let a16 = f32_to_f16_par(&v);

        // The resident weight is packed for the M >= 256 tiling (Mt is capped there and
        // the layout is M-independent). Small requests are padded up to 256 rows so they
        // can reuse the same resident weights instead of needing a re-pack; the extra
        // rows are ignored on readback.
        let padded_m = m.div_ceil(4) * 4;
        let padded_m = if padded_m < 256 { 256 } else { padded_m };
        let a16 = if padded_m == m {
            a16
        } else {
            burn_rocket::pad_rows(&a16, m, k, padded_m)
        };
        let mut c16 = vec![f16::ZERO; padded_m * n];
        T_CONVERT_US.fetch_add(t_conv.elapsed().as_micros() as u64, Ordering::Relaxed);

        let t_npu = Instant::now();
        self.ctx
            .matmul_prepacked(padded_m, k, n, &a16, &mut c16, &slot.weight)
            .unwrap_or_else(|e| panic!("NPU matmul failed (layer {layer}, {kind:?}): {e}"));
        T_NPU_US.fetch_add(t_npu.elapsed().as_micros() as u64, Ordering::Relaxed);
        NPU_CALLS.fetch_add(1, Ordering::Relaxed);

        let t_conv = Instant::now();
        let c32 = f16_to_f32_par(&c16[..m * n]);
        let out = Tensor::<3>::from_data(TensorData::new(c32, [1, m, n]), &device);
        T_CONVERT_US.fetch_add(t_conv.elapsed().as_micros() as u64, Ordering::Relaxed);
        out
    }

    pub fn n_layers(&self) -> usize {
        self.n_layers
    }
}

/// NPU attention (`rocket_flash_attn_fp16_ctx`): NPU QK/PV with a causal additive
/// mask, host softmax inside the library. One persistent fa context per process;
/// mask + scratch are cached per sequence length.
pub struct NpuAttention {
    fa: burn_rocket::RocketFaCtx,
    n_head: usize,
    n_kv: usize,
    head_dim: usize,
    scale: f32,
    state: std::sync::Mutex<AttnState>,
}

struct AttnState {
    n: usize,
    mask: Vec<f16>, // [n][n] additive: 0 for j<=t, -inf otherwise
    q: Vec<f16>,    // [n_head][n][head_dim]
    k: Vec<f16>,    // [n_kv][n][head_dim]
    v: Vec<f16>,    // [n_kv][head_dim][n]  (per-head transposed)
    out: Vec<f16>,  // [n_head][n][head_dim]
}

// SAFETY: as for `NpuModel`: the fa context is not thread-safe, but this handle is only
// used from one thread at a time (requests are serialized). Burn's `Module` trait
// requires its containers to be `Send + Sync`.
unsafe impl Send for NpuAttention {}
unsafe impl Sync for NpuAttention {}

impl std::fmt::Debug for NpuAttention {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "NpuAttention(n_head={}, n_kv={}, head_dim={})",
            self.n_head, self.n_kv, self.head_dim
        )
    }
}

impl NpuAttention {
    pub fn new(nthreads: usize, n_head: usize, n_kv: usize, head_dim: usize) -> anyhow::Result<Self> {        let fa = burn_rocket::RocketFaCtx::new(nthreads)
            .map_err(|e| anyhow::anyhow!("NPU attention context creation failed: {e}"))?;
        Ok(Self {
            fa,
            n_head,
            n_kv,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            state: std::sync::Mutex::new(AttnState {
                n: 0,
                mask: Vec::new(),
                q: Vec::new(),
                k: Vec::new(),
                v: Vec::new(),
                out: Vec::new(),
            }),
        })
    }

    /// `q` `[1, n, n_head*d]`, `k`/`v` `[1, n, n_kv*d]` (f32, RoPE + QK-norm applied).
    /// Returns `[1, n, n_head*d]` f32 for `o_proj`.
    pub fn forward(&self, q: Tensor<3>, k: Tensor<3>, v: Tensor<3>) -> Tensor<3> {
        let [b, n, hd] = q.dims();
        assert_eq!(b, 1, "the NPU attention path supports batch size 1");
        let (h, kv, d) = (self.n_head, self.n_kv, self.head_dim);
        assert_eq!(hd, h * d, "q width must be n_head*head_dim");
        let device = q.device();

        let qv: Vec<f32> = q.to_data().try_to_vec().expect("f32 q");
        let kvv: Vec<f32> = k.to_data().try_to_vec().expect("f32 k");
        let vvv: Vec<f32> = v.to_data().try_to_vec().expect("f32 v");

        let mut st = self.state.lock().expect("attn state");
        if st.n != n {
            st.n = n;
            st.mask = build_causal_mask(n);
            st.q = vec![f16::ZERO; h * n * d];
            st.k = vec![f16::ZERO; kv * n * d];
            st.v = vec![f16::ZERO; kv * d * n];
            st.out = vec![f16::ZERO; h * n * d];
        }
        let t_conv = Instant::now();
        let AttnState { mask, q, k, v, out, .. } = &mut *st;
        fill_heads(&qv, n, h, d, q);
        fill_heads(&kvv, n, kv, d, k);
        fill_heads_transposed(&vvv, n, kv, d, v);
        T_CONVERT_US.fetch_add(t_conv.elapsed().as_micros() as u64, Ordering::Relaxed);

        let t_npu = Instant::now();
        self.fa
            .flash_attn(n, n, d, d, h, kv, self.scale, q, k, v, Some(mask), out)
            .unwrap_or_else(|e| panic!("NPU attention failed: {e}"));
        T_NPU_US.fetch_add(t_npu.elapsed().as_micros() as u64, Ordering::Relaxed);
        NPU_CALLS.fetch_add(1, Ordering::Relaxed);

        let t_conv = Instant::now();
        let o = unpack_heads(out, n, h, d);
        let res = Tensor::<3>::from_data(TensorData::new(o, [1, n, h * d]), &device);
        T_CONVERT_US.fetch_add(t_conv.elapsed().as_micros() as u64, Ordering::Relaxed);
        res
    }
}

/// Parallel f32 -> f16 conversion (rayon; chunked to keep it SIMD-friendly).
fn f32_to_f16_par(src: &[f32]) -> Vec<f16> {
    use rayon::prelude::*;
    const CHUNK: usize = 8192;
    let mut out = vec![f16::ZERO; src.len()];
    out.par_chunks_mut(CHUNK)
        .zip(src.par_chunks(CHUNK))
        .for_each(|(o, s)| {
            for (d, &x) in o.iter_mut().zip(s) {
                *d = f16::from_f32(x);
            }
        });
    out
}

/// Parallel f16 -> f32 conversion.
fn f16_to_f32_par(src: &[f16]) -> Vec<f32> {
    use rayon::prelude::*;
    const CHUNK: usize = 8192;
    let mut out = vec![0f32; src.len()];
    out.par_chunks_mut(CHUNK)
        .zip(src.par_chunks(CHUNK))
        .for_each(|(o, s)| {
            for (d, &x) in o.iter_mut().zip(s) {
                *d = x.to_f32();
            }
        });
    out
}

/// Causal additive mask: `mask[t][j] = 0` for `j <= t`, `-inf` otherwise.
fn build_causal_mask(n: usize) -> Vec<f16> {    use rayon::prelude::*;
    let mut mask = vec![f16::ZERO; n * n];
    mask.par_chunks_mut(n).enumerate().for_each(|(t, row)| {
        for m in row[t + 1..].iter_mut() {
            *m = f16::NEG_INFINITY;
        }
    });
    mask
}

/// `src` is `[n, heads*d]` row-major; writes `dst[head][n][d]`.
fn fill_heads(src: &[f32], n: usize, heads: usize, d: usize, dst: &mut [f16]) {
    use rayon::prelude::*;
    let hd = heads * d;
    dst.par_chunks_mut(n * d).enumerate().for_each(|(hi, dst_h)| {
        for (s, dst_row) in dst_h.chunks_mut(d).enumerate() {
            let src_row = &src[s * hd + hi * d..s * hd + hi * d + d];
            for (o, &x) in dst_row.iter_mut().zip(src_row) {
                *o = f16::from_f32(x);
            }
        }
    });
}

/// `src` is `[n, heads*d]` row-major; writes the per-head transpose `dst[head][d][n]`
/// (the AV B-operand layout).
fn fill_heads_transposed(src: &[f32], n: usize, heads: usize, d: usize, dst: &mut [f16]) {
    use rayon::prelude::*;
    let hd = heads * d;
    dst.par_chunks_mut(d * n).enumerate().for_each(|(hi, dst_h)| {
        for s in 0..n {
            let src_row = &src[s * hd + hi * d..s * hd + hi * d + d];
            for (di, &x) in src_row.iter().enumerate() {
                dst_h[di * n + s] = f16::from_f32(x);
            }
        }
    });
}

/// `src` is `[heads][n][d]`; returns `[n, heads*d]` row-major f32.
fn unpack_heads(src: &[f16], n: usize, heads: usize, d: usize) -> Vec<f32> {
    use rayon::prelude::*;
    let hd = heads * d;
    let mut dst = vec![0f32; n * hd];
    dst.par_chunks_mut(hd).enumerate().for_each(|(s, row)| {
        for hi in 0..heads {
            let src_row = &src[hi * n * d + s * d..hi * n * d + s * d + d];
            for (o, &x) in row[hi * d..hi * d + d].iter_mut().zip(src_row) {
                *o = x.to_f32();
            }
        }
    });
    dst
}

/// A handle to one packed projection, stored in the model's `Proj` fields.
#[derive(Clone)]
pub struct NpuRef {
    model: Arc<NpuModel>,
    layer: usize,
    kind: ProjKind,
}

impl std::fmt::Debug for NpuRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NpuRef(layer={}, {:?})", self.layer, self.kind)
    }
}

impl NpuRef {
    pub fn new(model: Arc<NpuModel>, layer: usize, kind: ProjKind) -> Self {
        Self { model, layer, kind }
    }

    #[inline]
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        self.model.forward(self.layer, self.kind, x)
    }
}
