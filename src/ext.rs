//! NPU offload as a Burn backend extension (`#[backend_extension(Flex)]`).
//!
//! Burn tensors reach these operations through the macro-generated `Dispatch`
//! routing; the `Flex` implementation below runs the resident-weight matmul and
//! masked grouped-query attention on the RK3588 NPU through `librocketnpu`.
//!
//! Use the high-level helpers ([`pack`], [`pack2`], [`pack3`], [`matmul`],
//! [`attention`]) so model code only sees ordinary `Tensor`s:
//!
//! ```ignore
//! burn_rocket::init(5)?;
//! let w = burn_rocket::pack(weight); // pack-and-drop
//! let y = burn_rocket::matmul(x, &w);
//! ```
//!
//! Weights stay resident for the process lifetime. The ops are inference-only:
//! the extension macro generates no autodiff support, and calling them with an
//! autodiff context panics inside Burn.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::Instant;

use burn::backend::tensor::FloatTensor;
use burn::backend::{Backend, Dispatch, Flex, TensorMetadata, backend_extension};
use burn::tensor::{DType, Tensor, TensorData};
use half::f16;

use crate::masks::{build_causal_mask, build_causal_window_mask, build_window_mask};
use crate::{
    Error, OpFailure, RocketCtx, RocketFaCtx, RocketI8Ctx, RocketI8Weight, RocketWeight, pad_rows,
};

/// The Flex backend's float primitive, the concrete type the ops execute on.
type FlexTensor = FloatTensor<Flex>;

/// Handle to a weight resident in NPU memory, returned by [`pack`], [`pack2`]
/// or [`pack3`]. Weights are never released; their ids are valid for the
/// process lifetime (bench/serve load one model).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WeightId(u64);

impl WeightId {
    /// The raw registry id.
    pub fn id(self) -> u64 {
        self.0
    }
}

/// Burn backend extension: RK3588 NPU operations on the `Flex` backend.
///
/// Tensor arguments are owned float primitives; the plain-`u64` return of the
/// pack ops is passed back through the dispatch boundary unchanged (Burn treats
/// it as an ordinary value, not a tensor).
#[backend_extension(Flex)]
pub trait RocketOps: Backend {
    /// Pack a `[N, K]` f32 weight into resident NPU memory (the host copy is
    /// dropped) and return its handle.
    fn rocket_pack(t: FloatTensor<Self>) -> u64;
    /// Pack two weights sharing one input, concatenated along N.
    fn rocket_pack2(a: FloatTensor<Self>, b: FloatTensor<Self>) -> u64;
    /// Pack three weights sharing one input, concatenated along N.
    fn rocket_pack3(a: FloatTensor<Self>, b: FloatTensor<Self>, c: FloatTensor<Self>) -> u64;
    /// Pack an int8 group-wise weight (`[N, K]` f32 quantized host-side at pack
    /// time) into resident NPU memory; `group` is the K-group width (`% 32 == 0`).
    fn rocket_pack_i8(t: FloatTensor<Self>, group: usize) -> u64;
    /// Fused `gelu_approximate(a) * b` over equal-shaped f32 tensors, in one pass.
    fn rocket_gelu_mul(a: FloatTensor<Self>, b: FloatTensor<Self>) -> FloatTensor<Self>;
    /// Fused weighted RMSNorm over the last dim, in one pass:
    /// `x * (mean(x^2) + eps)^-0.5 * w` (`w` has the last dim's size).
    fn rocket_rms_norm(x: FloatTensor<Self>, w: FloatTensor<Self>, eps: f64) -> FloatTensor<Self>;
    /// Fused scale-free RMSNorm over the last dim: `x * (mean(x^2) + eps)^-0.5`.
    fn rocket_rms_norm_noscale(x: FloatTensor<Self>, eps: f64) -> FloatTensor<Self>;
    /// Fused rotate-half RoPE: `x` is `[B, S, H, D]`, `cos`/`sin` are `[S, D/2]`.
    fn rocket_rope(
        x: FloatTensor<Self>,
        cos: FloatTensor<Self>,
        sin: FloatTensor<Self>,
    ) -> FloatTensor<Self>;
    /// `[.., M, K] * [N, K]^T -> [.., M, N]` with a resident weight.
    fn rocket_matmul(x: FloatTensor<Self>, id: u64) -> FloatTensor<Self>;
    /// Masked grouped-query attention for `[1, S, H*D]` / `[1, S, KV*D]` inputs.
    #[allow(clippy::too_many_arguments)]
    fn rocket_attention(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        n_head: usize,
        n_kv_heads: usize,
        head_dim: usize,
        scale: f64,
        softcap: Option<f32>,
        is_causal: bool,
    ) -> FloatTensor<Self>;
    /// Bidirectional windowed attention: position `t` attends to `j` iff
    /// `|t - j| <= window`. `window < 0` disables the mask (full bidirectional).
    #[allow(clippy::too_many_arguments)]
    fn rocket_attention_window(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        n_head: usize,
        n_kv_heads: usize,
        head_dim: usize,
        scale: f64,
        softcap: Option<f32>,
        window: i64,
    ) -> FloatTensor<Self>;
    /// Causal sliding-window attention (generation prefill): position `t`
    /// attends to `j` iff `t - window < j <= t`. `window < 0` = plain causal.
    #[allow(clippy::too_many_arguments)]
    fn rocket_attention_causal_window(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        n_head: usize,
        n_kv_heads: usize,
        head_dim: usize,
        scale: f64,
        softcap: Option<f32>,
        window: i64,
    ) -> FloatTensor<Self>;
    /// Chunked bidirectional band attention: query `q_start + i` attends to key
    /// `kv_start + j` iff `|(q_start + i) - (kv_start + j)| <= window` (`< 0` =
    /// no mask). `q` is `[1, n_q, H*D]`, `k`/`v` are `[1, n_kv_len, KV*D]`; the
    /// key window typically spans only the band around the query chunk.
    #[allow(clippy::too_many_arguments)]
    fn rocket_attention_window_block(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        n_head: usize,
        n_kv_heads: usize,
        head_dim: usize,
        scale: f64,
        softcap: Option<f32>,
        window: i64,
        q_start: i64,
        kv_start: i64,
    ) -> FloatTensor<Self>;
}

// ---------------------------------------------------------------------------
// Wall-time instrumentation (microseconds): our f32<->f16 conversion work vs the
// library call (host pack + NPU wait). Shared with the app's bench summary.
// ---------------------------------------------------------------------------

static T_CONVERT_US: AtomicU64 = AtomicU64::new(0);
static T_NPU_US: AtomicU64 = AtomicU64::new(0);
static NPU_CALLS: AtomicU64 = AtomicU64::new(0);

/// Accumulated NPU wall times.
#[derive(Debug, Clone, Copy)]
pub struct Stats {
    /// f32<->f16 conversion and layout work on the CPU side.
    pub convert_s: f64,
    /// Time inside `librocketnpu` calls.
    pub npu_s: f64,
    /// Number of NPU calls (matmuls + attention).
    pub calls: u64,
}

/// `(convert_s, npu_s, calls)` accumulated since the last reset.
pub fn stats() -> Stats {
    Stats {
        convert_s: T_CONVERT_US.load(Ordering::Relaxed) as f64 / 1e6,
        npu_s: T_NPU_US.load(Ordering::Relaxed) as f64 / 1e6,
        calls: NPU_CALLS.load(Ordering::Relaxed),
    }
}

/// Reset the NPU wall-time counters.
pub fn stats_reset() {
    T_CONVERT_US.store(0, Ordering::Relaxed);
    T_NPU_US.store(0, Ordering::Relaxed);
    NPU_CALLS.store(0, Ordering::Relaxed);
}

// ---------------------------------------------------------------------------
// Global engine: one NPU context, the resident weights and the attention
// scratch/fa context. `librocketnpu` contexts are not thread-safe, so every op
// goes through this mutex.
// ---------------------------------------------------------------------------

/// M warm-up row count at pack time; with canonical tiling the resident weights
/// are reused for any M (>= 4).
const PACK_M: usize = 512;

/// Padding floor for the M dimension in *legacy* tiling mode (canonical tiling
/// off): resident weights are packed for the M >= 256 tiling and cannot serve
/// smaller matmuls directly. Canonical mode pads only to the M % 4 alignment.
const MIN_M: usize = 256;

/// Row chunks above this size are split into separate NPU matmuls: the per-call
/// scratch grows with M and a ~30k-row activation needs ~200 MB of input BOs,
/// enough to fail with `ROCKET_E_NOMEM` under memory pressure. Rows are
/// independent, so chunking changes no result. `ROCKET_MATMUL_CHUNK_M` overrides
/// the default (0 disables chunking).
const MATMUL_CHUNK_M: usize = 8192;

fn matmul_chunk_m() -> usize {
    static OVERRIDE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *OVERRIDE.get_or_init(|| {
        std::env::var("ROCKET_MATMUL_CHUNK_M")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(MATMUL_CHUNK_M)
    })
}

/// Abort the current op with a structured payload: the default panic hook would
/// print a bare `Box<dyn Any>`, so log the detail first.
fn op_failure(op: &'static str, rc: i32, m: usize, k: usize, n: usize) -> ! {
    eprintln!("burn-rocket: {op} failed: rc={rc} m={m} k={k} n={n}");
    std::panic::panic_any(OpFailure {
        error: Error { op, rc },
        m,
        k,
        n,
    })
}

struct Resident {
    weight: RocketWeight,
    k: usize,
    n: usize,
}

/// A resident group-wise int8 weight plus the host-side weight scales the
/// per-call API needs (`[N, K/group]`).
struct I8Resident {
    weight: RocketI8Weight,
    k: usize,
    n: usize,
    group: usize,
    b_scale: Vec<f32>,
}

struct FaState {
    fa: RocketFaCtx,
    n_head: usize,
    n_kv: usize,
    head_dim: usize,
    /// Query length the buffers/masks are currently sized for.
    n: usize,
    /// KV length the buffers are currently sized for (`== n` for the
    /// full-sequence mask modes).
    n_k: usize,
    mask: Vec<f16>, // [n][n] additive: 0 for j<=t, -inf otherwise
    /// Cached bidirectional band mask for the current `n` (`usize::MAX` = none).
    win: usize,
    win_mask: Vec<f16>, // [n][n] additive: 0 for |t-j|<=win, -inf otherwise
    /// Cached causal sliding-window mask for the current `n`.
    cw: usize,
    cw_mask: Vec<f16>, // [n][n] additive: 0 for t-w<j<=t, -inf otherwise
    /// Cached query-chunk band mask key: `(n_q, n_kv_len, window, q0 - k0)`.
    wb: Option<(usize, usize, usize, usize)>,
    wb_mask: Vec<f16>, // [n_q][n_kv_len] block band mask
    q: Vec<f16>,       // [n_head][n][head_dim]
    k: Vec<f16>,       // [n_kv_heads][n_k][head_dim]
    v: Vec<f16>,       // [n_kv_heads][head_dim][n_k]  (per-head transposed)
    out: Vec<f16>,     // [n_head][n][head_dim]
}

struct Engine {
    threads: usize,
    ctx: RocketCtx,
    fa: Option<FaState>,
    weights: HashMap<u64, Resident>,
    /// Resident int8 weights (the context is created lazily at the first int8
    /// pack; it owns its own worker fds).
    i8_ctx: Option<RocketI8Ctx>,
    i8_weights: HashMap<u64, I8Resident>,
    next_id: u64,
}

// SAFETY: `librocketnpu` contexts are not thread-safe and are deliberately not
// `Send`; the engine is only ever touched while holding `ENGINE`'s mutex, which
// serializes every access. This mirrors the previous per-model `unsafe impl
// Send + Sync` justification.
unsafe impl Send for Engine {}

static ENGINE: Mutex<Option<Engine>> = Mutex::new(None);

fn engine() -> MutexGuard<'static, Option<Engine>> {
    // A panic inside an op poisons the mutex; the engine itself stays usable.
    ENGINE.lock().unwrap_or_else(PoisonError::into_inner)
}

fn with_engine<R>(f: impl FnOnce(&mut Engine) -> R) -> R {
    let mut guard = engine();
    let e = guard
        .as_mut()
        .expect("burn_rocket::init() has not been called");
    f(e)
}

/// Create the global NPU context with `threads` worker threads. Must be called
/// once before any of the [`pack`]/[`matmul`]/[`attention`] helpers; later
/// calls are no-ops.
pub fn init(threads: usize) -> Result<(), Error> {
    let mut guard = engine();
    if guard.is_some() {
        return Ok(());
    }
    let ctx = RocketCtx::new(threads)?;
    *guard = Some(Engine {
        threads,
        ctx,
        fa: None,
        weights: HashMap::new(),
        i8_ctx: None,
        i8_weights: HashMap::new(),
        next_id: 1,
    });
    Ok(())
}

// ---------------------------------------------------------------------------
// Op implementations (concrete `Flex` backend)
// ---------------------------------------------------------------------------

impl RocketOps for Flex {
    fn rocket_pack(t: FloatTensor<Self>) -> u64 {
        pack_one(t, "weight")
    }

    fn rocket_pack2(a: FloatTensor<Self>, b: FloatTensor<Self>) -> u64 {
        pack_many(
            vec![to_host(a, "fused weight 0"), to_host(b, "fused weight 1")],
            "fused weight",
        )
    }

    fn rocket_pack3(a: FloatTensor<Self>, b: FloatTensor<Self>, c: FloatTensor<Self>) -> u64 {
        pack_many(
            vec![
                to_host(a, "fused weight 0"),
                to_host(b, "fused weight 1"),
                to_host(c, "fused weight 2"),
            ],
            "fused weight",
        )
    }

    fn rocket_pack_i8(t: FloatTensor<Self>, group: usize) -> u64 {
        pack_i8_one(t, group)
    }

    fn rocket_gelu_mul(a: FloatTensor<Self>, b: FloatTensor<Self>) -> FloatTensor<Self> {
        gelu_mul_impl(a, b)
    }

    fn rocket_rms_norm(x: FloatTensor<Self>, w: FloatTensor<Self>, eps: f64) -> FloatTensor<Self> {
        rms_norm_impl(x, w, eps)
    }

    fn rocket_rms_norm_noscale(x: FloatTensor<Self>, eps: f64) -> FloatTensor<Self> {
        rms_norm_noscale_impl(x, eps)
    }

    fn rocket_rope(
        x: FloatTensor<Self>,
        cos: FloatTensor<Self>,
        sin: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        rope_impl(x, cos, sin)
    }

    fn rocket_matmul(x: FloatTensor<Self>, id: u64) -> FloatTensor<Self> {
        matmul_impl(x, id)
    }

    #[allow(clippy::too_many_arguments)]
    fn rocket_attention(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        n_head: usize,
        n_kv_heads: usize,
        head_dim: usize,
        scale: f64,
        softcap: Option<f32>,
        is_causal: bool,
    ) -> FloatTensor<Self> {
        attention_impl(
            q,
            k,
            v,
            n_head,
            n_kv_heads,
            head_dim,
            scale,
            softcap,
            if is_causal {
                MaskMode::Causal
            } else {
                MaskMode::None
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn rocket_attention_window(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        n_head: usize,
        n_kv_heads: usize,
        head_dim: usize,
        scale: f64,
        softcap: Option<f32>,
        window: i64,
    ) -> FloatTensor<Self> {
        let mode = if window < 0 {
            MaskMode::None
        } else {
            MaskMode::Window(window as usize)
        };
        attention_impl(q, k, v, n_head, n_kv_heads, head_dim, scale, softcap, mode)
    }

    #[allow(clippy::too_many_arguments)]
    fn rocket_attention_causal_window(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        n_head: usize,
        n_kv_heads: usize,
        head_dim: usize,
        scale: f64,
        softcap: Option<f32>,
        window: i64,
    ) -> FloatTensor<Self> {
        let mode = if window < 0 {
            MaskMode::Causal
        } else {
            MaskMode::CausalWindow(window as usize)
        };
        attention_impl(q, k, v, n_head, n_kv_heads, head_dim, scale, softcap, mode)
    }

    #[allow(clippy::too_many_arguments)]
    fn rocket_attention_window_block(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        n_head: usize,
        n_kv_heads: usize,
        head_dim: usize,
        scale: f64,
        softcap: Option<f32>,
        window: i64,
        q_start: i64,
        kv_start: i64,
    ) -> FloatTensor<Self> {
        let mode = if window < 0 {
            MaskMode::None
        } else {
            MaskMode::WindowBlock {
                window: window as usize,
                q0: q_start.max(0) as usize,
                k0: kv_start.max(0) as usize,
            }
        };
        attention_impl(q, k, v, n_head, n_kv_heads, head_dim, scale, softcap, mode)
    }
}

/// `t` is a `[N, K]` f32 tensor; returns `(k, n, row-major [N, K] fp16)`.
fn to_host(t: FlexTensor, what: &str) -> (usize, usize, Vec<f16>) {
    assert_eq!(
        t.dtype(),
        DType::F32,
        "{what}: the NPU path needs f32 weights"
    );
    let [n, k] = t.shape().dims::<2>();
    let values: Vec<f32> = t
        .into_data()
        .try_to_vec()
        .unwrap_or_else(|e| panic!("{what}: expected f32 data ({e})"));
    (k, n, f32_to_f16_par(&values))
}

fn pack_one(t: FlexTensor, what: &str) -> u64 {
    let (k, n, b) = to_host(t, what);
    with_engine(|e| {
        let weight = e
            .ctx
            .pack_weight(PACK_M, k, n, &b)
            .unwrap_or_else(|err| op_failure("rocket_weights_pack", err.rc, PACK_M, k, n));
        let id = e.next_id;
        e.next_id += 1;
        e.weights.insert(id, Resident { weight, k, n });
        id
    })
}

/// Fused groups: every part is `[N_i, K]` with a shared `K`; the resident weight
/// is the concatenation along N, so one matmul produces all outputs.
fn pack_many(parts: Vec<(usize, usize, Vec<f16>)>, what: &str) -> u64 {
    let k = parts[0].0;
    assert!(
        parts.iter().all(|(kk, _, _)| *kk == k),
        "{what}: all parts must share K"
    );
    let bufs: Vec<&[f16]> = parts.iter().map(|(_, _, b)| b.as_slice()).collect();
    let n: usize = parts.iter().map(|(_, n, _)| *n).sum();
    with_engine(|e| {
        let weight = e
            .ctx
            .pack_weight_seg(PACK_M, k, &bufs)
            .unwrap_or_else(|err| op_failure("rocket_weights_pack_seg", err.rc, PACK_M, k, n));
        let id = e.next_id;
        e.next_id += 1;
        e.weights.insert(id, Resident { weight, k, n });
        id
    })
}

/// Pack one `[N, K]` f32 weight as a resident group-wise int8 weight: the codes
/// are quantized host-side (symmetric `max/127` per K-group) and scattered into
/// NPU BOs once; the weight scales stay host-side for the per-call dequant.
fn pack_i8_one(t: FlexTensor, group: usize) -> u64 {
    assert_eq!(t.dtype(), DType::F32, "the NPU path needs f32 weights");
    let [n, k] = t.shape().dims::<2>();
    assert!(
        group > 0 && group.is_multiple_of(32) && k.is_multiple_of(group),
        "int8 group {group} must be a positive multiple of 32 dividing K={k}"
    );
    assert_eq!(k % 32, 0, "the int8 path needs K % 32 == 0 (K={k})");
    assert_eq!(n % 32, 0, "the int8 path needs N % 32 == 0 (N={n})");
    let values: Vec<f32> = t
        .into_data()
        .try_to_vec()
        .expect("the NPU path needs f32 weights");
    let (q, b_scale) = quantize_i8_weight(&values, n, k, group);
    with_engine(|e| {
        let threads = e.threads;
        let ctx = e.i8_ctx.get_or_insert_with(|| {
            RocketI8Ctx::new(threads)
                .unwrap_or_else(|err| op_failure("rocket_i8_ctx_create", err.rc, 0, k, n))
        });
        let weight = ctx
            .pack_weight_gw(PACK_M, k, n, &q, group)
            .unwrap_or_else(|err| op_failure("rocket_i8_weights_pack_gw", err.rc, PACK_M, k, n));
        let id = e.next_id;
        e.next_id += 1;
        e.i8_weights.insert(
            id,
            I8Resident {
                weight,
                k,
                n,
                group,
                b_scale,
            },
        );
        id
    })
}

/// Quantize a `[N, K]` f32 weight into symmetric per-K-group int8 codes; returns
/// `(codes, scales [N, K/group])`, rayon-parallel over output channels.
fn quantize_i8_weight(values: &[f32], n: usize, k: usize, group: usize) -> (Vec<i8>, Vec<f32>) {
    use rayon::prelude::*;

    let n_groups = k / group;
    let mut q = vec![0i8; n * k];
    let mut scales = vec![0f32; n * n_groups];
    q.par_chunks_mut(k)
        .zip(scales.par_chunks_mut(n_groups))
        .zip(values.par_chunks(k))
        .for_each(|((qrow, srow), xrow)| crate::host::quantize_row_groups(xrow, qrow, srow, group));
    (q, scales)
}

/// Quantize `m` rows of `k` f32 values into int8 with per-row per-group scales;
/// the tail is padded to `M % 4 == 0` (zero codes, scales 1), because the int8
/// resident path has no `M == 1` pad and requires `M % 4 == 0`.
fn quantize_i8_rows(values: &[f32], m: usize, k: usize, group: usize) -> (Vec<i8>, Vec<f32>) {
    use rayon::prelude::*;

    let n_groups = k / group;
    let m4 = m.div_ceil(4) * 4;
    let mut q = vec![0i8; m4 * k];
    let mut scales = vec![1.0f32; m4 * n_groups];
    q.par_chunks_mut(k)
        .zip(scales.par_chunks_mut(n_groups))
        .zip(values.par_chunks(k))
        .for_each(|((qrow, srow), xrow)| crate::host::quantize_row_groups(xrow, qrow, srow, group));
    (q, scales)
}

/// `t` must be f32; returns its shape and values.
fn to_host_f32(t: FlexTensor, what: &str) -> (Vec<usize>, Vec<f32>) {
    assert_eq!(t.dtype(), DType::F32, "{what} must be f32");
    let dims: Vec<usize> = t.shape().into();
    let values: Vec<f32> = t
        .into_data()
        .try_to_vec()
        .unwrap_or_else(|e| panic!("{what}: expected f32 data ({e})"));
    (dims, values)
}

/// Fused `gelu_approximate(a) * b`: one rayon pass instead of the composite's
/// ~9 single-threaded flex ops (with a `tanh` and a `powf` per element).
fn gelu_mul_impl(a: FlexTensor, b: FlexTensor) -> FlexTensor {
    use rayon::prelude::*;
    const CHUNK: usize = 8192;

    let (dims, av) = to_host_f32(a, "gelu_mul input a");
    let (bdims, bv) = to_host_f32(b, "gelu_mul input b");
    assert_eq!(dims, bdims, "gelu_mul inputs must share a shape");
    let mut out = vec![0f32; av.len()];
    out.par_chunks_mut(CHUNK)
        .zip(av.par_chunks(CHUNK))
        .zip(bv.par_chunks(CHUNK))
        .for_each(|((o, g), u)| crate::host::gelu_mul(g, u, o));
    FlexTensor::from_data(TensorData::new(out, dims))
}

/// Fused weighted RMSNorm over the last dim: one pass per row instead of the
/// composite `square / mean / add / sqrt / div / mul` chain.
fn rms_norm_impl(x: FlexTensor, w: FlexTensor, eps: f64) -> FlexTensor {
    use rayon::prelude::*;

    let (dims, xv) = to_host_f32(x, "rms_norm input");
    let d = *dims.last().expect("rms_norm needs at least one dim");
    let (_, wv) = to_host_f32(w, "rms_norm weight");
    assert_eq!(wv.len(), d, "rms_norm: the weight has the last dim's size");
    let mut out = vec![0f32; xv.len()];
    out.par_chunks_mut(d)
        .zip(xv.par_chunks(d))
        .for_each(|(o, row)| crate::host::rms_norm_row(row, &wv, eps as f32, o));
    FlexTensor::from_data(TensorData::new(out, dims))
}

/// Fused scale-free RMSNorm over the last dim (`x * (mean(x^2) + eps)^-0.5`).
fn rms_norm_noscale_impl(x: FlexTensor, eps: f64) -> FlexTensor {
    use rayon::prelude::*;

    let (dims, xv) = to_host_f32(x, "rms_norm_noscale input");
    let d = *dims
        .last()
        .expect("rms_norm_noscale needs at least one dim");
    let mut out = vec![0f32; xv.len()];
    out.par_chunks_mut(d)
        .zip(xv.par_chunks(d))
        .for_each(|(o, row)| crate::host::rms_norm_row(row, &[], eps as f32, o));
    FlexTensor::from_data(TensorData::new(out, dims))
}

/// Fused rotate-half RoPE: one pass instead of slice/mul/sub/add/cat chains.
fn rope_impl(x: FlexTensor, cos: FlexTensor, sin: FlexTensor) -> FlexTensor {
    use rayon::prelude::*;

    let (dims, xv) = to_host_f32(x, "rope input");
    assert_eq!(dims.len(), 4, "rope: x must be [B, S, H, D]");
    let (s, h, d) = (dims[1], dims[2], dims[3]);
    assert_eq!(d % 2, 0, "rope: the head dim must be even");
    let half = d / 2;
    let (_, cv) = to_host_f32(cos, "rope cos");
    let (_, sv) = to_host_f32(sin, "rope sin");
    assert_eq!(cv.len(), s * half, "rope: cos must be [S, D/2]");
    assert_eq!(sv.len(), s * half, "rope: sin must be [S, D/2]");
    let mut out = vec![0f32; xv.len()];
    out.par_chunks_mut(d).enumerate().for_each(|(row, o)| {
        let si = (row / h) % s;
        crate::host::rope_row(
            &xv[row * d..(row + 1) * d],
            &cv[si * half..(si + 1) * half],
            &sv[si * half..(si + 1) * half],
            half,
            o,
        );
    });
    FlexTensor::from_data(TensorData::new(out, dims))
}

/// How a registry id is executed; resolved before the host-side conversion so
/// the (parallel) conversion runs outside the engine lock.
enum WeightKind {
    F16,
    I8 { group: usize },
}

fn weight_kind(id: u64, k: usize) -> WeightKind {
    with_engine(|e| {
        if let Some(r) = e.weights.get(&id) {
            assert_eq!(k, r.k, "activation K mismatch for NPU weight id {id}");
            WeightKind::F16
        } else if let Some(r) = e.i8_weights.get(&id) {
            assert_eq!(k, r.k, "activation K mismatch for NPU weight id {id}");
            WeightKind::I8 { group: r.group }
        } else {
            panic!("NPU weight id {id} is not packed");
        }
    })
}

fn matmul_impl(x: FlexTensor, id: u64) -> FlexTensor {
    let dims: Vec<usize> = x.shape().into();
    assert!(dims.len() >= 2, "NPU matmul needs at least [M, K]");
    let k = dims[dims.len() - 1];
    let m: usize = dims[..dims.len() - 1].iter().product();
    assert_eq!(x.dtype(), DType::F32, "the NPU path needs f32 activations");
    let values: Vec<f32> = x
        .into_data()
        .try_to_vec()
        .expect("the NPU path needs f32 activations");

    match weight_kind(id, k) {
        WeightKind::F16 => {
            let t_conv = Instant::now();
            let a16 = f32_to_f16_par(&values);
            T_CONVERT_US.fetch_add(t_conv.elapsed().as_micros() as u64, Ordering::Relaxed);
            matmul_f16(dims, m, k, id, &a16)
        }
        WeightKind::I8 { group } => {
            let t_conv = Instant::now();
            let (q, a_scale) = quantize_i8_rows(&values, m, k, group);
            T_CONVERT_US.fetch_add(t_conv.elapsed().as_micros() as u64, Ordering::Relaxed);
            matmul_i8(dims, m, k, id, group, &q, &a_scale)
        }
    }
}

fn matmul_f16(dims: Vec<usize>, m: usize, k: usize, id: u64, a16: &[f16]) -> FlexTensor {
    with_engine(|e| {
        let r = e
            .weights
            .get(&id)
            .unwrap_or_else(|| panic!("NPU weight id {id} is not packed"));
        let n = r.n;
        let chunk_m = matmul_chunk_m();

        // Rows are independent, so M is safe to split into bounded chunks
        // (extra pad rows are ignored on readback).
        let mut c32: Vec<f32> = Vec::with_capacity(m * n);
        let mut off = 0;
        while off < m {
            let rows = if chunk_m == 0 {
                m - off
            } else {
                (m - off).min(chunk_m)
            };
            // Rows are independent, so M is safe to split: full chunks are
            // matmul'd at their exact size (bounded NPU scratch). With the
            // canonical tiling the resident weight serves any M, so only the
            // M % 4 alignment pad remains; the legacy tiling needs the floor.
            let padded_m = if crate::canonical_tiling() {
                (rows.div_ceil(4) * 4).max(4)
            } else {
                (rows.div_ceil(4) * 4).max(MIN_M)
            };
            let src = &a16[off * k..(off + rows) * k];

            let t_conv = Instant::now();
            let padded = (padded_m != rows).then(|| pad_rows(src, rows, k, padded_m));
            let a: &[f16] = padded.as_deref().unwrap_or(src);
            let mut c16 = vec![f16::ZERO; padded_m * n];
            T_CONVERT_US.fetch_add(t_conv.elapsed().as_micros() as u64, Ordering::Relaxed);

            let t_npu = Instant::now();
            e.ctx
                .matmul_prepacked(padded_m, k, n, a, &mut c16, &r.weight)
                .unwrap_or_else(|err| {
                    op_failure("rocket_matmul_fp16_prepacked", err.rc, rows, k, n)
                });
            T_NPU_US.fetch_add(t_npu.elapsed().as_micros() as u64, Ordering::Relaxed);
            NPU_CALLS.fetch_add(1, Ordering::Relaxed);

            let t_conv = Instant::now();
            c32.extend(f16_to_f32_par(&c16[..rows * n]));
            T_CONVERT_US.fetch_add(t_conv.elapsed().as_micros() as u64, Ordering::Relaxed);
            off += rows;
        }

        let mut out_dims = dims;
        *out_dims.last_mut().unwrap() = n;
        FlexTensor::from_data(TensorData::new(c32, out_dims))
    })
}

/// Group-wise int8 matmul with a resident weight. The output buffer is sized to
/// the `M % 4` pad (`ceil4(m)`) so chunks write straight into it; the pad rows
/// are truncated before the tensor is built.
fn matmul_i8(
    dims: Vec<usize>,
    m: usize,
    k: usize,
    id: u64,
    group: usize,
    q: &[i8],
    a_scale: &[f32],
) -> FlexTensor {
    debug_assert_eq!(q.len(), m.div_ceil(4) * 4 * k);
    let n_groups = k / group;
    with_engine(|e| {
        let r = e
            .i8_weights
            .get(&id)
            .unwrap_or_else(|| panic!("NPU weight id {id} is not packed"));
        let n = r.n;
        // Chunk starts stay % 4 (the int8 path has no M == 1 pad and needs
        // M % 4 == 0); the `m % 4` tail is covered by `q`'s zero pad rows.
        let chunk_m = matmul_chunk_m();
        let chunk = if chunk_m == 0 { m } else { chunk_m.max(4) & !3 };
        let mut c32: Vec<f32> = vec![0f32; m.div_ceil(4) * 4 * n];
        let mut off = 0;
        while off < m {
            let rows = (m - off).min(chunk);
            let padded_m = (rows + 3) & !3;
            let a = &q[off * k..(off + padded_m) * k];
            let sa = &a_scale[off * n_groups..(off + padded_m) * n_groups];
            let cf = &mut c32[off * n..(off + padded_m) * n];

            let t_npu = Instant::now();
            e.i8_ctx
                .as_ref()
                .expect("the int8 context exists once an int8 weight is packed")
                .matmul_gw(padded_m, k, n, a, sa, &r.b_scale, cf, &r.weight)
                .unwrap_or_else(|err| {
                    op_failure("rocket_matmul_int8_prepacked_gw", err.rc, rows, k, n)
                });
            T_NPU_US.fetch_add(t_npu.elapsed().as_micros() as u64, Ordering::Relaxed);
            NPU_CALLS.fetch_add(1, Ordering::Relaxed);
            off += rows;
        }
        c32.truncate(m * n);

        let mut out_dims = dims;
        *out_dims.last_mut().unwrap() = n;
        FlexTensor::from_data(TensorData::new(c32, out_dims))
    })
}

/// Additive-mask selection for the NPU attention op.
#[derive(Clone, Copy, PartialEq, Eq)]
enum MaskMode {
    /// Full bidirectional attention (no mask).
    None,
    /// Causal: `j <= t`.
    Causal,
    /// Symmetric band: `|t - j| <= window`.
    Window(usize),
    /// Causal sliding window: `t - window < j <= t`.
    CausalWindow(usize),
    /// Band over a query chunk at absolute offset `q0`, against keys starting at
    /// `k0`: keep key `j` iff `|(q0 + i) - (k0 + j)| <= window`.
    WindowBlock { window: usize, q0: usize, k0: usize },
}

#[allow(clippy::too_many_arguments)]
fn attention_impl(
    q: FlexTensor,
    k: FlexTensor,
    v: FlexTensor,
    n_head: usize,
    n_kv: usize,
    head_dim: usize,
    scale: f64,
    softcap: Option<f32>,
    mode: MaskMode,
) -> FlexTensor {
    let q_dims: Vec<usize> = q.shape().into();
    assert_eq!(q_dims.len(), 3, "NPU attention input must be [1, S, H*D]");
    let (b, n) = (q_dims[0], q_dims[1]);
    assert_eq!(b, 1, "the NPU attention path supports batch size 1");
    assert_eq!(
        q_dims[2],
        n_head * head_dim,
        "q width must be n_head * head_dim"
    );
    let (k_dims, v_dims): (Vec<usize>, Vec<usize>) = (k.shape().into(), v.shape().into());
    let n_k = k_dims[1];
    assert_eq!(
        k_dims,
        vec![1, n_k, n_kv * head_dim],
        "NPU attention: k must be [1, S, KV*D]"
    );
    assert_eq!(
        v_dims,
        vec![1, n_k, n_kv * head_dim],
        "NPU attention: v must be [1, S, KV*D]"
    );
    if !matches!(mode, MaskMode::WindowBlock { .. }) {
        assert_eq!(
            n_k, n,
            "the full-sequence mask modes need equal query/key lengths"
        );
    }

    let qf: Vec<f32> = q
        .into_data()
        .try_to_vec()
        .expect("the NPU path needs f32 activations");
    let kf: Vec<f32> = k
        .into_data()
        .try_to_vec()
        .expect("the NPU path needs f32 activations");
    let vf: Vec<f32> = v
        .into_data()
        .try_to_vec()
        .expect("the NPU path needs f32 activations");

    let out = with_engine(|e| {
        let recreate = match &e.fa {
            Some(s) => s.n_head != n_head || s.n_kv != n_kv || s.head_dim != head_dim,
            None => true,
        };
        if recreate {
            let fa = RocketFaCtx::new(e.threads)
                .unwrap_or_else(|err| op_failure("rocket_fa_ctx_create", err.rc, n, n_k, head_dim));
            e.fa = Some(FaState {
                fa,
                n_head,
                n_kv,
                head_dim,
                n: 0,
                n_k: 0,
                mask: Vec::new(),
                win: usize::MAX,
                win_mask: Vec::new(),
                cw: usize::MAX,
                cw_mask: Vec::new(),
                wb: None,
                wb_mask: Vec::new(),
                q: Vec::new(),
                k: Vec::new(),
                v: Vec::new(),
                out: Vec::new(),
            });
        }
        let st = e.fa.as_mut().unwrap();
        if st.n != n || st.n_k != n_k {
            st.n = n;
            st.n_k = n_k;
            st.mask = build_causal_mask(n);
            st.win = usize::MAX;
            st.win_mask.clear();
            st.cw = usize::MAX;
            st.cw_mask.clear();
            st.wb = None;
            st.wb_mask.clear();
            st.q = vec![f16::ZERO; n_head * n * head_dim];
            st.k = vec![f16::ZERO; n_kv * n_k * head_dim];
            st.v = vec![f16::ZERO; n_kv * head_dim * n_k];
            st.out = vec![f16::ZERO; n_head * n * head_dim];
        }

        let t_conv = Instant::now();
        if let MaskMode::Window(w) = mode
            && st.win != w
        {
            st.win_mask = build_window_mask(n, w);
            st.win = w;
        }
        if let MaskMode::CausalWindow(w) = mode
            && st.cw != w
        {
            st.cw_mask = build_causal_window_mask(n, w);
            st.cw = w;
        }
        if let MaskMode::WindowBlock { window, q0, k0 } = mode {
            // The block mask depends only on the relative offset (translation
            // invariant), so a chunk run rebuilds it only when the key length or
            // offset changes (first / interior / last chunk).
            let key = (n, n_k, window, q0 - k0);
            if st.wb != Some(key) {
                st.wb_mask = crate::masks::build_window_mask_block(n, n_k, q0, k0, window);
                st.wb = Some(key);
            }
        }
        let FaState {
            fa,
            mask,
            win_mask,
            cw_mask,
            wb_mask,
            q,
            k,
            v,
            out,
            ..
        } = st;
        fill_heads(&qf, n, n_head, head_dim, q);
        fill_heads(&kf, n_k, n_kv, head_dim, k);
        fill_heads_transposed(&vf, n_k, n_kv, head_dim, v);
        T_CONVERT_US.fetch_add(t_conv.elapsed().as_micros() as u64, Ordering::Relaxed);

        let t_npu = Instant::now();
        let mask = match mode {
            MaskMode::None => None,
            MaskMode::Causal => Some(&mask[..]),
            MaskMode::Window(_) => Some(&win_mask[..]),
            MaskMode::CausalWindow(_) => Some(&cw_mask[..]),
            MaskMode::WindowBlock { .. } => Some(&wb_mask[..]),
        };
        fa.flash_attn(
            n,
            n_k,
            head_dim,
            head_dim,
            n_head,
            n_kv,
            scale as f32,
            softcap.unwrap_or(0.0),
            q,
            k,
            v,
            mask,
            out,
        )
        .unwrap_or_else(|err| op_failure("rocket_flash_attn_fp16_ctx", err.rc, n, n_k, head_dim));
        T_NPU_US.fetch_add(t_npu.elapsed().as_micros() as u64, Ordering::Relaxed);
        NPU_CALLS.fetch_add(1, Ordering::Relaxed);

        let t_conv = Instant::now();
        let o = unpack_heads(out, n, n_head, head_dim);
        T_CONVERT_US.fetch_add(t_conv.elapsed().as_micros() as u64, Ordering::Relaxed);
        o
    });

    FlexTensor::from_data(TensorData::new(out, q_dims))
}

// ---------------------------------------------------------------------------
// High-level helpers: ordinary `Tensor`s in and out.
// ---------------------------------------------------------------------------

/// Pack a `[N, K]` f32 weight into resident NPU memory; the host tensor is
/// consumed and its memory dropped (`pack-and-drop`).
pub fn pack<const D: usize>(w: Tensor<D>) -> WeightId {
    WeightId(<Dispatch as RocketOps>::rocket_pack(w.into_dispatch()))
}

/// Pack two weights sharing one input activation, concatenated along N.
pub fn pack2(a: Tensor<2>, b: Tensor<2>) -> WeightId {
    WeightId(<Dispatch as RocketOps>::rocket_pack2(
        a.into_dispatch(),
        b.into_dispatch(),
    ))
}

/// Pack three weights sharing one input activation, concatenated along N.
pub fn pack3(a: Tensor<2>, b: Tensor<2>, c: Tensor<2>) -> WeightId {
    WeightId(<Dispatch as RocketOps>::rocket_pack3(
        a.into_dispatch(),
        b.into_dispatch(),
        c.into_dispatch(),
    ))
}

/// Pack a `[N, K]` f32 weight into resident int8 NPU memory (W8A8): the weight
/// is quantized host-side (symmetric int8, one scale per `group`-wide K block,
/// `group % 32 == 0`) and the codes are scattered into NPU BOs once; `matmul`
/// routes to the int8 path for weights packed this way. The host tensor is
/// consumed and its memory dropped.
pub fn pack_i8(w: Tensor<2>, group: usize) -> WeightId {
    WeightId(<Dispatch as RocketOps>::rocket_pack_i8(
        w.into_dispatch(),
        group,
    ))
}

/// `x[.., M, K] @ w[N, K]^T -> [.., M, N]` on the NPU.
pub fn matmul<const D: usize>(x: Tensor<D>, w: &WeightId) -> Tensor<D> {
    Tensor::from_dispatch(<Dispatch as RocketOps>::rocket_matmul(
        x.into_dispatch(),
        w.0,
    ))
}

/// Masked grouped-query attention on the NPU, for `[1, S, H*D]` queries and
/// `[1, S, KV*D]` keys/values.
#[allow(clippy::too_many_arguments)]
pub fn attention<const D: usize>(
    q: Tensor<D>,
    k: Tensor<D>,
    v: Tensor<D>,
    n_head: usize,
    n_kv_heads: usize,
    head_dim: usize,
    scale: f64,
    softcap: Option<f32>,
    is_causal: bool,
) -> Tensor<D> {
    Tensor::from_dispatch(<Dispatch as RocketOps>::rocket_attention(
        q.into_dispatch(),
        k.into_dispatch(),
        v.into_dispatch(),
        n_head,
        n_kv_heads,
        head_dim,
        scale,
        softcap,
        is_causal,
    ))
}

/// Bidirectional windowed grouped-query attention on the NPU: position `t`
/// attends to `j` iff `|t - j| <= window` (`window < 0` = no mask). Same input
/// layout as [`attention`].
#[allow(clippy::too_many_arguments)]
pub fn attention_window<const D: usize>(
    q: Tensor<D>,
    k: Tensor<D>,
    v: Tensor<D>,
    n_head: usize,
    n_kv_heads: usize,
    head_dim: usize,
    scale: f64,
    softcap: Option<f32>,
    window: i64,
) -> Tensor<D> {
    Tensor::from_dispatch(<Dispatch as RocketOps>::rocket_attention_window(
        q.into_dispatch(),
        k.into_dispatch(),
        v.into_dispatch(),
        n_head,
        n_kv_heads,
        head_dim,
        scale,
        softcap,
        window,
    ))
}

/// Causal sliding-window grouped-query attention on the NPU (generation
/// prefill): position `t` attends to `j` iff `t - window < j <= t`
/// (`window < 0` = plain causal). Same input layout as [`attention`].
#[allow(clippy::too_many_arguments)]
pub fn attention_causal_window<const D: usize>(
    q: Tensor<D>,
    k: Tensor<D>,
    v: Tensor<D>,
    n_head: usize,
    n_kv_heads: usize,
    head_dim: usize,
    scale: f64,
    softcap: Option<f32>,
    window: i64,
) -> Tensor<D> {
    Tensor::from_dispatch(<Dispatch as RocketOps>::rocket_attention_causal_window(
        q.into_dispatch(),
        k.into_dispatch(),
        v.into_dispatch(),
        n_head,
        n_kv_heads,
        head_dim,
        scale,
        softcap,
        window,
    ))
}

/// Chunked bidirectional band grouped-query attention on the NPU: `q` is
/// `[1, n_q, H*D]`, `k`/`v` are `[1, n_kv_len, KV*D]`, and query `q_start + i`
/// attends to key `kv_start + j` iff `|(q_start + i) - (kv_start + j)| <= window`
/// (`window < 0` = no mask). Lets a sliding layer attend a query chunk against
/// only the keys near it; same input layout as [`attention`].
#[allow(clippy::too_many_arguments)]
pub fn attention_window_block<const D: usize>(
    q: Tensor<D>,
    k: Tensor<D>,
    v: Tensor<D>,
    n_head: usize,
    n_kv_heads: usize,
    head_dim: usize,
    scale: f64,
    softcap: Option<f32>,
    window: i64,
    q_start: i64,
    kv_start: i64,
) -> Tensor<D> {
    Tensor::from_dispatch(<Dispatch as RocketOps>::rocket_attention_window_block(
        q.into_dispatch(),
        k.into_dispatch(),
        v.into_dispatch(),
        n_head,
        n_kv_heads,
        head_dim,
        scale,
        softcap,
        window,
        q_start,
        kv_start,
    ))
}

/// Fused `gelu_approximate(a) * b` (one pass; the NPU build's CPU glue).
pub fn gelu_mul<const D: usize>(a: Tensor<D>, b: Tensor<D>) -> Tensor<D> {
    Tensor::from_dispatch(<Dispatch as RocketOps>::rocket_gelu_mul(
        a.into_dispatch(),
        b.into_dispatch(),
    ))
}

/// Fused weighted RMSNorm over the last dim
/// (`x * (mean(x^2) + eps)^-0.5 * w`; `w` has the last dim's size).
pub fn rms_norm<const D: usize>(x: Tensor<D>, w: Tensor<1>, eps: f64) -> Tensor<D> {
    Tensor::from_dispatch(<Dispatch as RocketOps>::rocket_rms_norm(
        x.into_dispatch(),
        w.into_dispatch(),
        eps,
    ))
}

/// Fused scale-free RMSNorm over the last dim (`x * (mean(x^2) + eps)^-0.5`).
pub fn rms_norm_noscale<const D: usize>(x: Tensor<D>, eps: f64) -> Tensor<D> {
    Tensor::from_dispatch(<Dispatch as RocketOps>::rocket_rms_norm_noscale(
        x.into_dispatch(),
        eps,
    ))
}

/// Fused rotate-half RoPE: `x` is `[B, S, H, D]`, `cos`/`sin` are `[S, D/2]`.
pub fn rope_apply(x: Tensor<4>, cos: Tensor<2>, sin: Tensor<2>) -> Tensor<4> {
    Tensor::from_dispatch(<Dispatch as RocketOps>::rocket_rope(
        x.into_dispatch(),
        cos.into_dispatch(),
        sin.into_dispatch(),
    ))
}

// ---------------------------------------------------------------------------
// Conversions and layouts (rayon-parallel)
// ---------------------------------------------------------------------------

/// Parallel f32 -> f16 conversion (chunked to keep it SIMD-friendly).
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

/// `src` is `[n, heads*d]` row-major; writes `dst[head][n][d]`.
fn fill_heads(src: &[f32], n: usize, heads: usize, d: usize, dst: &mut [f16]) {
    use rayon::prelude::*;
    let hd = heads * d;
    dst.par_chunks_mut(n * d)
        .enumerate()
        .for_each(|(hi, dst_h)| {
            for (s, dst_row) in dst_h.chunks_mut(d).enumerate() {
                let src_row = &src[s * hd + hi * d..s * hd + hi * d + d];
                for (o, &x) in dst_row.iter_mut().zip(src_row) {
                    *o = f16::from_f32(x);
                }
            }
        });
}

/// `src` is `[n, heads*d]` row-major; writes the per-head transpose
/// `dst[head][d][n]` (the AV B-operand layout).
fn fill_heads_transposed(src: &[f32], n: usize, heads: usize, d: usize, dst: &mut [f16]) {
    use rayon::prelude::*;
    let hd = heads * d;
    dst.par_chunks_mut(d * n)
        .enumerate()
        .for_each(|(hi, dst_h)| {
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
