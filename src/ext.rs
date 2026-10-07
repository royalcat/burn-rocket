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
use crate::{Error, OpFailure, RocketCtx, RocketFaCtx, RocketWeight, pad_rows};

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

/// M warm-up row count at pack time; the resident weights are reused for any M
/// (small requests are padded up to 256 rows, see `matmul_impl`).
const PACK_M: usize = 512;

/// Padding floor for the M dimension: resident weights are packed for the
/// M >= 256 tiling and cannot serve smaller matmuls directly.
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

struct FaState {
    fa: RocketFaCtx,
    n_head: usize,
    n_kv: usize,
    head_dim: usize,
    n: usize,
    mask: Vec<f16>, // [n][n] additive: 0 for j<=t, -inf otherwise
    /// Cached bidirectional band mask for the current `n` (`usize::MAX` = none).
    win: usize,
    win_mask: Vec<f16>, // [n][n] additive: 0 for |t-j|<=win, -inf otherwise
    /// Cached causal sliding-window mask for the current `n`.
    cw: usize,
    cw_mask: Vec<f16>, // [n][n] additive: 0 for t-w<j<=t, -inf otherwise
    q: Vec<f16>,       // [n_head][n][head_dim]
    k: Vec<f16>,       // [n_kv][n][head_dim]
    v: Vec<f16>,       // [n_kv][head_dim][n]  (per-head transposed)
    out: Vec<f16>,     // [n_head][n][head_dim]
}

struct Engine {
    threads: usize,
    ctx: RocketCtx,
    fa: Option<FaState>,
    weights: HashMap<u64, Resident>,
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

    let t_conv = Instant::now();
    let a16 = f32_to_f16_par(&values);
    T_CONVERT_US.fetch_add(t_conv.elapsed().as_micros() as u64, Ordering::Relaxed);

    with_engine(|e| {
        let r = e
            .weights
            .get(&id)
            .unwrap_or_else(|| panic!("NPU weight id {id} is not packed"));
        assert_eq!(k, r.k, "activation K mismatch for NPU weight id {id}");
        let n = r.n;
        let chunk_m = matmul_chunk_m();

        // Rows are independent, so M is safe to split: full chunks are matmul'd
        // at their exact size (bounded NPU scratch), the tail keeps the old
        // small-request padding (extra rows are ignored on readback).
        let mut c32: Vec<f32> = Vec::with_capacity(m * n);
        let mut off = 0;
        while off < m {
            let rows = if chunk_m == 0 {
                m - off
            } else {
                (m - off).min(chunk_m)
            };
            let padded_m = (rows.div_ceil(4) * 4).max(MIN_M);
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
    assert_eq!(
        k_dims,
        vec![1, n, n_kv * head_dim],
        "NPU attention: k must be [1, S, KV*D]"
    );
    assert_eq!(
        v_dims,
        vec![1, n, n_kv * head_dim],
        "NPU attention: v must be [1, S, KV*D]"
    );

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
                .unwrap_or_else(|err| op_failure("rocket_fa_ctx_create", err.rc, n, n, head_dim));
            e.fa = Some(FaState {
                fa,
                n_head,
                n_kv,
                head_dim,
                n: 0,
                mask: Vec::new(),
                win: usize::MAX,
                win_mask: Vec::new(),
                cw: usize::MAX,
                cw_mask: Vec::new(),
                q: Vec::new(),
                k: Vec::new(),
                v: Vec::new(),
                out: Vec::new(),
            });
        }
        let st = e.fa.as_mut().unwrap();
        if st.n != n {
            st.n = n;
            st.mask = build_causal_mask(n);
            st.win = usize::MAX;
            st.win_mask.clear();
            st.cw = usize::MAX;
            st.cw_mask.clear();
            st.q = vec![f16::ZERO; n_head * n * head_dim];
            st.k = vec![f16::ZERO; n_kv * n * head_dim];
            st.v = vec![f16::ZERO; n_kv * head_dim * n];
            st.out = vec![f16::ZERO; n_head * n * head_dim];
        }

        let t_conv = Instant::now();
        if let MaskMode::Window(w) = mode {
            if st.win != w {
                st.win_mask = build_window_mask(n, w);
                st.win = w;
            }
        }
        if let MaskMode::CausalWindow(w) = mode {
            if st.cw != w {
                st.cw_mask = build_causal_window_mask(n, w);
                st.cw = w;
            }
        }
        let FaState {
            fa,
            mask,
            win_mask,
            cw_mask,
            q,
            k,
            v,
            out,
            ..
        } = st;
        fill_heads(&qf, n, n_head, head_dim, q);
        fill_heads(&kf, n, n_kv, head_dim, k);
        fill_heads_transposed(&vf, n, n_kv, head_dim, v);
        T_CONVERT_US.fetch_add(t_conv.elapsed().as_micros() as u64, Ordering::Relaxed);

        let t_npu = Instant::now();
        let mask = match mode {
            MaskMode::None => None,
            MaskMode::Causal => Some(&mask[..]),
            MaskMode::Window(_) => Some(&win_mask[..]),
            MaskMode::CausalWindow(_) => Some(&cw_mask[..]),
        };
        fa.flash_attn(
            n,
            n,
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
        .unwrap_or_else(|err| op_failure("rocket_flash_attn_fp16_ctx", err.rc, n, n, head_dim));
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
