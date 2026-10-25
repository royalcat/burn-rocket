//! RK3588 NPU offload for Burn models, through the mainline `rocket` driver.
//!
//! The userspace driver is the pure-Rust `rocket-npu` crate; this crate keeps the
//! Burn-facing surface: the [`RocketOps`](ext::RocketOps) backend extension, the
//! resident-weight registry, the fused CPU glue, and the host-built attention masks.
//! Everything NPU-specific (device, register programs, matmul, attention) lives in
//! `rocket-npu` and needs no C toolchain or kernel headers.
//!
//! On the `Flex` backend the operations are exposed as Burn **backend extensions**
//! (`crate::ext`, feature `npu`): the [`ext::RocketOps`] trait is routed through
//! Burn's `Dispatch`, and the ordinary-`Tensor` helpers
//! [`pack`](ext::pack), [`pack2`](ext::pack2), [`pack3`](ext::pack3),
//! [`matmul`](ext::matmul) and [`attention`](ext::attention) wrap the dispatch
//! plumbing. Call [`ext::init`] once before using them.
//!
//! The extension ops are inference-only and serialize through one global engine.

/// The pure-Rust NPU driver this crate is built on.
#[cfg(feature = "npu")]
pub use rocket_npu as npu;

/// Fused host-side elementwise kernels for the NPU build's CPU glue
/// (unit tested on any host; the callers live in the `npu`-gated extension).
#[cfg_attr(not(feature = "npu"), allow(dead_code))]
pub(crate) mod host;

/// Host-side additive attention masks (unit tested on any host).
pub(crate) mod masks;

/// Burn backend-extension ops (`#[backend_extension(Flex)]`); NPU builds only.
#[cfg(feature = "npu")]
pub mod ext;

#[cfg(feature = "npu")]
pub use ext::{
    Stats, WeightId, attention, attention_causal_window, attention_window, attention_window_block,
    bf16_stream_mode, gelu_mul, init, matmul, pack, pack_i8, pack2, pack3, rms_norm,
    rms_norm_noscale, rope_apply, silu_mul, stats, stats_reset,
};
#[cfg(feature = "npu")]
pub use ext::{clear_scratch, mem_stats, trim_scratch};

/// One entry of the matmul per-shape scratch cache.
#[derive(Debug, Clone, Copy, Default)]
pub struct SlotMem {
    /// Capacity class height (the slot's `M`); calls with a smaller live `M` reuse it.
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub bo_bytes: usize,
    pub host_bytes: usize,
}

/// One attention worker's retained scratch.
#[derive(Debug, Clone, Copy, Default)]
pub struct AttnMem {
    /// Padded `(query, key)` shape the scratch is sized for.
    pub tp: usize,
    pub kn: usize,
    /// The `ROCKET_FA_TILE_KV` key-tile width this worker last used; 0 = the
    /// materialized (full-key-axis) path.
    pub tile: usize,
    pub bo_bytes: usize,
    pub host_bytes: usize,
}

/// Live NPU scratch accounting for the global engine.
///
/// `mem_stats` is `None` on builds without the `npu` feature (and before
/// `init`), so callers can log it unconditionally from shared server code.
#[derive(Debug, Clone, Default)]
pub struct MemStats {
    /// Resident weight BOs (the packed projections).
    pub weights_bytes: usize,
    pub matmul_bo_bytes: usize,
    pub matmul_host_bytes: usize,
    pub attention_bo_bytes: usize,
    pub attention_host_bytes: usize,
    pub slots: Vec<SlotMem>,
    pub attn_workers: Vec<AttnMem>,
}

#[cfg(not(feature = "npu"))]
pub fn mem_stats() -> Option<MemStats> {
    None
}

#[cfg(not(feature = "npu"))]
pub fn trim_scratch() {}

#[cfg(not(feature = "npu"))]
pub fn clear_scratch() {}

pub use half;
use half::f16;
use std::sync::OnceLock;

/// Canonical tiling for resident weights (default **on**): the NPU plans every
/// packed weight at the canonical mature tile, so one pack serves *any* M down
/// to 4 and small inputs are computed at their true M instead of being padded
/// up to the M >= 256 tiling. `ROCKET_CTX_CANONICAL=0` restores the legacy
/// behavior (small M padded to 256) for A/B runs without a rebuild.
pub fn canonical_tiling() -> bool {
    static ONCE: OnceLock<bool> = OnceLock::new();
    *ONCE.get_or_init(|| {
        std::env::var("ROCKET_CTX_CANONICAL")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

/// Error from an NPU operation, with the same negative status codes the old C
/// library used (`-1` shape, `-2` tiling, `-3` NOMEM, `-4` device, `-5`
/// unsupported).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Error {
    pub op: &'static str,
    pub rc: i32,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = match self.rc {
            -1 => "unsupported shape",
            -2 => "shape not tileable",
            -3 => "host allocation failed",
            -4 => "device/driver failure",
            -5 => "unsupported feature",
            _ => "unknown error",
        };
        write!(f, "{} failed: {} (rc={})", self.op, kind, self.rc)
    }
}

impl std::error::Error for Error {}

impl Error {
    /// True when the caller should fall back to a different path (`ROCKET_E_TILING`).
    pub fn is_fallback(&self) -> bool {
        self.rc == -2
    }
}

#[cfg(feature = "npu")]
impl From<rocket_npu::Error> for Error {
    fn from(value: rocket_npu::Error) -> Self {
        Error {
            op: value.op(),
            rc: value.status().code(),
        }
    }
}

/// Panic payload of a failed extension op: the raw [`Error`] plus the shape the
/// call used, so a host (e.g. the serving layer) can map failures to responses
/// instead of parsing panic strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpFailure {
    pub error: Error,
    pub m: usize,
    pub k: usize,
    pub n: usize,
}

/// Device/driver information: `Some("rocket")` when a device opens.
#[cfg(feature = "npu")]
pub fn driver_name() -> Option<String> {
    rocket_npu::Device::open()
        .ok()
        .map(|d| d.driver_name().to_string())
}

/// Number of big (A76) cores available for worker pinning, 0 if not applicable.
#[cfg(feature = "npu")]
pub fn num_big_cores() -> i32 {
    rocket_npu::device::affinity::num_big_cores() as i32
}

/// `(submit_ioctls, submitted_tasks)` since the last reset.
#[cfg(feature = "npu")]
pub fn submit_counters() -> (u64, u64) {
    (
        rocket_npu::device::submit::ioctl_submit_count(),
        rocket_npu::device::submit::submitted_task_count(),
    )
}

/// Reset the submit counters.
#[cfg(feature = "npu")]
pub fn reset_submit_counters() {
    rocket_npu::device::submit::reset_counters();
}

/// Compatibility shim: the busy-poll budget is resolved once from
/// `ROCKET_BUSY_POLL`; a post-init override is not supported by the port.
#[cfg(feature = "npu")]
pub fn set_busy_poll_us(_us: i64) {}

/// A persistent NPU context: worker fds plus per-shape scratch.
///
/// One context must not be used from several threads at once; the extension
/// engine serializes every access.
#[cfg(feature = "npu")]
pub struct RocketCtx {
    inner: rocket_npu::matmul::Context,
}

#[cfg(feature = "npu")]
impl RocketCtx {
    /// Open the device and create a context with `nthreads` worker fds.
    pub fn new(nthreads: usize) -> Result<Self, Error> {
        let inner = rocket_npu::matmul::Context::new(nthreads)?;
        Ok(Self { inner })
    }

    /// Pack a weight matrix `B` of shape `[N, K]` (row-major fp16) into resident
    /// NPU buffers, once. With the canonical tiling the handle serves any
    /// `M >= 4` with the same `K`/`N`.
    pub fn pack_weight(
        &mut self,
        m: usize,
        k: usize,
        n: usize,
        b: &[f16],
    ) -> Result<RocketWeight, Error> {
        Ok(self.inner.pack_weight(m, k, n, b)?)
    }

    /// Pack several weights that share one input, concatenated along N.
    pub fn pack_weight_seg(
        &mut self,
        m: usize,
        k: usize,
        parts: &[&[f16]],
    ) -> Result<RocketWeight, Error> {
        Ok(self.inner.pack_weight_seg(m, k, parts)?)
    }

    /// `C[M, N] = A[M, K] * B[N, K]^T` with a resident weight.
    pub fn matmul_prepacked(
        &mut self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f16],
        c: &mut [f16],
        w: &RocketWeight,
    ) -> Result<(), Error> {
        Ok(self.inner.matmul_prepacked(m, k, n, a, c, w)?)
    }

    /// Keep only the matmul shape set touched since the previous call.
    pub fn trim_scratch(&mut self) {
        self.inner.trim_scratch();
    }

    /// Drop every cached matmul shape set.
    pub fn clear_scratch(&mut self) {
        self.inner.clear_scratch();
    }

    /// Live per-shape scratch held by this context.
    pub fn mem_stats(&self) -> rocket_npu::matmul::MemStats {
        self.inner.mem_stats()
    }
}

/// A weight resident in NPU buffers.
#[cfg(feature = "npu")]
pub type RocketWeight = rocket_npu::matmul::Weight;

/// Persistent masked-GQA attention context (`rocket_flash_attn_fp16_ctx`).
#[cfg(feature = "npu")]
pub struct RocketFaCtx {
    inner: rocket_npu::attn::Attention,
}

#[cfg(feature = "npu")]
impl RocketFaCtx {
    pub fn new(nthreads: usize) -> Result<Self, Error> {
        Ok(Self {
            inner: rocket_npu::attn::Attention::new(nthreads)?,
        })
    }

    /// Masked GQA attention (fp16, head-major layouts); see
    /// [`rocket_npu::attn::Attention::flash`].
    #[allow(clippy::too_many_arguments)]
    pub fn flash_attn(
        &mut self,
        n_tokens: usize,
        n_kv: usize,
        head_dim: usize,
        dv: usize,
        n_head: usize,
        n_kv_heads: usize,
        scale: f32,
        softcap: f32,
        q: &[f16],
        k: &[f16],
        v: &[f16],
        mask: Option<&[f16]>,
        out: &mut [f16],
    ) -> Result<(), Error> {
        Ok(self.inner.flash(
            n_tokens, n_kv, head_dim, dv, n_head, n_kv_heads, scale, softcap, q, k, v, mask, out,
        )?)
    }

    /// Live per-worker attention scratch.
    pub fn mem_stats(&self) -> rocket_npu::attn::MemStats {
        self.inner.mem_stats()
    }

    /// Drop scratch classes from an earlier, much larger request.
    pub fn trim_scratch(&mut self) {
        self.inner.trim_scratch();
    }
}

// ---------------------------------------------------------------------------
// int8 and native bf16 stream
//
// The wrappers below expose the Rust `rocket-npu` resident group-wise int8 and
// bf16-stream contexts on the same surface the C library used.
// ---------------------------------------------------------------------------

/// Resident group-wise int8 context (`rocket_i8_ctx` in the C API).
#[cfg(feature = "npu")]
pub struct RocketI8Ctx {
    inner: rocket_npu::matmul::I8Context,
}

#[cfg(feature = "npu")]
impl RocketI8Ctx {
    pub fn new(nthreads: usize) -> Result<Self, Error> {
        Ok(Self {
            inner: rocket_npu::matmul::I8Context::new(nthreads)?,
        })
    }

    /// Pack a pre-quantized `[N, K]` int8 weight with K-groups of `group`.
    pub fn pack_weight_gw(
        &mut self,
        m: usize,
        k: usize,
        n: usize,
        b: &[i8],
        group: usize,
    ) -> Result<RocketI8Weight, Error> {
        Ok(self.inner.pack_weight_gw(m, k, n, b, group)?)
    }

    /// `C[m,n] = sum_g a_scale[m,g] * b_scale[n,g] * (A_g B_g^T)`, f32 output.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_gw(
        &mut self,
        m: usize,
        k: usize,
        n: usize,
        a: &[i8],
        a_scale: &[f32],
        b_scale: &[f32],
        cf: &mut [f32],
        w: &RocketI8Weight,
    ) -> Result<(), Error> {
        Ok(self.inner.matmul_gw(m, k, n, a, a_scale, b_scale, cf, w)?)
    }
}

/// A group-wise int8 weight resident in NPU buffers.
#[cfg(feature = "npu")]
pub type RocketI8Weight = rocket_npu::matmul::I8Weight;

/// Persistent native-bf16 stream: f32 in/out, bf16 device math, f64 host accumulation.
#[cfg(feature = "npu")]
pub struct RocketBf16Stream {
    inner: rocket_npu::matmul::Bf16Stream,
}

#[cfg(feature = "npu")]
impl RocketBf16Stream {
    pub fn new(nthreads: usize) -> Result<Self, Error> {
        Ok(Self {
            inner: rocket_npu::matmul::Bf16Stream::new(nthreads)?,
        })
    }

    /// Re-packs A and B every call; returns 0 or a negative status.
    pub fn matmul(
        &mut self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
    ) -> i32 {
        match self.inner.matmul(m, k, n, a, b, c) {
            Ok(()) => 0,
            Err(e) => e.status().code(),
        }
    }

    /// The per-call multi-core fallback (opens its own worker fds).
    pub fn matmul_mt(
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        nthreads: usize,
    ) -> Result<(), Error> {
        Ok(rocket_npu::matmul::Bf16Stream::matmul_mt(
            m, k, n, a, b, c, nthreads,
        )?)
    }
}

// ---------------------------------------------------------------------------
// f32 <-> f16 helpers
// ---------------------------------------------------------------------------

/// Convert an f32 slice to a fresh fp16 vector.
pub fn f32_to_f16(src: &[f32]) -> Vec<f16> {
    src.iter().map(|&x| f16::from_f32(x)).collect()
}

/// Convert an fp16 slice into `dst` as f32 (reusing the allocation).
pub fn f16_to_f32_into(src: &[f16], dst: &mut Vec<f32>) {
    dst.clear();
    dst.reserve(src.len());
    dst.extend(src.iter().map(|x| x.to_f32()));
}

/// Pad `[rows, k]` fp16 data to `padded_rows` rows by repeating the last row
/// (the padded rows are ignored by the caller; the NPU needs `M % 4 == 0`).
pub fn pad_rows(src: &[f16], rows: usize, k: usize, padded_rows: usize) -> Vec<f16> {
    debug_assert!(padded_rows >= rows);
    let mut out = Vec::with_capacity(padded_rows * k);
    out.extend_from_slice(src);
    if padded_rows > rows {
        let last = &src[(rows - 1) * k..rows * k];
        for _ in rows..padded_rows {
            out.extend_from_slice(last);
        }
    }
    out
}
