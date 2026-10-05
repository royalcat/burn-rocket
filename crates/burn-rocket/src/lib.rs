//! RK3588 NPU offload for Burn models, through the mainline `rocket` driver.
//!
//! This crate is a thin safe wrapper around `librocketnpu`
//! (<https://github.com/gregordinary/rocket-userspace>), the userspace driver for
//! the mainline `rocket` DRM-accel driver. It exposes the fp16 resident-weight
//! matmul (`C[M,N] = A[M,K] * B[N,K]^T`) that Burn models use to offload their
//! projections; everything else stays on the CPU backend.
//!
//! aarch64-only: it links `librocketnpu.a` (see `build.rs` and `ROCKETNPU_DIR`).
//!
//! Handles are **not** `Send`/`Sync`: `rocket_ctx` mutates shared per-shape
//! scratch every call and is documented as not thread-safe. Use one context per
//! concurrent caller.
//!
//! ```
//! # #[cfg(target_arch = "aarch64")] {
//! use burn_rocket::RocketCtx;
//! let ctx = RocketCtx::new(5).unwrap();
//! # }
//! ```

pub mod ffi;

pub use half;
use half::f16;
use std::ffi::CStr;
use std::sync::Arc;
/// Error from a librocketnpu call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Error {
    pub op: &'static str,
    pub rc: i32,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = match self.rc {
            ffi::ROCKET_E_SHAPE => "unsupported shape",
            ffi::ROCKET_E_TILING => "shape not tileable",
            ffi::ROCKET_E_NOMEM => "host allocation failed",
            ffi::ROCKET_E_DEVICE => "device/driver failure",
            ffi::ROCKET_E_UNSUPPORTED => "unsupported feature",
            _ => "unknown error",
        };
        write!(f, "{} failed: {} (rc={})", self.op, kind, self.rc)
    }
}

impl std::error::Error for Error {}

impl Error {
    /// True when the caller should fall back to a different path (`ROCKET_E_TILING`).
    pub fn is_fallback(&self) -> bool {
        self.rc == ffi::ROCKET_E_TILING
    }
}

fn check(op: &'static str, rc: i32) -> Result<(), Error> {
    if rc == ffi::ROCKET_OK {
        Ok(())
    } else {
        Err(Error { op, rc })
    }
}

/// Device/driver information.
pub fn driver_name() -> Option<String> {
    unsafe {
        let p = ffi::rocket_driver_name();
        if p.is_null() {
            return None;
        }
        Some(CStr::from_ptr(p).to_string_lossy().into_owned())
    }
}

/// Number of big (A76) cores available for worker pinning, 0 if not applicable.
pub fn num_big_cores() -> i32 {
    unsafe { ffi::rocket_num_big_cores() }
}

/// `(submit_ioctls, submitted_tasks)` since the last reset.
pub fn submit_counters() -> (u64, u64) {
    unsafe {
        (
            ffi::rocket_submit_ioctl_count(),
            ffi::rocket_submit_task_count(),
        )
    }
}

/// Reset the submit counters.
pub fn reset_submit_counters() {
    unsafe { ffi::rocket_submit_counters_reset() }
}

/// Spin-poll the completion fence for up to `us` microseconds before blocking.
pub fn set_busy_poll_us(us: i64) {
    unsafe { ffi::rocket_busy_poll_set_us(us as libc::c_long) }
}

struct CtxInner {
    ctx: *mut ffi::RocketCtxOpaque,
    fd: i32,
}

// The context owns worker fds and per-shape scratch; librocketnpu documents it as
// not thread-safe. `CtxInner` is therefore deliberately neither Send nor Sync.

impl Drop for CtxInner {
    fn drop(&mut self) {
        unsafe {
            ffi::rocket_ctx_free(self.ctx);
            ffi::rocket_close(self.fd);
        }
    }
}

/// A persistent NPU context: worker fds plus per-shape scratch.
///
/// Not `Send`/`Sync` — give each concurrent caller its own context. The per-call
/// fan-out across the three NPU cores happens inside one context.
pub struct RocketCtx {
    inner: Arc<CtxInner>,
}

impl RocketCtx {
    /// Open the device and create a context with `nthreads` worker threads/fds.
    pub fn new(nthreads: usize) -> Result<Self, Error> {
        let fd = unsafe { ffi::rocket_open() };
        if fd < 0 {
            return Err(Error {
                op: "rocket_open",
                rc: fd,
            });
        }
        let ctx = unsafe { ffi::rocket_ctx_create(nthreads as i32) };
        if ctx.is_null() {
            unsafe { ffi::rocket_close(fd) };
            return Err(Error {
                op: "rocket_ctx_create",
                rc: ffi::ROCKET_E_DEVICE,
            });
        }
        Ok(Self {
            inner: Arc::new(CtxInner { ctx, fd }),
        })
    }

    /// Pack a weight matrix `B` of shape `[N, K]` (row-major fp16) into resident
    /// NPU buffers, once. `m` is the warm-up row count used for tiling; the
    /// resulting handle can be reused for any `M >= 256` with the same `K`/`N`.
    pub fn pack_weight(
        &self,
        m: usize,
        k: usize,
        n: usize,
        b: &[f16],
    ) -> Result<RocketWeight, Error> {
        assert_eq!(b.len(), n * k, "weight buffer must be [N, K]");
        let w = unsafe {
            ffi::rocket_weights_pack(
                self.inner.ctx,
                m as i32,
                k as i32,
                n as i32,
                b.as_ptr() as *const ffi::F16,
            )
        };
        if w.is_null() {
            return Err(Error {
                op: "rocket_weights_pack",
                rc: ffi::ROCKET_E_SHAPE,
            });
        }
        Ok(RocketWeight {
            w,
            ctx: self.inner.clone(),
        })
    }

    /// Pack several weights that share one input activation into a single resident
    /// weight, concatenated along N: `C[:, sum(Ns)] = A * [B0; B1; ...]^T`.
    /// Each `parts[i]` is `[Ns[i], K]` row-major fp16; the total `N` is their sum.
    pub fn pack_weight_seg(
        &self,
        m: usize,
        k: usize,
        parts: &[&[f16]],
    ) -> Result<RocketWeight, Error> {
        let ns: Vec<i32> = parts.iter().map(|p| (p.len() / k) as i32).collect();
        let n: usize = ns.iter().map(|&x| x as usize).sum();
        let ptrs: Vec<*const ffi::F16> = parts
            .iter()
            .map(|p| {
                assert_eq!(p.len() % k, 0, "each segment must be [Ns, K]");
                p.as_ptr() as *const ffi::F16
            })
            .collect();
        let w = unsafe {
            ffi::rocket_weights_pack_seg(
                self.inner.ctx,
                m as i32,
                k as i32,
                n as i32,
                ptrs.as_ptr(),
                ns.as_ptr(),
                parts.len() as i32,
            )
        };
        if w.is_null() {
            return Err(Error {
                op: "rocket_weights_pack_seg",
                rc: ffi::ROCKET_E_SHAPE,
            });
        }
        Ok(RocketWeight {
            w,
            ctx: self.inner.clone(),
        })
    }

    /// `C[M, N] = A[M, K] * B[N, K]^T` with a resident weight.
    ///
    /// `a` is `[M, K]` and `c` is `[M, N]`, both row-major fp16. `m` may differ
    /// from the `m` used at pack time for `M >= 256`.
    pub fn matmul_prepacked(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f16],
        c: &mut [f16],
        w: &RocketWeight,
    ) -> Result<(), Error> {
        assert_eq!(a.len(), m * k, "A must be [M, K]");
        assert_eq!(c.len(), m * n, "C must be [M, N]");
        let rc = unsafe {
            ffi::rocket_matmul_fp16_prepacked(
                self.inner.ctx,
                m as i32,
                k as i32,
                n as i32,
                a.as_ptr() as *const ffi::F16,
                c.as_mut_ptr() as *mut ffi::F16,
                w.w,
            )
        };
        check("rocket_matmul_fp16_prepacked", rc)
    }

    /// Create a streaming context (weights re-packed on every call) bound to a
    /// fresh device fd. Useful when weights do not fit in resident memory.
    pub fn stream(&self, nthreads: usize) -> Result<RocketStream, Error> {
        let s = unsafe { ffi::rocket_stream_create(nthreads as i32) };
        if s.is_null() {
            return Err(Error {
                op: "rocket_stream_create",
                rc: ffi::ROCKET_E_DEVICE,
            });
        }
        Ok(RocketStream { s })
    }
}

/// A weight resident in NPU buffers, packed from a `[N, K]` row-major fp16 matrix.
pub struct RocketWeight {
    w: *mut ffi::RocketWeightsOpaque,
    ctx: Arc<CtxInner>,
}

impl Drop for RocketWeight {
    fn drop(&mut self) {
        unsafe { ffi::rocket_weights_free(self.ctx.ctx, self.w) }
    }
}

/// Streaming matmul context: keeps scratch/worker BOs but re-packs `B` per call.
pub struct RocketStream {
    s: *mut ffi::RocketStreamOpaque,
}

impl RocketStream {
    /// `C[M, N] = A[M, K] * B[N, K]^T`, re-packing `B` every call.
    pub fn matmul(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f16],
        b: &[f16],
        c: &mut [f16],
    ) -> Result<(), Error> {
        assert_eq!(a.len(), m * k, "A must be [M, K]");
        assert_eq!(b.len(), n * k, "B must be [N, K]");
        assert_eq!(c.len(), m * n, "C must be [M, N]");
        let rc = unsafe {
            ffi::rocket_matmul_fp16_stream(
                self.s,
                m as i32,
                k as i32,
                n as i32,
                a.as_ptr() as *const ffi::F16,
                b.as_ptr() as *const ffi::F16,
                c.as_mut_ptr() as *mut ffi::F16,
            )
        };
        check("rocket_matmul_fp16_stream", rc)
    }
}

impl Drop for RocketStream {
    fn drop(&mut self) {
        unsafe { ffi::rocket_stream_free(self.s) }
    }
}

/// Persistent context for masked grouped-query attention on the NPU
/// (`rocket_flash_attn_fp16_ctx`): worker fds and per-head scratch stay resident
/// across calls. Not `Send`/`Sync`.
pub struct RocketFaCtx {
    c: *mut ffi::RocketFaCtxOpaque,
}

impl RocketFaCtx {
    /// Create with `nthreads` workers (clamped by the library to `[1, min(8, n_head)]`).
    pub fn new(nthreads: usize) -> Result<Self, Error> {
        let c = unsafe { ffi::rocket_fa_ctx_create(nthreads as i32) };
        if c.is_null() {
            return Err(Error {
                op: "rocket_fa_ctx_create",
                rc: ffi::ROCKET_E_DEVICE,
            });
        }
        Ok(Self { c })
    }

    /// Masked GQA attention: `softmax(scale * Q K^T + mask) V`.
    ///
    /// Layouts (fp16, head-major):
    /// - `q`: `[n_head][n_tokens][head_dim]`
    /// - `k`: `[n_kv_heads][n_kv][head_dim]`
    /// - `v`: `[n_kv_heads][dv][n_kv]` (per-head transposed: the AV B-operand)
    /// - `mask`: `[n_tokens][n_kv]` additive (`-inf` masks), or `None` for unmasked
    /// - `out`: `[n_head][n_tokens][dv]`
    ///
    /// `n_head` must be a multiple of `n_kv_heads`, `head_dim % 32 == 0`, `dv % 16 == 0`.
    #[allow(clippy::too_many_arguments)]
    pub fn flash_attn(
        &self,
        n_tokens: usize,
        n_kv: usize,
        head_dim: usize,
        dv: usize,
        n_head: usize,
        n_kv_heads: usize,
        scale: f32,
        q: &[f16],
        k: &[f16],
        v: &[f16],
        mask: Option<&[f16]>,
        out: &mut [f16],
    ) -> Result<(), Error> {
        assert_eq!(q.len(), n_head * n_tokens * head_dim);
        assert_eq!(k.len(), n_kv_heads * n_kv * head_dim);
        assert_eq!(v.len(), n_kv_heads * dv * n_kv);
        assert_eq!(out.len(), n_head * n_tokens * dv);
        if let Some(m) = mask {
            assert_eq!(m.len(), n_tokens * n_kv);
        }
        let rc = unsafe {
            ffi::rocket_flash_attn_fp16_ctx(
                self.c,
                n_tokens as i32,
                n_kv as i32,
                head_dim as i32,
                dv as i32,
                n_head as i32,
                n_kv_heads as i32,
                scale,
                0.0,
                q.as_ptr() as *const ffi::F16,
                k.as_ptr() as *const ffi::F16,
                v.as_ptr() as *const ffi::F16,
                mask.map_or(std::ptr::null(), |m| m.as_ptr() as *const ffi::F16),
                out.as_mut_ptr() as *mut ffi::F16,
            )
        };
        check("rocket_flash_attn_fp16_ctx", rc)
    }
}

impl Drop for RocketFaCtx {
    fn drop(&mut self) {
        unsafe { ffi::rocket_fa_ctx_free(self.c) }
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
