//! Raw FFI declarations for `librocketnpu` (gregordinary/rocket-userspace).
//!
//! `_Float16` arrays are passed as raw pointers; the element type is `u16` here
//! (the bit pattern of the half-precision float), converted with the `half` crate.

#![allow(non_camel_case_types)]

use libc::{c_char, c_int, c_long, c_uint, c_void};

/// Element type of the `_Float16` buffers the library takes.
pub type F16 = u16;

/// A buffer object: NPU-visible memory that is also mmap-able on the CPU.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RocketBo {
    pub handle: u32,
    pub dma_address: u64,
    pub mmap_offset: u64,
    pub size: usize,
    pub ptr: *mut c_void,
    pub obj_addr: u64,
}

#[repr(C)]
pub struct RocketCtxOpaque {
    _private: [u8; 0],
}

#[repr(C)]
pub struct RocketWeightsOpaque {
    _private: [u8; 0],
}

#[repr(C)]
pub struct RocketStreamOpaque {
    _private: [u8; 0],
}

#[repr(C)]
pub struct RocketBf16StreamOpaque {
    _private: [u8; 0],
}

pub const ROCKET_OK: c_int = 0;
pub const ROCKET_E_SHAPE: c_int = -1;
pub const ROCKET_E_TILING: c_int = -2;
pub const ROCKET_E_NOMEM: c_int = -3;
pub const ROCKET_E_DEVICE: c_int = -4;
pub const ROCKET_E_UNSUPPORTED: c_int = -5;

/// `rocket_ctx_create_ex` flag: canonical tiling, M-independent down to M = 4
/// (one pack serves every M; small M no longer grows Kt / returns
/// `ROCKET_E_TILING`). See `rocket_matmul.h` in the library.
pub const ROCKET_CTX_TILING_CANONICAL: c_uint = 0x1;

unsafe extern "C" {
    // Device.
    pub fn rocket_open() -> c_int;
    pub fn rocket_close(fd: c_int);
    pub fn rocket_driver_name() -> *const c_char;
    pub fn rocket_num_big_cores() -> c_int;
    pub fn rocket_busy_poll_set_us(us: c_long);
    pub fn rocket_submit_ioctl_count() -> u64;
    pub fn rocket_submit_task_count() -> u64;
    pub fn rocket_submit_counters_reset();

    // Resident-weight path.
    #[allow(dead_code)] // kept for A/B against the canonical-tiling context
    pub fn rocket_ctx_create(nthreads: c_int) -> *mut RocketCtxOpaque;
    pub fn rocket_ctx_create_ex(nthreads: c_int, flags: c_uint) -> *mut RocketCtxOpaque;
    pub fn rocket_ctx_free(ctx: *mut RocketCtxOpaque);
    pub fn rocket_weights_pack(
        ctx: *mut RocketCtxOpaque,
        m: c_int,
        k: c_int,
        n: c_int,
        b: *const F16,
    ) -> *mut RocketWeightsOpaque;
    pub fn rocket_weights_free(ctx: *mut RocketCtxOpaque, w: *mut RocketWeightsOpaque);
    /// Pack several weights sharing one input as one resident weight, concatenated
    /// along N (segmented; the caller does not materialize the concatenation).
    pub fn rocket_weights_pack_seg(
        ctx: *mut RocketCtxOpaque,
        m: c_int,
        k: c_int,
        n: c_int,
        bs: *const *const F16,
        ns: *const c_int,
        nseg: c_int,
    ) -> *mut RocketWeightsOpaque;
    pub fn rocket_matmul_fp16_prepacked(
        ctx: *mut RocketCtxOpaque,
        m: c_int,
        k: c_int,
        n: c_int,
        a: *const F16,
        c: *mut F16,
        w: *mut RocketWeightsOpaque,
    ) -> c_int;

    // Streaming path (weights re-packed per call).
    pub fn rocket_stream_create(nthreads: c_int) -> *mut RocketStreamOpaque;
    pub fn rocket_stream_free(s: *mut RocketStreamOpaque);
    pub fn rocket_matmul_fp16_stream(
        s: *mut RocketStreamOpaque,
        m: c_int,
        k: c_int,
        n: c_int,
        a: *const F16,
        b: *const F16,
        c: *mut F16,
    ) -> c_int;

    // Streaming bf16 path: f32 A/B operands, truncated to bf16 during the NPU
    // scatter; fp32 output (host double K-accum). No resident-weight variant
    // exists, so the weight is re-packed per call. `_mt` is the per-call
    // multicore fallback the stream tells the caller to use on a refusal.
    pub fn rocket_bf16_stream_create(nthreads: c_int) -> *mut RocketBf16StreamOpaque;
    pub fn rocket_bf16_stream_free(s: *mut RocketBf16StreamOpaque);
    #[allow(clippy::too_many_arguments)]
    pub fn rocket_matmul_bf16_stream(
        s: *mut RocketBf16StreamOpaque,
        m: c_int,
        k: c_int,
        n: c_int,
        a: *const f32,
        b: *const f32,
        c: *mut f32,
    ) -> c_int;
    #[allow(clippy::too_many_arguments)]
    pub fn rocket_matmul_bf16_mt(
        m: c_int,
        k: c_int,
        n: c_int,
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        nthreads: c_int,
    ) -> c_int;

    // Masked grouped-query attention (LLM prefill): NPU QK/PV, additive mask,
    // host softmax inside the library.
    pub fn rocket_fa_ctx_create(nthreads: c_int) -> *mut RocketFaCtxOpaque;
    pub fn rocket_fa_ctx_free(c: *mut RocketFaCtxOpaque);
    #[allow(clippy::too_many_arguments)]
    pub fn rocket_flash_attn_fp16_ctx(
        c: *mut RocketFaCtxOpaque,
        n_tokens: c_int,
        n_kv: c_int,
        head_dim: c_int,
        dv: c_int,
        n_head: c_int,
        n_kv_heads: c_int,
        scale: f32,
        softcap: f32,
        q: *const F16,
        k: *const F16,
        v: *const F16,
        mask: *const F16,
        out: *mut F16,
    ) -> c_int;

    // Resident int8 (W8A8) group-wise path: pre-quantized int8 weights/A, per
    // K-group scales applied on readback (`C[M,N] = sum_g a_scale[m,g] *
    // b_scale[n,g] * int32_partial`). K % 32, N % 32, M % 4; the resident weight
    // is M-independent (canonical tile), so one pack serves every M.
    pub fn rocket_i8_ctx_create(nthreads: c_int) -> *mut RocketI8CtxOpaque;
    pub fn rocket_i8_ctx_free(ctx: *mut RocketI8CtxOpaque);
    pub fn rocket_i8_weights_pack_gw(
        ctx: *mut RocketI8CtxOpaque,
        m: c_int,
        k: c_int,
        n: c_int,
        b: *const i8,
        group: c_int,
    ) -> *mut RocketI8WeightsOpaque;
    pub fn rocket_i8_weights_free(ctx: *mut RocketI8CtxOpaque, w: *mut RocketI8WeightsOpaque);
    #[allow(clippy::too_many_arguments)]
    pub fn rocket_matmul_int8_prepacked_gw(
        ctx: *mut RocketI8CtxOpaque,
        m: c_int,
        k: c_int,
        n: c_int,
        a: *const i8,
        a_scale: *const f32,
        b_scale: *const f32,
        cf: *mut f32,
        w: *mut RocketI8WeightsOpaque,
    ) -> c_int;
}

#[repr(C)]
pub struct RocketI8CtxOpaque {
    _private: [u8; 0],
}

#[repr(C)]
pub struct RocketI8WeightsOpaque {
    _private: [u8; 0],
}

#[repr(C)]
pub struct RocketFaCtxOpaque {
    _private: [u8; 0],
}
