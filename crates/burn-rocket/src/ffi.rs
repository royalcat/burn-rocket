//! Raw FFI declarations for `librocketnpu` (gregordinary/rocket-userspace).
//!
//! `_Float16` arrays are passed as raw pointers; the element type is `u16` here
//! (the bit pattern of the half-precision float), converted with the `half` crate.

#![allow(non_camel_case_types)]

use libc::{c_char, c_int, c_long, c_void};

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

pub const ROCKET_OK: c_int = 0;
pub const ROCKET_E_SHAPE: c_int = -1;
pub const ROCKET_E_TILING: c_int = -2;
pub const ROCKET_E_NOMEM: c_int = -3;
pub const ROCKET_E_DEVICE: c_int = -4;
pub const ROCKET_E_UNSUPPORTED: c_int = -5;

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
    pub fn rocket_ctx_create(nthreads: c_int) -> *mut RocketCtxOpaque;
    pub fn rocket_ctx_free(ctx: *mut RocketCtxOpaque);
    pub fn rocket_weights_pack(
        ctx: *mut RocketCtxOpaque,
        m: c_int,
        k: c_int,
        n: c_int,
        b: *const F16,
    ) -> *mut RocketWeightsOpaque;
    pub fn rocket_weights_free(ctx: *mut RocketCtxOpaque, w: *mut RocketWeightsOpaque);
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
}
