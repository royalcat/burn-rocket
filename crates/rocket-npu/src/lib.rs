//! Pure-Rust userspace driver for the mainline RK3588 `rocket` NPU.
//!
//! This crate replaces the C `librocketnpu` with a native Rust implementation of the
//! same algorithms: DRM-accel device handling, register-command generation, the
//! tiled matmul engine (fp16, int8, int4, int16, bf16, tf32), resident weights,
//! multi-threaded fan-out, and flash attention. It is an independent crate: no Burn
//! dependency, no C toolchain, no kernel headers.
//!
//! The layout follows the upstream C library closely enough that a function can be
//! read side by side with its original (`rocket_matmul.c`, `npu_regcmd.c`, …):
//!
//! - [`device`] — `/dev/accel/accelN`, buffer objects, cache sync, job submission.
//! - [`regcmd`] — register-command encoding (the hardware program).
//! - [`matmul`] — geometry planning, host packing, resident/streaming engines.
//! - [`attn`] — flash attention on top of the matmul engine.
//! - [`knobs`], [`hw`], [`error`] — diagnostics, the RK3588 profile, typed failures.
//!
//! The crate is inference-only and targets the RK3588 only. Failure payloads carry
//! the same negative status codes as the C library so callers can keep their error
//! mapping (`-3` NOMEM, `-2` tiling, …).
//!
//! # Surface covered and deliberately missing
//!
//! This is a focused port of the C library's matmul/attention core, not a
//! function-for-function replacement of the full 367-entry API. Implemented:
//!
//! - device/uAPI: open/caps, BO allocation (32-bit IOVA checked), prep/fini
//!   (whole-object and ranged), one-job submit, submit/fence counters, sysfs core
//!   enumeration, big-core detection and worker pinning;
//! - fp16: one-shot, streaming, resident and segmented-resident matmul, canonical
//!   and legacy tiling, KACC and CPU-accumulation, per-shape scratch cache;
//! - int8: resident **group-wise** W8A8 only ([`matmul::I8Context`]);
//! - bf16: the native streaming path only ([`matmul::Bf16Stream`]);
//! - attention: masked GQA with host mask/soft-cap/softmax, per-worker head split
//!   ([`attn::Attention`]); the host softmax has a fast branch-free `exp` path
//!   ([`attn::Softmax`]) and keeps the original scalar one behind
//!   `SOFTMAX_EXACT=1`.
//!
//! Missing (no API in this crate yet; callers get a compile error, not a silent
//! wrong result):
//!
//! - fp16 `rocket_matmul_fp16_mt`, `_f32out`, `_batch`/`rocket_mm_batch_*` and the
//!   public `rocket_stream_*` aliases (use [`matmul::Context`] for the resident and
//!   streaming forms);
//! - dense/per-channel int8 (`rocket_matmul_int8*` without `_gw`), one-shot
//!   `rocket_matmul_int8_groupwise`, and the int8 planner's dense variant;
//! - all int4, int16 and tf32 entries (one-shot, resident, group-wise, plans);
//! - one-shot bf16 (`rocket_matmul_bf16`) and a resident bf16 weight (only the
//!   stream exists, matching the C library);
//! - `rocket_mha_self_fp16`, the `*_ref_fp16` host references, the `_planes` and
//!   transposed-output variants, and the RK3576-specific entries;
//! - batched/chained submit (`ROCKET_BATCH_SUBMIT`, `rocket_chain_*`), multi-job
//!   submit, and the public `rocket_submit_*` convenience wrappers;
//! - the `rocket_bo_ensure32`, `rocket_weights_bytes`/`rocket_i8_weights_bytes`,
//!   `rocket_ctx_nthreads` and post-init `rocket_busy_poll_set_us` helpers;
//! - the long-context tiled attention path (`ROCKET_FA_TILE_KV`), on-NPU softmax,
//!   and chained per-head attention batching;
//! - every non-matmul op family: conv, pool, activation, norm, FFN, softmax,
//!   reduce, resize/upsample, encoder, SAM/SigLIP and graph execution.
//!
//! Env knobs belonging to the missing dtypes are not read; the fp16/int8/bf16 and
//! attention knobs that are implemented keep their C names and semantics.

pub mod attn;
pub mod device;
pub mod error;
pub mod hw;
pub mod knobs;
pub mod matmul;
pub mod regcmd;

mod log;

pub use device::{Bo, Caps, Device, Version};
pub use error::{Error, Result, Status};
pub use hw::Profile;

/// Initialize the crate: resolve the process-wide hardware profile and the core
/// affinity set. Idempotent and cheap; callers need not call it, but doing so at
/// startup keeps the first matmul free of one-time probing.
pub fn init() {
    let _ = hw::current();
    let _ = device::affinity::num_big_cores();
}
