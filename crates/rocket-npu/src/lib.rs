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
//!   (whole-object and ranged), one-job and multi-job submit (`submit`,
//!   `submit_tasks`, `submit_tasks_pre`, `submit_jobs`), the capability probes
//!   (`batched_submit_supported`, `submit_batch_atomic`, `batch_completion_tracked`,
//!   `no_dpu_done_supported`, `ppu_done_supported`, `bo_ranges_supported`, and the
//!   device-less `batched_submit_supported_global`), `Bo::ensure32`, submit/fence
//!   counters and the post-init `fence::set_busy_poll_us`, sysfs core enumeration,
//!   big-core detection and worker pinning;
//! - the contiguous self-chaining regcmd layout (`matmul::chain`, the port of
//!   `rocket_chain.c`) as host-testable slice helpers: `enabled`, `words`,
//!   `amount_encode`, `pack`, `pack_at`, `verify`;
//! - fp16: one-shot, one-shot multi-fd (`_mt`) and fp32-output (`_f32out`),
//!   streaming, resident and segmented-resident matmul, canonical and legacy tiling,
//!   KACC and CPU-accumulation, per-shape scratch cache;
//! - int8: resident **group-wise** W8A8 only ([`matmul::I8Context`]);
//! - bf16: the native streaming path ([`matmul::Bf16Stream`]) plus the one-shot
//!   `_bf16`/`_mt` wrappers over it;
//! - the `rocket_weights_bytes`/`rocket_i8_weights_bytes` footprint helpers and
//!   `rocket_pad_m` ([`matmul::pad_m`]);
//! - attention: masked GQA with host mask/soft-cap/softmax, per-worker head split
//!   ([`attn::Attention`]); the host softmax has a fast branch-free `exp` path
//!   ([`attn::Softmax`]) and keeps the original scalar one behind
//!   `SOFTMAX_EXACT=1`.
//!
//! Missing (no API in this crate yet; callers get a compile error, not a silent
//! wrong result):
//!
//! - fp16 `_batch`/`rocket_mm_batch_*` and the public `rocket_stream_*` aliases (use
//!   [`matmul::Context`] for the resident and streaming forms);
//! - dense/per-channel int8 (`rocket_matmul_int8*` without `_gw`), one-shot
//!   `rocket_matmul_int8_groupwise`, and the int8 planner's dense variant;
//! - all int4, int16 and tf32 entries (one-shot, resident, group-wise, plans);
//! - a resident bf16 weight (the C library has none either);
//! - `rocket_mha_self_fp16`, the `*_ref_fp16` host references, the `_planes` and
//!   transposed-output variants, and the RK3576-specific entries;
//! - the matmul batch wrappers (`rocket_matmul_fp16_batch`, `rocket_mm_batch_*`) and
//!   the Context/attention paths that would *use* the chained layout — `matmul::chain`
//!   provides the layout and the submit seam it needs, but no engine batches yet, so
//!   `ROCKET_BATCH_SUBMIT` currently changes nothing;
//! - the `rocket_ctx_nthreads` helper;
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
