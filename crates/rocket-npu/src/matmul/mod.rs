//! The tiled matmul engine: geometry, planning, layout, and the compute paths.

pub mod batch;
pub mod bf16;
pub mod bos;
pub(crate) mod chain;
pub mod context;
pub mod f32out;
pub mod geom;
pub mod int8;
pub mod layout;
pub mod plan;
pub mod run;

pub use batch::{MmBatch, batch};
pub use bf16::Bf16Stream;
pub use bos::{Bos, Config};
/// Whether the contiguous self-chaining submit layout is active for this process
/// (`ROCKET_BATCH_SUBMIT=1` on a kernel that honors `DRM_ROCKET_JOB_BATCHED`). Resolved
/// once; the fp16 CPU-accumulation and batched matmuls use it per submit group.
pub use chain::enabled as chain_enabled;
pub use context::{Context, MemStats, SlotMem, Weight};
pub use f32out::one_shot as one_shot_f32out;
pub use int8::{I8Context, I8Weight};
pub use plan::{Plan, plan_geom};
pub use run::{
    compute_cpu, compute_kacc, i8_weights_bytes, one_shot, one_shot_bf16, one_shot_bf16_mt,
    one_shot_mt, pad_m, weights_bytes,
};
