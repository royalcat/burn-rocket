//! The tiled matmul engine: geometry, planning, layout, and the compute paths.

pub mod bf16;
pub mod bos;
pub mod context;
pub mod geom;
pub mod int8;
pub mod layout;
pub mod plan;
pub mod run;

pub use bf16::Bf16Stream;
pub use bos::{Bos, Config};
pub use context::{Context, Weight};
pub use int8::{I8Context, I8Weight};
pub use plan::{Plan, plan_geom};
pub use run::{compute_cpu, compute_kacc, one_shot};
