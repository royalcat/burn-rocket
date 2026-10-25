//! Register-command generation.
//!
//! One matmul task is a `u64` program stream: `(target << 48) | (value << 16) | reg`.
//! The emitters live per dtype; the shared geometry/planning code lives in
//! [`crate::matmul`].

pub mod activation;
pub mod bf16;
pub mod desc;
pub mod fp16;
pub mod int4;
pub mod int8;
pub mod tf32;
