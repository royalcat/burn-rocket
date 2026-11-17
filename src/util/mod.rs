//! Generic, family-agnostic helpers shared by every model in this crate.

pub mod causal;
pub mod checkpoint;
pub mod device;
pub mod gguf;
pub mod gguf_tokenizer;
#[cfg(all(feature = "npu", target_arch = "aarch64"))]
pub mod glue;
pub mod http;
pub mod hub;
pub mod mem;
pub mod proj;
pub mod quant;
pub mod rope;
pub mod store;

pub use device::device;
pub use mem::{log_npu_mem, rss_mib, trim_after_request, trim_heap, trim_npu_scratch};
