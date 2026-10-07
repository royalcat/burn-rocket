//! Generic, family-agnostic helpers shared by every model in this crate.

pub mod device;
pub mod http;
pub mod mem;
pub mod proj;
pub mod quant;
pub mod rope;
pub mod store;

pub use device::device;
pub use mem::rss_mib;
