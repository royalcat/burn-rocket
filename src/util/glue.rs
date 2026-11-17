//! Process-wide gate for the fused CPU glue kernels in NPU builds
//! (`burn_rocket::{gelu_mul, silu_mul, rms_norm, rms_norm_noscale, rope_apply}`).
//!
//! The NPU loaders set it; `ROCKET_GLUE=0` restores the composite ops (the A/B
//! that validates the kernels against them) and `ROCKET_GLUE=1` forces the
//! kernels in CPU modes. Like the Gemma gate, the env var is read once per
//! process (first use), after the loaders have set the flag.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

static GLUE: AtomicBool = AtomicBool::new(false);

/// Set by the NPU loaders (`--npu` modes).
pub fn set_glue(on: bool) {
    GLUE.store(on, Ordering::Relaxed);
}

/// Whether the fused glue kernels are active.
pub fn glue() -> bool {
    static ONCE: OnceLock<bool> = OnceLock::new();
    *ONCE.get_or_init(|| match std::env::var("ROCKET_GLUE") {
        Ok(v) => v != "0",
        Err(_) => GLUE.load(Ordering::Relaxed),
    })
}
