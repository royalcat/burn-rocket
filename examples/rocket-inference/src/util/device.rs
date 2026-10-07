//! `--backend` resolution.

use anyhow::{Result, bail};
use burn::prelude::*;

/// Resolve the `--backend` flag to a Burn device: `flex` is always available,
/// `cpu` only when the `cpu` feature (CubeCL/LLVM) is compiled in.
pub fn device(backend: &str) -> Result<Device> {
    match backend {
        #[cfg(feature = "cpu")]
        "cpu" => Ok(Device::cpu()),
        "flex" => Ok(Device::flex()),
        #[cfg(not(feature = "cpu"))]
        "cpu" => {
            bail!("this build has no 'cpu' backend (built without the 'cpu' feature); use flex")
        }
        other => bail!("unknown backend '{other}' (expected cpu|flex)"),
    }
}
