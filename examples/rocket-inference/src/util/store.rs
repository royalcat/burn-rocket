//! Safetensors load-report checks shared by the model loaders.

use anyhow::{Result, bail};
use burn_store::ApplyResult;

/// Fail on load errors, on missing parameters not tolerated by `missing_ok`,
/// and report file tensors the model tree never uses.
///
/// `missing_ok(name)` marks parameters a checkpoint may legitimately omit (for
/// example KV-shared layers that never compute K/V).
pub fn check_load_report(result: &ApplyResult, missing_ok: impl Fn(&str) -> bool) -> Result<()> {
    if !result.errors.is_empty() {
        bail!("load errors: {:?}", result.errors);
    }
    let missing: Vec<_> = result
        .missing
        .iter()
        .filter(|(name, _)| !missing_ok(name))
        .collect();
    if !missing.is_empty() {
        bail!(
            "{} model parameters missing from file (first: {:?})",
            missing.len(),
            missing.first()
        );
    }
    if !result.unused.is_empty() {
        let sample: Vec<&String> = result.unused.iter().take(8).collect();
        println!(
            "note: {} file tensors unused by the model tree (e.g. {sample:?})",
            result.unused.len()
        );
    }
    Ok(())
}
