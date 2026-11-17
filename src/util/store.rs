//! Safetensors load-report checks shared by the model loaders.

use std::path::Path;

use anyhow::{Result, bail};
use burn::tensor::DType;
use burn_store::ApplyResult;

/// The checkpoint's native float dtype, read from the safetensors header (the
/// first float tensor's dtype — packed-int QAT exports have bf16 scales, so the
/// first float entry still names the model dtype). Falls back to bf16, the
/// dtype of every current checkpoint, when the header cannot be parsed.
pub fn native_dtype(path: &Path) -> DType {
    fn first_float_dtype(path: &Path) -> Option<DType> {
        use std::io::Read;

        let mut f = std::fs::File::open(path).ok()?;
        let mut len = [0u8; 8];
        f.read_exact(&mut len).ok()?;
        let n = u64::from_le_bytes(len) as usize;
        let mut buf = vec![0u8; n];
        f.read_exact(&mut buf).ok()?;
        let header: serde_json::Value = serde_json::from_slice(&buf).ok()?;
        for (name, meta) in header.as_object()? {
            if name == "__metadata__" {
                continue;
            }
            match meta.get("dtype").and_then(|d| d.as_str()) {
                Some("F32") => return Some(DType::F32),
                Some("F16") => return Some(DType::F16),
                Some("BF16") => return Some(DType::BF16),
                _ => {}
            }
        }
        None
    }

    first_float_dtype(path).unwrap_or(DType::BF16)
}

/// Fail on load errors, on missing parameters not tolerated by `missing_ok`,
/// and report file tensors the model tree never uses.
///
/// `missing_ok(name)` marks parameters a checkpoint may legitimately omit (for
/// example KV-shared layers that never compute K/V).
pub fn check_load_report(result: &ApplyResult, missing_ok: impl Fn(&str) -> bool) -> Result<()> {
    check_missing(result, missing_ok)?;
    if !result.unused.is_empty() {
        let sample: Vec<&String> = result.unused.iter().take(8).collect();
        println!(
            "note: {} file tensors unused by the model tree (e.g. {sample:?})",
            result.unused.len()
        );
    }
    Ok(())
}

/// Fail on load errors and on parameters missing from the file. Unused file
/// tensors are expected: a focused pass (for example the towers-only load) sees
/// the rest of the checkpoint and ignores it.
pub fn check_missing(result: &ApplyResult, missing_ok: impl Fn(&str) -> bool) -> Result<()> {
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
    Ok(())
}
