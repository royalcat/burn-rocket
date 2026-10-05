//! Links `librocketnpu.a` (gregordinary/rocket-userspace) for aarch64 targets.
//!
//! The library is aarch64-only. Set `ROCKETNPU_DIR` to the directory holding
//! `librocketnpu.a`; the default is `<repo>/vendor/rocketnpu`.
//!
//! On the board the library can be used straight from its build tree:
//! `ROCKETNPU_DIR=/root/npu-poc/rocket-userspace/build`.

use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=ROCKETNPU_DIR");

    // Only `npu` builds actually call into librocketnpu; the extension code
    // (`flex` without `npu`) is checked but never linked against it.
    if std::env::var_os("CARGO_FEATURE_NPU").is_none() {
        return;
    }

    let dir: PathBuf = match std::env::var_os("ROCKETNPU_DIR") {
        Some(dir) => PathBuf::from(dir),
        None => {
            let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
            manifest.join("../../vendor/rocketnpu")
        }
    };

    let lib = dir.join("librocketnpu.a");
    if !lib.exists() {
        println!(
            "cargo:warning=librocketnpu.a not found at {} (set ROCKETNPU_DIR); \
             the crate compiles but will not link",
            lib.display()
        );
        return;
    }

    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if arch != "aarch64" {
        println!(
            "cargo:warning=burn-rocket targets aarch64 only (current arch: {arch}); \
             linking {} will fail",
            lib.display()
        );
    }

    if let Some(parent) = lib.parent() {
        println!("cargo:rustc-link-search=native={}", parent.display());
    }
    println!("cargo:rustc-link-lib=static=rocketnpu");
    // librocketnpu uses libm (erf/exp/log) and pthreads.
    println!("cargo:rustc-link-lib=m");
    println!("cargo:rustc-link-lib=pthread");

    // Keep the archive's mtime visible to cargo.
    let _ = Path::new(&lib).metadata();
}
