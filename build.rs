//! Links `librocketnpu.a` (gregordinary/rocket-userspace) for `npu` builds.
//!
//! Supply the archive with `scripts/build-rocketnpu.sh`: it builds a pinned
//! upstream commit on this host (aarch64 cross by default; `--target host`
//! makes a native archive for link checks) and installs it to
//! `<repo>/vendor/rocketnpu` together with `COMMIT`/`ARCH` provenance files.
//! Set `ROCKETNPU_DIR` to override the directory holding `librocketnpu.a`.

use std::path::PathBuf;

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
            manifest.join("vendor/rocketnpu")
        }
    };

    let lib = dir.join("librocketnpu.a");
    // Watch the archive: `scripts/build-rocketnpu.sh` replaces it in place and
    // the crate must relink when it changes. A missing file counts as changed,
    // so its creation is picked up too.
    println!("cargo:rerun-if-changed={}", lib.display());
    if !lib.exists() {
        println!(
            "cargo:warning=librocketnpu.a not found at {}; run scripts/build-rocketnpu.sh \
             (or set ROCKETNPU_DIR to a directory holding it). The crate compiles but \
             will not link",
            lib.display()
        );
        return;
    }

    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

    // The build script records the archive's architecture next to it; use it
    // to catch a stale archive of the other flavor (the link would otherwise
    // fail with confusing symbol errors).
    let built = std::fs::read_to_string(dir.join("ARCH"))
        .ok()
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty());
    match &built {
        Some(built) if *built != arch => {
            println!(
                "cargo:warning=librocketnpu.a at {} was built for {built}, not {arch}; \
                 rebuild it with scripts/build-rocketnpu.sh --target {}",
                lib.display(),
                if arch == "aarch64" { "aarch64" } else { "host" }
            );
        }
        None if arch != "aarch64" => {
            println!(
                "cargo:warning=burn-rocket targets aarch64; current target is {arch}. The \
                 archive at {} has no ARCH provenance — if it was built for aarch64 it \
                 will not link here; use scripts/build-rocketnpu.sh --target host for \
                 link checks",
                lib.display()
            );
        }
        _ => {}
    }

    if let Some(parent) = lib.parent() {
        println!("cargo:rustc-link-search=native={}", parent.display());
    }
    println!("cargo:rustc-link-lib=static=rocketnpu");
    // librocketnpu uses libm (erf/exp/log) and pthreads.
    println!("cargo:rustc-link-lib=m");
    println!("cargo:rustc-link-lib=pthread");

}
