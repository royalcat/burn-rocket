//! Links `librocketnpu.a` (gregordinary/rocket-userspace) for `npu` builds.
//!
//! Archive resolution, in order:
//!
//! 1. `ROCKETNPU_DIR`, if set (explicit override; never auto-built).
//! 2. `<repo>/vendor/rocketnpu`, when its `ARCH` provenance matches the target
//!    (an absent `ARCH` counts as aarch64 — legacy archives were aarch64-only).
//! 3. `$OUT_DIR/rocketnpu`, the archive left by a previous auto-build.
//! 4. Auto-build: `scripts/build-rocketnpu.sh --out $OUT_DIR/rocketnpu` (the
//!    script's clone/cmake cache lands in `$OUT_DIR/rocket-userspace`). Set
//!    `ROCKETNPU_AUTO=0` to disable; `ROCKETNPU_SRC` builds from an existing
//!    checkout (offline).
//!
//! A failed auto-build only warns — the crate still compiles, it just cannot
//! link — and a stamp in `$OUT_DIR` keeps later builds from retrying until the
//! script changes or `cargo clean -p burn-rocket` removes the stamp.

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=ROCKETNPU_DIR");
    println!("cargo:rerun-if-env-changed=ROCKETNPU_AUTO");
    println!("cargo:rerun-if-env-changed=ROCKETNPU_SRC");

    // Only `npu` builds actually call into librocketnpu; the extension code
    // (`flex` without `npu`) is checked but never linked against it.
    if std::env::var_os("CARGO_FEATURE_NPU").is_none() {
        return;
    }

    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

    // 1. Explicit override.
    if let Some(dir) = std::env::var_os("ROCKETNPU_DIR") {
        let dir = PathBuf::from(dir);
        watch_lib(&dir);
        if dir.join("librocketnpu.a").exists() {
            warn_arch_mismatch(&dir, &target_arch);
            link(&dir);
        } else {
            println!(
                "cargo:warning=librocketnpu.a not found in ROCKETNPU_DIR={}; run \
                 scripts/build-rocketnpu.sh (or unset ROCKETNPU_DIR to let it \
                 auto-build). The crate compiles but will not link",
                dir.display()
            );
        }
        return;
    }

    // 2. Vendor archive: prefer it when its provenance matches the target.
    let vendor = manifest.join("vendor/rocketnpu");
    watch_lib(&vendor);
    let mut vendor_note = None;
    match archive_arch(&vendor) {
        Some(arch) if arch == target_arch => {
            link(&vendor);
            return;
        }
        Some(arch) => {
            vendor_note = Some(format!(
                "the vendor archive at {} was built for {arch}, not {target_arch}",
                vendor.join("librocketnpu.a").display()
            ));
        }
        None if vendor.join("librocketnpu.a").exists() && target_arch == "aarch64" => {
            link(&vendor);
            return;
        }
        None if vendor.join("librocketnpu.a").exists() => {
            vendor_note = Some(format!(
                "the vendor archive at {} has no ARCH provenance and the target is {target_arch}",
                vendor.join("librocketnpu.a").display()
            ));
        }
        None => {}
    }

    // 3. Archive from a previous auto-build.
    let out_dir =
        PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR is set for build scripts"));
    let out_archive = out_dir.join("rocketnpu");
    if out_archive.join("librocketnpu.a").exists() {
        link(&out_archive);
        return;
    }

    // 4. Auto-build (on by default; ROCKETNPU_AUTO=0 turns it off).
    let script = manifest.join("scripts/build-rocketnpu.sh");
    if !auto_enabled() {
        warn_manual(
            &vendor_note,
            &format!(
                "auto-build disabled by ROCKETNPU_AUTO=0; run {} (or set ROCKETNPU_DIR)",
                script.display()
            ),
        );
        return;
    }
    println!("cargo:rerun-if-changed={}", script.display());
    if stamp_blocks(&out_dir, &script) {
        warn_manual(
            &vendor_note,
            &format!(
                "a previous auto-build failed; run {} manually to see why, or \
                 `cargo clean -p burn-rocket` to retry",
                script.display()
            ),
        );
        return;
    }
    match auto_build(&script, &out_archive, &target_arch) {
        Ok(()) => {
            watch_lib(&out_archive);
            link(&out_archive);
        }
        Err(reason) => {
            write_stamp(&out_dir, &script);
            println!(
                "cargo:warning=librocketnpu.a auto-build failed: {reason}. The crate \
                 compiles but will not link; run {} manually to see the full log, \
                 or `cargo clean -p burn-rocket` to retry",
                script.display()
            );
        }
    }
}

fn link(dir: &Path) {
    println!("cargo:rustc-link-search=native={}", dir.display());
    println!("cargo:rustc-link-lib=static=rocketnpu");
    // librocketnpu uses libm (erf/exp/log) and pthreads.
    println!("cargo:rustc-link-lib=m");
    println!("cargo:rustc-link-lib=pthread");
}

/// Watch the archive: `scripts/build-rocketnpu.sh` replaces it in place and the
/// crate must relink when it changes. A missing file counts as changed, so its
/// creation is picked up too.
fn watch_lib(dir: &Path) {
    println!(
        "cargo:rerun-if-changed={}",
        dir.join("librocketnpu.a").display()
    );
}

/// The architecture recorded by `scripts/build-rocketnpu.sh`, if any.
fn archive_arch(dir: &Path) -> Option<String> {
    std::fs::read_to_string(dir.join("ARCH"))
        .ok()
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
}

/// The build script records the archive's architecture next to it; use it to
/// catch a stale archive of the other flavor (the link would otherwise fail
/// with confusing symbol errors).
fn warn_arch_mismatch(dir: &Path, target_arch: &str) {
    match archive_arch(dir) {
        Some(arch) if arch != target_arch => {
            println!(
                "cargo:warning=librocketnpu.a at {} was built for {arch}, not {target_arch}; \
                 rebuild it with scripts/build-rocketnpu.sh --target {}",
                dir.join("librocketnpu.a").display(),
                if target_arch == "aarch64" {
                    "aarch64"
                } else {
                    "host"
                }
            );
        }
        None if target_arch != "aarch64" => {
            println!(
                "cargo:warning=burn-rocket targets aarch64; current target is {target_arch}. The \
                 archive at {} has no ARCH provenance — if it was built for aarch64 it \
                 will not link here; use scripts/build-rocketnpu.sh --target host for \
                 link checks",
                dir.join("librocketnpu.a").display()
            );
        }
        _ => {}
    }
}

fn warn_manual(vendor_note: &Option<String>, hint: &str) {
    let mut msg = String::from("librocketnpu.a not found for this target");
    if let Some(note) = vendor_note {
        msg.push_str(": ");
        msg.push_str(note);
    }
    msg.push_str("; ");
    msg.push_str(hint);
    msg.push_str(". The crate compiles but will not link");
    println!("cargo:warning={msg}");
}

/// Auto-build is on unless `ROCKETNPU_AUTO` is one of the falsy spellings.
fn auto_enabled() -> bool {
    match std::env::var("ROCKETNPU_AUTO") {
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "no" | "off"
        ),
        Err(_) => true,
    }
}

/// Run the build script for this target, installing into `$OUT_DIR/rocketnpu`.
/// The child inherits `OUT_DIR`, so its clone/cmake cache stays under the
/// cargo target dir.
fn auto_build(script: &Path, out: &Path, target_arch: &str) -> Result<(), String> {
    let host_arch = std::env::consts::ARCH;
    let target = if target_arch == host_arch {
        "host"
    } else if target_arch == "aarch64" {
        "aarch64"
    } else {
        return Err(format!(
            "no auto-build rule for target {target_arch} on {host_arch}"
        ));
    };
    let output = Command::new("bash")
        .arg(script)
        .arg("--target")
        .arg(target)
        .arg("--out")
        .arg(out)
        .output()
        .map_err(|e| format!("cannot run bash {}: {e}", script.display()))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let text = if stderr.trim().is_empty() {
            stdout
        } else {
            stderr
        };
        let tail: Vec<&str> = text.lines().rev().take(3).collect();
        let tail = tail.into_iter().rev().collect::<Vec<_>>().join(" / ");
        return Err(if tail.is_empty() {
            format!("{} exited with {}", script.display(), output.status)
        } else {
            format!("{} exited with {}: {tail}", script.display(), output.status)
        });
    }
    if !out.join("librocketnpu.a").exists() {
        return Err(format!(
            "{} succeeded but no archive was installed",
            script.display()
        ));
    }
    Ok(())
}

/// A failed auto-build is not retried until the script itself changes (a stamp
/// records its size+mtime) or the stamp is removed (`cargo clean`).
fn stamp_blocks(out_dir: &Path, script: &Path) -> bool {
    match std::fs::read_to_string(out_dir.join("rocketnpu-auto-failed")) {
        Ok(stamp) => stamp.trim() == script_fingerprint(script),
        Err(_) => false,
    }
}

fn write_stamp(out_dir: &Path, script: &Path) {
    let _ = std::fs::write(
        out_dir.join("rocketnpu-auto-failed"),
        script_fingerprint(script),
    );
}

fn script_fingerprint(script: &Path) -> String {
    match std::fs::metadata(script) {
        Ok(meta) => {
            let mtime = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            format!("{} {mtime}", meta.len())
        }
        Err(_) => "missing".to_owned(),
    }
}
