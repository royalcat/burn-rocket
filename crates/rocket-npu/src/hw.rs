//! The RK3588 hardware profile and the chip-detection front end.
//!
//! Profile values are machine parameters copied from the C library (and from the
//! vendor measurements recorded there). The detector follows the C order:
//! `ROCKET_CHIP`, the bound rocket device's DT compatible, `/proc/device-tree`,
//! then the RK3588 default. Other chips are named in a warning but run with the
//! RK3588 profile — this port implements the RK3588 encoding only.

use std::sync::OnceLock;

use crate::log::log_warn;

/// The machine parameters of one SoC's NPU.
#[derive(Debug, Clone, Copy)]
pub struct Profile {
    pub name: &'static str,
    /// CBUF banks used by the emitter's bank budget.
    pub cbuf_banks: usize,
    pub cbuf_bank_size: usize,
    /// Mt/Nt cap.
    pub max_tile: u32,
    pub kgroup_2b: u32,
    pub kgroup_4b: u32,
    pub ngroup: u32,
    pub default_workers: u32,
    /// Fenced-wait duration that marks a likely retired job.
    pub slow_wait_ms: u64,
    /// The driver's job watchdog.
    pub backstop_ms: u64,
}

/// RK3588 / RK3588S.
pub const RK3588: Profile = Profile {
    name: "rk3588",
    cbuf_banks: 12,
    cbuf_bank_size: 32768,
    max_tile: 256,
    kgroup_2b: 32,
    kgroup_4b: 16,
    ngroup: 16,
    default_workers: 8,
    slow_wait_ms: 450,
    backstop_ms: 500,
};

fn read_dt_strings(path: &str) -> Option<String> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.is_empty() {
        return None;
    }
    let mut s = String::with_capacity(bytes.len());
    for b in bytes {
        s.push(if b == 0 { ' ' } else { b as char });
    }
    Some(s)
}

fn bound_device_compatible() -> Option<String> {
    let entries = std::fs::read_dir("/sys/bus/platform/drivers/rocket").ok()?;
    for entry in entries.flatten() {
        let path = entry.path().join("of_node/compatible");
        if let Some(s) = read_dt_strings(path.to_str()?) {
            return Some(s);
        }
    }
    None
}

fn detect() -> &'static Profile {
    if let Some(forced) = crate::knobs::string("ROCKET_CHIP") {
        if forced == RK3588.name {
            return &RK3588;
        }
        log_warn!(
            "ROCKET_CHIP=\"{forced}\" names no profile; falling back to {}",
            RK3588.name
        );
        return &RK3588;
    }

    let (compat, src) = match bound_device_compatible() {
        Some(c) => (Some(c), "bound rocket device"),
        None => (
            read_dt_strings("/proc/device-tree/compatible"),
            "/proc/device-tree/compatible",
        ),
    };
    let Some(compat) = compat else {
        return &RK3588;
    };
    if compat.contains("rk3588") {
        return &RK3588;
    }
    if compat.contains("rk3576") {
        log_warn!(
            "rk3576 detected ({src}), but this port implements the rk3588 encoding only; \
             running with the rk3588 profile will not compute correctly"
        );
        return &RK3588;
    }
    if compat.contains("rk3568") || compat.contains("rk3566") {
        log_warn!(
            "unprofiled NPU detected ({src}, compatible \"{compat}\"); \
             the rk3588 profile and encoding do not apply"
        );
        return &RK3588;
    }
    log_warn!("unrecognized NPU compatible \"{compat}\" ({src}); falling back to rk3588");
    &RK3588
}

/// The active profile, resolved once per process.
pub fn current() -> &'static Profile {
    static PROFILE: OnceLock<&'static Profile> = OnceLock::new();
    PROFILE.get_or_init(detect)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_matches_the_c_values() {
        assert_eq!(RK3588.cbuf_banks * RK3588.cbuf_bank_size, 12 * 32768);
        assert_eq!(RK3588.max_tile, 256);
        assert_eq!(RK3588.kgroup_2b, 32);
        assert_eq!(RK3588.ngroup, 16);
    }
}
