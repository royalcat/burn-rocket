//! Sysfs enumeration of the rocket-bound NPU cores.

use std::sync::OnceLock;

const DRIVER_DIR: &str = "/sys/bus/platform/drivers/rocket";
const MAX_DEVS: usize = 8;

/// Names of the platform devices the rocket driver is bound to, capped at [`MAX_DEVS`].
///
/// The driver directory also holds `bind`, `unbind`, `uevent` and a `module` link; a
/// bound device is the entry carrying a `power/` subdirectory.
pub fn bound_devices() -> Vec<String> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(DRIVER_DIR) else {
        return out;
    };
    for entry in entries.flatten() {
        if out.len() >= MAX_DEVS {
            break;
        }
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with('.') {
            continue;
        }
        if entry.path().join("power").is_dir() {
            out.push(name.into_owned());
        }
    }
    out
}

/// How many NPU cores the driver has bound. Cached, like the C library, because the
/// bind set does not change under a process using the device.
pub fn bound_core_count() -> usize {
    static COUNT: OnceLock<usize> = OnceLock::new();
    *COUNT.get_or_init(|| bound_devices().len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enumeration_is_stable() {
        let a = bound_devices().len();
        let b = bound_core_count();
        assert_eq!(a, b);
    }
}
