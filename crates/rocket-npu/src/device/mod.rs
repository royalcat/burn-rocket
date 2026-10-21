//! Device, buffer-object, and submit primitives.
//!
//! One [`Device`] is one DRM file descriptor. Multi-threaded contexts open one device
//! per worker, exactly like the C library, so submits from different threads do not
//! contend on the driver's per-file lock.
//!
//! Buffer objects are GEM handles with a CPU mapping; [`Bo::prep`] waits for the
//! NPU's writes to land and hands the pages back to the CPU, [`Bo::fini`] hands them
//! back to the device. Both are whole-object syncs unless the range forms are
//! available (interface 1.5).

pub mod affinity;
pub mod bo;
pub mod fence;
pub mod submit;
pub mod sysfs;
pub mod uapi;

use std::ffi::CString;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::path::Path;
use std::sync::Arc;

use crate::error::{Error, Result};
use crate::log::log_debug;

pub use bo::{Bo, BoRange, MemRef};
pub use uapi::Handle;

/// Device permissions and driver capabilities, resolved at open time.
#[derive(Debug, Clone, Copy)]
pub struct Version {
    pub major: i32,
    pub minor: i32,
    pub patch: i32,
}

impl Version {
    fn at_least_minor(self, minor: i32) -> bool {
        self.major > 1 || (self.major == 1 && self.minor >= minor)
    }
}

/// What the running kernel supports.
#[derive(Debug, Clone, Copy)]
pub struct Caps {
    pub version: Version,
    /// `DRM_ROCKET_JOB_BATCHED` is honored (module parameter says so, or 1.1+).
    pub batched: bool,
    /// `DRM_ROCKET_JOB_NO_DPU_DONE` is honored (RK3576 only; always false here).
    pub no_dpu_done: bool,
    /// `DRM_ROCKET_JOB_PPU_DONE` is honored (1.3+).
    pub ppu_done: bool,
    /// Ranged cache maintenance ioctls exist (1.5+).
    pub ranges: bool,
    /// A batched job waits for all of its tasks (1.4+).
    pub batch_completion_tracked: bool,
}

pub(crate) struct DeviceInner {
    fd: OwnedFd,
    caps: Caps,
}

impl DeviceInner {
    pub(crate) fn fd(&self) -> RawFd {
        self.fd.as_raw_fd()
    }

    pub(crate) fn caps(&self) -> Caps {
        self.caps
    }
}

/// A DRM-accel handle for the mainline `rocket` driver.
#[derive(Clone)]
pub struct Device {
    inner: Arc<DeviceInner>,
}

impl Device {
    /// Open `$ROCKET_DEV` (default `/dev/accel/accel0`).
    pub fn open() -> Result<Device> {
        let path =
            crate::knobs::string("ROCKET_DEV").unwrap_or_else(|| "/dev/accel/accel0".to_string());
        Device::open_at(Path::new(&path))
    }

    /// Open a specific accel device node.
    pub fn open_at(path: &Path) -> Result<Device> {
        let cpath = CString::new(path.as_os_str().as_encoded_bytes())
            .map_err(|_| Error::io("open", libc::EINVAL))?;
        let fd = unsafe { libc::open(cpath.as_ptr(), libc::O_RDWR | libc::O_CLOEXEC) };
        if fd < 0 {
            let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
            return Err(Error::device("open", errno).with_detail(format!("{}", path.display())));
        }
        // SAFETY: fd is a fresh, uniquely owned descriptor.
        let fd = unsafe { OwnedFd::from_raw_fd(fd) };

        let mut name = [0u8; 32];
        let mut dv = uapi::DrmVersion {
            name_len: name.len() - 1,
            name: name.as_mut_ptr().cast(),
            ..Default::default()
        };
        let rc = unsafe { libc::ioctl(fd.as_raw_fd(), uapi::DRM_IOCTL_VERSION, &mut dv) };
        if rc < 0 {
            let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
            return Err(Error::device("DRM_IOCTL_VERSION", errno));
        }
        let end = name.iter().position(|b| *b == 0).unwrap_or(name.len());
        let driver = String::from_utf8_lossy(&name[..end]);
        if driver != "rocket" {
            return Err(Error::device("open", libc::ENODEV).with_detail(format!(
                "{} is '{driver}', expected 'rocket'",
                path.display()
            )));
        }
        log_debug!(
            "opened {driver} at {} ({}.{}.{})",
            path.display(),
            dv.version_major,
            dv.version_minor,
            dv.version_patchlevel
        );

        let version = Version {
            major: dv.version_major,
            minor: dv.version_minor,
            patch: dv.version_patchlevel,
        };
        let caps = Caps {
            version,
            batched: batched_supported(version),
            no_dpu_done: false,
            ppu_done: version.at_least_minor(3),
            ranges: version.at_least_minor(5),
            batch_completion_tracked: version.at_least_minor(4),
        };
        Ok(Device {
            inner: Arc::new(DeviceInner { fd, caps }),
        })
    }

    pub fn caps(&self) -> Caps {
        self.inner.caps
    }

    /// The kernel's driver name (`"rocket"` for a successfully opened device).
    pub fn driver_name(&self) -> &'static str {
        "rocket"
    }

    /// A fresh buffer object of `size` bytes.
    pub fn alloc_bo(&self, size: usize) -> Result<Bo> {
        Bo::new(&self.inner, size)
    }

    /// A buffer object whose last byte lies in the low 4 GiB of NPU IOVA, as every
    /// 32-bit regcmd address field requires.
    pub fn alloc_bo32(&self, size: usize) -> Result<Bo> {
        Bo::new_32(&self.inner, size)
    }

    /// Submit one job with `tasks`. Callers build the register programs; this is the
    /// raw seam used by the emitter tests and the matmul engine.
    pub fn submit(
        &self,
        tasks: &[uapi::DrmRocketTask],
        in_handles: &[Handle],
        out_handles: &[Handle],
        flags: u32,
    ) -> Result<()> {
        submit::submit(&self.inner, tasks, in_handles, out_handles, flags)
    }

    #[allow(dead_code)] // used by the matmul context
    pub(crate) fn inner(&self) -> &Arc<DeviceInner> {
        &self.inner
    }
}

fn batched_supported(version: Version) -> bool {
    if let Ok(s) = std::fs::read_to_string("/sys/module/rocket/parameters/rocket_batch_submit") {
        // The parameter exists only on a kernel carrying the batched-submit patch;
        // its value is the master switch.
        return s.trim().parse::<i64>().map(|v| v != 0).unwrap_or(false);
    }
    version.at_least_minor(1)
}

/// `CLOCK_MONOTONIC` in nanoseconds.
pub(crate) fn now_ns() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
}
