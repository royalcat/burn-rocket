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
use std::sync::{Arc, OnceLock};

use crate::error::{Error, Result};
use crate::log::log_debug;

pub use bo::{Bo, BoRange, MemRef};
pub use submit::Job;
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

    /// One job carrying `tasks`, with no `DRM_ROCKET_JOB_*` flags — the named alias of
    /// [`Device::submit`] (C `rocket_submit_tasks`).
    pub fn submit_tasks(
        &self,
        tasks: &[uapi::DrmRocketTask],
        in_handles: &[Handle],
        out_handles: &[Handle],
    ) -> Result<()> {
        submit::submit_tasks(&self.inner, tasks, in_handles, out_handles)
    }

    /// As [`Device::submit_tasks`], but copying the descriptors into a caller-owned
    /// `scratch` buffer first (C `rocket_submit_tasks_pre`). Rust's slice-based submit
    /// already builds the descriptor array on the stack, so this form exists for API
    /// parity rather than for speed.
    pub fn submit_tasks_pre(
        &self,
        scratch: &mut Vec<uapi::DrmRocketTask>,
        tasks: &[uapi::DrmRocketTask],
        in_handles: &[Handle],
        out_handles: &[Handle],
        flags: u32,
    ) -> Result<()> {
        submit::submit_tasks_pre(&self.inner, scratch, tasks, in_handles, out_handles, flags)
    }

    /// Submit one job per element of `jobs` in a single ioctl (C `rocket_submit_jobs`).
    ///
    /// A job's tasks run sequentially on one NPU core; several jobs in one submit are
    /// scheduled across the three cores, so real concurrency needs each job to write a
    /// separate output BO (a shared output BO serialises on the write-after-write
    /// dependency). Waits are per output BO, as with any other async submit.
    pub fn submit_jobs(&self, jobs: &[Job<'_>]) -> Result<()> {
        submit::submit_jobs(&self.inner, jobs)
    }

    /// Whether the running kernel honors `DRM_ROCKET_JOB_BATCHED`
    /// (C `rocket_batched_submit_supported`).
    pub fn batched_submit_supported(&self) -> bool {
        self.inner.caps.batched
    }

    /// Whether chained submit is this provider's *native* mode
    /// (C `rocket_batched_submit_native`).
    ///
    /// Always false here: the stock driver runs a job's tasks one per IRQ, so the
    /// contiguous self-chained layout stays an opt-in extension
    /// (`ROCKET_BATCH_SUBMIT=1`, see `matmul::chain`).
    pub fn batched_submit_native(&self) -> bool {
        false
    }

    /// Whether one submit is one job, i.e. the kernel keeps a core on a job's whole task
    /// sequence (C `rocket_submit_batch_atomic`).
    ///
    /// Structural on this driver, not versioned: `core->in_flight_job` is held across the
    /// tasks and the done fence is only signalled once the last task retires, so CBUF
    /// operand reuse holds across a batch without the caller doing anything.
    pub fn submit_batch_atomic(&self) -> bool {
        true
    }

    /// Whether a batched job's completion wait is armed on the task counter rather than
    /// on a per-task `PC_DONE` deadline (interface 1.4+, C
    /// `rocket_batch_completion_tracked`).
    pub fn batch_completion_tracked(&self) -> bool {
        self.inner.caps.batch_completion_tracked
    }

    /// Whether `DRM_ROCKET_JOB_NO_DPU_DONE` may be sent (RK3576 only; false here,
    /// C `rocket_no_dpu_done_supported`). The submit ioctl rejects unknown flags, so
    /// this is a safety gate, not advice.
    pub fn no_dpu_done_supported(&self) -> bool {
        self.inner.caps.no_dpu_done
    }

    /// Whether `DRM_ROCKET_JOB_PPU_DONE` may be sent (interface 1.3+,
    /// C `rocket_ppu_done_supported`).
    pub fn ppu_done_supported(&self) -> bool {
        self.inner.caps.ppu_done
    }

    /// Whether the ranged cache-maintenance ioctls exist (interface 1.5+,
    /// C `rocket_bo_ranges_supported`).
    pub fn bo_ranges_supported(&self) -> bool {
        self.inner.caps.ranges
    }

    #[allow(dead_code)] // used by the matmul context
    pub(crate) fn inner(&self) -> &Arc<DeviceInner> {
        &self.inner
    }
}

fn batched_supported(version: Version) -> bool {
    match batch_submit_param() {
        Some(v) => v,
        // The parameter exists only on a kernel carrying the batched-submit patch; a
        // kernel that implements the flag without the kill switch still reports >= 1.1.
        None => version.at_least_minor(1),
    }
}

/// The kernel's `rocket_batch_submit` module parameter, if the patch is present.
///
/// Its mere existence proves `DRM_ROCKET_JOB_BATCHED` is understood; its value is the
/// master switch (`0` forces every job down the per-task path, which is exactly the mix
/// userspace must not self-chain into).
fn batch_submit_param() -> Option<bool> {
    let s = std::fs::read_to_string("/sys/module/rocket/parameters/rocket_batch_submit").ok()?;
    Some(s.trim().parse::<i64>().map(|v| v != 0).unwrap_or(false))
}

/// Whether the running kernel honors `DRM_ROCKET_JOB_BATCHED`, probed without an open
/// [`Device`] (C `rocket_batched_submit_supported`).
///
/// Reads `/sys/module/rocket/parameters/rocket_batch_submit` first and otherwise falls
/// back to the advertised DRM interface version (`>= 1.1`), which needs a device node.
/// Probed once and cached, like the C code, so it is safe to call from a hot path —
/// `matmul::chain::enabled` does, because it must answer without a device.
pub fn batched_submit_supported_global() -> bool {
    static SUPPORTED: OnceLock<bool> = OnceLock::new();
    *SUPPORTED.get_or_init(|| {
        if let Some(forced) = batch_submit_param() {
            if !forced {
                log_debug!(
                    "rocket_batch_submit=0: the kernel forces every job to the per-task \
                     path, so chained submit stays off"
                );
            }
            return forced;
        }
        probe_version().is_some_and(|v| v.at_least_minor(1))
    })
}

/// The interface version the running kernel advertises, without keeping a [`Device`]
/// open (C `rocket_iface_minor`). `None` when no `rocket` device can be opened.
fn probe_version() -> Option<Version> {
    static VERSION: OnceLock<Option<Version>> = OnceLock::new();
    *VERSION.get_or_init(|| Device::open().ok().map(|d| d.caps().version))
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
