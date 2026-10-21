//! Job submission and the process-wide submit counters.

use std::sync::atomic::{AtomicU64, Ordering};

use super::DeviceInner;
use super::uapi::{self, DRM_ROCKET_JOB_BATCHED, DrmRocketJob, DrmRocketTask, Handle};
use crate::error::{Error, Result};
use crate::log::log_error;

static IOCTL_SUBMITS: AtomicU64 = AtomicU64::new(0);
static SUBMITTED_TASKS: AtomicU64 = AtomicU64::new(0);

/// Number of `ROCKET_SUBMIT` ioctls this process has issued.
pub fn ioctl_submit_count() -> u64 {
    IOCTL_SUBMITS.load(Ordering::Relaxed)
}

/// Number of tasks carried by those submits.
pub fn submitted_task_count() -> u64 {
    SUBMITTED_TASKS.load(Ordering::Relaxed)
}

pub fn reset_counters() {
    IOCTL_SUBMITS.store(0, Ordering::Relaxed);
    SUBMITTED_TASKS.store(0, Ordering::Relaxed);
}

pub(crate) fn submit(
    dev: &DeviceInner,
    tasks: &[DrmRocketTask],
    in_handles: &[Handle],
    out_handles: &[Handle],
    flags: u32,
) -> Result<()> {
    let mut job = DrmRocketJob {
        tasks: tasks.as_ptr() as u64,
        in_bo_handles: in_handles.as_ptr() as u64,
        out_bo_handles: out_handles.as_ptr() as u64,
        task_count: tasks.len() as u32,
        task_struct_size: std::mem::size_of::<DrmRocketTask>() as u32,
        in_bo_handle_count: in_handles.len() as u32,
        out_bo_handle_count: out_handles.len() as u32,
        flags,
        reserved: 0,
    };
    let mut submit = uapi::DrmRocketSubmit {
        jobs: (&mut job as *mut DrmRocketJob) as u64,
        job_count: 1,
        job_struct_size: std::mem::size_of::<DrmRocketJob>() as u32,
        reserved: 0,
    };

    IOCTL_SUBMITS.fetch_add(1, Ordering::Relaxed);
    SUBMITTED_TASKS.fetch_add(tasks.len() as u64, Ordering::Relaxed);
    let rc = unsafe { libc::ioctl(dev.fd(), uapi::DRM_IOCTL_ROCKET_SUBMIT, &mut submit) };
    if rc < 0 {
        let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
        log_error!(
            "ROCKET_SUBMIT({} tasks, flags={flags:#x}): {}",
            tasks.len(),
            std::io::Error::from_raw_os_error(errno)
        );
        return Err(Error::device("submit", errno));
    }
    Ok(())
}

/// Whether `flags` asks for the chained (batched) execution path.
pub fn is_batched(flags: u32) -> bool {
    flags & DRM_ROCKET_JOB_BATCHED != 0
}
