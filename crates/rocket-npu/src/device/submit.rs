//! Job submission and the process-wide submit counters.
//!
//! One `ROCKET_SUBMIT` ioctl carries one or more jobs. A job's tasks run sequentially
//! on one NPU core; several jobs in one submit are scheduled across the cores. The C
//! library's three entry points are all here:
//!
//! - `submit` — one job, `flags` and all (the raw seam; [`super::Device::submit`]);
//! - `submit_tasks` / `submit_tasks_pre` — the same thing under the C names, the `_pre`
//!   form writing into a caller-owned descriptor buffer;
//! - `submit_jobs` — N jobs in one ioctl ([`super::Device::submit_jobs`]).
//!
//! Chaining (the contiguous self-linked regcmd layout) is a *layout* property of the
//! tasks the caller built, not something this module does: see `matmul::chain`.

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

/// One job of a multi-job submit: its tasks (sequential on one NPU core) and the
/// buffer objects the job reads and writes — the C `rocket_job_desc`.
///
/// For real concurrency across cores each job must write a SEPARATE output BO: a shared
/// output BO serialises the jobs on the write-after-write dependency.
#[derive(Debug, Clone, Copy)]
pub struct Job<'a> {
    pub tasks: &'a [DrmRocketTask],
    pub in_handles: &'a [Handle],
    pub out_handles: &'a [Handle],
    /// `DRM_ROCKET_JOB_*` bits for this job.
    ///
    /// The C `rocket_submit_jobs` hardcodes 0 (the stock gapped per-task path). A
    /// chained job needs [`DRM_ROCKET_JOB_BATCHED`] *and* a kernel that honors it —
    /// check [`super::Device::batched_submit_supported`] first, because an older kernel
    /// ignores the bit and runs the self-chained layout down the per-task path, which
    /// stalls. A task-less job is rejected whatever the flags say.
    pub flags: u32,
}

impl<'a> Job<'a> {
    /// A job with no `DRM_ROCKET_JOB_*` flags (the stock gapped per-task path).
    pub fn new(
        tasks: &'a [DrmRocketTask],
        in_handles: &'a [Handle],
        out_handles: &'a [Handle],
    ) -> Job<'a> {
        Job {
            tasks,
            in_handles,
            out_handles,
            flags: 0,
        }
    }

    /// A job with explicit `DRM_ROCKET_JOB_*` flags.
    pub fn with_flags(
        tasks: &'a [DrmRocketTask],
        in_handles: &'a [Handle],
        out_handles: &'a [Handle],
        flags: u32,
    ) -> Job<'a> {
        Job {
            tasks,
            in_handles,
            out_handles,
            flags,
        }
    }

    /// The `drm_rocket_job` descriptor this job submits as.
    fn desc(&self) -> DrmRocketJob {
        job_desc(self.tasks, self.in_handles, self.out_handles, self.flags)
    }
}

/// Build the ABI descriptor for one job.
///
/// Every submit sends the full 48-byte `drm_rocket_job`, whatever header the kernel was
/// built against (see [`DrmRocketJob`]).
fn job_desc(
    tasks: &[DrmRocketTask],
    in_handles: &[Handle],
    out_handles: &[Handle],
    flags: u32,
) -> DrmRocketJob {
    DrmRocketJob {
        tasks: tasks.as_ptr() as u64,
        in_bo_handles: in_handles.as_ptr() as u64,
        out_bo_handles: out_handles.as_ptr() as u64,
        task_count: tasks.len() as u32,
        task_struct_size: std::mem::size_of::<DrmRocketTask>() as u32,
        in_bo_handle_count: in_handles.len() as u32,
        out_bo_handle_count: out_handles.len() as u32,
        flags,
        reserved: 0,
    }
}

/// Issue one `ROCKET_SUBMIT` for `jobs` and bump the counters.
fn ioctl_submit(
    dev: &DeviceInner,
    jobs: &mut [DrmRocketJob],
    what: &'static str,
    n_tasks: u64,
) -> Result<()> {
    let mut submit = uapi::DrmRocketSubmit {
        jobs: jobs.as_mut_ptr() as u64,
        job_count: jobs.len() as u32,
        job_struct_size: std::mem::size_of::<DrmRocketJob>() as u32,
        reserved: 0,
    };

    IOCTL_SUBMITS.fetch_add(1, Ordering::Relaxed);
    SUBMITTED_TASKS.fetch_add(n_tasks, Ordering::Relaxed);
    let rc = unsafe { libc::ioctl(dev.fd(), uapi::DRM_IOCTL_ROCKET_SUBMIT, &mut submit) };
    if rc < 0 {
        let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
        let err = std::io::Error::from_raw_os_error(errno);
        match jobs {
            [job] => log_error!(
                "ROCKET_SUBMIT(1 job, {n_tasks} task(s), flags={:#x}): {err}",
                job.flags
            ),
            many => log_error!(
                "ROCKET_SUBMIT({} jobs, {n_tasks} task(s)): {err}",
                many.len()
            ),
        }
        return Err(Error::device(what, errno));
    }
    Ok(())
}

pub(crate) fn submit(
    dev: &DeviceInner,
    tasks: &[DrmRocketTask],
    in_handles: &[Handle],
    out_handles: &[Handle],
    flags: u32,
) -> Result<()> {
    let mut jobs = [job_desc(tasks, in_handles, out_handles, flags)];
    ioctl_submit(dev, &mut jobs, "submit", tasks.len() as u64)
}

/// One job carrying `tasks`, with no `DRM_ROCKET_JOB_*` flags (C `rocket_submit_tasks`).
///
/// The C shim keeps the stock gapped per-task path; the chained path is opt-in through
/// [`submit`]'s `flags` on the hot path.
pub(crate) fn submit_tasks(
    dev: &DeviceInner,
    tasks: &[DrmRocketTask],
    in_handles: &[Handle],
    out_handles: &[Handle],
) -> Result<()> {
    submit(dev, tasks, in_handles, out_handles, 0)
}

/// Bytes a caller must reserve for `max_tasks` descriptors in `submit_tasks_pre`
/// (C `rocket_submit_scratch_size`).
pub fn submit_scratch_size(max_tasks: u32) -> usize {
    max_tasks as usize * std::mem::size_of::<DrmRocketTask>()
}

/// No-alloc core of [`submit_tasks`]: `scratch` is a caller-owned descriptor buffer,
/// reused across submits (C `rocket_submit_tasks_pre`).
///
/// Every element is fully rewritten, so nothing from a previous, longer submit survives.
/// Rust's slice-based [`submit`] already builds its `drm_rocket_task` array on the stack,
/// so this form exists for API parity with the C library rather than for speed.
pub(crate) fn submit_tasks_pre(
    dev: &DeviceInner,
    scratch: &mut Vec<DrmRocketTask>,
    tasks: &[DrmRocketTask],
    in_handles: &[Handle],
    out_handles: &[Handle],
    flags: u32,
) -> Result<()> {
    scratch.clear();
    scratch.extend_from_slice(tasks);
    submit(dev, scratch, in_handles, out_handles, flags)
}

/// Submit N jobs in ONE ioctl (C `rocket_submit_jobs`).
///
/// Counters: one ioctl, and the task count is the sum over the jobs. An empty `jobs`
/// slice submits nothing at all (no ioctl, no counters), as in the C shim; a job with no
/// tasks is a caller bug and is rejected with [`crate::Status::Shape`] before any ioctl.
pub(crate) fn submit_jobs(dev: &DeviceInner, jobs: &[Job<'_>]) -> Result<()> {
    if jobs.is_empty() {
        return Ok(());
    }
    let mut descs: Vec<DrmRocketJob> = Vec::with_capacity(jobs.len());
    let mut n_tasks = 0u64;
    for (j, job) in jobs.iter().enumerate() {
        if job.tasks.is_empty() {
            // Caught here rather than by an empty allocation: a zero-task job would
            // otherwise reach the kernel as a job that can never complete.
            log_error!("submit_jobs: job {j} carries no tasks");
            return Err(Error::shape("submit_jobs"));
        }
        n_tasks += job.tasks.len() as u64;
        descs.push(job.desc());
    }
    ioctl_submit(dev, &mut descs, "submit_jobs", n_tasks)
}

/// Whether `flags` asks for the chained (batched) execution path.
pub fn is_batched(flags: u32) -> bool {
    flags & DRM_ROCKET_JOB_BATCHED != 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn job_desc_carries_the_abi() {
        let tasks = [
            DrmRocketTask {
                regcmd: 0x1000,
                regcmd_count: 4,
            },
            DrmRocketTask {
                regcmd: 0x2000,
                regcmd_count: 6,
            },
        ];
        let ins = [7u32];
        let outs = [9u32, 11u32];
        let job = Job::new(&tasks, &ins, &outs);
        let d = job.desc();
        assert_eq!(d.tasks, tasks.as_ptr() as u64);
        assert_eq!(d.task_count, 2);
        assert_eq!(
            d.task_struct_size as usize,
            std::mem::size_of::<DrmRocketTask>()
        );
        assert_eq!(d.in_bo_handle_count, 1);
        assert_eq!(d.out_bo_handle_count, 2);
        assert_eq!(d.flags, 0);
        assert_eq!(d.reserved, 0);

        let chained = Job::with_flags(&tasks, &ins, &outs, DRM_ROCKET_JOB_BATCHED);
        assert!(is_batched(chained.desc().flags));
        assert!(!is_batched(job.desc().flags));
    }

    #[test]
    fn scratch_size_matches_the_descriptor() {
        assert_eq!(submit_scratch_size(0), 0);
        assert_eq!(submit_scratch_size(1), 8);
        assert_eq!(submit_scratch_size(64), 512);
    }
}
