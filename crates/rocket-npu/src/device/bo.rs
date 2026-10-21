//! Buffer objects (GEM handles with a persistent CPU mapping).

use std::ptr::NonNull;
use std::sync::Arc;

use super::uapi::{self, Handle};
use super::{DeviceInner, now_ns};
use crate::device::fence;
use crate::error::{Error, Result};
use crate::log::{log_error, log_warn};

/// The identity of a buffer object for the compute routines: its NPU address and GEM
/// handle. Copying it avoids borrowing the BO while other fields of the scratch are
/// mutably borrowed.
#[derive(Debug, Clone, Copy)]
pub struct MemRef {
    pub dma: u64,
    pub handle: Handle,
}

/// A byte range inside a buffer object for the ranged cache-maintenance ioctls.
#[derive(Debug, Clone, Copy)]
pub struct BoRange {
    pub offset: u64,
    pub size: u64,
}

impl From<BoRange> for uapi::DrmRocketBoRange {
    fn from(r: BoRange) -> Self {
        Self {
            offset: r.offset,
            size: r.size,
        }
    }
}

/// One NPU buffer: a GEM handle plus its CPU mapping.
///
/// The object is `Send + Sync` because the driver allows a file's ioctls from any
/// thread and the mapping is stable; mutation is gated by `&mut self`, and the
/// matmul engine gives each worker exclusive ownership of its scratch.
pub struct Bo {
    dev: Arc<DeviceInner>,
    handle: Handle,
    dma_address: u64,
    size: usize,
    ptr: NonNull<u8>,
}

// SAFETY: the mapping is process-wide and the API prevents concurrent mutation.
unsafe impl Send for Bo {}
unsafe impl Sync for Bo {}

impl Bo {
    pub(crate) fn new(dev: &Arc<DeviceInner>, size: usize) -> Result<Bo> {
        if size == 0 || size > u32::MAX as usize {
            return Err(Error::shape("alloc_bo"));
        }
        let mut create = uapi::DrmRocketCreateBo {
            size: size as u32,
            ..Default::default()
        };
        let rc = unsafe { libc::ioctl(dev.fd(), uapi::DRM_IOCTL_ROCKET_CREATE_BO, &mut create) };
        if rc < 0 {
            let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
            log_error!(
                "ROCKET_CREATE_BO({size}): {}",
                std::io::Error::from_raw_os_error(errno)
            );
            return Err(Error::device("alloc_bo", errno));
        }

        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                dev.fd(),
                create.offset as libc::off_t,
            )
        };
        if ptr == libc::MAP_FAILED {
            let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
            log_error!("mmap bo: {}", std::io::Error::from_raw_os_error(errno));
            let mut gc = uapi::DrmGemClose {
                handle: create.handle,
                pad: 0,
            };
            unsafe { libc::ioctl(dev.fd(), uapi::DRM_IOCTL_GEM_CLOSE, &mut gc) };
            return Err(Error::device("alloc_bo/mmap", errno));
        }

        Ok(Bo {
            dev: Arc::clone(dev),
            handle: create.handle,
            dma_address: create.dma_address,
            size,
            // SAFETY: mmap returned a valid, non-null mapping.
            ptr: unsafe { NonNull::new_unchecked(ptr.cast()) },
        })
    }

    pub(crate) fn new_32(dev: &Arc<DeviceInner>, size: usize) -> Result<Bo> {
        let bo = Bo::new(dev, size)?;
        if size != 0 && (bo.dma_address + size as u64 - 1) >> 32 != 0 {
            let lo = bo.dma_address;
            let hi = bo.dma_address + size as u64;
            log_error!(
                "alloc_bo32({size}): IOVA {lo:#x}..{hi:#x} leaves the low 4 GiB the \
                 regcmd address fields can encode"
            );
            drop(bo);
            return Err(Error::device("alloc_bo32", 0));
        }
        Ok(bo)
    }

    pub fn handle(&self) -> Handle {
        self.handle
    }

    /// The owning device (cloned handle) for submits issued while borrowing this BO.
    pub(crate) fn device(&self) -> super::Device {
        super::Device {
            inner: Arc::clone(&self.dev),
        }
    }

    pub fn mem_ref(&self) -> MemRef {
        MemRef {
            dma: self.dma_address,
            handle: self.handle,
        }
    }

    pub fn dma_address(&self) -> u64 {
        self.dma_address
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.size) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size) }
    }

    /// Raw host pointer. Callers may write through it only while no NPU job that reads
    /// or writes the object is in flight; the typed slices are the normal interface.
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Wait for the NPU's writes to this object and sync it for CPU access.
    ///
    /// `timeout_ns` is relative; `0` means a non-blocking poll/cache sync. The kernel
    /// takes an absolute deadline, which this converts. A negative result other than
    /// `EBUSY`/`ETIMEDOUT` is returned as a device error.
    pub fn prep(&mut self, timeout_ns: u64) -> Result<()> {
        if timeout_ns == 0 {
            return self.prep_wait(0);
        }
        let t0 = now_ns();
        let rc = self.prep_wait(timeout_ns);
        fence::note_wait(now_ns().saturating_sub(t0), self.handle);
        rc
    }

    /// Non-blocking completion probe: 0 when the writes have landed, `EBUSY` while
    /// the job is still in flight.
    pub fn poll(&mut self) -> Result<()> {
        self.prep_wait(0)
    }

    fn prep_wait(&mut self, timeout_ns: u64) -> Result<()> {
        if timeout_ns != 0 {
            let spin_us = fence::busy_poll_us();
            if spin_us > 0 {
                let deadline = now_ns() + spin_us * 1000;
                loop {
                    match self.prep_poll() {
                        Ok(()) => return Ok(()),
                        Err(e) if e.errno() == libc::EBUSY => {}
                        Err(e) => return Err(e),
                    }
                    if now_ns() >= deadline {
                        break;
                    }
                    std::hint::spin_loop();
                }
            }
        }

        let deadline = if timeout_ns == 0 {
            0
        } else {
            let now = now_ns() as i64;
            if timeout_ns > (i64::MAX - now) as u64 {
                i64::MAX
            } else {
                now + timeout_ns as i64
            }
        };
        let mut prep = uapi::DrmRocketPrepBo {
            handle: self.handle,
            reserved: 0,
            timeout_ns: deadline,
        };
        let mut rc;
        loop {
            rc = unsafe { libc::ioctl(self.dev.fd(), uapi::DRM_IOCTL_ROCKET_PREP_BO, &mut prep) };
            if rc >= 0 || std::io::Error::last_os_error().raw_os_error() != Some(libc::EINTR) {
                break;
            }
        }
        if rc < 0 {
            let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
            if timeout_ns == 0 {
                log_error!(
                    "bo prep sync handle={}: {}",
                    self.handle,
                    std::io::Error::from_raw_os_error(errno)
                );
            }
            return Err(Error::device("bo_prep", errno));
        }
        Ok(())
    }

    fn prep_poll(&mut self) -> Result<()> {
        let mut prep = uapi::DrmRocketPrepBo {
            handle: self.handle,
            reserved: 0,
            timeout_ns: 0,
        };
        let rc = unsafe { libc::ioctl(self.dev.fd(), uapi::DRM_IOCTL_ROCKET_PREP_BO, &mut prep) };
        if rc < 0 {
            let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
            return Err(Error::device("bo_poll", errno));
        }
        Ok(())
    }

    /// The ranged wait; falls back to the whole-object sync when the kernel predates
    /// interface 1.5.
    pub fn prep_ranges(&mut self, ranges: &[BoRange], timeout_ns: u64) -> Result<()> {
        if ranges.is_empty() || !self.dev.caps().ranges {
            return self.prep(timeout_ns);
        }
        let t0 = now_ns();
        let rc = self.prep_ranges_wait(ranges, timeout_ns);
        if timeout_ns != 0 {
            fence::note_wait(now_ns().saturating_sub(t0), self.handle);
        }
        rc
    }

    fn prep_ranges_wait(&mut self, ranges: &[BoRange], timeout_ns: u64) -> Result<()> {
        let raw: Vec<uapi::DrmRocketBoRange> = ranges.iter().copied().map(Into::into).collect();
        let deadline = if timeout_ns == 0 {
            0
        } else {
            let now = now_ns() as i64;
            if timeout_ns > (i64::MAX - now) as u64 {
                i64::MAX
            } else {
                now + timeout_ns as i64
            }
        };
        let mut prep = uapi::DrmRocketPrepBoRanges {
            handle: self.handle,
            range_count: raw.len() as u32,
            ranges: raw.as_ptr() as u64,
            timeout_ns: deadline,
            reserved: 0,
        };
        let mut rc;
        loop {
            rc = unsafe {
                libc::ioctl(
                    self.dev.fd(),
                    uapi::DRM_IOCTL_ROCKET_PREP_BO_RANGES,
                    &mut prep,
                )
            };
            if rc >= 0 || std::io::Error::last_os_error().raw_os_error() != Some(libc::EINTR) {
                break;
            }
        }
        if rc < 0 {
            let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
            return Err(Error::device("bo_prep_ranges", errno));
        }
        Ok(())
    }

    /// Hand the object's CPU writes back to the device.
    pub fn fini(&mut self) -> Result<()> {
        let mut fini = uapi::DrmRocketFiniBo {
            handle: self.handle,
            reserved: 0,
        };
        if unsafe { libc::ioctl(self.dev.fd(), uapi::DRM_IOCTL_ROCKET_FINI_BO, &mut fini) } < 0 {
            let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
            log_error!(
                "bo fini handle={}: {}",
                self.handle,
                std::io::Error::from_raw_os_error(errno)
            );
            return Err(Error::device("bo_fini", errno));
        }
        Ok(())
    }

    /// Ranged form of [`Bo::fini`]; falls back to the whole-object sync.
    pub fn fini_ranges(&mut self, ranges: &[BoRange]) -> Result<()> {
        if ranges.is_empty() || !self.dev.caps().ranges {
            return self.fini();
        }
        let raw: Vec<uapi::DrmRocketBoRange> = ranges.iter().copied().map(Into::into).collect();
        let mut fini = uapi::DrmRocketFiniBoRanges {
            handle: self.handle,
            range_count: raw.len() as u32,
            ranges: raw.as_ptr() as u64,
            reserved: 0,
        };
        if unsafe {
            libc::ioctl(
                self.dev.fd(),
                uapi::DRM_IOCTL_ROCKET_FINI_BO_RANGES,
                &mut fini,
            )
        } < 0
        {
            let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
            log_error!(
                "bo fini_ranges handle={}: {}",
                self.handle,
                std::io::Error::from_raw_os_error(errno)
            );
            return Err(Error::device("bo_fini_ranges", errno));
        }
        Ok(())
    }
}

impl Drop for Bo {
    fn drop(&mut self) {
        let rc = unsafe { libc::munmap(self.ptr.as_ptr().cast(), self.size) };
        if rc < 0 {
            let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
            log_warn!(
                "munmap(handle={}, size={}): {}",
                self.handle,
                self.size,
                std::io::Error::from_raw_os_error(errno)
            );
        }
        let mut gc = uapi::DrmGemClose {
            handle: self.handle,
            pad: 0,
        };
        if unsafe { libc::ioctl(self.dev.fd(), uapi::DRM_IOCTL_GEM_CLOSE, &mut gc) } < 0 {
            let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
            log_warn!(
                "GEM_CLOSE(handle={}): {} — leaking GEM handle/IOVA",
                self.handle,
                std::io::Error::from_raw_os_error(errno)
            );
        }
    }
}
