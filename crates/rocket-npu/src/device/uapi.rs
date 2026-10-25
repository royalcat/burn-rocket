//! The kernel uAPI used by the mainline `rocket` DRM-accel driver.
//!
//! These definitions are the subset of `<drm/rocket_accel.h>` (Linux, MIT) and
//! `<drm/drm.h>` this crate drives. They are written here rather than generated so a
//! build needs no kernel headers; the layout tests pin the ABI sizes and ioctl
//! numbers against the values the kernel advertises.

use std::ffi::c_char;

pub type Handle = u32;

pub const DRM_IOCTL_BASE: u8 = b'd';
pub const DRM_COMMAND_BASE: u8 = 0x40;

const IOC_WRITE: u32 = 1;
const IOC_READ: u32 = 2;
const IOC_READ_WRITE: u32 = IOC_READ | IOC_WRITE;

/// `_IOC(dir, type, nr, size)` from `<asm-generic/ioctl.h>` for a 64-bit target.
const fn ioc(dir: u32, ty: u8, nr: u8, size: usize) -> libc::c_ulong {
    ((dir as libc::c_ulong) << 30)
        | ((size as libc::c_ulong) << 16)
        | ((ty as libc::c_ulong) << 8)
        | (nr as libc::c_ulong)
}

const fn iow(nr: u8, size: usize) -> libc::c_ulong {
    ioc(IOC_WRITE, DRM_IOCTL_BASE, nr, size)
}

const fn iowr(nr: u8, size: usize) -> libc::c_ulong {
    ioc(IOC_READ_WRITE, DRM_IOCTL_BASE, nr, size)
}

/// `DRM_IOCTL_VERSION`.
pub const DRM_IOCTL_VERSION: libc::c_ulong = iowr(0x00, std::mem::size_of::<DrmVersion>());
/// `DRM_IOCTL_GEM_CLOSE`.
pub const DRM_IOCTL_GEM_CLOSE: libc::c_ulong = iow(0x09, std::mem::size_of::<DrmGemClose>());
/// `DRM_IOCTL_ROCKET_CREATE_BO`.
pub const DRM_IOCTL_ROCKET_CREATE_BO: libc::c_ulong =
    iowr(DRM_COMMAND_BASE, std::mem::size_of::<DrmRocketCreateBo>());
/// `DRM_IOCTL_ROCKET_SUBMIT`.
pub const DRM_IOCTL_ROCKET_SUBMIT: libc::c_ulong = iow(
    DRM_COMMAND_BASE + 0x01,
    std::mem::size_of::<DrmRocketSubmit>(),
);
/// `DRM_IOCTL_ROCKET_PREP_BO`.
pub const DRM_IOCTL_ROCKET_PREP_BO: libc::c_ulong = iow(
    DRM_COMMAND_BASE + 0x02,
    std::mem::size_of::<DrmRocketPrepBo>(),
);
/// `DRM_IOCTL_ROCKET_FINI_BO`.
pub const DRM_IOCTL_ROCKET_FINI_BO: libc::c_ulong = iow(
    DRM_COMMAND_BASE + 0x03,
    std::mem::size_of::<DrmRocketFiniBo>(),
);
/// `DRM_IOCTL_ROCKET_PREP_BO_RANGES` (interface 1.5).
pub const DRM_IOCTL_ROCKET_PREP_BO_RANGES: libc::c_ulong = iow(
    DRM_COMMAND_BASE + 0x04,
    std::mem::size_of::<DrmRocketPrepBoRanges>(),
);
/// `DRM_IOCTL_ROCKET_FINI_BO_RANGES` (interface 1.5).
pub const DRM_IOCTL_ROCKET_FINI_BO_RANGES: libc::c_ulong = iow(
    DRM_COMMAND_BASE + 0x05,
    std::mem::size_of::<DrmRocketFiniBoRanges>(),
);

pub const DRM_ROCKET_JOB_BATCHED: u32 = 1 << 0;
pub const DRM_ROCKET_JOB_NO_DPU_DONE: u32 = 1 << 1;
pub const DRM_ROCKET_JOB_PPU_DONE: u32 = 1 << 2;

/// `struct drm_version`.
#[repr(C)]
pub struct DrmVersion {
    pub version_major: i32,
    pub version_minor: i32,
    pub version_patchlevel: i32,
    pub name_len: usize,
    pub name: *mut c_char,
    pub date_len: usize,
    pub date: *mut c_char,
    pub desc_len: usize,
    pub desc: *mut c_char,
}

impl Default for DrmVersion {
    fn default() -> Self {
        Self {
            version_major: 0,
            version_minor: 0,
            version_patchlevel: 0,
            name_len: 0,
            name: std::ptr::null_mut(),
            date_len: 0,
            date: std::ptr::null_mut(),
            desc_len: 0,
            desc: std::ptr::null_mut(),
        }
    }
}

/// `struct drm_gem_close`.
#[repr(C)]
#[derive(Default)]
pub struct DrmGemClose {
    pub handle: Handle,
    pub pad: u32,
}

/// `struct drm_rocket_create_bo`.
#[repr(C)]
#[derive(Default)]
pub struct DrmRocketCreateBo {
    pub size: u32,
    pub handle: Handle,
    pub dma_address: u64,
    pub offset: u64,
}

/// `struct drm_rocket_task`.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct DrmRocketTask {
    pub regcmd: u32,
    pub regcmd_count: u32,
}

/// `struct drm_rocket_job` plus its trailing `flags`/`reserved` (interface 1.1+).
///
/// Every submit sends the full 48-byte ABI, regardless of which header the kernel was
/// built against: a kernel that predates the tail copies the fields it knows and
/// strides past the rest, while declaring the shorter size to a newer kernel trips
/// its size guard.
#[repr(C)]
#[derive(Default)]
pub struct DrmRocketJob {
    pub tasks: u64,
    pub in_bo_handles: u64,
    pub out_bo_handles: u64,
    pub task_count: u32,
    pub task_struct_size: u32,
    pub in_bo_handle_count: u32,
    pub out_bo_handle_count: u32,
    pub flags: u32,
    pub reserved: u32,
}

/// `struct drm_rocket_submit`.
#[repr(C)]
#[derive(Default)]
pub struct DrmRocketSubmit {
    pub jobs: u64,
    pub job_count: u32,
    pub job_struct_size: u32,
    pub reserved: u64,
}

/// `struct drm_rocket_prep_bo`.
#[repr(C)]
#[derive(Default)]
pub struct DrmRocketPrepBo {
    pub handle: Handle,
    pub reserved: u32,
    pub timeout_ns: i64,
}

/// `struct drm_rocket_fini_bo`.
#[repr(C)]
#[derive(Default)]
pub struct DrmRocketFiniBo {
    pub handle: Handle,
    pub reserved: u32,
}

/// `struct drm_rocket_bo_range`.
#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct DrmRocketBoRange {
    pub offset: u64,
    pub size: u64,
}

/// `struct drm_rocket_prep_bo_ranges`.
#[repr(C)]
#[derive(Default)]
pub struct DrmRocketPrepBoRanges {
    pub handle: Handle,
    pub range_count: u32,
    pub ranges: u64,
    pub timeout_ns: i64,
    pub reserved: u64,
}

/// `struct drm_rocket_fini_bo_ranges`.
#[repr(C)]
#[derive(Default)]
pub struct DrmRocketFiniBoRanges {
    pub handle: Handle,
    pub range_count: u32,
    pub ranges: u64,
    pub reserved: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;

    #[test]
    fn abi_sizes() {
        assert_eq!(size_of::<DrmVersion>(), 64);
        assert_eq!(size_of::<DrmGemClose>(), 8);
        assert_eq!(size_of::<DrmRocketCreateBo>(), 24);
        assert_eq!(size_of::<DrmRocketTask>(), 8);
        assert_eq!(size_of::<DrmRocketJob>(), 48);
        assert_eq!(size_of::<DrmRocketSubmit>(), 24);
        assert_eq!(size_of::<DrmRocketPrepBo>(), 16);
        assert_eq!(size_of::<DrmRocketFiniBo>(), 8);
        assert_eq!(size_of::<DrmRocketBoRange>(), 16);
        assert_eq!(size_of::<DrmRocketPrepBoRanges>(), 32);
        assert_eq!(size_of::<DrmRocketFiniBoRanges>(), 24);
    }

    #[test]
    fn ioctl_numbers_match_linux() {
        // Values the DRM core and the rocket driver expose on a 64-bit kernel.
        assert_eq!(DRM_IOCTL_VERSION, 0xc040_6400);
        assert_eq!(DRM_IOCTL_GEM_CLOSE, 0x4008_6409);
        assert_eq!(DRM_IOCTL_ROCKET_CREATE_BO, 0xc018_6440);
        assert_eq!(DRM_IOCTL_ROCKET_SUBMIT, 0x4018_6441);
        assert_eq!(DRM_IOCTL_ROCKET_PREP_BO, 0x4010_6442);
        assert_eq!(DRM_IOCTL_ROCKET_FINI_BO, 0x4008_6443);
        assert_eq!(DRM_IOCTL_ROCKET_PREP_BO_RANGES, 0x4020_6444);
        assert_eq!(DRM_IOCTL_ROCKET_FINI_BO_RANGES, 0x4018_6445);
    }
}
