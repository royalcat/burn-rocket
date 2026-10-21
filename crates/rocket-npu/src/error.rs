//! Typed failures with the C library's status codes.
//!
//! The C API returned `0` or a negative `ROCKET_E_*` code (or `-errno`). Callers in
//! `burn-rocket` already branch on those codes, so the Rust port keeps them: an
//! [`Error`] exposes [`Error::status`] and the numeric [`Status::code`].

use std::fmt;

/// The stable failure classes of the C library.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Status {
    /// A shape the hardware cannot run (`ROCKET_E_SHAPE`, -1).
    Shape,
    /// A shape the planner cannot tile (`ROCKET_E_TILING`, -2).
    Tiling,
    /// A host or device allocation failed (`ROCKET_E_NOMEM`, -3).
    NoMem,
    /// A driver/ioctl failure (`ROCKET_E_DEVICE`, -4).
    Device,
    /// A request the port does not implement (`ROCKET_E_UNSUPPORTED`, -5).
    Unsupported,
    /// A plain I/O or argument failure outside the C status set.
    Io,
}

impl Status {
    pub const fn code(self) -> i32 {
        match self {
            Status::Shape => -1,
            Status::Tiling => -2,
            Status::NoMem => -3,
            Status::Device => -4,
            Status::Unsupported => -5,
            Status::Io => -6,
        }
    }

    pub const fn from_code(code: i32) -> Option<Status> {
        match code {
            -1 => Some(Status::Shape),
            -2 => Some(Status::Tiling),
            -3 => Some(Status::NoMem),
            -4 => Some(Status::Device),
            -5 => Some(Status::Unsupported),
            -6 => Some(Status::Io),
            _ => None,
        }
    }
}

/// One failed operation: what was attempted, which class, and (for device failures)
/// the errno the kernel returned.
#[derive(Debug, Clone)]
pub struct Error {
    op: &'static str,
    status: Status,
    errno: i32,
    detail: Option<String>,
}

impl Error {
    pub fn new(status: Status, op: &'static str, errno: i32) -> Self {
        Self {
            op,
            status,
            errno,
            detail: None,
        }
    }

    pub fn shape(op: &'static str) -> Self {
        Self::new(Status::Shape, op, 0)
    }

    pub fn tiling(op: &'static str) -> Self {
        Self::new(Status::Tiling, op, 0)
    }

    pub fn nomem(op: &'static str) -> Self {
        Self::new(Status::NoMem, op, 0)
    }

    pub fn device(op: &'static str, errno: i32) -> Self {
        Self::new(Status::Device, op, errno)
    }

    pub fn unsupported(op: &'static str) -> Self {
        Self::new(Status::Unsupported, op, 0)
    }

    pub fn io(op: &'static str, errno: i32) -> Self {
        Self::new(Status::Io, op, errno)
    }

    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = Some(detail.into());
        self
    }

    pub fn status(&self) -> Status {
        self.status
    }

    pub fn op(&self) -> &'static str {
        self.op
    }

    pub fn errno(&self) -> i32 {
        self.errno
    }

    pub fn detail(&self) -> Option<&str> {
        self.detail.as_deref()
    }

    /// A fallback-eligible failure: the caller may run the same op on the CPU.
    pub fn is_fallback(&self) -> bool {
        matches!(self.status, Status::Shape | Status::Tiling)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let what = match self.status {
            Status::Shape => "unsupported shape",
            Status::Tiling => "shape not tileable",
            Status::NoMem => "out of memory",
            Status::Device => "device failure",
            Status::Unsupported => "unsupported feature",
            Status::Io => "I/O failure",
        };
        write!(f, "{}: {what}", self.op)?;
        if self.errno != 0 {
            let s = std::io::Error::from_raw_os_error(self.errno);
            write!(f, " ({s})")?;
        }
        if let Some(d) = &self.detail {
            write!(f, ": {d}")?;
        }
        Ok(())
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Error::io("io", value.raw_os_error().unwrap_or(0))
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_match_the_c_header() {
        assert_eq!(Status::Shape.code(), -1);
        assert_eq!(Status::Tiling.code(), -2);
        assert_eq!(Status::NoMem.code(), -3);
        assert_eq!(Status::Device.code(), -4);
        assert_eq!(Status::Unsupported.code(), -5);
        for s in [
            Status::Shape,
            Status::Tiling,
            Status::NoMem,
            Status::Device,
            Status::Unsupported,
            Status::Io,
        ] {
            assert_eq!(Status::from_code(s.code()), Some(s));
        }
    }

    #[test]
    fn fallback_covers_shape_and_tiling_only() {
        assert!(Error::shape("x").is_fallback());
        assert!(Error::tiling("x").is_fallback());
        assert!(!Error::nomem("x").is_fallback());
        assert!(!Error::device("x", 19).is_fallback());
    }
}
