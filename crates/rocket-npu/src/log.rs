//! The crate's diagnostic channel: stderr, filtered by `ROCKET_LOG_LEVEL`.
//!
//! Defaults match the C library: info threshold, stderr sink. `ROCKET_DEBUG` raises
//! the threshold to debug. The output wording is allowed to differ from the C
//! library, but the level gating and the env names are the same.

use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    Error = 0,
    Warn = 1,
    Info = 2,
    Debug = 3,
}

fn resolve_level() -> Level {
    let mut level = match crate::knobs::string("ROCKET_LOG_LEVEL").as_deref() {
        Some("error" | "0") => Level::Error,
        Some("warn" | "1") => Level::Warn,
        Some("debug" | "3") => Level::Debug,
        _ => Level::Info,
    };
    if crate::knobs::present("ROCKET_DEBUG") {
        level = Level::Debug;
    }
    level
}

fn level() -> Level {
    static LEVEL: OnceLock<Level> = OnceLock::new();
    *LEVEL.get_or_init(resolve_level)
}

pub(crate) fn log(at: Level, args: std::fmt::Arguments<'_>) {
    if at > level() {
        return;
    }
    let tag = match at {
        Level::Error => "error",
        Level::Warn => "warn",
        Level::Info => "info",
        Level::Debug => "debug",
    };
    eprintln!("rocket {tag}: {args}");
}

macro_rules! log_error {
    ($($arg:tt)*) => {
        $crate::log::log($crate::log::Level::Error, format_args!($($arg)*))
    };
}
macro_rules! log_warn {
    ($($arg:tt)*) => {
        $crate::log::log($crate::log::Level::Warn, format_args!($($arg)*))
    };
}
macro_rules! log_debug {
    ($($arg:tt)*) => {
        $crate::log::log($crate::log::Level::Debug, format_args!($($arg)*))
    };
}

pub(crate) use {log_debug, log_error, log_warn};
