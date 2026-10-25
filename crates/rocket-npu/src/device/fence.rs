//! Fenced-wait accounting and the busy-poll budget.
//!
//! The driver retires a job that outlasts its watchdog and signals the fence as if it
//! completed, so a wait's duration is the only trace a hung job leaves in the process.
//! These counters feed the parity harness and `burn_rocket::stats`.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::log::log_warn;

static WAITS: AtomicU64 = AtomicU64::new(0);
static SLOW_WAITS: AtomicU64 = AtomicU64::new(0);
static MAX_WAIT_NS: AtomicU64 = AtomicU64::new(0);

/// Sentinel for "the busy-poll budget has not been resolved yet" (C's `-1`).
const UNRESOLVED: u64 = u64::MAX;
static BUSY_POLL_US: AtomicU64 = AtomicU64::new(UNRESOLVED);

fn slow_mark_ms() -> u64 {
    static MARK: OnceLock<u64> = OnceLock::new();
    *MARK.get_or_init(|| {
        let dflt = crate::hw::current().slow_wait_ms.max(1);
        crate::knobs::positive("ROCKET_SLOW_WAIT_MS")
            .map(|v| v as u64)
            .unwrap_or(dflt)
    })
}

/// Record one timed wait. Waits at or past the slow mark warn: the output may be
/// unwritten because the kernel retired the job.
pub(crate) fn note_wait(ns: u64, handle: u32) {
    WAITS.fetch_add(1, Ordering::Relaxed);
    let mut prev = MAX_WAIT_NS.load(Ordering::Relaxed);
    while ns > prev {
        match MAX_WAIT_NS.compare_exchange_weak(prev, ns, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(p) => prev = p,
        }
    }
    let mark = slow_mark_ms();
    if ns < mark * 1_000_000 {
        return;
    }
    SLOW_WAITS.fetch_add(1, Ordering::Relaxed);
    log_warn!(
        "a fenced wait on BO {handle} took {:.1} ms, past the {mark} ms slow-wait mark; \
         the kernel may have retired the job",
        ns as f64 / 1e6
    );
}

/// `(waits, slow waits, max wait in µs)`.
pub fn counters() -> (u64, u64, u64) {
    (
        WAITS.load(Ordering::Relaxed),
        SLOW_WAITS.load(Ordering::Relaxed),
        MAX_WAIT_NS.load(Ordering::Relaxed) / 1000,
    )
}

pub fn reset_counters() {
    WAITS.store(0, Ordering::Relaxed);
    SLOW_WAITS.store(0, Ordering::Relaxed);
    MAX_WAIT_NS.store(0, Ordering::Relaxed);
}

/// `ROCKET_BUSY_POLL` in microseconds, resolved once, or 0 when off.
///
/// A caller may override it at any time with [`set_busy_poll_us`]; the sentinel says
/// "not resolved yet", so the override works before or after the first read.
pub(crate) fn busy_poll_us() -> u64 {
    let v = BUSY_POLL_US.load(Ordering::Relaxed);
    if v != UNRESOLVED {
        return v;
    }
    let resolved = crate::knobs::int("ROCKET_BUSY_POLL")
        .filter(|v| *v > 0)
        .map(|v| v as u64)
        .unwrap_or(0);
    match BUSY_POLL_US.compare_exchange(UNRESOLVED, resolved, Ordering::Relaxed, Ordering::Relaxed)
    {
        Ok(_) => resolved,
        // Another thread resolved it first (or a setter did); its value wins.
        Err(v) => v,
    }
}

/// Override the completion-wait spin budget (C `rocket_busy_poll_set_us`).
///
/// Negative values clamp to 0 (spin off). Takes effect immediately, whether or not
/// `ROCKET_BUSY_POLL` was already resolved, and applies to every wait that starts
/// afterwards.
pub fn set_busy_poll_us(us: i64) {
    BUSY_POLL_US.store(us.max(0) as u64, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters_accumulate() {
        reset_counters();
        note_wait(1_000, 7);
        note_wait(3_000, 7);
        let (waits, slow, max_us) = counters();
        assert_eq!(waits, 2);
        assert_eq!(slow, 0);
        assert_eq!(max_us, 3);
        reset_counters();
    }

    #[test]
    fn busy_poll_override() {
        // Process-wide, so save and restore: other tests may rely on the env value.
        let before = busy_poll_us();
        set_busy_poll_us(-5);
        assert_eq!(busy_poll_us(), 0, "negative clamps to off");
        set_busy_poll_us(250);
        assert_eq!(busy_poll_us(), 250, "the setter overrides a resolved value");
        set_busy_poll_us(0);
        assert_eq!(busy_poll_us(), 0);
        set_busy_poll_us(before as i64);
    }
}
