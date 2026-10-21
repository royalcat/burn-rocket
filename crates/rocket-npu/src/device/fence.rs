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
pub(crate) fn busy_poll_us() -> u64 {
    static US: OnceLock<u64> = OnceLock::new();
    *US.get_or_init(|| {
        crate::knobs::int("ROCKET_BUSY_POLL")
            .filter(|v| *v > 0)
            .map(|v| v as u64)
            .unwrap_or(0)
    })
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
}
