//! Small helpers shared by the model servers: mutex poisoning recovery and
//! panic-payload rendering.

use std::any::Any;
use std::sync::{Mutex, MutexGuard, PoisonError};

/// Lock a mutex, recovering the inner data if a previous holder panicked.
///
/// A panicking forward must not wedge a server, so poisoned locks are treated
/// as normal locks. Note: the error must be consumed with `into_inner()` — the
/// `PoisonError` holds the guard, so re-locking while it is alive deadlocks.
pub fn lock_or_recover<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Render a panic payload as a string.
pub fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}
