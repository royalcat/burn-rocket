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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn lock_or_recover_survives_a_poisoned_mutex() {
        let mutex = Arc::new(Mutex::new(7u32));
        let poisoned = mutex.clone();
        let joined = std::thread::spawn(move || {
            let _guard = poisoned.lock().unwrap();
            panic!("poison the mutex");
        })
        .join();
        assert!(joined.is_err(), "the helper thread must have panicked");
        assert_eq!(*lock_or_recover(&mutex), 7);
    }

    #[test]
    fn panic_message_renders_both_payload_kinds() {
        let text = panic_message(&("boom" as &str));
        assert_eq!(text, "boom");
        let owned = "owned".to_string();
        let text = panic_message(&owned);
        assert_eq!(text, "owned");
        let other = 3u32;
        assert_eq!(panic_message(&other), "unknown panic");
    }
}
