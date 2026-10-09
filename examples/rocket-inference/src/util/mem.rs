//! Process memory readout and heap release.

/// Anonymous resident set size (`RssAnon`), MiB.
pub fn rss_mib() -> f64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in s.lines() {
        if let Some(v) = line.strip_prefix("RssAnon:") {
            return v
                .trim()
                .trim_end_matches(" kB")
                .parse::<f64>()
                .unwrap_or(0.0)
                / 1024.0;
        }
    }
    0.0
}

/// Return the allocator's free pages to the OS (glibc keeps them in its arenas,
/// which would hide memory savings after large one-shot allocations: model
/// loads, quantization and the tokenizer parse).
///
/// A no-op on platforms without `malloc_trim`.
pub fn trim_heap() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    unsafe {
        libc::malloc_trim(0);
    }
}

/// Whether a completed served request should release its heap high-water mark.
///
/// On by default (long requests otherwise leave tens to hundreds of MiB
/// resident); `ROCKET_TRIM=0` disables it.
pub fn trim_after_requests() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("ROCKET_TRIM")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

/// Enable the heap release at the end of every request when configured
/// (see [`trim_after_requests`]).
pub fn trim_after_request() {
    if trim_after_requests() {
        trim_heap();
    }
}
