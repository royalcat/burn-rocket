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
        // Keep the just-used matmul shape set, drop per-shape scratch from
        // earlier requests before handing the transient heap back.
        trim_npu_scratch();
        trim_heap();
    }
}

/// Drop per-shape matmul scratch from earlier requests, keeping the just-used
/// set. A no-op without the `npu` feature.
pub fn trim_npu_scratch() {
    #[cfg(feature = "npu")]
    burn_rocket::trim_scratch();
}

/// Whether `ROCKET_MEM_LOG=1` enables the per-request NPU scratch line.
#[cfg_attr(not(feature = "npu"), allow(dead_code))]
pub fn mem_log_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ROCKET_MEM_LOG").is_ok_and(|v| v != "0"))
}

/// Print the driver's live BO/host scratch split (no-op without the `npu`
/// feature or before `burn_rocket::init`).
pub fn log_npu_mem() {
    #[cfg(feature = "npu")]
    {
        if mem_log_enabled() {
            log_npu_mem_impl();
        }
    }
}

#[cfg(feature = "npu")]
fn log_npu_mem_impl() {
    let Some(stats) = burn_rocket::mem_stats() else {
        return;
    };
    let mib = |b: usize| b as f64 / (1024.0 * 1024.0);
    let mut slots: Vec<_> = stats.slots.iter().collect();
    slots.sort_by_key(|s| std::cmp::Reverse(s.bo_bytes + s.host_bytes));
    let top_slots = slots
        .iter()
        .take(5)
        .map(|s| {
            format!(
                "{}x{}x{}:{:.1}/{:.1}",
                s.m,
                s.k,
                s.n,
                mib(s.bo_bytes),
                mib(s.host_bytes)
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let attn = stats
        .attn_workers
        .iter()
        .map(|w| {
            // `@t` is the tiled path's key-tile width (0 = materialized).
            format!(
                "{}x{}@t{}:{:.1}/{:.1}",
                w.tp,
                w.kn,
                w.tile,
                mib(w.bo_bytes),
                mib(w.host_bytes)
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    tracing::info!(
        target: "npu_mem",
        weights_mib = mib(stats.weights_bytes),
        matmul_bo_mib = mib(stats.matmul_bo_bytes),
        matmul_host_mib = mib(stats.matmul_host_bytes),
        attn_bo_mib = mib(stats.attention_bo_bytes),
        attn_host_mib = mib(stats.attention_host_bytes),
        slots = stats.slots.len(),
        top_slots = %top_slots,
        attn_workers = %attn,
        "npu scratch"
    );
}
