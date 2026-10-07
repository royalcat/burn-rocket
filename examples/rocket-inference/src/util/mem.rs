//! Process memory readout.

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
