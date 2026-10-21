//! Big-core detection and worker pinning.
//!
//! The matmul fan-out's workers do the pack/readback work on the CPU; parking one on
//! an A55 little core stalls the join, so workers are pinned to the A76 cluster. The
//! set is discovered from `cpuinfo_max_freq` (no hardcoded ids) unless
//! `ROCKET_CPU_AFFINITY` names one, as in the C library.

use std::sync::OnceLock;

const CPU_SETSIZE: usize = 1024;

fn read_max_freq(cpu: usize) -> Option<i64> {
    let path = format!("/sys/devices/system/cpu/cpu{cpu}/cpufreq/cpuinfo_max_freq");
    std::fs::read_to_string(path)
        .ok()?
        .trim()
        .parse::<i64>()
        .ok()
}

fn detect() -> Vec<usize> {
    if let Some(list) = crate::knobs::cpu_list("ROCKET_CPU_AFFINITY") {
        if list.is_empty() {
            return Vec::new();
        }
        return list;
    }
    if crate::knobs::present("ROCKET_CPU_AFFINITY") {
        // Set but empty: the C code treats this as "pinning disabled".
        return Vec::new();
    }

    let nconf = unsafe { libc::sysconf(libc::_SC_NPROCESSORS_CONF) };
    let nconf = if nconf < 1 {
        unsafe { libc::sysconf(libc::_SC_NPROCESSORS_ONLN) }
    } else {
        nconf
    };
    if nconf < 1 {
        return Vec::new();
    }
    let nconf = (nconf as usize).min(CPU_SETSIZE);

    let mut freqs = Vec::with_capacity(nconf);
    let mut max_freq = -1i64;
    for cpu in 0..nconf {
        let f = read_max_freq(cpu);
        if let Some(f) = f.filter(|f| *f > 0) {
            max_freq = max_freq.max(f);
        }
        freqs.push(f);
    }
    if max_freq <= 0 {
        return Vec::new();
    }

    let mut big = Vec::new();
    let mut known = 0usize;
    for (cpu, f) in freqs.iter().enumerate() {
        let Some(f) = f else { continue };
        known += 1;
        if *f == max_freq {
            big.push(cpu);
        }
    }
    // A homogeneous set constrains nothing; let the scheduler decide.
    if known == 0 || big.len() >= known {
        return Vec::new();
    }
    big
}

fn big_cores() -> &'static [usize] {
    static CORES: OnceLock<Vec<usize>> = OnceLock::new();
    CORES.get_or_init(detect)
}

/// Number of big cores, or 0 when pinning is disabled/unavailable.
pub fn num_big_cores() -> usize {
    big_cores().len()
}

/// The core a given worker is pinned to, or `None` when pinning is disabled.
pub fn core_for(worker_idx: usize, core_base: usize) -> Option<usize> {
    let cores = big_cores();
    if cores.is_empty() {
        return None;
    }
    Some(cores[(core_base + worker_idx) % cores.len()])
}

/// Pin the calling thread to this worker's big core. A no-op when pinning is off or
/// the syscall fails (the scheduler still runs the worker).
pub fn pin_current(worker_idx: usize, core_base: usize) {
    let Some(cpu) = core_for(worker_idx, core_base) else {
        return;
    };
    // SAFETY: cpu_set_t is a plain C bit set; CPU_ZERO/CPU_SET initialize it.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(cpu, &mut set);
        let rc = libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
        if rc != 0 {
            crate::log::log_debug!(
                "affinity: worker {worker_idx} -> cpu {cpu} failed: {}",
                std::io::Error::last_os_error()
            );
        }
    }
}

/// The calling thread's rotation base. Contexts inherit it into their workers so
/// several pools in one process spread over the cluster.
pub fn set_base(base: usize) {
    BASE.with(|b| b.set(base));
}

pub fn base() -> usize {
    BASE.with(|b| b.get())
}

thread_local! {
    static BASE: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}
