//! Clef-Omni phase-0 probe: can an RK3588 stream Qwen3-Omni-30B-A3B's experts
//! (128 per layer, top-8, hidden 2048, expert intermediate 768) from NVMe through
//! the NPU? It decides the resident weight format (see
//! `docs/experiments/clef-phase0.md`).
//!
//!   expert_bench [--mode all|matmul|rewrite|disk|pipeline] [--reps R]
//!                [--threads T] [--file PATH] [--chunk-mib 4.7] [--max-mib 2048]
//!                [--group 128] [--layers 3] [--tokens 128,1024,4096]
//!                [--hit N] [--depth 8] [--no-ref]
//!
//! Modes (default `all`; the file-dependent ones need `--file`):
//!
//! * `matmul`   — expert-shaped group-wise int8 (`pack_weight_gw` + `matmul_gw`,
//!   activation quantize included) vs the resident fp16 `Context`, for gate|up
//!   (K=2048, N=1536) and down (K=768, N=2048) at M in {16,32,64,128}, groups
//!   128/256/512 (K=768: 128/256/768); cosine vs an f64 reference, wall and CPU.
//!   Same conventions as `q8_bench`.
//! * `rewrite`  — `I8Context::rewrite_weight_gw` (re-scatter new codes into the
//!   existing BOs) vs a fresh `pack_weight_gw`, with an exactness check (the
//!   rewritten weight must compute bit-identically to a fresh pack of the same codes).
//! * `disk`     — sequential reads of `--file` in expert-sized chunks, O_DIRECT with
//!   aligned buffers (plain reads + `posix_fadvise(DONTNEED)` when unsupported).
//!   Reads only; the file is never written. Needs no NPU.
//! * `pipeline` — one streamed layer: read E hit experts from disk, rewrite the
//!   resident slot, run the fused gate|up and the down matmul, with the read of
//!   layer L+1 overlapped with compute of layer L (prefetch thread). Per `--tokens`
//!   S: hit = round(128*(1-(15/16)^S)), M_e = S*8/hit. Variants: disk read alone,
//!   compute alone (no disk), overlapped with one N-split context (`--threads`
//!   workers), and overlapped with 3 one-worker contexts on 3 consumer threads
//!   ("3 experts on 3 cores"). Prints per-layer and extrapolated 48-layer times.
//!   Without `--file` the disk is replaced by a pre-filled buffer (compute only).
//!
//! The device-free parts (`--mode disk`, the pipeline's disk-only row) run on any host.

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::ffi::CString;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Barrier, Mutex};
use std::time::Instant;

use half::f16;
use rocket_npu::matmul::{Context, I8Context};

const ALIGN: usize = 4096;
const LAYERS: usize = 48;
const EXPERTS: usize = 128;
const TOPK: usize = 8;
const HIDDEN: usize = 2048;
const INTER: usize = 768;
/// Resident-weight pack `M` (informational with canonical tiling).
const PACK_M: usize = 128;

type BoxErr = Box<dyn std::error::Error + Send + Sync>;

// ---------------------------------------------------------------- small helpers

fn rnd(seed: &mut u64) -> f32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    ((*seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
}

/// `(user + sys)` CPU seconds of this process.
fn cpu_s() -> f64 {
    // SAFETY: getrusage fills the zeroed struct on success, which is the only case read.
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) } != 0 {
        return 0.0;
    }
    let tv = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 * 1e-6;
    tv(ru.ru_utime) + tv(ru.ru_stime)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v.get(v.len() / 2).copied().unwrap_or(0.0)
}

fn mib(bytes: usize) -> f64 {
    bytes as f64 / (1u64 << 20) as f64
}

fn compare(got: &[f32], want: &[f32]) -> (f64, f64) {
    let (mut dot, mut na, mut nb, mut max) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (g, w) in got.iter().zip(want) {
        let (g, w) = (*g as f64, *w as f64);
        dot += g * w;
        na += g * g;
        nb += w * w;
        max = max.max((g - w).abs());
    }
    (dot / (na.sqrt() * nb.sqrt()), max)
}

/// Symmetric int8 group quantize: `scale = max|v|/127`, `round(v/scale)`.
fn quantize_group(seg: &[f32], q: &mut [i8], s: &mut f32) {
    let amax = seg.iter().fold(0f32, |a, &v| a.max(v.abs()));
    let scale = (amax / 127.0).max(1e-12);
    *s = scale;
    let inv = 1.0 / scale;
    for (qi, &v) in q.iter_mut().zip(seg) {
        *qi = (v * inv).round().clamp(-127.0, 127.0) as i8;
    }
}

/// Per-row per-group activation quantize over `threads` row blocks.
fn quantize_rows(
    x: &[f32],
    m: usize,
    k: usize,
    group: usize,
    q: &mut [i8],
    s: &mut [f32],
    threads: usize,
) {
    let ng = k / group;
    let step = m.div_ceil(threads.max(1)).max(1);
    let work = |xs: &[f32], qs: &mut [i8], ss: &mut [f32]| {
        for (xr, (qr, sr)) in xs.chunks(k).zip(qs.chunks_mut(k).zip(ss.chunks_mut(ng))) {
            for g in 0..ng {
                quantize_group(
                    &xr[g * group..(g + 1) * group],
                    &mut qr[g * group..(g + 1) * group],
                    &mut sr[g],
                );
            }
        }
    };
    if threads <= 1 || m <= step {
        work(x, q, s);
        return;
    }
    std::thread::scope(|scope| {
        for ((xs, qs), ss) in x
            .chunks(step * k)
            .zip(q.chunks_mut(step * k))
            .zip(s.chunks_mut(step * ng))
        {
            scope.spawn(move || work(xs, qs, ss));
        }
    });
}

/// `[N,K]` f32 weight -> group-wise codes + scales (`[N, K/group]`).
fn quantize_weight(w: &[f32], n: usize, k: usize, group: usize) -> (Vec<i8>, Vec<f32>) {
    let ng = k / group;
    let mut q = vec![0i8; n * k];
    let mut s = vec![0f32; n * ng];
    for r in 0..n {
        for g in 0..ng {
            let range = r * k + g * group..r * k + (g + 1) * group;
            quantize_group(&w[range.clone()], &mut q[range], &mut s[r * ng + g]);
        }
    }
    (q, s)
}

fn dequantize(q: &[i8], s: &[f32], n: usize, k: usize, group: usize) -> Vec<f32> {
    let ng = k / group;
    let mut out = vec![0f32; n * k];
    for r in 0..n {
        for g in 0..ng {
            let sc = s[r * ng + g];
            for i in 0..group {
                let idx = r * k + g * group + i;
                out[idx] = q[idx] as f32 * sc;
            }
        }
    }
    out
}

/// f64 reference `C[sample,N] = A[sample,K] * Bq[N,K]^T`.
fn reference(a: &[f32], bq: &[f32], k: usize, n: usize, sample: usize) -> Vec<f32> {
    let mut out = vec![0f32; sample * n];
    for i in 0..sample {
        for j in 0..n {
            let mut sum = 0f64;
            for p in 0..k {
                sum += a[i * k + p] as f64 * bq[j * k + p] as f64;
            }
            out[i * n + j] = sum as f32;
        }
    }
    out
}

fn groups_for(k: usize) -> Vec<usize> {
    if k == INTER {
        vec![128, 256, 768]
    } else {
        vec![128, 256, 512]
    }
}

/// The two expert matmul shapes: (name, K, N).
const SHAPES: [(&str, usize, usize); 2] = [("gate|up", HIDDEN, 2 * INTER), ("down", INTER, HIDDEN)];

// ---------------------------------------------------------------- disk

/// A page-aligned heap buffer (O_DIRECT needs aligned address, length and offset).
struct AlignedBuf {
    ptr: *mut u8,
    len: usize,
}

// SAFETY: the buffer is plain owned memory with no thread affinity.
unsafe impl Send for AlignedBuf {}

impl AlignedBuf {
    fn new(len: usize) -> AlignedBuf {
        assert!(len > 0 && len.is_multiple_of(ALIGN));
        let layout = Layout::from_size_align(len, ALIGN).expect("layout");
        // SAFETY: non-zero size, valid layout.
        let ptr = unsafe { alloc_zeroed(layout) };
        assert!(!ptr.is_null(), "out of memory");
        AlignedBuf { ptr, len }
    }
    fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr/len describe our live allocation.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as above, and `&mut self` is exclusive.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.len, ALIGN).expect("layout");
        // SAFETY: allocated with the same layout in `new`.
        unsafe { dealloc(self.ptr, layout) };
    }
}

/// A read-only file, opened O_DIRECT when the filesystem allows it.
struct DiskFile {
    fd: i32,
    direct: bool,
    len: u64,
}

impl DiskFile {
    fn open(path: &str) -> Result<DiskFile, BoxErr> {
        let c = CString::new(path)?;
        // SAFETY: valid NUL-terminated path; read-only flags.
        let try_open = |flags: i32| unsafe {
            libc::open(c.as_ptr(), libc::O_RDONLY | libc::O_CLOEXEC | flags)
        };
        let mut fd = try_open(libc::O_DIRECT);
        let mut direct = fd >= 0;
        if direct {
            // Some filesystems accept the flag at open and refuse the read.
            let mut probe = AlignedBuf::new(ALIGN);
            // SAFETY: aligned buffer of ALIGN bytes.
            let rc = unsafe { libc::pread(fd, probe.as_mut_slice().as_mut_ptr().cast(), ALIGN, 0) };
            if rc < 0 {
                // SAFETY: fd was just opened by us.
                unsafe { libc::close(fd) };
                fd = -1;
                direct = false;
            }
        }
        if fd < 0 {
            fd = try_open(0);
        }
        if fd < 0 {
            return Err(format!("open {path}: {}", std::io::Error::last_os_error()).into());
        }
        // SAFETY: zeroed stat filled by fstat.
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        if unsafe { libc::fstat(fd, &mut st) } != 0 {
            return Err(format!("fstat {path}: {}", std::io::Error::last_os_error()).into());
        }
        // Start cold: drop any cached pages of the file.
        // SAFETY: advisory call on our fd.
        unsafe { libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_DONTNEED) };
        Ok(DiskFile {
            fd,
            direct,
            len: st.st_size as u64,
        })
    }

    /// Fill `buf` from `off` (stops short only at EOF); returns the bytes read.
    fn read_at(&self, buf: &mut [u8], off: u64) -> std::io::Result<usize> {
        let mut got = 0usize;
        while got < buf.len() {
            // SAFETY: the destination range is inside `buf`.
            let rc = unsafe {
                libc::pread(
                    self.fd,
                    buf[got..].as_mut_ptr().cast(),
                    buf.len() - got,
                    (off + got as u64) as libc::off_t,
                )
            };
            if rc < 0 {
                let e = std::io::Error::last_os_error();
                if e.kind() == std::io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(e);
            }
            if rc == 0 {
                break;
            }
            got += rc as usize;
        }
        if !self.direct {
            // Plain reads fill the page cache: drop what we just read so the next
            // pass measures the device, not DRAM.
            // SAFETY: advisory call on our fd.
            unsafe {
                libc::posix_fadvise(
                    self.fd,
                    off as libc::off_t,
                    got as libc::off_t,
                    libc::POSIX_FADV_DONTNEED,
                )
            };
        }
        Ok(got)
    }

    fn mode(&self) -> &'static str {
        if self.direct {
            "O_DIRECT"
        } else {
            "buffered + posix_fadvise(DONTNEED)"
        }
    }
}

impl Drop for DiskFile {
    fn drop(&mut self) {
        // SAFETY: closing our own fd.
        unsafe { libc::close(self.fd) };
    }
}

fn round_up(v: usize, to: usize) -> usize {
    v.div_ceil(to) * to
}

fn disk_bench(
    file: &DiskFile,
    path: &str,
    chunk: usize,
    max_bytes: usize,
    reps: usize,
) -> Result<f64, BoxErr> {
    let total = (file.len as usize).min(max_bytes) / chunk * chunk;
    if total == 0 {
        return Err(format!(
            "{path} ({} B) is smaller than one {chunk} B chunk",
            file.len
        )
        .into());
    }
    println!(
        "\n== disk: {path} ({:.1} MiB) | {} | chunk {:.2} MiB | {:.0} MiB per pass ==",
        mib(file.len as usize),
        file.mode(),
        mib(chunk),
        mib(total)
    );
    let mut buf = AlignedBuf::new(chunk);
    let mut best = 0f64;
    for rep in 0..reps.max(1) {
        let mut lat = Vec::with_capacity(total / chunk);
        let c0 = cpu_s();
        let t = Instant::now();
        let mut off = 0u64;
        while (off as usize) < total {
            let tc = Instant::now();
            let n = file.read_at(buf.as_mut_slice(), off)?;
            lat.push(tc.elapsed().as_secs_f64() * 1e3);
            if n == 0 {
                break;
            }
            off += chunk as u64;
        }
        let secs = t.elapsed().as_secs_f64();
        let cpu = cpu_s() - c0;
        let gbs = off as f64 / secs / 1e9;
        best = best.max(gbs);
        let p50 = median(lat.clone());
        lat.sort_by(|a, b| a.total_cmp(b));
        let p99 = lat[(lat.len() * 99 / 100).min(lat.len() - 1)];
        println!(
            "  pass {rep}: {gbs:6.3} GB/s ({:.3} GiB/s) | {:.2}s | cpu {:.2}s | chunk p50 {p50:.2} ms p99 {p99:.2} ms",
            off as f64 / secs / (1u64 << 30) as f64,
            secs,
            cpu
        );
    }
    println!("  best {best:.3} GB/s");
    Ok(best)
}

// ---------------------------------------------------------------- modes 1 and 2

fn matmul_mode(reps: usize, threads: usize, with_ref: bool) -> Result<(), BoxErr> {
    let mut ctx = Context::new(threads)?;
    let mut ictx = I8Context::new(threads)?;
    println!(
        "\n== matmul: expert shapes, f16 resident vs int8 group-wise (threads={threads} reps={reps}, \
         activation quantize included) =="
    );
    struct Row {
        name: &'static str,
        m: usize,
        k: usize,
        n: usize,
        group: usize,
        f16_wall: f64,
        f16_cpu: f64,
        i8_wall: f64,
        i8_cpu: f64,
        quant: f64,
        cos: f64,
        f16_bo: usize,
        i8_bo: usize,
    }
    let mut table = Vec::new();
    for (name, k, n) in SHAPES {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64 ^ ((n as u64) << 16) ^ k as u64;
        let w32: Vec<f32> = (0..n * k).map(|_| rnd(&mut seed)).collect();
        for group in groups_for(k) {
            let (bq, bs) = quantize_weight(&w32, n, k, group);
            let bdq = dequantize(&bq, &bs, n, k, group);
            let b16: Vec<f16> = bdq.iter().map(|&v| f16::from_f32(v)).collect();
            let wf16 = ctx.pack_weight(PACK_M, k, n, &b16)?;
            let wi8 = ictx.pack_weight_gw(PACK_M, k, n, &bq, group)?;
            let ng = k / group;
            for m in [16usize, 32, 64, 128] {
                let a32: Vec<f32> = (0..m * k).map(|_| rnd(&mut seed) * 2.0).collect();
                let sample = m.min(16);
                let cref = if with_ref {
                    let b16f: Vec<f32> = b16.iter().map(|v| v.to_f32()).collect();
                    reference(&a32, &b16f, k, n, sample)
                } else {
                    Vec::new()
                };
                // f16 arm
                let (mut walls, mut cpus) = (Vec::new(), Vec::new());
                let mut c16f = vec![0f32; m * n];
                for _ in 0..reps {
                    let c0 = cpu_s();
                    let t = Instant::now();
                    let a16: Vec<f16> = a32.iter().map(|&v| f16::from_f32(v)).collect();
                    let mut c16 = vec![f16::ZERO; m * n];
                    ctx.matmul_prepacked(m, k, n, &a16, &mut c16, &wf16)?;
                    for (d, s) in c16f.iter_mut().zip(&c16) {
                        *d = s.to_f32();
                    }
                    walls.push(t.elapsed().as_secs_f64() * 1e3);
                    cpus.push((cpu_s() - c0) * 1e3);
                }
                let (f16_wall, f16_cpu) = (median(walls), median(cpus));
                // int8 arm
                let (mut walls, mut cpus, mut quants) = (Vec::new(), Vec::new(), Vec::new());
                let mut out = vec![0f32; m * n];
                for _ in 0..reps {
                    let c0 = cpu_s();
                    let t = Instant::now();
                    let mut q = vec![0i8; m * k];
                    let mut sa = vec![1f32; m * ng];
                    quantize_rows(&a32, m, k, group, &mut q, &mut sa, threads);
                    let tq = t.elapsed().as_secs_f64() * 1e3;
                    ictx.matmul_gw(m, k, n, &q, &sa, &bs, &mut out, &wi8)?;
                    walls.push(t.elapsed().as_secs_f64() * 1e3);
                    quants.push(tq);
                    cpus.push((cpu_s() - c0) * 1e3);
                }
                let (i8_wall, i8_cpu, quant) = (median(walls), median(cpus), median(quants));
                let cos = if with_ref {
                    compare(&out[..sample * n], &cref).0
                } else {
                    compare(&out[..sample * n], &c16f[..sample * n]).0
                };
                println!(
                    "  {name:>7} M={m:>3} K={k:>4} N={n:>4} g{group:<3} | f16 {f16_wall:7.2} ms cpu {f16_cpu:7.2} | \
                     int8 {i8_wall:7.2} ms cpu {i8_cpu:7.2} (quant {quant:.2}) | {:.2}x | cos {cos:.8}",
                    i8_wall / f16_wall
                );
                table.push(Row {
                    name,
                    m,
                    k,
                    n,
                    group,
                    f16_wall,
                    f16_cpu,
                    i8_wall,
                    i8_cpu,
                    quant,
                    cos,
                    f16_bo: wf16.bytes(),
                    i8_bo: wi8.bytes(),
                });
            }
        }
    }
    println!(
        "\n== matmul summary (ms, median of {reps}; cosine vs {}) ==",
        if with_ref {
            "f64 reference"
        } else {
            "the f16 arm"
        }
    );
    println!(
        "  shape      M    K    N  grp |  f16 wall   cpu | int8 wall   cpu  quant | ratio |   cosine | bo MiB f16/i8"
    );
    for r in &table {
        println!(
            "  {:>7} {:>4} {:>4} {:>4} {:>4} | {:>9.2} {:>5.1} | {:>9.2} {:>5.1} {:>6.2} | {:>4.2}x | {:.6} | {:.2}/{:.2}",
            r.name,
            r.m,
            r.k,
            r.n,
            r.group,
            r.f16_wall,
            r.f16_cpu,
            r.i8_wall,
            r.i8_cpu,
            r.quant,
            r.i8_wall / r.f16_wall,
            r.cos,
            mib(r.f16_bo),
            mib(r.i8_bo)
        );
    }
    Ok(())
}

fn rewrite_mode(reps: usize, threads: usize) -> Result<(), BoxErr> {
    let mut ictx = I8Context::new(threads)?;
    println!(
        "\n== rewrite: in-place `rewrite_weight_gw` vs fresh `pack_weight_gw` (threads={threads}) =="
    );
    for (name, k, n) in SHAPES {
        let mut seed = 0x1234_5678_9abc_def1u64 ^ ((n as u64) << 16) ^ k as u64;
        let codes = |s: &mut u64| -> Vec<i8> {
            (0..n * k)
                .map(|_| (rnd(s) * 254.0).round().clamp(-127.0, 127.0) as i8)
                .collect()
        };
        for group in groups_for(k) {
            let b0 = codes(&mut seed);
            let b1 = codes(&mut seed);
            let mut w = ictx.pack_weight_gw(PACK_M, k, n, &b0, group)?;
            let (mut pack, mut rewr) = (Vec::new(), Vec::new());
            for _ in 0..reps.max(1) {
                let t = Instant::now();
                let fresh = ictx.pack_weight_gw(PACK_M, k, n, &b1, group)?;
                pack.push(t.elapsed().as_secs_f64() * 1e3);
                drop(fresh);
                let t = Instant::now();
                ictx.rewrite_weight_gw(&mut w, &b1)?;
                rewr.push(t.elapsed().as_secs_f64() * 1e3);
                // Put old codes back so every rep rewrites different bytes.
                ictx.rewrite_weight_gw(&mut w, &b0)?;
            }
            ictx.rewrite_weight_gw(&mut w, &b1)?;
            // Exactness: the rewritten weight computes what a fresh pack computes.
            let fresh = ictx.pack_weight_gw(PACK_M, k, n, &b1, group)?;
            let (m, ng) = (16usize, k / group);
            let a32: Vec<f32> = (0..m * k).map(|_| rnd(&mut seed) * 2.0).collect();
            let mut q = vec![0i8; m * k];
            let mut sa = vec![1f32; m * ng];
            quantize_rows(&a32, m, k, group, &mut q, &mut sa, 1);
            let bs = vec![0.01f32; n * ng];
            let (mut o1, mut o2) = (vec![0f32; m * n], vec![0f32; m * n]);
            ictx.matmul_gw(m, k, n, &q, &sa, &bs, &mut o1, &w)?;
            ictx.matmul_gw(m, k, n, &q, &sa, &bs, &mut o2, &fresh)?;
            let (p, r) = (median(pack), median(rewr));
            println!(
                "  {name:>7} K={k:>4} N={n:>4} g{group:<3} | pack {p:7.2} ms | rewrite {r:7.2} ms ({:.2}x) | \
                 {:.2} MiB codes -> {:.0} MiB/s | rewritten == fresh: {}",
                p / r,
                mib(n * k),
                mib(n * k) / (r / 1e3),
                if o1 == o2 {
                    "bit-identical"
                } else {
                    "MISMATCH"
                }
            );
            if o1 != o2 {
                return Err("rewrite produced a different weight than a fresh pack".into());
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------- mode 4

struct Slot {
    gu: rocket_npu::matmul::I8Weight,
    dn: rocket_npu::matmul::I8Weight,
    gu_scales: Vec<f32>,
    dn_scales: Vec<f32>,
}

/// Per-consumer phase totals (seconds).
#[derive(Default, Clone, Copy)]
struct Phases {
    rewrite: f64,
    quant: f64,
    npu: f64,
    experts: usize,
}

enum Item {
    Expert(AlignedBuf),
    LayerEnd,
}

struct PipeCfg {
    group: usize,
    hit: usize,
    m: usize,
    layers: usize,
    depth: usize,
    /// Per-consumer worker count of its `I8Context`.
    ctx_threads: usize,
    consumers: usize,
    compute: bool,
    expert_bytes: usize,
}

struct PipeResult {
    layer_s: Vec<f64>,
    total_s: f64,
    phases: Phases,
}

/// Run `layers` streamed layers. `file` = `None` skips the read (pre-filled buffers).
fn run_pipeline(cfg: &PipeCfg, file: Option<&DiskFile>) -> Result<PipeResult, BoxErr> {
    let (full_tx, full_rx) = mpsc::sync_channel::<Item>(cfg.depth);
    let full_rx = Arc::new(Mutex::new(full_rx));
    let (free_tx, free_rx) = mpsc::channel::<AlignedBuf>();
    for _ in 0..cfg.depth + cfg.consumers {
        let mut b = AlignedBuf::new(cfg.expert_bytes);
        let mut s = 0x5eed_u64;
        for x in b.as_mut_slice() {
            *x = (rnd(&mut s) * 255.0) as i8 as u8;
        }
        free_tx.send(b).ok();
    }
    let barrier = Arc::new(Barrier::new(cfg.consumers));
    let layer_done = Arc::new(Mutex::new(Vec::<Instant>::new()));
    let failed = Arc::new(AtomicBool::new(false));
    let slots = (file.map(|f| f.len as usize).unwrap_or(0) / cfg.expert_bytes).max(1);
    let start = Instant::now();

    let result: Result<Vec<Phases>, BoxErr> = std::thread::scope(|scope| {
        // Reader: streams every expert of every layer in order, then the layer
        // markers (one per consumer, so each can meet the layer barrier).
        let reader = scope.spawn({
            move || -> Result<(), BoxErr> {
                let mut idx = 0usize;
                for _ in 0..cfg.layers {
                    for _ in 0..cfg.hit {
                        let Ok(mut buf) = free_rx.recv() else {
                            return Ok(());
                        };
                        if let Some(f) = file {
                            let off = (idx % slots * cfg.expert_bytes) as u64;
                            f.read_at(buf.as_mut_slice(), off)?;
                        }
                        idx += 1;
                        if full_tx.send(Item::Expert(buf)).is_err() {
                            return Ok(());
                        }
                    }
                    for _ in 0..cfg.consumers {
                        if full_tx.send(Item::LayerEnd).is_err() {
                            return Ok(());
                        }
                    }
                }
                Ok(())
            }
        });
        let mut consumers = Vec::new();
        for ci in 0..cfg.consumers {
            let full_rx = full_rx.clone();
            let free_tx = free_tx.clone();
            let barrier = barrier.clone();
            let layer_done = layer_done.clone();
            let failed = failed.clone();
            consumers.push(scope.spawn(move || -> Result<Phases, BoxErr> {
                let r = consumer(cfg, ci, &full_rx, &free_tx, &barrier, &layer_done);
                if r.is_err() {
                    failed.store(true, Ordering::SeqCst);
                }
                r
            }));
        }
        drop(free_tx);
        let mut phases = Vec::new();
        let mut err: Option<BoxErr> = None;
        for c in consumers {
            match c.join() {
                Ok(Ok(p)) => phases.push(p),
                Ok(Err(e)) => err = err.or(Some(e)),
                Err(_) => err = err.or(Some("consumer thread panicked".into())),
            }
        }
        // Consumers are gone: unblock a reader parked on the full queue.
        drop(full_rx);
        match reader.join() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => err = err.or(Some(e)),
            Err(_) => err = err.or(Some("reader thread panicked".into())),
        }
        match err {
            Some(e) => Err(e),
            None => Ok(phases),
        }
    });
    let phases = result?;
    let total_s = start.elapsed().as_secs_f64();
    let marks = layer_done.lock().unwrap();
    let mut layer_s = Vec::new();
    let mut prev = start;
    for t in marks.iter() {
        layer_s.push(t.duration_since(prev).as_secs_f64());
        prev = *t;
    }
    let mut sum = Phases::default();
    for p in &phases {
        sum.rewrite += p.rewrite;
        sum.quant += p.quant;
        sum.npu += p.npu;
        sum.experts += p.experts;
    }
    Ok(PipeResult {
        layer_s,
        total_s,
        phases: sum,
    })
}

fn consumer(
    cfg: &PipeCfg,
    ci: usize,
    rx: &Mutex<mpsc::Receiver<Item>>,
    free_tx: &mpsc::Sender<AlignedBuf>,
    barrier: &Barrier,
    layer_done: &Mutex<Vec<Instant>>,
) -> Result<Phases, BoxErr> {
    let g = cfg.group;
    let (k1, n1, k2, n2) = (HIDDEN, 2 * INTER, INTER, HIDDEN);
    let mut ctx = if cfg.compute {
        Some(I8Context::new_at(cfg.ctx_threads, ci * cfg.ctx_threads)?)
    } else {
        None
    };
    let mut slot = None;
    if let Some(ctx) = ctx.as_mut() {
        let mut s = 0xabcd_ef01u64 ^ ci as u64;
        let b1: Vec<i8> = (0..n1 * k1).map(|_| (rnd(&mut s) * 200.0) as i8).collect();
        let b2: Vec<i8> = (0..n2 * k2).map(|_| (rnd(&mut s) * 200.0) as i8).collect();
        slot = Some(Slot {
            gu: ctx.pack_weight_gw(PACK_M, k1, n1, &b1, g)?,
            dn: ctx.pack_weight_gw(PACK_M, k2, n2, &b2, g)?,
            gu_scales: vec![0.004; n1 * (k1 / g)],
            dn_scales: vec![0.004; n2 * (k2 / g)],
        });
    }
    let m = cfg.m;
    let mut seed = 0x77u64 ^ ci as u64;
    let a1: Vec<f32> = (0..m * k1).map(|_| rnd(&mut seed)).collect();
    let (mut q1, mut s1) = (vec![0i8; m * k1], vec![1f32; m * (k1 / g)]);
    let (mut q2, mut s2) = (vec![0i8; m * k2], vec![1f32; m * (k2 / g)]);
    let (mut c1, mut c2) = (vec![0f32; m * n1], vec![0f32; m * n2]);
    let mut ph = Phases::default();
    let mut ended = 0usize;
    loop {
        let item = {
            let guard = rx.lock().unwrap();
            guard.recv()
        };
        let Ok(item) = item else { break };
        match item {
            Item::Expert(buf) => {
                if let (Some(ctx), Some(slot)) = (ctx.as_mut(), slot.as_mut()) {
                    let bytes = buf.as_slice();
                    // SAFETY: u8 -> i8 reinterpretation of the same bytes.
                    let codes: &[i8] =
                        unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast(), bytes.len()) };
                    let t = Instant::now();
                    ctx.rewrite_weight_gw(&mut slot.gu, &codes[..n1 * k1])?;
                    ctx.rewrite_weight_gw(&mut slot.dn, &codes[n1 * k1..n1 * k1 + n2 * k2])?;
                    ph.rewrite += t.elapsed().as_secs_f64();
                    let t = Instant::now();
                    quantize_rows(&a1, m, k1, g, &mut q1, &mut s1, cfg.ctx_threads);
                    ph.quant += t.elapsed().as_secs_f64();
                    let t = Instant::now();
                    ctx.matmul_gw(m, k1, n1, &q1, &s1, &slot.gu_scales, &mut c1, &slot.gu)?;
                    ph.npu += t.elapsed().as_secs_f64();
                    // Stand-in for silu(gate)*up: the first 768 columns of each row.
                    let t = Instant::now();
                    let mut act = vec![0f32; m * k2];
                    for (dst, src) in act.chunks_mut(k2).zip(c1.chunks(n1)) {
                        dst.copy_from_slice(&src[..k2]);
                    }
                    quantize_rows(&act, m, k2, g, &mut q2, &mut s2, cfg.ctx_threads);
                    ph.quant += t.elapsed().as_secs_f64();
                    let t = Instant::now();
                    ctx.matmul_gw(m, k2, n2, &q2, &s2, &slot.dn_scales, &mut c2, &slot.dn)?;
                    ph.npu += t.elapsed().as_secs_f64();
                }
                ph.experts += 1;
                let _ = free_tx.send(buf);
            }
            Item::LayerEnd => {
                barrier.wait();
                ended += 1;
                if ci == 0 {
                    layer_done.lock().unwrap().push(Instant::now());
                }
                if ended == cfg.layers {
                    break;
                }
            }
        }
    }
    Ok(ph)
}

fn expected_hit(s: usize) -> usize {
    (EXPERTS as f64 * (1.0 - (1.0 - 1.0 / 16.0f64).powi(s as i32)))
        .round()
        .max(1.0) as usize
}

#[allow(clippy::too_many_arguments)]
fn pipeline_mode(
    file: Option<&DiskFile>,
    threads: usize,
    group: usize,
    tokens: &[usize],
    layers: usize,
    depth: usize,
    hit_override: Option<usize>,
    device: bool,
) -> Result<(), BoxErr> {
    let scale_bytes = ((2 * INTER) * (HIDDEN / group) + HIDDEN * (INTER / group)) * 2;
    let codes = 2 * INTER * HIDDEN + HIDDEN * INTER;
    let expert_bytes = round_up(codes + scale_bytes, ALIGN);
    println!(
        "\n== pipeline: group {group}, expert {:.3} MiB on disk ({:.3} codes + {:.3} fp16 scales), \
         {layers} layers simulated, prefetch depth {depth}, source: {} ==",
        mib(expert_bytes),
        mib(codes),
        mib(scale_bytes),
        match file {
            Some(f) => format!("disk ({})", f.mode()),
            None => "memory (no --file: compute only)".into(),
        }
    );
    println!(
        "  per layer the thinker reads the hit experts once; matmuls per expert: fused gate|up + down"
    );
    let mut summary = Vec::new();
    for &s in tokens {
        let hit = hit_override.unwrap_or_else(|| expected_hit(s));
        let m = round_up((s * TOPK).div_ceil(hit).max(4), 4);
        println!(
            "\n  -- S={s} tokens: {hit} hit experts/layer, M_e={m}, {:.0} MiB read per layer --",
            mib(hit * expert_bytes)
        );
        let base = |consumers: usize, ctx_threads: usize, compute: bool| PipeCfg {
            group,
            hit,
            m,
            layers,
            depth,
            ctx_threads,
            consumers,
            compute,
            expert_bytes,
        };
        let report = |label: &str, r: &PipeResult, consumers: usize| {
            let per = r.total_s / layers as f64;
            let n = r.phases.experts.max(1) as f64;
            println!(
                "  {label:<34} layer {:8.2} s (layers: {}) | x{LAYERS} = {:8.1} s ({:5.1} min) \
                 | per expert: rewrite {:6.1} quant {:5.1} npu {:6.1} ms (summed over {consumers} consumer(s))",
                per,
                r.layer_s
                    .iter()
                    .map(|v| format!("{v:.2}"))
                    .collect::<Vec<_>>()
                    .join("/"),
                per * LAYERS as f64,
                per * LAYERS as f64 / 60.0,
                r.phases.rewrite / n * 1e3,
                r.phases.quant / n * 1e3,
                r.phases.npu / n * 1e3,
            );
            per * LAYERS as f64
        };
        if let Some(f) = file {
            let r = run_pipeline(&base(1, 1, false), Some(f))?;
            let t = report("disk read only", &r, 1);
            println!(
                "      -> {:.3} GB/s effective",
                (hit * layers * expert_bytes) as f64 / r.total_s / 1e9
            );
            let _ = t;
        }
        if !device {
            println!("  (no NPU device: compute variants skipped)");
            continue;
        }
        let mut row = (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
        let r = run_pipeline(&base(1, threads, true), None)?;
        row.0 = report(&format!("compute only, 1 ctx x {threads}w"), &r, 1);
        if let Some(f) = file {
            let r = run_pipeline(&base(1, threads, true), Some(f))?;
            row.1 = report(&format!("overlapped, 1 ctx x {threads}w"), &r, 1);
        }
        let r = run_pipeline(&base(3, 1, true), None)?;
        row.2 = report("compute only, 3 ctx x 1w (3 cores)", &r, 3);
        if let Some(f) = file {
            let r = run_pipeline(&base(3, 1, true), Some(f))?;
            row.3 = report("overlapped, 3 ctx x 1w (3 cores)", &r, 3);
        }
        summary.push((s, hit, m, row));
    }
    if !summary.is_empty() {
        println!("\n== pipeline summary: extrapolated {LAYERS}-layer time per forward pass (s) ==");
        println!("    S   hit   M_e | compute 1xNw | overlap 1xNw | compute 3x1w | overlap 3x1w");
        for (s, hit, m, r) in &summary {
            println!(
                "  {s:>4} {hit:>5} {m:>5} | {:>12.1} | {:>12.1} | {:>12.1} | {:>12.1}",
                r.0, r.1, r.2, r.3
            );
        }
        if let Some((_, _, _, r)) = summary.iter().find(|(s, ..)| *s == 1024) {
            let best = [r.1, r.3, r.0, r.2]
                .into_iter()
                .filter(|v| v.is_finite())
                .fold(f64::INFINITY, f64::min);
            println!(
                "  1k tokens best arrangement: {best:.0} s ({:.1} min) -> {} (threshold 300 s{})",
                best / 60.0,
                if best < 300.0 { "GO" } else { "NO-GO" },
                if file.is_none() {
                    "; compute only, disk not included"
                } else {
                    ""
                }
            );
        }
    }
    Ok(())
}

fn main() -> Result<(), BoxErr> {
    let mut mode = "all".to_string();
    let mut reps = 3usize;
    let mut threads = 3usize;
    let mut file: Option<String> = None;
    let mut chunk_mib = 4.7f64;
    let mut max_mib = 2048usize;
    let mut group = 128usize;
    let mut layers = 3usize;
    let mut depth = 8usize;
    let mut tokens = vec![128usize, 1024, 4096];
    let mut hit: Option<usize> = None;
    let mut with_ref = true;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut val = || args.next().unwrap_or_default();
        match arg.as_str() {
            "--mode" => mode = val(),
            "--reps" => reps = val().parse().unwrap_or(reps),
            "--threads" => threads = val().parse().unwrap_or(threads),
            "--file" => file = Some(val()),
            "--chunk-mib" => chunk_mib = val().parse().unwrap_or(chunk_mib),
            "--max-mib" => max_mib = val().parse().unwrap_or(max_mib),
            "--group" => group = val().parse().unwrap_or(group),
            "--layers" => layers = val().parse().unwrap_or(layers),
            "--depth" => depth = val().parse().unwrap_or(depth),
            "--hit" => hit = val().parse().ok(),
            "--tokens" => {
                tokens = val()
                    .split(',')
                    .filter_map(|v| v.trim().parse().ok())
                    .collect()
            }
            "--no-ref" => with_ref = false,
            "-h" | "--help" => {
                println!(
                    "{}",
                    include_str!("expert_bench.rs")
                        .lines()
                        .take_while(|l| l.starts_with("//!"))
                        .map(|l| l.trim_start_matches("//! ").trim_start_matches("//!"))
                        .collect::<Vec<_>>()
                        .join("\n")
                );
                return Ok(());
            }
            other => return Err(format!("unknown argument {other} (try --help)").into()),
        }
    }
    if !["all", "matmul", "rewrite", "disk", "pipeline"].contains(&mode.as_str()) {
        return Err(format!("unknown --mode {mode}").into());
    }
    if ![128, 256, 512, 768].contains(&group) {
        return Err("--group must be 128, 256, 512 or 768 (it must divide K=2048 and K=768 for the pipeline: 128 or 256)".into());
    }
    if (mode == "pipeline" || mode == "all")
        && (!HIDDEN.is_multiple_of(group) || !INTER.is_multiple_of(group))
    {
        {
            return Err(
                format!("--group {group} must divide both 2048 and 768 (use 128 or 256)").into(),
            );
        }
    }
    let want = |m: &str| mode == "all" || mode == m;
    let chunk = round_up((chunk_mib * (1u64 << 20) as f64).ceil() as usize, ALIGN);
    println!("expert_bench: mode={mode} threads={threads} reps={reps}");

    let disk = match &file {
        Some(p) => Some(DiskFile::open(p)?),
        None => None,
    };
    if want("disk") {
        match (&disk, &file) {
            (Some(d), Some(p)) => {
                disk_bench(d, p, chunk, max_mib << 20, reps)?;
            }
            _ if mode == "disk" => return Err("--mode disk needs --file PATH".into()),
            _ => println!("\n(disk mode skipped: no --file)"),
        }
    }
    let mut device_ok = true;
    if want("matmul")
        && let Err(e) = matmul_mode(reps, threads, with_ref)
    {
        {
            device_ok = false;
            println!("\nmatmul mode failed: {e}");
            if mode == "matmul" {
                return Err(e);
            }
        }
    }
    if want("rewrite")
        && let Err(e) = rewrite_mode(reps, threads)
    {
        {
            device_ok = false;
            println!("\nrewrite mode failed: {e}");
            if mode == "rewrite" {
                return Err(e);
            }
        }
    }
    if want("pipeline") {
        // Probe the device once so a host without an NPU still gets the disk-only rows.
        let has_dev = device_ok && I8Context::new(1).is_ok();
        pipeline_mode(
            disk.as_ref(),
            threads,
            group,
            &tokens,
            layers,
            depth,
            hit,
            has_dev,
        )?;
    }
    Ok(())
}
