//! The resident matmul context: one device fd per worker, per-shape scratch cache,
//! resident weights, and the threaded fan-out.
//!
//! This is the Rust port of `rocket_prepacked.c` + `rocket_fanout.c`: N is split
//! across workers (one fd each, so the driver can run them in parallel), a scratch
//! is cached per `(M,K,N)` and shared by streaming and prepacked calls, and a
//! resident weight owns only its per-worker weight BOs.

use std::thread;
use std::time::Instant;

use half::f16;

use crate::device::Bo;
use crate::error::{Error, Result};

use super::bos::{self, Bos, Config};
use super::plan::Plan;
use super::run::{compute_cpu, compute_kacc};
use crate::Device;
use crate::device::affinity;

const MAX_WORKERS: usize = 8;
const MAX_SLOTS: usize = 32;

/// One worker's slice of the output and its private scratch.
struct Worker {
    n0: usize,
    nsub: usize,
    m0: usize,
    msub: usize,
    plan: Plan,
    bos: Bos,
    csub: Vec<f16>,
    /// `(mt, in_slot)` of the last input pack when the shared pack was not used;
    /// a layout change needs the BO's padding lanes re-zeroed.
    pack_sig: Option<(usize, usize)>,
}

struct Slot {
    m: usize,
    k: usize,
    n: usize,
    split: bool,
    stamp: u64,
    workers: Vec<Worker>,
    /// Shared A-pack (zeroed once; only live lanes are rewritten).
    a_buf: Option<Vec<f16>>,
    /// Layout of the last shared pack `(in_slot, input_elems)`.
    a_sig: Option<(usize, usize)>,
}

impl Slot {
    fn mem(&self) -> SlotMem {
        let mut bo_bytes = 0;
        let mut host_bytes = self.a_buf.as_ref().map_or(0, |b| b.capacity() * 2);
        for w in &self.workers {
            bo_bytes += w.bos.bytes();
            host_bytes += w.csub.capacity() * 2;
        }
        SlotMem {
            m: self.m,
            k: self.k,
            n: self.n,
            bo_bytes,
            host_bytes,
        }
    }
}

/// The resident-weight layout signature: everything that fixes where a weight
/// element lands. M-independent, so a weight packed at one M serves any M that
/// plans the same K/N tiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Sig {
    n0: usize,
    nsub: usize,
    kt: usize,
    nt: usize,
    nkt: usize,
    nnt: usize,
    wt_slot: usize,
    wt_bytes: usize,
}

/// Per-slot scratch accounting (host-mapped NPU BOs plus CPU-side buffers).
#[derive(Debug, Clone, Copy, Default)]
pub struct SlotMem {
    /// Capacity class height (the slot's `M`).
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub bo_bytes: usize,
    pub host_bytes: usize,
}

/// All scratch held by one matmul context.
#[derive(Debug, Clone, Default)]
pub struct MemStats {
    pub bo_bytes: usize,
    pub host_bytes: usize,
    pub slots: Vec<SlotMem>,
}

/// A weight matrix resident in per-worker NPU buffers.
pub struct Weight {
    m: usize,
    k: usize,
    n: usize,
    split: bool,
    bos: Vec<Bo>,
    sig: Vec<Sig>,
}

impl Weight {
    pub fn shape(&self) -> (usize, usize, usize) {
        (self.m, self.k, self.n)
    }

    /// Live bytes of the per-worker resident weight BOs.
    pub fn bytes(&self) -> usize {
        self.bos.iter().map(Bo::len).sum()
    }
}

/// A resident NPU matmul context.
///
/// One context must not be used from several threads at once; create one per
/// concurrent caller. The fan-out inside a call is threaded.
pub struct Context {
    devs: Vec<Device>,
    threads: usize,
    canonical: bool,
    /// Round a slot's `M` up to a capacity class (see [`m_class`]).
    bucket: bool,
    /// Scratch cache budget in bytes (`ROCKET_SCRATCH_MB`; 0 = unlimited).
    budget: usize,
    slots: Vec<Slot>,
    clock: u64,
    /// Clock value at the last [`Context::trim_scratch`]; slots at or below it
    /// belong to an earlier request and are the first eviction candidates.
    last_trim: u64,
}

/// The `M` capacity ladder. Classes are spaced so padding is at most ~25 % while
/// a serving process sees only a handful of distinct shapes instead of one per
/// token count. The cache key stays exact in K and N.
const M_CLASSES: &[usize] = &[
    4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384,
    448, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 3584, 4096, 5120, 6144,
    7168, 8192, 10240, 12288, 14336, 16384, 20480, 24576, 28672, 32768,
];

fn m_class(m: usize) -> usize {
    if m <= 4 {
        return 4;
    }
    match M_CLASSES.binary_search(&m) {
        Ok(i) => M_CLASSES[i],
        Err(i) => M_CLASSES
            .get(i)
            .copied()
            .unwrap_or_else(|| m.div_ceil(4096) * 4096),
    }
}

fn plan_for(canonical: bool, m: usize, k: usize, nsub: usize) -> Result<Plan> {
    if canonical && m < 256 && m.is_multiple_of(4) {
        Plan::canonical(m, k, nsub)
    } else {
        Plan::new(m, k, nsub)
    }
}

impl Context {
    /// Open `nthreads` device fds (clamped to 1..=8).
    pub fn new(nthreads: usize) -> Result<Context> {
        let threads = nthreads.clamp(1, MAX_WORKERS);
        let mut devs = Vec::with_capacity(threads);
        for _ in 0..threads {
            devs.push(Device::open()?);
        }
        let canonical = !matches!(
            crate::knobs::string("ROCKET_CTX_CANONICAL").as_deref(),
            Some("0" | "false" | "off" | "no")
        );
        let bucket = !matches!(
            crate::knobs::string("ROCKET_M_BUCKET").as_deref(),
            Some("0" | "false" | "off" | "no")
        );
        let budget_mb = match crate::knobs::int("ROCKET_SCRATCH_MB") {
            Some(v) if v >= 0 => v as usize,
            _ => 1024,
        };
        Ok(Context {
            devs,
            threads,
            canonical,
            bucket,
            budget: budget_mb * 1024 * 1024,
            slots: Vec::new(),
            clock: 0,
            last_trim: 0,
        })
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    pub fn canonical(&self) -> bool {
        self.canonical
    }

    /// Live scratch held by the per-shape cache: NPU BO bytes plus CPU buffers.
    pub fn mem_stats(&self) -> MemStats {
        let mut stats = MemStats::default();
        for slot in &self.slots {
            let m = slot.mem();
            stats.bo_bytes += m.bo_bytes;
            stats.host_bytes += m.host_bytes;
            stats.slots.push(m);
        }
        stats
    }

    /// Total live scratch bytes (BOs + CPU buffers) in the cache.
    pub fn scratch_bytes(&self) -> usize {
        self.slots
            .iter()
            .map(|s| {
                let m = s.mem();
                m.bo_bytes + m.host_bytes
            })
            .sum()
    }

    fn capacity_m(&self, m: usize, split: bool) -> usize {
        if split || !self.bucket { m } else { m_class(m) }
    }

    fn ensure(&mut self, m: usize, k: usize, n: usize, split: bool) -> Result<usize> {
        let cap = self.capacity_m(m, split);
        if let Some(i) = self
            .slots
            .iter()
            .position(|s| s.m == cap && s.k == k && s.n == n && s.split == split)
        {
            self.clock += 1;
            self.slots[i].stamp = self.clock;
            return Ok(i);
        }
        let estimate = self.estimate_slot_bytes(cap, k, n, split)?;
        if self.budget > 0 && self.scratch_bytes() + estimate > self.budget {
            self.evict_older(estimate);
        }
        let at = if self.slots.len() < MAX_SLOTS {
            None
        } else {
            Some(
                self.slots
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, s)| s.stamp)
                    .map(|(i, _)| i)
                    .unwrap_or(0),
            )
        };
        // Free the evicted slot before allocating the new one: the C library's
        // scratch cache exists to conserve each fd's 4 GiB IOVA window, and
        // holding both across the alloc is exactly the peak that window (and
        // the board's memory cap) cannot afford.
        if let Some(i) = at {
            self.slots.remove(i);
        }
        let slot = self.build_slot(cap, k, n, split)?;
        self.slots.push(slot);
        self.clock += 1;
        let at = self.slots.len() - 1;
        self.slots[at].stamp = self.clock;
        Ok(at)
    }

    /// Upper bound of a slot's bytes (BOs + host buffers), used for the budget.
    fn estimate_slot_bytes(&self, m: usize, k: usize, n: usize, split: bool) -> Result<usize> {
        let nstep = fanout_step(n, self.threads, 16);
        let mstep = fanout_step(m, self.threads, 4);
        let config = Config::from_env();
        let mut total = if split { 0 } else { k * m * 2 };
        for t in 0..self.threads {
            let (_n0, nsub, msub) = if split {
                let m0 = t * mstep;
                if m0 >= m {
                    break;
                }
                (0, n, (m - m0).min(mstep))
            } else {
                let n0 = t * nstep;
                if n0 >= n {
                    break;
                }
                (n0, (n - n0).min(nstep), m)
            };
            let plan = if split {
                Plan::new(msub, k, nsub)?
            } else {
                plan_for(self.canonical, msub, k, nsub)?
            };
            total += Bos::estimate_bytes(&plan, config) + msub * nsub * 2;
        }
        Ok(total)
    }

    /// Drop slots from earlier requests (oldest first) until `need` more bytes
    /// fit the budget. Slots touched by the current request are never evicted
    /// here; a single request is allowed to exceed the budget.
    fn evict_older(&mut self, need: usize) {
        while self.budget > 0 && self.scratch_bytes() + need > self.budget {
            let victim = self
                .slots
                .iter()
                .enumerate()
                .filter(|(_, s)| s.stamp <= self.last_trim)
                .min_by_key(|(_, s)| s.stamp)
                .map(|(i, _)| i);
            match victim {
                Some(i) => {
                    self.slots.remove(i);
                }
                None => break,
            }
        }
    }

    /// Keep only the slots touched since the previous call (the just-finished
    /// request), dropping every older per-shape scratch set. Call after a
    /// served request; repeated same-length requests keep their hot set.
    pub fn trim_scratch(&mut self) {
        let mark = self.clock;
        self.slots.retain(|s| s.stamp > self.last_trim);
        self.last_trim = mark;
    }

    /// Drop every cached scratch set.
    pub fn clear_scratch(&mut self) {
        self.slots.clear();
        self.last_trim = self.clock;
    }

    fn build_slot(&self, m: usize, k: usize, n: usize, split: bool) -> Result<Slot> {
        let nstep = fanout_step(n, self.threads, 16);
        let mstep = fanout_step(m, self.threads, 4);
        let config = Config::from_env();
        let mut workers = Vec::with_capacity(self.threads);
        for t in 0..self.threads {
            let (n0, nsub, m0, msub) = if split {
                let m0 = t * mstep;
                if m0 >= m {
                    break;
                }
                (0, n, m0, (m - m0).min(mstep))
            } else {
                let n0 = t * nstep;
                if n0 >= n {
                    break;
                }
                (n0, (n - n0).min(nstep), 0, m)
            };
            let plan = if split {
                Plan::new(msub, k, nsub)?
            } else {
                plan_for(self.canonical, msub, k, nsub)?
            };
            let bos = Bos::new(&self.devs[t], &plan, config)?;
            workers.push(Worker {
                n0,
                nsub,
                m0,
                msub,
                plan,
                bos,
                csub: Vec::new(),
                pack_sig: None,
            });
        }
        if workers.is_empty() {
            return Err(Error::shape("context/split"));
        }
        Ok(Slot {
            m,
            k,
            n,
            split,
            stamp: 0,
            workers,
            a_buf: None,
            a_sig: None,
        })
    }

    /// Pack a whole `[N,K]` weight into resident per-worker BOs.
    pub fn pack_weight(&mut self, m: usize, k: usize, n: usize, b: &[f16]) -> Result<Weight> {
        self.pack_weight_inner(m, k, n, Some(b), None)
    }

    /// Pack several weights concatenated along N (`parts[i]` is `[Ns[i],K]`).
    pub fn pack_weight_seg(&mut self, m: usize, k: usize, parts: &[&[f16]]) -> Result<Weight> {
        let n: usize = parts.iter().map(|p| p.len() / k).sum();
        let mut segs = Vec::with_capacity(parts.len());
        let mut g0 = 0;
        for p in parts {
            if p.len() % k != 0 {
                return Err(Error::shape("pack_weight_seg"));
            }
            let ns = p.len() / k;
            segs.push((g0, ns, *p));
            g0 += ns;
        }
        self.pack_weight_inner(m, k, n, None, Some(&segs))
    }

    fn pack_weight_inner(
        &mut self,
        m: usize,
        k: usize,
        n: usize,
        b: Option<&[f16]>,
        segs: Option<&[(usize, usize, &[f16])]>,
    ) -> Result<Weight> {
        if let Some(b) = b
            && b.len() != n * k
        {
            return Err(Error::shape("pack_weight"));
        }
        let si = self.ensure(m, k, n, false)?;
        let slot = &mut self.slots[si];
        let mut bos_out = Vec::with_capacity(slot.workers.len());
        let mut sig = Vec::with_capacity(slot.workers.len());
        for (i, w) in slot.workers.iter_mut().enumerate() {
            let wt_bytes = bos::wt_bytes(&w.plan);
            let mut bo = w.bos.dev.alloc_bo(wt_bytes)?;
            bos::check_iova(&[&bo])?;
            bos::zero_bo(&mut bo)?;
            if let Some(b) = b {
                bos::pack_weights_into(&mut bo, &w.plan, &b[w.n0 * k..][..w.nsub * k])?;
            } else if let Some(segs) = segs {
                bos::pack_weights_seg_into(&mut bo, &w.plan, segs, w.n0)?;
            }
            sig.push(Sig {
                n0: w.n0,
                nsub: w.nsub,
                kt: w.plan.kt,
                nt: w.plan.nt,
                nkt: w.plan.nkt,
                nnt: w.plan.nnt,
                wt_slot: w.plan.wt_slot,
                wt_bytes,
            });
            bos_out.push(bo);
            let _ = i;
        }
        Ok(Weight {
            m,
            k,
            n,
            split: false,
            bos: bos_out,
            sig,
        })
    }

    /// `C[M,N] = A[M,K] * B[N,K]^T` with a resident weight.
    pub fn matmul_prepacked(
        &mut self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f16],
        c: &mut [f16],
        w: &Weight,
    ) -> Result<()> {
        if k != w.k || n != w.n {
            return Err(Error::shape("matmul_prepacked/shape"));
        }
        let si = self.ensure(m, k, n, false)?;
        self.check_weight(si, w)?;
        self.run_slot(si, m, Some(a), None, None, Some(w), c)
    }

    /// Streaming form: B is re-packed on every call.
    pub fn matmul(
        &mut self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f16],
        b: &[f16],
        c: &mut [f16],
    ) -> Result<()> {
        if b.len() != n * k {
            return Err(Error::shape("matmul/weights"));
        }
        let si = self.ensure(m, k, n, false)?;
        self.run_slot(si, m, Some(a), Some(b), None, None, c)
    }

    /// Streaming form with a weight concatenated along N.
    pub fn matmul_seg(
        &mut self,
        m: usize,
        k: usize,
        parts: &[&[f16]],
        a: &[f16],
        c: &mut [f16],
    ) -> Result<()> {
        let n: usize = parts.iter().map(|p| p.len() / k).sum();
        let mut segs = Vec::with_capacity(parts.len());
        let mut g0 = 0;
        for p in parts {
            let ns = p.len() / k;
            segs.push((g0, ns, *p));
            g0 += ns;
        }
        let si = self.ensure(m, k, n, false)?;
        self.run_slot(si, m, Some(a), None, Some(&segs), None, c)
    }

    fn check_weight(&self, si: usize, w: &Weight) -> Result<()> {
        let slot = &self.slots[si];
        if w.split || w.sig.len() != slot.workers.len() {
            return Err(Error::tiling("weight/layout"));
        }
        for (sig, worker) in w.sig.iter().zip(&slot.workers) {
            let now = Sig {
                n0: worker.n0,
                nsub: worker.nsub,
                kt: worker.plan.kt,
                nt: worker.plan.nt,
                nkt: worker.plan.nkt,
                nnt: worker.plan.nnt,
                wt_slot: worker.plan.wt_slot,
                wt_bytes: bos::wt_bytes(&worker.plan),
            };
            if *sig != now {
                return Err(Error::tiling("weight/layout-mismatch"));
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_slot(
        &mut self,
        si: usize,
        m_live: usize,
        a: Option<&[f16]>,
        b: Option<&[f16]>,
        segs: Option<&[(usize, usize, &[f16])]>,
        resident: Option<&Weight>,
        c: &mut [f16],
    ) -> Result<()> {
        let canonical = self.canonical;
        let slot = &mut self.slots[si];
        let cap = slot.m;
        let n = slot.n;
        let k = slot.k;
        let split = slot.split;
        if m_live == 0 || m_live > cap {
            return Err(Error::shape("run/m"));
        }
        if c.len() < m_live * n {
            return Err(Error::shape("run/output"));
        }
        if split {
            if m_live != cap {
                return Err(Error::shape("run/split-m"));
            }
        } else {
            // Capacity classes over-allocate the BOs, not the compute: rebuild each
            // worker's plan for the live M so padded rows are never executed. The
            // planner's K/N fields are M-independent, so the resident weight and the
            // allocated tiles still match.
            for w in &mut slot.workers {
                w.plan = plan_for(canonical, m_live, k, w.nsub)?;
                w.m0 = 0;
                w.msub = m_live;
            }
        }
        let Slot {
            workers,
            a_buf,
            a_sig,
            ..
        } = slot;
        let prof = crate::knobs::present("ROCKET_MM_PROFILE");
        let t0 = Instant::now();
        let packed = match a {
            Some(a) => prepare_packed(workers, a_buf, a_sig, a),
            None => None,
        };
        let t_pack = t0.elapsed().as_secs_f64() * 1e3;
        let t1 = Instant::now();
        let rc = run_workers(workers, packed, a, b, segs, resident, k, n, c);
        if prof {
            eprintln!(
                "slot M={}/{} K={} N={} workers={}: pack={:.2} workers={:.2}",
                m_live,
                cap,
                k,
                n,
                workers.len(),
                t_pack,
                t1.elapsed().as_secs_f64() * 1e3
            );
        }
        rc
    }
}

fn fanout_step(n: usize, threads: usize, align: usize) -> usize {
    let threads = threads.max(1);
    let align = align.max(1);
    let s = n.div_ceil(threads).div_ceil(align) * align;
    s.max(align)
}

/// Scatter A once into the persistent shared buffer when every worker's plan has the
/// same input tiling; otherwise each worker packs its own copy.
fn prepare_packed<'a>(
    workers: &[Worker],
    buf: &'a mut Option<Vec<f16>>,
    sig: &'a mut Option<(usize, usize)>,
    a: &[f16],
) -> Option<&'a [f16]> {
    let p0 = &workers[0].plan;
    let shared = workers.len() > 1
        && !crate::knobs::present("ROCKET_NO_SHARED_PACK")
        && workers[1..].iter().all(|w| {
            w.plan.mt == p0.mt
                && w.plan.kt == p0.kt
                && w.plan.nmt == p0.nmt
                && w.plan.nkt == p0.nkt
                && w.plan.in_slot == p0.in_slot
        });
    if !shared {
        return None;
    }
    let elems = p0.input_elems();
    let now = (p0.in_slot, elems);
    if buf.as_ref().is_none_or(|v| v.len() < elems) {
        *buf = Some(vec![f16::ZERO; elems]);
    } else if *sig != Some(now) {
        // A smaller live M can change the tile width, so the padding lanes of
        // the old layout are no longer zero in the new one.
        buf.as_mut().unwrap()[..elems].fill(f16::ZERO);
    }
    *sig = Some(now);
    let v = buf.as_mut().unwrap();
    bos::scatter_input(p0, &mut v[..elems], a);
    let v = buf.as_ref().unwrap();
    Some(&v[..elems])
}

#[allow(clippy::too_many_arguments)]
fn run_workers(
    workers: &mut [Worker],
    packed: Option<&[f16]>,
    a: Option<&[f16]>,
    b: Option<&[f16]>,
    segs: Option<&[(usize, usize, &[f16])]>,
    resident: Option<&Weight>,
    k: usize,
    n: usize,
    c: &mut [f16],
) -> Result<()> {
    let base = affinity::base();
    let results: Vec<Result<()>> = thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers.len());
        for (i, w) in workers.iter_mut().enumerate() {
            let wt = resident.map(|r| r.bos[i].mem_ref());
            handles.push(scope.spawn(move || -> Result<()> {
                affinity::pin_current(i, base);
                if let Some(b) = b {
                    w.bos.pack_weights(&w.plan, &b[w.n0 * k..][..w.nsub * k])?;
                } else if let Some(segs) = segs {
                    w.bos.pack_weights_seg(&w.plan, segs, w.n0)?;
                }
                match packed {
                    Some(p) => w.bos.load_input(&w.plan, p)?,
                    None => {
                        // A capacity-class slot can serve several live M values; when
                        // the tile width changes, the BO's old non-padding lanes are
                        // stale in the new layout. Re-zero once per layout.
                        let sig = (w.plan.mt, w.plan.in_slot);
                        if w.pack_sig != Some(sig) {
                            bos::zero_bo(&mut w.bos.in_all)?;
                            w.pack_sig = Some(sig);
                        }
                        w.bos
                            .pack_input(&w.plan, a.ok_or_else(|| Error::shape("run/input"))?)?;
                    }
                }
                w.csub.resize(w.msub * w.nsub, f16::ZERO);
                let wt = match wt {
                    Some(wt) => wt,
                    None => w.bos.wt().mem_ref(),
                };
                let mut res = if w.bos.has_pong {
                    compute_kacc(&w.plan, &mut w.bos, wt, &mut w.csub)
                } else {
                    compute_cpu(&w.plan, &mut w.bos, wt, &mut w.csub)
                };
                if let Err(e) = &res
                    && e.status() == crate::Status::Tiling
                {
                    res = compute_cpu(&w.plan, &mut w.bos, wt, &mut w.csub);
                }
                res
            }));
        }
        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .unwrap_or_else(|_| Err(Error::device("worker/panic", 0)))
            })
            .collect()
    });

    if let Some(e) = results.into_iter().find_map(|r| r.err()) {
        return Err(e);
    }
    for w in workers.iter() {
        for row in 0..w.msub {
            c[(w.m0 + row) * n + w.n0..][..w.nsub]
                .copy_from_slice(&w.csub[row * w.nsub..][..w.nsub]);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn m_classes_cover_with_bounded_padding() {
        for m in 4..=32768usize {
            let c = m_class(m);
            assert!(c >= m, "class {c} < {m}");
            if m >= 64 {
                assert!(
                    (c as f64) <= m as f64 * 1.25,
                    "class {c} pads {m} by more than 25 %"
                );
            } else {
                assert!(c - m <= 8, "class {c} pads {m} by more than 8 rows");
            }
        }
        assert_eq!(m_class(7908), 8192);
        assert_eq!(m_class(8192), 8192);
        assert_eq!(m_class(8193), 10240);
    }

    #[test]
    fn capacity_class_keeps_kn_layout() {
        for (k, n) in [(512, 1024), (512, 2048), (2048, 512), (4096, 512)] {
            for m in [4, 200, 256, 7908] {
                let cap = m_class(m);
                let live = plan_for(true, m, k, n).unwrap();
                let alloc = plan_for(true, cap, k, n).unwrap();
                assert_eq!(
                    (live.kt, live.nt, live.nkt, live.nnt, live.wt_slot),
                    (alloc.kt, alloc.nt, alloc.nkt, alloc.nnt, alloc.wt_slot),
                    "m={m} cap={cap} k={k} n={n}"
                );
                assert!(live.in_slot <= alloc.in_slot);
                assert!(live.out_slot <= alloc.out_slot);
                assert!(live.input_elems() <= alloc.input_elems());
            }
        }
    }
}
