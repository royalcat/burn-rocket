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
}

/// A resident NPU matmul context.
///
/// One context must not be used from several threads at once; create one per
/// concurrent caller. The fan-out inside a call is threaded.
pub struct Context {
    devs: Vec<Device>,
    threads: usize,
    canonical: bool,
    slots: Vec<Slot>,
    clock: u64,
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
        Ok(Context {
            devs,
            threads,
            canonical,
            slots: Vec::new(),
            clock: 0,
        })
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    pub fn canonical(&self) -> bool {
        self.canonical
    }

    fn ensure(&mut self, m: usize, k: usize, n: usize, split: bool) -> Result<usize> {
        if let Some(i) = self
            .slots
            .iter()
            .position(|s| s.m == m && s.k == k && s.n == n && s.split == split)
        {
            self.clock += 1;
            self.slots[i].stamp = self.clock;
            return Ok(i);
        }
        let slot = self.build_slot(m, k, n, split)?;
        let at = if self.slots.len() < MAX_SLOTS {
            self.slots.push(slot);
            self.slots.len() - 1
        } else {
            let at = self
                .slots
                .iter()
                .enumerate()
                .min_by_key(|(_, s)| s.stamp)
                .map(|(i, _)| i)
                .unwrap_or(0);
            self.slots[at] = slot;
            at
        };
        self.clock += 1;
        self.slots[at].stamp = self.clock;
        Ok(at)
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
            let plan = if !split && self.canonical && msub < 256 && msub % 4 == 0 {
                Plan::canonical(msub, k, nsub)?
            } else {
                Plan::new(msub, k, nsub)?
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
            let mut bo = w.bos.dev.alloc_bo(w.bos.wt_all.len())?;
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
                wt_bytes: bo.len(),
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
        self.run_slot(si, Some(a), None, None, Some(w), c)
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
        self.run_slot(si, Some(a), Some(b), None, None, c)
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
        self.run_slot(si, Some(a), None, Some(&segs), None, c)
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
                wt_bytes: worker.bos.wt_all.len(),
            };
            if *sig != now {
                return Err(Error::tiling("weight/layout-mismatch"));
            }
        }
        Ok(())
    }

    fn run_slot(
        &mut self,
        si: usize,
        a: Option<&[f16]>,
        b: Option<&[f16]>,
        segs: Option<&[(usize, usize, &[f16])]>,
        resident: Option<&Weight>,
        c: &mut [f16],
    ) -> Result<()> {
        let slot = &mut self.slots[si];
        let m = slot.m;
        let n = slot.n;
        let k = slot.k;
        let Slot { workers, a_buf, .. } = slot;
        let prof = crate::knobs::present("ROCKET_MM_PROFILE");
        let t0 = Instant::now();
        let packed = match a {
            Some(a) => prepare_packed(workers, a_buf, a),
            None => None,
        };
        let t_pack = t0.elapsed().as_secs_f64() * 1e3;
        let t1 = Instant::now();
        let rc = run_workers(workers, packed, a, b, segs, resident, k, n, c);
        if prof {
            eprintln!(
                "slot M={} K={} N={} workers={}: pack={:.2} workers={:.2}",
                m,
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
    if buf.is_none() {
        let mut v = vec![f16::ZERO; p0.input_elems()];
        bos::scatter_input(p0, &mut v, a);
        *buf = Some(v);
    } else {
        let v = buf.as_mut().unwrap();
        bos::scatter_input(p0, v, a);
    }
    let v = buf.as_ref().unwrap();
    Some(&v[..p0.input_elems()])
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
                    bos::pack_weights_into(
                        &mut w.bos.wt_all,
                        &w.plan,
                        &b[w.n0 * k..][..w.nsub * k],
                    )?;
                } else if let Some(segs) = segs {
                    bos::pack_weights_seg_into(&mut w.bos.wt_all, &w.plan, segs, w.n0)?;
                }
                match packed {
                    Some(p) => w.bos.load_input(&w.plan, p)?,
                    None => w
                        .bos
                        .pack_input(&w.plan, a.ok_or_else(|| Error::shape("run/input"))?)?,
                }
                w.csub.resize(w.msub * w.nsub, f16::ZERO);
                let wt = match wt {
                    Some(wt) => wt,
                    None => w.bos.wt_all.mem_ref(),
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
