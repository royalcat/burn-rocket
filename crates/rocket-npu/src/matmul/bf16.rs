//! Native bf16 matmul (f32 operands, bf16 hardware, f64 host accumulation).
//!
//! Port of `rocket_bf16_stream.c`: N is split across worker fds, A and B are
//! truncated to bf16 during the scatter, the NPU writes fp32 partials, and the host
//! accumulates K-tiles in f64 before narrowing to f32.
//!
//! Two engines share that compute core:
//!
//! - [`Bf16Stream`] re-packs both operands on every call. This is the whole of the C
//!   library's bf16 surface: `rocket_bf16_stream.c` has no resident-weight variant
//!   (`bfs_plan_init` plans each call's `M`, and the weight scratch lives in the
//!   per-`(M,K,N)` scratch).
//! - [`Bf16Context`] + [`Bf16Weight`] pack the weight once into per-worker resident BOs
//!   and reuse it across `M` — a **Rust extension**, not a C port. The weight layout is
//!   pinned by the **canonical plan** ([`resident_plan`], i.e. [`Plan::canonical_geom`]):
//!   the streaming planner derives `Kt` — and with it `nKt` and `wt_slot` — from the
//!   call's `M`, so a stream weight is only valid for the `M` it was packed at. The
//!   canonical plan plans at the full tile height and shrinks only `Mt` to the call's
//!   `M`, which leaves `Kt`/`Nt`/`nNt`/`nKt`/`wt_slot` `M`-independent.
//!
//! The two arms run the same programs and the same f64 K-accumulation, so they are
//! bit-identical **at the same tiling**. Above `max_tile` rows the planned and the
//! canonical tiling coincide, hence `M >= max_tile` is bit-identical against the plain
//! stream too; below it the pinned `Kt` (the large-`M` one) re-partitions the K sum, so
//! the values differ in the last bits exactly like the C's own
//! "same valid bf16 product within float-accum reassociation" note for `nthreads > 1`.
//! [`Bf16Stream::new_canonical`] is the stream over the canonical tiling, i.e. the arm
//! that isolates the resident weight from the tiling choice.

use std::collections::{HashMap, VecDeque};
use std::thread;

use crate::Device;
use crate::device::uapi::DrmRocketTask;
use crate::device::{Bo, Handle, MemRef};
use crate::error::{Error, Result};
use crate::regcmd::bf16::emit;
use crate::regcmd::desc::{BATCH, CBUF_BANK_SIZE, RC_STRIDE};
use crate::regcmd::fp16::MatmulParams;

use super::bos;
use super::layout;
use super::plan::{BF16, Plan, plan_geom};
use super::run::wait_ns;

const MAX_WORKERS: usize = 8;
const MAX_SLOTS: usize = 32;

/// How a worker's tile geometry is chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tiling {
    /// The C planner on the call's `M` (`mm_plan_geom`): `Kt` grows as `M` shrinks,
    /// so the weight cube layout depends on `M`.
    Planned,
    /// [`Plan::canonical_geom`]: `Kt`/`Nt` pinned at the full tile height, only `Mt`
    /// shrinks — the layout a resident weight needs.
    Canonical,
}

impl Tiling {
    fn plan(self, m: usize, k: usize, nsub: usize) -> Result<Plan> {
        match self {
            Tiling::Planned => plan_for(m, k, nsub),
            Tiling::Canonical => Plan::canonical_geom(&BF16, m, k, nsub),
        }
    }
}

/// The M-independent plan a [`Bf16Context`] weight is packed with: the canonical plan at
/// the full tile height, so its `Kt`/`Nt`/`nNt`/`nKt`/`wt_slot` are the same for every
/// call's `M`.
///
/// This is the resident-weight contract (a Rust extension — the C library plans the
/// weight for the call's `M` and re-packs it every time). `k % 32 == 0`, `n % 16 == 0`
/// and a tile height of at least 4 rows, as everywhere in the planner.
pub fn resident_plan(k: usize, n: usize) -> Result<Plan> {
    let max_tile = crate::hw::current().max_tile as usize;
    Plan::canonical_geom(&BF16, max_tile.max(4), k, n)
}

struct Scratch {
    workers: Vec<Worker>,
    /// Per-worker streaming-weight BOs ([`Bf16Stream`]); empty in a resident-context
    /// scratch, whose weight lives in the [`Bf16Weight`]. Zeroed when allocated: the
    /// scatter only writes live tiles, so the padding lanes must start at 0.
    wt: Vec<Bo>,
    a_buf: Option<Vec<u16>>,
}

struct Worker {
    n0: usize,
    nsub: usize,
    plan: Plan,
    guard: Bo,
    regcmd: Bo,
    in_all: Bo,
    out_all: Bo,
    acc: Vec<f64>,
    csub: Vec<f32>,
    in_zeroed: bool,
}

/// A persistent bf16 streaming context (fds + per-shape scratch), the C shape: the
/// tiling follows each call's `M` and both operands are re-packed per call.
pub struct Bf16Stream {
    devs: Vec<Device>,
    threads: usize,
    tiling: Tiling,
    scratches: HashMap<(usize, usize, usize), Scratch>,
    order: VecDeque<(usize, usize, usize)>,
}

impl Bf16Stream {
    /// The C library's streaming engine (`rocket_bf16_stream_create`).
    pub fn new(nthreads: usize) -> Result<Bf16Stream> {
        Bf16Stream::create(nthreads, Tiling::Planned)
    }

    /// The same engine over the **canonical** tiling ([`Plan::canonical_geom`]): the
    /// weight layout matches [`Bf16Context`]'s resident weight, so the two arms are
    /// bit-identical at every `M`. A Rust extension — [`Bf16Stream::new`] is the C
    /// behaviour — used as the isolation arm in `bf16_selftest`.
    pub fn new_canonical(nthreads: usize) -> Result<Bf16Stream> {
        Bf16Stream::create(nthreads, Tiling::Canonical)
    }

    fn create(nthreads: usize, tiling: Tiling) -> Result<Bf16Stream> {
        let threads = nthreads.clamp(1, MAX_WORKERS);
        let mut devs = Vec::with_capacity(threads);
        for _ in 0..threads {
            devs.push(Device::open()?);
        }
        Ok(Bf16Stream {
            devs,
            threads,
            tiling,
            scratches: HashMap::new(),
            order: VecDeque::new(),
        })
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Whether this stream pins the tiling (see [`Bf16Stream::new_canonical`]).
    pub fn canonical(&self) -> bool {
        self.tiling == Tiling::Canonical
    }

    fn scratch(&mut self, m: usize, k: usize, n: usize) -> Result<&mut Scratch> {
        let key = (m, k, n);
        if !self.scratches.contains_key(&key) {
            let scratch = build_scratch(&self.devs, self.threads, m, k, n, self.tiling, true)?;
            if self.scratches.len() >= MAX_SLOTS
                && let Some(old) = self.order.pop_front()
            {
                self.scratches.remove(&old);
            }
            self.scratches.insert(key, scratch);
            self.order.push_back(key);
        }
        Ok(self.scratches.get_mut(&key).unwrap())
    }

    /// `C[M,N] = A[M,K] * B[N,K]^T`, f32 in and out, bf16 device math.
    pub fn matmul(
        &mut self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
    ) -> Result<()> {
        if a.len() != m * k || b.len() != n * k || c.len() != m * n {
            return Err(Error::shape("bf16/buffers"));
        }
        let Scratch { workers, wt, a_buf } = self.scratch(m, k, n)?;
        if workers.len() != wt.len() {
            return Err(Error::tiling("bf16/split"));
        }
        pack_stream_weights(workers, wt, b)?;
        let weights: Vec<MemRef> = wt.iter().map(Bo::mem_ref).collect();
        let packed = prepare_shared(workers, a_buf, a);
        run_split(workers, &weights, packed, a, m, n, c)
    }

    /// Per-call multi-core form, opening its own fds (the old `rocket_matmul_bf16_mt`).
    pub fn matmul_mt(
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        nthreads: usize,
    ) -> Result<()> {
        let mut s = Bf16Stream::new(nthreads)?;
        s.matmul(m, k, n, a, b, c)
    }
}

/// The resident-weight layout signature: everything that fixes where a weight element
/// lands. `M`-independent by construction (the canonical plan).
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

/// A bf16 `[N,K]` weight packed once into per-worker resident NPU buffers.
///
/// A Rust extension: `rocket_bf16_stream.c` streams both operands on every call, which
/// re-scatters the weight each time. The values (f32 truncated to bf16), the weight cube
/// layout and the f64 host K-accumulation are the stream's, and the packing is the same
/// scatter (`bfs_wt_scatter` -> [`layout::wt_scatter_tile_bf16`]).
pub struct Bf16Weight {
    k: usize,
    n: usize,
    bos: Vec<Bo>,
    sig: Vec<Sig>,
}

impl Bf16Weight {
    /// `(K, N)` of the packed weight.
    pub fn shape(&self) -> (usize, usize) {
        (self.k, self.n)
    }

    /// Live bytes of the per-worker resident weight BOs.
    pub fn bytes(&self) -> usize {
        self.bos.iter().map(Bo::len).sum()
    }
}

/// A persistent bf16 context with resident weights: pack once per `(K, N)`, then run any
/// `M` without re-packing the weight (a Rust extension, see [`resident_plan`]).
pub struct Bf16Context {
    devs: Vec<Device>,
    threads: usize,
    scratches: HashMap<(usize, usize, usize), Scratch>,
    order: VecDeque<(usize, usize, usize)>,
}

impl Bf16Context {
    /// Open `nthreads` device fds (clamped to 1..=8).
    pub fn new(nthreads: usize) -> Result<Bf16Context> {
        let threads = nthreads.clamp(1, MAX_WORKERS);
        let mut devs = Vec::with_capacity(threads);
        for _ in 0..threads {
            devs.push(Device::open()?);
        }
        Ok(Bf16Context {
            devs,
            threads,
            scratches: HashMap::new(),
            order: VecDeque::new(),
        })
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    /// The N-split step: `ceil(N / threads)` rounded up to the 16-column generator
    /// granule (the C `rocket_fanout_nstep`). The split is what makes a weight's
    /// per-worker layout, so it is fixed by `(N, threads)`.
    fn nstep(&self, n: usize) -> usize {
        fanout_step(n, self.threads, 16)
    }

    /// Pack `B[N,K]` (f32, truncated to bf16) into resident per-worker BOs.
    ///
    /// The layout is [`resident_plan`]'s, i.e. the `Kt`/`Nt` a call at `M >= max_tile`
    /// plans, so the same weight serves every `M`. The per-worker slice is
    /// `[n0, n0+nsub)` of N, exactly the slice [`Bf16Stream`] would pack.
    pub fn pack_weight(&mut self, k: usize, n: usize, b: &[f32]) -> Result<Bf16Weight> {
        if b.len() != n * k {
            return Err(Error::shape("bf16/pack_weight"));
        }
        // Validate `K`/`N` (and the CBUF fit) through the planner before allocating.
        let plan = resident_plan(k, n)?;
        let nstep = self.nstep(n);
        let mut bos_out = Vec::with_capacity(self.threads);
        let mut sig = Vec::with_capacity(self.threads);
        for t in 0..self.threads {
            let n0 = t * nstep;
            if n0 >= n {
                break;
            }
            let nsub = (n - n0).min(nstep);
            let wplan = Plan::canonical_geom(&BF16, plan.m, k, nsub)?;
            let mut bo = self.devs[t]
                .alloc_bo(wplan.nnt * wplan.nkt * wplan.wt_slot * 2 + CBUF_BANK_SIZE)?;
            bos::check_iova(&[&bo])?;
            bo.prep(0)?;
            {
                let words = u16_words(&mut bo);
                words.fill(0);
                scatter_weights(words, &wplan, n0, b);
            }
            bo.fini()?;
            sig.push(Sig {
                n0,
                nsub,
                kt: wplan.kt,
                nt: wplan.nt,
                nkt: wplan.nkt,
                nnt: wplan.nnt,
                wt_slot: wplan.wt_slot,
                wt_bytes: bo.len(),
            });
            bos_out.push(bo);
        }
        if bos_out.is_empty() {
            return Err(Error::shape("bf16/split"));
        }
        Ok(Bf16Weight {
            k,
            n,
            bos: bos_out,
            sig,
        })
    }

    fn scratch(&mut self, m: usize, k: usize, n: usize) -> Result<&mut Scratch> {
        let key = (m, k, n);
        if !self.scratches.contains_key(&key) {
            let scratch =
                build_scratch(&self.devs, self.threads, m, k, n, Tiling::Canonical, false)?;
            if self.scratches.len() >= MAX_SLOTS
                && let Some(old) = self.order.pop_front()
            {
                self.scratches.remove(&old);
            }
            self.scratches.insert(key, scratch);
            self.order.push_back(key);
        }
        Ok(self.scratches.get_mut(&key).unwrap())
    }

    /// `C[M,N] = A[M,K] * W^T` with a resident weight: only A is packed per call.
    pub fn matmul(
        &mut self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        w: &Bf16Weight,
        c: &mut [f32],
    ) -> Result<()> {
        if w.k != k || w.n != n {
            return Err(Error::shape("bf16/resident/shape"));
        }
        if a.len() != m * k || c.len() != m * n {
            return Err(Error::shape("bf16/buffers"));
        }
        let Scratch { workers, a_buf, .. } = self.scratch(m, k, n)?;
        check_weight(workers, w)?;
        let weights: Vec<MemRef> = w.bos.iter().map(Bo::mem_ref).collect();
        let packed = prepare_shared(workers, a_buf, a);
        run_split(workers, &weights, packed, a, m, n, c)
    }
}

/// A resident weight must have been packed for this context's N split *and* this call's
/// tiling: the canonical plan pins `Kt`/`Nt` per worker, so a mismatch is a layout error,
/// never a silently wrong product.
fn check_weight(workers: &[Worker], w: &Bf16Weight) -> Result<()> {
    if w.sig.len() != workers.len() || w.bos.len() != workers.len() {
        return Err(Error::tiling("bf16/weight/workers"));
    }
    for (sig, worker) in w.sig.iter().zip(workers) {
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
            return Err(Error::tiling("bf16/weight/layout"));
        }
    }
    Ok(())
}

fn build_scratch(
    devs: &[Device],
    threads: usize,
    m: usize,
    k: usize,
    n: usize,
    tiling: Tiling,
    with_wt: bool,
) -> Result<Scratch> {
    let nstep = fanout_step(n, threads, 16);
    let mut workers = Vec::with_capacity(threads);
    let mut wt = Vec::new();
    for (t, dev) in devs.iter().enumerate().take(threads) {
        let n0 = t * nstep;
        if n0 >= n {
            break;
        }
        let nsub = (n - n0).min(nstep);
        let plan = tiling.plan(m, k, nsub)?;
        let guard = dev.alloc_bo(4096)?;
        let regcmd = dev.alloc_bo(BATCH * RC_STRIDE * 8)?;
        let in_all = dev.alloc_bo(plan.nmt * plan.nkt * plan.in_slot * 2 + CBUF_BANK_SIZE)?;
        let out_all = dev.alloc_bo(BATCH * plan.out_slot * 4 + CBUF_BANK_SIZE)?;
        if with_wt {
            let mut bo = dev.alloc_bo(plan.nnt * plan.nkt * plan.wt_slot * 2 + CBUF_BANK_SIZE)?;
            bos::check_iova(&[&bo])?;
            bos::zero_bo(&mut bo)?;
            wt.push(bo);
        }
        workers.push(Worker {
            n0,
            nsub,
            plan,
            guard,
            regcmd,
            in_all,
            out_all,
            acc: Vec::new(),
            csub: Vec::new(),
            in_zeroed: false,
        });
    }
    if workers.is_empty() {
        return Err(Error::shape("bf16/split"));
    }
    Ok(Scratch {
        workers,
        wt,
        a_buf: None,
    })
}

/// Scatter the worker's `[n0, n0+nsub)` N slice of `B[N,K]` (f32 -> bf16) into its
/// weight cube slots. Padding lanes must already be zero.
fn scatter_weights(words: &mut [u16], plan: &Plan, nbase: usize, b: &[f32]) {
    for ni in 0..plan.nnt {
        let n0 = ni * plan.nt;
        let ntile = plan.n_tile(ni);
        for ki in 0..plan.nkt {
            let k0 = ki * plan.kt;
            let ktile = plan.k_tile(ki);
            let off = (ni * plan.nkt + ki) * plan.wt_slot;
            layout::wt_scatter_tile_bf16(
                &mut words[off..off + plan.wt_slot],
                b,
                plan.k,
                nbase + n0,
                k0,
                ntile,
                ktile,
            );
        }
    }
}

/// Re-pack every worker's weight slice of `b` into its own streaming BO (the C
/// `bfs_run_worker` weight pass), one thread per worker fd.
fn pack_stream_weights(workers: &mut [Worker], wt: &mut [Bo], b: &[f32]) -> Result<()> {
    let base = crate::device::affinity::base();
    let results: Vec<Result<()>> = thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers.len());
        for (i, (w, bo)) in workers.iter_mut().zip(wt.iter_mut()).enumerate() {
            handles.push(scope.spawn(move || -> Result<()> {
                crate::device::affinity::pin_current(i, base);
                bo.prep(0)?;
                scatter_weights(u16_words(bo), &w.plan, w.n0, b);
                bo.fini()
            }));
        }
        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .unwrap_or_else(|_| Err(Error::device("bf16/worker", 0)))
            })
            .collect()
    });
    join(results)
}

/// Run every worker's N slice: pack A (or load the shared pack), then `compute` against
/// the worker's weight BO, and gather the f32 partials into `c`.
fn run_split(
    workers: &mut [Worker],
    weights: &[MemRef],
    packed: Option<&[u16]>,
    a: &[f32],
    m: usize,
    n: usize,
    c: &mut [f32],
) -> Result<()> {
    let base = crate::device::affinity::base();
    let results: Vec<Result<()>> = thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers.len());
        for (i, worker) in workers.iter_mut().enumerate() {
            let wt = weights[i];
            handles.push(scope.spawn(move || -> Result<()> {
                crate::device::affinity::pin_current(i, base);
                match packed {
                    Some(p) => worker.load_input(p)?,
                    None => worker.pack_input(a)?,
                }
                worker.compute(m, wt)
            }));
        }
        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .unwrap_or_else(|_| Err(Error::device("bf16/worker", 0)))
            })
            .collect()
    });
    join(results)?;
    for worker in workers.iter() {
        for row in 0..m {
            c[row * n + worker.n0..][..worker.nsub]
                .copy_from_slice(&worker.csub[row * worker.nsub..][..worker.nsub]);
        }
    }
    Ok(())
}

fn join(results: Vec<Result<()>>) -> Result<()> {
    match results.into_iter().find_map(|r| r.err()) {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

impl Worker {
    fn load_input(&mut self, packed: &[u16]) -> Result<()> {
        let n = self.plan.input_elems();
        self.in_all.prep(0)?;
        {
            let words = u16_words(&mut self.in_all);
            words[..n].copy_from_slice(&packed[..n]);
        }
        self.in_all.fini()
    }

    fn pack_input(&mut self, a: &[f32]) -> Result<()> {
        let k = self.plan.k;
        self.in_all.prep(0)?;
        {
            let words = u16_words(&mut self.in_all);
            if !self.in_zeroed {
                words.fill(0);
                self.in_zeroed = true;
            }
            for mi in 0..self.plan.nmt {
                let m0 = mi * self.plan.mt;
                let mtile = self.plan.m_tile(mi);
                for ki in 0..self.plan.nkt {
                    let k0 = ki * self.plan.kt;
                    let ktile = self.plan.k_tile(ki);
                    let off = (mi * self.plan.nkt + ki) * self.plan.in_slot;
                    layout::feat_scatter_bf16(
                        &mut words[off..off + self.plan.in_slot],
                        a,
                        k,
                        m0,
                        k0,
                        mtile,
                        ktile,
                    );
                }
            }
        }
        self.in_all.fini()
    }

    fn compute(&mut self, m: usize, wt: MemRef) -> Result<()> {
        let plan = &self.plan;
        self.acc.clear();
        self.acc.resize(m * self.nsub, 0.0);
        let mut tasks: Vec<DrmRocketTask> = Vec::with_capacity(BATCH);
        let mut meta: Vec<(usize, usize, usize, usize, usize)> = Vec::with_capacity(BATCH);
        let mut ops = Vec::with_capacity(160);
        let mut done = 0usize;
        let total = plan.jobs();
        let mut nb = 0usize;

        for mi in 0..plan.nmt {
            let m0 = mi * plan.mt;
            let mtile = plan.m_tile(mi);
            for ni in 0..plan.nnt {
                let n0 = ni * plan.nt;
                let ntile = plan.n_tile(ni);
                for ki in 0..plan.nkt {
                    if nb == 0 {
                        self.regcmd.prep(0)?;
                    }
                    let ktile = plan.k_tile(ki);
                    let out_off = nb * plan.out_slot;
                    let input = self.in_all.dma_address()
                        + ((mi * plan.nkt + ki) * plan.in_slot * 2) as u64;
                    let weights = wt.dma + ((ni * plan.nkt + ki) * plan.wt_slot * 2) as u64;
                    let output = self.out_all.dma_address() + (out_off * 4) as u64;
                    let p = MatmulParams::new(
                        mtile as u32,
                        ktile as u32,
                        ntile as u32,
                        input as u32,
                        weights as u32,
                        output as u32,
                    );
                    ops.clear();
                    let count = emit(&p, &mut ops)?;
                    if ops.len() > RC_STRIDE {
                        return Err(Error::tiling("bf16/regcmd"));
                    }
                    {
                        let bytes = self.regcmd.as_mut_slice();
                        let base = nb * RC_STRIDE * 8;
                        for (j, op) in ops.iter().enumerate() {
                            bytes[base + j * 8..base + j * 8 + 8]
                                .copy_from_slice(&op.0.to_ne_bytes());
                        }
                    }
                    let regcmd = (self.regcmd.dma_address() + (nb * RC_STRIDE * 8) as u64) as u32;
                    tasks.push(DrmRocketTask {
                        regcmd,
                        regcmd_count: count as u32,
                    });
                    meta.push((m0, n0, mtile, ntile, out_off));
                    nb += 1;
                    done += 1;

                    if nb == BATCH || done == total {
                        self.regcmd.fini()?;
                        self.out_all.prep(0)?;
                        self.out_all.fini()?;
                        let in_h: [Handle; 3] =
                            [self.in_all.handle(), wt.handle, self.regcmd.handle()];
                        let out_h = [self.out_all.handle()];
                        self.guard.device().submit(&tasks, &in_h, &out_h, 0)?;
                        self.out_all.prep(wait_ns())?;
                        {
                            let words = f32_words(&self.out_all);
                            for &(m0, n0, mtile, ntile, off) in &meta {
                                let slot = &words[off..off + plan.out_slot];
                                layout::detile_accum_bf16(
                                    &mut self.acc,
                                    self.nsub,
                                    slot,
                                    m0,
                                    n0,
                                    mtile,
                                    ntile,
                                );
                            }
                        }
                        self.out_all.fini()?;
                        nb = 0;
                        tasks.clear();
                        meta.clear();
                    }
                }
            }
        }
        self.csub.clear();
        self.csub.extend(self.acc.iter().map(|v| *v as f32));
        Ok(())
    }
}

fn prepare_shared<'a>(
    workers: &[Worker],
    buf: &'a mut Option<Vec<u16>>,
    a: &[f32],
) -> Option<&'a [u16]> {
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
        let mut v = vec![0u16; p0.input_elems()];
        scatter_shared(p0, &mut v, a);
        *buf = Some(v);
    } else {
        let v = buf.as_mut().unwrap();
        v.fill(0);
        scatter_shared(p0, v, a);
    }
    let v = buf.as_ref().unwrap();
    Some(&v[..p0.input_elems()])
}

fn scatter_shared(plan: &Plan, dst: &mut [u16], a: &[f32]) {
    for mi in 0..plan.nmt {
        let m0 = mi * plan.mt;
        let mtile = plan.m_tile(mi);
        for ki in 0..plan.nkt {
            let k0 = ki * plan.kt;
            let ktile = plan.k_tile(ki);
            let off = (mi * plan.nkt + ki) * plan.in_slot;
            layout::feat_scatter_bf16(
                &mut dst[off..off + plan.in_slot],
                a,
                plan.k,
                m0,
                k0,
                mtile,
                ktile,
            );
        }
    }
}

fn plan_for(m: usize, k: usize, n: usize) -> Result<Plan> {
    let (mt, kt, nt) = plan_geom(&BF16, m, k, n, 0)?;
    Ok(Plan::derive_geom(&BF16, m, k, n, mt, kt, nt))
}

fn fanout_step(n: usize, threads: usize, align: usize) -> usize {
    let threads = threads.max(1);
    let align = align.max(1);
    n.div_ceil(threads).div_ceil(align) * align
}

fn u16_words(bo: &mut Bo) -> &mut [u16] {
    let bytes = bo.as_mut_slice();
    assert_eq!(bytes.len() % 2, 0);
    // SAFETY: `u16` is a 2-byte plain-old-data type, the BO mapping is page-aligned and
    // the length is even.
    unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr().cast::<u16>(), bytes.len() / 2) }
}

fn f32_words(bo: &Bo) -> &[f32] {
    let bytes = bo.as_slice();
    assert_eq!(bytes.len() % 4, 0);
    // SAFETY: the bf16 output cube is raw fp32 and the mapping is page-aligned.
    unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<f32>(), bytes.len() / 4) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shapes the app's bf16 projections use (q|k|v|o, gate|up and down), plus a
    /// narrow one.
    const SHAPES: &[(usize, usize)] = &[(1024, 768), (1024, 3072), (4096, 1024), (512, 2048)];

    /// The profile's `Mt`/`Nt` cap (256 on RK3588).
    fn max_tile() -> usize {
        crate::hw::current().max_tile as usize
    }

    /// The resident contract: everything that fixes where a weight element lands is the
    /// same for every call's `M`. `resident_plan` is [M-free] by construction, so the
    /// proof is that each call's canonical plan keeps the pinned K/N fields.
    #[test]
    fn the_canonical_plan_pins_the_weight_layout_across_m() {
        for &(k, n) in SHAPES {
            let weight = resident_plan(k, n).unwrap();
            for m in [4usize, 64, 256, 512] {
                let call = Plan::canonical_geom(&BF16, m, k, n).unwrap();
                assert_eq!(
                    (call.kt, call.nt, call.nnt, call.nkt, call.wt_slot),
                    (weight.kt, weight.nt, weight.nnt, weight.nkt, weight.wt_slot),
                    "m={m} k={k} n={n}"
                );
                // `Mt` is the only thing the call's M moves: `min(max_tile, M)`, and the
                // M-tile count follows from it.
                assert_eq!(call.mt, max_tile().min(m), "m={m} k={k} n={n}");
                assert_eq!(call.nmt, m.div_ceil(call.mt));
                assert!(call.in_slot <= weight.in_slot && call.out_slot <= weight.out_slot);
            }
        }
    }

    /// ... and that the pin is load-bearing: the C planner's `Kt` shrinks as `M` grows,
    /// which is exactly why a stream weight cannot be reused across `M`.
    #[test]
    fn the_streaming_planner_derives_kt_from_m() {
        let (k, n) = (1024usize, 768usize);
        let small = Plan::canonical_geom(&BF16, max_tile(), k, n).unwrap();
        let planned = plan_for(4, k, n).unwrap();
        assert_eq!(planned.mt, 4);
        assert!(
            planned.kt > small.kt,
            "the small-M planner must pick a larger Kt (got {} vs {})",
            planned.kt,
            small.kt
        );
        assert_ne!(
            (planned.nkt, planned.wt_slot),
            (small.nkt, small.wt_slot),
            "a different Kt must change the weight layout"
        );
        // At the full tile height the two planners agree, which is the M above which a
        // resident weight and the plain stream are bit-identical.
        for m in [256usize, 512, 4096] {
            let a = plan_for(m, k, n).unwrap();
            let b = Plan::canonical_geom(&BF16, m, k, n).unwrap();
            assert_eq!((a.mt, a.kt, a.nt), (b.mt, b.kt, b.nt), "m={m}");
        }
    }

    /// The N split is what a resident weight is packed against, so the per-worker plans
    /// must agree between `pack_weight`'s and `matmul`'s view of `(k, nsub)`.
    #[test]
    fn every_worker_slice_has_one_layout_per_m() {
        for &(k, n) in SHAPES {
            for threads in [1usize, 3, 5, 8] {
                let step = fanout_step(n, threads, 16);
                for t in 0..threads {
                    let n0 = t * step;
                    if n0 >= n {
                        break;
                    }
                    let nsub = (n - n0).min(step);
                    let pack = Plan::canonical_geom(&BF16, max_tile(), k, nsub).unwrap();
                    for m in [4usize, 64, 256, 512] {
                        let call = Plan::canonical_geom(&BF16, m, k, nsub).unwrap();
                        assert_eq!(
                            (call.kt, call.nt, call.nnt, call.nkt, call.wt_slot),
                            (pack.kt, pack.nt, pack.nnt, pack.nkt, pack.wt_slot),
                            "k={k} n={n} threads={threads} nsub={nsub} m={m}"
                        );
                    }
                }
            }
        }
    }

    /// `resident_plan` is the planner itself: a bad shape is a typed error, and the
    /// pinned K/N fields are the ones the planner would choose at the full tile height.
    #[test]
    fn resident_plan_validates_through_the_planner() {
        assert!(resident_plan(1024, 768).is_ok());
        assert!(resident_plan(1000, 768).is_err()); // K % 32
        assert!(resident_plan(1024, 770).is_err()); // N % 16
        let p = resident_plan(1024, 768).unwrap();
        let (mt, kt, nt) = plan_geom(&BF16, max_tile(), 1024, 768, 0).unwrap();
        assert_eq!((p.mt, p.kt, p.nt), (mt, kt, nt));
        assert_eq!(p.m, max_tile());
    }

    /// The weight scatter reads the **worker's** column slice of the whole `[N,K]`
    /// weight (`nbase = n0`), which is the one part the resident pack adds over the
    /// stream's per-slice buffer: a non-zero base must land the same lanes the device
    /// reads back, and the padding lanes must stay zero for the device's zero-weight
    /// muting.
    #[test]
    fn the_worker_weight_scatter_reads_its_own_column_slice() {
        let (k, n) = (1024usize, 768usize);
        let b: Vec<f32> = (0..n * k).map(|i| i as f32 * 1e-3 - 7.0).collect();
        // A whole-N slice (3 N-tiles, 3 K-tiles at the pinned Kt) and a tail slice at a
        // non-zero base (1 N-tile): both must land the device's lanes.
        for (n0, nsub) in [(0usize, 768usize), (640, 128)] {
            let plan = Plan::canonical_geom(&BF16, 256, k, nsub).unwrap();
            assert!(plan.nkt > 1, "nsub={nsub}");
            let mut words = vec![0u16; plan.nnt * plan.nkt * plan.wt_slot];
            scatter_weights(&mut words, &plan, n0, &b);

            let mut live = vec![false; plan.nnt * plan.nkt * plan.wt_slot];
            for ni in 0..plan.nnt {
                for ki in 0..plan.nkt {
                    let ntile = plan.n_tile(ni);
                    let ktile = plan.k_tile(ki);
                    let off = (ni * plan.nkt + ki) * plan.wt_slot;
                    let slot = &words[off..off + plan.wt_slot];
                    for kk in 0..ntile {
                        for cc in 0..ktile {
                            let lane = layout::wt_idx_i16(ktile, kk + 1, cc + 1);
                            live[off + lane] = true;
                            let want = b[(n0 + ni * plan.nt + kk) * k + ki * plan.kt + cc];
                            assert_eq!(
                                slot[lane],
                                layout::f32_to_bf16_bits(want),
                                "n0={n0} nsub={nsub} ni={ni} ki={ki} kk={kk} cc={cc}"
                            );
                        }
                    }
                }
            }
            for (i, is_live) in live.iter().enumerate() {
                if !is_live {
                    assert_eq!(words[i], 0, "padding lane {i} must stay zero");
                }
            }
        }
    }
}
