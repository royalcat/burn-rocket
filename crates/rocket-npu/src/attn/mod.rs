//! Masked grouped-query flash attention on the fp16 matmul engine.
//!
//! The layout and semantics match the C library's `rocket_flash_attn_fp16`:
//!
//! - `Q`: `[n_head][n_tokens][head_dim]`
//! - `K`: `[n_kv_heads][n_kv][head_dim]`
//! - `V`: `[n_kv_heads][dv][n_kv]`
//! - `mask`: `[n_tokens][n_kv]` additive f16 (or none)
//! - `out`: `[n_head][n_tokens][dv]`
//!
//! Scores are brought host-side for the mandatory additive mask, scale, optional
//! soft-cap and softmax, then uploaded as the AV operand (the C default,
//! `ROCKET_ATTN_HOST_SOFTMAX=1`). Heads split into contiguous ranges across worker
//! fds, one per NPU scheduling entity. The host softmax itself is
//! [`softmax::Softmax`] (fast polynomial `exp` by default, `SOFTMAX_EXACT=1` for
//! the original scalar path).
//!
//! # Chained per-head batching (`ROCKET_FA_CHAIN`, default ON)
//!
//! Port of the C `fa_heads_range_batched` (`ROCKET_FA_CHAIN`): every per-head QK matmul
//! of a worker shares one `(Tp, dk, Kn)` shape and its per-head AV matmuls one
//! `(Tp, Kn, dv)`, so a worker's head range runs in GROUPS — gather the group's
//! operands, ONE [`MmBatch`] run for its QK, the host mask + softmax per head, one
//! [`MmBatch`] run for its AV — instead of one submit + fence wait per matmul. Per-head
//! attention GEMMs are tiny and dispatch-bound, so this attacks the submit floor rather
//! than the MAC count. The mask and softmax sit BETWEEN the two batched jobs (never
//! inside a chained QK→softmax→AV), which is what keeps the host-softmax default.
//!
//! The group is bounded by the score-matrix budget `ROCKET_FA_CHAIN_ELEMS` (default 32M
//! padded `Tp*Kn` elements): `G = clamp(budget / (Tp*Kn), 1, heads_in_range)`, so a long
//! context shrinks toward one head and `G == 1` runs the per-head path outright.
//! `ROCKET_FA_CHAIN=0` forces the per-head path for every call (the A/B switch).
//!
//! Numerics: the batched path is bit-identical to the per-head one only while both
//! accumulate the same way. [`MmBatch`] always accumulates the K-tiles in fp32 on the
//! CPU, while the per-head [`Scratch`] prefers the on-NPU fp16 K-accumulation (KACC)
//! whenever its ping-pong BOs exist — and KACC rounds each K-tile's f16 partial into the
//! running f16 sum, which the CPU path does not. A shape with a SINGLE K-tile is one fp32
//! tile narrowed once either way, so the bits agree; a shape with more does not, and it
//! keeps the per-head path. `prepare` therefore resolves that per SHAPE: the QK shape's K
//! is the head dim (one tile for every geometry in use), the AV shape's K is the padded
//! key count (one tile up to `kt` keys, so a long context tiles it). `ROCKET_NO_KACC=1`
//! removes the divergence and chains both shapes at any length.

use std::thread;

use half::f16;

use crate::Device;
use crate::device::affinity;
use crate::error::{Error, Result};
use crate::knobs;

use crate::matmul::Config;
use crate::matmul::MmBatch;
use crate::matmul::bos::Bos;
use crate::matmul::plan::Plan;
use crate::matmul::run::{compute_cpu, compute_kacc};

mod softmax;

pub use softmax::{Softmax, SoftmaxMode};

const MAX_WORKERS: usize = 8;

/// Default score-matrix budget of one chained head group, in padded `Tp*Kn` elements
/// (C `ROCKET_FA_CHAIN_ELEMS`).
const CHAIN_ELEMS: u64 = 32 << 20;

/// Whether a worker batches its per-head QK/AV matmuls into one job per group
/// (C `fa_chain`, default ON).
///
/// Resolved per call rather than once per process like the C library: attention runs
/// once per layer, so a `getenv` here is free and it is what lets one process A/B the
/// two paths.
fn chain_enabled() -> bool {
    knobs::boolean("ROCKET_FA_CHAIN").unwrap_or(true)
}

/// The chained group's score-matrix budget in padded elements (C `fa_chain_elems`);
/// zero and negative values are clamped up to one, as in C.
fn chain_elems() -> u64 {
    knobs::int("ROCKET_FA_CHAIN_ELEMS")
        .map(|v| v.max(1) as u64)
        .unwrap_or(CHAIN_ELEMS)
}

/// Heads per chained group: `clamp(budget / (Tp*Kn), 1, heads)`.
///
/// The bound is the group's host score scratch (`G` heads' `[Tp,Kn]` matrices held at
/// once), which is why it falls toward one as the key axis grows.
fn group_size(budget: u64, tp: usize, kn: usize, heads: usize) -> usize {
    if heads == 0 {
        return 0;
    }
    let per_head = (tp as u64).saturating_mul(kn as u64);
    let g = budget.checked_div(per_head).unwrap_or(heads as u64);
    g.clamp(1, heads as u64) as usize
}

/// Copy `Q[h]`'s `t` live rows into a zero-padded `[tp, dk]` tile.
fn gather_q(dst: &mut [f16], q: &[f16], h: usize, t: usize, dk: usize) {
    dst.fill(f16::ZERO);
    for row in 0..t {
        let src = (h * t + row) * dk;
        let d = row * dk;
        dst[d..d + dk].copy_from_slice(&q[src..src + dk]);
    }
}

/// Copy `K[hk]`'s `kv` live rows into a zero-padded `[kn, dk]` tile.
fn gather_k(dst: &mut [f16], k: &[f16], hk: usize, kv: usize, dk: usize) {
    dst.fill(f16::ZERO);
    for row in 0..kv {
        let src = (hk * kv + row) * dk;
        let d = row * dk;
        dst[d..d + dk].copy_from_slice(&k[src..src + dk]);
    }
}

/// Transpose `V[hk]` (`[dv, kv]`) into a zero-padded `[dv, kn]` tile — the AV B operand.
fn gather_v(dst: &mut [f16], v: &[f16], hk: usize, kv: usize, dv: usize, kn: usize) {
    dst.fill(f16::ZERO);
    for c in 0..dv {
        let src = (hk * dv + c) * kv;
        let d = c * kn;
        dst[d..d + kv].copy_from_slice(&v[src..src + kv]);
    }
}

/// Copy head `oi`'s `t` live rows out of its padded `[tp, dv]` result tile.
fn scatter_ctx(out: &mut [f16], ctx: &[f16], oi: usize, t: usize, dv: usize) {
    for row in 0..t {
        let dst = (oi * t + row) * dv;
        let src = row * dv;
        out[dst..dst + dv].copy_from_slice(&ctx[src..src + dv]);
    }
}

/// Host scratch for one chained head group: `cap` heads' worth of every per-head matrix
/// the group path touches, head-major.
///
/// Grow-only and refilled per group — the gathers zero their own tile, the matmuls and
/// the softmax write their whole output — so a smaller request simply uses the head of
/// the allocation instead of reallocating (which at long context would cross glibc's
/// mmap threshold and mmap/fault/munmap the score matrices every layer).
#[derive(Debug, Default)]
struct GroupBufs {
    /// Heads the buffers are sized for.
    cap: usize,
    /// Group queries, `cap * tp*dk`.
    qh: Vec<f16>,
    /// Group keys, `cap * kn*dk`.
    kh: Vec<f16>,
    /// Group values, transposed: `cap * dv*kn`.
    vh: Vec<f16>,
    /// Group scores, `cap * tp*kn`.
    sc: Vec<f16>,
    /// Group probabilities, `cap * tp*kn`.
    p: Vec<f16>,
    /// Group context (one head's AV result), `cap * tp*dv`.
    ctx: Vec<f16>,
}

impl GroupBufs {
    /// Grow to `heads` heads of the `(tp, kn, dk, dv)` shape.
    fn ensure(&mut self, heads: usize, tp: usize, kn: usize, dk: usize, dv: usize) {
        for (buf, need) in [
            (&mut self.qh, heads * tp * dk),
            (&mut self.kh, heads * kn * dk),
            (&mut self.vh, heads * dv * kn),
            (&mut self.sc, heads * tp * kn),
            (&mut self.p, heads * tp * kn),
            (&mut self.ctx, heads * tp * dv),
        ] {
            if buf.len() < need {
                buf.resize(need, f16::ZERO);
            }
        }
        self.cap = self.cap.max(heads);
    }

    /// Live fp16 elements held by the group buffers.
    fn host_elems(&self) -> usize {
        self.qh.capacity()
            + self.kh.capacity()
            + self.vh.capacity()
            + self.sc.capacity()
            + self.p.capacity()
            + self.ctx.capacity()
    }

    /// Release the buffers; the next group reallocates.
    fn release(&mut self) {
        *self = GroupBufs::default();
    }

    /// Whether the scratch is far larger than the current working shape needs for the
    /// heads it holds — the same 4x rule [`Worker::shrink_host`] applies to the per-head
    /// matrices.
    fn oversized(&self, tp: usize, kn: usize) -> bool {
        self.sc.capacity() > self.cap.max(1) * tp * kn * 4
    }
}

struct Scratch {
    plan: Plan,
    bos: Bos,
    stamp: u64,
}

/// Per-worker attention scratch accounting.
#[derive(Debug, Clone, Copy, Default)]
pub struct ScratchMem {
    /// Padded query/key shape the worker's buffers are sized for.
    pub shape: (usize, usize),
    pub bo_bytes: usize,
    pub host_bytes: usize,
}

/// All scratch held by one attention context.
#[derive(Debug, Clone, Default)]
pub struct MemStats {
    pub bo_bytes: usize,
    pub host_bytes: usize,
    pub workers: Vec<ScratchMem>,
}

impl Scratch {
    fn new(dev: &Device, m: usize, k: usize, n: usize) -> Result<Self> {
        let plan = Plan::new(m, k, n)?;
        let bos = Bos::new(dev, &plan, Config::from_env())?;
        Ok(Scratch {
            plan,
            bos,
            stamp: 0,
        })
    }

    fn matmul(&mut self, a: &[f16], b: &[f16], c: &mut [f16]) -> Result<()> {
        self.bos.pack_input(&self.plan, a)?;
        self.bos.pack_weights(&self.plan, b)?;
        let wt = self.bos.wt().mem_ref();
        if self.bos.has_pong {
            match compute_kacc(&self.plan, &mut self.bos, wt, c) {
                Ok(()) => return Ok(()),
                Err(e) if e.status() == crate::Status::Tiling => {}
                Err(e) => return Err(e),
            }
        }
        compute_cpu(&self.plan, &mut self.bos, wt, c)
    }
}

/// A two-entry scratch cache. A forward alternates between a sliding-layer chunk
/// and a full-layer chunk; keeping both avoids reallocating the large `[Tp, Kn]`
/// BOs on every layer, while the bound stops a short request from retaining an
/// arbitrarily old shape forever.
struct ScratchCache {
    entries: Vec<Scratch>,
    clock: u64,
    /// Clock at the last [`ScratchCache::trim`]; entries at or below it belong
    /// to an earlier request.
    last_trim: u64,
}

impl ScratchCache {
    fn new(dev: &Device, m: usize, k: usize, n: usize) -> Result<Self> {
        Ok(ScratchCache {
            entries: vec![Scratch::new(dev, m, k, n)?],
            clock: 1,
            last_trim: 0,
        })
    }

    fn get(&mut self, dev: &Device, m: usize, k: usize, n: usize) -> Result<()> {
        self.clock += 1;
        if let Some(i) = self
            .entries
            .iter()
            .position(|s| s.plan.m == m && s.plan.k == k && s.plan.n == n)
        {
            self.entries[i].stamp = self.clock;
            // `current_mut` is the last entry, so a hit has to move to the back.
            if i + 1 != self.entries.len() {
                let hit = self.entries.remove(i);
                self.entries.push(hit);
            }
            return Ok(());
        }
        if self.entries.len() >= 2 {
            let oldest = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, s)| s.stamp)
                .map(|(i, _)| i)
                .unwrap_or(0);
            // Drop before allocating: the new BOs can be as large as the old.
            self.entries.remove(oldest);
        }
        let mut scratch = Scratch::new(dev, m, k, n)?;
        scratch.stamp = self.clock;
        self.entries.push(scratch);
        Ok(())
    }

    fn current_mut(&mut self) -> &mut Scratch {
        self.entries
            .last_mut()
            .expect("attention scratch cache is never empty")
    }

    fn bo_bytes(&self) -> usize {
        self.entries.iter().map(|s| s.bos.bytes()).sum()
    }

    /// Drop entries from earlier requests, keeping every shape touched by the
    /// just-finished one (the first and last chunk can differ in length).
    fn trim(&mut self) {
        let mark = self.clock;
        self.entries.retain(|s| s.stamp > self.last_trim);
        self.last_trim = mark;
    }
}

struct Worker {
    dev: Device,
    /// Per-head scratch, two shape classes: the `ROCKET_FA_CHAIN=0` path, the fallback
    /// for a shape whose plan splits K, and any group of one.
    qk: ScratchCache,
    av: ScratchCache,
    /// The chained path's persistent batch contexts (one shape each), created on first
    /// use so the per-head path allocates neither.
    bqk: Option<MmBatch>,
    bav: Option<MmBatch>,
    /// Host scratch for one chained head group.
    grp: GroupBufs,
    /// Whether this shape's QK / AV matmul may batch and stay bit-identical to the
    /// per-head path (see [`Worker::resolve_batching`]).
    qk_batch: bool,
    av_batch: bool,
    qh: Vec<f16>,
    kh: Vec<f16>,
    vh: Vec<f16>,
    p: Vec<f16>,
    scores: Vec<f16>,
    ctx: Vec<f16>,
    out: Vec<f16>,
    /// Reused row buffer + kernel mode for the host softmax.
    sm: Softmax,
    head_dim: usize,
    dv: usize,
    tp: usize,
    kn: usize,
}

impl Worker {
    fn mem(&self) -> ScratchMem {
        let host = self.qh.capacity()
            + self.kh.capacity()
            + self.vh.capacity()
            + self.p.capacity()
            + self.scores.capacity()
            + self.ctx.capacity()
            + self.out.capacity()
            + self.grp.host_elems();
        let batch = self.bqk.as_ref().map_or(0, MmBatch::bytes)
            + self.bav.as_ref().map_or(0, MmBatch::bytes);
        ScratchMem {
            bo_bytes: self.qk.bo_bytes() + self.av.bo_bytes() + batch,
            host_bytes: host * 2,
            shape: (self.tp, self.kn),
        }
    }

    fn new(dev: Device) -> Result<Self> {
        // Small placeholder shapes; replaced on the first real call.
        let qk = ScratchCache::new(&dev, 4, 32, 16)?;
        let av = ScratchCache::new(&dev, 4, 32, 16)?;
        Ok(Worker {
            dev,
            qk,
            av,
            bqk: None,
            bav: None,
            grp: GroupBufs::default(),
            qk_batch: false,
            av_batch: false,
            qh: Vec::new(),
            kh: Vec::new(),
            vh: Vec::new(),
            p: Vec::new(),
            scores: Vec::new(),
            ctx: Vec::new(),
            out: Vec::new(),
            sm: Softmax::new(),
            head_dim: 0,
            dv: 0,
            tp: 0,
            kn: 0,
        })
    }

    fn prepare(&mut self, t: usize, kv: usize, dk: usize, dv: usize) -> Result<()> {
        self.tp = t.div_ceil(4) * 4;
        self.kn = kv.div_ceil(32) * 32;
        self.head_dim = dk;
        self.dv = dv;
        if self.kn == 0 || self.tp == 0 {
            return Err(Error::shape("flash_attn/empty"));
        }
        self.qh.resize(self.tp * dk, f16::ZERO);
        self.kh.resize(self.kn * dk, f16::ZERO);
        self.vh.resize(dv * self.kn, f16::ZERO);
        self.scores.resize(self.tp * self.kn, f16::ZERO);
        self.p.resize(self.tp * self.kn, f16::ZERO);
        self.ctx.resize(self.tp * dv, f16::ZERO);
        self.shrink_host();
        self.qk.get(&self.dev, self.tp, dk, self.kn)?;
        self.av.get(&self.dev, self.tp, self.kn, dv)?;
        self.resolve_batching(dk, dv);
        Ok(())
    }

    /// Decide, per operand shape, whether the chained path can be bit-identical to the
    /// per-head one.
    ///
    /// One K-tile is one fp32 tile narrowed once on either path, so the bits match
    /// whether the per-head scratch takes KACC or the CPU accumulator. More than one
    /// K-tile differs: KACC adds each tile's fp16 partial into a running fp16 sum,
    /// [`MmBatch`] adds every tile in fp32 on the CPU. So a multi-K-tile shape keeps the
    /// per-head path — QK never trips this (K is the head dim, one tile), AV does once
    /// the padded key count exceeds the plan's `kt`. Without KACC (`ROCKET_NO_KACC=1`,
    /// or a scratch without its ping-pong BOs) both paths accumulate in fp32 and every
    /// shape may batch.
    fn resolve_batching(&mut self, dk: usize, dv: usize) {
        let kacc = self.qk.current_mut().bos.has_pong;
        self.qk_batch = batch_matches_per_head(kacc, Plan::new(self.tp, dk, self.kn));
        self.av_batch = batch_matches_per_head(kacc, Plan::new(self.tp, self.kn, dv));
    }

    /// Release the long-request host matrices once the working shape is much
    /// smaller; small changes keep the capacity.
    fn shrink_host(&mut self) {
        if self.scores.capacity() > self.tp * self.kn * 4 {
            self.qh.shrink_to_fit();
            self.kh.shrink_to_fit();
            self.vh.shrink_to_fit();
            self.scores.shrink_to_fit();
            self.p.shrink_to_fit();
            self.ctx.shrink_to_fit();
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run(
        &mut self,
        t: usize,
        kv: usize,
        heads: &[usize],
        gqa: usize,
        scale: f32,
        softcap: f32,
        q: &[f16],
        k: &[f16],
        v: &[f16],
        mask: Option<&[f16]>,
    ) -> Result<()> {
        let dv = self.dv;
        self.out.clear();
        self.out.resize(heads.len() * t * dv, f16::ZERO);
        // A group of one chains nothing, so it takes the per-head path — and so does a
        // worker whose two shapes both need their own fp16 K-accumulation.
        let g = if chain_enabled() && (self.qk_batch || self.av_batch) {
            group_size(chain_elems(), self.tp, self.kn, heads.len())
        } else {
            1
        };
        if g > 1 {
            self.run_grouped(t, kv, heads, gqa, scale, softcap, q, k, v, mask, g)
        } else {
            self.run_per_head(t, kv, heads, gqa, scale, softcap, q, k, v, mask)
        }
    }

    /// `ROCKET_FA_CHAIN=0`: one submit + fence wait per head matmul.
    #[allow(clippy::too_many_arguments)]
    fn run_per_head(
        &mut self,
        t: usize,
        kv: usize,
        heads: &[usize],
        gqa: usize,
        scale: f32,
        softcap: f32,
        q: &[f16],
        k: &[f16],
        v: &[f16],
        mask: Option<&[f16]>,
    ) -> Result<()> {
        let dk = self.head_dim;
        let dv = self.dv;
        let tp = self.tp;
        let kn = self.kn;
        for (oi, &h) in heads.iter().enumerate() {
            let hk = h / gqa;
            gather_q(&mut self.qh, q, h, t, dk);
            gather_k(&mut self.kh, k, hk, kv, dk);
            gather_v(&mut self.vh, v, hk, kv, dv, kn);

            self.qk
                .current_mut()
                .matmul(&self.qh, &self.kh, &mut self.scores)?;
            self.sm.run(
                &self.scores,
                &mut self.p,
                mask,
                t,
                kv,
                tp,
                kn,
                scale,
                softcap,
            );
            self.av
                .current_mut()
                .matmul(&self.p, &self.vh, &mut self.ctx)?;

            scatter_ctx(&mut self.out, &self.ctx, oi, t, dv);
        }
        Ok(())
    }

    /// `ROCKET_FA_CHAIN=1`: the worker's head range in groups of `gmax`, one batched QK
    /// job and one batched AV job per group (C `fa_heads_range_batched`).
    ///
    /// A shape that cannot batch (see [`Worker::resolve_batching`]) keeps its per-head
    /// `Scratch` submits inside the same group loop, reading and writing the group's
    /// buffers, so the two halves stay independent and the numerics stay bit-identical.
    #[allow(clippy::too_many_arguments)]
    fn run_grouped(
        &mut self,
        t: usize,
        kv: usize,
        heads: &[usize],
        gqa: usize,
        scale: f32,
        softcap: f32,
        q: &[f16],
        k: &[f16],
        v: &[f16],
        mask: Option<&[f16]>,
        gmax: usize,
    ) -> Result<()> {
        let dk = self.head_dim;
        let dv = self.dv;
        let tp = self.tp;
        let kn = self.kn;
        let (qd, kd, tk, cd) = (tp * dk, kn * dk, tp * kn, tp * dv);
        self.grp.ensure(gmax, tp, kn, dk, dv);
        // Split the fields the group loop drives: the batch contexts, the group scratch,
        // the per-head fallback scratch and the output are disjoint.
        let Worker {
            dev,
            qk,
            av,
            bqk,
            bav,
            grp,
            sm,
            out,
            qk_batch,
            av_batch,
            ..
        } = self;
        for (gi, range) in heads.chunks(gmax).enumerate() {
            let g = range.len();
            let base = gi * gmax;

            // The group's QK operands: every head's Q/K tile, exactly as the per-head
            // path gathers them.
            for (i, &h) in range.iter().enumerate() {
                gather_q(&mut grp.qh[i * qd..][..qd], q, h, t, dk);
                gather_k(&mut grp.kh[i * kd..][..kd], k, h / gqa, kv, dk);
            }
            if *qk_batch {
                let a: Vec<&[f16]> = grp.qh.chunks_exact(qd).take(g).collect();
                let b: Vec<&[f16]> = grp.kh.chunks_exact(kd).take(g).collect();
                let mut c: Vec<&mut [f16]> = grp.sc.chunks_exact_mut(tk).take(g).collect();
                batch_context(bqk, dev)?.run(tp, dk, kn, &a, &b, &mut c)?;
            } else {
                for i in 0..g {
                    qk.current_mut().matmul(
                        &grp.qh[i * qd..][..qd],
                        &grp.kh[i * kd..][..kd],
                        &mut grp.sc[i * tk..][..tk],
                    )?;
                }
            }

            // The mask + softmax sit between the two batched jobs, per head.
            for i in 0..g {
                sm.run(
                    &grp.sc[i * tk..][..tk],
                    &mut grp.p[i * tk..][..tk],
                    mask,
                    t,
                    kv,
                    tp,
                    kn,
                    scale,
                    softcap,
                );
            }

            for (i, &h) in range.iter().enumerate() {
                gather_v(&mut grp.vh[i * kd..][..kd], v, h / gqa, kv, dv, kn);
            }
            if *av_batch {
                let a: Vec<&[f16]> = grp.p.chunks_exact(tk).take(g).collect();
                let b: Vec<&[f16]> = grp.vh.chunks_exact(kd).take(g).collect();
                let mut c: Vec<&mut [f16]> = grp.ctx.chunks_exact_mut(cd).take(g).collect();
                batch_context(bav, dev)?.run(tp, kn, dv, &a, &b, &mut c)?;
            } else {
                for i in 0..g {
                    av.current_mut().matmul(
                        &grp.p[i * tk..][..tk],
                        &grp.vh[i * kd..][..kd],
                        &mut grp.ctx[i * cd..][..cd],
                    )?;
                }
            }

            for i in 0..g {
                let src = &grp.ctx[i * cd..][..cd];
                scatter_ctx(out, src, base + i, t, dv);
            }
        }
        Ok(())
    }
}

/// Whether a shape's batched matmul is bit-identical to its per-head one.
///
/// `kacc` says the per-head scratch routes to the on-NPU fp16 K-accumulation. A single
/// K-tile is one fp32 tile narrowed once on either path; more than one is not (KACC
/// rounds each tile's fp16 partial into the running fp16 sum, [`MmBatch`] keeps fp32 on
/// the CPU). A shape the planner rejects keeps the per-head path, which owns its error.
fn batch_matches_per_head(kacc: bool, plan: Result<Plan>) -> bool {
    match plan {
        Ok(p) => !kacc || p.nkt == 1,
        Err(_) => false,
    }
}

/// The worker's persistent batch context for one shape, created on first use.
fn batch_context<'a>(slot: &'a mut Option<MmBatch>, dev: &Device) -> Result<&'a mut MmBatch> {
    if slot.is_none() {
        *slot = Some(MmBatch::new(dev)?);
    }
    Ok(slot.as_mut().expect("created above"))
}

/// A persistent attention context: one device fd per worker, per-shape scratch.
pub struct Attention {
    workers: Vec<Worker>,
}

impl Attention {
    pub fn new(threads: usize) -> Result<Attention> {
        let big = affinity::num_big_cores();
        let cap = if big > 0 { big } else { MAX_WORKERS };
        let n = threads.clamp(1, MAX_WORKERS).min(cap);
        let mut workers = Vec::with_capacity(n);
        for _ in 0..n {
            workers.push(Worker::new(Device::open()?)?);
        }
        Ok(Attention { workers })
    }

    /// Live scratch held by the worker contexts.
    pub fn mem_stats(&self) -> MemStats {
        let mut stats = MemStats::default();
        for w in &self.workers {
            let m = w.mem();
            stats.bo_bytes += m.bo_bytes;
            stats.host_bytes += m.host_bytes;
            stats.workers.push(m);
        }
        stats
    }

    /// Drop shape classes from an earlier, much larger request and release the
    /// host matrices they held. The just-used class stays resident.
    pub fn trim_scratch(&mut self) {
        for w in &mut self.workers {
            w.qk.trim();
            w.av.trim();
            w.shrink_host();
            // The chained path's request-scoped scratch: sized by the same shapes, so
            // it follows the same 4x rule as the per-head host matrices and keeps the
            // just-used group (a steady request stream never reallocates).
            if w.grp.oversized(w.tp, w.kn) {
                w.grp.release();
                if let Some(b) = &mut w.bqk {
                    b.trim();
                }
                if let Some(b) = &mut w.bav {
                    b.trim();
                }
            }
        }
    }

    /// `out = softmax(scale * Q K^T + mask) V`, per head, GQA-aware.
    #[allow(clippy::too_many_arguments)]
    pub fn flash(
        &mut self,
        n_tokens: usize,
        n_kv: usize,
        head_dim: usize,
        dv: usize,
        n_head: usize,
        n_kv_heads: usize,
        scale: f32,
        softcap: f32,
        q: &[f16],
        k: &[f16],
        v: &[f16],
        mask: Option<&[f16]>,
        out: &mut [f16],
    ) -> Result<()> {
        if n_head == 0 || n_kv_heads == 0 || !n_head.is_multiple_of(n_kv_heads) {
            return Err(Error::shape("flash_attn/heads"));
        }
        if !head_dim.is_multiple_of(32) || !dv.is_multiple_of(16) {
            return Err(Error::shape("flash_attn/dims"));
        }
        let gqa = n_head / n_kv_heads;
        if q.len() < n_head * n_tokens * head_dim
            || k.len() < n_kv_heads * n_kv * head_dim
            || v.len() < n_kv_heads * dv * n_kv
            || out.len() < n_head * n_tokens * dv
        {
            return Err(Error::shape("flash_attn/buffers"));
        }
        if let Some(m) = mask
            && m.len() < n_tokens * n_kv
        {
            return Err(Error::shape("flash_attn/mask"));
        }

        let threads = self.workers.len().min(n_head);
        // Balanced contiguous ranges: 16 heads over 5 workers is 3/3/3/3/4, not
        // 4/4/4/4 with the fifth idle.
        let range = |i: usize| (i * n_head / threads, (i + 1) * n_head / threads);
        let base = affinity::base();
        let results: Vec<Result<()>> = thread::scope(|scope| {
            let mut handles = Vec::with_capacity(threads);
            for (i, w) in self.workers.iter_mut().take(threads).enumerate() {
                let (h0, h1) = range(i);
                if h0 >= h1 {
                    continue;
                }
                handles.push(scope.spawn(move || -> Result<()> {
                    affinity::pin_current(i, base);
                    w.prepare(n_tokens, n_kv, head_dim, dv)?;
                    let heads: Vec<usize> = (h0..h1).collect();
                    w.run(n_tokens, n_kv, &heads, gqa, scale, softcap, q, k, v, mask)
                }));
            }
            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .unwrap_or_else(|_| Err(Error::device("attn/worker", 0)))
                })
                .collect()
        });
        if let Some(e) = results.into_iter().find_map(|r| r.err()) {
            return Err(e);
        }

        // Copy each worker's contiguous head range into the caller's output.
        for (i, w) in self.workers.iter().take(threads).enumerate() {
            let (h0, h1) = range(i);
            if h0 >= h1 {
                continue;
            }
            let n = (h1 - h0) * n_tokens * dv;
            out[h0 * n_tokens * dv..h0 * n_tokens * dv + n].copy_from_slice(&w.out[..n]);
        }
        Ok(())
    }
}

/// A held-open attention context, matching the C `rocket_fa_ctx` lifetime.
pub type FaContext = Attention;

#[cfg(test)]
mod tests {
    use super::*;

    // Everything below is device-free: the planner, the group rule and the host
    // gather/scatter mapping are pure slice work.

    const M: u64 = 1 << 20;

    #[test]
    fn group_size_is_the_budget_over_one_score_matrix() {
        // The budget divides by one head's padded [Tp,Kn] score matrix.
        assert_eq!(group_size(32 * M, 1024, 1024, 8), 8, "32 heads fit in 32M");
        assert_eq!(group_size(32 * M, 2048, 2048, 8), 8, "8 heads fit in 32M");
        assert_eq!(group_size(32 * M, 4096, 4096, 8), 2, "2 heads fit at 4k");
        assert_eq!(group_size(32 * M, 8192, 8192, 8), 1, "1 head at 8k");
        assert_eq!(group_size(32 * M, 16384, 16384, 8), 1, "clamped up to 1");
    }

    #[test]
    fn group_size_clamps_to_the_head_range_and_a_zero_budget() {
        // Never more heads than the worker owns.
        assert_eq!(group_size(u64::MAX, 4, 32, 3), 3);
        // A budget below one score matrix (and the C's v<1 clamp) is one head.
        assert_eq!(group_size(1, 4, 32, 3), 1);
        assert_eq!(group_size(0, 4, 32, 3), 1);
        // No heads (a worker with an empty range never gets here, but be total).
        assert_eq!(group_size(32 * M, 4, 32, 0), 0);
        // A degenerate shape cannot divide by zero.
        assert_eq!(group_size(32 * M, 0, 0, 3), 3);
    }

    #[test]
    fn a_shape_may_batch_only_while_it_keeps_one_k_tile_or_drops_kacc() {
        // 4x64x16: one K-tile, so KACC and the CPU accumulation narrow the same tile.
        let one = Plan::new(4, 64, 16).unwrap();
        assert_eq!(one.nkt, 1);
        assert!(batch_matches_per_head(true, Ok(one)));
        assert!(batch_matches_per_head(false, Ok(one)));
        // A split K axis: KACC rounds per-tile, so only the no-KACC path may batch.
        let many = Plan::new(264, 1024, 512).unwrap();
        assert!(many.nkt > 1);
        assert!(!batch_matches_per_head(true, Ok(many)));
        assert!(batch_matches_per_head(false, Ok(many)));
        // An unplannable shape keeps the per-head path (which owns the error).
        assert!(!batch_matches_per_head(false, Plan::new(6, 64, 16)));
    }

    #[test]
    fn gathers_pad_every_tile_and_keep_the_live_rows() {
        // dk = 32 so Q/K tiles are whole rows; t = 3 pads Tp from 3 to 4.
        let dk = 32;
        let (t, tp) = (3usize, 4usize);
        let q: Vec<f16> = (0..2 * t * dk).map(|i| f16::from_f32(i as f32)).collect();
        let mut tile = vec![f16::ONE; tp * dk];
        gather_q(&mut tile, &q, 1, t, dk);
        assert_eq!(&tile[..t * dk], &q[t * dk..], "head 1's live rows");
        assert!(
            tile[t * dk..].iter().all(|v| *v == f16::ZERO),
            "the padding row is zeroed, not left over"
        );

        let (kv, kn) = (3usize, 32usize);
        let k: Vec<f16> = (0..2 * kv * dk).map(|i| f16::from_f32(i as f32)).collect();
        let mut ktile = vec![f16::ONE; kn * dk];
        gather_k(&mut ktile, &k, 0, kv, dk);
        assert_eq!(&ktile[..kv * dk], &k[..kv * dk]);
        assert!(ktile[kv * dk..].iter().all(|v| *v == f16::ZERO));

        // V is [dv, kv] and gathers into a [dv, kn] tile: rows keep their stride.
        let (dv, n_kv) = (2usize, 3usize);
        let v: Vec<f16> = (0..2 * dv * n_kv)
            .map(|i| f16::from_f32(i as f32))
            .collect();
        let mut vtile = vec![f16::ONE; dv * kn];
        gather_v(&mut vtile, &v, 1, n_kv, dv, kn);
        for c in 0..dv {
            let src = (dv + c) * n_kv;
            assert_eq!(&vtile[c * kn..c * kn + n_kv], &v[src..src + n_kv]);
            assert!(
                vtile[c * kn + n_kv..(c + 1) * kn]
                    .iter()
                    .all(|v| *v == f16::ZERO)
            );
        }
    }

    #[test]
    fn scatter_writes_head_oi_into_its_own_span() {
        // Two heads of t=2 rows over dv=4, from padded [tp=4, dv] tiles.
        let (t, tp, dv) = (2usize, 4usize, 4usize);
        let mut out = vec![f16::ZERO; 2 * t * dv];
        for head in 0..2 {
            let base = (head * 100) as f32;
            let ctx: Vec<f16> = (0..tp * dv)
                .map(|i| f16::from_f32(base + i as f32))
                .collect();
            scatter_ctx(&mut out, &ctx, head, t, dv);
        }
        for head in 0..2 {
            let base = (head * 100) as f32;
            for row in 0..t {
                for c in 0..dv {
                    let want = f16::from_f32(base + (row * dv + c) as f32);
                    assert_eq!(
                        out[(head * t + row) * dv + c],
                        want,
                        "head {head} row {row}"
                    );
                }
            }
        }
    }
}
