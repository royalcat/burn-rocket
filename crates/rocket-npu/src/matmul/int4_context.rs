//! Resident int4 matmul (W4A4), the int4 sibling of [`super::int8::I8Context`].
//!
//! Port of `rocket_prepacked_int4.c`: N is split across worker fds in 64-column
//! slices, a scratch is cached per `(M,K,N,group)`, and a resident weight holds the
//! nibble-packed codes only (the scales stay caller-side). The NPU writes int16
//! partials; this layer gathers each one into the host accumulator — int64 for the
//! dense/per-channel mode, f32 with the per-group scales for the group-wise one. There
//! is no NPU-side integer K-accumulation for int4 (the C leaves the DPU-EW path
//! unimplemented).
//!
//! Two modes share one compute core, selected by the weight's group
//! (`I4Context::pack_weight` / `_pack_gw`):
//!
//! - `group == 0` — **per-channel**: raw int32 `C[M,N]`, host int64 K-accumulation of
//!   the int16 partials (exact up to the hardware's per-tile int16 saturation).
//! - `group > 0` — **group-wise**: `Kt` is forced to `group`, so every K-tile stays
//!   inside one quant group and its int16 partial takes that group's
//!   `a_scale[m,g]*b_scale[n,g]` on readback, accumulated in f32. The in-model W4A4
//!   path.
//!
//! Three things differ from the int8 engine next door:
//!
//! - The operands are **nibble-packed** (two int4 codes per byte, one code per `i8` in
//!   `[-8,7]`), so a slot's byte length is its nibble count halved.
//! - The output cube is **int16** (`out_precision = 1`, C2=8), not int32 — hence the
//!   int16 de-tile and the per-tile saturation the C bounds with `kt_cap` / `group`.
//! - Contiguous self-chaining stays **off**, as the C leaves it: the int8 matmul chains
//!   bit-exactly (`tests/int8_chain_probe`), but this path writes int16 through its own
//!   output stage and no int4 chain has been scored against a CPU model. Gapped tasks
//!   still batch into one ioctl, which is the part that matters.

use std::collections::{HashMap, VecDeque};
use std::thread;

use crate::Device;
use crate::device::uapi::DrmRocketTask;
use crate::device::{Bo, Handle};
use crate::error::{Error, Result};
use crate::hw;
use crate::regcmd::desc::{BATCH, CBUF_BANK_SIZE, RC_STRIDE};
use crate::regcmd::fp16::MatmulParams;
use crate::regcmd::int4::emit;

use super::bos::i16_words_ref;
use super::int4::{check_group, int4_plan};
use super::layout;
use super::plan::{INT4, Plan, plan_geom};
use super::run::{wait_ns, write_regcmd};

const MAX_WORKERS: usize = 8;

struct Worker {
    dev: Device,
    n0: usize,
    nsub: usize,
    plan: Plan,
    _guard: Bo,
    regcmd: Bo,
    in_all: Bo,
    out_all: Bo,
    facc: Vec<f32>,
    iacc: Vec<i64>,
    in_zeroed: bool,
}

/// A weight resident in per-worker NPU buffers (int4 codes only, no scales).
///
/// `group == 0` is the per-channel (dense) layout, `group > 0` the group-wise one. The
/// codes are scattered `(N/64, K/32, 64, 32)` nibbles per K-tile either way; only the
/// tile boundaries differ (group-wise pins `Kt` to the group), so a weight's bytes are
/// the int4 codes and nothing else.
pub struct I4Weight {
    m: usize,
    k: usize,
    n: usize,
    group: usize,
    bos: Vec<Bo>,
    sig: Vec<Sig>,
}

impl I4Weight {
    pub fn shape(&self) -> (usize, usize, usize, usize) {
        (self.m, self.k, self.n, self.group)
    }

    /// Live bytes of the per-worker resident weight BOs (the C
    /// `rocket_i4_weights_bytes`, including each worker's CBUF-bank padding).
    pub fn bytes(&self) -> usize {
        self.bos.iter().map(Bo::len).sum()
    }
}

/// The per-worker tile layout a resident weight was scattered for.
///
/// A signature rather than a pointer, exactly as in the C: the scratch cache recycles
/// its least recently used shape, so a held pointer would be one unrelated shape away
/// from freed memory.
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

/// A persistent int4 context (per-channel and group-wise resident weights).
pub struct I4Context {
    devs: Vec<Device>,
    threads: usize,
    scratches: HashMap<(usize, usize, usize, usize), Vec<Worker>>,
    order: VecDeque<(usize, usize, usize, usize)>,
}

impl I4Context {
    pub fn new(nthreads: usize) -> Result<I4Context> {
        let threads = nthreads.clamp(1, MAX_WORKERS);
        let mut devs = Vec::with_capacity(threads);
        for _ in 0..threads {
            devs.push(Device::open()?);
        }
        Ok(I4Context {
            devs,
            threads,
            scratches: HashMap::new(),
            order: VecDeque::new(),
        })
    }

    fn build(&self, m: usize, k: usize, n: usize, group: usize) -> Result<Vec<Worker>> {
        if m == 0 || !m.is_multiple_of(4) {
            return Err(Error::shape("int4/m"));
        }
        if !k.is_multiple_of(32) || !n.is_multiple_of(64) {
            return Err(Error::shape("int4/k-n"));
        }
        if group != 0 {
            check_group(group, k)?;
        }
        let nstep = fanout_step(n, self.threads, 64);
        let mut workers = Vec::with_capacity(self.threads);
        for t in 0..self.threads {
            let n0 = t * nstep;
            if n0 >= n {
                break;
            }
            let nsub = (n - n0).min(nstep);
            let plan = plan_for(m, k, nsub, group)?;
            let tiles = plan.nmt * plan.nnt * plan.nkt;
            let slots = tiles.clamp(1, BATCH);
            let guard = self.devs[t].alloc_bo(4096)?;
            let regcmd = self.devs[t].alloc_bo(slots * RC_STRIDE * 8)?;
            let in_all =
                self.devs[t].alloc_bo(plan.nmt * plan.nkt * (plan.in_slot / 2) + CBUF_BANK_SIZE)?;
            // The C leaves the output BO at I4_BATCH unconditionally: a right-sized one
            // is bit-exact but buys nothing measurable here, because this path syncs its
            // output once per job instead of once per K-tile (the sync-bound paths, like
            // fp16 KACC, are the ones that gain). Right-sizing is still what this port
            // does, matching the int8 engine's scratch policy.
            let out_all = self.devs[t].alloc_bo(slots * plan.out_slot * 2 + CBUF_BANK_SIZE)?;
            workers.push(Worker {
                dev: self.devs[t].clone(),
                n0,
                nsub,
                plan,
                _guard: guard,
                regcmd,
                in_all,
                out_all,
                facc: Vec::new(),
                iacc: Vec::new(),
                in_zeroed: false,
            });
        }
        if workers.is_empty() {
            return Err(Error::shape("int4/split"));
        }
        Ok(workers)
    }

    fn scratch(&mut self, m: usize, k: usize, n: usize, group: usize) -> Result<&mut Vec<Worker>> {
        let key = (m, k, n, group);
        if !self.scratches.contains_key(&key) {
            let workers = self.build(m, k, n, group)?;
            if self.scratches.len() >= 32
                && let Some(old) = self.order.pop_front()
            {
                self.scratches.remove(&old);
            }
            self.scratches.insert(key, workers);
            self.order.push_back(key);
        }
        Ok(self.scratches.get_mut(&key).unwrap())
    }

    /// Pack a pre-quantized `[N,K]` int4 weight into resident per-worker BOs.
    ///
    /// Shared by both modes: the codes are scattered `(N/64, K/32, 64, 32)` nibbles per
    /// K-tile, and only the tile boundaries (fixed by the mode's tiling) move —
    /// group-wise is a re-tiling, not a requantization. No scales are stored: the output
    /// is raw int32 or caller-scaled f32, so the scales stay caller-side (the C
    /// `rk4_scatter_weights`).
    ///
    /// One scatter per tile, not the C's `Ntile` repeats: `rk4_scatter_weights` calls
    /// `i4_wt_scatter` inside a `for (kk = 1; kk <= Ntile; kk++)` loop that writes the
    /// same nibbles to the same slot every time, so the packed bytes are identical.
    fn pack_codes(
        &mut self,
        m: usize,
        k: usize,
        n: usize,
        b: &[i8],
        group: usize,
    ) -> Result<I4Weight> {
        if b.len() != n * k {
            return Err(Error::shape("int4/pack"));
        }
        let workers = self.scratch(m, k, n, group)?;
        let mut bos = Vec::with_capacity(workers.len());
        let mut sig = Vec::with_capacity(workers.len());
        for w in workers.iter_mut() {
            let wt_sz = expected_wt_bytes(&w.plan);
            let mut bo = w.dev.alloc_bo(wt_sz)?;
            super::bos::check_iova(&[&bo])?;
            bo.prep(0)?;
            {
                let bytes = bo.as_mut_slice();
                bytes.fill(0);
                for ni in 0..w.plan.nnt {
                    let n0 = ni * w.plan.nt;
                    let ntile = w.plan.n_tile(ni);
                    for ki in 0..w.plan.nkt {
                        let k0 = ki * w.plan.kt;
                        let ktile = w.plan.k_tile(ki);
                        let off = (ni * w.plan.nkt + ki) * (w.plan.wt_slot / 2);
                        layout::wt_scatter_tile_i4(
                            &mut bytes[off..off + w.plan.wt_slot / 2],
                            &b[w.n0 * k..],
                            k,
                            n0,
                            k0,
                            ntile,
                            ktile,
                        );
                    }
                }
            }
            bo.fini()?;
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
            bos.push(bo);
        }
        Ok(I4Weight {
            m,
            k,
            n,
            group,
            bos,
            sig,
        })
    }

    /// Pack a pre-quantized `[N,K]` int4 weight for the **per-channel** (dense) path
    /// (the C `rocket_i4_weights_pack`). The returned weight's group is 0.
    pub fn pack_weight(&mut self, m: usize, k: usize, n: usize, b: &[i8]) -> Result<I4Weight> {
        self.pack_codes(m, k, n, b, 0)
    }

    /// Pack a pre-quantized `[N,K]` int4 weight for the **group-wise** path (the C
    /// `rocket_i4_weights_pack_gw`). `group >= 32`, `group % 32 == 0`, `group | K`,
    /// `49*group < 32767`.
    pub fn pack_weight_gw(
        &mut self,
        m: usize,
        k: usize,
        n: usize,
        b: &[i8],
        group: usize,
    ) -> Result<I4Weight> {
        if group == 0 {
            return Err(Error::shape("int4/pack/group"));
        }
        self.pack_codes(m, k, n, b, group)
    }

    /// Run every worker over its N-slice of `a`, accumulating into the worker's `iacc`
    /// (dense) or `facc` (group-wise, `ng = K/group`).
    #[allow(clippy::too_many_arguments)]
    fn run_workers(
        &mut self,
        m: usize,
        k: usize,
        n: usize,
        group: usize,
        a: &[i8],
        ng: usize,
        a_scale: &[f32],
        b_scale: &[f32],
        w: &I4Weight,
    ) -> Result<()> {
        let workers = self.scratch(m, k, n, group)?;
        if workers.len() != w.bos.len() {
            return Err(Error::tiling("int4/weight-workers"));
        }
        for (i, worker) in workers.iter().enumerate() {
            let now = Sig {
                n0: worker.n0,
                nsub: worker.nsub,
                kt: worker.plan.kt,
                nt: worker.plan.nt,
                nkt: worker.plan.nkt,
                nnt: worker.plan.nnt,
                wt_slot: worker.plan.wt_slot,
                wt_bytes: expected_wt_bytes(&worker.plan),
            };
            // C returns -2 here (a genuine tiling mismatch, e.g. a ROCKET_MM_* override
            // between pack and compute) so the caller re-packs instead of miscomputing.
            if w.sig[i] != now {
                return Err(Error::tiling("int4/weight-layout"));
            }
        }

        let base = crate::device::affinity::base();
        let results: Vec<Result<()>> = thread::scope(|scope| {
            let mut handles = Vec::with_capacity(workers.len());
            for (i, worker) in workers.iter_mut().enumerate() {
                let wt = &w.bos[i];
                handles.push(scope.spawn(move || -> Result<()> {
                    crate::device::affinity::pin_current(i, base);
                    worker.pack_input(k, a)?;
                    worker.compute(m, k, n, ng, a_scale, b_scale, wt)
                }));
            }
            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .unwrap_or_else(|_| Err(Error::device("int4/worker", 0)))
                })
                .collect()
        });
        if let Some(e) = results.into_iter().find_map(|r| r.err()) {
            return Err(e);
        }
        Ok(())
    }

    /// `C[M,N] = A[M,K] * B[N,K]^T`, raw int32 (the C `rocket_matmul_int4_prepacked`).
    ///
    /// `w` must be a per-channel weight ([`I4Context::pack_weight`]); the group-wise
    /// form is [`I4Context::matmul_gw`]. `M % 4` is required — an unaligned `M`
    /// miscomputes on HW and the resident path cannot pad rows (C `rk4_call_m_ok`), so
    /// ragged row counts are padded caller-side. The pack's `M` is informational:
    /// canonical tiling makes the weight valid for every `M`.
    pub fn matmul(
        &mut self,
        m: usize,
        k: usize,
        n: usize,
        a: &[i8],
        c: &mut [i32],
        w: &I4Weight,
    ) -> Result<()> {
        if k != w.k || n != w.n || w.group != 0 {
            return Err(Error::shape("int4/matmul/shape"));
        }
        if a.len() != m * k || c.len() != m * n {
            return Err(Error::shape("int4/matmul/buffers"));
        }
        self.run_workers(m, k, n, 0, a, 0, &[], &[], w)?;
        for worker in self.scratch(m, k, n, 0)?.iter() {
            for row in 0..m {
                let dst = &mut c[row * n + worker.n0..][..worker.nsub];
                let src = &worker.iacc[row * worker.nsub..][..worker.nsub];
                for (d, s) in dst.iter_mut().zip(src) {
                    *d = *s as i32;
                }
            }
        }
        Ok(())
    }

    /// `C[M,N] = sum_g a_scale[m,g] * b_scale[n,g] * (A_g * B_g^T)`, f32 output (the C
    /// `rocket_matmul_int4_prepacked_gw`).
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_gw(
        &mut self,
        m: usize,
        k: usize,
        n: usize,
        a: &[i8],
        a_scale: &[f32],
        b_scale: &[f32],
        cf: &mut [f32],
        w: &I4Weight,
    ) -> Result<()> {
        if k != w.k || n != w.n || w.group == 0 {
            return Err(Error::shape("int4/matmul/shape"));
        }
        if a.len() != m * k || cf.len() != m * n {
            return Err(Error::shape("int4/matmul/buffers"));
        }
        let ng = k / w.group;
        if a_scale.len() != m * ng || b_scale.len() != n * ng {
            return Err(Error::shape("int4/matmul/scales"));
        }
        self.run_workers(m, k, n, w.group, a, ng, a_scale, b_scale, w)?;
        for worker in self.scratch(m, k, n, w.group)?.iter() {
            for row in 0..m {
                cf[row * n + worker.n0..][..worker.nsub]
                    .copy_from_slice(&worker.facc[row * worker.nsub..][..worker.nsub]);
            }
        }
        Ok(())
    }
}

impl Worker {
    fn pack_input(&mut self, k: usize, a: &[i8]) -> Result<()> {
        self.in_all.prep(0)?;
        {
            let bytes = self.in_all.as_mut_slice();
            if !self.in_zeroed {
                // Only live tile lanes are written; the padding past a ragged tile must
                // already be zero (as in the one-shot's `Shots`).
                bytes.fill(0);
                self.in_zeroed = true;
            }
            for mi in 0..self.plan.nmt {
                let m0 = mi * self.plan.mt;
                let mtile = self.plan.m_tile(mi);
                for ki in 0..self.plan.nkt {
                    let k0 = ki * self.plan.kt;
                    let ktile = self.plan.k_tile(ki);
                    let off = (mi * self.plan.nkt + ki) * (self.plan.in_slot / 2);
                    layout::feat_scatter_i4(
                        &mut bytes[off..off + self.plan.in_slot / 2],
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

    #[allow(clippy::too_many_arguments)]
    fn compute(
        &mut self,
        m: usize,
        k: usize,
        _n: usize,
        ng: usize,
        a_scale: &[f32],
        b_scale: &[f32],
        wt: &Bo,
    ) -> Result<()> {
        let plan = &self.plan;
        let dense = ng == 0;
        if dense {
            self.iacc.clear();
            self.iacc.resize(m * self.nsub, 0);
        } else {
            self.facc.clear();
            self.facc.resize(m * self.nsub, 0.0);
        }
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
                        + ((mi * plan.nkt + ki) * (plan.in_slot / 2)) as u64;
                    let weights =
                        wt.dma_address() + ((ni * plan.nkt + ki) * (plan.wt_slot / 2)) as u64;
                    let output = self.out_all.dma_address() + (out_off * 2) as u64;
                    let mut p = MatmulParams::new(
                        mtile as u32,
                        ktile as u32,
                        ntile as u32,
                        input as u32,
                        weights as u32,
                        output as u32,
                    );
                    // Not read by the int4 generator (it hardcodes the int16 writer and
                    // the plain-conv path), set for symmetry with the int8 engine.
                    p.fp32_to_fp16 = false;
                    ops.clear();
                    let count = emit(&p, &mut ops)?;
                    let regcmd = write_regcmd(&mut self.regcmd, nb, &ops)?;
                    tasks.push(DrmRocketTask {
                        regcmd,
                        regcmd_count: count as u32,
                    });
                    // Group-wise pins Kt = group, so one tile IS one quant group.
                    meta.push((m0, n0, mtile, ntile, ki));
                    nb += 1;
                    done += 1;

                    if nb == BATCH || done == total {
                        self.regcmd.fini()?;
                        self.out_all.prep(0)?;
                        self.out_all.fini()?;
                        let in_h: [Handle; 3] =
                            [self.in_all.handle(), wt.handle(), self.regcmd.handle()];
                        let out_h = [self.out_all.handle()];
                        self.dev.submit(&tasks, &in_h, &out_h, 0)?;
                        self.out_all.prep(wait_ns())?;
                        {
                            let words = i16_words_ref(&self.out_all);
                            let n0_base = self.n0;
                            for (j, &(m0, n0, mtile, ntile, g)) in meta.iter().enumerate() {
                                let slot = &words[j * plan.out_slot..(j + 1) * plan.out_slot];
                                if dense {
                                    layout::detile_accum_i16(
                                        &mut self.iacc,
                                        self.nsub,
                                        slot,
                                        m0,
                                        n0,
                                        mtile,
                                        ntile,
                                    );
                                } else {
                                    let facc = &mut self.facc;
                                    layout::detile_accum_i16_scaled(
                                        facc,
                                        self.nsub,
                                        slot,
                                        m0,
                                        n0,
                                        mtile,
                                        ntile,
                                        |row, col| {
                                            a_scale[row * ng + g]
                                                * b_scale[(n0_base + col) * ng + g]
                                        },
                                    );
                                }
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
        let _ = k;
        Ok(())
    }
}

/// The per-worker plan for a shape. The tiling is resolved at the **canonical** tile
/// height (`max_tile`), not the call's `M`, so a resident weight's scatter layout is
/// M-independent and one pack serves every `M` — the C `rk4_worker_alloc`. Only the
/// live `Mt` and `nMt` follow the call.
///
/// Group-wise pins `Kt` to the quant group after planning (one K-tile per group). The
/// group bound keeps that within the CBUF: at most 640 nibbles per row of an operand
/// tile, so the pair needs at most 3 + 3 of the 12 banks.
fn plan_for(m: usize, k: usize, n: usize, group: usize) -> Result<Plan> {
    let max_tile = hw::current().max_tile as usize;
    let (cmt, kt, nt) = plan_geom(&INT4, max_tile.max(4), k, n, 0)?;
    let kt = if group == 0 { kt } else { group };
    Ok(int4_plan(m, k, n, cmt.min(m), kt, nt))
}

/// Bytes of one worker's resident weight BO for `plan` (nibbles -> bytes, plus the
/// CBUF-bank pad the C allocates).
fn expected_wt_bytes(plan: &Plan) -> usize {
    plan.nnt * plan.nkt * (plan.wt_slot / 2) + CBUF_BANK_SIZE
}

/// N split across `threads` workers, each slice rounded up to the int4 N granularity
/// (the C `rocket_fanout_nstep`).
fn fanout_step(n: usize, threads: usize, align: usize) -> usize {
    let threads = threads.max(1);
    let align = align.max(1);
    n.div_ceil(threads).div_ceil(align) * align
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The weight scatter layout depends only on the N-split and on `Kt`/`Nt`, never on
    /// the call's `M` — that is what lets one pack serve every `M` (the C comment in
    /// `rk4_worker_alloc`).
    #[test]
    fn the_canonical_layout_is_m_independent() {
        for (k, n, group) in [
            (1024usize, 512usize, 0usize),
            (1024, 512, 32),
            (2560, 384, 128),
            (320, 320, 64),
        ] {
            let big = plan_for(256, k, n, group).unwrap();
            let small = plan_for(4, k, n, group).unwrap();
            assert_eq!(big.mt, 256.min(hw::current().max_tile as usize));
            assert_eq!(small.mt, 4, "only the live height follows the call");
            assert_eq!(
                (big.kt, big.nt, big.nkt, big.nnt, big.wt_slot),
                (small.kt, small.nt, small.nkt, small.nnt, small.wt_slot),
                "K={k} N={n} group={group}"
            );
            assert_eq!(expected_wt_bytes(&big), expected_wt_bytes(&small));
            if group > 0 {
                assert_eq!(big.kt, group);
                assert_eq!(big.nkt, k / group);
            }
        }
    }

    #[test]
    fn the_nsplit_rounds_each_slice_to_the_int4_n_group() {
        // 5 workers over N=384: ceil(384/5) = 77 -> a 128-column step, so three workers
        // take 128 each and the remaining two have nothing to do.
        assert_eq!(fanout_step(384, 5, 64), 128);
        assert_eq!(fanout_step(512, 5, 64), 128);
        assert_eq!(fanout_step(64, 5, 64), 64);
        assert_eq!(fanout_step(512, 1, 64), 512);
        assert_eq!(fanout_step(192, 3, 64), 64);
    }

    #[test]
    fn the_shape_contract_matches_the_c() {
        // The validation the C does up front in `rocket_i4_weights_pack*`. Every error
        // path returns before the per-worker BO allocation, so an empty context is
        // enough: no device is opened and this runs on the host.
        let ctx = I4Context {
            devs: Vec::new(),
            threads: 1,
            scratches: HashMap::new(),
            order: VecDeque::new(),
        };
        assert!(matches!(
            ctx.build(6, 64, 64, 0),
            Err(e) if e.status() == crate::Status::Shape
        ));
        assert!(ctx.build(4, 48, 64, 0).is_err()); // K % 32
        assert!(ctx.build(4, 64, 32, 0).is_err()); // N % 64
        assert!(ctx.build(4, 64, 64, 16).is_err()); // group < 32
        assert!(ctx.build(4, 96, 64, 64).is_err()); // group does not divide K
        assert!(ctx.build(4, 1344, 64, 672).is_err()); // 49*672 >= 32767
        // The accepted shapes go on to allocate worker BOs, so check their plan.
        let p = plan_for(4, 1280, 64, 640).unwrap();
        assert_eq!(p.kt, 640);
        assert_eq!(p.nkt, 2);
    }
}
