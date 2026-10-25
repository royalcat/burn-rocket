//! Resident int8 matmul (W8A8), the path behind `--npu-int8`.
//!
//! Port of `rocket_prepacked_int8.c`: N is split across worker fds in 32-column
//! slices, a scratch is cached per `(M,K,N,group)`, and a resident weight holds the
//! int8 codes only. The NPU writes int32 partials; this layer applies the per-row /
//! per-channel per-group scales and accumulates in f32, exactly like the C path.
//! There is no NPU-side int8 K-accumulation.
//!
//! Two modes share one compute core, selected by the weight's group
//! (`rocket_i8_weights_pack` / `_pack_gw`):
//!
//! - `group == 0` — **per-channel**: raw int32 `C[M,N]`, host int64 K-accumulation
//!   (integer-exact). The W8A8 in-model path; the caller owns the dequant.
//! - `group > 0` — **group-wise**: every K-tile stays inside one quant group, so its
//!   int32 partial takes that group's `a_scale[m,g]*b_scale[n,g]` on readback and is
//!   accumulated in f32. The natively-quantized-weight path.

use std::collections::{HashMap, VecDeque};
use std::thread;

use crate::Device;
use crate::device::uapi::DrmRocketTask;
use crate::device::{Bo, Handle};
use crate::error::{Error, Result};
use crate::hw;
use crate::regcmd::desc::{BATCH, CBUF_BANK_SIZE, RC_STRIDE};
use crate::regcmd::fp16::MatmulParams;
use crate::regcmd::int8::emit;

use super::bos::i8_words;
use super::int8_dense::int8_plan;
use super::layout;
use super::plan::{INT8, INT8_GW, Plan, plan_geom};
use super::run::wait_ns;

const MAX_WORKERS: usize = 8;

struct Worker {
    dev: Device,
    n0: usize,
    nsub: usize,
    plan: Plan,
    kt_per_group: usize,
    _guard: Bo,
    regcmd: Bo,
    in_all: Bo,
    out_all: Bo,
    facc: Vec<f32>,
    iacc: Vec<i64>,
    in_zeroed: bool,
}

/// A weight resident in per-worker NPU buffers (int8 codes only, no scales).
///
/// `group == 0` is the per-channel (dense) layout, `group > 0` the group-wise one.
/// The codes are scattered `(N/32, K/32, 32, 32)` per K-tile either way; only the tile
/// boundaries differ, so a weight's bytes are the int8 codes and nothing else.
pub struct I8Weight {
    m: usize,
    k: usize,
    n: usize,
    group: usize,
    bos: Vec<Bo>,
    sig: Vec<Sig>,
}

impl I8Weight {
    pub fn shape(&self) -> (usize, usize, usize, usize) {
        (self.m, self.k, self.n, self.group)
    }

    /// Live bytes of the per-worker resident weight BOs.
    pub fn bytes(&self) -> usize {
        self.bos.iter().map(Bo::len).sum()
    }
}

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

/// A persistent int8 context (per-channel and group-wise resident weights).
pub struct I8Context {
    devs: Vec<Device>,
    threads: usize,
    scratches: HashMap<(usize, usize, usize, usize), Vec<Worker>>,
    order: VecDeque<(usize, usize, usize, usize)>,
}

impl I8Context {
    pub fn new(nthreads: usize) -> Result<I8Context> {
        let threads = nthreads.clamp(1, MAX_WORKERS);
        let mut devs = Vec::with_capacity(threads);
        for _ in 0..threads {
            devs.push(Device::open()?);
        }
        Ok(I8Context {
            devs,
            threads,
            scratches: HashMap::new(),
            order: VecDeque::new(),
        })
    }

    fn build(&self, m: usize, k: usize, n: usize, group: usize) -> Result<Vec<Worker>> {
        if m == 0 || !m.is_multiple_of(4) {
            return Err(Error::shape("int8/m"));
        }
        if !k.is_multiple_of(32) || !n.is_multiple_of(32) {
            return Err(Error::shape("int8/k-n"));
        }
        if group != 0 && (group < 32 || !group.is_multiple_of(32) || !k.is_multiple_of(group)) {
            return Err(Error::shape("int8/group"));
        }
        let nstep = fanout_step(n, self.threads, 32);
        let mut workers = Vec::with_capacity(self.threads);
        for t in 0..self.threads {
            let n0 = t * nstep;
            if n0 >= n {
                break;
            }
            let nsub = (n - n0).min(nstep);
            let plan = plan_for(m, k, nsub, group)?;
            let kt_per_group = if group == 0 { 0 } else { group / plan.kt };
            debug_assert!(group == 0 || kt_per_group * plan.kt == group);
            let tiles = plan.nmt * plan.nnt * plan.nkt;
            let slots = tiles.clamp(1, BATCH);
            let guard = self.devs[t].alloc_bo(4096)?;
            let regcmd = self.devs[t].alloc_bo(slots * RC_STRIDE * 8)?;
            let in_all =
                self.devs[t].alloc_bo(plan.nmt * plan.nkt * plan.in_slot + CBUF_BANK_SIZE)?;
            let out_all = self.devs[t].alloc_bo(slots * plan.out_slot * 4 + CBUF_BANK_SIZE)?;
            workers.push(Worker {
                dev: self.devs[t].clone(),
                n0,
                nsub,
                plan,
                kt_per_group,
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
            return Err(Error::shape("int8/split"));
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

    /// Pack a pre-quantized `[N,K]` int8 weight into resident per-worker BOs.
    ///
    /// Shared by both modes: the codes are scattered `(N/32, K/32, 32, 32)` per K-tile,
    /// and only the tile boundaries (fixed by the mode's tiling) move — group-wise is a
    /// re-tiling, not a requantization. No scales are stored: the output is raw int32 or
    /// caller-scaled f32, so the scales stay caller-side (the C
    /// `rki_scatter_weights`).
    fn pack_codes(
        &mut self,
        m: usize,
        k: usize,
        n: usize,
        b: &[i8],
        group: usize,
    ) -> Result<I8Weight> {
        if b.len() != n * k {
            return Err(Error::shape("int8/pack"));
        }
        let workers = self.scratch(m, k, n, group)?;
        let mut bos = Vec::with_capacity(workers.len());
        let mut sig = Vec::with_capacity(workers.len());
        for w in workers.iter_mut() {
            let wt_sz = w.plan.nnt * w.plan.nkt * w.plan.wt_slot + CBUF_BANK_SIZE;
            let mut bo = w.dev.alloc_bo(wt_sz)?;
            bo.prep(0)?;
            {
                let words = i8_words(&mut bo);
                words.fill(0);
                for ni in 0..w.plan.nnt {
                    let n0 = ni * w.plan.nt;
                    let ntile = w.plan.n_tile(ni);
                    for ki in 0..w.plan.nkt {
                        let k0 = ki * w.plan.kt;
                        let ktile = w.plan.k_tile(ki);
                        let off = (ni * w.plan.nkt + ki) * w.plan.wt_slot;
                        layout::wt_scatter_tile_i8(
                            &mut words[off..off + w.plan.wt_slot],
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
        Ok(I8Weight {
            m,
            k,
            n,
            group,
            bos,
            sig,
        })
    }

    /// Pack a pre-quantized `[N,K]` int8 weight for the **per-channel** (dense) path
    /// (the C `rocket_i8_weights_pack`). The returned weight's group is 0.
    pub fn pack_weight(&mut self, m: usize, k: usize, n: usize, b: &[i8]) -> Result<I8Weight> {
        self.pack_codes(m, k, n, b, 0)
    }

    /// Pack a pre-quantized `[N,K]` int8 weight for the **group-wise** path (the C
    /// `rocket_i8_weights_pack_gw`). `group >= 32`, `group % 32 == 0`, `group | K`.
    pub fn pack_weight_gw(
        &mut self,
        m: usize,
        k: usize,
        n: usize,
        b: &[i8],
        group: usize,
    ) -> Result<I8Weight> {
        if group == 0 {
            return Err(Error::shape("int8/pack/group"));
        }
        self.pack_codes(m, k, n, b, group)
    }

    /// Run every worker over its N-slice of `a`, accumulating into the worker's
    /// `iacc` (dense) or `facc` (group-wise, `ng = K/group`).
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
        w: &I8Weight,
    ) -> Result<()> {
        let workers = self.scratch(m, k, n, group)?;
        if workers.len() != w.bos.len() {
            return Err(Error::tiling("int8/weight-workers"));
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
                wt_bytes: w.bos[i].len(),
            };
            if w.sig[i] != now {
                return Err(Error::tiling("int8/weight-layout"));
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
                        .unwrap_or_else(|_| Err(Error::device("int8/worker", 0)))
                })
                .collect()
        });
        if let Some(e) = results.into_iter().find_map(|r| r.err()) {
            return Err(e);
        }
        Ok(())
    }

    /// `C[M,N] = A[M,K] * B[N,K]^T`, raw int32 (the C `rocket_matmul_int8_prepacked`).
    ///
    /// `w` must be a per-channel weight ([`I8Context::pack_weight`]); the group-wise
    /// form is [`I8Context::matmul_gw`]. `M % 4` is required — an unaligned `M`
    /// miscomputes on HW and the resident path cannot pad rows (a padded row would need
    /// a matching `a_scale`), so ragged row counts are padded caller-side; the pack's
    /// `M` is informational (canonical tiling makes the weight valid for every `M`).
    pub fn matmul(
        &mut self,
        m: usize,
        k: usize,
        n: usize,
        a: &[i8],
        c: &mut [i32],
        w: &I8Weight,
    ) -> Result<()> {
        if k != w.k || n != w.n || w.group != 0 {
            return Err(Error::shape("int8/matmul/shape"));
        }
        if a.len() != m * k || c.len() != m * n {
            return Err(Error::shape("int8/matmul/buffers"));
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
    /// `rocket_matmul_int8_prepacked_gw`).
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
        w: &I8Weight,
    ) -> Result<()> {
        if k != w.k || n != w.n || w.group == 0 {
            return Err(Error::shape("int8/matmul/shape"));
        }
        if a.len() != m * k || cf.len() != m * n {
            return Err(Error::shape("int8/matmul/buffers"));
        }
        let ng = k / w.group;
        if a_scale.len() != m * ng || b_scale.len() != n * ng {
            return Err(Error::shape("int8/matmul/scales"));
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
            let words = i8_words(&mut self.in_all);
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
                    layout::feat_scatter_i8(
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
                    let input =
                        self.in_all.dma_address() + ((mi * plan.nkt + ki) * plan.in_slot) as u64;
                    let weights = wt.dma_address() + ((ni * plan.nkt + ki) * plan.wt_slot) as u64;
                    let output = self.out_all.dma_address() + (out_off * 4) as u64;
                    let mut p = MatmulParams::new(
                        mtile as u32,
                        ktile as u32,
                        ntile as u32,
                        input as u32,
                        weights as u32,
                        output as u32,
                    );
                    p.fp32_to_fp16 = false;
                    ops.clear();
                    let count = emit(&p, &mut ops)?;
                    if ops.len() > RC_STRIDE {
                        return Err(Error::tiling("int8/regcmd"));
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
                    let group_idx = ki / self.kt_per_group.max(1);
                    meta.push((m0, n0, mtile, ntile, group_idx));
                    nb += 1;
                    done += 1;

                    if nb == BATCH || done == total {
                        self.regcmd.fini()?;
                        self.out_all.prep(0)?;
                        self.out_all.fini()?;
                        let in_h: [Handle; 3] =
                            [self.in_all.handle(), wt.handle(), self.regcmd.handle()];
                        let out_h = [self.out_all.handle()];
                        // `self` also owns the BOs; submit through the BO's device.
                        self.dev.submit(&tasks, &in_h, &out_h, 0)?;
                        self.out_all.prep(wait_ns())?;
                        {
                            let words = super::bos::i32_words_ref(&self.out_all);
                            let n0_base = self.n0;
                            for (j, &(m0, n0, mtile, ntile, g)) in meta.iter().enumerate() {
                                let slot = &words[j * plan.out_slot..(j + 1) * plan.out_slot];
                                if dense {
                                    layout::detile_accum_i32(
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
                                    layout::detile_accum_i8_scaled(
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
/// M-independent and one pack serves every `M` — the C `rki_worker_alloc`. Only the
/// live `Mt` and `nMt` follow the call.
fn plan_for(m: usize, k: usize, n: usize, group: usize) -> Result<Plan> {
    let max_tile = hw::current().max_tile as usize;
    let geom = if group == 0 { &INT8 } else { &INT8_GW };
    let (cmt, kt, nt) = plan_geom(geom, max_tile.max(4), k, n, group)?;
    if group > 0 && !group.is_multiple_of(kt) {
        return Err(Error::shape("int8/group-kt"));
    }
    Ok(int8_plan(m, k, n, cmt.min(m), kt, nt))
}

fn fanout_step(n: usize, threads: usize, align: usize) -> usize {
    let threads = threads.max(1);
    let align = align.max(1);
    n.div_ceil(threads).div_ceil(align) * align
}
