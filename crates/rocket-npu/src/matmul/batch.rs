//! The batched fp16 matmul: `rocket_matmul_fp16_batch` and the persistent
//! `rocket_mm_batch` context.
//!
//! `nbatch` INDEPENDENT products `C[i] = A[i] · B[i]^T` that share one `(M, K, N)` run as
//! ONE job stream: every item's tiles are generated into a single regcmd BO and the
//! group's tasks go out in one submit + one fence wait, instead of `nbatch`
//! submit/wait round-trips. With `ROCKET_BATCH_SUBMIT=1` (plus a kernel that honors
//! `DRM_ROCKET_JOB_BATCHED`) the group's programs are also self-chained, so the whole
//! group fires one completion IRQ (the crate-internal `matmul::chain` layout;
//! [`super::chain_enabled`] reports whether it is active).
//!
//! Bit-identical to calling [`one_shot`] per item: same plan, same fp16 pack, same CPU
//! fp32 K-accumulation and de-tile/narrow, and the tiles are visited in `(item, mi, ni,
//! ki)` order with `ki` innermost, so each output tile's K-partials accumulate in
//! ascending `ki` exactly as [`super::compute_cpu`] does. A shape whose K axis is a
//! single tile skips the accumulator outright and stores the cube's f16 result straight
//! into the caller's output (what [`super::compute_kacc`] does) — the same bits, because
//! an f16 tile narrowed through f32 and back is unchanged. It deliberately does NOT
//! consult `ROCKET_KACC`/`ROCKET_REUSE`/`ROCKET_MM_PIPE` — those are single-matmul
//! performance variants, orthogonal to the batch lever. (A bit-for-bit A/B against
//! `one_shot` therefore needs `ROCKET_NO_KACC=1`, since `one_shot` may route to the
//! on-NPU KACC path, which rounds differently on purpose.)
//!
//! Shape contract: `M % 4 == 0`, `K % 32 == 0`, `N % 16 == 0`, as in C. `nbatch == 1`
//! falls through to [`one_shot`] on that item, which is also where the height-1 (GEMV)
//! pad lives — the batched path itself has no `M == 1` shim.

use half::f16;

use crate::device::uapi::{DRM_ROCKET_JOB_BATCHED, DrmRocketTask};
use crate::device::{Bo, Handle};
use crate::error::{Error, Result};
use crate::log::log_error;
use crate::regcmd::desc::{BATCH, CBUF_BANK_SIZE, NpuOp, RC_STRIDE};
use crate::regcmd::fp16::{MatmulParams, emit};

use super::bos;
use super::chain;
use super::layout;
use super::plan::Plan;
use super::run::{one_shot, wait_ns};
use crate::Device;

/// The `(M,K,N,nbatch)` layout of the resident pack. A context tracks ONE, so a caller
/// alternating two shapes should hold one context per shape — otherwise every run looks
/// like a layout switch and re-zeroes the BOs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Layout {
    m: usize,
    k: usize,
    n: usize,
    nbatch: usize,
}

/// One generated tile, kept until its group's submit has been read back.
#[derive(Debug, Clone, Copy)]
struct Tile {
    /// Index of the item (`A[item]`) this tile belongs to.
    item: usize,
    m0: usize,
    n0: usize,
    mtile: usize,
    ntile: usize,
    /// Output slot inside the rolling `out_all` BO, in fp16 elements.
    out_off: usize,
}

/// A persistent batched-matmul context: the resident BOs and host scratch reused across
/// runs of the same (or a new) shape.
///
/// The context is bound to the [`Device`] it was created from (the C context holds one
/// fd) and **is not thread-safe**: a run mutates the resident BOs, so every concurrent
/// worker needs its own context.
///
/// `in_all`/`wt_all`/`out_all` are allocated on the first run and grow only when a shape
/// needs more; the guard and the regcmd BO are fixed. The full-BO zero of the input and
/// weight packs is skipped whenever the layout repeats and no BO was reallocated, because
/// a fixed `(M,K,N,nbatch)` scatter rewrites the same live lanes and never touches the
/// padding — padding zeroed once stays zero. That skip is the per-call pack cost the
/// persistent form exists to reclaim over [`batch`].
pub struct MmBatch {
    dev: Device,
    /// Pushes the allocations off IOVA 0 (C `rocket_mm_batch_create`).
    guard: Bo,
    regcmd: Bo,
    in_all: Option<Bo>,
    wt_all: Option<Bo>,
    out_all: Option<Bo>,
    /// The resident `in_all`/`wt_all` are zero AND match `last`.
    prezeroed: bool,
    last: Option<Layout>,
    /// Per-item fp32 K-accumulators, `nbatch * M * N`; grow-only.
    acc: Vec<f32>,
}

impl MmBatch {
    /// Open a batch context on `dev` (C `rocket_mm_batch_create`).
    ///
    /// Allocates the guard and the `BATCH * RC_STRIDE` regcmd BO; the in/wt/out BOs are
    /// grown lazily by the first [`MmBatch::run`].
    pub fn new(dev: &Device) -> Result<MmBatch> {
        let guard = dev.alloc_bo(4096)?;
        let regcmd = dev.alloc_bo32(BATCH * RC_STRIDE * 8)?;
        Ok(MmBatch {
            dev: dev.clone(),
            guard,
            regcmd,
            in_all: None,
            wt_all: None,
            out_all: None,
            prezeroed: false,
            last: None,
            acc: Vec::new(),
        })
    }

    /// Drop the grow-only `in_all`/`wt_all`/`out_all` packs and the fp32 accumulators,
    /// keeping the (fixed-size) guard and regcmd BOs.
    ///
    /// The release half of a request-scoped trim: a caller that holds a context across
    /// requests (`attn::Attention` does, one per shape per worker) drops what a much
    /// larger request grew. Nothing is lost — every run re-packs its host matrices
    /// anyway — but the whole-BO zero of the input/weight packs is paid again on the
    /// next run, because "zeroed once stays zero" only holds for a BO that survived.
    pub fn trim(&mut self) {
        self.in_all = None;
        self.wt_all = None;
        self.out_all = None;
        self.prezeroed = false;
        self.last = None;
        self.acc = Vec::new();
    }

    /// Live bytes of the context's NPU buffer objects (guard, regcmd and the grown
    /// in/wt/out BOs).
    pub fn bytes(&self) -> usize {
        let mut n = self.guard.len() + self.regcmd.len();
        for bo in [&self.in_all, &self.wt_all, &self.out_all] {
            n += bo.as_ref().map_or(0, Bo::len);
        }
        n
    }

    /// `C[i] = A[i] · B[i]^T` for every item, in one job stream
    /// (C `rocket_mm_batch_run`).
    ///
    /// `a[i]` must be `[M,K]` and `b[i]` `[N,K]`; `c[i]` must hold `M*N` fp16 elements
    /// (a longer slice is filled at its head). `nbatch == 1` is the plain one-shot.
    ///
    /// A failure keeps the resident BOs for the next run, exactly like C: the packed
    /// layout is still self-consistent, so only the "already zeroed" claim is dropped
    /// when a pack could not finish.
    pub fn run(
        &mut self,
        m: usize,
        k: usize,
        n: usize,
        a: &[&[f16]],
        b: &[&[f16]],
        c: &mut [&mut [f16]],
    ) -> Result<()> {
        let nbatch = a.len();
        if nbatch == 0 || b.len() != nbatch || c.len() != nbatch {
            return Err(Error::shape("mm_batch/nbatch"));
        }
        if a.iter().any(|ai| ai.len() != m * k) {
            return Err(Error::shape("mm_batch/input"));
        }
        if b.iter().any(|bi| bi.len() != n * k) {
            return Err(Error::shape("mm_batch/weights"));
        }
        if c.iter().any(|ci| ci.len() < m * n) {
            return Err(Error::shape("mm_batch/output"));
        }
        if nbatch == 1 {
            // The C shim, including the height-1 pad that lives in `one_shot`.
            return one_shot(&self.dev, m, k, n, a[0], b[0], &mut c[0][..m * n]);
        }

        let plan = Plan::new(m, k, n)?;
        let in_item = plan.input_elems();
        let wt_item = plan.nnt * plan.nkt * plan.wt_slot;
        let in_sz = bytes_of(nbatch, in_item, "mm_batch/in-size")? + CBUF_BANK_SIZE;
        let wt_sz = bytes_of(nbatch, wt_item, "mm_batch/wt-size")? + CBUF_BANK_SIZE;
        let out_sz = BATCH * plan.out_slot * 2 + CBUF_BANK_SIZE;

        // A single K-tile needs no accumulation: the cube already holds this item's final
        // f16 result, so the readback stores it straight into the caller's output (what
        // `compute_kacc` does too) instead of round-tripping through the fp32
        // accumulator. Same bits — `detile_accum` + `narrow_f32_to_f16` is lossless on an
        // f16-origin tile — at a fraction of the host readback cost, which is the shape
        // attention's QK (K = head dim) and short-context AV (K = key axis) run.
        let direct_out = plan.nkt == 1;

        // From here to the end of the pack the padding state of in_all/wt_all is not
        // known: a failed grow leaves a fresh BO behind, so the "already zeroed" claim is
        // only re-established once both packs have finished. `prezeroed` therefore
        // records the PREVIOUS run's outcome (the C `b->prezeroed`), and `need_zero`
        // reads it below.
        let prezeroed = self.prezeroed;
        self.prezeroed = false;
        let grew_in = ensure(&self.dev, &mut self.in_all, in_sz)?;
        let grew_wt = ensure(&self.dev, &mut self.wt_all, wt_sz)?;
        // out_all is never pre-zeroed: the NPU writes it and only live slots are read.
        ensure(&self.dev, &mut self.out_all, out_sz)?;

        let layout = Layout { m, k, n, nbatch };
        let need_zero = zero_needed(grew_in || grew_wt, self.last, prezeroed, layout);

        {
            let bo = self.in_all.as_mut().expect("grown above");
            bo.prep(0)?;
            let words = bos::f16_words(bo);
            if need_zero {
                words.fill(f16::ZERO);
            }
            for (it, ai) in a.iter().enumerate() {
                bos::scatter_input(&plan, &mut words[it * in_item..], ai);
            }
            bo.fini()?;
        }
        {
            let bo = self.wt_all.as_mut().expect("grown above");
            bo.prep(0)?;
            let words = bos::f16_words(bo);
            if need_zero {
                words.fill(f16::ZERO);
            }
            for (it, bi) in b.iter().enumerate() {
                bos::scatter_weights(&plan, &mut words[it * wt_item..], bi);
            }
            bo.fini()?;
        }
        self.prezeroed = true;
        self.last = Some(layout);

        // `nbatch * M * N` fp32 accumulators: grow-only, zeroed every run (they
        // accumulate K-tiles). A single-K-tile shape stores its tiles directly instead.
        let acc_n = nbatch
            .checked_mul(m * n)
            .ok_or_else(|| Error::shape("mm_batch/acc"))?;
        if !direct_out {
            if self.acc.len() < acc_n {
                self.acc.resize(acc_n, 0.0);
            }
            self.acc[..acc_n].fill(0.0);
        }

        let jobs = plan.jobs();
        let total = nbatch * jobs;
        let chain_want = chain::enabled();
        // Hoisted per-run scratch; `group` holds the group's programs until the layout
        // decision is made (a chained task's trailer must encode its successor's length,
        // so the last task cannot be written before its successors are known).
        let mut group: Vec<u64> = Vec::new();
        let mut spans: Vec<(usize, u32)> = Vec::with_capacity(BATCH);
        let mut tiles: Vec<Tile> = Vec::with_capacity(BATCH);
        let mut tasks: Vec<DrmRocketTask> = Vec::with_capacity(BATCH);
        let mut ops: Vec<NpuOp> = Vec::with_capacity(RC_STRIDE);
        let mut done = 0usize;

        while done < total {
            let g = (total - done).min(BATCH);

            // Pass 1: generate the group's programs.
            group.clear();
            spans.clear();
            tiles.clear();
            for j in done..done + g {
                let item = j / jobs;
                let (mi, ni, ki) = plan.tile_of(j % jobs);

                let input = self.in_all.as_ref().expect("grown above").dma_address()
                    + ((item * in_item + (mi * plan.nkt + ki) * plan.in_slot) * 2) as u64;
                let weights = self.wt_all.as_ref().expect("grown above").dma_address()
                    + ((item * wt_item + (ni * plan.nkt + ki) * plan.wt_slot) * 2) as u64;
                let out_off = (j - done) * plan.out_slot;
                let output = self.out_all.as_ref().expect("grown above").dma_address()
                    + (out_off * 2) as u64;
                let mut p = MatmulParams::new(
                    plan.m_tile(mi) as u32,
                    plan.k_tile(ki) as u32,
                    plan.n_tile(ni) as u32,
                    input as u32,
                    weights as u32,
                    output as u32,
                );
                p.fp32_to_fp16 = true;
                ops.clear();
                let count = emit(&p, &mut ops)?;
                if count > RC_STRIDE {
                    log_error!("regcmd slot overflow: {count} > {RC_STRIDE}");
                    return Err(Error::tiling("mm_batch/regcmd"));
                }
                spans.push((group.len(), count as u32));
                group.extend(ops.iter().map(|op| op.0));
                tiles.push(Tile {
                    item,
                    m0: mi * plan.mt,
                    n0: ni * plan.nt,
                    mtile: plan.m_tile(mi),
                    ntile: plan.n_tile(ni),
                    out_off,
                });
            }

            // Pass 2: lay the group down and submit it.
            let chained =
                chain::group_chained(chain_want, g, spans.iter().map(|s| s.1), BATCH * RC_STRIDE);
            self.regcmd.prep(0)?;
            let base = self.regcmd.dma_address() as u32;
            tasks.clear();
            tasks.resize(g, DrmRocketTask::default());
            {
                let dst = bos::u64_words(&mut self.regcmd);
                for (i, &(off, count)) in spans.iter().enumerate() {
                    chain::pack(
                        chained,
                        dst,
                        base,
                        &mut tasks,
                        i,
                        &group[off..off + count as usize],
                        count,
                        RC_STRIDE,
                    )?;
                }
                if chained {
                    chain::verify(dst, base, &tasks, chain::words(spans[0].1))?;
                }
            }
            self.regcmd.fini()?;
            {
                let bo = self.out_all.as_mut().expect("grown above");
                bo.prep(0)?;
                bo.fini()?;
            }

            let in_handles: [Handle; 3] = [
                self.in_all.as_ref().expect("grown above").handle(),
                self.wt_all.as_ref().expect("grown above").handle(),
                self.regcmd.handle(),
            ];
            let out_handles = [self.out_all.as_ref().expect("grown above").handle()];
            let flags = if chained { DRM_ROCKET_JOB_BATCHED } else { 0 };
            self.dev.submit(&tasks, &in_handles, &out_handles, flags)?;

            if let Err(e) = self.out_all.as_mut().expect("grown above").prep(wait_ns()) {
                // A mis-chained group is exactly what shows up here, so name the layout.
                log_error!(
                    "mm_batch: group fence failed after {} ms (M={m} K={k} N={n} \
                     nbatch={nbatch}, chained={chained}, tasks={g}, tiles={done}/{total})",
                    wait_ns() / 1_000_000
                );
                return Err(e);
            }
            {
                let bo = self.out_all.as_ref().expect("grown above");
                let words = bos::f16_words_ref(bo);
                for t in &tiles {
                    let slot = &words[t.out_off..t.out_off + plan.out_slot];
                    if direct_out {
                        layout::detile_store(
                            &mut c[t.item][..m * n],
                            n,
                            slot,
                            t.m0,
                            t.n0,
                            t.mtile,
                            t.ntile,
                        );
                    } else {
                        let acc = &mut self.acc[t.item * m * n..(t.item + 1) * m * n];
                        layout::detile_accum(acc, n, slot, t.m0, t.n0, t.mtile, t.ntile);
                    }
                }
            }
            self.out_all.as_mut().expect("grown above").fini()?;

            done += g;
        }

        if !direct_out {
            for (it, ci) in c.iter_mut().enumerate() {
                layout::narrow_f32_to_f16(&mut ci[..m * n], &self.acc[it * m * n..][..m * n]);
            }
        }
        Ok(())
    }
}

/// `rocket_matmul_fp16_batch`: one transient batch context per call, so the BO lifetime
/// matches the stateless callers.
pub fn batch(
    dev: &Device,
    m: usize,
    k: usize,
    n: usize,
    a: &[&[f16]],
    b: &[&[f16]],
    c: &mut [&mut [f16]],
) -> Result<()> {
    if a.is_empty() || b.len() != a.len() || c.len() != a.len() {
        return Err(Error::shape("fp16_batch/nbatch"));
    }
    if a.len() == 1 {
        if a[0].len() != m * k || b[0].len() != n * k || c[0].len() < m * n {
            return Err(Error::shape("fp16_batch/buffers"));
        }
        return one_shot(dev, m, k, n, a[0], b[0], &mut c[0][..m * n]);
    }
    let mut ctx = MmBatch::new(dev)?;
    ctx.run(m, k, n, a, b, c)
}

/// The C's re-zero rule for the resident input/weight packs: zero the whole BO only when
/// an object was (re)allocated, the `(M,K,N,nbatch)` layout changed, or the previous run
/// did not finish its packs.
///
/// For a fixed layout the scatter rewrites the same live lanes and never touches padding,
/// so padding zeroed once stays zero — this predicate is exactly what the persistent
/// context exists to reclaim from [`batch`]'s per-call create/zero/free.
fn zero_needed(grew: bool, last: Option<Layout>, prezeroed: bool, layout: Layout) -> bool {
    grew || last != Some(layout) || !prezeroed
}

/// `n * elems * 2` bytes (fp16), with a typed `Shape` error instead of a wrap: `nbatch`
/// comes straight from the caller.
fn bytes_of(n: usize, elems: usize, op: &'static str) -> Result<usize> {
    n.checked_mul(elems)
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| Error::shape(op))
}

/// Grow `slot` to at least `need` bytes (C `rocket_bo_ensure32`): `Ok(true)` when the
/// object was (re)allocated, so its contents — and any zero padding — are gone.
fn ensure(dev: &Device, slot: &mut Option<Bo>, need: usize) -> Result<bool> {
    match slot {
        Some(bo) => bo.ensure32(need),
        None => {
            *slot = Some(dev.alloc_bo32(need)?);
            Ok(true)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The accumulator slice a tile of `item` writes into: `acc[item]` is `[M,N]`.
    fn acc_span(item: usize, m: usize, n: usize) -> (usize, usize) {
        (item * m * n, (item + 1) * m * n)
    }

    #[test]
    fn tiles_accumulate_into_their_own_item() {
        // Three items of a 4x64x16 shape: one tile each, so each item's fp32
        // accumulator is disjoint and no tile bleeds into a neighbour.
        let (m, n) = (4usize, 16usize);
        let mut acc = vec![0.0f32; 3 * m * n];
        for item in 0..3 {
            let (t0, t1) = acc_span(item, m, n);
            layout::detile_accum(&mut acc[t0..t1], n, &[f16::ONE; 64], 0, 0, m, n);
        }
        assert!(acc.iter().all(|v| *v == 1.0), "{acc:?}");
    }

    #[test]
    fn tile_order_is_item_major_with_ki_innermost() {
        // The global job index `run` walks decodes to (item, mi, ni, ki), item-major,
        // with ki innermost within each output tile.
        let plan = Plan::new(264, 1024, 512).unwrap();
        assert!(
            plan.nmt > 1 && plan.nnt > 1 && plan.nkt > 1,
            "the decode test must not be vacuous: {plan:?}"
        );
        let jobs = plan.jobs();
        let mut want = Vec::new();
        for item in 0..2 {
            for mi in 0..plan.nmt {
                for ni in 0..plan.nnt {
                    for ki in 0..plan.nkt {
                        want.push((item, mi, ni, ki));
                    }
                }
            }
        }
        let got: Vec<_> = (0..2 * jobs)
            .map(|j| {
                let (mi, ni, ki) = plan.tile_of(j % jobs);
                (j / jobs, mi, ni, ki)
            })
            .collect();
        assert_eq!(got, want);
    }

    #[test]
    fn ki_ascends_inside_one_output_tile() {
        // The property the fp32 accumulation order depends on: within one (item, mi, ni)
        // the ki run is contiguous and ascending, so each output tile's K-partials are
        // added in the same order `compute_cpu` adds them.
        let plan = Plan::new(264, 1024, 512).unwrap();
        assert!(plan.nkt > 1, "the shape must split K");
        let mut prev: Option<(usize, usize, usize)> = None;
        for j in 0..plan.jobs() {
            let (mi, ni, ki) = plan.tile_of(j);
            if let Some((pmi, pni, pki)) = prev
                && (pmi, pni) == (mi, ni)
            {
                assert_eq!(ki, pki + 1, "ki must ascend inside one output tile");
            }
            prev = Some((mi, ni, ki));
        }
    }

    #[test]
    fn a_uniform_group_chains_and_a_ragged_one_falls_back() {
        // The group layout decision itself lives in `chain::group_chained`; this pins the
        // shape of the `spans` a caller hands it (offset, count) end to end from `run`.
        let spans: Vec<(usize, u32)> = vec![(0, 126), (126, 126), (252, 126)];
        assert!(chain::group_chained(
            true,
            3,
            spans.iter().map(|s| s.1),
            BATCH * RC_STRIDE
        ));
        let ragged: Vec<(usize, u32)> = vec![(0, 126), (126, 128), (254, 126)];
        assert!(!chain::group_chained(
            true,
            3,
            ragged.iter().map(|s| s.1),
            BATCH * RC_STRIDE
        ));
    }

    #[test]
    fn the_resident_pack_is_zeroed_only_on_a_bo_or_layout_change() {
        let a = Layout {
            m: 4,
            k: 64,
            n: 16,
            nbatch: 2,
        };
        let b = Layout { nbatch: 3, ..a };
        assert!(
            zero_needed(true, Some(a), true, a),
            "a reallocated BO's contents are gone"
        );
        assert!(
            !zero_needed(false, Some(a), true, a),
            "a repeated layout keeps its zeroed padding"
        );
        assert!(zero_needed(false, None, true, a), "the first run zeroes");
        assert!(
            zero_needed(false, Some(a), true, b),
            "a new layout re-zeroes"
        );
        assert!(
            zero_needed(false, Some(a), false, a),
            "a run that did not finish its packs re-zeroes"
        );
        // Every field of the layout key is load-bearing.
        for other in [
            Layout { m: 8, ..a },
            Layout { k: 96, ..a },
            Layout { n: 32, ..a },
            Layout { nbatch: 1, ..a },
        ] {
            assert!(zero_needed(false, Some(a), true, other), "{other:?}");
        }
    }

    #[test]
    fn the_batched_path_has_no_gemv_pad() {
        // Only the nbatch==1 fall-through (`one_shot`) pads M==1; the plan is strict.
        assert!(Plan::new(1, 64, 16).is_err());
        assert!(Plan::new(4, 64, 16).is_ok());
    }
}
