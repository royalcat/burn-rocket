//! The compute paths: CPU-fp32 K-accumulation (the oracle) and NPU-side KACC.

use std::sync::OnceLock;
use std::time::Instant;

use half::f16;

use crate::device::uapi::{DRM_ROCKET_JOB_BATCHED, DrmRocketTask};
use crate::device::{Bo, Handle, MemRef};
use crate::error::{Error, Result};
use crate::knobs;
use crate::log::log_error;
use crate::regcmd::fp16::{MatmulParams, emit};

use super::bf16::Bf16Stream;
use super::bos::{Bos, Config};
use super::context::Weight;
use super::int8::I8Weight;
use super::layout;
use super::plan::Plan;
use crate::regcmd::desc::{BATCH, CBUF_BANK_SIZE, RC_STRIDE};

/// Phase timers for one compute path, gated by `ROCKET_MM_PROFILE`.
pub(crate) struct Prof {
    on: bool,
    gen_ms: f64,
    sync: f64,
    submit: f64,
    wait: f64,
    read: f64,
    t: Instant,
}

impl Prof {
    pub(crate) fn new() -> Self {
        Self {
            on: knobs::present("ROCKET_MM_PROFILE"),
            gen_ms: 0.0,
            sync: 0.0,
            submit: 0.0,
            wait: 0.0,
            read: 0.0,
            t: Instant::now(),
        }
    }
    pub(crate) fn mark(&mut self, bucket: u8) {
        if !self.on {
            return;
        }
        let ms = self.t.elapsed().as_secs_f64() * 1e3;
        self.t = Instant::now();
        match bucket {
            b'g' => self.gen_ms += ms,
            b's' => self.sync += ms,
            b'u' => self.submit += ms,
            b'w' => self.wait += ms,
            b'r' => self.read += ms,
            _ => {}
        }
    }
    pub(crate) fn dump(&self, plan: &Plan) {
        if !self.on {
            return;
        }
        eprintln!(
            "profile M={} K={} N={} jobs={}: gen={:.2} sync={:.2} submit={:.2} wait={:.2} read={:.2}",
            plan.m,
            plan.k,
            plan.n,
            plan.jobs(),
            self.gen_ms,
            self.sync,
            self.submit,
            self.wait,
            self.read
        );
    }
}

/// The output-fence deadline (`ROCKET_WAIT_MS`, default 8 s).
pub fn wait_ns() -> u64 {
    static NS: OnceLock<u64> = OnceLock::new();
    *NS.get_or_init(|| {
        let ms = knobs::positive("ROCKET_WAIT_MS").unwrap_or(8000);
        ms as u64 * 1_000_000
    })
}

/// `C[M,N] = A[M,K] * B[N,K]^T` with a fresh scratch set; M==1 is padded to 4.
pub fn one_shot(
    dev: &crate::Device,
    m: usize,
    k: usize,
    n: usize,
    a: &[f16],
    b: &[f16],
    c: &mut [f16],
) -> Result<()> {
    if m == 1 {
        let mut ap = vec![f16::ZERO; 4 * k];
        ap[..k].copy_from_slice(a);
        let mut cp = vec![f16::ZERO; 4 * n];
        one_shot(dev, 4, k, n, &ap, b, &mut cp)?;
        c.copy_from_slice(&cp[..n]);
        return Ok(());
    }
    let plan = Plan::new(m, k, n)?;
    let config = Config::from_env();
    let mut bos = Bos::new(dev, &plan, config)?;
    bos.pack_input(&plan, a)?;
    bos.pack_weights(&plan, b)?;
    let wt = bos.wt().mem_ref();
    let mut out = vec![f16::ZERO; m * n];
    if config.kacc {
        match compute_kacc(&plan, &mut bos, wt, &mut out) {
            Ok(()) => {
                c.copy_from_slice(&out);
                return Ok(());
            }
            Err(e) if e.status() == crate::Status::Tiling => {}
            Err(e) => return Err(e),
        }
    }
    compute_cpu(&plan, &mut bos, wt, &mut out)?;
    c.copy_from_slice(&out);
    Ok(())
}

/// K-accumulation routing for the cross-op cube API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KaccPolicy {
    /// KACC first, CPU accumulation when the EW stride floor refuses the shape
    /// (`rocket_matmul_fp16`); honours `ROCKET_NO_KACC`.
    Auto,
    /// KACC first regardless of `ROCKET_NO_KACC`, CPU accumulation on a `Tiling`
    /// refusal — the C fused path's down projection, which calls the KACC primitive
    /// directly and only falls back to `mm_compute` on the sentinel.
    Forced,
    /// KACC only; a `Tiling` refusal propagates (the C `mm_compute_kacc_cube`, whose
    /// caller falls back to the non-cube block).
    Only,
}

/// The A operand of a tiled matmul: where its `[M,K]` input cube lives.
///
/// Normally the per-call scratch [`Bos::in_all`]. The cross-op FFN/MLP hands in its own
/// BO instead, so the consumer matmul reads a *producer* matmul's output cube directly —
/// the C's `b.in_all = cube_g` alias. That is legal exactly when the two plans match tile
/// for tile (`out_slot == in_slot`, `nNt(producer) == nKt(consumer)`, same `nMt`/`Mt`),
/// which is the fused entries' matched-tiling gate.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AOperand {
    dma: u64,
    handle: Handle,
}

impl AOperand {
    /// The `Bos` per-call input scratch.
    pub(crate) fn scratch(bos: &Bos) -> AOperand {
        AOperand::of(bos.in_all.mem_ref())
    }

    pub(crate) fn of(m: MemRef) -> AOperand {
        AOperand {
            dma: m.dma,
            handle: m.handle,
        }
    }

    /// The A operand's GEM handle (the submit in-list).
    fn handle(&self) -> Handle {
        self.handle
    }

    /// DMA address of the `(mi, ki)` input tile (the C's
    /// `in_all.dma_address + (mi*nKt + ki)*in_slot*sizeof(_Float16)`).
    fn tile_dma(&self, plan: &Plan, mi: usize, ki: usize) -> u64 {
        self.dma + ((mi * plan.nkt + ki) * plan.in_slot * 2) as u64
    }
}

/// Resolve the compute-path choices for a policy: `Forced`/`Only` always provision the
/// KACC scratch (the C's fused entries call the KACC primitives directly, so they cannot
/// depend on `ROCKET_NO_KACC`), `Auto` honours the knob.
pub(crate) fn policy_config(policy: KaccPolicy) -> Config {
    let config = Config::from_env();
    Config {
        kacc: config.kacc || policy != KaccPolicy::Auto,
        pipe: config.pipe,
    }
}

/// Allocate (and zero) a cross-op output cube for `plan`: `cube_elems` fp16 plus the C's
/// one-CBUF-bank of slack (`mm_cube_elems(pl) * sizeof(_Float16) + 65536`).
///
/// Zeroing is not cosmetic: a ragged tile leaves lanes the device never writes, and the
/// on-cube activation would turn garbage there into NaN, which the consumer's zeroed
/// weight lanes cannot mute (`0 * NaN` is NaN).
pub(crate) fn alloc_cube(dev: &crate::Device, plan: &Plan) -> Result<Bo> {
    let mut bo = dev.alloc_bo(plan.cube_elems() * 2 + 2 * CBUF_BANK_SIZE)?;
    super::bos::check_iova(&[&bo])?;
    super::bos::zero_bo(&mut bo)?;
    Ok(bo)
}

pub(crate) fn write_regcmd(
    regcmd: &mut Bo,
    slot: usize,
    ops: &[crate::regcmd::desc::NpuOp],
) -> Result<u32> {
    if ops.len() > RC_STRIDE {
        log_error!("regcmd slot overflow: {} > {RC_STRIDE}", ops.len());
        return Err(Error::tiling("regcmd"));
    }
    let base = slot * RC_STRIDE * 8;
    {
        let bytes = regcmd.as_mut_slice();
        for (i, op) in ops.iter().enumerate() {
            bytes[base + i * 8..base + i * 8 + 8].copy_from_slice(&op.0.to_ne_bytes());
        }
    }
    Ok((regcmd.dma_address() + base as u64) as u32)
}

pub(crate) fn submit(
    dev: &crate::Device,
    tasks: &[DrmRocketTask],
    in_handles: &[Handle],
    out_handles: &[Handle],
    flags: u32,
) -> Result<()> {
    dev.submit(tasks, in_handles, out_handles, flags)
}

/// CPU fp32 accumulation over every K-tile (the byte-exact reference path).
///
/// Tiles are submitted in groups of up to [`BATCH`], one submit + one fence wait per
/// group. With `ROCKET_BATCH_SUBMIT=1` (see the crate-internal `matmul::chain`) a group
/// whose programs all
/// have the same word count is laid out contiguously and self-chained, so the group also
/// costs one completion IRQ; otherwise — and always when chaining is off, which is the
/// default — it keeps the gapped per-task layout byte-for-byte.
pub fn compute_cpu(plan: &Plan, bos: &mut Bos, wt: MemRef, c: &mut [f16]) -> Result<()> {
    let input = AOperand::scratch(bos);
    compute_cpu_from(plan, bos, wt, input, c)
}

/// [`compute_cpu`] reading its A operand from `input` (the cross-op cube consumer: the
/// C's `b.in_all = cube_g` then `mm_compute`, its tiny-M fallback for the down matmul).
pub(crate) fn compute_cpu_from(
    plan: &Plan,
    bos: &mut Bos,
    wt: MemRef,
    input: AOperand,
    c: &mut [f16],
) -> Result<()> {
    bos.ensure_out()?;
    let mut prof = Prof::new();
    let mut acc = vec![0.0f32; plan.m * plan.n];
    let total = plan.jobs();
    let chain_want = super::chain::enabled();
    // The group's programs are buffered until the layout is decided: a chained task's
    // trailer encodes its SUCCESSOR's length, so the last task cannot be written before
    // its successors are known. `spans` holds each program's (offset, word count).
    let mut group: Vec<u64> = Vec::new();
    let mut spans: Vec<(usize, u32)> = Vec::with_capacity(BATCH);
    let mut meta: Vec<(usize, usize)> = Vec::with_capacity(BATCH);
    let mut tasks: Vec<DrmRocketTask> = Vec::with_capacity(BATCH);
    let mut ops = Vec::with_capacity(160);
    let mut done = 0usize;

    while done < total {
        let g = (total - done).min(BATCH);

        // 1) Generate the group's programs, ki innermost.
        prof.mark(b'g');
        group.clear();
        spans.clear();
        meta.clear();
        for j in done..done + g {
            let (mi, ni, ki) = plan.tile_of(j);
            let out_off = (j - done) * plan.out_slot;
            let output_dma = (bos.out().dma_address() + (out_off * 2) as u64) as u32;
            let input_dma = input.tile_dma(plan, mi, ki);
            let weights = wt.dma + ((ni * plan.nkt + ki) * plan.wt_slot * 2) as u64;
            let mut p = MatmulParams::new(
                plan.m_tile(mi) as u32,
                plan.k_tile(ki) as u32,
                plan.n_tile(ni) as u32,
                input_dma as u32,
                weights as u32,
                output_dma,
            );
            p.fp32_to_fp16 = true;
            ops.clear();
            let count = emit(&p, &mut ops)?;
            if count > RC_STRIDE {
                log_error!("regcmd slot overflow: {count} > {RC_STRIDE}");
                return Err(Error::tiling("regcmd"));
            }
            spans.push((group.len(), count as u32));
            group.extend(ops.iter().map(|op| op.0));
            meta.push((mi, ni));
        }
        prof.mark(b'g');

        // 2) Lay the group down: contiguous + self-chained when it fits that contract,
        //    gapped per-task otherwise.
        let chained = super::chain::group_chained(
            chain_want,
            g,
            spans.iter().map(|s| s.1),
            BATCH * RC_STRIDE,
        );
        bos.regcmd.prep(0)?;
        let base = bos.regcmd.dma_address() as u32;
        tasks.clear();
        tasks.resize(g, DrmRocketTask::default());
        {
            let dst = super::bos::u64_words(&mut bos.regcmd);
            for (i, &(off, count)) in spans.iter().enumerate() {
                super::chain::pack(
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
                super::chain::verify(dst, base, &tasks, super::chain::words(spans[0].1))?;
            }
        }
        bos.regcmd.fini()?;
        prof.mark(b's');
        bos.out_mut().prep(0)?;
        bos.out_mut().fini()?;
        prof.mark(b's');
        let in_handles = [input.handle, wt.handle, bos.regcmd.handle()];
        let out_handles = [bos.out().handle()];
        // A chained group must be submitted with the batched flag, or the kernel runs the
        // self-linked layout down the per-task path and stalls.
        let flags = if chained { DRM_ROCKET_JOB_BATCHED } else { 0 };
        let dev = bos.dev.clone();
        submit(&dev, &tasks, &in_handles, &out_handles, flags)?;
        prof.mark(b'u');
        bos.out_mut().prep(wait_ns())?;
        prof.mark(b'w');

        let bytes = bos.out().as_slice();
        let words =
            unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<f16>(), bytes.len() / 2) };
        for (j, &(mi, ni)) in meta.iter().enumerate() {
            let slot = &words[j * plan.out_slot..(j + 1) * plan.out_slot];
            layout::detile_accum(
                &mut acc,
                plan.n,
                slot,
                mi * plan.mt,
                ni * plan.nt,
                plan.m_tile(mi),
                plan.n_tile(ni),
            );
        }
        bos.out_mut().fini()?;
        prof.mark(b'r');
        done += g;
    }

    let out = &mut c[..plan.m * plan.n];
    layout::narrow_f32_to_f16(out, &acc);
    prof.dump(plan);
    Ok(())
}

// ── NPU-side K-accumulation (the KACC family) ──────────────────────────────────

/// The KACC EW-stride floor: the C `mm_compute_kacc`'s tiny-M guards.
///
/// The accumulation job runs its EW add with a `MAX(M,12)` stride floor, so a last M-tile
/// below 12 rows would read a mis-strided operand. Both refusals are the `Tiling` routing
/// sentinel of the C library — not a hard error: the caller (and the fused FFN/MLP) treats
/// it as "fall back to the CPU-accumulation oracle for this shape".
fn guard_kacc_m(plan: &Plan) -> Result<()> {
    let last_m = plan.m - (plan.nmt - 1) * plan.mt;
    if last_m < 12 && last_m != plan.m {
        return Err(Error::tiling("kacc/last-m"));
    }
    if plan.m < 12 && plan.nmt == 1 && plan.nkt > 1 {
        return Err(Error::tiling("kacc/tiny-m"));
    }
    Ok(())
}

/// The device-free part of [`compute_kacc_cube`]'s contract.
///
/// The whole cube must be written by ONE NPU batch (`nMt*nNt <= BATCH`), because the
/// ping-pong scratch is sized for one batch and the de-tiling path is what normally reads
/// each batch's tiles back before the next batch reuses the buffers. `cube_bytes` is the
/// BO's size in bytes (the C's `cube->size` check, `ROCKET_E_NOMEM`).
fn guard_kacc_cube(plan: &Plan, cube_bytes: usize) -> Result<()> {
    guard_kacc_m(plan)?;
    let n_tiles = plan.nmt * plan.nnt;
    if n_tiles > BATCH {
        return Err(Error::tiling("kacc-cube/batch"));
    }
    if cube_bytes < n_tiles * plan.out_slot * 2 {
        return Err(Error::nomem("kacc-cube/cube"));
    }
    Ok(())
}

/// `ROCKET_REUSE` resolved for this shape: 0 = no CBUF operand reuse, 1 = weight-major
/// tile order, 2 = data-major (the canonical) order. AUTO picks the deeper consecutive
/// run, `mi` innermost when the plan has fewer M-tiles than N-tiles, as the C does.
fn kacc_rmode(plan: &Plan) -> u8 {
    match knobs::int("ROCKET_REUSE") {
        Some(v) => v.clamp(0, 2) as u8,
        None if plan.nnt >= plan.nmt => 2,
        None => 1,
    }
}

/// The `(mi, ni)` a KACC job's tile takes: the C's two compatible orderings.
///
/// Mode 1 walks `mi` innermost, so consecutive tiles share a weight column block (CBUF
/// weight reuse). Mode 2 — and mode 0, which has the same decode with both reuse bits
/// clear — walks the canonical `mi`-major order, so consecutive tiles share the input and
/// tile `gi` occupies the `gi`th output slot. **The cube-resident path must use mode 2**:
/// there the output slot index is the contract with the next op's input cube, not a
/// scheduling choice (the C says the same in `mm_compute_kacc_cube`).
fn kacc_decode(plan: &Plan, rmode: u8, tile: usize) -> (usize, usize) {
    if rmode == 1 {
        (tile % plan.nmt, tile / plan.nmt)
    } else {
        (tile / plan.nnt, tile % plan.nnt)
    }
}

/// The program parameters of one KACC tile: the C's `matmul_params_t` inside the
/// `ki` loop, shared by [`compute_kacc_from`] and [`compute_kacc_cube`] so the two paths
/// can only differ in where their buffers live.
///
/// `dst`/`src` are the ping-pong output buffers, `gi` is the tile's index within the job
/// (its slot is `gi * out_slot` inside both). `reuse` is `(weight_reuse, data_reuse)`.
#[allow(clippy::too_many_arguments)]
fn kacc_params(
    plan: &Plan,
    wt: MemRef,
    input: AOperand,
    ki: usize,
    gi: usize,
    mi: usize,
    ni: usize,
    dst: MemRef,
    src: Option<MemRef>,
    reuse: (bool, bool),
) -> MatmulParams {
    let out_off = (gi * plan.out_slot * 2) as u64;
    let mut p = MatmulParams::new(
        plan.m_tile(mi) as u32,
        plan.k_tile(ki) as u32,
        plan.n_tile(ni) as u32,
        input.tile_dma(plan, mi, ki) as u32,
        (wt.dma + ((ni * plan.nkt + ki) * plan.wt_slot * 2) as u64) as u32,
        (dst.dma + out_off) as u32,
    );
    p.accumulate = ki > 0;
    p.add_dma = match src {
        Some(s) if ki > 0 => (s.dma + out_off) as u32,
        _ => 0,
    };
    p.weight_reuse = reuse.0;
    p.data_reuse = reuse.1;
    p
}

/// Emit and run one KACC K-tile job: the `g` tiles of `[base, base + g)` against K-tile
/// `ki`, accumulating onto `src` and landing in `dst`.
///
/// This is the shared body of the C `mm_compute_kacc` and `mm_compute_kacc_cube` `ki`
/// loops; the two differ only in where the ping-pong buffers live, in the tile decode
/// (`rmode`), and in what happens to the final buffer (de-tile vs. leave the cube). The
/// submit keeps the C's stock gapped per-task layout (`write_regcmd`, one task per tile)
/// and one fence per K-tile; returns each tile's `(mi, ni)` in `gi` order so the caller
/// can de-tile its slot.
#[allow(clippy::too_many_arguments)]
fn kacc_job(
    plan: &Plan,
    regcmd: &mut Bo,
    dev: &crate::Device,
    wt: MemRef,
    input: AOperand,
    ki: usize,
    base: usize,
    g: usize,
    rmode: u8,
    dst: &mut Bo,
    src: Option<&Bo>,
    prof: &mut Prof,
) -> Result<Vec<(usize, usize)>> {
    let dst_ref = dst.mem_ref();
    let src_ref = src.map(Bo::mem_ref);
    let mut ops = Vec::with_capacity(160);
    let mut tasks: Vec<DrmRocketTask> = Vec::with_capacity(g);
    let mut meta: Vec<(usize, usize)> = Vec::with_capacity(g);

    regcmd.prep(0)?;
    prof.mark(b'g');
    // CBUF operand reuse is legal only between tiles of THIS job: a K-tile is its own
    // submit + fence, so the CBUF is cold at ki = 0 and the prev-key resets every ki. Both
    // orderings map tile `gi` to the same `(mi, ni)` across all ki, so the output slot is
    // unaffected.
    let mut prev: Option<(usize, usize, usize)> = None; // (reuse key, Mtile, Ntile)
    for gi in 0..g {
        let (mi, ni) = kacc_decode(plan, rmode, base + gi);
        let (mtile, ntile) = (plan.m_tile(mi), plan.n_tile(ni));
        let (wr, dr) = match prev {
            Some((key, pm, pn)) if gi > 0 && pm == mtile && pn == ntile => match rmode {
                1 => (key == ni, false),
                2 => (false, key == mi),
                _ => (false, false),
            },
            _ => (false, false),
        };
        let p = kacc_params(plan, wt, input, ki, gi, mi, ni, dst_ref, src_ref, (wr, dr));
        ops.clear();
        let count = emit(&p, &mut ops)?;
        let regcmd_addr = write_regcmd(regcmd, gi, &ops)?;
        tasks.push(DrmRocketTask {
            regcmd: regcmd_addr,
            regcmd_count: count as u32,
        });
        meta.push((mi, ni));
        prev = Some((if rmode == 1 { ni } else { mi }, mtile, ntile));
    }
    regcmd.fini()?;
    prof.mark(b's');

    // The C's prep/fini pair around a submit: wait for `dst`'s previous user, hand it
    // back to the device, kick, then fence on its completion.
    dst.prep(0)?;
    dst.fini()?;
    prof.mark(b's');

    // in-list: input, weights, regcmd, (+ src when accumulating: the EW add reads it).
    let mut in_handles = vec![input.handle(), wt.handle, regcmd.handle()];
    if let Some(src) = src {
        in_handles.push(src.handle());
    }
    let out_handles = [dst.handle()];
    submit(dev, &tasks, &in_handles, &out_handles, 0)?;
    prof.mark(b'u');

    dst.prep(wait_ns())?;
    dst.fini()?;
    prof.mark(b'w');
    Ok(meta)
}

/// NPU-side K-accumulation (fp16 eltwise add ping-pong). Returns a `Tiling` error as
/// a routing sentinel for shapes the EW stride floor cannot serve.
pub fn compute_kacc(plan: &Plan, bos: &mut Bos, wt: MemRef, c: &mut [f16]) -> Result<()> {
    let input = AOperand::scratch(bos);
    compute_kacc_from(plan, bos, wt, input, c)
}

/// [`compute_kacc`] reading its A operand from `input` instead of `Bos::in_all`.
///
/// The cross-op FFN/MLP's down matmul is the caller: it hands in the producer cube
/// (`AOperand::of(cube.mem_ref())`), the C's `b2m.in_all = cube_g`. The two plans must
/// match tile for tile (`out_slot == in_slot`, `nNt(producer) == nKt(consumer)`, same
/// `nMt`/`Mt`), which is the fused entries' matched-tiling gate; then tile `(mi, ni)`'s
/// slot in the cube *is* this plan's `(mi, ki)` input tile.
pub(crate) fn compute_kacc_from(
    plan: &Plan,
    bos: &mut Bos,
    wt: MemRef,
    input: AOperand,
    c: &mut [f16],
) -> Result<()> {
    guard_kacc_m(plan)?;
    if bos.okacc0.is_none() || bos.pong.is_none() {
        return Err(Error::tiling("kacc/no-scratch"));
    }

    let mut prof = Prof::new();
    let n_tiles = plan.nmt * plan.nnt;
    let rmode = kacc_rmode(plan);

    for base in (0..n_tiles).step_by(BATCH) {
        let g = (n_tiles - base).min(BATCH);
        // The last K-tile's decode is the whole group's: every ki walks the same tiles.
        let mut meta: Vec<(usize, usize)> = Vec::with_capacity(g);
        for ki in 0..plan.nkt {
            let dst_is_ping = ki.is_multiple_of(2);
            let (dst, src) = if dst_is_ping {
                (bos.okacc0.as_mut().unwrap(), bos.pong.as_ref().unwrap())
            } else {
                (bos.pong.as_mut().unwrap(), bos.okacc0.as_ref().unwrap())
            };
            meta = kacc_job(
                plan,
                &mut bos.regcmd,
                &bos.dev,
                wt,
                input,
                ki,
                base,
                g,
                rmode,
                dst,
                (ki > 0).then_some(src),
                &mut prof,
            )?;
        }

        let dst_is_ping = (plan.nkt - 1).is_multiple_of(2);
        let fin = if dst_is_ping {
            bos.okacc0.as_mut().unwrap()
        } else {
            bos.pong.as_mut().unwrap()
        };
        fin.prep(0)?;
        prof.mark(b'r');
        {
            let words = super::bos::f16_words_ref(fin);
            for (gi, &(mi, ni)) in meta.iter().enumerate() {
                let slot = &words[gi * plan.out_slot..(gi + 1) * plan.out_slot];
                layout::detile_store(
                    c,
                    plan.n,
                    slot,
                    mi * plan.mt,
                    ni * plan.nt,
                    plan.m_tile(mi),
                    plan.n_tile(ni),
                );
            }
        }
        fin.fini()?;
    }
    prof.dump(plan);
    Ok(())
}

/// Cross-op KACC compute: the same NPU-side fp16 K-accumulation as [`compute_kacc`], but
/// the COMPLETE output cube is left in `cube` (canonical `[nMt][nNt]` tile order) for the
/// next op to read directly, with **no de-tile and no fp32 host accumulation** (the C
/// `mm_compute_kacc_cube`).
///
/// Contract, all of it the C's:
/// - `nMt*nNt <= BATCH`: every tile of the cube must co-exist, so the whole cube has to be
///   one NPU batch.
/// - the tiny-M EW stride floor of [`guard_kacc_m`], as a `Tiling` sentinel the caller
///   answers with [`compute_cpu_from`] (the C's `mm_compute` fallback).
/// - `cube` holds at least `nMt*nNt*out_slot` fp16 — [`alloc_cube`] is the sized-to-fit
///   (and zeroed) allocator for exactly this.
///
/// The A operand is `input`: `Bos::in_all` for the first projection, a producer matmul's
/// cube for the consumer (the C's `b2m.in_all = cube_g`). The alternate ping-pong buffer
/// is the resident [`Bos::pong`] when it is already cube-sized, else a local scratch BO
/// (the C's `pong_local`); a single K-tile needs none. On return `cube` is CPU-visible
/// (the C's trailing `rocket_bo_prep(cube, 0)`), so the on-cube bias/activation pass can
/// read the mapping before handing it back with a `prep`/`fini` pair.
pub(crate) fn compute_kacc_cube(
    plan: &Plan,
    bos: &mut Bos,
    wt: MemRef,
    input: AOperand,
    cube: &mut Bo,
) -> Result<()> {
    guard_kacc_cube(plan, cube.len())?;
    let n_tiles = plan.nmt * plan.nnt;
    let need = n_tiles * plan.out_slot * 2;
    // The final buffer after nKt-1 accumulations is `cube`; the ping-pong is arranged so
    // that parity falls out of nKt (the C's `finidx`).
    let finidx = (plan.nkt - 1) & 1;

    let reuse_pong = plan.nkt > 1 && bos.pong.as_ref().is_some_and(|b| b.len() >= need);
    let mut alt: Option<Bo> = if plan.nkt > 1 {
        if reuse_pong {
            bos.pong.take()
        } else {
            let bo = bos.dev.alloc_bo(need + CBUF_BANK_SIZE)?;
            super::bos::check_iova(&[&bo])?;
            Some(bo)
        }
    } else {
        None
    };

    let mut prof = Prof::new();
    // DATA reuse is mandatory here, not a perf knob: it is what makes tile `gi` map to
    // `(mi = gi/nNt, ni = gi%nNt)`, so its output slot `gi*out_slot` *is* the canonical
    // `(mi*nNt + ni)` slot the consumer's input cube expects.
    let rmode = 2;
    let result = (|| -> Result<()> {
        for ki in 0..plan.nkt {
            let dst_is_cube = (ki & 1) == finidx;
            let (dst, src) = if dst_is_cube {
                (&mut *cube, alt.as_ref())
            } else {
                (
                    alt.as_mut()
                        .expect("a multi-K-tile cube job always has an alternate buffer"),
                    Some(&*cube),
                )
            };
            let src = if ki > 0 { src } else { None };
            kacc_job(
                plan,
                &mut bos.regcmd,
                &bos.dev,
                wt,
                input,
                ki,
                0,
                n_tiles,
                rmode,
                dst,
                src,
                &mut prof,
            )?;
        }
        // Leave the complete cube CPU-visible for the on-cube element-wise pass.
        cube.prep(0)?;
        Ok(())
    })();

    if reuse_pong {
        bos.pong = alt.take();
    }
    prof.mark(b'r');
    result?;
    prof.dump(plan);
    Ok(())
}

// ── the remaining one-shot entry points of the C library ────────────────────────

/// `rocket_matmul_fp16_mt`: N split across `nthreads` worker fds, each running the
/// tiled matmul on its column slice (one scheduling entity per fd is what lets the
/// driver spread the job over the NPU cores).
///
/// Opens and drops its own context, like the C per-call `_mt` wrapper; `nthreads` is
/// clamped to 1..=8 by [`Context::new`]. `M == 1` is padded to 4 rows (the C `_mt`
/// delegates to `rocket_matmul_fp16`, whose height-1 shim does the same); other
/// ragged `M` are rejected by the planner, as in C.
pub fn one_shot_mt(
    m: usize,
    k: usize,
    n: usize,
    a: &[f16],
    b: &[f16],
    c: &mut [f16],
    nthreads: usize,
) -> Result<()> {
    // C `rocket_matmul_fp16_mt`: split the output columns N into >=16-aligned slices,
    // give each worker its own fd, and run the unchanged single-fd path on its slice.
    // This is the stateless per-call fan-out; `Context` is the resident/streaming form.
    let threads = nthreads.clamp(1, 8);
    if m == 1 {
        if c.len() < n {
            return Err(Error::shape("fp16_mt/output"));
        }
        let (mp, ap) = pad_m(m, k, a);
        let mut cp = vec![f16::ZERO; mp * n];
        one_shot_mt(mp, k, n, &ap, b, &mut cp, threads)?;
        c[..n].copy_from_slice(&cp[..n]);
        return Ok(());
    }
    if a.len() < m * k || b.len() < n * k || c.len() < m * n {
        return Err(Error::shape("fp16_mt/buffers"));
    }
    let step = (n.div_ceil(threads).div_ceil(16) * 16).max(16);
    let slices: Vec<(usize, usize)> = (0..threads)
        .map(|t| (t * step, ((t + 1) * step).min(n)))
        .filter(|(n0, n1)| n0 < n1)
        .collect();
    let base = crate::device::affinity::base();
    let results: Vec<Result<(usize, Vec<f16>)>> = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(slices.len());
        for (i, &(n0, n1)) in slices.iter().enumerate() {
            handles.push(scope.spawn(move || -> Result<(usize, Vec<f16>)> {
                crate::device::affinity::pin_current(i, base);
                let dev = crate::Device::open()?;
                let nsub = n1 - n0;
                let mut sub = vec![f16::ZERO; m * nsub];
                one_shot(&dev, m, k, nsub, a, &b[n0 * k..n1 * k], &mut sub)?;
                Ok((n0, sub))
            }));
        }
        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .unwrap_or_else(|_| Err(Error::device("fp16_mt/worker", 0)))
            })
            .collect()
    });
    for r in results {
        let (n0, sub) = r?;
        let nsub = sub.len() / m;
        for row in 0..m {
            c[row * n + n0..][..nsub].copy_from_slice(&sub[row * nsub..][..nsub]);
        }
    }
    Ok(())
}

/// `rocket_matmul_bf16`: single-fd one-shot bf16 (f32 A/B, truncated to bf16 on the
/// scatter; f32 output).
///
/// A thin wrapper over the streaming engine, which is the only bf16 engine (the C
/// library documents the two as the same product). The shape contract is therefore
/// the stream's: `M % 4 == 0`, `K % 32 == 0`, `N % 16 == 0` — pad ragged row counts
/// with [`pad_m`] (the C one-shot pads `M == 1` itself; the resident stream does not).
pub fn one_shot_bf16(
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) -> Result<()> {
    let mut s = Bf16Stream::new(1)?;
    s.matmul(m, k, n, a, b, c)
}

/// `rocket_matmul_bf16_mt`: the bf16 one-shot with N spread over `nthreads` worker
/// fds (`nthreads` clamped to 1..=8). Same shape contract as [`one_shot_bf16`].
pub fn one_shot_bf16_mt(
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    nthreads: usize,
) -> Result<()> {
    Bf16Stream::matmul_mt(m, k, n, a, b, c, nthreads)
}

/// `rocket_weights_bytes`: resident NPU-BO footprint of a packed fp16 weight (the
/// per-worker weight tiles; the shared scratch is cached across weights and not
/// counted).
pub fn weights_bytes(w: &Weight) -> usize {
    w.bytes()
}

/// `rocket_i8_weights_bytes`: the int8 sibling of [`weights_bytes`].
pub fn i8_weights_bytes(w: &I8Weight) -> usize {
    w.bytes()
}

/// The documented `rocket_pad_m` contract: zero-pad `A[M,K]` to a multiple of 4 rows.
///
/// Returns `(M', A_padded)`. The height-1 GEMV is the case the HW mis-computes, which
/// is why every one-shot path in the C library pads `M == 1`; the resident int8 path
/// requires `M % 4` outright and leaves the pad to the caller. Note that the C tree in
/// this snapshot documents `rocket_pad_m` (API.md) but does not define it, so this is
/// the documented behaviour rather than a port of an existing body.
pub fn pad_m(m: usize, k: usize, a: &[f16]) -> (usize, Vec<f16>) {
    assert_eq!(a.len(), m * k, "pad_m: A is [M,K]");
    let mp = m.div_ceil(4) * 4;
    if mp == m {
        return (m, a.to_vec());
    }
    let mut out = vec![f16::ZERO; mp * k];
    out[..m * k].copy_from_slice(a);
    (mp, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_m_zero_pads_to_a_multiple_of_four_rows() {
        let k = 3;
        let a: Vec<f16> = (0..5 * k).map(|i| f16::from_f32(i as f32)).collect();
        let (mp, ap) = pad_m(5, k, &a);
        assert_eq!(mp, 8);
        assert_eq!(ap.len(), 8 * k);
        assert_eq!(&ap[..5 * k], &a[..]);
        assert!(ap[5 * k..].iter().all(|v| *v == f16::ZERO));

        let (m1, one) = pad_m(1, 2, &[f16::ONE, f16::from_f32(2.0)]);
        assert_eq!(m1, 4);
        assert_eq!(one.len(), 4 * 2);
        assert_eq!(&one[..2], &[f16::ONE, f16::from_f32(2.0)]);
        assert!(one[2..].iter().all(|v| *v == f16::ZERO));

        let (m4, aligned) = pad_m(4, 2, &[f16::ONE; 8]);
        assert_eq!(m4, 4);
        assert_eq!(aligned.len(), 8);
    }

    // ── the KACC routing guards and the cross-op cube contract ─────────────────

    fn mem(dma: u64, handle: Handle) -> MemRef {
        MemRef { dma, handle }
    }

    #[test]
    fn kacc_guards_refuse_the_c_tiny_m_shapes() {
        // A last M-tile below the EW stride floor with M spanning more than one tile.
        let ragged = Plan::new(264, 1024, 512).unwrap();
        assert!(ragged.nmt > 1 && ragged.m_tile(ragged.nmt - 1) == 8);
        assert_eq!(
            guard_kacc_m(&ragged).unwrap_err().status(),
            crate::Status::Tiling
        );
        // A whole M below the floor is fine with one K-tile (nothing to accumulate) and
        // refuses with more than one (the C's `pl->M < 12 && nMt == 1 && nKt > 1`).
        let tiny_one_kt = Plan::new(8, 8192, 16).unwrap();
        assert!(tiny_one_kt.nmt == 1 && tiny_one_kt.nkt == 1);
        assert!(guard_kacc_m(&tiny_one_kt).is_ok());
        let tiny = Plan::new(8, 32768, 16).unwrap();
        assert!(tiny.nmt == 1 && tiny.nkt > 1);
        assert_eq!(
            guard_kacc_m(&tiny).unwrap_err().status(),
            crate::Status::Tiling
        );
        // A ragged last tile at or above the floor is served.
        let ok = Plan::new(280, 1024, 512).unwrap();
        assert_eq!(ok.m_tile(ok.nmt - 1), 24);
        assert!(guard_kacc_m(&ok).is_ok());
    }

    #[test]
    fn a_cube_job_must_fit_one_batch_and_a_whole_cube() {
        // 2 M-tiles x 64 N-tiles: the cube cannot be one NPU batch, so no ping-pong
        // arrangement could keep every tile alive — the C's `ROCKET_E_TILING`.
        let wide = Plan::new(512, 64, 16384).unwrap();
        assert!(wide.nmt * wide.nnt > BATCH);
        assert_eq!(
            guard_kacc_cube(&wide, usize::MAX).unwrap_err().status(),
            crate::Status::Tiling
        );

        // A short cube is `ROCKET_E_NOMEM`, checked after the batch/groups guards.
        let small = Plan::new(64, 256, 128).unwrap();
        assert_eq!(small.nmt * small.nnt, 1);
        let need = small.nmt * small.nnt * small.out_slot * 2;
        assert_eq!(small.cube_elems() * 2, need);
        assert_eq!(
            guard_kacc_cube(&small, need - 1).unwrap_err().status(),
            crate::Status::NoMem
        );
        assert!(guard_kacc_cube(&small, need).is_ok());
        // `alloc_cube` is the sized-to-fit allocator: `cube_elems` fp16 plus the C's
        // one-CBUF-bank slack (`mm_cube_elems(pl)*sizeof(_Float16) + 65536`).
        assert!(small.cube_elems() * 2 + 2 * CBUF_BANK_SIZE >= need);
    }

    #[test]
    fn kacc_decodes_cover_every_tile_once() {
        let p = Plan::new(512, 1024, 512).unwrap();
        assert!(p.nmt > 1 && p.nnt > 1);
        for rmode in [0u8, 1, 2] {
            let mut seen = vec![false; p.nmt * p.nnt];
            for tile in 0..p.nmt * p.nnt {
                let (mi, ni) = kacc_decode(&p, rmode, tile);
                assert!(mi < p.nmt && ni < p.nnt, "mode {rmode} tile {tile}");
                let slot = mi * p.nnt + ni;
                assert!(!seen[slot], "mode {rmode} repeats tile {slot}");
                seen[slot] = true;
            }
            assert!(seen.iter().all(|s| *s), "mode {rmode} skips a tile");
        }
        // Mode 1 walks `mi` innermost (consecutive tiles share a weight column block);
        // every other mode keeps the canonical `mi`-major decode, where tile `gi`'s output
        // slot IS the `gi`th cube slot — the cross-op cube's contract.
        assert_eq!(kacc_decode(&p, 1, 1), (1, 0));
        assert_eq!(kacc_decode(&p, 2, 1), (0, 1));
        assert_eq!(kacc_decode(&p, 0, 1), (0, 1));
        assert_eq!(kacc_decode(&p, 2, p.nnt + 1), (1, 1));
        assert_eq!(kacc_decode(&p, 1, p.nmt + 1), (1, 1));
    }

    #[test]
    fn a_operand_tile_addresses_walk_mi_then_ki() {
        let p = Plan::new(512, 1024, 256).unwrap();
        assert!(p.nmt > 1 && p.nkt > 1);
        let a = AOperand::of(mem(0x1000, 7));
        assert_eq!(a.handle(), 7);
        for mi in 0..p.nmt {
            for ki in 0..p.nkt {
                let want = 0x1000 + ((mi * p.nkt + ki) * p.in_slot * 2) as u64;
                assert_eq!(a.tile_dma(&p, mi, ki), want, "mi={mi} ki={ki}");
            }
        }
        // The last slot ends exactly at the end of the packed-A region.
        let last = a.tile_dma(&p, p.nmt - 1, p.nkt - 1) + (p.in_slot * 2) as u64;
        assert_eq!(last, 0x1000 + (p.input_elems() * 2) as u64);
    }

    #[test]
    fn a_cube_job_runs_the_detiled_job_programs() {
        // nKt odd, so the cube (the C's `finidx == 0`) is written on the same K-tile
        // parity `compute_kacc_from` uses for its ping buffer. Give the two arms the same
        // two buffer addresses and their programs must be byte for byte identical: the
        // cube variant differs only in where the buffers live and in skipping the de-tile,
        // which is why its output is the same f16 values the de-tiled path produces.
        let p = Plan::new(256, 1152, 256).unwrap();
        assert_eq!((p.nmt, p.nnt, p.nkt), (1, 1, 3));
        assert_eq!((p.nkt - 1) & 1, 0);
        let wt = mem(0x4000, 2);
        let input = AOperand::of(mem(0x8000, 3));
        let (ping, pong) = (mem(0x1_0000, 4), mem(0x2_0000, 5));
        let finidx = (p.nkt - 1) & 1;

        let mut ops_a = Vec::new();
        let mut ops_b = Vec::new();
        for ki in 0..p.nkt {
            for gi in 0..p.nmt * p.nnt {
                let (mi, ni) = kacc_decode(&p, 2, gi);
                // `compute_kacc_from`: ping on even ki.
                let (dst_a, src_a) = if ki.is_multiple_of(2) {
                    (ping, pong)
                } else {
                    (pong, ping)
                };
                let a = kacc_params(
                    &p,
                    wt,
                    input,
                    ki,
                    gi,
                    mi,
                    ni,
                    dst_a,
                    Some(src_a),
                    (false, true),
                );
                // `compute_kacc_cube`: the cube holds the final buffer.
                let (dst, src) = if (ki & 1) == finidx {
                    (ping, pong)
                } else {
                    (pong, ping)
                };
                let b = kacc_params(&p, wt, input, ki, gi, mi, ni, dst, Some(src), (false, true));
                ops_a.clear();
                ops_b.clear();
                let count = emit(&a, &mut ops_a).unwrap();
                assert_eq!(count, emit(&b, &mut ops_b).unwrap(), "ki={ki} gi={gi}");
                assert_eq!(ops_a, ops_b, "ki={ki} gi={gi}");

                // The parameter set itself: the tile's live extents, the shared input and
                // weight tiles, the slot offset inside the ping-pong buffer, and the
                // accumulation pair.
                assert_eq!(
                    (a.m, a.k, a.n),
                    (
                        p.m_tile(mi) as u32,
                        p.k_tile(ki) as u32,
                        p.n_tile(ni) as u32
                    )
                );
                assert_eq!(
                    a.input_dma,
                    (0x8000 + (mi * p.nkt + ki) * p.in_slot * 2) as u32
                );
                assert_eq!(
                    a.weights_dma,
                    (0x4000 + (ni * p.nkt + ki) * p.wt_slot * 2) as u32
                );
                let off = (gi * p.out_slot * 2) as u64;
                assert_eq!(a.output_dma, (dst_a.dma + off) as u32);
                assert_eq!(a.accumulate, ki > 0);
                assert_eq!(a.add_dma, if ki > 0 { (src_a.dma + off) as u32 } else { 0 });
            }
        }
    }
}
