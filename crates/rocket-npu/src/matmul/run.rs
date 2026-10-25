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
use crate::regcmd::desc::{BATCH, RC_STRIDE};

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

/// [`one_shot`] under an explicit [`Plan`] (the cross-op FFN/MLP pinned tiling).
///
/// The C spellings this stands in for are `rocket_matmul_fp16` and `mm_compute_kacc` +
/// `mm_compute`: `policy` picks the routing. `a`, `b`, `c` must be exactly
/// `plan.m * plan.k`, `plan.n * plan.k` and `plan.m * plan.n` long (the padded cubes are
/// built here, exactly as the packing passes do).
pub(crate) fn one_shot_plan(
    dev: &crate::Device,
    plan: &Plan,
    policy: KaccPolicy,
    a: &[f16],
    b: &[f16],
    c: &mut [f16],
) -> Result<()> {
    if a.len() < plan.m * plan.k || b.len() < plan.n * plan.k || c.len() < plan.m * plan.n {
        return Err(Error::shape("one_shot_plan/buffers"));
    }
    let config = Config::from_env();
    let config = Config {
        kacc: config.kacc || policy != KaccPolicy::Auto,
        pipe: config.pipe,
    };
    let mut bos = Bos::new(dev, plan, config)?;
    bos.pack_input(plan, &a[..plan.m * plan.k])?;
    bos.pack_weights(plan, &b[..plan.n * plan.k])?;
    let wt = bos.wt().mem_ref();
    let mut out = vec![f16::ZERO; plan.m * plan.n];
    if config.kacc {
        match compute_kacc(plan, &mut bos, wt, &mut out) {
            Ok(()) => {
                c[..plan.m * plan.n].copy_from_slice(&out);
                return Ok(());
            }
            Err(e) if e.status() == crate::Status::Tiling => {
                if policy == KaccPolicy::Only {
                    // The C fused entries read this refusal as "not cube-resident" and
                    // fall back to the whole host-handoff block, not to another
                    // accumulation order.
                    return Err(e);
                }
            }
            Err(e) => return Err(e),
        }
    }
    // `Auto` with KACC turned off by the knob: the C's `rocket_matmul_fp16` goes straight
    // to CPU accumulation, and `Forced`/`Only` always take the KACC branch above.
    compute_cpu(plan, &mut bos, wt, &mut out)?;
    c[..plan.m * plan.n].copy_from_slice(&out);
    Ok(())
}

/// K-accumulation routing for [`one_shot_plan`].
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
            let input = bos.in_all.dma_address() + ((mi * plan.nkt + ki) * plan.in_slot * 2) as u64;
            let weights = wt.dma + ((ni * plan.nkt + ki) * plan.wt_slot * 2) as u64;
            let mut p = MatmulParams::new(
                plan.m_tile(mi) as u32,
                plan.k_tile(ki) as u32,
                plan.n_tile(ni) as u32,
                input as u32,
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
        let in_handles = [bos.in_all.handle(), wt.handle, bos.regcmd.handle()];
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

/// NPU-side K-accumulation (fp16 eltwise add ping-pong). Returns a `Tiling` error as
/// a routing sentinel for shapes the EW stride floor cannot serve.
pub fn compute_kacc(plan: &Plan, bos: &mut Bos, wt: MemRef, c: &mut [f16]) -> Result<()> {
    let last_m = plan.m - (plan.nmt - 1) * plan.mt;
    if last_m < 12 && last_m != plan.m {
        return Err(Error::tiling("kacc/last-m"));
    }
    if plan.m < 12 && plan.nmt == 1 && plan.nkt > 1 {
        return Err(Error::tiling("kacc/tiny-m"));
    }
    if bos.okacc0.is_none() || bos.pong.is_none() {
        return Err(Error::tiling("kacc/no-scratch"));
    }

    let mut prof = Prof::new();
    let n_tiles = plan.nmt * plan.nnt;
    let mut ops = Vec::with_capacity(160);
    let (mode, auto) = match knobs::int("ROCKET_REUSE") {
        Some(v) => (v.clamp(0, 2) as u8, false),
        None => (0, true),
    };
    let mut rmode = mode;
    if auto {
        rmode = if plan.nnt >= plan.nmt { 2 } else { 1 };
    }

    for base in (0..n_tiles).step_by(BATCH) {
        let g = (n_tiles - base).min(BATCH);
        let mut tasks: Vec<DrmRocketTask> = Vec::with_capacity(g);
        let mut bmi = vec![0usize; g];
        let mut bni = vec![0usize; g];

        for ki in 0..plan.nkt {
            bos.regcmd.prep(0)?;
            prof.mark(b'g');
            tasks.clear();
            let mut prev: Option<(usize, usize, usize, usize)> = None;
            for gi in 0..g {
                let tile = base + gi;
                let (mi, ni) = if rmode == 1 {
                    (tile % plan.nmt, tile / plan.nmt)
                } else {
                    (tile / plan.nnt, tile % plan.nnt)
                };
                let mtile = plan.m_tile(mi);
                let ntile = plan.n_tile(ni);
                let dst_is_ping = ki % 2 == 0;
                let (dst_dma, src_dma) = {
                    let ping = bos.okacc0.as_ref().unwrap();
                    let pong = bos.pong.as_ref().unwrap();
                    if dst_is_ping {
                        (ping.dma_address(), pong.dma_address())
                    } else {
                        (pong.dma_address(), ping.dma_address())
                    }
                };
                let out_off = gi * plan.out_slot;
                let output = (dst_dma + (out_off * 2) as u64) as u32;
                let add = (src_dma + (out_off * 2) as u64) as u32;
                let input =
                    bos.in_all.dma_address() + ((mi * plan.nkt + ki) * plan.in_slot * 2) as u64;
                let weights = wt.dma + ((ni * plan.nkt + ki) * plan.wt_slot * 2) as u64;
                let dims_ok = prev
                    .map(|(_, pm, pn, pk)| pm == mtile && pn == ntile && pk == plan.k_tile(ki))
                    .unwrap_or(false);
                let (wr, dr) = if gi > 0 && dims_ok {
                    match rmode {
                        1 => (prev.unwrap().0 == ni, false),
                        2 => (false, prev.unwrap().0 == mi),
                        _ => (false, false),
                    }
                } else {
                    (false, false)
                };
                let mut p = MatmulParams::new(
                    mtile as u32,
                    plan.k_tile(ki) as u32,
                    ntile as u32,
                    input as u32,
                    weights as u32,
                    output,
                );
                p.accumulate = ki > 0;
                p.add_dma = if ki > 0 { add } else { 0 };
                p.weight_reuse = wr;
                p.data_reuse = dr;
                ops.clear();
                let count = emit(&p, &mut ops)?;
                let regcmd = write_regcmd(&mut bos.regcmd, gi, &ops)?;
                tasks.push(DrmRocketTask {
                    regcmd,
                    regcmd_count: count as u32,
                });
                bmi[gi] = mi;
                bni[gi] = ni;
                prev = Some((
                    if rmode == 1 { ni } else { mi },
                    mtile,
                    ntile,
                    plan.k_tile(ki),
                ));
            }
            bos.regcmd.fini()?;
            prof.mark(b's');

            let dst_is_ping = ki % 2 == 0;
            {
                let dst = if dst_is_ping {
                    bos.okacc0.as_mut().unwrap()
                } else {
                    bos.pong.as_mut().unwrap()
                };
                dst.prep(0)?;
                dst.fini()?;
            }
            prof.mark(b's');
            let in_handles = if ki > 0 {
                [
                    bos.in_all.handle(),
                    wt.handle,
                    bos.regcmd.handle(),
                    if dst_is_ping {
                        bos.pong.as_ref().unwrap().handle()
                    } else {
                        bos.okacc0.as_ref().unwrap().handle()
                    },
                ]
                .to_vec()
            } else {
                vec![bos.in_all.handle(), wt.handle, bos.regcmd.handle()]
            };
            let out_handles = [if dst_is_ping {
                bos.okacc0.as_ref().unwrap().handle()
            } else {
                bos.pong.as_ref().unwrap().handle()
            }];
            let dev = bos.dev.clone();
            // KACC keeps the stock gapped per-task layout: it generates its groups one
            // K-tile at a time from its own loop, which this pass did not convert (the C
            // library does chain it).
            submit(&dev, &tasks, &in_handles, &out_handles, 0)?;
            prof.mark(b'u');
            {
                let dst = if dst_is_ping {
                    bos.okacc0.as_mut().unwrap()
                } else {
                    bos.pong.as_mut().unwrap()
                };
                dst.prep(wait_ns())?;
                dst.fini()?;
            }
            prof.mark(b'w');
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
            let bytes = fin.as_slice();
            let words = unsafe {
                std::slice::from_raw_parts(bytes.as_ptr().cast::<f16>(), bytes.len() / 2)
            };
            for gi in 0..g {
                let slot = &words[gi * plan.out_slot..(gi + 1) * plan.out_slot];
                layout::detile_store(
                    c,
                    plan.n,
                    slot,
                    bmi[gi] * plan.mt,
                    bni[gi] * plan.nt,
                    plan.m_tile(bmi[gi]),
                    plan.n_tile(bni[gi]),
                );
            }
        }
        fin.fini()?;
    }
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
}
