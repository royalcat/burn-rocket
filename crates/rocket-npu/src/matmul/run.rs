//! The compute paths: CPU-fp32 K-accumulation (the oracle) and NPU-side KACC.

use std::sync::OnceLock;
use std::time::Instant;

use half::f16;

use crate::device::uapi::DrmRocketTask;
use crate::device::{Handle, MemRef};
use crate::error::{Error, Result};
use crate::knobs;
use crate::log::log_error;
use crate::regcmd::fp16::{MatmulParams, emit};

use super::bos::{Bos, Config};
use super::layout;
use super::plan::Plan;
use crate::regcmd::desc::{BATCH, RC_STRIDE};

struct Prof {
    on: bool,
    gen_ms: f64,
    sync: f64,
    submit: f64,
    wait: f64,
    read: f64,
    t: Instant,
}

impl Prof {
    fn new() -> Self {
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
    fn mark(&mut self, bucket: u8) {
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
    fn dump(&self, plan: &Plan) {
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

fn write_regcmd(bos: &mut Bos, slot: usize, ops: &[crate::regcmd::desc::NpuOp]) -> Result<u32> {
    if ops.len() > RC_STRIDE {
        log_error!("regcmd slot overflow: {} > {RC_STRIDE}", ops.len());
        return Err(Error::tiling("regcmd"));
    }
    let base = slot * RC_STRIDE * 8;
    {
        let bytes = bos.regcmd.as_mut_slice();
        for (i, op) in ops.iter().enumerate() {
            bytes[base + i * 8..base + i * 8 + 8].copy_from_slice(&op.0.to_ne_bytes());
        }
    }
    Ok((bos.regcmd.dma_address() + base as u64) as u32)
}

fn submit(
    dev: &crate::Device,
    bos: &mut Bos,
    tasks: &[DrmRocketTask],
    in_handles: &[Handle],
    out_handles: &[Handle],
) -> Result<()> {
    dev.submit(tasks, in_handles, out_handles, 0)?;
    let _ = bos;
    Ok(())
}

/// CPU fp32 accumulation over every K-tile (the byte-exact reference path).
pub fn compute_cpu(plan: &Plan, bos: &mut Bos, wt: MemRef, c: &mut [f16]) -> Result<()> {
    bos.ensure_out()?;
    let mut prof = Prof::new();
    let mut acc = vec![0.0f32; plan.m * plan.n];
    let total = plan.jobs();
    let mut done = 0usize;
    let mut nb = 0usize;
    let mut tasks: Vec<DrmRocketTask> = Vec::with_capacity(BATCH);
    let mut meta: Vec<(usize, usize)> = Vec::with_capacity(BATCH);
    let mut ops = Vec::with_capacity(160);

    for mi in 0..plan.nmt {
        for ni in 0..plan.nnt {
            for ki in 0..plan.nkt {
                if nb == 0 {
                    bos.regcmd.prep(0)?;
                }
                prof.mark(b'g');
                let out_off = nb * plan.out_slot;
                let output_dma = (bos.out().dma_address() + (out_off * 2) as u64) as u32;
                let input =
                    bos.in_all.dma_address() + ((mi * plan.nkt + ki) * plan.in_slot * 2) as u64;
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
                let regcmd = write_regcmd(bos, nb, &ops)?;
                tasks.push(DrmRocketTask {
                    regcmd,
                    regcmd_count: count as u32,
                });
                meta.push((mi, ni));
                let _ = out_off;
                nb += 1;
                done += 1;

                if nb == BATCH || done == total {
                    bos.regcmd.fini()?;
                    prof.mark(b's');
                    bos.out_mut().prep(0)?;
                    bos.out_mut().fini()?;
                    prof.mark(b's');
                    let in_handles = [bos.in_all.handle(), wt.handle, bos.regcmd.handle()];
                    let out_handles = [bos.out().handle()];
                    let dev = bos.dev.clone();
                    submit(&dev, bos, &tasks, &in_handles, &out_handles)?;
                    prof.mark(b'u');
                    bos.out_mut().prep(wait_ns())?;
                    prof.mark(b'w');

                    let bytes = bos.out().as_slice();
                    let words = unsafe {
                        std::slice::from_raw_parts(bytes.as_ptr().cast::<f16>(), bytes.len() / 2)
                    };
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
                    nb = 0;
                    tasks.clear();
                    meta.clear();
                }
            }
        }
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
                let regcmd = write_regcmd(bos, gi, &ops)?;
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
            submit(&dev, bos, &tasks, &in_handles, &out_handles)?;
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
