//! fp16 inputs with an **fp32 output cube**: the port of `rocket_matmul_fp16_f32out`.
//!
//! Planning and the fp16 A/B packing are identical to [`super::run::one_shot`]; only
//! the output side differs, exactly like the C sibling:
//!
//! - the emitter runs with `fp32_to_fp16 = false`, so the DPU writes its full fp32
//!   accumulator (`size_e = 3`, `surf_add * 4`) instead of narrowing each K-partial to
//!   fp16;
//! - the output cube is therefore C2=4 and 4 B/elem, read back with
//!   [`layout::detile_accum_f32`] rather than the C2=8 fp16 de-tile;
//! - the host K-accumulation is `f64` and the result leaves as `(f32) acc[i]`, so
//!   neither a K-tile nor the final value is rounded to fp16. That is the genuine
//!   fp16-input dot product (the fp16-output path rounds each K-partial before the
//!   host sum).
//!
//! It costs 2x the output readback bytes, so it stays opt-in. The scratch BO set is
//! private to this module because the shared [`super::Bos`] sizes its output BO for
//! fp16 elements; it mirrors the C `guard`/`regcmd`/`in_all`/`wt_all`/`out_all`
//! allocation, and the tile batching, regcmd writer, submit seam and wait path are
//! the ones in [`super::run`].

use half::f16;

use crate::device::uapi::DrmRocketTask;
use crate::device::{Bo, Handle};
use crate::error::{Error, Result};
use crate::regcmd::desc::{BATCH, CBUF_BANK_SIZE, RC_STRIDE};
use crate::regcmd::fp16::{MatmulParams, emit};

use super::bos;
use super::layout;
use super::plan::Plan;
use super::run::{Prof, pad_m, submit, wait_ns, write_regcmd};
use crate::Device;

/// Bytes per output-cube element (the one geometry delta vs the fp16 path).
const OUT_ELEM: usize = 4;

/// The C path's BO set: eager, fp16 in/weight scratch, **f32** output scratch.
struct Scratch {
    _guard: Bo,
    regcmd: Bo,
    in_all: Bo,
    wt_all: Bo,
    out_all: Bo,
}

impl Scratch {
    fn new(dev: &Device, plan: &Plan) -> Result<Scratch> {
        let mut s = Scratch {
            _guard: dev.alloc_bo(4096)?,
            regcmd: dev.alloc_bo(BATCH * RC_STRIDE * 8)?,
            in_all: dev.alloc_bo(plan.nmt * plan.nkt * plan.in_slot * 2 + CBUF_BANK_SIZE)?,
            wt_all: dev.alloc_bo(plan.nnt * plan.nkt * plan.wt_slot * 2 + CBUF_BANK_SIZE)?,
            out_all: dev.alloc_bo(BATCH * plan.out_slot * OUT_ELEM + CBUF_BANK_SIZE)?,
        };
        bos::check_iova(&[&s.in_all, &s.wt_all, &s.out_all, &s.regcmd])?;
        // The scatters write live tiles only, so the padding lanes must be zero
        // before the first pack (the C memsets both BOs once, as here).
        bos::zero_bo(&mut s.in_all)?;
        bos::zero_bo(&mut s.wt_all)?;
        Ok(s)
    }
}

/// `C[M,N] = A[M,K] * B[N,K]^T` with the full fp32 accumulator; `M == 1` pads to 4
/// rows and returns row 0.
pub fn one_shot(
    dev: &Device,
    m: usize,
    k: usize,
    n: usize,
    a: &[f16],
    b: &[f16],
    c: &mut [f32],
) -> Result<()> {
    if m == 0 || k == 0 || n == 0 {
        return Err(Error::shape("f32out/shape"));
    }
    if a.len() != m * k || b.len() != n * k || c.len() != m * n {
        return Err(Error::shape("f32out/buffers"));
    }
    if m == 1 {
        let (mp, ap) = pad_m(m, k, a);
        let mut cp = vec![0.0f32; mp * n];
        one_shot(dev, mp, k, n, &ap, b, &mut cp)?;
        c.copy_from_slice(&cp[..n]);
        return Ok(());
    }
    let plan = Plan::new(m, k, n)?;
    let mut scratch = Scratch::new(dev, &plan)?;
    bos::pack_weights_into(&mut scratch.wt_all, &plan, b)?;
    bos::pack_input_into(&mut scratch.in_all, &plan, a)?;
    compute(dev, &plan, &mut scratch, c)
}

fn compute(dev: &Device, plan: &Plan, s: &mut Scratch, c: &mut [f32]) -> Result<()> {
    let mut prof = Prof::new();
    let mut acc = vec![0.0f64; plan.m * plan.n];
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
                    s.regcmd.prep(0)?;
                }
                prof.mark(b'g');
                let out_off = nb * plan.out_slot;
                let output = (s.out_all.dma_address() + (out_off * OUT_ELEM) as u64) as u32;
                let input =
                    s.in_all.dma_address() + ((mi * plan.nkt + ki) * plan.in_slot * 2) as u64;
                let weights =
                    s.wt_all.dma_address() + ((ni * plan.nkt + ki) * plan.wt_slot * 2) as u64;
                let mut p = MatmulParams::new(
                    plan.m_tile(mi) as u32,
                    plan.k_tile(ki) as u32,
                    plan.n_tile(ni) as u32,
                    input as u32,
                    weights as u32,
                    output,
                );
                p.fp32_to_fp16 = false;
                ops.clear();
                let count = emit(&p, &mut ops)?;
                let regcmd = write_regcmd(&mut s.regcmd, nb, &ops)?;
                tasks.push(DrmRocketTask {
                    regcmd,
                    regcmd_count: count as u32,
                });
                meta.push((mi, ni));
                nb += 1;
                done += 1;

                if nb == BATCH || done == total {
                    s.regcmd.fini()?;
                    prof.mark(b's');
                    s.out_all.prep(0)?;
                    s.out_all.fini()?;
                    prof.mark(b's');
                    let in_handles: [Handle; 3] =
                        [s.in_all.handle(), s.wt_all.handle(), s.regcmd.handle()];
                    let out_handles = [s.out_all.handle()];
                    submit(dev, &tasks, &in_handles, &out_handles, 0)?;
                    prof.mark(b'u');
                    s.out_all.prep(wait_ns())?;
                    prof.mark(b'w');

                    let bytes = s.out_all.as_slice();
                    let words = unsafe {
                        std::slice::from_raw_parts(bytes.as_ptr().cast::<f32>(), bytes.len() / 4)
                    };
                    for (j, &(mi, ni)) in meta.iter().enumerate() {
                        let slot = &words[j * plan.out_slot..(j + 1) * plan.out_slot];
                        layout::detile_accum_f32(
                            &mut acc,
                            plan.n,
                            slot,
                            mi * plan.mt,
                            ni * plan.nt,
                            plan.m_tile(mi),
                            plan.n_tile(ni),
                        );
                    }
                    s.out_all.fini()?;
                    prof.mark(b'r');
                    nb = 0;
                    tasks.clear();
                    meta.clear();
                }
            }
        }
    }

    for (d, v) in c.iter_mut().zip(&acc) {
        *d = *v as f32;
    }
    prof.dump(plan);
    Ok(())
}
