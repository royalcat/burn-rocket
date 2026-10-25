//! tf32: the planner and the one-shot matmul.
//!
//! Port of `rocket_matmul_plan_tf32` / `rocket_matmul_tf32` in `rocket_matmul.c`:
//! one fd, a transient BO set (in/wt/out all **4 B per element**), the tf32 geometry
//! of [`super::plan`] and [`crate::regcmd::tf32::emit`] for the program.
//!
//! tf32 = fp32 range (8-bit exponent) + fp16 precision (10-bit mantissa), carried in
//! an fp32 container. The caller feeds **raw fp32** and the MAC **truncates** each
//! operand's mantissa to 10 bits (zero the low 13 bits, round toward zero — the
//! rounding the C library's own `tf32rt` gate models); nothing is truncated host-side,
//! unlike the bf16 path, whose scatter narrows to the high 16 bits. The output is the
//! full fp32 accumulator (`size_e = 3`, `surf_add = stride * 4`) gathered at the C2=4
//! fp32 cube, and K-tile partials are summed on the host in **f64** — so there is no
//! saturation and no per-tile narrowing anywhere.
//!
//! The 4-byte geometry is the one thing that is not bf16's: the feature cube is C2=4
//! `(K/4, M, 4)`, the weight tile is `(N/16, K/16, 16, 16)` (the K-group halves from
//! fp16's 32; the N-group stays 16), and a 64-byte hardware "data entry" holds 16
//! elements instead of 32. `K % 16`, `N % 16` and `M % 4` (or `M == 1`) are required.
//!
//! tf32 is the lowest-value rung of the datatype matrix — half rate (256x3 MACs/cycle
//! against fp16's 512x3) and bf16 already gives fp32 range at full speed — so this is
//! completeness, not a workload. Numbers are measured against a tf32-rounded reference
//! in `src/bin/tf32_selftest.rs`.

use crate::Device;
use crate::device::uapi::DrmRocketTask;
use crate::device::{Bo, Handle};
use crate::error::{Error, Result};
use crate::regcmd::desc::{BATCH, CBUF_BANK_SIZE, RC_STRIDE};
use crate::regcmd::fp16::MatmulParams;
use crate::regcmd::tf32::emit;

use super::bos::{check_iova, f32_words, f32_words_ref, zero_bo};
use super::layout;
use super::plan::{Plan, TF32, plan_geom};
use super::run::{Prof, submit, wait_ns, write_regcmd};

/// The tf32 planner: `(Mt, Kt, Nt)` for the 4-byte geometry (`N % 16`, `K % 16`,
/// `M % 4`; Kt ceiling 8192, no bank slack). The C `rocket_matmul_plan_tf32`.
pub fn plan_tf32(m: usize, k: usize, n: usize) -> Result<(usize, usize, usize)> {
    plan_geom(&TF32, m, k, n, 0)
}

/// The tf32 tiling plan for a resolved `(Mt, Kt, Nt)`: the slots are fp32 element
/// counts, with the cube K granule 16 (not 32) as in the C's `rup(Kt, 16)`.
pub(crate) fn tf32_plan(m: usize, k: usize, n: usize, mt: usize, kt: usize, nt: usize) -> Plan {
    Plan::derive_geom(&TF32, m, k, n, mt, kt, nt)
}

/// The transient BO set one tf32 call drives — the C locals in `rocket_matmul_tf32`:
/// a 4 KiB guard (to push the real allocations off IOVA 0), the regcmd batch, the raw
/// fp32 A and B cubes and the fp32 output cube.
///
/// `in_all`/`wt_all` are zeroed once at allocation: the scatters write live tile lanes
/// only, so the padding lanes past a ragged tile must already be zero.
struct Shots {
    dev: Device,
    _guard: Bo,
    regcmd: Bo,
    in_all: Bo,
    wt_all: Bo,
    out_all: Bo,
}

impl Shots {
    fn new(dev: &Device, plan: &Plan) -> Result<Shots> {
        const ELEM: usize = 4;
        let in_sz = plan.nmt * plan.nkt * plan.in_slot * ELEM + CBUF_BANK_SIZE;
        let wt_sz = plan.nnt * plan.nkt * plan.wt_slot * ELEM + CBUF_BANK_SIZE;
        let rc_sz = BATCH * RC_STRIDE * 8;
        let out_sz = BATCH * plan.out_slot * ELEM + CBUF_BANK_SIZE;

        let guard = dev.alloc_bo(4096)?;
        let regcmd = dev.alloc_bo(rc_sz)?;
        let mut in_all = dev.alloc_bo(in_sz)?;
        let mut wt_all = dev.alloc_bo(wt_sz)?;
        let out_all = dev.alloc_bo(out_sz)?;
        check_iova(&[&in_all, &wt_all, &out_all, &regcmd])?;
        zero_bo(&mut in_all)?;
        zero_bo(&mut wt_all)?;
        Ok(Shots {
            dev: dev.clone(),
            _guard: guard,
            regcmd,
            in_all,
            wt_all,
            out_all,
        })
    }

    /// `A[M,K]` (raw f32) -> the `(K/4, M, 4)` feature cube.
    fn pack_input(&mut self, plan: &Plan, a: &[f32]) -> Result<()> {
        assert_eq!(a.len(), plan.m * plan.k);
        self.in_all.prep(0)?;
        {
            let words = f32_words(&mut self.in_all);
            for mi in 0..plan.nmt {
                let m0 = mi * plan.mt;
                let mtile = plan.m_tile(mi);
                for ki in 0..plan.nkt {
                    let k0 = ki * plan.kt;
                    let ktile = plan.k_tile(ki);
                    let off = (mi * plan.nkt + ki) * plan.in_slot;
                    layout::feat_scatter_tf32(
                        &mut words[off..off + plan.in_slot],
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
        self.in_all.fini()
    }

    /// `B[N,K]` (raw f32) -> the `(N/16, K/16, 16, 16)` weight cube.
    fn pack_weights(&mut self, plan: &Plan, b: &[f32]) -> Result<()> {
        assert_eq!(b.len(), plan.n * plan.k);
        self.wt_all.prep(0)?;
        {
            let words = f32_words(&mut self.wt_all);
            for ni in 0..plan.nnt {
                let n0 = ni * plan.nt;
                let ntile = plan.n_tile(ni);
                for ki in 0..plan.nkt {
                    let k0 = ki * plan.kt;
                    let ktile = plan.k_tile(ki);
                    let off = (ni * plan.nkt + ki) * plan.wt_slot;
                    layout::wt_scatter_tile_tf32(
                        &mut words[off..off + plan.wt_slot],
                        b,
                        plan.k,
                        n0,
                        k0,
                        ntile,
                        ktile,
                    );
                }
            }
        }
        self.wt_all.fini()
    }
}

/// `C[M,N] = A[M,K] * B[N,K]^T`, tf32 x tf32 -> fp32 (the C `rocket_matmul_tf32`).
///
/// `A` and `B` are raw fp32 and `C` comes back as fp32; the hardware rounds each
/// operand's mantissa to 10 bits, so the result matches an f64 dot product of the
/// **tf32-rounded** inputs to ~1.5e-7 relative (not the plain-f32 product). The host
/// K-accumulation is f64, which keeps a tiled K's partial sum exact.
///
/// `M == 1` is padded to a 4-row tile and row 0 is returned: the height-1 GEMV
/// geometry miscomputes on HW, which is why every one-shot path here pads it. Other
/// ragged `M` are rejected by the planner (`M % 4`), as are `K % 16` and `N % 16`.
pub fn one_shot_tf32(
    dev: &Device,
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) -> Result<()> {
    if m == 1 {
        if a.len() < k || b.len() < n * k || c.len() < n {
            return Err(Error::shape("tf32/buffers"));
        }
        let mut ap = vec![0.0f32; 4 * k];
        ap[..k].copy_from_slice(&a[..k]);
        let mut cp = vec![0.0f32; 4 * n];
        one_shot_tf32(dev, 4, k, n, &ap, b, &mut cp)?;
        c[..n].copy_from_slice(&cp[..n]);
        return Ok(());
    }
    if a.len() < m * k || b.len() < n * k || c.len() < m * n {
        return Err(Error::shape("tf32/buffers"));
    }
    let (mt, kt, nt) = plan_tf32(m, k, n)?;
    let plan = tf32_plan(m, k, n, mt, kt, nt);
    let mut shots = Shots::new(dev, &plan)?;
    shots.pack_weights(&plan, b)?;
    shots.pack_input(&plan, a)?;

    let mut acc = vec![0.0f64; m * n];
    run(&plan, &mut shots, &mut acc)?;
    for (d, s) in c[..m * n].iter_mut().zip(&acc) {
        *d = *s as f32;
    }
    Ok(())
}

/// Submit every tile in groups of [`BATCH`] and accumulate the fp32 partials in f64.
fn run(plan: &Plan, shots: &mut Shots, acc: &mut [f64]) -> Result<()> {
    let mut prof = Prof::new();
    let total = plan.jobs();
    let mut tasks: Vec<DrmRocketTask> = Vec::with_capacity(BATCH);
    let mut meta: Vec<(usize, usize)> = Vec::with_capacity(BATCH);
    let mut ops = Vec::with_capacity(160);
    let mut done = 0usize;
    let mut nb = 0usize;

    for mi in 0..plan.nmt {
        for ni in 0..plan.nnt {
            for ki in 0..plan.nkt {
                if nb == 0 {
                    shots.regcmd.prep(0)?;
                }
                prof.mark(b'g');
                let out_off = nb * plan.out_slot;
                let input =
                    shots.in_all.dma_address() + ((mi * plan.nkt + ki) * plan.in_slot * 4) as u64;
                let weights =
                    shots.wt_all.dma_address() + ((ni * plan.nkt + ki) * plan.wt_slot * 4) as u64;
                let output = shots.out_all.dma_address() + (out_off * 4) as u64;
                // Neither `fp32_to_fp16` nor `accumulate` is read: the tf32 generator
                // hardcodes the fp32 output writer and the plain-conv path, as in C.
                let p = MatmulParams::new(
                    plan.m_tile(mi) as u32,
                    plan.k_tile(ki) as u32,
                    plan.n_tile(ni) as u32,
                    input as u32,
                    weights as u32,
                    output as u32,
                );
                ops.clear();
                let count = emit(&p, &mut ops)?;
                let regcmd = write_regcmd(&mut shots.regcmd, nb, &ops)?;
                tasks.push(DrmRocketTask {
                    regcmd,
                    regcmd_count: count as u32,
                });
                meta.push((mi, ni));
                nb += 1;
                done += 1;

                if nb == BATCH || done == total {
                    shots.regcmd.fini()?;
                    prof.mark(b's');
                    shots.out_all.prep(0)?;
                    shots.out_all.fini()?;
                    prof.mark(b's');
                    let in_h: [Handle; 3] = [
                        shots.in_all.handle(),
                        shots.wt_all.handle(),
                        shots.regcmd.handle(),
                    ];
                    let out_h = [shots.out_all.handle()];
                    submit(&shots.dev, &tasks, &in_h, &out_h, 0)?;
                    prof.mark(b'u');
                    shots.out_all.prep(wait_ns())?;
                    prof.mark(b'w');
                    {
                        let words = f32_words_ref(&shots.out_all);
                        for (j, &(mi, ni)) in meta.iter().enumerate() {
                            let slot = &words[j * plan.out_slot..(j + 1) * plan.out_slot];
                            layout::detile_accum_tf32(
                                acc,
                                plan.n,
                                slot,
                                mi * plan.mt,
                                ni * plan.nt,
                                plan.m_tile(mi),
                                plan.n_tile(ni),
                            );
                        }
                    }
                    shots.out_all.fini()?;
                    prof.mark(b'r');
                    nb = 0;
                    tasks.clear();
                    meta.clear();
                }
            }
        }
    }
    prof.dump(plan);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::plan::rup;
    use super::*;

    #[test]
    fn the_tf32_plan_uses_the_4_byte_cube_slots() {
        // M=264 > max_tile keeps nMt > 1, so the slot math is exercised across several
        // tiles: cube K group 16 (not the other dtypes' 32), N group 16, fp32 slots.
        let (mt, kt, nt) = plan_tf32(264, 320, 320).unwrap();
        let p = tf32_plan(264, 320, 320, mt, kt, nt);
        assert!(
            p.nmt > 1,
            "the ragged-M assertion must not be vacuous: {p:?}"
        );
        assert_eq!(p.mt % 4, 0);
        assert_eq!(p.kt % 16, 0);
        assert_eq!(p.nt % 16, 0);
        assert_eq!(p.nmt, 264usize.div_ceil(mt));
        assert_eq!(p.in_slot, rup(mt, 4) * rup(kt, 16));
        assert_eq!(p.wt_slot, rup(nt, 16) * rup(kt, 16));
        assert_eq!(p.out_slot, rup(mt, 4) * rup(nt, 16));
        assert_eq!(p.jobs(), p.nmt * p.nnt * p.nkt);
        // The cube K granule is 16, so a K%32 tile is NOT padded to the next 32.
        let (mt, kt, nt) = plan_tf32(8, 48, 32).unwrap();
        assert_eq!(kt, 48);
        let p = tf32_plan(8, 48, 32, mt, kt, nt);
        assert_eq!(p.in_slot, rup(mt, 4) * 48);
        assert_eq!(p.wt_slot, rup(nt, 16) * 48);
    }

    #[test]
    fn the_tf32_planner_rejects_the_shapes_the_c_does() {
        assert!(plan_tf32(6, 64, 16).is_err()); // M % 4
        assert!(plan_tf32(4, 40, 16).is_err()); // K % 16 (legal for nothing here)
        assert!(plan_tf32(4, 48, 8).is_err()); // N % 16
        assert!(plan_tf32(4, 48, 16).is_ok()); // K % 16 but not % 32
    }
}
