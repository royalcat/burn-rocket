//! Dense (per-channel) int8 and the one-shot group-wise sibling.
//!
//! Port of `rocket_matmul_int8` / `rocket_matmul_int8_groupwise` in
//! `rocket_matmul.c`: one fd, a transient BO set (in/wt 1 B per element, out 4 B per
//! int32), the int8 geometries of [`super::plan`], and
//! [`crate::regcmd::int8::emit`] for the program. Every tile's int32 partial comes
//! back to the host — the C library implements no NPU-side integer K-accumulation, so
//! neither does this.
//!
//! The two entry points differ only in the readback:
//!
//! - [`one_shot_int8`] sums the raw int32 partials in int64 (integer-exact) and writes
//!   a raw int32 `C[M,N]` — the C `rocket_matmul_int8`.
//! - [`one_shot_int8_groupwise`] keeps every K-tile inside one quant group and applies
//!   that group's `a_scale[m,g] * b_scale[n,g]` as its partial is read back, in f32 —
//!   the C `rocket_matmul_int8_groupwise`, and the one-shot twin of the resident
//!   [`super::int8::I8Context::matmul_gw`].
//!
//! The BO set ([`Shots`]) and the tile loop ([`run`]) are shared. Like the C, a call
//! allocates and drops its whole scratch; the resident forms ([`super::int8`],
//! `burn_rocket`'s `--npu-int8`) are what a serving process uses.

use crate::Device;
use crate::device::uapi::DrmRocketTask;
use crate::device::{Bo, Handle};
use crate::error::{Error, Result};
use crate::regcmd::desc::{BATCH, CBUF_BANK_SIZE, RC_STRIDE};
use crate::regcmd::fp16::MatmulParams;
use crate::regcmd::int8::emit;

use super::bos::{check_iova, i8_words, i32_words_ref, zero_bo};
use super::layout;
use super::plan::{INT8, INT8_GW, Plan, plan_geom};
use super::run::{wait_ns, write_regcmd};

/// The int8 planner: `(Mt, Kt, Nt)` for the dense (per-channel) geometry.
///
/// The C `rocket_matmul_plan_int8`; the group-wise planner is
/// `plan_geom(&INT8_GW, m, k, n, group)`.
pub fn plan_int8(m: usize, k: usize, n: usize) -> Result<(usize, usize, usize)> {
    plan_geom(&INT8, m, k, n, 0)
}

/// The int8 tiling plan for a resolved `(Mt, Kt, Nt)`.
///
/// [`Plan`]'s fp16 constructor rounds the weight slot's N to 16; int8's weight cube
/// groups N and K in 32s (its feature cube groups K in 16s) and its output cube is
/// int32 with C2=4, so the slot sizes come from the [`INT8`] geometry rather than
/// from fp16's. `mt` is the live tile height the caller resolved (`plan_geom`'s `Mt`,
/// or the canonical clamp of it).
pub(crate) fn int8_plan(m: usize, k: usize, n: usize, mt: usize, kt: usize, nt: usize) -> Plan {
    Plan::derive_geom(&INT8, m, k, n, mt, kt, nt)
}

/// The transient BO set one int8 one-shot call drives — the C locals in
/// `rocket_matmul_int8`: a 4 KiB guard (to push the real allocations off IOVA 0), the
/// regcmd batch, packed A, packed B and the int32 output cube.
///
/// `in_all`/`wt_all` are zeroed once at allocation: the scatters write live tile lanes
/// only, so the padding lanes past a ragged tile must already be zero. One `Shots` is
/// good for one packed A and one packed B, which is exactly one call's lifetime.
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
        let in_sz = plan.nmt * plan.nkt * plan.in_slot + CBUF_BANK_SIZE;
        let wt_sz = plan.nnt * plan.nkt * plan.wt_slot + CBUF_BANK_SIZE;
        let rc_sz = BATCH * RC_STRIDE * 8;
        let out_sz = BATCH * plan.out_slot * 4 + CBUF_BANK_SIZE;

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

    /// `A[M,K]` -> the `(K/16, M, 16)` int8 feature cube.
    fn pack_input(&mut self, plan: &Plan, a: &[i8]) -> Result<()> {
        assert_eq!(a.len(), plan.m * plan.k);
        self.in_all.prep(0)?;
        {
            let words = i8_words(&mut self.in_all);
            for mi in 0..plan.nmt {
                let m0 = mi * plan.mt;
                let mtile = plan.m_tile(mi);
                for ki in 0..plan.nkt {
                    let k0 = ki * plan.kt;
                    let ktile = plan.k_tile(ki);
                    let off = (mi * plan.nkt + ki) * plan.in_slot;
                    layout::feat_scatter_i8(
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

    /// `B[N,K]` -> the `(N/32, K/32, 32, 32)` int8 weight cube.
    fn pack_weights(&mut self, plan: &Plan, b: &[i8]) -> Result<()> {
        assert_eq!(b.len(), plan.n * plan.k);
        self.wt_all.prep(0)?;
        {
            let words = i8_words(&mut self.wt_all);
            for ni in 0..plan.nnt {
                let n0 = ni * plan.nt;
                let ntile = plan.n_tile(ni);
                for ki in 0..plan.nkt {
                    let k0 = ki * plan.kt;
                    let ktile = plan.k_tile(ki);
                    let off = (ni * plan.nkt + ki) * plan.wt_slot;
                    layout::wt_scatter_tile_i8(
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

/// How a tile's int32 partial is folded into the host accumulator.
enum Readback<'a> {
    /// Dense: exact int64 sum of the raw int32 partials (the C `acc`), then narrowed.
    Int64(&'a mut [i64]),
    /// Group-wise: `f32 += a_scale[m,g] * b_scale[n,g] * partial`. `ng = K/group`.
    Scaled {
        acc: &'a mut [f32],
        a_scale: &'a [f32],
        b_scale: &'a [f32],
        ng: usize,
    },
}

/// Submit every tile in groups of [`BATCH`] and accumulate the int32 partials.
///
/// `kt_per_group` is the number of K-tiles inside one quant group (0 for dense); the
/// group-wise planner guarantees `kt | group`, so a tile never straddles two groups.
fn run(plan: &Plan, shots: &mut Shots, kt_per_group: usize, rb: &mut Readback<'_>) -> Result<()> {
    let total = plan.jobs();
    let mut tasks: Vec<DrmRocketTask> = Vec::with_capacity(BATCH);
    let mut meta: Vec<(usize, usize, usize, usize, usize)> = Vec::with_capacity(BATCH);
    let mut ops = Vec::with_capacity(160);
    let mut done = 0usize;
    let mut nb = 0usize;

    for mi in 0..plan.nmt {
        let m0 = mi * plan.mt;
        let mtile = plan.m_tile(mi);
        for ni in 0..plan.nnt {
            let n0 = ni * plan.nt;
            let ntile = plan.n_tile(ni);
            for ki in 0..plan.nkt {
                if nb == 0 {
                    shots.regcmd.prep(0)?;
                }
                let ktile = plan.k_tile(ki);
                let out_off = nb * plan.out_slot;
                let input =
                    shots.in_all.dma_address() + ((mi * plan.nkt + ki) * plan.in_slot) as u64;
                let weights =
                    shots.wt_all.dma_address() + ((ni * plan.nkt + ki) * plan.wt_slot) as u64;
                let output = shots.out_all.dma_address() + (out_off * 4) as u64;
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
                let regcmd = write_regcmd(&mut shots.regcmd, nb, &ops)?;
                tasks.push(DrmRocketTask {
                    regcmd,
                    regcmd_count: count as u32,
                });
                meta.push((m0, n0, mtile, ntile, ki / kt_per_group.max(1)));
                nb += 1;
                done += 1;

                if nb == BATCH || done == total {
                    shots.regcmd.fini()?;
                    shots.out_all.prep(0)?;
                    shots.out_all.fini()?;
                    let in_h: [Handle; 3] = [
                        shots.in_all.handle(),
                        shots.wt_all.handle(),
                        shots.regcmd.handle(),
                    ];
                    let out_h = [shots.out_all.handle()];
                    shots.dev.submit(&tasks, &in_h, &out_h, 0)?;
                    shots.out_all.prep(wait_ns())?;
                    {
                        let words = i32_words_ref(&shots.out_all);
                        for (j, &(m0, n0, mtile, ntile, g)) in meta.iter().enumerate() {
                            let slot = &words[j * plan.out_slot..(j + 1) * plan.out_slot];
                            match rb {
                                Readback::Int64(acc) => layout::detile_accum_i32(
                                    acc, plan.n, slot, m0, n0, mtile, ntile,
                                ),
                                Readback::Scaled {
                                    acc,
                                    a_scale,
                                    b_scale,
                                    ng,
                                } => layout::detile_accum_i8_scaled(
                                    acc,
                                    plan.n,
                                    slot,
                                    m0,
                                    n0,
                                    mtile,
                                    ntile,
                                    |row, col| a_scale[row * *ng + g] * b_scale[col * *ng + g],
                                ),
                            }
                        }
                    }
                    shots.out_all.fini()?;
                    nb = 0;
                    tasks.clear();
                    meta.clear();
                }
            }
        }
    }
    Ok(())
}

/// `C[M,N] = A[M,K] * B[N,K]^T`, int8 x int8 -> raw int32 (the C `rocket_matmul_int8`).
///
/// `M == 1` is padded to a 4-row tile and row 0 is returned: the height-1 GEMV
/// geometry miscomputes on HW, which is why every one-shot path in the C library pads
/// it. Other ragged `M` are rejected by the planner (`M % 4`), as in C; `K % 32` and
/// `N % 32` likewise.
pub fn one_shot_int8(
    dev: &Device,
    m: usize,
    k: usize,
    n: usize,
    a: &[i8],
    b: &[i8],
    c: &mut [i32],
) -> Result<()> {
    if m == 1 {
        if a.len() < k || b.len() < n * k || c.len() < n {
            return Err(Error::shape("int8/buffers"));
        }
        let mut ap = vec![0i8; 4 * k];
        ap[..k].copy_from_slice(a);
        let mut cp = vec![0i32; 4 * n];
        one_shot_int8(dev, 4, k, n, &ap, b, &mut cp)?;
        c[..n].copy_from_slice(&cp[..n]);
        return Ok(());
    }
    if a.len() < m * k || b.len() < n * k || c.len() < m * n {
        return Err(Error::shape("int8/buffers"));
    }
    let (mt, kt, nt) = plan_int8(m, k, n)?;
    let plan = int8_plan(m, k, n, mt, kt, nt);
    let mut shots = Shots::new(dev, &plan)?;
    shots.pack_input(&plan, a)?;
    shots.pack_weights(&plan, b)?;

    let mut acc = vec![0i64; m * n];
    run(&plan, &mut shots, 0, &mut Readback::Int64(&mut acc))?;
    for (d, s) in c[..m * n].iter_mut().zip(&acc) {
        *d = *s as i32;
    }
    Ok(())
}

/// `Cf[M,N] = sum_g a_scale[m,g] * b_scale[n,g] * (A_g * B_g^T)`, f32 output (the C
/// `rocket_matmul_int8_groupwise`).
///
/// `a_scale` is `[M, K/group]` and `b_scale` `[N, K/group]`; A and B are pre-quantized
/// int8. There is no `M == 1` pad here (a padded row would need a padded `a_scale`;
/// single vectors are padded caller-side), `group % 32 == 0` and `group | K`.
///
/// Bit-identical to the C one-shot for the same plan, and to
/// [`super::int8::I8Context::matmul_gw`] whenever both resolve the same `Kt`: the
/// resident path plans at the canonical tile height (`max_tile`), the one-shot at the
/// call's `M`, and `Kt` is what sets the number of readbacks — hence the f32
/// accumulation order. For the usual groups (`Kt` pinned by the group, e.g. 32) the two
/// agree exactly; a group wider than the CBUF's Kt cap can resolve a different `Kt` per
/// path and then differ by an ULP or two in f32, exactly as the two C entry points do.
#[allow(clippy::too_many_arguments)]
pub fn one_shot_int8_groupwise(
    dev: &Device,
    m: usize,
    k: usize,
    n: usize,
    a: &[i8],
    b: &[i8],
    a_scale: &[f32],
    b_scale: &[f32],
    cf: &mut [f32],
    group: usize,
) -> Result<()> {
    if group == 0 {
        return Err(Error::shape("int8_gw/group"));
    }
    let (mt, kt, nt) = plan_geom(&INT8_GW, m, k, n, group)?;
    if !group.is_multiple_of(kt) {
        return Err(Error::shape("int8_gw/group-kt"));
    }
    let kt_per_group = group / kt;
    if a.len() < m * k || b.len() < n * k || cf.len() < m * n {
        return Err(Error::shape("int8_gw/buffers"));
    }
    let ng = k / group;
    if a_scale.len() < m * ng || b_scale.len() < n * ng {
        return Err(Error::shape("int8_gw/scales"));
    }
    let plan = int8_plan(m, k, n, mt, kt, nt);
    let mut shots = Shots::new(dev, &plan)?;
    shots.pack_input(&plan, a)?;
    shots.pack_weights(&plan, b)?;

    let mut facc = vec![0f32; m * n];
    run(
        &plan,
        &mut shots,
        kt_per_group,
        &mut Readback::Scaled {
            acc: &mut facc,
            a_scale,
            b_scale,
            ng,
        },
    )?;
    cf[..m * n].copy_from_slice(&facc);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::plan::rup;
    use super::*;

    #[test]
    fn the_int8_plan_uses_the_int8_cube_slots() {
        // M=264 > max_tile keeps nMt > 1, so the slot math is exercised across several
        // tiles: weight N-group 32 (not fp16's 16), int32 out C2=4.
        let (mt, kt, nt) = plan_int8(264, 320, 288).unwrap();
        let p = int8_plan(264, 320, 288, mt, kt, nt);
        assert!(
            p.nmt > 1,
            "the ragged-M assertion must not be vacuous: {p:?}"
        );
        assert_eq!(p.mt % 4, 0);
        assert_eq!(p.kt % 32, 0);
        assert_eq!(p.nt % 32, 0);
        assert_eq!(p.nmt, 264usize.div_ceil(mt));
        assert_eq!(p.in_slot, rup(mt, 4) * rup(kt, 32));
        assert_eq!(p.wt_slot, rup(nt, 32) * rup(kt, 32));
        assert_eq!(p.out_slot, rup(mt, 4) * rup(nt, 16));
        assert_eq!(p.jobs(), p.nmt * p.nnt * p.nkt);
        // int8's weight cube groups N/K in 32s (fp16 rounds N to 16), which is what
        // this constructor derives instead of leaning on `Plan::derive`.
        assert_eq!(p.wt_slot, rup(p.nt, 32) * rup(p.kt, 32));
    }

    #[test]
    fn the_dense_planner_rejects_the_shapes_the_c_does() {
        assert!(plan_int8(6, 64, 32).is_err()); // M % 4
        assert!(plan_int8(4, 48, 32).is_err()); // K % 32
        assert!(plan_int8(4, 64, 16).is_err()); // N % 32 (fp16's 16 is not enough)
        assert!(plan_int8(4, 64, 32).is_ok());
    }

    #[test]
    fn the_group_wise_planner_keeps_kt_inside_a_group() {
        for group in [32usize, 64, 128, 256, 512] {
            let (_, kt, _) = plan_geom(&INT8_GW, 64, 1024, 768, group).unwrap();
            assert!(group.is_multiple_of(kt), "group={group} kt={kt}");
        }
    }
}
