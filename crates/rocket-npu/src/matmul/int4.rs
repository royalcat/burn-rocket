//! Dense int4: the planner, the tiling plan and the one-shot matmul.
//!
//! Port of `rocket_matmul_plan_int4` / `rocket_matmul_int4_ex` in `rocket_matmul.c`:
//! one fd, a transient BO set (in/wt **nibble-packed**, so half a byte per element,
//! out 2 B per int16), the int4 geometry of [`super::plan`] and
//! [`crate::regcmd::int4::emit`] for the program.
//!
//! Three things make this path unlike int8's:
//!
//! - A and B are pre-quantized int4 codes **one per `i8` in `[-8,7]`** and get packed
//!   two-to-a-byte by [`layout::feat_scatter_i4`] / [`layout::wt_scatter_tile_i4`].
//! - The conv writes **int16**, not int32 (`out_precision = 1`, C2=8), so a K-tile
//!   partial saturates above 32767 and the host sums partials in int64. `[-8,7]`
//!   operands bound `|partial| <= 64*Kt`, which is why the C's in-model callers cap
//!   `Kt` at 480; [`one_shot_int4_ex`] exposes that cap, [`one_shot_int4`] does not
//!   (the C `rocket_matmul_int4`, valid only for small-magnitude data).
//! - There is no NPU-side integer K-accumulation (the C leaves the int4 DPU-EW path
//!   unimplemented), so every tile's partial comes back to the host, as for int8.
//!
//! Two entry points differ only in the readback:
//!
//! - [`one_shot_int4`] / [`one_shot_int4_ex`] sum the raw int16 partials in int64
//!   (integer-exact up to the int16 saturation the hardware applies per tile) and
//!   write a raw int32 `C[M,N]` — the C `rocket_matmul_int4`.
//! - [`one_shot_int4_groupwise`] forces `Kt = group`, so every K-tile stays inside one
//!   quant group, and applies that group's `a_scale[m,g] * b_scale[n,g]` to its
//!   partial in f32 — the C `rocket_matmul_int4_groupwise`, and the one-shot twin of
//!   the resident [`super::int4_context::I4Context::matmul_gw`].

use crate::Device;
use crate::device::uapi::DrmRocketTask;
use crate::device::{Bo, Handle};
use crate::error::{Error, Result};
use crate::regcmd::desc::{BATCH, CBUF_BANK_SIZE, RC_STRIDE};
use crate::regcmd::fp16::MatmulParams;
use crate::regcmd::int4::emit;

use super::bos::{check_iova, i16_words_ref, zero_bo};
use super::layout;
use super::plan::{INT4, Plan, plan_geom};
use super::run::{wait_ns, write_regcmd};

/// The int4 planner: `(Mt, Kt, Nt)` for the dense int4 geometry (`N % 64`, `K % 32`,
/// `M % 4`; half a byte per element, no bank slack).
///
/// The C `rocket_matmul_plan_int4`. There is no group argument: the group-wise int4
/// geometry is M3b.
pub fn plan_int4(m: usize, k: usize, n: usize) -> Result<(usize, usize, usize)> {
    plan_geom(&INT4, m, k, n, 0)
}

/// The int4 tiling plan for a resolved `(Mt, Kt, Nt)`.
///
/// The int4 slots are the int8 ones except for the weight cube's N group (64, not 32):
/// `in_slot`/`wt_slot` count **nibbles** and `out_slot` int16 elements, exactly as the
/// C's `(size_t)rup(...) * rup(...)` locals, so the caller halves/doubles for bytes.
/// `mt`/`kt`/`nt` are the tile dims the caller resolved (after the `kt_cap` override,
/// which changes `Kt` and therefore `nkt`).
pub(crate) fn int4_plan(m: usize, k: usize, n: usize, mt: usize, kt: usize, nt: usize) -> Plan {
    Plan::derive_geom(&INT4, m, k, n, mt, kt, nt)
}

/// The transient BO set one int4 one-shot call drives — the C locals in
/// `rocket_matmul_int4_ex`: a 4 KiB guard (to push the real allocations off IOVA 0),
/// the regcmd batch, packed A and packed B (nibble halves) and the int16 output cube.
///
/// `in_all`/`wt_all` are zeroed once at allocation: the scatters write live tile
/// nibbles only, so the padding lanes past a ragged tile must already be zero.
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
        // Nibbles -> bytes: both slots are whole numbers of bytes (their K groups are
        // 32 nibbles = 16 bytes and their N groups 4 or 64).
        let in_sz = plan.nmt * plan.nkt * (plan.in_slot / 2) + CBUF_BANK_SIZE;
        let wt_sz = plan.nnt * plan.nkt * (plan.wt_slot / 2) + CBUF_BANK_SIZE;
        let rc_sz = BATCH * RC_STRIDE * 8;
        let out_sz = BATCH * plan.out_slot * 2 + CBUF_BANK_SIZE;

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

    /// `A[M,K]` (int4 code per `i8`) -> the `(K/32, M, 32)` nibble feature cube.
    fn pack_input(&mut self, plan: &Plan, a: &[i8]) -> Result<()> {
        assert_eq!(a.len(), plan.m * plan.k);
        let slot_bytes = plan.in_slot / 2;
        self.in_all.prep(0)?;
        {
            let words = self.in_all.as_mut_slice();
            for mi in 0..plan.nmt {
                let m0 = mi * plan.mt;
                let mtile = plan.m_tile(mi);
                for ki in 0..plan.nkt {
                    let k0 = ki * plan.kt;
                    let ktile = plan.k_tile(ki);
                    let off = (mi * plan.nkt + ki) * slot_bytes;
                    layout::feat_scatter_i4(
                        &mut words[off..off + slot_bytes],
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

    /// `B[N,K]` -> the `(N/64, K/32, 64, 32)` nibble weight cube.
    fn pack_weights(&mut self, plan: &Plan, b: &[i8]) -> Result<()> {
        assert_eq!(b.len(), plan.n * plan.k);
        let slot_bytes = plan.wt_slot / 2;
        self.wt_all.prep(0)?;
        {
            let words = self.wt_all.as_mut_slice();
            for ni in 0..plan.nnt {
                let n0 = ni * plan.nt;
                let ntile = plan.n_tile(ni);
                for ki in 0..plan.nkt {
                    let k0 = ki * plan.kt;
                    let ktile = plan.k_tile(ki);
                    let off = (ni * plan.nkt + ki) * slot_bytes;
                    layout::wt_scatter_tile_i4(
                        &mut words[off..off + slot_bytes],
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

/// How a tile's int16 partial is folded into the host accumulator.
enum Readback<'a> {
    /// Dense: exact int64 sum of the raw int16 partials (the C `acc`), then narrowed.
    Int64(&'a mut [i64]),
    /// Group-wise: `f32 += a_scale[m,g] * b_scale[n,g] * partial`. `ng = K/group`.
    Scaled {
        acc: &'a mut [f32],
        a_scale: &'a [f32],
        b_scale: &'a [f32],
        ng: usize,
    },
}

/// Submit every tile in groups of [`BATCH`] and accumulate the partials.
///
/// `kt_per_group` is the number of K-tiles inside one quant group (0 for the dense
/// path, 1 for group-wise, which forces one tile per group).
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
                    shots.in_all.dma_address() + ((mi * plan.nkt + ki) * (plan.in_slot / 2)) as u64;
                let weights =
                    shots.wt_all.dma_address() + ((ni * plan.nkt + ki) * (plan.wt_slot / 2)) as u64;
                let output = shots.out_all.dma_address() + (out_off * 2) as u64;
                // Neither `fp32_to_fp16` nor `accumulate` is read: the int4 generator
                // hardcodes the int16 output writer and the plain-conv path, as in C.
                let p = MatmulParams::new(
                    mtile as u32,
                    ktile as u32,
                    ntile as u32,
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
                        let words = i16_words_ref(&shots.out_all);
                        for (j, &(m0, n0, mtile, ntile, g)) in meta.iter().enumerate() {
                            let slot = &words[j * plan.out_slot..(j + 1) * plan.out_slot];
                            match rb {
                                Readback::Int64(acc) => layout::detile_accum_i16(
                                    acc, plan.n, slot, m0, n0, mtile, ntile,
                                ),
                                Readback::Scaled {
                                    acc,
                                    a_scale,
                                    b_scale,
                                    ng,
                                } => layout::detile_accum_i16_scaled(
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

/// `C[M,N] = A[M,K] * B[N,K]^T`, int4 x int4 -> raw int32 (the C `rocket_matmul_int4`).
///
/// The operands are pre-quantized int4 codes, one per `i8` in `[-8,7]`. There is no
/// `Kt` cap, so a K this large makes the int16 tile partials saturate: the C recommends
/// `kt_cap = 480` for full-range data and uses the uncapped entry only for
/// small-magnitude inputs. Use [`one_shot_int4_ex`] when that matters.
///
/// `M == 1` is padded to a 4-row tile and row 0 is returned: the height-1 GEMV
/// geometry miscomputes on HW, which is why every one-shot path in the C library pads
/// it. Other ragged `M` are rejected by the planner (`M % 4`), as in C; `K % 32` and
/// `N % 64` likewise.
pub fn one_shot_int4(
    dev: &Device,
    m: usize,
    k: usize,
    n: usize,
    a: &[i8],
    b: &[i8],
    c: &mut [i32],
) -> Result<()> {
    one_shot_int4_ex(dev, m, k, n, a, b, c, 0)
}

/// [`one_shot_int4`] with the C's int16-saturation cap on `Kt` (the C
/// `rocket_matmul_int4_ex`).
///
/// `kt_cap >= 32` lowers a planned `Kt` above the cap to `(kt_cap/32)*32` (floor 32),
/// which bounds each tile partial to `64*kt_cap` and keeps it inside int16; `0` leaves
/// the plan alone. Unlike the C, the cap also applies to the `M == 1` pad path (the C's
/// `MM_PAD_M1` macro re-enters the *uncapped* entry point and silently drops it, which
/// would return a saturated row for a large K).
#[allow(clippy::too_many_arguments)]
pub fn one_shot_int4_ex(
    dev: &Device,
    m: usize,
    k: usize,
    n: usize,
    a: &[i8],
    b: &[i8],
    c: &mut [i32],
    kt_cap: usize,
) -> Result<()> {
    if m == 1 {
        if a.len() < k || b.len() < n * k || c.len() < n {
            return Err(Error::shape("int4/buffers"));
        }
        let mut ap = vec![0i8; 4 * k];
        ap[..k].copy_from_slice(&a[..k]);
        let mut cp = vec![0i32; 4 * n];
        one_shot_int4_ex(dev, 4, k, n, &ap, b, &mut cp, kt_cap)?;
        c[..n].copy_from_slice(&cp[..n]);
        return Ok(());
    }
    if a.len() < m * k || b.len() < n * k || c.len() < m * n {
        return Err(Error::shape("int4/buffers"));
    }
    let (mt, mut kt, nt) = plan_int4(m, k, n)?;
    if kt_cap >= 32 && kt > kt_cap {
        kt = (kt_cap / 32) * 32;
        if kt < 32 {
            kt = 32;
        }
    }
    let plan = int4_plan(m, k, n, mt, kt, nt);
    let mut shots = Shots::new(dev, &plan)?;
    shots.pack_weights(&plan, b)?;
    shots.pack_input(&plan, a)?;

    let mut acc = vec![0i64; m * n];
    run(&plan, &mut shots, 0, &mut Readback::Int64(&mut acc))?;
    for (d, s) in c[..m * n].iter_mut().zip(&acc) {
        *d = *s as i32;
    }
    Ok(())
}

/// The group-wise int4 group contract: `group >= 32`, `group % 32 == 0`, `group | K`
/// and `49*group < 32767`.
///
/// The last bound is the C's int16-saturation guard: with `[-7,7]` codes a K-tile of
/// `group` elements bounds `|partial| <= 49*group`, so the tile partial the hardware
/// writes as int16 cannot saturate. `group | K` additionally makes every K-tile exactly
/// one group, which is what lets the readback apply a single scale per tile. Shared by
/// [`one_shot_int4_groupwise`] and [`super::int4_context::I4Context`].
pub(crate) fn check_group(group: usize, k: usize) -> Result<()> {
    if group < 32
        || !group.is_multiple_of(32)
        || !k.is_multiple_of(group)
        || 49 * group as u64 >= 32767
    {
        return Err(Error::shape("int4_gw/group"));
    }
    Ok(())
}

/// `Cf[M,N] = sum_g a_scale[m,g] * b_scale[n,g] * (A_g * B_g^T)`, f32 output (the C
/// `rocket_matmul_int4_groupwise`).
///
/// The W4 quality lever over the per-channel scales of [`one_shot_int4`]: one dequant
/// scale per K-group bounds the int4 error to a narrow slice of K. A and B are
/// pre-quantized int4 codes (one per `i8`), `a_scale` is `[M, K/group]` and `b_scale`
/// `[N, K/group]` row-major. The tiling is the dense int4 plan with `Kt` forced to
/// `group`, so each tile IS one quant group and its int16 partial is scaled on
/// readback and accumulated in f32.
///
/// There is no `M == 1` pad here (a padded row would need a padded `a_scale`; a single
/// vector is padded caller-side), so `M % 4 == 0` is required outright, as in C.
///
/// Bit-identical to [`super::int4_context::I4Context::matmul_gw`] for the same shape:
/// both pin `Kt` to `group` and int4's `Nt` is M-independent, so the two resolve the
/// same tiles and the same f32 accumulation order — unlike int8's group-wise pair,
/// where a group wider than the CBUF `Kt` can resolve a different `Kt` per path.
#[allow(clippy::too_many_arguments)]
pub fn one_shot_int4_groupwise(
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
    if m == 0 || !m.is_multiple_of(4) {
        return Err(Error::shape("int4_gw/m"));
    }
    check_group(group, k)?;
    if a.len() < m * k || b.len() < n * k || cf.len() < m * n {
        return Err(Error::shape("int4_gw/buffers"));
    }
    let ng = k / group;
    if a_scale.len() < m * ng || b_scale.len() < n * ng {
        return Err(Error::shape("int4_gw/scales"));
    }
    // Plan at the call's M and then override Kt, exactly as the C does: the group is
    // small enough (<= 640 by `check_group`) that the pair always fits the CBUF, and
    // `emit` rejects a tile that would not.
    let (mt, _kt, nt) = plan_int4(m, k, n)?;
    let plan = int4_plan(m, k, n, mt, group, nt);
    let mut shots = Shots::new(dev, &plan)?;
    shots.pack_weights(&plan, b)?;
    shots.pack_input(&plan, a)?;

    let mut facc = vec![0f32; m * n];
    run(
        &plan,
        &mut shots,
        1,
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
    fn the_int4_plan_uses_the_int4_cube_slots() {
        // M=264 > max_tile keeps nMt > 1, so the slot math is exercised across several
        // tiles: weight N group 64 (not int8's 32), nibble in/wt slots, int16 out C2=8.
        let (mt, kt, nt) = plan_int4(264, 320, 320).unwrap();
        let p = int4_plan(264, 320, 320, mt, kt, nt);
        assert!(
            p.nmt > 1,
            "the ragged-M assertion must not be vacuous: {p:?}"
        );
        assert_eq!(p.mt % 4, 0);
        assert_eq!(p.kt % 32, 0);
        assert_eq!(p.nt % 64, 0);
        assert_eq!(p.nmt, 264usize.div_ceil(mt));
        assert_eq!(p.in_slot, rup(mt, 4) * rup(kt, 32));
        assert_eq!(p.wt_slot, rup(nt, 64) * rup(kt, 32));
        assert_eq!(p.out_slot, rup(mt, 4) * rup(nt, 16));
        assert_eq!(p.jobs(), p.nmt * p.nnt * p.nkt);
        // Every slot is a whole number of bytes once halved (nibbles) — the BO sizes
        // and the scatter offsets both rely on that.
        assert_eq!(p.in_slot % 2, 0);
        assert_eq!(p.wt_slot % 2, 0);
    }

    #[test]
    fn the_int4_planner_rejects_the_shapes_the_c_does() {
        assert!(plan_int4(6, 64, 64).is_err()); // M % 4
        assert!(plan_int4(4, 48, 64).is_err()); // K % 32
        assert!(plan_int4(4, 64, 32).is_err()); // N % 64 (int8's 32 is not enough)
        assert!(plan_int4(4, 64, 64).is_ok());
    }

    #[test]
    fn the_group_contract_matches_the_c() {
        // `group >= 32`, `% 32`, `| K`, `49*group < 32767` (the C's int16-saturation
        // bound for [-7,7] codes). 640 is the widest group that passes; 672 does not.
        for (group, k) in [
            (32usize, 1280usize),
            (64, 1280),
            (128, 1280),
            (256, 1280),
            (512, 1024),
            (640, 1280),
        ] {
            assert!(check_group(group, k).is_ok(), "group={group} k={k}");
        }
        assert!(check_group(0, 1024).is_err()); // no group
        assert!(check_group(16, 1024).is_err()); // < 32
        assert!(check_group(48, 1024).is_err()); // not % 32
        assert!(check_group(64, 96).is_err()); // group does not divide K
        assert!(check_group(672, 1344).is_err()); // 49*672 = 32928 >= 32767
        assert!(check_group(640, 1280).is_ok());
    }

    #[test]
    fn the_group_wise_plan_is_one_k_tile_per_group() {
        let (m, k, n, group) = (264, 1024, 384, 128);
        let (mt, planned_kt, nt) = plan_int4(m, k, n).unwrap();
        let p = int4_plan(m, k, n, mt, group, nt);
        // The plan is the dense one with Kt overridden: Nt/Mt and the cube slots follow
        // the int4 geometry, and the K tiling is exactly the quant groups.
        assert_eq!(p.kt, group);
        assert_eq!(p.nkt, k / group);
        assert_ne!(planned_kt, group, "the override must actually bind here");
        assert_eq!(p.jobs(), p.nmt * p.nnt * p.nkt);
        assert_eq!(p.wt_slot, rup(nt, 64) * group);
        // One group's tile partial cannot saturate int16 for [-7,7] codes, which is
        // what `check_group` bounds (64*group would be the [-8,7] worst case).
        assert!(64 * group < 32767);
    }

    #[test]
    fn the_kt_cap_shrinks_kt_and_retiles_k() {
        // K=16384 plans Kt=11264 uncapped at (4, ·, 64); a 480 cap must land on 480 and
        // re-count the K tiles.
        let (mt, kt, nt) = plan_int4(4, 16384, 64).unwrap();
        assert!(kt > 480, "the cap must bind here: Kt={kt}");
        let capped = int4_plan(4, 16384, 64, mt, 480, nt);
        assert_eq!(capped.kt, 480);
        assert_eq!(capped.nkt, 16384usize.div_ceil(480));
        let full = int4_plan(4, 16384, 64, mt, kt, nt);
        assert_eq!(full.nkt, 2);
    }
}
