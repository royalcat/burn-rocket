//! The buffer-object set one matmul plan drives, and the host packing passes.

use half::f16;

use crate::device::Bo;
use crate::error::{Error, Result};

use super::layout;
use super::plan::Plan;
use crate::Device;
use crate::regcmd::desc::{BATCH, CBUF_BANK_SIZE, RC_STRIDE};

/// Resolved compute-path choices, fixed when the scratch is allocated (so the
/// dispatcher can never pick a path whose BOs were not provisioned).
#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub kacc: bool,
    pub pipe: bool,
}

impl Config {
    pub fn from_env() -> Config {
        let kacc = match crate::knobs::int("ROCKET_KACC") {
            Some(v) => v != 0,
            None => !crate::knobs::present("ROCKET_NO_KACC"),
        };
        Config {
            kacc,
            pipe: crate::knobs::present("ROCKET_MM_PIPE"),
        }
    }
}

/// All BOs of one tiled matmul.
pub struct Bos {
    pub dev: Device,
    pub guard: Bo,
    pub regcmd: Bo,
    pub in_all: Bo,
    pub wt_all: Bo,
    pub out_all: Bo,
    pub okacc0: Option<Bo>,
    pub pong: Option<Bo>,
    pub prezeroed: bool,
    pub has_pong: bool,
}

impl Bos {
    pub fn new(dev: &Device, plan: &Plan, config: Config) -> Result<Bos> {
        let in_sz = plan.nmt * plan.nkt * plan.in_slot * 2 + CBUF_BANK_SIZE;
        let wt_sz = plan.nnt * plan.nkt * plan.wt_slot * 2 + CBUF_BANK_SIZE;
        let rc_sz = BATCH * RC_STRIDE * 8;
        let out_sz = BATCH * plan.out_slot * 2 + CBUF_BANK_SIZE;

        let guard = dev.alloc_bo(4096)?;
        let regcmd = dev.alloc_bo(rc_sz)?;
        let in_all = dev.alloc_bo(in_sz)?;
        let wt_all = dev.alloc_bo(wt_sz)?;
        let out_all = dev.alloc_bo(out_sz)?;
        check_iova(&[&in_all, &wt_all, &out_all, &regcmd])?;

        let has_pong = config.kacc;
        let (okacc0, pong) = if has_pong {
            let ktiles = (plan.nmt * plan.nnt).clamp(1, BATCH);
            let sz = ktiles * plan.out_slot * 2 + CBUF_BANK_SIZE;
            let a = dev.alloc_bo(sz)?;
            let b = dev.alloc_bo(sz)?;
            check_iova(&[&a, &b])?;
            (Some(a), Some(b))
        } else {
            (None, None)
        };

        let mut bos = Bos {
            dev: dev.clone(),
            guard,
            regcmd,
            in_all,
            wt_all,
            out_all,
            okacc0,
            pong,
            prezeroed: false,
            has_pong,
        };
        zero_bo(&mut bos.in_all)?;
        zero_bo(&mut bos.wt_all)?;
        bos.prezeroed = true;
        Ok(bos)
    }

    pub fn pack_input(&mut self, plan: &Plan, a: &[f16]) -> Result<()> {
        pack_input_into(&mut self.in_all, plan, a)
    }

    pub fn load_input(&mut self, plan: &Plan, packed: &[f16]) -> Result<()> {
        load_input_into(&mut self.in_all, plan, packed)
    }

    pub fn pack_weights(&mut self, plan: &Plan, b: &[f16]) -> Result<()> {
        pack_weights_into(&mut self.wt_all, plan, b)
    }

    pub fn pack_weights_seg(
        &mut self,
        plan: &Plan,
        segs: &[(usize, usize, &[f16])],
        worker_g0: usize,
    ) -> Result<()> {
        pack_weights_seg_into(&mut self.wt_all, plan, segs, worker_g0)
    }
}

fn f16_words(bo: &mut Bo) -> &mut [f16] {
    let bytes = bo.as_mut_slice();
    assert_eq!(bytes.len() % 2, 0);
    // SAFETY: `f16` is a 2-byte plain-old-data type, the BO mapping is page-aligned,
    // and the length is even.
    unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr().cast::<f16>(), bytes.len() / 2) }
}

fn f16_words_ref(bo: &Bo) -> &[f16] {
    let bytes = bo.as_slice();
    assert_eq!(bytes.len() % 2, 0);
    // SAFETY: see f16_words.
    unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<f16>(), bytes.len() / 2) }
}

pub fn zero_bo(bo: &mut Bo) -> Result<()> {
    bo.prep(0)?;
    bo.as_mut_slice().fill(0);
    bo.fini()
}

/// Scatter A into `[M,K]`'s canonical cube slots. Padding lanes must already be zero.
pub fn scatter_input(plan: &Plan, dst: &mut [f16], a: &[f16]) {
    assert_eq!(a.len(), plan.m * plan.k);
    assert!(dst.len() >= plan.input_elems());
    for mi in 0..plan.nmt {
        let m0 = mi * plan.mt;
        let mtile = plan.m_tile(mi);
        for ki in 0..plan.nkt {
            let k0 = ki * plan.kt;
            let ktile = plan.k_tile(ki);
            let off = (mi * plan.nkt + ki) * plan.in_slot;
            layout::feat_scatter_tile(
                &mut dst[off..off + plan.in_slot],
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

/// Scatter A into a BO's cube slots. Padding lanes must already be zero.
pub fn pack_input_into(bo: &mut Bo, plan: &Plan, a: &[f16]) -> Result<()> {
    assert_eq!(a.len(), plan.m * plan.k);
    bo.prep(0)?;
    scatter_input(plan, f16_words(bo), a);
    bo.fini()
}

/// Copy a pre-scattered canonical A buffer into the input BO.
pub fn load_input_into(bo: &mut Bo, plan: &Plan, packed: &[f16]) -> Result<()> {
    let elems = plan.input_elems();
    assert!(packed.len() >= elems);
    bo.prep(0)?;
    {
        let words = f16_words(bo);
        words[..elems].copy_from_slice(&packed[..elems]);
    }
    bo.fini()
}

/// Scatter a whole `B[N,K]` into `bo`'s weight slots. Padding lanes must be zero.
pub fn pack_weights_into(bo: &mut Bo, plan: &Plan, b: &[f16]) -> Result<()> {
    assert_eq!(b.len(), plan.n * plan.k);
    bo.prep(0)?;
    {
        let words = f16_words(bo);
        for ni in 0..plan.nnt {
            let n0 = ni * plan.nt;
            let ntile = plan.n_tile(ni);
            for ki in 0..plan.nkt {
                let k0 = ki * plan.kt;
                let ktile = plan.k_tile(ki);
                let off = (ni * plan.nkt + ki) * plan.wt_slot;
                layout::wt_scatter_tile(
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
    bo.fini()
}

/// Scatter the worker's column slice of a concatenated-along-N weight.
pub fn pack_weights_seg_into(
    bo: &mut Bo,
    plan: &Plan,
    segs: &[(usize, usize, &[f16])],
    worker_g0: usize,
) -> Result<()> {
    bo.prep(0)?;
    {
        let words = f16_words(bo);
        for ni in 0..plan.nnt {
            let n0 = ni * plan.nt;
            let ntile = plan.n_tile(ni);
            for ki in 0..plan.nkt {
                let k0 = ki * plan.kt;
                let ktile = plan.k_tile(ki);
                let off = (ni * plan.nkt + ki) * plan.wt_slot;
                let slot = &mut words[off..off + plan.wt_slot];
                for kk in 0..ntile {
                    let g = worker_g0 + n0 + kk;
                    let Some(&(g0, nseg, src)) = segs
                        .iter()
                        .find(|(g0, nseg, _)| g >= *g0 && g < *g0 + *nseg)
                    else {
                        continue;
                    };
                    let _ = nseg;
                    let brow = &src[(g - g0) * plan.k + k0..][..ktile];
                    layout::wt_scatter_row(slot, brow, kk, ktile);
                }
            }
        }
    }
    bo.fini()
}

/// Scatter A given as K planes of M rows (plane k at `at + k*lda`).
pub fn pack_input_planes_into(
    bo: &mut Bo,
    plan: &Plan,
    at: &[f16],
    lda: usize,
    m_live: usize,
    k_live: usize,
) -> Result<()> {
    bo.prep(0)?;
    {
        let words = f16_words(bo);
        for mi in 0..plan.nmt {
            let m0 = mi * plan.mt;
            let rows = m_live.saturating_sub(m0).min(plan.m_tile(mi));
            for ki in 0..plan.nkt {
                let k0 = ki * plan.kt;
                let ktile = plan.k_tile(ki);
                let k_live = k_live.min(plan.k).saturating_sub(k0).min(ktile);
                let off = (mi * plan.nkt + ki) * plan.in_slot;
                let slot = &mut words[off..off + plan.in_slot];
                slot.fill(f16::ZERO);
                let h = plan.m_tile(mi);
                for row in 0..rows {
                    for col in 0..k_live {
                        let src = at[(k0 + col) * lda + m0 + row];
                        slot[layout::feat_idx(h, col + 1, row + 1)] = src;
                    }
                }
            }
        }
    }
    bo.fini()
}

/// Borrow the packed input region (for diagnostics/tests).
pub fn input_slice<'a>(bo: &'a Bo, plan: &Plan) -> &'a [f16] {
    &f16_words_ref(bo)[..plan.input_elems()]
}

fn check_iova(bos: &[&Bo]) -> Result<()> {
    for bo in bos {
        let end = bo.dma_address() + bo.len() as u64;
        if end > (1u64 << 32) {
            return Err(Error::device("alloc_bo/iova", 0));
        }
    }
    Ok(())
}
