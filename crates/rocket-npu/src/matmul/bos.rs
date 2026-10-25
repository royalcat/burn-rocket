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

/// Bytes of the streaming-weight scratch for `plan`.
pub fn wt_bytes(plan: &Plan) -> usize {
    plan.nnt * plan.nkt * plan.wt_slot * 2 + CBUF_BANK_SIZE
}

/// Bytes of the CPU-accumulation output scratch for `plan`.
pub fn out_bytes(plan: &Plan) -> usize {
    BATCH * plan.out_slot * 2 + CBUF_BANK_SIZE
}

/// All BOs of one tiled matmul.
///
/// The streaming-weight (`wt_all`) and CPU-accumulation (`out_all`) scratch are
/// allocated lazily: resident-weight compute and the KACC path never touch them,
/// and a serving process caches many shapes whose unused N*K and BATCH-tile
/// buffers would otherwise add up to hundreds of MiB.
pub struct Bos {
    pub dev: Device,
    pub guard: Bo,
    pub regcmd: Bo,
    pub in_all: Bo,
    pub wt_all: Option<Bo>,
    pub out_all: Option<Bo>,
    pub okacc0: Option<Bo>,
    pub pong: Option<Bo>,
    pub prezeroed: bool,
    pub has_pong: bool,
    wt_sz: usize,
    out_sz: usize,
}

impl Bos {
    pub fn new(dev: &Device, plan: &Plan, config: Config) -> Result<Bos> {
        let in_sz = plan.nmt * plan.nkt * plan.in_slot * 2 + CBUF_BANK_SIZE;
        let wt_sz = wt_bytes(plan);
        let rc_sz = BATCH * RC_STRIDE * 8;
        let out_sz = out_bytes(plan);

        let guard = dev.alloc_bo(4096)?;
        let regcmd = dev.alloc_bo(rc_sz)?;
        let in_all = dev.alloc_bo(in_sz)?;
        check_iova(&[&in_all, &regcmd])?;

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
            wt_all: None,
            out_all: None,
            okacc0,
            pong,
            prezeroed: false,
            has_pong,
            wt_sz,
            out_sz,
        };
        zero_bo(&mut bos.in_all)?;
        bos.prezeroed = true;
        Ok(bos)
    }

    /// Allocate (and zero once) the streaming-weight scratch if it is absent.
    ///
    /// Padding lanes must be zero; the scatter only writes live tiles, so the
    /// zeroing happens on allocation rather than on every pack.
    pub fn ensure_wt(&mut self) -> Result<&mut Bo> {
        if self.wt_all.is_none() {
            let mut bo = self.dev.alloc_bo(self.wt_sz)?;
            check_iova(&[&bo])?;
            zero_bo(&mut bo)?;
            self.wt_all = Some(bo);
        }
        Ok(self.wt_all.as_mut().unwrap())
    }

    /// Allocate the CPU-accumulation output scratch if it is absent.
    pub fn ensure_out(&mut self) -> Result<&mut Bo> {
        if self.out_all.is_none() {
            let bo = self.dev.alloc_bo(self.out_sz)?;
            check_iova(&[&bo])?;
            self.out_all = Some(bo);
        }
        Ok(self.out_all.as_mut().unwrap())
    }

    pub fn wt(&self) -> &Bo {
        self.wt_all
            .as_ref()
            .expect("streaming-weight scratch was not allocated")
    }

    pub fn out(&self) -> &Bo {
        self.out_all
            .as_ref()
            .expect("CPU-accumulation output scratch was not allocated")
    }

    pub fn out_mut(&mut self) -> &mut Bo {
        self.out_all
            .as_mut()
            .expect("CPU-accumulation output scratch was not allocated")
    }

    /// Live bytes of this scratch set (host-mapped BOs only).
    pub fn bytes(&self) -> usize {
        let mut n = self.guard.len() + self.regcmd.len() + self.in_all.len();
        if let Some(b) = &self.wt_all {
            n += b.len();
        }
        if let Some(b) = &self.out_all {
            n += b.len();
        }
        if let Some(b) = &self.okacc0 {
            n += b.len();
        }
        if let Some(b) = &self.pong {
            n += b.len();
        }
        n
    }

    /// Bytes [`Bos::new`] would hold before the lazy `wt_all`/`out_all` are used.
    pub fn estimate_bytes(plan: &Plan, config: Config) -> usize {
        let mut n =
            4096 + BATCH * RC_STRIDE * 8 + plan.nmt * plan.nkt * plan.in_slot * 2 + CBUF_BANK_SIZE;
        if config.kacc {
            let ktiles = (plan.nmt * plan.nnt).clamp(1, BATCH);
            n += 2 * (ktiles * plan.out_slot * 2 + CBUF_BANK_SIZE);
        }
        n
    }

    pub fn pack_input(&mut self, plan: &Plan, a: &[f16]) -> Result<()> {
        pack_input_into(&mut self.in_all, plan, a)
    }

    pub fn load_input(&mut self, plan: &Plan, packed: &[f16]) -> Result<()> {
        load_input_into(&mut self.in_all, plan, packed)
    }

    pub fn pack_weights(&mut self, plan: &Plan, b: &[f16]) -> Result<()> {
        let bo = self.ensure_wt()?;
        pack_weights_into(bo, plan, b)
    }

    pub fn pack_weights_seg(
        &mut self,
        plan: &Plan,
        segs: &[(usize, usize, &[f16])],
        worker_g0: usize,
    ) -> Result<()> {
        let bo = self.ensure_wt()?;
        pack_weights_seg_into(bo, plan, segs, worker_g0)
    }
}

pub(crate) fn f16_words(bo: &mut Bo) -> &mut [f16] {
    let bytes = bo.as_mut_slice();
    assert_eq!(bytes.len() % 2, 0);
    // SAFETY: `f16` is a 2-byte plain-old-data type, the BO mapping is page-aligned,
    // and the length is even.
    unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr().cast::<f16>(), bytes.len() / 2) }
}

pub(crate) fn f16_words_ref(bo: &Bo) -> &[f16] {
    let bytes = bo.as_slice();
    assert_eq!(bytes.len() % 2, 0);
    // SAFETY: see f16_words.
    unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<f16>(), bytes.len() / 2) }
}

/// The BO as `u64` words, the form a register program is laid out in.
pub(crate) fn u64_words(bo: &mut Bo) -> &mut [u64] {
    let bytes = bo.as_mut_slice();
    assert_eq!(bytes.len() % 8, 0);
    // SAFETY: a BO mapping is page-aligned and the regcmd BO is a multiple of 8 bytes.
    unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr().cast::<u64>(), bytes.len() / 8) }
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

/// Scatter a whole `B[N,K]` into the weight slots of `dst`. Padding lanes must already
/// be zero: only the live tile lanes are written.
pub fn scatter_weights(plan: &Plan, dst: &mut [f16], b: &[f16]) {
    assert_eq!(b.len(), plan.n * plan.k);
    assert!(dst.len() >= plan.nnt * plan.nkt * plan.wt_slot);
    for ni in 0..plan.nnt {
        let n0 = ni * plan.nt;
        let ntile = plan.n_tile(ni);
        for ki in 0..plan.nkt {
            let k0 = ki * plan.kt;
            let ktile = plan.k_tile(ki);
            let off = (ni * plan.nkt + ki) * plan.wt_slot;
            layout::wt_scatter_tile(
                &mut dst[off..off + plan.wt_slot],
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

/// Scatter a whole `B[N,K]` into `bo`'s weight slots. Padding lanes must be zero.
pub fn pack_weights_into(bo: &mut Bo, plan: &Plan, b: &[f16]) -> Result<()> {
    assert_eq!(b.len(), plan.n * plan.k);
    bo.prep(0)?;
    scatter_weights(plan, f16_words(bo), b);
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

pub(crate) fn check_iova(bos: &[&Bo]) -> Result<()> {
    for bo in bos {
        let end = bo.dma_address() + bo.len() as u64;
        if end > (1u64 << 32) {
            return Err(Error::device("alloc_bo/iova", 0));
        }
    }
    Ok(())
}
