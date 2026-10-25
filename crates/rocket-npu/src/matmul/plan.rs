//! Tiling plans for the RK3588 matmul geometries.
//!
//! The planner is a translation of `mm_plan_geom` with one [`Geometry`] per dtype
//! (fp16's `2 B`, N-align 16, asymmetric-Nt heuristic is the default of [`Plan::new`]).
//! Plans are pure: no device access.

use crate::error::{Error, Result};
use crate::hw;
use crate::knobs;

use crate::regcmd::desc::CBUF_BANK_SIZE;

const DIM_MAX: usize = 1 << 24;

/// One dtype's tiling constants (`struct mm_geom` in the C library).
///
/// The element size is a fraction because int4 is half a byte: `mm_banks` computes
/// `rows*Kt*ebytes_num/ebytes_den` in integer arithmetic, so 1/2 must stay exact.
#[derive(Debug, Clone, Copy)]
pub struct Geometry {
    pub name: &'static str,
    /// Bytes per element = `elem_num / elem_den` (int4 = 1/2).
    pub elem_num: usize,
    pub elem_den: usize,
    pub n_align: usize,
    pub k_align: usize,
    pub kt_ceil: usize,
    pub bank_slack: usize,
    /// The feature/weight cube's K granule: 32 for every 1/2/4-byte-per-two dtype,
    /// 16 for tf32's 4-byte tile (a 64-byte hardware "data entry" holds 32 fp16 but
    /// only 16 fp32).
    pub k_group: usize,
    /// The weight slot's N granularity (`weight_int8` groups 32, `weight_int4` 64).
    pub wt_n_align: usize,
    pub asym_nt: bool,
    pub pick_nt: bool,
    pub grouped: bool,
}

pub const FP16: Geometry = Geometry {
    name: "fp16",
    elem_num: 2,
    elem_den: 1,
    n_align: 16,
    k_align: 32,
    kt_ceil: 16384,
    bank_slack: 0,
    k_group: 32,
    wt_n_align: 16,
    asym_nt: true,
    pick_nt: false,
    grouped: false,
};

/// int8 dense geometry (`gen_matmul_int8`): 1 byte, N/K align 32, one bank of
/// feature-DMA slack.
pub const INT8: Geometry = Geometry {
    name: "int8",
    elem_num: 1,
    elem_den: 1,
    n_align: 32,
    k_align: 32,
    kt_ceil: 16384,
    bank_slack: 1,
    k_group: 32,
    wt_n_align: 32,
    asym_nt: false,
    pick_nt: false,
    grouped: false,
};

/// Group-wise int8: Kt must divide the quant group; pick the smallest Nt that
/// reaches the fewest tiles.
pub const INT8_GW: Geometry = Geometry {
    name: "int8 group-wise",
    elem_num: 1,
    elem_den: 1,
    n_align: 32,
    k_align: 32,
    kt_ceil: 16384,
    bank_slack: 1,
    k_group: 32,
    wt_n_align: 32,
    asym_nt: false,
    pick_nt: true,
    grouped: true,
};

/// bf16 shares the int16/fp16 2-byte geometry (N16, K32) and the fp32 output cube.
pub const BF16: Geometry = Geometry {
    name: "bf16",
    elem_num: 2,
    elem_den: 1,
    n_align: 16,
    k_align: 32,
    kt_ceil: 16384,
    bank_slack: 0,
    k_group: 32,
    wt_n_align: 16,
    asym_nt: false,
    pick_nt: false,
    grouped: false,
};

/// int4 dense geometry (`gen_matmul_int4`): half a byte per element, N align 64
/// (`weight_int4` groups output channels in 64s), K align 32, no bank slack.
pub const INT4: Geometry = Geometry {
    name: "int4",
    elem_num: 1,
    elem_den: 2,
    n_align: 64,
    k_align: 32,
    kt_ceil: 16384,
    bank_slack: 0,
    k_group: 32,
    wt_n_align: 64,
    asym_nt: false,
    pick_nt: false,
    grouped: false,
};

/// tf32 geometry (`gen_matmul_tf32`): 4 B per element in an fp32 container, N align
/// 16, **K align 16** (a 4-byte element halves the K-group) and the Kt ceiling halved
/// to 8192 — the weight tile is `Kt * 4` bytes and the generator refuses above the
/// 64 KiB CBUF bank. No bank slack, no asymmetric-Nt heuristic.
pub const TF32: Geometry = Geometry {
    name: "tf32",
    elem_num: 4,
    elem_den: 1,
    n_align: 16,
    k_align: 16,
    kt_ceil: 8192,
    bank_slack: 0,
    k_group: 16,
    wt_n_align: 16,
    asym_nt: false,
    pick_nt: false,
    grouped: false,
};

/// A resolved tiling plan plus its per-tile slot sizes, counted in the dtype's own
/// elements (fp32 for tf32, nibbles for int4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Plan {
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub mt: usize,
    pub kt: usize,
    pub nt: usize,
    pub nmt: usize,
    pub nnt: usize,
    pub nkt: usize,
    pub in_slot: usize,
    pub wt_slot: usize,
    pub out_slot: usize,
}

impl Plan {
    pub fn new(m: usize, k: usize, n: usize) -> Result<Plan> {
        let (mt, kt, nt) = plan_geom(&FP16, m, k, n, 0)?;
        Ok(Plan::derive(m, k, n, mt, kt, nt))
    }

    /// A plan for the canonical tiling: plan at the full tile height and shrink Mt to
    /// the call's M, which only frees CBUF and keeps the M-independent K/N layout.
    pub fn canonical(m: usize, k: usize, n: usize) -> Result<Plan> {
        let max_tile = hw::current().max_tile as usize;
        let (cmt, kt, nt) = plan_geom(&FP16, max_tile.max(4), k, n, 0)?;
        Ok(Plan::derive(m, k, n, cmt.min(m), kt, nt))
    }

    pub fn derive(m: usize, k: usize, n: usize, mt: usize, kt: usize, nt: usize) -> Plan {
        Plan::derive_geom(&FP16, m, k, n, mt, kt, nt)
    }

    /// [`Plan::derive`] for an explicit geometry: the slot sizes are the dtype's
    /// (weight-N granularity differs — 16 fp16/bf16/tf32, 32 int8, 64 int4 — and so
    /// does the cube K granule, 32 except tf32's 16, while the feature and output
    /// cubes keep their 4/16 element groups).
    pub fn derive_geom(
        geom: &Geometry,
        m: usize,
        k: usize,
        n: usize,
        mt: usize,
        kt: usize,
        nt: usize,
    ) -> Plan {
        Plan {
            m,
            k,
            n,
            mt,
            kt,
            nt,
            nmt: m.div_ceil(mt),
            nnt: n.div_ceil(nt),
            nkt: k.div_ceil(kt),
            in_slot: rup(mt, 4) * rup(kt, geom.k_group),
            wt_slot: rup(nt, geom.wt_n_align) * rup(kt, geom.k_group),
            out_slot: rup(mt, 4) * rup(nt, 16),
        }
    }

    pub fn jobs(&self) -> usize {
        self.nmt * self.nnt * self.nkt
    }

    /// fp16 elements in the packed-A region.
    pub fn input_elems(&self) -> usize {
        self.nmt * self.nkt * self.in_slot
    }

    /// fp16 elements in the full output cube.
    pub fn cube_elems(&self) -> usize {
        self.nmt * self.nnt * self.out_slot
    }

    pub fn m_tile(&self, mi: usize) -> usize {
        (self.m - mi * self.mt).min(self.mt)
    }

    pub fn n_tile(&self, ni: usize) -> usize {
        (self.n - ni * self.nt).min(self.nt)
    }

    pub fn k_tile(&self, ki: usize) -> usize {
        (self.k - ki * self.kt).min(self.kt)
    }

    /// Decode a global job index into `(mi, ni, ki)` in the submission order: `mi`
    /// outermost, then `ni`, then `ki` innermost.
    ///
    /// `ki` innermost is load-bearing, not cosmetic: the CPU-accumulation paths add one
    /// output tile's K-partials in ascending `ki`, so this order is what keeps the fp32
    /// sum identical across the one-shot, batched and resident forms.
    pub fn tile_of(&self, job: usize) -> (usize, usize, usize) {
        let job = job % self.jobs();
        (
            job / (self.nnt * self.nkt),
            (job / self.nkt) % self.nnt,
            job % self.nkt,
        )
    }
}

pub(crate) fn rup(x: usize, a: usize) -> usize {
    x.div_ceil(a) * a
}

/// CBUF banks a `rows` x `kt` operand tile of this dtype occupies.
///
/// The C truncates the byte count (`long bytes = rows*Kt*num/den`) before rounding up
/// to banks, which is what keeps the half-byte int4 geometry exact.
fn banks(geom: &Geometry, rows: usize, kt: usize) -> usize {
    (rows * kt * geom.elem_num / geom.elem_den).div_ceil(CBUF_BANK_SIZE)
}

fn fits(geom: &Geometry, mt: usize, nt: usize, kt: usize, budget: usize) -> bool {
    banks(geom, mt, kt) + banks(geom, nt, kt) <= budget
}

fn max_kt(geom: &Geometry, mt: usize, nt: usize, k: usize, budget: usize) -> usize {
    let ka = geom.k_align;
    let mut kt = k.min(geom.kt_ceil);
    kt = (kt / ka) * ka;
    if kt < ka {
        kt = ka;
    }
    while kt > ka && !fits(geom, mt, nt, kt, budget) {
        kt -= ka;
    }
    kt
}

fn pick_nt(n: usize, max_tile: usize, na: usize) -> usize {
    let mut nt = if n <= max_tile {
        n
    } else {
        let nnt = n.div_ceil(max_tile);
        n.div_ceil(nnt)
    };
    nt = nt.div_ceil(na) * na;
    if nt > max_tile {
        nt = (max_tile / na) * na;
    }
    if nt < na {
        nt = na;
    }
    nt
}

fn asym_on() -> bool {
    match knobs::int("ROCKET_MM_ASYM") {
        Some(v) => v > 0,
        None => true,
    }
}

/// The planner. Returns `(Mt, Kt, Nt)`, or a shape/tiling error.
pub fn plan_geom(
    geom: &Geometry,
    m: usize,
    k: usize,
    n: usize,
    group: usize,
) -> Result<(usize, usize, usize)> {
    let profile = hw::current();
    let max_tile = profile.max_tile as usize;
    let budget = profile.cbuf_banks.saturating_sub(geom.bank_slack);
    let na = geom.n_align;
    let ka = geom.k_align;

    if m == 0 || k == 0 || n == 0 || m > DIM_MAX || k > DIM_MAX || n > DIM_MAX {
        return Err(Error::shape("plan"));
    }
    if !k.is_multiple_of(ka) || !n.is_multiple_of(na) || !m.is_multiple_of(4) {
        return Err(Error::shape("plan"));
    }
    if geom.grouped {
        if group < ka || !group.is_multiple_of(ka) || !k.is_multiple_of(group) {
            return Err(Error::shape("plan/group"));
        }
    } else if group != 0 {
        return Err(Error::shape("plan/group"));
    }

    let mut mt = m.min(max_tile);
    let mut nt = if geom.pick_nt {
        pick_nt(n, max_tile, na)
    } else {
        let mut nt = n.min(max_tile);
        nt = (nt / na) * na;
        nt.max(na)
    };

    if let Some(v) = knobs::int("ROCKET_MM_MT")
        && v >= 4
    {
        mt = (v as usize / 4) * 4;
        mt = mt.min(m);
    }
    if let Some(v) = knobs::int("ROCKET_MM_NT") {
        let v = v as usize;
        if v >= na {
            nt = (v / na) * na;
            nt = nt.min(n);
        }
    }

    let mut kt;
    if geom.grouped {
        loop {
            let mut found = None;
            let mut d = group;
            while d >= ka {
                if group.is_multiple_of(d) && fits(geom, mt, nt, d, budget) {
                    found = Some(d);
                    break;
                }
                d -= ka;
            }
            if let Some(d) = found {
                kt = d;
                break;
            }
            if nt > na {
                nt -= na;
            } else if mt > 4 {
                mt -= 4;
            } else {
                return Err(Error::shape("plan/cbuf"));
            }
        }
    } else {
        kt = max_kt(geom, mt, nt, k, budget);
        if geom.asym_nt
            && asym_on()
            && knobs::int("ROCKET_MM_NT").is_none()
            && nt == max_tile
            && n > max_tile
            && max_tile / 2 >= na
            && k.div_ceil(kt) > 1
        {
            nt = max_tile / 2;
            kt = max_kt(geom, mt, nt, k, budget);
        }
        while !fits(geom, mt, nt, kt, budget) && nt > na {
            nt -= na;
        }
        while !fits(geom, mt, nt, kt, budget) && mt > 4 {
            mt -= 4;
        }
        if let Some(v) = knobs::int("ROCKET_MM_KT") {
            let v = v as usize;
            if v >= ka {
                kt = (v / ka) * ka;
                kt = kt.min((k / ka) * ka);
                kt = kt.min(geom.kt_ceil);
                while kt > ka && !fits(geom, mt, nt, kt, budget) {
                    kt -= ka;
                }
            }
        }
    }
    if !fits(geom, mt, nt, kt, budget) {
        return Err(Error::shape("plan/cbuf"));
    }

    let nmt = m.div_ceil(mt);
    let nnt = n.div_ceil(nt);
    let nkt = k.div_ceil(kt);
    if (nmt as u64) * (nnt as u64) * (nkt as u64) > i32::MAX as u64 {
        return Err(Error::shape("plan/jobs"));
    }
    Ok((mt, kt, nt))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_shape_is_one_tile() {
        let p = Plan::new(4, 64, 16).unwrap();
        assert_eq!((p.mt, p.kt, p.nt), (4, 64, 16));
        assert_eq!(p.jobs(), 1);
        assert_eq!(p.in_slot, 4 * 64);
        assert_eq!(p.wt_slot, 16 * 64);
        assert_eq!(p.out_slot, 4 * 16);
    }

    #[test]
    fn large_shape_tiles() {
        let p = Plan::new(512, 2560, 4096).unwrap();
        assert!(p.mt <= 256 && p.nt <= 256);
        assert_eq!(p.kt % 32, 0);
        assert_eq!(p.jobs(), p.nmt * p.nnt * p.nkt);
        assert_eq!(p.mt % 4, 0);
    }

    #[test]
    fn rejects_unaligned() {
        assert!(Plan::new(6, 64, 16).is_err()); // M%4
        assert!(Plan::new(4, 48, 16).is_err()); // K%32
        assert!(Plan::new(4, 64, 8).is_err()); // N%16
    }

    #[test]
    fn canonical_keeps_k_n_layout() {
        let full = Plan::new(256, 1024, 768).unwrap();
        let small = Plan::canonical(4, 1024, 768).unwrap();
        assert_eq!(full.kt, small.kt);
        assert_eq!(full.nt, small.nt);
        assert_eq!(small.mt, 4);
    }

    #[test]
    fn tile_of_walks_mi_ni_ki_with_ki_innermost() {
        for (m, k, n) in [(4, 64, 16), (264, 1024, 512), (512, 2560, 4096)] {
            let p = Plan::new(m, k, n).unwrap();
            let mut want = Vec::with_capacity(p.jobs());
            for mi in 0..p.nmt {
                for ni in 0..p.nnt {
                    for ki in 0..p.nkt {
                        want.push((mi, ni, ki));
                    }
                }
            }
            let got: Vec<_> = (0..p.jobs()).map(|j| p.tile_of(j)).collect();
            assert_eq!(got, want, "M={m} K={k} N={n}");
        }
        let p = Plan::new(264, 1024, 512).unwrap();
        assert!(
            p.nmt > 1 && p.nnt > 1 && p.nkt > 1,
            "the decode test must not be vacuous: {p:?}"
        );
    }
}
