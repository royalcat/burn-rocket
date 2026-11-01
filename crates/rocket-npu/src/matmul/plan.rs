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
    /// Bytes per output element; read by the streaming planner only.
    pub obytes: usize,
    /// Weight banks a streamed task needs past its atom's.
    pub wspare: usize,
    /// The weight-streaming plan applies (see [`plan_stream`]).
    pub stream: bool,
    /// A K split's partials accumulate on the NPU when [`kacc_on`] (fp16 only).
    pub kacc: bool,
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
    obytes: 2,
    wspare: 0,
    stream: true,
    kacc: true,
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
    obytes: 4,
    wspare: 1,
    stream: true,
    kacc: false,
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
    obytes: 4,
    wspare: 1,
    stream: false,
    kacc: false,
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
    obytes: 4,
    wspare: 0,
    stream: false,
    kacc: false,
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
    obytes: 4,
    wspare: 0,
    stream: false,
    kacc: false,
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
    obytes: 4,
    wspare: 0,
    stream: false,
    kacc: false,
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

    /// `mm_plan_init_pin`: the cross-op-fused sibling of [`Plan::new`], with `Nt`
    /// and/or `Kt` pinned.
    ///
    /// The pinned dimension is fixed (the fused FFN/MLP contract: `Nt(projections)`
    /// must equal `Kt(down)`, a multiple of 32, so the output cube of one matmul
    /// aliases the input cube of the next tile for tile) and the free ones shrink to
    /// fit the CBUF. **The pin is the contract, so no `ROCKET_MM_*` env override is
    /// read and the asymmetric-Nt heuristic does not run** — the same rule as in C.
    ///
    /// `pin_nt`/`pin_kt` of `None` mean "free"; the C spells this `0`. Both, when
    /// given, must be multiples of 32 (`Nt`/`Kt` hardware granule) and no larger than
    /// `n`/`k`. `m % 4`, `k % 32`, `n % 16` are required, as everywhere in the
    /// planner — a caller with `m == 1` pads first.
    pub fn pinned(
        m: usize,
        k: usize,
        n: usize,
        pin_nt: Option<usize>,
        pin_kt: Option<usize>,
    ) -> Result<Plan> {
        let (mt, kt, nt) = plan_pin(&FP16, m, k, n, pin_nt, pin_kt)?;
        Ok(Plan::derive(m, k, n, mt, kt, nt))
    }

    /// A plan for the canonical tiling: plan at the full tile height and shrink Mt to
    /// the call's M, which only frees CBUF and keeps the M-independent K/N layout.
    pub fn canonical(m: usize, k: usize, n: usize) -> Result<Plan> {
        Plan::canonical_geom(&FP16, m, k, n)
    }

    /// [`Plan::canonical`] for an explicit geometry: the same M-independent `Kt`/`Nt`
    /// pinning with that dtype's slot sizes (the resident bf16 weight,
    /// [`super::bf16::resident_plan`], is the second caller).
    ///
    /// The planner runs at `max_tile` rows — so `Kt` is the one a full-height call gets,
    /// which is what keeps the resident weight layout valid for every `M` — and only
    /// `Mt` shrinks to `m`, which frees CBUF and never grows a tile.
    pub fn canonical_geom(geom: &Geometry, m: usize, k: usize, n: usize) -> Result<Plan> {
        let max_tile = hw::current().max_tile as usize;
        let (cmt, kt, nt) = plan_geom(geom, max_tile.max(4), k, n, 0)?;
        Ok(Plan::derive_geom(geom, m, k, n, cmt.min(m), kt, nt))
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

/// The C `kacc_on()`: NPU-side K-accumulation, on unless opted out.
///
/// Read per call (the C caches it once per process); a caller that changes the knob
/// between packing and running must keep its plan and its BO provisioning in step.
pub(crate) fn kacc_on() -> bool {
    match knobs::int("ROCKET_KACC") {
        Some(v) => v != 0,
        None => !knobs::present("ROCKET_NO_KACC"),
    }
}

/// `ROCKET_MM_STREAM`: the weight-streaming plan, on by default.
fn stream_on() -> bool {
    match knobs::int("ROCKET_MM_STREAM") {
        Some(v) => v > 0,
        None => true,
    }
}

/// Whether a geometry's plan streams. int8 streams only when
/// `ROCKET_MM_STREAM_INT8=1` too (the C: detector 1x1 convs route to the same entry
/// and those shapes are unmeasured).
fn stream_on_geom(geom: &Geometry) -> bool {
    if !geom.stream || !stream_on() {
        return false;
    }
    if geom.obytes != 4 {
        return true;
    }
    match knobs::int("ROCKET_MM_STREAM_INT8") {
        Some(v) => v > 0,
        None => false,
    }
}

/// `ROCKET_MM_STREAM_KT` (debug): restrict a streamed plan to this `Kt`, rounded down
/// to the K granularity. 0 = the full search.
fn stream_kt_force() -> usize {
    match knobs::int("ROCKET_MM_STREAM_KT") {
        Some(v) if v > 0 => v as usize,
        _ => 0,
    }
}

/// The widest streamed N tile. 4096 is the widest measured; the DPU channel field is
/// 13 bits, and an external userspace reports an 8192-wide int4 task that corrupts the
/// job after it, so nothing past what was measured.
const MM_STREAM_NT_MAX: usize = 4096;

/// Rows an M tile must reach to hide its weight re-read (the cost fit's crossover).
const STREAM_XM: usize = 128;
/// A host-read K pass's cost, in bytes-equivalents of its output tile.
const STREAM_HOSTW: f64 = 8.0;

/// `mm_plan_stream`: choose `(Mt, Kt, Nt)` for a streamed plan, or `None` when none
/// fits.
///
/// The whole-tile rule ([`fits`]) makes the weight tile share the CBUF with the input
/// tile, which caps `Nt` and multiplies the task count. A streamed program instead
/// cycles a weight cube larger than its banks through the banks the input tile leaves,
/// so one task covers a whole N slice. Its cost, in DRAM-byte equivalents over one
/// `max_tile`-row slice, is the weight stream (the larger of its bytes and `STREAM_XM`
/// rows' MACs), the input once per N tile, and the partial C each K pass writes (read
/// back by the next pass when KACC accumulates on the NPU, at `STREAM_HOSTW` times its
/// bytes when the host sums every pass).
///
/// The slice is `min(M, max_tile)`, not all of `M`, so every M at or above the cap
/// plans the same `(Kt, Nt)` and a resident weight packed at one M serves every M.
fn plan_stream(
    geom: &Geometry,
    m: usize,
    k: usize,
    n: usize,
    budget: usize,
    max_mt: usize,
) -> Option<(usize, usize, usize)> {
    let profile = hw::current();
    let eb = geom.elem_num as f64 / geom.elem_den as f64;
    let ka = geom.k_align;
    let devacc = geom.kacc && kacc_on();
    let bank = profile.cbuf_bank_size as i64;
    let ms = m.min(max_mt);
    let nt = pick_nt(n, MM_STREAM_NT_MAX, geom.n_align);
    let nn = n.div_ceil(nt);
    let mut best_mt = 0usize;
    let mut best_kt = 0usize;
    let mut best = 0.0f64;
    let force = stream_kt_force() / ka * ka;

    for pass in if force > 0 { 0..2 } else { 1..2 } {
        let mut prev_kt = 0usize;
        let mut found = false;
        for nk in 1..=k / ka {
            let kt = rup(k.div_ceil(nk), ka);
            if kt > geom.kt_ceil || kt == prev_kt {
                continue;
            }
            prev_kt = kt;
            if pass == 0 && kt != force {
                continue;
            }
            let atom = crate::regcmd::fp16::stream_weight_banks(kt * geom.elem_num / geom.elem_den)
                + geom.wspare;
            if atom >= budget {
                continue;
            }
            let rows = (budget - atom) as i64 * bank * geom.elem_den as i64
                / (kt as i64 * geom.elem_num as i64);
            let mut mt = rows.min(ms as i64) as usize;
            mt &= !3;
            if mt < 4 {
                continue;
            }
            let nkp = k.div_ceil(kt);
            let mut tasks = 0.0f64;
            let mut m0 = 0usize;
            while m0 < ms {
                let tile = (ms - m0).min(mt);
                tasks += if tile > STREAM_XM {
                    tile as f64 / STREAM_XM as f64
                } else {
                    1.0
                };
                m0 += mt;
            }
            let cost = tasks * (n * k) as f64 * eb
                + (ms * k) as f64 * eb * nn as f64
                + (ms * n) as f64
                    * geom.obytes as f64
                    * if devacc {
                        (2 * nkp - 1) as f64
                    } else {
                        STREAM_HOSTW * nkp as f64
                    };
            if best_mt == 0 || cost < best {
                best = cost;
                best_mt = mt;
                best_kt = kt;
                found = true;
            }
            // The first Kt that holds the whole slice ends the search: a deeper split
            // moves the same weight bytes in more passes and more tasks.
            if mt == (ms & !3) {
                break;
            }
        }
        if found {
            return Some((best_mt, best_kt, nt));
        }
    }
    None
}

/// `matmul_plan_pin`: the pinned-tile planner behind [`Plan::pinned`].
///
/// `pin_nt`/`pin_kt` (`None` = free) follow the C's rules: a pinned dimension is
/// validated (multiple of 32, within the real extent) and then never touched; the
/// while loop shrinks only free dimensions, and a pinned pair that cannot fit is a
/// `Shape` error (which the FFN/MLP callers turn into their host-handoff fallback,
/// exactly as the C does).
///
/// Only the fp16 geometry has a caller (`mm_plan_init_pin` is fp16-only in C too), whose
/// `bank_slack` is 0 — the budget expression below is therefore identical to the C's
/// `CBUF_BANKS`; a future dtype with slack would need the C's no-slack form to stay
/// parity-exact.
fn plan_pin(
    geom: &Geometry,
    m: usize,
    k: usize,
    n: usize,
    pin_nt: Option<usize>,
    pin_kt: Option<usize>,
) -> Result<(usize, usize, usize)> {
    let profile = hw::current();
    let max_tile = profile.max_tile as usize;
    let budget = profile.cbuf_banks.saturating_sub(geom.bank_slack);
    let na = geom.n_align;
    let ka = geom.k_align;

    if m == 0 || k == 0 || n == 0 || m > DIM_MAX || k > DIM_MAX || n > DIM_MAX {
        return Err(Error::shape("plan_pin"));
    }
    if !k.is_multiple_of(ka) || !n.is_multiple_of(na) || !m.is_multiple_of(4) {
        return Err(Error::shape("plan_pin"));
    }
    if let Some(p) = pin_nt
        && (!p.is_multiple_of(32) || p > n)
    {
        return Err(Error::shape("plan_pin/nt"));
    }
    if let Some(p) = pin_kt
        && (!p.is_multiple_of(32) || p > k)
    {
        return Err(Error::shape("plan_pin/kt"));
    }

    let mut mt = m.min(max_tile);
    let mut nt = pin_nt.unwrap_or_else(|| n.min(max_tile));
    let mut kt = match pin_kt {
        Some(p) => p,
        None => max_kt(geom, mt, nt, k, budget),
    };
    while !fits(geom, mt, nt, kt, budget) {
        if pin_nt.is_none() && nt > 16 {
            nt -= na;
        } else if mt > 4 {
            mt -= 4;
        } else if pin_kt.is_none() && kt > 32 {
            kt -= ka;
        } else {
            return Err(Error::shape("plan_pin/cbuf"));
        }
    }

    let (nmt, nnt, nkt) = (m.div_ceil(mt), n.div_ceil(nt), k.div_ceil(kt));
    if (nmt as u64) * (nnt as u64) * (nkt as u64) > i32::MAX as u64 {
        return Err(Error::shape("plan_pin/jobs"));
    }
    Ok((mt, kt, nt))
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
    let mut streamed = false;
    if stream_on_geom(geom)
        && !geom.grouped
        && !knobs::present("ROCKET_MM_MT")
        && !knobs::present("ROCKET_MM_NT")
        && !knobs::present("ROCKET_MM_KT")
        && let Some((smt, skt, snt)) = plan_stream(geom, m, k, n, budget, max_tile)
    {
        // the streamed plan; see `plan_stream`
        mt = smt;
        kt = skt;
        nt = snt;
        streamed = true;
    } else if geom.grouped {
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
    if !streamed && !fits(geom, mt, nt, kt, budget) {
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
        // The streaming plan (the default) covers a whole N slice per task, so `nt`
        // may exceed the 256-row tile cap; `mt` and the alignments do not move.
        assert!(p.mt <= 256 && p.mt.is_multiple_of(4));
        assert!(p.nt >= 16 && p.nt <= 4096);
        assert_eq!(p.kt % 32, 0);
        assert!(p.nmt > 1);
        assert_eq!(p.jobs(), p.nmt * p.nnt * p.nkt);
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
    fn canonical_geom_is_canonical_for_the_fp16_geometry() {
        // The bf16 resident weight's planner is the general form; `Plan::canonical` is
        // its fp16 instantiation and must not drift from it.
        for (m, k, n) in [(4, 1024, 768), (64, 512, 2048), (256, 4096, 1024)] {
            assert_eq!(
                Plan::canonical(m, k, n).unwrap(),
                Plan::canonical_geom(&FP16, m, k, n).unwrap()
            );
        }
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
        let p = Plan::derive(264, 1024, 512, 256, 512, 128);
        assert!(
            p.nmt > 1 && p.nnt > 1 && p.nkt > 1,
            "the decode test must not be vacuous: {p:?}"
        );
    }
}
