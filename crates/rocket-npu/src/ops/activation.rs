//! Elementwise fp16 activation and elementwise binary ops on the DPU (RK3588).
//!
//! Port of `rocket_activation.c`. Two device datapaths live here:
//!
//! - the **DPU LUT** pass ([`activation`], [`leaky_relu`], [`elu`], [`selu`]): a
//!   standalone flying-mode DPU job with no conv — DPU_RDMA reads the fp16 input,
//!   the BN stage scales it into the LUT index domain, the two NVDLA SDP tables with
//!   hardware linear interpolation produce the fixed-point result and OUT_CVT casts
//!   it back to fp16. The result is an fp16 LUT *approximation*: gate it with a
//!   tolerance ([`lut_apply`] is the host model of the same chain).
//! - the **conv-main EW** path ([`ew_mul`], [`ew_add`], [`ew_sub`], [`ew_max`],
//!   [`ew_min`]): an identity matmul (`C = a·I32ᵀ`) feeds the DPU accumulator as the
//!   EW main input and the DPU_RDMA ERDMA operand (`b`) is combined with it, so
//!   `out[i] = a[i] (op) b[i]` runs on the NPU. fp16 add/mul are bit-exact against
//!   the host arithmetic.
//!
//! [`clip`] (constant-operand MAX then MIN), [`prelu`] (`max(x, α_c·x)` for the
//! universal `α∈[0,1]`, the general four-pass form otherwise) and [`div`] (the
//! reciprocal LUT then the EW multiply) compose those two primitives.
//!
//! The gated kinds (`HardSwish`/`SiLU`/`GELU`/`Mish`) run as `x · gate(x)`: a
//! `[0,1]`-output gate LUT on the NPU and then the multiply — on the host by default
//! (one cheap pass over the readback) or on the NPU with `ROCKET_ACT_NPU_MUL` present
//! (which rounds the product through the `_Float16` EW path, so the result changes
//! slightly). `ROCKET_ACT_WIDE_LUT` selects the single-pass wide-output LUT instead
//! (`HardSwish`/`SiLU`/`GELU` only) — the C's RE path, which spikes on the flat
//! negative tail of GELU.
//!
//! # Numerics and deviations from the C originals
//!
//! The device paths are line-for-line ports: the same LUT words (byte-compared with
//! the C builder), the same register program word count (1070), the same tile caps
//! (`DPU_LUT_MAXN` 32768 elements for the LUT path, `[M,32]` rows of 1020 for the EW
//! path with `M >= 12` and `M % 4`), one submit + one fence wait per tile, as in C.
//! The host references ([`activation_ref`] and the parametric ones) keep the C's
//! mixed precision: input read through `f32`, the function evaluated in `f64`
//! (glibc `erf` for the GELU/GELU-gate shapes, so the numbers match the C bit for
//! bit), one RNE rounding to fp16 at the end.
//!
//! The `n = 0` case is a `Shape` error, as in the C (`n <= 0` there); a valid call
//! pads the element count up to the 8-element cube atom internally.
//!
//! Cost shape (board, cores 0-3, n = 262144): a single LUT pass is 2-8 ms, while an EW
//! binary op is ~10-13 ms because it is submit-bound: one 1020-row `[M,32]` tile per
//! 32640 elements is one submit + one fence wait, i.e. 9 submits at that length
//! (the C library pays the same). That is exactly why the gated kinds multiply on the
//! host by default: `ROCKET_ACT_NPU_MUL` replaces ~n cheap host multiplies with a
//! second NPU round-trip.
//!
//! One deliberate deviation: the C's `rocket_leaky_relu_fp16` / `elu_run` keep their
//! 1026-word table in a `static`, so two threads calling them race on it. Every table
//! here is built on the stack per call (as `rocket_activation_fp16` already does), so
//! the entry points are thread-safe as long as the caller serializes the NPU itself.
//!
//! Env knobs keep the C names and are read per call: `ROCKET_ACT_NPU_MUL`,
//! `ROCKET_ACT_WIDE_LUT`, `ROCKET_LUT_QFLOOR`, `ROCKET_LUT_XLO`/`_XHI`/`_SEXP`/
//! `_OFFSET`/`_MINEXP`/`_SCALE`/`_BNMUL`/`_BNALU`/`_IDXSCALE`/`_LO`/`_LESLOPE`/
//! `_LOSLOPE`/`_LOSHIFT`/`_SHIFT`/`_SPTR`/`_ENABLE`, `ROCKET_LEAKY_R`,
//! `ROCKET_LEAKY_NOREPAIR`, `ROCKET_ELU_R`, `ROCKET_ELU_NOREPAIR`,
//! `ROCKET_EW_CFG`/`_ERDMA_CFG`/`_RDMA_FMC`/`_SURF`/`_BASE_OFF`/`_NOTCH`.

use std::f64::consts::SQRT_2;
use std::fmt;

use half::f16;

use crate::Device;
use crate::device::Bo;
use crate::device::uapi::DrmRocketTask;
use crate::error::{Error, Result};
use crate::knobs;
use crate::matmul::layout;
use crate::matmul::run::wait_ns;
use crate::regcmd::activation::{LutActParams, MAX_OPS, emit as emit_lut};
use crate::regcmd::desc::RC_STRIDE;
use crate::regcmd::fp16::{MatmulParams, emit as emit_matmul};

pub use crate::regcmd::activation::LUT_ENTRIES;

/// Per-op element cap for the LUT tiling (the C `DPU_LUT_MAXN`).
///
/// `DPU_DATA_CUBE_WIDTH` is 13 bits, so `n/8 <= 0x1FFF` (`n <= 65528`) is the hard
/// ceiling, but a 65528-element chunk corrupts a thin tail of cube positions on HW;
/// 32768 is clean.
pub const DPU_LUT_MAXN: usize = 32768;

/// Pad BOs so a feature DMA never reads past the allocation (the C `CBUF_BANK`).
const CBUF_BANK: usize = 4096;

/// EW elementwise geometry: `N == K == 32` (the identity matmul).
const EW_NN: usize = 32;
/// Rows per EW tile: `feature_grains = M + 1` must fit 10 bits, so `M <= 1022`.
const EW_M_TILE: usize = 1020;
/// The K-accumulation surface stride floors at `MAX(M, 12)`, so every tile keeps
/// `M >= 12` (the extra rows are zero-padded and unread).
const EW_M_MIN: usize = 12;
/// Regcmd words one EW tile reserves (C `regs[256]`).
const EW_RC_WORDS: usize = 256;

// ── Activation kinds ───────────────────────────────────────────────────────────

/// `enum rocket_activation_kind` — the `ROCKET_ACTIVATION_*` values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum Activation {
    /// `ROCKET_ACTIVATION_SIGMOID` (0), `[0,1]` output.
    Sigmoid = 0,
    /// `ROCKET_ACTIVATION_HARDSIGMOID` (1), TFLite `clip(x/6+0.5, 0, 1)`.
    HardSigmoid = 1,
    /// `ROCKET_ACTIVATION_HARDSWISH` (2), `x · hardsigmoid(x)`; two-pass.
    HardSwish = 2,
    /// `ROCKET_ACTIVATION_SILU` (3), `x · sigmoid(x)`; two-pass.
    Silu = 3,
    /// `ROCKET_ACTIVATION_TANH` (4), signed output via the bias trick.
    Tanh = 4,
    /// `ROCKET_ACTIVATION_GELU` (5), accurate two-pass `x · Φ(x)`.
    Gelu = 5,
    /// `ROCKET_ACTIVATION_LEAKY_RELU` (6); parametric, see [`leaky_relu`].
    LeakyRelu = 6,
    /// `ROCKET_ACTIVATION_SQRT` (7), positive domain.
    Sqrt = 7,
    /// `ROCKET_ACTIVATION_RSQRT` (8), positive domain.
    Rsqrt = 8,
    /// `ROCKET_ACTIVATION_RECIPROCAL` (9), positive domain.
    Reciprocal = 9,
    /// `ROCKET_ACTIVATION_EXP` (10), shifted table, default domain `[-16,0]`.
    Exp = 10,
    /// `ROCKET_ACTIVATION_GELU_GATE` (11), the Gaussian CDF `Φ(x)` gate.
    GeluGate = 11,
    /// `ROCKET_ACTIVATION_SOFTPLUS` (12), shifted table, default domain `[-16,16]`.
    Softplus = 12,
    /// `ROCKET_ACTIVATION_MISH_GATE` (13), `tanh(softplus(x))`.
    MishGate = 13,
    /// `ROCKET_ACTIVATION_MISH` (14), `x · tanh(softplus(x))`; two-pass.
    Mish = 14,
    /// `ROCKET_ACTIVATION_ABS` (15), symmetric shifted table (exact at the kink).
    Abs = 15,
    /// `ROCKET_ACTIVATION_ELU` (16); parametric, see [`elu`].
    Elu = 16,
    /// `ROCKET_ACTIVATION_SELU` (17); parametric, see [`selu`].
    Selu = 17,
    /// `ROCKET_ACTIVATION_LOG` (18), signed output, default domain `[0.25,32]`.
    Log = 18,
}

impl Activation {
    /// Every kind, in the C enum's order.
    pub const ALL: [Activation; 19] = [
        Activation::Sigmoid,
        Activation::HardSigmoid,
        Activation::HardSwish,
        Activation::Silu,
        Activation::Tanh,
        Activation::Gelu,
        Activation::LeakyRelu,
        Activation::Sqrt,
        Activation::Rsqrt,
        Activation::Reciprocal,
        Activation::Exp,
        Activation::GeluGate,
        Activation::Softplus,
        Activation::MishGate,
        Activation::Mish,
        Activation::Abs,
        Activation::Elu,
        Activation::Selu,
        Activation::Log,
    ];

    /// The enum value as an `i32` (the C `int kind`).
    pub const fn code(self) -> i32 {
        self as i32
    }

    /// Decode a C kind value.
    pub const fn from_code(code: i32) -> Option<Activation> {
        Some(match code {
            0 => Activation::Sigmoid,
            1 => Activation::HardSigmoid,
            2 => Activation::HardSwish,
            3 => Activation::Silu,
            4 => Activation::Tanh,
            5 => Activation::Gelu,
            6 => Activation::LeakyRelu,
            7 => Activation::Sqrt,
            8 => Activation::Rsqrt,
            9 => Activation::Reciprocal,
            10 => Activation::Exp,
            11 => Activation::GeluGate,
            12 => Activation::Softplus,
            13 => Activation::MishGate,
            14 => Activation::Mish,
            15 => Activation::Abs,
            16 => Activation::Elu,
            17 => Activation::Selu,
            18 => Activation::Log,
            _ => return None,
        })
    }

    /// `rocket_activation_name`.
    pub const fn name(self) -> &'static str {
        match self {
            Activation::Sigmoid => "sigmoid",
            Activation::HardSigmoid => "hardsigmoid",
            Activation::HardSwish => "hardswish",
            Activation::Silu => "silu",
            Activation::Tanh => "tanh",
            Activation::Gelu => "gelu",
            Activation::LeakyRelu => "leaky_relu",
            Activation::Sqrt => "sqrt",
            Activation::Rsqrt => "rsqrt",
            Activation::Reciprocal => "reciprocal",
            Activation::Log => "log",
            Activation::Exp => "exp",
            Activation::GeluGate => "gelu_gate",
            Activation::Softplus => "softplus",
            Activation::MishGate => "mish_gate",
            Activation::Mish => "mish",
            Activation::Abs => "abs",
            Activation::Elu => "elu",
            Activation::Selu => "selu",
        }
    }

    /// The kind's gate for the two-pass gated forms (`ROCKET_ACT_NPU_MUL` /
    /// `ROCKET_ACT_WIDE_LUT` decide the route in [`activation`]).
    const fn gate(self) -> Option<Activation> {
        match self {
            Activation::HardSwish => Some(Activation::HardSigmoid),
            Activation::Gelu => Some(Activation::GeluGate),
            Activation::Mish => Some(Activation::MishGate),
            Activation::Silu => Some(Activation::Sigmoid),
            _ => None,
        }
    }

    /// The single-pass kinds `rocket_activation_fp16` accepts.
    const fn single_pass(self) -> bool {
        !matches!(
            self,
            Activation::LeakyRelu | Activation::Elu | Activation::Selu
        )
    }
}

impl fmt::Display for Activation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

// ── LUT table + epilogue ───────────────────────────────────────────────────────

/// The two DPU LUT tables: `LE[0..=512]` and `LO[513..=1025]`, Q-format words.
#[derive(Clone, PartialEq, Eq)]
pub struct LutTable(pub [u16; LUT_ENTRIES]);

impl LutTable {
    /// A zeroed table (every builder writes all 1026 words).
    pub fn zeroed() -> Self {
        Self([0; LUT_ENTRIES])
    }

    /// The raw words, `LE` then `LO`.
    pub fn words(&self) -> &[u16; LUT_ENTRIES] {
        &self.0
    }

    /// The LE half (`table id 0`).
    pub fn le(&self) -> &[u16] {
        &self.0[..513]
    }

    /// The LO half (`table id 1`).
    pub fn lo(&self) -> &[u16] {
        &self.0[513..]
    }
}

impl fmt::Debug for LutTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LutTable(le[0]={} lo[0]={})", self.0[0], self.0[513])
    }
}

/// `lut_epilogue_t`: the LUT-activation register constants.
///
/// The C keeps a `lut` pointer here; the port passes the [`LutTable`] alongside.
/// The same struct drives the standalone op and the conv→activation fusion, so the
/// two cannot drift.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LutEpilogue {
    /// `ROCKET_ACTIVATION_*`; host reference only (the generator ignores it).
    pub kind: i32,
    /// fp16 bits: `x -> LUT index` scale.
    pub bn_mul_operand: u16,
    /// BN ALU bias word (`0x8000_0000` = fp32 `-0.0`, a no-op).
    pub bn_alu_cfg: u32,
    /// `DPU_OUT_CVT_OFFSET` (signed, pre-shift).
    pub out_cvt_offset: u32,
    /// `DPU_OUT_CVT_SCALE.OUT_CVT_SCALE` (1 on the validated path).
    pub out_cvt_scale: u16,
    /// `DPU_OUT_CVT_SHIFT.MINUS_EXP`: `out = (q + offset) * 2^-exp`.
    pub out_cvt_minus_exp: u8,
    /// `DPU_OUT_CVT_SHIFT.CVT_TYPE`.
    pub out_cvt_cvt_type: u8,
    /// LE table start (Q index; `0xffff_c000` = -16384).
    pub lut_le_start: u32,
    /// LO table end (`0x0000_4000` = 16384).
    pub lut_lo_end: u32,
    /// LE underflow extrapolation slope scale.
    pub le_slope_scale: u16,
    /// LE slope shift.
    pub le_slope_shift: u8,
    /// LO overflow extrapolation slope scale.
    pub lo_slope_scale: u16,
    /// LO slope shift.
    pub lo_slope_shift: u8,
    /// `log2(step)` for the LE table (5 => step 32).
    pub le_index_select: u8,
    /// `log2(step)` for the LO table.
    pub lo_index_select: u8,
}

// ── Host math (the C `act_eval` and the parametric kernels) ────────────────────

unsafe extern "C" {
    /// glibc `erf`, the function the C reference calls (so the LUT words match it
    /// bit for bit).
    fn erf(x: f64) -> f64;
}

pub(crate) fn erf64(x: f64) -> f64 {
    // SAFETY: the C `erf` is pure, takes one f64 and returns one f64.
    unsafe { erf(x) }
}

/// `SELU_ALPHA` (Klambauer et al. 2017).
pub const SELU_ALPHA: f64 = 1.673_263_242_354_377_2;
/// `SELU_LAMBDA`.
pub const SELU_LAMBDA: f64 = 1.050_700_987_355_480_5;

fn hardsigmoid_d(x: f64) -> f64 {
    let y = x / 6.0 + 0.5; // TFLite HARD_SIGMOID: clip(x/6+0.5, 0, 1)
    y.clamp(0.0, 1.0)
}

fn elu_d(alpha: f64, lambda: f64, x: f64) -> f64 {
    lambda * if x >= 0.0 { x } else { alpha * (x.exp() - 1.0) }
}

/// The C `act_eval`: the exact function the LUT samples (f64, glibc math).
///
/// The parametric kinds return `0.0`, exactly as the C does: they have dedicated
/// entry points and references ([`leaky_relu_ref`], [`elu_ref`], [`selu_ref`]).
pub fn eval(kind: Activation, x: f64) -> f64 {
    match kind {
        Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        Activation::HardSigmoid => hardsigmoid_d(x),
        Activation::HardSwish => x * hardsigmoid_d(x),
        Activation::Silu => x * (1.0 / (1.0 + (-x).exp())),
        Activation::Tanh => x.tanh(),
        Activation::Gelu => 0.5 * x * (1.0 + erf64(x / SQRT_2)),
        Activation::Sqrt => {
            if x > 0.0 {
                x.sqrt()
            } else {
                0.0
            }
        }
        Activation::Rsqrt => {
            if x > 0.0 {
                1.0 / x.sqrt()
            } else {
                0.0
            }
        }
        Activation::Reciprocal => {
            if x != 0.0 {
                1.0 / x
            } else {
                0.0
            }
        }
        Activation::Log => {
            if x > 0.0 {
                x.ln()
            } else {
                0.0
            }
        }
        Activation::Exp => x.exp(),
        Activation::GeluGate => 0.5 * (1.0 + erf64(x / SQRT_2)),
        Activation::Softplus => {
            if x > 0.0 {
                x + (-x).exp().ln_1p()
            } else {
                x.exp().ln_1p()
            }
        }
        Activation::MishGate => softplus_mish_d(x),
        Activation::Mish => x * softplus_mish_d(x),
        Activation::Abs => x.abs(),
        Activation::LeakyRelu | Activation::Elu | Activation::Selu => 0.0,
    }
}

/// `tanh(softplus(x))`, the stably-evaluated Mish gate.
fn softplus_mish_d(x: f64) -> f64 {
    if x > 0.0 {
        (x + (-x).exp().ln_1p()).tanh()
    } else {
        x.exp().ln_1p().tanh()
    }
}

// ── LUT builders (`build_lut_*`) ──────────────────────────────────────────────

/// fp16 bit pattern of a value, for the BN_MUL index-scale operand (C `fp16_bits`).
pub fn fp16_bits(v: f64) -> u16 {
    f16::from_f64(v).to_bits()
}

/// fp32 bit pattern (C `fp32_bits`), for the BN_ALU bias word.
pub fn fp32_bits(v: f64) -> u32 {
    (v as f32).to_bits()
}

/// C `lround(v)` then the table clamp: `q = clamp(lround(v*32768), floor, 32767)`.
fn q_word(v: f64, floor: i64) -> u16 {
    let q = (v * 32768.0).round() as i64;
    q.clamp(floor, 32767) as u16
}

/// `ROCKET_LUT_QFLOOR`: the shifted-table entry floor (default 1).
///
/// HW fact: a table entry of exactly `q = 0` mis-decodes to a garbage constant
/// (~4.0 in the fp16 output), so the shifted builds floor every entry to 1
/// (`0` reproduces the raw glitch for RE).
fn qfloor_knob() -> i64 {
    knobs::i64_auto("ROCKET_LUT_QFLOOR").unwrap_or(1)
}

/// `build_lut_unit`: the LE/LO Q0.15 tables of a `[0,1]`-output activation on the
/// sigmoid grid (`step = 32/index_scale` in x).
fn build_lut_unit(kind: Activation, index_scale: f64, lut: &mut LutTable) {
    let step = 32.0 / index_scale;
    for i in 0..=512 {
        let y = eval(kind, -((512 - i) as f64) * step);
        lut.0[i] = q_word(y, 0);
    }
    for i in 0..=512 {
        let y = eval(kind, i as f64 * step);
        lut.0[513 + i] = q_word(y, 0);
    }
}

/// `build_lut_affine`: the normalized `g(x) = (f(x) - lo)/(hi - lo)` on the same grid,
/// so OUT_CVT can affine-decode a signed output range.
fn build_lut_affine(kind: Activation, index_scale: f64, lo: f64, hi: f64, lut: &mut LutTable) {
    let step = 32.0 / index_scale;
    let span = if hi - lo != 0.0 { hi - lo } else { 1.0 };
    for i in 0..=512 {
        let g = (eval(kind, -((512 - i) as f64) * step) - lo) / span;
        lut.0[i] = q_word(g, 0);
    }
    for i in 0..=512 {
        let g = (eval(kind, i as f64 * step) - lo) / span;
        lut.0[513 + i] = q_word(g, 0);
    }
}

/// `build_lut_shifted`: one table over `[x_lo,x_hi]`, mapped onto the positive index
/// half by the BN-ALU bias (`index = (x - x_lo)·16384/(x_hi - x_lo)`), so `x` never
/// crosses 0 in the index domain and the LE/LO mux cannot glitch. `LE` is flat.
fn build_lut_shifted(
    kind: Activation,
    x_lo: f64,
    x_hi: f64,
    out_lo: f64,
    s: f64,
    lut: &mut LutTable,
) {
    let dx = (x_hi - x_lo) / 512.0;
    let qfloor = qfloor_knob();
    for j in 0..=512 {
        let g = (eval(kind, x_lo + j as f64 * dx) - out_lo) / s;
        lut.0[513 + j] = q_word(g, qfloor);
    }
    for j in 0..=512 {
        lut.0[j] = lut.0[513];
    }
}

/// `build_lut_leaky`: the natural LE (negative branch, slope `alpha`) / LO (positive
/// branch) split of LeakyReLU over `[-R,R]`.
fn build_lut_leaky(alpha: f64, index_scale: f64, out_lo: f64, s: f64, lut: &mut LutTable) {
    let step = 32.0 / index_scale;
    for i in 0..=512 {
        let x = -((512 - i) as f64) * step;
        let f = if x >= 0.0 { x } else { alpha * x };
        lut.0[i] = q_word((f - out_lo) / s, 0);
    }
    for i in 0..=512 {
        let x = i as f64 * step;
        let f = if x >= 0.0 { x } else { alpha * x };
        lut.0[513 + i] = q_word((f - out_lo) / s, 0);
    }
}

/// `build_lut_shifted_elu`: ELU/SELU over the symmetric domain `[-R,R]` (the Abs
/// trick: the `x=0` kink lands on the middle sample, so no host repair is needed).
fn build_lut_shifted_elu(
    alpha: f64,
    lambda: f64,
    x_lo: f64,
    x_hi: f64,
    out_lo: f64,
    s: f64,
    lut: &mut LutTable,
) {
    let dx = (x_hi - x_lo) / 512.0;
    let qfloor = qfloor_knob();
    for j in 0..=512 {
        let f = elu_d(alpha, lambda, x_lo + j as f64 * dx);
        lut.0[513 + j] = q_word((f - out_lo) / s, qfloor);
    }
    for j in 0..=512 {
        lut.0[j] = lut.0[513];
    }
}

// ── Domain + epilogue builder (`act_shifted_domain`, `rocket_lut_epilogue_build`) ─

/// `act_shifted_domain`: `(x_lo, x_hi, out_lo, Sexp)` for the shifted single-table
/// kinds, `None` otherwise.
fn shifted_domain(kind: Activation) -> Option<(f64, f64, f64, i32)> {
    Some(match kind {
        Activation::Sqrt => (0.25, 64.0, 0.0, 3),
        Activation::Rsqrt => (0.50, 64.0, 0.0, 1),
        Activation::Reciprocal => (0.25, 32.0, 0.0, 2),
        Activation::Log => (0.25, 32.0, 0.25f64.ln(), 3),
        Activation::Exp => (-16.0, 0.0, 0.0, 0),
        Activation::Softplus => (-16.0, 16.0, 0.0, 5),
        Activation::Abs => (-16.0, 16.0, 0.0, 5),
        _ => return None,
    })
}

/// `2^sexp` with the C `1u << sexp` wrap-around for a bad override.
fn pow2(sexp: i32) -> f64 {
    (1u32.wrapping_shl(sexp as u32)) as f64
}

/// `lround(15.0 - log2(S))` as the C's `uint8_t`.
fn minus_exp_from_s(s: f64) -> u8 {
    ((15.0 - s.log2()).round() as i64) as u8
}

/// C `(uint8_t)(15 - Sexp)`.
fn minus_exp_from_sexp(sexp: i32) -> u8 {
    (15i64 - sexp as i64) as u8
}

/// `lround(out_lo * 32768.0 / S)` as the C's signed-then-`uint32_t` offset.
fn offset_from(out_lo: f64, s: f64) -> u32 {
    ((out_lo * 32768.0 / s).round() as i32) as u32
}

/// `rocket_lut_epilogue_build`: the LUT + epilogue constants of a SMOOTH single-pass
/// activation (`SiLU` / `tanh` / `GELU`; `HardSwish` only over its curved knee) — and
/// of the shifted-table kinds (`sqrt`/`rsqrt`/`reciprocal`/`log`/`exp`/`softplus`/
/// `abs`), which the C's builder handles on its first branch.
///
/// `Err(Unsupported)` for a kind that is not single-pass (`sigmoid`/`hardsigmoid` and
/// the two `[0,1]` gates use the unit geometry inside [`activation`], `LeakyReLU` and
/// `ELU`/`SELU` have their own entry points).
pub fn lut_epilogue_build(kind: Activation, lut: &mut LutTable) -> Result<LutEpilogue> {
    let mut ep = LutEpilogue {
        kind: kind.code(),
        bn_alu_cfg: 0x8000_0000, // fp32 -0.0: a no-op BN ALU bias add
        out_cvt_scale: 1,
        out_cvt_cvt_type: 1,
        lut_le_start: 0xffff_c000, // -16384 (Q index)
        lut_lo_end: 0x0000_4000,   //  16384
        le_index_select: 5,        // step = 2^5 = 32 in the Q index domain
        lo_index_select: 5,
        ..Default::default()
    };

    // POSITIVE-DOMAIN kinds: always the shifted single-table mode, so `x` never
    // crosses 0 and the LE/LO sign mux never sees it.
    if let Some((mut x_lo, mut x_hi, out_lo, mut sexp)) = shifted_domain(kind) {
        if let Some(v) = knobs::float("ROCKET_LUT_XLO") {
            x_lo = v;
        }
        if let Some(v) = knobs::float("ROCKET_LUT_XHI") {
            x_hi = v;
        }
        if let Some(v) = knobs::int("ROCKET_LUT_SEXP") {
            sexp = v as i32;
        }
        let s = pow2(sexp);
        let scale = 16384.0 / (x_hi - x_lo);
        build_lut_shifted(kind, x_lo, x_hi, out_lo, s, lut);
        ep.bn_mul_operand = fp16_bits(scale);
        ep.bn_alu_cfg = fp32_bits(-x_lo * scale); // index = (x - x_lo)*scale
        ep.out_cvt_minus_exp = minus_exp_from_s(s);
        ep.out_cvt_offset = offset_from(out_lo, s);
        if let Some(v) = knobs::i64_auto("ROCKET_LUT_OFFSET") {
            ep.out_cvt_offset = v as u32;
        }
        if let Some(v) = knobs::u32_auto("ROCKET_LUT_MINEXP") {
            ep.out_cvt_minus_exp = v as u8;
        }
        return Ok(ep);
    }

    // SHIFTED single-table mode for the smooth kinds (ROCKET_LUT_SHIFT): map the whole
    // domain onto the positive index half so there is no x≈0 LE/LO boundary to glitch.
    if knobs::present("ROCKET_LUT_SHIFT") {
        let (mut x_lo, mut x_hi, out_lo, s) = match kind {
            Activation::Tanh => (-8.0, 8.0, -1.0, 2.0),
            Activation::Silu | Activation::Gelu => (-12.0, 12.0, -0.5, 16.0),
            _ => return Err(Error::unsupported("rocket_lut_epilogue_build")),
        };
        if let Some(v) = knobs::float("ROCKET_LUT_XLO") {
            x_lo = v;
        }
        if let Some(v) = knobs::float("ROCKET_LUT_XHI") {
            x_hi = v;
        }
        let scale = 16384.0 / (x_hi - x_lo);
        build_lut_shifted(kind, x_lo, x_hi, out_lo, s, lut);
        ep.bn_mul_operand = fp16_bits(scale);
        // The BN ALU bias is added POST-scale (index domain) as fp32, so
        // B = -x_lo*scale (8192, the index centre, for a symmetric domain).
        ep.bn_alu_cfg = fp32_bits(-x_lo * scale);
        ep.out_cvt_minus_exp = minus_exp_from_s(s);
        ep.out_cvt_offset = offset_from(out_lo, s);
        if let Some(v) = knobs::u32_auto("ROCKET_LUT_BNALU") {
            ep.bn_alu_cfg = v;
        }
        if let Some(v) = knobs::u32_auto("ROCKET_LUT_BNMUL") {
            ep.bn_mul_operand = v as u16;
        }
        if let Some(v) = knobs::i64_auto("ROCKET_LUT_OFFSET") {
            ep.out_cvt_offset = v as u32;
        }
        if let Some(v) = knobs::u32_auto("ROCKET_LUT_MINEXP") {
            ep.out_cvt_minus_exp = v as u8;
        }
        return Ok(ep);
    }

    let (mut index_scale, mut lo, mut sexp, le_slope_scale, le_slope_shift) = match kind {
        Activation::Tanh => (5216.0, -1.0, 1, 23107u16, 22u8),
        // silu/gelu: asymptotic (no flat run), wide +-16 table, S=32.
        Activation::Silu | Activation::Gelu => (1024.0, -0.5, 5, 0, 0),
        // hardswish: KNEE-ONLY +-3 (S=4) — the NVDLA flat-tail quirk forbids its
        // x<=-3 constant run.
        Activation::HardSwish => (5461.0, -0.5, 2, 0, 0),
        _ => return Err(Error::unsupported("rocket_lut_epilogue_build")),
    };
    if let Some(v) = knobs::float("ROCKET_LUT_IDXSCALE") {
        index_scale = v;
    }
    if let Some(v) = knobs::float("ROCKET_LUT_LO") {
        lo = v;
    }
    if let Some(v) = knobs::int("ROCKET_LUT_SEXP") {
        sexp = v as i32;
    }
    let s = pow2(sexp);
    build_lut_affine(kind, index_scale, lo, lo + s, lut);
    ep.bn_mul_operand = fp16_bits(index_scale);
    ep.out_cvt_minus_exp = minus_exp_from_sexp(sexp);
    ep.out_cvt_offset = offset_from(lo, s);
    ep.le_slope_scale = le_slope_scale;
    ep.le_slope_shift = le_slope_shift;
    ep.lo_slope_scale = 0;
    ep.lo_slope_shift = 0;
    if let Some(v) = knobs::i64_auto("ROCKET_LUT_OFFSET") {
        ep.out_cvt_offset = v as u32;
    }
    if let Some(v) = knobs::u32_auto("ROCKET_LUT_MINEXP") {
        ep.out_cvt_minus_exp = v as u8;
    }
    if let Some(v) = knobs::u32_auto("ROCKET_LUT_SCALE") {
        ep.out_cvt_scale = v as u16;
    }
    if let Some(v) = knobs::u32_auto("ROCKET_LUT_BNMUL") {
        ep.bn_mul_operand = v as u16;
    }
    if let Some(v) = knobs::u32_auto("ROCKET_LUT_LESLOPE") {
        ep.le_slope_scale = v as u16;
    }
    if let Some(v) = knobs::u32_auto("ROCKET_LUT_LOSLOPE") {
        ep.lo_slope_scale = v as u16;
    }
    if let Some(v) = knobs::u32_auto("ROCKET_LUT_LOSHIFT") {
        ep.lo_slope_shift = v as u8;
    }
    Ok(ep)
}

// ── Job assembly (the constants `rocket_activation_fp16` programs) ─────────────

/// One LUT pass: the table and the constants the DPU program carries.
struct LutJob {
    lut: LutTable,
    ep: LutEpilogue,
}

/// The `[0,1]`-output kinds share the validated sigmoid geometry; only the table
/// content differs (`bn_mul_operand` = fp16(2596), index scale 2596, +1 rounding
/// bias and the Q0.15 decode).
fn unit_job(kind: Activation) -> LutJob {
    let mut lut = LutTable::zeroed();
    build_lut_unit(kind, 2596.0, &mut lut);
    let ep = LutEpilogue {
        kind: kind.code(),
        bn_mul_operand: 0x6912, // fp16(2596)
        bn_alu_cfg: 0x8000_0000,
        out_cvt_offset: 1, // +1 LSB rounding bias (Q domain)
        out_cvt_scale: 1,
        out_cvt_minus_exp: 15, // Q0.15 -> fp16: multiply by 2^-15
        out_cvt_cvt_type: 1,
        lut_le_start: 0xffff_c000,
        lut_lo_end: 0x0000_4000,
        le_slope_scale: 23107,
        le_slope_shift: 22,
        lo_slope_scale: 0,
        lo_slope_shift: 0,
        le_index_select: 5,
        lo_index_select: 5,
    };
    LutJob { lut, ep }
}

/// Build the job of a single-pass kind exactly as `rocket_activation_fp16` does,
/// including its own RE knobs (which are a subset of the builder's, by design).
fn single_pass_job(kind: Activation) -> Result<LutJob> {
    let mut job = match kind {
        Activation::Sigmoid
        | Activation::HardSigmoid
        | Activation::GeluGate
        | Activation::MishGate => unit_job(kind),
        _ => {
            let mut lut = LutTable::zeroed();
            let ep = lut_epilogue_build(kind, &mut lut)?;
            LutJob { lut, ep }
        }
    };
    if let Some(v) = knobs::i64_auto("ROCKET_LUT_OFFSET") {
        job.ep.out_cvt_offset = v as u32;
    }
    if let Some(v) = knobs::u32_auto("ROCKET_LUT_MINEXP") {
        job.ep.out_cvt_minus_exp = v as u8;
    }
    if let Some(v) = knobs::u32_auto("ROCKET_LUT_SCALE") {
        job.ep.out_cvt_scale = v as u16;
    }
    if let Some(v) = knobs::u32_auto("ROCKET_LUT_BNMUL") {
        job.ep.bn_mul_operand = v as u16;
    }
    Ok(job)
}

/// The x range a single-pass kind's table covers: `(x_lo, x_hi)`.
///
/// For the shifted single-table kinds this is the function domain (`sqrt` ->
/// `[0.25, 64]`, `exp` -> `[-16, 0]`, ...); for the two-table kinds it is the range
/// of the LE/LO split (`sigmoid` -> `[-6.31, 6.31]`, `tanh` -> `[-3.14, 3.14]`).
/// Outside it the hardware saturates or extrapolates with the slope.
pub fn table_domain(kind: Activation) -> Result<(f32, f32)> {
    let job = single_pass_job(kind)?;
    let mul = f16::from_bits(job.ep.bn_mul_operand).to_f32();
    let bias = f32::from_bits(job.ep.bn_alu_cfg);
    let le_start = job.ep.lut_le_start as i32 as f32;
    let lo_end = job.ep.lut_lo_end as i32 as f32;
    // A flat LE half is the shifted single-table layout: the function starts at
    // index 0, not at LE_START.
    let le_flat = job.lut.le().iter().all(|&w| w == job.lut.le()[0]);
    let start = if le_flat { 0.0 } else { le_start };
    if mul == 0.0 {
        return Err(Error::shape("table_domain"));
    }
    Ok(((start - bias) / mul, (lo_end - bias) / mul))
}

/// `ROCKET_LEAKY_R` (default 16, values below 1 ignored as in C).
fn leaky_radius() -> f64 {
    match knobs::float("ROCKET_LEAKY_R") {
        Some(r) if r >= 1.0 => r,
        _ => 16.0,
    }
}

/// `ROCKET_ELU_R` (default 8).
fn elu_radius() -> f64 {
    match knobs::float("ROCKET_ELU_R") {
        Some(r) if r >= 1.0 => r,
        _ => 8.0,
    }
}

/// The LeakyReLU LUT job: the natural LE/LO split over `[-R,R]`, span
/// `[-alpha*R, R]` covered by the smallest power of two.
fn leaky_job(alpha: f32) -> Result<LutJob> {
    if alpha <= 0.0 {
        // alpha = 0 is plain ReLU; its all-zero negative branch trips the LUT quirk.
        return Err(Error::unsupported("rocket_leaky_relu_fp16/alpha"));
    }
    let r = leaky_radius();
    let index_scale = 16384.0 / r;
    let out_lo = -(alpha as f64) * r;
    let need = r - out_lo; // = R*(1+alpha)
    let mut sexp = 0i32;
    while pow2(sexp) < need {
        sexp += 1;
    }
    let s = pow2(sexp);
    let mut lut = LutTable::zeroed();
    build_lut_leaky(alpha as f64, index_scale, out_lo, s, &mut lut);
    let mut ep = LutEpilogue {
        kind: Activation::LeakyRelu.code(),
        bn_mul_operand: fp16_bits(index_scale),
        bn_alu_cfg: 0x8000_0000,
        out_cvt_offset: offset_from(out_lo, s),
        out_cvt_scale: 1,
        out_cvt_minus_exp: minus_exp_from_sexp(sexp),
        out_cvt_cvt_type: 1,
        lut_le_start: 0xffff_c000,
        lut_lo_end: 0x0000_4000,
        // Slopes stay 0: |x| > R saturates at the table edge.
        le_slope_scale: 0,
        le_slope_shift: 0,
        lo_slope_scale: 0,
        lo_slope_shift: 0,
        le_index_select: 5,
        lo_index_select: 5,
    };
    if let Some(v) = knobs::i64_auto("ROCKET_LUT_OFFSET") {
        ep.out_cvt_offset = v as u32;
    }
    if let Some(v) = knobs::u32_auto("ROCKET_LUT_MINEXP") {
        ep.out_cvt_minus_exp = v as u8;
    }
    if let Some(v) = knobs::u32_auto("ROCKET_LUT_BNMUL") {
        ep.bn_mul_operand = v as u16;
    }
    if let Some(v) = knobs::u32_auto("ROCKET_LUT_BNALU") {
        ep.bn_alu_cfg = v;
    }
    Ok(LutJob { lut, ep })
}

/// The ELU/SELU LUT job: `lambda*(x>=0 ? x : alpha*(e^x-1))` on the symmetric shifted
/// table over `[-R,R]` (the Abs trick; no host repair needed).
fn elu_job(alpha: f64, lambda: f64) -> Result<LutJob> {
    if alpha <= 0.0 {
        return Err(Error::unsupported("rocket_elu_fp16/alpha"));
    }
    let r = elu_radius();
    let (x_lo, x_hi) = (-r, r);
    let scale = 16384.0 / (x_hi - x_lo);
    let out_lo = -lambda * alpha;
    let need = lambda * r - out_lo; // = lambda*(R+alpha)
    let mut sexp = 0i32;
    while pow2(sexp) < need {
        sexp += 1;
    }
    let s = pow2(sexp);
    let mut lut = LutTable::zeroed();
    build_lut_shifted_elu(alpha, lambda, x_lo, x_hi, out_lo, s, &mut lut);
    let mut ep = LutEpilogue {
        kind: Activation::Elu.code(),
        bn_mul_operand: fp16_bits(scale),
        bn_alu_cfg: fp32_bits(-x_lo * scale), // index = (x - x_lo)*scale
        out_cvt_offset: offset_from(out_lo, s),
        out_cvt_scale: 1,
        out_cvt_minus_exp: minus_exp_from_s(s),
        out_cvt_cvt_type: 1,
        lut_le_start: 0xffff_c000,
        lut_lo_end: 0x0000_4000,
        le_slope_scale: 0,
        le_slope_shift: 0,
        lo_slope_scale: 0,
        lo_slope_shift: 0,
        le_index_select: 5,
        lo_index_select: 5,
    };
    if let Some(v) = knobs::i64_auto("ROCKET_LUT_OFFSET") {
        ep.out_cvt_offset = v as u32;
    }
    if let Some(v) = knobs::u32_auto("ROCKET_LUT_MINEXP") {
        ep.out_cvt_minus_exp = v as u8;
    }
    if let Some(v) = knobs::u32_auto("ROCKET_LUT_BNMUL") {
        ep.bn_mul_operand = v as u16;
    }
    Ok(LutJob { lut, ep })
}

// ── The host model of the DPU LUT chain ───────────────────────────────────────

/// Linear interpolation between table samples at `pos` (in sample units).
fn lerp(table: &[u16], pos: f32) -> f32 {
    let last = (table.len() - 1) as f32;
    let pos = pos.clamp(0.0, last);
    let j = (pos.floor() as usize).min(table.len() - 2);
    let frac = pos - j as f32;
    let a = table[j] as f32;
    let b = table[j + 1] as f32;
    a + frac * (b - a)
}

/// The host model of what the DPU computes for one input: BN multiply + ALU bias,
/// the LE/LO hybrid table with linear interpolation (slope extrapolation outside the
/// table range) and the affine OUT_CVT decode.
///
/// This is the reference the board self-test compares the NPU against for the LUT
/// kinds: the C LUT is approximate by design, so comparing against the *interpolated
/// table* isolates the hardware/register side, while [`activation_ref`] (the exact
/// math) measures the LUT's own design error.
pub fn lut_apply(ep: &LutEpilogue, lut: &LutTable, x: f32) -> f32 {
    let bn_mul = f16::from_bits(ep.bn_mul_operand).to_f32();
    let bn_alu = f32::from_bits(ep.bn_alu_cfg); // fp32 bias (0x8000_0000 = -0.0)
    let index = x * bn_mul + bn_alu;
    let le_start = ep.lut_le_start as i32 as f32;
    let lo_end = ep.lut_lo_end as i32 as f32;
    let le_step = pow2((ep.le_index_select & 0x1F) as i32) as f32;
    let lo_step = pow2((ep.lo_index_select & 0x1F) as i32) as f32;

    let q = if index < 0.0 {
        let pos = (index - le_start) / le_step;
        if pos >= 0.0 {
            lerp(lut.le(), pos)
        } else {
            let slope = ep.le_slope_scale as f32 / pow2((ep.le_slope_shift & 0x1F) as i32) as f32;
            lut.0[0] as f32 + (index - le_start) * slope
        }
    } else {
        let pos = index / lo_step;
        if pos <= 512.0 {
            lerp(lut.lo(), pos)
        } else {
            let slope = ep.lo_slope_scale as f32 / pow2((ep.lo_slope_shift & 0x1F) as i32) as f32;
            lut.0[LUT_ENTRIES - 1] as f32 + (index - lo_end) * slope
        }
    };
    let offset = ep.out_cvt_offset as i32 as f32;
    let scale = ep.out_cvt_scale as f32 / pow2(ep.out_cvt_minus_exp as i32) as f32;
    // HW fact (measured on the board): the interpolated/extrapolated table value is an
    // unsigned 16-bit quantity, so it saturates to [0, 32767] before OUT_CVT. That is
    // what makes tanh's LE-slope tail stop exactly at -1 (q = 0) instead of running
    // past it, and it floors a gate's negative slope tail at 0.
    let q = q.clamp(0.0, 32767.0);
    (q + offset) * scale
}

// ── Device runs (BOs, submit, fence) ──────────────────────────────────────────

/// The BO's words as `f16`s (the mapping is page-aligned, `f16` is 2-byte aligned).
fn bo_words(bo: &Bo) -> &[f16] {
    let bytes = bo.as_slice();
    // SAFETY: f16 is repr(transparent) over u16 with align 2 and every prefix of the
    // mapping is initialized.
    unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<f16>(), bytes.len() / 2) }
}

/// The BO's words as mutable `f16`s.
fn bo_words_mut(bo: &mut Bo) -> &mut [f16] {
    let len = bo.len();
    let ptr = bo.as_ptr();
    // SAFETY: exclusive borrow of the BO, page-aligned mapping, 2-byte elements.
    unsafe { std::slice::from_raw_parts_mut(ptr.cast::<f16>(), len / 2) }
}

/// Write a program into `bo` from offset 0 and return its device address.
fn write_program(bo: &mut Bo, ops: &[crate::regcmd::desc::NpuOp]) -> Result<u32> {
    let bytes = bo.as_mut_slice();
    if ops.len() * 8 > bytes.len() {
        return Err(Error::tiling("write_program"));
    }
    for (i, op) in ops.iter().enumerate() {
        bytes[i * 8..i * 8 + 8].copy_from_slice(&op.0.to_ne_bytes());
    }
    Ok(bo.dma_address() as u32)
}

/// `run_dpu_lut`: allocate the BOs, lay the flat fp16 cube, generate the program,
/// submit, wait for the fence and read back — tiled at [`DPU_LUT_MAXN`] with the
/// chunk padded to the 8-element cube atom, exactly as the C does.
fn run_dpu_lut(dev: &Device, job: &LutJob, input: &[f16], out: &mut [f16]) -> Result<()> {
    let n = input.len();
    let cap = n.min(DPU_LUT_MAXN);
    let cap_np = cap.div_ceil(8) * 8;

    let _guard = dev.alloc_bo(4096)?; // keeps the IOVA away from 0, as in C
    let mut in_bo = dev.alloc_bo(cap_np * 2 + CBUF_BANK)?;
    let mut out_bo = dev.alloc_bo(cap_np * 2 + CBUF_BANK)?;
    let mut rc_bo = dev.alloc_bo(MAX_OPS * 8)?;

    let mut ops = Vec::with_capacity(MAX_OPS);
    for base in (0..n).step_by(DPU_LUT_MAXN) {
        let cn = (n - base).min(DPU_LUT_MAXN);
        let np = cn.div_ceil(8) * 8;

        in_bo.prep(0)?;
        {
            let words = bo_words_mut(&mut in_bo);
            words.fill(f16::ZERO);
            words[..cn].copy_from_slice(&input[base..base + cn]);
        }
        in_bo.fini()?;

        ops.clear();
        let params = LutActParams {
            input_dma: in_bo.dma_address() as u32,
            output_dma: out_bo.dma_address() as u32,
            n: np as u32,
            lut: job.lut.words(),
            bn_mul_operand: job.ep.bn_mul_operand,
            bn_alu_cfg: job.ep.bn_alu_cfg,
            out_cvt_offset: job.ep.out_cvt_offset,
            out_cvt_scale: job.ep.out_cvt_scale,
            out_cvt_minus_exp: job.ep.out_cvt_minus_exp,
            out_cvt_cvt_type: job.ep.out_cvt_cvt_type,
            lut_le_start: job.ep.lut_le_start,
            lut_lo_end: job.ep.lut_lo_end,
            le_slope_scale: job.ep.le_slope_scale,
            le_slope_shift: job.ep.le_slope_shift,
            lo_slope_scale: job.ep.lo_slope_scale,
            lo_slope_shift: job.ep.lo_slope_shift,
            le_index_select: job.ep.le_index_select,
            lo_index_select: job.ep.lo_index_select,
        };
        let count = emit_lut(&params, &mut ops)?;
        if count > MAX_OPS {
            return Err(Error::tiling("run_dpu_lut/regcmd"));
        }

        out_bo.prep(0)?;
        bo_words_mut(&mut out_bo).fill(f16::ZERO);
        out_bo.fini()?;

        rc_bo.prep(0)?;
        let regcmd = write_program(&mut rc_bo, &ops)?;
        rc_bo.fini()?;

        let task = DrmRocketTask {
            regcmd,
            regcmd_count: count as u32,
        };
        dev.submit(
            &[task],
            &[in_bo.handle(), rc_bo.handle()],
            &[out_bo.handle()],
            0,
        )?;
        out_bo.prep(wait_ns())?;
        out[base..base + cn].copy_from_slice(&bo_words(&out_bo)[..cn]);
        out_bo.fini()?;
    }
    Ok(())
}

// ── Public activation entry points ────────────────────────────────────────────

fn check_io(name: &'static str, input: &[f16], out: &[f16]) -> Result<()> {
    if input.len() != out.len() {
        return Err(Error::shape(name).with_detail(format!(
            "in {} out {}",
            input.len(),
            out.len()
        )));
    }
    if input.is_empty() {
        return Err(Error::shape(name));
    }
    Ok(())
}

/// `rocket_activation_fp16`: `out[i] = f(in[i])` for the enum kinds.
///
/// The gated kinds (`HardSwish`/`SiLU`/`GELU`/`Mish`) run the two-pass route
/// (`x · gate(x)`, the gate LUT on the NPU and the multiply on the host unless
/// `ROCKET_ACT_NPU_MUL` is present) unless `ROCKET_ACT_WIDE_LUT` selects the
/// single-pass table. `LeakyReLU`/`ELU`/`SELU` are parametric and rejected here, as
/// in C: use [`leaky_relu`], [`elu`], [`selu`].
pub fn activation(dev: &Device, kind: Activation, input: &[f16], out: &mut [f16]) -> Result<()> {
    check_io("rocket_activation_fp16", input, out)?;
    let wide = knobs::present("ROCKET_ACT_WIDE_LUT");
    if let Some(gate) = kind.gate().filter(|_| !wide) {
        let mut g = vec![f16::ZERO; input.len()];
        activation(dev, gate, input, &mut g)?;
        if knobs::present("ROCKET_ACT_NPU_MUL") {
            ew_mul(dev, input, &g, out) // fully on the NPU
        } else {
            for ((o, x), y) in out.iter_mut().zip(input).zip(&g) {
                *o = f16::from_f32(x.to_f32() * y.to_f32());
            }
            Ok(())
        }
    } else {
        if !kind.single_pass() {
            return Err(Error::unsupported("rocket_activation_fp16/kind"));
        }
        let job = single_pass_job(kind)?;
        run_dpu_lut(dev, &job, input, out)
    }
}

/// The host model of [`activation`]: the same table, evaluated by [`lut_apply`]
/// instead of on the NPU (no device access).
///
/// The gated kinds compose the gate model with the multiply; `ROCKET_ACT_WIDE_LUT`
/// selects the single-pass table. The multiply is modelled on the host, so with
/// `ROCKET_ACT_NPU_MUL` present the device's EW rounding can differ by an fp16 ULP
/// (that is exactly the C library's documented reason for keeping the host multiply
/// the default).
pub fn activation_model(kind: Activation, input: &[f16], out: &mut [f16]) -> Result<()> {
    check_io("activation_model", input, out)?;
    let wide = knobs::present("ROCKET_ACT_WIDE_LUT");
    if let Some(gate) = kind.gate().filter(|_| !wide) {
        let job = single_pass_job(gate)?;
        for (o, x) in out.iter_mut().zip(input) {
            let g = lut_apply(&job.ep, &job.lut, x.to_f32());
            *o = f16::from_f32(x.to_f32() * g);
        }
        Ok(())
    } else {
        if !kind.single_pass() {
            return Err(Error::unsupported("activation_model/kind"));
        }
        let job = single_pass_job(kind)?;
        for (o, x) in out.iter_mut().zip(input) {
            *o = f16::from_f32(lut_apply(&job.ep, &job.lut, x.to_f32()));
        }
        Ok(())
    }
}

/// `rocket_leaky_relu_fp16`: `x >= 0 ? x : alpha*x` on the DPU LUT (`alpha > 0`).
///
/// Exact within `[-R,R]` (`ROCKET_LEAKY_R`, default 16); `|x| > R` saturates at the
/// table edge. The DPU's sign-based LE/LO mux spikes in a razor-thin band around
/// `x = 0`; the host patches that band with the exact value after the readback
/// (`ROCKET_LEAKY_NOREPAIR` disables it).
pub fn leaky_relu(dev: &Device, alpha: f32, input: &[f16], out: &mut [f16]) -> Result<()> {
    check_io("rocket_leaky_relu_fp16", input, out)?;
    let job = leaky_job(alpha)?;
    run_dpu_lut(dev, &job, input, out)?;
    if !knobs::present("ROCKET_LEAKY_NOREPAIR") {
        let guard = (leaky_radius() / 1024.0) as f32;
        for (o, x) in out.iter_mut().zip(input) {
            let xv = x.to_f32();
            if xv <= guard && xv >= -guard {
                *o = f16::from_f32(if xv >= 0.0 { xv } else { alpha * xv });
            }
        }
    }
    Ok(())
}

/// `elu_run` with an explicit `(alpha, lambda)`; `rocket_elu_fp16` is
/// `(alpha, 1.0)`, `rocket_selu_fp16` the fixed self-normalizing pair.
fn elu_inner(dev: &Device, alpha: f64, lambda: f64, input: &[f16], out: &mut [f16]) -> Result<()> {
    check_io("rocket_elu_fp16", input, out)?;
    let job = elu_job(alpha, lambda)?;
    run_dpu_lut(dev, &job, input, out)?;
    // The shifted table does not escape the mux when the domain straddles 0, so the
    // razor-thin x≈0 band is patched on the host with the exact value.
    if !knobs::present("ROCKET_ELU_NOREPAIR") {
        let guard = (elu_radius() / 1024.0) as f32;
        for (o, x) in out.iter_mut().zip(input) {
            let xv = x.to_f32();
            if xv <= guard && xv >= -guard {
                *o = f16::from_f64(elu_d(alpha, lambda, xv as f64));
            }
        }
    }
    Ok(())
}

/// `rocket_elu_fp16`: `x >= 0 ? x : alpha*(e^x - 1)` on the symmetric shifted LUT
/// (`alpha > 0`, `ROCKET_ELU_R` default 8).
pub fn elu(dev: &Device, alpha: f32, input: &[f16], out: &mut [f16]) -> Result<()> {
    elu_inner(dev, alpha as f64, 1.0, input, out)
}

/// `rocket_selu_fp16`: `lambda*(x >= 0 ? x : alpha*(e^x - 1))` with the fixed
/// `alpha = 1.6732632423543772`, `lambda = 1.0507009873554805`.
pub fn selu(dev: &Device, input: &[f16], out: &mut [f16]) -> Result<()> {
    elu_inner(dev, SELU_ALPHA, SELU_LAMBDA, input, out)
}

// ── Elementwise binary ops (the conv-main EW datapath) ────────────────────────

/// The EW ALU operation (`enum { EW_OP_ADD, ... }` in C).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EwOp {
    /// `a + b`.
    Add,
    /// `a * b`.
    Mul,
    /// `a - b` on the add datapath with `b` negated (an exact fp16 sign flip).
    Sub,
    /// `max(a, b)` (`EW_ALU_ALGO` MAX).
    Max,
    /// `min(a, b)` (`EW_ALU_ALGO` MIN).
    Min,
}

impl EwOp {
    /// The op's name.
    pub const fn name(self) -> &'static str {
        match self {
            EwOp::Add => "add",
            EwOp::Mul => "mul",
            EwOp::Sub => "sub",
            EwOp::Max => "max",
            EwOp::Min => "min",
        }
    }
}

/// The `[M,32]` feature-cube position of a flat element (C
/// `feature_data(32, M, 1, 8, c+1, h+1, 1)`).
#[inline]
fn ew_pos(m: usize, t: usize) -> usize {
    layout::feat_idx(m, t % EW_NN + 1, t / EW_NN + 1)
}

/// `ew_binary_fp16`: `out[i] = a[i] (op) b[i]` through the conv-main EW datapath.
fn ew_binary(dev: &Device, a: &[f16], b: &[f16], out: &mut [f16], op: EwOp) -> Result<()> {
    check_io("rocket_ew_fp16", a, out)?;
    if b.len() != a.len() {
        return Err(Error::shape("rocket_ew_fp16/b"));
    }
    let n = a.len();
    let total_rows = n.div_ceil(EW_NN);
    let alloc_m = total_rows.clamp(EW_M_MIN, EW_M_TILE).div_ceil(4) * 4;

    let wt_len = EW_NN * EW_NN * 2 + CBUF_BANK;
    let cube = alloc_m * EW_NN * 2 + CBUF_BANK;
    let mut wt = dev.alloc_bo(wt_len)?;
    let mut in_bo = dev.alloc_bo(cube)?;
    let mut op_bo = dev.alloc_bo(cube)?;
    let mut out_bo = dev.alloc_bo(cube)?;
    let mut rc_bo = dev.alloc_bo(EW_RC_WORDS * 8)?;

    // Identity weights I_32 (built once, reused for every tile).
    wt.prep(0)?;
    {
        let words = bo_words_mut(&mut wt);
        words.fill(f16::ZERO);
        for kk in 1..=EW_NN {
            for c in 1..=EW_NN {
                words[layout::wt_idx(EW_NN, kk, c)] = if kk == c { f16::ONE } else { f16::ZERO };
            }
        }
    }
    wt.fini()?;

    let mut ops = Vec::with_capacity(RC_STRIDE);
    for base in (0..n).step_by(EW_M_TILE * EW_NN) {
        let telems = (n - base).min(EW_M_TILE * EW_NN);
        let m = telems.div_ceil(EW_NN).max(EW_M_MIN).div_ceil(4) * 4;

        in_bo.prep(0)?;
        op_bo.prep(0)?;
        {
            let inp = bo_words_mut(&mut in_bo);
            inp.fill(f16::ZERO);
            let opb = bo_words_mut(&mut op_bo);
            opb.fill(f16::ZERO);
            for t in 0..telems {
                let pos = ew_pos(m, t);
                inp[pos] = a[base + t];
                opb[pos] = if op == EwOp::Sub {
                    f16::from_f32(-b[base + t].to_f32())
                } else {
                    b[base + t]
                };
            }
        }
        in_bo.fini()?;
        op_bo.fini()?;

        out_bo.prep(0)?;
        bo_words_mut(&mut out_bo).fill(f16::ZERO);
        out_bo.fini()?;

        let mut p = MatmulParams::new(
            m as u32,
            EW_NN as u32,
            EW_NN as u32,
            in_bo.dma_address() as u32,
            wt.dma_address() as u32,
            out_bo.dma_address() as u32,
        );
        p.fp32_to_fp16 = true;
        p.accumulate = true;
        p.add_dma = op_bo.dma_address() as u32;
        p.ew_mul = op == EwOp::Mul;
        p.ew_op = match op {
            EwOp::Max => 2,
            EwOp::Min => 3,
            _ => 0,
        };

        ops.clear();
        let count = emit_matmul(&p, &mut ops)?;
        if count > EW_RC_WORDS {
            return Err(Error::tiling("rocket_ew_fp16/regcmd"));
        }
        rc_bo.prep(0)?;
        let regcmd = write_program(&mut rc_bo, &ops)?;
        rc_bo.fini()?;

        let task = DrmRocketTask {
            regcmd,
            regcmd_count: count as u32,
        };
        dev.submit(
            &[task],
            &[in_bo.handle(), wt.handle(), op_bo.handle(), rc_bo.handle()],
            &[out_bo.handle()],
            0,
        )?;
        out_bo.prep(wait_ns())?;
        {
            let ob = bo_words(&out_bo);
            for t in 0..telems {
                out[base + t] = ob[ew_pos(m, t)];
            }
        }
        out_bo.fini()?;
    }
    Ok(())
}

/// `rocket_ew_mul_fp16`: `out[i] = a[i] * b[i]` on the NPU (bit-exact vs the host
/// product).
pub fn ew_mul(dev: &Device, a: &[f16], b: &[f16], out: &mut [f16]) -> Result<()> {
    ew_binary(dev, a, b, out, EwOp::Mul)
}

/// `rocket_ew_add_fp16`: `out[i] = a[i] + b[i]` on the NPU (the residual primitive).
pub fn ew_add(dev: &Device, a: &[f16], b: &[f16], out: &mut [f16]) -> Result<()> {
    ew_binary(dev, a, b, out, EwOp::Add)
}

/// `rocket_ew_sub_fp16`: `out[i] = a[i] - b[i]` on the NPU.
pub fn ew_sub(dev: &Device, a: &[f16], b: &[f16], out: &mut [f16]) -> Result<()> {
    ew_binary(dev, a, b, out, EwOp::Sub)
}

/// `rocket_ew_max_fp16`: `out[i] = max(a[i], b[i])` on the NPU.
pub fn ew_max(dev: &Device, a: &[f16], b: &[f16], out: &mut [f16]) -> Result<()> {
    ew_binary(dev, a, b, out, EwOp::Max)
}

/// `rocket_ew_min_fp16`: `out[i] = min(a[i], b[i])` on the NPU.
pub fn ew_min(dev: &Device, a: &[f16], b: &[f16], out: &mut [f16]) -> Result<()> {
    ew_binary(dev, a, b, out, EwOp::Min)
}

/// `rocket_clip_fp16`: `min(max(x, lo), hi)` as a constant-operand MAX then MIN.
///
/// Both bounds are rounded to fp16 first (the constant operand the device sees).
pub fn clip(dev: &Device, lo: f32, hi: f32, input: &[f16], out: &mut [f16]) -> Result<()> {
    check_io("rocket_clip_fp16", input, out)?;
    let n = input.len();
    let l = f16::from_f32(lo);
    let cst = vec![l; n];
    ew_max(dev, input, &cst, out)?;
    let h = f16::from_f32(hi);
    let cst = vec![h; n];
    let mut tmp = vec![f16::ZERO; n];
    ew_min(dev, out, &cst, &mut tmp)?;
    out.copy_from_slice(&tmp);
    Ok(())
}

/// `rocket_prelu_fp16`: per-channel negative slope, input `[C][S]`.
///
/// No LUT (so no `x≈0` mux glitch): for the universal `alpha ∈ [0,1]` it is
/// `max(x, alpha_c·x)` (a per-channel EW multiply then an EW max, bit-exact); for any
/// slope outside `[0,1]` the general `relu(x) + alpha_c·min(x,0)` (four passes).
pub fn prelu(
    dev: &Device,
    c: usize,
    s: usize,
    x: &[f16],
    alpha: &[f32],
    out: &mut [f16],
) -> Result<()> {
    if c < 1 || s < 1 {
        return Err(Error::shape("rocket_prelu_fp16"));
    }
    if alpha.len() < c || x.len() < c * s || out.len() < c * s {
        return Err(Error::shape("rocket_prelu_fp16/buffers"));
    }
    let n = c * s;
    let all01 = alpha[..c].iter().all(|a| (0.0..=1.0).contains(a));

    let mut ab = vec![f16::ZERO; n];
    for ch in 0..c {
        let a = f16::from_f32(alpha[ch]);
        ab[ch * s..(ch + 1) * s].fill(a);
    }
    let mut tmp = vec![f16::ZERO; n];
    if all01 {
        ew_mul(dev, &x[..n], &ab, &mut tmp)?; // s = alpha_c * x
        ew_max(dev, &x[..n], &tmp, out)?; // out = max(x, s)
    } else {
        let zeros = vec![f16::ZERO; n];
        let mut p = vec![f16::ZERO; n];
        ew_max(dev, &x[..n], &zeros, &mut p)?; // p = relu(x)
        let mut neg = vec![f16::ZERO; n];
        ew_sub(dev, &x[..n], &p, &mut neg)?; // neg = x - p = min(x, 0)
        ew_mul(dev, &neg, &ab, &mut tmp)?; // tmp = alpha_c*min(x,0)
        ew_add(dev, &p, &tmp, out)?; // out = p + alpha*neg
    }
    Ok(())
}

/// `rocket_ew_div_fp16`: `out = a / b` as `reciprocal(b)` (the LUT) then `a * recip`.
///
/// `b` must be positive and inside the reciprocal LUT domain (default `[0.25, 32]`,
/// `ROCKET_LUT_XLO`/`_XHI` override) — this is an fp16 LUT approximation.
pub fn div(dev: &Device, a: &[f16], b: &[f16], out: &mut [f16]) -> Result<()> {
    check_io("rocket_ew_div_fp16", a, out)?;
    if b.len() != a.len() {
        return Err(Error::shape("rocket_ew_div_fp16/b"));
    }
    let mut recip = vec![f16::ZERO; a.len()];
    activation(dev, Activation::Reciprocal, b, &mut recip)?;
    ew_mul(dev, a, &recip, out)
}

/// The host model of [`leaky_relu`]: the table interpolation plus the host repair of
/// the `x≈0` mux band (the same band and value the device patches).
pub fn leaky_relu_model(alpha: f32, input: &[f16], out: &mut [f16]) -> Result<()> {
    check_io("leaky_relu_model", input, out)?;
    let job = leaky_job(alpha)?;
    let guard = (leaky_radius() / 1024.0) as f32;
    for (o, x) in out.iter_mut().zip(input) {
        let xv = x.to_f32();
        *o = if xv <= guard && xv >= -guard {
            f16::from_f32(if xv >= 0.0 { xv } else { alpha * xv })
        } else {
            f16::from_f32(lut_apply(&job.ep, &job.lut, xv))
        };
    }
    Ok(())
}

/// The host model of [`elu`] / [`selu`] (`alpha`, `lambda`), including the `x≈0`
/// repair band.
fn elu_model(alpha: f64, lambda: f64, input: &[f16], out: &mut [f16]) -> Result<()> {
    check_io("elu_model", input, out)?;
    let job = elu_job(alpha, lambda)?;
    let guard = (elu_radius() / 1024.0) as f32;
    for (o, x) in out.iter_mut().zip(input) {
        let xv = x.to_f32();
        *o = if xv <= guard && xv >= -guard {
            f16::from_f64(elu_d(alpha, lambda, xv as f64))
        } else {
            f16::from_f32(lut_apply(&job.ep, &job.lut, xv))
        };
    }
    Ok(())
}

/// The host model of [`elu`].
pub fn elu_model_alpha(alpha: f32, input: &[f16], out: &mut [f16]) -> Result<()> {
    elu_model(alpha as f64, 1.0, input, out)
}

/// The host model of [`selu`].
pub fn selu_model(input: &[f16], out: &mut [f16]) -> Result<()> {
    elu_model(SELU_ALPHA, SELU_LAMBDA, input, out)
}

/// The host model of an EW binary op: the same fp16 arithmetic the device performs
/// (bit-exact, so this is the assertion reference).
pub fn ew_model(op: EwOp, a: &[f16], b: &[f16], out: &mut [f16]) {
    ew_ref(op, a, b, out);
}

// ── Host references (`*_ref_fp16`) ────────────────────────────────────────────

/// `rocket_activation_ref_fp16`: the exact function, one RNE rounding to fp16.
pub fn activation_ref(kind: Activation, input: &[f16], out: &mut [f16]) {
    for (o, x) in out.iter_mut().zip(input) {
        *o = f16::from_f64(eval(kind, x.to_f32() as f64));
    }
}

/// `rocket_leaky_relu_ref_fp16`.
pub fn leaky_relu_ref(alpha: f32, input: &[f16], out: &mut [f16]) {
    for (o, x) in out.iter_mut().zip(input) {
        let x = x.to_f32();
        *o = f16::from_f32(if x >= 0.0 { x } else { alpha * x });
    }
}

/// `rocket_elu_ref_fp16`.
pub fn elu_ref(alpha: f32, input: &[f16], out: &mut [f16]) {
    for (o, x) in out.iter_mut().zip(input) {
        *o = f16::from_f64(elu_d(alpha as f64, 1.0, x.to_f32() as f64));
    }
}

/// `rocket_selu_ref_fp16`.
pub fn selu_ref(input: &[f16], out: &mut [f16]) {
    for (o, x) in out.iter_mut().zip(input) {
        *o = f16::from_f64(elu_d(SELU_ALPHA, SELU_LAMBDA, x.to_f32() as f64));
    }
}

/// `rocket_prelu_ref_fp16`: the fp16-rounded slope, matching the NPU broadcast.
pub fn prelu_ref(c: usize, s: usize, x: &[f16], alpha: &[f32], out: &mut [f16]) -> Result<()> {
    if c < 1 || s < 1 {
        return Err(Error::shape("rocket_prelu_ref_fp16"));
    }
    if alpha.len() < c || x.len() < c * s || out.len() < c * s {
        return Err(Error::shape("rocket_prelu_ref_fp16/buffers"));
    }
    for ch in 0..c {
        let a = f16::from_f32(alpha[ch]).to_f32();
        for j in 0..s {
            let v = x[ch * s + j].to_f32();
            out[ch * s + j] = f16::from_f32(if v >= 0.0 { v } else { a * v });
        }
    }
    Ok(())
}

/// The host reference of an EW binary op (the device paths are bit-exact).
pub fn ew_ref(op: EwOp, a: &[f16], b: &[f16], out: &mut [f16]) {
    for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
        let (x, y) = (x.to_f32(), y.to_f32());
        *o = f16::from_f32(match op {
            EwOp::Add => x + y,
            EwOp::Mul => x * y,
            EwOp::Sub => x - y,
            EwOp::Max => {
                if x > y {
                    x
                } else {
                    y
                }
            }
            EwOp::Min => {
                if x < y {
                    x
                } else {
                    y
                }
            }
        });
    }
}

/// `rocket_ew_mul_fp16`'s host reference.
pub fn ew_mul_ref(a: &[f16], b: &[f16], out: &mut [f16]) {
    ew_ref(EwOp::Mul, a, b, out);
}

/// `rocket_ew_add_fp16`'s host reference.
pub fn ew_add_ref(a: &[f16], b: &[f16], out: &mut [f16]) {
    ew_ref(EwOp::Add, a, b, out);
}

/// `rocket_ew_sub_fp16`'s host reference.
pub fn ew_sub_ref(a: &[f16], b: &[f16], out: &mut [f16]) {
    ew_ref(EwOp::Sub, a, b, out);
}

/// `rocket_ew_max_fp16`'s host reference.
pub fn ew_max_ref(a: &[f16], b: &[f16], out: &mut [f16]) {
    ew_ref(EwOp::Max, a, b, out);
}

/// `rocket_ew_min_fp16`'s host reference.
pub fn ew_min_ref(a: &[f16], b: &[f16], out: &mut [f16]) {
    ew_ref(EwOp::Min, a, b, out);
}

/// `rocket_clip_fp16`'s host reference (`fd < 0` in C): both bounds rounded to fp16.
pub fn clip_ref(lo: f32, hi: f32, input: &[f16], out: &mut [f16]) {
    let l = f16::from_f32(lo);
    let h = f16::from_f32(hi);
    for (o, x) in out.iter_mut().zip(input) {
        *o = if *x < l {
            l
        } else if *x > h {
            h
        } else {
            *x
        };
    }
}

/// `rocket_ew_div_fp16`'s host reference (`fd < 0` in C): an exact divide.
pub fn div_ref(a: &[f16], b: &[f16], out: &mut [f16]) {
    for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
        *o = f16::from_f32(x.to_f32() / y.to_f32());
    }
}

// ── The text dump compared against the C builder ──────────────────────────────

/// A stable text dump of every `rocket_lut_epilogue_build` result (`kind 0..=18`),
/// byte-comparable with the C builder's output (see `activation_selftest --dump-lut`).
pub fn dump_epilogues() -> String {
    let mut s = String::new();
    for kind in Activation::ALL {
        let mut lut = LutTable::zeroed();
        match lut_epilogue_build(kind, &mut lut) {
            Ok(ep) => {
                s.push_str(&format!("kind {} ret 0\n", kind.code()));
                s.push_str(&format!(
                    "ep {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n",
                    ep.kind,
                    ep.bn_mul_operand,
                    ep.bn_alu_cfg,
                    ep.out_cvt_offset,
                    ep.out_cvt_scale,
                    ep.out_cvt_minus_exp,
                    ep.out_cvt_cvt_type,
                    ep.lut_le_start,
                    ep.lut_lo_end,
                    ep.le_slope_scale,
                    ep.le_slope_shift,
                    ep.lo_slope_scale,
                    ep.lo_slope_shift,
                    ep.le_index_select,
                    ep.lo_index_select
                ));
                s.push_str("lut");
                for w in lut.words() {
                    s.push_str(&format!(" {w}"));
                }
                s.push('\n');
            }
            Err(_) => s.push_str(&format!("kind {} ret -5\n", kind.code())),
        }
    }
    s
}

/// FNV-1a of [`dump_epilogues`] under the default knobs, as produced by the C
/// `rocket_lut_epilogue_build` (byte-compared on the host, see `AGENTS.md`).
#[cfg(test)]
const DUMP_FNV: u64 = 17528923911139717435;

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Every test here builds LUTs, and the builders read the process-wide
    /// `ROCKET_LUT_*` knobs, so the crate's env-touching tests run one at a time.
    fn env_guard() -> std::sync::MutexGuard<'static, ()> {
        crate::knobs::TEST_ENV
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    fn ramp(n: usize, x0: f32, x1: f32) -> Vec<f16> {
        (0..n)
            .map(|i| {
                let t = i as f32 / (n.max(2) - 1) as f32;
                f16::from_f32(x0 + (x1 - x0) * t)
            })
            .collect()
    }

    #[test]
    fn kinds_match_the_c_enum() {
        let _g = env_guard();
        for (code, kind) in Activation::ALL.iter().enumerate() {
            assert_eq!(kind.code(), code as i32);
            assert_eq!(Activation::from_code(code as i32), Some(*kind));
        }
        assert_eq!(Activation::from_code(19), None);
        assert_eq!(Activation::from_code(-1), None);
        assert_eq!(Activation::LeakyRelu.name(), "leaky_relu");
        assert_eq!(Activation::HardSwish.name(), "hardswish");
    }

    #[test]
    fn unit_lut_geometry_is_pinned() {
        let _g = env_guard();
        // sigmoid on the validated grid: x = +-6.31 at the table ends, 0 in the middle.
        let job = unit_job(Activation::Sigmoid);
        assert_eq!(job.ep.bn_mul_operand, 0x6912); // fp16(2596)
        assert_eq!(job.ep.out_cvt_offset, 1);
        assert_eq!(job.ep.out_cvt_minus_exp, 15);
        assert_eq!(job.ep.le_index_select, 5);
        assert_eq!(job.ep.lut_le_start, 0xffff_c000);
        assert_eq!(job.ep.lut_lo_end, 0x4000);
        // sigmoid(0) = 0.5 -> 16384, at both LE[512] and LO[0].
        assert_eq!(job.lut.le()[512], 16384);
        assert_eq!(job.lut.lo()[0], 16384);
        // The ends are +-6.31: sigmoid(-6.31) = 0.00181 -> 59, sigmoid(6.31) -> 32709.
        assert_eq!(job.lut.le()[0], 59);
        assert_eq!(job.lut.lo()[512], 32709);
        // The curve is monotone and the ends are the extremes.
        for pair in job.lut.words().windows(2) {
            assert!(pair[0] <= pair[1], "unit LUT must be monotone");
        }
    }

    #[test]
    fn hardsigmoid_unit_lut_is_exact_at_the_breakpoints() {
        let _g = env_guard();
        let job = unit_job(Activation::HardSigmoid);
        let step = 32.0f32 / 2596.0;
        // x = 0 -> 0.5, and the saturation at x = +-3 is the first fully clipped sample.
        assert_eq!(job.lut.le()[512], 16384);
        let j = (3.0 / step).ceil() as usize; // first sample at x >= 3
        assert_eq!(job.lut.lo()[j], 32767);
        assert!(job.lut.lo()[j - 1] < 32767);
        assert_eq!(job.lut.le()[512 - j], 0);
        assert!(job.lut.le()[512 - j + 1] > 0);
        // The -2 region: clip(-x/6+0.5) ~ 0.1667 -> 5461 (sample j=350 is x=-1.997).
        assert!((job.lut.le()[350] as i32 - 5478).abs() <= 1);
    }

    #[test]
    fn affine_luts_match_the_c_builder() {
        let _g = env_guard();
        // The pinned epilogue fields and table words the C builder prints for
        // TANH/SILU/HARDSWISH (byte-compared, see `dump_epilogues`).
        let mut lut = LutTable::zeroed();
        let ep = lut_epilogue_build(Activation::Tanh, &mut lut).unwrap();
        assert_eq!((ep.bn_mul_operand, ep.bn_alu_cfg), (27928, 0x8000_0000));
        assert_eq!((ep.out_cvt_offset, ep.out_cvt_minus_exp), (4294950912, 14));
        assert_eq!((ep.le_slope_scale, ep.le_slope_shift), (23107, 22));
        // tanh(0) = 0 -> (0 - lo)/span = 0.5 -> 16384, at the middle sample.
        assert_eq!(lut.le()[512], 16384);
        assert_eq!(lut.lo()[0], 16384);
        // The table covers +-3.14 (the LE/LO split), so the ends are tanh(+-3.14).
        assert_eq!(lut.le()[0], 61);
        assert_eq!(lut.lo()[512], 32707);

        let mut lut = LutTable::zeroed();
        let ep = lut_epilogue_build(Activation::Silu, &mut lut).unwrap();
        assert_eq!((ep.bn_mul_operand, ep.out_cvt_minus_exp), (25600, 10));
        assert_eq!((ep.out_cvt_offset, ep.le_slope_scale), (4294966784, 0));
        // silu(0) = 0 -> g = (0 - lo)/S = 0.5/32 -> 512 (the table's zero line).
        assert_eq!(lut.le()[512], 512);
        // silu(+-6.31) with S=32: -0.0115 -> 512, +6.31 -> 16896.
        assert_eq!(lut.le()[0], 512);
        assert_eq!(lut.lo()[512], 16896);

        let mut lut = LutTable::zeroed();
        let ep = lut_epilogue_build(Activation::HardSwish, &mut lut).unwrap();
        assert_eq!((ep.bn_mul_operand, ep.out_cvt_minus_exp), (27989, 13));
        assert_eq!((ep.out_cvt_offset, ep.le_slope_scale), (4294963200, 0));
        // The knee-only table covers [-3,3], where hardswish is exactly 0 below -3
        // (so the LE half is flat at the 0 point, q = 4096 for lo = -0.5, S = 4).
        assert_eq!(lut.le()[0], 4096);
        assert_eq!(lut.le()[512], 4096);
        assert_eq!(lut.lo()[512], 28674); // hardswish(3) = 3 -> (3.5)/4
    }

    #[test]
    fn shifted_luts_match_the_c_builder() {
        let _g = env_guard();
        // SQRT: x in [0.25, 64], out [0,8], S=8.
        let mut lut = LutTable::zeroed();
        let ep = lut_epilogue_build(Activation::Sqrt, &mut lut).unwrap();
        assert_eq!(ep.bn_mul_operand, 23556);
        assert_eq!(ep.bn_alu_cfg, 3263201409);
        assert_eq!((ep.out_cvt_offset, ep.out_cvt_minus_exp), (0, 12));
        // LE is flat = LO[0]; sqrt(0.25) = 0.5 -> q = 0.5/8*32768 = 2048.
        assert_eq!(lut.le()[0], 2048);
        assert_eq!(lut.lo()[0], 2048);
        assert!(lut.le().iter().all(|&w| w == lut.le()[0]));
        // LO[512] = sqrt(64) = 8 -> q = 8/8*32768 = 32768, clamped to 32767.
        assert_eq!(lut.lo()[512], 32767);
        // Monotone non-decreasing throughout.
        for pair in lut.lo().windows(2) {
            assert!(pair[0] <= pair[1]);
        }

        // RSQRT: out [0.125, 1.414], S=2 -> 1/sqrt(0.5) = 1.4142 -> 23170.
        let mut lut = LutTable::zeroed();
        let ep = lut_epilogue_build(Activation::Rsqrt, &mut lut).unwrap();
        assert_eq!((ep.bn_mul_operand, ep.out_cvt_minus_exp), (23560, 14));
        assert_eq!(lut.lo()[0], 23170);
        assert_eq!(lut.lo()[512], 2048); // 1/sqrt(64) = 0.125 -> 0.125/2*32768
        for pair in lut.lo().windows(2) {
            assert!(pair[0] >= pair[1]); // decreasing
        }

        // RECIPROCAL: 1/0.25 = 4 saturates the top of the S=4 range.
        let mut lut = LutTable::zeroed();
        let ep = lut_epilogue_build(Activation::Reciprocal, &mut lut).unwrap();
        assert_eq!((ep.bn_mul_operand, ep.out_cvt_minus_exp), (24584, 13));
        assert_eq!(lut.lo()[0], 32767);
        assert_eq!(lut.lo()[512], 256); // 1/32 = 0.03125 -> /4*32768

        // ABS: the symmetric domain puts the kink on the middle sample exactly.
        let mut lut = LutTable::zeroed();
        let ep = lut_epilogue_build(Activation::Abs, &mut lut).unwrap();
        assert_eq!((ep.bn_mul_operand, ep.bn_alu_cfg), (24576, 1174405120));
        // |0| = 0 -> q = 0, floored to 1 (a q=0 entry mis-decodes on HW); the kink
        // sample is exact apart from that floor.
        assert_eq!(lut.lo()[256], 1);
        // The domain is [-16,16] and S=32, so the top sample |16| is 16/32 -> 16384.
        assert_eq!(lut.lo()[512], 16384);
        // The left half mirrors the right (|x| is even).
        for j in 0..256 {
            assert_eq!(lut.lo()[j], lut.lo()[512 - j]);
        }

        // LOG: signed output decoded by the OUT_CVT offset.
        let mut lut = LutTable::zeroed();
        let ep = lut_epilogue_build(Activation::Log, &mut lut).unwrap();
        assert_eq!(ep.bn_mul_operand, 24584);
        assert_eq!(ep.out_cvt_minus_exp, 12);
        assert_eq!(ep.out_cvt_offset, 4294961618); // lround(ln(0.25)*4096)
        // log(0.25) -> out_lo exactly -> q = 1 (the floor).
        assert_eq!(lut.lo()[0], 1);
        // log(32) = 3.4657 -> (3.4657 + 1.3863)/8 -> 19874.
        assert_eq!(lut.lo()[512], 19874);

        // EXP: default softmax domain [-16,0], S=1: exp(0) = 1 -> 32767 (clamped).
        let mut lut = LutTable::zeroed();
        let ep = lut_epilogue_build(Activation::Exp, &mut lut).unwrap();
        assert_eq!((ep.bn_mul_operand, ep.bn_alu_cfg), (25600, 1182793728));
        assert_eq!(ep.out_cvt_minus_exp, 15);
        assert_eq!(lut.lo()[512], 32767);
        assert_eq!(lut.lo()[0], 1); // exp(-16) ~ 1e-7 -> floored
        // SOFTPLUS over [-16,16], S=32: softplus(16) ~ 16 -> q ~ 16384.
        let mut lut = LutTable::zeroed();
        let ep = lut_epilogue_build(Activation::Softplus, &mut lut).unwrap();
        assert_eq!((ep.bn_mul_operand, ep.out_cvt_minus_exp), (24576, 10));
        assert_eq!(lut.lo()[512], 16384);
        assert_eq!(lut.lo()[0], 1);
    }

    #[test]
    fn unsupported_kinds_report_unsupported() {
        let _g = env_guard();
        for kind in [
            Activation::Sigmoid,
            Activation::HardSigmoid,
            Activation::GeluGate,
            Activation::MishGate,
            Activation::LeakyRelu,
            Activation::Elu,
            Activation::Selu,
        ] {
            let mut lut = LutTable::zeroed();
            let e = lut_epilogue_build(kind, &mut lut).unwrap_err();
            assert_eq!(e.status(), crate::Status::Unsupported, "{kind}");
        }
        // The shifted-domain and affine kinds are all supported.
        for kind in [
            Activation::Sqrt,
            Activation::Rsqrt,
            Activation::Reciprocal,
            Activation::Log,
            Activation::Exp,
            Activation::Softplus,
            Activation::Abs,
            Activation::Tanh,
            Activation::Silu,
            Activation::Gelu,
            Activation::HardSwish,
        ] {
            let mut lut = LutTable::zeroed();
            assert!(lut_epilogue_build(kind, &mut lut).is_ok(), "{kind}");
        }
    }

    #[test]
    fn references_match_known_identities() {
        let _g = env_guard();
        // relu-style identities for the exact refs.
        let x = [f16::from_f32(-1.0), f16::from_f32(0.0), f16::from_f32(1.0)];
        let mut out = [f16::ZERO; 3];
        leaky_relu_ref(0.1, &x, &mut out);
        assert_eq!(out[0].to_f32(), f16::from_f32(-0.1).to_f32());
        assert_eq!(out[1].to_f32(), 0.0);
        assert_eq!(out[2].to_f32(), 1.0);

        // sigmoid(0) = 0.5, hardsigmoid(0) = 0.5, tanh(0) = 0, gelu(-0.75) ~ -0.1700.
        let z = [f16::ZERO];
        let mut o = [f16::ZERO; 1];
        activation_ref(Activation::Sigmoid, &z, &mut o);
        assert_eq!(o[0].to_f32(), 0.5);
        activation_ref(Activation::HardSigmoid, &z, &mut o);
        assert_eq!(o[0].to_f32(), 0.5);
        activation_ref(Activation::Tanh, &z, &mut o);
        assert_eq!(o[0].to_f32(), 0.0);
        let g = [f16::from_f32(-0.75)];
        activation_ref(Activation::Gelu, &g, &mut o);
        assert!((o[0].to_f32() - (-0.1700)).abs() < 2e-3, "{}", o[0]);

        // selu(0) = 0, elu(0) = 0, elu(-1) = alpha*(e^-1 - 1) = -0.6321*alpha.
        let mut o = [f16::ZERO; 1];
        selu_ref(&z, &mut o);
        assert_eq!(o[0].to_f32(), 0.0);
        elu_ref(0.5, &z, &mut o);
        assert_eq!(o[0].to_f32(), 0.0);
        let m1 = [f16::from_f32(-1.0)];
        elu_ref(0.5, &m1, &mut o);
        assert!((o[0].to_f32() - 0.5 * (1.0f32.exp().recip() - 1.0)).abs() < 2e-3);
        // SELU's negative saturation is -lambda*alpha ~ -1.7581.
        let big = [f16::from_f32(-30.0)];
        selu_ref(&big, &mut o);
        assert!((o[0].to_f32() + 1.7581).abs() < 1e-3, "{}", o[0]);

        // prelu channel mapping: the alpha of the channel each element belongs to.
        let x = [
            f16::from_f32(2.0),
            f16::from_f32(-2.0),
            f16::from_f32(-4.0),
            f16::from_f32(4.0),
        ];
        let alpha = [0.25f32, 0.5];
        let mut out = [f16::ZERO; 4];
        prelu_ref(2, 2, &x, &alpha, &mut out).unwrap();
        assert_eq!(out[0].to_f32(), 2.0);
        assert_eq!(out[1].to_f32(), -0.5); // 0.25 * -2
        assert_eq!(out[2].to_f32(), -2.0); // 0.5 * -4
        assert_eq!(out[3].to_f32(), 4.0);
    }

    #[test]
    fn ew_refs_are_the_obvious_arithmetic() {
        let _g = env_guard();
        let a = [f16::from_f32(2.0), f16::from_f32(-3.5), f16::from_f32(0.5)];
        let b = [f16::from_f32(4.0), f16::from_f32(2.0), f16::from_f32(-0.25)];
        let mut out = [f16::ZERO; 3];
        ew_mul_ref(&a, &b, &mut out);
        assert_eq!(
            out.iter().map(|v| v.to_f32()).collect::<Vec<_>>(),
            vec![8.0, -7.0, -0.125]
        );
        ew_add_ref(&a, &b, &mut out);
        assert_eq!(
            out.iter().map(|v| v.to_f32()).collect::<Vec<_>>(),
            vec![6.0, -1.5, 0.25]
        );
        ew_sub_ref(&a, &b, &mut out);
        assert_eq!(
            out.iter().map(|v| v.to_f32()).collect::<Vec<_>>(),
            vec![-2.0, -5.5, 0.75]
        );
        ew_max_ref(&a, &b, &mut out);
        assert_eq!(
            out.iter().map(|v| v.to_f32()).collect::<Vec<_>>(),
            vec![4.0, 2.0, 0.5]
        );
        ew_min_ref(&a, &b, &mut out);
        assert_eq!(
            out.iter().map(|v| v.to_f32()).collect::<Vec<_>>(),
            vec![2.0, -3.5, -0.25]
        );
        // clip rounds both bounds to fp16 first.
        let x = ramp(9, -4.0, 4.0);
        let mut o = vec![f16::ZERO; 9];
        clip_ref(0.1, 2.5, &x, &mut o);
        assert!(o.iter().all(|v| {
            let v = v.to_f32();
            v >= f16::from_f32(0.1).to_f32() && v <= f16::from_f32(2.5).to_f32()
        }));
    }

    #[test]
    fn model_reproduces_the_table_interpolation() {
        let _g = env_guard();
        // sigmoid at x=0 is the middle sample 16384 with the +1 rounding bias, so the
        // model reads 16385/32768 (the C's designed +1 LSB bias, not an error).
        let job = unit_job(Activation::Sigmoid);
        let zero = lut_apply(&job.ep, &job.lut, 0.0);
        assert!((zero - 16385.0 / 32768.0).abs() < 1e-7, "{zero}");
        let step = 32.0 / 2596.0;
        let x = step * 100.25; // a quarter of the way into cell 100
        let lo = job.lut.lo()[100] as f32;
        let hi = job.lut.lo()[101] as f32;
        let got = lut_apply(&job.ep, &job.lut, x);
        let want = (lo + 0.25 * (hi - lo) + 1.0) / 32768.0;
        assert!((got - want).abs() < 1e-6, "{got} vs {want}");
        // Beyond the positive table end the value holds (LO slope 0): the last LO
        // sample plus the +1 bias.
        let top = (job.lut.lo()[512] as f32 + 1.0) / 32768.0;
        assert!((lut_apply(&job.ep, &job.lut, 20.0) - top).abs() < 1e-7);

        // The tanh tail uses the LE slope (23107 >> 22 per index LSB).
        let mut lut = LutTable::zeroed();
        let ep = lut_epilogue_build(Activation::Tanh, &mut lut).unwrap();
        let tail = lut_apply(&ep, &lut, -8.0);
        assert!(tail > -1.02 && tail < -0.98, "{tail}");
    }

    #[test]
    fn model_matches_the_exact_function_inside_the_lut_error() {
        let _g = env_guard();
        // The LUT design error, sampled on the dense grid. Inside the table the
        // interpolation + Q quantization is tiny; outside it the design saturates or
        // extrapolates (the documented trade-off, not a port error).
        let job = unit_job(Activation::Sigmoid);
        let xs = ramp(2048, -6.0, 6.0);
        let mut worst = 0.0f32;
        for &x in &xs {
            let got = lut_apply(&job.ep, &job.lut, x.to_f32());
            let want = eval(Activation::Sigmoid, x.to_f32() as f64) as f32;
            worst = worst.max((got - want).abs());
        }
        assert!(worst < 2e-4, "sigmoid LUT worst {worst}");

        let mut lut = LutTable::zeroed();
        let ep = lut_epilogue_build(Activation::Tanh, &mut lut).unwrap();
        let (lo, hi) = table_domain(Activation::Tanh).unwrap();
        let xs = ramp(2048, lo * 0.98, hi * 0.98);
        let mut worst = 0.0f32;
        for &x in &xs {
            let got = lut_apply(&ep, &lut, x.to_f32());
            let want = eval(Activation::Tanh, x.to_f32() as f64) as f32;
            worst = worst.max((got - want).abs());
        }
        assert!(worst < 5e-4, "tanh LUT worst {worst} (inside the table)");
    }

    #[test]
    fn gated_model_is_x_times_the_gate() {
        let _g = env_guard();
        let x = ramp(64, -8.0, 8.0);
        let mut out = vec![f16::ZERO; x.len()];
        activation_model(Activation::Silu, &x, &mut out).unwrap();
        let job = single_pass_job(Activation::Sigmoid).unwrap();
        for i in 0..x.len() {
            let g = lut_apply(&job.ep, &job.lut, x[i].to_f32());
            let want = f16::from_f32(x[i].to_f32() * g);
            assert_eq!(out[i].to_f32(), want.to_f32());
        }
        // GELU routes through GELU_GATE, Mish through MISH_GATE.
        activation_model(Activation::Mish, &x, &mut out).unwrap();
        let job = single_pass_job(Activation::MishGate).unwrap();
        for i in 0..x.len() {
            let g = lut_apply(&job.ep, &job.lut, x[i].to_f32());
            let want = f16::from_f32(x[i].to_f32() * g);
            assert_eq!(out[i].to_f32(), want.to_f32());
        }
        activation_model(Activation::Gelu, &x, &mut out).unwrap();
        let job = single_pass_job(Activation::GeluGate).unwrap();
        for i in 0..x.len() {
            let g = lut_apply(&job.ep, &job.lut, x[i].to_f32());
            let want = f16::from_f32(x[i].to_f32() * g);
            assert_eq!(out[i].to_f32(), want.to_f32());
        }
    }

    #[test]
    fn n_zero_is_a_shape_error_as_in_c() {
        let _g = env_guard();
        // The C returns ROCKET_E_SHAPE for `n <= 0`; the check happens before any
        // device use, so the device-free model entry exercises the same guard.
        let e = activation_model(Activation::Sigmoid, &[], &mut []).unwrap_err();
        assert_eq!(e.status(), crate::Status::Shape);
        // A length mismatch is rejected too.
        let x = [f16::ZERO; 2];
        let mut out = [f16::ZERO; 3];
        let e = activation_model(Activation::Sigmoid, &x, &mut out).unwrap_err();
        assert_eq!(e.status(), crate::Status::Shape);
    }

    #[test]
    fn table_domains_are_the_documented_ranges() {
        let _g = env_guard();
        let d = |k| table_domain(k).unwrap();
        let (lo, hi) = d(Activation::Sigmoid);
        assert!(
            (lo + 6.31).abs() < 0.01 && (hi - 6.31).abs() < 0.01,
            "{lo} {hi}"
        );
        let (lo, hi) = d(Activation::Tanh);
        assert!(lo < -3.1 && lo > -3.2 && hi > 3.1 && hi < 3.2, "{lo} {hi}");
        // The shifted domains come back through the fp16 BN scale, so they are close
        // rather than exact.
        let near = |a: (f32, f32), b: (f32, f32)| {
            (a.0 - b.0).abs() < 1e-3 * (1.0 + b.0.abs())
                && (a.1 - b.1).abs() < 1e-3 * (1.0 + b.1.abs())
        };
        assert!(
            near(d(Activation::Sqrt), (0.25, 64.0)),
            "{:?}",
            d(Activation::Sqrt)
        );
        assert!(near(d(Activation::Rsqrt), (0.5, 64.0)));
        assert!(near(d(Activation::Reciprocal), (0.25, 32.0)));
        assert!(near(d(Activation::Log), (0.25, 32.0)));
        assert!(near(d(Activation::Exp), (-16.0, 0.0)));
        assert!(near(d(Activation::Softplus), (-16.0, 16.0)));
        assert!(near(d(Activation::Abs), (-16.0, 16.0)));
    }

    #[test]
    fn lut_knobs_override_the_geometry() {
        let _g = env_guard();
        // The shifted-domain builder takes XLO/XHI/SEXP.
        unsafe {
            std::env::set_var("ROCKET_LUT_XLO", "1.0");
            std::env::set_var("ROCKET_LUT_XHI", "4.0");
            std::env::set_var("ROCKET_LUT_SEXP", "2");
        }
        let mut lut = LutTable::zeroed();
        let ep = lut_epilogue_build(Activation::Sqrt, &mut lut).unwrap();
        let scale = 16384.0 / 3.0;
        assert_eq!(ep.bn_mul_operand, fp16_bits(scale));
        assert_eq!(ep.bn_alu_cfg, fp32_bits(-scale));
        assert_eq!(ep.out_cvt_minus_exp, 13); // 15 - log2(4)
        // sqrt(1) = 1 with S=4 -> q = 0.25*32768 = 8192.
        assert_eq!(lut.lo()[0], 8192);
        unsafe {
            std::env::remove_var("ROCKET_LUT_XLO");
            std::env::remove_var("ROCKET_LUT_XHI");
            std::env::remove_var("ROCKET_LUT_SEXP");
        }

        // The affine builder takes IDXSCALE/LO/SEXP and the slope knobs.
        unsafe {
            std::env::set_var("ROCKET_LUT_IDXSCALE", "2048");
            std::env::set_var("ROCKET_LUT_LO", "-0.25");
            std::env::set_var("ROCKET_LUT_LESLOPE", "1234");
            std::env::set_var("ROCKET_LUT_LOSHIFT", "7");
        }
        let mut lut = LutTable::zeroed();
        let ep = lut_epilogue_build(Activation::Silu, &mut lut).unwrap();
        assert_eq!(ep.bn_mul_operand, fp16_bits(2048.0));
        assert_eq!(ep.le_slope_scale, 1234);
        assert_eq!(ep.lo_slope_shift, 7);
        // lo = -0.25 with S = 32 (Sexp unchanged) -> offset = -256.
        assert_eq!(ep.out_cvt_offset as i32, -256);
        unsafe {
            std::env::remove_var("ROCKET_LUT_IDXSCALE");
            std::env::remove_var("ROCKET_LUT_LO");
            std::env::remove_var("ROCKET_LUT_LESLOPE");
            std::env::remove_var("ROCKET_LUT_LOSHIFT");
        }

        // The standalone op's own overrides land on top of the builder's.
        unsafe {
            std::env::set_var("ROCKET_LUT_MINEXP", "9");
            std::env::set_var("ROCKET_LUT_BNMUL", "0x1111");
        }
        let job = single_pass_job(Activation::Tanh).unwrap();
        assert_eq!(job.ep.out_cvt_minus_exp, 9);
        assert_eq!(job.ep.bn_mul_operand, 0x1111);
        unsafe {
            std::env::remove_var("ROCKET_LUT_MINEXP");
            std::env::remove_var("ROCKET_LUT_BNMUL");
        }

        // The unit kinds keep the sigmoid geometry unless the generic knobs override it.
        let job = single_pass_job(Activation::Sigmoid).unwrap();
        assert_eq!(job.ep.out_cvt_minus_exp, 15);
        assert_eq!(job.ep.bn_mul_operand, 0x6912);

        // ROCKET_LUT_SHIFT moves tanh onto the shifted single-table geometry.
        unsafe { std::env::set_var("ROCKET_LUT_SHIFT", "1") };
        let mut lut = LutTable::zeroed();
        let ep = lut_epilogue_build(Activation::Tanh, &mut lut).unwrap();
        assert_eq!(ep.out_cvt_minus_exp, 14); // S = 2
        assert_ne!(ep.bn_alu_cfg, 0x8000_0000); // the index shift is real
        assert_eq!(ep.le_slope_scale, 0); // the shifted table has no LE slope
        assert!(lut.le().iter().all(|&w| w == lut.le()[0]));
        unsafe { std::env::remove_var("ROCKET_LUT_SHIFT") };
    }

    #[test]
    fn knobs_route_the_gated_kinds() {
        let _g = env_guard();
        // Default: SiLU = x * sigmoid(x) (the unit table's gate).
        let x = ramp(16, -3.0, 3.0);
        let mut wide = vec![f16::ZERO; x.len()];
        activation_model(Activation::Silu, &x, &mut wide).unwrap();
        let gate = single_pass_job(Activation::Sigmoid).unwrap();
        for i in 0..x.len() {
            let g = lut_apply(&gate.ep, &gate.lut, x[i].to_f32());
            assert_eq!(wide[i].to_f32(), f16::from_f32(x[i].to_f32() * g).to_f32());
        }
        // ROCKET_ACT_WIDE_LUT: the single-pass SiLU table instead.
        unsafe { std::env::set_var("ROCKET_ACT_WIDE_LUT", "1") };
        let mut single = vec![f16::ZERO; x.len()];
        activation_model(Activation::Silu, &x, &mut single).unwrap();
        let job = single_pass_job(Activation::Silu).unwrap();
        for i in 0..x.len() {
            let want = lut_apply(&job.ep, &job.lut, x[i].to_f32());
            assert_eq!(single[i].to_f32(), f16::from_f32(want).to_f32());
        }
        assert_ne!(wide, single, "the wide-LUT route must differ");
        unsafe { std::env::remove_var("ROCKET_ACT_WIDE_LUT") };
    }

    #[test]
    fn dump_is_stable() {
        let _g = env_guard();
        // FNV-1a over the whole epilogue dump: pins the LUT geometry and the
        // epilogue fields against the C builder (see the module docs).
        let dump = dump_epilogues();
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for b in dump.as_bytes() {
            hash ^= *b as u64;
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
        assert_eq!(hash, DUMP_FNV, "LUT dump drifted from the C builder");
    }
}
