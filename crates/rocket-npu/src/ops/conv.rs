//! fp16 CONV_2D on the RK3588 NPU (direct, depthwise, 1D, and the conv→activation
//! fusion).
//!
//! Port of `rocket_conv.c` (the RK3588 fp16 half; the native int8 entries of that file
//! are not part of this milestone) plus the `rocket_conv_transpose` lowering is in
//! [`crate::ops::conv_transpose`].
//!
//! The NPU CNA is a native convolution engine, so a K×K / stride / pad / dilation conv
//! runs directly (no im2col): the host scatters the input feature into the NC1HWC2 cube
//! (`C2 = 8`) and the weights into the conv weight cube
//! (`(OC/16, IC/32, KH, KW, 16, 32)`, the Mesa-confirmed `oc1/ic1/kh/kw/oc2/ic2`
//! reorder), and [`crate::regcmd::conv`] programs the sliding-window MAC.
//!
//! Tensor layouts (row-major, batch 1):
//!
//! ```text
//! input    in [IC][IH][IW]
//! weights  W  [OC][IC][KH][KW]   (direct);  [OC][1][KH][KW]  (depthwise, OC == IC)
//! output   out[OC][OH][OW]
//! out[oc][oh][ow] = sum_{ic,kh,kw} W[oc][ic][kh][kw] *
//!     in[ic][oh*stride_y + kh*dil_y - pad_top][ow*stride_x + kw*dil_x - pad_left]
//! ```
//!
//! Out-of-range input contributes 0 (that is what the padding means).
//!
//! # Tiling
//!
//! The **direct** path pads `IC` up to 32 (the first RGB layer — `IC < 32` is accepted
//! and zero-padded, so no `IC % 32` alignment is required of the caller) and `OC` up to
//! 16, then tiles over output channels (mult-16) and output **space** (OH rows, then OW
//! columns) when the problem does not fit one CBUF pass. Each spatial tile
//! **materialises its edge padding into the sub-input** — the CNA has only a symmetric
//! `pad_top`/`pad_left`, so a per-band pad would also pad the band's interior bottom —
//! and then runs with `pad_top = pad_left = 0` over real-halo + explicit-zero rows and
//! columns. The row budget never shrinks below a materialised height of 4 rows
//! (`CONV_MIN_DATAIN_H`): the CNA feature DMA is height-blocked in 4-row blocks and a
//! shorter sub-input produces wrong output at every `IC` (a hardware finding), so a
//! budget that a full-width band overflows narrows the **columns** instead.
//!
//! The **depthwise** path tiles over channels in multiples of the depthwise group `G`
//! (32 for fp16, `ROCKET_CONV_DW_GROUP` overrides) — channels are independent, so no
//! halo is involved — and additionally over space (`dw_spatial`) when even one
//! `G`-channel chunk overflows the validated feature budget. The upstream header's
//! "not yet implemented" note about spatial depthwise tiling predates the code that
//! implements it; this port follows the code.
//!
//! # Bands
//!
//! [`balance_bands`] partitions an axis into the *fewest* near-equal bands, so a
//! budget-derived maximum never leaves a remainder stub below the 4-row floor.
//!
//! # Depthwise weight cube
//!
//! Depthwise weights live in a `(C/G, KH, KW, G)` cube with one filter per channel;
//! [`weight_cube_dw_index`] is the pure index function and the emitter must be told the
//! same `G` (both sides read `ROCKET_CONV_DW_GROUP`).
//!
//! # Fused activation
//!
//! [`conv2d_act`] post-processes the conv result inside the same NPU job through the
//! DPU LUT epilogue built by [`crate::ops::activation::lut_epilogue_build`], so
//! `out = f(conv(...))` with no second round-trip. The fused `f` is the same fp16 table
//! interpolation the standalone [`crate::ops::activation::activation`] uses, so the
//! result is an fp16 LUT approximation: gate it with a tolerance. Only the smooth
//! single-pass kinds (`Silu`, `Tanh`, `Gelu`) are accepted (`-11` otherwise, as in C),
//! and depthwise is refused (`-10`) because the fusion is the direct path's.
//!
//! # Env knobs (the C names and semantics, read per call)
//!
//! `ROCKET_CONV_DW_GROUP`, `ROCKET_CONV_DW_FEAT_BANKS`, `ROCKET_CONV_DW_SIZEE`,
//! `ROCKET_CONV_DW_BSOWOP`, `ROCKET_CONV_DW_OCPAD`, `ROCKET_CONV_DW_DEBUG`,
//! `ROCKET_CONV_DW_NATIVE` (accepted no-op), `ROCKET_CONV_MIN_IH`, `ROCKET_CONV_GRAINS`,
//! `ROCKET_CONV_DENTRIES`, `ROCKET_CONV_STRIDE_SWAP`, `ROCKET_CONV_DIL_RAW`, plus the
//! LUT builder's own set through `lut_epilogue_build`.

use half::f16;

use crate::Device;
use crate::device::Bo;
use crate::device::uapi::DrmRocketTask;
use crate::error::{Error, Result};
use crate::knobs;
use crate::log::{log_debug, log_error};
use crate::ops::activation::{Activation, lut_epilogue_build};
use crate::ops::reduce::feature_data;
use crate::regcmd::conv::{ConvAct, ConvParams, gen_conv2d_dw_fp16, gen_conv2d_fp16};
use crate::regcmd::desc::CBUF_BANK_SIZE;

/// Slices a fused conv waits for the output BO (the C's hardcoded 2 s).
const CONV_WAIT_NS: u64 = 2_000_000_000;

/// Feature-tile budget: a tile's `rows*IC*IW*2` must leave room for the weight tile in
/// the 12×32 KiB CBUF. Keeping the feature within ~8 banks leaves the weight ≥ 4; the
/// tilers shrink it further when one OC group's weight cube needs more than 4 banks.
const CONV_FEAT_BUDGET: usize = 8 * CBUF_BANK_SIZE;

/// The CNA feature DMA is height-blocked in 4-row blocks: a materialised sub-input with
/// fewer than 4 rows produces wrong hardware output at every `IC`.
pub const CONV_MIN_DATAIN_H: usize = 4;

/// The C `rocket_conv2d_desc`: shapes and hyperparameters, no buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conv2dDesc {
    /// Input channels / height / width.
    pub ic: usize,
    pub ih: usize,
    pub iw: usize,
    /// Output channels.
    pub oc: usize,
    /// Kernel height / width.
    pub kh: usize,
    pub kw: usize,
    /// Window strides.
    pub stride_y: usize,
    pub stride_x: usize,
    /// Symmetric leading padding.
    pub pad_top: usize,
    pub pad_left: usize,
    /// Dilation rate (`1` = none).
    pub dil_y: usize,
    pub dil_x: usize,
    /// `true` = depthwise (`OC == IC`, one filter per channel).
    pub depthwise: bool,
    /// An explicit output extent, which is how an asymmetric pad is expressed.
    ///
    /// `0` means "whatever [`conv2d_oh`]/[`conv2d_ow`] derive", which is every caller.
    /// The RK3576 int8 entries honour a non-zero value inside a decoded bound; the fp16
    /// and RK3588 paths refuse an extent they did not derive themselves (the CNA's pad
    /// registers have only ever been driven symmetrically here).
    pub oh: usize,
    pub ow: usize,
    /// Run a narrow input-channel count on the direct datapath.
    ///
    /// The RK3576 CNA has a second, packed-image encoding for ≤ 4 channels; the RK3588
    /// paths never take it, so here the flag is satisfied by construction (the direct
    /// path zero-pads `IC < 32` up to 32, exactly as the flag asks) and is accepted.
    pub direct_datapath: bool,
}

/// A zero shape with the neutral hyperparameters (stride 1, dilation 1, no padding,
/// direct, no explicit extent) — so `..Default::default()` in a literal means "a plain
/// convolution".
impl Default for Conv2dDesc {
    fn default() -> Self {
        Self {
            ic: 0,
            ih: 0,
            iw: 0,
            oc: 0,
            kh: 0,
            kw: 0,
            stride_y: 1,
            stride_x: 1,
            pad_top: 0,
            pad_left: 0,
            dil_y: 1,
            dil_x: 1,
            depthwise: false,
            oh: 0,
            ow: 0,
            direct_datapath: false,
        }
    }
}

impl Conv2dDesc {
    /// The plain NCHW convolution descriptor (pads 0, dilation 1, direct).
    pub fn new(ic: usize, ih: usize, iw: usize, oc: usize, kh: usize, kw: usize) -> Self {
        Self {
            ic,
            ih,
            iw,
            oc,
            kh,
            kw,
            ..Default::default()
        }
    }
}

/// The output extent one axis: `(in + 2*pad - (dil*(k-1) + 1)) / stride + 1`.
///
/// Integer division with C semantics; a negative numerator truncates toward zero (the
/// C `int` division), so `(0 + 0 - 3)/1 + 1 = -2` rather than `-1`.
pub fn conv_out_dim(input: usize, k: usize, stride: usize, pad: usize, dil: usize) -> isize {
    let n = input as isize + 2 * pad as isize - (dil as isize * (k as isize - 1) + 1);
    n / stride as isize + 1
}

/// [`conv_out_dim`] clamped to a `usize` (a non-positive extent becomes 0).
fn out_dim_usize(input: usize, k: usize, stride: usize, pad: usize, dil: usize) -> usize {
    let v = conv_out_dim(input, k, stride, pad, dil);
    if v < 0 { 0 } else { v as usize }
}

/// `rocket_conv2d_oh`: the derived (or explicit) output height.
pub fn conv2d_oh(d: &Conv2dDesc) -> usize {
    if d.oh != 0 {
        d.oh
    } else {
        out_dim_usize(d.ih, d.kh, d.stride_y, d.pad_top, d.dil_y)
    }
}

/// `rocket_conv2d_ow`: the derived (or explicit) output width.
pub fn conv2d_ow(d: &Conv2dDesc) -> usize {
    if d.ow != 0 {
        d.ow
    } else {
        out_dim_usize(d.iw, d.kw, d.stride_x, d.pad_left, d.dil_x)
    }
}

/// The pad the last window **consumes** past the plane, which is what the CNA programs.
///
/// Zero for a symmetric descriptor. The fp16 path always runs symmetric pads (it
/// materialises the asymmetric part instead), so this is a pure helper here — it is what
/// a caller needs to decide whether the trailing side needs materialising.
pub fn conv2d_trail_y(d: &Conv2dDesc) -> usize {
    let reach = (conv2d_oh(d) as isize - 1) * d.stride_y as isize
        + d.dil_y.max(1) as isize * (d.kh as isize - 1)
        + 1;
    let have = d.pad_top as isize + d.ih as isize;
    if reach > have {
        (reach - have) as usize
    } else {
        0
    }
}

/// [`conv2d_trail_y`] on the width axis.
pub fn conv2d_trail_x(d: &Conv2dDesc) -> usize {
    let reach = (conv2d_ow(d) as isize - 1) * d.stride_x as isize
        + d.dil_x.max(1) as isize * (d.kw as isize - 1)
        + 1;
    let have = d.pad_left as isize + d.iw as isize;
    if reach > have {
        (reach - have) as usize
    } else {
        0
    }
}

/// The depthwise channel group `G` (`ROCKET_CONV_DW_GROUP`, default 32 for fp16).
///
/// The host weight cube and the generator must agree, so both read this knob.
pub fn conv_dw_group() -> usize {
    match knobs::int("ROCKET_CONV_DW_GROUP") {
        Some(g) if g > 0 => g as usize,
        _ => 32,
    }
}

/// The depthwise feature budget in bytes (`ROCKET_CONV_DW_FEAT_BANKS`, default 8 banks).
///
/// The single-job depthwise feature uses the same CNA input DMA (surface-strided,
/// height-blocked) as the direct path, so a depthwise job is only inside a
/// hardware-validated envelope when its feature stays within the direct path's eight
/// banks; an unbounded 12-bank depthwise feature overflows on hardware.
fn dw_feat_budget() -> usize {
    let banks = match knobs::int("ROCKET_CONV_DW_FEAT_BANKS") {
        Some(v) => v,
        None => (CONV_FEAT_BUDGET / CBUF_BANK_SIZE) as i64,
    };
    let banks = banks.clamp(1, 11) as usize; // leave >= 1 bank for the weight
    banks * CBUF_BANK_SIZE
}

/// Does a depthwise chunk of `cc` channels fit one CBUF pass at the full `IH×IW` feature?
pub fn dw_chunk_fits(cc: usize, ih: usize, iw: usize, kh: usize, kw: usize) -> bool {
    if cc == 0 {
        return false;
    }
    let wpk = kw * kh * cc * 2;
    if wpk > CBUF_BANK_SIZE {
        return false;
    }
    cc * ih * iw * 2 <= dw_feat_budget()
}

/// The materialised sub-input height of `out_rows` output rows.
pub fn datain_h(out_rows: usize, sy: usize, kh: usize, dy: usize) -> usize {
    out_rows.saturating_sub(1) * sy + (kh - 1) * dy + 1
}

/// Partition `total` into the fewest near-equal bands no larger than `maxb`.
///
/// Every band is `total/n` or `total/n + 1`, so a budget-derived maximum never leaves a
/// remainder stub below the 4-row floor. Returns the band sizes, descending.
pub fn balance_bands(total: usize, maxb: usize, sizes: &mut Vec<usize>) -> usize {
    let maxb = maxb.max(1);
    let n = total.div_ceil(maxb).max(1);
    let base = total / n;
    let extra = total % n;
    sizes.clear();
    sizes.extend((0..n).map(|i| base + usize::from(i < extra)));
    n
}

/// `rocket_conv2d_plan`, as the C's raw return code (`0`, `-1`..`-4`).
///
/// The classes are the C's: `-1` a malformed descriptor (non-positive dims/strides/
/// dilation, or an explicit output extent the generator did not derive), `-2` a
/// depthwise mismatch (`OC != IC` or `IC % G`), `-3` a non-positive derived output
/// extent, `-4` a tile that cannot fit the CBUF geometry even at its minimum.
pub fn conv2d_plan_code(d: &Conv2dDesc) -> i32 {
    if d.ic == 0 || d.ih == 0 || d.iw == 0 || d.oc == 0 || d.kh == 0 || d.kw == 0 {
        return -1;
    }
    if d.stride_x == 0 || d.stride_y == 0 || d.dil_x == 0 || d.dil_y == 0 {
        return -1;
    }
    // An explicit output extent is the RK3576 int8 path's trailing-pad request; this
    // planner serves the RK3588 fp16 generator, so an extent it did not derive is
    // refused rather than programmed on an untested encoding.
    if (d.oh != 0 && d.oh != out_dim_usize(d.ih, d.kh, d.stride_y, d.pad_top, d.dil_y))
        || (d.ow != 0 && d.ow != out_dim_usize(d.iw, d.kw, d.stride_x, d.pad_left, d.dil_x))
    {
        return -1;
    }
    if d.depthwise && d.oc != d.ic {
        return -2;
    }
    if d.depthwise && !d.ic.is_multiple_of(conv_dw_group()) {
        return -2;
    }
    // Direct: IC < 32 (the RGB first layer) is zero-padded to 32 by the driver, so any
    // IC is accepted; OC need not be a multiple of 16 either (it is padded, not
    // refused).
    let ic = if d.depthwise {
        d.ic
    } else {
        d.ic.div_ceil(32) * 32
    };
    let oh = conv2d_oh(d);
    let ow = conv2d_ow(d);
    if oh == 0 || ow == 0 {
        return -3;
    }
    if d.depthwise {
        // Depthwise tiles over channels and, when even one group overflows, over space,
        // so only one `G`-channel weight cube in a bank and the minimal spatial tile
        // must fit.
        let g = conv_dw_group();
        let wpk = d.kw * d.kh * g * 2;
        if wpk > CBUF_BANK_SIZE {
            return -4;
        }
        let ih_min = (d.kh - 1) * d.dil_y + 1;
        let iw_min = (d.kw - 1) * d.dil_x + 1;
        if g * ih_min * iw_min * 2 > dw_feat_budget() {
            return -4;
        }
    } else {
        let wpk = d.kw * d.kh * ic * 2; // one OC kernel
        if wpk > CBUF_BANK_SIZE {
            return -4;
        }
        let ih_min = (d.kh - 1) * d.dil_y + 1;
        let iw_min = (d.kw - 1) * d.dil_x + 1;
        if ic * ih_min * iw_min * 2 > CONV_FEAT_BUDGET {
            return -4;
        }
    }
    0
}

/// `rocket_conv2d_plan`: validate a descriptor against the supported set.
///
/// Pure — no hardware, no allocation. The C's negated reason becomes a
/// [`crate::Status`] (`-1`/`-3` → [`crate::Status::Shape`], `-2`/`-4` →
/// [`crate::Status::Tiling`]); use [`conv2d_plan_code`] when the exact C code matters.
pub fn conv2d_plan(d: &Conv2dDesc) -> Result<()> {
    match conv2d_plan_code(d) {
        0 => Ok(()),
        -2 => Err(Error::tiling("rocket_conv2d_plan/depthwise")),
        -3 => Err(Error::shape("rocket_conv2d_plan/out-extent")),
        -4 => Err(Error::tiling("rocket_conv2d_plan/cbuf")),
        _ => Err(Error::shape("rocket_conv2d_plan")),
    }
}

// ── Host cube index math ──────────────────────────────────────────────────────

/// The conv weight-cube element index (`weight_conv_fp16`): the Mesa-confirmed
/// `oc1/ic1/kh/kw/oc2/ic2` reorder with `oc` group 16 and `ic` group 32.
///
/// `oc`/`ic`/`kh_i`/`kw_i` are 1-based. At `KH = KW = 1` this collapses to the matmul
/// weight layout.
pub fn weight_cube_index(
    ic_total: usize,
    kh: usize,
    kw: usize,
    oc: usize,
    ic: usize,
    kh_i: usize,
    kw_i: usize,
) -> usize {
    let oc1 = (oc - 1) / 16;
    let oc2 = (oc - 1) % 16;
    let ic1 = (ic - 1) / 32;
    let ic2 = (ic - 1) % 32;
    let n_ic1 = ic_total.div_ceil(32);
    ((((oc1 * n_ic1 + ic1) * kh + (kh_i - 1)) * kw + (kw_i - 1)) * 16 + oc2) * 32 + ic2
}

/// The depthwise weight-cube element index (`weight_conv_dw_fp16`):
/// `(C/G, KH, KW, G)`, one filter per channel, `kh` outer / `kw` inner.
pub fn weight_cube_dw_index(
    kh: usize,
    kw: usize,
    g: usize,
    c: usize,
    kh_i: usize,
    kw_i: usize,
) -> usize {
    let ic1 = (c - 1) / g;
    let ic2 = (c - 1) % g;
    ((ic1 * kh + (kh_i - 1)) * kw + (kw_i - 1)) * g + ic2
}

/// The NC1HWC2 feature/output-cube element index for the fp16 atom.
///
/// This is the C `feature_data` (re-exported from [`crate::ops::reduce`], which already
/// ports the same index math for the PPU path): `C2 = 8` for the fp16 feature and
/// output cubes.
pub fn feature_cube_index(
    channels: usize,
    pix_h: usize,
    pix_w: usize,
    ch: usize,
    row: usize,
    col: usize,
) -> usize {
    feature_data(channels, pix_h, pix_w, 8, ch, row, col)
}

/// Scatter a row-major `[c][h][w]` feature into an NC1HWC2 cube (`C2 = 8`).
///
/// `cube` must hold `ceil(c/8)*8*h*w` elements; it is not zeroed here (the caller
/// zeroes, exactly as the C job does with its `memset`).
pub fn scatter_feature(cube: &mut [f16], c: usize, h: usize, w: usize, src: &[f16]) {
    for ch in 0..c {
        for row in 0..h {
            for col in 0..w {
                cube[feature_cube_index(c, h, w, ch + 1, row + 1, col + 1)] =
                    src[(ch * h + row) * w + col];
            }
        }
    }
}

/// Gather an NC1HWC2 output cube (`C2 = 8`) back to row-major `[oc][oh][ow]`.
pub fn gather_output(cube: &[f16], oc: usize, oh: usize, ow: usize, out: &mut [f16]) {
    for ch in 0..oc {
        for row in 0..oh {
            for col in 0..ow {
                out[(ch * oh + row) * ow + col] =
                    cube[feature_cube_index(oc, oh, ow, ch + 1, row + 1, col + 1)];
            }
        }
    }
}

/// Scatter row-major direct weights `[oc][ic][kh][kw]` into the conv weight cube.
///
/// The cube is `(OC/16, ceil(IC/32), KH, KW, 16, 32)`; `oc` must be a multiple of 16
/// and `ic` is zero-padded up to 32 by the caller (`cube` pre-zeroed).
pub fn scatter_weights(cube: &mut [f16], oc: usize, ic: usize, kh: usize, kw: usize, w: &[f16]) {
    for o in 0..oc {
        for k in 0..kh {
            for j in 0..kw {
                for i in 0..ic {
                    cube[weight_cube_index(ic, kh, kw, o + 1, i + 1, k + 1, j + 1)] =
                        w[((o * ic + i) * kh + k) * kw + j];
                }
            }
        }
    }
}

/// Scatter row-major depthwise weights `[c][kh][kw]` into the `(C/G, KH, KW, G)` cube.
pub fn scatter_weights_dw(cube: &mut [f16], c: usize, kh: usize, kw: usize, g: usize, w: &[f16]) {
    for ch in 0..c {
        for k in 0..kh {
            for j in 0..kw {
                cube[weight_cube_dw_index(kh, kw, g, ch + 1, k + 1, j + 1)] =
                    w[(ch * kh + k) * kw + j];
            }
        }
    }
}

// ── Host reference ────────────────────────────────────────────────────────────

/// `rocket_conv2d_ref_fp16`: the fp32-accumulate scatter reference (the golden oracle
/// and the host fallback).
///
/// Writes `out[OC*OH*OW]` as fp16 (fp32 accumulate, fp16 store), matching the NPU's
/// fp16-narrowed output. Depthwise reduces only each output channel's own input channel
/// with the `[OC][1][KH][KW]` weight layout; out-of-range input contributes 0.
pub fn conv2d_ref(d: &Conv2dDesc, input: &[f16], w: &[f16], out: &mut [f16]) {
    let (oh, ow) = (conv2d_oh(d), conv2d_ow(d));
    let (ic, ih, iw, oc, kh, kw) = (d.ic, d.ih, d.iw, d.oc, d.kh, d.kw);
    for o in 0..oc {
        for y in 0..oh {
            for x in 0..ow {
                let mut s = 0.0f32;
                let (c_lo, c_hi) = if d.depthwise { (o, o + 1) } else { (0, ic) };
                for c in c_lo..c_hi {
                    let wic = if d.depthwise { 0 } else { c };
                    let wspan = if d.depthwise { 1 } else { ic };
                    for k in 0..kh {
                        let yy = y * d.stride_y + k * d.dil_y;
                        if yy < d.pad_top {
                            continue;
                        }
                        let yy = yy - d.pad_top;
                        if yy >= ih {
                            continue;
                        }
                        for j in 0..kw {
                            let xx = x * d.stride_x + j * d.dil_x;
                            if xx < d.pad_left {
                                continue;
                            }
                            let xx = xx - d.pad_left;
                            if xx >= iw {
                                continue;
                            }
                            let a = input[(c * ih + yy) * iw + xx].to_f32();
                            let wv = w[((o * wspan + wic) * kh + k) * kw + j].to_f32();
                            s += a * wv;
                        }
                    }
                }
                out[(o * oh + y) * ow + x] = f16::from_f32(s);
            }
        }
    }
}

// ── Resident-BO context ───────────────────────────────────────────────────────

/// The resident-BO conv context (`rocket_conv_ctx`).
///
/// [`conv2d`] and friends allocate and free their NPU buffer objects (input, weight,
/// regcmd, output, plus an IOVA guard) on **every** call — and per *tile* for a tiled
/// conv. A context caches those BOs on the borrowed device and grows each to the
/// largest tile it has seen, so repeat calls reuse them. Every job still zeroes and
/// refills its BOs, so reuse is bit-identical to a fresh allocation.
///
/// One context may serve convs of different shapes (the pool grows to the max), and it
/// also owns the host scratch the spatial tilers use, grow-only in the same way.
pub struct ConvContext {
    dev: Device,
    guard: Option<Bo>,
    input: Option<Bo>,
    weight: Option<Bo>,
    rc: Option<Bo>,
    output: Option<Bo>,
    /// Grow-only host scratch for the tilers.
    sub_in: Vec<f16>,
    sub_out: Vec<f16>,
    rband: Vec<usize>,
    cband: Vec<usize>,
}

impl ConvContext {
    /// A context on a borrowed device (nothing is allocated until the first call).
    pub fn new(dev: &Device) -> Self {
        Self {
            dev: dev.clone(),
            guard: None,
            input: None,
            weight: None,
            rc: None,
            output: None,
            sub_in: Vec::new(),
            sub_out: Vec::new(),
            rband: Vec::new(),
            cband: Vec::new(),
        }
    }

    /// The device this context runs on.
    pub fn device(&self) -> &Device {
        &self.dev
    }

    /// `rocket_conv2d_fp16_ctx`: run a direct/depthwise fp16 conv on the resident BOs.
    pub fn conv2d(
        &mut self,
        d: &Conv2dDesc,
        input: &[f16],
        w: &[f16],
        out: &mut [f16],
    ) -> Result<()> {
        let dev = self.dev.clone();
        conv2d_run(&dev, self, d, None, input, w, out)
    }

    /// `rocket_conv2d_act_fp16_ctx`: the same conv with the fused activation epilogue.
    pub fn conv2d_act(
        &mut self,
        d: &Conv2dDesc,
        kind: Activation,
        input: &[f16],
        w: &[f16],
        out: &mut [f16],
    ) -> Result<()> {
        let dev = self.dev.clone();
        conv2d_act_run(&dev, self, d, kind, input, w, out)
    }

    /// `rocket_conv_transpose2d_fp16_ctx`: a transposed conv on the resident BOs.
    pub fn conv_transpose2d(
        &mut self,
        d: &crate::ops::conv_transpose::ConvTranspose2dDesc,
        input: &[f16],
        w: &[f16],
        out: &mut [f16],
    ) -> Result<()> {
        crate::ops::conv_transpose::conv_transpose2d_ctx(self, d, input, w, out)
    }
}

/// Ensure `slot` holds at least `need` bytes in the low 4 GiB (C `rocket_bo_ensure32`).
fn ensure_bo(dev: &Device, slot: &mut Option<Bo>, need: usize) -> Result<()> {
    match slot {
        Some(bo) => {
            bo.ensure32(need)?;
        }
        None => *slot = Some(dev.alloc_bo32(need)?),
    }
    Ok(())
}

fn bo_words(bo: &Bo) -> &[f16] {
    let bytes = bo.as_slice();
    // SAFETY: f16 is repr(transparent) over u16 with align 2 and every prefix of the
    // mapping is initialized.
    unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<f16>(), bytes.len() / 2) }
}

fn bo_words_mut(bo: &mut Bo) -> &mut [f16] {
    let len = bo.len();
    let ptr = bo.as_ptr();
    // SAFETY: exclusive borrow of the BO, page-aligned mapping, 2-byte elements.
    unsafe { std::slice::from_raw_parts_mut(ptr.cast::<f16>(), len / 2) }
}

// ── One job ───────────────────────────────────────────────────────────────────

/// Run ONE conv as a single NPU job (must fit one CBUF pass).
///
/// Explicit shapes on row-major host buffers; `dsy`/`dsx` are the deconvolution
/// interior dilation (1 = off). Returns the generator's refusal when the tile
/// overflows the CBUF.
#[allow(clippy::too_many_arguments)]
fn conv2d_one_job(
    dev: &Device,
    ctx: &mut ConvContext,
    ic: usize,
    ih: usize,
    iw: usize,
    oc: usize,
    oh: usize,
    ow: usize,
    kh: usize,
    kw: usize,
    sy: usize,
    sx: usize,
    pt: usize,
    pl: usize,
    dy: usize,
    dx: usize,
    depthwise: bool,
    g: usize,
    act: Option<&ConvAct<'_>>,
    input: &[f16],
    w: &[f16],
    out: &mut [f16],
    dsy: usize,
    dsx: usize,
) -> Result<()> {
    let cpad = if depthwise { ic.div_ceil(g) * g } else { ic };
    // The fusion is the direct path only (the smooth FFN/Whisper gates); depthwise
    // stays plain.
    let jact = if depthwise { None } else { act };

    // Pad the job's input height up to the 4-row CNA DMA floor with zero rows below the
    // real data: the cube gets IHj rows, datain_height is IHj, but OH stays the real
    // output count, so the padded rows never enter a computed output's receptive field.
    let ihj = ih.max(CONV_MIN_DATAIN_H);
    let cube_in = ic.div_ceil(8) * 8 * ihj * iw;
    let wt_elems = if depthwise {
        cpad * kh * kw
    } else {
        oc * ic * kh * kw
    };
    let cube_out = oc.div_ceil(8) * 8 * oh * ow;
    let nregs = if jact.is_some() {
        crate::regcmd::conv::MAX_OPS_ACT
    } else {
        crate::regcmd::conv::MAX_OPS
    };

    ensure_bo(dev, &mut ctx.guard, 4096)?; // keep the IOVA away from 0
    ensure_bo(dev, &mut ctx.input, cube_in * 2 + CBUF_BANK_SIZE)?;
    ensure_bo(dev, &mut ctx.weight, wt_elems * 2 + CBUF_BANK_SIZE)?;
    ensure_bo(dev, &mut ctx.rc, nregs * 8)?;
    ensure_bo(dev, &mut ctx.output, cube_out * 2 + CBUF_BANK_SIZE)?;

    // scatter the input feature into the NC1HWC2 cube (C2 = 8); rows >= IH stay 0
    {
        let bo = ctx.input.as_mut().expect("input BO");
        bo.prep(0)?;
        bo_words_mut(bo).fill(f16::ZERO);
        scatter_feature(bo_words_mut(bo), ic, ihj, iw, input);
        bo.fini()?;
    }

    // scatter the weights into the conv weight cube
    {
        let bo = ctx.weight.as_mut().expect("weight BO");
        bo.prep(0)?;
        let words = bo_words_mut(bo);
        words.fill(f16::ZERO);
        if depthwise {
            // OC == IC, one filter per channel: W[c][0][kh][kw]
            scatter_weights_dw(words, ic, kh, kw, g, w);
        } else {
            scatter_weights(words, oc, ic, kh, kw, w);
        }
        bo.fini()?;
    }

    let input_dma = ctx.input.as_ref().expect("input BO").dma_address() as u32;
    let weights_dma = ctx.weight.as_ref().expect("weight BO").dma_address() as u32;
    let output_dma = ctx.output.as_ref().expect("output BO").dma_address() as u32;

    let mut p = ConvParams::new(
        ic as u16,
        ihj as u16,
        iw as u16,
        oc as u16,
        oh as u16,
        ow as u16,
        kh as u16,
        kw as u16,
        input_dma,
        weights_dma,
        output_dma,
    );
    p.stride_y = sy as u8;
    p.stride_x = sx as u8;
    p.pad_top = pt as u8;
    p.pad_left = pl as u8;
    p.dil_y = dy as u8;
    p.dil_x = dx as u8;
    p.fp32_to_fp16 = true;
    p.dw_group = if depthwise { g as u8 } else { 0 };
    p.deconv_sy = dsy as u8;
    p.deconv_sx = dsx as u8;
    if let Some(act) = jact {
        p.act = Some(*act);
    }

    let mut ops = Vec::with_capacity(nregs);
    let count = if depthwise {
        gen_conv2d_dw_fp16(&mut p, &mut ops)?
    } else {
        gen_conv2d_fp16(&mut p, &mut ops)?
    };
    if count > nregs {
        log_error!("rocket_conv2d_fp16: regcmd overflow (task_count {count} > {nregs} words)");
        return Err(Error::tiling("rocket_conv2d_fp16/regcmd"));
    }

    {
        let bo = ctx.rc.as_mut().expect("rc BO");
        bo.prep(0)?;
        let bytes = bo.as_mut_slice();
        for (i, op) in ops.iter().enumerate() {
            bytes[i * 8..i * 8 + 8].copy_from_slice(&op.0.to_ne_bytes());
        }
        bo.fini()?;
    }
    {
        let bo = ctx.output.as_mut().expect("output BO");
        bo.prep(0)?;
        bo_words_mut(bo).fill(f16::ZERO);
        bo.fini()?;
    }

    let task = DrmRocketTask {
        regcmd: ctx.rc.as_ref().expect("rc BO").dma_address() as u32,
        regcmd_count: count as u32,
    };
    let in_handles = [
        ctx.input.as_ref().expect("input BO").handle(),
        ctx.weight.as_ref().expect("weight BO").handle(),
        ctx.rc.as_ref().expect("rc BO").handle(),
    ];
    let out_handles = [ctx.output.as_ref().expect("output BO").handle()];
    dev.submit(&[task], &in_handles, &out_handles, 0)?;

    // read back: de-scatter the output cube (OC/8, OH, OW, 8)
    {
        let bo = ctx.output.as_mut().expect("output BO");
        bo.prep(CONV_WAIT_NS)?;
        gather_output(bo_words(bo), oc, oh, ow, out);
        bo.fini()?;
    }
    Ok(())
}

/// One direct fp16 conv job with the output extent **given** rather than derived
/// (`rocket_conv2d_fp16_job_extent`).
///
/// The forward entries always derive the extent; the CNA deconvolution mode needs the
/// transposed one, which the forward arithmetic cannot produce. Stride and dilation
/// are 1, the only values that mode runs with.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_job_extent(
    dev: &Device,
    ic: usize,
    ih: usize,
    iw: usize,
    oc: usize,
    oh: usize,
    ow: usize,
    kh: usize,
    kw: usize,
    pt: usize,
    pl: usize,
    input: &[f16],
    w: &[f16],
    out: &mut [f16],
) -> Result<()> {
    let mut ctx = ConvContext::new(dev);
    conv2d_one_job(
        dev, &mut ctx, ic, ih, iw, oc, oh, ow, kh, kw, 1, 1, pt, pl, 1, 1, false, 0, None, input,
        w, out, 1, 1,
    )
}

/// `rocket_conv2d_fp16_deconv_fits`: does one job per output-channel tile hold the
/// shape?
///
/// `IC` in whole 32-channel groups, the **compact** input within one CBUF pass beside a
/// 16-channel weight tile, one kernel within one bank, and an output extent the geometry
/// fields carry (`OW`/`OH` <= 2047 and `OH*OW` <= 262143). Pure.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_deconv_fits(
    ic: usize,
    ih: usize,
    iw: usize,
    oc: usize,
    oh: usize,
    ow: usize,
    kh: usize,
    kw: usize,
) -> bool {
    let banks = crate::hw::current().cbuf_banks;
    if ic == 0 || ih == 0 || iw == 0 || oc == 0 || oh == 0 || ow == 0 {
        return false;
    }
    if !ic.is_multiple_of(32) {
        return false; // whole K-groups only
    }
    let ihj = ih.max(CONV_MIN_DATAIN_H);
    let feat = ic * ihj * iw * 2;
    if feat > CONV_FEAT_BUDGET {
        return false;
    }
    let fbanks = feat.div_ceil(CBUF_BANK_SIZE);
    let w16 = (16 * ic * kh * kw * 2).div_ceil(CBUF_BANK_SIZE);
    if kh * kw * ic * 2 > CBUF_BANK_SIZE {
        return false; // a kernel per bank
    }
    if fbanks + w16 > banks {
        return false;
    }
    if ow > 0x7FF || oh > 0x7FF || oh * ow > 0x3FFFF {
        return false;
    }
    true
}

/// The CNA deconvolution mode as a route (`rocket_conv2d_fp16_deconv`).
///
/// One direct fp16 job per output-channel tile with the deconvolution mode on, the
/// **compact** input, the transposed output extent and a pad of `k-1-p`. `wf` is the
/// forward kernel `[OC][IC][KH][KW]`, i.e. the transposed conv's weight already rotated
/// 180° and channel-transposed.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_deconv(
    dev: &Device,
    ctx: Option<&mut ConvContext>,
    ic: usize,
    ih: usize,
    iw: usize,
    oc: usize,
    oh: usize,
    ow: usize,
    kh: usize,
    kw: usize,
    sy: usize,
    sx: usize,
    pt: usize,
    pl: usize,
    input: &[f16],
    wf: &[f16],
    out: &mut [f16],
) -> Result<()> {
    if !conv2d_deconv_fits(ic, ih, iw, oc, oh, ow, kh, kw) {
        return Err(
            Error::coded(crate::Status::Tiling, -4, "rocket_conv2d_fp16_deconv/fit").with_detail(
                "the compact input + a 16-channel weight tile do not fit one CBUF pass",
            ),
        );
    }
    // A local context costs nothing when the caller brings one (no BO is allocated
    // before the first job).
    let mut local = ConvContext::new(dev);
    let ctx: &mut ConvContext = match ctx {
        Some(c) => c,
        None => &mut local,
    };

    // OC not a multiple of 16: zero kernels pad the tile and the real channels are
    // sliced off the front, as the forward path does.
    let ocp = oc.div_ceil(16) * 16;
    let wpad = if ocp != oc {
        let mut wp = vec![f16::ZERO; ocp * ic * kh * kw];
        wp[..oc * ic * kh * kw].copy_from_slice(wf);
        Some(wp)
    } else {
        None
    };
    let wuse: &[f16] = wpad.as_deref().unwrap_or(wf);
    let mut opad = vec![f16::ZERO; ocp * oh * ow];

    // The weight tile takes what the compact input leaves, capped at 4 banks as the
    // forward tiler keeps it; one 16-channel tile always fits (the fit check).
    let banks = crate::hw::current().cbuf_banks;
    let ihj = ih.max(CONV_MIN_DATAIN_H);
    let fbanks = (ic * ihj * iw * 2).div_ceil(CBUF_BANK_SIZE);
    let wcap = (banks - fbanks).min(4);
    let mut oct = ocp;
    while oct > 16 && (oct * ic * kh * kw * 2).div_ceil(CBUF_BANK_SIZE) > wcap {
        oct -= 16;
    }

    let mut tile = vec![f16::ZERO; oct * oh * ow];
    let mut oc0 = 0;
    while oc0 < ocp {
        let n = (ocp - oc0).min(oct);
        let wslice = &wuse[oc0 * ic * kh * kw..(oc0 + n) * ic * kh * kw];
        conv2d_one_job(
            dev,
            ctx,
            ic,
            ih,
            iw,
            n,
            oh,
            ow,
            kh,
            kw,
            1,
            1,
            pt,
            pl,
            1,
            1,
            false,
            0,
            None,
            input,
            wslice,
            &mut tile[..n * oh * ow],
            sy,
            sx,
        )?;
        opad[oc0 * oh * ow..(oc0 + n) * oh * ow].copy_from_slice(&tile[..n * oh * ow]);
        oc0 += n;
    }
    out[..oc * oh * ow].copy_from_slice(&opad[..oc * oh * ow]);
    Ok(())
}

/// Spatially tile one depthwise channel chunk whose feature is too big for one CBUF
/// pass (`dw_spatial`).
///
/// Mirrors the direct path's band tiler exactly — materialised halo + explicit zero pad
/// so each band runs with `pad_top = pad_left = 0`, the 4-row floor, columns narrowed
/// only when a full-width band still overflows — but for the depthwise job (each channel
/// reduces only itself, so there is no OC loop and the weight cube is shared by every
/// band).
#[allow(clippy::too_many_arguments)]
fn dw_spatial(
    dev: &Device,
    ctx: &mut ConvContext,
    cn: usize,
    ih: usize,
    iw: usize,
    oh: usize,
    ow: usize,
    kh: usize,
    kw: usize,
    sy: usize,
    sx: usize,
    pt: usize,
    pl: usize,
    dy: usize,
    dx: usize,
    g: usize,
    input: &[f16],
    w: &[f16],
    out: &mut [f16],
) -> Result<()> {
    let budget = dw_feat_budget();
    let feat = |rh: usize, cw: usize| cn * datain_h(rh, sy, kh, dy) * datain_h(cw, sx, kw, dx) * 2;
    let min_ihs = {
        let v = match knobs::int("ROCKET_CONV_MIN_IH") {
            Some(v) if v > 0 => v as usize,
            _ => 6,
        };
        v.max(CONV_MIN_DATAIN_H) // never below the hardware row floor
    };
    let mut rht_lo = 1;
    while rht_lo < oh && datain_h(rht_lo, sy, kh, dy) < min_ihs {
        rht_lo += 1;
    }
    let mut rht = oh;
    let mut cwt = ow;
    while rht > rht_lo && feat(rht, cwt) > budget {
        rht -= 1;
    }
    if feat(rht, cwt) > budget {
        while cwt > 1 && feat(rht, cwt) > budget {
            cwt -= 1;
        }
    }
    if feat(rht, cwt) > budget {
        return Err(Error::tiling("rocket_dw_spatial/cbuf"));
    }

    let ihs = datain_h(rht, sy, kh, dy);
    let iws = datain_h(cwt, sx, kw, dx);
    // BALANCED row/col bands: no tiny remainder stub below the 4-row floor.
    let mut rband = std::mem::take(&mut ctx.rband);
    let mut cband = std::mem::take(&mut ctx.cband);
    let nr = balance_bands(oh, rht, &mut rband);
    let nc = balance_bands(ow, cwt, &mut cband);
    let mut sub_in = std::mem::take(&mut ctx.sub_in);
    let mut sub_out = std::mem::take(&mut ctx.sub_out);
    sub_in.resize(cn * ihs * iws, f16::ZERO);
    sub_out.resize(cn * rht * cwt, f16::ZERO);

    let mut ret = Ok(());
    let mut r0 = 0;
    'rows: for &rh in &rband[..nr] {
        let ih_sub = datain_h(rh, sy, kh, dy);
        let gh0 = r0 as isize * sy as isize - pt as isize;
        let mut c0 = 0;
        for &cw in &cband[..nc] {
            let iw_sub = datain_h(cw, sx, kw, dx);
            let gw0 = c0 as isize * sx as isize - pl as isize;

            sub_in.fill(f16::ZERO);
            for c in 0..cn {
                for j in 0..ih_sub {
                    let gih = gh0 + j as isize;
                    if gih < 0 || gih >= ih as isize {
                        continue;
                    }
                    let cs = gw0.max(0) as usize;
                    let ce = ((gw0 + iw_sub as isize).min(iw as isize)).max(0) as usize;
                    if cs >= ce {
                        continue;
                    }
                    let dst_off = (c * ih_sub + j) * iw_sub + (cs as isize - gw0) as usize;
                    let src_off = (c * ih + gih as usize) * iw + cs;
                    sub_in[dst_off..dst_off + (ce - cs)]
                        .copy_from_slice(&input[src_off..src_off + (ce - cs)]);
                }
            }

            ret = conv2d_one_job(
                dev,
                ctx,
                cn,
                ih_sub,
                iw_sub,
                cn,
                rh,
                cw,
                kh,
                kw,
                sy,
                sx,
                0,
                0,
                dy,
                dx,
                true,
                g,
                None,
                &sub_in,
                w,
                &mut sub_out[..cn * rh * cw],
                1,
                1,
            );
            if ret.is_err() {
                break 'rows;
            }

            for c in 0..cn {
                for r in 0..rh {
                    let dst = ((c * oh) + (r0 + r)) * ow + c0;
                    let src = (c * rh + r) * cw;
                    out[dst..dst + cw].copy_from_slice(&sub_out[src..src + cw]);
                }
            }
            c0 += cw;
        }
        r0 += rh;
    }
    ctx.rband = rband;
    ctx.cband = cband;
    ctx.sub_in = sub_in;
    ctx.sub_out = sub_out;
    ret
}

/// The conv tiling driver (`conv2d_run`).
///
/// `fd < 0`'s CPU fallback of the C has no counterpart here: [`conv2d_ref`] is the host
/// path and the caller picks it explicitly.
fn conv2d_run(
    dev: &Device,
    ctx: &mut ConvContext,
    d: &Conv2dDesc,
    act: Option<&ConvAct<'_>>,
    input: &[f16],
    w: &[f16],
    out: &mut [f16],
) -> Result<()> {
    // Direct OC % 16 != 0: pad OC up to the conv weight oc group, run the padded conv,
    // then slice the real OC channels off the front. The extra kernels are zero and each
    // direct output channel is independent, so the first OC channels are bit-exact.
    if !d.depthwise && !d.oc.is_multiple_of(16) {
        let oc = d.oc;
        let ocp = oc.div_ceil(16) * 16;
        let (oh, ow) = (conv2d_oh(d), conv2d_ow(d));
        if oh == 0 || ow == 0 {
            return Err(Error::shape("rocket_conv2d_fp16/out-extent"));
        }
        let mut dp = *d;
        dp.oc = ocp;
        let mut wp = vec![f16::ZERO; ocp * d.ic * d.kh * d.kw];
        wp[..oc * d.ic * d.kh * d.kw].copy_from_slice(&w[..oc * d.ic * d.kh * d.kw]);
        let mut op = vec![f16::ZERO; ocp * oh * ow];
        conv2d_run(dev, ctx, &dp, act, input, &wp, &mut op)?;
        out[..oc * oh * ow].copy_from_slice(&op[..oc * oh * ow]);
        return Ok(());
    }

    conv2d_plan(d)?;

    let icr = d.ic;
    let (ih, iw, oc, kh, kw) = (d.ih, d.iw, d.oc, d.kh, d.kw);
    let (oh, ow) = (conv2d_oh(d), conv2d_ow(d));
    let (sy, sx, pt, pl) = (d.stride_y, d.stride_x, d.pad_top, d.pad_left);
    let (dy, dx) = (d.dil_y, d.dil_x);

    if d.depthwise {
        // Two nested, independent tilings: channels (each channel reduces only itself)
        // and, when even one group overflows, space.
        let g = conv_dw_group();
        let mut cc = icr;
        while cc > g && !dw_chunk_fits(cc, ih, iw, kh, kw) {
            cc -= g;
        }
        let spatial = !dw_chunk_fits(cc, ih, iw, kh, kw);
        if knobs::present("ROCKET_CONV_DW_DEBUG") {
            log_debug!(
                "[rocket_conv dw] C={icr} IH={ih} IW={iw} -> chunk Cc={cc} ({} chunk(s)){}",
                icr.div_ceil(cc),
                if spatial { " +spatial-tile" } else { "" }
            );
        }
        let mut c0 = 0;
        while c0 < icr {
            let cn = (icr - c0).min(cc);
            let in_c = &input[c0 * ih * iw..(c0 + cn) * ih * iw];
            let w_c = &w[c0 * kh * kw..(c0 + cn) * kh * kw];
            let out_c = &mut out[c0 * oh * ow..(c0 + cn) * oh * ow];
            if spatial {
                dw_spatial(
                    dev, ctx, cn, ih, iw, oh, ow, kh, kw, sy, sx, pt, pl, dy, dx, g, in_c, w_c,
                    out_c,
                )?;
            } else {
                conv2d_one_job(
                    dev, ctx, cn, ih, iw, cn, oh, ow, kh, kw, sy, sx, pt, pl, dy, dx, true, g,
                    None, in_c, w_c, out_c, 1, 1,
                )?;
            }
            c0 += cn;
        }
        return Ok(());
    }

    // First-layer IC < 32 (e.g. RGB): zero-pad the input channels up to the weight ic
    // group of 32. The padded channels contribute 0. Build a padded weight once; the
    // input is padded lazily in the per-tile materialisation (ic >= ICr stays zero).
    let ic = icr.div_ceil(32) * 32;
    let wpad = if ic != icr {
        let mut wp = vec![f16::ZERO; oc * ic * kh * kw];
        for o in 0..oc {
            for c in 0..icr {
                let dst = ((o * ic + c) * kh) * kw;
                let src = ((o * icr + c) * kh) * kw;
                wp[dst..dst + kh * kw].copy_from_slice(&w[src..src + kh * kw]);
            }
        }
        Some(wp)
    } else {
        None
    };
    let wuse: &[f16] = wpad.as_deref().unwrap_or(w);

    // fast path: no channel pad AND the whole problem fits one CBUF pass
    {
        let ih_full = datain_h(oh, sy, kh, dy);
        let iw_full = datain_h(ow, sx, kw, dx);
        if ic == icr
            && oc * ic * kh * kw * 2 <= 4 * CBUF_BANK_SIZE
            && ic * ih_full * iw_full * 2 <= CONV_FEAT_BUDGET
        {
            return conv2d_one_job(
                dev, ctx, ic, ih, iw, oc, oh, ow, kh, kw, sy, sx, pt, pl, dy, dx, false, 0, act,
                input, w, out, 1, 1,
            );
        }
    }

    // Feature bytes for a (rh output rows x cw output columns) tile's materialised
    // sub-input.
    let feat = |rh: usize, cw: usize| ic * datain_h(rh, sy, kh, dy) * datain_h(cw, sx, kw, dx) * 2;

    let min_ihs = {
        let v = match knobs::int("ROCKET_CONV_MIN_IH") {
            Some(v) if v > 0 => v as usize,
            _ => 6,
        };
        v.max(CONV_MIN_DATAIN_H)
    };

    // OC tile (mult-16) and the feature budget jointly share the 12 CBUF banks (the
    // generator gives the weight `12 - data_bank` banks). Shrink OCt toward the 16
    // oc-group to keep the weight ~4 banks; when even one oc-group's weight cube needs
    // more (a KxK conv at large IC), accept the bigger weight and shrink the feature
    // budget so the two still sum to <= 12 banks.
    let banks = crate::hw::current().cbuf_banks;
    let mut oct = oc;
    while oct > 16 && (oct * ic * kh * kw * 2).div_ceil(CBUF_BANK_SIZE) > 4 {
        oct -= 16;
    }
    let wbanks = (oct * ic * kh * kw * 2).div_ceil(CBUF_BANK_SIZE);
    if wbanks > banks - 1 {
        return Err(Error::tiling("rocket_conv2d_fp16/banks"));
    }
    let feat_budget = ((banks - wbanks) * CBUF_BANK_SIZE).min(CONV_FEAT_BUDGET);

    // The smallest row tile whose materialised sub-input is >= min_ihs rows.
    let mut rht_lo = 1;
    while rht_lo < oh && datain_h(rht_lo, sy, kh, dy) < min_ihs {
        rht_lo += 1;
    }
    let mut rht = oh;
    let mut cwt = ow;
    while rht > rht_lo && feat(rht, cwt) > feat_budget {
        rht -= 1;
    }
    if feat(rht, cwt) > feat_budget {
        while cwt > 1 && feat(rht, cwt) > feat_budget {
            cwt -= 1;
        }
    }
    if feat(rht, cwt) > feat_budget {
        return Err(Error::tiling("rocket_conv2d_fp16/cbuf"));
    }

    let ihs = datain_h(rht, sy, kh, dy);
    let iws = datain_h(cwt, sx, kw, dx);
    let mut rband = std::mem::take(&mut ctx.rband);
    let mut cband = std::mem::take(&mut ctx.cband);
    let nr = balance_bands(oh, rht, &mut rband);
    let nc = balance_bands(ow, cwt, &mut cband);
    let mut sub_in = std::mem::take(&mut ctx.sub_in);
    let mut sub_out = std::mem::take(&mut ctx.sub_out);
    sub_in.resize(ic * ihs * iws, f16::ZERO);
    sub_out.resize(oct * rht * cwt, f16::ZERO);

    let mut ret = Ok(());
    let mut oc0 = 0;
    'tiles: while oc0 < oc {
        let ocn = (oc - oc0).min(oct);
        let wslice = &wuse[oc0 * ic * kh * kw..(oc0 + ocn) * ic * kh * kw];
        let mut r0 = 0;
        for &rh in &rband[..nr] {
            let ih_sub = datain_h(rh, sy, kh, dy);
            let gh0 = r0 as isize * sy as isize - pt as isize;
            let mut c0 = 0;
            for &cw in &cband[..nc] {
                let iw_sub = datain_h(cw, sx, kw, dx);
                let gw0 = c0 as isize * sx as isize - pl as isize;

                // materialise the tile's sub-input: real halo + zero pad, with the
                // channel pad (ic >= ICr stays zero from the fill)
                sub_in.fill(f16::ZERO);
                for c in 0..icr {
                    for j in 0..ih_sub {
                        let gih = gh0 + j as isize;
                        if gih < 0 || gih >= ih as isize {
                            continue;
                        }
                        let cs = gw0.max(0) as usize;
                        let ce = ((gw0 + iw_sub as isize).min(iw as isize)).max(0) as usize;
                        if cs >= ce {
                            continue;
                        }
                        let dst_off = (c * ih_sub + j) * iw_sub + (cs as isize - gw0) as usize;
                        let src_off = (c * ih + gih as usize) * iw + cs;
                        sub_in[dst_off..dst_off + (ce - cs)]
                            .copy_from_slice(&input[src_off..src_off + (ce - cs)]);
                    }
                }

                // sub-conv: both pads materialised -> pad_top = pad_left = 0; the
                // output is exactly rh x cw.
                ret = conv2d_one_job(
                    dev,
                    ctx,
                    ic,
                    ih_sub,
                    iw_sub,
                    ocn,
                    rh,
                    cw,
                    kh,
                    kw,
                    sy,
                    sx,
                    0,
                    0,
                    dy,
                    dx,
                    false,
                    0,
                    act,
                    &sub_in,
                    wslice,
                    &mut sub_out[..ocn * rh * cw],
                    1,
                    1,
                );
                if ret.is_err() {
                    break 'tiles;
                }

                for o in 0..ocn {
                    for r in 0..rh {
                        let dst = ((oc0 + o) * oh + (r0 + r)) * ow + c0;
                        let src = (o * rh + r) * cw;
                        out[dst..dst + cw].copy_from_slice(&sub_out[src..src + cw]);
                    }
                }
                c0 += cw;
            }
            r0 += rh;
        }
        oc0 += ocn;
    }

    ctx.rband = rband;
    ctx.cband = cband;
    ctx.sub_in = sub_in;
    ctx.sub_out = sub_out;
    ret
}

// ── Public entries ────────────────────────────────────────────────────────────

/// `rocket_conv2d_fp16`: run a direct or depthwise fp16 conv on the NPU.
pub fn conv2d(
    dev: &Device,
    d: &Conv2dDesc,
    input: &[f16],
    w: &[f16],
    out: &mut [f16],
) -> Result<()> {
    let mut ctx = ConvContext::new(dev);
    conv2d_run(dev, &mut ctx, d, None, input, w, out)
}

/// `rocket_conv2d_fp16_ctx`: the same conv on a resident-BO context.
///
/// The free-function counterpart of [`ConvContext::conv2d`], kept for C-API parity.
pub fn conv2d_ctx(
    ctx: &mut ConvContext,
    d: &Conv2dDesc,
    input: &[f16],
    w: &[f16],
    out: &mut [f16],
) -> Result<()> {
    ctx.conv2d(d, input, w, out)
}

/// The C's conv→activation refusal codes (its own local numbers, not `ROCKET_E_*`).
pub const CONV_ACT_ERR_DEPTHWISE: i32 = -10;
/// A fused activation kind outside `Silu`/`Tanh`/`Gelu`.
pub const CONV_ACT_ERR_KIND: i32 = -11;

/// [`conv2d_act_code`]'s pure predicate: `0`, `-10` (depthwise) or `-11` (kind).
pub fn conv2d_act_code(d: &Conv2dDesc, kind: Activation) -> i32 {
    if d.depthwise {
        return CONV_ACT_ERR_DEPTHWISE;
    }
    match kind {
        Activation::Silu | Activation::Tanh | Activation::Gelu => 0,
        _ => CONV_ACT_ERR_KIND,
    }
}

/// The conv→activation fusion body (`conv2d_act_run`).
fn conv2d_act_run(
    dev: &Device,
    ctx: &mut ConvContext,
    d: &Conv2dDesc,
    kind: Activation,
    input: &[f16],
    w: &[f16],
    out: &mut [f16],
) -> Result<()> {
    match conv2d_act_code(d, kind) {
        CONV_ACT_ERR_DEPTHWISE => {
            return Err(Error::coded(
                crate::Status::Unsupported,
                CONV_ACT_ERR_DEPTHWISE,
                "rocket_conv2d_act_fp16",
            )
            .with_detail("depthwise has no fused-activation form (-10)"));
        }
        CONV_ACT_ERR_KIND => {
            return Err(Error::coded(
                crate::Status::Unsupported,
                CONV_ACT_ERR_KIND,
                "rocket_conv2d_act_fp16",
            )
            .with_detail("only SiLU / tanh / GELU fuse (-11)"));
        }
        _ => {}
    }
    let mut lut = crate::ops::activation::LutTable::zeroed();
    let ep = lut_epilogue_build(kind, &mut lut)?;
    let act = ConvAct {
        lut: lut.words(),
        ep: &ep,
    };
    conv2d_run(dev, ctx, d, Some(&act), input, w, out)
}

/// `rocket_conv2d_act_fp16`: a direct fp16 conv with `f(x)` fused into the same NPU
/// job's DPU epilogue.
///
/// Only the smooth single-pass kinds (`Silu`, `Tanh`, `Gelu`) are accepted; `HardSwish`
/// is refused (its exactly-flat `x <= -3` tail trips the LE/LO mux) and depthwise is
/// refused (direct scope). The C's codes are `-10`/`-11`; see [`conv2d_act_code`] and
/// [`Error::c_code`].
pub fn conv2d_act(
    dev: &Device,
    d: &Conv2dDesc,
    kind: Activation,
    input: &[f16],
    w: &[f16],
    out: &mut [f16],
) -> Result<()> {
    let mut ctx = ConvContext::new(dev);
    conv2d_act_run(dev, &mut ctx, d, kind, input, w, out)
}

/// `rocket_conv2d_act_fp16_ctx`: the fused conv on a resident-BO context.
pub fn conv2d_act_ctx(
    ctx: &mut ConvContext,
    d: &Conv2dDesc,
    kind: Activation,
    input: &[f16],
    w: &[f16],
    out: &mut [f16],
) -> Result<()> {
    ctx.conv2d_act(d, kind, input, w, out)
}

/// `rocket_conv1d_fp16`: a width-only 1D convolution over the time axis.
///
/// Lowered onto the height-1 conv (`IH = KH = OH = 1`, the time axis on the **height**
/// axis — the tiler tiles output *rows* first, so a long / many-channel sequence shrinks
/// the per-tile height until the feature fits one CBUF pass, where a width-on-time
/// layout would leave `OH = 1` untileable):
///
/// ```text
/// in  [IC][IT]        W [OC][IC][KW]        out [OC][OT]
/// out[oc][t] = sum_{ic,kw} W[oc][ic][kw] * in[ic][t*stride + kw - pad]
/// ```
#[allow(clippy::too_many_arguments)]
pub fn conv1d(
    dev: &Device,
    ic: usize,
    it: usize,
    oc: usize,
    kw: usize,
    stride: usize,
    pad: usize,
    input: &[f16],
    w: &[f16],
    out: &mut [f16],
) -> Result<()> {
    if ic == 0 || it == 0 || oc == 0 || kw == 0 || stride == 0 {
        return Err(Error::shape("rocket_conv1d_fp16"));
    }
    let d = Conv2dDesc {
        ic,
        ih: it,
        iw: 1,
        oc,
        kh: kw,
        kw: 1,
        stride_y: stride,
        stride_x: 1,
        pad_top: pad,
        pad_left: 0,
        dil_y: 1,
        dil_x: 1,
        depthwise: false,
        ..Default::default()
    };
    conv2d(dev, &d, input, w, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn desc(ic: usize, ih: usize, iw: usize, oc: usize, kh: usize, kw: usize) -> Conv2dDesc {
        Conv2dDesc::new(ic, ih, iw, oc, kh, kw)
    }

    #[test]
    fn output_dims_and_trails() {
        // 3x3, pad 1, stride 1 keeps the plane.
        let mut d = desc(3, 8, 8, 16, 3, 3);
        d.pad_top = 1;
        d.pad_left = 1;
        assert_eq!((conv2d_oh(&d), conv2d_ow(&d)), (8, 8));
        assert_eq!((conv2d_trail_y(&d), conv2d_trail_x(&d)), (1, 1));

        // 3x3, pad 0, stride 2 on 8 -> 3.
        let mut d = desc(3, 8, 10, 16, 3, 3);
        d.stride_y = 2;
        d.stride_x = 2;
        assert_eq!((conv2d_oh(&d), conv2d_ow(&d)), (3, 4));
        assert_eq!((conv2d_trail_y(&d), conv2d_trail_x(&d)), (0, 0));

        // Dilation 2 reaches no further than a pad-less 3x3 at the same extent: the last
        // window's last tap lands exactly on the last row.
        let mut d = desc(32, 8, 8, 16, 3, 3);
        d.dil_y = 2;
        d.dil_x = 2;
        assert_eq!((conv2d_oh(&d), conv2d_ow(&d)), (4, 4));
        assert_eq!((conv2d_trail_y(&d), conv2d_trail_x(&d)), (0, 0));

        // An explicit extent wins; an extent the formula did not derive is refused.
        let mut d = desc(32, 8, 8, 16, 3, 3);
        d.pad_top = 1;
        d.pad_left = 1;
        d.oh = 9;
        assert_eq!(conv2d_oh(&d), 9);
        assert_eq!(conv2d_plan_code(&d), -1);
        d.oh = 8;
        assert_eq!(conv2d_plan_code(&d), 0);

        // 1x1 is the degenerate case: out == in, no trail.
        let d = desc(32, 5, 7, 16, 1, 1);
        assert_eq!((conv2d_oh(&d), conv2d_ow(&d)), (5, 7));
        assert_eq!(conv_out_dim(5, 1, 1, 0, 1), 5);
    }

    #[test]
    fn plan_accepts_and_refuses() {
        // Direct 3x3 at IC=3 (the RGB stem: padded to 32, not refused).
        let mut d = desc(3, 56, 56, 64, 3, 3);
        d.pad_top = 1;
        d.pad_left = 1;
        assert_eq!(conv2d_plan_code(&d), 0);
        assert!(conv2d_plan(&d).is_ok());

        // OC=24 is padded to 32, not refused (an SSD head).
        let d = desc(32, 8, 8, 24, 1, 1);
        assert_eq!(conv2d_plan_code(&d), 0);

        // A kernel bigger than one CBUF bank at 3x3: IC 6000 -> 3*3*6016*2 > 32768.
        let d = desc(6000, 4, 4, 16, 3, 3);
        assert_eq!(conv2d_plan_code(&d), -4);

        // A kernel that fits a bank but whose minimal tile overflows the feature budget
        // is impossible for the direct path at reasonable IC; the depthwise branch is
        // where the spatial floor bites (below).

        // Malformed descriptors.
        assert_eq!(conv2d_plan_code(&desc(0, 4, 4, 16, 1, 1)), -1);
        let mut d = desc(32, 4, 4, 16, 3, 3);
        d.stride_y = 0;
        assert_eq!(conv2d_plan_code(&d), -1);
        let mut d = desc(32, 4, 4, 16, 3, 3);
        d.dil_x = 0;
        assert_eq!(conv2d_plan_code(&d), -1);

        // An output extent that collapses: 4x4 with a 5x5 kernel.
        let d = desc(32, 4, 4, 16, 5, 5);
        assert_eq!(conv2d_oh(&d), 0);
        assert_eq!(conv2d_plan_code(&d), -3);

        // Depthwise: OC must equal IC and IC must be a multiple of G.
        let mut d = desc(32, 8, 8, 32, 3, 3);
        d.depthwise = true;
        assert_eq!(conv2d_plan_code(&d), 0);
        let mut d = desc(32, 8, 8, 16, 3, 3);
        d.depthwise = true;
        assert_eq!(conv2d_plan_code(&d), -2);
        let mut d = desc(24, 8, 8, 24, 3, 3);
        d.depthwise = true;
        assert_eq!(conv2d_plan_code(&d), -2);

        // A single depthwise channel whose minimal tile overflows the feature budget:
        // a 7x7 kernel at 5 banks' worth of taps.
        let mut d = desc(32, 8, 8, 32, 7, 7);
        d.depthwise = true;
        assert_eq!(conv2d_plan_code(&d), 0, "32*13*13*2 < 8 banks");
        // ... and one whose G-channel weight cube cannot fit a bank: 25x25 taps at
        // G = 32 is 40000 B > 32768, with a 32x32 input to keep the extent positive.
        let mut d = desc(32, 32, 32, 32, 25, 25);
        d.depthwise = true;
        assert_eq!(conv2d_oh(&d), 8);
        assert_eq!(conv2d_plan_code(&d), -4);

        // Status mapping.
        let mut d = desc(24, 8, 8, 24, 3, 3);
        d.depthwise = true;
        assert_eq!(conv2d_plan(&d).unwrap_err().status(), crate::Status::Tiling);
    }

    #[test]
    fn cube_scatter_gather_round_trip() {
        // A partial channel group (IC = 3) exercises the C2 = 8 padding, and a partial
        // output group too.
        let (c, h, w) = (3usize, 5usize, 7usize);
        let src: Vec<f16> = (0..c * h * w).map(|i| f16::from_f32(i as f32)).collect();
        let mut cube = vec![f16::ZERO; c.div_ceil(8) * 8 * h * w];
        scatter_feature(&mut cube, c, h, w, &src);
        let mut back = vec![f16::ZERO; c * h * w];
        gather_output(&cube, c, h, w, &mut back);
        assert_eq!(src, back);
        // Channels live 8 apart at one pixel, and the plane stride is h*w*8.
        assert_eq!(feature_cube_index(c, h, w, 1, 1, 1), 0);
        assert_eq!(feature_cube_index(c, h, w, 2, 1, 1), 1);
        assert_eq!(feature_cube_index(c, h, w, 1, 1, 2), 8);
        assert_eq!(feature_cube_index(c, h, w, 9, 1, 1), h * w * 8);
    }

    #[test]
    fn weight_cube_layout() {
        // 1x1 at IC = 32 collapses to the matmul weight layout: (n/16, k/32, 16, 32).
        assert_eq!(weight_cube_index(32, 1, 1, 1, 1, 1, 1), 0);
        assert_eq!(weight_cube_index(32, 1, 1, 16, 32, 1, 1), 15 * 32 + 31);
        assert_eq!(weight_cube_index(32, 1, 1, 17, 1, 1, 1), 16 * 32);
        // The kernel spatial sits between the ic1 and oc2 groups.
        let idx = weight_cube_index(64, 3, 3, 1, 1, 1, 2);
        assert_eq!(idx, 16 * 32); // kh = 1, kw = 2 -> one kernel's worth
        let idx = weight_cube_index(64, 3, 3, 2, 1, 1, 1);
        assert_eq!(idx, 32); // oc2 = 1
        // oc1 = 0, ic1 = 1 (IC = 64 -> nIC1 = 2), kh = kw = 3:
        // ((0*2 + 1)*3 + 2)*3 + 2 = 17 kernel slots, each 16*32.
        let idx = weight_cube_index(64, 3, 3, 1, 33, 3, 3);
        assert_eq!(idx, 17 * 16 * 32);
        // Round trip through the scatter: every element lands where the index says.
        let (oc, ic, kh, kw) = (16usize, 32usize, 3usize, 3usize);
        let w: Vec<f16> = (0..oc * ic * kh * kw)
            .map(|i| f16::from_f32((i % 97) as f32))
            .collect();
        let mut cube = vec![f16::ZERO; oc * ic * kh * kw];
        scatter_weights(&mut cube, oc, ic, kh, kw, &w);
        for o in 0..oc {
            for c in 0..ic {
                for k in 0..kh {
                    for j in 0..kw {
                        let src = w[((o * ic + c) * kh + k) * kw + j];
                        let idx = weight_cube_index(ic, kh, kw, o + 1, c + 1, k + 1, j + 1);
                        assert_eq!(cube[idx], src);
                    }
                }
            }
        }
    }

    #[test]
    fn depthwise_weight_cube_layout() {
        // (C/G, KH, KW, G), kh outer / kw inner, G channels innermost.
        assert_eq!(weight_cube_dw_index(3, 3, 32, 1, 1, 1), 0);
        assert_eq!(weight_cube_dw_index(3, 3, 32, 2, 1, 1), 1);
        assert_eq!(weight_cube_dw_index(3, 3, 32, 1, 1, 2), 32);
        assert_eq!(weight_cube_dw_index(3, 3, 32, 1, 2, 1), 3 * 32);
        assert_eq!(weight_cube_dw_index(3, 3, 32, 33, 1, 1), 3 * 3 * 32);
        let (c, kh, kw, g) = (64usize, 3usize, 3usize, 32usize);
        let w: Vec<f16> = (0..c * kh * kw)
            .map(|i| f16::from_f32((i % 89) as f32))
            .collect();
        let mut cube = vec![f16::ZERO; c * kh * kw];
        scatter_weights_dw(&mut cube, c, kh, kw, g, &w);
        for ch in 0..c {
            for k in 0..kh {
                for j in 0..kw {
                    let src = w[(ch * kh + k) * kw + j];
                    assert_eq!(
                        cube[weight_cube_dw_index(kh, kw, g, ch + 1, k + 1, j + 1)],
                        src
                    );
                }
            }
        }
    }

    #[test]
    fn reference_hand_cases() {
        // 1x1 identity: out == in*tap.
        let mut d = desc(2, 2, 2, 2, 1, 1);
        d.oc = 2;
        let input: Vec<f16> = [1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0]
            .iter()
            .map(|v| f16::from_f32(*v))
            .collect();
        let mut w = vec![f16::ZERO; 2 * 2];
        w[0] = f16::from_f32(2.0); // oc0 <- ic0
        w[3] = f16::from_f32(0.5); // oc1 <- ic1
        let mut out = vec![f16::ZERO; 2 * 2 * 2];
        conv2d_ref(&d, &input, &w, &mut out);
        assert_eq!(out[0].to_f32(), 2.0);
        assert_eq!(out[3].to_f32(), 8.0);
        assert_eq!(out[4].to_f32(), -0.5);
        assert_eq!(out[7].to_f32(), -2.0);

        // 3x3 pad 1 sum kernel: every interior output is the 3x3 window sum, the
        // border loses the out-of-range taps (zero padding).
        let mut d = desc(1, 3, 3, 1, 3, 3);
        d.pad_top = 1;
        d.pad_left = 1;
        let input: Vec<f16> = (1..=9).map(|v| f16::from_f32(v as f32)).collect();
        let w = vec![f16::from_f32(1.0); 9];
        let mut out = vec![f16::ZERO; 9];
        conv2d_ref(&d, &input, &w, &mut out);
        assert_eq!(out[0].to_f32(), 12.0); // 1+2+4+5
        assert_eq!(out[1].to_f32(), 21.0); // 1+2+3+4+5+6
        assert_eq!(out[4].to_f32(), 45.0); // the whole plane
        assert_eq!(out[8].to_f32(), 28.0); // 5+6+8+9

        // Stride 2 on a 4x4 with a 2x2 kernel: four disjoint 2x2 windows.
        let mut d = desc(1, 4, 4, 1, 2, 2);
        d.stride_y = 2;
        d.stride_x = 2;
        let input: Vec<f16> = (0..16).map(|v| f16::from_f32(v as f32)).collect();
        let w = vec![f16::from_f32(1.0); 4];
        let mut out = vec![f16::ZERO; 4];
        conv2d_ref(&d, &input, &w, &mut out);
        assert_eq!(
            out.iter().map(|v| v.to_f32()).collect::<Vec<_>>(),
            vec![10.0, 18.0, 42.0, 50.0]
        );

        // Dilation 2, 2x2 kernel: the taps are one row/col apart.
        let mut d = desc(1, 4, 4, 1, 2, 2);
        d.dil_y = 2;
        d.dil_x = 2;
        let mut out = vec![f16::ZERO; 4];
        conv2d_ref(&d, &input, &w, &mut out);
        assert_eq!(
            out[..2].iter().map(|v| v.to_f32()).collect::<Vec<_>>(),
            vec![20.0, 24.0]
        );

        // Depthwise: each channel keeps its own kernel.
        let mut d = desc(2, 2, 2, 2, 1, 1);
        d.depthwise = true;
        let input: Vec<f16> = (1..=8).map(|v| f16::from_f32(v as f32)).collect();
        let w = vec![f16::from_f32(2.0), f16::from_f32(3.0)];
        let mut out = vec![f16::ZERO; 8];
        conv2d_ref(&d, &input, &w, &mut out);
        assert_eq!(out[3].to_f32(), 8.0); // ch0 * 2
        assert_eq!(out[4].to_f32(), 15.0); // ch1 * 3
    }

    #[test]
    fn bands_are_balanced_and_bounded() {
        let mut sizes = Vec::new();
        // 56 rows in bands of at most 16: four bands of 14, not 16/16/16/8.
        let n = balance_bands(56, 16, &mut sizes);
        assert_eq!(n, 4);
        assert_eq!(sizes, vec![14, 14, 14, 14]);
        // 57 -> 15/14/14/14 (descending, the extra first).
        let n = balance_bands(57, 16, &mut sizes);
        assert_eq!(n, 4);
        assert_eq!(sizes, vec![15, 14, 14, 14]);
        // A maximum larger than the total is one band.
        let n = balance_bands(5, 16, &mut sizes);
        assert_eq!((n, sizes.clone()), (1, vec![5]));
        // Never exceeds maxb.
        for total in 1..200usize {
            let n = balance_bands(total, 7, &mut sizes);
            assert!(n >= total.div_ceil(7));
            assert_eq!(sizes.iter().sum::<usize>(), total);
            assert!(sizes.iter().all(|&s| s <= 7));
        }
    }

    #[test]
    fn conv1d_maps_onto_a_height_one_conv() {
        // conv1d is the height-1 conv with the time axis on the height axis; the same
        // weights and data must produce the same numbers through the reference.
        let (ic, it, oc, kw, stride, pad) = (3usize, 9usize, 5usize, 3usize, 2usize, 1usize);
        let mut d = Conv2dDesc {
            ic,
            ih: it,
            iw: 1,
            oc,
            kh: kw,
            kw: 1,
            stride_y: stride,
            stride_x: 1,
            pad_top: pad,
            pad_left: 0,
            ..Default::default()
        };
        let ot = conv2d_oh(&d);
        assert_eq!(ot, (9 + 2 - 3) / 2 + 1);
        let input: Vec<f16> = (0..ic * it)
            .map(|i| f16::from_f32(i as f32 * 0.1))
            .collect();
        let w: Vec<f16> = (0..oc * ic * kw)
            .map(|i| f16::from_f32((i % 7) as f32 * 0.25 - 0.5))
            .collect();
        let mut out = vec![f16::ZERO; oc * ot];
        conv2d_ref(&d, &input, &w, &mut out);
        // The kernel is [OC][IC][KH][KW] with KW == 1, i.e. the conv1d [OC][IC][KW].
        for o in 0..oc {
            for t in 0..ot {
                let mut s = 0.0f32;
                for c in 0..ic {
                    for k in 0..kw {
                        let pos = t as isize * stride as isize + k as isize - pad as isize;
                        if pos < 0 || pos >= it as isize {
                            continue;
                        }
                        s += input[c * it + pos as usize].to_f32()
                            * w[(o * ic + c) * kw + k].to_f32();
                    }
                }
                assert!((out[o * ot + t].to_f32() - s).abs() < 0.02);
            }
        }
        // The degenerate descriptor is refused like the C's conv1d guard.
        assert_eq!(conv2d_plan_code(&d), 0);
        d.pad_top = 0;
        d.stride_y = 1;
        assert_eq!(conv2d_plan_code(&d), 0);
    }

    #[test]
    fn fused_activation_refusals_keep_the_c_codes() {
        let d = desc(32, 8, 8, 16, 3, 3);
        for kind in [Activation::Silu, Activation::Tanh, Activation::Gelu] {
            assert_eq!(conv2d_act_code(&d, kind), 0);
        }
        assert_eq!(
            conv2d_act_code(&d, Activation::HardSwish),
            CONV_ACT_ERR_KIND
        );
        assert_eq!(conv2d_act_code(&d, Activation::Sigmoid), CONV_ACT_ERR_KIND);
        let mut dw = d;
        dw.depthwise = true;
        assert_eq!(
            conv2d_act_code(&dw, Activation::Silu),
            CONV_ACT_ERR_DEPTHWISE
        );
    }
}
