//! fp16 and int8 CONV_2D on the RK3588 NPU (direct, depthwise, 1D, and the
//! conv→activation fusion), plus the int8 multicore pool.
//!
//! Port of `rocket_conv.c` (both halves: the fp16 family and the native int8
//! direct/depthwise entries) plus `rocket_conv_transpose.c`, which lives in
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
//! `ROCKET_CONV_DENTRIES`, `ROCKET_CONV_STRIDE_SWAP`, `ROCKET_CONV_DIL_RAW`,
//! `ROCKET_CONV_BATCH` (the int8 direct conv's coalesced multi-task submit, off by
//! default), plus the LUT builder's own set through `lut_epilogue_build`.
//!
//! # Batched int8 tiles
//!
//! `ROCKET_CONV_BATCH=1` runs a contiguous run of the int8 direct conv's independent tiles
//! as **one** gapped multi-task job (C `conv2d_int8_batch_tiles`): one submit ioctl, one
//! IOMMU attach and one fence wait for the whole run, with the tiles' complete programs in
//! fixed-size slots of one regcmd BO and their feature/weight cubes in bank-aligned slots
//! of one input/weight BO. The hardware still runs the tiles as separate kicks and each
//! writes a disjoint region of the output, so the result is **bit-identical** to the
//! per-tile path, which stays the default.

use half::f16;

use crate::Device;
use crate::device::Bo;
use crate::device::uapi::{DRM_ROCKET_JOB_BATCHED, DrmRocketTask};
use crate::error::{Error, Result};
use crate::knobs;
use crate::log::{log_debug, log_error, log_info};
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
    /// The int8 depthwise output's per-OC int32 bias cube (BRDMA).
    bias: Option<Bo>,
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
            bias: None,
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

    /// `rocket_conv2d_int8_ctx`: the native int8 direct conv on the resident BOs.
    pub fn conv2d_int8(
        &mut self,
        d: &Conv2dDesc,
        input: &[i8],
        w: &[i8],
        out: &mut [i32],
    ) -> Result<()> {
        conv2d_int8_ctx(self, d, input, w, out)
    }

    /// `rocket_conv2d_dw_int8_ctx`: the int8 depthwise int8-out conv on the resident
    /// BOs.
    #[allow(clippy::too_many_arguments)]
    pub fn conv2d_dw_int8(
        &mut self,
        d: &Conv2dDesc,
        input: &[i8],
        w: &[i8],
        bias: Option<&[i32]>,
        in_scale: f32,
        w_scale: f32,
        out_scale: f32,
        in_zp: i32,
        w_zp: i32,
        out_zp: i32,
        out: &mut [i8],
    ) -> Result<()> {
        conv2d_dw_int8_ctx(
            self, d, input, w, bias, in_scale, w_scale, out_scale, in_zp, w_zp, out_zp, out,
        )
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

// ############################################################################
// PART 5 — native int8 CONV_2D (direct, int32-raw)
// ############################################################################

/// The C `DW_INT8_G`: the int8 depthwise weight-cube channel group (Mesa's value —
/// feature atom 16 × 4; the fp16 group is half that at 32).
pub const DW_INT8_G: usize = 64;

/// The int8 feature-tile budget: 8 banks **minus one slack bank**. The int8 `C2 = 16`
/// feature-cube DMA over-reads one CBUF bank past its `ceil` allocation at
/// near-bank-full geometries (the same hardware signature `gen_matmul_int8` has), so
/// the tiler keeps `feature_banks + 1 + weight_banks <= 12`; fp16's 2-byte cube is
/// immune and uses the full 8 (`CONV_FEAT_BUDGET`).
const CONV_FEAT_BUDGET_I8: usize = 7 * CBUF_BANK_SIZE;

/// The C `rocket_conv2d_pool` cap (`ROCKET_CONV_POOL_MAX`).
pub const CONV_POOL_MAX: usize = 8;

/// `rocket_conv2d_ref_int8`: the int64-accumulate → int32 reference.
///
/// int8 × int8 reduced over `KH*KW*IC` is exact in int64 (a 7x7x512 conv sums to ~406M,
/// past int32), and the store is the raw int32 accumulator the NPU writes — the caller
/// applies the per-output-channel requant, exactly as it must for the device output.
/// Depthwise (here: the int32-raw diagnostic form) reduces only each output channel's
/// own input channel with the `[OC][1][KH][KW]` layout, as [`conv2d_ref`] does.
pub fn conv2d_ref_int8(d: &Conv2dDesc, input: &[i8], w: &[i8], out: &mut [i32]) {
    let (oh, ow) = (conv2d_oh(d), conv2d_ow(d));
    let (ic, ih, iw, oc, kh, kw) = (d.ic, d.ih, d.iw, d.oc, d.kh, d.kw);
    for o in 0..oc {
        for y in 0..oh {
            for x in 0..ow {
                let mut s = 0i64;
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
                            let a = input[(c * ih + yy) * iw + xx] as i64;
                            let wv = w[((o * wspan + wic) * kh + k) * kw + j] as i64;
                            s += a * wv;
                        }
                    }
                }
                out[(o * oh + y) * ow + x] = s as i32;
            }
        }
    }
}

/// The int8 conv weight-cube element index (`weight_conv_int8`): the fp16 reorder with
/// the int8 N group of **32** instead of 16 (the K group stays 32).
///
/// `oc`/`ic`/`kh_i`/`kw_i` are 1-based. At `KH = KW = 1` it collapses to the int8
/// matmul's `(N/32, K/32, 32, 32)` weight tile.
pub fn weight_cube_index_i8(
    ic_total: usize,
    kh: usize,
    kw: usize,
    oc: usize,
    ic: usize,
    kh_i: usize,
    kw_i: usize,
) -> usize {
    let oc1 = (oc - 1) / 32;
    let oc2 = (oc - 1) % 32;
    let ic1 = (ic - 1) / 32;
    let ic2 = (ic - 1) % 32;
    let n_ic1 = ic_total.div_ceil(32);
    ((((oc1 * n_ic1 + ic1) * kh + (kh_i - 1)) * kw + (kw_i - 1)) * 32 + oc2) * 32 + ic2
}

/// The int8 feature-cube element index: the same NC1HWC2 math as the fp16 cube with the
/// int8 channel atom `C2 = 16`.
pub fn feature_cube_index_i8(
    channels: usize,
    pix_h: usize,
    pix_w: usize,
    ch: usize,
    row: usize,
    col: usize,
) -> usize {
    feature_data(channels, pix_h, pix_w, 16, ch, row, col)
}

/// Scatter a row-major int8 `[c][h][w]` feature into an NC1HWC2 cube (`C2 = 16`).
pub fn scatter_feature_i8(cube: &mut [i8], c: usize, h: usize, w: usize, src: &[i8]) {
    for ch in 0..c {
        for row in 0..h {
            for col in 0..w {
                cube[feature_cube_index_i8(c, h, w, ch + 1, row + 1, col + 1)] =
                    src[(ch * h + row) * w + col];
            }
        }
    }
}

/// Scatter row-major int8 direct weights `[oc][ic][kh][kw]` into the int8 weight cube.
pub fn scatter_weights_i8(cube: &mut [i8], oc: usize, ic: usize, kh: usize, kw: usize, w: &[i8]) {
    for o in 0..oc {
        for k in 0..kh {
            for j in 0..kw {
                for i in 0..ic {
                    cube[weight_cube_index_i8(ic, kh, kw, o + 1, i + 1, k + 1, j + 1)] =
                        w[((o * ic + i) * kh + k) * kw + j];
                }
            }
        }
    }
}

/// Scatter row-major int8 depthwise weights `[c][kh][kw]` into the `(C/G, KH, KW, G)`
/// cube — the same element math as [`scatter_weights_dw`], only the group differs.
pub fn scatter_weights_dw_i8(cube: &mut [i8], c: usize, kh: usize, kw: usize, g: usize, w: &[i8]) {
    for ch in 0..c {
        for k in 0..kh {
            for j in 0..kw {
                cube[weight_cube_dw_index(kh, kw, g, ch + 1, k + 1, j + 1)] =
                    w[(ch * kh + k) * kw + j];
            }
        }
    }
}

/// Gather an int32 output cube (`C2 = 4`) back to row-major `[oc][oh][ow]`.
pub fn gather_output_i32(cube: &[i32], oc: usize, oh: usize, ow: usize, out: &mut [i32]) {
    for ch in 0..oc {
        for row in 0..oh {
            for col in 0..ow {
                out[(ch * oh + row) * ow + col] =
                    cube[feature_data(oc, oh, ow, 4, ch + 1, row + 1, col + 1)];
            }
        }
    }
}

/// Gather an int8 output cube (`C2 = 16`) back to row-major `[c][oh][ow]`.
pub fn gather_output_i8(cube: &[i8], c: usize, oh: usize, ow: usize, out: &mut [i8]) {
    for ch in 0..c {
        for row in 0..oh {
            for col in 0..ow {
                out[(ch * oh + row) * ow + col] =
                    cube[feature_data(c, oh, ow, 16, ch + 1, row + 1, col + 1)];
            }
        }
    }
}

fn bo_i8(bo: &Bo) -> &[i8] {
    let bytes = bo.as_slice();
    // SAFETY: i8 has no alignment or validity requirement beyond a byte.
    unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<i8>(), bytes.len()) }
}

fn bo_i8_mut(bo: &mut Bo) -> &mut [i8] {
    let len = bo.len();
    let ptr = bo.as_ptr();
    // SAFETY: exclusive borrow of the BO, page-aligned mapping, 1-byte elements.
    unsafe { std::slice::from_raw_parts_mut(ptr.cast::<i8>(), len) }
}

fn bo_i32(bo: &Bo) -> &[i32] {
    let bytes = bo.as_slice();
    // SAFETY: the mapping is page-aligned (so 4-byte aligned) and every prefix of it is
    // initialized; the length is truncated to whole words.
    unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<i32>(), bytes.len() / 4) }
}

/// `rocket_conv2d_int8_run`'s pre-flight checks, as one pure predicate (the C folds
/// them into the run; this port keeps them addressable for tests and callers).
///
/// `0` runnable, `-1` malformed, `-2` depthwise (this is the direct entry), `-3` a
/// collapsed output extent, `-4` a tile that cannot fit the CBUF geometry.
pub fn conv2d_int8_plan_code(d: &Conv2dDesc) -> i32 {
    if d.depthwise {
        return -2;
    }
    if d.ic == 0 || d.ih == 0 || d.iw == 0 || d.oc == 0 || d.kh == 0 || d.kw == 0 {
        return -1;
    }
    if d.stride_x == 0 || d.stride_y == 0 || d.dil_x == 0 || d.dil_y == 0 {
        return -1;
    }
    let (oh, ow) = (conv2d_oh(d), conv2d_ow(d));
    if oh == 0 || ow == 0 {
        return -3;
    }
    let ic = d.ic.div_ceil(32) * 32;
    if d.kw * d.kh * ic > CBUF_BANK_SIZE {
        return -4;
    }
    let ih_min = (d.kh - 1) * d.dil_y + 1;
    let iw_min = (d.kw - 1) * d.dil_x + 1;
    if ic * ih_min * iw_min > CONV_FEAT_BUDGET_I8 {
        return -4;
    }
    0
}

/// `rocket_conv2d_dw_int8_run`'s pre-flight checks.
///
/// `0` runnable, `-1` malformed, `-2` not depthwise / `OC != IC`, `-3` a collapsed
/// output extent, `-4` a channel chunk that cannot fit one CBUF pass, `-5` a channel
/// count that is not a multiple of 16 (the `C2 = 16` feature-cube grain).
pub fn conv2d_dw_int8_plan_code(d: &Conv2dDesc) -> i32 {
    if !d.depthwise || d.oc != d.ic {
        return -2;
    }
    if d.ic == 0 || d.ih == 0 || d.iw == 0 || d.kh == 0 || d.kw == 0 {
        return -1;
    }
    if d.stride_x == 0 || d.stride_y == 0 || d.dil_x == 0 || d.dil_y == 0 {
        return -1;
    }
    if !d.ic.is_multiple_of(16) {
        return -5;
    }
    let (oh, ow) = (conv2d_oh(d), conv2d_ow(d));
    if oh == 0 || ow == 0 {
        return -3;
    }
    if d.kw * d.kh * DW_INT8_G > CBUF_BANK_SIZE {
        return -4;
    }
    let cc = dw_int8_chunk(d.ic, d.ih, d.iw);
    if cc * d.ih * d.iw > CONV_FEAT_BUDGET {
        return -4;
    }
    0
}

/// The largest `G`-multiple channel chunk whose full feature fits one CBUF pass
/// (`conv2d_dw_int8_run`'s `Cc`; there is no spatial tile on the int8 DW path).
pub fn dw_int8_chunk(c: usize, ih: usize, iw: usize) -> usize {
    let mut cc = c;
    while cc > DW_INT8_G && cc * ih * iw > CONV_FEAT_BUDGET {
        cc -= DW_INT8_G;
    }
    cc
}

/// One independent int8 direct-conv tile (`i8tile`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvInt8Tile {
    /// First output channel and how many this tile writes.
    pub oc0: usize,
    pub oc_n: usize,
    /// First output row / how many rows.
    pub r0: usize,
    pub rh: usize,
    /// First output column / how many columns.
    pub c0: usize,
    pub cw: usize,
}

/// The int8 direct tiler's decomposition, as the C computes it inline before
/// enumerating tiles.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvInt8Tiles {
    /// Input channels including the 32-channel pad.
    pub ic: usize,
    /// Output-channel tile (a multiple of the 32-channel weight group).
    pub oc_tile: usize,
    /// Balanced output-row bands.
    pub rband: Vec<usize>,
    /// Balanced output-column bands.
    pub cband: Vec<usize>,
}

impl ConvInt8Tiles {
    /// The independent `(oc-group × row-band × col-band)` tiles, in the C's order
    /// (output channel outer, then row band, then column band). Each writes a disjoint
    /// region of the output, so they may run on several cores.
    pub fn tiles(&self, d: &Conv2dDesc) -> Vec<ConvInt8Tile> {
        let mut out = Vec::new();
        let mut oc0 = 0;
        while oc0 < d.oc {
            let oc_n = (d.oc - oc0).min(self.oc_tile);
            let mut r0 = 0;
            for &rh in &self.rband {
                let mut c0 = 0;
                for &cw in &self.cband {
                    out.push(ConvInt8Tile {
                        oc0,
                        oc_n,
                        r0,
                        rh,
                        c0,
                        cw,
                    });
                    c0 += cw;
                }
                r0 += rh;
            }
            oc0 += oc_n;
        }
        out
    }

    /// Tiles the plan produces.
    pub fn count(&self, d: &Conv2dDesc) -> usize {
        let nrow: usize = self.rband.len();
        let ncol: usize = self.cband.len();
        d.oc.div_ceil(self.oc_tile) * nrow * ncol
    }
}

/// The int8 direct tiler's budgets (`conv2d_int8_run`'s OC/row/column decomposition).
///
/// OC tiles in multiples of 32 (the int8 weight oc group) and the feature budget
/// jointly share the 12 CBUF banks: the generator gives the weight `banks - (fd+1)`, so
/// a tile is valid only when its weight cube fits what the feature leaves. Prefer a
/// weight within ~4 banks and shrink `OCt` toward 32; when even one 32-group needs more
/// (a KxK conv at large IC), accept the bigger weight and shrink the feature budget.
pub fn conv_int8_tile_plan(d: &Conv2dDesc) -> Result<ConvInt8Tiles> {
    let ic = d.ic.div_ceil(32) * 32;
    if d.kw * d.kh * ic > CBUF_BANK_SIZE {
        return Err(Error::tiling("rocket_conv2d_int8/weight-bank"));
    }
    let banks = crate::hw::current().cbuf_banks;
    let ocw = |oct: usize| (oct * ic * d.kh * d.kw).div_ceil(CBUF_BANK_SIZE);
    let mut oc_tile = d.oc;
    while oc_tile > 32 && ocw(oc_tile) > 4 {
        oc_tile -= 32;
    }
    let wbanks = ocw(oc_tile);
    // Leave >= 1 feature bank plus the C2=16 slack bank.
    if wbanks > banks - 2 {
        return Err(Error::tiling("rocket_conv2d_int8/banks"));
    }
    let feat_budget = ((banks - 1 - wbanks) * CBUF_BANK_SIZE).min(CONV_FEAT_BUDGET_I8);

    let (oh, ow) = (conv2d_oh(d), conv2d_ow(d));
    let min_ihs = {
        let v = match knobs::int("ROCKET_CONV_MIN_IH") {
            Some(v) if v > 0 => v as usize,
            _ => 6,
        };
        v.max(CONV_MIN_DATAIN_H) // never below the hardware row floor
    };
    let feat = |rh: usize, cw: usize| {
        ic * datain_h(rh, d.stride_y, d.kh, d.dil_y) * datain_h(cw, d.stride_x, d.kw, d.dil_x)
    };
    let mut rht_lo = 1;
    while rht_lo < oh && datain_h(rht_lo, d.stride_y, d.kh, d.dil_y) < min_ihs {
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
        return Err(Error::tiling("rocket_conv2d_int8/cbuf"));
    }
    let mut rband = Vec::new();
    let mut cband = Vec::new();
    balance_bands(oh, rht, &mut rband);
    balance_bands(ow, cwt, &mut cband);
    Ok(ConvInt8Tiles {
        ic,
        oc_tile,
        rband,
        cband,
    })
}

/// Run ONE int8 direct conv as a single NPU job (must fit one CBUF pass).
///
/// `in [IC][IH][IW]` int8, `W [OC][IC][KH][KW]` int8, `out [OC][OH][OW]` int32. The
/// caller has already zero-padded `IC` to 32 and `OC` to 32 (the two cube groups).
#[allow(clippy::too_many_arguments)]
fn conv2d_int8_one_job(
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
    input: &[i8],
    w: &[i8],
    out: &mut [i32],
) -> Result<()> {
    let dev = ctx.dev.clone();
    // A datain_height below 4 reads the cube wrong at every IC: pad the job's input
    // height with zero rows below the real data. OH stays the real output count, so the
    // padded rows never enter a computed output's receptive field.
    let ihj = ih.max(CONV_MIN_DATAIN_H);
    let cube_in = ic.div_ceil(16) * 16 * ihj * iw; // int8 bytes
    let wt_elems = oc * ic * kh * kw;
    let cube_out = oc.div_ceil(4) * 4 * oh * ow; // int32 words
    let nregs = crate::regcmd::conv::MAX_OPS;

    ensure_bo(&dev, &mut ctx.guard, 4096)?;
    ensure_bo(&dev, &mut ctx.input, cube_in + CBUF_BANK_SIZE)?;
    ensure_bo(&dev, &mut ctx.weight, wt_elems + CBUF_BANK_SIZE)?;
    ensure_bo(&dev, &mut ctx.rc, nregs * 8)?;
    ensure_bo(&dev, &mut ctx.output, cube_out * 4 + CBUF_BANK_SIZE)?;

    {
        let bo = ctx.input.as_mut().expect("input BO");
        bo.prep(0)?;
        let words = bo_i8_mut(bo);
        words.fill(0);
        scatter_feature_i8(words, ic, ihj, iw, input);
        bo.fini()?;
    }
    {
        let bo = ctx.weight.as_mut().expect("weight BO");
        bo.prep(0)?;
        let words = bo_i8_mut(bo);
        words.fill(0);
        scatter_weights_i8(words, oc, ic, kh, kw, w);
        bo.fini()?;
    }

    let mut p = crate::regcmd::conv::ConvInt8Params::new(
        ic as u16,
        ihj as u16,
        iw as u16,
        oc as u16,
        oh as u16,
        ow as u16,
        kh as u16,
        kw as u16,
        ctx.input.as_ref().expect("input BO").dma_address() as u32,
        ctx.weight.as_ref().expect("weight BO").dma_address() as u32,
        ctx.output.as_ref().expect("output BO").dma_address() as u32,
    );
    p.stride_y = sy as u8;
    p.stride_x = sx as u8;
    p.pad_top = pt as u8;
    p.pad_left = pl as u8;
    p.dil_y = dy as u8;
    p.dil_x = dx as u8;

    let mut ops = Vec::with_capacity(nregs);
    let count = crate::regcmd::conv::gen_conv2d_int8(&mut p, &mut ops)?;
    if count > nregs {
        log_error!("rocket_conv2d_int8: regcmd overflow (task_count {count} > {nregs} words)");
        return Err(Error::tiling("rocket_conv2d_int8/regcmd"));
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
        bo_i8_mut(bo).fill(0);
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

    {
        let bo = ctx.output.as_mut().expect("output BO");
        bo.prep(CONV_WAIT_NS)?;
        gather_output_i32(&bo_i32(bo)[..cube_out], oc, oh, ow, out);
        bo.fini()?;
    }
    Ok(())
}

/// Materialise one tile's sub-input: the real halo rows/columns plus the explicit zero
/// pad, with the channel pad (`ic >= icr` stays zero from the fill).
///
/// `gh0`/`gw0` are the sub-input's top-left corner in the global input, i.e.
/// `r0 * stride_y - pad_top` and `c0 * stride_x - pad_left`. Shared by the per-tile job
/// and the batched slots so both hand the CNA the same bytes.
#[allow(clippy::too_many_arguments)]
fn materialize_sub_input(
    sub_in: &mut [i8],
    ic: usize,
    icr: usize,
    ih: usize,
    iw: usize,
    ih_sub: usize,
    iw_sub: usize,
    gh0: isize,
    gw0: isize,
    input: &[i8],
) {
    sub_in[..ic * ih_sub * iw_sub].fill(0);
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
}

/// One independent OC/OH/OW tile of the int8 direct conv (`conv2d_int8_one_tile`):
/// materialise the tile's sub-input (real halo + zero pad + channel pad), run the
/// single job, scatter the int32 result into the output's disjoint region.
#[allow(clippy::too_many_arguments)]
fn conv2d_int8_one_tile(
    ctx: &mut ConvContext,
    ic: usize,
    icr: usize,
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
    input: &[i8],
    wuse: &[i8],
    out: &mut [i32],
    w_oc0: usize,
    out_oc0: usize,
    oc_n: usize,
    r0: usize,
    rh: usize,
    c0: usize,
    cw: usize,
    sub_in: &mut [i8],
    sub_out: &mut [i32],
) -> Result<()> {
    let ih_sub = (rh - 1) * sy + (kh - 1) * dy + 1;
    let gh0 = r0 as isize * sy as isize - pt as isize;
    let iw_sub = (cw - 1) * sx + (kw - 1) * dx + 1;
    let gw0 = c0 as isize * sx as isize - pl as isize;

    materialize_sub_input(sub_in, ic, icr, ih, iw, ih_sub, iw_sub, gh0, gw0, input);

    // `w_oc0` is the tile's place in the *weight* tensor (always the global output
    // channel); `out_oc0` is its place in the caller's `out`, which a multicore worker
    // has already offset to its own output-channel band.
    let wslice = &wuse[w_oc0 * ic * kh * kw..(w_oc0 + oc_n) * ic * kh * kw];
    conv2d_int8_one_job(
        ctx, ic, ih_sub, iw_sub, oc_n, rh, cw, kh, kw, sy, sx, 0, 0, dy, dx, sub_in, wslice,
        sub_out,
    )?;

    for o in 0..oc_n {
        for r in 0..rh {
            let dst = ((out_oc0 + o) * oh + (r0 + r)) * ow + c0;
            let src = (o * rh + r) * cw;
            out[dst..dst + cw].copy_from_slice(&sub_out[src..src + cw]);
        }
    }
    Ok(())
}

// ── Batched (one gapped multi-task job) int8 direct tiles ─────────────────────

/// The C `CONV_RC_STRIDE`: the per-task regcmd slot of a batched int8 conv job.
///
/// It is the generator's own word cap ([`MAX_OPS`](crate::regcmd::conv::MAX_OPS), 256
/// words), so a slot always holds a whole program and the layout never has to vary its
/// stride: the slots are fixed-size, exactly as the C lays them out.
pub const CONV_RC_STRIDE: usize = crate::regcmd::conv::MAX_OPS;

/// Whether the int8 direct conv coalesces a run of tiles into one multi-task submit
/// (C `conv_batch_enabled`, `ROCKET_CONV_BATCH`).
///
/// The C's rule is `atoi(getenv(...)) > 0` with the answer cached once per process; this
/// reads it per call so an A/B can flip it inside one process. Unset (or `0`) keeps the
/// per-tile path, which is the default and stays byte-for-byte what it was.
pub fn conv_batch_enabled() -> bool {
    knobs::int("ROCKET_CONV_BATCH").is_some_and(|v| v > 0)
}

/// Round `n` bytes up to a whole CBUF bank (C `bank_round`).
///
/// Every slot of a batched job starts on a bank boundary, so its base has the alignment a
/// standalone job's freshly allocated BO has and the int8 `C2 = 16` feature DMA's one-bank
/// over-read lands in the zeroed spare bank that follows it.
fn bank_round(n: usize) -> usize {
    n.div_ceil(CBUF_BANK_SIZE) * CBUF_BANK_SIZE
}

/// One tile's slots inside a batched int8 conv job (C `conv2d_int8_batch_tiles` pass 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvBatchSlot {
    /// The tile this slot serves.
    pub tile: ConvInt8Tile,
    /// The tile's real (halo-grown) sub-input extent, and its height at the 4-row CNA
    /// DMA floor.
    pub ih_sub: usize,
    pub iw_sub: usize,
    pub ihj: usize,
    /// Bytes of row-major sub-input scratch this tile needs.
    pub sub_in: usize,
    /// Byte offsets of this tile's input / weight / output slot in the job's BOs.
    pub in_off: usize,
    pub wt_off: usize,
    pub out_off: usize,
}

impl ConvBatchSlot {
    /// The tile's scattered feature-cube size in bytes (`C2 = 16`, always `ihj` rows).
    pub fn in_bytes(&self, ic: usize) -> usize {
        ic.div_ceil(16) * 16 * self.ihj * self.iw_sub
    }

    /// The tile's scattered weight-cube size in bytes (the int8 `oc` group is 32).
    ///
    /// The tiler keeps `oc_n` a multiple of 32, so this is the C's `OCn * IC * KH * KW`;
    /// the rounding only guards a hand-built tile list from writing past its slot.
    pub fn wt_bytes(&self, ic: usize, kh: usize, kw: usize) -> usize {
        self.tile.oc_n.div_ceil(32) * 32 * ic * kh * kw
    }

    /// The tile's int32 output-cube size in bytes (`C2 = 4`).
    pub fn out_bytes(&self) -> usize {
        self.tile.oc_n.div_ceil(4) * 4 * self.tile.rh * self.tile.cw * 4
    }
}

/// A batched job's slot layout: the per-tile slots and the BO sizes they need
/// (C `conv2d_int8_batch_tiles` pass 1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvBatchLayout {
    /// One slot per tile, in tile order.
    pub slots: Vec<ConvBatchSlot>,
    /// Bytes the job's input / weight / output BOs must hold.
    pub in_bytes: usize,
    pub wt_bytes: usize,
    pub out_bytes: usize,
    /// Largest single-tile sub-input, for the host scratch.
    pub max_sub_in: usize,
}

/// Lay `tiles` into bank-aligned slots of one batched job.
///
/// `ic` is the padded input-channel count the tiler resolved (`plan.ic`), i.e. what the
/// per-tile job scatters. Each region is `bank_round(cube) + CBUF_BANK` bytes, so every
/// slot starts on a bank boundary and is followed by the zeroed spare bank the int8
/// feature DMA over-reads into — exactly the bytes a standalone job's own BO provides.
pub fn conv_batch_layout(d: &Conv2dDesc, ic: usize, tiles: &[ConvInt8Tile]) -> ConvBatchLayout {
    let (mut in_bytes, mut wt_bytes, mut out_bytes, mut max_sub_in) = (0, 0, 0, 0);
    let mut slots = Vec::with_capacity(tiles.len());
    for &tile in tiles {
        let ih_sub = (tile.rh - 1) * d.stride_y + (d.kh - 1) * d.dil_y + 1;
        let iw_sub = (tile.cw - 1) * d.stride_x + (d.kw - 1) * d.dil_x + 1;
        let slot = ConvBatchSlot {
            tile,
            ih_sub,
            iw_sub,
            ihj: ih_sub.max(CONV_MIN_DATAIN_H),
            sub_in: ic * ih_sub * iw_sub,
            in_off: in_bytes,
            wt_off: wt_bytes,
            out_off: out_bytes,
        };
        max_sub_in = max_sub_in.max(slot.sub_in);
        in_bytes += bank_round(slot.in_bytes(ic)) + CBUF_BANK_SIZE;
        wt_bytes += bank_round(slot.wt_bytes(ic, d.kh, d.kw)) + CBUF_BANK_SIZE;
        out_bytes += bank_round(slot.out_bytes()) + CBUF_BANK_SIZE;
        slots.push(slot);
    }
    ConvBatchLayout {
        slots,
        in_bytes,
        wt_bytes,
        out_bytes,
        max_sub_in,
    }
}

/// Every tile's program of a batched job, concatenated, with each tile's `(first word,
/// word count)` in the run (C `conv2d_int8_batch_tiles`' program pass).
struct BatchPrograms {
    group: Vec<u64>,
    spans: Vec<(usize, u32)>,
}

/// Generate each slot's program against the slot's addresses.
///
/// Device-free (the emitters are pure), so a host test can gate the batched slot programs
/// against the per-tile job's. The geometry is `conv2d_int8_one_tile`'s: the tile's real
/// output extent, the halo-grown sub-input (raised to the 4-row DMA floor), and
/// `pad_top = pad_left = 0` because the halo is materialised.
fn batch_programs(
    layout: &ConvBatchLayout,
    ic: usize,
    d: &Conv2dDesc,
    in_dma: u32,
    wt_dma: u32,
    out_dma: u32,
) -> Result<BatchPrograms> {
    let mut group: Vec<u64> = Vec::with_capacity(layout.slots.len() * CONV_RC_STRIDE);
    let mut spans: Vec<(usize, u32)> = Vec::with_capacity(layout.slots.len());
    for slot in &layout.slots {
        let t = slot.tile;
        let mut p = crate::regcmd::conv::ConvInt8Params::new(
            ic as u16,
            slot.ihj as u16,
            slot.iw_sub as u16,
            t.oc_n as u16,
            t.rh as u16,
            t.cw as u16,
            d.kh as u16,
            d.kw as u16,
            in_dma.wrapping_add(slot.in_off as u32),
            wt_dma.wrapping_add(slot.wt_off as u32),
            out_dma.wrapping_add(slot.out_off as u32),
        );
        p.stride_y = d.stride_y as u8;
        p.stride_x = d.stride_x as u8;
        p.dil_y = d.dil_y as u8;
        p.dil_x = d.dil_x as u8;
        let mut ops = Vec::with_capacity(CONV_RC_STRIDE);
        let count = crate::regcmd::conv::gen_conv2d_int8(&mut p, &mut ops)?;
        if count > CONV_RC_STRIDE {
            log_error!(
                "rocket_conv2d_int8 batch: regcmd overflow (task_count {count} > \
                 {CONV_RC_STRIDE} words)"
            );
            return Err(Error::tiling("rocket_conv2d_int8/regcmd"));
        }
        spans.push((group.len(), count as u32));
        group.extend(ops.iter().map(|op| op.0));
    }
    Ok(BatchPrograms { group, spans })
}

/// Whether a group's programs go down the contiguous self-chained layout, and the regcmd
/// words that needs. `Some(words)` = chained, `None` = gapped slots.
///
/// Chaining is **this port's** extension — the C's conv batch always packs gapped slots
/// (`rkt_chain_pack(0, ...)`) — and it needs every program the same length, so a run whose
/// lengths disagree keeps the gapped slots. A chained run must also fit the gapped
/// layout's own budget, so a batch never reserves more regcmd BO for the chain than it
/// would for fixed slots. `want` is [`chain::enabled`](crate::matmul::chain::enabled),
/// passed in so the decision is host-testable.
fn batch_chain_plan(want: bool, spans: &[(usize, u32)], gapped_words: usize) -> Option<usize> {
    let uniform = spans.first().is_some_and(|first| {
        spans
            .iter()
            .all(|s| s.1 == first.1 && s.1 <= CONV_RC_STRIDE as u32)
    });
    let chained = uniform
        && crate::matmul::chain::group_chained(
            want,
            spans.len(),
            spans.iter().map(|s| s.1),
            gapped_words,
        );
    chained.then(|| crate::matmul::chain::words(spans[0].1) * spans.len())
}

/// Split a run of `tn` tiles into the groups one batched job carries.
///
/// A group holds at most [`BATCH`](crate::regcmd::desc::BATCH) tiles — the matmul batch's
/// job size, which bounds the batched BOs — and never a single trailing tile: one tile is
/// what the per-tile path is for, so a lone tail folds into the group before it.
fn batch_groups(tn: usize) -> Vec<std::ops::Range<usize>> {
    let mut groups = Vec::new();
    let mut start = 0;
    while start < tn {
        let mut end = (start + crate::regcmd::desc::BATCH).min(tn);
        if tn - end == 1 {
            end += 1;
        }
        groups.push(start..end);
        start = end;
    }
    groups
}

/// De-scatter one tile's int32 output cube straight into its region of `out`.
///
/// The batch has no per-tile `sub_out` scratch: every tile writes only its own rows and
/// columns of the caller's output (the C's de-scatter does the same), so `out_oc0` is the
/// tile's first channel *in this slice* — a multicore worker passes its band's base.
fn gather_output_i32_tile(
    cube: &[i32],
    t: &ConvInt8Tile,
    oh: usize,
    ow: usize,
    out_oc0: usize,
    out: &mut [i32],
) {
    for o in 0..t.oc_n {
        for r in 0..t.rh {
            for c in 0..t.cw {
                out[((out_oc0 + o) * oh + (t.r0 + r)) * ow + t.c0 + c] =
                    cube[feature_data(t.oc_n, t.rh, t.cw, 4, o + 1, r + 1, c + 1)];
            }
        }
    }
}

/// Batch `tiles` into ONE multi-task int8 conv job (C `conv2d_int8_batch_tiles`).
///
/// Every tile's complete program goes into its own [`CONV_RC_STRIDE`]-word slot of one
/// regcmd BO, and its feature/weight cubes into bank-aligned slots of one input/weight BO;
/// the whole run is then submitted as a single job — one ioctl, one IOMMU attach and one
/// fence wait, while the hardware still runs the tiles as separate kicks. Every tile
/// carries the program the per-tile path emits against the same cube bytes, so the batch
/// is **bit-identical** to [`conv2d_int8_one_tile`] per tile; only the submit is coalesced.
/// Opt-in through [`conv_batch_enabled`], which `conv2d_int8_run` reads once per call.
///
/// No capability gate is needed: the gapped multi-task job is the stock driver's own shape
/// (the matmul batch already relies on it), and the **chained** layout — this port's
/// extension when [`chain::enabled`](crate::matmul::chain::enabled) — is gated by
/// [`batch_chain_plan`], which asks the kernel's batched-submit probe itself.
///
/// `out_oc0_base` is the output channel `out[0]` stands for: `0` on the serial path, a
/// multicore worker's band base otherwise (the C's `out_oc0`).
#[allow(clippy::too_many_arguments)]
fn conv2d_int8_batch_tiles(
    ctx: &mut ConvContext,
    ic: usize,
    d: &Conv2dDesc,
    input: &[i8],
    wuse: &[i8],
    out: &mut [i32],
    out_oc0_base: usize,
    tiles: &[ConvInt8Tile],
) -> Result<()> {
    if tiles.is_empty() {
        return Ok(());
    }
    let dev = ctx.dev.clone();
    let layout = conv_batch_layout(d, ic, tiles);
    let (ih, iw) = (d.ih, d.iw);
    let (oh, ow) = (conv2d_oh(d), conv2d_ow(d));
    let (kh, kw) = (d.kh, d.kw);

    ensure_bo(&dev, &mut ctx.guard, 4096)?;
    ensure_bo(&dev, &mut ctx.input, layout.in_bytes)?;
    ensure_bo(&dev, &mut ctx.weight, layout.wt_bytes)?;
    ensure_bo(&dev, &mut ctx.output, layout.out_bytes)?;

    // The emitters read their slots' addresses, so every BO but the regcmd one is grown
    // first (growing a BO moves it).
    let progs = batch_programs(
        &layout,
        ic,
        d,
        ctx.input.as_ref().expect("input BO").dma_address() as u32,
        ctx.weight.as_ref().expect("weight BO").dma_address() as u32,
        ctx.output.as_ref().expect("output BO").dma_address() as u32,
    )?;
    let g = progs.spans.len();
    let gapped_words = g * CONV_RC_STRIDE;
    let chain_words = batch_chain_plan(crate::matmul::chain::enabled(), &progs.spans, gapped_words);
    let chained = chain_words.is_some();
    ensure_bo(&dev, &mut ctx.rc, chain_words.unwrap_or(gapped_words) * 8)?;
    if knobs::present("ROCKET_CONV_BATCH_DEBUG") {
        let second = layout.slots.get(1).or_else(|| layout.slots.first());
        log_info!(
            "conv int8 batch: {g} tile(s) of {} words, {}, rc {} words, in {} B + wt {} B + \
             out {} B, slot 1 at in+{} wt+{} out+{}",
            progs.spans[0].1,
            if chained {
                "contiguous + self-chained"
            } else {
                "gapped slots"
            },
            chain_words.unwrap_or(gapped_words),
            layout.in_bytes,
            layout.wt_bytes,
            layout.out_bytes,
            second.map_or(0, |s| s.in_off),
            second.map_or(0, |s| s.wt_off),
            second.map_or(0, |s| s.out_off),
        );
    }

    // Feature cubes: every tile's halo/pad sub-input, scattered into its own slot. The
    // whole BO is zeroed, so the slots' spare banks and the channel pad read zero exactly
    // as a fresh single-job BO does.
    let mut sub_in = vec![0i8; layout.max_sub_in.max(1)];
    {
        let bo = ctx.input.as_mut().expect("input BO");
        bo.prep(0)?;
        let words = bo_i8_mut(bo);
        words.fill(0);
        for slot in &layout.slots {
            let t = slot.tile;
            let gh0 = t.r0 as isize * d.stride_y as isize - d.pad_top as isize;
            let gw0 = t.c0 as isize * d.stride_x as isize - d.pad_left as isize;
            materialize_sub_input(
                &mut sub_in,
                ic,
                d.ic,
                ih,
                iw,
                slot.ih_sub,
                slot.iw_sub,
                gh0,
                gw0,
                input,
            );
            let cube = &mut words[slot.in_off..slot.in_off + slot.in_bytes(ic)];
            scatter_feature_i8(cube, ic, slot.ihj, slot.iw_sub, &sub_in[..slot.sub_in]);
        }
        bo.fini()?;
    }
    // Weight cubes: each tile's own OC slice of the (already IC-padded) weights.
    {
        let bo = ctx.weight.as_mut().expect("weight BO");
        bo.prep(0)?;
        let words = bo_i8_mut(bo);
        words.fill(0);
        let span = ic * kh * kw;
        for slot in &layout.slots {
            let t = slot.tile;
            let wslice = &wuse[t.oc0 * span..(t.oc0 + t.oc_n) * span];
            let cube = &mut words[slot.wt_off..slot.wt_off + slot.wt_bytes(ic, kh, kw)];
            scatter_weights_i8(cube, t.oc_n, ic, kh, kw, wslice);
        }
        bo.fini()?;
    }
    {
        let bo = ctx.output.as_mut().expect("output BO");
        bo.prep(0)?;
        bo_i8_mut(bo).fill(0);
        bo.fini()?;
    }

    // Regcmd: each program in its slot (gapped, or contiguous and self-chained).
    let mut tasks = vec![DrmRocketTask::default(); g];
    {
        let bo = ctx.rc.as_mut().expect("rc BO");
        bo.prep(0)?;
        let base = bo.dma_address() as u32;
        let dst = crate::matmul::bos::u64_words(bo);
        for (i, &(off, count)) in progs.spans.iter().enumerate() {
            crate::matmul::chain::pack(
                chained,
                dst,
                base,
                &mut tasks,
                i,
                &progs.group[off..off + count as usize],
                count,
                CONV_RC_STRIDE,
            )?;
        }
        if chained {
            crate::matmul::chain::verify(
                dst,
                base,
                &tasks,
                crate::matmul::chain::words(progs.spans[0].1),
            )?;
        }
        bo.fini()?;
    }

    let in_handles = [
        ctx.input.as_ref().expect("input BO").handle(),
        ctx.weight.as_ref().expect("weight BO").handle(),
        ctx.rc.as_ref().expect("rc BO").handle(),
    ];
    let out_handles = [ctx.output.as_ref().expect("output BO").handle()];
    let flags = if chained { DRM_ROCKET_JOB_BATCHED } else { 0 };
    let mut scratch: Vec<DrmRocketTask> = Vec::with_capacity(g);
    dev.submit_tasks_pre(&mut scratch, &tasks, &in_handles, &out_handles, flags)?;

    // ONE fence wait for the whole job, then every tile's de-scatter.
    {
        let bo = ctx.output.as_mut().expect("output BO");
        bo.prep(CONV_WAIT_NS)?;
        let words = bo_i32(bo);
        for slot in &layout.slots {
            let cube = &words[slot.out_off / 4..];
            gather_output_i32_tile(cube, &slot.tile, oh, ow, slot.tile.oc0 - out_oc0_base, out);
        }
        bo.fini()?;
    }
    Ok(())
}

/// The int8 direct tiling driver (`conv2d_int8_run`).
///
/// `ctxs[0]` is the coordinator (the C's `fd`/`ctx`), and with more than one context the
/// independent tiles fan out across them — one NPU core each, `ctxs.len()` workers.
fn conv2d_int8_run(
    ctxs: &mut [ConvContext],
    d: &Conv2dDesc,
    input: &[i8],
    w: &[i8],
    out: &mut [i32],
) -> Result<()> {
    if d.depthwise {
        // int8 depthwise is the int8-out entry (`conv2d_dw_int8`).
        return Err(Error::coded(
            crate::Status::Shape,
            -2,
            "rocket_conv2d_int8/depthwise",
        ));
    }
    if ctxs.is_empty() {
        return Err(Error::shape("rocket_conv2d_int8/no-context"));
    }

    // OC % 32 != 0: pad OC up to the int8 weight oc group, run, slice the real OC off
    // the front. The extra kernels are zero and each direct output channel is
    // independent, so the first OC channels are bit-exact.
    if !d.oc.is_multiple_of(32) {
        let oc = d.oc;
        let ocp = oc.div_ceil(32) * 32;
        let (oh, ow) = (conv2d_oh(d), conv2d_ow(d));
        if oh == 0 || ow == 0 {
            return Err(Error::coded(
                crate::Status::Shape,
                -3,
                "rocket_conv2d_int8/out-extent",
            ));
        }
        let mut dp = *d;
        dp.oc = ocp;
        let mut wp = vec![0i8; ocp * d.ic * d.kh * d.kw];
        wp[..oc * d.ic * d.kh * d.kw].copy_from_slice(&w[..oc * d.ic * d.kh * d.kw]);
        let mut op = vec![0i32; ocp * oh * ow];
        conv2d_int8_run(ctxs, &dp, input, &wp, &mut op)?;
        out[..oc * oh * ow].copy_from_slice(&op[..oc * oh * ow]);
        return Ok(());
    }

    match conv2d_int8_plan_code(d) {
        0 => {}
        -3 => {
            return Err(Error::coded(
                crate::Status::Shape,
                -3,
                "rocket_conv2d_int8/out-extent",
            ));
        }
        -4 => return Err(Error::tiling("rocket_conv2d_int8/cbuf")),
        _ => return Err(Error::shape("rocket_conv2d_int8")),
    }

    let icr = d.ic;
    let (ih, iw, oc, kh, kw) = (d.ih, d.iw, d.oc, d.kh, d.kw);
    let (oh, ow) = (conv2d_oh(d), conv2d_ow(d));
    let (sy, sx, pt, pl) = (d.stride_y, d.stride_x, d.pad_top, d.pad_left);
    let (dy, dx) = (d.dil_y, d.dil_x);

    // First-layer IC < 32 (RGB): zero-pad the input channels up to the ic group of 32.
    // The padded weight is built once; the input is padded lazily in the per-tile
    // materialisation (ic >= ICr stays zero).
    let ic = icr.div_ceil(32) * 32;
    let wpad = if ic != icr {
        let mut wp = vec![0i8; oc * ic * kh * kw];
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
    let wuse: &[i8] = wpad.as_deref().unwrap_or(w);

    // fast path: no channel pad AND the whole problem fits one CBUF pass
    {
        let ih_full = datain_h(oh, sy, kh, dy);
        let iw_full = datain_h(ow, sx, kw, dx);
        if ic == icr
            && oc * ic * kh * kw <= 4 * CBUF_BANK_SIZE
            && ic * ih_full * iw_full <= CONV_FEAT_BUDGET_I8
        {
            return conv2d_int8_one_job(
                &mut ctxs[0],
                ic,
                ih,
                iw,
                oc,
                oh,
                ow,
                kh,
                kw,
                sy,
                sx,
                pt,
                pl,
                dy,
                dx,
                input,
                w,
                out,
            );
        }
    }

    let plan = conv_int8_tile_plan(d)?;
    let tiles = plan.tiles(d);

    // `ROCKET_CONV_BATCH`: coalesce a run of the independent tiles into ONE multi-task
    // job instead of one job per tile. Read once per call; off keeps the per-tile path
    // byte-for-byte. The stock driver already runs a job's tasks back to back on one core
    // (`matmul::batch` relies on it), so this needs no extra capability — only the
    // optional chained layout does, and `batch_chain_plan` asks the kernel itself.
    let batch = conv_batch_enabled();

    // Largest tile's materialised sub-input / output, for the per-worker scratch.
    let rh_max = *plan.rband.iter().max().expect("bands");
    let cw_max = *plan.cband.iter().max().expect("bands");
    let ihs = datain_h(rh_max, sy, kh, dy);
    let iws = datain_h(cw_max, sx, kw, dx);
    let sub_in_sz = plan.ic * ihs * iws;
    let sub_out_elems = plan.oc_tile * rh_max * cw_max;

    // Multicore: hand each worker a CONTIGUOUS run of output-channel groups. That is
    // what makes the output split expressible as disjoint slices — the C fans the same
    // independent tiles over its worker fds, and tile assignment cannot change the
    // result (each tile writes a disjoint region), so this is bit-identical.
    let ngroups = d.oc.div_ceil(plan.oc_tile);
    let nw = if ctxs.len() > 1 && ngroups > 1 {
        ctxs.len().min(ngroups)
    } else {
        0
    };
    if nw > 1 {
        let base = ngroups / nw;
        let extra = ngroups % nw;
        // (first output channel, channel count, this worker's output slice)
        let mut bands: Vec<(usize, usize, &mut [i32])> = Vec::with_capacity(nw);
        let mut rest = &mut out[..];
        let mut cursor = 0;
        for w in 0..nw {
            let n = base + usize::from(w < extra);
            let oc_base = cursor * plan.oc_tile;
            let oc_n = (d.oc - oc_base).min(n * plan.oc_tile);
            let (band, tail) = rest.split_at_mut(oc_n * oh * ow);
            bands.push((oc_base, oc_n, band));
            rest = tail;
            cursor += n;
        }

        let base_aff = crate::device::affinity::base();
        let results: Vec<Result<()>> = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(nw);
            for (w, (ctx, (s, oc_n, band))) in ctxs.iter_mut().zip(bands).enumerate() {
                let tiles = &tiles;
                handles.push(scope.spawn(move || -> Result<()> {
                    crate::device::affinity::pin_current(w, base_aff);
                    // A worker's tiles are a CONTIGUOUS run of the plan (the output-channel
                    // group is its outer axis), which is what makes the band a disjoint
                    // slice of `out`.
                    let lo = tiles.partition_point(|t| t.oc0 < s);
                    let hi = lo + tiles[lo..].partition_point(|t| t.oc0 < s + oc_n);
                    let band_tiles = &tiles[lo..hi];
                    if batch && band_tiles.len() > 1 {
                        for g in batch_groups(band_tiles.len()) {
                            conv2d_int8_batch_tiles(
                                ctx,
                                plan.ic,
                                d,
                                input,
                                wuse,
                                band,
                                s,
                                &band_tiles[g],
                            )?;
                        }
                        return Ok(());
                    }
                    let mut sub_in = vec![0i8; sub_in_sz];
                    let mut sub_out = vec![0i32; sub_out_elems];
                    for tl in band_tiles.iter() {
                        conv2d_int8_one_tile(
                            ctx,
                            plan.ic,
                            icr,
                            ih,
                            iw,
                            oh,
                            ow,
                            kh,
                            kw,
                            sy,
                            sx,
                            pt,
                            pl,
                            dy,
                            dx,
                            input,
                            wuse,
                            band,
                            tl.oc0,
                            tl.oc0 - s,
                            tl.oc_n,
                            tl.r0,
                            tl.rh,
                            tl.c0,
                            tl.cw,
                            &mut sub_in,
                            &mut sub_out,
                        )?;
                    }
                    Ok(())
                }));
            }
            handles
                .into_iter()
                .map(|h| {
                    h.join().unwrap_or_else(|_| {
                        Err(Error::device("rocket_conv2d_int8/worker-panic", 0))
                    })
                })
                .collect()
        });
        return results.into_iter().find(|r| r.is_err()).unwrap_or(Ok(()));
    }

    // Serial: one scratch pair on the coordinator context, or — with
    // `ROCKET_CONV_BATCH` — each group of tiles coalesced into one multi-task job.
    if batch && tiles.len() > 1 {
        for g in batch_groups(tiles.len()) {
            conv2d_int8_batch_tiles(&mut ctxs[0], plan.ic, d, input, wuse, out, 0, &tiles[g])?;
        }
        return Ok(());
    }
    let mut sub_in = vec![0i8; sub_in_sz];
    let mut sub_out = vec![0i32; sub_out_elems];
    for tl in &tiles {
        conv2d_int8_one_tile(
            &mut ctxs[0],
            plan.ic,
            icr,
            ih,
            iw,
            oh,
            ow,
            kh,
            kw,
            sy,
            sx,
            pt,
            pl,
            dy,
            dx,
            input,
            wuse,
            out,
            tl.oc0,
            tl.oc0,
            tl.oc_n,
            tl.r0,
            tl.rh,
            tl.c0,
            tl.cw,
            &mut sub_in,
            &mut sub_out,
        )?;
    }
    Ok(())
}

/// `rocket_conv2d_int8`: the native int8 direct conv, int32-raw output.
///
/// `in [IC][IH][IW]` int8, `W [OC][IC][KH][KW]` int8, `out [OC][OH][OW]` int32.
/// `OC` is zero-padded to 32 and `IC` to 32 internally; `depthwise` must be `false`
/// (the `-2` of the C entry — int8 depthwise is [`conv2d_dw_int8`]).
pub fn conv2d_int8(
    dev: &Device,
    d: &Conv2dDesc,
    input: &[i8],
    w: &[i8],
    out: &mut [i32],
) -> Result<()> {
    let mut ctx = ConvContext::new(dev);
    conv2d_int8_run(std::slice::from_mut(&mut ctx), d, input, w, out)
}

/// `rocket_conv2d_int8_ctx`: the same conv on a resident-BO context.
pub fn conv2d_int8_ctx(
    ctx: &mut ConvContext,
    d: &Conv2dDesc,
    input: &[i8],
    w: &[i8],
    out: &mut [i32],
) -> Result<()> {
    conv2d_int8_run(std::slice::from_mut(ctx), d, input, w, out)
}

/// The native int8/uint8 direct conv's multicore pool (`rocket_conv_pool`).
///
/// The driver pins one fd to one NPU core while it has queued work, so the single-fd
/// entry serializes its independent OC/OH/OW tiles onto one core. A pool holds N
/// persistent devices, each with its own resident [`ConvContext`], so the tiles fan out
/// across the cores while keeping the resident-BO win. The pool **owns** its devices.
///
/// [`conv2d_int8_mt`] is bit-identical to [`conv2d_int8`] — same tiles, same single
/// jobs — and falls back to the serial path for a single-tile conv.
pub struct ConvInt8Pool {
    ctxs: Vec<ConvContext>,
}

impl ConvInt8Pool {
    /// `rocket_conv_pool_create`: open `nthreads` devices (clamped to `[1, 8]`).
    ///
    /// Degrades to however many devices opened, as C does; an `Err` only when none did.
    pub fn new(nthreads: usize) -> Result<Self> {
        let n = nthreads.clamp(1, CONV_POOL_MAX);
        let mut ctxs = Vec::with_capacity(n);
        for _ in 0..n {
            match Device::open() {
                Ok(dev) => ctxs.push(ConvContext::new(&dev)),
                Err(e) => {
                    log_debug!(
                        "rocket_conv_pool_create: opened {} of {n} devices",
                        ctxs.len()
                    );
                    if ctxs.is_empty() {
                        return Err(e);
                    }
                    break;
                }
            }
        }
        Ok(Self { ctxs })
    }

    /// Worker contexts actually opened.
    pub fn len(&self) -> usize {
        self.ctxs.len()
    }

    /// True when no worker context exists (a pool is never constructed empty).
    pub fn is_empty(&self) -> bool {
        self.ctxs.is_empty()
    }

    /// The first worker's device.
    pub fn device(&self) -> &Device {
        self.ctxs[0].device()
    }

    /// `rocket_conv2d_int8_mt`: run a direct int8 conv across the pool's workers.
    pub fn conv2d(
        &mut self,
        d: &Conv2dDesc,
        input: &[i8],
        w: &[i8],
        out: &mut [i32],
    ) -> Result<()> {
        conv2d_int8_mt(self, d, input, w, out)
    }
}

/// `rocket_conv2d_int8_mt`: the multicore direct int8 conv, bit-identical to
/// [`conv2d_int8`].
pub fn conv2d_int8_mt(
    pool: &mut ConvInt8Pool,
    d: &Conv2dDesc,
    input: &[i8],
    w: &[i8],
    out: &mut [i32],
) -> Result<()> {
    if pool.ctxs.is_empty() {
        return Err(Error::shape("rocket_conv2d_int8_mt/pool"));
    }
    conv2d_int8_run(&mut pool.ctxs, d, input, w, out)
}

// ############################################################################
// PART 6 — native int8 DEPTHWISE conv (int8-out, on-chip requant)
// ############################################################################

/// The DPU output convertor's scale → register pair (`requant_params` in the C test's
/// `requant_model.h`, the QNNPACK derivation the generator programs).
///
/// Returns `(mul, shift)` with `shift` the **register** value (already pre-decremented),
/// so the hardware computes `sat8(round_half_even(acc * mul >> shift) + out_zp)`.
pub fn requant_params(conv_scale: f32) -> (u32, u32) {
    let bits = conv_scale.to_bits();
    let shift = (127u32 + 31 - 32).wrapping_sub(bits >> 23) + 16 - 1;
    let mut mul = ((bits >> 9) & 0x7FFF) + 1;
    if mul < (1 << 14) {
        mul |= 1 << 14;
    }
    (mul, shift)
}

/// The host model of the DPU's rounding shift: nearest, ties to **even** (measured on
/// hardware — not the arithmetic-shift `(x + half) >> shift`, which takes ties toward
/// +inf, and not round-half-away-from-zero).
pub fn requant_round_shift(p: i64, shift: u32) -> i64 {
    if shift == 0 {
        return p;
    }
    let half = 1i64 << (shift - 1);
    let mut v = (p + half) >> shift;
    let rem = p & ((1i64 << shift) - 1);
    if rem == half && (v & 1) != 0 {
        v -= 1;
    }
    v
}

/// `acc * mul >> shift`, saturated to int8 (`requant_apply`).
pub fn requant_apply(acc: i64, mul: u32, shift: u32) -> i8 {
    let v = requant_round_shift(acc * mul as i64, shift);
    v.clamp(-128, 127) as i8
}

/// The same, then the output zero point the OUT_CVT offset carries (`requant_apply_zp`).
pub fn requant_apply_zp(acc: i64, mul: u32, shift: u32, out_zp: i32) -> i8 {
    let v = requant_round_shift(acc * mul as i64, shift) + out_zp as i64;
    v.clamp(-128, 127) as i8
}

/// The int8 depthwise int8-out host model: TFLite's accumulator, then the DPU requant.
///
/// This is the exact model of the program [`conv2d_dw_int8`] emits, so a device result
/// must match it **bit for bit**. (The C library's `dw_int8_ref` is the same
/// accumulator with TFLite's `lrintf` + round-to-nearest requant instead, which can
/// differ by one where a value sits on a rounding boundary; the C self-test gates the
/// device against this model and the library oracle within one, and this port does the
/// same.)
#[allow(clippy::too_many_arguments)]
pub fn conv2d_dw_int8_model(
    d: &Conv2dDesc,
    input: &[i8],
    w: &[i8],
    bias: Option<&[i32]>,
    in_scale: f32,
    w_scale: f32,
    out_scale: f32,
    in_zp: i32,
    out_zp: i32,
    out: &mut [i8],
) {
    let (mul, shift) = requant_params((in_scale * w_scale) / out_scale);
    let (ih, iw, kh, kw) = (d.ih, d.iw, d.kh, d.kw);
    let (oh, ow) = (conv2d_oh(d), conv2d_ow(d));
    for c in 0..d.ic {
        for y in 0..oh {
            for x in 0..ow {
                let mut acc = bias.map(|b| b[c] as i64).unwrap_or(0);
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
                        acc += (input[(c * ih + yy) * iw + xx] as i64 - in_zp as i64)
                            * w[(c * kh + k) * kw + j] as i64;
                    }
                }
                out[(c * oh + y) * ow + x] = requant_apply_zp(acc, mul, shift, out_zp);
            }
        }
    }
}

/// Run ONE int8 depthwise int8-out job (single CBUF pass).
///
/// `in [C][IH][IW]` int8 (model domain), `w [C][KH][KW]` int8, `bias_q [C]` int32
/// (optional), `out [C][OH][OW]` int8 (model domain).
#[allow(clippy::too_many_arguments)]
fn conv2d_dw_int8_one_job(
    ctx: &mut ConvContext,
    c: usize,
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
    input: &[i8],
    w: &[i8],
    bias: Option<&[i32]>,
    in_scale: f32,
    w_scale: f32,
    out_scale: f32,
    in_zp: i32,
    w_zp: i32,
    out_zp: i32,
    out: &mut [i8],
) -> Result<()> {
    let dev = ctx.dev.clone();
    let g = DW_INT8_G;
    let cpad = c.div_ceil(g) * g;
    let ihj = ih.max(CONV_MIN_DATAIN_H);
    let cube_in = c.div_ceil(16) * 16 * ihj * iw;
    let wt_elems = cpad * kh * kw;
    let cube_out = c * oh * ow;

    ensure_bo(&dev, &mut ctx.guard, 4096)?;
    ensure_bo(&dev, &mut ctx.input, cube_in + CBUF_BANK_SIZE)?;
    ensure_bo(&dev, &mut ctx.weight, wt_elems + CBUF_BANK_SIZE)?;
    ensure_bo(&dev, &mut ctx.rc, crate::regcmd::conv::MAX_OPS * 8)?;
    ensure_bo(&dev, &mut ctx.bias, cpad * 4 + CBUF_BANK_SIZE)?;
    ensure_bo(&dev, &mut ctx.output, cube_out + CBUF_BANK_SIZE)?;

    // Feature cube (C2 = 16): the raw int8 values, which the CNA reads signed.
    {
        let bo = ctx.input.as_mut().expect("input BO");
        bo.prep(0)?;
        let words = bo_i8_mut(bo);
        words.fill(0);
        scatter_feature_i8(words, c, ihj, iw, input);
        bo.fini()?;
    }
    // DW weight cube (group G): the raw int8 values.
    {
        let bo = ctx.weight.as_mut().expect("weight BO");
        bo.prep(0)?;
        let words = bo_i8_mut(bo);
        words.fill(0);
        scatter_weights_dw_i8(words, c, kh, kw, g, w);
        bo.fini()?;
    }
    // The per-OC int32 bias cube: bias_q - in_zp * sum_kernel(w - w_zp). The CNA pads
    // with in_zp, so a padded tap adds in_zp * w and the fold takes it back out.
    {
        let bo = ctx.bias.as_mut().expect("bias BO");
        bo.prep(0)?;
        let bytes = bo.as_mut_slice();
        let dst = &mut bytes[..cpad * 4];
        dst.fill(0);
        for ch in 0..c {
            let mut sw = 0i32;
            for k in 0..kh {
                for j in 0..kw {
                    sw += w[(ch * kh + k) * kw + j] as i32 - w_zp;
                }
            }
            let b = bias.map(|b| b[ch]).unwrap_or(0) - in_zp * sw;
            dst[ch * 4..ch * 4 + 4].copy_from_slice(&b.to_ne_bytes());
        }
        bo.fini()?;
    }

    let mut p = crate::regcmd::conv::ConvInt8Params::new(
        c as u16,
        ihj as u16,
        iw as u16,
        c as u16,
        oh as u16,
        ow as u16,
        kh as u16,
        kw as u16,
        ctx.input.as_ref().expect("input BO").dma_address() as u32,
        ctx.weight.as_ref().expect("weight BO").dma_address() as u32,
        ctx.output.as_ref().expect("output BO").dma_address() as u32,
    );
    p.stride_y = sy as u8;
    p.stride_x = sx as u8;
    p.pad_top = pt as u8;
    p.pad_left = pl as u8;
    p.dil_y = dy as u8;
    p.dil_x = dx as u8;
    p.dw_group = g as u8;
    p.int8_out = true;
    p.in_scale = in_scale;
    p.w_scale = w_scale;
    p.out_scale = out_scale;
    // The generator carries Mesa's uint8 formulas, so it takes the uint8-equivalent
    // zero points: the pad lands on in_zp, the offset on out_zp, and the CPEND operand
    // on 0 (a raw w_zp of 0 would give 0x80 and nearly every output would be wrong).
    p.input_zero_point = in_zp + 0x80;
    p.output_zero_point = out_zp + 0x80;
    p.weight_zero_point = w_zp + 0x80;
    p.bias_dma = ctx.bias.as_ref().expect("bias BO").dma_address() as u32;

    let mut ops = Vec::with_capacity(crate::regcmd::conv::MAX_OPS);
    let count = crate::regcmd::conv::gen_conv2d_dw_int8(&mut p, &mut ops)?;
    if count > crate::regcmd::conv::MAX_OPS {
        log_error!("rocket_conv2d_dw_int8: regcmd overflow (task_count {count})");
        return Err(Error::tiling("rocket_conv2d_dw_int8/regcmd"));
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
        bo_i8_mut(bo).fill(0);
        bo.fini()?;
    }

    let task = DrmRocketTask {
        regcmd: ctx.rc.as_ref().expect("rc BO").dma_address() as u32,
        regcmd_count: count as u32,
    };
    let in_handles = [
        ctx.input.as_ref().expect("input BO").handle(),
        ctx.weight.as_ref().expect("weight BO").handle(),
        ctx.bias.as_ref().expect("bias BO").handle(),
        ctx.rc.as_ref().expect("rc BO").handle(),
    ];
    let out_handles = [ctx.output.as_ref().expect("output BO").handle()];
    dev.submit(&[task], &in_handles, &out_handles, 0)?;

    {
        let bo = ctx.output.as_mut().expect("output BO");
        bo.prep(CONV_WAIT_NS)?;
        gather_output_i8(&bo_i8(bo)[..cube_out], c, oh, ow, out);
        bo.fini()?;
    }
    Ok(())
}

/// The int8 depthwise driver (`conv2d_dw_int8_run`): validate, then channel-tile in
/// `G = 64` chunks (channels are independent — each chunk is one job). There is no
/// spatial tiling on this path (a single channel whose feature overflows one CBUF pass
/// is refused, as in C).
#[allow(clippy::too_many_arguments)]
fn conv2d_dw_int8_run(
    ctx: &mut ConvContext,
    d: &Conv2dDesc,
    input: &[i8],
    w: &[i8],
    bias: Option<&[i32]>,
    in_scale: f32,
    w_scale: f32,
    out_scale: f32,
    in_zp: i32,
    w_zp: i32,
    out_zp: i32,
    out: &mut [i8],
) -> Result<()> {
    match conv2d_dw_int8_plan_code(d) {
        0 => {}
        -2 => {
            return Err(Error::coded(
                crate::Status::Shape,
                -2,
                "rocket_conv2d_dw_int8/depthwise",
            ));
        }
        -3 => {
            return Err(Error::coded(
                crate::Status::Shape,
                -3,
                "rocket_conv2d_dw_int8/out-extent",
            ));
        }
        -5 => {
            log_error!(
                "rocket_conv2d_dw_int8: channel count C={} must be a multiple of 16 \
                 (C2 feature-cube grain)",
                d.ic
            );
            return Err(Error::coded(
                crate::Status::Tiling,
                -5,
                "rocket_conv2d_dw_int8/channels",
            ));
        }
        -4 => return Err(Error::tiling("rocket_conv2d_dw_int8/cbuf")),
        _ => return Err(Error::shape("rocket_conv2d_dw_int8")),
    }
    // The fold carries in_zp * sum(w); a weight zero point would add w_zp * sum(x),
    // which depends on the input and cannot be folded. TFLite's int8 weights are
    // symmetric (the C refuses a non-zero w_zp on the device path only).
    if w_zp != 0 {
        log_error!(
            "rocket_conv2d_dw_int8: weight zero point {w_zp}; the NPU path takes \
             symmetric int8 weights only"
        );
        return Err(Error::unsupported("rocket_conv2d_dw_int8/weight-zp"));
    }
    let (ih, iw, kh, kw) = (d.ih, d.iw, d.kh, d.kw);
    let (oh, ow) = (conv2d_oh(d), conv2d_ow(d));
    let (sy, sx, pt, pl) = (d.stride_y, d.stride_x, d.pad_top, d.pad_left);
    let (dy, dx) = (d.dil_y, d.dil_x);
    let cc = dw_int8_chunk(d.ic, ih, iw);
    let mut c0 = 0;
    while c0 < d.ic {
        let cn = (d.ic - c0).min(cc);
        conv2d_dw_int8_one_job(
            ctx,
            cn,
            ih,
            iw,
            oh,
            ow,
            kh,
            kw,
            sy,
            sx,
            pt,
            pl,
            dy,
            dx,
            &input[c0 * ih * iw..(c0 + cn) * ih * iw],
            &w[c0 * kh * kw..(c0 + cn) * kh * kw],
            bias.map(|b| &b[c0..c0 + cn]),
            in_scale,
            w_scale,
            out_scale,
            in_zp,
            w_zp,
            out_zp,
            &mut out[c0 * oh * ow..(c0 + cn) * oh * ow],
        )?;
        c0 += cn;
    }
    Ok(())
}

/// `rocket_conv2d_dw_int8`: int8-in / int8-out depthwise conv with the on-chip requant.
///
/// The caller passes the **raw model tensors**: `in [C][IH][IW]`, `w [C][KH][KW]`,
/// `bias_q [C]` int32 (optional), `out [C][OH][OW]`, all in the model's int8 domain,
/// plus the per-tensor scales and the model-domain zero points. `w_zp != 0` is
/// [`crate::Status::Unsupported`] (the fold cannot carry a weight zero point), and `C`
/// must be a multiple of 16 (the `C2` feature-cube grain).
#[allow(clippy::too_many_arguments)]
pub fn conv2d_dw_int8(
    dev: &Device,
    d: &Conv2dDesc,
    input: &[i8],
    w: &[i8],
    bias: Option<&[i32]>,
    in_scale: f32,
    w_scale: f32,
    out_scale: f32,
    in_zp: i32,
    w_zp: i32,
    out_zp: i32,
    out: &mut [i8],
) -> Result<()> {
    let mut ctx = ConvContext::new(dev);
    conv2d_dw_int8_run(
        &mut ctx, d, input, w, bias, in_scale, w_scale, out_scale, in_zp, w_zp, out_zp, out,
    )
}

/// `rocket_conv2d_dw_int8_ctx`: the same depthwise conv on a resident-BO context.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_dw_int8_ctx(
    ctx: &mut ConvContext,
    d: &Conv2dDesc,
    input: &[i8],
    w: &[i8],
    bias: Option<&[i32]>,
    in_scale: f32,
    w_scale: f32,
    out_scale: f32,
    in_zp: i32,
    w_zp: i32,
    out_zp: i32,
    out: &mut [i8],
) -> Result<()> {
    conv2d_dw_int8_run(
        ctx, d, input, w, bias, in_scale, w_scale, out_scale, in_zp, w_zp, out_zp, out,
    )
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

    // ── int8 conv ─────────────────────────────────────────────────────────────

    #[test]
    fn int8_reference_hand_cases() {
        // 1x1: one tap per output, exact integer product.
        let mut d = desc(2, 1, 2, 2, 1, 1);
        d.ic = 2;
        d.oc = 2;
        let input = [1i8, -2, 3, 4]; // [ic][1][2]
        let w = [2i8, 1, -1, 3]; // [oc][ic][1][1]
        let mut out = [0i32; 4];
        conv2d_ref_int8(&d, &input, &w, &mut out);
        assert_eq!(out, [5, 0, 8, 14]);

        // 3x3 pad 1 on a 2x2 input: out[0][0] only sees in[0][0] at tap (1,1).
        let mut d = desc(1, 2, 2, 1, 3, 3);
        d.pad_top = 1;
        d.pad_left = 1;
        let input = [5i8, -7, 11, 13];
        let w: Vec<i8> = (1..=9).collect();
        let mut out = [0i32; 4];
        conv2d_ref_int8(&d, &input, &w, &mut out);
        // out[0][0] = 5*5 + (-7)*6 + 11*8 + 13*9;
        // out[0][1] = 5*4 + (-7)*5 + 11*7 + 13*8;
        // out[1][0] = 5*2 + (-7)*3 + 11*5 + 13*6; out[1][1] = 5*1 + (-7)*2 + 11*4 + 13*5.
        assert_eq!(out, [188, 166, 122, 100]);

        // Depthwise (int32-raw): each channel reduces only itself.
        let mut d = desc(2, 2, 1, 2, 2, 1);
        d.depthwise = true;
        let input = [1i8, 2, 3, 4];
        let w = [1i8, 10, 2, 20]; // [c][kh][kw]
        let mut out = [0i32; 2];
        conv2d_ref_int8(&d, &input, &w, &mut out);
        // oh = 2 - 2 + 1 = 1, ow = 1: c0 = 1*1 + 2*10 = 21; c1 = 3*2 + 4*20 = 86.
        assert_eq!(out, [21, 86]);

        // A stride-2 3x3 reads only the taps that land on the plane.
        let mut d = desc(1, 4, 4, 1, 3, 3);
        d.ic = 1;
        d.oc = 1;
        d.stride_y = 2;
        d.stride_x = 2;
        d.pad_top = 1;
        d.pad_left = 1;
        let input: Vec<i8> = (0..16).collect();
        let w = [1i8; 9];
        let mut out = [0i32; 4];
        conv2d_ref_int8(&d, &input, &w, &mut out);
        assert_eq!(
            out,
            [
                1 + 4 + 5,
                1 + 2 + 3 + 5 + 6 + 7,
                4 + 5 + 8 + 9 + 12 + 13,
                90
            ]
        );
    }

    #[test]
    fn int8_plan_codes_match_the_run_classes() {
        // Direct 3x3 at the RGB stem IC=3 (padded to 32) is accepted.
        let mut d = desc(3, 56, 56, 64, 3, 3);
        d.pad_top = 1;
        d.pad_left = 1;
        assert_eq!(conv2d_int8_plan_code(&d), 0);

        // A depthwise descriptor belongs to the other entry.
        let mut dw = desc(64, 56, 56, 64, 3, 3);
        dw.pad_top = 1;
        dw.pad_left = 1;
        dw.depthwise = true;
        assert_eq!(conv2d_int8_plan_code(&dw), -2);
        assert_eq!(conv2d_dw_int8_plan_code(&dw), 0);
        assert_eq!(conv2d_dw_int8_plan_code(&d), -2);
        // An RGB-stem depthwise form hits the C2 = 16 channel grain (the C's -5, which
        // it checks after the depthwise/OC match).
        let mut dw3 = dw;
        dw3.ic = 3;
        dw3.oc = 3;
        assert_eq!(conv2d_dw_int8_plan_code(&dw3), -5);

        // A 3x3 kernel at IC 12000 cannot fit one bank even at 1 B/elem.
        let d = desc(12000, 4, 4, 32, 3, 3);
        assert_eq!(conv2d_int8_plan_code(&d), -4);
        // The int8 depthwise group is 64: 64*25*25 > 32768 at k25.
        let mut d = desc(64, 64, 64, 64, 25, 25);
        d.depthwise = true;
        assert_eq!(conv2d_dw_int8_plan_code(&d), -4);
        // The C2 = 16 feature-cube grain refuses a C that is not a multiple of it.
        let mut d = desc(24, 8, 8, 24, 3, 3);
        d.depthwise = true;
        assert_eq!(conv2d_dw_int8_plan_code(&d), -5);
        // A collapsed extent.
        let d = desc(32, 4, 4, 32, 5, 5);
        assert_eq!(conv2d_int8_plan_code(&d), -3);
        // Malformed.
        let d = desc(0, 4, 4, 32, 1, 1);
        assert_eq!(conv2d_int8_plan_code(&d), -1);
    }

    #[test]
    fn int8_tile_plan_covers_the_output_once() {
        // 128x112x112 at k3 p1: the weight tile shrinks to 96 (4 banks) and the rows
        // band because 128 * 114 * 114 overflows the 7-bank int8 feature budget.
        let mut d = desc(128, 112, 112, 128, 3, 3);
        d.pad_top = 1;
        d.pad_left = 1;
        let plan = conv_int8_tile_plan(&d).expect("tileable");
        assert_eq!(plan.ic, 128);
        assert_eq!(plan.oc_tile, 96);
        assert!(plan.rband.len() > 1, "row bands: {:?}", plan.rband);
        assert_eq!(plan.rband.iter().sum::<usize>(), 112);
        assert_eq!(plan.cband.iter().sum::<usize>(), 112);
        let tiles = plan.tiles(&d);
        assert_eq!(plan.count(&d), tiles.len());
        let (oh, ow) = (conv2d_oh(&d), conv2d_ow(&d));
        let mut covered = 0usize;
        let mut chan = vec![0usize; d.oc * oh * ow];
        for t in &tiles {
            covered += t.oc_n * t.rh * t.cw;
            assert!(t.oc0 + t.oc_n <= d.oc && t.r0 + t.rh <= oh && t.c0 + t.cw <= ow);
            for o in t.oc0..t.oc0 + t.oc_n {
                for r in t.r0..t.r0 + t.rh {
                    for c in t.c0..t.c0 + t.cw {
                        chan[(o * oh + r) * ow + c] += 1;
                    }
                }
            }
        }
        assert_eq!(covered, d.oc * oh * ow);
        assert!(chan.iter().all(|&n| n == 1), "every element covered once");

        // A small problem is a single tile (the fast path would serve it anyway).
        let mut d = desc(32, 8, 8, 32, 3, 3);
        d.pad_top = 1;
        d.pad_left = 1;
        let plan = conv_int8_tile_plan(&d).expect("tileable");
        assert_eq!(plan.oc_tile, 32);
        assert_eq!(plan.tiles(&d).len(), 1);
    }

    #[test]
    fn int8_requant_model_and_registers_agree() {
        // The register pair the emitter programs and the host model share one
        // derivation; pin both against the C test's `requant_model.h` numbers.
        let (mul, shift) = requant_params(0.0213 * 0.00347 / 0.0925);
        assert_eq!((mul, shift), (0x68bc, 25));
        // ... and the emitter writes exactly that (scale, shift-1 register).
        let mut p = crate::regcmd::conv::ConvInt8Params::new(64, 8, 8, 64, 8, 8, 3, 3, 0, 0, 0);
        p.pad_top = 1;
        p.pad_left = 1;
        p.dw_group = 64;
        p.int8_out = true;
        p.in_scale = 0.0213;
        p.w_scale = 0.00347;
        p.out_scale = 0.0925;
        p.input_zero_point = 0x80;
        p.output_zero_point = 0x80;
        p.weight_zero_point = 0x80;
        let mut ops = Vec::new();
        crate::regcmd::conv::gen_conv2d_dw_int8(&mut p, &mut ops).unwrap();
        let scale = ops
            .iter()
            .find(|o| {
                o.target() == crate::regcmd::desc::DPU
                    && o.reg() == crate::regcmd::desc::DPU_OUT_CVT_SCALE
            })
            .unwrap()
            .value()
            & 0xFFFF;
        let reg_shift = ops
            .iter()
            .find(|o| {
                o.target() == crate::regcmd::desc::DPU
                    && o.reg() == crate::regcmd::desc::DPU_OUT_CVT_SHIFT
            })
            .unwrap()
            .value()
            & 0x3F;
        assert_eq!(scale, mul);
        assert_eq!(reg_shift, shift);

        // Round half to even, both signs, and saturation after the offset.
        assert_eq!(requant_round_shift(2, 1), 1); // 1.0
        assert_eq!(requant_round_shift(1, 1), 0); // 0.5 -> 0
        assert_eq!(requant_round_shift(3, 1), 2); // 1.5 -> 2
        assert_eq!(requant_round_shift(-1, 1), 0); // -0.5 -> 0
        assert_eq!(requant_round_shift(-3, 1), -2); // -1.5 -> -2
        assert_eq!(requant_round_shift(7, 0), 7); // no shift
        assert_eq!(requant_apply(1, 0x4000, 14), 1);
        assert_eq!(requant_apply(1 << 40, 0x4000, 14), 127);
        assert_eq!(requant_apply(-(1 << 40), 0x4000, 14), -128);
        assert_eq!(requant_apply_zp(0, 0x4000, 14, 5), 5);
    }

    #[test]
    fn int8_cube_round_trips() {
        // int8 feature cube (C2 = 16) with a partial channel group.
        let (c, h, w) = (3usize, 5usize, 7usize);
        let src: Vec<i8> = (0..c * h * w).map(|i| (i as i8).wrapping_mul(3)).collect();
        let mut cube = vec![0i8; c.div_ceil(16) * 16 * h * w];
        scatter_feature_i8(&mut cube, c, h, w, &src);
        let mut back = vec![0i8; c * h * w];
        gather_output_i8(&cube, c, h, w, &mut back);
        assert_eq!(src, back);
        assert_eq!(feature_cube_index_i8(c, h, w, 1, 1, 1), 0);
        assert_eq!(feature_cube_index_i8(c, h, w, 2, 1, 1), 1);
        assert_eq!(feature_cube_index_i8(c, h, w, 1, 1, 2), 16);
        assert_eq!(feature_cube_index_i8(c, h, w, 17, 1, 1), h * w * 16);

        // int32 output cube (C2 = 4).
        let (oc, oh, ow) = (6usize, 3usize, 4usize);
        let src: Vec<i32> = (0..oc * oh * ow).map(|i| i as i32 - 7).collect();
        let mut cube = vec![0i32; oc.div_ceil(4) * 4 * oh * ow];
        for ch in 0..oc {
            for r in 0..oh {
                for col in 0..ow {
                    cube[feature_data(oc, oh, ow, 4, ch + 1, r + 1, col + 1)] =
                        src[(ch * oh + r) * ow + col];
                }
            }
        }
        let mut back = vec![0i32; oc * oh * ow];
        gather_output_i32(&cube, oc, oh, ow, &mut back);
        assert_eq!(src, back);

        // The int8 direct weight cube at 1x1 collapses to the (N/32, K/32, 32, 32)
        // matmul tile; the kernel spatial sits between the ic1 and oc2 groups. (OC and
        // IC must be the group-padded extents the one-job path passes: 32/32 here.)
        assert_eq!(weight_cube_index_i8(32, 1, 1, 1, 1, 1, 1), 0);
        assert_eq!(weight_cube_index_i8(32, 1, 1, 32, 32, 1, 1), 31 * 32 + 31);
        assert_eq!(weight_cube_index_i8(32, 1, 1, 33, 1, 1, 1), 32 * 32);
        assert_eq!(weight_cube_index_i8(64, 3, 3, 1, 1, 1, 2), 32 * 32);
        assert_eq!(weight_cube_index_i8(64, 3, 3, 2, 1, 1, 1), 32);

        // Round trip the direct int8 weight scatter at the padded groups.
        let wsrc: Vec<i8> = (0..32 * 32 * 3 * 3).map(|i| (i % 251) as i8).collect();
        let mut wcube = vec![0i8; 32 * 32 * 3 * 3];
        scatter_weights_i8(&mut wcube, 32, 32, 3, 3, &wsrc);
        for o in 0..32 {
            for k in 0..3 {
                for j in 0..3 {
                    for i in 0..32 {
                        assert_eq!(
                            wcube[weight_cube_index_i8(32, 3, 3, o + 1, i + 1, k + 1, j + 1)],
                            wsrc[((o * 32 + i) * 3 + k) * 3 + j]
                        );
                    }
                }
            }
        }
    }

    // ── int8 batched submit (ROCKET_CONV_BATCH) ───────────────────────────────

    #[test]
    fn int8_batch_slots_are_bank_aligned_and_disjoint() {
        // 128x112x112 k3 p1 is the tiler's own multi-tile shape: OC shrinks to 96 and the
        // rows band, so the slots get several geometries.
        let mut d = desc(128, 112, 112, 128, 3, 3);
        d.pad_top = 1;
        d.pad_left = 1;
        let plan = conv_int8_tile_plan(&d).expect("tileable");
        let tiles = plan.tiles(&d);
        assert!(tiles.len() > 1, "tiles: {}", tiles.len());

        let layout = conv_batch_layout(&d, plan.ic, &tiles);
        assert_eq!(layout.slots.len(), tiles.len());

        let (mut in_end, mut wt_end, mut out_end, mut max_sub) = (0, 0, 0, 0);
        for (slot, &tile) in layout.slots.iter().zip(&tiles) {
            assert_eq!(slot.tile, tile);
            // Every slot starts on a whole CBUF bank, so its feature DMA base has the
            // alignment a fresh single-job BO has.
            for off in [slot.in_off, slot.wt_off, slot.out_off] {
                assert!(
                    off.is_multiple_of(CBUF_BANK_SIZE),
                    "slot offset {off} is not bank-aligned"
                );
            }
            // ... and the slots are laid end to end, in tile order.
            assert_eq!(
                (slot.in_off, slot.wt_off, slot.out_off),
                (in_end, wt_end, out_end)
            );

            // The tile's real (halo-grown) sub-input, raised to the 4-row DMA floor.
            let ih_sub = (tile.rh - 1) * d.stride_y + (d.kh - 1) * d.dil_y + 1;
            let iw_sub = (tile.cw - 1) * d.stride_x + (d.kw - 1) * d.dil_x + 1;
            assert_eq!((slot.ih_sub, slot.iw_sub), (ih_sub, iw_sub));
            assert_eq!(slot.ihj, ih_sub.max(CONV_MIN_DATAIN_H));
            assert_eq!(slot.sub_in, plan.ic * ih_sub * iw_sub);
            max_sub = max_sub.max(slot.sub_in);

            // Each region holds its cube plus one spare bank (the int8 over-read's zeros).
            let in_slot = bank_round(slot.in_bytes(plan.ic)) + CBUF_BANK_SIZE;
            let wt_slot = bank_round(slot.wt_bytes(plan.ic, d.kh, d.kw)) + CBUF_BANK_SIZE;
            let out_slot = bank_round(slot.out_bytes()) + CBUF_BANK_SIZE;
            assert!(slot.in_bytes(plan.ic) <= in_slot - CBUF_BANK_SIZE);
            assert!(slot.wt_bytes(plan.ic, d.kh, d.kw) <= wt_slot - CBUF_BANK_SIZE);
            assert!(slot.out_bytes() <= out_slot - CBUF_BANK_SIZE);
            in_end += in_slot;
            wt_end += wt_slot;
            out_end += out_slot;
        }
        assert_eq!(
            (layout.in_bytes, layout.wt_bytes, layout.out_bytes),
            (in_end, wt_end, out_end)
        );
        assert_eq!(layout.max_sub_in, max_sub);

        // Batching changes only the submit: the tiles still cover the output exactly once.
        let (oh, ow) = (conv2d_oh(&d), conv2d_ow(&d));
        let mut covered = 0usize;
        let mut chan = vec![0usize; d.oc * oh * ow];
        for t in &tiles {
            covered += t.oc_n * t.rh * t.cw;
            for o in t.oc0..t.oc0 + t.oc_n {
                for r in t.r0..t.r0 + t.rh {
                    for c in t.c0..t.c0 + t.cw {
                        chan[(o * oh + r) * ow + c] += 1;
                    }
                }
            }
        }
        assert_eq!(covered, d.oc * oh * ow);
        assert!(chan.iter().all(|&n| n == 1), "every element covered once");
    }

    #[test]
    fn an_int8_batch_slot_runs_the_per_tile_program() {
        let mut d = desc(128, 112, 112, 128, 3, 3);
        d.pad_top = 1;
        d.pad_left = 1;
        let plan = conv_int8_tile_plan(&d).expect("tileable");
        let tiles = plan.tiles(&d);
        let batch = &tiles[..3];

        // BO base 0, so a slot's program is addressed at its own slot offsets and the
        // per-tile job below can be handed the very same addresses.
        let layout = conv_batch_layout(&d, plan.ic, batch);
        let progs = batch_programs(&layout, plan.ic, &d, 0, 0, 0).expect("emitters");

        for (i, slot) in layout.slots.iter().enumerate() {
            let t = slot.tile;
            // The per-tile path's program for the same tile (`conv2d_int8_one_job`'s call:
            // the halo-grown sub-input, the real output extent, materialised pad).
            let mut p = crate::regcmd::conv::ConvInt8Params::new(
                plan.ic as u16,
                slot.ihj as u16,
                slot.iw_sub as u16,
                t.oc_n as u16,
                t.rh as u16,
                t.cw as u16,
                d.kh as u16,
                d.kw as u16,
                slot.in_off as u32,
                slot.wt_off as u32,
                slot.out_off as u32,
            );
            p.stride_y = d.stride_y as u8;
            p.stride_x = d.stride_x as u8;
            p.dil_y = d.dil_y as u8;
            p.dil_x = d.dil_x as u8;
            let mut ops = Vec::new();
            let count = crate::regcmd::conv::gen_conv2d_int8(&mut p, &mut ops).unwrap();

            let (off, span) = progs.spans[i];
            assert_eq!(span as usize, count, "tile {i} op count");
            let want: Vec<u64> = ops.iter().map(|op| op.0).collect();
            assert_eq!(
                &progs.group[off..off + count],
                &want[..],
                "tile {i} program"
            );
        }

        // The gapped slots put each program at its own descriptor, one CONV_RC_STRIDE
        // apart — the layout the per-tile path's jobs would each get on their own.
        let base = 0x1000_0000u32;
        let g = layout.slots.len();
        let mut dst = vec![0u64; g * CONV_RC_STRIDE];
        let mut tasks = vec![DrmRocketTask::default(); g];
        for (i, &(off, count)) in progs.spans.iter().enumerate() {
            crate::matmul::chain::pack(
                false,
                &mut dst,
                base,
                &mut tasks,
                i,
                &progs.group[off..off + count as usize],
                count,
                CONV_RC_STRIDE,
            )
            .unwrap();
            assert_eq!(tasks[i].regcmd, base + (i * CONV_RC_STRIDE * 8) as u32);
            assert_eq!(tasks[i].regcmd_count, count);
            let slot = &dst[i * CONV_RC_STRIDE..i * CONV_RC_STRIDE + count as usize];
            assert_eq!(slot, &progs.group[off..off + count as usize]);
        }
    }

    #[test]
    fn int8_batch_groups_cover_the_tiles_once() {
        assert!(batch_groups(0).is_empty());
        assert_eq!(batch_groups(1), vec![0..1]);
        for tn in [2usize, 3, 7, 63, 64, 65, 66, 127, 128, 129, 130, 1000] {
            let groups = batch_groups(tn);
            assert_eq!(groups.first().unwrap().start, 0);
            assert_eq!(groups.last().unwrap().end, tn);
            let mut next = 0;
            let mut total = 0;
            for g in &groups {
                assert_eq!(g.start, next, "groups are contiguous at tn={tn}");
                assert!(g.len() <= crate::regcmd::desc::BATCH + 1, "tn={tn}");
                assert!(
                    g.len() > 1,
                    "a lone tile belongs on the per-tile path (tn={tn})"
                );
                next = g.end;
                total += g.len();
            }
            assert_eq!((next, total), (tn, tn));
        }
    }

    #[test]
    fn int8_batch_chain_layout_needs_uniform_programs() {
        let uniform = [(0usize, 10u32), (10, 10), (20, 10)];
        // Chaining is opt-in (`ROCKET_BATCH_SUBMIT`), and the gapped slots are the C's.
        assert_eq!(batch_chain_plan(false, &uniform, 300), None);
        assert_eq!(batch_chain_plan(true, &uniform, 300), Some(30));
        // A chain may not need more regcmd BO than the fixed slots it replaces.
        assert_eq!(batch_chain_plan(true, &uniform, 29), None);
        // One PC stride cannot describe mixed program lengths.
        assert_eq!(batch_chain_plan(true, &[(0, 10), (10, 12)], 512), None);
        // A program longer than a slot is not a state this tiler produces.
        assert_eq!(batch_chain_plan(true, &[(0, 300), (300, 300)], 4096), None);
        assert_eq!(batch_chain_plan(true, &[], 0), None);
    }

    #[test]
    fn conv_batch_knob_is_read_per_call() {
        let _guard = crate::knobs::TEST_ENV
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        unsafe { std::env::remove_var("ROCKET_CONV_BATCH") };
        assert!(!conv_batch_enabled(), "the per-tile path is the default");
        for (v, want) in [
            ("0", false),
            ("1", true),
            ("2", true),
            ("-1", false),
            ("junk", false),
        ] {
            unsafe { std::env::set_var("ROCKET_CONV_BATCH", v) };
            assert_eq!(conv_batch_enabled(), want, "ROCKET_CONV_BATCH={v}");
        }
        unsafe { std::env::remove_var("ROCKET_CONV_BATCH") };
        assert!(!conv_batch_enabled());
    }
}
