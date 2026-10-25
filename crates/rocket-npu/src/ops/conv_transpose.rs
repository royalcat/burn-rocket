//! Transposed convolution (ConvTranspose2d / "deconvolution") on the RK3588 NPU.
//!
//! Port of `rocket_conv_transpose.c`. Two routes, chosen per descriptor by
//! [`conv_transpose2d_route`]:
//!
//! - the CNA's **hardware deconvolution mode** ([`ConvTransposeRoute::Deconv`], via
//!   [`crate::ops::conv::conv2d_deconv`]): a direct conv at power-of-two strides whose
//!   *compact* input fits one CBUF pass, one task per output-channel tile. The part
//!   interior-dilates the input itself, the output extent is programmed as the
//!   transposed one, and the pad is `k-1-p`; the kernel is rotated and
//!   channel-transposed on the host. Cheaper than the lowering at decoder shapes.
//! - the **lowering** ([`ConvTransposeRoute::Lowered`], everything else): the input is
//!   interior-dilated (stride-1 zeros inserted between pixels) and border-padded on the
//!   host, the kernel is spatially rotated 180° with its in/out channels transposed, and
//!   a **stride-1 forward conv** produces the result, inheriting the forward tiling
//!   bit-for-bit.
//!
//! Tensor layouts (row-major, batch 1):
//!
//! ```text
//! input    in [IC][IH][IW]
//! weights  W  [IC][OC][KH][KW]   (direct; PyTorch/ONNX order: IN channels first)
//!          W  [C][1][KH][KW]     (depthwise, OC == IC == C: one kernel per channel)
//! output   out[OC][OH][OW]
//!
//! OH = (IH-1)*stride_y - 2*pad_top  + dil_y*(KH-1) + opad_y + 1
//! OW = (IW-1)*stride_x - 2*pad_left + dil_x*(KW-1) + opad_x + 1
//!
//! out[oc][ph][pw] = sum_{ic,kh,kw : ih,iw integral and in-range}
//!     in[ic][ih][iw] * W[ic][oc][kh][kw]
//!   where ph = ih*stride_y - pad_top + kh*dil_y   (and the analogous pw)
//! ```
//!
//! Depthwise sums only `ic == oc`, so each channel is upsampled by its own kernel.
//! `opad_*` is the ONNX/PyTorch `output_padding` (a trailing-only extra border that
//! disambiguates the output size when `stride > 1`; `0` = none, must be `< stride`).
//!
//! # The lowering's identity
//!
//! ```text
//! ConvTranspose(X; W, s, p, d, opad) == Conv(dilate_and_pad(X), rot180(W^T); 1, 0, d)
//! ```
//!
//! with `lead = d*(K-1) - p` on the leading edge and `lead + opad` on the trailing
//! edge. Derivation: ConvTranspose puts `in[ih]` at `ph = ih*s - p + kh*d`; in the
//! lowered input `in[ih]` sits at `r = lead + ih*s`, and the forward conv reads
//! `xd[ph + kf*d]`, which equals `lead + ih*s` exactly when `kf = K-1-kh`. `xd` is zero
//! off the stride lattice and in the border, so only integral, in-range `ih` contribute.
//!
//! # Limits (this bring-up)
//!
//! `pad_top <= dil_y*(KH-1)` and `pad_left <= dil_x*(KW-1)`, so the lowered border pad
//! is non-negative (a larger pad would CROP; the plan answers `-2`). The lowering
//! materialises the dilated input, so its cost scales with the **upsampled** size — the
//! inserted zeros are still MAC'd. `ROCKET_CONV_TRANSPOSE_HW=0` forces the lowering.

use half::f16;

use crate::Device;
use crate::knobs;
use crate::ops::conv::{Conv2dDesc, ConvContext, conv2d_deconv, conv2d_deconv_fits};
use crate::ops::conv::{conv2d, conv2d_plan_code};

/// `ROCKET_CONV_TRANSPOSE_LOWERED`: the dilated-input lowering.
pub const ROCKET_CONV_TRANSPOSE_LOWERED: i32 = 0;
/// `ROCKET_CONV_TRANSPOSE_DECONV`: the CNA's hardware deconvolution mode.
pub const ROCKET_CONV_TRANSPOSE_DECONV: i32 = 1;

/// The C `rocket_conv_transpose2d_desc`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvTranspose2dDesc {
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
    /// Output padding (a trailing extra border; `0` = none, must be `< stride`).
    pub opad_y: usize,
    pub opad_x: usize,
    /// Kernel dilation rate (`1` = none).
    pub dil_y: usize,
    pub dil_x: usize,
    /// `true` = depthwise (`OC == IC`, `W [C][1][KH][KW]`).
    pub depthwise: bool,
}

/// A zero shape with the neutral hyperparameters (stride 1, dilation 1, no padding, no
/// output padding, direct).
impl Default for ConvTranspose2dDesc {
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
            opad_y: 0,
            opad_x: 0,
            dil_y: 1,
            dil_x: 1,
            depthwise: false,
        }
    }
}

impl ConvTranspose2dDesc {
    /// The plain strided transposed conv (pads 0, no output padding, dilation 1).
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

/// The output height implied by a descriptor.
pub fn conv_transpose2d_oh(d: &ConvTranspose2dDesc) -> isize {
    (d.ih as isize - 1) * d.stride_y as isize - 2 * d.pad_top as isize
        + d.dil_y as isize * (d.kh as isize - 1)
        + d.opad_y as isize
        + 1
}

/// The output width implied by a descriptor.
pub fn conv_transpose2d_ow(d: &ConvTranspose2dDesc) -> isize {
    (d.iw as isize - 1) * d.stride_x as isize - 2 * d.pad_left as isize
        + d.dil_x as isize * (d.kw as isize - 1)
        + d.opad_x as isize
        + 1
}

/// Which route [`conv_transpose2d`] takes for a descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvTransposeRoute {
    /// The dilated-input lowering onto a stride-1 forward conv.
    Lowered,
    /// The CNA's hardware deconvolution mode.
    Deconv,
}

impl ConvTransposeRoute {
    /// The C route constant (`0`/`1`).
    pub const fn code(self) -> i32 {
        match self {
            ConvTransposeRoute::Lowered => ROCKET_CONV_TRANSPOSE_LOWERED,
            ConvTransposeRoute::Deconv => ROCKET_CONV_TRANSPOSE_DECONV,
        }
    }

    /// Decode a non-negative C route constant.
    pub const fn from_code(code: i32) -> Option<ConvTransposeRoute> {
        Some(match code {
            ROCKET_CONV_TRANSPOSE_LOWERED => ConvTransposeRoute::Lowered,
            ROCKET_CONV_TRANSPOSE_DECONV => ConvTransposeRoute::Deconv,
            _ => return None,
        })
    }

    /// The route's name.
    pub const fn name(self) -> &'static str {
        match self {
            ConvTransposeRoute::Lowered => "lowered",
            ConvTransposeRoute::Deconv => "deconv",
        }
    }
}

/// The forward-conv descriptor the transpose lowers to, plus the lowered geometry.
///
/// Returns `(fwd, ihd, iwd, lead_y, lead_x)`; `fwd` is stride-1, pad-0 over the
/// dilated + bordered input.
pub fn lower_desc(d: &ConvTranspose2dDesc) -> (Conv2dDesc, usize, usize, isize, isize) {
    let ly = d.dil_y as isize * (d.kh as isize - 1) - d.pad_top as isize;
    let lx = d.dil_x as isize * (d.kw as isize - 1) - d.pad_left as isize;
    let ty = ly + d.opad_y as isize; // trailing border = leading + opad
    let tx = lx + d.opad_x as isize;
    let core_h = (d.ih - 1) * d.stride_y + 1; // interior-dilated input height
    let core_w = (d.iw - 1) * d.stride_x + 1;
    let ihd = core_h as isize + ly + ty;
    let iwd = core_w as isize + lx + tx;

    let fwd = Conv2dDesc {
        ic: d.ic,
        ih: ihd.max(0) as usize,
        iw: iwd.max(0) as usize,
        oc: d.oc,
        kh: d.kh,
        kw: d.kw,
        stride_y: 1,
        stride_x: 1,
        pad_top: 0,
        pad_left: 0,
        dil_y: d.dil_y,
        dil_x: d.dil_x,
        depthwise: d.depthwise,
        ..Default::default()
    };
    (fwd, ihd.max(0) as usize, iwd.max(0) as usize, ly, lx)
}

/// The descriptor checks shared by both routes (`desc_valid`): `0` or a negative reason.
pub fn conv_transpose2d_desc_code(d: &ConvTranspose2dDesc) -> i32 {
    if d.ic == 0 || d.ih == 0 || d.iw == 0 || d.oc == 0 || d.kh == 0 || d.kw == 0 {
        return -1;
    }
    if d.stride_x == 0 || d.stride_y == 0 || d.dil_x == 0 || d.dil_y == 0 {
        return -1;
    }
    if d.depthwise && d.oc != d.ic {
        return -1; // one kernel per channel
    }
    // A pad larger than the (dilated) kernel reach would CROP the output: the lowered
    // leading border goes negative, which this bring-up does not implement.
    if d.pad_top > d.dil_y * (d.kh - 1) {
        return -2;
    }
    if d.pad_left > d.dil_x * (d.kw - 1) {
        return -2;
    }
    if conv_transpose2d_oh(d) <= 0 || conv_transpose2d_ow(d) <= 0 {
        return -3;
    }
    0
}

fn pow2_stride(s: usize) -> bool {
    matches!(s, 1 | 2 | 4 | 8)
}

/// Whether the hardware deconvolution route applies (`deconv_eligible`).
///
/// The CNA deconvolution mode dilates by a power-of-two stride only, has no depthwise or
/// dilated form measured, and its pad field (`k-1-p`) has an undecoded anomaly at 12-15
/// when `s = 2`. `ROCKET_CONV_TRANSPOSE_HW=0` forces the lowering.
pub fn conv_transpose2d_deconv_eligible(d: &ConvTranspose2dDesc) -> bool {
    if let Some(v) = knobs::string("ROCKET_CONV_TRANSPOSE_HW")
        && v.starts_with('0')
    {
        return false;
    }
    if crate::hw::current().name != "rk3588" {
        return false;
    }
    if d.depthwise || d.dil_y != 1 || d.dil_x != 1 {
        return false;
    }
    if !pow2_stride(d.stride_y) || !pow2_stride(d.stride_x) {
        return false;
    }
    if d.stride_y == 1 && d.stride_x == 1 {
        return false; // nothing to dilate
    }
    if d.opad_y >= d.stride_y || d.opad_x >= d.stride_x {
        return false;
    }
    if d.kh as isize - 1 - d.pad_top as isize > 11 || d.kw as isize - 1 - d.pad_left as isize > 11 {
        return false;
    }
    conv2d_deconv_fits(
        d.ic,
        d.ih,
        d.iw,
        d.oc,
        conv_transpose2d_oh(d) as usize,
        conv_transpose2d_ow(d) as usize,
        d.kh,
        d.kw,
    )
}

/// `rocket_conv_transpose2d_route`: the C's raw return (`0` lowered, `1` deconv, `<0`
/// refused with [`conv_transpose2d_plan_code`]'s reasons).
pub fn conv_transpose2d_route_code(d: &ConvTranspose2dDesc) -> i32 {
    let r = conv_transpose2d_desc_code(d);
    if r != 0 {
        return r;
    }
    if conv_transpose2d_deconv_eligible(d) {
        return ROCKET_CONV_TRANSPOSE_DECONV;
    }
    let (fwd, ..) = lower_desc(d);
    // Propagate the forward planner's CBUF-fit / alignment refusal.
    conv2d_plan_code(&fwd)
}

/// The route [`conv_transpose2d`] takes, or the C's refusal as a typed error.
pub fn conv_transpose2d_route(d: &ConvTranspose2dDesc) -> crate::Result<ConvTransposeRoute> {
    match conv_transpose2d_route_code(d) {
        0 => Ok(ConvTransposeRoute::Lowered),
        1 => Ok(ConvTransposeRoute::Deconv),
        -2 => Err(crate::Error::shape(
            "rocket_conv_transpose2d_route/pad-crops",
        )),
        -3 => Err(crate::Error::shape(
            "rocket_conv_transpose2d_route/out-extent",
        )),
        -4 => Err(crate::Error::tiling("rocket_conv_transpose2d_route/cbuf")),
        _ => Err(crate::Error::shape("rocket_conv_transpose2d_route")),
    }
}

/// `rocket_conv_transpose2d_plan`: `0` if runnable, else the C's negated reason code.
pub fn conv_transpose2d_plan_code(d: &ConvTranspose2dDesc) -> i32 {
    let r = conv_transpose2d_route_code(d);
    if r < 0 { r } else { 0 }
}

/// [`conv_transpose2d_plan_code`] as a typed result (the C's code classes).
pub fn conv_transpose2d_plan(d: &ConvTranspose2dDesc) -> crate::Result<()> {
    match conv_transpose2d_plan_code(d) {
        0 => Ok(()),
        -2 => Err(crate::Error::shape(
            "rocket_conv_transpose2d_plan/pad-crops",
        )),
        -3 => Err(crate::Error::shape(
            "rocket_conv_transpose2d_plan/out-extent",
        )),
        -4 => Err(crate::Error::tiling("rocket_conv_transpose2d_plan/cbuf")),
        _ => Err(crate::Error::shape("rocket_conv_transpose2d_plan")),
    }
}

/// `rocket_conv_transpose2d_ref_fp16`: the fp32-accumulate scatter-add reference.
///
/// Independent of the route — it is the definition, so the gate validates the lowering
/// (and the hardware deconvolution) rather than just the forward conv.
pub fn conv_transpose2d_ref(d: &ConvTranspose2dDesc, input: &[f16], w: &[f16], out: &mut [f16]) {
    let (ic, ih, iw, oc, kh, kw) = (d.ic, d.ih, d.iw, d.oc, d.kh, d.kw);
    let oh = conv_transpose2d_oh(d) as usize;
    let ow = conv_transpose2d_ow(d) as usize;
    let mut acc = vec![0.0f32; oc * oh * ow];

    let span = if d.depthwise { 1 } else { oc }; // weight is [IC][OC..] direct / [C][1..]
    for c in 0..ic {
        let (o_lo, o_hi) = if d.depthwise { (c, c + 1) } else { (0, oc) };
        for y in 0..ih {
            for x in 0..iw {
                let xv = input[(c * ih + y) * iw + x].to_f32();
                if xv == 0.0 {
                    continue;
                }
                for o in o_lo..o_hi {
                    let woc = if d.depthwise { 0 } else { o };
                    for k in 0..kh {
                        let ph = y as isize * d.stride_y as isize - d.pad_top as isize
                            + k as isize * d.dil_y as isize;
                        if ph < 0 || ph >= oh as isize {
                            continue;
                        }
                        for j in 0..kw {
                            let pw = x as isize * d.stride_x as isize - d.pad_left as isize
                                + j as isize * d.dil_x as isize;
                            if pw < 0 || pw >= ow as isize {
                                continue;
                            }
                            let wv = w[((c * span + woc) * kh + k) * kw + j].to_f32();
                            acc[(o * oh + ph as usize) * ow + pw as usize] += xv * wv;
                        }
                    }
                }
            }
        }
    }
    for (o, a) in out.iter_mut().zip(&acc) {
        *o = f16::from_f32(*a);
    }
}

/// The shared body (`transpose_run`): build the lowered input + flipped weights, run the
/// forward conv (or the hardware route).
fn transpose_run(
    dev: &Device,
    ctx: Option<&mut ConvContext>,
    d: &ConvTranspose2dDesc,
    input: &[f16],
    w: &[f16],
    out: &mut [f16],
) -> crate::Result<()> {
    let route = conv_transpose2d_route(d)?;
    let (ic, ih, iw, oc, kh, kw) = (d.ic, d.ih, d.iw, d.oc, d.kh, d.kw);
    let (fwd, ihd, iwd, ly, lx) = lower_desc(d);
    let deconv = route == ConvTransposeRoute::Deconv;

    let mut xd = if deconv {
        Vec::new()
    } else {
        vec![f16::ZERO; ic * ihd * iwd]
    };
    let wf_n = if d.depthwise {
        ic * kh * kw
    } else {
        oc * ic * kh * kw
    };
    let mut wf = vec![f16::ZERO; wf_n];

    // scatter the input onto the stride lattice inside the leading border
    if !deconv {
        for c in 0..ic {
            for y in 0..ih {
                for x in 0..iw {
                    let r = ly + (y * d.stride_y) as isize;
                    let cc = lx + (x * d.stride_x) as isize;
                    if r < 0 || cc < 0 {
                        continue;
                    }
                    let r = r as usize;
                    let cc = cc as usize;
                    if r >= ihd || cc >= iwd {
                        continue;
                    }
                    xd[(c * ihd + r) * iwd + cc] = input[(c * ih + y) * iw + x];
                }
            }
        }
    }

    if d.depthwise {
        // wf[c][0][kh][kw] = W[c][0][KH-1-kh][KW-1-kw] (spatial flip only)
        for c in 0..ic {
            for k in 0..kh {
                for j in 0..kw {
                    wf[(c * kh + k) * kw + j] = w[(c * kh + (kh - 1 - k)) * kw + (kw - 1 - j)];
                }
            }
        }
    } else {
        // wf[oc][ic][kh][kw] = W[ic][oc][KH-1-kh][KW-1-kw] (180° flip + channel transpose)
        for o in 0..oc {
            for c in 0..ic {
                for k in 0..kh {
                    for j in 0..kw {
                        wf[((o * ic + c) * kh + k) * kw + j] =
                            w[((c * oc + o) * kh + (kh - 1 - k)) * kw + (kw - 1 - j)];
                    }
                }
            }
        }
    }

    // The pad on the dilated surface is k-1-p, which puts row 0 of the result at row 0
    // of the transposed conv; the output extent is the transposed one.
    let (oh, ow) = (
        conv_transpose2d_oh(d) as usize,
        conv_transpose2d_ow(d) as usize,
    );
    if deconv {
        return conv2d_deconv(
            dev,
            ctx,
            ic,
            ih,
            iw,
            oc,
            oh,
            ow,
            kh,
            kw,
            d.stride_y,
            d.stride_x,
            kh - 1 - d.pad_top,
            kw - 1 - d.pad_left,
            input,
            &wf,
            out,
        );
    }
    match ctx {
        Some(ctx) => ctx.conv2d(&fwd, &xd, &wf, out),
        None => conv2d(dev, &fwd, &xd, &wf, out),
    }
}

/// `rocket_conv_transpose2d_fp16`: run a transposed conv on the NPU.
pub fn conv_transpose2d(
    dev: &Device,
    d: &ConvTranspose2dDesc,
    input: &[f16],
    w: &[f16],
    out: &mut [f16],
) -> crate::Result<()> {
    transpose_run(dev, None, d, input, w, out)
}

/// `rocket_conv_transpose2d_fp16_ctx`: the same on a resident-BO context.
pub fn conv_transpose2d_ctx(
    ctx: &mut ConvContext,
    d: &ConvTranspose2dDesc,
    input: &[f16],
    w: &[f16],
    out: &mut [f16],
) -> crate::Result<()> {
    let dev = ctx.device().clone();
    transpose_run(&dev, Some(ctx), d, input, w, out)
}

/// The transposed-conv element `(oh, ow)` helper for a width-only 1D transpose.
///
/// Convenience for callers that keep the time axis on one dimension, mirroring
/// [`crate::ops::conv::conv_out_dim`]'s role for the forward family.
pub fn conv_transpose_out_dim(
    input: usize,
    k: usize,
    stride: usize,
    pad: usize,
    dil: usize,
    opad: usize,
) -> isize {
    (input as isize - 1) * stride as isize - 2 * pad as isize
        + dil as isize * (k as isize - 1)
        + opad as isize
        + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn desc(
        ic: usize,
        ih: usize,
        iw: usize,
        oc: usize,
        kh: usize,
        kw: usize,
    ) -> ConvTranspose2dDesc {
        ConvTranspose2dDesc::new(ic, ih, iw, oc, kh, kw)
    }

    #[test]
    fn output_dims() {
        // Stride 2, 3x3, pad 1, opad 1: exactly double the plane (the usual decoder form).
        let mut d = desc(32, 8, 8, 16, 3, 3);
        d.stride_y = 2;
        d.stride_x = 2;
        d.pad_top = 1;
        d.pad_left = 1;
        d.opad_y = 1;
        d.opad_x = 1;
        assert_eq!((conv_transpose2d_oh(&d), conv_transpose2d_ow(&d)), (16, 16));
        // Without the output padding the same geometry gives 15.
        d.opad_y = 0;
        d.opad_x = 0;
        assert_eq!((conv_transpose2d_oh(&d), conv_transpose2d_ow(&d)), (15, 15));
        // Stride 1 with pad 0 and a 3x3 kernel grows by two.
        let mut d = desc(16, 5, 6, 16, 3, 3);
        assert_eq!((conv_transpose2d_oh(&d), conv_transpose2d_ow(&d)), (7, 8));
        // Dilation 2 adds four instead of two.
        d.dil_y = 2;
        d.dil_x = 2;
        assert_eq!((conv_transpose2d_oh(&d), conv_transpose2d_ow(&d)), (9, 10));
        assert_eq!(conv_transpose_out_dim(5, 3, 1, 0, 2, 0), 9);
    }

    #[test]
    fn plan_and_route_codes() {
        // A direct stride-2 decoder shape takes the hardware route.
        let mut d = desc(32, 8, 8, 16, 3, 3);
        d.stride_y = 2;
        d.stride_x = 2;
        d.pad_top = 1;
        d.pad_left = 1;
        d.opad_y = 1;
        d.opad_x = 1;
        assert_eq!(conv_transpose2d_plan_code(&d), 0);
        assert_eq!(
            conv_transpose2d_route(&d).unwrap(),
            ConvTransposeRoute::Deconv
        );

        // IC % 32 != 0 forces the lowering (the deconv route needs whole K-groups).
        let mut d = desc(24, 8, 8, 24, 3, 3);
        d.stride_y = 2;
        d.stride_x = 2;
        d.pad_top = 1;
        d.pad_left = 1;
        assert_eq!(
            conv_transpose2d_route(&d).unwrap(),
            ConvTransposeRoute::Lowered
        );

        // Depthwise / dilated / stride-1 all lower.
        let mut d = desc(32, 4, 4, 32, 2, 2);
        d.depthwise = true;
        d.stride_y = 2;
        d.stride_x = 2;
        assert_eq!(
            conv_transpose2d_route(&d).unwrap(),
            ConvTransposeRoute::Lowered
        );
        let mut d = desc(32, 4, 4, 16, 3, 3);
        d.dil_y = 2;
        d.dil_x = 2;
        assert_eq!(
            conv_transpose2d_route(&d).unwrap(),
            ConvTransposeRoute::Lowered
        );
        let d = desc(32, 4, 4, 16, 3, 3);
        assert_eq!(
            conv_transpose2d_route(&d).unwrap(),
            ConvTransposeRoute::Lowered
        );

        // `ROCKET_CONV_TRANSPOSE_HW=0` forces the lowering even where the mode fits.
        let mut d = desc(32, 8, 8, 16, 3, 3);
        d.stride_y = 2;
        d.stride_x = 2;
        d.pad_top = 1;
        d.pad_left = 1;
        assert_eq!(
            conv_transpose2d_route_code(&d),
            ROCKET_CONV_TRANSPOSE_DECONV
        );
        let _g = crate::knobs::TEST_ENV
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        // SAFETY: the environment is process-wide; the crate's tests serialize here.
        unsafe { std::env::set_var("ROCKET_CONV_TRANSPOSE_HW", "0") };
        assert_eq!(
            conv_transpose2d_route_code(&d),
            ROCKET_CONV_TRANSPOSE_LOWERED
        );
        assert_eq!(
            conv_transpose2d_route(&d).unwrap(),
            ConvTransposeRoute::Lowered
        );
        unsafe { std::env::remove_var("ROCKET_CONV_TRANSPOSE_HW") };

        // Refusals: a pad larger than the kernel reach crops (-2); a non-positive extent
        // is -3; a malformed descriptor -1.
        let mut d = desc(32, 4, 4, 16, 3, 3);
        d.pad_top = 3;
        assert_eq!(conv_transpose2d_plan_code(&d), -2);
        assert_eq!(
            conv_transpose2d_plan(&d).unwrap_err().status(),
            crate::Status::Shape
        );
        // A pad equal to the kernel reach is allowed and leaves a 1x1 output; one
        // larger collapses it, which is -3.
        let mut d = desc(32, 3, 3, 16, 3, 3);
        d.pad_top = 2;
        d.pad_left = 2;
        assert_eq!((conv_transpose2d_oh(&d), conv_transpose2d_ow(&d)), (1, 1));
        assert_eq!(conv_transpose2d_plan_code(&d), 0);
        let mut d = desc(32, 1, 1, 16, 3, 3);
        d.pad_top = 2;
        d.pad_left = 2;
        assert_eq!(conv_transpose2d_oh(&d), -1);
        assert_eq!(conv_transpose2d_plan_code(&d), -3);
        let mut d = desc(32, 1, 1, 16, 3, 3);
        d.pad_top = 0;
        d.pad_left = 0;
        assert_eq!(conv_transpose2d_oh(&d), 3);
        assert_eq!(conv_transpose2d_plan_code(&d), 0);
        let mut d = desc(32, 4, 4, 16, 3, 3);
        d.depthwise = true; // OC != IC
        assert_eq!(conv_transpose2d_plan_code(&d), -1);
        let mut d = desc(0, 4, 4, 16, 3, 3);
        d.ic = 0;
        assert_eq!(conv_transpose2d_plan_code(&d), -1);
        let mut d = desc(32, 4, 4, 16, 3, 3);
        d.stride_y = 0;
        assert_eq!(conv_transpose2d_plan_code(&d), -1);
    }

    #[test]
    fn route_enum_round_trips() {
        assert_eq!(ConvTransposeRoute::Lowered.code(), 0);
        assert_eq!(ConvTransposeRoute::Deconv.code(), 1);
        for r in [ConvTransposeRoute::Lowered, ConvTransposeRoute::Deconv] {
            assert_eq!(ConvTransposeRoute::from_code(r.code()), Some(r));
        }
        assert_eq!(ConvTransposeRoute::from_code(-1), None);
        assert_eq!(ConvTransposeRoute::Deconv.name(), "deconv");
    }

    #[test]
    fn reference_hand_cases() {
        // 1x1 stride 1: out == W^T applied per pixel. W is [IC][OC][1][1].
        let mut d = desc(2, 2, 1, 3, 1, 1);
        d.oc = 3;
        let input: Vec<f16> = [1.0, 2.0, 3.0, 4.0]
            .iter()
            .map(|v| f16::from_f32(*v))
            .collect();
        let mut w = vec![f16::ZERO; 2 * 3];
        w[0] = f16::from_f32(1.0); // ic0 -> oc0
        w[1] = f16::from_f32(2.0); // ic0 -> oc1
        w[5] = f16::from_f32(-1.0); // ic1 -> oc2
        let mut out = vec![f16::ZERO; 3 * 2];
        conv_transpose2d_ref(&d, &input, &w, &mut out);
        // oc0: in[ic0] * 1.0; oc1: in[ic0] * 2.0; oc2: in[ic1] * -1.0
        assert_eq!(out[0].to_f32(), 1.0);
        assert_eq!(out[1].to_f32(), 2.0);
        assert_eq!(out[2].to_f32(), 2.0);
        assert_eq!(out[3].to_f32(), 4.0);
        assert_eq!(out[4].to_f32(), -3.0);
        assert_eq!(out[5].to_f32(), -4.0);

        // Stride 2 with a 1x1 kernel: the pixels land on the even lattice, odd output
        // positions stay zero, and a trailing opad adds one zero row/col.
        let mut d = desc(1, 2, 2, 1, 1, 1);
        d.stride_y = 2;
        d.stride_x = 2;
        d.opad_y = 1;
        d.opad_x = 1;
        assert_eq!((conv_transpose2d_oh(&d), conv_transpose2d_ow(&d)), (4, 4));
        let input: Vec<f16> = (1..=4).map(|v| f16::from_f32(v as f32)).collect();
        let w = vec![f16::from_f32(1.0)];
        let mut out = vec![f16::ZERO; 16];
        conv_transpose2d_ref(&d, &input, &w, &mut out);
        let got: Vec<f32> = out.iter().map(|v| v.to_f32()).collect();
        assert_eq!(
            got,
            vec![
                1.0, 0.0, 2.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, //
                3.0, 0.0, 4.0, 0.0, //
                0.0, 0.0, 0.0, 0.0
            ]
        );

        // 3x3 stride 1 pad 0 with a full-ones kernel: every input pixel scatters into
        // its trailing 3x3 neighbourhood, so the output is the overlap count of the
        // input plane. `input` here is the 2x2 [1,2;3,4] plane.
        let d = desc(1, 2, 2, 1, 3, 3);
        assert_eq!((conv_transpose2d_oh(&d), conv_transpose2d_ow(&d)), (4, 4));
        let w = vec![f16::from_f32(1.0); 9];
        let mut out = vec![f16::ZERO; 16];
        conv_transpose2d_ref(&d, &input, &w, &mut out);
        let got: Vec<f32> = out.iter().map(|v| v.to_f32()).collect();
        assert_eq!(
            got,
            vec![
                1.0, 3.0, 3.0, 2.0, //
                4.0, 10.0, 10.0, 6.0, //
                4.0, 10.0, 10.0, 6.0, //
                3.0, 7.0, 7.0, 4.0
            ]
        );

        // Depthwise stride 2: each channel is upsampled by its own 1x1 tap.
        let mut d = desc(2, 2, 1, 2, 1, 1);
        d.depthwise = true;
        d.stride_y = 2;
        d.stride_x = 2;
        let input: Vec<f16> = [1.0, 2.0, 3.0, 4.0]
            .iter()
            .map(|v| f16::from_f32(*v))
            .collect();
        let w = vec![f16::from_f32(2.0), f16::from_f32(0.5)];
        let oh = conv_transpose2d_oh(&d) as usize;
        let ow = conv_transpose2d_ow(&d) as usize;
        let mut out = vec![f16::ZERO; 2 * oh * ow];
        conv_transpose2d_ref(&d, &input, &w, &mut out);
        assert_eq!(out[0].to_f32(), 2.0); // ch0 y0 x0
        assert_eq!(out[ow].to_f32(), 0.0);
        assert_eq!(out[2 * ow].to_f32(), 4.0); // ch0 y1
        assert_eq!(out[oh * ow].to_f32(), 1.5); // ch1 y0, *0.5
    }

    #[test]
    fn lower_desc_borders_match_the_transposed_extent() {
        // The whole identity: the lowered input's extent minus the kernel reach equals
        // the transposed output extent, with the leading border `d*(K-1) - p`.
        for &(ih, iw, k, s, p, opad, dil) in &[
            (8usize, 8usize, 3usize, 2usize, 1usize, 1usize, 1usize),
            (7, 5, 3, 2, 1, 0, 1),
            (5, 6, 5, 1, 2, 0, 1),
            (4, 4, 3, 4, 1, 3, 1),
            (6, 6, 3, 1, 1, 0, 2),
        ] {
            let mut d = desc(32, ih, iw, 16, k, k);
            d.stride_y = s;
            d.stride_x = s;
            d.pad_top = p;
            d.pad_left = p;
            d.opad_y = opad;
            d.opad_x = opad;
            d.dil_y = dil;
            d.dil_x = dil;
            let (fwd, ihd, iwd, ly, lx) = lower_desc(&d);
            let ty = ly + opad as isize;
            let tx = lx + opad as isize;
            assert_eq!(ly, dil as isize * (k as isize - 1) - p as isize);
            assert_eq!(lx, ly);
            assert_eq!(ihd as isize, (ih - 1) as isize * s as isize + 1 + ly + ty);
            assert_eq!(iwd as isize, (iw - 1) as isize * s as isize + 1 + lx + tx);
            // The forward conv over the lowered input reproduces the extent:
            let foh = crate::ops::conv::conv_out_dim(ihd, k, 1, 0, dil);
            assert_eq!(foh, conv_transpose2d_oh(&d));
            let fow = crate::ops::conv::conv_out_dim(iwd, k, 1, 0, dil);
            assert_eq!(fow, conv_transpose2d_ow(&d));
            assert_eq!(fwd.stride_y, 1);
            assert_eq!(fwd.pad_top, 0);
            assert_eq!(fwd.dil_y, dil);
            assert_eq!(fwd.depthwise, d.depthwise);
        }
    }
}
