//! fp16 CONV_2D register-command emission (RK3588, direct + depthwise).
//!
//! Direct translation of `gen_conv2d_task` + `gen_conv2d_fill` — the bodies behind
//! `gen_conv2d_fp16` and `gen_conv2d_dw_fp16` in `npu_regcmd.c`. One call emits the
//! complete program for one conv tile; the orchestration (host scatter, tiling,
//! submission) lives in [`crate::ops::conv`].
//!
//! The CNA here is a real convolution engine: `KH`/`KW`, the strides, the symmetric
//! `pad_top`/`pad_left` and the ATROUS dilation rates are programmed into
//! `CONV_CON1..3`, the weight cube is `(OC/16, IC/32, KH, KW, 16, 32)`, and no im2col
//! happens anywhere. The task emitter is a conv-aware sibling of the matmul one: same
//! CNA/CORE/DPU sequence and the same plain-conv DPU_RDMA block, plus
//!   * the `CONV_CON3` dilation fields (rate - 1; `0` is "no dilation" and is
//!     byte-identical to the matmul emitter),
//!   * the `CONV_CON3` deconvolution-stride fields and the `CONV_CON1` `DECONV` bit
//!     (the CNA's transposed-conv mode, used by [`crate::ops::conv_transpose`]),
//!   * `CORE_MISC_CFG.DW_EN` plus `conv_mode = 3` for depthwise,
//!   * the fused-activation epilogue (`lut`: BN index scale -> EW LUT -> affine
//!     OUT_CVT) on the direct path.
//!
//! # Depthwise output-writer fields
//!
//! A depthwise job that copies the direct-conv values is channel-plausible and wrong.
//! The four fields the upstream bring-up had to change are baked in here exactly as
//! in C: `size_e = 3`, `od_bypass = 0`, `surf_add *= 2`, `bs_ow_op = 0x80`, and
//! `feature_grains = 50 + stride_y + 1` when the caller did not override it. The
//! `ROCKET_CONV_DW_SIZEE` / `ROCKET_CONV_DW_BSOWOP` / `ROCKET_CONV_DW_OCPAD` knobs stay
//! available for a single-variable hardware sweep.
//!
//! # Refusals
//!
//! The generator's own classes: a feature/weight tile that overflows the CBUF
//! (`-1`/`-2` in C) is [`Error::tiling`], a deconvolution request outside the mode's
//! contract is [`Error::shape`], and a 10-bit `feature_grains` overflow is
//! [`Error::tiling`]. The C's `-1` for a shape the *plan* should have refused is not
//! reproducible here because [`crate::ops::conv`] never calls the emitter with one.

use crate::error::{Error, Result};
use crate::knobs;
use crate::log::log_error;

use crate::ops::activation::LutEpilogue;

use super::desc::*;

/// The fused conv->activation epilogue: the 1026-entry table and the register word
/// set built by [`crate::ops::activation::lut_epilogue_build`].
///
/// Mirrors the C `npu_dpu_desc.lut_ep` (`lut_epilogue_t` plus its `const uint16_t *`
/// table, which the C struct carries as two pointers).
#[derive(Debug, Clone, Copy)]
pub struct ConvAct<'a> {
    /// `LE[0..=512]`, `LO[513..=1025]`.
    pub lut: &'a [u16],
    /// The epilogue register words.
    pub ep: &'a LutEpilogue,
}

/// Parameters of one fp16 conv tile (the RK3588 subset of the C `conv_params_t`).
///
/// The fields the C struct carries for the int8/out and RK3576 paths
/// (`int8_out`, the quant scales, `bias_dma`, `lut`, `ih_full`/`oh_full`,
/// `in_surf_elems`/`in_pitch_w`) are absent: those encodings are not this family's.
#[derive(Debug, Clone, Copy, Default)]
pub struct ConvParams<'a> {
    /// Input channels (direct: the driver zero-pads up to 32; depthwise: `IC == OC`).
    pub ic: u16,
    /// Input height. Must be the *job's* height, already padded to the 4-row DMA floor
    /// by the caller when the tile is shorter (see [`crate::ops::conv`]).
    pub ih: u16,
    /// Input width.
    pub iw: u16,
    /// Output channels (direct) / depthwise channels.
    pub oc: u16,
    /// Output height (the caller derives it; the deconv route passes the transposed
    /// extent).
    pub oh: u16,
    /// Output width.
    pub ow: u16,
    /// Kernel height / width.
    pub kh: u16,
    pub kw: u16,
    /// Window strides (`1` on the deconvolution route).
    pub stride_y: u8,
    pub stride_x: u8,
    /// Dilation rate (`1` = none); emitted as rate - 1.
    pub dil_y: u8,
    pub dil_x: u8,
    /// Symmetric leading pad (the CNA has no trailing-pad register).
    pub pad_top: u8,
    pub pad_left: u8,
    /// NPU IOVA of the scattered feature cube.
    pub input_dma: u32,
    /// NPU IOVA of the scattered weight cube.
    pub weights_dma: u32,
    /// NPU IOVA of the output cube.
    pub output_dma: u32,
    /// `true` narrows the output to fp16 (the C2=8 cube); `false` keeps fp32.
    pub fp32_to_fp16: bool,
    /// Depthwise channel group (the weight cube's innermost atom). `0` -> 32 for
    /// fp16; `ROCKET_CONV_DW_GROUP` overrides it (and the host scatter must agree).
    pub dw_group: u8,
    /// Fused activation (direct path only, `fp32_to_fp16` required).
    pub act: Option<ConvAct<'a>>,
    /// The CNA deconvolution mode's interior-dilation factor per axis; `1` is off.
    /// `2`, `4` or `8` turn it on.
    pub deconv_sy: u8,
    pub deconv_sx: u8,
    /// OUT: words written (the C `task_count`).
    pub task_count: u32,
}

impl<'a> ConvParams<'a> {
    /// The plain-conv parameter set: fp16 out, no activation, no deconvolution.
    ///
    /// The C `conv_params_t` field order, minus the DMA/callback arguments that have
    /// their own struct fields.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ic: u16,
        ih: u16,
        iw: u16,
        oc: u16,
        oh: u16,
        ow: u16,
        kh: u16,
        kw: u16,
        input_dma: u32,
        weights_dma: u32,
        output_dma: u32,
    ) -> Self {
        Self {
            ic,
            ih,
            iw,
            oc,
            oh,
            ow,
            kh,
            kw,
            stride_y: 1,
            stride_x: 1,
            dil_y: 1,
            dil_x: 1,
            pad_top: 0,
            pad_left: 0,
            input_dma,
            weights_dma,
            output_dma,
            fp32_to_fp16: true,
            dw_group: 0,
            act: None,
            deconv_sy: 1,
            deconv_sx: 1,
            task_count: 0,
        }
    }
}

/// Words a fused-activation conv program needs (the C `regs[2048]` bound).
pub const MAX_OPS_ACT: usize = 2048;
/// Words a plain conv program needs (the C `regs[256]` bound).
pub const MAX_OPS: usize = 256;

#[derive(Default, Clone)]
struct Cna {
    conv_mode: u32,
    in_precision: u32,
    proc_precision: u32,
    kernel_groups: u32,
    feature_grains: u32,
    conv_y_stride: u32,
    conv_x_stride: u32,
    atrous_y: u32,
    atrous_x: u32,
    deconv: bool,
    deconv_y: u32,
    deconv_x: u32,
    datain_width: u32,
    datain_height: u32,
    datain_channel: u32,
    dataout_width: u32,
    dataout_height: u32,
    dataout_atomics: u32,
    weight_width: u32,
    weight_height: u32,
    weight_kernels: u32,
    weight_bytes: u32,
    weight_bytes_per_kernel: u32,
    weight_bank: u32,
    data_bank: u32,
    data_entries: u32,
    data_sign: u32,
    cvt_type: u32,
    cvt_bypass: u32,
    cvt_scale0: u32,
    cvt_scale1: u32,
    cvt_scale2: u32,
    cvt_scale3: u32,
    fc_skip_en: u32,
    data_offset: u32,
    pad_left: u32,
    pad_top: u32,
    feature_base_addr: u32,
    weight_offset: u32,
    weight_burst_len: u32,
    data_burst_len: u32,
    line_stride: u32,
    surf_stride: u32,
    dma_width: u32,
    dma_height: u32,
    dma_channel: u32,
    decompress_addr0: u32,
    pad_con1: u32,
}

#[derive(Default, Clone)]
struct Core {
    proc_precision: u32,
    qd_en: u32,
    dw_en: u32,
    dataout_height: u32,
    dataout_width: u32,
    dataout_channel: u32,
}

#[derive(Default, Clone)]
struct Dpu {
    burst_len: u32,
    conv_mode: u32,
    output_mode: u32,
    flying_mode: u32,
    out_precision: u32,
    in_precision: u32,
    proc_precision: u32,
    dst_base_addr: u32,
    dst_surf_stride: u32,
    width: u32,
    height: u32,
    channel: u32,
    bs_bypass: u32,
    bs_alu_bypass: u32,
    bs_mul_bypass: u32,
    bs_relu_bypass: u32,
    bs_ow_op: u32,
    od_bypass: u32,
    size_e_2: u32,
    size_e_1: u32,
    size_e_0: u32,
    channel_wdma: u32,
    height_wdma: u32,
    width_wdma: u32,
    bn_relu_bypass: u32,
    bn_mul_bypass: u32,
    bn_alu_bypass: u32,
    bn_bypass: u32,
    ew_bypass: u32,
    ew_op_bypass: u32,
    ew_lut_bypass: u32,
    ew_op_cvt_bypass: u32,
    ew_relu_bypass: u32,
    fp32tofp16_en: u32,
    out_cvt_scale: u32,
    out_cvt_offset: u32,
    out_cvt_shift: u32,
    out_cvt_round: u8,
    surf_add: u32,
    /// The int8-out depthwise path's per-OC int32 bias add in the BS ALU (the C
    /// `npu_dpu_desc.bias_en`): the BS block is driven in add mode and BRDMA fetches
    /// the bias cube. `false` keeps the BS block fully bypassed, which is what every
    /// fp16 and int32-raw program does (`bias_en = 0` in the C default).
    bias_en: bool,
    /// `DPU_RDMA_BS_BASE_ADDR`, the int32 bias cube's IOVA (read only with `bias_en`).
    bias_base_addr: u32,
}

/// `gen_conv2d_fp16`: emit one direct fp16 conv program into `ops`.
///
/// Returns the word count and stores it in [`ConvParams::task_count`] as well (the C
/// struct's out-parameter), so a caller sizing a regcmd buffer sees it.
pub fn gen_conv2d_fp16(p: &mut ConvParams<'_>, ops: &mut Vec<NpuOp>) -> Result<usize> {
    fill(p, false, ops)
}

/// `gen_conv2d_dw_fp16`: the depthwise variant (`CONV_MODE = 3` + `DW_EN`).
pub fn gen_conv2d_dw_fp16(p: &mut ConvParams<'_>, ops: &mut Vec<NpuOp>) -> Result<usize> {
    fill(p, true, ops)
}

fn fill(p: &mut ConvParams<'_>, depthwise: bool, ops: &mut Vec<NpuOp>) -> Result<usize> {
    let banks = crate::hw::current().cbuf_banks;
    let bank = CBUF_BANK_SIZE;
    let ic = p.ic as u32;
    let ih = p.ih as u32;
    let iw = p.iw as u32;
    let oc = p.oc as u32;
    let oh = p.oh as u32;
    let ow = p.ow as u32;
    let kh = p.kh as u32;
    let kw = p.kw as u32;

    let mut cna = Cna {
        conv_mode: if depthwise { 3 } else { DIRECT_CONVOLUTION },
        in_precision: PRECISION_FLOAT16,
        proc_precision: PRECISION_FLOAT16,
        kernel_groups: 0,
        feature_grains: ih + 1,
        ..Default::default()
    };
    if let Some(v) = knobs::u32_auto("ROCKET_CONV_GRAINS") {
        cna.feature_grains = v & 0xFFFF;
    }
    {
        let mut sx = p.stride_x as u32;
        let mut sy = p.stride_y as u32;
        if knobs::present("ROCKET_CONV_STRIDE_SWAP") {
            std::mem::swap(&mut sx, &mut sy);
        }
        cna.conv_x_stride = sx;
        cna.conv_y_stride = sy;
    }
    {
        let dx = if p.dil_x > 0 { p.dil_x as u32 } else { 1 };
        let dy = if p.dil_y > 0 { p.dil_y as u32 } else { 1 };
        let bias = if knobs::present("ROCKET_CONV_DIL_RAW") {
            0
        } else {
            1
        };
        cna.atrous_x = dx.wrapping_sub(bias);
        cna.atrous_y = dy.wrapping_sub(bias);
    }

    // The deconvolution mode (the transposed-expansion route): direct only, forward
    // stride 1, power-of-two interior dilation up to 8.
    if p.deconv_sy > 1 || p.deconv_sx > 1 {
        let dsy = if p.deconv_sy > 1 {
            p.deconv_sy as u32
        } else {
            1
        };
        let dsx = if p.deconv_sx > 1 {
            p.deconv_sx as u32
        } else {
            1
        };
        if depthwise
            || p.stride_y > 1
            || p.stride_x > 1
            || (dsy & (dsy - 1)) != 0
            || (dsx & (dsx - 1)) != 0
            || dsy > 8
            || dsx > 8
        {
            log_error!(
                "gen_conv2d_fp16: the deconvolution mode takes a direct stride-1 conv \
                 and power-of-two strides up to 8 (got {dsy} x {dsx})"
            );
            return Err(Error::shape("gen_conv2d_fp16/deconv"));
        }
        cna.deconv = true;
        cna.deconv_y = dsy - 1;
        cna.deconv_x = dsx - 1;
    }

    cna.datain_width = iw;
    cna.datain_height = ih;
    cna.datain_channel = ic;
    cna.dataout_width = ow;
    cna.dataout_height = oh;
    cna.dataout_atomics = ow * oh;
    cna.weight_width = kw;
    cna.weight_height = kh;

    let fd_bytes = (iw * ih * ic) as usize * 2;
    let fd_banks = fd_bytes.div_ceil(bank);
    if fd_banks > banks - 1 {
        return Err(Error::tiling("gen_conv2d_fp16/feature-banks"));
    }

    let weight_banks;
    if depthwise {
        // One filter per channel; the whole KH*KW*C cube spans banks by
        // KW*KH*IC (not *OC), and G is the in-cube channel group.
        let mut g = if p.dw_group != 0 {
            p.dw_group as u32
        } else {
            32
        };
        if let Some(v) = knobs::u32_auto("ROCKET_CONV_DW_GROUP") {
            g = v;
        }
        if g == 0 {
            g = 32;
        }
        let cpad = ic.div_ceil(g) * g;
        cna.weight_kernels = 1;
        cna.weight_bytes_per_kernel = kw * kh * ic * 2;
        cna.weight_bytes = cpad * kh * kw * 2;
        let wb = (cna.weight_bytes as usize).div_ceil(bank);
        if fd_banks + wb > banks {
            return Err(Error::tiling("gen_conv2d_fp16/dw-banks"));
        }
        weight_banks = banks - fd_banks;
    } else {
        cna.weight_kernels = oc;
        cna.weight_bytes_per_kernel = kw * kh * ic * 2;
        cna.weight_bytes = cna.weight_bytes_per_kernel * cna.weight_kernels;
        if cna.weight_bytes_per_kernel as usize <= bank {
            weight_banks = banks - fd_banks;
        } else {
            return Err(Error::tiling("gen_conv2d_fp16/weight-bank"));
        }
    }

    cna.weight_bank = weight_banks as u32;
    cna.data_bank = fd_banks as u32;
    cna.data_entries = (iw * ic).div_ceil(32);
    if let Some(v) = knobs::u32_auto("ROCKET_CONV_DENTRIES") {
        cna.data_entries = v & 0xFFFF;
    }
    cna.data_sign = 1;
    cna.cvt_type = 1;
    cna.cvt_bypass = 1;
    cna.cvt_scale0 = 1;
    cna.cvt_scale1 = 1;
    cna.cvt_scale2 = 1;
    cna.cvt_scale3 = 1;
    cna.fc_skip_en = 0;
    cna.data_offset = 0;
    cna.pad_left = p.pad_left as u32;
    cna.pad_top = p.pad_top as u32;
    cna.feature_base_addr = p.input_dma;
    cna.weight_offset = 0;
    cna.weight_burst_len = 0xF;
    cna.data_burst_len = 0xF;
    cna.line_stride = iw * 4;
    // The feature surface (channel-plane) stride. The matmul uses
    // `line_stride*(IH/4 - 1)`, which truncates for an IH that is not a multiple of 4;
    // Mesa computes it as `line_stride*(IH/4 - 1)` in float == `IW*(IH-4)` here.
    let surf = cna.line_stride as i64 * (ih as i64 - 4) / 4;
    cna.surf_stride = if surf < 0 { 0 } else { surf as u32 };
    cna.dma_width = iw;
    cna.dma_height = ih;
    cna.dma_channel = ic;
    cna.decompress_addr0 = p.weights_dma;

    let mut core = Core {
        proc_precision: PRECISION_FLOAT16,
        qd_en: 1,
        dw_en: depthwise as u32,
        dataout_height: oh - 1,
        dataout_width: ow - 1,
        dataout_channel: oc - 1,
    };

    let mut dpu = Dpu {
        burst_len: 0xF,
        conv_mode: if depthwise { 3 } else { DIRECT_CONVOLUTION },
        output_mode: 0x2,
        flying_mode: 0,
        out_precision: if p.fp32_to_fp16 {
            PRECISION_FLOAT16
        } else {
            PRECISION_FLOAT32
        },
        in_precision: PRECISION_FLOAT16,
        proc_precision: PRECISION_FLOAT16,
        dst_base_addr: p.output_dma,
        dst_surf_stride: oh * ow,
        width: core.dataout_width,
        height: core.dataout_height,
        channel: core.dataout_channel,
        bs_bypass: 1,
        bs_alu_bypass: 1,
        bs_mul_bypass: 1,
        bs_relu_bypass: 1,
        bn_bypass: 1,
        bn_alu_bypass: 1,
        bn_mul_bypass: 1,
        bn_relu_bypass: 1,
        ew_bypass: 1,
        ew_op_bypass: 1,
        ew_lut_bypass: 1,
        ew_op_cvt_bypass: 1,
        ew_relu_bypass: 1,
        fp32tofp16_en: p.fp32_to_fp16 as u32,
        out_cvt_scale: 1,
        ..Default::default()
    };
    if p.fp32_to_fp16 {
        dpu.size_e_2 = 1;
        dpu.size_e_1 = 1;
        dpu.size_e_0 = 1;
    } else {
        dpu.size_e_2 = 3;
        dpu.size_e_1 = 3;
        dpu.size_e_0 = 3;
    }
    dpu.od_bypass = 1;
    dpu.width_wdma = core.dataout_width;
    dpu.height_wdma = core.dataout_height;
    dpu.channel_wdma = core.dataout_channel;
    dpu.surf_add = if p.fp32_to_fp16 {
        dpu.dst_surf_stride * 2
    } else {
        dpu.dst_surf_stride * 4
    };

    if depthwise {
        // The depthwise output-writer set (see the module docs).
        dpu.size_e_2 = 3;
        dpu.size_e_1 = 3;
        dpu.size_e_0 = 3;
        dpu.od_bypass = 0;
        dpu.surf_add *= 2;
        if !knobs::present("ROCKET_CONV_GRAINS") {
            cna.feature_grains = 50 + p.stride_y as u32 + 1;
        }
        dpu.bs_ow_op = 0x80; // 0x80 - weight_zp (symmetric fp16: zp 0)
        if let Some(v) = knobs::u32_auto("ROCKET_CONV_DW_SIZEE") {
            let se = v & 0x7;
            dpu.size_e_2 = se;
            dpu.size_e_1 = se;
            dpu.size_e_0 = se;
        }
        if let Some(v) = knobs::u32_auto("ROCKET_CONV_DW_BSOWOP") {
            dpu.bs_ow_op = v;
        }
        if knobs::present("ROCKET_CONV_DW_OCPAD") {
            let mut oc_eff = oc.div_ceil(32) * 32;
            if oc <= 32 {
                oc_eff *= 2;
            }
            oc_eff = oc_eff.div_ceil(64) * 64;
            core.dataout_channel = oc_eff - 1;
            dpu.channel = oc_eff - 1;
            dpu.channel_wdma = oc_eff - 1;
        }
    }

    // The fused epilogue requires the fp16-out writer (the LUT result is fp16); the C
    // generator does not force it — the contract is the caller's, and
    // `ops::conv` only ever asks for it with `fp32_to_fp16 = true`.

    let start = ops.len();
    emit_task(ops, &cna, &core, &dpu, p.act.as_ref())?;
    let count = ops.len() - start;
    p.task_count = count as u32;
    Ok(count)
}

// ── int8 conv (the int8 sibling of `fill`) ────────────────────────────────────

/// Parameters of one int8 conv tile (the int8 subset of the C `conv_params_t`).
///
/// The C reuses one `conv_params_t` for all four generators; this port keeps the fp16
/// struct free of the quant fields ([`ConvInt8Params`] carries them) so neither has
/// fields its own path cannot use. The direct int8 path ignores the scales, the zero
/// points and `bias_dma` (it writes a raw int32 accumulator that the caller requants);
/// the depthwise path with `int8_out` is the on-chip-requant writer and uses all of
/// them.
#[derive(Debug, Clone, Copy)]
pub struct ConvInt8Params {
    /// Input channels. The direct path is called with `IC` already padded to 32 (the
    /// weight `ic` group); depthwise with the channel count of this job's chunk.
    pub ic: u16,
    /// The job's input height, already padded to the 4-row CNA DMA floor when short.
    pub ih: u16,
    pub iw: u16,
    /// Output channels (direct: padded to the 32-channel weight oc group by the caller).
    pub oc: u16,
    pub oh: u16,
    pub ow: u16,
    pub kh: u16,
    pub kw: u16,
    /// Window strides.
    pub stride_y: u8,
    pub stride_x: u8,
    /// Dilation rate (`1` = none); emitted as rate - 1.
    pub dil_y: u8,
    pub dil_x: u8,
    /// Symmetric leading pad.
    pub pad_top: u8,
    pub pad_left: u8,
    /// NPU IOVA of the scattered int8 feature cube (`C2 = 16`).
    pub input_dma: u32,
    /// NPU IOVA of the scattered int8 weight cube (oc group 32 / ic group 32).
    pub weights_dma: u32,
    /// NPU IOVA of the int32 (direct) or int8 (depthwise) output cube.
    pub output_dma: u32,
    /// Depthwise channel group: `0` -> 64 (Mesa's int8 value);
    /// `ROCKET_CONV_DW_GROUP` overrides it, as in C.
    pub dw_group: u8,
    /// `true` selects Mesa's int8-output writer with the on-chip requant (the depthwise
    /// path); `false` keeps the validated int32-raw datapath.
    pub int8_out: bool,
    /// Per-tensor quant scales (`in_scale * w_scale / out_scale` is the requant scale).
    pub in_scale: f32,
    pub w_scale: f32,
    pub out_scale: f32,
    /// Zero points in the **uint8-centred** domain the generator's formulas use, i.e.
    /// `model_zp + 0x80` (see [`crate::ops::conv`] and the C's note).
    pub input_zero_point: i32,
    pub output_zero_point: i32,
    pub weight_zero_point: i32,
    /// NPU IOVA of the per-OC int32 bias cube (BRDMA), read only with `int8_out`.
    pub bias_dma: u32,
    /// OUT: words written (the C `task_count`).
    pub task_count: u32,
}

impl ConvInt8Params {
    /// A plain direct int8 conv: int32-raw output, no depthwise, no quant fields.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ic: u16,
        ih: u16,
        iw: u16,
        oc: u16,
        oh: u16,
        ow: u16,
        kh: u16,
        kw: u16,
        input_dma: u32,
        weights_dma: u32,
        output_dma: u32,
    ) -> Self {
        Self {
            ic,
            ih,
            iw,
            oc,
            oh,
            ow,
            kh,
            kw,
            stride_y: 1,
            stride_x: 1,
            dil_y: 1,
            dil_x: 1,
            pad_top: 0,
            pad_left: 0,
            input_dma,
            weights_dma,
            output_dma,
            dw_group: 0,
            int8_out: false,
            in_scale: 0.0,
            w_scale: 0.0,
            out_scale: 1.0,
            input_zero_point: 0,
            output_zero_point: 0,
            weight_zero_point: 0,
            bias_dma: 0,
            task_count: 0,
        }
    }
}

/// `gen_conv2d_int8`: emit one direct int8 conv program (`int8 x int8 -> int32`).
///
/// The mechanical int8 deltas vs [`gen_conv2d_fp16`]: 1 B/element, `C2 = 16` feature
/// cube, `OC`-group-32 weight cube, `data_entries = ceil(IW*IC/64)`, `QD_EN = 0`, the
/// int32 output writer (`OUT_PRECISION = int32`, `size_e = 7`, `surf_add = stride*8`)
/// and — the hardware finding the C records — **one extra CBUF bank for the feature**
/// (`data_bank = fd_banks + 1`), because the int8 C2=16 feature-cube DMA over-reads one
/// bank past its `ceil` allocation at near-bank-full geometries.
pub fn gen_conv2d_int8(p: &mut ConvInt8Params, ops: &mut Vec<NpuOp>) -> Result<usize> {
    fill_int8(p, false, ops)
}

/// `gen_conv2d_dw_int8`: the depthwise variant. With [`ConvInt8Params::int8_out`] it is
/// Mesa's int8-output writer with the on-chip requant (`QD_EN = 1`, the BS ALU bias add
/// from BRDMA, the three-field OUT_CVT, `size_e = 3` / `surf_add = stride*4`); without
/// it the int32-raw depthwise encoding (kept for the hardware sweep, as in C).
pub fn gen_conv2d_dw_int8(p: &mut ConvInt8Params, ops: &mut Vec<NpuOp>) -> Result<usize> {
    fill_int8(p, true, ops)
}

/// The shared int8 body (`gen_conv2d_int8_fill`).
fn fill_int8(p: &mut ConvInt8Params, depthwise: bool, ops: &mut Vec<NpuOp>) -> Result<usize> {
    let banks = crate::hw::current().cbuf_banks;
    let bank = CBUF_BANK_SIZE;
    let ic = p.ic as u32;
    let ih = p.ih as u32;
    let iw = p.iw as u32;
    let oc = p.oc as u32;
    let oh = p.oh as u32;
    let ow = p.ow as u32;
    let kh = p.kh as u32;
    let kw = p.kw as u32;

    let mut cna = Cna {
        conv_mode: if depthwise { 3 } else { DIRECT_CONVOLUTION },
        in_precision: PRECISION_INT8,
        proc_precision: PRECISION_INT8,
        kernel_groups: 0,
        feature_grains: ih + 1,
        ..Default::default()
    };
    if let Some(v) = knobs::u32_auto("ROCKET_CONV_GRAINS") {
        cna.feature_grains = v & 0xFFFF;
    }
    {
        let mut sx = p.stride_x as u32;
        let mut sy = p.stride_y as u32;
        if knobs::present("ROCKET_CONV_STRIDE_SWAP") {
            std::mem::swap(&mut sx, &mut sy);
        }
        cna.conv_x_stride = sx;
        cna.conv_y_stride = sy;
    }
    {
        let dx = if p.dil_x > 0 { p.dil_x as u32 } else { 1 };
        let dy = if p.dil_y > 0 { p.dil_y as u32 } else { 1 };
        let bias = if knobs::present("ROCKET_CONV_DIL_RAW") {
            0
        } else {
            1
        };
        cna.atrous_x = dx.wrapping_sub(bias);
        cna.atrous_y = dy.wrapping_sub(bias);
    }

    cna.datain_width = iw;
    cna.datain_height = ih;
    cna.datain_channel = ic;
    cna.dataout_width = ow;
    cna.dataout_height = oh;
    cna.dataout_atomics = ow * oh;
    cna.weight_width = kw;
    cna.weight_height = kh;

    // 1 byte per element on this path.
    let fd_bytes = (iw * ih * ic) as usize;
    let fd_banks = fd_bytes.div_ceil(bank);
    if fd_banks > banks - 1 {
        return Err(Error::tiling("gen_conv2d_int8/feature-banks"));
    }

    let weight_banks;
    if depthwise {
        // Mesa's int8 group is 64 (feature atom 16 x 4); the fp16 value is 32. The
        // HW-unconfirmed int8 group is an env sweep, as in C.
        let mut g = if p.dw_group != 0 {
            p.dw_group as u32
        } else {
            64
        };
        if let Some(v) = knobs::u32_auto("ROCKET_CONV_DW_GROUP") {
            g = v;
        }
        if g == 0 {
            g = 64;
        }
        let cpad = ic.div_ceil(g) * g;
        cna.weight_kernels = 1;
        cna.weight_bytes_per_kernel = kw * kh * ic; // one int8 filter per channel
        cna.weight_bytes = cpad * kh * kw;
        let wb = (cna.weight_bytes as usize).div_ceil(bank);
        if fd_banks + wb > banks {
            return Err(Error::tiling("gen_conv2d_int8/dw-banks"));
        }
        weight_banks = banks - fd_banks;
    } else {
        cna.weight_kernels = oc;
        cna.weight_bytes_per_kernel = kw * kh * ic;
        cna.weight_bytes = cna.weight_bytes_per_kernel * cna.weight_kernels;
        if cna.weight_bytes_per_kernel as usize <= bank {
            weight_banks = banks - fd_banks;
        } else {
            return Err(Error::tiling("gen_conv2d_int8/weight-bank"));
        }
    }

    cna.weight_bank = weight_banks as u32;
    cna.data_bank = fd_banks as u32;

    // The int8 C2=16 feature-cube DMA over-reads ONE CBUF bank past its ceil
    // allocation at near-bank-full geometries (HW-proven for this conv path, the
    // identical signature `gen_matmul_int8` has). Reserve it; the weight takes the
    // rest. The tiler reserves the matching bank, so `fd_banks + 1 + weight <= 12`.
    if fd_banks + 1 > banks - 1 {
        log_error!(
            "gen_conv2d_int8: int8 CBUF bank-slack does not fit (fd_banks={fd_banks}, \
             need fd_banks+1 <= {}) — use the int8 conv tiler",
            banks - 1
        );
        return Err(Error::tiling("gen_conv2d_int8/bank-slack"));
    }
    {
        let mut data_bank = fd_banks + 1;
        cna.data_bank = data_bank as u32;
        cna.weight_bank = (banks - data_bank) as u32;
        // The audit sentinel, kept for future RE exactly as C (`ROCKET_CONV_I8_FDBANK_EXTRA`):
        // -1 reproduces the old exact fit, +1 over-reserves.
        if let Some(x) = knobs::int("ROCKET_CONV_I8_FDBANK_EXTRA") {
            let mut nb = data_bank as i64 + x;
            if nb < 1 {
                nb = 1;
            }
            if nb > (banks - 1) as i64 {
                nb = (banks - 1) as i64;
            }
            data_bank = nb as usize;
            cna.data_bank = data_bank as u32;
            cna.weight_bank = (banks - data_bank) as u32;
        }
    }
    cna.data_entries = (iw * ic).div_ceil(64);
    if let Some(v) = knobs::u32_auto("ROCKET_CONV_DENTRIES") {
        cna.data_entries = v & 0xFFFF;
    }
    cna.data_sign = 1;
    cna.cvt_type = 1;
    cna.cvt_bypass = 1;
    cna.cvt_scale0 = 1;
    cna.cvt_scale1 = 1;
    cna.cvt_scale2 = 1;
    cna.cvt_scale3 = 1;
    cna.fc_skip_en = 0;
    cna.data_offset = 0;
    cna.pad_left = p.pad_left as u32;
    cna.pad_top = p.pad_top as u32;
    cna.feature_base_addr = p.input_dma;
    cna.weight_offset = 0;
    cna.weight_burst_len = 0xF;
    cna.data_burst_len = 0xF;
    // Same float-correct feature surface stride as the fp16 conv (`IW*(IH-4)`): the cube
    // atom is 16 B for both dtypes, so the stride is element-size independent.
    cna.line_stride = iw * 4;
    let surf = cna.line_stride as i64 * (ih as i64 - 4) / 4;
    cna.surf_stride = if surf < 0 { 0 } else { surf as u32 };
    cna.dma_width = iw;
    cna.dma_height = ih;
    cna.dma_channel = ic;
    cna.decompress_addr0 = p.weights_dma;

    let mut core = Core {
        proc_precision: PRECISION_INT8,
        qd_en: 0,
        dw_en: depthwise as u32,
        dataout_height: oh - 1,
        dataout_width: ow - 1,
        dataout_channel: oc - 1,
    };

    let mut dpu = Dpu {
        burst_len: 0xF,
        conv_mode: if depthwise { 3 } else { DIRECT_CONVOLUTION },
        output_mode: 0x2,
        flying_mode: 0,
        out_precision: PRECISION_INT32,
        in_precision: PRECISION_INT8,
        proc_precision: PRECISION_INT8,
        dst_base_addr: p.output_dma,
        dst_surf_stride: oh * ow,
        width: core.dataout_width,
        height: core.dataout_height,
        channel: core.dataout_channel,
        bs_bypass: 1,
        bs_alu_bypass: 1,
        bs_mul_bypass: 1,
        bs_relu_bypass: 1,
        bn_bypass: 1,
        bn_alu_bypass: 1,
        bn_mul_bypass: 1,
        bn_relu_bypass: 1,
        ew_bypass: 1,
        ew_op_bypass: 1,
        ew_lut_bypass: 1,
        ew_op_cvt_bypass: 1,
        ew_relu_bypass: 1,
        fp32tofp16_en: 0,
        out_cvt_scale: 1,
        ..Default::default()
    };
    // The int32-output stride quirk (== gen_matmul_int8): size_e = 7, surf_add =
    // stride * 8. 3/4 (the float-native 4-byte pattern) strides the int32 wrong.
    {
        let mut se = 7u32;
        let mut sm = 8u32;
        if let Some(v) = knobs::u32_auto("ROCKET_CONV_INT8_SIZE_E") {
            se = v;
        }
        if let Some(v) = knobs::u32_auto("ROCKET_CONV_INT8_SURF_MULT") {
            sm = v;
        }
        dpu.size_e_2 = se & 0x7;
        dpu.size_e_1 = se & 0x7;
        dpu.size_e_0 = se & 0x7;
        dpu.surf_add = dpu.dst_surf_stride * sm;
    }
    dpu.od_bypass = 1;
    dpu.width_wdma = core.dataout_width;
    dpu.height_wdma = core.dataout_height;
    dpu.channel_wdma = core.dataout_channel;

    if depthwise {
        // Depthwise output geometry: the universal DW fields, then the int8-out writer.
        dpu.od_bypass = 0;
        if !knobs::present("ROCKET_CONV_GRAINS") {
            cna.feature_grains = 50 + p.stride_y as u32 + 1;
        }
        dpu.bs_ow_op = 0x80u32.wrapping_sub(p.weight_zero_point as u32);

        if p.int8_out {
            // Mesa's DW int8-output regcmd: QD_EN required, int8 border pad, int8 out.
            core.qd_en = 1;
            // The CNA border pad: the zero point in the uint8-centred domain, one BYTE
            // per lane (the odd channels of a group read byte 1 of the word).
            let pb = ((p.input_zero_point & 0xFF) - 0x80) & 0xFF;
            cna.pad_con1 = (pb as u32) * 0x0101_0101;
            dpu.out_precision = PRECISION_INT8;
            // int8-out writer stride: size_e = 3, surf_add = stride * 4 (NOT 7/8).
            dpu.size_e_2 = 3;
            dpu.size_e_1 = 3;
            dpu.size_e_0 = 3;
            dpu.surf_add = dpu.dst_surf_stride * 4;

            // The OUT_CVT requant triple, derived exactly as Mesa (QNNPACK
            // requantization.h): conv_scale bits -> shift 126 - exp + 16 - 1, an odd
            // 15-bit mantissa forced into [2^14, 2^15), offset = out_zp - 0x80.
            let conv_scale = (p.in_scale * p.w_scale) / p.out_scale;
            let bits = conv_scale.to_bits();
            // C unsigned arithmetic: an exponent above 126 wraps, exactly as it does
            // in the reference emitter.
            let shift = (127u32 + 31 - 32).wrapping_sub(bits >> 23).wrapping_add(16);
            let mut scale = ((bits >> 9) & 0x7FFF) + 1;
            if scale < (1 << 14) {
                scale |= 1 << 14;
            }
            dpu.out_cvt_scale = scale & 0xFFFF;
            dpu.out_cvt_shift = shift.wrapping_sub(1);
            dpu.out_cvt_offset = (p.output_zero_point - 0x80) as u32;

            // The per-OC int32 bias add in the BS ALU, fetched by BRDMA.
            dpu.bias_en = true;
            dpu.bias_base_addr = p.bias_dma;
        }

        if let Some(v) = knobs::u32_auto("ROCKET_CONV_DW_SIZEE") {
            let se = v & 0x7;
            dpu.size_e_2 = se;
            dpu.size_e_1 = se;
            dpu.size_e_0 = se;
        }
        if let Some(v) = knobs::u32_auto("ROCKET_CONV_DW_SURF_MULT") {
            dpu.surf_add = dpu.dst_surf_stride * v;
        }
        if let Some(v) = knobs::u32_auto("ROCKET_CONV_DW_BSOWOP") {
            dpu.bs_ow_op = v;
        }
    }

    let start = ops.len();
    emit_task(ops, &cna, &core, &dpu, None)?;
    let count = ops.len() - start;
    p.task_count = count as u32;
    Ok(count)
}

fn emit_task(
    ops: &mut Vec<NpuOp>,
    cna: &Cna,
    core: &Core,
    dpu: &Dpu,
    act: Option<&ConvAct<'_>>,
) -> Result<()> {
    // 10-bit CNA field (the C refuses unconditionally; `-DNDEBUG` would strip an
    // assert, which is why this is a returned error).
    if cna.feature_grains > 0x3FF {
        log_error!(
            "npu_regcmd: feature_grains {} exceeds 10-bit CNA field (max 1023)",
            cna.feature_grains
        );
        return Err(Error::tiling("gen_conv2d_fp16/grains"));
    }
    let mut push = |target, reg, value| ops.push(NpuOp::new(target, reg, value));

    push(DPU, DPU_S_POINTER, 0xE);
    push(DPU_RDMA, DPU_RDMA_S_POINTER, 0xE);

    // The deconvolution mode: CONV_CON1 bit 16 DECONV and CONV_CON3 [13:11]/[10:8]
    // holding stride-1. The C's raw diagnostic overrides keep their semantics.
    {
        let mut ds = cna.deconv as u32;
        let mut fx = cna.deconv_x;
        let mut fy = cna.deconv_y;
        if let Some(v) = knobs::u32_auto("ROCKET_CNA_DECONV") {
            ds = v;
        }
        if let Some(v) = knobs::u32_auto("ROCKET_CNA_DECONV_X") {
            fx = v;
        }
        if let Some(v) = knobs::u32_auto("ROCKET_CNA_DECONV_Y") {
            fy = v;
        }
        let mut value = ((cna.proc_precision & 0x7) << 7)
            | ((cna.in_precision & 0x7) << 4)
            | (cna.conv_mode & 0xF);
        if ds != 0 {
            value |= 1 << 16;
        }
        push(CNA, CNA_CONV_CON1, value);
        let value = ((cna.kernel_groups & 0xFF) << 16) | ((cna.feature_grains & 0x3FF) << 4);
        push(CNA, CNA_CONV_CON2, value);
        let mut value = ((cna.atrous_y & 0x1F) << 21)
            | ((cna.atrous_x & 0x1F) << 16)
            | ((cna.conv_y_stride & 0x7) << 3)
            | (cna.conv_x_stride & 0x7);
        value |= ((fy & 0x7) << 11) | ((fx & 0x7) << 8);
        push(CNA, CNA_CONV_CON3, value);
    }
    push(
        CNA,
        CNA_DATA_SIZE0,
        ((cna.datain_width & 0x7FF) << 16) | (cna.datain_height & 0x7FF),
    );
    push(
        CNA,
        CNA_DATA_SIZE1,
        ((cna.datain_channel.wrapping_sub(1) & 0xFFFF) << 16) | (cna.datain_channel & 0xFFFF),
    );
    push(CNA, CNA_DATA_SIZE2, cna.dataout_width & 0x7FF);
    push(CNA, CNA_DATA_SIZE3, cna.dataout_atomics & 0x3FFFF);
    push(CNA, CNA_WEIGHT_SIZE0, cna.weight_bytes);
    push(CNA, CNA_WEIGHT_SIZE1, cna.weight_bytes_per_kernel & 0x7FFFF);
    push(
        CNA,
        CNA_WEIGHT_SIZE2,
        ((cna.weight_width & 0x1F) << 24)
            | ((cna.weight_height & 0x1F) << 16)
            | (cna.weight_kernels & 0x3FFF),
    );
    push(
        CNA,
        CNA_CBUF_CON0,
        ((cna.weight_bank & 0xF) << 4) | (cna.data_bank & 0xF),
    );
    push(CNA, CNA_CBUF_CON1, cna.data_entries & 0x1FFF);
    push(
        CNA,
        CNA_CVT_CON0,
        ((cna.data_sign & 1) << 3) | ((cna.cvt_type & 1) << 1) | (cna.cvt_bypass & 1),
    );
    push(CNA, CNA_CVT_CON1, (cna.cvt_scale0 & 0xFFFF) << 16);
    push(CNA, CNA_CVT_CON2, (cna.cvt_scale1 & 0xFFFF) << 16);
    push(CNA, CNA_CVT_CON3, (cna.cvt_scale2 & 0xFFFF) << 16);
    push(CNA, CNA_CVT_CON4, (cna.cvt_scale3 & 0xFFFF) << 16);
    push(CNA, CNA_FC_CON0, cna.fc_skip_en & 1);
    push(CNA, CNA_FC_CON1, cna.data_offset & 0x1FFFF);
    push(
        CNA,
        CNA_PAD_CON0,
        ((cna.pad_left & 0xF) << 4) | (cna.pad_top & 0xF),
    );
    push(CNA, CNA_FEATURE_DATA_ADDR, cna.feature_base_addr);
    push(CNA, CNA_FC_CON2, cna.weight_offset & 0x1FFFF);
    push(
        CNA,
        CNA_DMA_CON0,
        ((cna.weight_burst_len & 0xF) << 16) | (cna.data_burst_len & 0xF),
    );
    push(CNA, CNA_DMA_CON1, cna.line_stride & 0x0FFF_FFFF);
    push(CNA, CNA_DMA_CON2, cna.surf_stride & 0x0FFF_FFFF);
    push(
        CNA,
        CNA_FC_DATA_SIZE0,
        ((cna.dma_width & 0x7FF) << 16) | (cna.dma_height & 0x7FF),
    );
    push(CNA, CNA_FC_DATA_SIZE1, cna.dma_channel & 0xFFFF);
    push(CNA, CNA_DCOMP_CTRL, 0);
    push(CNA, CNA_DCOMP_REGNUM, 0);
    push(CNA, CNA_DCOMP_ADDR0, cna.decompress_addr0);
    for reg in [
        CNA_DCOMP_AMOUNT,
        0x1144,
        0x1148,
        0x114C,
        0x1150,
        0x1154,
        0x1158,
        0x115C,
        0x1160,
        0x1164,
        0x1168,
        0x116C,
        0x1170,
        0x1174,
        0x1178,
        0x117C,
    ] {
        push(CNA, reg, 0);
    }
    push(CNA, CNA_CVT_CON5, 0);
    push(CNA, CNA_PAD_CON1, cna.pad_con1);

    // CORE_MISC_CFG carries DW_EN for depthwise (the conv delta vs the matmul emitter).
    let value = ((core.proc_precision & 0x7) << 8) | ((core.dw_en & 1) << 1) | (core.qd_en & 1);
    push(CORE, CORE_MISC_CFG, value);
    push(
        CORE,
        CORE_DATAOUT_SIZE_0,
        ((core.dataout_height & 0xFFFF) << 16) | (core.dataout_width & 0xFFFF),
    );
    push(CORE, CORE_DATAOUT_SIZE_1, core.dataout_channel & 0xFFFF);
    push(CORE, CORE_CLIP_TRUNCATE, 0);
    push(CORE, CORE_3030, 0);

    push(
        DPU,
        DPU_FEATURE_MODE_CFG,
        ((dpu.burst_len & 0xF) << 5)
            | ((dpu.conv_mode & 0x3) << 3)
            | ((dpu.output_mode & 0x3) << 1)
            | (dpu.flying_mode & 1),
    );
    push(
        DPU,
        DPU_DATA_FORMAT,
        ((dpu.out_precision & 0x7) << 29)
            | ((dpu.in_precision & 0x7) << 26)
            | (dpu.proc_precision & 0x7),
    );
    push(DPU, DPU_OFFSET_PEND, 0);
    push(DPU, DPU_DST_BASE_ADD, dpu.dst_base_addr);
    push(
        DPU,
        DPU_DST_SURF_STRIDE,
        (dpu.dst_surf_stride & 0x0FFF_FFFF) << 4,
    );
    push(DPU, DPU_DATA_CUBE_WIDTH, dpu.width & 0x1FFF);
    push(DPU, DPU_DATA_CUBE_HEIGHT, dpu.height & 0x1FFF);
    push(DPU, DPU_DATA_CUBE_NOTCH_ADDR, 0);
    push(
        DPU,
        DPU_DATA_CUBE_CHANNEL,
        ((dpu.channel & 0x1FFF) << 16) | (dpu.channel & 0x1FFF),
    );
    push(
        DPU,
        DPU_BS_CFG,
        if dpu.bias_en {
            // The BS ALU add mode reading the per-OC int32 bias from BRDMA (Mesa's
            // int8 word: BS_ALU_ALGO(2) | BS_ALU_SRC(1) | RELU/MUL bypass).
            (2 << 16) | (1 << 8) | (1 << 6) | (1 << 4)
        } else {
            ((dpu.bs_relu_bypass & 1) << 6)
                | ((dpu.bs_mul_bypass & 1) << 4)
                | ((dpu.bs_alu_bypass & 1) << 1)
                | (dpu.bs_bypass & 1)
        },
    );
    push(DPU, DPU_BS_ALU_CFG, 0);
    push(DPU, DPU_BS_MUL_CFG, 0);
    push(DPU, DPU_BS_RELUX_CMP_VALUE, 0);
    push(
        DPU,
        DPU_BS_OW_CFG,
        ((dpu.size_e_2 & 0x7) << 8)
            | ((dpu.size_e_1 & 0x7) << 5)
            | ((dpu.size_e_0 & 0x7) << 2)
            | ((dpu.od_bypass & 1) << 1),
    );
    push(DPU, DPU_BS_OW_OP, dpu.bs_ow_op & 0xFFFF);
    // `tp_precision` / `size_c_wdma` stay 0 on the fp16 path, so only the channel
    // field is live (the C word is `channel_wdma & 0x1FFF`).
    push(DPU, DPU_WDMA_SIZE_0, dpu.channel_wdma & 0x1FFF);
    push(
        DPU,
        DPU_WDMA_SIZE_1,
        ((dpu.height_wdma & 0x1FFF) << 16) | (dpu.width_wdma & 0x1FFF),
    );
    if let Some(act) = act {
        // The conv result rides the BN index scale into the LUT grid exactly as the
        // standalone flying-mode activation programs it.
        push(DPU, DPU_BN_CFG, (2 << 16) | (1 << 6));
        push(DPU, DPU_BN_ALU_CFG, act.ep.bn_alu_cfg);
        push(
            DPU,
            DPU_BN_MUL_CFG,
            ((act.ep.bn_mul_operand as u32) & 0xFFFF) << 16,
        );
    } else {
        push(
            DPU,
            DPU_BN_CFG,
            ((dpu.bn_relu_bypass & 1) << 6)
                | ((dpu.bn_mul_bypass & 1) << 4)
                | ((dpu.bn_alu_bypass & 1) << 1)
                | (dpu.bn_bypass & 1),
        );
        push(DPU, DPU_BN_ALU_CFG, 0);
        push(DPU, DPU_BN_MUL_CFG, 0);
    }
    push(DPU, DPU_BN_RELUX_CMP_VALUE, 0);
    if act.is_some() {
        push(DPU, DPU_EW_CFG, (1 << 9) | (1 << 8));
    } else {
        push(
            DPU,
            DPU_EW_CFG,
            ((dpu.ew_relu_bypass & 1) << 9)
                | ((dpu.ew_op_cvt_bypass & 1) << 8)
                | ((dpu.ew_lut_bypass & 1) << 7)
                | ((dpu.ew_op_bypass & 1) << 1)
                | (dpu.ew_bypass & 1),
        );
    }
    push(DPU, DPU_EW_CVT_OFFSET_VALUE, 0);
    push(DPU, DPU_EW_CVT_SCALE_VALUE, 1);
    push(DPU, DPU_EW_RELUX_CMP_VALUE, 0);
    if let Some(act) = act {
        let ep = act.ep;
        push(DPU, DPU_OUT_CVT_OFFSET, ep.out_cvt_offset);
        push(
            DPU,
            DPU_OUT_CVT_SCALE,
            (1 << 16) | ((ep.out_cvt_scale as u32) & 0xFFFF),
        );
        push(
            DPU,
            DPU_OUT_CVT_SHIFT,
            (((ep.out_cvt_cvt_type as u32) & 1) << 31)
                | (((ep.out_cvt_minus_exp as u32) & 0xFF) << 12),
        );
    } else {
        push(DPU, DPU_OUT_CVT_OFFSET, dpu.out_cvt_offset);
        push(
            DPU,
            DPU_OUT_CVT_SCALE,
            ((dpu.fp32tofp16_en & 1) << 16) | (dpu.out_cvt_scale & 0xFFFF),
        );
        let mut round = (dpu.out_cvt_round & 1) as u32;
        if let Some(v) = knobs::u32_auto("ROCKET_OUT_CVT_ROUND") {
            round = v & 1;
        }
        push(
            DPU,
            DPU_OUT_CVT_SHIFT,
            (round << 30) | (dpu.out_cvt_shift & 0x3F),
        );
    }
    for i in 0..8 {
        push(DPU, DPU_EW_OP_VALUE_0 + i * 4, 0);
    }
    push(DPU, DPU_SURFACE_ADD, (dpu.surf_add & 0x0FFF_FFFF) << 4);
    push(DPU, DPU_40C4, 0);
    if let Some(act) = act {
        // The `emit_lut_tables` helper: two `LUT_ACCESS_CFG` bursts of 513 words.
        push(DPU, DPU_LUT_ACCESS_CFG, 1 << 17);
        for &word in &act.lut[..=512] {
            push(DPU, DPU_LUT_ACCESS_DATA, word as u32);
        }
        push(DPU, DPU_LUT_ACCESS_CFG, (1 << 17) | (1 << 16));
        for &word in &act.lut[513..=1025] {
            push(DPU, DPU_LUT_ACCESS_DATA, word as u32);
        }
        let ep = act.ep;
        push(DPU, DPU_LUT_CFG, (1 << 6) | (1 << 5) | (2 << 2));
        push(
            DPU,
            DPU_LUT_INFO,
            (((ep.lo_index_select as u32) & 0xFF) << 16)
                | (((ep.le_index_select as u32) & 0xFF) << 8),
        );
        push(DPU, DPU_LUT_LE_START, ep.lut_le_start);
        push(DPU, DPU_LUT_LE_END, 0);
        push(DPU, DPU_LUT_LO_START, 0);
        push(DPU, DPU_LUT_LO_END, ep.lut_lo_end);
        push(
            DPU,
            DPU_LUT_LE_SLOPE_SCALE,
            (ep.le_slope_scale as u32) & 0xFFFF,
        );
        push(
            DPU,
            DPU_LUT_LE_SLOPE_SHIFT,
            (ep.le_slope_shift as u32) & 0x1F,
        );
        push(
            DPU,
            DPU_LUT_LO_SLOPE_SCALE,
            (ep.lo_slope_scale as u32) & 0xFFFF,
        );
        push(
            DPU,
            DPU_LUT_LO_SLOPE_SHIFT,
            (ep.lo_slope_shift as u32) & 0x1F,
        );
    } else {
        for reg in [
            DPU_LUT_ACCESS_CFG,
            DPU_LUT_ACCESS_DATA,
            DPU_LUT_CFG,
            DPU_LUT_INFO,
            DPU_LUT_LE_START,
            DPU_LUT_LE_END,
            DPU_LUT_LO_START,
            DPU_LUT_LO_END,
            DPU_LUT_LE_SLOPE_SCALE,
            DPU_LUT_LE_SLOPE_SHIFT,
            DPU_LUT_LO_SLOPE_SCALE,
            DPU_LUT_LO_SLOPE_SHIFT,
        ] {
            push(DPU, reg, 0);
        }
    }

    // DPU_RDMA: plain conv (no eltwise feed) — MRDMA + ERDMA disabled.
    push(DPU_RDMA, DPU_RDMA_DATA_CUBE_WIDTH, dpu.width & 0x1FFF);
    push(DPU_RDMA, DPU_RDMA_DATA_CUBE_HEIGHT, dpu.height & 0x1FFF);
    push(DPU_RDMA, DPU_RDMA_DATA_CUBE_CHANNEL, dpu.channel & 0x1FFF);
    push(DPU_RDMA, DPU_RDMA_SRC_BASE_ADDR, 0);
    push(
        DPU_RDMA,
        DPU_RDMA_BRDMA_CFG,
        if dpu.bias_en { 1 << 1 } else { 0 },
    );
    push(
        DPU_RDMA,
        DPU_RDMA_BS_BASE_ADDR,
        if dpu.bias_en { dpu.bias_base_addr } else { 0 },
    );
    push(DPU_RDMA, DPU_RDMA_NRDMA_CFG, 0);
    push(DPU_RDMA, DPU_RDMA_BN_BASE_ADDR, 0);
    push(DPU_RDMA, DPU_RDMA_ERDMA_CFG, RDMA_ERDMA_DISABLE);
    push(DPU_RDMA, DPU_RDMA_EW_BASE_ADDR, 0);
    push(DPU_RDMA, DPU_RDMA_EW_SURF_STRIDE, 0);
    let mut fmc = rdma_fmc_burst_len(15)
        | rdma_fmc_mrdma_disable(1)
        | if dpu.fp32tofp16_en != 0 {
            rdma_fmc_mrdma_fp16tofp32(1)
        } else {
            0
        };
    fmc |= (dpu.conv_mode & 0x3) << 1;
    push(DPU_RDMA, DPU_RDMA_FEATURE_MODE_CFG, fmc);
    push(DPU_RDMA, DPU_RDMA_SRC_DMA_CFG, 0);
    push(DPU_RDMA, DPU_RDMA_SURF_NOTCH, 0);
    push(DPU_RDMA, DPU_RDMA_PAD_CFG, 0);
    push(DPU_RDMA, DPU_RDMA_WEIGHT, RDMA_WEIGHT_ALL1);
    push(DPU_RDMA, DPU_RDMA_EW_SURF_NOTCH, 0);

    drop(push);
    ops.push(NpuOp(0));
    ops.push(NpuOp::new(PC, PC_REGISTER_AMOUNTS, 0));
    ops.push(NpuOp::new(OP_40, 0, 0));
    ops.push(NpuOp::new(OP_ENABLE, PC_OPERATION_ENABLE, 0x1D));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params<'a>(ic: u16, ih: u16, iw: u16, oc: u16, kh: u16, kw: u16) -> ConvParams<'a> {
        let oh = (ih as usize + 2 - kh as usize) + 1;
        let ow = (iw as usize + 2 - kw as usize) + 1;
        let mut p = ConvParams::new(
            ic, ih, iw, oc, oh as u16, ow as u16, kh, kw, 0x1000, 0x2000, 0x3000,
        );
        p.pad_top = 1;
        p.pad_left = 1;
        p
    }

    #[test]
    fn emits_a_plausible_direct_program() {
        let mut p = params(32, 8, 8, 16, 3, 3);
        let mut ops = Vec::new();
        let count = gen_conv2d_fp16(&mut p, &mut ops).unwrap();
        assert_eq!(count as u32, p.task_count);
        assert!(count > 90 && count <= MAX_OPS, "count {count}");
        assert_eq!(ops[0], NpuOp::new(DPU, DPU_S_POINTER, 0xE));
        assert_eq!(ops[1], NpuOp::new(DPU_RDMA, DPU_RDMA_S_POINTER, 0xE));
        // The kernel geometry lands in the weight-size register, not in a weight count.
        let w2 = ops
            .iter()
            .find(|op| op.target() == CNA && op.reg() == CNA_WEIGHT_SIZE2)
            .unwrap();
        assert_eq!(w2.value(), (3 << 24) | (3 << 16) | 16);
        // Pad registers are pad_left[7:4] | pad_top[3:0].
        let pad = ops
            .iter()
            .find(|op| op.target() == CNA && op.reg() == CNA_PAD_CON0)
            .unwrap();
        assert_eq!(pad.value(), (1 << 4) | 1);
        // Plain conv: no dilation fields, no deconv bit.
        let c3 = ops
            .iter()
            .find(|op| op.target() == CNA && op.reg() == CNA_CONV_CON3)
            .unwrap();
        assert_eq!(c3.value(), (1 << 3) | 1);
        let c1 = ops
            .iter()
            .find(|op| op.target() == CNA && op.reg() == CNA_CONV_CON1)
            .unwrap();
        assert_eq!(c1.value() & (1 << 16), 0);
        // DPU_RDMA stays visible: MRDMA disabled or the job hangs.
        let fmc = ops
            .iter()
            .find(|op| op.target() == DPU_RDMA && op.reg() == DPU_RDMA_FEATURE_MODE_CFG)
            .unwrap();
        assert_ne!(fmc.value() & (1 << 4), 0);
        assert_eq!(
            ops[count - 1],
            NpuOp::new(OP_ENABLE, PC_OPERATION_ENABLE, 0x1D)
        );
    }

    #[test]
    fn depthwise_sets_dw_en_and_the_output_writer_fields() {
        let mut p = ConvParams::new(32, 8, 8, 32, 8, 8, 3, 3, 0x1000, 0x2000, 0x3000);
        p.pad_top = 1;
        p.pad_left = 1;
        let mut ops = Vec::new();
        gen_conv2d_dw_fp16(&mut p, &mut ops).unwrap();
        let misc = ops
            .iter()
            .find(|op| op.target() == CORE && op.reg() == CORE_MISC_CFG)
            .unwrap();
        assert_ne!(misc.value() & (1 << 1), 0, "DW_EN");
        let mode = ops
            .iter()
            .find(|op| op.target() == DPU && op.reg() == DPU_FEATURE_MODE_CFG)
            .unwrap();
        assert_eq!((mode.value() >> 3) & 0x3, 3, "conv_mode 3");
        let ow = ops
            .iter()
            .find(|op| op.target() == DPU && op.reg() == DPU_BS_OW_CFG)
            .unwrap();
        assert_eq!((ow.value() >> 2) & 0x7, 3, "size_e 3");
        assert_eq!((ow.value() >> 1) & 1, 0, "od_bypass dropped");
        let owop = ops
            .iter()
            .find(|op| op.target() == DPU && op.reg() == DPU_BS_OW_OP)
            .unwrap();
        assert_eq!(owop.value(), 0x80);
        // One weight kernel, G-padded cube bytes.
        let w2 = ops
            .iter()
            .find(|op| op.target() == CNA && op.reg() == CNA_WEIGHT_SIZE2)
            .unwrap();
        assert_eq!(w2.value() & 0x3FFF, 1);
        let w0 = ops
            .iter()
            .find(|op| op.target() == CNA && op.reg() == CNA_WEIGHT_SIZE0)
            .unwrap();
        assert_eq!(w0.value(), 32 * 3 * 3 * 2);
    }

    #[test]
    fn refuses_over_budget_tiles() {
        // A kernel bigger than one CBUF bank (32768 B): 3x3x6000 fp16.
        let mut p = ConvParams::new(6000, 1, 1, 16, 1, 1, 3, 3, 0, 0, 0);
        let mut ops = Vec::new();
        assert_eq!(
            gen_conv2d_fp16(&mut p, &mut ops).unwrap_err().status(),
            crate::Status::Tiling
        );
        // The feature alone cannot exceed 11 banks.
        let mut p = ConvParams::new(32, 400, 400, 16, 1, 1, 1, 1, 0, 0, 0);
        let mut ops = Vec::new();
        assert_eq!(
            gen_conv2d_fp16(&mut p, &mut ops).unwrap_err().status(),
            crate::Status::Tiling
        );
    }

    #[test]
    fn deconv_mode_contract() {
        // Direct, stride 1, power-of-two interior dilation: accepted.
        let mut p = ConvParams::new(32, 4, 4, 16, 8, 8, 3, 3, 0x1000, 0x2000, 0x3000);
        p.deconv_sy = 2;
        p.deconv_sx = 2;
        let mut ops = Vec::new();
        gen_conv2d_fp16(&mut p, &mut ops).unwrap();
        let c1 = ops
            .iter()
            .find(|op| op.target() == CNA && op.reg() == CNA_CONV_CON1)
            .unwrap();
        assert_ne!(c1.value() & (1 << 16), 0, "DECONV bit");
        let c3 = ops
            .iter()
            .find(|op| op.target() == CNA && op.reg() == CNA_CONV_CON3)
            .unwrap();
        assert_eq!((c3.value() >> 11) & 0x7, 1, "deconv y stride - 1");
        assert_eq!((c3.value() >> 8) & 0x7, 1, "deconv x stride - 1");
        // A stride-2 forward conv cannot also be a deconvolution.
        let mut p = ConvParams::new(32, 4, 4, 16, 8, 8, 3, 3, 0x1000, 0x2000, 0x3000);
        p.deconv_sy = 2;
        p.stride_y = 2;
        let mut ops = Vec::new();
        assert_eq!(
            gen_conv2d_fp16(&mut p, &mut ops).unwrap_err().status(),
            crate::Status::Shape
        );
        // Nor can a depthwise one.
        let mut p = ConvParams::new(32, 4, 4, 32, 8, 8, 3, 3, 0x1000, 0x2000, 0x3000);
        p.deconv_sx = 2;
        let mut ops = Vec::new();
        assert_eq!(
            gen_conv2d_dw_fp16(&mut p, &mut ops).unwrap_err().status(),
            crate::Status::Shape
        );
    }

    #[test]
    fn fused_activation_uploads_the_tables() {
        use crate::ops::activation::{Activation, LutTable, lut_epilogue_build};
        let _g = crate::knobs::TEST_ENV
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut lut = LutTable::zeroed();
        let ep = lut_epilogue_build(Activation::Silu, &mut lut).unwrap();
        let mut p = params(32, 8, 8, 16, 3, 3);
        p.act = Some(ConvAct {
            lut: lut.words(),
            ep: &ep,
        });
        let mut ops = Vec::new();
        let count = gen_conv2d_fp16(&mut p, &mut ops).unwrap();
        assert!(count <= MAX_OPS_ACT, "count {count}");
        // Two table uploads of 513 data words, and the LUT registers at the tail.
        let data = ops
            .iter()
            .filter(|op| op.target() == DPU && op.reg() == DPU_LUT_ACCESS_DATA)
            .count();
        assert_eq!(data, 1026);
        let cfg = ops
            .iter()
            .find(|op| op.target() == DPU && op.reg() == DPU_LUT_CFG)
            .unwrap();
        assert_eq!(cfg.value(), (1 << 6) | (1 << 5) | (2 << 2));
        let bn = ops
            .iter()
            .find(|op| op.target() == DPU && op.reg() == DPU_BN_CFG)
            .unwrap();
        assert_eq!(bn.value(), (2 << 16) | (1 << 6));
    }

    fn int8_params(ic: u16, ih: u16, iw: u16, oc: u16, kh: u16, kw: u16) -> ConvInt8Params {
        let oh = (ih as usize + 2 - kh as usize) + 1;
        let ow = (iw as usize + 2 - kw as usize) + 1;
        let mut p = ConvInt8Params::new(
            ic, ih, iw, oc, oh as u16, ow as u16, kh, kw, 0x1000, 0x2000, 0x3000,
        );
        p.pad_top = 1;
        p.pad_left = 1;
        p
    }

    fn word(ops: &[NpuOp], target: u64, reg: u32) -> u32 {
        ops.iter()
            .find(|op| op.target() == target && op.reg() == reg)
            .unwrap_or_else(|| panic!("missing {target:#x}/{reg:#x}"))
            .value()
    }

    /// The direct int8 deltas: 1 B/element, the +1 feature slack bank, `C2 = 16`-
    /// independent `data_entries = ceil(IW*IC/64)`, and the int32 output writer.
    #[test]
    fn int8_direct_packs_the_slack_bank_and_int32_writer() {
        let mut p = int8_params(32, 8, 8, 16, 3, 3);
        let mut ops = Vec::new();
        let count = gen_conv2d_int8(&mut p, &mut ops).unwrap();
        assert_eq!(count as u32, p.task_count);
        assert!(count <= MAX_OPS, "count {count}");

        // 8*8*32 = 2048 B -> 1 bank; the feature takes fd_banks + 1 and the weight the
        // remaining 10 of the 12 CBUF banks.
        assert_eq!(word(&ops, CNA, CNA_CBUF_CON0), (10 << 4) | 2);
        assert_eq!(word(&ops, CNA, CNA_CBUF_CON1), 4); // ceil(8*32/64)
        // int8 precisions on the CNA/CORE/DPU; the output is a raw int32 accumulator.
        assert_eq!(word(&ops, CNA, CNA_CONV_CON1) & 0x7F, 0);
        assert_eq!(word(&ops, CORE, CORE_MISC_CFG) & 0x700, 0);
        // OUT_PRECISION = int32 (4), IN/PROC = int8 (0).
        assert_eq!(word(&ops, DPU, DPU_DATA_FORMAT), 4 << 29);
        // size_e = 7 and surf_add = OH*OW*8 (the int32 stride quirk).
        assert_eq!((word(&ops, DPU, DPU_BS_OW_CFG) >> 2) & 0x7, 7);
        assert_eq!((word(&ops, DPU, DPU_BS_OW_CFG) >> 1) & 1, 1, "od_bypass");
        assert_eq!(word(&ops, DPU, DPU_SURFACE_ADD), (8 * 8 * 8) << 4);
        // The BS block stays bypassed and BRDMA is not armed on the int32 path.
        assert_eq!(
            word(&ops, DPU, DPU_BS_CFG),
            (1 << 6) | (1 << 4) | (1 << 1) | 1
        );
        assert_eq!(word(&ops, DPU_RDMA, DPU_RDMA_BRDMA_CFG), 0);
        assert_eq!(word(&ops, DPU_RDMA, DPU_RDMA_BS_BASE_ADDR), 0);
        // 1-byte weight cube: KW*KH*IC per kernel.
        assert_eq!(word(&ops, CNA, CNA_WEIGHT_SIZE1), 3 * 3 * 32);
        assert_eq!(word(&ops, CNA, CNA_WEIGHT_SIZE0), 3 * 3 * 32 * 16);
        // The trailer is the plain conv one.
        assert_eq!(
            ops[count - 1],
            NpuOp::new(OP_ENABLE, PC_OPERATION_ENABLE, 0x1D)
        );
    }

    /// The int8-out depthwise writer: QD_EN, the byte-replicated border pad, the int32
    /// bias in the BS ALU (BRDMA armed) and the three OUT_CVT requant fields.
    #[test]
    fn int8_depthwise_packs_the_requant_registers() {
        let mut p = int8_params(64, 8, 8, 64, 3, 3);
        p.int8_out = true;
        p.in_scale = 0.0213;
        p.w_scale = 0.00347;
        p.out_scale = 0.0925;
        p.input_zero_point = -2 + 0x80;
        p.output_zero_point = 5 + 0x80;
        p.weight_zero_point = 0x80; // raw w_zp 0 -> the uint8-centred 0x80
        p.bias_dma = 0x4000;
        let mut ops = Vec::new();
        gen_conv2d_dw_int8(&mut p, &mut ops).unwrap();

        assert_eq!(word(&ops, CORE, CORE_MISC_CFG) & 1, 1, "QD_EN");
        assert_eq!((word(&ops, CORE, CORE_MISC_CFG) >> 1) & 1, 1, "DW_EN");
        // The pad byte is (in_zp & 0xff) - 0x80, repeated in all four lanes: the
        // signed model zero point -2 as a byte (0xFE).
        assert_eq!(word(&ops, CNA, CNA_PAD_CON1), 0xFEFE_FEFE);
        // int8 out/proc precision, the int8 writer stride and od_bypass dropped.
        assert_eq!(word(&ops, DPU, DPU_DATA_FORMAT), 0);
        assert_eq!((word(&ops, DPU, DPU_BS_OW_CFG) >> 2) & 0x7, 3);
        assert_eq!((word(&ops, DPU, DPU_BS_OW_CFG) >> 1) & 1, 0);
        assert_eq!(word(&ops, DPU, DPU_SURFACE_ADD), (8 * 8 * 4) << 4);
        // 0x80 - the uint8-centred weight zero point (0x80 for a symmetric int8 weight
        // set), i.e. 0 — the CPEND operand the C notes must be 0.
        assert_eq!(word(&ops, DPU, DPU_BS_OW_OP), 0);
        // The BS ALU add + BRDMA bias fetch.
        let bs = word(&ops, DPU, DPU_BS_CFG);
        assert_eq!(bs, (2 << 16) | (1 << 8) | (1 << 6) | (1 << 4));
        assert_eq!(word(&ops, DPU_RDMA, DPU_RDMA_BRDMA_CFG), 1 << 1);
        assert_eq!(word(&ops, DPU_RDMA, DPU_RDMA_BS_BASE_ADDR), 0x4000);
        // The hand-derived OUT_CVT triple (conv_scale 0.0213*0.00347/0.0925).
        assert_eq!(word(&ops, DPU, DPU_OUT_CVT_SCALE), 0x68bc);
        assert_eq!(word(&ops, DPU, DPU_OUT_CVT_SHIFT), 25);
        assert_eq!(word(&ops, DPU, DPU_OUT_CVT_OFFSET), 5);
        // feature_grains = 50 + stride_y + 1 for the DW path.
        assert_eq!((word(&ops, CNA, CNA_CONV_CON2) >> 4) & 0x3FF, 52);
        // G-padded weight cube bytes: 64*3*3.
        assert_eq!(word(&ops, CNA, CNA_WEIGHT_SIZE0), 64 * 3 * 3);
    }

    /// The C generator's own refusals: a feature above 11 banks, a kernel above one bank.
    #[test]
    fn int8_refuses_over_budget_tiles() {
        let mut p = int8_params(32, 400, 400, 16, 1, 1);
        p.pad_top = 0;
        p.pad_left = 0;
        let mut ops = Vec::new();
        assert_eq!(
            gen_conv2d_int8(&mut p, &mut ops).unwrap_err().status(),
            crate::Status::Tiling
        );
        let mut p = int8_params(6000, 1, 1, 16, 3, 3);
        p.pad_top = 0;
        p.pad_left = 0;
        let mut ops = Vec::new();
        assert_eq!(
            gen_conv2d_int8(&mut p, &mut ops).unwrap_err().status(),
            crate::Status::Tiling
        );
    }
}
