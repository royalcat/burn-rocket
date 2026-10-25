//! fp16 matmul register-command emission (RK3588 1x1-conv geometry).
//!
//! Direct translation of `gen_matmul_fp16` + the shared `gen_matmul_task` from
//! `npu_regcmd.c`. One call emits the complete program for one tile; the caller
//! appends it to a regcmd buffer and submits it as one task.

use crate::error::{Error, Result};
use crate::knobs;
use crate::log::log_error;

use super::desc::*;

/// Parameters of one fp16 matmul tile.
#[derive(Debug, Clone)]
pub struct MatmulParams {
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub input_dma: u32,
    pub weights_dma: u32,
    pub output_dma: u32,
    /// `false` writes the fp32 accumulator (fp32 output cube).
    pub fp32_to_fp16: bool,
    /// Adds `add_dma` (the running partial) into the conv result on the NPU.
    pub accumulate: bool,
    pub add_dma: u32,
    /// `EW_OP_TYPE = multiply` instead of the accumulated add (the conv-main EW
    /// multiply; only meaningful with [`MatmulParams::accumulate`]).
    pub ew_mul: bool,
    /// Selects the EW ALU algo on the same conv-main datapath: `0` keeps the
    /// add/multiply encoding, `2` is MAX, `3` is MIN.
    pub ew_op: u32,
    pub weight_reuse: bool,
    pub data_reuse: bool,
}

impl MatmulParams {
    pub fn new(m: u32, k: u32, n: u32, input_dma: u32, weights_dma: u32, output_dma: u32) -> Self {
        Self {
            m,
            k,
            n,
            input_dma,
            weights_dma,
            output_dma,
            fp32_to_fp16: true,
            accumulate: false,
            add_dma: 0,
            ew_mul: false,
            ew_op: 0,
            weight_reuse: false,
            data_reuse: false,
        }
    }
}

#[derive(Default, Clone)]
pub(crate) struct CnaDesc {
    pub(crate) conv_mode: u32,
    pub(crate) in_precision: u32,
    pub(crate) proc_precision: u32,
    pub(crate) kernel_groups: u32,
    pub(crate) feature_grains: u32,
    pub(crate) conv_y_stride: u32,
    pub(crate) conv_x_stride: u32,
    pub(crate) datain_width: u32,
    pub(crate) datain_height: u32,
    pub(crate) datain_channel: u32,
    pub(crate) dataout_width: u32,
    pub(crate) dataout_height: u32,
    pub(crate) dataout_atomics: u32,
    pub(crate) weight_bytes: u32,
    pub(crate) weight_bytes_per_kernel: u32,
    pub(crate) weight_width: u32,
    pub(crate) weight_height: u32,
    pub(crate) weight_kernels: u32,
    pub(crate) weight_bank: u32,
    pub(crate) data_bank: u32,
    pub(crate) weight_reuse: bool,
    pub(crate) data_reuse: bool,
    pub(crate) data_entries: u32,
    pub(crate) data_sign: u32,
    pub(crate) cvt_type: u32,
    pub(crate) cvt_bypass: u32,
    pub(crate) cvt_scale0: u32,
    pub(crate) cvt_scale1: u32,
    pub(crate) cvt_scale2: u32,
    pub(crate) cvt_scale3: u32,
    pub(crate) fc_skip_en: u32,
    pub(crate) data_offset: u32,
    pub(crate) pad_left: u32,
    pub(crate) pad_top: u32,
    pub(crate) feature_base_addr: u32,
    pub(crate) weight_offset: u32,
    pub(crate) weight_burst_len: u32,
    pub(crate) data_burst_len: u32,
    pub(crate) line_stride: u32,
    pub(crate) surf_stride: u32,
    pub(crate) dma_width: u32,
    pub(crate) dma_height: u32,
    pub(crate) dma_channel: u32,
    pub(crate) decompress_addr0: u32,
}

#[derive(Default, Clone)]
pub(crate) struct CoreDesc {
    pub(crate) proc_precision: u32,
    pub(crate) qd_en: u32,
    pub(crate) dataout_height: u32,
    pub(crate) dataout_width: u32,
    pub(crate) dataout_channel: u32,
}

#[derive(Default, Clone)]
pub(crate) struct DpuDesc {
    pub(crate) burst_len: u32,
    pub(crate) conv_mode: u32,
    pub(crate) output_mode: u32,
    pub(crate) flying_mode: u32,
    pub(crate) out_precision: u32,
    pub(crate) in_precision: u32,
    pub(crate) proc_precision: u32,
    pub(crate) dst_base_addr: u32,
    pub(crate) dst_surf_stride: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) channel: u32,
    pub(crate) bs_bypass: u32,
    pub(crate) bs_alu_bypass: u32,
    pub(crate) bs_mul_bypass: u32,
    pub(crate) bs_relu_bypass: u32,
    pub(crate) bs_ow_op: u32,
    pub(crate) od_bypass: u32,
    pub(crate) size_e_2: u32,
    pub(crate) size_e_1: u32,
    pub(crate) size_e_0: u32,
    pub(crate) channel_wdma: u32,
    pub(crate) height_wdma: u32,
    pub(crate) width_wdma: u32,
    pub(crate) bn_relu_bypass: u32,
    pub(crate) bn_mul_bypass: u32,
    pub(crate) bn_alu_bypass: u32,
    pub(crate) bn_bypass: u32,
    pub(crate) ew_bypass: u32,
    pub(crate) ew_op_bypass: u32,
    pub(crate) ew_lut_bypass: u32,
    pub(crate) ew_op_cvt_bypass: u32,
    pub(crate) ew_relu_bypass: u32,
    pub(crate) fp32tofp16_en: u32,
    pub(crate) out_cvt_scale: u32,
    pub(crate) out_cvt_offset: u32,
    pub(crate) out_cvt_shift: u8,
    pub(crate) out_cvt_minus_exp: u8,
    pub(crate) out_cvt_cvt_type: u8,
    pub(crate) out_cvt_round: u8,
    pub(crate) bn_mul_operand: u16,
    pub(crate) mc_surf_out: u8,
    pub(crate) tp_precision: u8,
    pub(crate) size_c_wdma: u16,
    pub(crate) tp_org_en: u8,
    pub(crate) surf_add: u32,
    pub(crate) ew_accumulate: bool,
    pub(crate) ew_base_addr: u32,
    pub(crate) ew_src_base_addr: u32,
    pub(crate) ew_cfg_val: u32,
    pub(crate) erdma_cfg_val: u32,
    pub(crate) ew_surf_stride: u32,
    pub(crate) rdma_fmc_val: u32,
    pub(crate) ew_surf_notch: u32,
}

/// Emit one matmul program into `ops`, returning the word count.
pub fn emit(params: &MatmulParams, ops: &mut Vec<NpuOp>) -> Result<usize> {
    let start = ops.len();
    emit_fp16(params, ops)?;
    Ok(ops.len() - start)
}

fn emit_fp16(params: &MatmulParams, ops: &mut Vec<NpuOp>) -> Result<()> {
    let mut cna = CnaDesc {
        conv_mode: DIRECT_CONVOLUTION,
        in_precision: PRECISION_FLOAT16,
        proc_precision: PRECISION_FLOAT16,
        kernel_groups: 0,
        feature_grains: params.m + 1,
        conv_x_stride: 1,
        conv_y_stride: 1,
        datain_width: 1,
        datain_height: params.m,
        datain_channel: params.k,
        dataout_width: 1,
        dataout_height: params.m,
        dataout_atomics: params.m,
        weight_width: 1,
        weight_height: 1,
        weight_kernels: params.n,
        weight_bytes_per_kernel: params.k * 2,
        weight_bytes: params.k * 2 * params.n,
        ..Default::default()
    };
    if cna.feature_grains > 0x3FF {
        log_error!("feature_grains {} exceeds 10 bits", cna.feature_grains);
        return Err(Error::tiling("gen_matmul_fp16"));
    }

    let fd_bytes = cna.datain_width * cna.datain_height * cna.datain_channel * 2;
    let fd_banks = (fd_bytes as usize).div_ceil(CBUF_BANK_SIZE);
    let weight_bytes = cna.weight_bytes as usize;
    let weight_banks = weight_bytes.div_ceil(CBUF_BANK_SIZE);
    if fd_banks > CBUF_BANKS - 1 {
        return Err(Error::tiling("gen_matmul_fp16/cbuf"));
    }
    if cna.weight_bytes_per_kernel as usize > CBUF_BANK_SIZE {
        return Err(Error::tiling("gen_matmul_fp16/weight-kernel"));
    }
    cna.weight_bank = (CBUF_BANKS - fd_banks) as u32;
    let _ = weight_banks;
    cna.data_bank = fd_banks as u32;
    cna.weight_reuse = params.weight_reuse;
    cna.data_reuse = params.data_reuse;
    cna.data_entries = (cna.datain_width * cna.datain_channel).div_ceil(32);
    cna.data_sign = 1;
    cna.cvt_type = 1;
    cna.cvt_bypass = 1;
    cna.cvt_scale0 = 1;
    cna.cvt_scale1 = 1;
    cna.cvt_scale2 = 1;
    cna.cvt_scale3 = 1;
    cna.fc_skip_en = 0;
    cna.data_offset = 0;
    cna.pad_left = 0;
    cna.pad_top = 0;
    cna.feature_base_addr = params.input_dma;
    cna.weight_offset = 0;
    cna.weight_burst_len = 0xF;
    cna.data_burst_len = 0xF;
    cna.line_stride = cna.datain_width * 4;
    cna.surf_stride = feature_surf_stride(cna.line_stride, cna.datain_height);
    cna.dma_width = cna.datain_width;
    cna.dma_height = cna.datain_height;
    cna.dma_channel = cna.datain_channel;
    cna.decompress_addr0 = params.weights_dma;

    let core = CoreDesc {
        proc_precision: PRECISION_FLOAT16,
        qd_en: 1,
        dataout_height: cna.dataout_height - 1,
        dataout_width: cna.dataout_width - 1,
        dataout_channel: cna.weight_kernels - 1,
    };

    let mut dpu = DpuDesc {
        burst_len: 0xF,
        conv_mode: DIRECT_CONVOLUTION,
        output_mode: 0x2,
        flying_mode: 0,
        out_precision: if params.fp32_to_fp16 {
            PRECISION_FLOAT16
        } else {
            PRECISION_FLOAT32
        },
        in_precision: PRECISION_FLOAT16,
        proc_precision: PRECISION_FLOAT16,
        dst_base_addr: params.output_dma,
        dst_surf_stride: cna.dataout_height * cna.dataout_width,
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
        fp32tofp16_en: params.fp32_to_fp16 as u32,
        out_cvt_scale: 1,
        ..Default::default()
    };
    if params.fp32_to_fp16 {
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
    dpu.surf_add = if params.fp32_to_fp16 {
        dpu.dst_surf_stride * 2
    } else {
        dpu.dst_surf_stride * 4
    };

    // K-accumulation geometry: the running partial is one surface after the conv
    // output's first surface.
    dpu.ew_accumulate = params.accumulate;
    {
        let m = params.m;
        let mut ew_stride = m.max(12);
        let mut base_off = m * 16;
        if let Some(v) = knobs::u32_auto("ROCKET_EW_SURF") {
            ew_stride = v;
        }
        if let Some(v) = knobs::u32_auto("ROCKET_EW_BASE_OFF") {
            base_off = v;
        }
        let mut notch = ew_stride;
        if let Some(v) = knobs::u32_auto("ROCKET_EW_NOTCH") {
            notch = v;
        }
        dpu.ew_src_base_addr = params.add_dma;
        dpu.ew_base_addr = params.add_dma.wrapping_add(base_off);
        dpu.ew_surf_stride = ew_stride << 4;
        dpu.ew_surf_notch = notch << 4;
    }
    {
        let mut ew = match params.ew_op {
            2 => EW_CFG_ALU_BASE,             // MAX (EW_ALU_ALGO 0)
            3 => EW_CFG_ALU_BASE | (1 << 16), // MIN (EW_ALU_ALGO 1)
            _ if params.ew_mul => EW_CFG_MUL, // EW_OP_TYPE = multiply
            _ => EW_CFG_ALU_BASE | (2 << 16), // add
        };
        let mut erd = ERDMA_CFG_16B;
        let mut fmc = rdma_fmc_burst_len(15)
            | rdma_fmc_comb_use(5)
            | rdma_fmc_in_precision(PRECISION_FLOAT16)
            | rdma_fmc_proc_precision(PRECISION_FLOAT16);
        if let Some(v) = knobs::u32_auto("ROCKET_EW_CFG") {
            ew = v;
        }
        if let Some(v) = knobs::u32_auto("ROCKET_ERDMA_CFG") {
            erd = v;
        }
        if let Some(v) = knobs::u32_auto("ROCKET_RDMA_FMC") {
            fmc = v;
        }
        dpu.ew_cfg_val = ew;
        dpu.erdma_cfg_val = erd;
        dpu.rdma_fmc_val = fmc;
    }

    emit_task(ops, &cna, &core, &dpu)
}

pub(crate) fn feature_surf_stride(line_stride: u32, datain_height: u32) -> u32 {
    let s = line_stride as i64 * ((datain_height / 4) as i64 - 1);
    if s < 0 { 0 } else { s as u32 }
}

pub(crate) fn geometry_fits(cna: &CnaDesc, dpu: &DpuDesc) -> bool {
    let fields: &[(&str, u32, u32)] = &[
        ("CNA datain_width", cna.datain_width, 0x7FF),
        ("CNA datain_height", cna.datain_height, 0x7FF),
        ("CNA dataout_width", cna.dataout_width, 0x7FF),
        ("CNA dataout_atomics", cna.dataout_atomics, 0x3FFFF),
        ("CNA dma_width", cna.dma_width, 0x7FF),
        ("CNA dma_height", cna.dma_height, 0x7FF),
        ("DPU width", dpu.width, 0x1FFF),
        ("DPU height", dpu.height, 0x1FFF),
        ("DPU channel", dpu.channel, 0x1FFF),
        ("DPU width_wdma", dpu.width_wdma, 0x1FFF),
        ("DPU height_wdma", dpu.height_wdma, 0x1FFF),
        ("DPU channel_wdma", dpu.channel_wdma, 0x1FFF),
    ];
    for &(name, v, max) in fields {
        if v > max {
            log_error!("{name} {v} exceeds its field (max {max})");
            return false;
        }
    }
    true
}

pub(crate) fn emit_task(
    ops: &mut Vec<NpuOp>,
    cna: &CnaDesc,
    core: &CoreDesc,
    dpu: &DpuDesc,
) -> Result<()> {
    if !geometry_fits(cna, dpu) {
        return Err(Error::shape("gen_matmul_task"));
    }
    let mut push = |target, reg, value| ops.push(NpuOp::new(target, reg, value));

    push(DPU, DPU_S_POINTER, 0xE);
    push(DPU_RDMA, DPU_RDMA_S_POINTER, 0xE);

    let value =
        ((cna.proc_precision & 0x7) << 7) | ((cna.in_precision & 0x7) << 4) | (cna.conv_mode & 0xF);
    push(CNA, CNA_CONV_CON1, value);
    let value = ((cna.kernel_groups & 0xFF) << 16) | ((cna.feature_grains & 0x3FF) << 4);
    push(CNA, CNA_CONV_CON2, value);
    let value = ((cna.conv_y_stride & 0x7) << 3) | (cna.conv_x_stride & 0x7);
    push(CNA, CNA_CONV_CON3, value);
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
    let mut cbuf = ((cna.weight_bank & 0xF) << 4)
        | (cna.data_bank & 0xF)
        | (if cna.weight_reuse { 0x2000 } else { 0 })
        | (if cna.data_reuse { 0x1000 } else { 0 });
    if let Some(v) = knobs::u32_auto("ROCKET_FC_DATA_BANK") {
        cbuf |= (v & 0x7) << 8;
    }
    push(CNA, CNA_CBUF_CON0, cbuf);
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
    push(CNA, CNA_PAD_CON1, 0);

    // Diagnostic clock-gating knobs (unset = byte-identical program).
    if let Some(v) = knobs::u32_auto("ROCKET_CNA_CLK_GATE") {
        push(CNA, CNA_CLK_GATE, v & 0x17);
    }
    let mut misc = ((core.proc_precision & 0x7) << 8) | (core.qd_en & 1);
    if let Some(v) = knobs::u32_auto("ROCKET_CORE_SOFT_GATING") {
        misc |= (v & 0x3F) << 14;
    }
    push(CORE, CORE_MISC_CFG, misc);
    if let Some(v) = knobs::u32_auto("ROCKET_CORE_MAC_GATING") {
        push(CORE, CORE_MAC_GATING, v & 0x07FF_FFFF);
    }
    push(
        CORE,
        CORE_DATAOUT_SIZE_0,
        ((core.dataout_height & 0xFFFF) << 16) | (core.dataout_width & 0xFFFF),
    );
    push(CORE, CORE_DATAOUT_SIZE_1, core.dataout_channel & 0xFFFF);
    push(CORE, CORE_CLIP_TRUNCATE, 0);
    push(
        CORE,
        CORE_3030,
        knobs::u32_auto("ROCKET_CORE_3030").unwrap_or(0),
    );

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
            | (((dpu.mc_surf_out & 1) as u32) << 3)
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
        ((dpu.bs_relu_bypass & 1) << 6)
            | ((dpu.bs_mul_bypass & 1) << 4)
            | ((dpu.bs_alu_bypass & 1) << 1)
            | (dpu.bs_bypass & 1),
    );
    push(DPU, DPU_BS_ALU_CFG, 0);
    push(DPU, DPU_BS_MUL_CFG, 0);
    push(DPU, DPU_BS_RELUX_CMP_VALUE, 0);
    push(
        DPU,
        DPU_BS_OW_CFG,
        (((dpu.tp_org_en & 1) as u32) << 27)
            | ((dpu.size_e_2 & 0x7) << 8)
            | ((dpu.size_e_1 & 0x7) << 5)
            | ((dpu.size_e_0 & 0x7) << 2)
            | ((dpu.od_bypass & 1) << 1),
    );
    push(DPU, DPU_BS_OW_OP, dpu.bs_ow_op & 0xFFFF);
    push(
        DPU,
        DPU_WDMA_SIZE_0,
        (((dpu.tp_precision & 1) as u32) << 27)
            | ((dpu.size_c_wdma as u32 & 0x7FF) << 16)
            | (dpu.channel_wdma & 0x1FFF),
    );
    push(
        DPU,
        DPU_WDMA_SIZE_1,
        ((dpu.height_wdma & 0x1FFF) << 16) | (dpu.width_wdma & 0x1FFF),
    );
    push(
        DPU,
        DPU_BN_CFG,
        ((dpu.bn_relu_bypass & 1) << 6)
            | ((dpu.bn_mul_bypass & 1) << 4)
            | ((dpu.bn_alu_bypass & 1) << 1)
            | (dpu.bn_bypass & 1),
    );
    push(DPU, DPU_BN_ALU_CFG, 0);
    push(
        DPU,
        DPU_BN_MUL_CFG,
        (dpu.bn_mul_operand as u32 & 0xFFFF) << 16,
    );
    push(DPU, DPU_BN_RELUX_CMP_VALUE, 0);
    let ew_cfg = if dpu.ew_accumulate {
        dpu.ew_cfg_val
    } else {
        ((dpu.ew_relu_bypass & 1) << 9)
            | ((dpu.ew_op_cvt_bypass & 1) << 8)
            | ((dpu.ew_lut_bypass & 1) << 7)
            | ((dpu.ew_op_bypass & 1) << 1)
            | (dpu.ew_bypass & 1)
    };
    push(DPU, DPU_EW_CFG, ew_cfg);
    push(DPU, DPU_EW_CVT_OFFSET_VALUE, 0);
    push(DPU, DPU_EW_CVT_SCALE_VALUE, 1);
    push(DPU, DPU_EW_RELUX_CMP_VALUE, 0);
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
    let out_cvt_shift = (((dpu.out_cvt_cvt_type & 1) as u32) << 31)
        | (round << 30)
        | ((dpu.out_cvt_minus_exp as u32) << 12)
        | ((dpu.out_cvt_shift & 0x3F) as u32);
    push(DPU, DPU_OUT_CVT_SHIFT, out_cvt_shift);
    for i in 0..8 {
        push(DPU, DPU_EW_OP_VALUE_0 + i * 4, 0);
    }
    push(DPU, DPU_SURFACE_ADD, (dpu.surf_add & 0x0FFF_FFFF) << 4);
    push(DPU, DPU_40C4, 0);
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

    push(DPU_RDMA, DPU_RDMA_DATA_CUBE_WIDTH, dpu.width & 0x1FFF);
    push(DPU_RDMA, DPU_RDMA_DATA_CUBE_HEIGHT, dpu.height & 0x1FFF);
    push(DPU_RDMA, DPU_RDMA_DATA_CUBE_CHANNEL, dpu.channel & 0x1FFF);
    push(
        DPU_RDMA,
        DPU_RDMA_SRC_BASE_ADDR,
        if dpu.ew_accumulate {
            dpu.ew_src_base_addr
        } else {
            0
        },
    );
    push(DPU_RDMA, DPU_RDMA_BRDMA_CFG, 0);
    push(DPU_RDMA, DPU_RDMA_BS_BASE_ADDR, 0);
    push(DPU_RDMA, DPU_RDMA_NRDMA_CFG, 0);
    push(DPU_RDMA, DPU_RDMA_BN_BASE_ADDR, 0);
    if dpu.ew_accumulate {
        push(DPU_RDMA, DPU_RDMA_ERDMA_CFG, dpu.erdma_cfg_val);
        push(DPU_RDMA, DPU_RDMA_EW_BASE_ADDR, dpu.ew_base_addr);
        push(DPU_RDMA, DPU_RDMA_EW_SURF_STRIDE, dpu.ew_surf_stride);
    } else {
        push(DPU_RDMA, DPU_RDMA_ERDMA_CFG, RDMA_ERDMA_DISABLE);
        push(DPU_RDMA, DPU_RDMA_EW_BASE_ADDR, 0);
        push(DPU_RDMA, DPU_RDMA_EW_SURF_STRIDE, 0);
    }
    let fmc = if dpu.ew_accumulate {
        dpu.rdma_fmc_val
    } else {
        rdma_fmc_burst_len(15)
            | rdma_fmc_mrdma_disable(1)
            | rdma_fmc_mrdma_fp16tofp32(dpu.fp32tofp16_en)
    };
    push(DPU_RDMA, DPU_RDMA_FEATURE_MODE_CFG, fmc);
    push(DPU_RDMA, DPU_RDMA_SRC_DMA_CFG, 0);
    push(
        DPU_RDMA,
        DPU_RDMA_SURF_NOTCH,
        if dpu.ew_accumulate {
            dpu.ew_surf_notch
        } else {
            0
        },
    );
    push(DPU_RDMA, DPU_RDMA_PAD_CFG, 0);
    push(DPU_RDMA, DPU_RDMA_WEIGHT, RDMA_WEIGHT_ALL1);
    push(
        DPU_RDMA,
        DPU_RDMA_EW_SURF_NOTCH,
        if dpu.ew_accumulate {
            dpu.ew_surf_notch
        } else {
            0
        },
    );

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

    fn params(m: u32, k: u32, n: u32) -> MatmulParams {
        MatmulParams::new(m, k, n, 0x1000, 0x2000, 0x3000)
    }

    #[test]
    fn emits_a_plausible_program() {
        let mut ops = Vec::new();
        let count = emit(&params(4, 32, 16), &mut ops).unwrap();
        assert!(count > 90 && count <= RC_STRIDE, "count {count}");
        assert_eq!(ops[0], NpuOp::new(DPU, DPU_S_POINTER, 0xE));
        assert_eq!(ops[1], NpuOp::new(DPU_RDMA, DPU_RDMA_S_POINTER, 0xE));
        // The DPU_RDMA block must be armed and MRDMA disabled or the job hangs.
        let fmc = ops
            .iter()
            .find(|op| op.target() == DPU_RDMA && op.reg() == DPU_RDMA_FEATURE_MODE_CFG)
            .unwrap();
        assert_ne!(fmc.value() & (1 << 4), 0);
        // PC trailer.
        assert_eq!(
            ops[count - 1],
            NpuOp::new(OP_ENABLE, PC_OPERATION_ENABLE, 0x1D)
        );
    }

    #[test]
    fn reject_unencodable_weights() {
        // K=64 KiB per kernel overflows the 64 KiB weight-tile rule.
        let mut ops = Vec::new();
        assert!(emit(&params(4, 32768, 16), &mut ops).is_err());
    }
}
