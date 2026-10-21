//! int8 matmul register-command emission (RK3588).
//!
//! Direct translation of `gen_matmul_int8`: int8 x int8 -> int32, with the
//! feature-DMA bank slack (`data_bank = fd_banks + 1`) and the int8 output-writer
//! quirk (`size_e = 7`, `surf_add = stride * 8`). No NPU-side K-accumulation: the
//! C library never uses it in production.

use crate::error::{Error, Result};
use crate::knobs;
use crate::log::log_error;

use super::desc::*;
use super::fp16::{CnaDesc, CoreDesc, DpuDesc, MatmulParams, emit_task, feature_surf_stride};

pub fn emit(params: &MatmulParams, ops: &mut Vec<NpuOp>) -> Result<usize> {
    let start = ops.len();
    emit_int8(params, ops)?;
    Ok(ops.len() - start)
}

fn emit_int8(params: &MatmulParams, ops: &mut Vec<NpuOp>) -> Result<()> {
    let mut cna = CnaDesc {
        conv_mode: DIRECT_CONVOLUTION,
        in_precision: PRECISION_INT8,
        proc_precision: PRECISION_INT8,
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
        weight_bytes_per_kernel: params.k,
        weight_bytes: params.k * params.n,
        ..Default::default()
    };
    if cna.feature_grains > 0x3FF {
        log_error!("int8 feature_grains {} exceeds 10 bits", cna.feature_grains);
        return Err(Error::tiling("gen_matmul_int8"));
    }

    let fd_bytes = (cna.datain_width * cna.datain_height * cna.datain_channel) as usize;
    let fd_banks = fd_bytes.div_ceil(CBUF_BANK_SIZE);
    if fd_banks > CBUF_BANKS - 1 {
        return Err(Error::tiling("gen_matmul_int8/fd-banks"));
    }
    // The int8 C2=16 feature DMA reads one bank past the exact-fit allocation at
    // resonant geometries; reserve that bank and fail loudly if it cannot fit.
    if fd_banks + 1 > CBUF_BANKS - 1 {
        log_error!("int8 CBUF bank slack does not fit (fd_banks={fd_banks})");
        return Err(Error::tiling("gen_matmul_int8/bank-slack"));
    }
    if cna.weight_bytes_per_kernel as usize > CBUF_BANK_SIZE {
        return Err(Error::tiling("gen_matmul_int8/weight-kernel"));
    }
    cna.data_bank = (fd_banks + 1) as u32;
    cna.weight_bank = (CBUF_BANKS - (fd_banks + 1)) as u32;
    cna.weight_reuse = params.weight_reuse;
    cna.data_reuse = params.data_reuse;
    cna.data_entries = (cna.datain_width * cna.datain_channel).div_ceil(64);
    cna.data_sign = 1;
    cna.cvt_type = 1;
    cna.cvt_bypass = 1;
    cna.cvt_scale0 = 1;
    cna.cvt_scale1 = 1;
    cna.cvt_scale2 = 1;
    cna.cvt_scale3 = 1;
    cna.feature_base_addr = params.input_dma;
    cna.weight_burst_len = 0xF;
    cna.data_burst_len = 0xF;
    cna.line_stride = cna.datain_width * 4;
    cna.surf_stride = feature_surf_stride(cna.line_stride, cna.datain_height);
    cna.dma_width = cna.datain_width;
    cna.dma_height = cna.datain_height;
    cna.dma_channel = cna.datain_channel;
    cna.decompress_addr0 = params.weights_dma;

    if let Some(v) = knobs::u32_auto("ROCKET_I8_FDBANK_EXTRA") {
        let mut nb = cna.data_bank as i64 + v as i64;
        nb = nb.clamp(1, (CBUF_BANKS - 1) as i64);
        cna.data_bank = nb as u32;
        cna.weight_bank = (CBUF_BANKS as i64 - nb) as u32;
    }
    if let Some(v) = knobs::u32_auto("ROCKET_I8_GRAINS") {
        cna.feature_grains = v;
    }
    if let Some(v) = knobs::u32_auto("ROCKET_I8_SURF") {
        cna.surf_stride = v;
    }
    if let Some(v) = knobs::u32_auto("ROCKET_I8_ENTRIES") {
        cna.data_entries = v;
    }
    if let Some(v) = knobs::u32_auto("ROCKET_I8_LINESTRIDE") {
        cna.line_stride = v;
    }

    let core = CoreDesc {
        proc_precision: PRECISION_INT8,
        qd_en: 0,
        dataout_height: cna.dataout_height - 1,
        dataout_width: cna.dataout_width - 1,
        dataout_channel: cna.weight_kernels - 1,
    };

    let fp32_out = knobs::present("ROCKET_INT8_FP32_OUT");
    let mut dpu = DpuDesc {
        burst_len: 0xF,
        conv_mode: DIRECT_CONVOLUTION,
        output_mode: 0x2,
        flying_mode: 0,
        out_precision: if fp32_out {
            PRECISION_FLOAT32
        } else {
            PRECISION_INT32
        },
        in_precision: PRECISION_INT8,
        proc_precision: PRECISION_INT8,
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
        fp32tofp16_en: 0,
        out_cvt_scale: 1,
        ..Default::default()
    };
    let mut se = 7u32;
    let mut sm = 8u32;
    if let Some(v) = knobs::u32_auto("ROCKET_INT8_SIZE_E") {
        se = v;
    }
    if let Some(v) = knobs::u32_auto("ROCKET_INT8_SURF_MULT") {
        sm = v;
    }
    dpu.size_e_2 = se & 0x7;
    dpu.size_e_1 = se & 0x7;
    dpu.size_e_0 = se & 0x7;
    dpu.surf_add = dpu.dst_surf_stride * sm;
    dpu.od_bypass = 1;
    dpu.width_wdma = core.dataout_width;
    dpu.height_wdma = core.dataout_height;
    dpu.channel_wdma = core.dataout_channel;
    dpu.ew_accumulate = false;

    emit_task(ops, &cna, &core, &dpu)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emits_int8_program_with_bank_slack() {
        let p = MatmulParams::new(4, 64, 32, 0x1000, 0x2000, 0x3000);
        let mut ops = Vec::new();
        let n = emit(&p, &mut ops).unwrap();
        assert!(n > 90 && n <= RC_STRIDE);
        let cbuf = ops
            .iter()
            .find(|o| o.target() == CNA && o.reg() == CNA_CBUF_CON0)
            .unwrap()
            .value();
        let fd_banks = (4 * 64usize).div_ceil(CBUF_BANK_SIZE);
        assert_eq!((cbuf >> 4) & 0xF, (CBUF_BANKS - (fd_banks + 1)) as u32);
        assert_eq!(cbuf & 0xF, (fd_banks + 1) as u32);
    }

    #[test]
    fn rejects_too_many_feature_banks() {
        // 256 rows * 16384 K = 4 MiB, far beyond the CBUF.
        let p = MatmulParams::new(256, 16384, 32, 0, 0, 0);
        assert!(emit(&p, &mut Vec::new()).is_err());
    }
}
