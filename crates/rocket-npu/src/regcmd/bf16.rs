//! bf16 matmul register-command emission (RK3588).
//!
//! Direct translation of `gen_matmul_bf16`: bf16 x bf16 -> fp32, the same 2-byte
//! input geometry as fp16/int16 and the fp32 output writer (`size_e=3`, `surf*4`).
//! There is no NPU-side K-accumulation on this path; partials sum on the host in
//! f64 (the stream context).

use crate::error::{Error, Result};
use crate::knobs;
use crate::log::log_error;

use super::desc::*;
use super::fp16::{CnaDesc, CoreDesc, DpuDesc, MatmulParams, emit_task, feature_surf_stride};

pub fn emit(params: &MatmulParams, ops: &mut Vec<NpuOp>) -> Result<usize> {
    let start = ops.len();
    emit_bf16(params, ops)?;
    Ok(ops.len() - start)
}

fn emit_bf16(params: &MatmulParams, ops: &mut Vec<NpuOp>) -> Result<()> {
    let in_prec = knobs::u32_auto("ROCKET_BF16_PREC").unwrap_or(PRECISION_BFLOAT16);
    let out_prec = knobs::u32_auto("ROCKET_BF16_OUT_PREC").unwrap_or(PRECISION_FLOAT32);
    let mut se = knobs::u32_auto("ROCKET_BF16_SIZE_E").unwrap_or(3);
    let mut sm = knobs::u32_auto("ROCKET_BF16_SURF_MULT").unwrap_or(4);
    let dentries_div = knobs::u32_auto("ROCKET_BF16_DENTRIES_DIV")
        .filter(|v| *v != 0)
        .unwrap_or(32);

    let mut cna = CnaDesc {
        conv_mode: DIRECT_CONVOLUTION,
        in_precision: in_prec & 0x7,
        proc_precision: in_prec & 0x7,
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
        log_error!("bf16 feature_grains {} exceeds 10 bits", cna.feature_grains);
        return Err(Error::tiling("gen_matmul_bf16"));
    }
    let fd_bytes = (cna.datain_width * cna.datain_height * cna.datain_channel * 2) as usize;
    let fd_banks = fd_bytes.div_ceil(CBUF_BANK_SIZE);
    if fd_banks > CBUF_BANKS - 1 {
        return Err(Error::tiling("gen_matmul_bf16/fd-banks"));
    }
    if cna.weight_bytes_per_kernel as usize > CBUF_BANK_SIZE {
        return Err(Error::tiling("gen_matmul_bf16/weight-kernel"));
    }
    cna.weight_bank = (CBUF_BANKS - fd_banks) as u32;
    cna.data_bank = fd_banks as u32;
    cna.weight_reuse = params.weight_reuse;
    cna.data_reuse = params.data_reuse;
    cna.data_entries = (cna.datain_width * cna.datain_channel).div_ceil(dentries_div);
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

    let core = CoreDesc {
        proc_precision: in_prec & 0x7,
        qd_en: 1,
        dataout_height: cna.dataout_height - 1,
        dataout_width: cna.dataout_width - 1,
        dataout_channel: cna.weight_kernels - 1,
    };

    if let Some(v) = knobs::u32_auto("ROCKET_BF16_SIZE_E") {
        se = v;
    }
    if let Some(v) = knobs::u32_auto("ROCKET_BF16_SURF_MULT") {
        sm = v;
    }
    let dpu = DpuDesc {
        burst_len: 0xF,
        conv_mode: DIRECT_CONVOLUTION,
        output_mode: 0x2,
        flying_mode: 0,
        out_precision: out_prec & 0x7,
        in_precision: in_prec & 0x7,
        proc_precision: in_prec & 0x7,
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
        size_e_2: se & 0x7,
        size_e_1: se & 0x7,
        size_e_0: se & 0x7,
        surf_add: cna.dataout_height * cna.dataout_width * sm,
        od_bypass: 1,
        width_wdma: core.dataout_width,
        height_wdma: core.dataout_height,
        channel_wdma: core.dataout_channel,
        ..Default::default()
    };
    // `mut` is only needed for the struct-update form above with env reads before it.
    emit_task(ops, &cna, &core, &dpu)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emits_bf16_fp32_out_program() {
        let p = MatmulParams::new(4, 64, 32, 0x1000, 0x2000, 0x3000);
        let mut ops = Vec::new();
        let n = emit(&p, &mut ops).unwrap();
        assert!(n > 90 && n <= RC_STRIDE);
        let fmt = ops
            .iter()
            .find(|o| o.target() == DPU && o.reg() == DPU_DATA_FORMAT)
            .unwrap()
            .value();
        assert_eq!((fmt >> 29) & 0x7, PRECISION_FLOAT32);
        assert_eq!((fmt >> 26) & 0x7, PRECISION_BFLOAT16);
    }
}
