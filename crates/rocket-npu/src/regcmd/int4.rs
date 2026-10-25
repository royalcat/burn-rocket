//! int4 matmul register-command emission (RK3588).
//!
//! Direct translation of `gen_matmul_int4`: int4 x int4 -> int16, the structural
//! sibling of [`super::int8`] with a different precision triple (in/proc int4 = 6,
//! out int16 = 1), half-byte operand byte math, a doubled feature-DMA density
//! (`data_entries` divisor 128) and **no** CBUF bank slack.
//!
//! The output writer keeps the int8 conv's integer-output quirk (`size_e = 7`,
//! `surf_add = dst_surf_stride * 8`): it is HW-proven for int8's int32 write and
//! carries over unchanged to int4's int16 one. The path is plain-conv
//! (`ew_accumulate = 0`, MRDMA/ERDMA disabled), so no NPU-side K-accumulation: K-tile
//! partials come back to the host as int16.

use crate::error::{Error, Result};
use crate::knobs;
use crate::log::log_error;

use super::desc::*;
use super::fp16::{CnaDesc, CoreDesc, DpuDesc, MatmulParams, emit_task, feature_surf_stride};

pub fn emit(params: &MatmulParams, ops: &mut Vec<NpuOp>) -> Result<usize> {
    let start = ops.len();
    emit_int4(params, ops)?;
    Ok(ops.len() - start)
}

fn emit_int4(params: &MatmulParams, ops: &mut Vec<NpuOp>) -> Result<()> {
    // HW-confirmed encodings (`matmul_int4_rocket`, bit-exact), kept env-overridable as
    // the C's diagnostics are: in/proc precision int4 = 6 is the only value that gives
    // a correct int4 MAC, and out precision int16 = 1 is the int4 accumulator's width.
    let mut in_prec = PRECISION_INT4;
    let mut out_prec = PRECISION_INT16;
    if let Some(v) = knobs::u32_auto("ROCKET_INT4_PREC") {
        in_prec = v & 0x7;
    }
    if let Some(v) = knobs::u32_auto("ROCKET_INT4_OUT_PREC") {
        out_prec = v & 0x7;
    }
    let mut dentries_div = 128u32;
    if let Some(v) = knobs::u32_auto("ROCKET_INT4_DENTRIES_DIV")
        && v != 0
    {
        dentries_div = v;
    }

    let mut cna = CnaDesc {
        conv_mode: DIRECT_CONVOLUTION,
        in_precision: in_prec,
        proc_precision: in_prec,
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
        // int4 = 0.5 B/elem: weight bytes = W*H*K/2 per kernel (K is even, K%32==0).
        weight_bytes_per_kernel: params.k.div_ceil(2),
        weight_bytes: params.k.div_ceil(2) * params.n,
        ..Default::default()
    };
    if cna.feature_grains > 0x3FF {
        log_error!("int4 feature_grains {} exceeds 10 bits", cna.feature_grains);
        return Err(Error::tiling("gen_matmul_int4"));
    }

    let fd_bytes = (cna.datain_width * cna.datain_height * cna.datain_channel).div_ceil(2) as usize;

    let fd_banks = fd_bytes.div_ceil(CBUF_BANK_SIZE);
    if fd_banks > CBUF_BANKS - 1 {
        return Err(Error::tiling("gen_matmul_int4/fd-banks"));
    }
    if cna.weight_bytes_per_kernel as usize > CBUF_BANK_SIZE {
        return Err(Error::tiling("gen_matmul_int4/weight-kernel"));
    }
    // No slack bank here, unlike int8: int4's C2=32 feature cube is a whole number of
    // 16-byte atoms per group and the resonance is an int8-C2=16 DMA artefact.
    cna.data_bank = fd_banks as u32;
    cna.weight_bank = (CBUF_BANKS - fd_banks) as u32;
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
        proc_precision: in_prec,
        qd_en: 0,
        dataout_height: cna.dataout_height - 1,
        dataout_width: cna.dataout_width - 1,
        dataout_channel: cna.weight_kernels - 1,
    };

    let mut dpu = DpuDesc {
        burst_len: 0xF,
        conv_mode: DIRECT_CONVOLUTION,
        output_mode: 0x2,
        flying_mode: 0,
        out_precision: out_prec,
        in_precision: in_prec,
        proc_precision: in_prec,
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
    // The integer-conv output writer: SIZE_E 1->16 columns, 3->32, 7->all 64, and the
    // same `surf * 8` surface add the int8 int32 write uses.
    let mut se = 7u32;
    let mut sm = 8u32;
    if let Some(v) = knobs::u32_auto("ROCKET_INT4_SIZE_E") {
        se = v;
    }
    if let Some(v) = knobs::u32_auto("ROCKET_INT4_SURF_MULT") {
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

    fn params(m: u32, k: u32, n: u32) -> MatmulParams {
        MatmulParams::new(m, k, n, 0x1000, 0x2000, 0x3000)
    }

    #[test]
    fn emits_int4_program_without_bank_slack() {
        let p = params(4, 64, 64);
        let mut ops = Vec::new();
        let n = emit(&p, &mut ops).unwrap();
        // The same fixed program shape int8 emits (only field values differ).
        assert_eq!(n, 126);
        assert_eq!(ops.len(), n);
        let cbuf = ops
            .iter()
            .find(|o| o.target() == CNA && o.reg() == CNA_CBUF_CON0)
            .unwrap()
            .value();
        // fd_bytes = (4*64+1)/2 = 128 -> one bank; no int8-style slack bank.
        assert_eq!(cbuf & 0xF, 1);
        assert_eq!((cbuf >> 4) & 0xF, (CBUF_BANKS - 1) as u32);
        // Trailer: magic word, PC, op-40, enable.
        assert_eq!(ops[n - 1], NpuOp::new(OP_ENABLE, PC_OPERATION_ENABLE, 0x1D));
        assert_eq!(ops[n - 4], NpuOp(0));

        // A bigger tile keeps the same program length: the op count is geometry-free.
        let mut big = Vec::new();
        assert_eq!(emit(&params(256, 1024, 256), &mut big).unwrap(), n);
        let fmt = big
            .iter()
            .find(|o| o.target() == DPU && o.reg() == DPU_DATA_FORMAT)
            .unwrap()
            .value();
        assert_eq!(fmt >> 29, PRECISION_INT16);
        assert_eq!((fmt >> 26) & 0x7, PRECISION_INT4);
        assert_eq!(fmt & 0x7, PRECISION_INT4);
    }

    #[test]
    fn rejects_too_many_feature_banks() {
        // 256 rows * 16384 K = 2 MiB of int4, far beyond the CBUF.
        let p = params(256, 16384, 64);
        assert!(emit(&p, &mut Vec::new()).is_err());
    }
}
