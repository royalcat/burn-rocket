//! tf32 matmul register-command emission (RK3588).
//!
//! Direct translation of `gen_matmul_tf32`: tf32 x tf32 -> fp32, the only **4-byte
//! input** matmul on this stack (fp16/bf16/int16 are 2-byte, int8 1-byte, int4 half a
//! byte; fp32/int32 have only ever been *outputs*). You feed raw fp32 — the MAC
//! truncates the mantissa to 10 bits, toward zero (tf32 = fp32 range, fp16 precision,
//! in a 4-byte container) — and the output side is the proven fp32-out writer bf16 and
//! fp16's `_f32out` use.
//!
//! Four encodings are tf32-specific:
//!
//! - **The precision is per-stage**: tf32 = 7 is a valid code only at the *front* of
//!   the pipe. `CNA` in/proc and `CORE` proc are 7, but the DPU's data-format enums
//!   list only 0..6 (no tf32 code) — the C's first HW run set all stages to 7, the job
//!   ran, and the DPU wrote nothing. The DPU only ever sees the fp32 accumulator, so
//!   its in/proc/out are fp32 (5).
//! - **4-byte operands**: `weight_bytes_per_kernel = K * 4`, `fd_bytes = M * K * 4`, and
//!   the feature-DMA `data_entries` divisor halves to 16 (a "data entry" is a fixed
//!   64 bytes: 32 fp16, 64 int8, 16 tf32).
//! - **`size_e = 3` with `surf_add = dst_surf_stride * 4`** — the fp32-out stride, as
//!   for bf16 and the fp16 fp32-out branch.
//! - **`core_desc.qd_en = 1`**, like the fp16/bf16 writers (int8/int4 emit 0).
//!
//! Plain-conv path only (`ew_accumulate = 0`): no MRDMA/ERDMA, so no NPU-side
//! K-accumulation — K-tile partials are summed on the host in f64.

use crate::error::{Error, Result};
use crate::knobs;
use crate::log::log_error;

use super::desc::*;
use super::fp16::{CnaDesc, CoreDesc, DpuDesc, MatmulParams, emit_task, feature_surf_stride};

pub fn emit(params: &MatmulParams, ops: &mut Vec<NpuOp>) -> Result<usize> {
    let start = ops.len();
    emit_tf32(params, ops)?;
    Ok(ops.len() - start)
}

fn emit_tf32(params: &MatmulParams, ops: &mut Vec<NpuOp>) -> Result<()> {
    // Per-stage precision (the key tf32 lesson): 7 at CNA/CORE, fp32(5) at the DPU.
    // Every field stays env-overridable so a sweep can isolate which stage takes 7.
    let in_prec = knobs::u32_auto("ROCKET_TF32_PREC").unwrap_or(PRECISION_TF32) & 0x7;
    let core_prec = knobs::u32_auto("ROCKET_TF32_CORE_PREC").unwrap_or(PRECISION_TF32) & 0x7;
    let dpu_prec = knobs::u32_auto("ROCKET_TF32_DPU_PREC").unwrap_or(PRECISION_FLOAT32) & 0x7;
    let out_prec = knobs::u32_auto("ROCKET_TF32_OUT_PREC").unwrap_or(PRECISION_FLOAT32) & 0x7;
    let se = knobs::u32_auto("ROCKET_TF32_SIZE_E").unwrap_or(3);
    let sm = knobs::u32_auto("ROCKET_TF32_SURF_MULT").unwrap_or(4);
    let dentries_div = knobs::u32_auto("ROCKET_TF32_DENTRIES_DIV")
        .filter(|v| *v != 0)
        .unwrap_or(16);
    let elem_bytes = knobs::u32_auto("ROCKET_TF32_ELEM_BYTES")
        .filter(|v| *v != 0)
        .unwrap_or(4);
    let output_mode = knobs::u32_auto("ROCKET_TF32_OUTPUT_MODE").unwrap_or(0x2);
    let mc_surf_out = knobs::u32_auto("ROCKET_TF32_MC_SURF").unwrap_or(0) as u8;
    let fgrains = match knobs::u32_auto("ROCKET_TF32_FGRAINS") {
        Some(v) => v,
        None => params.m + 1,
    };

    let mut cna = CnaDesc {
        conv_mode: DIRECT_CONVOLUTION,
        in_precision: in_prec,
        proc_precision: in_prec,
        kernel_groups: 0,
        feature_grains: fgrains,
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
        // tf32 = 4 B/elem: weight bytes = W*H*K * 4 per kernel.
        weight_bytes_per_kernel: params.k * elem_bytes,
        weight_bytes: params.k * elem_bytes * params.n,
        ..Default::default()
    };
    if cna.feature_grains > 0x3FF {
        log_error!("tf32 feature_grains {} exceeds 10 bits", cna.feature_grains);
        return Err(Error::tiling("gen_matmul_tf32"));
    }

    let fd_bytes =
        (cna.datain_width * cna.datain_height * cna.datain_channel * elem_bytes) as usize;
    let fd_banks = fd_bytes.div_ceil(CBUF_BANK_SIZE);
    if fd_banks > CBUF_BANKS - 1 {
        return Err(Error::tiling("gen_matmul_tf32/fd-banks"));
    }
    if cna.weight_bytes_per_kernel as usize > CBUF_BANK_SIZE {
        return Err(Error::tiling("gen_matmul_tf32/weight-kernel"));
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
        proc_precision: core_prec,
        qd_en: 1,
        dataout_height: cna.dataout_height - 1,
        dataout_width: cna.dataout_width - 1,
        dataout_channel: cna.weight_kernels - 1,
    };

    let dpu = DpuDesc {
        burst_len: 0xF,
        conv_mode: DIRECT_CONVOLUTION,
        output_mode: output_mode & 0x3,
        flying_mode: 0,
        mc_surf_out,
        out_precision: out_prec,
        // The DPU rides the fp32 accumulator; there is no tf32 code in its enum.
        in_precision: dpu_prec,
        proc_precision: dpu_prec,
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
    emit_task(ops, &cna, &core, &dpu)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(m: u32, k: u32, n: u32) -> MatmulParams {
        MatmulParams::new(m, k, n, 0x1000, 0x2000, 0x3000)
    }

    fn value(ops: &[NpuOp], target: u64, reg: u32) -> u32 {
        ops.iter()
            .find(|o| o.target() == target && o.reg() == reg)
            .unwrap_or_else(|| panic!("missing op target={target:#x} reg={reg:#x}"))
            .value()
    }

    #[test]
    fn emits_the_tf32_per_stage_precision_program() {
        let p = params(4, 64, 32);
        let mut ops = Vec::new();
        let n = emit(&p, &mut ops).unwrap();
        // One fixed program shape, independent of the tile: the same 126 words the
        // fp16/bf16/int4 generators emit (they share `emit_task`).
        assert_eq!(n, 126);
        assert_eq!(ops.len(), n);

        // CNA in/proc = tf32 (7): the operand read is the stage that accepts 7.
        let conv_con1 = value(&ops, CNA, CNA_CONV_CON1);
        assert_eq!((conv_con1 >> 7) & 0x7, PRECISION_TF32);
        assert_eq!((conv_con1 >> 4) & 0x7, PRECISION_TF32);
        // CORE proc = tf32 and qd_en = 1 (the fp16/bf16 writer's bit, not int4's 0).
        let misc = value(&ops, CORE, CORE_MISC_CFG);
        assert_eq!((misc >> 8) & 0x7, PRECISION_TF32);
        assert_eq!(misc & 1, 1);
        // DPU in/proc/out = fp32 (5): there is no tf32 code at the output stage.
        let fmt = value(&ops, DPU, DPU_DATA_FORMAT);
        assert_eq!((fmt >> 29) & 0x7, PRECISION_FLOAT32);
        assert_eq!((fmt >> 26) & 0x7, PRECISION_FLOAT32);
        assert_eq!(fmt & 0x7, PRECISION_FLOAT32);
        assert_eq!((fmt >> 3) & 1, 0, "mc_surf_out stays 0");

        // The fp32-out writer: size_e = 3 in all three fields, surf_add = stride * 4.
        let ow = value(&ops, DPU, DPU_BS_OW_CFG);
        assert_eq!((ow >> 8) & 0x7, 3);
        assert_eq!((ow >> 5) & 0x7, 3);
        assert_eq!((ow >> 2) & 0x7, 3);
        assert_eq!((ow >> 1) & 1, 1, "od_bypass");
        assert_eq!(value(&ops, DPU, DPU_SURFACE_ADD), (4 * 4) << 4);
        assert_eq!(value(&ops, DPU, DPU_OUT_CVT_SCALE) & (1 << 16), 0);

        // 4-byte feature DMA: data_entries = K / 16 (a 64-byte entry holds 16 fp32).
        assert_eq!(value(&ops, CNA, CNA_CBUF_CON1), 64 / 16);
        // fd_bytes = 4*64*4 = 1 KiB -> one feature bank, the rest to the weights.
        let cbuf = value(&ops, CNA, CNA_CBUF_CON0);
        assert_eq!(cbuf & 0xF, 1);
        assert_eq!((cbuf >> 4) & 0xF, (CBUF_BANKS - 1) as u32);
        // 4-byte weight geometry: K * 4 bytes per kernel, N kernels.
        assert_eq!(value(&ops, CNA, CNA_WEIGHT_SIZE1), 64 * 4);
        assert_eq!(value(&ops, CNA, CNA_WEIGHT_SIZE0), 64 * 4 * 32);

        // Trailer: magic word, PC, op-40, enable.
        assert_eq!(ops[n - 4], NpuOp(0));
        assert_eq!(ops[n - 1], NpuOp::new(OP_ENABLE, PC_OPERATION_ENABLE, 0x1D));
    }

    #[test]
    fn the_program_length_is_geometry_free() {
        let mut small = Vec::new();
        let mut big = Vec::new();
        let a = emit(&params(4, 16, 16), &mut small).unwrap();
        // The largest tile the planner can hand out at the 256-row cap: fd =
        // 256 * 352 * 4 = 352 KiB is 11 banks, one short of the 12-bank CBUF.
        let b = emit(&params(256, 352, 256), &mut big).unwrap();
        assert_eq!(a, b);
        // One K-alignment step further and the feature tile no longer fits.
        assert!(emit(&params(256, 368, 256), &mut Vec::new()).is_err());
    }

    #[test]
    fn rejects_an_oversized_weight_kernel() {
        // Kt * 4 > 64 KiB: the generator refuses rather than tiling a CBUF bank.
        let mut ops = Vec::new();
        assert!(emit(&params(4, 32768, 16), &mut ops).is_err());
    }
}
