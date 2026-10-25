//! DPU LUT activation register-command emission (RK3588, flying mode).
//!
//! Direct translation of `gen_lut_activation_fp16` + `emit_lut_tables` from
//! `npu_regcmd.c`. A standalone NVDLA-SDP LUT pass with no conv (no CNA/CORE):
//! DPU_RDMA reads the fp16 input cube in flying mode, the BN stage multiplies it by
//! `bn_mul_operand` into the LUT index domain, the LE/LO hybrid table with hardware
//! linear interpolation produces the fixed-point result and OUT_CVT casts it back to
//! fp16. Only the DPU and DPU_RDMA blocks run (the enable word is `0x18`).
//!
//! The cube dims only partition the data: `C2 = 8` channels, `width = cols = n/8`,
//! `height = 1`, and the read/write strides are identical, so a flat fp16 vector
//! comes back element-for-element (`out[i] = f(in[i])`).
//!
//! The two 513-entry tables are uploaded through the LUT access port: one
//! `DPU_LUT_ACCESS_CFG` per table (ACCESS_TYPE write + TABLE_ID) followed by 513
//! contiguous `DPU_LUT_ACCESS_DATA` writes, which auto-increment the table address.

use crate::error::{Error, Result};
use crate::knobs;

use super::desc::*;

/// Entries in the two DPU LUT tables (`LE[0..=512]`, `LO[513..=1025]`).
pub const LUT_ENTRIES: usize = 1026;

/// Words one activation program needs (the C's `regs[2048]` bound; 1070 measured).
pub const MAX_OPS: usize = 1088;

/// Parameters of one flying-mode DPU LUT activation tile.
///
/// Mirrors the C `lut_act_params_t`: the caller owns the table and every constant.
#[derive(Debug, Clone)]
pub struct LutActParams<'a> {
    pub input_dma: u32,
    pub output_dma: u32,
    /// Element count; must be a positive multiple of 8 (the C2 cube atom).
    pub n: u32,
    /// 1026 Q-format entries: `LE[0..=512]`, `LO[513..=1025]`.
    pub lut: &'a [u16],
    /// fp16 bits: input -> LUT-index scale.
    pub bn_mul_operand: u16,
    /// BN ALU bias word (`0x8000_0000` = fp32 `-0.0`, a no-op).
    pub bn_alu_cfg: u32,
    pub out_cvt_offset: u32,
    pub out_cvt_scale: u16,
    pub out_cvt_minus_exp: u8,
    pub out_cvt_cvt_type: u8,
    pub lut_le_start: u32,
    pub lut_lo_end: u32,
    pub le_slope_scale: u16,
    pub le_slope_shift: u8,
    pub lo_slope_scale: u16,
    pub lo_slope_shift: u8,
    pub le_index_select: u8,
    pub lo_index_select: u8,
}

/// Emit one LUT activation program into `ops`, returning the word count.
///
/// `ROCKET_LUT_SPTR` / `ROCKET_LUT_ENABLE` keep the C generator's bring-up overrides.
pub fn emit(p: &LutActParams<'_>, ops: &mut Vec<NpuOp>) -> Result<usize> {
    if p.n == 0 || !p.n.is_multiple_of(8) {
        return Err(Error::shape("gen_lut_activation_fp16"));
    }
    if p.lut.len() < LUT_ENTRIES {
        return Err(Error::shape("gen_lut_activation_fp16/lut"));
    }
    let cols = p.n / 8;
    if cols > 0x1FFF {
        // DATA_CUBE_WIDTH is 13 bits, so n/8 must fit.
        return Err(Error::tiling("gen_lut_activation_fp16/width"));
    }
    let width = cols - 1; // the width register is W-1
    let sptr = knobs::u32_auto("ROCKET_LUT_SPTR").unwrap_or(0xE);
    let enable = knobs::u32_auto("ROCKET_LUT_ENABLE").unwrap_or(0x18);

    let start = ops.len();
    let mut push = |target, reg, value| ops.push(NpuOp::new(target, reg, value));

    push(DPU, DPU_S_POINTER, sptr);
    push(DPU_RDMA, DPU_RDMA_S_POINTER, sptr);

    // Upload the two LUT tables (513 Q-format entries each) via the access port.
    push(DPU, DPU_LUT_ACCESS_CFG, 1 << 17);
    for &word in &p.lut[..513] {
        push(DPU, DPU_LUT_ACCESS_DATA, word as u32);
    }
    push(DPU, DPU_LUT_ACCESS_CFG, (1 << 17) | (1 << 16));
    for &word in &p.lut[513..LUT_ENTRIES] {
        push(DPU, DPU_LUT_ACCESS_DATA, word as u32);
    }

    // Flying mode, output_mode 2, fp16 in/proc/out.
    push(DPU, DPU_FEATURE_MODE_CFG, (15 << 5) | (2 << 1) | 1);
    push(
        DPU,
        DPU_DATA_FORMAT,
        (PRECISION_FLOAT16 << 29) | (PRECISION_FLOAT16 << 26) | PRECISION_FLOAT16,
    );
    push(DPU, DPU_DST_BASE_ADD, p.output_dma);
    push(DPU, DPU_DST_SURF_STRIDE, (cols & 0x0FFF_FFFF) << 4);
    push(DPU, DPU_DATA_CUBE_WIDTH, width & 0x1FFF);
    push(DPU, DPU_DATA_CUBE_HEIGHT, 0);
    push(DPU, DPU_DATA_CUBE_CHANNEL, (7 << 16) | 7); // C = 8

    // BS fully bypassed; the OW writer only bypasses OD.
    push(DPU, DPU_BS_CFG, (1 << 6) | (1 << 4) | (1 << 1) | 1);
    push(DPU, DPU_BS_OW_CFG, 1 << 1);
    push(DPU, DPU_WDMA_SIZE_0, 7);
    push(DPU, DPU_WDMA_SIZE_1, width & 0x1FFF);

    // BN: ALU_ALGO(2) bias, MUL by the index scale; RELU bypassed.
    push(DPU, DPU_BN_CFG, (2 << 16) | (1 << 6));
    push(DPU, DPU_BN_ALU_CFG, p.bn_alu_cfg);
    push(
        DPU,
        DPU_BN_MUL_CFG,
        ((p.bn_mul_operand as u32) & 0xFFFF) << 16,
    );

    // EW: RELU + OP_CVT bypassed, LUT left active (bit7 clear); cvt scale 1.
    push(DPU, DPU_EW_CFG, (1 << 9) | (1 << 8));
    push(DPU, DPU_EW_CVT_SCALE_VALUE, 1);

    // Output converter: fixed-point LUT result -> fp16.
    push(DPU, DPU_OUT_CVT_OFFSET, p.out_cvt_offset);
    push(
        DPU,
        DPU_OUT_CVT_SCALE,
        (1 << 16) | ((p.out_cvt_scale as u32) & 0xFFFF),
    );
    push(
        DPU,
        DPU_OUT_CVT_SHIFT,
        (((p.out_cvt_cvt_type as u32) & 1) << 31) | (((p.out_cvt_minus_exp as u32) & 0xFF) << 12),
    );

    push(DPU, DPU_SURFACE_ADD, (cols & 0x0FFF_FFFF) << 4);
    push(DPU, DPU_40C4, 0);

    // LUT control: hybrid LE+LO mux, index selects, table range + extrapolation.
    push(DPU, DPU_LUT_CFG, (1 << 6) | (1 << 5) | (2 << 2));
    push(
        DPU,
        DPU_LUT_INFO,
        (((p.lo_index_select as u32) & 0xFF) << 16) | (((p.le_index_select as u32) & 0xFF) << 8),
    );
    push(DPU, DPU_LUT_LE_START, p.lut_le_start);
    push(DPU, DPU_LUT_LO_END, p.lut_lo_end);
    push(
        DPU,
        DPU_LUT_LE_SLOPE_SCALE,
        (p.le_slope_scale as u32) & 0xFFFF,
    );
    push(
        DPU,
        DPU_LUT_LE_SLOPE_SHIFT,
        (p.le_slope_shift as u32) & 0x1F,
    );
    push(
        DPU,
        DPU_LUT_LO_SLOPE_SCALE,
        (p.lo_slope_scale as u32) & 0xFFFF,
    );
    push(
        DPU,
        DPU_LUT_LO_SLOPE_SHIFT,
        (p.lo_slope_shift as u32) & 0x1F,
    );

    // DPU_RDMA: the input feed. MRDMA flying, fp16->fp32 on read, ERDMA disabled.
    push(DPU_RDMA, DPU_RDMA_DATA_CUBE_WIDTH, width & 0x1FFF);
    push(DPU_RDMA, DPU_RDMA_DATA_CUBE_HEIGHT, 0);
    push(DPU_RDMA, DPU_RDMA_DATA_CUBE_CHANNEL, 7);
    push(DPU_RDMA, DPU_RDMA_SRC_BASE_ADDR, p.input_dma);
    push(DPU_RDMA, DPU_RDMA_ERDMA_CFG, RDMA_ERDMA_DISABLE);
    push(
        DPU_RDMA,
        DPU_RDMA_FEATURE_MODE_CFG,
        rdma_fmc_in_precision(PRECISION_FLOAT16)
            | rdma_fmc_burst_len(15)
            | rdma_fmc_proc_precision(PRECISION_FLOAT16)
            | rdma_fmc_mrdma_fp16tofp32(1)
            | 1, // FLYING_MODE
    );
    push(DPU_RDMA, DPU_RDMA_WEIGHT, RDMA_WEIGHT_ALL1);

    // PC trailer (rocket framing): enable = DPU|DPU_RDMA block bits.
    drop(push);
    ops.push(NpuOp(0));
    ops.push(NpuOp::new(PC, PC_REGISTER_AMOUNTS, 0));
    ops.push(NpuOp::new(OP_40, 0, 0));
    ops.push(NpuOp::new(OP_ENABLE, PC_OPERATION_ENABLE, enable));

    Ok(ops.len() - start)
}

/// FNV-1a of the C `gen_lut_activation_fp16` program (TANH epilogue, n = 1024).
#[cfg(test)]
const PROGRAM_FNV: u64 = 695932668960196215;

#[cfg(test)]
mod tests {
    use super::*;

    fn params<'a>(lut: &'a [u16], n: u32) -> LutActParams<'a> {
        LutActParams {
            input_dma: 0x1000,
            output_dma: 0x2000,
            n,
            lut,
            bn_mul_operand: 0x6912,
            bn_alu_cfg: 0x8000_0000,
            out_cvt_offset: 1,
            out_cvt_scale: 1,
            out_cvt_minus_exp: 15,
            out_cvt_cvt_type: 1,
            lut_le_start: 0xffff_c000,
            lut_lo_end: 0x4000,
            le_slope_scale: 0,
            le_slope_shift: 0,
            lo_slope_scale: 0,
            lo_slope_shift: 0,
            le_index_select: 5,
            lo_index_select: 5,
        }
    }

    #[test]
    fn emits_the_c_program_shape() {
        let lut = vec![7u16; LUT_ENTRIES];
        let mut ops = Vec::new();
        let count = emit(&params(&lut, 1024), &mut ops).unwrap();
        assert_eq!(count, 1070, "the C generator's word count");
        assert!(count <= MAX_OPS);
        // Trailer.
        assert_eq!(
            ops[count - 1],
            NpuOp::new(OP_ENABLE, PC_OPERATION_ENABLE, 0x18)
        );
        assert_eq!(ops[count - 2], NpuOp::new(OP_40, 0, 0));
        assert_eq!(ops[count - 3], NpuOp::new(PC, PC_REGISTER_AMOUNTS, 0));
        assert_eq!(ops[count - 4], NpuOp(0));
        // Table upload: 513 contiguous DATA writes per table, cfg at the head.
        assert_eq!(ops[2], NpuOp::new(DPU, DPU_LUT_ACCESS_CFG, 1 << 17));
        assert_eq!(ops[3], NpuOp::new(DPU, DPU_LUT_ACCESS_DATA, 7));
        assert_eq!(ops[515], NpuOp::new(DPU, DPU_LUT_ACCESS_DATA, 7));
        assert_eq!(
            ops[516],
            NpuOp::new(DPU, DPU_LUT_ACCESS_CFG, (1 << 17) | (1 << 16))
        );
        // Width register is cols-1 (1024 elements -> 128 cube columns).
        let w = ops
            .iter()
            .find(|op| op.target() == DPU && op.reg() == DPU_DATA_CUBE_WIDTH)
            .unwrap();
        assert_eq!(w.value(), 127);
    }

    #[test]
    fn rejects_bad_counts() {
        let lut = vec![0u16; LUT_ENTRIES];
        let mut ops = Vec::new();
        assert_eq!(
            emit(&params(&lut, 0), &mut ops).unwrap_err().status(),
            crate::Status::Shape
        );
        assert_eq!(
            emit(&params(&lut, 1023), &mut ops).unwrap_err().status(),
            crate::Status::Shape
        );
        assert_eq!(
            emit(&params(&lut, 8 * 0x2000), &mut ops)
                .unwrap_err()
                .status(),
            crate::Status::Tiling
        );
        assert!(emit(&params(&lut[..10], 8), &mut ops).is_err());
    }

    #[test]
    fn program_matches_the_c_word_for_word() {
        // FNV-1a over the C `gen_lut_activation_fp16` dump (n=1024, in 0x1000,
        // out 0x2000, the default TANH epilogue's LUT + constants), byte-compared
        // against the host C oracle's `--dump-gen` output.
        use crate::ops::activation::{Activation, LutTable, lut_epilogue_build};
        // The builder reads the process-wide LUT knobs, so serialize with the ops tests.
        let _g = crate::knobs::TEST_ENV
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut lut = LutTable::zeroed();
        let ep = lut_epilogue_build(Activation::Tanh, &mut lut).unwrap();
        let mut ops = Vec::new();
        let count = emit(
            &LutActParams {
                input_dma: 0x1000,
                output_dma: 0x2000,
                n: 1024,
                lut: lut.words(),
                bn_mul_operand: ep.bn_mul_operand,
                bn_alu_cfg: ep.bn_alu_cfg,
                out_cvt_offset: ep.out_cvt_offset,
                out_cvt_scale: ep.out_cvt_scale,
                out_cvt_minus_exp: ep.out_cvt_minus_exp,
                out_cvt_cvt_type: ep.out_cvt_cvt_type,
                lut_le_start: ep.lut_le_start,
                lut_lo_end: ep.lut_lo_end,
                le_slope_scale: ep.le_slope_scale,
                le_slope_shift: ep.le_slope_shift,
                lo_slope_scale: ep.lo_slope_scale,
                lo_slope_shift: ep.lo_slope_shift,
                le_index_select: ep.le_index_select,
                lo_index_select: ep.lo_index_select,
            },
            &mut ops,
        )
        .unwrap();
        assert_eq!(count, 1070, "the C generator's word count");
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for op in &ops {
            for b in op.0.to_le_bytes() {
                hash ^= b as u64;
                hash = hash.wrapping_mul(0x0100_0000_01b3);
            }
        }
        assert_eq!(hash, PROGRAM_FNV, "the program drifted from the C emission");
    }
}
