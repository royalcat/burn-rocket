//! RK3588 register offsets, block targets, and precision encodings.
//!
//! These values are from the mainline/rocket userspace documentation and are
//! IP-revision-specific to the RK3588. The encoding is
//! `(target << 48) | (value << 16) | reg`.

/// One register write in an NPU program.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NpuOp(pub u64);

impl NpuOp {
    pub const fn new(target: u64, reg: u32, value: u32) -> Self {
        Self((target << 48) | ((value as u64) << 16) | reg as u64)
    }

    pub const fn target(self) -> u64 {
        self.0 >> 48
    }

    pub const fn reg(self) -> u32 {
        (self.0 & 0xffff) as u32
    }

    pub const fn value(self) -> u32 {
        ((self.0 >> 16) & 0xffff_ffff) as u32
    }
}

// ── Block targets ──────────────────────────────────────────────────────────────
pub const PC: u64 = 0x0101;
pub const CNA: u64 = 0x0201;
pub const CORE: u64 = 0x0801;
pub const DPU: u64 = 0x1001;
pub const DPU_RDMA: u64 = 0x2001;
pub const OP_40: u64 = 0x0041;
pub const OP_ENABLE: u64 = 0x0081;

// ── PC ─────────────────────────────────────────────────────────────────────────
pub const PC_OPERATION_ENABLE: u32 = 0x0008;
pub const PC_BASE_ADDRESS: u32 = 0x0010;
pub const PC_REGISTER_AMOUNTS: u32 = 0x0014;

// ── CNA ────────────────────────────────────────────────────────────────────────
pub const CNA_S_POINTER: u32 = 0x1004;
pub const CNA_CONV_CON1: u32 = 0x100C;
pub const CNA_CONV_CON2: u32 = 0x1010;
pub const CNA_CONV_CON3: u32 = 0x1014;
pub const CNA_DATA_SIZE0: u32 = 0x1020;
pub const CNA_DATA_SIZE1: u32 = 0x1024;
pub const CNA_DATA_SIZE2: u32 = 0x1028;
pub const CNA_DATA_SIZE3: u32 = 0x102C;
pub const CNA_WEIGHT_SIZE0: u32 = 0x1030;
pub const CNA_WEIGHT_SIZE1: u32 = 0x1034;
pub const CNA_WEIGHT_SIZE2: u32 = 0x1038;
pub const CNA_CBUF_CON0: u32 = 0x1040;
pub const CNA_CBUF_CON1: u32 = 0x1044;
pub const CNA_CVT_CON0: u32 = 0x104C;
pub const CNA_CVT_CON1: u32 = 0x1050;
pub const CNA_CVT_CON2: u32 = 0x1054;
pub const CNA_CVT_CON3: u32 = 0x1058;
pub const CNA_CVT_CON4: u32 = 0x105C;
pub const CNA_FC_CON0: u32 = 0x1060;
pub const CNA_FC_CON1: u32 = 0x1064;
pub const CNA_PAD_CON0: u32 = 0x1068;
pub const CNA_FEATURE_DATA_ADDR: u32 = 0x1070;
pub const CNA_FC_CON2: u32 = 0x1074;
pub const CNA_DMA_CON0: u32 = 0x1078;
pub const CNA_DMA_CON1: u32 = 0x107C;
pub const CNA_DMA_CON2: u32 = 0x1080;
pub const CNA_FC_DATA_SIZE0: u32 = 0x1084;
pub const CNA_FC_DATA_SIZE1: u32 = 0x1088;
pub const CNA_CLK_GATE: u32 = 0x1090;
pub const CNA_DCOMP_CTRL: u32 = 0x1100;
pub const CNA_DCOMP_REGNUM: u32 = 0x1104;
pub const CNA_DCOMP_ADDR0: u32 = 0x1110;
pub const CNA_DCOMP_AMOUNT: u32 = 0x1140;
pub const CNA_CVT_CON5: u32 = 0x1180;
pub const CNA_PAD_CON1: u32 = 0x1184;

// ── CORE ───────────────────────────────────────────────────────────────────────
pub const CORE_MAC_GATING: u32 = 0x300C;
pub const CORE_MISC_CFG: u32 = 0x3010;
pub const CORE_DATAOUT_SIZE_0: u32 = 0x3014;
pub const CORE_DATAOUT_SIZE_1: u32 = 0x3018;
pub const CORE_CLIP_TRUNCATE: u32 = 0x301C;
pub const CORE_3030: u32 = 0x3030;

// ── DPU ────────────────────────────────────────────────────────────────────────
pub const DPU_S_POINTER: u32 = 0x4004;
pub const DPU_FEATURE_MODE_CFG: u32 = 0x400C;
pub const DPU_DATA_FORMAT: u32 = 0x4010;
pub const DPU_OFFSET_PEND: u32 = 0x4014;
pub const DPU_DST_BASE_ADD: u32 = 0x4020;
pub const DPU_DST_SURF_STRIDE: u32 = 0x4024;
pub const DPU_DATA_CUBE_WIDTH: u32 = 0x4030;
pub const DPU_DATA_CUBE_HEIGHT: u32 = 0x4034;
pub const DPU_DATA_CUBE_NOTCH_ADDR: u32 = 0x4038;
pub const DPU_DATA_CUBE_CHANNEL: u32 = 0x403C;
pub const DPU_BS_CFG: u32 = 0x4040;
pub const DPU_BS_ALU_CFG: u32 = 0x4044;
pub const DPU_BS_MUL_CFG: u32 = 0x4048;
pub const DPU_BS_RELUX_CMP_VALUE: u32 = 0x404C;
pub const DPU_BS_OW_CFG: u32 = 0x4050;
pub const DPU_BS_OW_OP: u32 = 0x4054;
pub const DPU_WDMA_SIZE_0: u32 = 0x4058;
pub const DPU_WDMA_SIZE_1: u32 = 0x405C;
pub const DPU_BN_CFG: u32 = 0x4060;
pub const DPU_BN_ALU_CFG: u32 = 0x4064;
pub const DPU_BN_MUL_CFG: u32 = 0x4068;
pub const DPU_BN_RELUX_CMP_VALUE: u32 = 0x406C;
pub const DPU_EW_CFG: u32 = 0x4070;
pub const DPU_EW_CVT_OFFSET_VALUE: u32 = 0x4074;
pub const DPU_EW_CVT_SCALE_VALUE: u32 = 0x4078;
pub const DPU_EW_RELUX_CMP_VALUE: u32 = 0x407C;
pub const DPU_OUT_CVT_OFFSET: u32 = 0x4080;
pub const DPU_OUT_CVT_SCALE: u32 = 0x4084;
pub const DPU_OUT_CVT_SHIFT: u32 = 0x4088;
pub const DPU_EW_OP_VALUE_0: u32 = 0x4090;
pub const DPU_SURFACE_ADD: u32 = 0x40C0;
pub const DPU_40C4: u32 = 0x40C4;
pub const DPU_LUT_ACCESS_CFG: u32 = 0x4100;
pub const DPU_LUT_ACCESS_DATA: u32 = 0x4104;
pub const DPU_LUT_CFG: u32 = 0x4108;
pub const DPU_LUT_INFO: u32 = 0x410C;
pub const DPU_LUT_LE_START: u32 = 0x4110;
pub const DPU_LUT_LE_END: u32 = 0x4114;
pub const DPU_LUT_LO_START: u32 = 0x4118;
pub const DPU_LUT_LO_END: u32 = 0x411C;
pub const DPU_LUT_LE_SLOPE_SCALE: u32 = 0x4120;
pub const DPU_LUT_LE_SLOPE_SHIFT: u32 = 0x4124;
pub const DPU_LUT_LO_SLOPE_SCALE: u32 = 0x4128;
pub const DPU_LUT_LO_SLOPE_SHIFT: u32 = 0x412C;

// ── DPU read-DMA ───────────────────────────────────────────────────────────────
pub const DPU_RDMA_S_POINTER: u32 = 0x5004;
pub const DPU_RDMA_DATA_CUBE_WIDTH: u32 = 0x500C;
pub const DPU_RDMA_DATA_CUBE_HEIGHT: u32 = 0x5010;
pub const DPU_RDMA_DATA_CUBE_CHANNEL: u32 = 0x5014;
pub const DPU_RDMA_SRC_BASE_ADDR: u32 = 0x5018;
pub const DPU_RDMA_BRDMA_CFG: u32 = 0x501C;
pub const DPU_RDMA_BS_BASE_ADDR: u32 = 0x5020;
pub const DPU_RDMA_NRDMA_CFG: u32 = 0x5028;
pub const DPU_RDMA_BN_BASE_ADDR: u32 = 0x502C;
pub const DPU_RDMA_ERDMA_CFG: u32 = 0x5034;
pub const DPU_RDMA_EW_BASE_ADDR: u32 = 0x5038;
pub const DPU_RDMA_EW_SURF_STRIDE: u32 = 0x5040;
pub const DPU_RDMA_FEATURE_MODE_CFG: u32 = 0x5044;
pub const DPU_RDMA_SRC_DMA_CFG: u32 = 0x5048;
pub const DPU_RDMA_SURF_NOTCH: u32 = 0x504C;
pub const DPU_RDMA_PAD_CFG: u32 = 0x5064;
pub const DPU_RDMA_WEIGHT: u32 = 0x5068;
pub const DPU_RDMA_EW_SURF_NOTCH: u32 = 0x506C;

// ── Field builders ─────────────────────────────────────────────────────────────
pub const fn rdma_fmc_burst_len(x: u32) -> u32 {
    (x & 0xF) << 11
}
pub const fn rdma_fmc_mrdma_disable(x: u32) -> u32 {
    (x & 1) << 4
}
pub const fn rdma_fmc_mrdma_fp16tofp32(x: u32) -> u32 {
    (x & 1) << 3
}
pub const fn rdma_fmc_comb_use(x: u32) -> u32 {
    (x & 0x7) << 8
}
pub const fn rdma_fmc_in_precision(x: u32) -> u32 {
    (x & 0x7) << 15
}
pub const fn rdma_fmc_proc_precision(x: u32) -> u32 {
    (x & 0x7) << 5
}

pub const RDMA_WEIGHT_ALL1: u32 = (1 << 24) | (1 << 16) | (1 << 8) | 1;
pub const RDMA_ERDMA_DISABLE: u32 = 1;
pub const EW_CFG_ALU_BASE: u32 = 0x1080_02C0;
pub const EW_CFG_MUL: u32 = 0x1080_03C4;
pub const ERDMA_CFG_16B: u32 = 0x4000_0008;

// ── Precision / conv encodings ─────────────────────────────────────────────────
pub const DIRECT_CONVOLUTION: u32 = 0;
pub const PRECISION_INT8: u32 = 0;
pub const PRECISION_INT16: u32 = 1;
pub const PRECISION_FLOAT16: u32 = 2;
pub const PRECISION_BFLOAT16: u32 = 3;
pub const PRECISION_INT32: u32 = 4;
pub const PRECISION_FLOAT32: u32 = 5;
pub const PRECISION_INT4: u32 = 6;
pub const PRECISION_TF32: u32 = 7;

/// `CBUF` bank size and count on the RK3588.
pub const CBUF_BANK_SIZE: usize = 32768;
pub const CBUF_BANKS: usize = 12;
/// Tiles per job and u64 words reserved per task in a gapped regcmd BO.
pub const BATCH: usize = 64;
pub const RC_STRIDE: usize = 128;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn npuop_layout() {
        let op = NpuOp::new(DPU, 0x4004, 0xE);
        assert_eq!(op.target(), DPU);
        assert_eq!(op.reg(), 0x4004);
        assert_eq!(op.value(), 0xE);
        assert_eq!(op.0, (0x1001u64 << 48) | (0xEu64 << 16) | 0x4004);
    }

    #[test]
    fn fetches_are_constant_builders() {
        assert_eq!(rdma_fmc_burst_len(15), 0xF << 11);
        assert_eq!(rdma_fmc_comb_use(5), 5 << 8);
    }
}
