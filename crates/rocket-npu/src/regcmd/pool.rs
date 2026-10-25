//! PPU pooling register-command emission (RK3588, fp16).
//!
//! Direct translation of `gen_pool_fp16` from `npu_regcmd.c`. A pooling job is a
//! **self-contained PPU + PPU_RDMA job**: PPU_RDMA reads the input NC1HWC2 cube, the
//! PPU reduces each kernel window and writes the output cube in the same layout.
//! There is no CNA/CORE/DPU work and no weight, so the PC enable trailer brings only
//! those two blocks into `op_en` (`0x60`).
//!
//! Conventions (HW-validated upstream): every cube-geometry field is `(value - 1)`
//! and 13 bits wide, every kernel/stride field is also `(value - 1)` and 4 bits wide,
//! cube strides are **bytes** (16-aligned because the fp16 channel atom is `C2 = 8`),
//! and the padding fields are counts (not decremented). Mode is `FLYING_MODE = 1` with
//! `DPU_FLYIN = 0`, i.e. a standalone RDMA-fed pool.
//!
//! # Average pooling has no divider
//!
//! The PPU multiplies the window sum by a per-axis reciprocal and shifts:
//! `avg = sum * recip_w * recip_h * 2^-32`, so the register carries
//! `fp16(65536.0 / k_axis)` as that field's bit pattern ([`ppu_recip_kernel_fp16`]).
//! `k >= 2` is the useful domain (`k = 1` would need 65536, which saturates fp16);
//! the register still encodes it, and the C generator does not reject it. MAX/MIN need
//! no reciprocal — they are idempotent, so the tiled result is bit-exact.
//!
//! The generator emits the fp16 precision pair only (`PROC_PRECISION` /
//! `IN_PRECISION` = 2). The PPU also pools natively in int8, but that is not plumbed
//! upstream either (int8 routes through this fp16 path with a cube-boundary convert).

use crate::error::{Error, Result};

use super::desc::*;

/// `PPU_OPERATION_MODE_CFG.POOLING_METHOD` (NVDLA PDP lineage).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum PoolMethod {
    /// Window average (`0`); needs `recip_w`/`recip_h`.
    #[default]
    Avg = 0,
    /// Window maximum (`1`); padded cells read `-inf` so they never win.
    Max = 1,
    /// Window minimum (`2`); the padded-cell `-inf` fill does not apply.
    Min = 2,
}

impl PoolMethod {
    /// The register code (`POOL_METHOD_*`).
    pub const fn code(self) -> u8 {
        self as u8
    }

    /// Decode a C `POOL_METHOD_*` value.
    pub const fn from_code(code: u8) -> Option<PoolMethod> {
        Some(match code {
            0 => PoolMethod::Avg,
            1 => PoolMethod::Max,
            2 => PoolMethod::Min,
            _ => return None,
        })
    }

    /// The method's name.
    pub const fn name(self) -> &'static str {
        match self {
            PoolMethod::Avg => "avg",
            PoolMethod::Max => "max",
            PoolMethod::Min => "min",
        }
    }
}

/// Parameters of one fp16 pool (the C `pool_params_t`, minus the output pointer).
///
/// Geometry is `u16`/`u8` because the registers are 13-bit (dims, value-1) and 4-bit
/// (kernel/stride, value-1) fields; [`gen_pool_fp16`] checks the ranges exactly as the
/// C generator does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PoolParams {
    /// Channels (per-channel pooling; the cube channel atom is 8 for fp16).
    pub c: u16,
    /// Input height / width.
    pub ih: u16,
    pub iw: u16,
    /// Output height / width (caller-computed).
    pub oh: u16,
    pub ow: u16,
    /// Pooling kernel height / width (>= 1; average pooling needs >= 2 to divide).
    pub kh: u8,
    pub kw: u8,
    /// Window strides.
    pub stride_y: u8,
    pub stride_x: u8,
    /// Padding counts (not decremented).
    pub pad_top: u8,
    pub pad_left: u8,
    pub pad_bottom: u8,
    pub pad_right: u8,
    /// Pooling method.
    pub method: PoolMethod,
    /// Average-pool per-axis reciprocal as the raw fp16 bit pattern
    /// ([`ppu_recip_kernel_fp16`]). Ignored unless `method == Avg`.
    pub recip_w: u32,
    pub recip_h: u32,
    /// NPU IOVA of the input NC1HWC2 cube.
    pub input_dma: u32,
    /// NPU IOVA of the output NC1HWC2 cube.
    pub output_dma: u32,
}

/// `ppu_recip_kernel_fp16`: the fp16 bit pattern of `65536.0 / k`.
///
/// This is the PPU's only division: `avg = sum * recip_w * recip_h * 2^-32`. It is
/// exact for powers of two and quantized otherwise (~1e-3 relative), which is why an
/// average-pool result is gated with a tolerance while MAX/MIN are bit-exact.
pub fn ppu_recip_kernel_fp16(k: u32) -> u32 {
    let k = k.max(1);
    let h = half::f16::from_f32(65536.0 / k as f32);
    h.to_bits() as u32
}

/// Words one pool program needs (30 measured; 64 covers any future field).
pub const MAX_OPS: usize = 64;

/// `gen_pool_fp16`: emit the PPU + PPU_RDMA program for one fp16 pool into `ops`,
/// returning the number of words appended.
///
/// Failure classes follow the C generator: bad geometry (`-1`) is [`Error::shape`],
/// a 13-bit cube-dimension overflow (`-2`) or a 4-bit kernel/stride overflow (`-3`)
/// is [`Error::tiling`].
pub fn gen_pool_fp16(p: &PoolParams, ops: &mut Vec<NpuOp>) -> Result<usize> {
    if p.iw < 1 || p.ih < 1 || p.ow < 1 || p.oh < 1 || p.c < 1 {
        return Err(Error::shape("gen_pool_fp16"));
    }
    if p.kw < 1 || p.kh < 1 || p.stride_x < 1 || p.stride_y < 1 {
        return Err(Error::shape("gen_pool_fp16/kernel"));
    }
    // CUBE_* dims are 13-bit (value-1); kernel/stride are 4-bit (value-1).
    let dims = (p.iw as u32 - 1)
        | (p.ih as u32 - 1)
        | (p.ow as u32 - 1)
        | (p.oh as u32 - 1)
        | (p.c as u32 - 1);
    if dims >> 13 != 0 {
        return Err(Error::tiling("gen_pool_fp16/cube-dims"));
    }
    let geom =
        (p.kw as u32 - 1) | (p.kh as u32 - 1) | (p.stride_x as u32 - 1) | (p.stride_y as u32 - 1);
    if geom >> 4 != 0 {
        return Err(Error::tiling("gen_pool_fp16/kernel-stride"));
    }

    const C2: u32 = 8; // fp16 feature-cube channel atom
    const ESZ: u32 = 2; // fp16 element bytes
    let in_line_stride = p.iw as u32 * C2 * ESZ; // bytes/row
    let in_surf_stride = p.ih as u32 * p.iw as u32 * C2 * ESZ; // bytes/C-plane
    let out_surf_stride = p.oh as u32 * p.ow as u32 * C2 * ESZ;

    let start = ops.len();
    let mut push = |target, reg, value| ops.push(NpuOp::new(target, reg, value));

    // Arm the PPU + PPU_RDMA single-register groups (the same 0xE as the DPU path).
    push(PPU, PPU_S_POINTER, 0xE);
    push(PPU_RDMA, PPU_RDMA_S_POINTER, 0xE);

    // PPU input/output cube geometry (field = dim - 1).
    push(PPU, PPU_DATA_CUBE_IN_WIDTH, p.iw as u32 - 1);
    push(PPU, PPU_DATA_CUBE_IN_HEIGHT, p.ih as u32 - 1);
    push(PPU, PPU_DATA_CUBE_IN_CHANNEL, p.c as u32 - 1);
    push(PPU, PPU_DATA_CUBE_OUT_WIDTH, p.ow as u32 - 1);
    push(PPU, PPU_DATA_CUBE_OUT_HEIGHT, p.oh as u32 - 1);
    push(PPU, PPU_DATA_CUBE_OUT_CHANNEL, p.c as u32 - 1);

    // Operation mode: FLYING_MODE always, POOLING_METHOD in bits[1:0].
    push(
        PPU,
        PPU_OPERATION_MODE_CFG,
        PPU_FLYING_MODE | (p.method.code() as u32 & 0x3),
    );

    // Kernel + stride, each field = value-1: KW[3:0] KH[11:8] SX[19:16] SY[23:20].
    let kcfg = (p.kw as u32 - 1) & 0xF
        | ((p.kh as u32 - 1) & 0xF) << 8
        | ((p.stride_x as u32 - 1) & 0xF) << 16
        | ((p.stride_y as u32 - 1) & 0xF) << 20;
    push(PPU, PPU_POOLING_KERNEL_CFG, kcfg);

    // Average reciprocal: per-axis fp16(65536/k) in its 17-bit field; 0 otherwise.
    let avg = p.method == PoolMethod::Avg;
    push(
        PPU,
        PPU_RECIP_KERNEL_WIDTH,
        if avg { p.recip_w & 0x1_FFFF } else { 0 },
    );
    push(
        PPU,
        PPU_RECIP_KERNEL_HEIGHT,
        if avg { p.recip_h & 0x1_FFFF } else { 0 },
    );

    // Padding, counts not value-1: L[2:0] T[6:4] R[10:8] B[14:12].
    let pad = (p.pad_left as u32 & 0x7)
        | ((p.pad_top as u32 & 0x7) << 4)
        | ((p.pad_right as u32 & 0x7) << 8)
        | ((p.pad_bottom as u32 & 0x7) << 12);
    push(PPU, PPU_POOLING_PADDING_CFG, pad);
    // Pad fill: 0 for average, -inf for max so a padded cell never wins.
    let fill = if p.method == PoolMethod::Max && pad != 0 {
        PPU_PAD_VALUE_NEG_INF
    } else {
        0
    };
    push(PPU, PPU_PADDING_VALUE_1_CFG, fill);
    push(PPU, PPU_PADDING_VALUE_2_CFG, 0);

    // Output write-out. DST_BASE_ADDR holds the raw IOVA (the field is [31:4] and BOs
    // are page-aligned); DATA_FORMAT mirrors the output surface stride into
    // INDEX_ADD[31:4] and sets PROC_PRECISION[2:0] = 2 (fp16). The surface stride is
    // 16-aligned, so the OR is clean.
    push(PPU, PPU_DST_BASE_ADDR, p.output_dma);
    push(PPU, PPU_DST_SURF_STRIDE, out_surf_stride);
    push(PPU, PPU_DATA_FORMAT, out_surf_stride | 0x2);
    push(PPU, PPU_MISC_CTRL, 0x3); // BURST_LEN = 3

    // PPU_RDMA: the input feed (own source base + strides; IN_PRECISION = 2, fp16).
    push(PPU_RDMA, PPU_RDMA_CUBE_IN_WIDTH, p.iw as u32 - 1);
    push(PPU_RDMA, PPU_RDMA_CUBE_IN_HEIGHT, p.ih as u32 - 1);
    push(PPU_RDMA, PPU_RDMA_CUBE_IN_CHANNEL, p.c as u32 - 1);
    push(PPU_RDMA, PPU_RDMA_SRC_BASE_ADDR, p.input_dma);
    push(PPU_RDMA, PPU_RDMA_SRC_LINE_STRIDE, in_line_stride);
    push(PPU_RDMA, PPU_RDMA_SRC_SURF_STRIDE, in_surf_stride);
    push(PPU_RDMA, PPU_RDMA_DATA_FORMAT, 0x2);

    // PC trailer: enable PPU + PPU_RDMA only. The framing mirrors the validated
    // matmul trailer, so the chained layout can rewrite the filler word.
    push(OP_NONE, 0x0, 0x0);
    push(PC, PC_REGISTER_AMOUNTS, 0x0);
    push(OP_40, 0x0, 0x0);
    push(OP_ENABLE, PC_OPERATION_ENABLE, PC_ENABLE_PPU_PPU_RDMA);

    Ok(ops.len() - start)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One pinned program: a 3x7x7 -> 3x1x1 average pool (kernel 7, no pad), the
    /// smallest shape that exercises every field with a non-trivial value.
    #[test]
    fn encodes_a_pinned_average_pool() {
        let p = PoolParams {
            c: 3,
            ih: 7,
            iw: 7,
            oh: 1,
            ow: 1,
            kh: 7,
            kw: 7,
            stride_y: 7,
            stride_x: 7,
            pad_top: 0,
            pad_left: 0,
            pad_bottom: 0,
            pad_right: 0,
            method: PoolMethod::Avg,
            recip_w: ppu_recip_kernel_fp16(7),
            recip_h: ppu_recip_kernel_fp16(7),
            input_dma: 0x0001_0000,
            output_dma: 0x0002_0000,
        };
        let mut ops = Vec::new();
        let n = gen_pool_fp16(&p, &mut ops).expect("emit");
        assert_eq!(n, 30);
        assert_eq!(ops.len(), 30);

        let recip = ppu_recip_kernel_fp16(7);
        // fp16(65536/7) = 9362.2857 -> 9360.0 (the spacing is 8 in that binade) =
        // 0x7092; cross-checked against the C's ppu_recip_kernel_fp16 on the host.
        assert_eq!(recip, 0x7092);
        let expected = [
            NpuOp::new(PPU, PPU_S_POINTER, 0xE).0,
            NpuOp::new(PPU_RDMA, PPU_RDMA_S_POINTER, 0xE).0,
            NpuOp::new(PPU, PPU_DATA_CUBE_IN_WIDTH, 6).0,
            NpuOp::new(PPU, PPU_DATA_CUBE_IN_HEIGHT, 6).0,
            NpuOp::new(PPU, PPU_DATA_CUBE_IN_CHANNEL, 2).0,
            NpuOp::new(PPU, PPU_DATA_CUBE_OUT_WIDTH, 0).0,
            NpuOp::new(PPU, PPU_DATA_CUBE_OUT_HEIGHT, 0).0,
            NpuOp::new(PPU, PPU_DATA_CUBE_OUT_CHANNEL, 2).0,
            NpuOp::new(PPU, PPU_OPERATION_MODE_CFG, 0x10).0,
            NpuOp::new(
                PPU,
                PPU_POOLING_KERNEL_CFG,
                6 | (6 << 8) | (6 << 16) | (6 << 20),
            )
            .0,
            NpuOp::new(PPU, PPU_RECIP_KERNEL_WIDTH, recip).0,
            NpuOp::new(PPU, PPU_RECIP_KERNEL_HEIGHT, recip).0,
            NpuOp::new(PPU, PPU_POOLING_PADDING_CFG, 0).0,
            NpuOp::new(PPU, PPU_PADDING_VALUE_1_CFG, 0).0,
            NpuOp::new(PPU, PPU_PADDING_VALUE_2_CFG, 0).0,
            NpuOp::new(PPU, PPU_DST_BASE_ADDR, 0x0002_0000).0,
            NpuOp::new(PPU, PPU_DST_SURF_STRIDE, 16).0,
            NpuOp::new(PPU, PPU_DATA_FORMAT, 16 | 0x2).0,
            NpuOp::new(PPU, PPU_MISC_CTRL, 0x3).0,
            NpuOp::new(PPU_RDMA, PPU_RDMA_CUBE_IN_WIDTH, 6).0,
            NpuOp::new(PPU_RDMA, PPU_RDMA_CUBE_IN_HEIGHT, 6).0,
            NpuOp::new(PPU_RDMA, PPU_RDMA_CUBE_IN_CHANNEL, 2).0,
            NpuOp::new(PPU_RDMA, PPU_RDMA_SRC_BASE_ADDR, 0x0001_0000).0,
            NpuOp::new(PPU_RDMA, PPU_RDMA_SRC_LINE_STRIDE, 7 * 8 * 2).0,
            NpuOp::new(PPU_RDMA, PPU_RDMA_SRC_SURF_STRIDE, 7 * 7 * 8 * 2).0,
            NpuOp::new(PPU_RDMA, PPU_RDMA_DATA_FORMAT, 0x2).0,
            NpuOp::new(OP_NONE, 0x0, 0x0).0,
            NpuOp::new(PC, PC_REGISTER_AMOUNTS, 0x0).0,
            NpuOp::new(OP_40, 0x0, 0x0).0,
            NpuOp::new(OP_ENABLE, PC_OPERATION_ENABLE, 0x60).0,
        ];
        assert_eq!(ops.iter().map(|o| o.0).collect::<Vec<_>>(), expected);
    }

    /// Max with padding sets the `-inf` fill; min never does, and neither emits a
    /// reciprocal.
    #[test]
    fn pads_and_reciprocals_follow_the_method() {
        let base = PoolParams {
            c: 8,
            ih: 5,
            iw: 5,
            oh: 3,
            ow: 3,
            kh: 3,
            kw: 3,
            stride_y: 3,
            stride_x: 3,
            pad_top: 1,
            pad_left: 1,
            pad_bottom: 1,
            pad_right: 1,
            method: PoolMethod::Max,
            recip_w: 0xDEAD,
            recip_h: 0xBEEF,
            input_dma: 0x1000,
            output_dma: 0x2000,
        };
        let mut max_ops = Vec::new();
        gen_pool_fp16(&base, &mut max_ops).unwrap();
        assert_eq!(max_ops[8].value(), 0x10 | 1); // FLYING_MODE | MAX
        assert_eq!(max_ops[10].value(), 0); // recip_w ignored
        assert_eq!(max_ops[11].value(), 0); // recip_h ignored
        assert_eq!(max_ops[12].value(), 0x1111); // pad L=1 T=1 R=1 B=1

        let min = PoolParams {
            method: PoolMethod::Min,
            ..base
        };
        let mut min_ops = Vec::new();
        gen_pool_fp16(&min, &mut min_ops).unwrap();
        assert_eq!(min_ops[8].value(), 0x10 | 2);
        assert_eq!(min_ops[13].value(), 0); // min keeps the 0 pad fill
        assert_eq!(min_ops[10].value(), 0);

        let avg = PoolParams {
            method: PoolMethod::Avg,
            recip_w: ppu_recip_kernel_fp16(3),
            recip_h: ppu_recip_kernel_fp16(3),
            ..base
        };
        let mut avg_ops = Vec::new();
        gen_pool_fp16(&avg, &mut avg_ops).unwrap();
        assert_eq!(avg_ops[13].value(), 0); // avg never fills -inf
        assert_eq!(avg_ops[10].value(), ppu_recip_kernel_fp16(3));
    }

    #[test]
    fn range_checks_match_the_c_classes() {
        let ok = PoolParams {
            c: 8,
            ih: 4,
            iw: 4,
            oh: 2,
            ow: 2,
            kh: 2,
            kw: 2,
            stride_y: 2,
            stride_x: 2,
            method: PoolMethod::Avg,
            recip_w: ppu_recip_kernel_fp16(2),
            recip_h: ppu_recip_kernel_fp16(2),
            input_dma: 0x1000,
            output_dma: 0x2000,
            ..PoolParams::default()
        };
        let mut ops = Vec::new();
        assert!(gen_pool_fp16(&ok, &mut ops).is_ok());

        let zero_w = PoolParams { iw: 0, ..ok };
        assert_eq!(
            gen_pool_fp16(&zero_w, &mut ops).unwrap_err().status(),
            crate::error::Status::Shape
        );
        let zero_k = PoolParams { kh: 0, ..ok };
        assert_eq!(
            gen_pool_fp16(&zero_k, &mut ops).unwrap_err().status(),
            crate::error::Status::Shape
        );
        let wide_k = PoolParams { kw: 17, ..ok };
        assert_eq!(
            gen_pool_fp16(&wide_k, &mut ops).unwrap_err().status(),
            crate::error::Status::Tiling
        );
        let big_c = PoolParams { c: 8193, ..ok };
        assert_eq!(
            gen_pool_fp16(&big_c, &mut ops).unwrap_err().status(),
            crate::error::Status::Tiling
        );
        // 8192 channels is the last encodable value (field = 8191).
        let max_c = PoolParams { c: 8192, ..ok };
        assert!(gen_pool_fp16(&max_c, &mut ops).is_ok());
    }

    #[test]
    fn reciprocals_are_the_fp16_bits() {
        assert_eq!(
            ppu_recip_kernel_fp16(2),
            half::f16::from_f32(32768.0).to_bits() as u32
        );
        assert_eq!(
            ppu_recip_kernel_fp16(16),
            half::f16::from_f32(4096.0).to_bits() as u32
        );
        // k=1 saturates (the C header's warning): 65536 is beyond fp16's max.
        assert_eq!(
            ppu_recip_kernel_fp16(1),
            half::f16::INFINITY.to_bits() as u32
        );
        // k=0 clamps to 1, as the C does.
        assert_eq!(ppu_recip_kernel_fp16(0), ppu_recip_kernel_fp16(1));
    }
}
