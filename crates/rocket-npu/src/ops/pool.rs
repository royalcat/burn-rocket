//! MaxPool / AveragePool on the PPU (`rocket_pool.c`).
//!
//! A pooling job is a **self-contained PPU + PPU_RDMA job**: the host scatters the
//! input feature into the NC1HWC2 cube (`C2 = 8`, the same cube the conv path uses),
//! the PPU reduces each kernel window per channel and writes the output cube in the
//! same layout, and the host de-scatters it. There are no weights, no CNA/CORE/DPU.
//!
//! Layouts (row-major, batch 1, channels-major — the same as
//! [`crate::ops::conv::conv2d`]):
//!
//! ```text
//! input   in [C][IH][IW]
//! output  out[C][OH][OW]
//! out[c][oh][ow] = reduce_{kh,kw} in[c][oh*stride_y + kh - pad_top,
//!                                     ow*stride_x + kw - pad_left]
//! ```
//!
//! Out-of-range reads do not contribute to a maximum (the pad fill is `-inf`); an
//! average divides by `kh*kw` — **count-include-pad**, the only divisor the PPU can do.
//!
//! # Average pooling has no divider
//!
//! The PPU multiplies the window sum by a per-axis reciprocal and shifts:
//! `avg = sum * recip_w * recip_h * 2^-32`, so the register carries
//! `fp16(65536.0 / k_axis)` ([`crate::regcmd::pool::ppu_recip_kernel_fp16`]). At pad 0
//! that is the exact mean; with padding it differs from TFLite's `AVERAGE_POOL_2D`
//! (which divides by the valid tap count) at the border windows. `avg_exclude_pad` is
//! therefore **refused** by every RK3588 entry (`-10`), exactly as in C: the two are
//! the same function only at pad 0 and a caller that asked has a padded window in mind.
//!
//! # int8 / uint8
//!
//! int8 and uint8 pooling run **through the fp16 path**, as the C library does: every
//! int8 value is exact in fp16, so lifting the feature, running the proven fp16 PPU
//! job and narrowing back is correct by construction — MAX is bit-exact, an average
//! matches the `fp16(65536/k)` reciprocal and is then rounded. uint8 recentres by
//! `-128` (an order-preserving `byte ^ 0x80` map) so the fp16 domain stays small, and
//! adds it back with a `[0, 255]` clamp.
//!
//! The PPU *does* have a native int8 pooling precision (`PPU_DATA_FORMAT = 0` over a
//! packed `C2 = 16` cube, bit-exact for MAX/MIN/AVG with an integer Q16 reciprocal),
//! but the C library does not plumb it — `gen_pool_fp16` emits the fp16 pair only —
//! and this port follows the reference rather than inventing a second encoding. That
//! is also why the parity harness has no separate int8 pool program to compare.

use crate::Device;
use crate::device::uapi::DrmRocketTask;
use crate::error::{Error, Result};
use crate::ops::reduce::feature_data;
use crate::regcmd::pool::{PoolMethod, PoolParams, gen_pool_fp16, ppu_recip_kernel_fp16};

/// The C `rocket_pool_fp16_plan`'s refusals that are its own local codes rather than
/// `ROCKET_E_*` classes.
pub const POOL_ERR_METHOD: i32 = -4;
/// Output extent that collapses.
pub const POOL_ERR_EXTENT: i32 = -5;
/// A cube-geometry field overflow (13-bit, value-1).
pub const POOL_ERR_CUBE: i32 = -6;
/// A kernel/stride field overflow (4-bit, value-1).
pub const POOL_ERR_KERNEL: i32 = -7;
/// A padding field overflow (3-bit count).
pub const POOL_ERR_PAD: i32 = -8;
/// An average kernel below 2 on an axis (the reciprocal would saturate fp16).
pub const POOL_ERR_AVG_K: i32 = -9;
/// `avg_exclude_pad`, which no RK3588 entry implements.
pub const POOL_ERR_EXCLUDE_PAD: i32 = -10;

/// Slack past the scattered cube, so a DMA burst over-read cannot run off the BO (the
/// C `POOL_BO_SLACK`).
const POOL_BO_SLACK: usize = 32768;
/// Words one pool program needs (30 measured; the generator's own bound is 64).
const RC_WORDS: usize = crate::regcmd::pool::MAX_OPS;
/// The C's hardcoded two-second wait for a one-shot op.
const WAIT_NS: u64 = 2_000_000_000;

/// `rocket_pool_desc`: the pooling geometry and method, no buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolDesc {
    /// Channels / input height / input width.
    pub c: usize,
    pub ih: usize,
    pub iw: usize,
    /// Pooling kernel height / width.
    pub kh: usize,
    pub kw: usize,
    /// Window strides.
    pub stride_y: usize,
    pub stride_x: usize,
    /// Padding counts (asymmetric pads are legal here, unlike the CNA's leading-only
    /// pad: the PPU has all four).
    pub pad_top: usize,
    pub pad_left: usize,
    pub pad_bottom: usize,
    pub pad_right: usize,
    /// Max or average. `Min` is encodable by the generator but refused by the runtime
    /// plan (`-4`), exactly as in C.
    pub method: PoolMethod,
    /// Divide an average by the tap count that fell inside the plane instead of
    /// `kh*kw`. The RK3576 int8 path implements it; every RK3588 entry refuses it
    /// (`-10`), so this port refuses it too.
    pub avg_exclude_pad: bool,
}

impl Default for PoolDesc {
    fn default() -> Self {
        Self {
            c: 0,
            ih: 0,
            iw: 0,
            kh: 1,
            kw: 1,
            stride_y: 1,
            stride_x: 1,
            pad_top: 0,
            pad_left: 0,
            pad_bottom: 0,
            pad_right: 0,
            method: PoolMethod::Max,
            avg_exclude_pad: false,
        }
    }
}

impl PoolDesc {
    /// A zero-padded pool with unit strides.
    pub fn new(c: usize, ih: usize, iw: usize, kh: usize, kw: usize) -> Self {
        Self {
            c,
            ih,
            iw,
            kh,
            kw,
            ..Default::default()
        }
    }

    /// `rocket_pool_oh`: the derived output height.
    pub fn oh(&self) -> isize {
        pool_out_dim(
            self.ih,
            self.kh,
            self.stride_y,
            self.pad_top,
            self.pad_bottom,
        )
    }

    /// `rocket_pool_ow`: the derived output width.
    pub fn ow(&self) -> isize {
        pool_out_dim(
            self.iw,
            self.kw,
            self.stride_x,
            self.pad_left,
            self.pad_right,
        )
    }
}

/// `rocket_pool_out_dim`: one output axis (floor; VALID / explicit-pad semantics).
///
/// `isize`, because a kernel that reaches past the padded extent yields a non-positive
/// extent and the planner's `-5` is about that value, not about a saturated zero.
pub fn pool_out_dim(input: usize, k: usize, stride: usize, pad_lo: usize, pad_hi: usize) -> isize {
    let stride = stride as isize;
    if stride == 0 {
        // The C divides by the caller's stride; a zero stride is caught by the plan
        // before this is called. Guard the division rather than panicking here.
        return 0;
    }
    (input as isize + pad_lo as isize + pad_hi as isize - k as isize) / stride + 1
}

/// `rocket_pool_fp16_plan`, as the C's raw return code (`0`, `-3`, `-4`..`-10`).
///
/// The classes are the C's: `-3` (`ROCKET_E_NOMEM`) a non-positive channel/extent,
/// `-5` (`ROCKET_E_UNSUPPORTED`) a non-positive kernel or stride, `-4` a method outside
/// MAX/AVG, `-10` `avg_exclude_pad`, `-5` a collapsed output extent, `-6`/`-7`/`-8` a
/// cube/kernel/pad field overflow, and `-9` an average kernel below 2.
pub fn pool_fp16_plan_code(d: &PoolDesc) -> i32 {
    if d.c < 1 || d.ih < 1 || d.iw < 1 {
        return crate::Status::NoMem.code();
    }
    if d.kh < 1 || d.kw < 1 || d.stride_y < 1 || d.stride_x < 1 {
        return crate::Status::Unsupported.code();
    }
    if d.method != PoolMethod::Max && d.method != PoolMethod::Avg {
        return POOL_ERR_METHOD;
    }
    if d.avg_exclude_pad {
        return POOL_ERR_EXCLUDE_PAD;
    }
    let (oh, ow) = (d.oh(), d.ow());
    if oh < 1 || ow < 1 {
        return POOL_ERR_EXTENT;
    }
    // CUBE_* dims are 13-bit (value-1); kernel/stride are 4-bit (value-1); pad 3-bit.
    if ((d.iw - 1) | (d.ih - 1) | (ow as usize - 1) | (oh as usize - 1) | (d.c - 1)) >> 13 != 0 {
        return POOL_ERR_CUBE;
    }
    if ((d.kw - 1) | (d.kh - 1) | (d.stride_x - 1) | (d.stride_y - 1)) >> 4 != 0 {
        return POOL_ERR_KERNEL;
    }
    if (d.pad_left | d.pad_top | d.pad_right | d.pad_bottom) >> 3 != 0 {
        return POOL_ERR_PAD;
    }
    // The average reciprocal fp16(65536/k) overflows fp16 at k = 1, so an average needs
    // k >= 2 per axis (a 1-wide average kernel is degenerate anyway). MAX is fine at 1.
    if d.method == PoolMethod::Avg && (d.kw < 2 || d.kh < 2) {
        return POOL_ERR_AVG_K;
    }
    0
}

/// `rocket_pool_fp16_plan`: validate a pool descriptor.
pub fn pool_fp16_plan(d: &PoolDesc) -> Result<()> {
    match pool_fp16_plan_code(d) {
        0 => Ok(()),
        c if c == crate::Status::NoMem.code() => Err(Error::nomem("rocket_pool_fp16_plan/dims")),
        c if c == crate::Status::Unsupported.code() => {
            Err(Error::unsupported("rocket_pool_fp16_plan/geometry"))
        }
        POOL_ERR_EXCLUDE_PAD => Err(Error::unsupported("rocket_pool_fp16_plan/exclude-pad")),
        c if c == POOL_ERR_METHOD || c == POOL_ERR_AVG_K => Err(Error::coded(
            crate::Status::Shape,
            c,
            "rocket_pool_fp16_plan",
        )),
        c => Err(Error::coded(
            crate::Status::Tiling,
            c,
            "rocket_pool_fp16_plan",
        )),
    }
}

/// `rocket_pool_ref_fp16`: the fp32-accumulate host reference.
///
/// An average divides by `kh*kw` (count-include-pad) to match the PPU reciprocal; a max
/// uses `-inf` for out-of-range reads.
pub fn pool_ref_fp16(d: &PoolDesc, input: &[half::f16], out: &mut [half::f16]) {
    let (c, ih, iw) = (d.c, d.ih, d.iw);
    let (oh, ow) = (d.oh().max(0) as usize, d.ow().max(0) as usize);
    for ch in 0..c {
        for y in 0..oh {
            for x in 0..ow {
                let mut acc = if d.method == PoolMethod::Max {
                    f32::NEG_INFINITY
                } else {
                    0.0f32
                };
                for k in 0..d.kh {
                    let yy = (y * d.stride_y + k) as isize - d.pad_top as isize;
                    if yy < 0 || yy >= ih as isize {
                        continue;
                    }
                    for j in 0..d.kw {
                        let xx = (x * d.stride_x + j) as isize - d.pad_left as isize;
                        if xx < 0 || xx >= iw as isize {
                            continue;
                        }
                        let v = input[(ch * ih + yy as usize) * iw + xx as usize].to_f32();
                        if d.method == PoolMethod::Max {
                            if v > acc {
                                acc = v;
                            }
                        } else {
                            acc += v;
                        }
                    }
                }
                let o = if d.method == PoolMethod::Avg {
                    acc / (d.kh * d.kw) as f32
                } else {
                    acc
                };
                out[(ch * oh + y) * ow + x] = half::f16::from_f32(o);
            }
        }
    }
}

/// `rocket_pool_ref_int8`: the int8 host reference.
///
/// int8 values are exactly representable in fp16, so the int8 pool is the fp16 pool of
/// the same values rounded back to int8: MAX is the exact integer max, an average
/// matches the `fp16(65536/k)` reciprocal path. The reference is therefore the fp16
/// oracle composed with a round-to-nearest + clamp.
pub fn pool_ref_int8(d: &PoolDesc, input: &[i8], out: &mut [i8]) {
    let ni = d.c * d.ih * d.iw;
    let no = d.c * d.oh().max(0) as usize * d.ow().max(0) as usize;
    let mut fi = vec![half::f16::ZERO; ni];
    for (dst, &v) in fi.iter_mut().zip(input) {
        *dst = half::f16::from_f32(v as f32);
    }
    let mut fo = vec![half::f16::ZERO; no];
    pool_ref_fp16(d, &fi, &mut fo);
    for (dst, v) in out.iter_mut().zip(&fo) {
        let r = v.to_f32().round_ties_even();
        *dst = r.clamp(-128.0, 127.0) as i8;
    }
}

/// The port's own uint8 reference (the C library has none): int8 recentred by 128.
pub fn pool_ref_uint8(d: &PoolDesc, input: &[u8], out: &mut [u8]) {
    let ni = d.c * d.ih * d.iw;
    let no = d.c * d.oh().max(0) as usize * d.ow().max(0) as usize;
    let signed: Vec<i8> = input.iter().map(|&v| (v as i32 - 128) as i8).collect();
    debug_assert_eq!(signed.len(), ni);
    let mut mid = vec![0i8; no];
    pool_ref_int8(d, &signed, &mut mid);
    for (dst, &v) in out.iter_mut().zip(&mid) {
        *dst = (v as i32 + 128).clamp(0, 255) as u8;
    }
}

fn bo_words(bo: &crate::device::Bo) -> &[half::f16] {
    let bytes = bo.as_slice();
    // SAFETY: f16 is repr(transparent) over u16 with align 2 and every prefix of the
    // mapping is initialized.
    unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<half::f16>(), bytes.len() / 2) }
}

fn bo_words_mut(bo: &mut crate::device::Bo) -> &mut [half::f16] {
    let len = bo.len();
    let ptr = bo.as_ptr();
    // SAFETY: exclusive borrow of the BO, page-aligned mapping, 2-byte elements.
    unsafe { std::slice::from_raw_parts_mut(ptr.cast::<half::f16>(), len / 2) }
}

/// The fp16 PPU pool job (`rocket_pool_fp16`).
///
/// `in` is `[C][IH][IW]`, `out` is `[C][OH][OW]`, both row-major fp16.
pub fn pool_fp16(
    dev: &Device,
    d: &PoolDesc,
    input: &[half::f16],
    out: &mut [half::f16],
) -> Result<()> {
    pool_fp16_plan(d)?;
    let (c, ih, iw) = (d.c, d.ih, d.iw);
    let (oh, ow) = (d.oh() as usize, d.ow() as usize);

    let c1 = c.div_ceil(8);
    let in_cube = c1 * ih * iw * 8;
    let out_cube = c1 * oh * ow * 8;

    let _guard = dev.alloc_bo(4096)?; // keeps IOVA 0 unused, as in C
    let mut src = dev.alloc_bo(in_cube * 2 + POOL_BO_SLACK)?;
    let mut rc = dev.alloc_bo(RC_WORDS * 8)?;
    let mut dst = dev.alloc_bo(out_cube * 2 + POOL_BO_SLACK)?;
    crate::matmul::bos::check_iova(&[&src, &rc, &dst])?;

    // Scatter the input feature [C][IH][IW] into the NC1HWC2 cube (C2 = 8).
    src.prep(0)?;
    {
        let words = bo_words_mut(&mut src);
        words.fill(half::f16::ZERO);
        for ch in 0..c {
            for y in 0..ih {
                for x in 0..iw {
                    words[feature_data(c, ih, iw, 8, ch + 1, y + 1, x + 1)] =
                        input[(ch * ih + y) * iw + x];
                }
            }
        }
    }
    src.fini()?;

    let (rw, rh) = if d.method == PoolMethod::Avg {
        (
            ppu_recip_kernel_fp16(d.kw as u32),
            ppu_recip_kernel_fp16(d.kh as u32),
        )
    } else {
        (0, 0)
    };
    let p = PoolParams {
        c: c as u16,
        ih: ih as u16,
        iw: iw as u16,
        oh: oh as u16,
        ow: ow as u16,
        kh: d.kh as u8,
        kw: d.kw as u8,
        stride_y: d.stride_y as u8,
        stride_x: d.stride_x as u8,
        pad_top: d.pad_top as u8,
        pad_left: d.pad_left as u8,
        pad_bottom: d.pad_bottom as u8,
        pad_right: d.pad_right as u8,
        method: d.method,
        recip_w: rw,
        recip_h: rh,
        input_dma: src.dma_address() as u32,
        output_dma: dst.dma_address() as u32,
    };
    let mut ops = Vec::with_capacity(RC_WORDS);
    let count = gen_pool_fp16(&p, &mut ops)?;
    if count > RC_WORDS {
        return Err(Error::tiling("rocket_pool_fp16/regcmd"));
    }
    rc.prep(0)?;
    {
        let bytes = rc.as_mut_slice();
        for (i, op) in ops.iter().enumerate() {
            bytes[i * 8..i * 8 + 8].copy_from_slice(&op.0.to_ne_bytes());
        }
    }
    rc.fini()?;

    // Prezero the output: a partial channel group's dead lanes are read back with the
    // live ones and the de-scatter indexes past them.
    dst.prep(0)?;
    bo_words_mut(&mut dst).fill(half::f16::ZERO);
    dst.fini()?;

    let task = DrmRocketTask {
        regcmd: rc.dma_address() as u32,
        regcmd_count: count as u32,
    };
    // This program finishes on the PPU, not the DPU; naming the block is what lets a
    // driver that waits for the named block retire the job at all.
    let flags = if dev.ppu_done_supported() {
        crate::device::uapi::DRM_ROCKET_JOB_PPU_DONE
    } else {
        0
    };
    dev.submit(
        &[task],
        &[src.handle(), rc.handle()],
        &[dst.handle()],
        flags,
    )?;

    dst.prep(WAIT_NS)?;
    {
        let words = bo_words(&dst);
        for ch in 0..c {
            for y in 0..oh {
                for x in 0..ow {
                    out[(ch * oh + y) * ow + x] =
                        words[feature_data(c, oh, ow, 8, ch + 1, y + 1, x + 1)];
                }
            }
        }
    }
    dst.fini()?;
    Ok(())
}

/// `rocket_pool_int8_core`: int8/uint8 pooling routed through the fp16 PPU job.
///
/// Every int8 value is exact in fp16, so lifting, running the proven fp16 program and
/// narrowing back is correct: MAX bit-exact, an average within the reciprocal tolerance
/// and then rounded. `recenter` selects the uint8 byte interpretation and the `[0, 255]`
/// output clamp.
/// Narrow one pool result back to the int8/uint8 byte domain.
///
/// `lrintf` semantics (nearest, ties to even) then the zero-point re-centring and the
/// dtype clamp. The uint8 arm must go through `u8` — a `f32 as i8` cast would saturate
/// at 127 and turn every bright pixel into mid-grey.
fn narrow_pool(f: f32, bias: i32, recenter: bool) -> i8 {
    let r = f.round_ties_even() + bias as f32;
    if recenter {
        (r.clamp(0.0, 255.0) as u8) as i8
    } else {
        r.clamp(-128.0, 127.0) as i8
    }
}

fn pool_int8_core(
    dev: &Device,
    d: &PoolDesc,
    input: &[i8],
    out: &mut [i8],
    recenter: bool,
) -> Result<()> {
    // The geometry rules are dtype-independent.
    pool_fp16_plan(d)?;
    let ni = d.c * d.ih * d.iw;
    let no = d.c * d.oh() as usize * d.ow() as usize;
    let bias = if recenter { 128 } else { 0 };

    let mut fi = vec![half::f16::ZERO; ni];
    for (dst, &v) in fi.iter_mut().zip(input) {
        let val = if recenter { (v as u8) as i32 } else { v as i32 };
        *dst = half::f16::from_f32((val - bias) as f32);
    }
    let mut fo = vec![half::f16::ZERO; no];
    pool_fp16(dev, d, &fi, &mut fo)?;
    for (dst, v) in out.iter_mut().zip(&fo) {
        *dst = narrow_pool(v.to_f32(), bias, recenter);
    }
    Ok(())
}

/// `rocket_pool_int8`: signed int8 pooling (in/out `[C][IH][IW]` / `[C][OH][OW]`).
pub fn pool_int8(dev: &Device, d: &PoolDesc, input: &[i8], out: &mut [i8]) -> Result<()> {
    pool_int8_core(dev, d, input, out, false)
}

/// `rocket_pool_uint8`: uint8 pooling, recentred around the signed PPU compare.
///
/// The byte layout is passed through untouched (`out` is the caller's `[u8]` seen as
/// `[i8]`), so `v` is the raw byte and the recentring happens inside the core.
pub fn pool_uint8(dev: &Device, d: &PoolDesc, input: &[u8], out: &mut [u8]) -> Result<()> {
    let n = d.c * d.ih * d.iw;
    let no = d.c * d.oh() as usize * d.ow() as usize;
    if input.len() < n || out.len() < no {
        return Err(Error::shape("rocket_pool_uint8/buffers"));
    }
    // SAFETY: u8 and i8 have the same size and no validity requirement beyond a byte;
    // the slices are the caller's and stay alive for the call.
    let src = unsafe { std::slice::from_raw_parts(input.as_ptr().cast::<i8>(), n) };
    let dst = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<i8>(), no) };
    pool_int8_core(dev, d, src, dst, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn desc(c: usize, ih: usize, iw: usize, kh: usize, kw: usize) -> PoolDesc {
        PoolDesc::new(c, ih, iw, kh, kw)
    }

    #[test]
    fn output_dims_follow_the_padded_extent() {
        assert_eq!(pool_out_dim(7, 7, 7, 0, 0), 1);
        assert_eq!(pool_out_dim(5, 3, 1, 1, 1), 5); // stride 1, symmetric pad 1
        // C integer division truncates toward zero: (-1)/2 == 0, so this degenerate
        // extent is 1, not 0 — the plan accepts it (the C does too).
        assert_eq!(pool_out_dim(4, 5, 2, 0, 0), 1);
        assert_eq!(pool_out_dim(4, 5, 1, 0, 0), 0); // collapsed
        // Truncation toward zero, as the C's int division does.
        assert_eq!(pool_out_dim(1, 3, 2, 0, 0), 0);
        let mut d = desc(8, 5, 5, 3, 3);
        d.stride_y = 3;
        d.stride_x = 3;
        d.pad_top = 1;
        d.pad_left = 1;
        d.pad_bottom = 1;
        d.pad_right = 1;
        // (5 + 1 + 1 - 3) / 3 + 1 = 2 on each axis.
        assert_eq!((d.oh(), d.ow()), (2, 2));
    }

    #[test]
    fn plan_accepts_and_refuses_the_c_classes() {
        // The global-pool shape and a padded, overlapping average.
        assert_eq!(pool_fp16_plan_code(&desc(8, 7, 7, 7, 7)), 0);
        let mut d = desc(8, 5, 5, 3, 3);
        d.stride_y = 1;
        d.stride_x = 1;
        d.method = PoolMethod::Avg;
        assert_eq!(pool_fp16_plan_code(&d), 0);
        // avg needs k >= 2 per axis; max is fine at 1.
        let mut d = desc(2, 4, 4, 1, 1);
        d.method = PoolMethod::Avg;
        assert_eq!(pool_fp16_plan_code(&d), POOL_ERR_AVG_K);
        let mut d = desc(2, 4, 4, 1, 1);
        d.method = PoolMethod::Max;
        assert_eq!(pool_fp16_plan_code(&d), 0);
        // Min is encodable but the runtime plan refuses it.
        let mut d = desc(2, 4, 4, 2, 2);
        d.method = PoolMethod::Min;
        assert_eq!(pool_fp16_plan_code(&d), POOL_ERR_METHOD);
        // avg_exclude_pad is refused rather than silently dropped.
        let mut d = desc(8, 5, 5, 3, 3);
        d.method = PoolMethod::Avg;
        d.avg_exclude_pad = true;
        assert_eq!(pool_fp16_plan_code(&d), POOL_ERR_EXCLUDE_PAD);
        assert_eq!(
            pool_fp16_plan(&d).unwrap_err().status(),
            crate::Status::Unsupported
        );
        // Field widths: kernel 17 (4-bit, value-1), channel 8193 (13-bit). The kernel
        // case needs a plane tall enough to keep the extent positive — the extent check
        // comes first, as in C.
        let d = desc(8, 17, 17, 17, 2);
        assert_eq!(pool_fp16_plan_code(&d), POOL_ERR_KERNEL);
        let d = desc(8193, 4, 4, 2, 2);
        assert_eq!(pool_fp16_plan_code(&d), POOL_ERR_CUBE);
        // ... and 8192 is the last encodable channel count.
        assert_eq!(pool_fp16_plan_code(&desc(8192, 4, 4, 2, 2)), 0);
        // A pad of 8 overflows the 3-bit count.
        let mut d = desc(8, 8, 8, 3, 3);
        d.method = PoolMethod::Avg;
        d.pad_left = 8;
        assert_eq!(pool_fp16_plan_code(&d), POOL_ERR_PAD);
        // Zero extents.
        let d = desc(0, 4, 4, 2, 2);
        assert_eq!(pool_fp16_plan_code(&d), crate::Status::NoMem.code());
        let mut d = desc(8, 4, 4, 2, 2);
        d.stride_x = 0;
        assert_eq!(pool_fp16_plan_code(&d), crate::Status::Unsupported.code());
    }

    #[test]
    fn reference_hand_cases() {
        use half::f16;
        // 2x2 max, stride 2, no pad: one window per plane.
        let d = {
            let mut d = desc(1, 2, 2, 2, 2);
            d.stride_y = 2;
            d.stride_x = 2;
            d
        };
        let input = [
            f16::from_f32(1.0),
            f16::from_f32(-3.0),
            f16::from_f32(2.5),
            f16::from_f32(0.5),
        ];
        let mut out = [f16::ZERO; 1];
        pool_ref_fp16(&d, &input, &mut out);
        assert_eq!(out[0], f16::from_f32(2.5));

        // 2x2 average over the same window: mean of the four.
        let mut d = d;
        d.method = PoolMethod::Avg;
        pool_ref_fp16(&d, &input, &mut out);
        assert_eq!(out[0], f16::from_f32(0.25));

        // A padded average divides by kh*kw (count-include-pad): a 3x3 window at the
        // corner of a 1x1 plane is that pixel / 9.
        let mut d = desc(1, 1, 1, 3, 3);
        d.method = PoolMethod::Avg;
        d.pad_top = 1;
        d.pad_left = 1;
        d.pad_bottom = 1;
        d.pad_right = 1;
        assert_eq!((d.oh(), d.ow()), (1, 1));
        let input = [f16::from_f32(9.0)];
        pool_ref_fp16(&d, &input, &mut out);
        assert_eq!(out[0], f16::from_f32(1.0));

        // Max: the corner window only reaches the in-range part of the plane (the
        // padded cells read -inf and never win), the far corner reaches all of it.
        let mut d = desc(1, 3, 3, 3, 3);
        d.pad_top = 1;
        d.pad_left = 1;
        assert_eq!((d.oh(), d.ow()), (2, 2));
        let input: Vec<f16> = (0..9).map(|i| f16::from_f32(i as f32 - 4.0)).collect();
        let mut out = [f16::ZERO; 4];
        pool_ref_fp16(&d, &input, &mut out);
        assert_eq!(out[0], f16::ZERO); // rows -1..1 x cols -1..1 -> -4..1
        assert_eq!(out[3], f16::from_f32(4.0)); // rows 0..2 x cols 0..2 -> the plane

        // int8 max is the exact integer max, including both ends of the range.
        let mut d = desc(1, 4, 4, 2, 2);
        d.stride_y = 2;
        d.stride_x = 2;
        let input = [
            -128i8, 127, 0, 5, -7, -128, 127, -1, 3, 4, 5, 6, 7, 8, 9, 10,
        ];
        let mut out = [0i8; 4];
        pool_ref_int8(&d, &input, &mut out);
        // (0,1)/(1,0)/(1,1) windows: {0,5,127,-1}, {3,4,7,8}, {5,6,9,10}.
        assert_eq!(out, [127, 127, 8, 10]);

        // int8 average: the C routes through fp16 and rounds to nearest (the header's
        // "truncates toward zero" predates the fp16 routing; the code is the contract).
        let mut d = desc(1, 1, 4, 1, 4);
        d.method = PoolMethod::Avg;
        let input = [1i8, 2, 2, 2]; // 7/4 = 1.75 -> 2
        let mut out = [0i8; 1];
        pool_ref_int8(&d, &input, &mut out);
        assert_eq!(out[0], 2);
        let input = [-1i8, -2, -1, -2]; // -1.5 -> ties to even -> -2
        pool_ref_int8(&d, &input, &mut out);
        assert_eq!(out[0], -2);

        // uint8 recentring: max of {0, 255} is 255, and a mean stays in range.
        let d = {
            let mut d = desc(1, 1, 4, 1, 4);
            d.method = PoolMethod::Avg;
            d
        };
        let input = [0u8, 255, 255, 255];
        let mut out = [0u8; 1];
        pool_ref_uint8(&d, &input, &mut out);
        // (-128 + 3*127) / 4 = 63.25 -> 63, then + 128 = 191.
        assert_eq!(out[0], 191);
        let mut d = desc(1, 1, 4, 1, 4);
        d.method = PoolMethod::Max;
        pool_ref_uint8(&d, &input, &mut out);
        assert_eq!(out[0], 255);
    }

    #[test]
    fn uint8_narrowing_keeps_the_full_byte_range() {
        // The bug this pins: a `f32 as i8` cast saturates at 127, so every bright
        // uint8 result would read back as 127.
        assert_eq!(narrow_pool(255.0, 128, true), -1i8); // 0xFF
        assert_eq!(narrow_pool(72.0, 128, true) as u8, 200);
        assert_eq!(narrow_pool(-128.0, 128, true) as u8, 0);
        assert_eq!(narrow_pool(-1.0, 128, true) as u8, 127);
        assert_eq!(narrow_pool(127.0, 128, true) as u8, 255);
        assert_eq!(narrow_pool(300.0, 128, true) as u8, 255); // clamped
        assert_eq!(narrow_pool(-200.0, 128, true) as u8, 0);
        // int8 stays a signed clamp.
        assert_eq!(narrow_pool(127.4, 0, false), 127);
        assert_eq!(narrow_pool(-200.0, 0, false), -128);
        assert_eq!(narrow_pool(-0.5, 0, false), 0); // ties to even
        assert_eq!(narrow_pool(-1.5, 0, false), -2);
    }

    #[test]
    fn int8_plan_is_the_geometry_plan() {
        // The int8/uint8 entries run the fp16 geometry rules, so the same descriptor
        // accepts and refuses identically (a dtype-independent plan).
        let mut d = desc(8, 7, 7, 7, 7);
        d.method = PoolMethod::Avg;
        assert_eq!(pool_fp16_plan_code(&d), 0);
    }
}
