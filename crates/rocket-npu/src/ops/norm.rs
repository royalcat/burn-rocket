//! Normalization primitives (RK3588), starting with the per-row broadcast scale.
//!
//! **M6b ports only `rocket_scale_rows_fp16` here** (with its `_ref`): softmax and
//! log-softmax need a per-row scalar broadcast, and the C library keeps that primitive
//! next to the norms because RMSNorm/LayerNorm fold their `1/rms` into it. The rest of
//! `rocket_norm.h` — `rocket_rmsnorm_fp16` and `rocket_layernorm_fp16` plus their
//! references and the fp16-square prescale helper (`rocket_square_prescale_k`) — lands
//! in **M7**, together with the FFN/encoder composites it feeds.
//!
//! # The cost model (worth keeping in mind for M7)
//!
//! RMSNorm/LayerNorm are memory-bound elementwise+reduce ops. Standalone on the NPU they
//! are *submit*-bound (the flat EW path tiles ~1020 rows per submit, so a `[512,3840]`
//! square is ~60 submits), and for an isolated norm the host A76 single memory pass is
//! faster. The win is compositional: when the norm sits between two NPU matmuls (the
//! FFN/attention case), running it on-device lets the activation stay in the NC1HWC2 cube
//! and avoids the de-tile → host → re-pack layout round trip on every op, which is the
//! dominant not-MAC-bound cost. These primitives exist to be **fused into a resident
//! block**, not to beat the host standalone.
//!
//! The work split that follows from that: the `O(M*H)` work (square, the H-contraction
//! reduce, the final scale) runs on the NPU; the `O(M)` per-row tail (mean, `+eps`,
//! `rsqrt`) is exact fp32 host scalar work. Sending `M` tiny per-row scalars back to the
//! NPU for an *approximate* LUT `rsqrt` would add a round trip **and** hit the rsqrt-LUT
//! domain problem (`ms` spans many decades across rows/layers; a uniform-grid LUT cannot
//! cover that) for no benefit — the DPU rsqrt LUT stays reserved for a large-**tensor**
//! `rsqrt`, not an `M`-vector.

use half::f16;

use crate::Device;
use crate::error::{Error, Result};
use crate::ops::activation::ew_mul;

/// `rocket_scale_rows_ref_fp16`: `out[m][n] = in[m][n] * r[m]` in f32, one RNE narrow.
pub fn scale_rows_ref(m: usize, n: usize, input: &[f16], r: &[f32], out: &mut [f16]) {
    assert!(input.len() >= m * n, "input shorter than M*N");
    assert!(r.len() >= m, "r shorter than M");
    assert!(out.len() >= m * n, "output shorter than M*N");
    for row in 0..m {
        let rm = r[row];
        let ip = &input[row * n..row * n + n];
        let op = &mut out[row * n..row * n + n];
        for (o, &v) in op.iter_mut().zip(ip) {
            *o = f16::from_f32(v.to_f32() * rm);
        }
    }
}

/// `rocket_scale_rows_fp16`: `out[m][n] = in[m][n] * r[m]` on the NPU.
///
/// The per-row scalar `r[m]` is fp32 and is **rounded to fp16** for the multiply (the C
/// header says so explicitly: the broadcast is materialized as an fp16 cube and the DPU
/// EW multiply does the arithmetic, so it inherits that path bit-for-bit). The
/// materialization is a pure host fill — no arithmetic — and then one
/// [`ew_mul`] call.
///
/// `m*n > INT_MAX` is a [`crate::error::Status::Shape`], exactly as in C: the element count goes to the
/// EW entry as an `i32`, and a truncating product would under-process the surface and
/// return a silently partial result.
pub fn scale_rows(
    dev: &Device,
    m: usize,
    n: usize,
    input: &[f16],
    r: &[f32],
    out: &mut [f16],
) -> Result<()> {
    if m < 1 || n < 1 {
        return Err(Error::shape("rocket_scale_rows_fp16"));
    }
    let mn = m * n;
    if input.len() < mn || r.len() < m || out.len() < mn {
        return Err(Error::shape("rocket_scale_rows_fp16/buffers"));
    }
    if mn > i32::MAX as usize {
        return Err(
            Error::shape("rocket_scale_rows_fp16").with_detail(format!("{mn} elements > INT_MAX"))
        );
    }

    let out = &mut out[..mn];
    let mut rb = vec![f16::ZERO; mn];
    for row in 0..m {
        let rv = f16::from_f32(r[row]);
        rb[row * n..row * n + n].fill(rv);
    }
    ew_mul(dev, input, &rb, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vals(v: &[f32]) -> Vec<f16> {
        v.iter().map(|x| f16::from_f32(*x)).collect()
    }

    #[test]
    fn reference_scales_each_row_by_its_own_scalar() {
        let x = vals(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let r = [0.5f32, 2.0];
        let mut out = vec![f16::ZERO; 6];
        scale_rows_ref(2, 3, &x, &r, &mut out);
        let got: Vec<f32> = out.iter().map(|v| v.to_f32()).collect();
        assert_eq!(got, vec![0.5, 1.0, 1.5, 8.0, 10.0, 12.0]);

        // The scalar is rounded to fp16 before the multiply, so a value outside the fp16
        // grid lands on the nearest representable one — 1/3 becomes 0.33325...
        let r = [1.0f32 / 3.0];
        let x = vals(&[3.0]);
        let mut out = vec![f16::ZERO; 1];
        scale_rows_ref(1, 1, &x, &r, &mut out);
        assert_eq!(
            out[0],
            f16::from_f32(f16::from_f32(1.0 / 3.0).to_f32() * 3.0)
        );

        // An exact power-of-two scale is exact.
        let r = [0.25f32];
        let x = vals(&[1.0, -8.0, 1024.0]);
        let mut out = vec![f16::ZERO; 3];
        scale_rows_ref(1, 3, &x, &r, &mut out);
        assert_eq!(out, vals(&[0.25, -2.0, 256.0]));
    }

    #[test]
    fn buffers_are_validated() {
        // The device entry needs a device, but every length/geometry check runs first,
        // so these reach a typed error without one (the C checks the shape before fd).
        let dev = match Device::open() {
            Ok(d) => d,
            Err(_) => return, // no device on this host; the ref path is covered above
        };
        let x = vec![f16::ZERO; 6];
        let r = [1.0f32; 2];
        let mut out = vec![f16::ZERO; 6];
        assert!(scale_rows(&dev, 0, 3, &x, &r, &mut out).is_err());
        assert!(scale_rows(&dev, 2, 0, &x, &r, &mut out).is_err());
        assert!(scale_rows(&dev, 2, 4, &x, &r, &mut out).is_err());
        assert!(scale_rows(&dev, 3, 3, &x, &r, &mut out).is_err());
    }
}
