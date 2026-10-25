//! Integer-factor spatial upsample (`rocket_resize.c`).
//!
//! An upsample **is** a transposed conv: scatter each input pixel onto a stride-`scale`
//! lattice and convolve with a small kernel, so both modes ride the already-validated
//! depthwise [`crate::ops::conv_transpose`] path rather than adding an NPU primitive.
//!
//! ```text
//! nearest  — kernel = ones(scale): each input pixel replicated into a scale x scale block.
//! bilinear — kernel = a separable triangle of size k = 2*scale - scale%2,
//!            tri[i] = 1 - |i - (k-1)/2| / scale.
//! ```
//!
//! The stride-`scale` subsample of that triangle is a partition of unity (each output
//! phase's taps sum to 1), so the interior is exact 2-tap bilinear interpolation with
//! the half-pixel map `src = (o + 0.5)/scale - 0.5` and a **zero** boundary (matching
//! `F.interpolate(mode='bilinear', align_corners=False)` in the interior; the framework's
//! own `align_corners` / clamp-boundary modes are a delegate-wiring concern).
//!
//! Both use `pad = (k - scale)/2`, `opad = 0`, so the output is exactly
//! `IH*scale_y × IW*scale_x` (the `k` cancels:
//! `OH = (IH-1)*scale - 2*pad + (k-1) + 1 = IH*scale`).
//!
//! Layouts (row-major, batch 1): `in [C][IH][IW]`, `out [C][IH*scale_y][IW*scale_x]`,
//! fp16. Constraint: `C % G == 0` (`G = 32`, the depthwise channel group) — inherited
//! from the depthwise forward conv.

use half::f16;

use crate::Device;
use crate::error::Result;
use crate::ops::conv_transpose::{
    ConvTranspose2dDesc, conv_transpose2d, conv_transpose2d_plan, conv_transpose2d_plan_code,
};

/// The bilinear 1-D triangle kernel size for an integer factor (the FCN
/// "bilinear deconv" form).
pub fn bilinear_ksize(scale: usize) -> usize {
    2 * scale - (scale % 2)
}

/// Fill the per-channel depthwise kernel `W[C][1][KH][KW]`.
///
/// `mode` 0 = nearest (ones), 1 = bilinear (the separable triangle). All channels share
/// the same kernel, exactly as the C `build_kernel`.
pub fn build_kernel(mode: u8, c: usize, kh: usize, kw: usize, sy: usize, sx: usize) -> Vec<f16> {
    let mut ty = vec![0.0f64; kh];
    let mut tx = vec![0.0f64; kw];
    if mode == 0 {
        ty.fill(1.0);
        tx.fill(1.0);
    } else {
        let cy = (kh - 1) as f64 / 2.0;
        let cx = (kw - 1) as f64 / 2.0;
        for (i, t) in ty.iter_mut().enumerate() {
            *t = 1.0 - (i as f64 - cy).abs() / sy as f64;
        }
        for (j, t) in tx.iter_mut().enumerate() {
            *t = 1.0 - (j as f64 - cx).abs() / sx as f64;
        }
    }
    let mut w = vec![f16::ZERO; c * kh * kw];
    for k in 0..kh {
        for j in 0..kw {
            let v = f16::from_f64(ty[k] * tx[j]);
            for ch in 0..c {
                w[(ch * kh + k) * kw + j] = v;
            }
        }
    }
    w
}

/// The depthwise-transpose descriptor of an upsample (`upsample_desc`).
///
/// `mode` 0 = nearest, 1 = bilinear. Returns the descriptor plus its kernel extent.
pub fn upsample_desc(
    mode: u8,
    c: usize,
    ih: usize,
    iw: usize,
    sy: usize,
    sx: usize,
) -> (ConvTranspose2dDesc, usize, usize) {
    let kh = if mode == 0 { sy } else { bilinear_ksize(sy) };
    let kw = if mode == 0 { sx } else { bilinear_ksize(sx) };
    let mut d = ConvTranspose2dDesc::new(c, ih, iw, c, kh, kw);
    d.stride_y = sy;
    d.stride_x = sx;
    d.pad_top = (kh - sy) / 2;
    d.pad_left = (kw - sx) / 2;
    d.opad_y = 0;
    d.opad_x = 0;
    d.dil_y = 1;
    d.dil_x = 1;
    d.depthwise = true;
    (d, kh, kw)
}

fn upsample_plan_code(mode: u8, c: usize, ih: usize, iw: usize, sy: usize, sx: usize) -> i32 {
    if c == 0 || ih == 0 || iw == 0 || sy == 0 || sx == 0 {
        return -1;
    }
    let (d, _, _) = upsample_desc(mode, c, ih, iw, sy, sx);
    conv_transpose2d_plan_code(&d)
}

/// `rocket_upsample_nearest_plan`: validate an integer-factor nearest upsample.
pub fn upsample_nearest_plan(c: usize, ih: usize, iw: usize, sy: usize, sx: usize) -> i32 {
    upsample_plan_code(0, c, ih, iw, sy, sx)
}

/// `rocket_upsample_bilinear_plan`: validate an integer-factor bilinear upsample.
pub fn upsample_bilinear_plan(c: usize, ih: usize, iw: usize, sy: usize, sx: usize) -> i32 {
    upsample_plan_code(1, c, ih, iw, sy, sx)
}

/// Shared device path: build the kernel and run the depthwise transposed conv.
#[allow(clippy::too_many_arguments)]
fn upsample_run(
    dev: &Device,
    mode: u8,
    input: &[f16],
    out: &mut [f16],
    c: usize,
    ih: usize,
    iw: usize,
    sy: usize,
    sx: usize,
) -> Result<()> {
    let (d, kh, kw) = upsample_desc(mode, c, ih, iw, sy, sx);
    conv_transpose2d_plan(&d)?;
    let w = build_kernel(mode, c, kh, kw, sy, sx);
    conv_transpose2d(dev, &d, input, &w, out)
}

/// `rocket_upsample_nearest_fp16`: block replication.
#[allow(clippy::too_many_arguments)]
pub fn upsample_nearest(
    dev: &Device,
    input: &[f16],
    out: &mut [f16],
    c: usize,
    ih: usize,
    iw: usize,
    sy: usize,
    sx: usize,
) -> Result<()> {
    upsample_run(dev, 0, input, out, c, ih, iw, sy, sx)
}

/// `rocket_upsample_bilinear_fp16`: half-pixel bilinear interpolation, zero boundary.
#[allow(clippy::too_many_arguments)]
pub fn upsample_bilinear(
    dev: &Device,
    input: &[f16],
    out: &mut [f16],
    c: usize,
    ih: usize,
    iw: usize,
    sy: usize,
    sx: usize,
) -> Result<()> {
    upsample_run(dev, 1, input, out, c, ih, iw, sy, sx)
}

/// `rocket_upsample_nearest_ref_fp16`: the independent nearest oracle
/// (`out[c][oh][ow] = in[c][oh/sy][ow/sx]`).
pub fn upsample_nearest_ref(
    input: &[f16],
    out: &mut [f16],
    c: usize,
    ih: usize,
    iw: usize,
    sy: usize,
    sx: usize,
) {
    let (oh, ow) = (ih * sy, iw * sx);
    for ch in 0..c {
        for y in 0..oh {
            for x in 0..ow {
                out[(ch * oh + y) * ow + x] = input[(ch * ih + y / sy) * iw + x / sx];
            }
        }
    }
}

/// `rocket_upsample_bilinear_ref_fp16`: the independent half-pixel oracle.
///
/// `src = (o + 0.5)/scale - 0.5`, two taps per axis, out-of-range taps contribute 0.
pub fn upsample_bilinear_ref(
    input: &[f16],
    out: &mut [f16],
    c: usize,
    ih: usize,
    iw: usize,
    sy: usize,
    sx: usize,
) {
    let (oh, ow) = (ih * sy, iw * sx);
    for ch in 0..c {
        let ic = &input[ch * ih * iw..(ch + 1) * ih * iw];
        for y in 0..oh {
            let scy = (y as f64 + 0.5) / sy as f64 - 0.5;
            let y0 = scy.floor() as i64;
            let dy = scy - y0 as f64;
            for x in 0..ow {
                let scx = (x as f64 + 0.5) / sx as f64 - 0.5;
                let x0 = scx.floor() as i64;
                let dx = scx - x0 as f64;
                let tap = |yy: i64, xx: i64| -> f64 {
                    if yy >= 0 && yy < ih as i64 && xx >= 0 && xx < iw as i64 {
                        ic[(yy as usize) * iw + xx as usize].to_f64()
                    } else {
                        0.0
                    }
                };
                let p00 = tap(y0, x0);
                let p01 = tap(y0, x0 + 1);
                let p10 = tap(y0 + 1, x0);
                let p11 = tap(y0 + 1, x0 + 1);
                let v =
                    (1.0 - dy) * ((1.0 - dx) * p00 + dx * p01) + dy * ((1.0 - dx) * p10 + dx * p11);
                out[(ch * oh + y) * ow + x] = f16::from_f64(v);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::conv_transpose::{conv_transpose2d_oh, conv_transpose2d_ow};

    #[test]
    fn kernel_sizes_and_geometry() {
        // Nearest reuses the scale as the kernel; bilinear's is 2s - s%2.
        assert_eq!(bilinear_ksize(1), 1);
        assert_eq!(bilinear_ksize(2), 4);
        assert_eq!(bilinear_ksize(3), 5);
        assert_eq!(bilinear_ksize(4), 8);
        for s in 1..=4 {
            let (d, kh, kw) = upsample_desc(0, 32, 5, 7, s, s);
            assert_eq!((kh, kw), (s, s));
            assert_eq!((d.pad_top, d.pad_left), (0, 0));
            assert!(d.depthwise);
            assert_eq!(conv_transpose2d_oh(&d) as usize, 5 * s);
            assert_eq!(conv_transpose2d_ow(&d) as usize, 7 * s);
            let (d, kh, kw) = upsample_desc(1, 32, 5, 7, s, s);
            assert_eq!((kh, kw), (bilinear_ksize(s), bilinear_ksize(s)));
            assert_eq!((d.pad_top, d.pad_left), ((kh - s) / 2, (kw - s) / 2));
            assert_eq!(conv_transpose2d_oh(&d) as usize, 5 * s);
            assert_eq!(conv_transpose2d_ow(&d) as usize, 7 * s);
        }
        // Per-axis scales differ, and only the odd axis grows the triangle.
        let (d, kh, kw) = upsample_desc(1, 32, 3, 3, 2, 3);
        assert_eq!((kh, kw), (4, 5));
        assert_eq!((d.pad_top, d.pad_left), (1, 1));
        assert_eq!(conv_transpose2d_oh(&d) as usize, 6);
        assert_eq!(conv_transpose2d_ow(&d) as usize, 9);
    }

    #[test]
    fn plans_refuse_what_the_transpose_refuses() {
        // C % G == 0 is the inherited depthwise constraint.
        assert_eq!(upsample_nearest_plan(32, 8, 8, 2, 2), 0);
        assert_eq!(upsample_bilinear_plan(32, 8, 8, 2, 2), 0);
        assert_ne!(upsample_nearest_plan(24, 8, 8, 2, 2), 0);
        assert_eq!(upsample_nearest_plan(0, 8, 8, 2, 2), -1);
        assert_eq!(upsample_bilinear_plan(32, 8, 8, 0, 2), -1);
    }

    #[test]
    fn nearest_reference_replicates_blocks() {
        let (c, ih, iw) = (1usize, 2usize, 2usize);
        let input = [
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
        ];
        let mut out = vec![f16::ZERO; c * ih * 2 * iw * 2];
        upsample_nearest_ref(&input, &mut out, c, ih, iw, 2, 2);
        let got: Vec<f32> = out.iter().map(|v| v.to_f32()).collect();
        assert_eq!(
            got,
            vec![
                1.0, 1.0, 2.0, 2.0, //
                1.0, 1.0, 2.0, 2.0, //
                3.0, 3.0, 4.0, 4.0, //
                3.0, 3.0, 4.0, 4.0,
            ]
        );
    }

    #[test]
    fn bilinear_reference_is_half_pixel_with_a_zero_boundary() {
        // A 1x2 ramp, x2: the half-pixel map puts 0.25/0.75 between the samples at the
        // edges, and the out-of-range taps contribute 0 (so the first and last outputs
        // are 0.75 and 0.5 of the edge sample).
        let input = [f16::from_f32(2.0), f16::from_f32(4.0)];
        let mut out = vec![f16::ZERO; 4];
        upsample_bilinear_ref(&input, &mut out, 1, 1, 2, 1, 2);
        let got: Vec<f32> = out.iter().map(|v| v.to_f32()).collect();
        // x = 0: src = -0.25 -> x0 = -1, dx = 0.75 -> 0.75*in[0] = 1.5
        // x = 1: src = 0.25  -> 0.75*2 + 0.25*4 = 2.5
        // x = 2: src = 0.75  -> 0.25*2 + 0.75*4 = 3.5
        // x = 3: src = 1.25  -> x0 = 1, dx = 0.25 -> 0.75*4 = 3.0
        assert_eq!(got, vec![1.5, 2.5, 3.5, 3.0]);

        // A single sample upscaled by 3 (k = 5, pad 1, tri [1/3, 2/3, 1, 2/3, 1/3]):
        // output phase o reads tap o+1, so the block is the outer product of
        // [2/3, 1, 2/3] — the kernel truncates at the plane boundary exactly as the C
        // reference does.
        let input = [f16::from_f32(-1.5)];
        let mut out = vec![f16::ZERO; 9];
        upsample_bilinear_ref(&input, &mut out, 1, 1, 1, 3, 3);
        let row = [2.0f32 / 3.0, 1.0, 2.0 / 3.0];
        for y in 0..3 {
            for x in 0..3 {
                let want = f16::from_f64(-1.5 * (row[y] * row[x]) as f64).to_f32();
                assert_eq!(out[y * 3 + x].to_f32(), want, "at ({y},{x})");
            }
        }

        // Interior parity with the exact 2-tap formula on a 4x4 plane at x2.
        let input: Vec<f16> = (0..16).map(|i| f16::from_f32(i as f32)).collect();
        let mut out = vec![f16::ZERO; 64];
        upsample_bilinear_ref(&input, &mut out, 1, 4, 4, 2, 2);
        // (oh, ow) = (3, 3): src = (3.5)/2 - 0.5 = 1.25 -> 0.25*in[1][1] ... in[2][2].
        let v = 0.75 * 0.75 * 5.0 + 0.75 * 0.25 * 6.0 + 0.25 * 0.75 * 9.0 + 0.25 * 0.25 * 10.0;
        assert_eq!(out[3 * 8 + 3].to_f32(), v as f32);
    }

    #[test]
    fn kernels_are_plain_or_triangular() {
        // Nearest: ones at every tap, for every channel.
        let w = build_kernel(0, 2, 2, 2, 2, 2);
        assert_eq!(w.len(), 2 * 4);
        assert!(w.iter().all(|v| v.to_f32() == 1.0));
        // Bilinear at scale 2 (k = 4, centre 1.5): tri = [0.25, 0.75, 0.75, 0.25]. Its
        // stride-2 subsample sums to 1 at both phases.
        let w = build_kernel(1, 1, 4, 4, 2, 2);
        let tri = [0.25f32, 0.75, 0.75, 0.25];
        assert_eq!(tri[0] + tri[2], 1.0);
        assert_eq!(tri[1] + tri[3], 1.0);
        for k in 0..4 {
            for j in 0..4 {
                assert_eq!(w[k * 4 + j].to_f32(), tri[k] * tri[j]);
            }
        }
        // At scale 3 (k = 5): tri = [1/3, 2/3, 1, 2/3, 1/3].
        let w = build_kernel(1, 1, 5, 5, 3, 3);
        let tri = [1.0 / 3.0f32, 2.0 / 3.0, 1.0, 2.0 / 3.0, 1.0 / 3.0];
        for k in 0..5 {
            for j in 0..5 {
                let want = f16::from_f64(tri[k] as f64 * tri[j] as f64).to_f32();
                assert_eq!(w[k * 5 + j].to_f32(), want);
            }
        }
        // Every channel shares the same kernel.
        let w = build_kernel(1, 3, 4, 4, 2, 2);
        for ch in 1..3 {
            assert_eq!(&w[..16], &w[ch * 16..(ch + 1) * 16]);
        }
    }
}
