//! Qwen2-VL image preprocessing for d3 (the Qwen3.5-VL processor).
//!
//! The checkpoint's processor is `Qwen2VLImageProcessorFast` with the runtime's
//! 1.6 MP budget, used through the torchvision backend. For one image that is:
//!
//! 1. decode to RGB8 (PIL `convert("RGB")`, no EXIF rotation, no crop);
//! 2. `smart_resize(h, w, 32, 65_536, 1_638_400)` — the patch factor is
//!    `patch_size * merge_size = 32`, and the d3 runtime overrides the
//!    checkpoint's own `[65_536, 16_777_216]` budget with `[65_536, 1_638_400]`
//!    (`d3_runtime.IMAGE_MIN_PIXELS`/`IMAGE_MAX_PIXELS`);
//! 3. `torchvision.transforms.v2.functional.resize(..., BICUBIC, antialias=True,
//!    align_corners=False)` on the **uint8** tensor (the resize runs before the
//!    normalize, so its rounding is the uint8 one);
//! 4. normalize `(x - 127.5) / 127.5` in f32 (`image_mean = image_std = 0.5`
//!    fused with the `1/255` rescale);
//! 5. patchify in **merge-block order** — patch order is
//!    `(block_row, block_col, in_row, in_col)` — with each patch's values in
//!    `[channel, temporal, patch_row, patch_col]` order, the single frame
//!    duplicated across `temporal_patch_size`.
//!
//! The result is one `[patches, 3 * T * p * p]` f32 matrix (1536 wide for the
//! d3 checkpoints), the `[1, H/16, W/16]` grid, and the merged token count
//! `H * W / (16 * 16) / merge^2` the prompt reserves.
//!
//! The resize reuses [`crate::gemma::media::resize_bicubic_aa`], the ATen uint8
//! antialiased-bicubic port (two separable passes, int16 weights, uint8
//! intermediate) that EmbeddedGemma 2's tower validated bit-for-bit against
//! torchvision; the parity probe checks it again on the d3 assets.

use anyhow::{Context, Result, bail};
use burn::prelude::*;
use burn::tensor::TensorData;
use image::RgbImage;

use super::config::VisionConfig;
use crate::gemma::media::{load_image, resize_bicubic_aa};

/// `d3_runtime.IMAGE_MIN_PIXELS` (the processor's `shortest_edge`).
pub const MIN_PIXELS: usize = 65_536;
/// `d3_runtime.IMAGE_MAX_PIXELS` (the processor's `longest_edge`).
pub const MAX_PIXELS: usize = 1_638_400;
/// `smart_resize`'s aspect-ratio ceiling.
pub const MAX_ASPECT_RATIO: f64 = 200.0;

/// One preprocessed image, ready for the vision tower.
#[derive(Debug, Clone)]
pub struct PreparedImage {
    /// `[patches, in_channels * temporal_patch_size * patch_size^2]` f32 pixel
    /// values in `[-1, 1]`, patches in merge-block order.
    pub pixel_values: Tensor<2>,
    /// `[1, grid_h, grid_w]` patch grid (`grid_h * 16 = resized height`).
    pub grid: [usize; 3],
    /// Merged image tokens this image contributes to the prompt.
    pub tokens: usize,
    /// `(height, width)` of the decoded source.
    pub source: (usize, usize),
    /// `(height, width)` after `smart_resize`.
    pub resized: (usize, usize),
}

impl PreparedImage {
    /// Patches in the image (`grid_h * grid_w`).
    pub fn patches(&self) -> usize {
        self.grid[1] * self.grid[2]
    }

    /// The patch grid as `(rows, cols)`.
    pub fn patch_grid(&self) -> (usize, usize) {
        (self.grid[1], self.grid[2])
    }
}

/// A float as Python's `repr` prints it: the shortest round-trip form, with a
/// `.0` on an integral value (`400`, never `400.0`-less).
fn python_float(value: f64) -> String {
    let text = format!("{value}");
    if text.contains(['.', 'e', 'E']) || !value.is_finite() {
        text
    } else {
        format!("{text}.0")
    }
}

/// Python's `round(x)` for `x >= 0`: ties go to the **even** integer.
///
/// `smart_resize` uses Python's `round`, which is half-to-even; Rust's
/// `f64::round` is half-away-from-zero, so `h/32 = 32.5` would give 33 instead
/// of 32 and change the grid.
fn round_half_even(x: f64) -> f64 {
    let floor = x.floor();
    let frac = x - floor;
    // Above .5 rounds up; exactly .5 rounds to the even neighbour.
    let up = frac > 0.5 || (frac == 0.5 && (floor as i64) & 1 == 1);
    if up { floor + 1.0 } else { floor }
}

/// `Qwen2VLImageProcessor.smart_resize`: round both sides to a multiple of
/// `factor`, then shrink/grow so the pixel count lands in
/// `[min_pixels, max_pixels]` while keeping the aspect ratio.
pub fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize)> {
    if height == 0 || width == 0 {
        bail!("the image has no pixels ({width}x{height})");
    }
    if factor == 0 || min_pixels == 0 || max_pixels < min_pixels {
        bail!("invalid processor budget (factor {factor}, min {min_pixels}, max {max_pixels})");
    }
    let (h, w) = (height as f64, width as f64);
    let aspect = h.max(w) / h.min(w);
    if aspect > MAX_ASPECT_RATIO {
        // The reference's message, verbatim (`Qwen2VLImageProcessor.smart_resize`):
        // a Python float repr of the ratio, so an integral one keeps its `.0`.
        bail!(
            "absolute aspect ratio must be smaller than {MAX_ASPECT_RATIO}, got {}",
            python_float(aspect)
        );
    }
    let factor_f = factor as f64;
    let mut h_bar = round_half_even(h / factor_f) * factor_f;
    let mut w_bar = round_half_even(w / factor_f) * factor_f;
    if h_bar * w_bar > max_pixels as f64 {
        let beta = ((h * w) / max_pixels as f64).sqrt();
        h_bar = ((h / beta / factor_f).floor() * factor_f).max(factor_f);
        w_bar = ((w / beta / factor_f).floor() * factor_f).max(factor_f);
    } else if h_bar * w_bar < min_pixels as f64 {
        let beta = (min_pixels as f64 / (h * w)).sqrt();
        h_bar = (h * beta / factor_f).ceil() * factor_f;
        w_bar = (w * beta / factor_f).ceil() * factor_f;
    }
    Ok((h_bar as usize, w_bar as usize))
}

/// Normalize one uint8 channel value exactly as the processor does
/// (`(x - 127.5) / 127.5` after `tvF.normalize` fuses the `1/255` rescale).
#[inline]
pub fn normalize_channel(value: u8) -> f32 {
    (value as f32 - 127.5) / 127.5
}

/// Patchify an already resized RGB8 image into the processor's flat layout.
///
/// Returns `(pixel_values, patches)`; the flat row layout is
/// `[channel, temporal, patch_row, patch_col]` with the frame duplicated across
/// the temporal patches, and rows are in merge-block order.
pub fn patchify(img: &RgbImage, config: &VisionConfig) -> Result<(Vec<f32>, usize)> {
    let p = config.patch_size;
    let m = config.spatial_merge_size;
    let t = config.temporal_patch_size;
    let channels = config.in_channels;
    let (w, h) = (img.width() as usize, img.height() as usize);
    let factor = p * m;
    if h == 0 || w == 0 {
        bail!("the image has no pixels ({w}x{h})");
    }
    if !h.is_multiple_of(factor) || !w.is_multiple_of(factor) {
        bail!(
            "image {w}x{h} is not divisible by the patch factor {factor} \
             (patch {p} * merge {m})"
        );
    }
    let (grid_h, grid_w) = (h / p, w / p);
    let patch_dim = config.patch_dim();
    let patches = grid_h * grid_w;
    let raw = img.as_raw();
    let stride = w * channels;
    let mut values = Vec::with_capacity(patches * patch_dim);
    for block_row in 0..grid_h / m {
        for block_col in 0..grid_w / m {
            for in_row in 0..m {
                for in_col in 0..m {
                    let row = block_row * m + in_row;
                    let col = block_col * m + in_col;
                    for c in 0..channels {
                        for _t in 0..t {
                            for i in 0..p {
                                let y = row * p + i;
                                for j in 0..p {
                                    let x = col * p + j;
                                    values.push(normalize_channel(
                                        raw[y * stride + x * channels + c],
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if values.len() != patches * patch_dim {
        bail!(
            "patchify produced {} values, expected {} ({patches} patches of {patch_dim})",
            values.len(),
            patches * patch_dim
        );
    }
    Ok((values, patches))
}

/// Preprocess one decoded RGB8 image: `smart_resize`, resize, normalize, patchify.
pub fn prepare_from_rgb(
    img: &RgbImage,
    config: &VisionConfig,
    device: &Device,
) -> Result<PreparedImage> {
    let (source_h, source_w) = (img.height() as usize, img.width() as usize);
    let (resized_h, resized_w) = smart_resize(
        source_h,
        source_w,
        config.pixel_factor(),
        MIN_PIXELS,
        MAX_PIXELS,
    )?;
    let resized = if (resized_h, resized_w) == (source_h, source_w) {
        img.clone()
    } else {
        resize_bicubic_aa(img, resized_w, resized_h)
    };
    let (values, patches) = patchify(&resized, config)?;
    let grid = [
        1,
        resized_h / config.patch_size,
        resized_w / config.patch_size,
    ];
    let tokens = config.merged_tokens(grid[1], grid[2]);
    let pixel_values = Tensor::<2>::from_data(
        TensorData::new(values, [patches, config.patch_dim()]),
        device,
    );
    Ok(PreparedImage {
        pixel_values,
        grid,
        tokens,
        source: (source_h, source_w),
        resized: (resized_h, resized_w),
    })
}

/// Decode and preprocess one image file (PIL `convert("RGB")` semantics: JPEGs
/// go through mozjpeg/libjpeg-turbo, like the deployed image paths).
pub fn prepare_image(
    path: &std::path::Path,
    config: &VisionConfig,
    device: &Device,
) -> Result<PreparedImage> {
    let img = load_image(path).with_context(|| format!("load the d3 image {}", path.display()))?;
    prepare_from_rgb(&img, config, device)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> VisionConfig {
        VisionConfig {
            depth: 1,
            hidden_size: 8,
            num_heads: 2,
            intermediate_size: 16,
            out_hidden_size: 4,
            patch_size: 2,
            temporal_patch_size: 2,
            spatial_merge_size: 2,
            in_channels: 3,
            num_position_embeddings: 4,
            rope_theta: 10_000.0,
            hidden_act: "gelu_pytorch_tanh".to_string(),
        }
    }

    #[test]
    fn python_rounding_is_half_to_even() {
        assert_eq!(round_half_even(31.25), 31.0);
        assert_eq!(round_half_even(28.125), 28.0);
        assert_eq!(round_half_even(32.5), 32.0, "32.5 -> 32 (even)");
        assert_eq!(round_half_even(33.5), 34.0, "33.5 -> 34 (even)");
        assert_eq!(round_half_even(30.5), 30.0);
        assert_eq!(round_half_even(0.5), 0.0);
        assert_eq!(round_half_even(1.5), 2.0);
        // Rust's own round would give 33 here.
        assert_eq!(32.5f64.round(), 33.0);
    }

    #[test]
    fn smart_resize_rounds_to_the_patch_factor() {
        // The receipt probe: 900x1000 -> 896x992 (grid [1, 62, 56]).
        assert_eq!(
            smart_resize(1000, 900, 32, MIN_PIXELS, MAX_PIXELS).unwrap(),
            (992, 896)
        );
        // Data URL / chart sizes from the image fixtures.
        assert_eq!(
            smart_resize(1320, 2200, 32, MIN_PIXELS, MAX_PIXELS).unwrap(),
            (960, 1632)
        );
        assert_eq!(
            smart_resize(1080, 1920, 32, MIN_PIXELS, MAX_PIXELS).unwrap(),
            (960, 1696)
        );
        assert_eq!(
            smart_resize(1120, 2200, 32, MIN_PIXELS, MAX_PIXELS).unwrap(),
            (896, 1792)
        );
        assert_eq!(
            smart_resize(683, 960, 32, MIN_PIXELS, MAX_PIXELS).unwrap(),
            (672, 960)
        );
    }

    #[test]
    fn smart_resize_grows_a_tiny_image() {
        // 64x64 = 4096 px < 65536: beta = sqrt(65536/4096) = 4 -> 256x256.
        assert_eq!(
            smart_resize(64, 64, 32, MIN_PIXELS, MAX_PIXELS).unwrap(),
            (256, 256)
        );
        // 100x100 = 10000: beta = sqrt(6.5536) = 2.56 -> ceil(100*2.56/32)*32 = 256.
        assert_eq!(
            smart_resize(100, 100, 32, MIN_PIXELS, MAX_PIXELS).unwrap(),
            (256, 256)
        );
        // A 1x1 image: beta = 256 -> ceil(256/32)*32 = 256.
        assert_eq!(
            smart_resize(1, 1, 32, MIN_PIXELS, MAX_PIXELS).unwrap(),
            (256, 256)
        );
    }

    #[test]
    fn smart_resize_refuses_extreme_aspect_and_empty_images() {
        let err = smart_resize(1, 300, 32, MIN_PIXELS, MAX_PIXELS).unwrap_err();
        assert_eq!(
            err.to_string(),
            "absolute aspect ratio must be smaller than 200, got 300.0"
        );
        // The control's 4000x10 case: Python's repr keeps the `.0`.
        let err = smart_resize(10, 4000, 32, MIN_PIXELS, MAX_PIXELS).unwrap_err();
        assert_eq!(
            err.to_string(),
            "absolute aspect ratio must be smaller than 200, got 400.0"
        );
        // A fractional ratio prints like Python too (no trailing zero padding).
        let err = smart_resize(10, 2010, 32, MIN_PIXELS, MAX_PIXELS).unwrap_err();
        assert_eq!(
            err.to_string(),
            "absolute aspect ratio must be smaller than 200, got 201.0"
        );
        assert!(smart_resize(0, 64, 32, MIN_PIXELS, MAX_PIXELS).is_err());
        // 200:1 is allowed (the check is `> 200`).
        assert!(smart_resize(1, 200, 32, MIN_PIXELS, MAX_PIXELS).is_ok());
    }

    #[test]
    fn a_python_float_keeps_its_decimal_point() {
        assert_eq!(python_float(400.0), "400.0");
        assert_eq!(python_float(200.5), "200.5");
        assert_eq!(python_float(201.0), "201.0");
        assert_eq!(python_float(1.0 / 3.0), "0.3333333333333333");
    }

    #[test]
    fn smart_resize_matches_the_runtime_budget_exactly() {
        // Exactly at the maximum: 1600x1024 = 1,638,400 px is kept.
        let (h, w) = smart_resize(1024, 1600, 32, MIN_PIXELS, MAX_PIXELS).unwrap();
        assert_eq!(h * w, 1024 * 1600);
    }

    /// A 4x4 image with a distinct gray value per pixel; patch 2, merge 2.
    ///
    /// The merge-block order of the four patches is
    /// `(0,0) (0,1) (1,0) (1,1)` in *blocks*, i.e. patch (row, col):
    /// block (0,0) -> patches (0,0) (0,1) (1,0) (1,1).
    #[test]
    fn patchify_uses_merge_block_order_and_channel_major_values() {
        let mut img = RgbImage::new(4, 4);
        for y in 0..4u32 {
            for x in 0..4u32 {
                let v = (y * 4 + x) as u8;
                img.put_pixel(
                    x,
                    y,
                    image::Rgb([v, v.wrapping_add(50), v.wrapping_add(100)]),
                );
            }
        }
        let cfg = config();
        let (values, patches) = patchify(&img, &cfg).unwrap();
        assert_eq!(patches, 4);
        // 3 channels * 2 temporal * 2 * 2 = 24 values per patch.
        assert_eq!(values.len(), 4 * 24);
        // First patch = image pixels (0,0) (0,1) (1,0) (1,1) in channel-major,
        // temporal-duplicated, row-major-within-the-patch order.
        let expected_pixels = [(0usize, 0usize), (0, 1), (1, 0), (1, 1)];
        let mut expected = Vec::new();
        for c in 0..3usize {
            for _t in 0..2 {
                for (y, x) in expected_pixels {
                    let v = (y * 4 + x) as u8;
                    let value = [v, v.wrapping_add(50), v.wrapping_add(100)][c];
                    expected.push(normalize_channel(value));
                }
            }
        }
        assert_eq!(&values[..24], &expected[..]);
        // Patch 1 is block (0, 1): pixels (0, 2) (0, 3) (1, 2) (1, 3).
        let second = normalize_channel(2);
        assert_eq!(values[24], second);
        // Block (1,0) starts at patch 2: pixels (2,0) (2,1) (3,0) (3,1).
        assert_eq!(values[48], normalize_channel(8));
    }

    #[test]
    fn patchify_refuses_a_misaligned_image() {
        let img = RgbImage::new(5, 4);
        assert!(patchify(&img, &config()).is_err_and(|e| e.to_string().contains("not divisible")));
    }

    #[test]
    fn a_prepared_image_carries_the_grid_and_token_count() {
        let cfg = config();
        let img = RgbImage::new(4, 4);
        let device = Device::default();
        let (values, patches) = patchify(&img, &cfg).unwrap();
        let grid = [1, 2, 2];
        let prepared = PreparedImage {
            pixel_values: Tensor::<2>::from_data(
                TensorData::new(values, [patches, cfg.patch_dim()]),
                &device,
            ),
            grid,
            tokens: cfg.merged_tokens(grid[1], grid[2]),
            source: (4, 4),
            resized: (4, 4),
        };
        assert_eq!(prepared.grid, [1, 2, 2]);
        assert_eq!(prepared.patches(), 4);
        assert_eq!(prepared.tokens, 1, "a 2x2 patch grid merges into one token");
        assert_eq!(prepared.patch_grid(), (2, 2));
        assert_eq!(prepared.pixel_values.dims(), [4, 24]);
    }

    #[test]
    fn normalization_endpoints_are_exact() {
        assert_eq!(normalize_channel(0), -1.0);
        assert!(((normalize_channel(255) - 1.0) as f64).abs() < 1e-7);
        assert_eq!(normalize_channel(127), (127.0f32 - 127.5) / 127.5);
    }
}
