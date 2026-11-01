//! Media loading and preprocessing (image, then audio/video).
//!
//! The image pipeline mirrors `Gemma4ImageProcessor`:
//! aspect-ratio-preserving resize (BICUBIC, antialias) to the largest size that
//! fits the patch budget and is divisible by `pooling_kernel_size * patch_size`,
//! rescale to [0, 1], patchify channel-last within each patch, and emit the
//! (x, y) patch position ids.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use base64::Engine;
use image::{DynamicImage, RgbImage};
use reqwest::header::CONTENT_TYPE;

use crate::server::telemetry;

/// One preprocessed image, ready for the vision tower.
pub struct PreparedImage {
    /// Patch pixels, `[num_patches, 3 * patch_size^2]`, channel-last within each patch.
    pub patches: Vec<f32>,
    /// Patch (x, y) grid coordinates (`x` = column, `y` = row).
    pub xs: Vec<u32>,
    pub ys: Vec<u32>,
    pub patch_h: usize,
    pub patch_w: usize,
    /// Patches in each dimension after pooling (`patch_h / k`, `patch_w / k`).
    pub soft_h: usize,
    pub soft_w: usize,
}

impl PreparedImage {
    pub fn num_patches(&self) -> usize {
        self.patch_h * self.patch_w
    }

    pub fn num_soft_tokens(&self) -> usize {
        self.soft_h * self.soft_w
    }
}

/// Largest aspect-ratio-preserving target size (divisible by
/// `pooling_kernel_size * patch_size`) that fits the patch budget. Mirrors
/// `get_aspect_ratio_preserving_size` in `image_processing_gemma4.py`.
pub fn aspect_ratio_preserving_size(
    height: usize,
    width: usize,
    patch_size: usize,
    max_patches: usize,
    pooling_kernel_size: usize,
) -> Result<(usize, usize)> {
    let total_px = (height * width) as f64;
    let target_px = (max_patches * patch_size * patch_size) as f64;
    let factor = (target_px / total_px).sqrt();
    let ideal_height = factor * height as f64;
    let ideal_width = factor * width as f64;
    let side_mult = pooling_kernel_size * patch_size;

    let mut target_height = (ideal_height / side_mult as f64).floor() as usize * side_mult;
    let mut target_width = (ideal_width / side_mult as f64).floor() as usize * side_mult;

    if target_height == 0 && target_width == 0 {
        bail!("cannot resize {height}x{width} to a nonzero size divisible by {side_mult}");
    }

    let max_side_length = (max_patches / (pooling_kernel_size * pooling_kernel_size)) * side_mult;
    if target_height == 0 {
        target_height = side_mult;
        target_width = ((width / height) * side_mult).min(max_side_length);
    } else if target_width == 0 {
        target_width = side_mult;
        target_height = ((height / width) * side_mult).min(max_side_length);
    }

    if target_height * target_width > max_patches * patch_size * patch_size {
        bail!(
            "resizing {height}x{width} to {target_height}x{target_width} exceeds \
             {max_patches} patches at patch_size {patch_size}"
        );
    }
    Ok((target_height, target_width))
}

/// Video metadata needed to replicate `sample_frames` (fps=1, max_frames=32,
/// uniform overflow).
pub struct VideoInfo {
    pub width: usize,
    pub height: usize,
    pub fps: f64,
    pub total_frames: usize,
    pub duration: f64,
}

pub fn video_info(path: &Path) -> Result<VideoInfo> {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames,duration",
            "-of",
            "json",
        ])
        .arg(path)
        .output()
        .context("run ffprobe (is it installed?)")?;
    if !out.status.success() {
        bail!(
            "ffprobe failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    let json: serde_json::Value =
        serde_json::from_slice(&out.stdout).context("parse ffprobe output")?;
    let stream = &json["streams"][0];
    let width = stream["width"].as_u64().context("ffprobe: no width")? as usize;
    let height = stream["height"].as_u64().context("ffprobe: no height")? as usize;
    let rate = stream["r_frame_rate"].as_str().unwrap_or("25/1");
    let fps = parse_rate(rate)?;
    let duration = stream["duration"]
        .as_str()
        .and_then(|d| d.parse::<f64>().ok())
        .unwrap_or(0.0);
    let total_frames = stream["nb_frames"]
        .as_str()
        .and_then(|n| n.parse::<usize>().ok())
        .unwrap_or_else(|| (duration * fps).round() as usize);
    if total_frames == 0 {
        bail!("ffprobe: video has no frames");
    }
    Ok(VideoInfo {
        width,
        height,
        fps,
        total_frames,
        duration,
    })
}

fn parse_rate(rate: &str) -> Result<f64> {
    let (num, den) = rate.split_once('/').context("unexpected r_frame_rate")?;
    let num: f64 = num.parse()?;
    let den: f64 = den.parse()?;
    if den == 0.0 {
        bail!("invalid frame rate {rate}");
    }
    Ok(num / den)
}

/// Frame indices for the EmbeddingGemma 2 video processor defaults
/// (`fps=1`, `max_frames=32`, `overflow_strategy="uniform"`).
pub fn sample_frame_indices(info: &VideoInfo, fps: f64, max_frames: usize) -> Vec<usize> {
    let step = info.fps / fps;
    let num_sampled = ((info.duration * fps) as usize).max(1);
    let mut idx: Vec<usize> = (0..num_sampled)
        .map(|i| ((i as f64 * step) as usize).min(info.total_frames - 1))
        .collect();
    if idx.len() > max_frames && max_frames > 1 {
        let n = idx.len();
        let picked: Vec<usize> = (0..max_frames)
            .map(|i| idx[(i as f64 * (n - 1) as f64 / (max_frames - 1) as f64) as usize])
            .collect();
        idx = picked;
    } else if max_frames == 1 {
        idx.truncate(1);
    }
    idx
}

/// Extract the selected frames (by native index) as RGB images.
///
/// `select` matches a frame at most once, so duplicates in `indices` (the
/// sampler clamps, so a duration that overestimates the frame count can repeat
/// the last index) are requested once and replicated afterwards — the reference
/// indexes the decoded array (`video[indices]`) and keeps every repeat.
#[telemetry::instrument(name = "video.frames", skip_all, fields(frames = indices.len()))]
pub fn extract_frames(path: &Path, info: &VideoInfo, indices: &[usize]) -> Result<Vec<RgbImage>> {
    let mut unique = indices.to_vec();
    unique.sort_unstable();
    unique.dedup();
    let sel = unique
        .iter()
        .map(|i| format!("eq(n\\,{i})"))
        .collect::<Vec<_>>()
        .join("+");
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args([
            "-vf",
            &format!("select='{sel}'"),
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ])
        .output()
        .context("run ffmpeg for frame extraction")?;
    if !out.status.success() {
        bail!(
            "ffmpeg frame extraction failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    let frame_bytes = info.width * info.height * 3;
    let data = out.stdout;
    if data.len() < frame_bytes {
        bail!(
            "ffmpeg returned {} bytes, expected at least one {frame_bytes}-byte frame",
            data.len()
        );
    }
    let mut decoded = Vec::with_capacity(unique.len());
    for chunk in data.chunks_exact(frame_bytes) {
        let img = RgbImage::from_raw(info.width as u32, info.height as u32, chunk.to_vec())
            .context("build frame image")?;
        decoded.push(img);
    }
    if decoded.len() != unique.len() {
        bail!(
            "ffmpeg returned {} frames for {} requested indices",
            decoded.len(),
            unique.len()
        );
    }
    // Frames come back in decode (index) order, so a binary search maps each
    // requested index (duplicates included) to its frame.
    let mut frames = Vec::with_capacity(indices.len());
    for &index in indices {
        let pos = unique.binary_search(&index).expect("requested index");
        frames.push(decoded[pos].clone());
    }
    Ok(frames)
}

/// Decode an image file to RGB8 (PIL `convert('RGB')` equivalent).
///
/// The decoder is chosen by the file's **content** (magic number), like
/// `PIL.Image.open`: JPEG goes through [`load_jpeg`] (mozjpeg/libjpeg-turbo), the
/// rest through the `image` crate.  The extension is not consulted — a JPEG saved
/// as `.png` still takes libjpeg, which is what PIL does.
#[telemetry::instrument(name = "image.decode", skip_all)]
pub fn load_image(path: &Path) -> Result<RgbImage> {
    let bytes = std::fs::read(path).with_context(|| format!("read image {}", path.display()))?;
    decode_image(&bytes).with_context(|| format!("decode image {}", path.display()))
}

/// Decode already-read image bytes to RGB8, choosing the decoder by content.
///
/// JPEGs go through mozjpeg (libjpeg-turbo), matching PIL's decoder; the Rust
/// `image` crate's zune-jpeg decoder differs by up to ±4 LSB, which is the only
/// remaining gap versus the HF reference on JPEG inputs.  A JPEG libjpeg cannot
/// handle (the CMYK/YCCK Adobe ones: libjpeg has no CMYK -> RGB conversion) falls
/// back to the `image` crate instead of aborting the process.
///
/// Every JPEG goes through [`load_jpeg`], which checks the container
/// ([`check_jpeg`]) before decoding: a truncated file (missing EOI, a cut segment)
/// or an unsupported sample precision is refused there instead of being silently
/// accepted by the fallback decoder, which is what PIL does (`OSError: image file
/// is truncated` / `Unsupported precision`).
///
/// 16-bit PNGs get PIL's own conversion rules rather than `to_rgb8`'s rounding,
/// because the checkpoints' processors see PIL pixels (see [`pil_rgb8`]).
pub fn decode_image(bytes: &[u8]) -> Result<RgbImage> {
    if is_jpeg(bytes) {
        // The container check comes first and never falls back: a truncated file
        // must not reach the more permissive `image` decoder.
        check_jpeg(bytes)?;
        match load_jpeg(bytes) {
            JpegOutcome::Rgb(img) => return Ok(img),
            JpegOutcome::Unsupported(error) => eprintln!(
                "mozjpeg cannot decode this JPEG ({error:#}); falling back to the image crate"
            ),
        }
    }
    let img = image::load_from_memory(bytes).context("decode the image bytes")?;
    Ok(pil_rgb8(&img))
}

/// What a mozjpeg decode produced.
enum JpegOutcome {
    /// The decoded pixels.
    Rgb(RgbImage),
    /// A JPEG libjpeg cannot convert to RGB (CMYK/YCCK, which has no libjpeg
    /// conversion); the `image` crate can decode it, so the caller falls back.
    Unsupported(anyhow::Error),
}

/// Whether these bytes start like a JPEG (`FF D8 FF`).
pub fn is_jpeg(bytes: &[u8]) -> bool {
    bytes.len() >= 3 && bytes[0] == 0xFF && bytes[1] == 0xD8 && bytes[2] == 0xFF
}

/// The RGB8 pixels of a decoded image with PIL's `convert("RGB")` semantics.
///
/// 8-bit, palette and float images go through `to_rgb8` (which already matches
/// PIL: luma is replicated, alpha dropped).  16-bit PNGs need Pillow's rules
/// instead, because `to_rgb8` scales them (`(v + 128) / 257`) while PIL does not:
///
/// * a 16-bit **grayscale** PNG opens as `I;16` and `convert("RGB")` **clips each
///   sample to `[0, 255]`** (so a 16-bit image reads as near-white);
/// * a 16-bit **colour** PNG (RGB or RGBA) opens as `RGB`/`RGBA` and keeps the
///   **high byte** (`v >> 8`), alpha dropped by the conversion;
/// * a 16-bit **gray+alpha** PNG is opened by Pillow as `RGBA`, i.e. the same
///   high-byte rule (`v >> 8`) replicated to RGB.
///
/// These are byte-exact against `PIL.Image.convert("RGB")` (see the unit tests).
fn pil_rgb8(img: &DynamicImage) -> RgbImage {
    fn from_rgb(mut pixels: impl Iterator<Item = [u8; 3]>, (w, h): (u32, u32)) -> RgbImage {
        let mut out = Vec::with_capacity((w as usize) * (h as usize) * 3);
        for pixel in &mut pixels {
            out.extend_from_slice(&pixel);
        }
        RgbImage::from_raw(w, h, out).expect("pil_rgb8: buffer sized from the image")
    }
    match img {
        DynamicImage::ImageLuma16(luma) => {
            let (w, h) = luma.dimensions();
            let pixels = luma.pixels().map(|p| {
                let v = p.0[0].min(255) as u8;
                [v, v, v]
            });
            from_rgb(pixels, (w, h))
        }
        DynamicImage::ImageLumaA16(luma) => {
            let (w, h) = luma.dimensions();
            let pixels = luma.pixels().map(|p| {
                let v = (p.0[0] >> 8) as u8;
                [v, v, v]
            });
            from_rgb(pixels, (w, h))
        }
        DynamicImage::ImageRgb16(rgb) => {
            let (w, h) = rgb.dimensions();
            let pixels = rgb
                .pixels()
                .map(|p| [p.0[0] >> 8, p.0[1] >> 8, p.0[2] >> 8].map(|v| v as u8));
            from_rgb(pixels, (w, h))
        }
        DynamicImage::ImageRgba16(rgba) => {
            let (w, h) = rgba.dimensions();
            let pixels = rgba
                .pixels()
                .map(|p| [p.0[0] >> 8, p.0[1] >> 8, p.0[2] >> 8].map(|v| v as u8));
            from_rgb(pixels, (w, h))
        }
        other => other.to_rgb8(),
    }
}

/// The SOF markers that describe a frame (`0xC4`/`0xC8`/`0xCC` are not SOFs).
fn is_sof(marker: u8) -> bool {
    (0xC0..=0xCF).contains(&marker) && !matches!(marker, 0xC4 | 0xC8 | 0xCC)
}

/// The big-endian segment length at `at`, which must cover at least its own two
/// bytes and stay inside `limit`.
fn segment_length(bytes: &[u8], at: usize, limit: usize) -> Result<usize> {
    if at + 2 > limit {
        bail!("truncated JPEG segment (no length)");
    }
    let length = usize::from(bytes[at]) << 8 | usize::from(bytes[at + 1]);
    if length < 2 {
        bail!("malformed JPEG segment length {length}");
    }
    if at + length > limit {
        bail!(
            "truncated JPEG segment ({length} bytes declared, {} available)",
            limit - at
        );
    }
    Ok(length)
}

/// Structural integrity checks on an encoded JPEG, the ones PIL's loader gets for
/// free: an SOI, well-formed marker segments, an 8-bit frame, and an EOI after the
/// first scan.
///
/// This is deliberately a *container* check, not a decoder: libjpeg tolerates a
/// missing EOI (and the `image` crate decodes the visible rows), while PIL raises
/// `image file is truncated`, so a cut-off upload must be refused before either
/// decoder runs.  A 12-bit frame is refused here too (libjpeg and the standard PIL
/// wheels are 8-bit only) rather than aborting inside libjpeg.
pub fn check_jpeg(bytes: &[u8]) -> Result<()> {
    if bytes.len() < 4 || bytes[0] != 0xFF || bytes[1] != 0xD8 {
        bail!("not a JPEG (no SOI marker)");
    }
    let limit = bytes.len();
    let mut at = 2usize;
    let mut frame = false;
    loop {
        // Every segment starts with 0xFF (fill bytes may repeat it).
        if at >= limit {
            bail!("truncated JPEG (no EOI marker)");
        }
        if bytes[at] != 0xFF {
            bail!("malformed JPEG (expected a marker at offset {at})");
        }
        while at < limit && bytes[at] == 0xFF {
            at += 1;
        }
        if at >= limit {
            bail!("truncated JPEG (no EOI marker)");
        }
        let marker = bytes[at];
        at += 1;
        match marker {
            // EOI.
            0xD9 => {
                if !frame {
                    bail!("malformed JPEG (EOI before any frame header)");
                }
                return Ok(());
            }
            // A stuffed 0xFF inside entropy data, or a stand-alone marker.
            0x00 | 0x01 | 0xD0..=0xD7 => continue,
            0xD8 => bail!("malformed JPEG (a second SOI after the first)"),
            0xDA => {
                // SOS: skip its header, then walk the entropy-coded data to the
                // next real marker (FF 00 is stuffing, FF D0-D7 a restart).
                let length = segment_length(bytes, at, limit)?;
                at += length;
                loop {
                    if at + 1 >= limit {
                        bail!("truncated JPEG (the scan has no EOI marker)");
                    }
                    if bytes[at] != 0xFF {
                        at += 1;
                        continue;
                    }
                    let next = bytes[at + 1];
                    if next == 0x00 || (0xD0..=0xD7).contains(&next) {
                        at += 2;
                        continue;
                    }
                    break;
                }
            }
            _ => {
                let length = segment_length(bytes, at, limit)?;
                if is_sof(marker) {
                    if length < 8 {
                        bail!("malformed JPEG frame header ({length} bytes)");
                    }
                    let precision = bytes[at + 2];
                    if precision != 8 {
                        bail!("unsupported JPEG sample precision {precision} (only 8-bit)");
                    }
                    frame = true;
                }
                at += length;
            }
        }
    }
}

/// JPEG decode via mozjpeg (libjpeg-turbo), PIL-compatible.
///
/// libjpeg reports a fatal problem by calling `error_exit`, which mozjpeg turns
/// into a `resume_unwind` (an abort for an unguarded call) — and that fires from
/// `jpeg_read_header` too, i.e. inside [`mozjpeg::Decompress::new_mem`] on a
/// malformed header.  The whole decode, header included, therefore runs inside
/// `catch_unwind`; a CMYK or YCCK source comes back as [`JpegOutcome::Unsupported`]
/// before decompression starts (libjpeg has no CMYK -> RGB conversion), so
/// [`decode_image`] can fall back to the `image` crate for exactly that case and
/// no other.
fn load_jpeg(bytes: &[u8]) -> JpegOutcome {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| decode_jpeg(bytes))) {
        Ok(Ok(image)) => JpegOutcome::Rgb(image),
        Ok(Err(error)) => JpegOutcome::Unsupported(error),
        // libjpeg's `unwind_error_exit` resumes with a `String` — the formatted
        // error message — and deliberately skips the panic hook.  A `String`
        // payload is therefore libjpeg's own fatal; any other payload is a real
        // panic (an `expect` in mozjpeg, our own code) and is re-raised so it
        // still reaches the caller's panic policy.
        Err(payload) => match payload.downcast::<String>() {
            Ok(message) => {
                JpegOutcome::Unsupported(anyhow!("libjpeg aborted the decode: {message}"))
            }
            Err(payload) => std::panic::resume_unwind(payload),
        },
    }
}

/// Open, vet and decode one JPEG.  Every libjpeg fatal in here is a panic.
///
/// `Err` is reserved for the CMYK/YCCK refusal; any other libjpeg fatal arrives as
/// a panic payload and reaches [`load_jpeg`]'s guard, which also treats it as
/// "mozjpeg cannot decode this": the container was already checked, so what is
/// left is a decoder limitation (the CMYK-class files the `image` crate handles),
/// not a broken upload.
fn decode_jpeg(bytes: &[u8]) -> Result<RgbImage> {
    let decompress = mozjpeg::Decompress::new_mem(bytes).context("open jpeg")?;
    let space = decompress.color_space();
    if matches!(
        space,
        mozjpeg::ColorSpace::JCS_CMYK | mozjpeg::ColorSpace::JCS_YCCK
    ) {
        bail!("the JPEG stores CMYK/YCCK (libjpeg cannot convert it to RGB)");
    }
    decode_rgb(decompress)
}

fn decode_rgb<R: std::io::Read>(decompress: mozjpeg::Decompress<R>) -> Result<RgbImage> {
    let mut image = decompress.rgb().context("convert the jpeg to RGB")?;
    let width = image.width() as u32;
    let height = image.height() as u32;
    let pixels: Vec<u8> = image.read_scanlines().context("read jpeg scanlines")?;
    let buf = RgbImage::from_raw(width, height, pixels)
        .context("jpeg scanline buffer has an unexpected size")?;
    Ok(buf)
}

/// Cubic convolution kernel (Keys, a = -0.5), matching PIL and ATen's
/// `HelperInterpCubic::aa_filter`.
fn bicubic_kernel(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 {
        ((1.5 * x - 2.5) * x) * x + 1.0
    } else if x < 2.0 {
        (((-0.5 * x + 2.5) * x) - 4.0) * x + 2.0
    } else {
        0.0
    }
}

/// Per-axis antialias resampling plan (ATen `_compute_indices_min_size_weights_aa`
/// + `_compute_index_ranges_int16_weights`): integer taps and weights with a
///   per-axis precision, applied in integer arithmetic.
struct AxisPlan {
    xmin: Vec<usize>,
    size: Vec<usize>,
    weights: Vec<i16>,
    max_interp: usize,
    precision: u32,
}

impl AxisPlan {
    /// `interp_size` 4 (bicubic); antialias support `2 * max(1, scale)`.
    fn new(input_size: usize, output_size: usize) -> Self {
        let scale = input_size as f64 / output_size as f64;
        let support = if scale >= 1.0 { 2.0 * scale } else { 2.0 };
        let max_interp = (support.ceil() as usize) * 2 + 1;
        let invscale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };

        let mut xmin = vec![0usize; output_size];
        let mut size = vec![0usize; output_size];
        let mut ws = vec![0f64; output_size * max_interp];
        let mut wt_max = 0f64;
        for i in 0..output_size {
            let center = scale * (i as f64 + 0.5);
            let lo = ((center - support + 0.5) as i64).max(0) as usize;
            let hi = ((center + support + 0.5) as i64).min(input_size as i64) as usize;
            let xsize = hi.saturating_sub(lo).min(max_interp);
            let mut total = 0.0;
            for j in 0..xsize {
                let w = bicubic_kernel((j as f64 + lo as f64 - center + 0.5) * invscale);
                ws[i * max_interp + j] = w;
                total += w;
            }
            if total != 0.0 {
                for j in 0..xsize {
                    let w = ws[i * max_interp + j] / total;
                    ws[i * max_interp + j] = w;
                    wt_max = wt_max.max(w);
                }
            }
            xmin[i] = lo;
            size[i] = xsize;
        }

        // Choose the largest precision that keeps the biggest weight in int16.
        let mut precision = 0u32;
        while precision < 22 {
            let next = (0.5 + wt_max * ((1u64 << (precision + 1)) as f64)) as i64;
            if next >= (1 << 15) {
                break;
            }
            precision += 1;
        }
        let mut weights = vec![0i16; output_size * max_interp];
        for i in 0..output_size {
            for j in 0..size[i] {
                let v = ws[i * max_interp + j] * ((1u64 << precision) as f64);
                weights[i * max_interp + j] = if v < 0.0 {
                    (-0.5 + v) as i32 as i16
                } else {
                    (0.5 + v) as i32 as i16
                };
            }
        }
        Self {
            xmin,
            size,
            weights,
            max_interp,
            precision,
        }
    }

    /// One pass along a contiguous axis: `src` is `[rows, in_len]`, `out` is
    /// `[rows, out_len]`, both u8 (ATen's uint8 separable loop).
    fn apply(&self, src: &[u8], rows: usize, in_len: usize, out_len: usize) -> Vec<u8> {
        let mut out = vec![0u8; rows * out_len];
        for r in 0..rows {
            let row = &src[r * in_len..(r + 1) * in_len];
            for o in 0..out_len {
                let lo = self.xmin[o];
                let n = self.size[o];
                let mut acc: i32 = if self.precision > 0 {
                    1 << (self.precision - 1)
                } else {
                    0
                };
                for j in 0..n {
                    let w = self.weights[o * self.max_interp + j] as i32;
                    acc += row[lo + j] as i32 * w;
                }
                let v = if self.precision > 0 {
                    acc >> self.precision
                } else {
                    acc
                };
                out[r * out_len + o] = v.clamp(0, 255) as u8;
            }
        }
        out
    }
}

/// BICUBIC resize with the antialias window, matching the uint8 torchvision/PIL
/// resample path bit-for-bit (ATen `upsample_bicubic2d_aa` uint8 kernel):
/// horizontal pass into a uint8 intermediate, then a vertical pass, both in
/// integer arithmetic with int16 weights.
pub fn resize_bicubic_aa(img: &RgbImage, out_w: usize, out_h: usize) -> RgbImage {
    let (in_w, in_h) = (img.width() as usize, img.height() as usize);
    let raw = img.as_raw();
    let plane_len = in_w * in_h;
    // Split into per-channel planes (ATen treats N*C as the outer dimension).
    let mut planes = vec![vec![0u8; plane_len]; 3];
    for (i, chunk) in raw.as_chunks::<3>().0.iter().enumerate() {
        planes[0][i] = chunk[0];
        planes[1][i] = chunk[1];
        planes[2][i] = chunk[2];
    }

    let hplan = AxisPlan::new(in_w, out_w);
    let vplan = AxisPlan::new(in_h, out_h);
    let mut out = vec![0u8; out_w * out_h * 3];
    for c in 0..3 {
        let horiz = hplan.apply(&planes[c], in_h, in_w, out_w);
        // Transpose to [out_w, in_h] so the vertical pass sees contiguous columns.
        let mut transposed = vec![0u8; out_w * in_h];
        for y in 0..in_h {
            for x in 0..out_w {
                transposed[x * in_h + y] = horiz[y * out_w + x];
            }
        }
        let vert = vplan.apply(&transposed, out_w, in_h, out_h);
        for x in 0..out_w {
            for y in 0..out_h {
                out[(y * out_w + x) * 3 + c] = vert[x * out_h + y];
            }
        }
    }
    RgbImage::from_raw(out_w as u32, out_h as u32, out).expect("resized image buffer")
}

/// Aspect-ratio-preserving resize (BICUBIC) to the patch-budget target size.
#[telemetry::instrument(name = "image.resize", skip_all)]
pub fn resize_rgb(
    img: &RgbImage,
    patch_size: usize,
    max_soft_tokens: usize,
    pooling_kernel_size: usize,
) -> Result<RgbImage> {
    let (h, w) = (img.height() as usize, img.width() as usize);
    let max_patches = max_soft_tokens * pooling_kernel_size * pooling_kernel_size;
    let (target_h, target_w) =
        aspect_ratio_preserving_size(h, w, patch_size, max_patches, pooling_kernel_size)?;
    if (target_h, target_w) == (h, w) {
        return Ok(img.clone());
    }
    Ok(resize_bicubic_aa(img, target_w, target_h))
}

/// Dump an RGB8 image as `width:u32 | height:u32 | rgb bytes` for debugging.
pub fn dump_rgb(path: &Path, img: &RgbImage) -> Result<()> {
    let (w, h) = (img.width(), img.height());
    let mut bytes = Vec::with_capacity(8 + img.as_raw().len());
    bytes.extend_from_slice(&w.to_le_bytes());
    bytes.extend_from_slice(&h.to_le_bytes());
    bytes.extend_from_slice(img.as_raw());
    std::fs::write(path, bytes).with_context(|| format!("write {}", path.display()))?;
    println!("dumped image {w}x{h} to {}", path.display());
    Ok(())
}

/// Patchify an already-resized RGB image: patches in row-major order, each
/// flattened as (patch-row, patch-col, channel), rescaled to [0, 1].
pub fn patchify(
    img: &RgbImage,
    patch_size: usize,
    pooling_kernel_size: usize,
) -> Result<PreparedImage> {
    let (h, w) = (img.height() as usize, img.width() as usize);
    if h % patch_size != 0 || w % patch_size != 0 {
        bail!("image {w}x{h} is not divisible by patch size {patch_size}");
    }
    let patch_h = h / patch_size;
    let patch_w = w / patch_size;
    let channels = 3;
    let mut patches = Vec::with_capacity(patch_h * patch_w * patch_size * patch_size * channels);
    let raw = img.as_raw();
    let stride = w * channels;
    for py in 0..patch_h {
        for px in 0..patch_w {
            for i in 0..patch_size {
                let y = py * patch_size + i;
                for j in 0..patch_size {
                    let x = px * patch_size + j;
                    let o = y * stride + x * channels;
                    patches.push(raw[o] as f32 / 255.0);
                    patches.push(raw[o + 1] as f32 / 255.0);
                    patches.push(raw[o + 2] as f32 / 255.0);
                }
            }
        }
    }
    let mut xs = Vec::with_capacity(patch_h * patch_w);
    let mut ys = Vec::with_capacity(patch_h * patch_w);
    for py in 0..patch_h {
        for px in 0..patch_w {
            xs.push(px as u32);
            ys.push(py as u32);
        }
    }
    let k = pooling_kernel_size;
    if !patch_h.is_multiple_of(k) || !patch_w.is_multiple_of(k) {
        bail!("patch grid {patch_w}x{patch_h} is not divisible by pooling {k}");
    }
    Ok(PreparedImage {
        patches,
        xs,
        ys,
        patch_h,
        patch_w,
        soft_h: patch_h / k,
        soft_w: patch_w / k,
    })
}

/// Upper bound for a fetched media file (base64 uploads are bounded by the
/// HTTP body limit instead).
pub(crate) const MAX_FETCH_BYTES: u64 = 20 * 1024 * 1024;

/// A media value resolved to a local file. Uploads (base64 data URIs) and
/// downloaded URLs live in the temp dir and are removed when the guard drops,
/// so the caller must keep it alive until the media has been decoded.
#[derive(Debug)]
pub(crate) struct ResolvedMedia {
    path: PathBuf,
    temp: bool,
}

impl ResolvedMedia {
    fn local(path: PathBuf) -> Self {
        Self { path, temp: false }
    }

    fn temp(path: PathBuf) -> Self {
        Self { path, temp: true }
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for ResolvedMedia {
    fn drop(&mut self) {
        if self.temp {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// A media value is either a filesystem path or a `data:<mime>;base64,<payload>`
/// blob (written to a temp file so ffmpeg/hound can read it).
pub(crate) fn resolve_media(value: &str, ext: &str) -> Result<ResolvedMedia, String> {
    let Some(rest) = value.strip_prefix("data:") else {
        return Ok(ResolvedMedia::local(PathBuf::from(value)));
    };
    let (mime, b64) = rest
        .split_once(";base64,")
        .ok_or("bad data URI (expected data:<mime>;base64,<payload>)")?;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64)
        .map_err(|e| format!("base64: {e}"))?;
    let path = write_temp(&bytes, &extension_for_mime(mime, ext))?;
    Ok(ResolvedMedia::temp(path))
}

/// Like [`resolve_media`], but also downloads `http(s)` URLs. Blocking: call
/// this from a blocking thread (the server's compute path), never from an async
/// handler. `ROCKET_MEDIA_FETCH=0` disables the download.
#[telemetry::instrument(name = "media.fetch", skip_all)]
pub(crate) fn resolve_media_remote(value: &str, ext: &str) -> Result<ResolvedMedia, String> {
    if !value.starts_with("http://") && !value.starts_with("https://") {
        return resolve_media(value, ext);
    }
    if std::env::var("ROCKET_MEDIA_FETCH").ok().as_deref() == Some("0") {
        return Err("remote media fetching is disabled (ROCKET_MEDIA_FETCH=0)".to_string());
    }
    fetch_remote(value, ext)
}

fn http_client() -> &'static reqwest::blocking::Client {
    static CLIENT: OnceLock<reqwest::blocking::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(10))
            .redirect(reqwest::redirect::Policy::limited(3))
            // Some hosts (Wikimedia included) reject requests without one.
            .user_agent(concat!("rocket-inference/", env!("CARGO_PKG_VERSION")))
            .build()
            .expect("build the media HTTP client")
    })
}

fn fetch_remote(url: &str, ext: &str) -> Result<ResolvedMedia, String> {
    let resp = http_client()
        .get(url)
        .send()
        .map_err(|e| format!("fetch {url}: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("fetch {url}: HTTP {}", resp.status()));
    }
    if let Some(len) = resp.content_length()
        && len > MAX_FETCH_BYTES
    {
        return Err(format!(
            "fetch {url}: {len} bytes exceeds the {} MiB limit",
            MAX_FETCH_BYTES / (1024 * 1024)
        ));
    }
    // The decoders pick their backend by file extension, so take it from the
    // response's MIME type when the server sends one.
    let mime = resp
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|v| v.split(';').next().unwrap_or(v).trim().to_string())
        .unwrap_or_default();
    let mut bytes = Vec::new();
    resp.take(MAX_FETCH_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| format!("read {url}: {e}"))?;
    if bytes.len() as u64 > MAX_FETCH_BYTES {
        return Err(format!(
            "fetch {url}: response exceeds the {} MiB limit",
            MAX_FETCH_BYTES / (1024 * 1024)
        ));
    }
    let path = write_temp(&bytes, &extension_for_mime(&mime, ext))?;
    Ok(ResolvedMedia::temp(path))
}

fn extension_for_mime(mime: &str, fallback: &str) -> String {
    match mime.split('/').nth(1) {
        Some("jpeg") => "jpg".to_string(),
        Some("quicktime") => "mp4".to_string(),
        Some(other) if !other.is_empty() => other.to_string(),
        _ => fallback.to_string(),
    }
}

fn write_temp(bytes: &[u8], ext: &str) -> Result<PathBuf, String> {
    let dir = std::env::temp_dir().join("rocket-inference");
    std::fs::create_dir_all(&dir).map_err(|e| format!("temp dir: {e}"))?;
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let path = dir.join(format!("upload-{}-{nanos}.{ext}", std::process::id()));
    std::fs::write(&path, bytes).map_err(|e| format!("write {}: {e}", path.display()))?;
    Ok(path)
}

/// Write `bytes` to a temporary file with this extension and return it as a
/// [`ResolvedMedia`], so the file is removed when the guard drops.
///
/// The decoders (ffmpeg, hound) pick their backend by file extension, so a caller
/// that only has encoded bytes needs a named file on disk; the d3 video path uses
/// this for `data:video/...` uploads and removes the file after decoding.
pub(crate) fn temporary_file(bytes: &[u8], ext: &str) -> Result<ResolvedMedia, String> {
    Ok(ResolvedMedia::temp(write_temp(bytes, ext)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(fps: f64, duration: f64, total_frames: usize) -> VideoInfo {
        VideoInfo {
            width: 480,
            height: 360,
            fps,
            duration,
            total_frames,
        }
    }

    /// Expected indices from the reference `EmbeddingGemma2VideoProcessor`
    /// (`sample_frames`: fps sampling, then a `linspace` uniform overflow cap).
    #[test]
    fn frame_indices_match_the_reference_processor() {
        let cases: &[(f64, f64, usize, f64, usize, &[usize])] = &[
            // `data/test.mp4`: 4 s at 8 fps -> 4 frames at the 1 fps default.
            (8.0, 4.0, 32, 1.0, 32, &[0, 8, 16, 24]),
            // 40 s at 8 fps -> 40 sampled -> uniform overflow to the 32-frame budget.
            (
                8.0,
                40.0,
                320,
                1.0,
                32,
                &[
                    0, 8, 16, 24, 40, 48, 56, 64, 80, 88, 96, 104, 120, 128, 136, 144, 160, 168,
                    176, 184, 200, 208, 216, 224, 240, 248, 256, 264, 280, 288, 296, 312,
                ],
            ),
            // 10 s at 30 fps -> 10 frames, no overflow.
            (
                30.0,
                10.0,
                300,
                1.0,
                32,
                &[0, 30, 60, 90, 120, 150, 180, 210, 240, 270],
            ),
            // NTSC rate (30000/1001); `int(i * step)` truncates each index.
            (
                30000.0 / 1001.0,
                10.01,
                300,
                1.0,
                32,
                &[0, 29, 59, 89, 119, 149, 179, 209, 239, 269],
            ),
            // Sampling at the native rate keeps every frame.
            (
                8.0,
                4.0,
                32,
                8.0,
                32,
                &[
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                    22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                ],
            ),
            // A one-frame budget keeps the first sampled frame.
            (8.0, 4.0, 32, 1.0, 1, &[0]),
        ];
        for &(fps, duration, total_frames, target_fps, max_frames, expected) in cases {
            let indices =
                sample_frame_indices(&info(fps, duration, total_frames), target_fps, max_frames);
            assert_eq!(
                indices.as_slice(),
                expected,
                "fps {fps} duration {duration} total {total_frames} \
                 target {target_fps} max {max_frames}"
            );
        }
    }

    /// The index is clamped to the last decoded frame when the container
    /// duration overestimates the frame count.
    #[test]
    fn frame_indices_are_clamped_to_the_last_frame() {
        let indices = sample_frame_indices(&info(8.0, 5.0, 32), 1.0, 32);
        assert_eq!(indices, vec![0, 8, 16, 24, 31]);
    }

    #[test]
    fn parses_rational_frame_rates() {
        assert!((parse_rate("30000/1001").unwrap() - 29.970_029_970_03).abs() < 1e-9);
        assert!(parse_rate("0/0").is_err());
        assert!(parse_rate("25").is_err());
    }

    /// PIL's `convert("RGB")` rules for 16-bit PNGs, with the exact samples the
    /// review probe used (`0, 255, 256, 1000, 65535`).
    ///
    /// The expectations are `PIL.Image.convert("RGB")` output, measured on
    /// Pillow 11: a 16-bit grayscale (`I;16`) is **clipped** to `[0, 255]`, while
    /// 16-bit colour keeps the **high byte** and Pillow's 16-bit gray+alpha is
    /// opened as RGBA and follows the high-byte rule too (alpha dropped).
    #[test]
    fn sixteen_bit_pngs_follow_pils_convert_rules() {
        use image::{ImageBuffer, Luma, LumaA, Rgb, Rgba};
        // `Gray16Image`/`GrayAlpha16Image` are crate-private in `image`, so name
        // the buffers by their pixel types.
        type Luma16 = ImageBuffer<Luma<u16>, Vec<u16>>;
        type LumaA16 = ImageBuffer<LumaA<u16>, Vec<u16>>;
        type Rgb16 = ImageBuffer<Rgb<u16>, Vec<u16>>;
        type Rgba16 = ImageBuffer<Rgba<u16>, Vec<u16>>;

        let luma: Luma16 =
            Luma16::from_raw(3, 2, vec![0, 255, 256, 1000, 32768, 65535]).expect("luma buffer");
        let rgb = pil_rgb8(&DynamicImage::ImageLuma16(luma));
        assert_eq!(
            rgb.as_raw(),
            &[
                0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255
            ],
            "a 16-bit gray sample is clipped, not scaled"
        );

        let rgb16: Rgb16 = Rgb16::from_raw(
            2,
            2,
            vec![
                0, 255, 256, // -> (0, 0, 1)
                1000, 65535, 32768, // -> (3, 255, 128)
                257, 512, 1023, // -> (1, 2, 3)
                65535, 256, 0, // -> (255, 1, 0)
            ],
        )
        .expect("rgb16 buffer");
        let rgb = pil_rgb8(&DynamicImage::ImageRgb16(rgb16));
        assert_eq!(rgb.as_raw(), &[0, 0, 1, 3, 255, 128, 1, 2, 3, 255, 1, 0]);

        let rgba16: Rgba16 =
            Rgba16::from_raw(2, 1, vec![0, 255, 256, 7, 1000, 65535, 32768, 65535])
                .expect("rgba16 buffer");
        let rgb = pil_rgb8(&DynamicImage::ImageRgba16(rgba16));
        assert_eq!(rgb.as_raw(), &[0, 0, 1, 3, 255, 128], "alpha is dropped");

        let la16: LumaA16 = LumaA16::from_raw(2, 1, vec![0, 7, 1000, 65535]).expect("la16 buffer");
        let rgb = pil_rgb8(&DynamicImage::ImageLumaA16(la16));
        assert_eq!(
            rgb.as_raw(),
            &[0, 0, 0, 3, 3, 3],
            "Pillow opens LA16 as RGBA"
        );
    }

    /// 8-bit images keep the `to_rgb8` mapping (luma replicated, alpha dropped),
    /// which is PIL's: the EmbeddingGemma tower's pixels must not move.
    #[test]
    fn eight_bit_images_keep_the_to_rgb8_mapping() {
        use image::{GrayAlphaImage, GrayImage, RgbaImage};

        let luma: GrayImage = GrayImage::from_raw(2, 1, vec![0, 200]).unwrap();
        assert_eq!(
            pil_rgb8(&DynamicImage::ImageLuma8(luma)).as_raw(),
            &[0, 0, 0, 200, 200, 200]
        );

        let la: GrayAlphaImage = GrayAlphaImage::from_raw(2, 1, vec![7, 0, 9, 255]).unwrap();
        assert_eq!(
            pil_rgb8(&DynamicImage::ImageLumaA8(la)).as_raw(),
            &[7, 7, 7, 9, 9, 9]
        );

        let rgba: RgbaImage = RgbaImage::from_raw(1, 1, vec![10, 20, 30, 40]).expect("rgba buffer");
        assert_eq!(
            pil_rgb8(&DynamicImage::ImageRgba8(rgba)).as_raw(),
            &[10, 20, 30]
        );

        let rgb = RgbImage::from_raw(1, 1, vec![1, 2, 3]).unwrap();
        assert_eq!(pil_rgb8(&DynamicImage::ImageRgb8(rgb)).as_raw(), &[1, 2, 3]);
    }

    /// A CMYK (Adobe) JPEG must not abort the process: libjpeg has no CMYK to RGB
    /// conversion and mozjpeg raises it as a `resume_unwind`, so the loader has to
    /// fall back to the `image` crate.  `data/cmyk-test-rgb.raw` is PIL's
    /// `convert("RGB")` of `data/cmyk-test.jpg`; zune-jpeg agrees within one LSB.
    #[test]
    fn a_cmyk_jpeg_decodes_without_aborting() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
        let expected = std::fs::read(dir.join("cmyk-test-rgb.raw")).expect("cmyk-test-rgb.raw");
        let image = load_image(&dir.join("cmyk-test.jpg")).expect("a CMYK JPEG decodes");
        assert_eq!((image.width(), image.height()), (32, 24));
        let ours = image.as_raw();
        assert_eq!(ours.len(), expected.len());
        let worst = ours
            .iter()
            .zip(&expected)
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap_or(0);
        assert!(
            worst <= 2,
            "CMYK fallback differs from PIL by up to {worst} LSB"
        );
    }

    /// The marker offsets (the `0xFF` of each segment) of a JPEG, in file order.
    fn jpeg_markers(data: &[u8]) -> Vec<usize> {
        let mut offsets = Vec::new();
        let mut i = 2; // past SOI
        while i + 4 <= data.len() {
            if data[i] != 0xFF {
                i += 1;
                continue;
            }
            let marker = data[i + 1];
            offsets.push(i);
            if marker == 0xD9 {
                break;
            }
            if marker == 0xD8 || (0xD0..=0xD7).contains(&marker) {
                i += 2;
                continue;
            }
            let length = usize::from(data[i + 2]) << 8 | usize::from(data[i + 3]);
            if length < 2 {
                break;
            }
            i += 2 + length;
        }
        offsets
    }

    /// A malformed JPEG must never abort the process, and must not be accepted by
    /// the fallback decoder either.
    ///
    /// `jpeg_read_header` — called inside `Decompress::new_mem`, before any pixel
    /// is read — fires libjpeg's `error_exit` for an unsupported SOF precision or a
    /// bogus segment length, and mozjpeg raises that as a resumed unwind (a silent
    /// exit 101 before the guard covered the header).  `check_jpeg` now refuses
    /// both structurally, so the file is a clean error: the `image` crate would
    /// otherwise accept a 12-bit frame or read past a nonsense segment length.
    #[test]
    fn a_malformed_jpeg_header_does_not_abort() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
        let source = std::fs::read(dir.join("cat.jpeg")).expect("cat.jpeg");
        let markers = jpeg_markers(&source);
        let sof = markers
            .iter()
            .copied()
            .find(|&i| {
                matches!(
                    source[i + 1],
                    0xC0 | 0xC1 | 0xC2 | 0xC3 | 0xC5 | 0xC6 | 0xC7 | 0xC9 | 0xCA | 0xCB
                )
            })
            .expect("cat.jpeg has an SOF segment");
        assert_eq!(source[sof + 4], 8, "cat.jpeg is an 8-bit JPEG");
        let app0 = markers
            .iter()
            .copied()
            .find(|&i| source[i + 1] == 0xE0)
            .expect("cat.jpeg has an APP0 segment");

        let mut sof12 = source.clone();
        sof12[sof + 4] = 12;
        let mut long_app0 = source.clone();
        long_app0[app0 + 2..app0 + 4].copy_from_slice(&[0xFF, 0xFF]);

        check_jpeg(&source).expect("cat.jpeg is structurally sound");

        for (label, bytes, expected) in [
            (
                "SOF-precision-12",
                sof12,
                "unsupported JPEG sample precision 12",
            ),
            ("APP0-length-ffff", long_app0, "truncated JPEG segment"),
        ] {
            let message = format!("{:#}", check_jpeg(&bytes).expect_err(label));
            assert!(message.contains(expected), "{label}: {message}");
            // Every entry point reports the same clean error, no decode attempt.
            let path = std::env::temp_dir().join(format!(
                "rocket-malformed-{}-{label}.jpg",
                std::process::id()
            ));
            std::fs::write(&path, &bytes).expect("write the malformed jpeg");
            assert!(decode_image(&bytes).is_err(), "{label}");
            let err = load_image(&path).expect_err(label);
            assert!(
                format!("{err:#}").contains("decode image"),
                "{label}: {err:#}"
            );
            std::fs::remove_file(&path).ok();
        }
    }

    /// A truncated JPEG (the scan cut off, no EOI) is refused even though libjpeg
    /// and the `image` crate both decode the visible rows; PIL raises
    /// `image file is truncated`.  A missing EOI and a cut entropy scan are the two
    /// shapes an interrupted upload takes.
    #[test]
    fn a_truncated_jpeg_is_refused() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
        let source = std::fs::read(dir.join("cat.jpeg")).expect("cat.jpeg");
        assert!(source.ends_with(&[0xFF, 0xD9]), "cat.jpeg ends with EOI");
        // The control's cut: the last 36 bytes (EOI + trailing scan) are gone.
        let no_eoi = &source[..source.len() - 36];
        let message = format!("{:#}", check_jpeg(no_eoi).expect_err("no EOI"));
        assert!(message.contains("truncated JPEG"), "{message}");
        // A hard cut in the middle of the entropy-coded data.
        let half = &source[..source.len() / 2];
        assert!(check_jpeg(half).is_err());
        // Both must fail every entry point, and neither may fall back.
        assert!(decode_image(no_eoi).is_err());
        assert!(decode_image(half).is_err());
        // The intact file still decodes to the same pixels as before the check.
        let intact = decode_image(&source).expect("cat.jpeg decodes");
        assert_eq!((intact.width(), intact.height()), (960, 686));
    }

    /// `decode_image` picks the decoder by content, not by what a caller claims.
    #[test]
    fn the_decoder_is_chosen_by_magic() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
        let jpeg = std::fs::read(dir.join("cat.jpeg")).expect("cat.jpeg");
        assert!(is_jpeg(&jpeg));
        // The same bytes under a `.png` name and with no extension at all.
        for name in ["magic-lie.png", "magic-other.bin"] {
            let path = std::env::temp_dir().join(format!("rocket-{}-{name}", std::process::id()));
            std::fs::write(&path, &jpeg).expect("write the jpeg");
            let image = load_image(&path).expect("a JPEG decodes whatever its name says");
            assert_eq!((image.width(), image.height()), (960, 686), "{name}");
            std::fs::remove_file(&path).ok();
        }
        // A PNG is not a JPEG and keeps the `image` decoder (and PIL's RGB
        // mapping): encode one in memory so the test needs no fixture.
        let mut encoded = std::io::Cursor::new(Vec::new());
        let rgb = RgbImage::from_raw(2, 1, vec![1, 2, 3, 4, 5, 6]).expect("2x1 rgb");
        DynamicImage::ImageRgb8(rgb)
            .write_to(&mut encoded, image::ImageFormat::Png)
            .expect("encode a png");
        let png = encoded.into_inner();
        assert!(!is_jpeg(&png));
        let decoded = decode_image(&png).expect("a PNG decodes");
        assert_eq!(decoded.as_raw(), &[1, 2, 3, 4, 5, 6]);
    }
}
