//! Media loading and preprocessing (image, then audio/video).
//!
//! The image pipeline mirrors `Gemma4ImageProcessor`:
//! aspect-ratio-preserving resize (BICUBIC, antialias) to the largest size that
//! fits the patch budget and is divisible by `pooling_kernel_size * patch_size`,
//! rescale to [0, 1], patchify channel-last within each patch, and emit the
//! (x, y) patch position ids.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use base64::Engine;
use image::RgbImage;


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
        bail!(
            "cannot resize {height}x{width} to a nonzero size divisible by {side_mult}"
        );
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
    let (num, den) = rate
        .split_once('/')
        .context("unexpected r_frame_rate")?;
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
pub fn extract_frames(path: &Path, info: &VideoInfo, indices: &[usize]) -> Result<Vec<RgbImage>> {
    let sel = indices
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
    let mut frames = Vec::with_capacity(data.len() / frame_bytes);
    for chunk in data.chunks_exact(frame_bytes) {
        let img = RgbImage::from_raw(info.width as u32, info.height as u32, chunk.to_vec())
            .context("build frame image")?;
        frames.push(img);
    }
    Ok(frames)
}

/// Decode an image file to RGB8 (PIL `convert('RGB')` equivalent).
///
/// JPEGs go through mozjpeg (libjpeg-turbo), matching PIL's decoder; the Rust
/// `image` crate's zune-jpeg decoder differs by up to ±4 LSB, which is the only
/// remaining gap versus the HF reference on JPEG inputs.
pub fn load_image(path: &Path) -> Result<RgbImage> {
    let is_jpeg = path
        .extension()
        .map(|e| matches!(e.to_string_lossy().to_ascii_lowercase().as_str(), "jpg" | "jpeg"))
        .unwrap_or(false);
    if is_jpeg {
        match load_jpeg(path) {
            Ok(img) => return Ok(img),
            Err(e) => eprintln!("mozjpeg decode failed ({e}); falling back to the image crate"),
        }
    }
    let img = image::open(path).with_context(|| format!("decode image {}", path.display()))?;
    Ok(img.to_rgb8())
}

/// JPEG decode via mozjpeg (libjpeg-turbo), PIL-compatible.
fn load_jpeg(path: &Path) -> Result<RgbImage> {
    let decompress = mozjpeg::Decompress::new_path(path)
        .with_context(|| format!("open jpeg {}", path.display()))?;
    let mut image = decompress
        .rgb()
        .with_context(|| format!("decode jpeg {}", path.display()))?;
    let width = image.width() as u32;
    let height = image.height() as u32;
    let pixels: Vec<u8> = image
        .read_scanlines()
        .with_context(|| format!("read jpeg scanlines {}", path.display()))?;
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

/// A media value is either a filesystem path or a `data:<mime>;base64,<payload>`
/// blob (written to a temp file so ffmpeg/hound can read it).
pub(crate) fn resolve_media(value: &str, ext: &str) -> Result<PathBuf, String> {
    let Some(rest) = value.strip_prefix("data:") else {
        return Ok(PathBuf::from(value));
    };
    let (mime, b64) = rest
        .split_once(";base64,")
        .ok_or("bad data URI (expected data:<mime>;base64,<payload>)")?;
    // The decoders pick their backend by file extension, so take it from the MIME
    // type (`image/jpeg` -> jpg, `audio/wav` -> wav) instead of a generic name.
    let ext = match mime.split('/').nth(1) {
        Some("jpeg") => "jpg",
        Some(other) if !other.is_empty() => other,
        _ => ext,
    };
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64)
        .map_err(|e| format!("base64: {e}"))?;
    let dir = std::env::temp_dir().join("rocket-inference");
    std::fs::create_dir_all(&dir).map_err(|e| format!("temp dir: {e}"))?;
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    let path = dir.join(format!("upload-{}-{nanos}.{ext}", std::process::id()));
    std::fs::write(&path, bytes).map_err(|e| format!("write {}: {e}", path.display()))?;
    Ok(path)
}
