//! Image preprocessing of Clef-Omni (the Qwen3-Omni image processor, as the release's
//! `encode_record` drives it): decode -> `smart_resize` -> antialiased bicubic resize on the
//! uint8 pixels -> `(x - 127.5) / 127.5` -> patchify in merge-block order.
//!
//! * `size = {"shortest_edge": 65536, "longest_edge": 1048576}` are PIXEL budgets here
//!   ([`IMAGE_MIN_PIXELS`] / [`IMAGE_MAX_PIXELS`]), factor 32 (patch 16 x merge 2).
//! * The resize is `torchvision.transforms.v2.functional.resize(uint8, BICUBIC,
//!   antialias=True)`: the ATen uint8 kernel port that EmbeddingGemma 2 / d3 validated
//!   bit for bit ([`crate::gemma::media::resize_bicubic_aa`]).
//! * The normalization `(v - 127.5) / 127.5` equals the processor's fused
//!   `rescale(1/255)` + `normalize(0.5, 0.5)` bit for bit (checked against the oracle's
//!   `media_pixel_values`; `(v / 255 - 0.5) / 0.5` is not).
//! * Patch rows are `[channel 3, temporal 2, py 16, px 16]`; patches are stored per temporal
//!   group, then per 2x2 merge block in raster order, then in-block row-major (so 4
//!   consecutive rows are one merged token). A still image is one frame duplicated into the
//!   two temporal slots; [`patchify_pairs`] takes real frame pairs (video, phase 8e).
//!
//! Sources are what the other servers accept: a path, a `data:<mime>;base64,...` URI, or an
//! `http(s)` URL (the existing blocking fetch with its 20 MiB / 10 s / 3-redirect limits and
//! the `ROCKET_MEDIA_FETCH=0` switch). Anything the client can fix is an [`Invalid`].

use super::VisionConfig;
use crate::clef::media::smart_resize;
use crate::clef::request::Invalid;
use crate::gemma::media::{load_image, resize_bicubic_aa, resolve_media_remote};
use anyhow::{Result, bail};
use image::RgbImage;

/// The image processor's `shortest_edge` (a pixel count).
pub const IMAGE_MIN_PIXELS: usize = 65_536;
/// The image processor's `longest_edge` (a pixel count).
pub const IMAGE_MAX_PIXELS: usize = 1_048_576;
/// A decoded source above this many pixels is refused before any resize (the release has no
/// cap; this bounds the memory of a hostile upload: 16 MP = 48 MB of RGB8).
pub const MAX_SOURCE_PIXELS: usize = 16_000_000;

/// One image ready for the tower.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedImage {
    /// `[patches, 1536]` row-major, values in `[-1, 1]`, merge-block order.
    pub pixels: Vec<f32>,
    /// `[t = 1, h, w]` patch grid (`h * 16` = resized height).
    pub grid: [usize; 3],
    /// `(height, width)` of the decoded source.
    pub source: (usize, usize),
    /// `(height, width)` after `smart_resize`.
    pub resized: (usize, usize),
}

impl PreparedImage {
    pub fn patches(&self) -> usize {
        self.grid.iter().product()
    }

    /// Merged tokens the image contributes to the prompt (`t*h*w / 4`).
    pub fn tokens(&self) -> usize {
        self.patches() / 4
    }
}

/// `(v - 127.5) / 127.5`, the processor's fused rescale + normalize.
#[inline]
pub fn normalize_channel(v: u8) -> f32 {
    (f32::from(v) - 127.5) / 127.5
}

/// Patchify temporal frame pairs (each `(frame0, frame1)` is one temporal group of the same
/// `h x w` size) into `[groups * (h/16) * (w/16), 1536]` rows, plus the `[groups, h/16, w/16]`
/// grid. A still image is `[(img, img)]`.
pub fn patchify_pairs(
    pairs: &[(&RgbImage, &RgbImage)],
    cfg: &VisionConfig,
) -> Result<(Vec<f32>, [usize; 3])> {
    let (p, m) = (cfg.patch, cfg.merge);
    let Some((first, _)) = pairs.first() else {
        bail!("no frames to patchify");
    };
    let (w, h) = (first.width() as usize, first.height() as usize);
    if h == 0 || w == 0 || h % (p * m) != 0 || w % (p * m) != 0 {
        bail!(
            "frame {w}x{h} is not a multiple of the patch factor {}",
            p * m
        );
    }
    let (gh, gw) = (h / p, w / p);
    let dim = cfg.patch_dim();
    let mut out = Vec::with_capacity(pairs.len() * gh * gw * dim);
    for (a, b) in pairs {
        if a.dimensions() != first.dimensions() || b.dimensions() != first.dimensions() {
            bail!("the frames of a video must share one size");
        }
        let frames = [a.as_raw(), b.as_raw()];
        for br in 0..gh / m {
            for bc in 0..gw / m {
                for ir in 0..m {
                    for ic in 0..m {
                        let (row, col) = (br * m + ir, bc * m + ic);
                        for c in 0..cfg.in_channels {
                            for raw in frames.iter().take(cfg.temporal) {
                                for i in 0..p {
                                    let base = ((row * p + i) * w + col * p) * 3 + c;
                                    out.extend(
                                        (0..p).map(|j| normalize_channel(raw[base + j * 3])),
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok((out, [pairs.len(), gh, gw]))
}

/// Preprocess one decoded RGB8 image (a still: the frame fills both temporal slots).
pub fn prepare_image(img: &RgbImage, cfg: &VisionConfig) -> Result<PreparedImage> {
    let source = (img.height() as usize, img.width() as usize);
    if source.0 * source.1 > MAX_SOURCE_PIXELS {
        return Err(Invalid(format!(
            "the image is {}x{} pixels, over the {MAX_SOURCE_PIXELS}-pixel limit",
            source.1, source.0
        ))
        .into());
    }
    let (hb, wb) = smart_resize(
        source.0,
        source.1,
        cfg.factor(),
        IMAGE_MIN_PIXELS,
        IMAGE_MAX_PIXELS,
    )
    .map_err(|e| Invalid(format!("{e}")))?;
    let resized = if (hb, wb) == source {
        std::borrow::Cow::Borrowed(img)
    } else {
        std::borrow::Cow::Owned(resize_bicubic_aa(img, wb, hb))
    };
    let (pixels, grid) = patchify_pairs(&[(&resized, &resized)], cfg)?;
    Ok(PreparedImage {
        pixels,
        grid,
        source,
        resized: (hb, wb),
    })
}

/// Fetch (path / data URI / `http(s)`) and decode one image value of a request
/// (`images[index]`); PIL `convert("RGB")` semantics, JPEGs through libjpeg(-turbo).
pub fn load_rgb(value: &str, index: usize) -> Result<RgbImage> {
    let bad = |what: &dyn std::fmt::Display| Invalid(format!("images[{index}]: {what}"));
    let resolved = resolve_media_remote(value, "img").map_err(|e| bad(&e))?;
    load_image(resolved.path()).map_err(|e| {
        if value.starts_with("data:") || value.starts_with("http") {
            // the decoder saw a temp file: do not leak its name, give the cause
            bad(&format_args!(
                "could not decode the image ({})",
                e.root_cause()
            ))
            .into()
        } else {
            bad(&format_args!("{e:#}")).into()
        }
    })
}

/// [`load_rgb`] + [`prepare_image`].
pub fn prepare_value(value: &str, index: usize, cfg: &VisionConfig) -> Result<PreparedImage> {
    let img = load_rgb(value, index)?;
    prepare_image(&img, cfg).map_err(|e| match e.downcast::<Invalid>() {
        Ok(Invalid(m)) => Invalid(format!("images[{index}]: {m}")).into(),
        Err(e) => e,
    })
}
