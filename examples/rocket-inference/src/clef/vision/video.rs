//! Video preprocessing of Clef-Omni (phase 8d): what the release's `load_video` plus the
//! Qwen3-Omni video processor do, with ffmpeg/ffprobe instead of PyAV.
//!
//! 1. **Sampling is a time rule** (`load_video`): decoded frames in presentation order; a frame is
//!    skipped when `frame.time is None or frame.time + 1e-6 < len(kept) / 2.0`, i.e. the first
//!    frame at or after every 0.5 s multiple is kept (`VIDEO_FPS = 2.0`). `frame.time` is
//!    `pts * time_base` (ffprobe's `pts` of the first video stream and its `time_base`; `pts`
//!    unset = never kept). An odd count duplicates the last frame. There is no cap in the
//!    release; this port adds one ([`max_video_frames`]) and refuses above it with a 422.
//! 2. **Per-frame cap**: with `scale = min(1, sqrt(262144 / (w*h)))` the frame is converted to
//!    `width = max(2, round(w*scale/2)*2)` x `height = ...` (Python `round`, ties to even) by
//!    swscale (`frame.to_ndarray(format="rgb24", width, height)`, PyAV's default bilinear). Here
//!    one ffmpeg run does it: `select` the kept indices, `scale=W:H:flags=bilinear,format=rgb24`.
//!    `-noautorotate` because PyAV ignores the rotation side data.
//! 3. **Processor**: `smart_resize` with the VIDEO table (131,072..786,432 pixels, factor 32;
//!    NOT the image table), the ATen antialiased bicubic uint8 resize per frame, `(v - 127.5) /
//!    127.5`, frames paired into temporal groups of 2 (`patchify_pairs`), rows in merge-block
//!    order per group. The grid is `[frames/2, h/16, w/16]`; the tower attends per temporal group.
//! 4. **Soundtrack**: the release uses the audio tracks of the videos only when EVERY video has
//!    one (`use_audio_in_video`, all-or-nothing); a record where any video is silent ignores all
//!    tracks. Since phase 8e the tracks are decoded ([`ProbedVideo::soundtrack`], 16 kHz mono)
//!    and go through the audio tower, interleaved with the video tokens (`encode::encode_request`).
//!
//! Decoder-version sensitivity: swscale's YUV->RGB conversion is what PyAV runs; the ffmpeg CLI
//! of the same libswscale generation is bit-identical on the repo fixtures (checked by the
//! `decode_and_preprocessing_match_the_oracle_for_videos` gate), but a different ffmpeg major
//! can differ by 1-2/255 per channel (see the d3 notes in AGENTS.md); not measured here.

use super::VisionConfig;
use super::preprocess::{MAX_SOURCE_PIXELS, patchify_pairs};
use crate::clef::media::smart_resize;
use crate::clef::request::Invalid;
use crate::gemma::media::{resize_bicubic_aa, resolve_media_remote};
use anyhow::{Context, Result, bail};
use image::RgbImage;
use rayon::prelude::*;
use std::path::Path;
use std::process::Command;

/// `VIDEO_FPS` of the release.
pub const VIDEO_FPS: f64 = 2.0;
/// `MAX_VIDEO_FRAME_PIXELS = 256*32*32`.
pub const VIDEO_FRAME_MAX_PIXELS: usize = 262_144;
/// The video processor's class defaults (`128*32*32`, `768*32*32`), applied because the
/// release passes no `size` for videos.
pub const VIDEO_MIN_PIXELS: usize = 131_072;
pub const VIDEO_MAX_PIXELS: usize = 786_432;
/// Sampled frames a request may carry (this port's guard; the release has none). 64 frames =
/// 32 temporal groups, 4,480 merged tokens at the smallest per-frame size.
pub const DEFAULT_MAX_VIDEO_FRAMES: usize = 64;

/// The `scale` filter options that make the ffmpeg CLI reproduce PyAV's
/// `frame.to_ndarray(format="rgb24", width, height)`: PyAV's default interpolation is bilinear
/// and it calls swscale with its default chroma siting, which for 4:2:0 is horizontal 0 /
/// vertical 128. The CLI's scale filter otherwise derives the siting from the stream's
/// chroma-location metadata (and differs by up to 66/255 on a 1280x720 -> 682x384 clip;
/// measured with PyAV 19.0.2 / ffmpeg n9.0.2, `scaled_frames_match_pyav_*`).
const SCALE_OPTIONS: &str = "flags=bilinear:in_h_chr_pos=0:in_v_chr_pos=128";

/// `ROCKET_CLEF_VIDEO_MAX_FRAMES` (default [`DEFAULT_MAX_VIDEO_FRAMES`]).
pub fn max_video_frames() -> usize {
    std::env::var("ROCKET_CLEF_VIDEO_MAX_FRAMES")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n: &usize| n >= 2)
        .unwrap_or(DEFAULT_MAX_VIDEO_FRAMES)
}

/// What ffprobe says about a video file.
#[derive(Debug, Clone, PartialEq)]
pub struct VideoProbe {
    pub width: usize,
    pub height: usize,
    /// `pts` of every decoded frame of the first video stream, presentation order (`None` =
    /// unset, PyAV's `frame.time is None`).
    pub frame_pts: Vec<Option<i64>>,
    /// Stream `time_base` as `(numerator, denominator)`.
    pub time_base: (i64, i64),
    /// The file has an audio stream.
    pub has_audio_stream: bool,
}

impl VideoProbe {
    /// `frame.time` of every frame (seconds): `pts * num / den` as PyAV computes it.
    pub fn frame_times(&self) -> Vec<Option<f64>> {
        self.frame_pts
            .iter()
            .map(|p| p.map(|p| p as f64 * self.time_base.0 as f64 / self.time_base.1 as f64))
            .collect()
    }
}

/// The 2 fps time rule of `load_video`: indices of the frames to keep.
pub fn sample_frames(times: &[Option<f64>]) -> Vec<usize> {
    let mut kept = Vec::new();
    for (i, t) in times.iter().enumerate() {
        let Some(t) = t else { continue };
        if t + 1e-6 < kept.len() as f64 / VIDEO_FPS {
            continue;
        }
        kept.push(i);
    }
    kept
}

/// The per-frame cap: `(width, height)` the release converts a `w x h` frame to.
pub fn capped_size(w: usize, h: usize) -> (usize, usize) {
    let scale = (VIDEO_FRAME_MAX_PIXELS as f64 / (w * h) as f64)
        .sqrt()
        .min(1.0);
    let f = |v: usize| (((v as f64 * scale / 2.0).round_ties_even() as usize) * 2).max(2);
    (f(w), f(h))
}

fn run_ffprobe(path: &Path, args: &[&str]) -> Result<serde_json::Value> {
    let out = Command::new("ffprobe")
        .args(["-v", "error", "-of", "json"])
        .args(args)
        .arg(path)
        .output()
        .context("run ffprobe (is it installed?)")?;
    if !out.status.success() {
        return Err(Invalid(format!(
            "could not read the video ({})",
            String::from_utf8_lossy(&out.stderr).trim()
        ))
        .into());
    }
    serde_json::from_slice(&out.stdout).context("parse ffprobe output")
}

/// Probe `path`: stream geometry, audio presence and the frame pts list. A file ffprobe cannot
/// read, or without a video stream, is [`Invalid`].
pub fn probe_video(path: &Path) -> Result<VideoProbe> {
    let streams = run_ffprobe(
        path,
        &[
            "-show_entries",
            "stream=index,codec_type,width,height,time_base",
        ],
    )?;
    let list = streams["streams"].as_array().cloned().unwrap_or_default();
    let video = list
        .iter()
        .find(|s| s["codec_type"] == "video")
        .ok_or_else(|| Invalid("the file has no video stream".into()))?;
    let (width, height) = (
        video["width"].as_u64().unwrap_or(0) as usize,
        video["height"].as_u64().unwrap_or(0) as usize,
    );
    if width == 0 || height == 0 {
        return Err(Invalid("the video has no frame size".into()).into());
    }
    let (num, den) = video["time_base"]
        .as_str()
        .and_then(|s| s.split_once('/'))
        .and_then(|(a, b)| Some((a.parse::<i64>().ok()?, b.parse::<i64>().ok()?)))
        .filter(|&(a, b)| a > 0 && b > 0)
        .ok_or_else(|| Invalid("the video stream has no time base".into()))?;
    let frames = run_ffprobe(
        path,
        &[
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=pts",
            "-show_frames",
        ],
    )?;
    let frame_pts = frames["frames"]
        .as_array()
        .map(|f| f.iter().map(|f| f["pts"].as_i64()).collect())
        .unwrap_or_default();
    Ok(VideoProbe {
        width,
        height,
        frame_pts,
        time_base: (num, den),
        has_audio_stream: list.iter().any(|s| s["codec_type"] == "audio"),
    })
}

/// Whether the first audio stream decodes to at least one sample (the release's `_decode_audio`
/// returns `None` for 0 samples, which makes the video "silent").
pub fn audio_has_samples(path: &Path) -> Result<bool> {
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args([
            "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-f", "f32le", "-",
        ])
        .output()
        .context("run ffmpeg for the soundtrack probe")?;
    if !out.status.success() {
        return Err(Invalid(format!(
            "could not decode the soundtrack of the video ({})",
            String::from_utf8_lossy(&out.stderr).trim()
        ))
        .into());
    }
    Ok(out.stdout.len() >= 4)
}

/// Decode the frames `indices` (ascending, unique, as numbered in `probe.frame_pts`) as RGB8
/// at the capped size.
pub fn decode_frames(path: &Path, probe: &VideoProbe, indices: &[usize]) -> Result<Vec<RgbImage>> {
    let (w, h) = capped_size(probe.width, probe.height);
    let sel = indices
        .iter()
        .map(|i| format!("eq(n\\,{i})"))
        .collect::<Vec<_>>()
        .join("+");
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-noautorotate", "-i"])
        .arg(path)
        .args([
            "-map",
            "0:v:0",
            "-vf",
            &format!("select='{sel}',scale={w}:{h}:{SCALE_OPTIONS},format=rgb24"),
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
        return Err(Invalid(format!(
            "could not decode the video ({})",
            String::from_utf8_lossy(&out.stderr).trim()
        ))
        .into());
    }
    let frame_bytes = w * h * 3;
    if out.stdout.len() != indices.len() * frame_bytes {
        bail!(
            "ffmpeg returned {} bytes for {} frames of {w}x{h} (expected {})",
            out.stdout.len(),
            indices.len(),
            indices.len() * frame_bytes
        );
    }
    out.stdout
        .chunks_exact(frame_bytes)
        .map(|c| RgbImage::from_raw(w as u32, h as u32, c.to_vec()).context("build a frame image"))
        .collect()
}

/// A video ready for the tower.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedVideo {
    /// `[patches, 1536]` row-major, values in `[-1, 1]`, per temporal group in merge-block order.
    pub pixels: Vec<f32>,
    /// `[frames / 2, h, w]` patch grid.
    pub grid: [usize; 3],
    /// `(height, width)` of the decoded source stream.
    pub source: (usize, usize),
    /// `(height, width)` after the per-frame cap.
    pub capped: (usize, usize),
    /// `(height, width)` after `smart_resize`.
    pub resized: (usize, usize),
    /// Frames sampled at 2 fps (before the even-count duplication).
    pub sampled: usize,
}

impl PreparedVideo {
    pub fn patches(&self) -> usize {
        self.grid.iter().product()
    }

    /// Merged tokens the video contributes to the prompt (`t*h*w / 4`).
    pub fn tokens(&self) -> usize {
        self.patches() / 4
    }
}

/// The video processor on decoded frames (already capped): smart-resize, resize every frame,
/// pair them and patchify. `max_tokens` bounds the merged tokens this video may produce (the
/// grid is known before any pixel work, so a video over it is refused cheaply).
pub fn prepare_video(
    frames: &[RgbImage],
    sampled: usize,
    source: (usize, usize),
    cfg: &VisionConfig,
    max_tokens: usize,
) -> Result<PreparedVideo> {
    let Some(first) = frames.first() else {
        return Err(Invalid("the video has no frames".into()).into());
    };
    let capped = (first.height() as usize, first.width() as usize);
    let (hb, wb) = smart_resize(
        capped.0,
        capped.1,
        cfg.factor(),
        VIDEO_MIN_PIXELS,
        VIDEO_MAX_PIXELS,
    )
    .map_err(|e| Invalid(format!("{e}")))?;
    let groups = frames.len().div_ceil(cfg.temporal);
    let tokens = groups * (hb / cfg.patch) * (wb / cfg.patch) / cfg.merge_unit();
    if tokens > max_tokens {
        return Err(Invalid(format!(
            "the video needs {tokens} tokens ({sampled} sampled frames at {wb}x{hb}), over the \
             {max_tokens} still free under the record's token cap"
        ))
        .into());
    }
    let resized: Vec<std::borrow::Cow<'_, RgbImage>> = frames
        .par_iter()
        .map(|f| {
            if (hb, wb) == capped {
                std::borrow::Cow::Borrowed(f)
            } else {
                std::borrow::Cow::Owned(resize_bicubic_aa(f, wb, hb))
            }
        })
        .collect();
    // temporal groups of 2; an odd tail pairs the last frame with itself (the release already
    // evened the count, so this only matters to direct callers)
    let pairs: Vec<(&RgbImage, &RgbImage)> = resized
        .chunks(cfg.temporal)
        .map(|c| (&*c[0], &*c[c.len() - 1]))
        .collect();
    let (pixels, grid) = patchify_pairs(&pairs, cfg)?;
    Ok(PreparedVideo {
        pixels,
        grid,
        source,
        capped,
        resized: (hb, wb),
        sampled,
    })
}

/// A video value resolved, probed and sampled but not decoded yet.
pub struct ProbedVideo {
    resolved: crate::gemma::media::ResolvedMedia,
    pub probe: VideoProbe,
    /// Frame indices the time rule keeps (`probe.frame_pts` numbering).
    pub keep: Vec<usize>,
    index: usize,
}

impl ProbedVideo {
    /// Fetch (path / data URI / `http(s)`), probe and sample `videos[index]`.
    pub fn open(value: &str, index: usize) -> Result<Self> {
        let bad = |what: &dyn std::fmt::Display| Invalid(format!("videos[{index}]: {what}"));
        let resolved = resolve_media_remote(value, "mp4").map_err(|e| bad(&e))?;
        let probe = probe_video(resolved.path()).map_err(|e| match e.downcast::<Invalid>() {
            Ok(Invalid(m)) => anyhow::Error::from(bad(&m)),
            Err(e) => e,
        })?;
        if probe.width * probe.height > MAX_SOURCE_PIXELS {
            return Err(bad(&format_args!(
                "the video is {}x{} pixels, over the {MAX_SOURCE_PIXELS}-pixel limit",
                probe.width, probe.height
            ))
            .into());
        }
        let keep = sample_frames(&probe.frame_times());
        if keep.is_empty() {
            return Err(bad(&"no decodable frames").into());
        }
        let cap = max_video_frames();
        if keep.len() > cap {
            return Err(bad(&format_args!(
                "{} frames sampled at {VIDEO_FPS} fps, over the limit of {cap} \
                 (ROCKET_CLEF_VIDEO_MAX_FRAMES); send a shorter clip",
                keep.len()
            ))
            .into());
        }
        Ok(Self {
            resolved,
            probe,
            keep,
            index,
        })
    }

    /// The video's soundtrack (16 kHz mono f32 of the first audio stream) or `None` when it has
    /// no audio stream / no samples: the release's `_decode_audio is None`, which makes the
    /// video "silent" for the whole record (`use_audio_in_video = all(track is not None)`).
    pub fn soundtrack(&self) -> Result<Option<Vec<f32>>> {
        if !self.probe.has_audio_stream {
            return Ok(None);
        }
        crate::clef::audio::soundtrack(self.resolved.path(), self.index)
    }

    /// Decode the sampled frames (at the capped size) and duplicate the last one for an odd
    /// count: exactly the release's `load_video` frames (`media_video_frames_*` of the oracle).
    /// Returns them with the sampled count before the duplication.
    pub fn decode_even(&self) -> Result<(Vec<RgbImage>, usize)> {
        let mut frames =
            decode_frames(self.resolved.path(), &self.probe, &self.keep).map_err(|e| match e
                .downcast::<Invalid>()
            {
                Ok(Invalid(m)) => Invalid(format!("videos[{}]: {m}", self.index)).into(),
                Err(e) => e,
            })?;
        let sampled = frames.len();
        if sampled % 2 == 1 {
            frames.push(frames[sampled - 1].clone());
        }
        Ok((frames, sampled))
    }

    /// Decode and run the video processor. `max_tokens` bounds this video's merged tokens.
    pub fn prepare(&self, cfg: &VisionConfig, max_tokens: usize) -> Result<PreparedVideo> {
        let (frames, sampled) = self.decode_even()?;
        let source = (self.probe.height, self.probe.width);
        prepare_video(&frames, sampled, source, cfg, max_tokens).map_err(|e| {
            match e.downcast::<Invalid>() {
                Ok(Invalid(m)) => Invalid(format!("videos[{}]: {m}", self.index)).into(),
                Err(e) => e,
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_time_rule_keeps_the_first_frame_at_or_after_each_half_second() {
        // 8 fps, 4 s: frames every 0.125 s -> indices 0, 4, 8, ... (0.5 s apart)
        let t: Vec<Option<f64>> = (0..32).map(|i| Some(i as f64 / 8.0)).collect();
        assert_eq!(sample_frames(&t), [0, 4, 8, 12, 16, 20, 24, 28]);
        // 30 fps: 0.5 s = frame 15
        let t: Vec<Option<f64>> = (0..60).map(|i| Some(i as f64 / 30.0)).collect();
        assert_eq!(sample_frames(&t), [0, 15, 30, 45]);
        // 25 fps: 0.5 s = frame 12.5 -> 13 (the first frame AFTER the boundary), then 25, 38
        let t: Vec<Option<f64>> = (0..75).map(|i| Some(i as f64 / 25.0)).collect();
        assert_eq!(sample_frames(&t), [0, 13, 25, 38, 50, 63]);
        // a frame whose time is within the 1e-6 tolerance below the boundary counts
        let t = [Some(0.0), Some(0.499_999_5), Some(0.6)];
        assert_eq!(sample_frames(&t), [0, 1]);
        // frames without a time are never kept; the rule counts only kept frames
        let t = [Some(0.0), None, Some(0.2), None, Some(0.55), Some(1.0)];
        assert_eq!(sample_frames(&t), [0, 4, 5]);
        // a clip shorter than 0.5 s keeps its first frame only
        assert_eq!(sample_frames(&[Some(0.0), Some(0.1)]), [0]);
        assert!(sample_frames(&[None, None]).is_empty());
    }

    #[test]
    fn frame_times_are_pts_times_the_time_base() {
        let p = VideoProbe {
            width: 2,
            height: 2,
            frame_pts: vec![Some(0), None, Some(15360)],
            time_base: (1, 15360),
            has_audio_stream: false,
        };
        assert_eq!(p.frame_times(), [Some(0.0), None, Some(1.0)]);
    }

    #[test]
    fn the_frame_cap_matches_the_release_rounding() {
        // at or below 262,144 pixels nothing changes (even sizes)
        assert_eq!(capped_size(192, 144), (192, 144));
        assert_eq!(capped_size(480, 360), (480, 360));
        assert_eq!(capped_size(512, 512), (512, 512));
        // above it both sides scale by sqrt(cap / px) and round to even
        assert_eq!(capped_size(1280, 720), (682, 384));
        assert_eq!(capped_size(1920, 1080), (682, 384));
        // odd sizes round to even even without scaling (Python round: ties to even)
        assert_eq!(capped_size(481, 361), (480, 360));
        assert_eq!(capped_size(483, 361), (484, 360));
        // never below 2
        assert_eq!(capped_size(1, 1), (2, 2));
    }

    #[test]
    fn the_video_tables_are_not_the_image_tables() {
        // small clips are UPSCALED to >= 131,072 pixels
        assert_eq!(
            smart_resize(144, 192, 32, VIDEO_MIN_PIXELS, VIDEO_MAX_PIXELS).unwrap(),
            (320, 448)
        );
        assert_eq!(
            smart_resize(180, 240, 32, VIDEO_MIN_PIXELS, VIDEO_MAX_PIXELS).unwrap(),
            (320, 448)
        );
        assert_eq!(
            smart_resize(360, 480, 32, VIDEO_MIN_PIXELS, VIDEO_MAX_PIXELS).unwrap(),
            (352, 480)
        );
    }

    #[test]
    fn an_odd_frame_count_is_paired_with_a_duplicate_and_the_budget_is_checked_first() {
        let cfg = VisionConfig::shipped();
        let frame = |v: u8| RgbImage::from_pixel(320, 448, image::Rgb([v, v, v]));
        // 2 frames of 320x448 (already the smart-resize target): one group, 20x28 patches
        let v = prepare_video(&[frame(10), frame(200)], 2, (448, 320), &cfg, 10_000).unwrap();
        assert_eq!(v.grid, [1, 28, 20]);
        assert_eq!(v.tokens(), 140);
        // the first patch row interleaves the two frames per channel: [c, t, py, px]
        let (lo, hi) = (
            super::super::preprocess::normalize_channel(10),
            super::super::preprocess::normalize_channel(200),
        );
        assert_eq!(v.pixels[0], lo);
        assert_eq!(v.pixels[256], hi);
        assert_eq!(v.pixels[512], lo);
        // over the token budget: a client error before any pixel work
        let err = prepare_video(&[frame(1), frame(2)], 2, (448, 320), &cfg, 100).unwrap_err();
        assert!(err.downcast_ref::<Invalid>().is_some(), "{err:#}");
        // no frames
        assert!(prepare_video(&[], 0, (0, 0), &cfg, 100).is_err());
    }
}
