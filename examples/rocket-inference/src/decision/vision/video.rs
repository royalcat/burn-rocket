//! d3 video decoding and frame sampling (Phase 3).
//!
//! The runtime reads a video with OpenCV, samples it at 2 frames per second and
//! feeds the frames to the checkpoint's `Qwen3VLVideoProcessor`, which groups them
//! in temporal pairs. This module is the sampling half: the source metadata
//! (`ffprobe`), the reference's index arithmetic, the even-pair duplication and the
//! per-pair timestamps the prompt prints.
//!
//! Reference: `d3_runtime.video_sample_indices` / `_decode_video` / `_video`:
//!
//! * `count = int(total / fps * 2)`, clamped to `[4, 32]` and `<= total`, then
//!   `np.linspace(0, total - 1, count).round()` — half-to-even on an exact `.5`,
//!   with the last index exactly `total - 1` (numpy's endpoint correction);
//! * an odd frame count duplicates the **last frame and its index** (the temporal
//!   pair, and the timestamp pair, then see two copies of it);
//! * the source fps is the container's rate, and an unknown or implausible one
//!   (`< 0.1` or `> 1000`, or not finite) is 24;
//! * the total comes from the container; a container that does not say gets its
//!   frames counted (the server stops counting at `300 * fps`);
//! * a truncated clip keeps only the frames that decode, **with their actual
//!   source indices** (`_decode_video` appends `kept`), so the pair count, grid and
//!   timestamps come from those frames; no decoded frame at all is
//!   [`NO_FRAME`], an unreadable container is [`COULD_NOT_OPEN`] — the reference's
//!   own texts, without the decoder's stderr (which can name a temporary upload).
//!
//! The frames themselves come from `ffmpeg` (see [`crate::gemma::media`]); a probe
//! over the repo's two fixtures showed `ffmpeg`'s `rgb24` output **bit-identical**
//! to OpenCV's `BGR -> RGB` on every sampled frame, so the decoder is not a parity
//! variable here.

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result, bail};
use image::RgbImage;

use crate::gemma::media::{self, VideoInfo};

/// Sampled frames per second (`d3_runtime.VIDEO_FPS`).
pub const FPS: f64 = 2.0;
/// At least this many frames (`VIDEO_MIN_FRAMES`).
pub const MIN_FRAMES: usize = 4;
/// At most this many frames (`VIDEO_MAX_FRAMES`).
pub const MAX_FRAMES: usize = 32;
/// The served source-fps window (`VIDEO_SOURCE_FPS`).
pub const SOURCE_FPS: (f64, f64) = (0.1, 1000.0);
/// The fps assumed when the container's is unknown (`VIDEO_UNKNOWN_FPS`).
pub const UNKNOWN_FPS: f64 = 24.0;
/// The served duration limit, seconds (`MAX_VIDEO_SECONDS`).
pub const MAX_SECONDS: f64 = 300.0;
/// The served per-frame pixel limit (`MAX_VIDEO_SOURCE_PIXELS`).
pub const MAX_SOURCE_PIXELS: u64 = 8_294_400;
/// The served decoded-byte limit (`MAX_VIDEO_BYTES`).
pub const MAX_BYTES: usize = 32_000_000;
/// The data-URL subtypes the server accepts (`VIDEO_FORMATS`).
pub const FORMATS: [&str; 4] = ["mp4", "webm", "quicktime", "x-matroska"];
/// `d3_runtime._decode_video`'s refusal of a clip that does not open.
pub const COULD_NOT_OPEN: &str = "the video could not be opened";
/// `d3_runtime._decode_video`'s refusal of a clip none of whose frames decode.
pub const NO_FRAME: &str = "no frame of the video could be decoded";

/// One decoded video: the sampled frames, in order, and their source metadata.
#[derive(Debug, Clone)]
pub struct DecodedVideo {
    /// RGB8 samples; always an even count (the last frame repeats when the
    /// sampled count is odd) — one temporal pair per two frames.
    pub frames: Vec<RgbImage>,
    /// The source frame number of every sample (the duplicate of the last frame
    /// repeats its index too).
    pub indices: Vec<usize>,
    /// The source frame rate, normalized to [`UNKNOWN_FPS`] when implausible.
    pub fps: f64,
    /// The source frame count the sampling ran over.
    pub total: usize,
    /// Frame size of the source.
    pub width: usize,
    pub height: usize,
}

impl DecodedVideo {
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// One timestamp per temporal pair: the average of the pair's source times
    /// (`Qwen3VLProcessor._calculate_timestamps` with `merge_size = 2`).
    pub fn timestamps(&self, temporal: usize) -> Vec<f64> {
        timestamps(&self.indices, self.fps, temporal)
    }
}

/// The source frames a video is sampled at: 2 per second, at least 4 and at
/// most 32, spread evenly over the whole clip (`d3_runtime.video_sample_indices`).
pub fn sample_indices(total: usize, fps: f64) -> Vec<usize> {
    let raw = (total as f64 / fps * FPS) as usize;
    let count = raw.clamp(MIN_FRAMES, MAX_FRAMES).min(total);
    linspace_rounded(total, count)
}

/// `np.linspace(0, total - 1, count).round()`: `count` points from 0 to
/// `total - 1` inclusive, every interior one rounded half-to-even.
fn linspace_rounded(total: usize, count: usize) -> Vec<usize> {
    if count <= 1 || total <= 1 {
        return vec![0; count.max(1)];
    }
    let last = (total - 1) as f64;
    let step = last / (count - 1) as f64;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let value = if i + 1 == count {
            last
        } else {
            i as f64 * step
        };
        out.push(round_half_even(value) as usize);
    }
    out
}

/// Python's `round` for a non-negative value: ties go to the even integer
/// (Rust's `f64::round` is half-away-from-zero, which would give 3 for 2.5).
fn round_half_even(value: f64) -> f64 {
    let floor = value.floor();
    let frac = value - floor;
    let up = frac > 0.5 || (frac == 0.5 && (floor as i64) & 1 == 1);
    if up { floor + 1.0 } else { floor }
}

/// The prompt timestamps of one video's temporal pairs: `average(idx / fps)`
/// per pair, with an odd index list padded by its last entry
/// (`Qwen3VLProcessor._calculate_timestamps`).
pub fn timestamps(indices: &[usize], fps: f64, temporal: usize) -> Vec<f64> {
    let temporal = temporal.max(1);
    let mut times: Vec<f64> = indices.iter().map(|index| *index as f64 / fps).collect();
    while !times.len().is_multiple_of(temporal) {
        times.push(*times.last().expect("a video has at least one frame"));
    }
    times
        .chunks(temporal)
        .map(|pair| (pair[0] + pair[pair.len() - 1]) / 2.0)
        .collect()
}

/// The source metadata of a video file: `(width, height, fps, total frames)`.
///
/// `nb_frames` is the container's answer, which is what OpenCV's
/// `CAP_PROP_FRAME_COUNT` reports; when the container does not say, the frames are
/// counted (`ffprobe -count_frames`). The server's 300-second guard bounds that
/// count.
fn probe(path: &Path) -> Result<(usize, usize, f64, usize), String> {
    let metadata = ffprobe(path, false)?;
    let Probe {
        width,
        height,
        fps,
        total,
    } = metadata;
    if width == 0 || height == 0 {
        return Err("the video has no frame size".to_string());
    }
    let fps = match fps {
        Some(fps) if fps.is_finite() && (SOURCE_FPS.0..=SOURCE_FPS.1).contains(&fps) => fps,
        _ => UNKNOWN_FPS,
    };
    let total = match total {
        Some(total) if total > 0 => total,
        _ => ffprobe(path, true)?.total.unwrap_or(0),
    };
    if total == 0 {
        return Err("the video has no frames".to_string());
    }
    Ok((width, height, fps, total))
}

/// The ffprobe fields this module reads.
struct Probe {
    width: usize,
    height: usize,
    fps: Option<f64>,
    total: Option<usize>,
}

/// One `ffprobe` call: the first video stream's size, rate and frame count.
///
/// `count` asks ffprobe to decode the stream and count its frames (the fallback
/// for a container without `nb_frames`).
fn ffprobe(path: &Path, count: bool) -> Result<Probe, String> {
    let mut command = Command::new("ffprobe");
    command.args(["-v", "error", "-select_streams", "v:0"]);
    if count {
        command.args(["-count_frames"]);
    }
    let out = command
        .args([
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames,nb_read_frames",
            "-of",
            "json",
        ])
        .arg(path)
        .output()
        .map_err(|error| {
            tracing::warn!(%error, "ffprobe could not be run");
            COULD_NOT_OPEN.to_string()
        })?;
    if !out.status.success() {
        // The reference (OpenCV) reports only that the clip did not open, and
        // ffprobe's stderr names the file it read — which for a `data:video/...`
        // upload is our temporary upload path. Log it, return the reference's text.
        tracing::warn!(
            stderr = %String::from_utf8_lossy(&out.stderr).trim(),
            "ffprobe rejected the video"
        );
        return Err(COULD_NOT_OPEN.to_string());
    }
    let json: serde_json::Value = serde_json::from_slice(&out.stdout).map_err(|error| {
        tracing::warn!(%error, "ffprobe output is not JSON");
        COULD_NOT_OPEN.to_string()
    })?;
    let stream = &json["streams"][0];
    let number = |value: &serde_json::Value| -> Option<usize> {
        value
            .as_u64()
            .map(|v| v as usize)
            .or_else(|| value.as_str().and_then(|text| text.parse().ok()))
    };
    let fps = stream["r_frame_rate"].as_str().and_then(|rate| {
        let (num, den) = rate.split_once('/')?;
        let (num, den): (f64, f64) = (num.parse().ok()?, den.parse().ok()?);
        (den != 0.0).then(|| num / den)
    });
    let total = if count {
        number(&stream["nb_read_frames"])
    } else {
        number(&stream["nb_frames"])
    };
    Ok(Probe {
        width: number(&stream["width"]).unwrap_or(0),
        height: number(&stream["height"]).unwrap_or(0),
        fps,
        total,
    })
}

/// Decode one video file and sample its frames (`d3_runtime.load_video` for a
/// local path).
///
/// `strict` is the server's contract: frames of at most [`MAX_SOURCE_PIXELS`]
/// pixels and a clip of at most [`MAX_SECONDS`]. The messages are the reference's.
pub fn decode(path: &Path, strict: bool) -> Result<DecodedVideo, String> {
    let (width, height, fps, total) = probe(path)?;
    if strict && (width as u64) * (height as u64) > MAX_SOURCE_PIXELS {
        return Err(format!(
            "video frames must have at most {} pixels",
            thousands(MAX_SOURCE_PIXELS)
        ));
    }
    if strict && total as f64 / fps > MAX_SECONDS {
        return Err(format!(
            "each video must be at most {} seconds long",
            MAX_SECONDS as u64
        ));
    }
    let info = VideoInfo {
        width,
        height,
        fps,
        total_frames: total,
        duration: total as f64 / fps,
    };
    let mut decoded = decode_sampled(path, &info, &sample_indices(total, fps));
    if decoded.is_empty() {
        return Err(NO_FRAME.to_string());
    }
    // `_video`: an odd decode repeats its last frame **and its index**, so every
    // frame belongs to a temporal pair and every pair has a timestamp. The count is
    // the number of frames the file actually yielded, which a truncated clip can
    // make smaller than the plan (`_decode_video` keeps the ones OpenCV retrieved).
    if !decoded.len().is_multiple_of(2) {
        decoded.push(decoded.last().expect("nonempty").clone());
    }
    let (indices, frames) = decoded.into_iter().unzip();
    Ok(DecodedVideo {
        frames,
        indices,
        fps,
        total,
        width,
        height,
    })
}

/// The sampled frames the file actually yields, with their source indices.
///
/// `d3_runtime._decode_video` keeps a frame only when OpenCV retrieves it, so a
/// truncated clip decodes fewer frames than were planned and the prompt is built
/// from the ones that came back — with their **actual** indices, not the planned
/// ones. One `ffmpeg` pass covers the whole list; when it comes back short the
/// decoder is probed per index, which is the only way to learn which planned frame
/// numbers survived (the bulk `select` output does not say).
fn decode_sampled(path: &Path, info: &VideoInfo, planned: &[usize]) -> Vec<(usize, RgbImage)> {
    match media::extract_frames(path, info, planned) {
        Ok(frames) if frames.len() == planned.len() => {
            return planned.iter().copied().zip(frames).collect();
        }
        Ok(frames) => tracing::warn!(
            decoded = frames.len(),
            planned = planned.len(),
            "the video yields fewer frames than were sampled; probing each index"
        ),
        Err(error) => tracing::warn!(
            error = %format!("{error:#}"),
            "ffmpeg frame extraction failed; probing each index"
        ),
    }
    let mut kept = Vec::new();
    for &index in planned {
        match extract_frame(path, info, index) {
            Some(frame) => kept.push((index, frame)),
            None => tracing::warn!(index, "a sampled frame did not decode"),
        }
    }
    kept
}

/// One source frame by native index, or `None` when it does not decode.
fn extract_frame(path: &Path, info: &VideoInfo, index: usize) -> Option<RgbImage> {
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args([
            "-vf",
            &format!("select='eq(n\\,{index})'"),
            "-fps_mode",
            "passthrough",
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ])
        .output()
        .ok()?;
    let frame_bytes = info.width * info.height * 3;
    if out.stdout.len() < frame_bytes {
        return None;
    }
    RgbImage::from_raw(
        info.width as u32,
        info.height as u32,
        out.stdout[..frame_bytes].to_vec(),
    )
}

/// Decode a video from bytes (`data:video/...` uploads): written to a temporary
/// file so it decodes exactly like the same clip given by path
/// (`_decode_video`'s `NamedTemporaryFile`). The file is removed afterwards.
pub fn decode_bytes(bytes: &[u8], extension: &str, strict: bool) -> Result<DecodedVideo, String> {
    let file = media::temporary_file(bytes, extension)?;
    decode(file.path(), strict)
}

/// `{value:,}` — the reference's digit grouping in its size messages.
pub(crate) fn thousands(value: u64) -> String {
    let digits = value.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            out.push(',');
        }
        out.push(digit);
    }
    out
}

/// The OpenAI `input_video`/data-URL extension a decoded payload is written with.
pub fn extension_for(subtype: &str) -> &'static str {
    match subtype {
        "quicktime" => "mp4",
        "x-matroska" => "mkv",
        "webm" => "webm",
        _ => "mp4",
    }
}

/// The `Qwen3VLProcessor.replace_video_token` string of one video: one group per
/// temporal pair, `<{t:.1} seconds><|vision_start|><|video_pad|>*frame_seqlen
/// <|vision_end|>`, concatenated with no separator.
///
/// `grid` is the `[T, H_patches, W_patches]` patch grid and `merge` the spatial
/// merge factor, so `frame_seqlen = H * W / merge^2` is one pair's merged tokens.
pub fn video_replacement(
    grid: [usize; 3],
    merge: usize,
    timestamps: &[f64],
    video_token: &str,
) -> Result<String> {
    let [frames, height, width] = grid;
    if merge == 0 || !height.is_multiple_of(merge) || !width.is_multiple_of(merge) {
        bail!("the {height}x{width} patch grid is not divisible by the merge size {merge}");
    }
    if frames != timestamps.len() {
        bail!(
            "the video has {frames} temporal pairs but {} timestamps",
            timestamps.len()
        );
    }
    let frame_seqlen = height * width / (merge * merge);
    let mut out = String::new();
    for stamp in timestamps {
        out.push_str(&format!("<{stamp:.1} seconds>"));
        out.push_str("<|vision_start|>");
        for _ in 0..frame_seqlen {
            out.push_str(video_token);
        }
        out.push_str("<|vision_end|>");
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `np.linspace(0, total - 1, count).round()` from the reference venv.
    #[test]
    fn the_frame_indices_are_the_reference_linspace() {
        let cases: &[(usize, f64, &[usize])] = &[
            // `assets/example-video.mp4`: 4 s at 8 fps -> 8 samples (the dump's own
            // `frames_indices`).
            (32, 8.0, &[0, 4, 9, 13, 18, 22, 27, 31]),
            (
                300,
                30.0,
                &[
                    0, 16, 31, 47, 63, 79, 94, 110, 126, 142, 157, 173, 189, 205, 220, 236, 252,
                    268, 283, 299,
                ],
            ),
            // 40 s at 8 fps -> 80 samples, capped at 32.
            (
                320,
                8.0,
                &[
                    0, 10, 21, 31, 41, 51, 62, 72, 82, 93, 103, 113, 123, 134, 144, 154, 165, 175,
                    185, 196, 206, 216, 226, 237, 247, 257, 268, 278, 288, 298, 309, 319,
                ],
            ),
            // A one-second clip: 2 samples, lifted to the 4-frame floor.
            (8, 8.0, &[0, 2, 5, 7]),
            (10, 8.0, &[0, 3, 6, 9]),
            // 2.5 s at 8 fps -> 5 samples (odd: `_video` duplicates the last).
            (20, 8.0, &[0, 5, 10, 14, 19]),
            (1, 8.0, &[0]),
            (2, 8.0, &[0, 1]),
            // A non-integral rate.
            (100, 30000.0 / 1001.0, &[0, 20, 40, 59, 79, 99]),
            // 129 exact steps.
            (
                4000,
                25.0,
                &[
                    0, 129, 258, 387, 516, 645, 774, 903, 1032, 1161, 1290, 1419, 1548, 1677, 1806,
                    1935, 2064, 2193, 2322, 2451, 2580, 2709, 2838, 2967, 3096, 3225, 3354, 3483,
                    3612, 3741, 3870, 3999,
                ],
            ),
            (24, 24.0, &[0, 8, 15, 23]),
            (48, 24.0, &[0, 16, 31, 47]),
            (
                720,
                120.0,
                &[0, 65, 131, 196, 261, 327, 392, 458, 523, 588, 654, 719],
            ),
            (7, 24.0, &[0, 2, 4, 6]),
            (
                3000,
                60.0,
                &[
                    0, 97, 193, 290, 387, 484, 580, 677, 774, 871, 967, 1064, 1161, 1258, 1354,
                    1451, 1548, 1645, 1741, 1838, 1935, 2032, 2128, 2225, 2322, 2419, 2515, 2612,
                    2709, 2806, 2902, 2999,
                ],
            ),
            // An exact .5 tie at an even floor rounds down (numpy's half-to-even).
            (11, 4.0, &[0, 2, 5, 8, 10]),
            (15, 4.0, &[0, 2, 5, 7, 9, 12, 14]),
            (19, 5.0, &[0, 3, 6, 9, 12, 15, 18]),
            (7, 4.0, &[0, 2, 4, 6]),
            (9, 5.0, &[0, 3, 5, 8]),
            (13, 6.0, &[0, 4, 8, 12]),
        ];
        for &(total, fps, expected) in cases {
            assert_eq!(
                sample_indices(total, fps).as_slice(),
                expected,
                "total {total}, fps {fps}"
            );
        }
    }

    #[test]
    fn the_rounding_is_half_to_even() {
        assert_eq!(round_half_even(2.5), 2.0);
        assert_eq!(round_half_even(3.5), 4.0);
        assert_eq!(round_half_even(2.4999), 2.0);
        assert_eq!(round_half_even(2.5001), 3.0);
        // Rust's own round would give 3 here.
        assert_eq!(2.5f64.round(), 3.0);
    }

    /// The reference's `_calculate_timestamps`: pair averages of `idx / fps`, the
    /// exact values the dumped prompts print.
    #[test]
    fn the_pair_timestamps_are_the_reference_averages() {
        let indices = [0, 4, 9, 13, 18, 22, 27, 31];
        assert_eq!(timestamps(&indices, 8.0, 2), vec![0.25, 1.375, 2.5, 3.625]);
        let printed: Vec<String> = timestamps(&indices, 8.0, 2)
            .iter()
            .map(|value| format!("{value:.1}"))
            .collect();
        assert_eq!(printed, ["0.2", "1.4", "2.5", "3.6"]);
        // An odd list is padded by its last entry before pairing: 0, .375, .75,
        // 1.125, 1.5 -> (0 + .375)/2, (.75 + 1.125)/2, (1.5 + 1.5)/2.
        assert_eq!(
            timestamps(&[0, 3, 6, 9, 12], 8.0, 2),
            vec![0.1875, 0.9375, 1.5]
        );
        // A single pair.
        assert_eq!(timestamps(&[0, 31], 8.0, 2), vec![1.9375]);
    }

    #[test]
    fn the_video_replacement_groups_per_pair() {
        let text = video_replacement([2, 4, 4], 2, &[0.2, 1.4], "<|video_pad|>").unwrap();
        let group =
            "<|vision_start|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|vision_end|>";
        assert_eq!(text, format!("<0.2 seconds>{group}<1.4 seconds>{group}"));
        // Shape refusals.
        assert!(video_replacement([2, 4, 6], 4, &[0.2, 1.4], "x").is_err());
        assert!(video_replacement([2, 4, 4], 2, &[0.2], "x").is_err());
        assert!(video_replacement([2, 4, 4], 0, &[0.2, 5.0], "x").is_err());
    }

    /// The fixtures' own metadata and sampled indices (the dumped
    /// `video_metadata.json`): both are 4 s at 8 fps, so 8 samples.
    #[test]
    fn the_fixture_videos_sample_eight_frames() {
        for (path, width, height) in [
            (
                std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/test.mp4"),
                480usize,
                360usize,
            ),
            (
                std::path::Path::new("/tmp/opencode/d3/models/d3-lite/assets/example-video.mp4")
                    .to_path_buf(),
                640,
                360,
            ),
        ] {
            if !path.is_file() {
                eprintln!("skipping: {} is not available", path.display());
                continue;
            }
            let video = decode(&path, true).expect("the fixture decodes");
            assert_eq!((video.width, video.height), (width, height));
            assert_eq!(video.fps, 8.0);
            assert_eq!(video.total, 32);
            assert_eq!(video.indices, vec![0, 4, 9, 13, 18, 22, 27, 31]);
            assert_eq!(video.frame_count(), 8);
            assert_eq!(video.frames[0].dimensions(), (width as u32, height as u32));
            assert_eq!(
                video
                    .timestamps(2)
                    .iter()
                    .map(|v| format!("{v:.1}"))
                    .collect::<Vec<_>>(),
                ["0.2", "1.4", "2.5", "3.6"]
            );
        }
    }

    #[test]
    fn size_messages_group_the_digits() {
        assert_eq!(thousands(8_294_400), "8,294,400");
        assert_eq!(thousands(32_000_000), "32,000,000");
        assert_eq!(thousands(999), "999");
        assert_eq!(thousands(1_000), "1,000");
    }

    #[test]
    fn a_missing_file_is_a_clean_error() {
        let error = decode(Path::new("/nonexistent/clip.mp4"), true).unwrap_err();
        assert_eq!(error, COULD_NOT_OPEN);
    }

    /// F2: a refused upload carries the reference's text, never the decoder's
    /// stderr — which names the temporary file a `data:video/...` payload was
    /// written to (`/tmp/rocket-inference/upload-<pid>-<nanos>.mp4`).
    #[test]
    fn a_refused_upload_does_not_leak_the_temporary_path() {
        let dir = std::env::temp_dir().join("d3-video-refusals");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("upload-4242-1234.mp4");
        std::fs::write(&path, b"this is not a video at all\n").expect("write the upload");
        let error = decode(&path, true).unwrap_err();
        assert_eq!(error, COULD_NOT_OPEN);
        for leak in ["/tmp/", "upload-", dir.to_str().unwrap()] {
            assert!(!error.contains(leak), "the detail leaks {leak:?}: {error}");
        }
        std::fs::remove_file(&path).ok();
    }

    /// F3: a truncated (but faststart) clip decodes the frames that survive,
    /// exactly like `_decode_video`: the reference keeps `[0, 4, 9, 13]` of the
    /// eight samples of this fixture and answers with those four frames.
    /// A container whose media data is gone decodes nothing and is [`NO_FRAME`].
    #[test]
    fn a_truncated_clip_keeps_the_frames_that_decode() {
        let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/test.mp4");
        if !source.is_file() {
            eprintln!("skipping: {} is not available", source.display());
            return;
        }
        let dir = std::env::temp_dir().join(format!("d3-video-truncated-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let fast = dir.join("faststart.mp4");
        let status = std::process::Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-i"])
            .arg(&source)
            .args(["-c", "copy", "-movflags", "+faststart"])
            .arg(&fast)
            .status();
        let Ok(status) = status else {
            eprintln!("skipping: ffmpeg is not available");
            return;
        };
        if !status.success() {
            eprintln!("skipping: ffmpeg could not remux the fixture");
            return;
        }
        let bytes = std::fs::read(&fast).expect("read the faststart copy");
        // Cut the media data at 55 % of the file; the header (ftyp + moov) is at the
        // front, so the stream still opens and only the tail frames are lost.
        let cut = dir.join("cut55.mp4");
        std::fs::write(&cut, &bytes[..bytes.len() * 55 / 100]).expect("write the cut");
        let video = decode(&cut, true).expect("a truncated clip still decodes its frames");
        assert_eq!((video.fps, video.total), (8.0, 32));
        assert!(
            video.frames.len() >= 2 && video.frames.len() < 8,
            "{:?} frames",
            video.frames.len()
        );
        assert_eq!(video.frames.len() % 2, 0, "pairs are padded, not dropped");
        assert_eq!(video.frames.len(), video.indices.len());
        let planned = sample_indices(video.total, video.fps);
        assert!(
            video.indices.iter().all(|index| planned.contains(index)),
            "indices {:?} are planned samples {:?}",
            video.indices,
            planned
        );
        assert!(
            video.indices.windows(2).all(|pair| pair[0] < pair[1]),
            "indices stay in source order: {:?}",
            video.indices
        );
        assert!(video.indices.len() < planned.len());
        assert_eq!(video.indices[0], 0, "the head frames decode");

        // The same header with no media data left: nothing decodes.
        let mut at = 0usize;
        let mdat = loop {
            assert!(at + 8 <= bytes.len(), "no mdat box in the fixture");
            let size = u32::from_be_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
            let kind = &bytes[at + 4..at + 8];
            if kind == b"mdat" {
                break at;
            }
            assert!(size >= 8, "bad box size at {at}");
            at += size;
        };
        let head = dir.join("header-only.mp4");
        std::fs::write(&head, &bytes[..mdat]).expect("write the header-only copy");
        let error = decode(&head, true).unwrap_err();
        assert_eq!(error, NO_FRAME);
        assert!(!error.contains("/tmp/"), "{error}");
        std::fs::remove_dir_all(&dir).ok();
    }
}
