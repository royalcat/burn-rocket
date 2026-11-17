//! Audio decoding of Clef-Omni (phase 8e): the release's `_decode_audio` (PyAV: the FIRST audio
//! stream, `AudioResampler(format="flt", layout="mono", rate=16000)`, frames concatenated) done
//! with the ffmpeg CLI: `-map 0:a:0 -ac 1 -ar 16000 -f f32le`. The oracle notes (docs/experiments/clef-oracle.md
//! §7.8) measured the CLI bit-identical to PyAV on `test.mp4` (AAC 44.1 kHz -> 16 kHz); the
//! gates in `audio::tests` repeat that against the dumped samples of every fixture.
//!
//! Limits (this port's, the release has none): a clip longer than
//! [`max_audio_seconds`] (`ROCKET_CLEF_AUDIO_MAX_SECONDS`, default 300) is a client error.
//! Decoding stops one second past the cap so a huge file costs bounded memory.

use super::mel::SAMPLE_RATE;
use crate::clef::request::Invalid;
use crate::gemma::media::resolve_media_remote;
use anyhow::Result;
use std::path::Path;
use std::process::Command;

/// Default for `ROCKET_CLEF_AUDIO_MAX_SECONDS`.
pub const DEFAULT_MAX_AUDIO_SECONDS: usize = 300;

/// `ROCKET_CLEF_AUDIO_MAX_SECONDS` (default 300 s): the longest clip a request may carry.
pub fn max_audio_seconds() -> usize {
    std::env::var("ROCKET_CLEF_AUDIO_MAX_SECONDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n: &usize| n >= 1)
        .unwrap_or(DEFAULT_MAX_AUDIO_SECONDS)
}

/// Why a first-audio-stream decode produced no samples.
#[derive(Debug)]
pub enum NoAudio {
    /// The file has no audio stream.
    NoStream,
    /// ffmpeg could not decode it (the reason is ffmpeg's stderr).
    Failed(String),
}

/// Decode the first audio stream of `path` to 16 kHz mono f32, at most `max_samples + 1 s`
/// worth of it. `Ok(samples)` may be empty (a stream without samples).
pub fn decode_first_stream(
    path: &Path,
    max_samples: usize,
) -> std::result::Result<Vec<f32>, NoAudio> {
    let limit = format!(
        "{:.3}",
        (max_samples + SAMPLE_RATE) as f64 / SAMPLE_RATE as f64
    );
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args([
            "-vn", "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-t", &limit, "-f", "f32le", "-",
        ])
        .output()
        .map_err(|e| NoAudio::Failed(format!("run ffmpeg (is it installed?): {e}")))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr).trim().to_string();
        return Err(if err.contains("matches no streams") {
            NoAudio::NoStream
        } else {
            NoAudio::Failed(err)
        });
    }
    Ok(out
        .stdout
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

/// Refuse a clip over the length cap.
pub fn check_length(samples: usize, what: &str) -> Result<()> {
    check_length_with(samples, max_audio_seconds(), what)
}

/// [`check_length`] against an explicit cap in seconds.
pub fn check_length_with(samples: usize, cap: usize, what: &str) -> Result<()> {
    if samples > cap * SAMPLE_RATE {
        return Err(Invalid(format!(
            "{what}: the audio is longer than {cap} s (ROCKET_CLEF_AUDIO_MAX_SECONDS); send a \
             shorter clip"
        ))
        .into());
    }
    Ok(())
}

/// The shortest clip this port accepts: one STFT window (25 ms). torch's reflect padding needs
/// more than `n_fft / 2` samples in the batch's longest clip and the model needs one mel frame.
pub const MIN_SAMPLES: usize = super::mel::N_FFT;

/// Fetch (path / `data:` URL / `http(s)`) and decode `audio[index]`.
pub fn load_audio(value: &str, index: usize) -> Result<Vec<f32>> {
    let what = format!("audio[{index}]");
    let bad = |m: &dyn std::fmt::Display| Invalid(format!("{what}: {m}"));
    let resolved = resolve_media_remote(value, "wav").map_err(|e| bad(&e))?;
    let samples = decode_first_stream(resolved.path(), max_audio_seconds() * SAMPLE_RATE).map_err(
        |e| match e {
            NoAudio::NoStream => bad(&"the file has no audio stream"),
            NoAudio::Failed(m) => bad(&format_args!("could not decode the audio ({m})")),
        },
    )?;
    if samples.is_empty() {
        return Err(bad(&"the audio has no samples").into());
    }
    check_length(samples.len(), &what)?;
    Ok(samples)
}

/// A clip that decodes to fewer samples than [`MIN_SAMPLES`] is a client error.
pub fn check_clip_len(samples: usize, what: &str) -> Result<()> {
    if samples < MIN_SAMPLES {
        return Err(Invalid(format!(
            "{what}: the audio is too short ({samples} samples; the minimum is {MIN_SAMPLES})"
        ))
        .into());
    }
    Ok(())
}

/// The soundtrack of a video file: `None` when it has no audio stream or no samples (the
/// release's `_decode_audio is None`: that video is then "silent" for the whole record).
pub fn soundtrack(path: &Path, index: usize) -> Result<Option<Vec<f32>>> {
    let what = format!("videos[{index}]");
    match decode_first_stream(path, max_audio_seconds() * SAMPLE_RATE) {
        Ok(s) if s.is_empty() => Ok(None),
        Ok(s) => {
            check_length(s.len(), &format!("{what} soundtrack"))?;
            Ok(Some(s))
        }
        Err(NoAudio::NoStream) => Ok(None),
        Err(NoAudio::Failed(m)) => Err(Invalid(format!(
            "{what}: could not decode the soundtrack of the video ({m})"
        ))
        .into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_length_cap_is_enforced_with_the_env_knob() {
        // default
        assert_eq!(DEFAULT_MAX_AUDIO_SECONDS, 300);
        assert!(check_length(300 * SAMPLE_RATE, "audio[0]").is_ok());
        let e = check_length(300 * SAMPLE_RATE + 1, "audio[0]").unwrap_err();
        assert!(e.downcast_ref::<Invalid>().is_some());
        assert!(format!("{e}").contains("300 s"));
    }

    #[test]
    fn too_short_clips_are_client_errors() {
        assert!(check_clip_len(399, "audio[0]").is_err());
        assert!(check_clip_len(400, "audio[0]").is_ok());
    }
}
