//! Audio frontend: waveform decoding (16 kHz mono f32) and the USM-style
//! log-mel feature extractor used by the Gemma 4 / EmbeddingGemma 2 audio tower.
//!
//! Mirrors `Gemma4AudioFeatureExtractor` (`feature_extraction_gemma4.py`):
//! semicausal padding, periodic Hann window, 512-point rFFT (single precision,
//! like `np.fft.rfft` on f32 input), HTK mel filter bank with `norm=None`,
//! `log(x + 1e-3)`, frame mask, and padding to a multiple of 128 samples.

use std::path::Path;
use std::process::Command;
use std::sync::OnceLock;

use anyhow::{Context, Result, bail};
use rustfft::num_complex::Complex32;
use rustfft::{FftPlanner, Fft};

pub const SAMPLE_RATE: usize = 16_000;
pub const FRAME_LENGTH: usize = 320;
pub const HOP_LENGTH: usize = 160;
pub const FFT_LENGTH: usize = 512;
pub const MEL_BINS: usize = 128;
/// `pad_to_multiple_of` in the feature extractor.
pub const PAD_TO_MULTIPLE: usize = 128;
/// `max_length` in the feature extractor (30 s), with truncation.
pub const MAX_SAMPLES: usize = 480_000;
const MEL_FLOOR: f64 = 1e-3;
const PAD_LEFT: usize = FRAME_LENGTH / 2;
const FREQ_BINS: usize = FFT_LENGTH / 2 + 1;

/// Log-mel features of one waveform.
pub struct AudioFeatures {
    /// `[frames, MEL_BINS]` row-major, invalid frames zeroed.
    pub mel: Vec<f32>,
    /// Frame validity (from the padded-sample mask).
    pub mask: Vec<bool>,
    pub frames: usize,
    /// Truncated waveform length in samples (before padding to a multiple of 128).
    pub samples: usize,
}

impl AudioFeatures {
    pub fn frame(&self, i: usize) -> &[f32] {
        &self.mel[i * MEL_BINS..(i + 1) * MEL_BINS]
    }

    /// Number of soft tokens: two stride-2 (k=3, p=1) convolutions subsample the
    /// frame axis twice.
    pub fn num_soft_tokens(&self) -> usize {
        subsample_count(subsample_count(self.frames))
    }

    /// Valid soft tokens (the tower's output mask subsampled the same way).
    pub fn valid_soft_tokens(&self) -> usize {
        let (m1, _) = subsample_mask(&self.mask);
        let (m2, _) = subsample_mask(&m1);
        m2.iter().filter(|&&v| v).count()
    }
}

/// Conv output length for kernel 3, stride 2, padding 1.
fn subsample_count(t: usize) -> usize {
    if t == 0 { 0 } else { (t - 1) / 2 + 1 }
}

/// Subsample a mask like the two conv layers: `mask[::2][:t_out]`.
pub fn subsample_mask(mask: &[bool]) -> (Vec<bool>, usize) {
    let t_out = subsample_count(mask.len());
    let out: Vec<bool> = mask.iter().step_by(2).take(t_out).copied().collect();
    (out, t_out)
}

/// Decode an audio file to 16 kHz mono f32 samples. WAV goes through `hound`
/// (16 kHz mono only); anything else is piped through `ffmpeg`.
pub fn load_waveform(path: &Path) -> Result<Vec<f32>> {
    let is_wav = path
        .extension()
        .map(|e| e.eq_ignore_ascii_case("wav"))
        .unwrap_or(false);
    if is_wav {
        match read_wav(path) {
            Ok(samples) => return Ok(samples),
            Err(e) => eprintln!("wav decode failed ({e}); falling back to ffmpeg"),
        }
    }
    read_ffmpeg(path)
}

fn read_wav(path: &Path) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("open wav {}", path.display()))?;
    let spec = reader.spec();
    if spec.sample_rate != SAMPLE_RATE as u32 {
        bail!("wav sample rate is {} (need 16000)", spec.sample_rate);
    }
    if spec.channels != 1 {
        bail!("wav has {} channels (need mono)", spec.channels);
    }
    match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .context("read f32 wav samples"),
        hound::SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / scale))
                .collect::<Result<Vec<_>, _>>()
                .context("read int wav samples")
        }
    }
}

fn read_ffmpeg(path: &Path) -> Result<Vec<f32>> {
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args(["-f", "f32le", "-ac", "1", "-ar", "16000", "-"])
        .output()
        .context("run ffmpeg (is it installed?)")?;
    if !out.status.success() {
        bail!(
            "ffmpeg failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(out
        .stdout
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn window() -> &'static [f32] {
    static W: OnceLock<Vec<f32>> = OnceLock::new();
    W.get_or_init(|| {
        (0..FRAME_LENGTH)
            .map(|n| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / FRAME_LENGTH as f64).cos()) as f32)
            .collect()
    })
}

fn fft() -> &'static std::sync::Arc<dyn Fft<f32>> {
    static F: OnceLock<std::sync::Arc<dyn Fft<f32>>> = OnceLock::new();
    F.get_or_init(|| FftPlanner::<f32>::new().plan_fft_forward(FFT_LENGTH))
}

fn hz_to_mel_htk(f: f64) -> f64 {
    2595.0 * (1.0 + f / 700.0).log10()
}

fn mel_to_hz_htk(m: f64) -> f64 {
    700.0 * (10f64.powf(m / 2595.0) - 1.0)
}

/// `[FREQ_BINS, MEL_BINS]` HTK triangular filter bank (`norm=None`).
fn mel_filter_bank() -> &'static [f64] {
    static M: OnceLock<Vec<f64>> = OnceLock::new();
    M.get_or_init(|| {
        let mel_min = hz_to_mel_htk(0.0);
        let mel_max = hz_to_mel_htk(8000.0);
        let mel_freqs: Vec<f64> = (0..MEL_BINS + 2)
            .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (MEL_BINS + 1) as f64)
            .collect();
        let filter_freqs: Vec<f64> = mel_freqs.iter().map(|m| mel_to_hz_htk(*m)).collect();
        let fft_freqs: Vec<f64> = (0..FREQ_BINS)
            .map(|i| (SAMPLE_RATE as f64 / 2.0) * i as f64 / (FREQ_BINS - 1) as f64)
            .collect();
        let filter_diff: Vec<f64> = (0..MEL_BINS + 1)
            .map(|m| filter_freqs[m + 1] - filter_freqs[m])
            .collect();
        let mut out = vec![0.0f64; FREQ_BINS * MEL_BINS];
        for k in 0..FREQ_BINS {
            for m in 0..MEL_BINS {
                let down = (fft_freqs[k] - filter_freqs[m]) / filter_diff[m];
                let up = (filter_freqs[m + 2] - fft_freqs[k]) / filter_diff[m + 1];
                out[k * MEL_BINS + m] = down.min(up).max(0.0);
            }
        }
        out
    })
}

/// Extract log-mel features from a 16 kHz mono waveform.
pub fn mel_features(waveform: &[f32]) -> AudioFeatures {
    let real = waveform.len().min(MAX_SAMPLES);
    let padded_len = real.div_ceil(PAD_TO_MULTIPLE) * PAD_TO_MULTIPLE;
    let mut padded = vec![0.0f32; PAD_LEFT + padded_len];
    padded[PAD_LEFT..PAD_LEFT + real].copy_from_slice(&waveform[..real]);

    let total = padded.len();
    let frames = if total >= FRAME_LENGTH + 1 {
        (total - (FRAME_LENGTH + 1)) / HOP_LENGTH + 1
    } else {
        0
    };

    let w = window();
    let filters = mel_filter_bank();
    let fft = fft();
    let mut mel = vec![0.0f32; frames * MEL_BINS];
    let mut mask = vec![false; frames];
    let mut buf = vec![Complex32::new(0.0, 0.0); FFT_LENGTH];

    for f in 0..frames {
        let start = f * HOP_LENGTH;
        for n in 0..FRAME_LENGTH {
            buf[n] = Complex32::new(padded[start + n] * w[n], 0.0);
        }
        for b in buf.iter_mut().skip(FRAME_LENGTH) {
            *b = Complex32::new(0.0, 0.0);
        }
        fft.process(&mut buf);

        // A frame is valid when every sample in its analysis window is real audio.
        mask[f] = f * HOP_LENGTH + FRAME_LENGTH - PAD_LEFT < real;
        if !mask[f] {
            continue;
        }
        let out = &mut mel[f * MEL_BINS..(f + 1) * MEL_BINS];
        for m in 0..MEL_BINS {
            let mut acc = 0.0f64;
            for k in 0..FREQ_BINS {
                acc += buf[k].norm() as f64 * filters[k * MEL_BINS + m];
            }
            out[m] = (acc + MEL_FLOOR).ln() as f32;
        }
    }

    AudioFeatures {
        mel,
        mask,
        frames,
        samples: real,
    }
}
