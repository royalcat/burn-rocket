//! The Qwen3-Omni audio feature extractor: `WhisperFeatureExtractor` with `feature_size 128`,
//! `n_fft 400`, `hop 160`, 16 kHz, `padding=True` (zero-pad every clip of the BATCH to the longest
//! one), `truncation=False`, `return_attention_mask=True`.
//!
//! What the HF code (`_torch_extract_fbank_features`) computes, on the zero-padded `[n, N]`
//! waveforms:
//!
//! 1. `torch.stft(n_fft 400, hop 160, periodic Hann(400), center=True)`: reflect-pad 200 samples
//!    on both sides, `1 + N/160` frames; the LAST frame is dropped, `F = N / 160` frames remain;
//! 2. `power = |X|^2` over the 201 bins, `mel = filters^T @ power` (Slaney scale + Slaney norm,
//!    128 filters, 0-8000 Hz), `log10(clamp(mel, 1e-10))`;
//! 3. per CLIP (the whole padded `[128, F]` array): `max(x, max(x) - 8)`, `(x + 4) / 4`;
//! 4. the attention mask `[::160]` of the sample mask, trimmed by one when `N % 160 != 0`
//!    (`min(ceil(n_i / 160), F)` valid frames per clip).
//!
//! **A clip's features depend on its batch mates**: a short clip is zero-padded to the longest
//! clip BEFORE the STFT, so its last valid frames see zeros where an unpadded clip would see
//! reflect padding (docs/experiments/clef-oracle.md §1 / §7.4). The explicit `audio` clips and the soundtracks
//! of audio-in-video records are ONE batch.

use rayon::prelude::*;
use rustfft::num_complex::Complex64;
use rustfft::{Fft, FftPlanner};
use std::sync::{Arc, OnceLock};

pub const SAMPLE_RATE: usize = 16_000;
pub const N_FFT: usize = 400;
pub const HOP: usize = 160;
pub const MEL_BINS: usize = 128;
const FREQ_BINS: usize = N_FFT / 2 + 1;

/// The log-mel features of a batch of clips.
#[derive(Debug, Clone, PartialEq)]
pub struct MelBatch {
    /// `F`: frames of every clip's `[128, F]` array (`N / 160`, `N` = the longest clip).
    pub frames: usize,
    /// Valid frames of every clip (`min(ceil(samples / 160), F)`): the feature length the
    /// placeholder count and the tower derive from.
    pub valid: Vec<usize>,
    /// Per clip, `[128, F]` row-major (mel bin major).
    pub feats: Vec<Vec<f32>>,
}

impl MelBatch {
    pub fn clips(&self) -> usize {
        self.valid.len()
    }
}

/// Slaney `hz_to_mel` (`audio_utils.hz_to_mel(mel_scale="slaney")`).
fn hz_to_mel(f: f64) -> f64 {
    let (min_log_hz, min_log_mel) = (1000.0, 15.0);
    let logstep = 27.0 / 6.4f64.ln();
    if f >= min_log_hz {
        min_log_mel + (f / min_log_hz).ln() * logstep
    } else {
        3.0 * f / 200.0
    }
}

fn mel_to_hz(m: f64) -> f64 {
    let (min_log_hz, min_log_mel) = (1000.0, 15.0);
    let logstep = 6.4f64.ln() / 27.0;
    if m >= min_log_mel {
        min_log_hz * (logstep * (m - min_log_mel)).exp()
    } else {
        200.0 * m / 3.0
    }
}

/// `[201, 128]` Slaney-normalized triangular filter bank (`mel_filter_bank(..., norm="slaney",
/// mel_scale="slaney")`), rounded to f32 as the extractor does (`.to(torch.float32)`).
pub fn mel_filters() -> &'static [f32] {
    static F: OnceLock<Vec<f32>> = OnceLock::new();
    F.get_or_init(|| {
        let (mel_min, mel_max) = (hz_to_mel(0.0), hz_to_mel(8000.0));
        let filter_freqs: Vec<f64> = (0..MEL_BINS + 2)
            .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f64 / (MEL_BINS + 1) as f64))
            .collect();
        let fft_freqs: Vec<f64> = (0..FREQ_BINS)
            .map(|i| (SAMPLE_RATE / 2) as f64 * i as f64 / (FREQ_BINS - 1) as f64)
            .collect();
        let mut out = vec![0f32; FREQ_BINS * MEL_BINS];
        for (k, &fq) in fft_freqs.iter().enumerate() {
            for m in 0..MEL_BINS {
                let down = -(filter_freqs[m] - fq) / (filter_freqs[m + 1] - filter_freqs[m]);
                let up = (filter_freqs[m + 2] - fq) / (filter_freqs[m + 2] - filter_freqs[m + 1]);
                let enorm = 2.0 / (filter_freqs[m + 2] - filter_freqs[m]);
                out[k * MEL_BINS + m] = (down.min(up).max(0.0) * enorm) as f32;
            }
        }
        out
    })
}

fn hann() -> &'static [f64] {
    static W: OnceLock<Vec<f64>> = OnceLock::new();
    W.get_or_init(|| {
        (0..N_FFT)
            .map(|n| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / N_FFT as f64).cos())
            .collect()
    })
}

fn fft() -> &'static Arc<dyn Fft<f64>> {
    static F: OnceLock<Arc<dyn Fft<f64>>> = OnceLock::new();
    F.get_or_init(|| FftPlanner::<f64>::new().plan_fft_forward(N_FFT))
}

/// `torch.stft` input of one clip: zero-padded to `n`, then reflect-padded by `N_FFT / 2`.
fn reflect_padded(clip: &[f32], n: usize) -> Vec<f32> {
    let pad = N_FFT / 2;
    let mut x = vec![0f32; n + 2 * pad];
    x[pad..pad + clip.len()].copy_from_slice(clip);
    // the reflection reads the ZERO-padded signal (`n` samples), as torch does
    for i in 0..pad {
        x[pad - 1 - i] = x[pad + 1 + i];
        x[pad + n + i] = x[pad + n - 2 - i];
    }
    x
}

/// `log_mel_batch` of the module doc. `clips` are 16 kHz mono f32. The caller guarantees the
/// longest clip has more than `N_FFT / 2` samples (torch's reflect padding needs it) and that
/// every clip is non-empty.
pub fn log_mel_batch(clips: &[&[f32]]) -> MelBatch {
    let n = clips.iter().map(|c| c.len()).max().unwrap_or(0);
    assert!(
        n > N_FFT / 2,
        "the longest clip needs more than 200 samples"
    );
    let frames = n / HOP;
    let filters = mel_filters();
    let win = hann();
    let valid = clips
        .iter()
        .map(|c| c.len().div_ceil(HOP).min(frames))
        .collect();
    let feats = clips
        .iter()
        .map(|clip| {
            let x = reflect_padded(clip, n);
            // power spectra [frames][201]
            let mut power = vec![0f64; frames * FREQ_BINS];
            power.par_chunks_mut(FREQ_BINS).enumerate().for_each_init(
                || vec![Complex64::new(0.0, 0.0); N_FFT],
                |buf, (f, row)| {
                    for (i, b) in buf.iter_mut().enumerate() {
                        *b = Complex64::new(f64::from(x[f * HOP + i]) * win[i], 0.0);
                    }
                    fft().process(buf);
                    for (p, c) in row.iter_mut().zip(buf.iter()) {
                        *p = c.norm_sqr();
                    }
                },
            );
            // mel[m][f] = sum_k filters[k][m] * power[f][k]; log10, per-clip floor, scale
            let mut out = vec![0f32; MEL_BINS * frames];
            out.par_chunks_mut(frames).enumerate().for_each(|(m, row)| {
                for (f, o) in row.iter_mut().enumerate() {
                    let p = &power[f * FREQ_BINS..(f + 1) * FREQ_BINS];
                    let mut acc = 0f64;
                    for (k, &pk) in p.iter().enumerate() {
                        acc += f64::from(filters[k * MEL_BINS + m]) * pk;
                    }
                    *o = acc.max(1e-10).log10() as f32;
                }
            });
            let max = out.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let floor = max - 8.0;
            out.par_iter_mut()
                .for_each(|v| *v = (v.max(floor) + 4.0) / 4.0);
            out
        })
        .collect();
    MelBatch {
        frames,
        valid,
        feats,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slaney_scale_round_trips_and_has_the_known_breakpoints() {
        assert!((hz_to_mel(1000.0) - 15.0).abs() < 1e-12);
        assert!((hz_to_mel(200.0) - 3.0).abs() < 1e-12);
        for hz in [0.0, 100.0, 999.0, 1000.0, 4321.0, 8000.0] {
            assert!((mel_to_hz(hz_to_mel(hz)) - hz).abs() < 1e-9, "{hz}");
        }
    }

    #[test]
    fn the_filter_bank_is_slaney_normalized() {
        let f = mel_filters();
        assert_eq!(f.len(), 201 * 128);
        // every filter is a nonnegative triangle; the area under it is ~1 (slaney norm 2/width
        // over the Hz-spaced bins: sum_k f[k][m] * 40 Hz ~ 1 for the wide filters)
        for m in 0..MEL_BINS {
            let col: Vec<f32> = (0..201).map(|k| f[k * 128 + m]).collect();
            assert!(col.iter().all(|&v| v >= 0.0));
            assert!(col.iter().any(|&v| v > 0.0), "filter {m} is empty");
        }
        let area: f32 = (0..201).map(|k| f[k * 128 + 100]).sum::<f32>() * 40.0;
        assert!((area - 1.0).abs() < 0.05, "area {area}");
    }

    #[test]
    fn a_short_clip_is_padded_to_the_longest_before_the_stft() {
        // a clip alone vs the same clip next to a longer one: only the last valid frames differ
        let a: Vec<f32> = (0..8000).map(|i| (i as f32 * 0.07).sin() * 0.3).collect();
        let b: Vec<f32> = (0..16000).map(|i| (i as f32 * 0.02).sin() * 0.2).collect();
        let alone = log_mel_batch(&[&a]);
        let both = log_mel_batch(&[&a, &b]);
        assert_eq!(alone.frames, 50);
        assert_eq!(both.frames, 100);
        assert_eq!(both.valid, [50, 100]);
        // frames well before the end are identical in the two batches (same signal around)
        for m in [0usize, 40, 127] {
            for f in 0..45 {
                let (x, y) = (alone.feats[0][m * 50 + f], both.feats[0][m * 100 + f]);
                assert!((x - y).abs() < 1e-6, "m{m} f{f}: {x} vs {y}");
            }
        }
        // the last frame sees reflect padding alone, zeros next to the longer clip
        let last: f32 = (0..128)
            .map(|m| (alone.feats[0][m * 50 + 49] - both.feats[0][m * 100 + 49]).abs())
            .fold(0.0, f32::max);
        assert!(last > 1e-3, "the padding artefact vanished: {last}");
    }

    #[test]
    fn the_valid_length_follows_the_trimmed_attention_mask() {
        // N = 16100: F = 100, a 16100-sample clip has ceil(16100/160) = 101 mask entries, trimmed
        let a = vec![0.1f32; 16100];
        let b = vec![0.1f32; 8000];
        let m = log_mel_batch(&[&a, &b]);
        assert_eq!((m.frames, m.valid.clone()), (100, vec![100, 50]));
    }

    #[test]
    fn values_are_floored_eight_decades_under_the_peak() {
        let a: Vec<f32> = (0..4800).map(|i| (i as f32 * 0.3).sin()).collect();
        let m = log_mel_batch(&[&a]);
        let (mx, mn) = m.feats[0]
            .iter()
            .fold((f32::MIN, f32::MAX), |(a, b), &v| (a.max(v), b.min(v)));
        // (log10 + 4) / 4: the floor is peak - 8/4
        assert!(mx - mn <= 2.0 + 1e-6, "{mx} {mn}");
    }
}
