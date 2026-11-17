//! The Qwen3-Omni audio tower ("AuT" encoder) of Clef-Omni (`thinker.audio_tower.*`, phase 8e):
//! 128-bin log-mel features in, `[tokens, 2048]` features out, as the thinker splices them over
//! the `<|audio_pad|>` embeddings ([`TowerOutputs::audio`](super::media::TowerOutputs)).
//!
//! Port spec: `tools/clef_media.py::manual_audio` (plain torch, checked against the HF module),
//! reproduced on the host in f32:
//!
//! 1. each clip's valid mel frames `[128, L]` are cut into chunks of 100 frames (`2 * n_window`);
//!    the last chunk is shorter and zero-padded to 100 on the right;
//! 2. three `Conv2d(k3, s2, p1)` + GELU(erf) (480 channels): `[480, 16, 13]` per chunk (time
//!    100 -> 50 -> 25 -> 13, mel 128 -> 64 -> 32 -> 16);
//! 3. `[chunk, 13, 480*16]` (channel-major `(c, f)` flatten) -> `conv_out` Linear(7680 -> 1280,
//!    no bias) + the sinusoid table (positions 0..12, restarting in EVERY chunk);
//! 4. only the valid steps of each chunk are kept (`audio_out_len` of the chunk), chunks of all
//!    clips concatenate to `[A, 1280]`;
//! 5. attention windows over that sequence, per clip, of `max_valid * (n_window_infer / 100)` =
//!    104 tokens (a shorter last window): full attention inside a window, never across;
//! 6. 32 pre-LN layers (LayerNorm eps 1e-5): `x += attn(LN(x))` (q/k/v/out with bias, 20 x 64
//!    heads, no rotary), `x += fc2(gelu(fc1(LN(x))))` (FFN 5120);
//! 7. `ln_post`, `proj1` (1280 -> 1280) + GELU(erf), `proj2` (1280 -> 2048).
//!
//! **Weights** are BF16 in both checkpoints (the BF16 release: shard 1; the AutoRound repo: shard
//! 6; the recipe quantizes only `thinker.model.layers`), mmapped and widened per use.
//!
//! **Placement** ([`TowerDevice`], shared with the vision tower and loaded lazily and
//! independently of it): `Cpu` is f32; `Emu16` the host emulation of the fp16 NPU numerics;
//! `Npu` (aarch64 `npu` builds) packs every linear as a resident fp16 weight (the two conv
//! layers that matter, `conv2d2`/`conv2d3`, run as im2col matmuls with `K = 4320`; `conv2d1`
//! has `K = 9` and always runs on the host). Attention is per 104-token window: ~1,200 tiny
//! dispatches for 300 s, which the driver pays more for than the 2 GFLOP/layer the CPU needs,
//! so the window attention stays on the host in f32 in EVERY placement unless
//! `ROCKET_CLEF_AUDIO_ATTN=device` asks the placement to run it (`flash_attn` per window).

use super::media::audio_out_len;
use super::model::Shards;
use super::vision::compute::{CpuExec, Exec, Fp16Range, TowerDevice, full_attention, make_exec};
use super::vision::{Lin, widen_par};
use anyhow::{Context, Result, ensure};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Mutex;

pub mod decode;
pub mod mel;
#[cfg(test)]
mod tests;

#[allow(unused_imports)]
pub use decode::{
    DEFAULT_MAX_AUDIO_SECONDS, MIN_SAMPLES, check_clip_len, load_audio, max_audio_seconds,
    soundtrack,
};
pub use mel::{MelBatch, log_mel_batch};

/// `thinker.audio_tower.` tensors of the checkpoint.
pub const PREFIX: &str = "thinker.audio_tower.";
/// LayerNorm epsilon of every tower norm (torch default).
pub const LN_EPS: f32 = 1e-5;
/// Mel frames of one chunk (`2 * n_window`).
pub const CHUNK_FRAMES: usize = 100;
/// Chunks per conv stem batch (bounds the im2col matrices to ~110 MiB).
const STEM_BATCH: usize = 8;

/// `thinker_config.audio_config` (the fields the forward uses).
#[derive(Debug, Clone, PartialEq)]
pub struct AudioConfig {
    pub layers: usize,
    pub d_model: usize,
    pub heads: usize,
    pub ffn: usize,
    pub mel_bins: usize,
    pub downsample: usize,
    pub out_dim: usize,
    pub n_window: usize,
    pub n_window_infer: usize,
}

impl AudioConfig {
    /// The shipped Clef-Omni geometry (what `config.json` of both checkpoints says).
    pub fn shipped() -> Self {
        Self {
            layers: 32,
            d_model: 1280,
            heads: 20,
            ffn: 5120,
            mel_bins: 128,
            downsample: 480,
            out_dim: 2048,
            n_window: 50,
            n_window_infer: 800,
        }
    }

    /// Read `thinker_config.audio_config` of `dir/config.json`.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let p = dir.join("config.json");
        let v: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&p).with_context(|| format!("read {}", p.display()))?,
        )?;
        let ac = &v["thinker_config"]["audio_config"];
        ensure!(
            ac.is_object(),
            "config.json has no thinker_config.audio_config"
        );
        let get = |k: &str| -> Result<usize> {
            ac[k]
                .as_u64()
                .map(|v| v as usize)
                .with_context(|| format!("audio_config.{k} missing"))
        };
        let cfg = Self {
            layers: get("encoder_layers")?,
            d_model: get("d_model")?,
            heads: get("encoder_attention_heads")?,
            ffn: get("encoder_ffn_dim")?,
            mel_bins: get("num_mel_bins")?,
            downsample: get("downsample_hidden_size")?,
            out_dim: get("output_dim")?,
            n_window: get("n_window")?,
            n_window_infer: get("n_window_infer")?,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.heads > 0 && self.d_model.is_multiple_of(self.heads),
            "audio d_model {} / heads {}",
            self.d_model,
            self.heads
        );
        ensure!(
            self.n_window == 50 && self.mel_bins == 128 && self.n_window_infer.is_multiple_of(100),
            "this port is written for n_window 50, 128 mel bins and a window that is a multiple of \
             100 frames (got {} / {} / {})",
            self.n_window,
            self.mel_bins,
            self.n_window_infer
        );
        Ok(())
    }

    pub fn head_dim(&self) -> usize {
        self.d_model / self.heads
    }

    /// Mel rows after the three stride-2 convs (128 -> 16).
    pub fn mel_out(&self) -> usize {
        sub2(sub2(sub2(self.mel_bins)))
    }

    /// Time steps of a full chunk after the convs (100 -> 13).
    pub fn steps(&self) -> usize {
        sub2(sub2(sub2(2 * self.n_window)))
    }

    /// `conv_out`'s input width: `downsample * mel_out` (7680).
    pub fn conv_flat(&self) -> usize {
        self.downsample * self.mel_out()
    }

    /// Attention window in tokens for a batch whose longest chunk keeps `max_valid` steps.
    pub fn window(&self, max_valid: usize) -> usize {
        max_valid * (self.n_window_infer / (2 * self.n_window))
    }
}

/// `Conv2d(k3, s2, p1)` output length.
fn sub2(n: usize) -> usize {
    (n - 1) / 2 + 1
}

struct Layer {
    n1w: Vec<f32>,
    n1b: Vec<f32>,
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    n2w: Vec<f32>,
    n2b: Vec<f32>,
    fc1: Lin,
    fc2: Lin,
}

/// Everything small and the linear table of the tower (the big weights stay in the shards).
pub struct Plan {
    pub cfg: AudioConfig,
    conv1_w: Vec<f32>,
    conv1_b: Vec<f32>,
    conv2: Lin,
    conv3: Lin,
    conv_out: Lin,
    layers: Vec<Layer>,
    ln_w: Vec<f32>,
    ln_b: Vec<f32>,
    proj1: Lin,
    proj2: Lin,
    pub lins: usize,
}

impl Plan {
    /// Every linear of the tower in slot order.
    pub fn linears(&self) -> Vec<&Lin> {
        let mut v = vec![&self.conv2, &self.conv3, &self.conv_out];
        for l in &self.layers {
            v.extend([&l.q, &l.k, &l.v, &l.o, &l.fc1, &l.fc2]);
        }
        v.extend([&self.proj1, &self.proj2]);
        v
    }

    /// Index the tower: validate every tensor's shape against `cfg` and read the small ones.
    pub fn build(shards: &Shards, cfg: &AudioConfig) -> Result<Self> {
        let (d, ffn, ds) = (cfg.d_model, cfg.ffn, cfg.downsample);
        let mut slot = 0usize;
        let mut lin = |w: &str, b: Option<&str>, n: usize, k: usize, shape: &[usize]| {
            let name = format!("{PREFIX}{w}");
            let (_, info) = shards.get(&name)?;
            ensure!(
                info.shape == shape,
                "{name}: shape {:?}, expected {shape:?}",
                info.shape
            );
            let l = Lin {
                name,
                k,
                n,
                bias: match b {
                    Some(b) => shards.f32(&format!("{PREFIX}{b}"), &[n])?,
                    None => Vec::new(),
                },
                slot,
            };
            slot += 1;
            Ok::<_, anyhow::Error>(l)
        };
        let conv2 = lin(
            "conv2d2.weight",
            Some("conv2d2.bias"),
            ds,
            ds * 9,
            &[ds, ds, 3, 3],
        )?;
        let conv3 = lin(
            "conv2d3.weight",
            Some("conv2d3.bias"),
            ds,
            ds * 9,
            &[ds, ds, 3, 3],
        )?;
        let conv_out = lin(
            "conv_out.weight",
            None,
            d,
            cfg.conv_flat(),
            &[d, cfg.conv_flat()],
        )?;
        let vec_of = |n: &str, len: usize| shards.f32(&format!("{PREFIX}{n}"), &[len]);
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let p = format!("layers.{i}.");
            let mut l = |n: &str, out: usize, inp: usize| {
                lin(
                    &format!("{p}{n}.weight"),
                    Some(&format!("{p}{n}.bias")),
                    out,
                    inp,
                    &[out, inp],
                )
            };
            layers.push(Layer {
                n1w: vec_of(&format!("{p}self_attn_layer_norm.weight"), d)?,
                n1b: vec_of(&format!("{p}self_attn_layer_norm.bias"), d)?,
                q: l("self_attn.q_proj", d, d)?,
                k: l("self_attn.k_proj", d, d)?,
                v: l("self_attn.v_proj", d, d)?,
                o: l("self_attn.out_proj", d, d)?,
                n2w: vec_of(&format!("{p}final_layer_norm.weight"), d)?,
                n2b: vec_of(&format!("{p}final_layer_norm.bias"), d)?,
                fc1: l("fc1", ffn, d)?,
                fc2: l("fc2", d, ffn)?,
            });
        }
        let proj1 = lin("proj1.weight", Some("proj1.bias"), d, d, &[d, d])?;
        let proj2 = lin(
            "proj2.weight",
            Some("proj2.bias"),
            cfg.out_dim,
            d,
            &[cfg.out_dim, d],
        )?;
        Ok(Self {
            cfg: cfg.clone(),
            conv1_w: shards.f32(&format!("{PREFIX}conv2d1.weight"), &[ds, 1, 3, 3])?,
            conv1_b: vec_of("conv2d1.bias", ds)?,
            conv2,
            conv3,
            conv_out,
            layers,
            ln_w: vec_of("ln_post.weight", d)?,
            ln_b: vec_of("ln_post.bias", d)?,
            proj1,
            proj2,
            lins: slot,
        })
    }
}

/// What a tower run returns.
#[derive(Debug, Default, Clone)]
pub struct AudioOut {
    /// `[A, 2048]`: what replaces the `<|audio_pad|>` embeddings (all clips concatenated).
    pub features: Vec<f32>,
    /// Debug taps by name: `conv_out` (`[A, 1280]` after the sinusoid), `layer_{k}`, `ln_post`.
    pub taps: BTreeMap<String, Vec<f32>>,
    /// Tokens of every clip (`audio_out_len(valid frames)`).
    pub lens: Vec<usize>,
}

/// The sinusoid table `[steps, d]`: `[sin(t w_j) | cos(t w_j)]`, `w_j = exp(-ln(10000)/(d/2-1) j)`
/// (`SinusoidsPositionEmbedding`, computed in f32 like torch).
pub fn sinusoid_table(steps: usize, d: usize) -> Vec<f32> {
    let half = d / 2;
    let inc = (10000f64.ln() / (half - 1) as f64) as f32;
    let inv: Vec<f32> = (0..half).map(|j| (-inc * j as f32).exp()).collect();
    let mut t = vec![0f32; steps * d];
    for (s, row) in t.chunks_mut(d).enumerate() {
        for (j, &w) in inv.iter().enumerate() {
            let a = s as f32 * w;
            row[j] = a.sin();
            row[half + j] = a.cos();
        }
    }
    t
}

/// `conv2d1` (1 -> C channels, k3 s2 p1) + GELU on zero-padded chunks `[b, 128, 100]`:
/// channel-major `[b, C, 64, 50]`.
fn conv1(x: &[f32], b: usize, w: &[f32], bias: &[f32], mels: usize, frames: usize) -> Vec<f32> {
    let c = bias.len();
    let (ho, wo) = (sub2(mels), sub2(frames));
    let mut out = vec![0f32; b * c * ho * wo];
    out.par_chunks_mut(ho * wo).enumerate().for_each(|(i, o)| {
        let (bi, oc) = (i / c, i % c);
        let src = &x[bi * mels * frames..(bi + 1) * mels * frames];
        let k = &w[oc * 9..oc * 9 + 9];
        for f in 0..ho {
            for t in 0..wo {
                let mut acc = bias[oc];
                for kh in 0..3 {
                    let r = (2 * f + kh) as isize - 1;
                    if r < 0 || r >= mels as isize {
                        continue;
                    }
                    for kw in 0..3 {
                        let q = (2 * t + kw) as isize - 1;
                        if q < 0 || q >= frames as isize {
                            continue;
                        }
                        acc += k[kh * 3 + kw] * src[r as usize * frames + q as usize];
                    }
                }
                o[f * wo + t] = super::math::gelu(acc);
            }
        }
    });
    out
}

/// im2col of a channel-major `[b, c, h, w]` activation for a `3x3 / s2 / p1` conv: rows
/// `(b, fo, to)`, columns `(ic, kh, kw)` (the stored `[oc, ic, kh, kw]` weight order).
fn im2col(src: &[f32], b: usize, c: usize, h: usize, w: usize) -> (Vec<f32>, usize, usize) {
    let (ho, wo) = (sub2(h), sub2(w));
    let cols = c * 9;
    let mut out = vec![0f32; b * ho * wo * cols];
    out.par_chunks_mut(cols).enumerate().for_each(|(r, row)| {
        let (bi, rem) = (r / (ho * wo), r % (ho * wo));
        let (fo, to) = (rem / wo, rem % wo);
        for ic in 0..c {
            let plane = &src[(bi * c + ic) * h * w..(bi * c + ic + 1) * h * w];
            for kh in 0..3 {
                let y = (2 * fo + kh) as isize - 1;
                if y < 0 || y >= h as isize {
                    continue;
                }
                for kw in 0..3 {
                    let x = (2 * to + kw) as isize - 1;
                    if x < 0 || x >= w as isize {
                        continue;
                    }
                    row[ic * 9 + kh * 3 + kw] = plane[y as usize * w + x as usize];
                }
            }
        }
    });
    (out, ho, wo)
}

/// One chunk of a clip: `len` valid mel frames starting at `start`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Chunk {
    clip: usize,
    start: usize,
    len: usize,
}

fn chunks_of(valid: &[usize]) -> Vec<Chunk> {
    let mut v = Vec::new();
    for (clip, &l) in valid.iter().enumerate() {
        for c in 0..l.div_ceil(CHUNK_FRAMES) {
            let start = c * CHUNK_FRAMES;
            v.push(Chunk {
                clip,
                start,
                len: CHUNK_FRAMES.min(l - start),
            });
        }
    }
    v
}

/// The attention windows `(start, end)` over the concatenated token sequence: per clip, windows
/// of `window` tokens from the clip's start (a shorter last one).
pub fn window_bounds(tokens_per_clip: &[usize], window: usize) -> Vec<(usize, usize)> {
    let mut segs = Vec::new();
    let mut off = 0;
    for &cnt in tokens_per_clip {
        let mut s = 0;
        while s < cnt {
            let e = (s + window).min(cnt);
            segs.push((off + s, off + e));
            s = e;
        }
        off += cnt;
    }
    segs
}

fn add_in_place(x: &mut [f32], y: &[f32]) {
    x.par_chunks_mut(1 << 14)
        .zip(y.par_chunks(1 << 14))
        .for_each(|(a, b)| a.iter_mut().zip(b).for_each(|(a, b)| *a += b));
}

/// `(cosine, max |x - y| / max |y|)` of two equally long vectors.
fn row_agreement(x: &[f32], y: &[f32]) -> (f64, f64) {
    let (mut dot, mut nx, mut ny, mut md, mut peak) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (&a, &b) in x.iter().zip(y) {
        let (a, b) = (f64::from(a), f64::from(b));
        dot += a * b;
        nx += a * a;
        ny += b * b;
        md = md.max((a - b).abs());
        peak = peak.max(b.abs());
    }
    (
        dot / (nx.sqrt() * ny.sqrt()).max(1e-300),
        md / peak.max(1e-30),
    )
}

/// The audio tower: the index of its tensors, the mmapped shards and the compute placement.
pub struct AudioTower {
    plan: Plan,
    shards: Shards,
    exec: Box<dyn Exec>,
    /// The CPU fallback of an fp16 placement whose range overflowed.
    cpu: CpuExec,
    device: TowerDevice,
    /// Window attention on the placement instead of the host (`ROCKET_CLEF_AUDIO_ATTN=device`).
    attn_on_device: bool,
    sin: Vec<f32>,
    fallbacks: Mutex<usize>,
}

impl std::fmt::Debug for AudioTower {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioTower")
            .field("device", &self.device)
            .field("layers", &self.plan.cfg.layers)
            .finish_non_exhaustive()
    }
}

impl AudioTower {
    /// Whether the checkpoint in `dir` has the tower's tensors (the BF16 checkpoint and the
    /// AutoRound shards do; a `clef-pack` skips them).
    pub fn available(dir: &Path) -> bool {
        let Ok(bytes) = std::fs::read(dir.join("model.safetensors.index.json")) else {
            return false;
        };
        let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
            return false;
        };
        v["weight_map"]
            .as_object()
            .is_some_and(|m| m.contains_key(&format!("{PREFIX}conv2d1.weight")))
    }

    /// Open the tower of the checkpoint in `dir` on `device`. `threads` sizes the NPU contexts.
    pub fn open(dir: &Path, device: TowerDevice, threads: usize) -> Result<Self> {
        let cfg = AudioConfig::from_dir(dir)?;
        let shards = Shards::open_with(dir, |k| k.starts_with(PREFIX))
            .context("open the shards that hold the audio tower")?;
        let plan = Plan::build(&shards, &cfg).context("index the audio tower")?;
        Self::from_parts(plan, shards, device, threads)
    }

    pub fn from_parts(
        plan: Plan,
        shards: Shards,
        device: TowerDevice,
        threads: usize,
    ) -> Result<Self> {
        let exec = make_exec(device, &plan.linears(), &shards, threads)?;
        let attn_on_device = matches!(
            std::env::var("ROCKET_CLEF_AUDIO_ATTN").ok().as_deref(),
            Some("device" | "npu")
        );
        let sin = sinusoid_table(plan.cfg.steps(), plan.cfg.d_model);
        Ok(Self {
            plan,
            shards,
            exec,
            cpu: CpuExec,
            device,
            attn_on_device,
            sin,
            fallbacks: Mutex::new(0),
        })
    }

    pub fn config(&self) -> &AudioConfig {
        &self.plan.cfg
    }

    pub fn device(&self) -> TowerDevice {
        self.device
    }

    pub fn label(&self) -> String {
        format!(
            "{}, window attention on the {}",
            self.exec.label(),
            if self.attn_on_device {
                "placement"
            } else {
                "host (f32)"
            }
        )
    }

    /// Encode a batch of clips (record order: explicit clips, then soundtracks): features of
    /// every clip concatenated. An fp16 range overflow of the placement reruns the whole tower on
    /// the CPU in f32. `taps` names the intermediate arrays to keep (`conv_out`, `layer_{k}`,
    /// `ln_post`).
    pub fn forward(&self, mel: &MelBatch, threads: usize, taps: &[String]) -> Result<AudioOut> {
        match self.run(&*self.exec, mel, threads, taps) {
            Err(e) if e.downcast_ref::<Fp16Range>().is_some() => {
                eprintln!("clef: audio tower: {e:#}; rerunning this batch in f32 on the CPU");
                *self.fallbacks.lock().unwrap_or_else(|p| p.into_inner()) += 1;
                self.run(&self.cpu, mel, threads, taps)
            }
            r => r,
        }
    }

    /// Debug aid (`ROCKET_CLEF_TOWER_CHECK=1`): run `mel` through the placement and through the
    /// f32 CPU reference and describe the agreement (cosine, max |d| relative to the peak).
    pub fn check_against_cpu(&self, mel: &MelBatch, threads: usize) -> Result<String> {
        let a = self.forward(mel, threads, &[])?;
        let b = self.run(&self.cpu, mel, threads, &[])?;
        let (cos, rel) = row_agreement(&a.features, &b.features);
        Ok(format!("features cos {cos:.7} rel-to-peak {rel:.2e}"))
    }

    pub fn fallbacks(&self) -> usize {
        *self.fallbacks.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Release the mapped shard pages and the placement's scratch after a request.
    pub fn trim(&self, keep_bytes: usize) {
        self.shards.release();
        self.exec.trim(keep_bytes);
    }

    fn linear(
        &self,
        exec: &dyn Exec,
        lin: &Lin,
        x: &[f32],
        m: usize,
        threads: usize,
    ) -> Result<Vec<f32>> {
        let mut y = exec.matmul(lin, &self.shards, x, m, threads)?;
        if !lin.bias.is_empty() {
            y.par_chunks_mut(lin.n).for_each(|r| {
                for (v, b) in r.iter_mut().zip(&lin.bias) {
                    *v += b;
                }
            });
        }
        Ok(y)
    }

    /// The conv stem + `conv_out` + sinusoid of a batch of chunks: `[chunks * steps, d]`.
    fn stem(
        &self,
        exec: &dyn Exec,
        mel: &MelBatch,
        chunks: &[Chunk],
        threads: usize,
    ) -> Result<Vec<f32>> {
        let cfg = &self.plan.cfg;
        let (mels, steps, ds) = (cfg.mel_bins, cfg.steps(), cfg.downsample);
        let (fo, d) = (cfg.mel_out(), cfg.d_model);
        let b = chunks.len();
        // zero-padded chunks [b, 128, 100]
        let mut x = vec![0f32; b * mels * CHUNK_FRAMES];
        for (i, ch) in chunks.iter().enumerate() {
            let feats = &mel.feats[ch.clip];
            for m in 0..mels {
                x[(i * mels + m) * CHUNK_FRAMES..][..ch.len]
                    .copy_from_slice(&feats[m * mel.frames + ch.start..][..ch.len]);
            }
        }
        let h1 = conv1(
            &x,
            b,
            &self.plan.conv1_w,
            &self.plan.conv1_b,
            mels,
            CHUNK_FRAMES,
        );
        drop(x);
        let (h, w) = (sub2(mels), sub2(CHUNK_FRAMES));
        // conv2 as an im2col matmul, GELU, back to channel-major for the next im2col
        let (cols, ho, wo) = im2col(&h1, b, ds, h, w);
        drop(h1);
        let mut y = self.linear(exec, &self.plan.conv2, &cols, b * ho * wo, threads)?;
        drop(cols);
        super::math::gelu_in_place(&mut y);
        let mut h2 = vec![0f32; b * ds * ho * wo];
        h2.par_chunks_mut(ho * wo).enumerate().for_each(|(i, o)| {
            let (bi, c) = (i / ds, i % ds);
            for (p, v) in o.iter_mut().enumerate() {
                *v = y[(bi * ho * wo + p) * ds + c];
            }
        });
        drop(y);
        // conv3 (channels-last output), GELU
        let (cols, ho3, wo3) = im2col(&h2, b, ds, ho, wo);
        drop(h2);
        let mut y3 = self.linear(exec, &self.plan.conv3, &cols, b * ho3 * wo3, threads)?;
        drop(cols);
        super::math::gelu_in_place(&mut y3);
        ensure!(
            ho3 == fo && wo3 == steps,
            "audio conv stem produced {ho3}x{wo3}, expected {fo}x{steps}"
        );
        // [b, steps, ds * fo] with the (c, f) flatten
        let flat = ds * fo;
        let mut z = vec![0f32; b * steps * flat];
        z.par_chunks_mut(flat).enumerate().for_each(|(r, row)| {
            let (bi, t) = (r / steps, r % steps);
            for c in 0..ds {
                for f in 0..fo {
                    row[c * fo + f] = y3[(bi * fo * steps + f * steps + t) * ds + c];
                }
            }
        });
        drop(y3);
        let mut out = self.linear(exec, &self.plan.conv_out, &z, b * steps, threads)?;
        out.par_chunks_mut(d).enumerate().for_each(|(r, row)| {
            for (v, s) in row.iter_mut().zip(&self.sin[(r % steps) * d..][..d]) {
                *v += s;
            }
        });
        Ok(out)
    }

    fn run(
        &self,
        exec: &dyn Exec,
        mel: &MelBatch,
        threads: usize,
        taps: &[String],
    ) -> Result<AudioOut> {
        let plan = &self.plan;
        let cfg = &plan.cfg;
        let (d, heads, hd) = (cfg.d_model, cfg.heads, cfg.head_dim());
        let steps = cfg.steps();
        ensure!(
            mel.valid.iter().all(|&v| v > 0) && mel.feats.len() == mel.valid.len(),
            "an audio clip without mel frames"
        );
        let chunks = chunks_of(&mel.valid);
        let want = |n: &str| taps.iter().any(|t| t == n);

        // 1-4: the stem, valid steps only, clips concatenated
        let mut x: Vec<f32> = Vec::new();
        let mut lens = vec![0usize; mel.valid.len()];
        let mut max_valid = 0;
        for group in chunks.chunks(STEM_BATCH) {
            let out = self.stem(exec, mel, group, threads)?;
            for (i, ch) in group.iter().enumerate() {
                let v = audio_out_len(ch.len);
                max_valid = max_valid.max(v);
                lens[ch.clip] += v;
                x.extend_from_slice(&out[i * steps * d..][..v * d]);
            }
        }
        let a = x.len() / d;
        let mut res = AudioOut {
            lens: lens.clone(),
            ..Default::default()
        };
        if want("conv_out") {
            res.taps.insert("conv_out".into(), x.clone());
        }
        // 5: attention windows
        let segs = window_bounds(&lens, cfg.window(max_valid));
        // 6: the layers
        for (li, l) in plan.layers.iter().enumerate() {
            let n1 = super::math::layer_norm(&x, a, d, &l.n1w, &l.n1b, LN_EPS);
            let q = self.linear(exec, &l.q, &n1, a, threads)?;
            let k = self.linear(exec, &l.k, &n1, a, threads)?;
            let v = self.linear(exec, &l.v, &n1, a, threads)?;
            drop(n1);
            let ctx = if self.attn_on_device {
                exec.attention(&q, &k, &v, &segs, heads, hd, threads)?
            } else {
                full_attention(&q, &k, &v, &segs, heads, hd)
            };
            drop((q, k, v));
            let o = self.linear(exec, &l.o, &ctx, a, threads)?;
            add_in_place(&mut x, &o);
            drop((ctx, o));
            let n2 = super::math::layer_norm(&x, a, d, &l.n2w, &l.n2b, LN_EPS);
            let mut f = self.linear(exec, &l.fc1, &n2, a, threads)?;
            drop(n2);
            super::math::gelu_in_place(&mut f);
            let m = self.linear(exec, &l.fc2, &f, a, threads)?;
            add_in_place(&mut x, &m);
            drop((f, m));
            if want(&format!("layer_{li}")) {
                res.taps.insert(format!("layer_{li}"), x.clone());
            }
        }
        // 7: ln_post, proj1 + GELU, proj2
        let ln = super::math::layer_norm(&x, a, d, &plan.ln_w, &plan.ln_b, LN_EPS);
        if want("ln_post") {
            res.taps.insert("ln_post".into(), ln.clone());
        }
        let mut p = self.linear(exec, &plan.proj1, &ln, a, threads)?;
        super::math::gelu_in_place(&mut p);
        res.features = self.linear(exec, &plan.proj2, &p, a, threads)?;
        Ok(res)
    }
}

/// Widen the tower tensor `name` (debug/test helper).
pub fn read_tensor(shards: &Shards, name: &str) -> Result<Vec<f32>> {
    let (f, i) = shards.get(name)?;
    Ok(widen_par(f, i))
}
