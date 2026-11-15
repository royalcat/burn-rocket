//! The Qwen3-Omni vision tower of Clef-Omni (`thinker.visual.*`, phase 8c): images in,
//! merged `[tokens, 2048]` features + three DeepStack taps out, as the thinker consumes
//! them ([`TowerOutputs`] / [`MediaFeatures::assemble`](super::media::MediaFeatures)).
//!
//! Port spec: `tools/clef_media.py::manual_vision` (plain torch, checked against the HF
//! module), reproduced here on the host in f32:
//!
//! 1. patch embed = the Conv3d(3,1152,(2,16,16)) as a linear over the 1536-value patch rows;
//! 2. plus the learned 48x48 position table resampled bilinearly (`align_corners=True`) to
//!    the frame's grid, in merge-block order, repeated per temporal group;
//! 3. 27 pre-LN blocks (LayerNorm eps 1e-6): `qkv` (bias), 2-D axial rotary (theta 1e4,
//!    rotate-half over the 72-wide head, row/col of the patch in its FRAME), full attention
//!    inside one frame (never across frames), `proj`; MLP `fc1 -> gelu(tanh) -> fc2`;
//! 4. final merger: LayerNorm(1152) per patch, then 4 consecutive patches -> Linear(4608) ->
//!    GELU(erf) -> Linear(2048); the three DeepStack mergers (after blocks 8/16/24) group 4
//!    patches FIRST and LayerNorm the 4608-wide row.
//!
//! **Weights** are stored as they are in the checkpoint: BF16 (all 351 tensors, in the BF16
//! release and in the AutoRound int8 repo alike - the AutoRound recipe quantizes only
//! `thinker.model.layers`; the tower sits in shards 6-7 there, shard 1 in the BF16 repo). They
//! stay in the mmapped shards and are widened to f32 per use, so the host holds no weight copy.
//!
//! **Compute placement** ([`TowerDevice`]): `Cpu` is plain f32 (gemm); `Npu` (aarch64 `npu`
//! builds) packs every linear as a resident fp16 weight (the path the EmbeddingGemma 2 / d3
//! towers use; native bf16 has no speed advantage on this part and would need new
//! plumbing - the exact bf16 -> f16 conversion is lossless for these weights) and runs the
//! per-frame attention through the fp16 flash-attention driver, with the 72-wide head
//! zero-padded to 96 (QK) / 80 (V) because the driver wants multiples of 32 / 16 (padding with
//! zeros changes nothing: dot products and the softmax are unchanged, the padded output
//! columns are dropped). `Emu16` is the host emulation of the NPU numerics (fp16 operands and
//! results, f32 accumulation) used to gate the fp16 path without a board.

use super::media::{DEEPSTACK_LAYERS, TowerOutputs};
use super::model::Shards;
use super::st::{Dtype, SafeFile, TensorInfo};
use crate::decision::vision::model::{patch_positions, position_taps};
use anyhow::{Context, Result, bail, ensure};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Mutex;

mod compute;
#[cfg(all(feature = "npu", target_arch = "aarch64"))]
mod npu;
mod preprocess;
#[cfg(test)]
mod tests;
mod video;

pub use compute::{Exec, TowerDevice};
#[allow(unused_imports)]
pub use preprocess::{
    IMAGE_MAX_PIXELS, IMAGE_MIN_PIXELS, MAX_SOURCE_PIXELS, PreparedImage, load_rgb,
    normalize_channel, patchify_pairs, prepare_image, prepare_value,
};
#[allow(unused_imports)]
pub use video::{
    DEFAULT_MAX_VIDEO_FRAMES, PreparedVideo, ProbedVideo, SOUNDTRACK_NOT_IMPLEMENTED, VIDEO_FPS,
    VIDEO_FRAME_MAX_PIXELS, VIDEO_MAX_PIXELS, VIDEO_MIN_PIXELS, VideoProbe, audio_has_samples,
    capped_size, decode_frames, max_video_frames, prepare_video, probe_video, sample_frames,
};

/// `thinker.visual.` tensors of the checkpoint.
pub const PREFIX: &str = "thinker.visual.";
/// LayerNorm epsilon of every tower norm.
pub const LN_EPS: f32 = 1e-6;

/// `thinker_config.vision_config` (the fields the forward uses).
#[derive(Debug, Clone, PartialEq)]
pub struct VisionConfig {
    pub depth: usize,
    pub hidden: usize,
    pub heads: usize,
    pub inter: usize,
    pub patch: usize,
    pub temporal: usize,
    pub merge: usize,
    pub out_hidden: usize,
    pub in_channels: usize,
    pub num_pos: usize,
    pub deepstack: Vec<usize>,
    pub theta: f64,
}

impl VisionConfig {
    /// The shipped Clef-Omni geometry (what `config.json` of both checkpoints says).
    pub fn shipped() -> Self {
        Self {
            depth: 27,
            hidden: 1152,
            heads: 16,
            inter: 4304,
            patch: 16,
            temporal: 2,
            merge: 2,
            out_hidden: 2048,
            in_channels: 3,
            num_pos: 2304,
            deepstack: vec![8, 16, 24],
            theta: 10000.0,
        }
    }

    /// Read `thinker_config.vision_config` of `dir/config.json`.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let p = dir.join("config.json");
        let v: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&p).with_context(|| format!("read {}", p.display()))?,
        )?;
        let vc = &v["thinker_config"]["vision_config"];
        ensure!(
            vc.is_object(),
            "config.json has no thinker_config.vision_config"
        );
        let get = |k: &str| -> Result<usize> {
            vc[k]
                .as_u64()
                .map(|v| v as usize)
                .with_context(|| format!("vision_config.{k} missing"))
        };
        let deepstack = vc["deepstack_visual_indexes"]
            .as_array()
            .context("vision_config.deepstack_visual_indexes missing")?
            .iter()
            .map(|d| d.as_u64().map(|d| d as usize))
            .collect::<Option<Vec<_>>>()
            .context("deepstack_visual_indexes must be integers")?;
        let cfg = Self {
            depth: get("depth")?,
            hidden: get("hidden_size")?,
            heads: get("num_heads")?,
            inter: get("intermediate_size")?,
            patch: get("patch_size")?,
            temporal: get("temporal_patch_size")?,
            merge: get("spatial_merge_size")?,
            out_hidden: get("out_hidden_size")?,
            in_channels: vc["in_channels"].as_u64().unwrap_or(3) as usize,
            num_pos: get("num_position_embeddings").unwrap_or(2304),
            deepstack,
            // the HF default; `rope_parameters` is not written by the shipped config
            theta: vc["rope_parameters"]["rope_theta"]
                .as_f64()
                .or_else(|| vc["rope_theta"].as_f64())
                .unwrap_or(10000.0),
        };
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.heads > 0
                && self.hidden.is_multiple_of(self.heads)
                && self.head_dim().is_multiple_of(4),
            "vision hidden {} / heads {} must give a head dim divisible by 4",
            self.hidden,
            self.heads
        );
        ensure!(
            self.deepstack.len() == DEEPSTACK_LAYERS
                && self.deepstack.iter().all(|&b| b < self.depth),
            "expected {DEEPSTACK_LAYERS} DeepStack taps inside the {} blocks, got {:?}",
            self.depth,
            self.deepstack
        );
        let side = self.pos_side();
        ensure!(
            side * side == self.num_pos,
            "the position table is not square"
        );
        ensure!(
            self.merge == 2 && self.temporal == 2,
            "this port is written for merge 2 / temporal 2"
        );
        Ok(())
    }

    pub fn head_dim(&self) -> usize {
        self.hidden / self.heads
    }

    /// Values of one patch row: `3 * temporal * patch^2` (1536).
    pub fn patch_dim(&self) -> usize {
        self.in_channels * self.temporal * self.patch * self.patch
    }

    /// `patch * merge` (32): the smart-resize factor.
    pub fn factor(&self) -> usize {
        self.patch * self.merge
    }

    /// Patches per merged token (`merge^2`).
    pub fn merge_unit(&self) -> usize {
        self.merge * self.merge
    }

    /// Side of the square learned position table (48).
    pub fn pos_side(&self) -> usize {
        (self.num_pos as f64).sqrt().round() as usize
    }
}

/// One linear layer: where its weight sits, its f32 bias, and a dense id the NPU placement
/// packs under.
#[derive(Debug, Clone)]
pub struct Lin {
    pub name: String,
    pub k: usize,
    pub n: usize,
    pub bias: Vec<f32>,
    pub slot: usize,
}

#[derive(Debug, Clone)]
struct Block {
    n1w: Vec<f32>,
    n1b: Vec<f32>,
    qkv: Lin,
    proj: Lin,
    n2w: Vec<f32>,
    n2b: Vec<f32>,
    fc1: Lin,
    fc2: Lin,
}

#[derive(Debug, Clone)]
struct Merger {
    ln_w: Vec<f32>,
    ln_b: Vec<f32>,
    fc1: Lin,
    fc2: Lin,
}

/// Everything small and the linear table of the tower (the big weights stay in the shards).
pub struct Plan {
    pub cfg: VisionConfig,
    patch: Lin,
    pos: Vec<f32>,
    blocks: Vec<Block>,
    merger: Merger,
    deep: Vec<Merger>,
    pub lins: usize,
}

impl Plan {
    /// Every linear of the tower in slot order.
    pub fn linears(&self) -> Vec<&Lin> {
        let mut v = vec![&self.patch];
        for b in &self.blocks {
            v.extend([&b.qkv, &b.proj, &b.fc1, &b.fc2]);
        }
        for m in std::iter::once(&self.merger).chain(&self.deep) {
            v.extend([&m.fc1, &m.fc2]);
        }
        v
    }
}

fn tensor<'a>(shards: &'a Shards, name: &str) -> Result<(&'a SafeFile, &'a TensorInfo)> {
    shards.get(name)
}

fn vec_of(shards: &Shards, name: &str, n: usize) -> Result<Vec<f32>> {
    shards.f32(name, &[n])
}

impl Plan {
    /// Index the tower: validate every tensor's dtype/shape against `cfg` and read the small
    /// ones (norms, biases, the position table).
    pub fn build(shards: &Shards, cfg: &VisionConfig) -> Result<Self> {
        let (h, inter, out) = (cfg.hidden, cfg.inter, cfg.out_hidden);
        let mu = cfg.merge_unit() * h;
        let mut slot = 0usize;
        let mut lin =
            |wname: &str, bname: &str, n: usize, k: usize, shape: &[usize]| -> Result<Lin> {
                let (_, info) = tensor(shards, &format!("{PREFIX}{wname}"))?;
                ensure!(
                    info.shape == shape
                        && matches!(info.dtype, Dtype::Bf16 | Dtype::F16 | Dtype::F32),
                    "{PREFIX}{wname}: {:?} {:?}, expected a float tensor of shape {shape:?}",
                    info.dtype,
                    info.shape
                );
                let l = Lin {
                    name: format!("{PREFIX}{wname}"),
                    k,
                    n,
                    bias: vec_of(shards, &format!("{PREFIX}{bname}"), n)?,
                    slot,
                };
                slot += 1;
                Ok(l)
            };
        let patch = lin(
            "patch_embed.proj.weight",
            "patch_embed.proj.bias",
            h,
            cfg.patch_dim(),
            &[h, cfg.in_channels, cfg.temporal, cfg.patch, cfg.patch],
        )?;
        let pos = shards.f32(&format!("{PREFIX}pos_embed.weight"), &[cfg.num_pos, h])?;
        let mut blocks = Vec::with_capacity(cfg.depth);
        for b in 0..cfg.depth {
            let p = format!("blocks.{b}.");
            blocks.push(Block {
                n1w: vec_of(shards, &format!("{PREFIX}{p}norm1.weight"), h)?,
                n1b: vec_of(shards, &format!("{PREFIX}{p}norm1.bias"), h)?,
                qkv: lin(
                    &format!("{p}attn.qkv.weight"),
                    &format!("{p}attn.qkv.bias"),
                    3 * h,
                    h,
                    &[3 * h, h],
                )?,
                proj: lin(
                    &format!("{p}attn.proj.weight"),
                    &format!("{p}attn.proj.bias"),
                    h,
                    h,
                    &[h, h],
                )?,
                n2w: vec_of(shards, &format!("{PREFIX}{p}norm2.weight"), h)?,
                n2b: vec_of(shards, &format!("{PREFIX}{p}norm2.bias"), h)?,
                fc1: lin(
                    &format!("{p}mlp.linear_fc1.weight"),
                    &format!("{p}mlp.linear_fc1.bias"),
                    inter,
                    h,
                    &[inter, h],
                )?,
                fc2: lin(
                    &format!("{p}mlp.linear_fc2.weight"),
                    &format!("{p}mlp.linear_fc2.bias"),
                    h,
                    inter,
                    &[h, inter],
                )?,
            });
        }
        let mut merger = |pre: &str, ln: usize| -> Result<Merger> {
            Ok(Merger {
                ln_w: vec_of(shards, &format!("{PREFIX}{pre}.ln_q.weight"), ln)?,
                ln_b: vec_of(shards, &format!("{PREFIX}{pre}.ln_q.bias"), ln)?,
                fc1: lin(
                    &format!("{pre}.mlp.0.weight"),
                    &format!("{pre}.mlp.0.bias"),
                    mu,
                    mu,
                    &[mu, mu],
                )?,
                fc2: lin(
                    &format!("{pre}.mlp.2.weight"),
                    &format!("{pre}.mlp.2.bias"),
                    out,
                    mu,
                    &[out, mu],
                )?,
            })
        };
        let merger_final = merger("merger", h)?;
        let deep = (0..DEEPSTACK_LAYERS)
            .map(|j| merger(&format!("merger_list.{j}"), mu))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            cfg: cfg.clone(),
            patch,
            pos,
            blocks,
            merger: merger_final,
            deep,
            lins: slot,
        })
    }
}

/// What a tower run returns.
#[derive(Debug, Default, Clone)]
pub struct VisionOut {
    /// `[P/4, 2048]` merged tokens (what replaces the `<|image_pad|>` embeddings).
    pub merged: Vec<f32>,
    /// DeepStack taps (after blocks 8/16/24), each `[P/4, 2048]`.
    pub deepstack: [Vec<f32>; DEEPSTACK_LAYERS],
    /// Block outputs (pre-merge, `[P, 1152]`) the caller asked to keep.
    pub taps: BTreeMap<usize, Vec<f32>>,
}

/// Widen a stored float tensor to f32 in parallel (bf16 is a shift).
pub(crate) fn widen_par(file: &SafeFile, info: &TensorInfo) -> Vec<f32> {
    let bytes = file.bytes(info);
    match info.dtype {
        Dtype::Bf16 => {
            let mut out = vec![0f32; info.numel()];
            out.par_chunks_mut(1 << 16)
                .zip(bytes.par_chunks(2 << 16))
                .for_each(|(o, b)| {
                    for (d, c) in o.iter_mut().zip(b.as_chunks::<2>().0) {
                        *d = f32::from_bits(u32::from(u16::from_le_bytes(*c)) << 16);
                    }
                });
            out
        }
        _ => file.to_f32(info),
    }
}

/// `x * 0.5 (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))` (`gelu_pytorch_tanh`).
#[inline]
pub fn gelu_tanh(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044_715 * x * x * x)).tanh())
}

/// `[patches, 36]` cos/sin tables of one frame: `[cos(row f) (18) | cos(col f) (18)]`.
fn rope_tables(grid_h: usize, grid_w: usize, cfg: &VisionConfig) -> (Vec<f32>, Vec<f32>) {
    let spatial = cfg.head_dim() / 2;
    let nf = spatial / 2;
    let inv: Vec<f32> = (0..nf)
        .map(|i| 1.0 / (cfg.theta as f32).powf((2 * i) as f32 / spatial as f32))
        .collect();
    let pos = patch_positions(grid_h, grid_w, cfg.merge);
    let (mut cos, mut sin) = (
        Vec::with_capacity(pos.len() * spatial),
        Vec::with_capacity(pos.len() * spatial),
    );
    for &(r, c) in &pos {
        for axis in [r, c] {
            for &f in &inv {
                let a = axis as f32 * f;
                cos.push(a.cos());
                sin.push(a.sin());
            }
        }
    }
    (cos, sin)
}

/// Rotate-half rotary on `x [patches, heads*hd]` in place; `cos`/`sin` are the one-frame
/// tables (`patch i` uses row `i % frame_patches`).
fn apply_rope(
    x: &mut [f32],
    heads: usize,
    hd: usize,
    cos: &[f32],
    sin: &[f32],
    frame_patches: usize,
) {
    let half = hd / 2;
    x.par_chunks_mut(heads * hd)
        .enumerate()
        .for_each(|(i, row)| {
            let t = (i % frame_patches) * half;
            let (c, s) = (&cos[t..t + half], &sin[t..t + half]);
            for head in row.chunks_mut(hd) {
                for j in 0..half {
                    let (a, b) = (head[j], head[j + half]);
                    head[j] = a * c[j] - b * s[j];
                    head[j + half] = b * c[j] + a * s[j];
                }
            }
        });
}

/// Bilinear position embeddings `[patches, hidden]` of the grid, merge-block order, repeated per
/// temporal group (`F.interpolate(..., align_corners=True)` as four weighted taps).
fn position_embeddings(table: &[f32], grid: [usize; 3], cfg: &VisionConfig) -> Vec<f32> {
    let (t, gh, gw) = (grid[0], grid[1], grid[2]);
    let h = cfg.hidden;
    let fp = gh * gw;
    let (idx, wt) = position_taps(gh, gw, cfg.pos_side(), cfg.merge);
    let mut out = vec![0f32; t * fp * h];
    out.par_chunks_mut(h).enumerate().for_each(|(i, row)| {
        let p = i % fp;
        for tap in 0..4 {
            let src = &table[idx[tap][p] as usize * h..][..h];
            let w = wt[tap][p];
            if tap == 0 {
                for (o, &s) in row.iter_mut().zip(src) {
                    *o = s * w;
                }
            } else {
                for (o, &s) in row.iter_mut().zip(src) {
                    *o += s * w;
                }
            }
        }
    });
    out
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

/// The vision tower: the index of its tensors, the mmapped shards and the compute placement.
pub struct VisionTower {
    plan: Plan,
    shards: Shards,
    exec: Box<dyn Exec>,
    /// The CPU fallback of an fp16 placement whose range overflowed.
    cpu: compute::CpuExec,
    device: TowerDevice,
    /// Requests that overflowed the fp16 range and reran on the CPU.
    fallbacks: Mutex<usize>,
}

impl std::fmt::Debug for VisionTower {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VisionTower")
            .field("device", &self.device)
            .field("depth", &self.plan.cfg.depth)
            .finish_non_exhaustive()
    }
}

impl VisionTower {
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
            .is_some_and(|m| m.contains_key(&format!("{PREFIX}patch_embed.proj.weight")))
    }

    /// Open the tower of the checkpoint in `dir` (BF16 or AutoRound: the shards that hold
    /// `thinker.visual.*`), on `device`. `threads` sizes the NPU contexts.
    pub fn open(dir: &Path, device: TowerDevice, threads: usize) -> Result<Self> {
        let cfg = VisionConfig::from_dir(dir)?;
        let shards = Shards::open_with(dir, |k| k.starts_with(PREFIX))
            .context("open the shards that hold the vision tower")?;
        let plan = Plan::build(&shards, &cfg).context("index the vision tower")?;
        Self::from_parts(plan, shards, device, threads)
    }

    pub fn from_parts(
        plan: Plan,
        shards: Shards,
        device: TowerDevice,
        threads: usize,
    ) -> Result<Self> {
        let exec = compute::make_exec(device, &plan, &shards, threads)?;
        Ok(Self {
            plan,
            shards,
            exec,
            cpu: compute::CpuExec,
            device,
            fallbacks: Mutex::new(0),
        })
    }

    pub fn config(&self) -> &VisionConfig {
        &self.plan.cfg
    }

    pub fn device(&self) -> TowerDevice {
        self.device
    }

    pub fn label(&self) -> String {
        self.exec.label()
    }

    /// Encode images (record order) into the thinker's [`TowerOutputs`] rows: merged tokens and
    /// the three DeepStack taps, each image after the other. Each image is its own tower run.
    pub fn encode_images(&self, images: &[PreparedImage], threads: usize) -> Result<TowerOutputs> {
        let mut out = TowerOutputs::default();
        for img in images {
            let r = self.forward(&img.pixels, img.grid, threads, &[])?;
            out.image.extend_from_slice(&r.merged);
            for (dst, src) in out.image_deepstack.iter_mut().zip(&r.deepstack) {
                dst.extend_from_slice(src);
            }
        }
        Ok(out)
    }

    /// Encode a record's images and videos into the thinker's [`TowerOutputs`]: images first
    /// (their rows and DeepStack taps in `image*`), then videos (`video*`), each item its own
    /// tower run in record order (HF runs `get_image_features` and `get_video_features`
    /// apart; the tower attends per temporal group, so a video is `t` independent segments).
    pub fn encode_media(
        &self,
        images: &[PreparedImage],
        videos: &[PreparedVideo],
        threads: usize,
    ) -> Result<TowerOutputs> {
        let mut out = self.encode_images(images, threads)?;
        for v in videos {
            let r = self.forward(&v.pixels, v.grid, threads, &[])?;
            out.video.extend_from_slice(&r.merged);
            for (dst, src) in out.video_deepstack.iter_mut().zip(&r.deepstack) {
                dst.extend_from_slice(src);
            }
        }
        Ok(out)
    }

    /// One tower run over `[P, 1536]` pixel rows of a `[t, h, w]` patch grid; `taps` lists block
    /// outputs to keep (pre-merge). An fp16 range overflow of the placement reruns the whole
    /// tower on the CPU in f32.
    pub fn forward(
        &self,
        pixels: &[f32],
        grid: [usize; 3],
        threads: usize,
        taps: &[usize],
    ) -> Result<VisionOut> {
        match self.run(&*self.exec, pixels, grid, threads, taps) {
            Err(e) if e.downcast_ref::<compute::Fp16Range>().is_some() => {
                eprintln!("clef: vision tower: {e:#}; rerunning this image in f32 on the CPU");
                *self.fallbacks.lock().unwrap_or_else(|p| p.into_inner()) += 1;
                self.run(&self.cpu, pixels, grid, threads, taps)
            }
            r => r,
        }
    }

    /// Debug aid (`ROCKET_CLEF_TOWER_CHECK=1`): run `img` through the placement and through the
    /// f32 CPU reference and describe how far the placement's merged tokens and DeepStack taps
    /// are (cosine, max |d| relative to the peak). This is how the NPU numerics are checked on
    /// the board without a separate probe binary.
    pub fn check_against_cpu(&self, img: &PreparedImage, threads: usize) -> Result<String> {
        self.check_pixels(&img.pixels, img.grid, threads)
    }

    /// [`VisionTower::check_against_cpu`] for any `[P, 1536]` rows of a `[t, h, w]` grid (a
    /// video's temporal groups included).
    pub fn check_pixels(&self, pixels: &[f32], grid: [usize; 3], threads: usize) -> Result<String> {
        let a = self.forward(pixels, grid, threads, &[])?;
        let b = self.run(&self.cpu, pixels, grid, threads, &[])?;
        let mut parts = Vec::new();
        for (name, x, y) in std::iter::once(("merged", &a.merged, &b.merged)).chain(
            ["deepstack0", "deepstack1", "deepstack2"]
                .into_iter()
                .zip(&a.deepstack)
                .zip(&b.deepstack)
                .map(|((n, x), y)| (n, x, y)),
        ) {
            let (cos, rel) = row_agreement(x, y);
            parts.push(format!("{name} cos {cos:.7} rel-to-peak {rel:.2e}"));
        }
        Ok(parts.join(", "))
    }

    /// Number of runs that fell back to the CPU after an fp16 overflow.
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
        y.par_chunks_mut(lin.n).for_each(|r| {
            for (v, b) in r.iter_mut().zip(&lin.bias) {
                *v += b;
            }
        });
        Ok(y)
    }

    fn merge_mlp(
        &self,
        exec: &dyn Exec,
        m: &Merger,
        x: &[f32],
        rows: usize,
        threads: usize,
    ) -> Result<Vec<f32>> {
        let mut h = self.linear(exec, &m.fc1, x, rows, threads)?;
        super::math::gelu_in_place(&mut h);
        self.linear(exec, &m.fc2, &h, rows, threads)
    }

    fn run(
        &self,
        exec: &dyn Exec,
        pixels: &[f32],
        grid: [usize; 3],
        threads: usize,
        taps: &[usize],
    ) -> Result<VisionOut> {
        let plan = &self.plan;
        let cfg = &plan.cfg;
        let (t, gh, gw) = (grid[0], grid[1], grid[2]);
        let (h, heads, hd) = (cfg.hidden, cfg.heads, cfg.head_dim());
        ensure!(
            t > 0 && gh > 0 && gw > 0 && gh % cfg.merge == 0 && gw % cfg.merge == 0,
            "bad patch grid {grid:?}"
        );
        let (fp, p) = (gh * gw, t * gh * gw);
        ensure!(
            pixels.len() == p * cfg.patch_dim(),
            "{} pixel values for a {grid:?} grid of {}-value patches",
            pixels.len(),
            cfg.patch_dim()
        );
        // attention segments: one per frame (temporal group)
        let segs: Vec<(usize, usize)> = (0..t).map(|i| (i * fp, (i + 1) * fp)).collect();

        let mut x = self.linear(exec, &plan.patch, pixels, p, threads)?;
        add_in_place(&mut x, &position_embeddings(&plan.pos, grid, cfg));
        let (cos, sin) = rope_tables(gh, gw, cfg);

        let mut out = VisionOut::default();
        for (bi, b) in plan.blocks.iter().enumerate() {
            let n1 = super::math::layer_norm(&x, p, h, &b.n1w, &b.n1b, LN_EPS);
            let qkv = self.linear(exec, &b.qkv, &n1, p, threads)?;
            drop(n1);
            let (mut q, mut k, mut v) = (vec![0f32; p * h], vec![0f32; p * h], vec![0f32; p * h]);
            for (i, row) in qkv.chunks(3 * h).enumerate() {
                q[i * h..(i + 1) * h].copy_from_slice(&row[..h]);
                k[i * h..(i + 1) * h].copy_from_slice(&row[h..2 * h]);
                v[i * h..(i + 1) * h].copy_from_slice(&row[2 * h..]);
            }
            drop(qkv);
            apply_rope(&mut q, heads, hd, &cos, &sin, fp);
            apply_rope(&mut k, heads, hd, &cos, &sin, fp);
            let ctx = exec.attention(&q, &k, &v, &segs, heads, hd, threads)?;
            drop((q, k, v));
            let o = self.linear(exec, &b.proj, &ctx, p, threads)?;
            add_in_place(&mut x, &o);
            drop((ctx, o));

            let n2 = super::math::layer_norm(&x, p, h, &b.n2w, &b.n2b, LN_EPS);
            let mut f = self.linear(exec, &b.fc1, &n2, p, threads)?;
            drop(n2);
            f.par_chunks_mut(4096)
                .for_each(|c| c.iter_mut().for_each(|v| *v = gelu_tanh(*v)));
            let m = self.linear(exec, &b.fc2, &f, p, threads)?;
            add_in_place(&mut x, &m);
            drop((f, m));

            if let Some(j) = cfg.deepstack.iter().position(|&d| d == bi) {
                // postshuffle merger: group 4 patches, THEN LayerNorm the 4608-wide row
                let mg = &plan.deep[j];
                let mu = cfg.merge_unit() * h;
                let ln = super::math::layer_norm(
                    &x,
                    p / cfg.merge_unit(),
                    mu,
                    &mg.ln_w,
                    &mg.ln_b,
                    LN_EPS,
                );
                out.deepstack[j] = self.merge_mlp(exec, mg, &ln, p / cfg.merge_unit(), threads)?;
            }
            if taps.contains(&bi) {
                out.taps.insert(bi, x.clone());
            }
        }
        // final merger: LayerNorm each 1152-wide patch, THEN group 4 rows
        let ln = super::math::layer_norm(&x, p, h, &plan.merger.ln_w, &plan.merger.ln_b, LN_EPS);
        out.merged = self.merge_mlp(exec, &plan.merger, &ln, p / cfg.merge_unit(), threads)?;
        Ok(out)
    }
}

/// Reject a tower of the wrong size early (`encode` guards).
pub fn ensure_supported(cfg: &VisionConfig) -> Result<()> {
    if cfg.in_channels != 3 {
        bail!("only RGB towers are supported");
    }
    Ok(())
}
