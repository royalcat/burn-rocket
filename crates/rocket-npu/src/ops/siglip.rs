//! The SigLIP-B/16 vision encoder (RK3588) — the M9 capstone, part 2.
//!
//! Port of `rocket_siglip_encoder.c` + `rocket_siglip.h` (the SmolVLM-256M front-end,
//! `google/siglip-base-patch16-*`). The graph, from the C header:
//!
//! ```text
//! patch-embed   Conv2d(ic->d, k=patch, s=patch)  ==  im2col + matmul (host gather, NPU GEMM)
//! + pos-embed   add the learned [L,d] position table (host glue)
//! 12 x block    pre-norm (LN->MHA->res, LN->GELU-MLP->res): the pre-norm encoder block
//! post-LN       a final LayerNorm over the d axis
//! ```
//!
//! The big GEMMs (q/k/v/o, fc1, fc2, the patch projection), every softmax, both
//! LayerNorms per block, the residual adds and the MLP GELU run on the NPU in the
//! one-shot path ([`siglip_encode`], the C `rocket_siglip_encode(fd >= 0)`, assembled
//! from [`super::encoder::encoder_block`]); the only host work there is the `O(L·d)`
//! glue (the im2col gather, the patch-bias + position adds). The resident path
//! ([`SiglipContext`], the C `rocket_siglip_encode_ctx`) keeps the projections
//! prepacked and runs the LayerNorm / bias adds / GELU / residual on the host, where
//! the data is already de-tiled.
//!
//! # GELU note
//!
//! The encoder block's device path uses the **exact-erf** 2-pass GELU (`x·Φ(x)`);
//! SigLIP trains with `gelu_pytorch_tanh`. The two differ by ~1e-3, well under the
//! fidelity target — no tanh LUT is needed unless a cosine gate misses. The C's
//! resident path uses the tanh form on the host (an exact match to SigLIP's training
//! activation); so does this port's, and [`SiglipVariant::act_kind`] selects
//! quick-GELU (CLIP) or exact-erf (DINOv2) for a foreign checkpoint.
//!
//! # Variants (`rocket_siglip_model`'s "plain-ViT knobs")
//!
//! The blob loader leaves [`SiglipVariant`] at the SigLIP defaults (no prefix token, no
//! pre-encoder LayerNorm, gelu-tanh, single exit), so a v1 blob loads unchanged. The
//! C also lets a frontend set them for a CLIP-shaped or DPT-style checkpoint; this port
//! exposes them as a plain struct ([`SiglipVariant::default`] = SigLIP) and honours them
//! in the resident path, which is the only place the C honours them too — the one-shot
//! [`siglip_encode`] refuses a non-SigLIP configuration with the C's `-3`.
//!
//! `patch_dim_pad` is the patch projection's matmul K: the fp16 matmul needs `K % 32`
//! and `ic*kh*kw` is only sometimes a multiple of 32 (768 for patch 16, but 588 for
//! patch 14). Lay `patch_w` out at that row stride with a zeroed tail and the im2col
//! zero-pads to match; `0` means "no padding needed", which is what the blob loader
//! sets.
//!
//! # Blob format (`tools/siglip_extract.py`)
//!
//! Little-endian, a 16 x int32 header (64 bytes) then fp16 weights in declaration
//! order: `patch_W [d][patch_dim]`, `patch_b [d]`, `pos [L][d]`, per layer
//! `ln1_g ln1_b Wq bq Wk bk Wv bv Wo bo ln2_g ln2_b Wf1 bf1 Wf2 bf2`, then
//! `post_g post_b`. All linears stay row-major `[out,in]` (a PyTorch `nn.Linear`), so
//! no transpose. Loader errors keep the C codes: `-1` open, `-2` size, `-3`
//! magic/version, `-4` inconsistent dims / blob size.

use std::path::Path;
use std::sync::OnceLock;

use half::f16;

use crate::Device;
use crate::attn::Attention;
use crate::error::{Error, Result};
use crate::matmul::run::one_shot;
use crate::matmul::{Context, Weight};
use crate::ops::encoder::{EncoderParams, encoder_block, encoder_block_ref};
use crate::ops::norm::{layernorm, layernorm_ref};

use super::sam::{Gen, add_bias, patchify, residual};

/// `SIGLIP_MAGIC` ("SGLP", little-endian).
pub const SIGLIP_MAGIC: u32 = 0x5347_4C50;
/// `SIGLIP_VERSION`.
pub const SIGLIP_VERSION: i32 = 1;
/// `ROCKET_SIGLIP_MAX_TAPS` (DPT heads tap 4; headroom for wider variants).
pub const SIGLIP_MAX_TAPS: usize = 8;

// ── Model ─────────────────────────────────────────────────────────────────────

/// The blob-header geometry (`rocket_siglip_model`'s scalar fields).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SiglipGeometry {
    /// Hidden size (768).
    pub d: usize,
    /// Encoder layers (12).
    pub n_layers: usize,
    /// Attention heads (12).
    pub n_head: usize,
    /// MLP intermediate size (3072).
    pub d_ff: usize,
    /// Number of patches = `side*side` (1024).
    pub l: usize,
    /// `ic*kh*kw` (768).
    pub patch_dim: usize,
    /// Input channels (3).
    pub ic: usize,
    /// Patch kernel height / width (16 / 16).
    pub kh: usize,
    pub kw: usize,
    /// Patch stride (16).
    pub stride: usize,
    /// Input image side (512).
    pub image_size: usize,
    /// `layer_norm_eps` (1e-6).
    pub eps: f32,
}

impl SiglipGeometry {
    /// The small NPU-legal geometry the self-test fixtures use: `d=64`, 2 layers,
    /// 2 heads (`dh=32`), `d_ff=64`, an 8x8 patch grid (`L=64`), 8x8 patches on a 64px
    /// image, `ic=3` (`patch_dim=192`). Every matmul meets the fp16 align; `patch_dim`
    /// is already a multiple of 32, so no `patch_dim_pad` is needed (the padded form is
    /// covered by its own unit test).
    pub fn fixture() -> SiglipGeometry {
        SiglipGeometry {
            d: 64,
            n_layers: 2,
            n_head: 2,
            d_ff: 64,
            l: 64,
            patch_dim: 192,
            ic: 3,
            kh: 8,
            kw: 8,
            stride: 8,
            image_size: 64,
            eps: 1e-6,
        }
    }

    /// The per-layer fp16 element count (`layer_stride_elems` in the C loader).
    pub fn layer_stride(&self) -> usize {
        let (d, dff) = (self.d, self.d_ff);
        2 * d + 4 * (d * d + d) + 2 * d + (dff * d + dff) + (d * dff + d)
    }

    /// The blob-header geometry contract the C loader enforces (`-4`).
    fn check(&self) -> bool {
        self.d > 0
            && self.n_layers > 0
            && self.n_head > 0
            && self.d_ff > 0
            && self.l > 0
            && self.patch_dim > 0
            && self.d.is_multiple_of(self.n_head)
    }
}

/// The plain-ViT variant knobs the C's frontend path sets (SigLIP leaves them at the
/// defaults, so a v1 blob is unaffected).
#[derive(Debug, Clone, PartialEq)]
pub struct SiglipVariant {
    /// The patch projection's matmul K (`patch_dim` rounded up to `%32`), `0` = no pad.
    pub patch_dim_pad: usize,
    /// Prepended tokens (CLIP `cls` = 1); 0 for SigLIP.
    pub n_prefix: usize,
    /// `[n_prefix][d]` prefix (cls) embeddings.
    pub prefix: Vec<f16>,
    /// Pre-encoder LayerNorm (CLIP `pre_layrnorm`), applied to the token sequence.
    pub pre_g: Option<Vec<f16>>,
    pub pre_b: Option<Vec<f16>>,
    /// The MLP activation: gelu-tanh (SigLIP), quick-GELU (CLIP) or exact erf (DINOv2).
    pub act_kind: SiglipAct,
    /// Multi-tap exit count (a DPT-style dense head consumes several layer outputs).
    pub taps: Vec<usize>,
}

impl Default for SiglipVariant {
    fn default() -> SiglipVariant {
        SiglipVariant {
            patch_dim_pad: 0,
            n_prefix: 0,
            prefix: Vec::new(),
            pre_g: None,
            pre_b: None,
            act_kind: SiglipAct::GeluTanh,
            taps: Vec::new(),
        }
    }
}

/// The MLP activation selectors of the C's `act_kind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SiglipAct {
    /// `gelu_pytorch_tanh`: `0.5x(1+tanh(√(2/π)(x+0.044715x³)))` — SigLIP's training form.
    GeluTanh = 0,
    /// CLIP's quick-GELU: `x·sigmoid(1.702x)`.
    QuickGelu = 1,
    /// The exact erf GELU `x·Φ(x)` (DINOv2).
    ErfGelu = 2,
}

impl SiglipAct {
    /// The C `act_kind` code.
    pub fn code(self) -> i32 {
        match self {
            SiglipAct::GeluTanh => 0,
            SiglipAct::QuickGelu => 1,
            SiglipAct::ErfGelu => 2,
        }
    }
}

/// One encoder layer's weights (row-major `[out,in]` linears, `[d]` affines/biases).
#[derive(Debug, Clone)]
pub struct SiglipLayerWeights {
    /// LayerNorm 1 gamma / beta `[d]`.
    pub ln1_g: Vec<f16>,
    pub ln1_b: Vec<f16>,
    /// q / k / v / o projections `[d,d]` with their `[d]` biases.
    pub wq: Vec<f16>,
    pub bq: Vec<f16>,
    pub wk: Vec<f16>,
    pub bk: Vec<f16>,
    pub wv: Vec<f16>,
    pub bv: Vec<f16>,
    pub wo: Vec<f16>,
    pub bo: Vec<f16>,
    /// LayerNorm 2 gamma / beta `[d]`.
    pub ln2_g: Vec<f16>,
    pub ln2_b: Vec<f16>,
    /// fc1 `[d_ff,d]` + bias `[d_ff]`, fc2 `[d,d_ff]` + bias `[d]`.
    pub wf1: Vec<f16>,
    pub bf1: Vec<f16>,
    pub wf2: Vec<f16>,
    pub bf2: Vec<f16>,
}

/// A loaded SigLIP vision encoder: geometry, variant knobs and owned fp16 tensors.
#[derive(Debug, Clone)]
pub struct SiglipModel {
    /// The blob-header geometry.
    pub geom: SiglipGeometry,
    /// The plain-ViT knobs (`default` = SigLIP).
    pub variant: SiglipVariant,
    /// Patch projection `[d][patch_dim]`, laid out at `variant.patch_dim_pad` stride.
    pub patch_w: Vec<f16>,
    /// Patch bias `[d]`.
    pub patch_b: Vec<f16>,
    /// Position table `[ntok][d]` (`ntok = L + n_prefix`).
    pub pos: Vec<f16>,
    /// Per-layer weights, in blob order.
    pub layers: Vec<SiglipLayerWeights>,
    /// Final LayerNorm weight (`None` = CLIP's raw last residual).
    pub post_g: Option<Vec<f16>>,
    /// Final LayerNorm bias.
    pub post_b: Option<Vec<f16>>,
}

impl SiglipModel {
    /// Build a model from owned tensors (the "frontend fills the struct" path), with the
    /// same consistency checks the blob loader applies.
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        geom: SiglipGeometry,
        variant: SiglipVariant,
        patch_w: Vec<f16>,
        patch_b: Vec<f16>,
        pos: Vec<f16>,
        layers: Vec<SiglipLayerWeights>,
        post_g: Option<Vec<f16>>,
        post_b: Option<Vec<f16>>,
    ) -> Result<SiglipModel> {
        let m = SiglipModel {
            geom,
            variant,
            patch_w,
            patch_b,
            pos,
            layers,
            post_g,
            post_b,
        };
        m.validate("rocket_siglip_load")?;
        Ok(m)
    }

    /// A deterministic pseudo-random model of `geom` (an LCG, not a checkpoint).
    ///
    /// The self-test / A-B fixture: [`Self::to_blob`]-able, so the C library can encode
    /// the same weights and the two host references can be compared bit for bit. The
    /// variant stays at the plain SigLIP defaults.
    pub fn synthetic(geom: SiglipGeometry, seed: u32) -> SiglipModel {
        let d = geom.d;
        let mut r = Gen::new(seed);
        let mut layers = Vec::new();
        for _ in 0..geom.n_layers {
            layers.push(SiglipLayerWeights {
                ln1_g: r.g(d),
                ln1_b: r.b(d),
                wq: r.w(d * d),
                bq: r.w(d),
                wk: r.w(d * d),
                bk: r.w(d),
                wv: r.w(d * d),
                bv: r.w(d),
                wo: r.w(d * d),
                bo: r.w(d),
                ln2_g: r.g(d),
                ln2_b: r.b(d),
                wf1: r.w(geom.d_ff * d),
                bf1: r.w(geom.d_ff),
                wf2: r.w(d * geom.d_ff),
                bf2: r.w(d),
            });
        }
        let (patch_w, patch_b, pos) = (r.w(d * geom.patch_dim), r.w(d), r.w(geom.l * d));
        let (post_g, post_b) = (r.g(d), r.b(d));
        SiglipModel::from_parts(
            geom,
            SiglipVariant::default(),
            patch_w,
            patch_b,
            pos,
            layers,
            Some(post_g),
            Some(post_b),
        )
        .expect("the synthetic geometry is consistent")
    }

    /// Serialize to the `tools/siglip_extract.py` blob format (the inverse of
    /// [`Self::from_blob`]); used by the self-test's `--write-blob` and the C A/B.
    ///
    /// A v1 blob lays `patch_w` out at `patch_dim`, so a model built with a
    /// [`SiglipVariant::patch_dim_pad`] must be narrowed first (the fixture never pads).
    pub fn to_blob(&self) -> Vec<u8> {
        let g = &self.geom;
        let mut b = Vec::new();
        let h: [i32; 16] = [
            SIGLIP_MAGIC as i32,
            SIGLIP_VERSION,
            g.d as i32,
            g.n_layers as i32,
            g.n_head as i32,
            g.d_ff as i32,
            g.l as i32,
            g.patch_dim as i32,
            g.ic as i32,
            g.kh as i32,
            g.kw as i32,
            g.stride as i32,
            g.image_size as i32,
            g.eps.to_bits() as i32,
            0,
            0,
        ];
        for v in h {
            b.extend_from_slice(&v.to_le_bytes());
        }
        let mut put = |t: &[f16]| {
            for v in t {
                b.extend_from_slice(&v.to_bits().to_le_bytes());
            }
        };
        let pdp = self.patch_stride();
        for row in 0..g.d {
            put(&self.patch_w[row * pdp..row * pdp + g.patch_dim]);
        }
        put(&self.patch_b);
        put(&self.pos);
        for w in &self.layers {
            for t in [
                &w.ln1_g, &w.ln1_b, &w.wq, &w.bq, &w.wk, &w.bk, &w.wv, &w.bv, &w.wo, &w.bo,
                &w.ln2_g, &w.ln2_b, &w.wf1, &w.bf1, &w.wf2, &w.bf2,
            ] {
                put(t);
            }
        }
        put(self.post_g.as_deref().unwrap_or(&[]));
        put(self.post_b.as_deref().unwrap_or(&[]));
        b
    }

    /// `rocket_siglip_load`: read and validate a weight blob from
    /// `tools/siglip_extract.py`.
    pub fn load(path: impl AsRef<Path>) -> Result<SiglipModel> {
        let bytes = std::fs::read(path.as_ref())
            .map_err(|_| Error::coded(crate::Status::Io, -1, "rocket_siglip_load"))?;
        Self::from_blob(&bytes)
    }

    /// Parse a weight blob from memory (the C `mmap`, minus the mmap).
    pub fn from_blob(bytes: &[u8]) -> Result<SiglipModel> {
        let bad_size = || Error::coded(crate::Status::Io, -2, "rocket_siglip_load");
        if bytes.len() < 64 {
            return Err(bad_size());
        }
        let h = |i: usize| -> i32 {
            let o = i * 4;
            i32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]])
        };
        if h(0) as u32 != SIGLIP_MAGIC || h(1) != SIGLIP_VERSION {
            return Err(Error::coded(
                crate::Status::Unsupported,
                -3,
                "rocket_siglip_load",
            ));
        }
        let geom = SiglipGeometry {
            d: h(2).max(0) as usize,
            n_layers: h(3).max(0) as usize,
            n_head: h(4).max(0) as usize,
            d_ff: h(5).max(0) as usize,
            l: h(6).max(0) as usize,
            patch_dim: h(7).max(0) as usize,
            ic: h(8).max(0) as usize,
            kh: h(9).max(0) as usize,
            kw: h(10).max(0) as usize,
            stride: h(11).max(0) as usize,
            image_size: h(12).max(0) as usize,
            eps: f32::from_bits(h(13) as u32),
        };
        let bad_dims = || Error::coded(crate::Status::Shape, -4, "rocket_siglip_load");
        if !geom.check() {
            return Err(bad_dims());
        }

        let (d, dff, l, n) = (geom.d, geom.d_ff, geom.l, geom.n_layers);
        let mut off = 64usize;
        let mut read = |count: usize| -> Result<Vec<f16>> {
            let end = off
                .checked_add(count * 2)
                .filter(|e| *e <= bytes.len())
                .ok_or_else(bad_dims)?;
            let mut v = Vec::with_capacity(count);
            for c in bytes[off..end].as_chunks::<2>().0 {
                v.push(f16::from_bits(u16::from_le_bytes(*c)));
            }
            off = end;
            Ok(v)
        };

        let patch_w = read(d * geom.patch_dim)?;
        let patch_b = read(d)?;
        let pos = read(l * d)?;
        let mut layers = Vec::with_capacity(n);
        for _ in 0..n {
            layers.push(SiglipLayerWeights {
                ln1_g: read(d)?,
                ln1_b: read(d)?,
                wq: read(d * d)?,
                bq: read(d)?,
                wk: read(d * d)?,
                bk: read(d)?,
                wv: read(d * d)?,
                bv: read(d)?,
                wo: read(d * d)?,
                bo: read(d)?,
                ln2_g: read(d)?,
                ln2_b: read(d)?,
                wf1: read(dff * d)?,
                bf1: read(dff)?,
                wf2: read(d * dff)?,
                bf2: read(d)?,
            });
        }
        let post_g = read(d)?;
        let post_b = read(d)?;
        // The C requires the blob to be exactly the size the dims imply.
        if off != bytes.len() {
            return Err(bad_dims());
        }
        let model = SiglipModel {
            geom,
            variant: SiglipVariant::default(),
            patch_w,
            patch_b,
            pos,
            layers,
            post_g: Some(post_g),
            post_b: Some(post_b),
        };
        model.validate("rocket_siglip_load")?;
        Ok(model)
    }

    /// Tensor-length validation (the `-4` class for a hand-built model).
    fn validate(&self, op: &'static str) -> Result<()> {
        let g = &self.geom;
        let v = &self.variant;
        let ntok = self.ntok();
        let pdp = self.patch_stride();
        if !g.check() || self.layers.len() != g.n_layers || pdp < g.patch_dim {
            return Err(Error::coded(crate::Status::Shape, -4, op));
        }
        if v.n_prefix > 0 && v.prefix.len() != v.n_prefix * g.d {
            return Err(Error::coded(crate::Status::Shape, -4, op));
        }
        if v.pre_g.as_ref().is_some_and(|t| t.len() != g.d)
            || v.pre_b.as_ref().is_some_and(|t| t.len() != g.d)
        {
            return Err(Error::coded(crate::Status::Shape, -4, op));
        }
        // Multi-tap exits: at most ROCKET_SIGLIP_MAX_TAPS, ascending and in range.
        if v.taps.len() > SIGLIP_MAX_TAPS
            || v.taps
                .iter()
                .enumerate()
                .any(|(t, &layer)| layer >= g.n_layers || (t > 0 && layer <= v.taps[t - 1]))
        {
            return Err(Error::coded(crate::Status::Shape, -4, op));
        }
        let mut ok = self.patch_w.len() == g.d * pdp
            && self.patch_b.len() == g.d
            && self.pos.len() == ntok * g.d;
        let (d, dff) = (g.d, g.d_ff);
        for w in &self.layers {
            ok &= w.ln1_g.len() == d
                && w.ln1_b.len() == d
                && w.ln2_g.len() == d
                && w.ln2_b.len() == d
                && w.wq.len() == d * d
                && w.wk.len() == d * d
                && w.wv.len() == d * d
                && w.wo.len() == d * d
                && w.bq.len() == d
                && w.bk.len() == d
                && w.bv.len() == d
                && w.bo.len() == d
                && w.wf1.len() == dff * d
                && w.bf1.len() == dff
                && w.wf2.len() == d * dff
                && w.bf2.len() == d;
        }
        ok &= self.post_g.as_ref().is_none_or(|t| t.len() == d)
            && self.post_b.as_ref().is_none_or(|t| t.len() == d);
        if !ok {
            return Err(Error::coded(crate::Status::Shape, -4, op));
        }
        Ok(())
    }

    /// The token-sequence length `ntok = L + n_prefix`.
    pub fn ntok(&self) -> usize {
        self.geom.l + self.variant.n_prefix
    }

    /// The patch projection's matmul K (`patch_dim_pad` or `patch_dim`).
    pub fn patch_stride(&self) -> usize {
        if self.variant.patch_dim_pad == 0 {
            self.geom.patch_dim
        } else {
            self.variant.patch_dim_pad
        }
    }

    /// The token-projection row count padded to `M % 4 == 0`.
    pub fn mp(&self) -> usize {
        (self.ntok() + 3) & !3
    }

    /// The patch-projection row count padded to `M % 4 == 0`.
    pub fn lp(&self) -> usize {
        (self.geom.l + 3) & !3
    }

    /// Whether this model is the plain SigLIP configuration the one-shot path handles
    /// (the C's `-3` test).
    pub fn is_plain_siglip(&self) -> bool {
        let v = &self.variant;
        v.n_prefix == 0
            && v.prefix.is_empty()
            && v.pre_g.is_none()
            && v.act_kind == SiglipAct::GeluTanh
            && v.taps.is_empty()
            && self.post_g.is_some()
    }

    /// The `EncoderParams` of layer `l` (borrowed).
    fn layer_params(&self, l: usize) -> EncoderParams<'_> {
        let w = &self.layers[l];
        EncoderParams {
            ln1_g: &w.ln1_g,
            ln1_b: Some(&w.ln1_b),
            wq: &w.wq,
            bq: Some(&w.bq),
            wk: &w.wk,
            bk: Some(&w.bk),
            wv: &w.wv,
            bv: Some(&w.bv),
            wo: &w.wo,
            bo: Some(&w.bo),
            ln2_g: &w.ln2_g,
            ln2_b: Some(&w.ln2_b),
            wf1: &w.wf1,
            bf1: Some(&w.bf1),
            wf2: &w.wf2,
            bf2: Some(&w.bf2),
        }
    }
}

// ── Host math ────────────────────────────────────────────────────────────────

/// `C[M][N] = A[M][K] · B[N][K]^T` with an **f64** accumulator and an fp16 store — the
/// C `rocket_siglip_encoder.c` `host_matmul` (the `fd < 0` patch-embed path). Note this
/// differs from SAM's f32 accumulator; each port keeps its own.
fn host_matmul_f64(m: usize, k: usize, n: usize, a: &[f16], b: &[f16], c: &mut [f16]) {
    for i in 0..m {
        let ar = &a[i * k..i * k + k];
        for j in 0..n {
            let br = &b[j * k..j * k + k];
            let mut acc = 0.0f64;
            for t in 0..k {
                acc += ar[t].to_f64() * br[t].to_f64();
            }
            c[i * n + j] = crate::ops::f16_from_f64(acc);
        }
    }
}

/// The resident path's LayerNorm: two-pass f64 mean/variance, fp16 store (the C
/// `h_layernorm` in `rocket_siglip_encoder.c`). The one-shot path uses
/// [`layernorm_ref`] instead, which is the C's `rocket_layernorm_ref_fp16`
/// (sum + sum-of-squares); the two are the same function rounded two ways.
fn layernorm_two_pass(
    m: usize,
    d: usize,
    x: &[f16],
    g: &[f16],
    b: Option<&[f16]>,
    eps: f32,
    out: &mut [f16],
) {
    let inv_h = 1.0 / d as f64;
    for row in 0..m {
        let xr = &x[row * d..row * d + d];
        let mut mean = 0.0f64;
        for v in xr {
            mean += v.to_f64();
        }
        mean *= inv_h;
        let mut var = 0.0f64;
        for v in xr {
            let t = v.to_f64() - mean;
            var += t * t;
        }
        let inv = 1.0 / (var * inv_h + f64::from(eps)).sqrt();
        let o = &mut out[row * d..row * d + d];
        for j in 0..d {
            let bv = b.map_or(0.0, |b| b[j].to_f64());
            o[j] = crate::ops::f16_from_f64((xr[j].to_f64() - mean) * inv * g[j].to_f64() + bv);
        }
    }
}

/// `gelu_pytorch_tanh`, exactly as the C `gelu1` (f32 arithmetic, one fp16 rounding).
fn gelu_tanh1(x: f16) -> f16 {
    let v = x.to_f32();
    // sqrt(2/pi), written the way the C source spells it
    let c = 0.797_884_6f32;
    let t = (c * (v + 0.044_715f32 * v * v * v)).tanh();
    f16::from_f32(0.5 * v * (1.0 + t))
}

/// CLIP's quick-GELU `x·sigmoid(1.702x)`, as the C `quickgelu1`.
fn quick_gelu1(x: f16) -> f16 {
    let v = x.to_f32();
    f16::from_f32(v / (1.0 + (-1.702f32 * v).exp()))
}

/// The exact erf GELU `x·Φ(x)` (f32, glibc `erff`), as the C `erfgelu1`.
fn erf_gelu1(x: f16) -> f16 {
    let v = x.to_f32();
    f16::from_f32(0.5 * v * (1.0 + super::sam::erff32(v * std::f32::consts::FRAC_1_SQRT_2)))
}

/// The fp16-input activation tables (all 65536 outputs fit one), built once per kind and
/// bit-exact to the scalar path — the C's `g_gelu_lut` / `g_qgelu_lut` / `g_egelu_lut`.
fn act_lut(kind: SiglipAct) -> &'static [f16; 65536] {
    static TANH: OnceLock<Box<[f16; 65536]>> = OnceLock::new();
    static QUICK: OnceLock<Box<[f16; 65536]>> = OnceLock::new();
    static ERF: OnceLock<Box<[f16; 65536]>> = OnceLock::new();
    type Lut = OnceLock<Box<[f16; 65536]>>;
    let (cell, f): (&Lut, fn(f16) -> f16) = match kind {
        SiglipAct::GeluTanh => (&TANH, gelu_tanh1),
        SiglipAct::QuickGelu => (&QUICK, quick_gelu1),
        SiglipAct::ErfGelu => (&ERF, erf_gelu1),
    };
    cell.get_or_init(|| {
        let mut t = Box::new([f16::ZERO; 65536]);
        for (u, slot) in t.iter_mut().enumerate() {
            *slot = f(f16::from_bits(u as u16));
        }
        t
    })
}

fn act_in_place(kind: SiglipAct, x: &mut [f16]) {
    let lut = act_lut(kind);
    for v in x.iter_mut() {
        *v = lut[v.to_bits() as usize];
    }
}

/// A row-wise softmax in place on fp16 scores: f32 max, `expf` of the *rounded* score
/// with the **unrounded** exponential in the denominator, then one fp16 multiply — the
/// C `sm_rows`.
fn softmax_rows(x: &mut [f16], rows: usize, n: usize) {
    for r in 0..rows {
        let row = &mut x[r * n..(r + 1) * n];
        let mut mx = f32::NEG_INFINITY;
        for v in row.iter() {
            let v = v.to_f32();
            if v > mx {
                mx = v;
            }
        }
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            let e = (v.to_f32() - mx).exp();
            *v = f16::from_f32(e);
            sum += e;
        }
        let inv = 1.0 / sum;
        for v in row.iter_mut() {
            *v = f16::from_f32(v.to_f32() * inv);
        }
    }
}

/// Where the token sequence is assembled: the patch rows get `proj + patch_b + pos`, the
/// prefix rows `prefix + pos` (the C order, so the SigLIP path stays bit-identical).
fn assemble_tokens(m: &SiglipModel, xp: &[f16], x: &mut [f16]) {
    let g = &m.geom;
    let (d, l, npre) = (g.d, g.l, m.variant.n_prefix);
    for j in 0..npre {
        let xr = &mut x[j * d..j * d + d];
        let cr = &m.variant.prefix[j * d..j * d + d];
        let pr = &m.pos[j * d..j * d + d];
        for e in 0..d {
            xr[e] = f16::from_f32(cr[e].to_f32() + pr[e].to_f32());
        }
    }
    for p in 0..l {
        let xr = &mut x[(npre + p) * d..(npre + p + 1) * d];
        let sr = &xp[p * d..p * d + d];
        let pr = &m.pos[(npre + p) * d..(npre + p + 1) * d];
        for e in 0..d {
            xr[e] = f16::from_f32(sr[e].to_f32() + m.patch_b[e].to_f32() + pr[e].to_f32());
        }
    }
}

fn check_encode(
    m: &SiglipModel,
    pixels: &[f16],
    out: &[f16],
    hidden: Option<&[f16]>,
) -> Result<()> {
    let g = &m.geom;
    let pix = g.ic * g.image_size * g.image_size;
    let ntok = m.ntok();
    let taps = m.variant.taps.len().max(1);
    let out_n = taps * ntok * g.d;
    let hid_n = (g.n_layers + 1) * ntok * g.d;
    if pixels.len() < pix || out.len() < out_n || hidden.is_some_and(|h| h.len() < hid_n) {
        return Err(
            Error::shape("rocket_siglip_encode/buffers").with_detail(format!(
                "pixels {} (need {pix}) out {} (need {out_n}) hidden (need {hid_n})",
                pixels.len(),
                out.len()
            )),
        );
    }
    Ok(())
}

// ── The one-shot path ────────────────────────────────────────────────────────

/// `rocket_siglip_encode(fd >= 0, ...)`: encode one image on the NPU, one-shot.
///
/// `pixels` is the preprocessed input `[ic][image_size][image_size]` fp16; `out`
/// receives `[ntok][d]` (the final LayerNorm applied when `post_g` is set, else the raw
/// last residual). `hidden` (when given, at least `(n_layers+1) * ntok * d`) receives
/// the intermediate hidden states, index 0 = the embeddings.
///
/// This per-call path handles **only** the SigLIP configuration
/// ([`SiglipModel::is_plain_siglip`]); a CLIP- or DPT-shaped model must use
/// [`SiglipContext`], exactly as in C (which returns `-3`).
pub fn siglip_encode(
    dev: &Device,
    m: &SiglipModel,
    pixels: &[f16],
    out: &mut [f16],
    hidden: Option<&mut [f16]>,
) -> Result<()> {
    if !m.is_plain_siglip() {
        return Err(Error::coded(
            crate::Status::Unsupported,
            -3,
            "rocket_siglip_encode",
        ));
    }
    encode_oneshot(dev, m, pixels, out, hidden)
}

fn encode_oneshot(
    dev: &Device,
    m: &SiglipModel,
    pixels: &[f16],
    out: &mut [f16],
    mut hidden: Option<&mut [f16]>,
) -> Result<()> {
    check_encode(m, pixels, out, hidden.as_deref())?;
    let g = &m.geom;
    let (d, l) = (g.d, g.l);
    let pdp = m.patch_stride();
    let ntok = m.ntok();
    let td = ntok * d;

    let mut patches = vec![f16::ZERO; l * pdp];
    patchify(
        pixels,
        g.image_size,
        g.ic,
        g.kh,
        g.kw,
        g.stride,
        g.image_size / g.stride,
        pdp,
        &mut patches,
    );
    let mut xp = vec![f16::ZERO; l * d];
    let mut xa = vec![f16::ZERO; td];
    let mut xb = vec![f16::ZERO; td];
    one_shot(dev, l, pdp, d, &patches, &m.patch_w, &mut xp)?;
    assemble_tokens(m, &xp, &mut xa);
    if let Some(hid) = hidden.as_deref_mut() {
        hid[..td].copy_from_slice(&xa);
    }
    for layer in 0..g.n_layers {
        let p = m.layer_params(layer);
        encoder_block(dev, ntok, d, g.n_head, g.d_ff, &xa, &p, g.eps, &mut xb)?;
        std::mem::swap(&mut xa, &mut xb);
        if let Some(hid) = hidden.as_deref_mut() {
            hid[(layer + 1) * td..(layer + 2) * td].copy_from_slice(&xa);
        }
    }
    match (&m.post_g, &m.post_b) {
        (Some(g_), b) => layernorm(dev, ntok, d, &xa, g_, b.as_deref(), g.eps, out),
        (None, _) => {
            out[..td].copy_from_slice(&xa);
            Ok(())
        }
    }
}

/// `rocket_siglip_encode(fd < 0, ...)`: the exact host reference.
///
/// Bit-identical to the C: the f64-accumulate patch matmul,
/// [`super::encoder::encoder_block_ref`] per layer (f64 norms and matmuls, glibc `erf`
/// for the GELU) and [`layernorm_ref`] for the final norm.
pub fn siglip_encode_ref(
    m: &SiglipModel,
    pixels: &[f16],
    out: &mut [f16],
    mut hidden: Option<&mut [f16]>,
) -> Result<()> {
    if !m.is_plain_siglip() {
        return Err(Error::coded(
            crate::Status::Unsupported,
            -3,
            "rocket_siglip_encode",
        ));
    }
    check_encode(m, pixels, out, hidden.as_deref())?;
    let g = &m.geom;
    let (d, l) = (g.d, g.l);
    let pdp = m.patch_stride();
    let ntok = m.ntok();
    let td = ntok * d;

    let mut patches = vec![f16::ZERO; l * pdp];
    patchify(
        pixels,
        g.image_size,
        g.ic,
        g.kh,
        g.kw,
        g.stride,
        g.image_size / g.stride,
        pdp,
        &mut patches,
    );
    let mut xp = vec![f16::ZERO; l * d];
    let mut xa = vec![f16::ZERO; td];
    let mut xb = vec![f16::ZERO; td];
    host_matmul_f64(l, pdp, d, &patches, &m.patch_w, &mut xp);
    assemble_tokens(m, &xp, &mut xa);
    if let Some(hid) = hidden.as_deref_mut() {
        hid[..td].copy_from_slice(&xa);
    }
    for layer in 0..g.n_layers {
        let p = m.layer_params(layer);
        encoder_block_ref(ntok, d, g.n_head, g.d_ff, &xa, &p, g.eps, &mut xb)?;
        std::mem::swap(&mut xa, &mut xb);
        if let Some(hid) = hidden.as_deref_mut() {
            hid[(layer + 1) * td..(layer + 2) * td].copy_from_slice(&xa);
        }
    }
    match (&m.post_g, &m.post_b) {
        (Some(g_), b) => {
            layernorm_ref(ntok, d, &xa, g_, b.as_deref(), g.eps, out);
            Ok(())
        }
        (None, _) => {
            out[..td].copy_from_slice(&xa);
            Ok(())
        }
    }
}

/// `rocket_siglip_load`: load a weight blob (see [`SiglipModel::load`]).
pub fn siglip_load(path: impl AsRef<Path>) -> Result<SiglipModel> {
    SiglipModel::load(path)
}

// ── The resident path ────────────────────────────────────────────────────────

/// One layer's resident projections.
struct LayerResident {
    wq: Weight,
    wk: Weight,
    wv: Weight,
    wo: Weight,
    wf1: Weight,
    wf2: Weight,
}

/// The resident (prepacked, multicore) SigLIP encoder — the latency path.
///
/// Packs the seven static GEMMs per layer (q/k/v/o, fc1, fc2) plus the patch projection
/// into resident NPU BOs once at create and reuses them; per image only the activations
/// are packed. The attention matmuls (scores, `P·V`) run through a resident streaming
/// context, softmax on the host, and the LayerNorm / GELU / residual / bias adds on the
/// host (memory-bound, and the data is already de-tiled). Bit-faithful to
/// [`siglip_encode`] within fp16.
pub struct SiglipContext<'m> {
    model: &'m SiglipModel,
    mm: Context,
    /// The multicore MHA fan-out (`None` = the single-context per-head path).
    fa: Option<Attention>,
    nthreads: usize,
    w_patch: Weight,
    layers: Vec<LayerResident>,
}

impl<'m> SiglipContext<'m> {
    /// `rocket_siglip_ctx_create`: pack every static weight and open the fan-out
    /// contexts. `nthreads` is clamped to `1..=8`.
    pub fn create(model: &'m SiglipModel, nthreads: usize) -> Result<SiglipContext<'m>> {
        model.validate("rocket_siglip_ctx_create")?;
        let g = &model.geom;
        let nthreads = nthreads.clamp(1, 8);
        let mut mm = Context::new(nthreads)?;
        let mp = model.mp();
        let lp = model.lp();
        let pdp = model.patch_stride();
        let w_patch = mm.pack_weight(lp, pdp, g.d, &model.patch_w)?;
        let mut layers = Vec::with_capacity(g.n_layers);
        for w in &model.layers {
            layers.push(LayerResident {
                wq: mm.pack_weight(mp, g.d, g.d, &w.wq)?,
                wk: mm.pack_weight(mp, g.d, g.d, &w.wk)?,
                wv: mm.pack_weight(mp, g.d, g.d, &w.wv)?,
                wo: mm.pack_weight(mp, g.d, g.d, &w.wo)?,
                wf1: mm.pack_weight(mp, g.d, g.d_ff, &w.wf1)?,
                wf2: mm.pack_weight(mp, g.d_ff, g.d, &w.wf2)?,
            });
        }
        let use_fa = nthreads > 1
            && !matches!(
                crate::knobs::string("ROCKET_SIGLIP_FA").as_deref(),
                Some("0" | "false" | "off" | "no")
            );
        let fa = if use_fa {
            Attention::new(nthreads).ok()
        } else {
            None
        };
        Ok(SiglipContext {
            model,
            mm,
            fa,
            nthreads,
            w_patch,
            layers,
        })
    }

    /// The worker count the context was created with (clamped `1..=8`).
    pub fn nthreads(&self) -> usize {
        self.nthreads
    }

    /// Whether the flash-attention fan-out is active (the C's `use_fa`).
    pub fn uses_fa(&self) -> bool {
        self.fa.is_some()
    }

    /// The model this context packs.
    pub fn model(&self) -> &'m SiglipModel {
        self.model
    }

    /// `rocket_siglip_encode_ctx`: encode one image with the resident weights.
    ///
    /// `out` receives the single normalized exit, or — with
    /// [`SiglipVariant::taps`] non-empty — the `n_taps` normalized tap blocks
    /// `[n_taps][ntok][d]`.
    pub fn encode(
        &mut self,
        pixels: &[f16],
        out: &mut [f16],
        mut hidden: Option<&mut [f16]>,
    ) -> Result<()> {
        let m = self.model;
        check_encode(m, pixels, out, hidden.as_deref())?;
        let g = &m.geom;
        let (d, dff, l, nh) = (g.d, g.d_ff, g.l, g.n_head);
        let dh = d / nh;
        let ntok = m.ntok();
        let (mp, lp, pdp) = (m.mp(), m.lp(), m.patch_stride());
        let td = ntok * d;
        let scale = 1.0 / (dh as f32).sqrt();
        let mut tap_next = 0usize;

        // Token buffers hold Mp rows (M % 4); the host elementwise touches only the
        // first ntok, and every projection runs at M = Mp with the pad rows held at
        // zero, so a pad row's output stays zero and never poisons a real token.
        let mut patches = vec![f16::ZERO; lp * pdp];
        let mut xp = vec![f16::ZERO; lp * d];
        let mut x = vec![f16::ZERO; mp * d];
        let mut ln = vec![f16::ZERO; mp * d];
        let mut q = vec![f16::ZERO; mp * d];
        let mut k = vec![f16::ZERO; mp * d];
        let mut v = vec![f16::ZERO; mp * d];
        let mut cc = vec![f16::ZERO; mp * d];
        let mut y = vec![f16::ZERO; mp * dff];

        patchify(
            pixels,
            g.image_size,
            g.ic,
            g.kh,
            g.kw,
            g.stride,
            g.image_size / g.stride,
            pdp,
            &mut patches,
        );
        self.mm
            .matmul_prepacked(lp, pdp, d, &patches, &mut xp, &self.w_patch)?;
        assemble_tokens(m, &xp[..l * d], &mut x);
        if let Some(g_) = &m.variant.pre_g {
            layernorm_two_pass(ntok, d, &x, g_, m.variant.pre_b.as_deref(), g.eps, &mut ln);
            x[..td].copy_from_slice(&ln[..td]);
        }
        if let Some(hid) = hidden.as_deref_mut() {
            hid[..td].copy_from_slice(&x[..td]);
        }

        // The attention scratch: the FA path's head-major operands, or the per-head
        // gather buffers + all heads' scores.
        let use_fa = self.fa.is_some();
        let mut qh = vec![f16::ZERO; ntok * dh];
        let mut khh = vec![f16::ZERO; ntok * dh];
        let mut vht = vec![f16::ZERO; dh * ntok];
        let mut sc = vec![f16::ZERO; if use_fa { 0 } else { nh * ntok * ntok }];
        let (mut qhm, mut khm, mut vhm, mut ohm) = if use_fa {
            (
                vec![f16::ZERO; td],
                vec![f16::ZERO; td],
                vec![f16::ZERO; td],
                vec![f16::ZERO; td],
            )
        } else {
            (Vec::new(), Vec::new(), Vec::new(), Vec::new())
        };

        for layer in 0..g.n_layers {
            let r = &self.layers[layer];
            let w = &m.layers[layer];

            // --- attention sublayer: x += Wo·MHA(LN1(x)) ---
            layernorm_two_pass(ntok, d, &x, &w.ln1_g, Some(&w.ln1_b), g.eps, &mut ln);
            self.mm.matmul_prepacked(mp, d, d, &ln, &mut q, &r.wq)?;
            self.mm.matmul_prepacked(mp, d, d, &ln, &mut k, &r.wk)?;
            self.mm.matmul_prepacked(mp, d, d, &ln, &mut v, &r.wv)?;
            add_bias(ntok, d, &mut q, &w.bq);
            add_bias(ntok, d, &mut k, &w.bk);
            add_bias(ntok, d, &mut v, &w.bv);
            if !use_fa {
                // fold 1/sqrt(dh) into q; the FA primitive applies the scale itself
                for cell in q[..td].iter_mut() {
                    *cell = f16::from_f32(cell.to_f32() * scale);
                }
            }

            if use_fa {
                for h in 0..nh {
                    let off = h * dh;
                    let (qd, kd) = (h * ntok * dh, h * ntok * dh);
                    let vd = h * dh * ntok;
                    for t in 0..ntok {
                        qhm[qd + t * dh..qd + (t + 1) * dh]
                            .copy_from_slice(&q[t * d + off..t * d + off + dh]);
                        khm[kd + t * dh..kd + (t + 1) * dh]
                            .copy_from_slice(&k[t * d + off..t * d + off + dh]);
                        for cx in 0..dh {
                            vhm[vd + cx * ntok + t] = v[t * d + off + cx];
                        }
                    }
                }
                let fa = self.fa.as_mut().expect("use_fa implies a context");
                fa.flash(
                    ntok, ntok, dh, dh, nh, nh, scale, 0.0, &qhm, &khm, &vhm, None, &mut ohm,
                )?;
                for h in 0..nh {
                    let off = h * dh;
                    for t in 0..ntok {
                        cc[t * d + off..t * d + off + dh]
                            .copy_from_slice(&ohm[h * ntok * dh + t * dh..][..dh]);
                    }
                }
            } else {
                // phase 1: every head's scores q_h·k_h^T into sc[h] (stacked by row)
                for h in 0..nh {
                    let off = h * dh;
                    for t in 0..ntok {
                        qh[t * dh..(t + 1) * dh].copy_from_slice(&q[t * d + off..t * d + off + dh]);
                        khh[t * dh..(t + 1) * dh]
                            .copy_from_slice(&k[t * d + off..t * d + off + dh]);
                    }
                    self.mm.matmul(
                        ntok,
                        dh,
                        ntok,
                        &qh[..ntok * dh],
                        &khh[..ntok * dh],
                        &mut sc[h * ntok * ntok..(h + 1) * ntok * ntok],
                    )?;
                }
                // phase 2: row-wise softmax on the host, in place (no NPU round-trip)
                softmax_rows(&mut sc, nh * ntok, ntok);
                // phase 3: every head's ctx_h = P_h·v_h, scattered into cc columns
                for h in 0..nh {
                    let off = h * dh;
                    for t in 0..ntok {
                        for cx in 0..dh {
                            vht[cx * ntok + t] = v[t * d + off + cx];
                        }
                    }
                    self.mm.matmul(
                        ntok,
                        ntok,
                        dh,
                        &sc[h * ntok * ntok..(h + 1) * ntok * ntok],
                        &vht[..dh * ntok],
                        &mut qh[..ntok * dh],
                    )?;
                    for t in 0..ntok {
                        cc[t * d + off..t * d + off + dh].copy_from_slice(&qh[t * dh..][..dh]);
                    }
                }
            }

            self.mm.matmul_prepacked(mp, d, d, &cc, &mut y, &r.wo)?;
            add_bias(ntok, d, &mut y, &w.bo);
            residual(&mut x[..td], &y[..td]);

            // --- FFN sublayer: x += fc2(ACT(fc1(LN2(x)))) ---
            layernorm_two_pass(ntok, d, &x, &w.ln2_g, Some(&w.ln2_b), g.eps, &mut ln);
            self.mm.matmul_prepacked(mp, d, dff, &ln, &mut y, &r.wf1)?;
            add_bias(ntok, dff, &mut y, &w.bf1);
            act_in_place(m.variant.act_kind, &mut y[..ntok * dff]);
            self.mm.matmul_prepacked(mp, dff, d, &y, &mut cc, &r.wf2)?;
            add_bias(ntok, d, &mut cc, &w.bf2);
            residual(&mut x[..td], &cc[..td]);

            if let Some(hid) = hidden.as_deref_mut() {
                hid[(layer + 1) * td..(layer + 2) * td].copy_from_slice(&x[..td]);
            }
            // multi-tap exit (DPT heads): emit this layer's output NOW, while it is the
            // live residual — the next layer adds into x in place.
            if tap_next < m.variant.taps.len() && m.variant.taps[tap_next] == layer {
                let ob = &mut out[tap_next * td..(tap_next + 1) * td];
                match (&m.post_g, &m.post_b) {
                    (Some(g_), b) => {
                        layernorm_two_pass(ntok, d, &x, g_, b.as_deref(), g.eps, ob);
                    }
                    (None, _) => ob.copy_from_slice(&x[..td]),
                }
                tap_next += 1;
            }
        }

        if m.variant.taps.is_empty() {
            match (&m.post_g, &m.post_b) {
                (Some(g_), b) => {
                    layernorm_two_pass(ntok, d, &x, g_, b.as_deref(), g.eps, out);
                }
                (None, _) => out[..td].copy_from_slice(&x[..td]),
            }
        }
        Ok(())
    }
}

/// `rocket_siglip_encode_ctx`: the free-function form of [`SiglipContext::encode`].
pub fn siglip_encode_ctx(
    c: &mut SiglipContext<'_>,
    pixels: &[f16],
    out: &mut [f16],
    hidden: Option<&mut [f16]>,
) -> Result<()> {
    c.encode(pixels, out, hidden)
}

#[cfg(test)]
mod tests {
    use super::super::sam::lcg_f16;
    use super::*;

    /// The synthetic geometry (the shared fixture).
    fn geom() -> SiglipGeometry {
        SiglipGeometry::fixture()
    }

    fn model(seed: u32) -> SiglipModel {
        SiglipModel::synthetic(geom(), seed)
    }

    #[test]
    fn blob_round_trip_is_lossless_and_matches_the_analytic_size() {
        let m = model(7);
        let plain = m.clone();
        let blob = m.to_blob();
        let g = &m.geom;
        // header + weights, exactly the analytic element count
        let expect = 64
            + 2 * (g.d * g.patch_dim + g.d + g.l * g.d + g.n_layers * g.layer_stride() + 2 * g.d);
        assert_eq!(blob.len(), expect, "blob size != the analytic layout");
        let back = SiglipModel::from_blob(&blob).unwrap();
        assert_eq!(back.geom, plain.geom);
        assert_eq!(back.patch_w, plain.patch_w);
        assert_eq!(back.pos, plain.pos);
        assert_eq!(back.variant, SiglipVariant::default());
        for (a, b) in back.layers.iter().zip(&plain.layers) {
            assert_eq!(a.wq, b.wq);
            assert_eq!(a.bv, b.bv);
            assert_eq!(a.wf2, b.wf2);
        }
        assert_eq!(back.post_g, plain.post_g);
        assert_eq!(back.post_b, plain.post_b);
    }

    #[test]
    fn loader_error_classes_match_the_c() {
        let blob = model(1).to_blob();
        assert_eq!(SiglipModel::from_blob(&[0u8; 63]).unwrap_err().c_code(), -2);
        let mut bad = blob.clone();
        bad[0] ^= 0x01;
        assert_eq!(SiglipModel::from_blob(&bad).unwrap_err().c_code(), -3);
        let mut bad = blob.clone();
        bad[4] = 2; // version
        assert_eq!(SiglipModel::from_blob(&bad).unwrap_err().c_code(), -3);
        // d % n_head != 0
        let mut bad = blob.clone();
        bad[16..20].copy_from_slice(&(5i32).to_le_bytes()); // n_head
        assert_eq!(SiglipModel::from_blob(&bad).unwrap_err().c_code(), -4);
        // a truncated weight region
        assert_eq!(
            SiglipModel::from_blob(&blob[..blob.len() - 2])
                .unwrap_err()
                .c_code(),
            -4
        );
        assert_eq!(
            SiglipModel::load("/nonexistent/siglip_weights.f16")
                .unwrap_err()
                .c_code(),
            -1
        );
    }

    #[test]
    fn the_activation_tables_are_the_float_formulas() {
        for &v in &[-4.0f32, -1.0, -0.0, 0.0, 0.5, 1.5, 3.0, 20.0] {
            let h = f16::from_f32(v);
            let lut = act_lut(SiglipAct::GeluTanh);
            let c = 0.797_884_6f32;
            let t = (c * (v + 0.044_715f32 * v * v * v)).tanh();
            assert_eq!(
                lut[h.to_bits() as usize],
                f16::from_f32(0.5 * v * (1.0 + t))
            );
            let lut = act_lut(SiglipAct::QuickGelu);
            assert_eq!(
                lut[h.to_bits() as usize],
                f16::from_f32(v / (1.0 + (-1.702f32 * v).exp()))
            );
            let lut = act_lut(SiglipAct::ErfGelu);
            assert_eq!(lut[h.to_bits() as usize], erf_gelu1(h));
        }
        // The table lookup is a pure function of the fp16 bit pattern.
        let mut x = lcg_f16(1024, -6.0, 6.0, 41);
        let mut once = x.clone();
        act_in_place(SiglipAct::GeluTanh, &mut once);
        act_in_place(SiglipAct::GeluTanh, &mut x);
        assert_eq!(x, once);
    }

    #[test]
    fn the_row_softmax_is_the_c_form() {
        let (rows, n) = (3usize, 8usize);
        let mut x = lcg_f16(rows * n, -3.0, 3.0, 51);
        let orig = x.clone();
        softmax_rows(&mut x, rows, n);
        for r in 0..rows {
            let row = &orig[r * n..(r + 1) * n];
            let mx = row
                .iter()
                .map(|v| v.to_f32())
                .fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|v| (v.to_f32() - mx).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let total: f32 = x[r * n..(r + 1) * n].iter().map(|v| v.to_f32()).sum();
            assert!((total - 1.0).abs() < 2e-3, "row {r} sums to {total}");
            for j in 0..n {
                let want = f16::from_f32(f16::from_f32(exps[j]).to_f32() * (1.0 / sum));
                assert!((x[r * n + j].to_f32() - want.to_f32()).abs() <= 2e-4);
            }
        }
    }

    #[test]
    fn the_host_reference_is_deterministic() {
        let m = model(11);
        let g = &m.geom;
        let pix = lcg_f16(g.ic * g.image_size * g.image_size, -1.0, 1.0, 3);
        let mut a = vec![f16::ZERO; m.ntok() * g.d];
        let mut b = vec![f16::ZERO; m.ntok() * g.d];
        let mut ha = vec![f16::ZERO; (g.n_layers + 1) * m.ntok() * g.d];
        let mut hb = vec![f16::ZERO; (g.n_layers + 1) * m.ntok() * g.d];
        siglip_encode_ref(&m, &pix, &mut a, Some(&mut ha)).unwrap();
        siglip_encode_ref(&m, &pix, &mut b, Some(&mut hb)).unwrap();
        assert_eq!(a, b);
        assert_eq!(ha, hb);
        assert!(a.iter().all(|v| v.to_f32().is_finite()));
        // The final LayerNorm standardizes rows: mean ~0, variance ~1 per row.
        let d = g.d;
        let row = &a[..d];
        let mean: f64 = row.iter().map(|v| v.to_f64()).sum::<f64>() / d as f64;
        assert!(mean.abs() < 0.05, "row mean {mean}");
        let var: f64 = row.iter().map(|v| (v.to_f64() - mean).powi(2)).sum::<f64>() / d as f64;
        assert!((var - 1.0).abs() < 0.2, "row variance {var}");
    }

    #[test]
    fn the_one_shot_path_refuses_a_non_siglip_model() {
        let mut m = model(5);
        let g = m.geom;
        let pix = vec![f16::ZERO; g.ic * g.image_size * g.image_size];
        let mut out = vec![f16::ZERO; m.ntok() * g.d];
        // A CLIP-shaped variant (prefix token + pre-LN) is the resident path's job.
        m.variant.n_prefix = 1;
        m.variant.prefix = vec![f16::ZERO; g.d];
        let e = siglip_encode_ref(&m, &pix, &mut out, None).unwrap_err();
        assert_eq!(e.c_code(), -3);
        // Short buffers are refused before any work.
        let mut m = model(5);
        m.variant.n_prefix = 0;
        m.variant.prefix.clear();
        let mut short = vec![f16::ZERO; 4];
        assert!(siglip_encode_ref(&m, &pix, &mut short, None).is_err());
    }

    #[test]
    fn the_variant_knobs_are_validated() {
        let m = model(13);
        let g = m.geom;
        let build = |v: SiglipVariant| {
            // `pos` covers the whole token sequence (`ntok = L + n_prefix`); a blob only
            // carries the patch rows, so a prefixed model extends the table.
            let mut pos = vec![f16::ZERO; v.n_prefix * g.d];
            pos.extend_from_slice(&m.pos);
            SiglipModel::from_parts(
                g,
                v,
                m.patch_w.clone(),
                m.patch_b.clone(),
                pos,
                m.layers.clone(),
                m.post_g.clone(),
                m.post_b.clone(),
            )
        };
        // A CLIP-shaped variant (a cls prefix row + a pre-encoder LayerNorm) loads, and is
        // refused by the one-shot path (it is the resident path's job).
        let clip = SiglipVariant {
            n_prefix: 1,
            prefix: vec![f16::ZERO; g.d],
            pre_g: Some(vec![f16::ONE; g.d]),
            pre_b: Some(vec![f16::ZERO; g.d]),
            act_kind: SiglipAct::QuickGelu,
            ..Default::default()
        };
        let mm = build(clip.clone()).unwrap();
        assert!(!mm.is_plain_siglip());
        assert_eq!(mm.ntok(), g.l + 1);
        let pix = vec![f16::ZERO; g.ic * g.image_size * g.image_size];
        let mut out = vec![f16::ZERO; mm.ntok() * g.d];
        assert_eq!(
            siglip_encode_ref(&mm, &pix, &mut out, None)
                .unwrap_err()
                .c_code(),
            -3
        );
        // A prefix whose table is the wrong length is a `-4`.
        let mut bad = clip.clone();
        bad.prefix = vec![f16::ZERO; g.d - 1];
        assert_eq!(build(bad).unwrap_err().c_code(), -4);
        // Multi-tap exits must be ascending and in range.
        let taps = |t: &[usize]| SiglipVariant {
            taps: t.to_vec(),
            ..Default::default()
        };
        assert!(build(taps(&[0, 1])).is_ok());
        assert_eq!(build(taps(&[1, 0])).unwrap_err().c_code(), -4);
        assert_eq!(build(taps(&[g.n_layers])).unwrap_err().c_code(), -4);
        assert_eq!(
            build(taps(&[0, 1, 2, 3, 4, 5, 6, 7, 8]))
                .unwrap_err()
                .c_code(),
            -4
        );
    }

    #[test]
    fn the_patch_projection_pads_the_im2col_columns() {
        // A model whose `patch_dim` is not a multiple of 32 lays `patch_w` out at
        // `patch_dim_pad` with a zeroed tail; the im2col zero-fills the columns to match,
        // so a padded and an unpadded model must produce the same embeddings.
        let m = model(9);
        let g = m.geom;
        let pdp = g.patch_dim.div_ceil(32) * 32 + 32; // 192 -> 224
        let mut padded = m.clone();
        let mut w = vec![f16::ZERO; g.d * pdp];
        for row in 0..g.d {
            w[row * pdp..row * pdp + g.patch_dim]
                .copy_from_slice(&m.patch_w[row * g.patch_dim..][..g.patch_dim]);
        }
        padded.patch_w = w;
        padded.variant.patch_dim_pad = pdp;

        let pix = lcg_f16(g.ic * g.image_size * g.image_size, -1.0, 1.0, 6);
        let mut a = vec![f16::ZERO; m.ntok() * g.d];
        let mut b = vec![f16::ZERO; m.ntok() * g.d];
        siglip_encode_ref(&m, &pix, &mut a, None).unwrap();
        siglip_encode_ref(&padded, &pix, &mut b, None).unwrap();
        let max = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x.to_f32() - y.to_f32()).abs())
            .fold(0.0f32, f32::max);
        assert!(max < 2e-3, "padded vs unpadded patch projection: max {max}");
    }
}
