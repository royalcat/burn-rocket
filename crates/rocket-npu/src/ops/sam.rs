//! The SAM ViT-Det image encoder (RK3588) — the M9 capstone, part 1.
//!
//! Port of `rocket_sam_encoder.c` + `rocket_sam.h` (`facebook/sam-vit-base`). The
//! graph, straight from the C header (per HF `modeling_sam.py`, verified against the
//! ONNX export):
//!
//! ```text
//! patch-embed   Conv2d(ic->d, k=patch, s=patch) WITH bias   ==  im2col + matmul
//! + pos-embed   add the learned 2D [grid,grid,d] table (host glue)
//! 12 x block    pre-norm (LN1 -> attn -> res, LN2 -> erf-GELU MLP -> res):
//!                 - a FUSED qkv Linear (d -> 3d), split into q/k/v
//!                 - WINDOWED attention on all layers except the periodic global
//!                   ones: the 2D token grid is partitioned into `win`x`win`
//!                   windows (the grid is zero-padded up to a multiple of `win`,
//!                   and those pad tokens DO participate as keys/queries in
//!                   boundary windows — faithful repro); global layers run full
//!                   self-attention over the whole grid
//!                 - scale folded into the softmax (DINOv2-style, NOT SigLIP)
//!                 - a decomposed RELATIVE-POSITION bias added to the QK^T scores
//!                   BEFORE softmax (computed from the UNSCALED q), so attention
//!                   runs matmul -> +bias -> softmax -> AV (not a fused flash-attn)
//! neck          Transpose NHWC->NCHW -> conv1x1 d->neck_out -> channels-first
//!               LayerNorm -> conv3x3 pad1 -> channels-first LayerNorm
//!               -> [1,neck_out,grid,grid]
//! ```
//!
//! The big GEMMs (qkv/o, fc1/fc2, the patch projection, the neck convs), the
//! per-window/per-head QK^T and P·V matmuls run on the NPU; the `O(tokens·d)` glue
//! (im2col, pos add, window partition/unpartition, the LayerNorms, the erf-GELU, the
//! softmax, and the decomposed rel-pos matmul+add) runs on the host. The rel-pos add
//! is faithful in fp16 (the C measures cos >= 0.9999991/layer), so there is no
//! host-f32 fallback for it.
//!
//! # Decomposed rel-pos
//!
//! Per head, with q reshaped to `[B, side, side, dhead]` (`B = windows·heads`):
//!
//! ```text
//! rel_h[b,i,j,i'] = sum_c q[b,i,j,c] * Rh[i,i',c]      (Rh is [side,side,dhead])
//! rel_w[b,i,j,j'] = sum_c q[b,i,j,c] * Rw[j,j',c]      (Rw is [side,side,dhead])
//! bias[b,(i,j),(i',j')] = rel_h[b,i,j,i'] + rel_w[b,i,j,j']
//! scores += bias                                      (added to scale*QK^T)
//! ```
//!
//! The `[side,side,dhead]` tables are the resolved `get_rel_pos` output
//! (`q_size == k_size`, so the `F.interpolate` is identity):
//! `Rh[i,j,c] = rel_pos_param[(i-j)+(side-1), c]`. `side` is `win` for a windowed
//! layer and `grid` for a global one.
//!
//! # The two entry points
//!
//! [`sam_encode`] is the one-shot path: every call re-packs the static weights and
//! re-allocates its BOs (the C `rocket_sam_encode(fd >= 0, ...)`), and
//! [`sam_encode_ref`] is its exact host reference (`fd < 0`; the off-device datapath
//! self-check — bit-identical to the C, see the module tests and the `ab_board` A/B).
//! [`SamContext`] is the resident latency path: the static GEMM weights (patch, the
//! per-layer qkv/o/fc1/fc2, the neck conv1) are packed once at create and reused
//! across images, so a per-image call only packs activations.
//!
//! The C keeps one computed-matmul *stream* (its own fd → its own NPU core) and a
//! head scratch per worker and fans the `(window, head)` tasks across pthreads. This
//! port instead runs the same computed matmuls through the resident
//! [`crate::matmul::Context`], which fans N across its own worker fds; the task split
//! differs, the arithmetic does not (both are the same fp16 one-shot matmul on the
//! same operands).
//!
//! # Blob format (`tools/sam_extract.py`)
//!
//! Little-endian. A 16 x int32 header (64 bytes) then `n_layers` int32 windowed flags
//! then fp16 weights in declaration order. The C mmaps the file; this port owns the
//! decoded tensors (a `Vec<f16>` per tensor), so [`SamModel::from_blob`] and
//! [`SamModel::from_parts`] are the two constructors. Loader errors keep the C codes:
//! `-1` open, `-2` size, `-3` magic/version, `-4` inconsistent dims / blob size.

use std::path::Path;
use std::sync::OnceLock;

use half::f16;

use crate::Device;
use crate::error::{Error, Result};
use crate::matmul::run::one_shot;
use crate::matmul::{Context, Weight};
use crate::ops::conv::{Conv2dDesc, conv2d, conv2d_ref};

/// `SAM_MAGIC` ("SAMB", little-endian).
pub const SAM_MAGIC: u32 = 0x5341_4D42;
/// `SAM_VERSION`.
pub const SAM_VERSION: i32 = 1;
/// `ROCKET_SAM_MAX_LAYERS` (SAM-B 12, ViT-L 24, ViT-H 32).
pub const SAM_MAX_LAYERS: usize = 32;

// ── Model ─────────────────────────────────────────────────────────────────────

/// The blob-header geometry (`rocket_sam_model`'s scalar fields).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamGeometry {
    /// Hidden size (768 for SAM-B).
    pub d: usize,
    /// Encoder layers (12).
    pub n_layers: usize,
    /// Attention heads (12).
    pub n_head: usize,
    /// `d / n_head` (64).
    pub dhead: usize,
    /// MLP intermediate size (3072).
    pub d_ff: usize,
    /// Token-grid side = `image_size / patch` (64).
    pub grid: usize,
    /// Window side (14).
    pub win: usize,
    /// Patch conv kernel **and** stride (16).
    pub patch: usize,
    /// Input image side (1024).
    pub image_size: usize,
    /// Input channels (3).
    pub ic: usize,
    /// `ic * patch * patch` (768).
    pub patch_dim: usize,
    /// Neck output channels (256).
    pub neck_out: usize,
    /// `layer_norm_eps` (1e-6).
    pub eps: f32,
}

impl SamGeometry {
    /// Token count of the real grid.
    pub fn tokens(&self) -> usize {
        self.grid * self.grid
    }

    /// The small NPU-legal geometry the self-test fixtures use: `d=64`, 2 layers,
    /// 2 heads (`dhead=32`), `d_ff=64`, `grid=8`, `win=4`, `patch=8`, `ic=3`
    /// (`patch_dim=192`), `neck_out=16`. Every matmul meets the fp16 align
    /// (`K % 32`, `N % 16`, `M % 4`), and layer 0 is windowed while layer 1 is global.
    pub fn fixture() -> SamGeometry {
        SamGeometry {
            d: 64,
            n_layers: 2,
            n_head: 2,
            dhead: 32,
            d_ff: 64,
            grid: 8,
            win: 4,
            patch: 8,
            image_size: 64,
            ic: 3,
            patch_dim: 192,
            neck_out: 16,
            eps: 1e-6,
        }
    }

    /// Window-partitioned token count: `gp * gp` with `gp = grid` rounded up to a
    /// multiple of `win` (`grid*grid` when the grid already divides evenly).
    pub fn window_tokens(&self) -> usize {
        let gp = self.grid.div_ceil(self.win) * self.win;
        gp * gp
    }

    /// The geometry contract the C loader enforces (`-4`).
    fn check(&self) -> bool {
        self.d > 0
            && self.n_layers > 0
            && self.n_layers <= SAM_MAX_LAYERS
            && self.n_head > 0
            && self.d_ff > 0
            && self.grid > 0
            && self.win > 0
            && self.dhead > 0
            && self.d.is_multiple_of(self.n_head)
            && self.dhead == self.d / self.n_head
            && self.patch_dim == self.ic * self.patch * self.patch
    }
}

/// One encoder layer's weights (row-major `[out,in]`, i.e. a PyTorch `nn.Linear`).
#[derive(Debug, Clone)]
pub struct SamLayerWeights {
    /// `1` = windowed attention, `0` = global.
    pub windowed: bool,
    /// LayerNorm 1 gamma / beta `[d]`.
    pub ln1_g: Vec<f16>,
    pub ln1_b: Vec<f16>,
    /// Fused qkv `[3d,d]` and bias `[3d]` (split into q/k/v by row thirds).
    pub wqkv: Vec<f16>,
    pub bqkv: Vec<f16>,
    /// Output projection `[d,d]` and bias `[d]`.
    pub wo: Vec<f16>,
    pub bo: Vec<f16>,
    /// LayerNorm 2 gamma / beta `[d]`.
    pub ln2_g: Vec<f16>,
    pub ln2_b: Vec<f16>,
    /// fc1 `[d_ff,d]` + bias `[d_ff]`.
    pub wf1: Vec<f16>,
    pub bf1: Vec<f16>,
    /// fc2 `[d,d_ff]` + bias `[d]`.
    pub wf2: Vec<f16>,
    pub bf2: Vec<f16>,
    /// The resolved rel-pos tables `[side,side,dhead]`.
    pub rh: Vec<f16>,
    pub rw: Vec<f16>,
}

/// The conv neck's weights.
#[derive(Debug, Clone)]
pub struct SamNeckWeights {
    /// 1x1 conv weight squeezed to `[neck_out,d]`.
    pub c1: Vec<f16>,
    /// Channels-first LayerNorm 1 `[neck_out]`.
    pub ln1_g: Vec<f16>,
    pub ln1_b: Vec<f16>,
    /// 3x3 conv weight `[neck_out][neck_out][3][3]`.
    pub c2: Vec<f16>,
    /// Channels-first LayerNorm 2 `[neck_out]`.
    pub ln2_g: Vec<f16>,
    pub ln2_b: Vec<f16>,
}

/// A loaded SAM image encoder: geometry plus owned fp16 tensors.
#[derive(Debug, Clone)]
pub struct SamModel {
    /// The blob-header geometry.
    pub geom: SamGeometry,
    /// Patch projection `[d][patch_dim]` (row-major `[oc][ic*kh*kw]`).
    pub patch_w: Vec<f16>,
    /// Patch bias `[d]`.
    pub patch_b: Vec<f16>,
    /// Learned position table `[grid*grid][d]` (the `[1,grid,grid,d]` table flattened).
    pub pos: Vec<f16>,
    /// Per-layer weights, in blob order.
    pub layers: Vec<SamLayerWeights>,
    /// The conv neck.
    pub neck: SamNeckWeights,
}

impl SamModel {
    /// Build a model from owned tensors (the "frontend fills the struct" path), with
    /// the same consistency checks the blob loader applies.
    pub fn from_parts(
        geom: SamGeometry,
        patch_w: Vec<f16>,
        patch_b: Vec<f16>,
        pos: Vec<f16>,
        layers: Vec<SamLayerWeights>,
        neck: SamNeckWeights,
    ) -> Result<SamModel> {
        let m = SamModel {
            geom,
            patch_w,
            patch_b,
            pos,
            layers,
            neck,
        };
        m.validate("rocket_sam_load")?;
        Ok(m)
    }

    /// A deterministic pseudo-random model of `geom` (an LCG, not a checkpoint).
    ///
    /// The self-test / A-B fixture: [`Self::to_blob`]-able, so the C library can encode
    /// the same weights and the two host references can be compared bit for bit.
    pub fn synthetic(geom: SamGeometry, seed: u32) -> SamModel {
        let d = geom.d;
        let gg = geom.tokens();
        let mut r = Gen::new(seed);
        let mut layers = Vec::new();
        for l in 0..geom.n_layers {
            // Alternate windowed / global, so both attention shapes are exercised.
            let windowed = l % 2 == 0;
            let side = if windowed { geom.win } else { geom.grid };
            layers.push(SamLayerWeights {
                windowed,
                ln1_g: r.g(d),
                ln1_b: r.b(d),
                wqkv: r.w(3 * d * d),
                bqkv: r.w(3 * d),
                wo: r.w(d * d),
                bo: r.w(d),
                ln2_g: r.g(d),
                ln2_b: r.b(d),
                wf1: r.w(geom.d_ff * d),
                bf1: r.w(geom.d_ff),
                wf2: r.w(d * geom.d_ff),
                bf2: r.w(d),
                rh: r.w(side * side * geom.dhead),
                rw: r.w(side * side * geom.dhead),
            });
        }
        let no = geom.neck_out;
        let (patch_w, patch_b, pos) = (r.w(d * geom.patch_dim), r.w(d), r.w(gg * d));
        let (c1, ln1_g, ln1_b) = (r.w(no * d), r.g(no), r.b(no));
        let (c2, ln2_g, ln2_b) = (r.w(no * no * 9), r.g(no), r.b(no));
        SamModel::from_parts(
            geom,
            patch_w,
            patch_b,
            pos,
            layers,
            SamNeckWeights {
                c1,
                ln1_g,
                ln1_b,
                c2,
                ln2_g,
                ln2_b,
            },
        )
        .expect("the synthetic geometry is consistent")
    }

    /// Serialize to the `tools/sam_extract.py` blob format (the inverse of
    /// [`Self::from_blob`]); used by the self-test's `--write-blob` and the C A/B.
    pub fn to_blob(&self) -> Vec<u8> {
        let g = &self.geom;
        let mut b = Vec::new();
        let h: [i32; 16] = [
            SAM_MAGIC as i32,
            SAM_VERSION,
            g.d as i32,
            g.n_layers as i32,
            g.n_head as i32,
            g.d_ff as i32,
            g.grid as i32,
            g.win as i32,
            g.patch as i32,
            g.ic as i32,
            g.neck_out as i32,
            g.eps.to_bits() as i32,
            g.image_size as i32,
            g.dhead as i32,
            g.patch_dim as i32,
            0,
        ];
        for v in h {
            b.extend_from_slice(&v.to_le_bytes());
        }
        for w in &self.layers {
            b.extend_from_slice(&(w.windowed as i32).to_le_bytes());
        }
        let mut put = |t: &[f16]| {
            for v in t {
                b.extend_from_slice(&v.to_bits().to_le_bytes());
            }
        };
        put(&self.patch_w);
        put(&self.patch_b);
        put(&self.pos);
        for w in &self.layers {
            for t in [
                &w.ln1_g, &w.ln1_b, &w.wqkv, &w.bqkv, &w.wo, &w.bo, &w.ln2_g, &w.ln2_b, &w.wf1,
                &w.bf1, &w.wf2, &w.bf2, &w.rh, &w.rw,
            ] {
                put(t);
            }
        }
        put(&self.neck.c1);
        put(&self.neck.ln1_g);
        put(&self.neck.ln1_b);
        put(&self.neck.c2);
        put(&self.neck.ln2_g);
        put(&self.neck.ln2_b);
        b
    }

    /// `rocket_sam_load`: read and validate a weight blob from `tools/sam_extract.py`.
    pub fn load(path: impl AsRef<Path>) -> Result<SamModel> {
        let bytes = std::fs::read(path.as_ref())
            .map_err(|_| Error::coded(crate::Status::Io, -1, "rocket_sam_load"))?;
        Self::from_blob(&bytes)
    }

    /// Parse a weight blob from memory (the C `mmap`, minus the mmap).
    pub fn from_blob(bytes: &[u8]) -> Result<SamModel> {
        let bad_size = || Error::coded(crate::Status::Io, -2, "rocket_sam_load");
        if bytes.len() < 64 {
            return Err(bad_size());
        }
        let h = |i: usize| -> i32 {
            let o = i * 4;
            i32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]])
        };
        if h(0) as u32 != SAM_MAGIC || h(1) != SAM_VERSION {
            return Err(Error::coded(
                crate::Status::Unsupported,
                -3,
                "rocket_sam_load",
            ));
        }
        let geom = SamGeometry {
            d: h(2).max(0) as usize,
            n_layers: h(3).max(0) as usize,
            n_head: h(4).max(0) as usize,
            d_ff: h(5).max(0) as usize,
            grid: h(6).max(0) as usize,
            win: h(7).max(0) as usize,
            patch: h(8).max(0) as usize,
            ic: h(9).max(0) as usize,
            neck_out: h(10).max(0) as usize,
            eps: f32::from_bits(h(11) as u32),
            image_size: h(12).max(0) as usize,
            dhead: h(13).max(0) as usize,
            patch_dim: h(14).max(0) as usize,
        };
        let bad_dims = || Error::coded(crate::Status::Shape, -4, "rocket_sam_load");
        if !geom.check() {
            return Err(bad_dims());
        }

        let d = geom.d;
        let dff = geom.d_ff;
        let dh = geom.dhead;
        let no = geom.neck_out;
        let gg = geom.tokens();
        let n = geom.n_layers;

        // The weights start after the 64-byte header and the layer flags.
        let wstart = 64 + n * 4;
        if bytes.len() < wstart {
            return Err(bad_size());
        }
        let flags: Vec<bool> = (0..n).map(|l| h(16 + l) != 0).collect();
        let mut r = BlobReader {
            bytes,
            off: wstart,
            op: "rocket_sam_load",
        };

        let patch_w = r.take(d * geom.patch_dim)?;
        let patch_b = r.take(d)?;
        let pos = r.take(gg * d)?;
        let mut layers = Vec::with_capacity(n);
        for &windowed in &flags {
            let side = if windowed { geom.win } else { geom.grid };
            layers.push(SamLayerWeights {
                windowed,
                ln1_g: r.take(d)?,
                ln1_b: r.take(d)?,
                wqkv: r.take(3 * d * d)?,
                bqkv: r.take(3 * d)?,
                wo: r.take(d * d)?,
                bo: r.take(d)?,
                ln2_g: r.take(d)?,
                ln2_b: r.take(d)?,
                wf1: r.take(dff * d)?,
                bf1: r.take(dff)?,
                wf2: r.take(d * dff)?,
                bf2: r.take(d)?,
                rh: r.take(side * side * dh)?,
                rw: r.take(side * side * dh)?,
            });
        }
        let neck = SamNeckWeights {
            c1: r.take(no * d)?,
            ln1_g: r.take(no)?,
            ln1_b: r.take(no)?,
            c2: r.take(no * no * 9)?,
            ln2_g: r.take(no)?,
            ln2_b: r.take(no)?,
        };
        // The C requires the blob to be exactly the size the dims imply.
        if r.off != bytes.len() {
            return Err(bad_dims());
        }
        let model = SamModel {
            geom,
            patch_w,
            patch_b,
            pos,
            layers,
            neck,
        };
        model.validate("rocket_sam_load")?;
        Ok(model)
    }

    /// Tensor-length validation (the `-4` class for a hand-built model).
    fn validate(&self, op: &'static str) -> Result<()> {
        let g = &self.geom;
        if !g.check() || self.layers.len() != g.n_layers {
            return Err(Error::coded(crate::Status::Shape, -4, op));
        }
        let (d, dff, no, gg) = (g.d, g.d_ff, g.neck_out, g.tokens());
        let mut ok = self.patch_w.len() == d * g.patch_dim
            && self.patch_b.len() == d
            && self.pos.len() == gg * d;
        for (l, w) in self.layers.iter().enumerate() {
            let side = if w.windowed { g.win } else { g.grid };
            ok &= w.ln1_g.len() == d
                && w.ln1_b.len() == d
                && w.wqkv.len() == 3 * d * d
                && w.bqkv.len() == 3 * d
                && w.wo.len() == d * d
                && w.bo.len() == d
                && w.ln2_g.len() == d
                && w.ln2_b.len() == d
                && w.wf1.len() == dff * d
                && w.bf1.len() == dff
                && w.wf2.len() == d * dff
                && w.bf2.len() == d
                && w.rh.len() == side * side * g.dhead
                && w.rw.len() == side * side * g.dhead;
            let _ = l;
        }
        ok &= self.neck.c1.len() == no * d
            && self.neck.ln1_g.len() == no
            && self.neck.ln1_b.len() == no
            && self.neck.c2.len() == no * no * 9
            && self.neck.ln2_g.len() == no
            && self.neck.ln2_b.len() == no;
        if !ok {
            return Err(Error::coded(crate::Status::Shape, -4, op));
        }
        Ok(())
    }

    /// The rel-pos side of layer `l` (`win` windowed, `grid` global).
    pub fn side(&self, l: usize) -> usize {
        if self.layers[l].windowed {
            self.geom.win
        } else {
            self.geom.grid
        }
    }

    /// Window-partitioned token count (see [`SamGeometry::window_tokens`]).
    pub fn window_tokens(&self) -> usize {
        self.geom.window_tokens()
    }
}

/// A deterministic tensor generator for the self-test fixtures (an LCG, so a case is
/// reproducible on any host and the C A/B can be driven from the same blob).
pub(super) struct Gen(u32);

impl Gen {
    pub(super) fn new(seed: u32) -> Gen {
        Gen(0x9e37_79b9 ^ seed)
    }

    fn seed(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        self.0
    }

    /// A weight-like tensor in `[-0.4, 0.4)`.
    pub(super) fn w(&mut self, n: usize) -> Vec<f16> {
        let s = self.seed();
        lcg_f16(n, -0.4, 0.4, s)
    }

    /// A LayerNorm-gamma-like tensor in `[0.5, 1.5)`.
    pub(super) fn g(&mut self, n: usize) -> Vec<f16> {
        let s = self.seed();
        lcg_f16(n, 0.5, 1.5, s)
    }

    /// A bias-like tensor in `[-0.2, 0.2)`.
    pub(super) fn b(&mut self, n: usize) -> Vec<f16> {
        let s = self.seed();
        lcg_f16(n, -0.2, 0.2, s)
    }
}

/// A deterministic uniform fp16 tensor (the shared self-test PRNG).
pub(super) fn lcg_f16(n: usize, lo: f32, hi: f32, seed: u32) -> Vec<f16> {
    let mut st = 0x9e37_79b9u32 ^ seed;
    (0..n)
        .map(|_| {
            st = st.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let u = ((st >> 8) & 0xffff) as f32 / 65535.0;
            f16::from_f32(lo + (hi - lo) * u)
        })
        .collect()
}

/// A bounds-checked little-endian fp16 reader over the blob's weight region.
struct BlobReader<'a> {
    bytes: &'a [u8],
    off: usize,
    op: &'static str,
}

impl BlobReader<'_> {
    fn take(&mut self, n: usize) -> Result<Vec<f16>> {
        let end = self
            .off
            .checked_add(n * 2)
            .filter(|e| *e <= self.bytes.len())
            .ok_or_else(|| Error::coded(crate::Status::Shape, -4, self.op))?;
        let src = &self.bytes[self.off..end];
        let mut out = Vec::with_capacity(n);
        for c in src.as_chunks::<2>().0 {
            out.push(f16::from_bits(u16::from_le_bytes(*c)));
        }
        self.off = end;
        Ok(out)
    }
}

// ── Host math (shared by the reference and the device path's glue) ─────────────

/// `C[M][N] = A[M][K] · B[N][K]^T` with an f32 accumulator and an fp16 store — the C
/// `host_matmul`. The products are exact in f32 (an fp16 significand squared needs 22
/// of the 24 bits), so an fma contraction cannot change the result.
fn host_matmul(m: usize, k: usize, n: usize, a: &[f16], b: &[f16], c: &mut [f16]) {
    for i in 0..m {
        let ar = &a[i * k..i * k + k];
        for j in 0..n {
            let br = &b[j * k..j * k + k];
            let mut acc = 0.0f32;
            for t in 0..k {
                acc += ar[t].to_f32() * br[t].to_f32();
            }
            c[i * n + j] = f16::from_f32(acc);
        }
    }
}

/// LayerNorm over the last axis: two-pass f64 mean/variance, fp16 store (the C
/// `h_layernorm`). This is deliberately **not** [`super::norm::layernorm_ref`]: SAM's
/// reference averages the squared deviation from the mean instead of the sum of
/// squares, and the two round differently.
fn h_layernorm(m: usize, d: usize, x: &[f16], g: &[f16], b: &[f16], eps: f32, out: &mut [f16]) {
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
            o[j] = crate::ops::f16_from_f64(
                (xr[j].to_f64() - mean) * inv * g[j].to_f64() + b[j].to_f64(),
            );
        }
    }
}

/// Channels-first LayerNorm (`SamLayerNorm`): for each of the `p` spatial positions,
/// normalize over the `c` channels (stride `p` in a `[C,P]` channels-major buffer),
/// affine per channel, eps 1e-6 (the C `h_layernorm2d`).
fn h_layernorm2d(c: usize, p: usize, x: &[f16], g: &[f16], b: &[f16], eps: f32, out: &mut [f16]) {
    let inv_c = 1.0 / c as f64;
    for pos in 0..p {
        let mut mean = 0.0f64;
        for ch in 0..c {
            mean += x[ch * p + pos].to_f64();
        }
        mean *= inv_c;
        let mut var = 0.0f64;
        for ch in 0..c {
            let t = x[ch * p + pos].to_f64() - mean;
            var += t * t;
        }
        let inv = 1.0 / (var * inv_c + f64::from(eps)).sqrt();
        for ch in 0..c {
            out[ch * p + pos] = crate::ops::f16_from_f64(
                (x[ch * p + pos].to_f64() - mean) * inv * g[ch].to_f64() + b[ch].to_f64(),
            );
        }
    }
}

/// The erf-GELU fp16 table (`0.5·x·(1+erf(x/√2))` over every fp16 input), built once.
///
/// The C keeps a `static` table under a `pthread_once`; this is the same table behind a
/// [`OnceLock`], and it is bit-exact to the C's because it evaluates the identical
/// **f32** expression with glibc's `erff` and rounds once to fp16.
fn gelu_erf_lut() -> &'static [f16; 65536] {
    static LUT: OnceLock<Box<[f16; 65536]>> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut t = Box::new([f16::ZERO; 65536]);
        for (u, slot) in t.iter_mut().enumerate() {
            let v = f16::from_bits(u as u16).to_f32();
            *slot = f16::from_f32(0.5 * v * (1.0 + erff32(v * INV_SQRT_2)));
        }
        t
    })
}

/// `1/√2` (the C `inv_sqrt2`).
const INV_SQRT_2: f32 = std::f32::consts::FRAC_1_SQRT_2;

unsafe extern "C" {
    /// glibc `erff`, the single-precision function the C `h_gelu_erf` table calls (so
    /// the table words match it bit for bit).
    fn erff(x: f32) -> f32;
}

/// glibc's single-precision `erf`, shared with the SigLIP activation tables.
pub(super) fn erff32(x: f32) -> f32 {
    // SAFETY: the C `erff` is pure, takes one f32 and returns one f32.
    unsafe { erff(x) }
}

fn erf_gelu_in_place(x: &mut [f16]) {
    let lut = gelu_erf_lut();
    for v in x.iter_mut() {
        *v = lut[v.to_bits() as usize];
    }
}

pub(super) fn add_bias(m: usize, n: usize, c: &mut [f16], b: &[f16]) {
    for row in 0..m {
        let r = &mut c[row * n..row * n + n];
        for (cell, &bv) in r.iter_mut().zip(b) {
            *cell = f16::from_f32(cell.to_f32() + bv.to_f32());
        }
    }
}

pub(super) fn residual(acc: &mut [f16], add: &[f16]) {
    for (a, &b) in acc.iter_mut().zip(add) {
        *a = f16::from_f32(a.to_f32() + b.to_f32());
    }
}

/// Round up to a multiple of 32 (the key pad for the QK `N % 16` / PV `K % 32` align).
fn round32(n: usize) -> usize {
    n.div_ceil(32) * 32
}

// ── Windowing (host) ─────────────────────────────────────────────────────────

/// `window_partition`: `h[gg][d]` → `hw[nWin*S2][d]`, window-major, pad tokens zero.
/// Returns `nWin = (gp/win)^2` with `gp = grid` rounded up to a multiple of `win`.
fn window_partition(h: &[f16], grid: usize, win: usize, d: usize, hw: &mut [f16]) -> usize {
    let gp = grid.div_ceil(win) * win;
    let nw = gp / win;
    let s2 = win * win;
    hw[..nw * nw * s2 * d].fill(f16::ZERO);
    for wr in 0..nw {
        for wc in 0..nw {
            let base = (wr * nw + wc) * s2 * d;
            for i in 0..win {
                let gr = wr * win + i;
                if gr >= grid {
                    continue;
                }
                for j in 0..win {
                    let gc = wc * win + j;
                    if gc >= grid {
                        continue;
                    }
                    let dst = base + (i * win + j) * d;
                    let src = (gr * grid + gc) * d;
                    hw[dst..dst + d].copy_from_slice(&h[src..src + d]);
                }
            }
        }
    }
    nw * nw
}

/// `window_unpartition`: `cc[nWin*S2][d]` → `ctxf[grid*grid][d]`, dropping pad tokens.
fn window_unpartition(cc: &[f16], grid: usize, win: usize, d: usize, ctxf: &mut [f16]) {
    let gp = grid.div_ceil(win) * win;
    let nw = gp / win;
    let s2 = win * win;
    for wr in 0..nw {
        for wc in 0..nw {
            let base = (wr * nw + wc) * s2 * d;
            for i in 0..win {
                let gr = wr * win + i;
                if gr >= grid {
                    continue;
                }
                for j in 0..win {
                    let gc = wc * win + j;
                    if gc >= grid {
                        continue;
                    }
                    let src = base + (i * win + j) * d;
                    let dst = (gr * grid + gc) * d;
                    ctxf[dst..dst + d].copy_from_slice(&cc[src..src + d]);
                }
            }
        }
    }
}

/// The patch-embedding im2col (stride == kernel for both encoders, pad 0):
/// `patches[p][ci*kh*kw + r*kw + c]`, matching `patch_W`'s `[oc][ic][kh][kw]` flatten.
///
/// `patches` is laid out at a `pdim_pad` row stride, and the caller zeroes it, so the
/// `[pdim, pdim_pad)` tail stays 0 (SigLIP's `patch_dim_pad`, see
/// [`super::siglip::SiglipVariant::patch_dim_pad`]); SAM passes `pdim_pad == pdim`.
#[allow(clippy::too_many_arguments)]
pub(super) fn patchify(
    pix: &[f16],
    image_size: usize,
    ic: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    side: usize,
    pdim_pad: usize,
    patches: &mut [f16],
) {
    for ph in 0..side {
        for pw in 0..side {
            let row = (ph * side + pw) * pdim_pad;
            for ci in 0..ic {
                let chan = &pix[ci * image_size * image_size..];
                for r in 0..kh {
                    let src = (ph * stride + r) * image_size + pw * stride;
                    let dst = row + ci * kh * kw + r * kw;
                    patches[dst..dst + kw].copy_from_slice(&chan[src..src + kw]);
                }
            }
        }
    }
}

/// SAM's fixed-geometry im2col (`stride == kernel == patch`, no pad, no tail).
fn im2col(g: &SamGeometry, pix: &[f16], patches: &mut [f16]) {
    patchify(
        pix,
        g.image_size,
        g.ic,
        g.patch,
        g.patch,
        g.patch,
        g.grid,
        g.patch_dim,
        patches,
    );
}

// ── The computed-matmul provider ─────────────────────────────────────────────

/// A matmul provider, so the one-shot (device), the resident (ctx) and the host
/// reference paths share the exact attention/block datapath — the C's `mm_provider`.
trait Mm {
    /// `C[M,N] = A[M,K] · B[N,K]^T` for two activation operands.
    fn mm(
        &mut self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f16],
        b: &[f16],
        c: &mut [f16],
    ) -> Result<()>;
    /// The same, with a static weight `plain` and its optional resident form.
    #[allow(clippy::too_many_arguments)]
    fn proj(
        &mut self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f16],
        plain: &[f16],
        res: Option<&Weight>,
        c: &mut [f16],
    ) -> Result<()>;
    /// A conv (the neck's 3x3); the host arm is the fp32-accumulate oracle.
    fn conv(&mut self, d: &Conv2dDesc, input: &[f16], w: &[f16], out: &mut [f16]) -> Result<()>;
}

struct HostMm;

impl Mm for HostMm {
    fn mm(
        &mut self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f16],
        b: &[f16],
        c: &mut [f16],
    ) -> Result<()> {
        host_matmul(m, k, n, a, b, c);
        Ok(())
    }

    fn proj(
        &mut self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f16],
        plain: &[f16],
        _res: Option<&Weight>,
        c: &mut [f16],
    ) -> Result<()> {
        host_matmul(m, k, n, a, plain, c);
        Ok(())
    }

    fn conv(&mut self, d: &Conv2dDesc, input: &[f16], w: &[f16], out: &mut [f16]) -> Result<()> {
        conv2d_ref(d, input, w, out);
        Ok(())
    }
}

struct DevMm<'a> {
    dev: &'a Device,
}

impl Mm for DevMm<'_> {
    fn mm(
        &mut self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f16],
        b: &[f16],
        c: &mut [f16],
    ) -> Result<()> {
        one_shot(self.dev, m, k, n, a, b, c)
    }

    fn proj(
        &mut self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f16],
        plain: &[f16],
        _res: Option<&Weight>,
        c: &mut [f16],
    ) -> Result<()> {
        one_shot(self.dev, m, k, n, a, plain, c)
    }

    fn conv(&mut self, d: &Conv2dDesc, input: &[f16], w: &[f16], out: &mut [f16]) -> Result<()> {
        conv2d(self.dev, d, input, w, out)
    }
}

struct CtxMm<'a, 'c> {
    ctx: &'a mut Context,
    dev: &'c Device,
}

impl Mm for CtxMm<'_, '_> {
    fn mm(
        &mut self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f16],
        b: &[f16],
        c: &mut [f16],
    ) -> Result<()> {
        self.ctx.matmul(m, k, n, a, b, c)
    }

    fn proj(
        &mut self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f16],
        plain: &[f16],
        res: Option<&Weight>,
        c: &mut [f16],
    ) -> Result<()> {
        match res {
            Some(w) => self.ctx.matmul_prepacked(m, k, n, a, c, w),
            None => self.ctx.matmul(m, k, n, a, plain, c),
        }
    }

    fn conv(&mut self, d: &Conv2dDesc, input: &[f16], w: &[f16], out: &mut [f16]) -> Result<()> {
        conv2d(self.dev, d, input, w, out)
    }
}

// ── Attention scratch ────────────────────────────────────────────────────────

/// The per-encode attention scratch shared by every layer (the C `attn_scratch`).
struct AttnScratch {
    /// Window-partitioned LN1 output `[Mwin][d]`.
    hw: Vec<f16>,
    /// Fused qkv `[Mtok][3d]`.
    qkv: Vec<f16>,
    /// Per-token attention context `[gg][d]`.
    ctxf: Vec<f16>,
    /// Key row/col index tables `[S2max]`.
    ip: Vec<usize>,
    jp: Vec<usize>,
}

impl AttnScratch {
    fn new(g: &SamGeometry) -> AttnScratch {
        let smax = g.grid.max(g.win);
        AttnScratch {
            hw: vec![f16::ZERO; g.window_tokens() * g.d],
            qkv: vec![f16::ZERO; g.window_tokens() * 3 * g.d],
            ctxf: vec![f16::ZERO; g.tokens() * g.d],
            ip: vec![0; smax * smax],
            jp: vec![0; smax * smax],
        }
    }
}

/// One head's scratch (the C `head_scratch`).
struct HeadScratch {
    qh: Vec<f16>,
    kh: Vec<f16>,
    vh: Vec<f16>,
    khp: Vec<f16>,
    vtp: Vec<f16>,
    sc: Vec<f16>,
    outh: Vec<f16>,
    relh: Vec<f16>,
    relw: Vec<f16>,
    qcol: Vec<f16>,
    blk: Vec<f16>,
}

impl HeadScratch {
    fn new(g: &SamGeometry) -> HeadScratch {
        let smax = g.grid.max(g.win);
        let s2max = smax * smax;
        let kpad = round32(s2max);
        let dh = g.dhead;
        HeadScratch {
            qh: vec![f16::ZERO; s2max * dh],
            kh: vec![f16::ZERO; s2max * dh],
            vh: vec![f16::ZERO; s2max * dh],
            khp: vec![f16::ZERO; kpad * dh],
            vtp: vec![f16::ZERO; dh * kpad],
            sc: vec![f16::ZERO; s2max * kpad],
            outh: vec![f16::ZERO; s2max * dh],
            relh: vec![f16::ZERO; s2max * smax],
            relw: vec![f16::ZERO; s2max * smax],
            qcol: vec![f16::ZERO; smax * dh],
            blk: vec![f16::ZERO; smax * smax],
        }
    }
}

/// `rel_pos`: fill `relh[t][i']` and `relw[t][j']` for `qh[S2][dh]`, side `S` (host).
fn rel_pos(hs: &mut HeadScratch, rh: &[f16], rw: &[f16], s: usize, dh: usize) {
    // rel_h: rows i*S..(i+1)*S of qh are contiguous -> block = qh_i · Rh[i]^T [S(j),S(i')]
    for i in 0..s {
        let qrows = i * s * dh;
        let rrows = i * s * dh;
        let dst = i * s * s;
        let (qh, relh) = (&hs.qh, &mut hs.relh);
        host_matmul(
            s,
            dh,
            s,
            &qh[qrows..qrows + s * dh],
            &rh[rrows..rrows + s * dh],
            &mut relh[dst..dst + s * s],
        );
    }
    // rel_w: gather qh column j (rows j, j+S, ...) -> block = qcol · Rw[j]^T [S(i),S(j')]
    for j in 0..s {
        for i in 0..s {
            let src = (i * s + j) * dh;
            hs.qcol[i * dh..(i + 1) * dh].copy_from_slice(&hs.qh[src..src + dh]);
        }
        host_matmul(
            s,
            dh,
            s,
            &hs.qcol[..s * dh],
            &rw[j * s * dh..(j + 1) * s * dh],
            &mut hs.blk[..s * s],
        );
        for i in 0..s {
            let dst = (i * s + j) * s;
            let src = i * s;
            hs.relw[dst..dst + s].copy_from_slice(&hs.blk[src..src + s]);
        }
    }
}

/// The biased softmax over one query row:
/// `p[t'] = softmax(scale·sc[t'] + relh[i'] + relw[j'])`, pad columns `[S2,Kpad)`
/// zeroed (excluded). `sc_row` is overwritten with `p` (the C `attn_softmax_row`).
#[allow(clippy::too_many_arguments)]
fn attn_softmax_row(
    sc_row: &mut [f16],
    s2: usize,
    kpad: usize,
    scale: f32,
    relh_t: &[f16],
    relw_t: &[f16],
    ip: &[usize],
    jp: &[usize],
) {
    let mut mx = f32::NEG_INFINITY;
    for (t, cell) in sc_row[..s2].iter_mut().enumerate() {
        let v = cell.to_f32() * scale + relh_t[ip[t]].to_f32() + relw_t[jp[t]].to_f32();
        *cell = f16::from_f32(v);
        if v > mx {
            mx = v;
        }
    }
    // The denominator accumulates the *unrounded* f32 exponential, exactly as in C.
    let mut sum = 0.0f32;
    for cell in sc_row[..s2].iter_mut() {
        let e = (cell.to_f32() - mx).exp();
        *cell = f16::from_f32(e);
        sum += e;
    }
    sc_row[s2..kpad].fill(f16::ZERO);
    let inv = 1.0 / sum;
    for v in sc_row[..s2].iter_mut() {
        *v = f16::from_f32(v.to_f32() * inv);
    }
}

/// Gather head `h`'s q/k/v for window `w` out of the fused `qkv[Mtok][3d]` into
/// `hs.qh/kh/vh` (the C `gather_head`; the fused layout is `[q|k|v]` per row).
fn gather_head(
    hs: &mut HeadScratch,
    qkv: &[f16],
    w: usize,
    h: usize,
    s2: usize,
    d: usize,
    dh: usize,
) {
    let off = h * dh;
    for t in 0..s2 {
        let base = (w * s2 + t) * 3 * d + off;
        hs.qh[t * dh..(t + 1) * dh].copy_from_slice(&qkv[base..base + dh]);
        hs.kh[t * dh..(t + 1) * dh].copy_from_slice(&qkv[base + d..base + d + dh]);
        hs.vh[t * dh..(t + 1) * dh].copy_from_slice(&qkv[base + 2 * d..base + 2 * d + dh]);
    }
}

/// One attention head over one window: `qh/kh/vh [S2][dhead]` pre-gathered, side `S`.
/// Writes `hs.outh[S2][dhead]` — the projection-free per-head context (the C
/// `attn_one_head`).
#[allow(clippy::too_many_arguments)]
fn attn_one_head<P: Mm>(
    p: &mut P,
    hs: &mut HeadScratch,
    rh: &[f16],
    rw: &[f16],
    s: usize,
    dh: usize,
    scale: f32,
    ip: &[usize],
    jp: &[usize],
) -> Result<()> {
    let s2 = s * s;
    let kpad = round32(s2);
    // pad keys to Kpad rows (zero) and v^T to Kpad cols (zero)
    hs.khp[..kpad * dh].fill(f16::ZERO);
    hs.khp[..s2 * dh].copy_from_slice(&hs.kh[..s2 * dh]);
    hs.vtp[..dh * kpad].fill(f16::ZERO);
    for t in 0..s2 {
        for c in 0..dh {
            hs.vtp[c * kpad + t] = hs.vh[t * dh + c];
        }
    }
    // scores = qh · khp^T  [S2][Kpad] (unscaled; the scale is folded in the softmax).
    // The device entries require exactly M*K / N*K / M*N elements, so every operand is
    // the live prefix of its scratch buffer.
    p.mm(
        s2,
        dh,
        kpad,
        &hs.qh[..s2 * dh],
        &hs.khp[..kpad * dh],
        &mut hs.sc[..s2 * kpad],
    )?;
    rel_pos(hs, rh, rw, s, dh);
    for t in 0..s2 {
        let (relh_t, relw_t) = (&hs.relh[t * s..t * s + s], &hs.relw[t * s..t * s + s]);
        attn_softmax_row(
            &mut hs.sc[t * kpad..(t + 1) * kpad],
            s2,
            kpad,
            scale,
            relh_t,
            relw_t,
            ip,
            jp,
        );
    }
    // context = P · v  [S2][dhead] = sc · vtp^T
    p.mm(
        s2,
        kpad,
        dh,
        &hs.sc[..s2 * kpad],
        &hs.vtp[..dh * kpad],
        &mut hs.outh[..s2 * dh],
    )
}

// ── The per-layer body and the neck ──────────────────────────────────────────

/// The resident weights of one layer (packed once by [`SamContext`]).
struct LayerResident {
    wqkv: Weight,
    wo: Weight,
    wf1: Weight,
    wf2: Weight,
}

/// The shared per-layer body: attention + FFN, updating `x` in place (the C `sam_layer`
/// / `ctx_layer`). `p` provides every matmul (projections + the computed QK/PV).
fn sam_layer<P: Mm>(
    p: &mut P,
    m: &SamModel,
    l: usize,
    x: &mut [f16],
    s: &mut AttnScratch,
    hs: &mut HeadScratch,
    res: Option<&LayerResident>,
) -> Result<()> {
    let g = &m.geom;
    let (d, dff, nh, dh, gg) = (g.d, g.d_ff, g.n_head, g.dhead, g.tokens());
    let w = &m.layers[l];
    let side = m.side(l);
    let s2 = side * side;
    let scale = 1.0 / (dh as f32).sqrt();
    let (wqkv_res, wo_res, wf1_res, wf2_res) = match res {
        Some(r) => (Some(&r.wqkv), Some(&r.wo), Some(&r.wf1), Some(&r.wf2)),
        None => (None, None, None, None),
    };

    let mut ln = vec![f16::ZERO; gg * d];
    let mut y = vec![f16::ZERO; gg * dff.max(d)];

    // --- attention: x += o( attn( LN1(x) ) ) ---
    h_layernorm(gg, d, x, &w.ln1_g, &w.ln1_b, g.eps, &mut ln);

    let (n_win, mtok, src): (usize, usize, &[f16]) = if w.windowed {
        let n = window_partition(&ln, g.grid, g.win, d, &mut s.hw);
        (n, n * s2, &s.hw[..n * s2 * d])
    } else {
        (1, gg, &ln[..gg * d])
    };
    // fused qkv: [Mtok][3d]
    p.proj(
        mtok,
        d,
        3 * d,
        src,
        &w.wqkv,
        wqkv_res,
        &mut s.qkv[..mtok * 3 * d],
    )?;
    add_bias(mtok, 3 * d, &mut s.qkv[..mtok * 3 * d], &w.bqkv);
    for t in 0..s2 {
        s.ip[t] = t / side;
        s.jp[t] = t % side;
    }

    // per window, per head: attention with rel-pos; scatter into cc (window-major)
    let mut cc = vec![f16::ZERO; mtok * d];
    for win in 0..n_win {
        for h in 0..nh {
            gather_head(hs, &s.qkv, win, h, s2, d, dh);
            attn_one_head(p, hs, &w.rh, &w.rw, side, dh, scale, &s.ip, &s.jp)?;
            for t in 0..s2 {
                let dst = (win * s2 + t) * d + h * dh;
                cc[dst..dst + dh].copy_from_slice(&hs.outh[t * dh..(t + 1) * dh]);
            }
        }
    }

    // un-partition (windowed) then o-proj at M = grid*grid
    let ctxf: &[f16] = if w.windowed {
        window_unpartition(&cc, g.grid, g.win, d, &mut s.ctxf);
        &s.ctxf[..gg * d]
    } else {
        &cc[..gg * d]
    };
    p.proj(gg, d, d, ctxf, &w.wo, wo_res, &mut y[..gg * d])?;
    add_bias(gg, d, &mut y[..gg * d], &w.bo);
    residual(x, &y[..gg * d]);

    // --- FFN: x += fc2( gelu( fc1( LN2(x) ) ) ) ---
    h_layernorm(gg, d, x, &w.ln2_g, &w.ln2_b, g.eps, &mut ln);
    p.proj(gg, d, dff, &ln, &w.wf1, wf1_res, &mut y[..gg * dff])?;
    add_bias(gg, dff, &mut y[..gg * dff], &w.bf1);
    erf_gelu_in_place(&mut y[..gg * dff]);
    p.proj(gg, dff, d, &y[..gg * dff], &w.wf2, wf2_res, &mut ln)?;
    add_bias(gg, d, &mut ln, &w.bf2);
    residual(x, &ln);
    Ok(())
}

/// The neck: `x_final[gg][d]` token-major → `out[neck_out][grid][grid]` (the C
/// `sam_neck`): conv1x1 → channels-first LN → conv3x3 pad1 → channels-first LN.
fn sam_neck<P: Mm>(
    p: &mut P,
    m: &SamModel,
    xf: &[f16],
    c1_res: Option<&Weight>,
    out: &mut [f16],
) -> Result<()> {
    let g = &m.geom;
    let (d, no, side, gg) = (g.d, g.neck_out, g.grid, g.tokens());
    let n = &m.neck;

    let mut y1 = vec![f16::ZERO; gg * no];
    let mut ncw = vec![f16::ZERO; no * gg];
    let mut ln = vec![f16::ZERO; no * gg];
    let mut c2 = vec![f16::ZERO; no * gg];

    // conv1 1x1: y1[gg][no] = xf[gg][d] · neck_c1[no][d]^T
    p.proj(gg, d, no, xf, &n.c1, c1_res, &mut y1)?;
    // -> channels-first [no][gg]
    for pos in 0..gg {
        for c in 0..no {
            ncw[c * gg + pos] = y1[pos * no + c];
        }
    }
    h_layernorm2d(no, gg, &ncw, &n.ln1_g, &n.ln1_b, 1e-6, &mut ln);
    // conv2 3x3 pad1, no bias
    let desc = Conv2dDesc {
        ic: no,
        ih: side,
        iw: side,
        oc: no,
        kh: 3,
        kw: 3,
        stride_y: 1,
        stride_x: 1,
        pad_top: 1,
        pad_left: 1,
        dil_y: 1,
        dil_x: 1,
        depthwise: false,
        ..Default::default()
    };
    p.conv(&desc, &ln, &n.c2, &mut c2)?;
    h_layernorm2d(no, gg, &c2, &n.ln2_g, &n.ln2_b, 1e-6, out);
    Ok(())
}

/// Check the caller's buffers (the C's `!m || !pixels || !out` plus a length check).
fn check_encode(m: &SamModel, pixels: &[f16], out: &[f16], hidden: Option<&[f16]>) -> Result<()> {
    let g = &m.geom;
    let pix = g.ic * g.image_size * g.image_size;
    let nn = g.neck_out * g.tokens();
    let hn = (g.n_layers + 1) * g.tokens() * g.d;
    if pixels.len() < pix || out.len() < nn || hidden.is_some_and(|h| h.len() < hn) {
        return Err(
            Error::shape("rocket_sam_encode/buffers").with_detail(format!(
                "pixels {} (need {pix}) out {} (need {nn}) hidden (need {hn})",
                pixels.len(),
                out.len()
            )),
        );
    }
    Ok(())
}

/// The resident weights of a whole model (what the ctx packs once).
struct ResidentRef<'a> {
    layers: &'a [LayerResident],
    patch: &'a Weight,
    neck_c1: &'a Weight,
}

/// The stem + the layer loop + the neck, shared by every arm (the C `rocket_sam_encode`
/// body): the caller supplies the provider and the optional resident weights.
fn encode_common<P: Mm>(
    p: &mut P,
    m: &SamModel,
    pixels: &[f16],
    out: &mut [f16],
    mut hidden: Option<&mut [f16]>,
    resident: Option<&ResidentRef<'_>>,
) -> Result<()> {
    check_encode(m, pixels, out, hidden.as_deref())?;
    let g = &m.geom;
    let d = g.d;
    let gg = g.tokens();
    let gd = gg * d;

    let mut s = AttnScratch::new(g);
    // One head scratch per encode (sized for the larger of the window and the grid):
    // the C's one-shot re-allocates it per layer and its resident path keeps one per
    // worker; hoisting it is numerically identical and avoids re-zeroing ~S2max*Kpad
    // fp16 per layer.
    let mut hs = HeadScratch::new(g);
    let mut patches = vec![f16::ZERO; gg * g.patch_dim];
    let mut x = vec![f16::ZERO; gd];

    // stem: im2col -> patch proj -> + patch_b + pos
    im2col(g, pixels, &mut patches);
    p.proj(
        gg,
        g.patch_dim,
        d,
        &patches,
        &m.patch_w,
        resident.map(|r| r.patch),
        &mut x,
    )?;
    for pos in 0..gg {
        let xr = &mut x[pos * d..pos * d + d];
        let pr = &m.pos[pos * d..pos * d + d];
        for j in 0..d {
            xr[j] = f16::from_f32(xr[j].to_f32() + m.patch_b[j].to_f32() + pr[j].to_f32());
        }
    }
    if let Some(hid) = hidden.as_deref_mut() {
        hid[..gd].copy_from_slice(&x);
    }

    for l in 0..g.n_layers {
        sam_layer(
            p,
            m,
            l,
            &mut x,
            &mut s,
            &mut hs,
            resident.map(|r| &r.layers[l]),
        )?;
        if let Some(hid) = hidden.as_deref_mut() {
            hid[(l + 1) * gd..(l + 2) * gd].copy_from_slice(&x);
        }
    }
    sam_neck(p, m, &x, resident.map(|r| r.neck_c1), out)
}

// ── The public entry points ──────────────────────────────────────────────────

/// `rocket_sam_encode(fd >= 0, ...)`: encode one image on the NPU (one-shot — every
/// call re-packs the static weights).
///
/// `pixels` is the preprocessed input `[ic][image_size][image_size]` fp16; `out`
/// receives the neck output `[neck_out][grid][grid]` (the ONNX `last_hidden_state`
/// `[1,neck_out,grid,grid]`). `hidden` (when given, at least
/// `(n_layers+1) * grid*grid * d`) receives the intermediate token grids, index 0 =
/// the embeddings, index `k` = the output of encoder layer `k-1`.
pub fn sam_encode(
    dev: &Device,
    m: &SamModel,
    pixels: &[f16],
    out: &mut [f16],
    hidden: Option<&mut [f16]>,
) -> Result<()> {
    encode_common(&mut DevMm { dev }, m, pixels, out, hidden, None)
}

/// `rocket_sam_encode(fd < 0, ...)`: the exact host reference.
///
/// Bit-identical to the C: an f32-accumulate host matmul, the two-pass f64 LayerNorms,
/// the fp16 erf-GELU table, the fp32 softmax row with the rel-pos bias, and
/// [`super::conv::conv2d_ref`] for the neck's 3x3.
pub fn sam_encode_ref(
    m: &SamModel,
    pixels: &[f16],
    out: &mut [f16],
    hidden: Option<&mut [f16]>,
) -> Result<()> {
    encode_common(&mut HostMm, m, pixels, out, hidden, None)
}

/// `rocket_sam_load`: load a weight blob (see [`SamModel::load`]).
pub fn sam_load(path: impl AsRef<Path>) -> Result<SamModel> {
    SamModel::load(path)
}

/// The resident (prepacked, multicore) SAM encoder — the latency path.
///
/// Packs the static GEMM weights (patch, per-layer qkv/o/fc1/fc2, neck conv1) into
/// resident NPU BOs once at create and reuses them across images; per image only the
/// activations are packed. Bit-faithful to [`sam_encode`] within fp16.
pub struct SamContext<'m> {
    model: &'m SamModel,
    mm: Context,
    /// The neck conv's device (the C `aux_fd`).
    dev: Device,
    nworkers: usize,
    w_patch: Weight,
    w_neck_c1: Weight,
    layers: Vec<LayerResident>,
}

impl<'m> SamContext<'m> {
    /// `rocket_sam_ctx_create`: pack every static weight. `nthreads` is clamped to
    /// `1..=8` (the C `SAM_MAX_WORKERS`).
    pub fn create(model: &'m SamModel, nthreads: usize) -> Result<SamContext<'m>> {
        model.validate("rocket_sam_ctx_create")?;
        let g = &model.geom;
        let nworkers = nthreads.clamp(1, 8);
        let mut mm = Context::new(nworkers)?;
        let dev = Device::open()?;
        let gg = g.tokens();
        let win_tokens = model.window_tokens();

        let w_patch = mm.pack_weight(gg, g.patch_dim, g.d, &model.patch_w)?;
        let w_neck_c1 = mm.pack_weight(gg, g.d, g.neck_out, &model.neck.c1)?;
        let mut layers = Vec::with_capacity(g.n_layers);
        for l in 0..g.n_layers {
            let w = &model.layers[l];
            // The fused qkv runs over the window partition (or the whole grid), so it is
            // packed at the M the attention will call it with (the resident layout needs
            // the pack-M and call-M plans to agree).
            let mqkv = if w.windowed { win_tokens } else { gg };
            layers.push(LayerResident {
                wqkv: mm.pack_weight(mqkv, g.d, 3 * g.d, &w.wqkv)?,
                wo: mm.pack_weight(gg, g.d, g.d, &w.wo)?,
                wf1: mm.pack_weight(gg, g.d, g.d_ff, &w.wf1)?,
                wf2: mm.pack_weight(gg, g.d_ff, g.d, &w.wf2)?,
            });
        }
        Ok(SamContext {
            model,
            mm,
            dev,
            nworkers,
            w_patch,
            w_neck_c1,
            layers,
        })
    }

    /// The worker count the context was created with (clamped `1..=8`).
    pub fn nthreads(&self) -> usize {
        self.nworkers
    }

    /// The model this context packs.
    pub fn model(&self) -> &'m SamModel {
        self.model
    }

    /// `rocket_sam_encode_ctx`: encode one image with the resident weights.
    pub fn encode(
        &mut self,
        pixels: &[f16],
        out: &mut [f16],
        hidden: Option<&mut [f16]>,
    ) -> Result<()> {
        let model = self.model;
        let mut p = CtxMm {
            ctx: &mut self.mm,
            dev: &self.dev,
        };
        let resident = ResidentRef {
            layers: &self.layers,
            patch: &self.w_patch,
            neck_c1: &self.w_neck_c1,
        };
        encode_common(&mut p, model, pixels, out, hidden, Some(&resident))
    }
}

/// `rocket_sam_encode_ctx`: the free-function form of [`SamContext::encode`].
pub fn sam_encode_ctx(
    c: &mut SamContext<'_>,
    pixels: &[f16],
    out: &mut [f16],
    hidden: Option<&mut [f16]>,
) -> Result<()> {
    c.encode(pixels, out, hidden)
}

#[cfg(test)]
mod tests {
    use super::*;

    use super::lcg_f16;

    /// The synthetic geometry (the shared fixture).
    fn geom() -> SamGeometry {
        SamGeometry::fixture()
    }

    #[test]
    fn blob_round_trip_is_lossless() {
        let m = SamModel::synthetic(geom(), 7);
        let blob = m.to_blob();
        let back = SamModel::from_blob(&blob).unwrap();
        assert_eq!(back.geom, m.geom);
        assert_eq!(back.patch_w, m.patch_w);
        assert_eq!(back.pos, m.pos);
        assert_eq!(back.layers.len(), m.layers.len());
        for (a, b) in back.layers.iter().zip(&m.layers) {
            assert_eq!(a.windowed, b.windowed);
            assert_eq!(a.wqkv, b.wqkv);
            assert_eq!(a.rh, b.rh);
            assert_eq!(a.rw, b.rw);
        }
        assert_eq!(back.neck.c2, m.neck.c2);
    }

    #[test]
    fn loader_error_classes_match_the_c() {
        // -2: shorter than the 64-byte header.
        let e = SamModel::from_blob(&[0u8; 63]).unwrap_err();
        assert_eq!(e.c_code(), -2);
        // -3: bad magic.
        let m = SamModel::synthetic(geom(), 1);
        let mut blob = m.to_blob();
        blob[0] ^= 0xff;
        assert_eq!(SamModel::from_blob(&blob).unwrap_err().c_code(), -3);
        // -3: bad version.
        let mut blob = m.to_blob();
        blob[4] = 9;
        assert_eq!(SamModel::from_blob(&blob).unwrap_err().c_code(), -3);
        // -4: an inconsistent dimension (dhead != d / n_head).
        let mut blob = m.to_blob();
        let bad = (13 * 4) as usize;
        blob[bad..bad + 4].copy_from_slice(&(13i32).to_le_bytes());
        assert_eq!(SamModel::from_blob(&blob).unwrap_err().c_code(), -4);
        // -4: a truncated weight region (the C's exact-size check).
        let blen = blob.len();
        assert_eq!(
            SamModel::from_blob(&blob[..blen - 2]).unwrap_err().c_code(),
            -4
        );
        // -1: a missing file.
        let e = SamModel::load("/nonexistent/sam_weights.f16").unwrap_err();
        assert_eq!(e.c_code(), -1);
    }

    #[test]
    fn host_reference_is_deterministic_and_uses_the_whole_graph() {
        let m = SamModel::synthetic(geom(), 11);
        let g = &m.geom;
        let pix = lcg_f16(g.ic * g.image_size * g.image_size, -1.0, 1.0, 3);
        let mut a = vec![f16::ZERO; g.neck_out * g.tokens()];
        let mut b = vec![f16::ZERO; g.neck_out * g.tokens()];
        let mut ha = vec![f16::ZERO; (g.n_layers + 1) * g.tokens() * g.d];
        let mut hb = vec![f16::ZERO; (g.n_layers + 1) * g.tokens() * g.d];
        sam_encode_ref(&m, &pix, &mut a, Some(&mut ha)).unwrap();
        sam_encode_ref(&m, &pix, &mut b, Some(&mut hb)).unwrap();
        assert_eq!(a, b, "the reference must be deterministic");
        assert_eq!(ha, hb);
        assert!(a.iter().all(|v| v.to_f32().is_finite()));
        // The windowed layer only mixes inside its window, so the embedding block must
        // still be bit-identical to the patch+pos stem.
        let gd = g.tokens() * g.d;
        assert_eq!(&ha[..gd], &ha[..gd]);
        // Layer 0 is windowed, layer 1 global: their outputs differ.
        assert_ne!(&ha[gd..2 * gd], &ha[2 * gd..3 * gd]);
        // ... and the neck is not the identity of the last hidden block.
        assert_ne!(&a[..16], &ha[2 * gd..2 * gd + 16]);
    }

    #[test]
    fn the_windowed_layer_only_mixes_inside_its_window() {
        // With a single windowed layer whose o-projection and MLP are zero, the attention
        // context of a token can only come from its own window: check the window
        // partition/unpartition mapping directly instead (cheaper and exact).
        let (grid, win, d) = (8usize, 4usize, 4usize);
        let mut h = vec![f16::ZERO; grid * grid * d];
        for t in 0..grid * grid {
            for c in 0..d {
                h[t * d + c] = f16::from_f32((t * d + c) as f32);
            }
        }
        let mut hw = vec![f16::ZERO; grid * grid * d];
        let n = window_partition(&h, grid, win, d, &mut hw);
        assert_eq!(n, 4);
        // window 0 holds rows 0-3 x cols 0-3 of the grid, raster order
        for i in 0..win {
            for j in 0..win {
                let src = (i * grid + j) * d;
                let dst = (i * win + j) * d;
                assert_eq!(&hw[dst..dst + d], &h[src..src + d]);
            }
        }
        // window 3 (bottom-right) starts at grid row 4, col 4
        let base = 3 * win * win * d;
        let src = (4 * grid + 4) * d;
        assert_eq!(&hw[base..base + d], &h[src..src + d]);
        // and the round trip is the identity
        let mut back = vec![f16::ZERO; grid * grid * d];
        window_unpartition(&hw, grid, win, d, &mut back);
        assert_eq!(back, h);
    }

    #[test]
    fn a_padded_grid_keeps_the_pad_tokens_inside_their_windows() {
        // grid 6, win 4 -> gp 8, nWin 4, S2 16: the pad rows/cols stay zero.
        let (grid, win, d) = (6usize, 4usize, 4usize);
        let h = lcg_f16(grid * grid * d, -1.0, 1.0, 5);
        let mut hw = vec![f16::from_f32(9.0); 4 * win * win * d];
        let n = window_partition(&h, grid, win, d, &mut hw);
        assert_eq!(n, 4);
        // Window 1 (rows 0-3, cols 4-7): cols 6,7 are pad -> zero.
        for i in 0..win {
            for j in 2..win {
                let dst = (win * win + i * win + j) * d;
                assert!(hw[dst..dst + d].iter().all(|v| *v == f16::ZERO));
            }
        }
        // The round trip drops the pad tokens and reproduces the grid exactly.
        let mut back = vec![f16::ZERO; grid * grid * d];
        window_unpartition(&hw, grid, win, d, &mut back);
        assert_eq!(back, h);
    }

    #[test]
    fn rel_pos_tables_are_indexed_by_distance() {
        // qh = all ones -> relh[i][j][i'] = sum_c Rh[i][i'][c]; check row by row.
        let (s, dh) = (4usize, 2usize);
        let mut hs = HeadScratch::new(&geom());
        hs.qh[..s * s * dh].fill(f16::ONE);
        let rh = lcg_f16(s * s * dh, -1.0, 1.0, 13);
        let rw = lcg_f16(s * s * dh, -1.0, 1.0, 17);
        rel_pos(&mut hs, &rh, &rw, s, dh);
        for i in 0..s {
            for j in 0..s {
                for ip in 0..s {
                    let mut want = 0.0f32;
                    for c in 0..dh {
                        want += rh[(i * s + ip) * dh + c].to_f32();
                    }
                    let got = hs.relh[(i * s + j) * s + ip].to_f32();
                    assert!(
                        (got - want).abs() < 2e-2,
                        "relh {i} {j} {ip}: {got} vs {want}"
                    );
                }
                for jp in 0..s {
                    let mut want = 0.0f32;
                    for c in 0..dh {
                        want += rw[(j * s + jp) * dh + c].to_f32();
                    }
                    let got = hs.relw[(i * s + j) * s + jp].to_f32();
                    assert!(
                        (got - want).abs() < 2e-2,
                        "relw {i} {j} {jp}: {got} vs {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_biased_softmax_matches_its_definition() {
        let (s2, kpad) = (4usize, 32usize);
        let sc = lcg_f16(s2, -2.0, 2.0, 21);
        let relh = lcg_f16(s2, -0.5, 0.5, 22);
        let relw = lcg_f16(s2, -0.5, 0.5, 23);
        let ip: Vec<usize> = (0..s2).map(|t| t / 2).collect();
        let jp: Vec<usize> = (0..s2).map(|t| t % 2).collect();
        let mut row = vec![f16::ZERO; kpad];
        row[..s2].copy_from_slice(&sc);
        attn_softmax_row(&mut row, s2, kpad, 0.5, &relh, &relw, &ip, &jp);
        // The definition: f32 logits, f16-rounded, then the f32 exponentials of the
        // *rounded* logits, and 1/sum of the *unrounded* ones.
        let mut logits = vec![0.0f32; s2];
        for t in 0..s2 {
            let v = sc[t].to_f32() * 0.5 + relh[ip[t]].to_f32() + relw[jp[t]].to_f32();
            logits[t] = f16::from_f32(v).to_f32();
        }
        let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|v| (v - mx).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for t in 0..s2 {
            let want = f16::from_f32(f16::from_f32(exps[t]).to_f32() * (1.0 / sum));
            assert!((row[t].to_f32() - want.to_f32()).abs() <= 1e-4);
        }
        assert!(
            row[s2..].iter().all(|v| *v == f16::ZERO),
            "the pad columns must be zero"
        );
        let total: f32 = row[..s2].iter().map(|v| v.to_f32()).sum();
        assert!((total - 1.0).abs() < 2e-3, "softmax sums to {total}");
    }

    #[test]
    fn the_erf_gelu_table_is_the_float_formula() {
        let lut = gelu_erf_lut();
        for &v in &[-3.0f32, -1.5, -0.5, -0.0, 0.0, 0.25, 1.0, 2.0, 5.0, 40.0] {
            let h = f16::from_f32(v);
            let want = f16::from_f32(0.5 * v * (1.0 + erff32(v * INV_SQRT_2)));
            assert_eq!(lut[h.to_bits() as usize], want, "x={v}");
        }
        // The table covers every fp16 bit pattern, so it is a pure lookup: apply twice.
        let mut x = lcg_f16(64, -4.0, 4.0, 31);
        let once = {
            let mut y = x.clone();
            erf_gelu_in_place(&mut y);
            y
        };
        erf_gelu_in_place(&mut x);
        assert_eq!(x, once);
    }

    #[test]
    fn encode_rejects_short_buffers_before_any_work() {
        let m = SamModel::synthetic(geom(), 3);
        let g = &m.geom;
        let pix = vec![f16::ZERO; g.ic * g.image_size * g.image_size];
        let mut out = vec![f16::ZERO; g.neck_out * g.tokens()];
        let mut short = vec![f16::ZERO; g.neck_out * g.tokens() - 1];
        assert!(sam_encode_ref(&m, &pix, &mut short, None).is_err());
        let mut hid = vec![f16::ZERO; (g.n_layers + 1) * g.tokens() * g.d - 1];
        assert!(sam_encode_ref(&m, &pix, &mut out, Some(&mut hid)).is_err());
        let shortpix = vec![f16::ZERO; 3];
        assert!(sam_encode_ref(&m, &shortpix, &mut out, None).is_err());
    }

    #[test]
    fn a_hand_built_model_is_validated() {
        let g = geom();
        let bad = SamGeometry { d: 48, ..g };
        let m = SamModel::synthetic(geom(), 5);
        let e = SamModel::from_parts(
            bad,
            m.patch_w.clone(),
            m.patch_b.clone(),
            m.pos.clone(),
            m.layers.clone(),
            m.neck.clone(),
        )
        .unwrap_err();
        assert_eq!(e.c_code(), -4);
    }
}
