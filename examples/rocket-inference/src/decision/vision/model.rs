//! The Qwen3.5 vision tower (d3 image path), f32 inference only.
//!
//! One image = patch embeddings + interpolated position table, then `depth`
//! pre-norm blocks with **full** (non-causal) self-attention over the image's
//! own patches, then a 2x2 patch merger. `merge_size^2` consecutive patches (in
//! merge-block order, which is what the processor emits) become one image token.
//!
//! Reference: `transformers/models/qwen3_5/modeling_qwen3_5.py`
//! (`Qwen3_5VisionModel` and friends) and `transformers/vision_utils.py`
//! (`get_vision_position_ids`, `get_vision_interpolation_indices_and_weights`):
//!
//! * patch embedding is the Conv3d whose kernel equals its stride, i.e. the
//!   plain `pixel_values @ W^T + b` the runtime linearizes it to;
//! * the learned `[48, 48]` position table is bilinearly resampled onto the
//!   image's `[grid_h, grid_w]` patch grid with `align_corners=True` and border
//!   padding, summed over the 4 taps (the reference's `(embed(indices) * weights)
//!   .sum(1)`);
//! * axial 2-D RoPE over the *whole* head: `head_dim / 4` frequencies per axis,
//!   angles `[h*f, w*f]` doubled to `head_dim`, `rotate_half` applied to q and k;
//! * per-image attention (each image is its own sequence, no mask), scale
//!   `1/sqrt(head_dim)`, `qkv` and `proj` with bias;
//! * MLP `fc1 -> gelu_pytorch_tanh -> fc2`, both with bias;
//! * merger: LayerNorm(hidden) per patch, `[patches/4, 4*hidden]` reshape in
//!   block order, `fc1 (4h->4h) -> gelu -> fc2 (4h->out_hidden)`.
//!
//! The two GELUs differ and both are load-bearing: the blocks use the
//! checkpoint's `hidden_act` (`gelu_pytorch_tanh`, the tanh approximation), while
//! `Qwen3_5VisionPatchMerger` hard-codes `nn.GELU()` — the exact **erf** form.
//! The two differ by up to 4.7e-4 per element (4.7e-4 at `|x| ~ 2.7`), which is
//! enough to move the pooled output by 1.3e-3 and the cosine off 1.
//!
//! Every norm is `nn.LayerNorm(..., eps=1e-6)` (see [`LAYER_NORM_EPS`]). The
//! tower runs in f32: a d3 checkpoint stores bf16 weights, and Phase 2a's gate is
//! an f32 reference.

use anyhow::{Result, bail};
use burn::prelude::*;
use burn::tensor::activation::{gelu, gelu_approximate};
use burn::tensor::module::attention;
use burn::tensor::ops::AttentionModuleOptions;
use burn::tensor::{Int, TensorData, s};

use super::config::{LAYER_NORM_EPS, VisionConfig};
use super::processor::{PreparedImage, PreparedVideo};

/// One image's (or one video's, per frame) axial RoPE tables: `(cos, sin)` of
/// shape `[patches, head_dim/2]`.
///
/// The reference builds `[cos(h*f) | cos(w*f)]` and doubles it over the head, so
/// the first half of the head sees the h angles and the second half the w angles
/// with the same values — one `head_dim/2` table per patch is all of it. A video
/// has no temporal axis in the tower: `get_vision_position_ids` repeats each
/// frame's `(h, w)` indices `t` times, so its table is the single-frame table
/// repeated once per sampled frame.
pub fn axial_rope_tables(
    grid: [usize; 3],
    config: &VisionConfig,
    device: &Device,
) -> (Tensor<2>, Tensor<2>) {
    let merge = config.spatial_merge_size;
    let head_dim = config.head_dim();
    let spatial = head_dim / 2;
    let n_freq = spatial / 2;
    // inv_freq = 1 / theta^(arange(0, spatial, 2) / spatial), computed in f32
    // like the torch buffer the reference multiplies with.
    let theta = config.rope_theta as f32;
    let inv_freq: Vec<f32> = (0..n_freq)
        .map(|i| 1.0 / theta.powf((2 * i) as f32 / spatial as f32))
        .collect();
    let (frames, grid_h, grid_w) = (grid[0], grid[1], grid[2]);
    let frame_patches = grid_h * grid_w;
    let patches = frames * frame_patches;
    let positions = patch_positions(grid_h, grid_w, merge);
    let mut cos = Vec::with_capacity(patches * spatial);
    let mut sin = Vec::with_capacity(patches * spatial);
    for _ in 0..frames {
        for (row, col) in &positions {
            for &freq in &inv_freq {
                let angle = *row as f32 * freq;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
            for &freq in &inv_freq {
                let angle = *col as f32 * freq;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
    }
    let cos = Tensor::<2>::from_data(TensorData::new(cos, [patches, spatial]), device);
    let sin = Tensor::<2>::from_data(TensorData::new(sin, [patches, spatial]), device);
    (cos, sin)
}

/// Patch `(row, col)` coordinates in merge-block order — the order both the
/// processor's patches and `get_vision_position_ids` use.
pub fn patch_positions(grid_h: usize, grid_w: usize, merge: usize) -> Vec<(usize, usize)> {
    let mut positions = Vec::with_capacity(grid_h * grid_w);
    for block_row in 0..grid_h / merge {
        for block_col in 0..grid_w / merge {
            for in_row in 0..merge {
                for in_col in 0..merge {
                    positions.push((block_row * merge + in_row, block_col * merge + in_col));
                }
            }
        }
    }
    positions
}

/// Per-axis bilinear taps and weights into a `side`-length table for `size`
/// target positions (`align_corners=True`, border padding):
/// `(floor, min(floor+1, side-1), w_floor, w_ceil)`.
fn axis_taps(size: usize, side: usize) -> (Vec<usize>, Vec<usize>, Vec<f32>, Vec<f32>) {
    let denominator = size.saturating_sub(1).max(1) as f32;
    let mut floor = Vec::with_capacity(size);
    let mut ceil = Vec::with_capacity(size);
    let mut w_floor = Vec::with_capacity(size);
    let mut w_ceil = Vec::with_capacity(size);
    for index in 0..size {
        let src = index as f32 * (side.saturating_sub(1)) as f32 / denominator;
        let base = src.floor();
        let frac = src - base;
        let base = base as usize;
        floor.push(base.min(side - 1));
        ceil.push((base + 1).min(side - 1));
        w_floor.push((1.0 - frac).max(0.0));
        w_ceil.push(frac.max(0.0));
    }
    (floor, ceil, w_floor, w_ceil)
}

/// The four `(table_index, weight)` taps of every patch, patch-major in
/// merge-block order, in the reference's `(h_tap outer, w_tap inner)` order.
pub fn position_taps(
    grid_h: usize,
    grid_w: usize,
    side: usize,
    merge: usize,
) -> ([Vec<i32>; 4], [Vec<f32>; 4]) {
    let (h0, h1, hw0, hw1) = axis_taps(grid_h, side);
    let (w0, w1, ww0, ww1) = axis_taps(grid_w, side);
    let patches = grid_h * grid_w;
    let mut indices = [
        Vec::with_capacity(patches),
        Vec::with_capacity(patches),
        Vec::with_capacity(patches),
        Vec::with_capacity(patches),
    ];
    let mut weights = [
        Vec::with_capacity(patches),
        Vec::with_capacity(patches),
        Vec::with_capacity(patches),
        Vec::with_capacity(patches),
    ];
    for (row, col) in patch_positions(grid_h, grid_w, merge) {
        indices[0].push((h0[row] * side + w0[col]) as i32);
        indices[1].push((h0[row] * side + w1[col]) as i32);
        indices[2].push((h1[row] * side + w0[col]) as i32);
        indices[3].push((h1[row] * side + w1[col]) as i32);
        weights[0].push(hw0[row] * ww0[col]);
        weights[1].push(hw0[row] * ww1[col]);
        weights[2].push(hw1[row] * ww0[col]);
        weights[3].push(hw1[row] * ww1[col]);
    }
    (indices, weights)
}

/// `q @ cos + rotate_half(q) * sin` over the full head dim (`x` is
/// `[patches, heads, head_dim]`, `cos`/`sin` are `[patches, head_dim/2]`).
fn apply_axial_rope(x: Tensor<3>, cos: &Tensor<2>, sin: &Tensor<2>) -> Tensor<3> {
    let [patches, heads, head_dim] = x.dims();
    let half = head_dim / 2;
    let x1 = x.clone().slice(s![.., .., 0..half]);
    let x2 = x.slice(s![.., .., half..head_dim]);
    let cos = cos.clone().reshape([patches, 1, half]);
    let sin = sin.clone().reshape([patches, 1, half]);
    let y1 = x1.clone() * cos.clone() - x2.clone() * sin.clone();
    let y2 = x2 * cos + x1 * sin;
    Tensor::cat(vec![y1, y2], 2).reshape([patches, heads, head_dim])
}

/// `y = x @ W + b` with `W` stored `[in, out]`.
fn linear(x: Tensor<2>, weight: &Tensor<2>, bias: &Tensor<1>) -> Tensor<2> {
    let [_, out] = weight.dims();
    x.matmul(weight.clone()) + bias.clone().reshape([1, out])
}

/// `nn.LayerNorm(hidden, eps=1e-6)`: f32 mean/variance over the last dim.
fn layer_norm(x: Tensor<2>, weight: &Tensor<1>, bias: &Tensor<1>) -> Tensor<2> {
    let [_, hidden] = x.dims();
    let mean = x.clone().mean_dim(1);
    let centered = x - mean;
    let var = centered.clone().square().mean_dim(1);
    let normed = centered / (var + LAYER_NORM_EPS).sqrt();
    normed * weight.clone().reshape([1, hidden]) + bias.clone().reshape([1, hidden])
}

#[derive(Debug)]
struct VisionBlock {
    /// `[hidden]`, `[hidden]`.
    norm1_w: Tensor<1>,
    norm1_b: Tensor<1>,
    /// `[hidden, 3 * hidden]`, `[3 * hidden]`.
    qkv_w: Tensor<2>,
    qkv_b: Tensor<1>,
    /// `[hidden, hidden]`, `[hidden]`.
    proj_w: Tensor<2>,
    proj_b: Tensor<1>,
    norm2_w: Tensor<1>,
    norm2_b: Tensor<1>,
    /// `[hidden, intermediate]`, `[intermediate]`.
    fc1_w: Tensor<2>,
    fc1_b: Tensor<1>,
    /// `[intermediate, hidden]`, `[hidden]`.
    fc2_w: Tensor<2>,
    fc2_b: Tensor<1>,
}

#[derive(Debug)]
struct PatchMerger {
    norm_w: Tensor<1>,
    norm_b: Tensor<1>,
    /// `[merge_unit * hidden, merge_unit * hidden]`, same width.
    fc1_w: Tensor<2>,
    fc1_b: Tensor<1>,
    /// `[merge_unit * hidden, out_hidden]`, `[out_hidden]`.
    fc2_w: Tensor<2>,
    fc2_b: Tensor<1>,
}

/// The vision tower: patch embedding + position table, the blocks, the merger.
#[derive(Debug)]
pub struct VisionTower {
    pub config: VisionConfig,
    /// `[patch_dim, hidden]`, `[hidden]`.
    patch_w: Tensor<2>,
    patch_b: Tensor<1>,
    /// `[num_position_embeddings, hidden]`.
    position_table: Tensor<2>,
    blocks: Vec<VisionBlock>,
    merger: PatchMerger,
}

impl VisionTower {
    /// A zero-initialized tower of this geometry (the loader replaces every
    /// tensor; the tests fill the ones they exercise).
    pub fn new_stub(config: VisionConfig, device: &Device) -> Self {
        let hidden = config.hidden_size;
        let inter = config.intermediate_size;
        let zeros1 = |n: usize| Tensor::<1>::zeros([n], device);
        let zeros2 = |a: usize, b: usize| Tensor::<2>::zeros([a, b], device);
        let merged = config.merge_unit() * hidden;
        let blocks = (0..config.depth)
            .map(|_| VisionBlock {
                norm1_w: zeros1(hidden),
                norm1_b: zeros1(hidden),
                qkv_w: zeros2(hidden, 3 * hidden),
                qkv_b: zeros1(3 * hidden),
                proj_w: zeros2(hidden, hidden),
                proj_b: zeros1(hidden),
                norm2_w: zeros1(hidden),
                norm2_b: zeros1(hidden),
                fc1_w: zeros2(hidden, inter),
                fc1_b: zeros1(inter),
                fc2_w: zeros2(inter, hidden),
                fc2_b: zeros1(hidden),
            })
            .collect();
        Self {
            patch_w: zeros2(config.patch_dim(), hidden),
            patch_b: zeros1(hidden),
            position_table: zeros2(config.num_position_embeddings, hidden),
            blocks,
            merger: PatchMerger {
                norm_w: zeros1(hidden),
                norm_b: zeros1(hidden),
                fc1_w: zeros2(merged, merged),
                fc1_b: zeros1(merged),
                fc2_w: zeros2(merged, config.out_hidden_size),
                fc2_b: zeros1(config.out_hidden_size),
            },
            config,
        }
    }

    /// Replace one `visual.*` tensor by its suffix (`patch_embed.proj.weight`,
    /// `blocks.3.mlp.linear_fc2.bias`, ...). Every weight is stored in the
    /// tower's own `[in, out]` orientation (the checkpoint's `[out, in]` is
    /// transposed by [`VisionTower::load`]).
    pub(crate) fn set(&mut self, suffix: &str, tensor: Tensor<2>) -> Result<()> {
        let hidden = self.config.hidden_size;
        let patch_dim = self.config.patch_dim();
        let positions = self.config.num_position_embeddings;
        let intermediate = self.config.intermediate_size;
        let out_hidden = self.config.out_hidden_size;
        let merged = self.config.merge_unit() * hidden;
        let as1 = |t: Tensor<2>| -> Result<Tensor<1>> {
            let [r, c] = t.dims();
            if r != 1 && c != 1 {
                bail!("expected a bias vector stored as [1, n] or [n, 1], got {r}x{c}");
            }
            Ok(t.reshape([r * c]))
        };
        match suffix {
            "patch_embed.proj.weight" => {
                let [rows, out] = tensor.dims();
                if rows != patch_dim || out != hidden {
                    bail!("patch_embed.proj.weight is {rows}x{out}, expected {patch_dim}x{hidden}");
                }
                self.patch_w = tensor;
            }
            "patch_embed.proj.bias" => self.patch_b = as1(tensor)?,
            "pos_embed.weight" => {
                let [rows, out] = tensor.dims();
                if rows != positions || out != hidden {
                    bail!("pos_embed.weight is {rows}x{out}, expected {positions}x{hidden}");
                }
                self.position_table = tensor;
            }
            "merger.norm.weight" => self.merger.norm_w = as1(tensor)?,
            "merger.norm.bias" => self.merger.norm_b = as1(tensor)?,
            "merger.linear_fc1.weight" => self.merger.fc1_w = check(tensor, merged, merged)?,
            "merger.linear_fc1.bias" => self.merger.fc1_b = check1(tensor, merged)?,
            "merger.linear_fc2.weight" => self.merger.fc2_w = check(tensor, merged, out_hidden)?,
            "merger.linear_fc2.bias" => self.merger.fc2_b = check1(tensor, out_hidden)?,
            other => {
                let (index, field) = other
                    .strip_prefix("blocks.")
                    .and_then(|rest| rest.split_once('.'))
                    .ok_or_else(|| anyhow::anyhow!("unknown vision tensor visual.{other}"))?;
                let index: usize = index
                    .parse()
                    .map_err(|_| anyhow::anyhow!("unknown vision tensor visual.{other}"))?;
                let block = self
                    .blocks
                    .get_mut(index)
                    .ok_or_else(|| anyhow::anyhow!("visual.{other}: block {index} out of range"))?;
                match field {
                    "norm1.weight" => block.norm1_w = as1(tensor)?,
                    "norm1.bias" => block.norm1_b = as1(tensor)?,
                    "norm2.weight" => block.norm2_w = as1(tensor)?,
                    "norm2.bias" => block.norm2_b = as1(tensor)?,
                    "attn.qkv.weight" => block.qkv_w = check(tensor, hidden, 3 * hidden)?,
                    "attn.qkv.bias" => block.qkv_b = check1(tensor, 3 * hidden)?,
                    "attn.proj.weight" => block.proj_w = check(tensor, hidden, hidden)?,
                    "attn.proj.bias" => block.proj_b = check1(tensor, hidden)?,
                    "mlp.linear_fc1.weight" => block.fc1_w = check(tensor, hidden, intermediate)?,
                    "mlp.linear_fc1.bias" => block.fc1_b = check1(tensor, intermediate)?,
                    "mlp.linear_fc2.weight" => block.fc2_w = check(tensor, intermediate, hidden)?,
                    "mlp.linear_fc2.bias" => block.fc2_b = check1(tensor, hidden)?,
                    _ => bail!("unknown vision tensor visual.{other}"),
                }
            }
        }
        Ok(())
    }

    /// Encode one preprocessed image into `[tokens, out_hidden]` merged tokens.
    pub fn encode(&self, image: &PreparedImage) -> Result<Tensor<2>> {
        let (grid_h, grid_w) = image.patch_grid();
        self.check_grid(
            image.pixel_values.clone(),
            [1, grid_h, grid_w],
            grid_h * grid_w,
            "image",
        )?;
        self.encode_grid(image.pixel_values.clone(), image.grid)
    }

    /// Encode one preprocessed video into `[tokens, out_hidden]` merged tokens.
    ///
    /// A video is a multi-frame grid `[T, H, W]`: the position table resamples each
    /// frame's `H x W` patches (`within % (H*W)`), the axial rope is the single
    /// frame's table repeated `T` times (there is no temporal axis), and every
    /// frame is its **own attention sequence** over `H*W` patches — the
    /// reference's `get_vision_cu_seqlens(grid_thw, merge_temporal=False)` makes
    /// `T` segments of `H*W` per video (measured: the whole-clip variant moves the
    /// pooled output to cosine 0.68 against the reference dump, the per-frame one
    /// reproduces it bit for bit). The merger then groups 2x2 patches within each
    /// frame, giving `T * H * W / merge^2` tokens.
    pub fn encode_video(&self, video: &PreparedVideo) -> Result<Tensor<2>> {
        let [frames, grid_h, grid_w] = video.grid;
        self.check_grid(
            video.pixel_values.clone(),
            video.grid,
            frames * grid_h * grid_w,
            "video",
        )?;
        self.encode_grid(video.pixel_values.clone(), video.grid)
    }

    /// Shape checks of one `[T, H, W]` patch grid.
    fn check_grid(
        &self,
        pixel_values: Tensor<2>,
        grid: [usize; 3],
        expected: usize,
        what: &str,
    ) -> Result<()> {
        let config = &self.config;
        let [patches, dim] = pixel_values.dims();
        let (frames, grid_h, grid_w) = (grid[0], grid[1], grid[2]);
        if dim != config.patch_dim() {
            bail!(
                "the {what} has {dim} values per patch, the tower takes {}",
                config.patch_dim()
            );
        }
        if frames == 0 || grid_h == 0 || grid_w == 0 {
            bail!("the {what} has an empty {grid:?} patch grid");
        }
        if patches != expected {
            bail!("{patches} patch rows do not match the {grid:?} grid");
        }
        let merge = config.spatial_merge_size;
        if !grid_h.is_multiple_of(merge) || !grid_w.is_multiple_of(merge) {
            bail!("the {grid_h}x{grid_w} patch grid is not divisible by the merge size {merge}");
        }
        Ok(())
    }

    /// The patch-embedding + position-table + blocks + merger forward over a
    /// `[T, H, W]` patch grid.
    fn encode_grid(&self, pixel_values: Tensor<2>, grid: [usize; 3]) -> Result<Tensor<2>> {
        let config = &self.config;
        let (frames, grid_h, grid_w) = (grid[0], grid[1], grid[2]);
        let frame_patches = grid_h * grid_w;
        let patches = frames * frame_patches;
        let merge = config.spatial_merge_size;
        let hidden = config.hidden_size;
        let heads = config.num_heads;
        let head_dim = config.head_dim();

        // Patch embedding + bilinear position table. A video resamples each frame's
        // grid, i.e. the single-frame taps repeat once per frame.
        let mut x = linear(pixel_values, &self.patch_w, &self.patch_b);
        let device = x.device();
        let (indices, weights) = position_taps(grid_h, grid_w, config.position_grid_side(), merge);
        let mut position: Option<Tensor<2>> = None;
        for tap in 0..4 {
            let mut index = indices[tap].clone();
            let mut weight = weights[tap].clone();
            for _ in 1..frames {
                index.extend_from_slice(&indices[tap]);
                weight.extend_from_slice(&weights[tap]);
            }
            debug_assert_eq!(index.len(), patches);
            let index = Tensor::<1, Int>::from_data(TensorData::new(index, [patches]), &device);
            let weight = Tensor::<1>::from_data(TensorData::new(weight, [patches]), &device);
            let row = self.position_table.clone().select(0, index) * weight.reshape([patches, 1]);
            position = Some(match position {
                Some(sum) => sum + row,
                None => row,
            });
        }
        if let Some(position) = position {
            x = x + position;
        }

        // Axial RoPE tables for this grid (repeated per frame for a video).
        let (cos, sin) = axial_rope_tables(grid, config, &x.device());
        let scale = (head_dim as f64).powf(-0.5);

        for block in &self.blocks {
            // Attention: pre-LN, q|k|v, rope, per-frame full attention, proj.
            let h = layer_norm(x.clone(), &block.norm1_w, &block.norm1_b);
            let qkv = linear(h, &block.qkv_w, &block.qkv_b).reshape([patches, 3, heads, head_dim]);
            let q = qkv
                .clone()
                .slice(s![.., 0..1, .., ..])
                .reshape([patches, heads, head_dim]);
            let k = qkv
                .clone()
                .slice(s![.., 1..2, .., ..])
                .reshape([patches, heads, head_dim]);
            let v = qkv
                .slice(s![.., 2..3, .., ..])
                .reshape([patches, heads, head_dim]);
            let q = apply_axial_rope(q, &cos, &sin);
            let k = apply_axial_rope(k, &cos, &sin);
            let mut frames_out = Vec::with_capacity(frames);
            for frame in 0..frames {
                let start = frame * frame_patches;
                let end = start + frame_patches;
                let qf = q
                    .clone()
                    .slice(s![start..end, .., ..])
                    .swap_dims(0, 1)
                    .unsqueeze_dim::<4>(0);
                let kf = k
                    .clone()
                    .slice(s![start..end, .., ..])
                    .swap_dims(0, 1)
                    .unsqueeze_dim::<4>(0);
                let vf = v
                    .clone()
                    .slice(s![start..end, .., ..])
                    .swap_dims(0, 1)
                    .unsqueeze_dim::<4>(0);
                frames_out.push(
                    attention(
                        qf,
                        kf,
                        vf,
                        None,
                        None,
                        AttentionModuleOptions {
                            scale: Some(scale),
                            softcap: None,
                            is_causal: false,
                        },
                    )
                    .reshape([heads, frame_patches, head_dim])
                    .swap_dims(0, 1)
                    .reshape([frame_patches, hidden]),
                );
            }
            let attn = if frames_out.len() == 1 {
                frames_out.remove(0)
            } else {
                Tensor::cat(frames_out, 0)
            };
            x = x + linear(attn, &block.proj_w, &block.proj_b);

            // MLP: pre-LN, fc1, gelu_pytorch_tanh, fc2.
            let h = layer_norm(x.clone(), &block.norm2_w, &block.norm2_b);
            let h = linear(h, &block.fc1_w, &block.fc1_b);
            let h = gelu_approximate(h);
            x = x + linear(h, &block.fc2_w, &block.fc2_b);
        }

        // Merger: LayerNorm per patch, merge-block reshape, fc1, exact GELU, fc2.
        //
        // The reference merger is `nn.GELU()` (the **erf** form), unlike the
        // blocks' `gelu_pytorch_tanh` (`config.hidden_act`): the two differ by up
        // to 4.7e-4 per element (9.8e-5 at x = 2), which was the whole pooled
        // residual of the first cut.
        let h = layer_norm(x, &self.merger.norm_w, &self.merger.norm_b);
        let tokens = config.merged_tokens(grid_h, grid_w) * frames;
        let h = h.reshape([tokens, config.merge_unit() * hidden]);
        let h = linear(h, &self.merger.fc1_w, &self.merger.fc1_b);
        let h = gelu(h);
        Ok(linear(h, &self.merger.fc2_w, &self.merger.fc2_b))
    }

    /// Encode several images, concatenating their merged tokens in image order
    /// (the reference's `pixel_values`/`pooler_output` contract).
    pub fn encode_all(&self, images: &[PreparedImage]) -> Result<Tensor<2>> {
        if images.is_empty() {
            bail!("encode_all takes at least one image");
        }
        let mut rows = Vec::with_capacity(images.len());
        for image in images {
            rows.push(self.encode(image)?);
        }
        Ok(if rows.len() == 1 {
            rows.remove(0)
        } else {
            Tensor::cat(rows, 0)
        })
    }
}

/// Shape check for a stored `[in, out]` weight.
fn check(tensor: Tensor<2>, rows: usize, cols: usize) -> Result<Tensor<2>> {
    let [r, c] = tensor.dims();
    if (r, c) != (rows, cols) {
        bail!("weight is {r}x{c}, expected {rows}x{cols}");
    }
    Ok(tensor)
}

/// Shape check for a `[n]` bias stored as `[1, n]` (or `[n, 1]`).
fn check1(tensor: Tensor<2>, n: usize) -> Result<Tensor<1>> {
    let [r, c] = tensor.dims();
    if (r != 1 && c != 1) || r * c != n {
        bail!("bias is {r}x{c}, expected 1x{n}");
    }
    Ok(tensor.reshape([n]))
}

#[cfg(test)]
mod tests {
    use super::super::processor::patchify;
    use super::*;
    use image::RgbImage;

    fn config() -> VisionConfig {
        VisionConfig {
            depth: 2,
            hidden_size: 8,
            num_heads: 2,
            intermediate_size: 16,
            out_hidden_size: 4,
            patch_size: 2,
            temporal_patch_size: 2,
            spatial_merge_size: 2,
            in_channels: 3,
            num_position_embeddings: 9,
            rope_theta: 10_000.0,
            hidden_act: "gelu_pytorch_tanh".to_string(),
        }
    }

    #[test]
    fn patch_positions_are_merge_block_major() {
        assert_eq!(
            patch_positions(4, 4, 2),
            vec![
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 1),
                (0, 2),
                (0, 3),
                (1, 2),
                (1, 3),
                (2, 0),
                (2, 1),
                (3, 0),
                (3, 1),
                (2, 2),
                (2, 3),
                (3, 2),
                (3, 3)
            ]
        );
        assert_eq!(patch_positions(1, 2, 1), vec![(0, 0), (0, 1)]);
    }

    #[test]
    fn bilinear_taps_map_the_corners_onto_the_table() {
        // A 3x3 patch grid onto a 3x3 table: identity (align_corners edges).
        let (index, weights) = position_taps(3, 3, 3, 1);
        assert_eq!(index[0], vec![0, 1, 2, 3, 4, 5, 6, 7, 8]);
        for w in &weights[0] {
            assert!((w - 1.0).abs() < 1e-7, "{:?}", weights);
        }
        // The upper tap is the clamped edge and carries no weight.
        for w in &weights[1] {
            assert!(w.abs() < 1e-7);
        }
        // A 2x4 grid onto the 3x3 table (`align_corners=True`): the four
        // columns land at source 0, 2/3, 4/3, 2 of the 0..2 table.
        let (index, weights) = position_taps(2, 4, 3, 1);
        // Row 0: patches (0,0) (0,1) (0,2) (0,3) — lower taps, then upper taps.
        assert_eq!(&index[0][..4], &[0, 0, 1, 2]);
        // The upper tap is floor + 1, clamped to the last table row (so the
        // last column's two taps coincide at 2).
        assert_eq!(&index[1][..4], &[1, 1, 2, 2]);
        for (weight, expected) in weights[0][..4].iter().zip([1.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]) {
            assert!((weight - expected).abs() < 1e-7, "{:?}", &weights[0][..4]);
        }
        for (weight, expected) in weights[1][..4].iter().zip([0.0, 2.0 / 3.0, 1.0 / 3.0, 0.0]) {
            assert!((weight - expected).abs() < 1e-7, "{:?}", &weights[1][..4]);
        }
        // The second row of the grid maps to source 2 (the last table row).
        assert_eq!(&index[0][4..], &[6, 6, 7, 8]);
        // A 1-wide grid divides by `max(size-1, 1)` = 1, i.e. index 0 -> 0.
        let (index, weights) = position_taps(1, 1, 5, 1);
        assert_eq!(index[0], vec![0]);
        assert!((weights[0][0] - 1.0).abs() < 1e-7);
    }

    #[test]
    fn taps_sum_to_one_everywhere() {
        // Merge-block order needs an even grid per side; odd sides are covered
        // above with merge 1.
        for (gh, gw) in [(2, 2), (2, 4), (8, 6), (48, 48), (62, 56)] {
            let (indices, weights) = position_taps(gh, gw, 48, 2);
            assert_eq!(indices[0].len(), gh * gw);
            for (patch, ((w0, w1), (w2, w3))) in weights[0]
                .iter()
                .zip(&weights[1])
                .zip(weights[2].iter().zip(&weights[3]))
                .enumerate()
            {
                let total = w0 + w1 + w2 + w3;
                assert!(
                    (total - 1.0).abs() < 1e-5,
                    "{gh}x{gw} patch {patch}: {total}"
                );
            }
        }
    }

    #[test]
    fn axial_rope_uses_h_then_w_angles_over_twice_the_span() {
        let cfg = config(); // head_dim 4 -> spatial 2 -> one frequency
        let device = Device::default();
        let (cos, sin) = axial_rope_tables([1, 2, 2], &cfg, &device);
        assert_eq!(cos.dims(), [4, 2]);
        // head_dim 4: spatial = 2, n_freq = 1, inv_freq = 1/theta^0 = 1.
        let values: Vec<f32> = cos.to_data().try_to_vec::<f32>().unwrap();
        let sines: Vec<f32> = sin.to_data().try_to_vec::<f32>().unwrap();
        assert!((values[0] - 0.0f32.cos()).abs() < 1e-7); // patch (0,0): h=0
        assert!((values[1] - 0.0f32.cos()).abs() < 1e-7); // w=0
        assert!((values[2] - 0.0f32.cos()).abs() < 1e-7); // patch (0,1): h=0
        assert!((values[3] - 1.0f32.cos()).abs() < 1e-7); // w=1
        assert!((values[4] - 1.0f32.cos()).abs() < 1e-7); // patch (1,0): h=1
        assert!((values[5] - 0.0f32.cos()).abs() < 1e-7);
        assert!((sines[3] - 1.0f32.sin()).abs() < 4e-7);
        assert!((sines[4] - 1.0f32.sin()).abs() < 4e-7);
        assert!(sines[0].abs() < 1e-7);
    }

    #[test]
    fn axial_rope_has_one_frequency_per_quarter_head() {
        let mut cfg = config();
        cfg.hidden_size = 32;
        cfg.num_heads = 2; // head_dim 16 -> spatial 8 -> 4 frequencies
        let device = Device::default();
        let (cos, sin) = axial_rope_tables([1, 2, 2], &cfg, &device);
        assert_eq!(cos.dims(), [4, 8]);
        let values: Vec<f32> = cos.to_data().try_to_vec::<f32>().unwrap();
        // The first patch is at (h, w) = (0, 0): every angle is 0, so cos is 1
        // and sin is 0 for its 8 angles (4 h, then 4 w).
        for value in &values[..8] {
            assert!((value - 1.0).abs() < 1e-7);
        }
        let sines: Vec<f32> = sin.to_data().try_to_vec::<f32>().unwrap();
        for value in &sines[..8] {
            assert!(value.abs() < 1e-7);
        }
        // Patch 1 is (0, 1): the h angles stay 0 and the w angles use the
        // frequency ladder 1, 1e-0.5, 1e-1, 1e-1.5 for theta 10_000 and 8 taps.
        let ladder: Vec<f32> = (0..4)
            .map(|i| 1.0 / 10_000.0f32.powf((2 * i) as f32 / 8.0))
            .collect();
        assert!(ladder[0] > ladder[1] && ladder[1] > ladder[2] && ladder[2] > ladder[3]);
        for (i, freq) in ladder.iter().enumerate() {
            assert!((values[8 + 4 + i] - freq.cos()).abs() < 1e-6);
        }
    }

    #[test]
    fn rope_rotates_the_two_head_halves_against_each_other() {
        let device = Device::default();
        // head_dim 4, one patch at h=1 (angle 1 rad): cos = 0.5403, sin = 0.8415,
        // and the w angle is 0 (cos 1, sin 0) in the second half of the table.
        let cos = Tensor::<2>::from_data([[0.5f32, 1.0]], &device);
        let sin = Tensor::<2>::from_data([[0.5f32, 0.0]], &device);
        let x = Tensor::<3>::from_data(
            [[[1.0f32, 2.0, 3.0, 4.0]]], // [1 patch, 1 head, 4]
            &device,
        );
        let y: Vec<f32> = apply_axial_rope(x, &cos, &sin)
            .to_data()
            .try_to_vec::<f32>()
            .unwrap();
        // half 0 rotates (1, 3) by angle with cos 0.5 / sin 0.5;
        // half 1 rotates (2, 4) by angle 0 (identity).
        assert!((y[0] - (1.0 * 0.5 - 3.0 * 0.5)).abs() < 1e-6);
        assert!((y[2] - (3.0 * 0.5 + 1.0 * 0.5)).abs() < 1e-6);
        assert!((y[1] - 2.0).abs() < 1e-6);
        assert!((y[3] - 4.0).abs() < 1e-6);
    }

    /// The merger groups `merge_size^2` consecutive patches in block order: with
    /// a hidden of 2 and merge 2 a token is the concatenation of 4 patches.
    #[test]
    fn the_merger_reshape_groups_patches_in_block_order() {
        let device = Device::default();
        let x = Tensor::<2>::from_data(
            [
                [1.0f32, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
                [4.0, 40.0],
                [5.0, 50.0],
                [6.0, 60.0],
                [7.0, 70.0],
                [8.0, 80.0],
            ],
            &device,
        );
        let merged = x.reshape([2, 8]);
        let values: Vec<f32> = merged.to_data().try_to_vec::<f32>().unwrap();
        assert_eq!(
            values,
            vec![
                1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 5.0, 50.0, 6.0, 60.0, 7.0, 70.0, 8.0,
                80.0
            ]
        );
    }

    #[test]
    fn a_stub_tower_has_the_configuration_shapes() {
        let cfg = config();
        let device = Device::default();
        let tower = VisionTower::new_stub(cfg.clone(), &device);
        assert_eq!(tower.blocks.len(), 2);
        assert_eq!(tower.patch_w.dims(), [24, 8]);
        assert_eq!(tower.position_table.dims(), [9, 8]);
        assert_eq!(tower.merger.fc1_w.dims(), [32, 32]);
        assert_eq!(tower.merger.fc2_w.dims(), [32, 4]);
        assert_eq!(tower.blocks[0].qkv_w.dims(), [8, 24]);
    }

    /// The merger's activation is `nn.GELU()` (exact erf), not the blocks'
    /// `gelu_pytorch_tanh`.  With the merger's `fc1` biased to a fixed input and
    /// `fc2` summing, the whole tower output is `merge_unit * hidden * gelu(2)`;
    /// the two forms differ by 9.8e-5 per element (3.2e-3 here), which is the
    /// size of the pooled residual Phase 2a's review measured.
    #[test]
    fn the_merger_uses_the_exact_erf_gelu() {
        let cfg = config();
        let device = Device::default();
        let mut tower = VisionTower::new_stub(cfg.clone(), &device);
        let merged = cfg.merge_unit() * cfg.hidden_size; // 32 values per token
        let x = 2.0f32;
        // `fc1` ignores its input and emits a constant 2.0; `fc2` sums it.
        tower.merger.norm_w = Tensor::<1>::ones([cfg.hidden_size], &device);
        tower.merger.norm_b = Tensor::<1>::zeros([cfg.hidden_size], &device);
        tower.merger.fc1_w = Tensor::<2>::zeros([merged, merged], &device);
        tower.merger.fc1_b = Tensor::<1>::full([merged], x, &device);
        tower.merger.fc2_w = Tensor::<2>::ones([merged, cfg.out_hidden_size], &device);
        tower.merger.fc2_b = Tensor::<1>::zeros([cfg.out_hidden_size], &device);

        let image = PreparedImage {
            pixel_values: Tensor::<2>::zeros([cfg.merge_unit(), cfg.patch_dim()], &device),
            grid: [1, 2, 2],
            tokens: 1,
            source: (4, 4),
            resized: (4, 4),
        };
        let out: Vec<f32> = tower
            .encode(&image)
            .unwrap()
            .to_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert_eq!(out.len(), cfg.out_hidden_size);

        // gelu(2) exact = 2 * Phi(2), and the tanh approximation as the blocks
        // compute it.  Both constants are f64 and narrowed once (an f32 literal
        // cannot hold these digits).
        const PHI_2: f64 = 0.977_249_868_051_820_8;
        const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;
        let x64 = x as f64;
        let erf_gelu = (2.0 * PHI_2) as f32;
        let tanh_gelu =
            (0.5 * x64 * (1.0 + (SQRT_2_OVER_PI * (x64 + 0.044715 * x64 * x64 * x64)).tanh()))
                as f32;

        let expected = merged as f32 * erf_gelu;
        let approximate = merged as f32 * tanh_gelu;
        assert!(
            (out[0] - expected).abs() < 1e-3,
            "merger output {} should be the erf gelu ({expected})",
            out[0]
        );
        assert!(
            (out[0] - approximate).abs() > 1e-3,
            "merger output {} must not be the tanh gelu ({approximate})",
            out[0]
        );
    }

    #[test]
    fn a_zero_tower_runs_end_to_end() {
        // With every weight zero the output is zero but every shape and index
        // path runs: patch embed, position gather, rope, attention, merger.
        let cfg = config();
        let device = Device::default();
        let tower = VisionTower::new_stub(cfg.clone(), &device);
        let img = RgbImage::new(4, 4);
        let (values, patches) = patchify(&img, &cfg).unwrap();
        let image = PreparedImage {
            pixel_values: Tensor::<2>::from_data(
                TensorData::new(values, [patches, cfg.patch_dim()]),
                &device,
            ),
            grid: [1, 2, 2],
            tokens: 1,
            source: (4, 4),
            resized: (4, 4),
        };
        let out = tower.encode(&image).unwrap();
        assert_eq!(out.dims(), [1, 4]);
        assert!(
            out.to_data()
                .try_to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|v| *v == 0.0)
        );
        // The same image twice concatenates the merged tokens.
        let both = tower.encode_all(&[image.clone(), image]).unwrap();
        assert_eq!(both.dims(), [2, 4]);
    }

    #[test]
    fn encode_rejects_a_mismatched_grid() {
        let cfg = config();
        let device = Device::default();
        let tower = VisionTower::new_stub(cfg.clone(), &device);
        let mut image = PreparedImage {
            pixel_values: Tensor::<2>::zeros([4, cfg.patch_dim()], &device),
            grid: [1, 2, 2],
            tokens: 1,
            source: (4, 4),
            resized: (4, 4),
        };
        assert!(tower.encode(&image).is_ok());
        image.grid = [1, 4, 2];
        assert!(
            tower
                .encode(&image)
                .is_err_and(|e| e.to_string().contains("do not match"))
        );
        image.grid = [1, 2, 2];
        image.pixel_values = Tensor::<2>::zeros([4, 8], &device);
        assert!(
            tower
                .encode(&image)
                .is_err_and(|e| e.to_string().contains("values per patch"))
        );
        image.pixel_values = Tensor::<2>::zeros([6, cfg.patch_dim()], &device);
        image.grid = [1, 3, 2];
        assert!(
            tower
                .encode(&image)
                .is_err_and(|e| e.to_string().contains("not divisible"))
        );
        assert!(tower.encode_all(&[]).is_err());
    }

    #[test]
    fn set_checks_the_tensor_shapes() {
        let cfg = config();
        let device = Device::default();
        let mut tower = VisionTower::new_stub(cfg, &device);
        tower
            .set(
                "patch_embed.proj.weight",
                Tensor::<2>::zeros([24, 8], &device),
            )
            .unwrap();
        assert!(
            tower
                .set(
                    "patch_embed.proj.weight",
                    Tensor::<2>::zeros([8, 24], &device)
                )
                .is_err_and(|e| e.to_string().contains("expected 24x8"))
        );
        tower
            .set(
                "blocks.1.mlp.linear_fc2.bias",
                Tensor::<2>::zeros([1, 8], &device),
            )
            .unwrap();
        assert!(
            tower
                .set("blocks.9.norm1.weight", Tensor::<2>::zeros([8, 1], &device))
                .is_err_and(|e| e.to_string().contains("out of range"))
        );
        assert!(
            tower
                .set(
                    "blocks.0.attn.rotary.weight",
                    Tensor::<2>::zeros([8, 1], &device)
                )
                .is_err_and(|e| e.to_string().contains("unknown vision tensor"))
        );
        assert!(
            tower
                .set("visual.something", Tensor::<2>::zeros([8, 1], &device))
                .is_err_and(|e| e.to_string().contains("unknown vision tensor"))
        );
    }
}
