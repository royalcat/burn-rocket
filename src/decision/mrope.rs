//! Multimodal RoPE positions (mrope) for d3 prompts with images and videos.
//!
//! A text prompt rotates token `i` by `i * inv_freq`; an image or video prompt
//! needs three positions per token — temporal, height and width — because the
//! vision tokens form a 2-D grid per frame. The reference (`get_rope_index` +
//! `Qwen3_5TextRotaryEmbedding.recomposition_frequencies`, see
//! `transformers/models/qwen3_5/modeling_qwen3_5.py`) builds them like this:
//!
//! 1. Walk `mm_token_type_ids` in runs (`0` text, `1` image, `2` video), keeping a
//!    running `current_pos`:
//!    - a text run of `len` tokens takes `current_pos .. current_pos + len` on
//!      **all three** rows; `current_pos += len`.
//!    - an image with `grid_thw = [1, H_patches, W_patches]` becomes
//!      `llm_h = H_patches / merge`, `llm_w = W_patches / merge` merged rows in
//!      row-major order (`h` outer, `w` inner): `T = 0`, `H = 0..llm_h`,
//!      `W = 0..llm_w`, each **plus `current_pos`**;
//!      `current_pos += max(H_patches, W_patches) / merge`.
//!    - a video run is one **temporal pair**: the reference repeats every
//!      `video_grid_thw` row by its `T` and sets the temporal extent to 1
//!      (`repeat_interleave(video_grid_thw, video_grid_thw[:, 0])`), so a video of
//!      `T` pairs is `T` runs of `H_patches * W_patches / merge^2` tokens, each
//!      positioned as a `[1, H, W]` grid at the current position, and every run
//!      advances `current_pos` by `max(H, W) / merge` — the timestamp and
//!      `<|vision_start|>`/`<|vision_end|>` text between two runs consumes
//!      positions like any other text.
//! 2. Per axis, `freqs[axis] = pos[axis] * inv_freq` (`[S, rotary/2]`, f32).
//! 3. Recompose: start from the temporal row and overwrite the height indices
//!    `(1..3*section[1]).step_by(3)` with the height row and the width indices
//!    `(2..3*section[2]).step_by(3)` with the width row — with the d3
//!    `mrope_section = [11, 11, 10]` that is `1, 4, .. 31` and `2, 5, .. 29`.
//! 4. `cos = sin = cos/sin(cat([freqs, freqs]))`, i.e. the mixed `rotary/2`
//!    frequencies once per half of the rotary width.
//!
//! The tables go to [`crate::qwen35_intent::model::RopeTables`], which applies the
//! reference's `q * cos + rotate_half(q) * sin` over the first `rotary` dims of
//! the head.

use anyhow::{Result, bail};
use burn::prelude::*;
use burn::tensor::TensorData;

use crate::qwen35_intent::model::{IntentTextConfig, RopeTables};

/// `mrope_section` of the d3 text config (`rope_parameters.mrope_section`).
pub const DEFAULT_SECTIONS: [usize; 3] = [11, 11, 10];

/// Text (`0`), image (`1`) and video (`2`) run markers of `mm_token_type_ids`.
pub const TYPE_TEXT: u8 = 0;
pub const TYPE_IMAGE: u8 = 1;
pub const TYPE_VIDEO: u8 = 2;

/// `Qwen3_5TextRotaryEmbedding.compute_default_rope_parameters`:
/// `1 / theta ** (arange(0, rotary, 2) / rotary)`, computed in **f32** like the
/// torch buffer the reference multiplies with (a f64 pow followed by a narrowing
/// cast can differ in the last bit, and the angle error grows with the
/// position).
pub fn inv_freq(rotary: usize, theta: f64) -> Vec<f32> {
    let theta = theta as f32;
    (0..rotary / 2)
        .map(|i| 1.0 / theta.powf((2 * i) as f32 / rotary as f32))
        .collect()
}

/// One position triplet `(temporal, height, width)` per token.
pub type Positions = Vec<[u32; 3]>;

/// `get_rope_index` for one sequence.
///
/// `mm_types` is `[S]` of [`TYPE_TEXT`]/[`TYPE_IMAGE`]/[`TYPE_VIDEO`]; `images[i]`
/// is the `[1, H_patches, W_patches]` grid of the i-th image run and `videos[j]`
/// the `[T, H_patches, W_patches]` grid of the j-th video, both in run order (the
/// reference's `image_grid_thw`/`video_grid_thw` rows). A video contributes one
/// run per temporal pair, so `videos[j].0` must equal the number of type-2 runs
/// that video owns. Every run must have a grid and every grid must be consumed.
pub fn positions_media(
    mm_types: &[u8],
    images: &[[usize; 3]],
    videos: &[[usize; 3]],
    merge: usize,
) -> Result<Positions> {
    if merge == 0 {
        bail!("the vision merge size must be positive");
    }
    let mut out: Positions = Vec::with_capacity(mm_types.len());
    let mut next_image = 0usize;
    let mut next_video = 0usize;
    // The temporal-pair index of the video whose runs are being consumed, and how
    // many of its runs have been seen.
    let mut video_run = 0usize;
    let mut current = 0u32;
    let mut index = 0usize;
    while index < mm_types.len() {
        let kind = mm_types[index];
        let start = index;
        while index < mm_types.len() && mm_types[index] == kind {
            index += 1;
        }
        let len = index - start;
        match kind {
            TYPE_TEXT => {
                for i in 0..len {
                    let pos = current + i as u32;
                    out.push([pos, pos, pos]);
                }
                current += len as u32;
            }
            TYPE_IMAGE => {
                let grid = images.get(next_image).ok_or_else(|| {
                    anyhow::anyhow!(
                        "the prompt has more image runs than the {} image grid(s) of the request",
                        images.len()
                    )
                })?;
                next_image += 1;
                let [temporal, height, width] = *grid;
                if temporal != 1 {
                    bail!("an image grid must have one frame, got {temporal}");
                }
                push_grid(&mut out, &mut current, height, width, len, merge)?;
            }
            TYPE_VIDEO => {
                // Each run is one temporal pair of the current video; the video ends
                // when its `T` runs have been consumed.
                let grid = *videos.get(next_video).ok_or_else(|| {
                    anyhow::anyhow!(
                        "the prompt has more video runs than the {} video(s) of the request",
                        videos.len()
                    )
                })?;
                let [pairs, height, width] = grid;
                push_grid(&mut out, &mut current, height, width, len, merge)?;
                video_run += 1;
                if video_run == pairs {
                    next_video += 1;
                    video_run = 0;
                }
            }
            other => {
                bail!("unknown multimodal token type {other} (text is 0, image is 1, video is 2)")
            }
        }
    }
    if next_image != images.len() {
        bail!(
            "the prompt uses {next_image} of the {} image grid(s) of the request",
            images.len()
        );
    }
    if next_video != videos.len() {
        bail!(
            "the prompt uses {next_video} of the {} video(s) of the request",
            videos.len()
        );
    }
    Ok(out)
}

/// Push one `[1, height, width]` frame grid's merged positions and advance
/// `current` by `max(height, width) / merge`.
fn push_grid(
    out: &mut Positions,
    current: &mut u32,
    height: usize,
    width: usize,
    len: usize,
    merge: usize,
) -> Result<()> {
    if !height.is_multiple_of(merge) || !width.is_multiple_of(merge) {
        bail!("the {height}x{width} patch grid is not divisible by the merge size {merge}");
    }
    let llm_h = height / merge;
    let llm_w = width / merge;
    if llm_h * llm_w != len {
        bail!("a media run of {len} tokens does not match its {llm_h}x{llm_w} merged grid");
    }
    for h in 0..llm_h {
        for w in 0..llm_w {
            out.push([*current, *current + h as u32, *current + w as u32]);
        }
    }
    *current += (height.max(width) / merge) as u32;
    Ok(())
}

/// [`positions_media`] for an image-only prompt (no video grids): the image
/// tests and any image-only caller.
#[allow(dead_code)]
pub fn positions(mm_types: &[u8], grids: &[[usize; 3]], merge: usize) -> Result<Positions> {
    positions_media(mm_types, grids, &[], merge)
}

/// The recomposed `rotary`-wide frequency row of one token: the temporal row with
/// the height/width sections overwritten, doubled over the two halves of the
/// rotary width.
pub fn frequencies(positions: &[u32; 3], inv_freq: &[f32], sections: [usize; 3]) -> Vec<f32> {
    let half = inv_freq.len();
    let mut mixed = vec![0f32; half];
    for (k, freq) in inv_freq.iter().enumerate() {
        mixed[k] = positions[0] as f32 * *freq;
    }
    for axis in 1..3 {
        let length = sections[axis] * 3;
        let mut index = axis;
        while index < length {
            if index >= half {
                break;
            }
            mixed[index] = positions[axis] as f32 * inv_freq[index];
            index += 3;
        }
    }
    let mut row = mixed.clone();
    row.extend_from_slice(&mixed);
    row
}

/// Build the `[S, rotary]` cos/sin tables of one prompt. Every position's
/// frequency row is a pure function of `inv_freq`, so the tables are built once
/// per forward on the host (a 32k prompt is 2 x 8 MiB of f32).
pub fn tables(
    positions: &Positions,
    rotary: usize,
    theta: f64,
    sections: [usize; 3],
    device: &Device,
) -> Result<RopeTables> {
    if positions.is_empty() {
        bail!("an mrope table needs at least one position");
    }
    if rotary == 0 || !rotary.is_multiple_of(2) {
        bail!("the rotary width must be a positive even number, got {rotary}");
    }
    let inv = inv_freq(rotary, theta);
    let seq = positions.len();
    let mut cos = Vec::with_capacity(seq * rotary);
    let mut sin = Vec::with_capacity(seq * rotary);
    for position in positions {
        for angle in frequencies(position, &inv, sections) {
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    Ok(RopeTables::new(
        Tensor::<2>::from_data(TensorData::new(cos, [seq, rotary]), device),
        Tensor::<2>::from_data(TensorData::new(sin, [seq, rotary]), device),
    ))
}

/// The `mrope_section` of a text config (the reference's 3-value list).
pub fn sections(config: &IntentTextConfig) -> Result<[usize; 3]> {
    let raw = config.rope_parameters.mrope_section.as_deref();
    let Some(raw) = raw else {
        return Ok(DEFAULT_SECTIONS);
    };
    let [a, b, c]: [usize; 3] = raw
        .try_into()
        .map_err(|_| anyhow::anyhow!("mrope_section has {} values, expected 3", raw.len()))?;
    Ok([a, b, c])
}

/// The per-prompt tables of the checkpoint's text config.
pub fn rope_tables(
    mm_types: &[u8],
    images: &[[usize; 3]],
    videos: &[[usize; 3]],
    config: &IntentTextConfig,
    merge: usize,
    device: &Device,
) -> Result<RopeTables> {
    let positions = positions_media(mm_types, images, videos, merge)?;
    tables(
        &positions,
        config.rotary_dim(),
        config.rope_parameters.rope_theta,
        sections(config)?,
        device,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference's recomposition indices for the d3 sections.
    #[test]
    fn the_recomposition_sections_are_the_reference_slices() {
        let inv = inv_freq(64, 10_000_000.0);
        assert_eq!(inv.len(), 32, "64 rotary dims take 32 frequencies");
        let mut height = Vec::new();
        let mut width = Vec::new();
        for index in 1..DEFAULT_SECTIONS[1] * 3 {
            if (index - 1) % 3 == 0 {
                height.push(index);
            }
        }
        for index in 2..DEFAULT_SECTIONS[2] * 3 {
            if (index - 2) % 3 == 0 {
                width.push(index);
            }
        }
        assert_eq!(height, (1..33).step_by(3).collect::<Vec<_>>());
        assert_eq!(width, (2..30).step_by(3).collect::<Vec<_>>());
        assert_eq!(height.len(), 11);
        assert_eq!(width.len(), 10);
        // A token at (t, h, w) = (5, 2, 7): the T row fills everything, H takes
        // the 11 height slots and W the 10 width slots.
        let row = frequencies(&[5, 2, 7], &inv, DEFAULT_SECTIONS);
        assert_eq!(row.len(), 64);
        for (k, freq) in inv.iter().enumerate() {
            let expected = match k % 3 {
                0 => 5.0 * freq,
                1 if k <= 31 => 2.0 * freq,
                2 if k <= 29 => 7.0 * freq,
                _ => 5.0 * freq,
            };
            assert_eq!(row[k], expected, "frequency {k}");
            assert_eq!(row[32 + k], expected, "doubled frequency {k}");
        }
        // Slots 30 and 31 fall outside the width section (30 > 29) and beyond the
        // height section's last index only for 31 < 32 -> still height.
        assert_eq!(row[30], 5.0 * inv[30]);
        assert_eq!(row[31], 2.0 * inv[31]);
    }

    /// A text-only prompt is the plain `0..S` ramp on every row.
    #[test]
    fn a_text_prompt_is_the_stock_ramp() {
        let positions = positions(&[0; 5], &[], 2).unwrap();
        for (i, pos) in positions.iter().enumerate() {
            assert_eq!(*pos, [i as u32; 3]);
        }
        let inv = inv_freq(64, 10_000_000.0);
        let row = frequencies(&positions[3], &inv, DEFAULT_SECTIONS);
        for (k, freq) in inv.iter().enumerate() {
            assert_eq!(
                row[k],
                3.0 * freq,
                "text frequencies are position * inv_freq"
            );
        }
        // ... and the table matches a stock housekeeping ramp: cos/sin of
        // `pos * inv_freq` over the doubled width.
        let tables = tables(
            &positions,
            64,
            10_000_000.0,
            DEFAULT_SECTIONS,
            &Device::default(),
        )
        .unwrap();
        assert_eq!(tables.len(), 5);
        assert_eq!(tables.rotary(), 64);
    }

    /// One 2x2 merged image between text: the run bookkeeping and the positions.
    #[test]
    fn a_merged_image_grid_is_row_major() {
        // 3 text, a 4x6 patch grid (merge 2 -> 2x3 = 6 tokens), 2 text.
        let mut types = vec![TYPE_TEXT; 3];
        types.extend(std::iter::repeat_n(TYPE_IMAGE, 6));
        types.extend(std::iter::repeat_n(TYPE_TEXT, 2));
        let positions = positions(&types, &[[1, 4, 6]], 2).unwrap();
        assert_eq!(positions.len(), 11);
        assert_eq!(positions[2], [2, 2, 2]);
        // The image starts at current_pos = 3: h in 0..2, w in 0..3, row-major.
        assert_eq!(positions[3], [3, 3, 3]);
        assert_eq!(positions[4], [3, 3, 4]);
        assert_eq!(positions[5], [3, 3, 5]);
        assert_eq!(positions[6], [3, 4, 3]);
        assert_eq!(positions[8], [3, 4, 5]);
        // current_pos += max(4, 6) / 2 = 3.
        assert_eq!(positions[9], [6, 6, 6]);
        assert_eq!(positions[10], [7, 7, 7]);
    }

    /// Two images with a text run between them: each image advances
    /// `current_pos` by `max(h, w) / merge`.
    #[test]
    fn a_text_run_between_two_images_keeps_advancing() {
        let mut types = vec![TYPE_TEXT; 2];
        types.extend(std::iter::repeat_n(TYPE_IMAGE, 4)); // 4x4 grid, merge 2 -> 2x2
        types.extend(std::iter::repeat_n(TYPE_TEXT, 1));
        types.extend(std::iter::repeat_n(TYPE_IMAGE, 6)); // 4x6 grid -> 2x3
        types.push(TYPE_TEXT);
        let positions = positions(&types, &[[1, 4, 4], [1, 4, 6]], 2).unwrap();
        assert_eq!(positions[0], [0, 0, 0]);
        assert_eq!(positions[2], [2, 2, 2], "first image starts at 2");
        assert_eq!(positions[5], [2, 3, 3], "last row of the 2x2 grid");
        // 2 + max(4,4)/2 = 4: the middle text token sits at 4.
        assert_eq!(positions[6], [4, 4, 4]);
        assert_eq!(positions[7], [5, 5, 5], "second image starts at 5");
        assert_eq!(positions[12], [5, 6, 7]);
        // 5 + max(4,6)/2 = 8.
        assert_eq!(positions[13], [8, 8, 8]);
    }

    /// The d3 receipt case, against the official runtime's dumped `mrope`
    /// (`/tmp/opencode/d3/dump-images-lite-f32`): 42 text tokens, a 62x56 grid
    /// (868 merged tokens) and 71 text tokens.
    #[test]
    fn the_receipt_positions_match_the_reference_dump() {
        let mut types = vec![TYPE_TEXT; 43];
        types.extend(std::iter::repeat_n(TYPE_IMAGE, 868));
        types.extend(std::iter::repeat_n(TYPE_TEXT, 71));
        let positions = positions(&types, &[[1, 62, 56]], 2).unwrap();
        assert_eq!(positions.len(), 982);
        assert_eq!(positions[42], [42, 42, 42]);
        assert_eq!(positions[43], [43, 43, 43]);
        assert_eq!(positions[44], [43, 43, 44]);
        assert_eq!(positions[43 + 867], [43, 73, 70]);
        assert_eq!(positions[911], [74, 74, 74]);
        assert_eq!(positions[981], [144, 144, 144]);
    }

    /// The video case from the reference dump (`dump-videos-lite-f32`,
    /// `video-example__direction`): 50 text tokens (system turn, `<|vision_start|>`
    /// and the first timestamp), then four temporal pairs of a 20x36 grid (180
    /// merged tokens each) separated by 8 text tokens, then 74 text tokens. Each
    /// pair is a `[1, 20, 36]` grid at the current position and advances it by
    /// `max(20, 36) / 2 = 18`, so the T row of the last token is 219 while the
    /// sequence is 868 long.
    #[test]
    fn the_video_positions_match_the_reference_dump() {
        let mut types = vec![TYPE_TEXT; 50];
        for pair in 0..4 {
            if pair > 0 {
                types.extend(std::iter::repeat_n(TYPE_TEXT, 8));
            }
            types.extend(std::iter::repeat_n(TYPE_VIDEO, 180));
        }
        types.extend(std::iter::repeat_n(TYPE_TEXT, 74));
        assert_eq!(types.len(), 868);
        let positions = positions_media(&types, &[], &[[4, 20, 36]], 2).unwrap();
        assert_eq!(positions.len(), 868);
        // Text before the first pair.
        assert_eq!(positions[49], [49, 49, 49]);
        // Inside the first pair: T fixed, H row-major over 10, W over 18.
        assert_eq!(positions[50], [50, 50, 50]);
        assert_eq!(positions[51], [50, 50, 51]);
        assert_eq!(positions[50 + 17], [50, 50, 67]);
        assert_eq!(positions[50 + 18], [50, 51, 50]);
        assert_eq!(positions[50 + 179], [50, 59, 67]);
        // The 8 text tokens after it continue at 50 + 18 = 68.
        assert_eq!(positions[230], [68, 68, 68]);
        assert_eq!(positions[237], [75, 75, 75]);
        // The next pair starts after those 8 tokens.
        assert_eq!(positions[238], [76, 76, 76]);
        // The last pair (formula: 50 + 3 * 188).
        assert_eq!(positions[614], [128, 128, 128]);
        assert_eq!(positions[614 + 179], [128, 137, 145]);
        // The closing text run of 74 tokens ends at 128 + 18 + 74 - 1.
        assert_eq!(positions[867], [219, 219, 219]);
    }

    /// Two videos: every temporal pair is its own run (a separator between them),
    /// and each pair advances `current_pos` by `max(H, W) / merge` of its grid.
    #[test]
    fn a_second_video_continues_after_the_first() {
        let mut types = vec![TYPE_TEXT; 2];
        types.extend(std::iter::repeat_n(TYPE_VIDEO, 4)); // 4x4 grid, pair 1
        types.push(TYPE_TEXT);
        types.extend(std::iter::repeat_n(TYPE_VIDEO, 4)); // pair 2
        types.push(TYPE_TEXT);
        types.extend(std::iter::repeat_n(TYPE_VIDEO, 6)); // 4x6 grid, pair 1
        types.push(TYPE_TEXT);
        types.extend(std::iter::repeat_n(TYPE_VIDEO, 6)); // pair 2
        types.push(TYPE_TEXT);
        assert_eq!(types.len(), 26);
        let positions = positions_media(&types, &[], &[[2, 4, 4], [2, 4, 6]], 2).unwrap();
        assert_eq!(positions[0], [0, 0, 0]);
        assert_eq!(positions[2], [2, 2, 2]);
        assert_eq!(positions[3], [2, 2, 3]);
        // Pair 1 advanced by max(4, 4)/2 = 2: the separator sits at 4.
        assert_eq!(positions[6], [4, 4, 4]);
        assert_eq!(positions[7], [5, 5, 5]);
        assert_eq!(positions[10], [5, 6, 6]);
        // 5 + 2 = 7: the separator, then the second video at 8.
        assert_eq!(positions[11], [7, 7, 7]);
        assert_eq!(positions[12], [8, 8, 8]);
        assert_eq!(positions[17], [8, 9, 10]);
        // 8 + max(4, 6)/2 = 11, then the separator at 12.
        assert_eq!(positions[18], [11, 11, 11]);
        assert_eq!(positions[19], [12, 12, 12]);
        assert_eq!(positions[24], [12, 13, 14]);
        assert_eq!(positions[25], [15, 15, 15]);
    }

    #[test]
    fn mismatched_grids_are_refused() {
        // An image run without a grid.
        assert!(positions(&[TYPE_IMAGE; 4], &[], 2).is_err());
        // A grid nobody consumed.
        assert!(positions(&[TYPE_TEXT; 2], &[[1, 4, 4]], 2).is_err());
        // A run that does not match the merged grid.
        assert!(positions(&[TYPE_IMAGE; 5], &[[1, 4, 4]], 2).is_err());
        // An image grid with more than one frame.
        assert!(positions(&[TYPE_IMAGE; 4], &[[2, 4, 4]], 2).is_err());
        // A grid not divisible by the merge size.
        assert!(positions(&[TYPE_IMAGE; 4], &[[1, 3, 4]], 2).is_err());
        // An unknown token type.
        assert!(positions(&[7; 4], &[], 2).is_err());
        // A video run without a video grid, and a video grid nobody consumed.
        assert!(positions_media(&[TYPE_VIDEO; 4], &[], &[], 2).is_err());
        assert!(positions_media(&[TYPE_TEXT; 2], &[], &[[1, 4, 4]], 2).is_err());
        // More runs than the video's temporal pairs.
        assert!(positions_media(&[TYPE_VIDEO; 8], &[], &[[1, 4, 4]], 2).is_err());
        // Fewer runs than its pairs.
        assert!(positions_media(&[TYPE_VIDEO; 4], &[], &[[2, 4, 4]], 2).is_err());
    }

    #[test]
    fn the_inverse_frequencies_are_the_f32_formula() {
        let inv = inv_freq(64, 10_000_000.0);
        assert_eq!(inv[0], 1.0, "the first frequency is one");
        let expected = 1.0f32 / 10_000_000f32.powf(62.0 / 64.0);
        assert_eq!(inv[31], expected);
        assert!(inv.windows(2).all(|w| w[0] > w[1]), "{inv:?}");
    }

    #[test]
    fn the_config_sections_are_read_and_validated() {
        let mut config = crate::qwen35_intent::model::IntentTextConfig::from_value(
            serde_json::json!({"text_config": {
                "vocab_size": 8,
                "hidden_size": 8,
                "intermediate_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "head_dim": 8,
                "rms_norm_eps": 1e-6,
                "max_position_embeddings": 8,
                "layer_types": ["full_attention"],
                "linear_conv_kernel_dim": 2,
                "linear_key_head_dim": 2,
                "linear_value_head_dim": 2,
                "linear_num_key_heads": 1,
                "linear_num_value_heads": 1,
                "partial_rotary_factor": 0.5,
                "rope_parameters": {"rope_theta": 10000.0, "mrope_section": [11, 11, 10]},
            }}),
        )
        .unwrap();
        assert_eq!(sections(&config).unwrap(), [11, 11, 10]);
        config.rope_parameters.mrope_section = Some(vec![11, 11]);
        let error = sections(&config).unwrap_err().to_string();
        assert!(error.contains("expected 3"), "{error}");
        config.rope_parameters.mrope_section = None;
        assert_eq!(sections(&config).unwrap(), DEFAULT_SECTIONS);
    }
}
