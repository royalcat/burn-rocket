//! Media plumbing of the Clef-Omni thinker (phase 8b): everything between "the towers
//! produced features" and "the text decoder ran", with NO tower code.
//!
//! * [`audio_out_len`], [`MediaLayout`] and [`build_media_ids`]: the placeholder block
//!   the Qwen3-Omni processor puts after the record prefix (`<|vision_start|>` +
//!   `<|image_pad|>` x n ... + `\n`), including the audio-in-video interleave. Everything
//!   is derived from grids / mel-frame counts, so the tokenizer is only needed for the
//!   `\n` token.
//! * [`rope_index`]: a literal port of HF `get_rope_index` (3-D interleaved mRoPE
//!   position streams `[3, S]`, `position_id_per_seconds = 13`), as a pure function of the
//!   token ids + grids + audio lengths. HF takes the 3-D branch only when an image or
//!   video grid exists; text-only and audio-only records get plain `0..S` positions.
//! * [`MediaFeatures`]: what the backbone consumes: positions, the rows spliced over the
//!   `*_pad` embeddings, and the DeepStack rows added after decoder layers 0, 1 and 2 at
//!   the visual pad positions only. [`MediaFeatures::assemble`] builds it from the tower
//!   outputs and the token ids.
//!
//! Reference: `tools/clef_media.py` and `docs/recipes/clef-oracle.md` "Media records".

use anyhow::{Result, bail, ensure};

pub const VISION_START: u32 = 151652;
pub const VISION_END: u32 = 151653;
pub const IMAGE_PAD: u32 = 151655;
pub const VIDEO_PAD: u32 = 151656;
pub const AUDIO_START: u32 = 151669;
pub const AUDIO_END: u32 = 151670;
pub const AUDIO_PAD: u32 = 151675;
/// `thinker_config.position_id_per_seconds`.
pub const POSITION_ID_PER_SECOND: f32 = 13.0;
/// Vision `spatial_merge_size`.
pub const SPATIAL_MERGE: usize = 2;
/// `rope_scaling.mrope_section` (interleaved).
pub const MROPE_SECTION: [usize; 3] = [24, 20, 20];
/// DeepStack taps: decoder layers after which features are added.
pub const DEEPSTACK_LAYERS: usize = 3;
/// `temporal_patch_size / VIDEO_FPS` = 2 / 2.0.
pub const VIDEO_SECOND_PER_GRID: f32 = 1.0;

/// Audio tower output length for `n` mel frames (`_get_feat_extract_output_lengths`,
/// `n_window = 50`): 13 tokens per full 100-frame chunk plus the strided tail.
pub fn audio_out_len(n: usize) -> usize {
    let (chunks, l) = ((n / 100) as i64, (n % 100) as i64);
    // Python floor division; every operand is >= -1 so `div_euclid` is the floor.
    let feat = (l - 1).div_euclid(2) + 1;
    let tail = ((feat - 1).div_euclid(2) + 1 - 1).div_euclid(2) + 1;
    (tail + chunks * 13) as usize
}

/// Tokens of one image/video grid after the spatial merge: `t*h*w / 4`.
pub fn grid_tokens(g: [usize; 3]) -> usize {
    g[0] * g[1] * g[2] / (SPATIAL_MERGE * SPATIAL_MERGE)
}

/// `smart_resize` of the Qwen3-VL image processors: `(h_bar, w_bar)` in pixels for an
/// `h x w` input (`factor` 32, `min_px`/`max_px` pixel budgets).
pub fn smart_resize(
    h: usize,
    w: usize,
    factor: usize,
    min_px: usize,
    max_px: usize,
) -> Result<(usize, usize)> {
    ensure!(h > 0 && w > 0, "empty image");
    ensure!(
        h.max(w) as f64 / h.min(w) as f64 <= 200.0,
        "absolute aspect ratio must be smaller than 200, got {}",
        h.max(w) as f64 / h.min(w) as f64
    );
    let round_to = |v: f64| ((v / factor as f64).round_ties_even() as usize).max(1) * factor;
    let mut hb = round_to(h as f64);
    let mut wb = round_to(w as f64);
    if hb * wb > max_px {
        let beta = ((h * w) as f64 / max_px as f64).sqrt();
        hb = factor.max(((h as f64 / beta / factor as f64).floor() as usize) * factor);
        wb = factor.max(((w as f64 / beta / factor as f64).floor() as usize) * factor);
    } else if hb * wb < min_px {
        let beta = (min_px as f64 / (h * w) as f64).sqrt();
        hb = ((h as f64 * beta / factor as f64).ceil() as usize) * factor;
        wb = ((w as f64 * beta / factor as f64).ceil() as usize) * factor;
    }
    Ok((hb, wb))
}

/// What the placeholder block and the positions of a record are derived from.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MediaLayout {
    /// `[t, h, w]` patch grid of every image (record order).
    pub image_grids: Vec<[usize; 3]>,
    /// `[t, h, w]` patch grid of every video (record order).
    pub video_grids: Vec<[usize; 3]>,
    /// Mel frames of every audio clip, in the processor's order: the explicit `audio`
    /// clips first, then (iff `use_audio_in_video`) one soundtrack per video.
    pub audio_frames: Vec<usize>,
    /// Number of explicit `audio` clips (the leading entries of `audio_frames`).
    pub explicit_audio: usize,
    /// All videos have a soundtrack: their audio is interleaved into the video block.
    pub use_audio_in_video: bool,
    /// Seconds per temporal grid of each video (`VIDEO_SECOND_PER_GRID` when empty).
    pub second_per_grid: Vec<f32>,
}

impl MediaLayout {
    pub fn is_empty(&self) -> bool {
        self.image_grids.is_empty() && self.video_grids.is_empty() && self.audio_frames.is_empty()
    }

    fn spg(&self, video: usize) -> f32 {
        self.second_per_grid
            .get(video)
            .copied()
            .unwrap_or(VIDEO_SECOND_PER_GRID)
    }

    fn validate(&self) -> Result<()> {
        let want = self.explicit_audio
            + if self.use_audio_in_video {
                self.video_grids.len()
            } else {
                0
            };
        ensure!(
            self.audio_frames.len() == want,
            "{} audio lengths for {} explicit clips and {} videos (use_audio_in_video={})",
            self.audio_frames.len(),
            self.explicit_audio,
            self.video_grids.len(),
            self.use_audio_in_video
        );
        for g in self.image_grids.iter().chain(&self.video_grids) {
            ensure!(
                g[0] > 0
                    && g[1] > 0
                    && g[2] > 0
                    && g[1] % SPATIAL_MERGE == 0
                    && g[2] % SPATIAL_MERGE == 0,
                "bad patch grid {g:?}"
            );
        }
        Ok(())
    }
}

/// `replace_video_with_audio_token`: the interleaved body between `<|audio_start|>` and
/// `<|audio_end|>` of one video. Video token k (raster over `t` groups of
/// `(h/2)*(w/2)` merged tokens) sits at time `group * second_per_grid * 13`, audio token
/// j at time j; a video token goes first when its time is `<=` the audio token's.
/// `true` = video pad, `false` = audio pad.
pub fn interleave_video_audio(
    grid: [usize; 3],
    second_per_grid: f32,
    audio_len: usize,
) -> Vec<bool> {
    let per_group = grid[1] / SPATIAL_MERGE * (grid[2] / SPATIAL_MERGE);
    let video_len = grid[0] * per_group;
    let vtime = |k: usize| (k / per_group) as f32 * second_per_grid * POSITION_ID_PER_SECOND;
    let (mut v, mut a) = (0usize, 0usize);
    let mut out = Vec::with_capacity(video_len + audio_len);
    while v < video_len && a < audio_len {
        if vtime(v) <= a as f32 {
            out.push(true);
            v += 1;
        } else {
            out.push(false);
            a += 1;
        }
    }
    out.extend(std::iter::repeat_n(true, video_len - v));
    out.extend(std::iter::repeat_n(false, audio_len - a));
    out
}

/// The media block of `encode_record`: images, then explicit audio, then videos
/// (whatever the JSON field order), each `<start> pads <end>\n`. `newline` is the token
/// id of `"\n"` (a separate token after every placeholder).
pub fn build_media_ids(layout: &MediaLayout, newline: u32) -> Result<Vec<u32>> {
    layout.validate()?;
    let mut ids = Vec::new();
    for &g in &layout.image_grids {
        ids.push(VISION_START);
        ids.extend(std::iter::repeat_n(IMAGE_PAD, grid_tokens(g)));
        ids.extend([VISION_END, newline]);
    }
    for &frames in &layout.audio_frames[..layout.explicit_audio] {
        ids.push(AUDIO_START);
        ids.extend(std::iter::repeat_n(AUDIO_PAD, audio_out_len(frames)));
        ids.extend([AUDIO_END, newline]);
    }
    for (i, &g) in layout.video_grids.iter().enumerate() {
        ids.push(VISION_START);
        if layout.use_audio_in_video {
            let frames = layout.audio_frames[layout.explicit_audio + i];
            ids.push(AUDIO_START);
            for is_video in interleave_video_audio(g, layout.spg(i), audio_out_len(frames)) {
                ids.push(if is_video { VIDEO_PAD } else { AUDIO_PAD });
            }
            ids.push(AUDIO_END);
        } else {
            ids.extend(std::iter::repeat_n(VIDEO_PAD, grid_tokens(g)));
        }
        ids.extend([VISION_END, newline]);
    }
    Ok(ids)
}

/// `[3, S]` mRoPE position streams (time, height, width). Integral in practice (HF
/// stores floats); kept as f32 like the reference.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Positions {
    pub t: Vec<f32>,
    pub h: Vec<f32>,
    pub w: Vec<f32>,
}

impl Positions {
    pub fn len(&self) -> usize {
        self.t.len()
    }

    pub fn is_empty(&self) -> bool {
        self.t.is_empty()
    }

    /// Plain text positions `0..s` in all three streams.
    pub fn text(s: usize) -> Self {
        let a: Vec<f32> = (0..s).map(|i| i as f32).collect();
        Self {
            t: a.clone(),
            h: a.clone(),
            w: a,
        }
    }

    /// From a row-major `[3, S]` array (the oracle's `position_ids`).
    pub fn from_flat(flat: &[f32]) -> Self {
        let s = flat.len() / 3;
        Self {
            t: flat[..s].to_vec(),
            h: flat[s..2 * s].to_vec(),
            w: flat[2 * s..].to_vec(),
        }
    }

    pub fn max(&self) -> f32 {
        self.t
            .iter()
            .chain(&self.h)
            .chain(&self.w)
            .fold(f32::NEG_INFINITY, |m, &v| m.max(v))
    }

    /// `max(pos) + 1 - S`.
    pub fn rope_delta(&self) -> f32 {
        self.max() + 1.0 - self.len() as f32
    }

    /// Which stream drives frequency slot `j` of `half` slots (interleaved mRoPE:
    /// `h` for `j % 3 == 1` below `section[1] * 3`, `w` for `j % 3 == 2` below
    /// `section[2] * 3`, else `t`).
    pub fn stream_of_slot(j: usize) -> usize {
        if j % 3 == 1 && j < MROPE_SECTION[1] * 3 {
            1
        } else if j % 3 == 2 && j < MROPE_SECTION[2] * 3 {
            2
        } else {
            0
        }
    }

    pub fn stream(&self, s: usize) -> &[f32] {
        match s {
            0 => &self.t,
            1 => &self.h,
            _ => &self.w,
        }
    }
}

/// Block accumulator of `get_rope_index`'s `llm_pos_ids_list`; `last_max` is the max of
/// the LAST appended block (HF's `llm_pos_ids_list[-1].max()`), not of everything.
struct PosList {
    p: Positions,
    last_max: Option<f32>,
}

impl PosList {
    fn next_start(&self) -> f32 {
        self.last_max.map_or(0.0, |m| m + 1.0)
    }

    fn push_cols(&mut self, cols: impl Iterator<Item = [f32; 3]>) {
        let mut m = f32::NEG_INFINITY;
        for c in cols {
            self.p.t.push(c[0]);
            self.p.h.push(c[1]);
            self.p.w.push(c[2]);
            m = m.max(c[0]).max(c[1]).max(c[2]);
        }
        if m.is_finite() {
            self.last_max = Some(m);
        }
    }

    /// `arange(n) + start` in all three streams.
    fn push_arange(&mut self, n: usize, start: f32) {
        self.push_cols((0..n).map(|i| [start + i as f32; 3]));
    }
}

/// `get_llm_pos_ids_for_vision`: columns `(t_index[g], row, col)` over the merged grid,
/// all shifted by `start`.
fn vision_columns(start: f32, grid: [usize; 3], t_index: &[f32]) -> Vec<[f32; 3]> {
    let (gh, gw) = (grid[1] / SPATIAL_MERGE, grid[2] / SPATIAL_MERGE);
    let mut v = Vec::with_capacity(t_index.len() * gh * gw);
    for &t in t_index {
        for r in 0..gh {
            for c in 0..gw {
                v.push([t + start, r as f32 + start, c as f32 + start]);
            }
        }
    }
    v
}

/// HF `Qwen3OmniMoePreTrainedModelForConditionalGeneration.get_rope_index` for one
/// unpadded record. `ids` carries the placeholder layout; the grids, audio lengths and
/// the flag come from the preprocessing. An error means the layout and the lengths
/// disagree (HF would raise or silently mis-position).
pub fn rope_index(ids: &[u32], layout: &MediaLayout) -> Result<Positions> {
    let n = ids.len();
    // HF: the 3-D branch needs an image or a video grid; otherwise plain cumsum - 1.
    if layout.image_grids.is_empty() && layout.video_grids.is_empty() {
        return Ok(Positions::text(n));
    }
    layout.validate()?;
    // HF would silently mis-position a record whose pads disagree with the lengths.
    let count = |tok: u32| ids.iter().filter(|&&t| t == tok).count();
    for (name, tok, want) in [
        (
            "image",
            IMAGE_PAD,
            layout
                .image_grids
                .iter()
                .map(|&g| grid_tokens(g))
                .sum::<usize>(),
        ),
        (
            "video",
            VIDEO_PAD,
            layout.video_grids.iter().map(|&g| grid_tokens(g)).sum(),
        ),
        (
            "audio",
            AUDIO_PAD,
            layout.audio_frames.iter().map(|&f| audio_out_len(f)).sum(),
        ),
    ] {
        ensure!(
            count(tok) == want,
            "{} {name} pads but the lengths imply {want}",
            count(tok)
        );
    }
    let aiv = layout.use_audio_in_video;
    let after_start: Vec<u32> = ids
        .iter()
        .enumerate()
        .filter(|&(_, &t)| t == VISION_START)
        .filter_map(|(i, _)| ids.get(i + 1).copied())
        .collect();
    let audio_nums = ids.iter().filter(|&&t| t == AUDIO_START).count();
    let image_nums = after_start.iter().filter(|&&t| t == IMAGE_PAD).count();
    let video_nums = after_start
        .iter()
        .filter(|&&t| t == if aiv { AUDIO_START } else { VIDEO_PAD })
        .count();
    let multimodal = if aiv {
        image_nums + audio_nums
    } else {
        image_nums + video_nums + audio_nums
    };
    let has_vision_pad = ids.contains(&IMAGE_PAD) || ids.contains(&VIDEO_PAD);
    let has_audio_pad = ids.contains(&AUDIO_PAD);
    let find = |tok: u32, from: usize| -> Result<usize> {
        match ids[from.min(n)..].iter().position(|&t| t == tok) {
            Some(i) => Ok(from + i),
            None => bail!("token {tok} not found after position {from}"),
        }
    };

    let mut list = PosList {
        p: Positions::default(),
        last_max: None,
    };
    let (mut image_idx, mut video_idx, mut audio_idx) = (0usize, 0usize, 0usize);
    let (mut remain_images, mut remain_videos, mut remain_audios) =
        (image_nums, video_nums, audio_nums);
    let mut st = 0usize;
    for _ in 0..multimodal {
        let mut st_idx = list.next_start();
        let ed_vision = if has_vision_pad && (remain_videos > 0 || remain_images > 0) {
            find(VISION_START, st)?
        } else {
            n + 1
        };
        let ed_audio = if has_audio_pad && remain_audios > 0 {
            find(AUDIO_START, st)?
        } else {
            n + 1
        };
        let min_ed = ed_vision.min(ed_audio);
        ensure!(min_ed <= n, "no placeholder left but a media item remains");
        let text_len = min_ed - st;
        if text_len != 0 {
            list.push_arange(text_len, st_idx);
            st_idx += text_len as f32;
        }
        let in_video_audio = min_ed == ed_vision && ed_vision + 1 == ed_audio;
        let (bos_len, eos_len) = if in_video_audio { (2, 2) } else { (1, 1) };
        list.push_arange(bos_len, st_idx);
        st_idx += bos_len as f32;
        let audio_len = |idx: usize| -> Result<usize> {
            Ok(audio_out_len(*layout.audio_frames.get(idx).ok_or_else(
                || anyhow::anyhow!("audio clip {idx} has no length"),
            )?))
        };
        let video_t = |idx: usize| -> Vec<f32> {
            let spg = layout.spg(idx);
            (0..layout.video_grids[idx][0])
                .map(|g| g as f32 * spg * POSITION_ID_PER_SECOND)
                .collect()
        };
        if min_ed == ed_audio {
            let len = audio_len(audio_idx)?;
            list.push_arange(len, st_idx);
            st += text_len + bos_len + len + eos_len;
            audio_idx += 1;
            remain_audios -= 1;
        } else if ids.get(ed_vision + 1) == Some(&IMAGE_PAD) {
            let g = *layout
                .image_grids
                .get(image_idx)
                .ok_or_else(|| anyhow::anyhow!("image {image_idx} has no grid"))?;
            let t_index: Vec<f32> = (0..g[0])
                .map(|i| i as f32 * POSITION_ID_PER_SECOND)
                .collect();
            list.push_cols(vision_columns(st_idx, g, &t_index).into_iter());
            st += text_len + bos_len + grid_tokens(g) + eos_len;
            image_idx += 1;
            remain_images -= 1;
        } else if ids.get(ed_vision + 1) == Some(&VIDEO_PAD) {
            let g = *layout
                .video_grids
                .get(video_idx)
                .ok_or_else(|| anyhow::anyhow!("video {video_idx} has no grid"))?;
            list.push_cols(vision_columns(st_idx, g, &video_t(video_idx)).into_iter());
            st += text_len + bos_len + grid_tokens(g) + eos_len;
            video_idx += 1;
            remain_videos -= 1;
        } else if in_video_audio {
            let alen = audio_len(audio_idx)?;
            let g = *layout
                .video_grids
                .get(video_idx)
                .ok_or_else(|| anyhow::anyhow!("video {video_idx} has no grid"))?;
            let audio: Vec<[f32; 3]> = (0..alen).map(|j| [st_idx + j as f32; 3]).collect();
            let video = vision_columns(st_idx, g, &video_t(video_idx));
            let (mut vi, mut ai) = (0usize, 0usize);
            while vi < video.len() && ai < audio.len() {
                if video[vi][0] <= audio[ai][0] {
                    list.push_cols(std::iter::once(video[vi]));
                    vi += 1;
                } else {
                    list.push_cols(std::iter::once(audio[ai]));
                    ai += 1;
                }
            }
            if vi < video.len() {
                list.push_cols(video[vi..].iter().copied());
            }
            if ai < audio.len() {
                list.push_cols(audio[ai..].iter().copied());
            }
            st += text_len + bos_len + alen + grid_tokens(g) + eos_len;
            audio_idx += 1;
            video_idx += 1;
            remain_videos -= 1;
            remain_audios -= 1;
        } else {
            bail!("vision_start at {ed_vision} is not followed by an image, video or audio token");
        }
        let st_idx = list.next_start();
        list.push_arange(eos_len, st_idx);
    }
    if st < n {
        let st_idx = list.next_start();
        list.push_arange(n - st, st_idx);
    }
    ensure!(
        list.p.len() == n,
        "positions cover {} tokens but the record has {n} (placeholder counts and lengths disagree)",
        list.p.len()
    );
    Ok(list.p)
}

/// Features the backbone consumes for one record. Rows are `hidden` floats each.
#[derive(Debug, Clone, Default)]
pub struct MediaFeatures {
    /// `[3, S]` mRoPE positions (`None` = plain text positions).
    pub positions: Option<Positions>,
    /// Sequence positions whose embedding is replaced (image, video and audio pads).
    pub splice_idx: Vec<usize>,
    /// `[splice_idx.len(), hidden]` replacement rows.
    pub splice_rows: Vec<f32>,
    /// Visual pad positions (sorted): the rows DeepStack adds to.
    pub deepstack_idx: Vec<usize>,
    /// `[deepstack_idx.len(), hidden]` features added after decoder layer 0, 1, 2.
    pub deepstack: [Vec<f32>; DEEPSTACK_LAYERS],
}

/// Tower outputs of one record, merged rows of `hidden` floats (image rows concatenated
/// in record order, then likewise video and audio).
#[derive(Debug, Clone, Default)]
pub struct TowerOutputs {
    pub image: Vec<f32>,
    pub video: Vec<f32>,
    pub audio: Vec<f32>,
    /// DeepStack taps of the image tower run and of the video tower run.
    pub image_deepstack: [Vec<f32>; DEEPSTACK_LAYERS],
    pub video_deepstack: [Vec<f32>; DEEPSTACK_LAYERS],
}

impl MediaFeatures {
    /// Locate the pad tokens of `ids` and line the tower rows up with them. The counts
    /// must match exactly (HF asserts the same).
    pub fn assemble(
        ids: &[u32],
        positions: Option<Positions>,
        t: &TowerOutputs,
        hidden: usize,
    ) -> Result<Self> {
        let rows = |v: &[f32]| v.len() / hidden;
        let counts = |tok: u32| ids.iter().filter(|&&x| x == tok).count();
        for (name, tok, got) in [
            ("image", IMAGE_PAD, rows(&t.image)),
            ("video", VIDEO_PAD, rows(&t.video)),
            ("audio", AUDIO_PAD, rows(&t.audio)),
        ] {
            ensure!(
                counts(tok) == got,
                "{} {name} placeholders but {got} {name} feature rows",
                counts(tok)
            );
        }
        for (what, pads, ds) in [
            ("image", counts(IMAGE_PAD), &t.image_deepstack),
            ("video", counts(VIDEO_PAD), &t.video_deepstack),
        ] {
            for (k, d) in ds.iter().enumerate() {
                ensure!(
                    d.len() == pads * hidden,
                    "{what} deepstack {k}: {} rows for {pads} placeholders",
                    d.len() / hidden
                );
            }
        }
        let mut f = MediaFeatures {
            positions,
            ..Default::default()
        };
        let (mut ni, mut nv, mut na) = (0usize, 0usize, 0usize);
        for (i, &tok) in ids.iter().enumerate() {
            let take =
                |src: &[f32], n: usize| -> Vec<f32> { src[n * hidden..(n + 1) * hidden].to_vec() };
            match tok {
                IMAGE_PAD => {
                    f.splice_idx.push(i);
                    f.splice_rows.extend_from_slice(&take(&t.image, ni));
                    f.deepstack_idx.push(i);
                    for k in 0..DEEPSTACK_LAYERS {
                        f.deepstack[k].extend_from_slice(&take(&t.image_deepstack[k], ni));
                    }
                    ni += 1;
                }
                VIDEO_PAD => {
                    f.splice_idx.push(i);
                    f.splice_rows.extend_from_slice(&take(&t.video, nv));
                    f.deepstack_idx.push(i);
                    for k in 0..DEEPSTACK_LAYERS {
                        f.deepstack[k].extend_from_slice(&take(&t.video_deepstack[k], nv));
                    }
                    nv += 1;
                }
                AUDIO_PAD => {
                    f.splice_idx.push(i);
                    f.splice_rows.extend_from_slice(&take(&t.audio, na));
                    na += 1;
                }
                _ => {}
            }
        }
        f.validate(ids.len(), hidden)?;
        Ok(f)
    }

    /// Shape checks against a record of `s` tokens.
    pub fn validate(&self, s: usize, hidden: usize) -> Result<()> {
        if let Some(p) = &self.positions {
            ensure!(
                p.t.len() == s && p.h.len() == s && p.w.len() == s,
                "positions cover {} tokens, the record has {s}",
                p.len()
            );
        }
        ensure!(
            self.splice_rows.len() == self.splice_idx.len() * hidden,
            "splice rows do not match the splice positions"
        );
        ensure!(
            self.splice_idx
                .iter()
                .chain(&self.deepstack_idx)
                .all(|&i| i < s),
            "a media position is outside the record"
        );
        ensure!(
            self.deepstack_idx.windows(2).all(|w| w[0] < w[1]),
            "deepstack positions must be strictly increasing"
        );
        for (k, d) in self.deepstack.iter().enumerate() {
            ensure!(
                d.len() == self.deepstack_idx.len() * hidden
                    || (d.is_empty() && self.deepstack_idx.is_empty()),
                "deepstack {k}: {} floats for {} positions",
                d.len(),
                self.deepstack_idx.len()
            );
        }
        Ok(())
    }

    /// `x[splice_idx] = splice_rows` on the `[s, hidden]` embeddings.
    pub fn splice(&self, x: &mut [f32], hidden: usize) {
        for (k, &i) in self.splice_idx.iter().enumerate() {
            x[i * hidden..(i + 1) * hidden]
                .copy_from_slice(&self.splice_rows[k * hidden..(k + 1) * hidden]);
        }
    }

    /// Add DeepStack feature `layer` (0-based decoder layer that just ran) to the visual
    /// rows of the `[s, hidden]` hidden state; a no-op for `layer >= 3`.
    pub fn inject_deepstack(&self, layer: usize, x: &mut [f32], hidden: usize) {
        let Some(d) = self.deepstack.get(layer) else {
            return;
        };
        for (k, &i) in self.deepstack_idx.iter().enumerate() {
            for (a, b) in x[i * hidden..(i + 1) * hidden]
                .iter_mut()
                .zip(&d[k * hidden..(k + 1) * hidden])
            {
                *a += b;
            }
        }
    }
}

#[cfg(test)]
mod tests;
