//! The loaded d3 model: one noncausal backbone pass per question, then the
//! 255-code readout.
//!
//! The flow mirrors `d3_runtime.D3`: [`DecisionModel::prepare`] validates,
//! tokenizes and expands every question (`invalid_question` /
//! `max_length_exceeded` land in the per-question errors, nothing runs yet),
//! [`DecisionModel::run`] forwards the runnable ones and
//! [`DecisionModel::respond`] builds the official answer object. The HTTP handler
//! refuses a request whose `prepare` reported an `invalid_question` (422, nothing
//! runs); the CLI keeps the runtime's per-question errors in a 200-shaped answer.
//!
//! Media requests: the request's images and videos are shared by every question,
//! preprocessed and encoded once by the vision tower, and each question's
//! `<|image_pad|>` is expanded to that image's merged token count. A video
//! contributes one `<|video_pad|>` per merged vision token of **every temporal
//! pair** — the reference's `replace_video_token` renders
//! `<t seconds><|vision_start|><|video_pad|>*frame_seqlen<|vision_end|>` once per
//! pair, so a video's feature rows land in `T` separate runs of the prompt with
//! the timestamps in between. The forward embeds the expanded ids, splices the
//! vision features into those runs and rotates the full-attention layers with the
//! multimodal positions of [`mrope`](crate::decision::mrope); a text-only question
//! takes the unchanged `forward_last_normed` path.

use std::collections::HashMap;

use anyhow::{Result, anyhow, bail};
use burn::prelude::*;
use image::RgbImage;
use serde_json::{Map, Value};
use tokenizers::Tokenizer;

use super::config::{DecisionConfig, MAX_OPTIONS};
use super::format::{self, Question};
use super::mrope;
use super::request::{DecisionRequest, MediaMode};
use super::vision::processor::{VIDEO_MAX_TOKENS, planned_video_tokens, prepare_video};
use super::vision::video::{DecodedVideo, thousands, video_replacement};
use super::vision::{PreparedImage, VisionTower, prepare_from_rgb};
use crate::qwen35_intent::model::IntentModel;

/// Why a text-only d3 package refuses images: the reference's own
/// `image_unavailable` text, verbatim.
pub const NO_VISION_TOWER: &str =
    "image inputs are not available: the checkpoint has no vision tower (visual.* weights)";

/// Why a text-only d3 package refuses videos: the reference reuses its image
/// reason under the `video inputs are not available:` prefix.
pub const NO_VIDEO_TOWER: &str =
    "video inputs are not available: the checkpoint has no vision tower (visual.* weights)";

/// Why a media request cannot be served without a vision tower, images before
/// videos: the reference loads `images` first (`d3_runtime.prepare` ->
/// `load_images` then `load_videos`), so an image+video request on a text-only
/// package reports the **image** reason.
pub(crate) fn unavailable_media(images: usize, videos: usize) -> Option<&'static str> {
    if images > 0 {
        Some(NO_VISION_TOWER)
    } else if videos > 0 {
        Some(NO_VIDEO_TOWER)
    } else {
        None
    }
}

/// Why [`DecisionModel::prepare`] failed, split by what the client can do about
/// it: media the request supplied but the processor refuses is the reference's
/// `422 {"detail": ...}` string, anything else is a server fault.
#[derive(Debug)]
pub enum PrepareError {
    /// The request's images or videos cannot be used: a frame the processor
    /// refuses (its aspect ratio, its budgets), a checkpoint with no vision
    /// tower, a media item that produced no tokens, or videos over the
    /// per-request video token budget. The message is the reference's.
    InvalidMedia(String),
    /// Tokenizer, tower or device failures: `500`.
    Internal(anyhow::Error),
}

impl PrepareError {
    fn media(message: impl Into<String>) -> Self {
        Self::InvalidMedia(message.into())
    }
}

impl std::fmt::Display for PrepareError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidMedia(message) => f.write_str(message),
            Self::Internal(error) => write!(f, "{error:#}"),
        }
    }
}

impl std::error::Error for PrepareError {}

impl From<anyhow::Error> for PrepareError {
    fn from(error: anyhow::Error) -> Self {
        Self::Internal(error)
    }
}

/// A loaded decision checkpoint, ready to answer System One requests.
#[derive(Debug)]
pub struct DecisionModel {
    pub backbone: IntentModel,
    /// `[255, hidden]` f32 readout rows, one per answer code.
    pub readout: Tensor<2>,
    pub config: DecisionConfig,
    pub tokenizer: Tokenizer,
    /// The vision tower of an image/video-capable checkpoint (`config.json`
    /// `vision_config` + `visual.*`); `None` for a text-only package.
    pub vision: Option<VisionTower>,
    /// `config.json` `image_token_id`: the id one `<|image_pad|>` expands from.
    pub image_token_id: u32,
    /// `config.json` `video_token_id`: the id of one `<|video_pad|>`.
    pub video_token_id: u32,
}

/// Which media a [`MediaSpan`] belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaKind {
    Image,
    Video,
}

/// Where one media item's feature rows go in an expanded prompt.
///
/// An image is one contiguous run. A video is one run **per temporal pair** (the
/// timestamps and vision markers between two pairs stay text), so its features
/// are spliced pair by pair in order.
#[derive(Debug, Clone)]
pub struct MediaSpan {
    pub kind: MediaKind,
    pub runs: Vec<(usize, usize)>,
}

#[allow(dead_code)] // `len` is read by the tests and the probe's reports
impl MediaSpan {
    /// The rows this span holds in total.
    pub fn len(&self) -> usize {
        self.runs.iter().map(|(start, end)| end - start).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.runs.is_empty()
    }
}

/// One runnable question: its expanded prompt and where the media features go.
#[derive(Debug, Clone)]
pub struct PreparedSequence {
    /// Question id.
    pub id: String,
    /// Token ids with every placeholder expanded: an image to one id per merged
    /// token, a video already one id per merged token of each temporal pair.
    pub ids: Vec<u32>,
    /// `[ids.len()]`, `1` on image rows and `2` on video rows
    /// (the reference's `mm_token_type_ids`).
    pub mm_types: Vec<u8>,
    /// The media spans in prompt order (images first, then videos).
    pub spans: Vec<MediaSpan>,
}

/// One request image: the preprocessed pixels and the tower's merged features
/// (`[tokens, out_hidden]`), shared by every question of the request.
#[derive(Debug)]
pub struct PreparedVision {
    pub image: PreparedImage,
    pub features: Tensor<2>,
}

/// One request video: the sampled frames it was read at, its patch grid, the
/// prompt group its timestamps produced and the tower's merged features
/// (`[tokens, out_hidden]`), shared by every question of the request.
///
/// The sampling metadata (`frames`, `indices`, `fps`, `total`, `tokens`) is kept
/// as the provenance of the prompt's timestamps: `--dump-prompts` and the video
/// probe report it, the forward needs only `grid`/`frame_tokens`/`features`.
#[allow(dead_code)]
#[derive(Debug)]
pub struct PreparedVideo {
    /// Sampled frames (even: one temporal pair per two frames).
    pub frames: usize,
    /// The source frame number of every sample.
    pub indices: Vec<usize>,
    /// The source frame rate the timestamps are computed from.
    pub fps: f64,
    /// The source frame count the sampling ran over.
    pub total: usize,
    /// `[T = frames / 2, H_patches, W_patches]`.
    pub grid: [usize; 3],
    /// Merged tokens the video contributes (`T * H * W / merge^2`).
    pub tokens: usize,
    /// Merged tokens of one temporal pair (`H * W / merge^2`): the number of
    /// `<|video_pad|>` ids in one prompt group.
    pub frame_tokens: usize,
    /// The reference's `replace_video_token` string (one timestamped group per
    /// pair), inserted between the chat template's `vision_start`/`vision_end`.
    pub replacement: String,
    /// Tokens the replacement adds to a prompt (`expanded` in the cap message).
    pub replacement_tokens: usize,
    pub features: Tensor<2>,
}

/// One validated request: the questions that will run and the errors of those
/// that will not (`d3_runtime.Prepared`).
#[derive(Debug)]
pub struct Prepared {
    /// Question ids in request order.
    pub keys: Vec<String>,
    /// The validated question of every entry that got past normalization.
    pub questions: HashMap<String, Question>,
    /// Question id -> its answer error record (`invalid_question` or
    /// `max_length_exceeded`).
    pub errors: Map<String, Value>,
    /// The runnable questions, request order.
    pub sequences: Vec<PreparedSequence>,
    /// The request's images, in list order (empty for a text request).
    pub images: Vec<PreparedVision>,
    /// The request's videos, in list order (empty unless it has videos).
    pub videos: Vec<PreparedVideo>,
}

impl Prepared {
    /// The merged token count of every request image, in list order.
    pub fn image_token_counts(&self) -> Vec<usize> {
        self.images.iter().map(|media| media.image.tokens).collect()
    }

    /// Question id -> message of every question that failed to normalize, in
    /// request order. The HTTP server refuses the request (422) when this is not
    /// empty; `max_length_exceeded` is not part of it (that stays a 200 answer).
    pub fn invalid_questions(&self) -> Map<String, Value> {
        let mut invalid = Map::new();
        for key in &self.keys {
            if let Some(error) = self.errors.get(key)
                && error.get("error").and_then(Value::as_str) == Some("invalid_question")
                && let Some(message) = error.get("message")
            {
                invalid.insert(key.clone(), message.clone());
            }
        }
        invalid
    }
}

impl DecisionModel {
    /// The per-question prompt-token cap for a request (`--max-state-tokens` /
    /// the server's `--max-tokens`), narrowed by the checkpoint's own
    /// `max_length` when it declares one.
    pub fn prompt_cap(&self, max_tokens: usize) -> usize {
        match self.config.max_length {
            Some(limit) => max_tokens.min(limit),
            None => max_tokens,
        }
    }

    /// Whether the checkpoint carries a vision tower (a media request needs one).
    pub fn has_vision(&self) -> bool {
        self.vision.is_some()
    }

    /// Preprocess and encode the request's images and videos once.
    ///
    /// The media are already decoded (see
    /// [`DecisionRequest::decode_images`]/[`DecisionRequest::decode_videos`]);
    /// this runs the Qwen2-VL/Qwen3-VL processors, the per-request video token
    /// budget and the vision tower. Returns the images and videos with their
    /// merged features, in list order.
    pub fn prepare_media(
        &self,
        images: &[RgbImage],
        videos: &[DecodedVideo],
    ) -> Result<(Vec<PreparedVision>, Vec<PreparedVideo>), PrepareError> {
        if images.is_empty() && videos.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        let Some(vision) = self.vision.as_ref() else {
            let reason = unavailable_media(images.len(), videos.len())
                .expect("media was checked to be nonempty");
            return Err(PrepareError::media(reason));
        };
        let config = &vision.config;
        let device = &self.backbone.device;
        // Images first: `_prepare_videos` renders (and so refuses) every image
        // before it plans the video budget, so a request with a bad image **and**
        // an over-budget video reports the image.
        let mut pixels = Vec::with_capacity(images.len());
        for (index, image) in images.iter().enumerate() {
            // The processor's own refusals (aspect ratio, budgets) are the
            // client's problem; they carry the reference's text.
            let image = prepare_from_rgb(image, config, device).map_err(|error| {
                PrepareError::media(format!("image rejected by the processor: {error:#}"))
            })?;
            if image.tokens == 0 {
                return Err(PrepareError::media(format!(
                    "images[{index}]: the processor produced no image tokens"
                )));
            }
            pixels.push(image);
        }
        if !videos.is_empty() && self.video_token_id == u32::MAX {
            return Err(PrepareError::media(
                "video inputs are not available: the checkpoint declares no video token",
            ));
        }
        // Then the video budget: all videos of one request together may not exceed
        // 16,384 tokens.
        let mut planned = 0usize;
        for video in videos {
            planned += planned_video_tokens(video.frame_count(), video.height, video.width, config)
                .map_err(|error| {
                    PrepareError::media(format!("video rejected by the processor: {error:#}"))
                })?;
        }
        if planned > VIDEO_MAX_TOKENS {
            return Err(PrepareError::media(format!(
                "the videos take {} input tokens, over the {} tokens a request may spend on \
                 videos; send fewer or shorter videos",
                thousands(planned as u64),
                thousands(VIDEO_MAX_TOKENS as u64)
            )));
        }
        let mut prepared_images = Vec::with_capacity(pixels.len());
        if !pixels.is_empty() {
            let features = vision.encode_all(&pixels)?;
            let [rows, _] = features.dims();
            let total: usize = pixels.iter().map(|image| image.tokens).sum();
            if rows != total {
                return Err(PrepareError::Internal(anyhow!(
                    "the vision tower produced {rows} rows for {total} image tokens"
                )));
            }
            let mut offset = 0usize;
            for image in pixels {
                let tokens = image.tokens;
                let rows = features.clone().slice(s![offset..offset + tokens, ..]);
                offset += tokens;
                prepared_images.push(PreparedVision {
                    image,
                    features: rows,
                });
            }
        }
        let merge = config.spatial_merge_size;
        let mut prepared_videos = Vec::with_capacity(videos.len());
        for (index, video) in videos.iter().enumerate() {
            let pixel = prepare_video(&video.frames, config, device).map_err(|error| {
                PrepareError::media(format!(
                    "videos[{index}]: video rejected by the processor: {error:#}"
                ))
            })?;
            let features = vision.encode_video(&pixel)?;
            let [rows, _] = features.dims();
            if rows != pixel.tokens {
                return Err(PrepareError::Internal(anyhow!(
                    "the vision tower produced {rows} rows for the {} video tokens of videos[{index}]",
                    pixel.tokens
                )));
            }
            let stamps = video.timestamps(config.temporal_patch_size);
            let replacement = video_replacement(pixel.grid, merge, &stamps, format::VIDEO_TOKEN)
                .map_err(|error| PrepareError::media(format!("videos[{index}]: {error:#}")))?;
            let replacement_tokens = self
                .tokenizer
                .encode(replacement.as_str(), false)
                .map_err(|error| anyhow!("tokenize the videos[{index}] prompt group: {error}"))?
                .get_ids()
                .len();
            prepared_videos.push(PreparedVideo {
                frames: video.frame_count(),
                indices: video.indices.clone(),
                fps: video.fps,
                total: video.total,
                grid: pixel.grid,
                tokens: pixel.tokens,
                frame_tokens: pixel.frame_tokens,
                replacement,
                replacement_tokens,
                features,
            });
        }
        Ok((prepared_images, prepared_videos))
    }

    /// Validate, render, tokenize and expand every question of one request.
    ///
    /// Envelope problems are already refused by [`DecisionRequest::from_value`];
    /// `images`/`videos` are the decoded request media (see
    /// [`DecisionModel::prepare_media`]). A question that cannot be answered ends
    /// up in [`Prepared::errors`] and the rest still runs.
    pub fn prepare(
        &self,
        req: &DecisionRequest,
        images: &[RgbImage],
        videos: &[DecodedVideo],
        max_tokens: usize,
    ) -> Result<Prepared, PrepareError> {
        let cap = self.prompt_cap(max_tokens);
        let (image_media, video_media) = self.prepare_media(images, videos)?;
        let image_tokens: Vec<usize> = image_media.iter().map(|media| media.image.tokens).collect();
        let video_layouts: Vec<(usize, usize)> = video_media
            .iter()
            .map(|video| (video.frame_tokens, video.grid[0]))
            .collect();
        let n_images = image_tokens.len();
        let n_videos = video_layouts.len();
        let image_total: usize = image_tokens.iter().sum();
        let video_total: usize = video_media
            .iter()
            .map(|video| video.replacement_tokens)
            .sum();
        let visual = image_total + video_total;
        let mut prepared = Prepared {
            keys: req.questions.keys().cloned().collect(),
            questions: HashMap::new(),
            errors: Map::new(),
            sequences: Vec::new(),
            images: image_media,
            videos: video_media,
        };
        for (id, value) in &req.questions {
            let raw_kind = value.get("type").cloned().unwrap_or(Value::Null);
            let question = match format::normalize_question(value) {
                Ok(question) => question,
                Err(message) => {
                    prepared.errors.insert(
                        id.clone(),
                        error_answer(raw_kind, "invalid_question", message),
                    );
                    continue;
                }
            };
            let kind = Value::String(question.kind.as_str().to_string());
            let user = match format::user_prompt(&req.state, &question, &self.config.codes) {
                Ok(user) => user,
                Err(message) => {
                    prepared.errors.insert(
                        id.clone(),
                        error_answer(kind.clone(), "invalid_question", message.to_string()),
                    );
                    prepared.questions.insert(id.clone(), question);
                    continue;
                }
            };
            // A media prompt's placeholder count must match the request, or a
            // state/option carrying a literal placeholder would splice or drop
            // rows; the reference refuses that question. A **text** prompt has no
            // such check (`d3_runtime._prepare_text`), so a literal token there is
            // just a token the model reads — as it was before images landed.
            if n_images > 0 || n_videos > 0 {
                let template = format::chatml_template(&user, n_images, n_videos);
                if let Some(message) = placeholder_problem(&template, n_images, n_videos) {
                    prepared.errors.insert(
                        id.clone(),
                        error_answer(kind.clone(), "invalid_question", message.to_string()),
                    );
                    prepared.questions.insert(id.clone(), question);
                    continue;
                }
            }
            let prompt = media_prompt(&user, n_images, &prepared.videos);
            let encoding = self
                .tokenizer
                .encode(prompt.as_str(), false)
                .map_err(|e| anyhow!("tokenize the '{id}' prompt: {e}"))?;
            let raw = encoding.get_ids();
            let sequence = match self.expand(id, raw, &image_tokens, &video_layouts) {
                Ok(sequence) => sequence,
                Err(message) => {
                    prepared.errors.insert(
                        id.clone(),
                        error_answer(kind.clone(), "invalid_question", message.to_string()),
                    );
                    prepared.questions.insert(id.clone(), question);
                    continue;
                }
            };
            prepared.questions.insert(id.clone(), question);
            let length = sequence.ids.len();
            if length > cap {
                let message = if n_images == 0 && n_videos == 0 {
                    format!(
                        "the question prompt has {length} tokens, over the maximum context \
                         length of {cap} tokens; nothing was truncated"
                    )
                } else if n_videos == 0 {
                    format!(
                        "the question prompt has {length} tokens ({visual} for {n_images} \
                         image(s)), over the maximum context length of {cap} tokens; \
                         nothing was truncated"
                    )
                } else {
                    format!(
                        "the question prompt has {length} tokens ({visual} for {n_images} \
                         image(s) and {n_videos} video(s)), over the maximum context length \
                         of {cap} tokens; nothing was truncated"
                    )
                };
                prepared.errors.insert(
                    id.clone(),
                    error_answer(kind, "max_length_exceeded", message),
                );
                continue;
            }
            prepared.sequences.push(sequence);
        }
        Ok(prepared)
    }

    /// Expand the placeholders of one tokenized prompt: each `<|image_pad|>`
    /// becomes `images[i]` copies of itself (the i-th image's merged token
    /// count), and each maximal run of `<|video_pad|>` is one temporal pair that
    /// must hold `videos[j].0` ids (the pair's merged token count), grouped into
    /// the j-th video's span.
    fn expand(
        &self,
        id: &str,
        raw: &[u32],
        images: &[usize],
        videos: &[(usize, usize)],
    ) -> Result<PreparedSequence, String> {
        expand_placeholders(
            id,
            raw,
            self.image_token_id,
            self.video_token_id,
            images,
            videos,
        )
    }

    /// Forward every runnable question (request order) and return its option
    /// probabilities plus the answered questions' prompt-token sum.
    pub fn run(&mut self, prepared: &Prepared) -> Result<(HashMap<String, Vec<f64>>, usize)> {
        let mut probabilities = HashMap::with_capacity(prepared.sequences.len());
        let mut tokens = 0usize;
        for sequence in &prepared.sequences {
            let id = &sequence.id;
            let question = prepared
                .questions
                .get(id)
                .ok_or_else(|| anyhow!("internal error: '{id}' has no validated question"))?;
            probabilities.insert(
                id.clone(),
                self.probabilities(question, sequence, &prepared.images, &prepared.videos)?,
            );
            tokens += sequence.ids.len();
        }
        Ok((probabilities, tokens))
    }

    /// The official System One response for one prepared request
    /// (`d3_runtime.D3.respond`): every question id in request order, either its
    /// answer or its error record.
    pub fn respond(
        &self,
        model: &str,
        prepared: &Prepared,
        probabilities: &HashMap<String, Vec<f64>>,
        tokens: usize,
    ) -> Value {
        let mut answers = Map::new();
        for id in &prepared.keys {
            if let Some(error) = prepared.errors.get(id) {
                answers.insert(id.clone(), error.clone());
                continue;
            }
            let Some(question) = prepared.questions.get(id) else {
                continue;
            };
            let Some(values) = probabilities.get(id) else {
                continue;
            };
            let answer = match format::product_answer(question, values) {
                Ok(answer) => answer,
                Err(message) => error_answer(
                    Value::String(question.kind.as_str().to_string()),
                    "invalid_model_output",
                    message,
                ),
            };
            answers.insert(id.clone(), answer);
        }
        serde_json::json!({
            "model": model,
            "answers": Value::Object(answers),
            "usage": {"input_tokens": tokens, "output_tokens": 0},
        })
    }

    /// Answer one request with the runtime's own semantics (per-question errors
    /// stay in the answer). Returns `(response, input_tokens, answered_questions)`;
    /// only envelope and media problems are errors.
    pub fn system_one(
        &mut self,
        model: &str,
        req: DecisionRequest,
        max_tokens: usize,
        media: MediaMode,
    ) -> Result<(Value, usize, usize)> {
        let images = req.decode_images(media).map_err(|e| anyhow!("{e}"))?;
        let videos = req.decode_videos(media).map_err(|e| anyhow!("{e}"))?;
        let prepared = self.prepare(&req, &images, &videos, max_tokens)?;
        let (probabilities, tokens) = self.run(&prepared)?;
        let answered = prepared.sequences.len();
        Ok((
            self.respond(model, &prepared, &probabilities, tokens),
            tokens,
            answered,
        ))
    }

    /// One forward pass: the final-normed last hidden through the readout, the
    /// invalid codes masked to `-inf`, the temperature and the softmax (the
    /// reference's `logits` + `probabilities`).
    fn probabilities(
        &mut self,
        question: &Question,
        sequence: &PreparedSequence,
        images: &[PreparedVision],
        videos: &[PreparedVideo],
    ) -> Result<Vec<f64>> {
        self.backbone.cache.reset();
        let hidden = if sequence.spans.is_empty() {
            self.backbone.forward_last_normed(&sequence.ids)
        } else {
            let embeds = self.backbone.embed_ids(&sequence.ids);
            let embeds = splice(embeds, sequence, images, videos)?;
            let mut image_grids: Vec<[usize; 3]> = Vec::new();
            let mut video_grids: Vec<[usize; 3]> = Vec::new();
            for span in &sequence.spans {
                match span.kind {
                    MediaKind::Image => {
                        let image = images.get(image_grids.len()).ok_or_else(|| {
                            anyhow!(
                                "the '{}' prompt has more image spans than the request's images",
                                sequence.id
                            )
                        })?;
                        image_grids.push(image.image.grid);
                    }
                    MediaKind::Video => {
                        let video = videos.get(video_grids.len()).ok_or_else(|| {
                            anyhow!(
                                "the '{}' prompt has more video spans than the request's videos",
                                sequence.id
                            )
                        })?;
                        video_grids.push(video.grid);
                    }
                }
            }
            let merge = self
                .vision
                .as_ref()
                .map(|vision| vision.config.spatial_merge_size)
                .unwrap_or(2);
            let tables = mrope::rope_tables(
                &sequence.mm_types,
                &image_grids,
                &video_grids,
                &self.backbone.cfg,
                merge,
                &self.backbone.device,
            )?;
            self.backbone
                .forward_last_normed_embeds(embeds, Some(&tables))
        };
        let logits = hidden.matmul(self.readout.clone().transpose());
        let values: Vec<f32> = logits
            .to_data()
            .try_to_vec()
            .map_err(|e| anyhow!("read the readout logits: {e}"))?;
        if values.len() < MAX_OPTIONS {
            return Err(anyhow!(
                "the readout returned {} logits, expected {MAX_OPTIONS}",
                values.len()
            ));
        }
        Ok(code_probabilities(
            &values,
            question.keys.len(),
            self.config.temperature,
        ))
    }
}

/// The prompt of one user turn for a request's media: what the model tokenizes.
///
/// It is the chat template with one `<|vision_start|><|image_pad|><|vision_end|>`
/// group per image, then (when the request has videos) `<|vision_start|>` + the
/// video's replacement + `<|vision_end|>` per video. Image placeholders are
/// expanded to their merged token counts when the ids are assembled, not here.
pub fn media_prompt(user: &str, n_images: usize, videos: &[PreparedVideo]) -> String {
    if videos.is_empty() {
        return if n_images == 0 {
            format::chatml(user)
        } else {
            format::chatml_images(user, n_images)
        };
    }
    media_prompt_with(user, &format::image_groups(n_images), videos)
}

/// [`media_prompt`] with an explicit image part.
fn media_prompt_with(user: &str, images: &str, videos: &[PreparedVideo]) -> String {
    let mut videos_part = String::new();
    for video in videos {
        videos_part.push_str("<|vision_start|>");
        videos_part.push_str(&video.replacement);
        videos_part.push_str("<|vision_end|>");
    }
    format::chatml_media(user, images, &videos_part)
}

/// The reference's own `expanded_text` of one question, for `--dump-prompts` and
/// byte comparisons: the same prompt as [`media_prompt`], except that a **video**
/// request expands each image placeholder to its merged token count — the
/// reference's `video_logits` feeds `replace_image_token`'s output to its
/// tokenizer, while its image path leaves one placeholder per image. Both
/// renderings tokenize to identical ids.
pub fn dump_prompt(user: &str, image_tokens: &[usize], videos: &[PreparedVideo]) -> String {
    if videos.is_empty() {
        return media_prompt(user, image_tokens.len(), videos);
    }
    let mut images = String::new();
    for tokens in image_tokens {
        images.push_str("<|vision_start|>");
        for _ in 0..*tokens {
            images.push_str(format::IMAGE_TOKEN);
        }
        images.push_str("<|vision_end|>");
    }
    media_prompt_with(user, &images, videos)
}

/// Expand the placeholders of one tokenized prompt.
///
/// Each `<|image_pad|>` becomes `images[i]` copies of itself (the i-th image's
/// merged token count) and is marked as image rows. A maximal run of
/// `<|video_pad|>` is one temporal pair of the j-th video: it must already hold
/// `videos[j].0` ids (the pair's merged token count — the reference's
/// `replace_video_token` emitted them), is marked as video rows, and is appended
/// to that video's span as one run. `videos[j].1` is the video's pair count, so
/// the span closes after its last pair.
///
/// `images` and `videos` both empty is the **text** path: a literal
/// `<|image_pad|>`/`<|video_pad|>` in the state is then just a token the model
/// reads (the reference's text path expands nothing either), so nothing is
/// refused.
pub(crate) fn expand_placeholders(
    id: &str,
    raw: &[u32],
    image_token_id: u32,
    video_token_id: u32,
    images: &[usize],
    videos: &[(usize, usize)],
) -> Result<PreparedSequence, String> {
    let mut ids: Vec<u32> = Vec::with_capacity(raw.len());
    let mut mm_types: Vec<u8> = Vec::with_capacity(raw.len());
    let mut spans: Vec<MediaSpan> = Vec::new();
    let mut next_image = 0usize;
    let mut next_video = 0usize;
    let mut next_pair = 0usize;
    let mut index = 0usize;
    while index < raw.len() {
        let token = raw[index];
        if token == image_token_id && !images.is_empty() {
            let Some(count) = images.get(next_image) else {
                return Err(format!(
                    "the '{id}' prompt has more image placeholders than the {} image(s) of \
                     the request",
                    images.len()
                ));
            };
            let start = ids.len();
            ids.extend(std::iter::repeat_n(token, *count));
            mm_types.extend(std::iter::repeat_n(mrope::TYPE_IMAGE, *count));
            spans.push(MediaSpan {
                kind: MediaKind::Image,
                runs: vec![(start, ids.len())],
            });
            next_image += 1;
            index += 1;
        } else if token == video_token_id && !videos.is_empty() {
            let Some(&(frame_tokens, pairs)) = videos.get(next_video) else {
                return Err(format!(
                    "the '{id}' prompt has more video placeholders than the {} video(s) of \
                     the request",
                    videos.len()
                ));
            };
            let start = ids.len();
            let mut run = 0usize;
            while index < raw.len() && raw[index] == video_token_id {
                ids.push(video_token_id);
                mm_types.push(mrope::TYPE_VIDEO);
                run += 1;
                index += 1;
            }
            if run != frame_tokens {
                return Err(format!(
                    "the '{id}' prompt has a video group of {run} tokens, but one temporal \
                     pair of videos[{next_video}] holds {frame_tokens}"
                ));
            }
            if next_pair == 0 {
                spans.push(MediaSpan {
                    kind: MediaKind::Video,
                    runs: vec![(start, ids.len())],
                });
            } else if let Some(span) = spans.last_mut() {
                span.runs.push((start, ids.len()));
            }
            next_pair += 1;
            if next_pair == pairs {
                next_video += 1;
                next_pair = 0;
            }
        } else {
            ids.push(token);
            mm_types.push(mrope::TYPE_TEXT);
            index += 1;
        }
    }
    if next_image != images.len() {
        return Err(format!(
            "the '{id}' prompt uses {next_image} of the {} image(s) of the request",
            images.len()
        ));
    }
    if next_video != videos.len() {
        return Err(format!(
            "the '{id}' prompt uses {next_video} of the {} video(s) of the request",
            videos.len()
        ));
    }
    Ok(PreparedSequence {
        id: id.to_string(),
        ids,
        mm_types,
        spans,
    })
}

/// Why a media prompt is unusable, or `None` when it is fine.
///
/// The reference (`d3_runtime._prepare_images` / `_prepare_videos`) refuses a
/// question whose rendered prompt carries a different number of
/// `<|image_pad|>`/`<|video_pad|>` placeholders than the request has images /
/// videos, because the expansion would then disagree with them. Its **text**
/// path has no such check, so a text-only request is never refused for a literal
/// placeholder token.
pub(crate) fn placeholder_problem(
    prompt: &str,
    n_images: usize,
    n_videos: usize,
) -> Option<&'static str> {
    if n_images == 0 && n_videos == 0 {
        return None;
    }
    if prompt.matches(format::IMAGE_TOKEN).count() != n_images
        || prompt.matches(format::VIDEO_TOKEN).count() != n_videos
    {
        return Some("the state or question contains a literal image or video placeholder token");
    }
    None
}

/// Write each media item's features into its span's runs of `[1, seq, hidden]`.
///
/// Row replacement keeps the length, so the spans (computed on the expanded
/// prompt) stay valid while the rows are spliced in prompt order. A video's rows
/// go in pair by pair: the first `frame_tokens` rows fill the first run, the next
/// `frame_tokens` the second, and so on.
fn splice(
    mut embeds: Tensor<3>,
    sequence: &PreparedSequence,
    images: &[PreparedVision],
    videos: &[PreparedVideo],
) -> Result<Tensor<3>> {
    let [batch, seq, hidden] = embeds.dims();
    let mut next_image = 0usize;
    let mut next_video = 0usize;
    for span in &sequence.spans {
        let features = match span.kind {
            MediaKind::Image => {
                let image = images.get(next_image).ok_or_else(|| {
                    anyhow!(
                        "the '{}' prompt has more image spans than the request's images",
                        sequence.id
                    )
                })?;
                next_image += 1;
                image.features.clone()
            }
            MediaKind::Video => {
                let video = videos.get(next_video).ok_or_else(|| {
                    anyhow!(
                        "the '{}' prompt has more video spans than the request's videos",
                        sequence.id
                    )
                })?;
                next_video += 1;
                video.features.clone()
            }
        };
        let [rows, width] = features.dims();
        if width != hidden {
            bail!("the vision features are {width} wide, the text embeddings {hidden}");
        }
        let mut row = 0usize;
        for (start, end) in &span.runs {
            let len = end - start;
            if row + len > rows {
                bail!(
                    "the media span [{start}, {end}) runs past its {rows} feature rows \
                     ({row} placed so far)"
                );
            }
            let block = features
                .clone()
                .slice(s![row..row + len, ..])
                .reshape([batch, len, hidden]);
            let before = embeds.clone().slice(s![.., 0..*start, ..]);
            let after = embeds.slice(s![.., *end..seq, ..]);
            embeds = Tensor::cat(vec![before, block, after], 1);
            row += len;
        }
        if row != rows {
            bail!("the media's {rows} feature rows do not fill its spans ({row} placed)");
        }
    }
    Ok(embeds)
}

/// The softmax over the first `n` codes of one masked readout row.
///
/// The reference divides the f32 logits by the temperature, softmaxes (with the
/// row max subtracted, over the valid codes only) and reads the first `n`
/// entries back; invalid codes are `-inf` and contribute nothing.
pub fn code_probabilities(logits: &[f32], n: usize, temperature: f64) -> Vec<f64> {
    let n = n.min(logits.len());
    let mut max = f32::NEG_INFINITY;
    for &value in &logits[..n] {
        if value > max {
            max = value;
        }
    }
    let mut exps = Vec::with_capacity(n);
    let mut sum = 0f32;
    for &value in &logits[..n] {
        let scaled = ((value - max) as f64 / temperature).exp() as f32;
        exps.push(scaled);
        sum += scaled;
    }
    if sum <= 0.0 || sum.is_nan() {
        // Every code was masked (no valid option); the caller reports the answer
        // as invalid output rather than dividing by zero.
        return vec![f32::NAN as f64; n];
    }
    exps.iter().map(|value| (*value / sum) as f64).collect()
}

/// One per-question error record (`{"type", "error", "message"}`).
fn error_answer(kind: Value, error: &'static str, message: String) -> Value {
    serde_json::json!({"type": kind, "error": error, "message": message})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_readout_softmax_masks_later_codes() {
        let mut logits = vec![f32::NEG_INFINITY; MAX_OPTIONS];
        logits[0] = 2.0;
        logits[1] = 1.0;
        logits[2] = -1.0;
        let probabilities = code_probabilities(&logits, 3, 1.0);
        assert_eq!(probabilities.len(), 3);
        let total: f64 = probabilities.iter().sum();
        assert!((total - 1.0).abs() < 1e-6, "{probabilities:?}");
        assert!(probabilities[0] > probabilities[1]);
        assert!(probabilities[1] > probabilities[2]);
        // The same row cut to two options renormalizes over them.
        let two = code_probabilities(&logits, 2, 1.0);
        assert!((two.iter().sum::<f64>() - 1.0).abs() < 1e-6);
        assert!((two[0] - probabilities[0] / (probabilities[0] + probabilities[1])).abs() < 1e-6);
    }

    #[test]
    fn the_temperature_flattens_the_distribution() {
        let mut logits = vec![f32::NEG_INFINITY; MAX_OPTIONS];
        logits[0] = 2.0;
        logits[1] = 0.0;
        let hot = code_probabilities(&logits, 2, 1.0);
        let warm = code_probabilities(&logits, 2, 4.0);
        assert!(warm[0] < hot[0], "{hot:?} vs {warm:?}");
        assert!(warm[1] > hot[1]);
        assert!((warm.iter().sum::<f64>() - 1.0).abs() < 1e-6);
    }

    fn prepared_with(errors: Vec<(&str, &'static str)>) -> Prepared {
        let mut prepared = Prepared {
            keys: errors.iter().map(|(id, _)| id.to_string()).collect(),
            questions: HashMap::new(),
            errors: Map::new(),
            sequences: Vec::new(),
            images: Vec::new(),
            videos: Vec::new(),
        };
        for (id, error) in errors {
            prepared.errors.insert(
                id.to_string(),
                error_answer(
                    Value::String("choice".to_string()),
                    error,
                    format!("{error} for {id}"),
                ),
            );
        }
        prepared
    }

    /// F4/F5: images are reported before videos, with the reference's wording.
    #[test]
    fn the_media_unavailable_reason_names_images_first() {
        assert_eq!(unavailable_media(0, 0), None);
        assert_eq!(unavailable_media(1, 0), Some(NO_VISION_TOWER));
        assert_eq!(unavailable_media(0, 1), Some(NO_VIDEO_TOWER));
        assert_eq!(unavailable_media(2, 3), Some(NO_VISION_TOWER));
        assert!(NO_VISION_TOWER.starts_with("image inputs are not available: "));
        assert!(NO_VIDEO_TOWER.starts_with("video inputs are not available: "));
        for message in [NO_VISION_TOWER, NO_VIDEO_TOWER] {
            assert!(
                message.ends_with("the checkpoint has no vision tower (visual.* weights)"),
                "{message}"
            );
        }
    }

    #[test]
    fn only_invalid_questions_refuse_a_request() {
        let prepared = prepared_with(vec![
            ("a", "invalid_question"),
            ("b", "max_length_exceeded"),
        ]);
        let invalid = prepared.invalid_questions();
        assert_eq!(invalid.len(), 1);
        assert_eq!(invalid["a"], "invalid_question for a");
        assert!(
            prepared.sequences.is_empty(),
            "nothing runs for a refused question"
        );
        assert!(prepared_with(vec![]).invalid_questions().is_empty());
    }

    /// The splice replaces exactly the expanded rows and leaves the rest alone.
    #[test]
    fn spliced_features_replace_their_rows() {
        let device = Device::default();
        let embeds = Tensor::<3>::from_data(
            burn::tensor::TensorData::new((0..12).map(|i| i as f32).collect::<Vec<_>>(), [1, 4, 3]),
            &device,
        );
        let features = Tensor::<2>::from_data(
            burn::tensor::TensorData::new(vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0], [2, 3]),
            &device,
        );
        let sequence = PreparedSequence {
            id: "q".to_string(),
            ids: vec![1, 2, 3, 4],
            mm_types: vec![
                mrope::TYPE_TEXT,
                mrope::TYPE_IMAGE,
                mrope::TYPE_IMAGE,
                mrope::TYPE_TEXT,
            ],
            spans: vec![MediaSpan {
                kind: MediaKind::Image,
                runs: vec![(1, 3)],
            }],
        };
        let vision = PreparedVision {
            image: PreparedImage {
                pixel_values: Tensor::zeros([0, 0], &device),
                grid: [1, 2, 2],
                tokens: 2,
                source: (4, 4),
                resized: (4, 4),
            },
            features,
        };
        let out = splice(embeds, &sequence, &[vision], &[]).unwrap();
        let values: Vec<f32> = out.to_data().try_to_vec().unwrap();
        assert_eq!(
            values,
            vec![
                0.0, 1.0, 2.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 9.0, 10.0, 11.0
            ]
        );
    }

    /// A video's feature rows go in pair by pair: the first `frame_tokens` rows
    /// into the first run, the rest into the second, with the timestamp rows in
    /// between untouched.
    #[test]
    fn a_videos_features_fill_its_pair_runs() {
        let device = Device::default();
        let embeds = Tensor::<3>::from_data(
            burn::tensor::TensorData::new((0..18).map(|i| i as f32).collect::<Vec<_>>(), [1, 6, 3]),
            &device,
        );
        let features = Tensor::<2>::from_data(
            burn::tensor::TensorData::new((100..112).map(|i| i as f32).collect::<Vec<_>>(), [4, 3]),
            &device,
        );
        let sequence = PreparedSequence {
            id: "q".to_string(),
            ids: vec![1, 8, 8, 2, 8, 8],
            mm_types: vec![
                mrope::TYPE_TEXT,
                mrope::TYPE_VIDEO,
                mrope::TYPE_VIDEO,
                mrope::TYPE_TEXT,
                mrope::TYPE_VIDEO,
                mrope::TYPE_VIDEO,
            ],
            spans: vec![MediaSpan {
                kind: MediaKind::Video,
                runs: vec![(1, 3), (4, 6)],
            }],
        };
        let video = PreparedVideo {
            frames: 4,
            indices: vec![0, 1, 2, 3],
            fps: 8.0,
            total: 4,
            grid: [2, 2, 2],
            tokens: 4,
            frame_tokens: 2,
            replacement: String::new(),
            replacement_tokens: 0,
            features,
        };
        let out = splice(embeds, &sequence, &[], &[video]).unwrap();
        let values: Vec<f32> = out.to_data().try_to_vec().unwrap();
        assert_eq!(
            values,
            vec![
                0.0, 1.0, 2.0, // text
                100.0, 101.0, 102.0, 103.0, 104.0, 105.0, // pair 1
                9.0, 10.0, 11.0, // the timestamp text stays
                106.0, 107.0, 108.0, 109.0, 110.0, 111.0, // pair 2
            ]
        );
    }

    /// The expansion: text prompts keep every token (even a literal placeholder),
    /// image prompts turn each placeholder into its image's rows, and a video's
    /// pair runs become one span.
    #[test]
    fn placeholders_expand_only_for_media_prompts() {
        let (image, video) = (7u32, 8u32);
        let raw = [1, image, 2, video, 3];
        // No media: the literal tokens are text and nothing is expanded.
        let text = expand_placeholders("q", &raw, image, video, &[], &[]).unwrap();
        assert_eq!(text.ids, raw);
        assert_eq!(text.mm_types, vec![mrope::TYPE_TEXT; raw.len()]);
        assert!(text.spans.is_empty());

        // One image of two merged tokens.
        let one = expand_placeholders("q", &[1, image, 2], image, video, &[2], &[]).unwrap();
        assert_eq!(one.ids, [1, image, image, 2]);
        assert_eq!(
            one.mm_types,
            [
                mrope::TYPE_TEXT,
                mrope::TYPE_IMAGE,
                mrope::TYPE_IMAGE,
                mrope::TYPE_TEXT
            ]
        );
        assert_eq!(one.spans.len(), 1);
        assert_eq!(one.spans[0].kind, MediaKind::Image);
        assert_eq!(one.spans[0].runs, [(1, 3)]);

        // Two images: each placeholder takes its own image's count.
        let two = expand_placeholders("q", &[image, image], image, video, &[1, 3], &[]).unwrap();
        assert_eq!(two.ids.len(), 4);
        assert_eq!(two.spans[0].runs, [(0, 1)]);
        assert_eq!(two.spans[1].runs, [(1, 4)]);

        // A video's pairs: each run of `frame_tokens` pads is one run of its span,
        // and the text between two runs stays text.
        let raw = [1, video, video, 2, video, video, 3];
        let run = expand_placeholders("q", &raw, image, video, &[], &[(2, 2)]).unwrap();
        assert_eq!(run.ids, raw, "video pads are already expanded");
        assert_eq!(
            run.mm_types,
            [
                mrope::TYPE_TEXT,
                mrope::TYPE_VIDEO,
                mrope::TYPE_VIDEO,
                mrope::TYPE_TEXT,
                mrope::TYPE_VIDEO,
                mrope::TYPE_VIDEO,
                mrope::TYPE_TEXT
            ]
        );
        assert_eq!(run.spans.len(), 1);
        assert_eq!(run.spans[0].kind, MediaKind::Video);
        assert_eq!(run.spans[0].runs, [(1, 3), (4, 6)]);
        assert_eq!(run.spans[0].len(), 4);

        // Two videos keep one span each (each with a single pair here).
        let both = expand_placeholders(
            "q",
            &[video, video, 2, video, video],
            image,
            video,
            &[],
            &[(2, 1), (2, 1)],
        )
        .unwrap();
        assert_eq!(both.spans.len(), 2);
        assert_eq!(both.spans[0].runs, [(0, 2)]);
        assert_eq!(both.spans[1].runs, [(3, 5)]);

        // Placeholder counts that disagree with the request are refused.
        assert!(expand_placeholders("q", &[image, image], image, video, &[1], &[]).is_err());
        assert!(expand_placeholders("q", &[image], image, video, &[1, 1], &[]).is_err());
        assert!(expand_placeholders("q", &[video, video], image, video, &[], &[(2, 2)]).is_err());
        // A group whose length is not one pair's token count.
        assert!(
            expand_placeholders(
                "q",
                &[video, video, video, 2, video, video],
                image,
                video,
                &[],
                &[(2, 2)]
            )
            .is_err()
        );
    }

    /// A text prompt is never refused for a literal placeholder token: the
    /// reference's `_prepare_text` has no count check, so a state that quotes
    /// `<|image_pad|>` is answered like any other text (the token is just a token).
    /// A media prompt does need the counts to match the request.
    #[test]
    fn only_media_prompts_check_their_placeholder_count() {
        let literal = format!("State:\nplease echo {} here\n", format::IMAGE_TOKEN);
        assert_eq!(placeholder_problem(&literal, 0, 0), None);
        // Two literal image tokens and no media: still a text prompt.
        let two = format!("{}{}", literal, format::IMAGE_TOKEN);
        assert_eq!(placeholder_problem(&two, 0, 0), None);
        // A literal video token in a text prompt is plain text too.
        assert_eq!(placeholder_problem(format::VIDEO_TOKEN, 0, 0), None);

        // With an image, an extra literal token breaks the count.
        let prompt = format::chatml_template("State:\necho {} here", 1, 0);
        assert_eq!(placeholder_problem(&prompt, 1, 0), None);
        assert!(
            placeholder_problem(&prompt, 2, 0).is_some(),
            "fewer placeholders than images"
        );
        let with_literal = format!("{}{}", prompt, format::IMAGE_TOKEN);
        assert!(
            placeholder_problem(&with_literal, 1, 0).is_some(),
            "a literal token in the state would add a placeholder"
        );
        // A video placeholder in an image prompt is refused by the reference.
        assert!(placeholder_problem(&format!("{prompt}{}", format::VIDEO_TOKEN), 1, 0).is_some());

        // A video prompt counts both media kinds.
        let video_prompt = format::chatml_template("State:\necho {} here", 1, 2);
        assert_eq!(placeholder_problem(&video_prompt, 1, 2), None);
        assert!(placeholder_problem(&video_prompt, 1, 1).is_some());
        assert!(placeholder_problem(&video_prompt, 0, 2).is_some());
        assert!(
            placeholder_problem(&format!("{video_prompt}{}", format::IMAGE_TOKEN), 1, 2).is_some()
        );
        assert!(
            placeholder_problem(&format!("{video_prompt}{}", format::VIDEO_TOKEN), 1, 2).is_some()
        );
    }

    /// A span that does not match its features is a hard error, not a silent
    /// re-shuffle.
    #[test]
    fn a_mismatched_span_is_refused() {
        let device = Device::default();
        let embeds = Tensor::<3>::zeros([1, 4, 3], &device);
        let sequence = PreparedSequence {
            id: "q".to_string(),
            ids: vec![1, 2, 3, 4],
            mm_types: vec![0; 4],
            spans: vec![MediaSpan {
                kind: MediaKind::Image,
                runs: vec![(1, 3)],
            }],
        };
        let vision = PreparedVision {
            image: PreparedImage {
                pixel_values: Tensor::zeros([0, 0], &device),
                grid: [1, 2, 2],
                tokens: 1,
                source: (4, 4),
                resized: (4, 4),
            },
            features: Tensor::zeros([1, 3], &device),
        };
        let error = splice(embeds, &sequence, &[vision], &[])
            .unwrap_err()
            .to_string();
        assert!(error.contains("runs past"), "{error}");
    }
}
