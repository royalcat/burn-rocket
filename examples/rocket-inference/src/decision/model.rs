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
//! Image requests (Phase 2b): the request's images are shared by every question,
//! preprocessed once, encoded by the vision tower once ([`VisionTower`]), and each
//! question's `<|image_pad|>` placeholder is expanded to that image's merged token
//! count. The forward then embeds the text ids, splices the vision features into
//! the expanded rows and rotates the full-attention layers with the multimodal
//! positions of [`mrope`](crate::decision::mrope); a text-only question takes the
//! unchanged `forward_last_normed` path.

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
use super::vision::{PreparedImage, VisionTower, prepare_from_rgb};
use crate::qwen35_intent::model::IntentModel;

/// Why a text-only d3 package refuses images (`d3_runtime`'s
/// `image_unavailable` message shape).
pub const NO_VISION_TOWER: &str =
    "image inputs are not available: this checkpoint has no vision tower";

/// A loaded decision checkpoint, ready to answer System One requests.
#[derive(Debug)]
pub struct DecisionModel {
    pub backbone: IntentModel,
    /// `[255, hidden]` f32 readout rows, one per answer code.
    pub readout: Tensor<2>,
    pub config: DecisionConfig,
    pub tokenizer: Tokenizer,
    /// The vision tower of an image-capable checkpoint (`config.json`
    /// `vision_config` + `visual.*`); `None` for a text-only package.
    pub vision: Option<VisionTower>,
    /// `config.json` `image_token_id`: the id one `<|image_pad|>` expands from.
    pub image_token_id: u32,
    /// `config.json` `video_token_id` (phase 3: a prompt that carries one is
    /// refused).
    pub video_token_id: u32,
}

/// One runnable question: its expanded prompt and where the image features go.
#[derive(Debug, Clone)]
pub struct PreparedSequence {
    /// Question id.
    pub id: String,
    /// Expanded token ids: every `<|image_pad|>` placeholder replaced by one id
    /// per merged image token.
    pub ids: Vec<u32>,
    /// `[ids.len()]`, `1` on the expanded image rows and `0` elsewhere (the
    /// reference's `mm_token_type_ids`).
    pub mm_types: Vec<u8>,
    /// `(start, end)` of each image's expanded rows, image order.
    pub image_spans: Vec<(usize, usize)>,
}

/// One request image: the preprocessed pixels and the tower's merged features
/// (`[tokens, out_hidden]`), shared by every question of the request.
#[derive(Debug)]
pub struct PreparedVision {
    pub image: PreparedImage,
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
}

impl Prepared {
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

    /// Whether the checkpoint carries a vision tower (an image request needs one).
    pub fn has_vision(&self) -> bool {
        self.vision.is_some()
    }

    /// Preprocess and encode the request's images once.
    ///
    /// `images` are already decoded (see [`DecisionRequest::decode_images`]);
    /// this runs the Qwen2-VL processor and the vision tower. Returns the images
    /// with their merged features, in list order.
    pub fn prepare_images(&self, images: &[RgbImage]) -> Result<Vec<PreparedVision>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }
        let Some(vision) = self.vision.as_ref() else {
            bail!("{NO_VISION_TOWER}");
        };
        let device = &self.backbone.device;
        let mut prepared = Vec::with_capacity(images.len());
        for (index, image) in images.iter().enumerate() {
            let image = prepare_from_rgb(image, &vision.config, device)
                .map_err(|e| anyhow!("image rejected by the processor: {e:#}"))?;
            if image.tokens == 0 {
                bail!("images[{index}]: the processor produced no image tokens");
            }
            prepared.push(image);
        }
        let pixels: Vec<PreparedImage> = prepared.to_vec();
        let features = vision.encode_all(&pixels)?;
        let [rows, _] = features.dims();
        let total: usize = prepared.iter().map(|image| image.tokens).sum();
        if rows != total {
            bail!("the vision tower produced {rows} rows for {total} image tokens");
        }
        let mut offset = 0usize;
        let mut out = Vec::with_capacity(prepared.len());
        for image in prepared {
            let tokens = image.tokens;
            let rows = features.clone().slice(s![offset..offset + tokens, ..]);
            out.push(PreparedVision {
                image,
                features: rows,
            });
            offset += tokens;
        }
        Ok(out)
    }

    /// Validate, render, tokenize and expand every question of one request.
    ///
    /// Envelope problems are already refused by [`DecisionRequest::from_value`];
    /// `images` are the decoded request images (see
    /// [`DecisionModel::prepare_images`]). A question that cannot be answered ends
    /// up in [`Prepared::errors`] and the rest still runs.
    pub fn prepare(
        &self,
        req: &DecisionRequest,
        images: &[RgbImage],
        max_tokens: usize,
    ) -> Result<Prepared> {
        let cap = self.prompt_cap(max_tokens);
        let media = self.prepare_images(images)?;
        let n_images = media.len();
        let visual = media.iter().map(|image| image.image.tokens).sum::<usize>();
        let mut prepared = Prepared {
            keys: req.questions.keys().cloned().collect(),
            questions: HashMap::new(),
            errors: Map::new(),
            sequences: Vec::new(),
            images: media,
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
                        error_answer(kind.clone(), "invalid_question", message),
                    );
                    prepared.questions.insert(id.clone(), question);
                    continue;
                }
            };
            // No image: the phase-1 text prompt, byte for byte.
            let prompt = if n_images == 0 {
                format::chatml(&user)
            } else {
                format::chatml_images(&user, n_images)
            };
            // A literal placeholder in the state or an option would make the plan
            // ambiguous; the reference refuses that question.
            if prompt.matches(format::IMAGE_TOKEN).count() != n_images
                || prompt.contains(format::VIDEO_TOKEN)
            {
                prepared.errors.insert(
                    id.clone(),
                    error_answer(
                        kind.clone(),
                        "invalid_question",
                        "the state or question contains a literal image or video placeholder token"
                            .to_string(),
                    ),
                );
                prepared.questions.insert(id.clone(), question);
                continue;
            }
            let encoding = self
                .tokenizer
                .encode(prompt.as_str(), false)
                .map_err(|e| anyhow!("tokenize the '{id}' prompt: {e}"))?;
            let raw = encoding.get_ids();
            let sequence = match self.expand(id, raw, &prepared.images) {
                Ok(sequence) => sequence,
                Err(message) => {
                    prepared.errors.insert(
                        id.clone(),
                        error_answer(kind.clone(), "invalid_question", message),
                    );
                    prepared.questions.insert(id.clone(), question);
                    continue;
                }
            };
            prepared.questions.insert(id.clone(), question);
            let length = sequence.ids.len();
            if length > cap {
                prepared.errors.insert(
                    id.clone(),
                    error_answer(
                        kind,
                        "max_length_exceeded",
                        if n_images == 0 {
                            format!(
                                "the question prompt has {length} tokens, over the maximum context \
                                 length of {cap} tokens; nothing was truncated"
                            )
                        } else {
                            format!(
                                "the question prompt has {length} tokens ({visual} for {n_images} \
                                 image(s)), over the maximum context length of {cap} tokens; \
                                 nothing was truncated"
                            )
                        },
                    ),
                );
                continue;
            }
            prepared.sequences.push(sequence);
        }
        Ok(prepared)
    }

    /// Expand every `<|image_pad|>` placeholder of one tokenized prompt to its
    /// image's merged token count and record the rows `mm_types` marks.
    fn expand(
        &self,
        id: &str,
        raw: &[u32],
        images: &[PreparedVision],
    ) -> Result<PreparedSequence, String> {
        let mut ids: Vec<u32> = Vec::with_capacity(raw.len());
        let mut mm_types: Vec<u8> = Vec::with_capacity(raw.len());
        let mut image_spans = Vec::with_capacity(images.len());
        let mut next = 0usize;
        for &token in raw {
            if token == self.image_token_id {
                let Some(image) = images.get(next) else {
                    return Err(format!(
                        "the '{id}' prompt has more image placeholders than the {} image(s) of \
                         the request",
                        images.len()
                    ));
                };
                let start = ids.len();
                ids.extend(std::iter::repeat_n(token, image.image.tokens));
                mm_types.extend(std::iter::repeat_n(mrope::TYPE_IMAGE, image.image.tokens));
                image_spans.push((start, ids.len()));
                next += 1;
            } else if token == self.video_token_id {
                return Err("video placeholders are not supported yet".to_string());
            } else {
                ids.push(token);
                mm_types.push(mrope::TYPE_TEXT);
            }
        }
        if next != images.len() {
            return Err(format!(
                "the '{id}' prompt uses {next} of the {} image(s) of the request",
                images.len()
            ));
        }
        Ok(PreparedSequence {
            id: id.to_string(),
            ids,
            mm_types,
            image_spans,
        })
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
                self.probabilities(question, sequence, &prepared.images)?,
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
        let prepared = self.prepare(&req, &images, max_tokens)?;
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
    ) -> Result<Vec<f64>> {
        self.backbone.cache.reset();
        let hidden = if sequence.image_spans.is_empty() {
            self.backbone.forward_last_normed(&sequence.ids)
        } else {
            let embeds = self.backbone.embed_ids(&sequence.ids);
            let embeds = splice(embeds, sequence, images)?;
            let grids: Vec<[usize; 3]> = sequence
                .image_spans
                .iter()
                .zip(images)
                .map(|(_, image)| image.image.grid)
                .collect();
            let merge = self
                .vision
                .as_ref()
                .map(|vision| vision.config.spatial_merge_size)
                .unwrap_or(2);
            let tables = mrope::rope_tables(
                &sequence.mm_types,
                &grids,
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

/// Write each image's features into its expanded rows of `[1, seq, hidden]`.
///
/// Row replacement keeps the length, so the spans (computed on the expanded
/// prompt) stay valid while the rows are spliced in image order.
fn splice(
    mut embeds: Tensor<3>,
    sequence: &PreparedSequence,
    images: &[PreparedVision],
) -> Result<Tensor<3>> {
    let [batch, seq, hidden] = embeds.dims();
    for ((start, end), image) in sequence.image_spans.iter().zip(images) {
        let features = image.features.clone();
        let [tokens, width] = features.dims();
        if width != hidden {
            bail!("the vision features are {width} wide, the text embeddings {hidden}");
        }
        if tokens != end - start {
            bail!("the image span [{start}, {end}) does not hold its {tokens} feature rows");
        }
        let before = embeds.clone().slice(s![.., 0..*start, ..]);
        let after = embeds.slice(s![.., *end..seq, ..]);
        embeds = Tensor::cat(
            vec![before, features.reshape([batch, tokens, hidden]), after],
            1,
        );
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
            image_spans: vec![(1, 3)],
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
        let out = splice(embeds, &sequence, &[vision]).unwrap();
        let values: Vec<f32> = out.to_data().try_to_vec().unwrap();
        assert_eq!(
            values,
            vec![
                0.0, 1.0, 2.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 9.0, 10.0, 11.0
            ]
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
            image_spans: vec![(1, 3)],
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
        let error = splice(embeds, &sequence, &[vision])
            .unwrap_err()
            .to_string();
        assert!(error.contains("does not hold"), "{error}");
    }
}
