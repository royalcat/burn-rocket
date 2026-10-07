//! The loaded model: family detection, capabilities, and the compute entry
//! points the HTTP handlers call (one resident model per process).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use burn::prelude::*;
use burn::tensor::{DType, Int, TensorData};
use tokenizers::Tokenizer;

use crate::gemma::embeddinggemma::model::Emb2Model;
use crate::gemma::gemma4::chat::{self, GenOptions};
use crate::gemma::gemma4::model::GenRoot;
use crate::gemma::inputs::{self, DebugPaths, MediaInputs};
use crate::qwen35_intent::model::IntentModel;
use crate::qwen3_embedding::model::{Qwen3Config, Qwen3Embedding};
use crate::util::rope::RopeCache;

use super::error::ApiError;

/// The model family loaded in this process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    Qwen3Embed,
    Emb2,
    Intent,
    Gemma4,
}

impl Family {
    /// Name for the `--family` flag and the `/v1/models` capability listing.
    pub fn name(self) -> &'static str {
        match self {
            Self::Qwen3Embed => "qwen3",
            Self::Emb2 => "embeddinggemma",
            Self::Intent => "intent",
            Self::Gemma4 => "gemma4",
        }
    }

    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "qwen3" => Ok(Self::Qwen3Embed),
            "embeddinggemma" => Ok(Self::Emb2),
            "intent" => Ok(Self::Intent),
            "gemma4" => Ok(Self::Gemma4),
            other => bail!(
                "unknown family '{other}' (expected auto|qwen3|embeddinggemma|intent|gemma4)"
            ),
        }
    }

    /// Ollama's `general.architecture` for this family (chat families only).
    pub fn ollama_architecture(self) -> &'static str {
        match self {
            Self::Qwen3Embed => "qwen3",
            Self::Emb2 => "embedding_gemma2",
            Self::Intent => "qwen3_5",
            Self::Gemma4 => "gemma4",
        }
    }
}

/// Detect the model family from a checkpoint's `config.json` (`model_type`).
pub fn detect_family(model_dir: &Path) -> Result<Family> {
    let path = model_dir.join("config.json");
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("read {}", path.display()))?;
    let json: serde_json::Value = serde_json::from_str(&text)
        .with_context(|| format!("parse {}", path.display()))?;
    detect_from_value(&json)
        .with_context(|| format!("detect the model family in {}", model_dir.display()))
}

fn detect_from_value(json: &serde_json::Value) -> Result<Family> {
    match json.get("model_type").and_then(|v| v.as_str()) {
        Some("qwen3") => Ok(Family::Qwen3Embed),
        Some("embedding_gemma2") => Ok(Family::Emb2),
        Some("qwen3_5") => Ok(Family::Intent),
        Some("gemma4") => Ok(Family::Gemma4),
        Some(other) => bail!(
            "unsupported model_type '{other}' \
             (pass --family qwen3|embeddinggemma|intent|gemma4)"
        ),
        None => bail!("config.json has no model_type (pass --family qwen3|embeddinggemma|intent|gemma4)"),
    }
}

/// What the loaded model can serve; the router registers only these routes.
#[derive(Debug, Clone, Copy, Default)]
pub struct Capabilities {
    pub embeddings: bool,
    pub multimodal: bool,
    pub chat: bool,
}

impl Capabilities {
    pub fn names(self) -> Vec<&'static str> {
        let mut names = Vec::new();
        if self.embeddings {
            names.push("embeddings");
        }
        if self.multimodal {
            names.push("multimodal");
        }
        if self.chat {
            names.push("chat");
        }
        names
    }
}

/// A chat message with plain-text content (media parts become placeholders and
/// are carried separately).
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// One chat/generate request, already resolved against the server settings.
#[derive(Debug, Clone)]
pub struct ChatTurn {
    pub messages: Vec<ChatMessage>,
    /// `/api/generate` prompt (mutually exclusive with `messages` in practice).
    pub prompt: Option<String>,
    /// `/api/generate` system prompt.
    pub system: Option<String>,
    /// `/api/generate` `raw: true`: skip the chat template.
    pub raw: bool,
    pub image: Option<PathBuf>,
    pub audio: Option<PathBuf>,
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub enable_thinking: bool,
    pub format_json: bool,
    pub stop_strings: Vec<String>,
    pub seed: u64,
    /// Prompt-token budget (Ollama `num_ctx`), already capped by the server.
    pub num_ctx: Option<usize>,
}

/// The result of one generation, in the terms both APIs report.
pub struct Completion {
    pub text: String,
    /// Generated token ids (Ollama's `context`).
    pub ids: Vec<u32>,
    pub prompt_tokens: usize,
    pub eval_count: usize,
    pub prompt_eval_ms: f64,
    pub eval_ms: f64,
    /// `"stop"` (EOS / stop string) or `"length"`.
    pub done_reason: &'static str,
}

/// One input to `/v1/embeddings`.
pub enum EmbedInput {
    Text(String),
    Tokens(Vec<u32>),
}

pub struct EmbedTextsRequest {
    pub inputs: Vec<EmbedInput>,
    /// Task-prompt name (`config_sentence_transformers.json`); supported by Emb2.
    pub prompt: Option<String>,
    /// MRL truncation; supported by Emb2.
    pub dim: Option<usize>,
}

pub struct EmbedMediaRequest {
    pub text: Option<String>,
    pub image: Option<PathBuf>,
    pub video: Option<PathBuf>,
    pub audio: Option<PathBuf>,
    pub prompt: Option<String>,
    pub dim: Option<usize>,
}

pub struct EmbedResult {
    pub vectors: Vec<Vec<f32>>,
    /// Prompt tokens across all inputs (OpenAI `usage`).
    pub tokens: usize,
}

/// The loaded model and its runtime state.
pub enum Engine {
    Qwen3(Qwen3Engine),
    Emb2(Emb2Engine),
    Intent(IntentEngine),
    Gemma4(Gemma4Engine),
}

pub struct Qwen3Engine {
    pub model: Qwen3Embedding,
    pub cfg: Qwen3Config,
    pub tokenizer: Tokenizer,
    pub device: Device,
    pub chunk: usize,
    pub key_block: usize,
    pub attn_fused: bool,
    pub max_tokens: usize,
}

pub struct Emb2Engine {
    pub model: Emb2Model,
    pub tokenizer: Tokenizer,
    pub device: Device,
    pub dtype: DType,
    pub max_tokens: usize,
    pub attn_chunk: usize,
    pub image_soft_tokens: usize,
    pub video_soft_tokens: usize,
    pub video_fps: f64,
    pub video_max_frames: usize,
    pub prompts: std::collections::HashMap<String, String>,
}

pub struct IntentEngine {
    pub model: IntentModel,
    pub tokenizer: Tokenizer,
    pub stop_ids: Vec<u32>,
    /// Prompt-token budget (context cap). `None` = server `--max-tokens`.
    pub max_tokens: usize,
}

pub struct Gemma4Engine {
    pub model: GenRoot,
    pub lm_head: Tensor<2>,
    pub tokenizer: Tokenizer,
    pub device: Device,
    pub pad_id: u32,
    pub eos: Vec<u32>,
    pub attn_chunk: usize,
    pub image_soft_tokens: usize,
    pub video_soft_tokens: usize,
}

impl Engine {
    pub fn embed_texts(&mut self, req: EmbedTextsRequest) -> Result<EmbedResult, ApiError> {
        match self {
            Self::Qwen3(engine) => engine.embed_texts(req),
            Self::Emb2(engine) => engine.embed_texts(req),
            Self::Intent(_) | Self::Gemma4(_) => {
                Err(ApiError::bad_request("this model does not produce embeddings"))
            }
        }
    }

    pub fn embed_media(&mut self, req: EmbedMediaRequest) -> Result<EmbedResult, ApiError> {
        match self {
            Self::Emb2(engine) => engine.embed_media(req),
            _ => Err(ApiError::bad_request(
                "this model does not accept image/video/audio input",
            )),
        }
    }

    pub fn chat(&mut self, turn: ChatTurn) -> Result<Completion, ApiError> {
        match self {
            Self::Intent(engine) => engine.chat(turn),
            Self::Gemma4(engine) => engine.chat(turn),
            Self::Qwen3(_) | Self::Emb2(_) => {
                Err(ApiError::bad_request("this model does not generate text"))
            }
        }
    }
}

impl Qwen3Engine {
    fn embed_texts(&mut self, req: EmbedTextsRequest) -> Result<EmbedResult, ApiError> {
        if req.prompt.is_some() || req.dim.is_some() {
            return Err(ApiError::bad_request(
                "dim/prompt are not supported by this model (they apply to EmbeddingGemma 2)",
            ));
        }
        if req.inputs.is_empty() {
            return Err(ApiError::bad_request("input must not be empty"));
        }
        let mut vectors = Vec::with_capacity(req.inputs.len());
        let mut tokens = 0;
        for input in req.inputs {
            let ids: Vec<i64> = match input {
                EmbedInput::Text(text) => {
                    let enc = self
                        .tokenizer
                        .encode(text.as_str(), true)
                        .map_err(|e| ApiError::bad_request(format!("tokenize: {e}")))?;
                    let mut ids = enc.get_ids().to_vec();
                    if ids.len() > self.max_tokens {
                        ids.truncate(self.max_tokens);
                    }
                    if ids.is_empty() {
                        return Err(ApiError::bad_request("input must not be empty"));
                    }
                    ids.iter().map(|&t| t as i64).collect()
                }
                EmbedInput::Tokens(ids) => {
                    if ids.is_empty() {
                        return Err(ApiError::bad_request("input must not be empty"));
                    }
                    if ids.len() > self.max_tokens {
                        return Err(ApiError::bad_request(format!(
                            "raw token input has {} tokens, over the server limit of {}",
                            ids.len(),
                            self.max_tokens
                        )));
                    }
                    ids.iter().map(|&t| t as i64).collect()
                }
            };
            tokens += ids.len();
            vectors.push(self.embed_tokens(ids));
        }
        Ok(EmbedResult { vectors, tokens })
    }

    /// Runs the embedding model on one token sequence (Qwen3-Embedding pools the
    /// last token itself and returns the embedding vector directly).
    fn embed_tokens(&self, ids: Vec<i64>) -> Vec<f32> {
        let n = ids.len();
        let input =
            Tensor::<2, Int>::from_data(TensorData::new(ids, [1, n]), &self.device);
        let rope = RopeCache::new(n, self.cfg.head_dim, self.cfg.rope_theta, DType::F32, &self.device);
        let out = self
            .model
            .forward(input, &rope, self.chunk, self.key_block, self.attn_fused);
        out.cast(DType::F32)
            .to_data()
            .try_to_vec()
            .unwrap_or_default()
    }
}

impl Emb2Engine {
    fn prompt_prefix(&self, prompt: Option<&str>) -> Result<String, ApiError> {
        match prompt {
            Some(name) => self
                .prompts
                .get(&name.to_lowercase())
                .cloned()
                .ok_or_else(|| ApiError::bad_request(format!("unknown prompt '{name}'"))),
            None => Ok(String::new()),
        }
    }

    fn embed_texts(&mut self, req: EmbedTextsRequest) -> Result<EmbedResult, ApiError> {
        let prefix = self.prompt_prefix(req.prompt.as_deref())?;
        let mut inputs = Vec::with_capacity(req.inputs.len());
        for input in req.inputs {
            match input {
                EmbedInput::Text(text) => inputs.push(text),
                EmbedInput::Tokens(_) => {
                    return Err(ApiError::bad_request(
                        "raw token ids are not supported by this model",
                    ));
                }
            }
        }
        if inputs.is_empty() {
            return Err(ApiError::bad_request("empty input"));
        }
        let mut vectors = Vec::with_capacity(inputs.len());
        let mut tokens = 0;
        for text in inputs {
            let (vector, n) =
                self.embed_one(Some(text), None, None, None, &prefix, req.dim)?;
            tokens += n;
            vectors.push(vector);
        }
        Ok(EmbedResult { vectors, tokens })
    }

    fn embed_media(&mut self, req: EmbedMediaRequest) -> Result<EmbedResult, ApiError> {
        let prefix = self.prompt_prefix(req.prompt.as_deref())?;
        let (vector, tokens) = self.embed_one(
            req.text,
            req.image,
            req.video,
            req.audio,
            &prefix,
            req.dim,
        )?;
        Ok(EmbedResult {
            vectors: vec![vector],
            tokens,
        })
    }

    /// One embedding forward (text and/or one media input), L2-normalized.
    #[allow(clippy::too_many_arguments)]
    fn embed_one(
        &mut self,
        text: Option<String>,
        image: Option<PathBuf>,
        video: Option<PathBuf>,
        audio: Option<PathBuf>,
        prompt_prefix: &str,
        dim: Option<usize>,
    ) -> Result<(Vec<f32>, usize), ApiError> {
        let has_media = image.is_some() || video.is_some() || audio.is_some();
        let req = MediaInputs {
            text,
            image,
            video,
            audio,
            prompt_prefix: prompt_prefix.to_string(),
            max_tokens: if has_media { None } else { Some(self.max_tokens) },
            image_soft_tokens: self.image_soft_tokens,
            video_soft_tokens: self.video_soft_tokens,
            video_fps: self.video_fps,
            video_max_frames: self.video_max_frames,
            attn_chunk: self.attn_chunk,
        };
        let prepared = inputs::prepare(&self.model, &self.tokenizer, &req, &DebugPaths::default())
            .map_err(|e| ApiError::bad_request(format!("{e:#}")))?;
        let n = prepared.ids.len();
        let input = inputs::make_input(&prepared.ids, &self.device);
        let rope = self.model.text().rope_tables(n, self.dtype, &self.device);
        let out = self
            .model
            .embed_ids(input, &rope, self.attn_chunk, prepared.soft);
        let mut v: Vec<f32> = out
            .cast(DType::F32)
            .into_data()
            .try_to_vec()
            .map_err(|e| ApiError::internal(format!("read embedding: {e}")))?;
        if let Some(d) = dim {
            v.truncate(d);
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for x in &mut v {
            *x /= norm;
        }
        Ok((v, n))
    }
}

impl IntentEngine {
    /// ChatML rendering, mirroring the model's Ollama template.
    fn render(&self, turn: &ChatTurn) -> Result<String, ApiError> {
        if turn.raw {
            return Ok(turn.prompt.clone().unwrap_or_default());
        }
        let mut out = String::new();
        if let Some(sys) = &turn.system {
            if !sys.is_empty() {
                out.push_str("<|im_start|>system\n");
                out.push_str(sys);
                out.push_str("<|im_end|>\n");
            }
        }
        for m in &turn.messages {
            if !matches!(m.role.as_str(), "system" | "user" | "assistant") {
                return Err(ApiError::bad_request(format!(
                    "unsupported role '{}'",
                    m.role
                )));
            }
            out.push_str("<|im_start|>");
            out.push_str(&m.role);
            out.push('\n');
            out.push_str(&m.content);
            out.push_str("<|im_end|>\n");
        }
        if let Some(prompt) = &turn.prompt {
            out.push_str("<|im_start|>user\n");
            out.push_str(prompt);
            out.push_str("<|im_end|>\n");
        }
        out.push_str("<|im_start|>assistant\n");
        Ok(out)
    }

    fn chat(&mut self, turn: ChatTurn) -> Result<Completion, ApiError> {
        if turn.image.is_some() || turn.audio.is_some() {
            return Err(ApiError::bad_request(
                "this model does not accept image/audio input",
            ));
        }
        let prompt = self.render(&turn)?;
        let enc = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| ApiError::bad_request(format!("tokenize: {e}")))?;
        let mut ids: Vec<u32> = enc.get_ids().to_vec();
        let num_ctx = turn.num_ctx.unwrap_or(self.max_tokens).min(self.max_tokens);
        if ids.len() > num_ctx {
            // Keep the tail of the conversation (Ollama truncates the context too).
            ids.drain(0..ids.len() - num_ctx);
        }
        if ids.is_empty() {
            return Err(ApiError::bad_request("empty prompt"));
        }

        let (new_ids, stats) = self
            .model
            .generate(&ids, turn.max_new_tokens, &self.stop_ids, turn.temperature, turn.seed);
        let mut text = self
            .tokenizer
            .decode(&new_ids, false)
            .map_err(|e| ApiError::internal(format!("decode: {e}")))?;
        let mut done_reason = if stats.stopped { "stop" } else { "length" };
        if let Some(cut) = stop_string_cut(&text, &turn.stop_strings) {
            text.truncate(cut);
            done_reason = "stop";
        }
        if turn.format_json {
            // The model is SFT'd to emit a JSON object; trim any stray wrapper.
            if let Some(c) = extract_json(&text) {
                text = c;
            }
        }
        Ok(Completion {
            text,
            ids: new_ids,
            prompt_tokens: ids.len(),
            eval_count: stats.steps,
            prompt_eval_ms: stats.prefill_s * 1000.0,
            eval_ms: stats.decode_s * 1000.0,
            done_reason,
        })
    }
}

impl Gemma4Engine {
    fn chat(&mut self, turn: ChatTurn) -> Result<Completion, ApiError> {
        if turn.format_json {
            return Err(ApiError::bad_request(
                "format 'json' is not supported by this model",
            ));
        }
        let (ids, soft) = if turn.raw {
            let prompt = turn.prompt.clone().unwrap_or_default();
            let ids = chat::encode(&self.tokenizer, &prompt)
                .map_err(|e| ApiError::bad_request(format!("{e:#}")))?;
            (ids, None)
        } else if turn.messages.is_empty() {
            let prompt = turn.prompt.clone().unwrap_or_default();
            let messages = [chat::Message::user(prompt)];
            self.encode_messages(&messages, turn.enable_thinking, turn.image, turn.audio)?
        } else {
            let messages: Vec<chat::Message> = turn
                .messages
                .iter()
                .map(|m| chat::Message {
                    role: m.role.clone(),
                    content: m.content.clone(),
                })
                .collect();
            self.encode_messages(&messages, turn.enable_thinking, turn.image, turn.audio)?
        };

        let opts = GenOptions {
            max_new_tokens: turn.max_new_tokens,
            greedy: turn.temperature == 0.0,
            temperature: if turn.temperature == 0.0 {
                1.0
            } else {
                turn.temperature
            },
            top_k: turn.top_k.unwrap_or(64),
            top_p: turn.top_p.unwrap_or(0.95),
            eos: self.eos.clone(),
        };
        let (out, stats) = chat::generate_with_media(
            &self.model,
            &self.lm_head,
            &ids,
            soft,
            self.pad_id,
            &opts,
            self.attn_chunk,
            &self.device,
        )
        .map_err(|e| ApiError::internal(format!("{e:#}")))?;
        let text = self
            .tokenizer
            .decode(&out, true)
            .map_err(|e| ApiError::internal(format!("decode: {e}")))?;
        Ok(Completion {
            text,
            ids: out.clone(),
            prompt_tokens: ids.len(),
            eval_count: out.len(),
            prompt_eval_ms: stats.prefill_s * 1000.0,
            eval_ms: stats.decode_s * 1000.0,
            done_reason: if stats.stopped { "stop" } else { "length" },
        })
    }

    /// Render chat messages and assemble the (ids, soft tokens) input; media
    /// parts (already resolved to paths) expand their placeholders.
    fn encode_messages(
        &mut self,
        messages: &[chat::Message],
        thinking: bool,
        image: Option<PathBuf>,
        audio: Option<PathBuf>,
    ) -> Result<(Vec<u32>, Option<(Vec<usize>, Tensor<2>)>), ApiError> {
        let rendered = chat::render(messages, thinking);
        if image.is_none() && audio.is_none() {
            let ids = chat::encode(&self.tokenizer, &rendered)
                .map_err(|e| ApiError::bad_request(format!("{e:#}")))?;
            return Ok((ids, None));
        }
        let req = MediaInputs {
            text: Some(rendered),
            image,
            video: None,
            audio,
            prompt_prefix: String::new(),
            max_tokens: None,
            image_soft_tokens: self.image_soft_tokens,
            video_soft_tokens: self.video_soft_tokens,
            video_fps: 1.0,
            video_max_frames: 32,
            attn_chunk: self.attn_chunk,
        };
        let prepared = inputs::prepare(&self.model, &self.tokenizer, &req, &DebugPaths::default())
            .map_err(|e| ApiError::bad_request(format!("{e:#}")))?;
        Ok((prepared.ids, prepared.soft))
    }
}

/// Earliest cutoff at any non-empty stop string.
fn stop_string_cut(text: &str, stops: &[String]) -> Option<usize> {
    let mut cut = None;
    for s in stops {
        if s.is_empty() {
            continue;
        }
        if let Some(pos) = text.find(s.as_str()) {
            cut = Some(cut.map_or(pos, |c: usize| c.min(pos)));
        }
    }
    cut
}

/// First balanced `{...}` object in `text`, if any.
fn extract_json(text: &str) -> Option<String> {
    let start = text.find('{')?;
    let bytes = text.as_bytes();
    let mut depth = 0i32;
    let mut in_str = false;
    let mut esc = false;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_str {
            if esc {
                esc = false;
            } else if b == b'\\' {
                esc = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(text[start..=i].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(json: &str) -> serde_json::Value {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn detect_known_families() {
        assert_eq!(
            detect_from_value(&config(r#"{"model_type": "qwen3"}"#)).unwrap(),
            Family::Qwen3Embed
        );
        assert_eq!(
            detect_from_value(&config(r#"{"model_type": "embedding_gemma2"}"#)).unwrap(),
            Family::Emb2
        );
        assert_eq!(
            detect_from_value(&config(r#"{"model_type": "qwen3_5"}"#)).unwrap(),
            Family::Intent
        );
        assert_eq!(
            detect_from_value(&config(r#"{"model_type": "gemma4"}"#)).unwrap(),
            Family::Gemma4
        );
    }

    #[test]
    fn detect_rejects_unknown_or_missing() {
        assert!(detect_from_value(&config(r#"{"model_type": "llama"}"#)).is_err());
        assert!(detect_from_value(&config(r#"{}"#)).is_err());
    }

    #[test]
    fn stop_string_cut_takes_the_earliest() {
        let stops = vec!["END".to_string(), "STOP".to_string()];
        assert_eq!(stop_string_cut("abc STOP def END", &stops), Some(4));
        assert_eq!(stop_string_cut("plain", &stops), None);
        assert_eq!(stop_string_cut("x", &["".to_string()]), None);
    }

    #[test]
    fn extract_json_returns_the_first_object() {
        assert_eq!(
            extract_json("noise {\"a\": {\"b\": 1}} tail"),
            Some("{\"a\": {\"b\": 1}}".to_string())
        );
        assert_eq!(extract_json("no object"), None);
    }
}
