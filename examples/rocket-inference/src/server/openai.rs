//! OpenAI-compatible endpoints: `/v1/embeddings` and `/v1/chat/completions`.

use axum::{Json, extract::State};
use base64::Engine as _;
use serde::{Deserialize, Serialize};

use super::AppState;
use super::engine::{ChatMessage, ChatTurn, EmbedBatchRequest, EmbedInput, EmbedPart};
use super::error::{ApiError, OpenAiError};
use super::log;
use crate::gemma::media::ResolvedMedia;

// ---------------------------------------------------------------------------
// /v1/embeddings
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub(crate) struct EmbeddingRequest {
    input: Input,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    encoding_format: Option<String>,
    /// MRL truncation (OpenAI/OpenRouter name; `dim` is the native alias).
    #[serde(default)]
    dimensions: Option<usize>,
    #[serde(default)]
    dim: Option<usize>,
    #[serde(default)]
    prompt: Option<String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
pub(crate) enum Input {
    One(String),
    Many(Vec<String>),
    Tokens(Vec<i64>),
    TokenBatches(Vec<Vec<i64>>),
    /// OpenRouter multimodal input: one block per embedding.
    Blocks(Vec<ContentBlock>),
    /// Convenience: a single block without the outer array.
    Block(ContentBlock),
}

/// One OpenRouter multimodal input block (`{"content": [...]}`).
#[derive(Deserialize)]
pub(crate) struct ContentBlock {
    content: Vec<serde_json::Value>,
}

/// Map an OpenRouter/OpenAI content block to engine parts, preserving order.
fn block_parts(block: ContentBlock) -> Result<Vec<EmbedPart>, ApiError> {
    if block.content.is_empty() {
        return Err(ApiError::bad_request("content block must not be empty"));
    }
    let mut parts = Vec::with_capacity(block.content.len());
    for p in block.content {
        match p.get("type").and_then(|t| t.as_str()) {
            Some("text") => {
                let text = p
                    .get("text")
                    .and_then(|t| t.as_str())
                    .ok_or_else(|| ApiError::bad_request("text part needs 'text'"))?;
                parts.push(EmbedPart::Text(text.to_string()));
            }
            Some("image_url") => {
                // OpenRouter's schema uses `{"url": ...}`; a bare string is
                // accepted too (some OpenAI-compatible clients send that).
                let url = p
                    .get("image_url")
                    .and_then(|u| u.as_str().or_else(|| u.get("url").and_then(|v| v.as_str())))
                    .ok_or_else(|| ApiError::bad_request("image_url part needs image_url.url"))?;
                parts.push(EmbedPart::Image(url.to_string()));
            }
            Some("input_audio") => {
                let (data, format) = media_field(&p, "input_audio")?;
                parts.push(EmbedPart::Audio(media_value(
                    &data,
                    format.as_deref(),
                    "audio/wav",
                )));
            }
            Some("input_video") => {
                let (data, format) = media_field(&p, "input_video")?;
                parts.push(EmbedPart::Video(media_value(
                    &data,
                    format.as_deref(),
                    "video/mp4",
                )));
            }
            Some("input_file") => {
                return Err(ApiError::bad_request(
                    "input_file content parts are not supported",
                ));
            }
            Some(other) => {
                return Err(ApiError::bad_request(format!(
                    "unsupported content part type '{other}'"
                )));
            }
            None => return Err(ApiError::bad_request("content part without a type")),
        }
    }
    Ok(parts)
}

/// `(data, format)` of an `input_audio` / `input_video` payload.
fn media_field(part: &serde_json::Value, name: &str) -> Result<(String, Option<String>), ApiError> {
    let media = part
        .get(name)
        .ok_or_else(|| ApiError::bad_request(format!("{name} part is missing its payload")))?;
    let data = media
        .get("data")
        .and_then(|d| d.as_str())
        .ok_or_else(|| ApiError::bad_request(format!("{name}.data is required")))?;
    let format = media.get("format").and_then(|f| f.as_str());
    Ok((data.to_string(), format.map(|f| f.to_string())))
}

/// The schema sends data URIs; bare base64 (as in the chat API) is wrapped with
/// the part's `format` or a default MIME type.
fn media_value(data: &str, format: Option<&str>, default_mime: &str) -> String {
    if data.starts_with("data:") {
        return data.to_string();
    }
    let mime = match format {
        Some(f) => {
            let kind = default_mime.split('/').next().unwrap_or("application");
            format!("{kind}/{f}")
        }
        None => default_mime.to_string(),
    };
    format!("data:{mime};base64,{data}")
}

#[derive(Serialize)]
struct EmbeddingData {
    object: &'static str,
    index: usize,
    embedding: Embedding,
}

#[derive(Serialize)]
#[serde(untagged)]
enum Embedding {
    Float(Vec<f32>),
    Base64(String),
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    total_tokens: usize,
}

pub(crate) async fn embeddings(
    State(state): State<AppState>,
    Json(req): Json<EmbeddingRequest>,
) -> Result<Json<serde_json::Value>, OpenAiError> {
    let encoding_format = req.encoding_format.unwrap_or_else(|| "float".to_string());
    if encoding_format != "float" && encoding_format != "base64" {
        return Err(ApiError::bad_request(format!(
            "unsupported encoding_format '{encoding_format}'"
        ))
        .into());
    }
    let model_name = req
        .model
        .clone()
        .unwrap_or_else(|| state.settings.model_name.clone());
    let inputs = match req.input {
        Input::One(text) => vec![EmbedInput::Text(text)],
        Input::Many(texts) => texts.into_iter().map(EmbedInput::Text).collect(),
        Input::Tokens(tokens) => vec![EmbedInput::Tokens(to_u32(tokens))],
        Input::TokenBatches(batches) => batches
            .into_iter()
            .map(|tokens| EmbedInput::Tokens(to_u32(tokens)))
            .collect(),
        Input::Blocks(blocks) => blocks
            .into_iter()
            .map(|block| block_parts(block).map(EmbedInput::Parts))
            .collect::<Result<Vec<_>, _>>()?,
        Input::Block(block) => vec![EmbedInput::Parts(block_parts(block)?)],
    };
    let request = EmbedBatchRequest {
        inputs,
        prompt: req.prompt,
        dim: req.dimensions.or(req.dim),
    };

    let inputs = request.inputs.len();
    let (result, timing) = state
        .compute(move |engine| engine.embed_texts(request))
        .await?;
    log::embedding(&state, "embeddings", inputs, result.tokens, &timing);

    let data: Vec<EmbeddingData> = result
        .vectors
        .into_iter()
        .enumerate()
        .map(|(index, vec)| EmbeddingData {
            object: "embedding",
            index,
            embedding: if encoding_format == "base64" {
                let mut bytes = Vec::with_capacity(vec.len() * 4);
                for v in &vec {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                Embedding::Base64(base64::engine::general_purpose::STANDARD.encode(bytes))
            } else {
                Embedding::Float(vec)
            },
        })
        .collect();
    Ok(Json(serde_json::json!({
        "id": format!("embd-{}", now_millis()),
        "object": "list",
        "data": data,
        "model": model_name,
        "usage": Usage {
            prompt_tokens: result.tokens,
            total_tokens: result.tokens,
        },
    })))
}

fn to_u32(tokens: Vec<i64>) -> Vec<u32> {
    tokens.into_iter().map(|t| t as u32).collect()
}

// ---------------------------------------------------------------------------
// /v1/chat/completions
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub(crate) struct ChatRequest {
    #[serde(default)]
    #[allow(dead_code)]
    model: Option<String>,
    messages: Vec<MessageIn>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    stop: Option<StopField>,
    #[serde(default)]
    enable_thinking: Option<bool>,
    #[serde(default)]
    stream: Option<bool>,
}

#[derive(Deserialize)]
pub(crate) struct MessageIn {
    role: String,
    content: serde_json::Value,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum StopField {
    One(String),
    Many(Vec<String>),
}

impl StopField {
    fn into_vec(self) -> Vec<String> {
        match self {
            Self::One(s) => vec![s],
            Self::Many(v) => v,
        }
    }
}

/// OpenAI content parts: text is kept as-is, an `image_url` / `input_audio`
/// part becomes the `<|image|>` / `<|audio|>` placeholder (its payload is
/// resolved to a local file for the towers).
fn content_parts(
    v: &serde_json::Value,
    images: &mut Vec<ResolvedMedia>,
    audios: &mut Vec<ResolvedMedia>,
) -> Result<String, ApiError> {
    match v {
        serde_json::Value::String(s) => Ok(s.clone()),
        serde_json::Value::Array(parts) => {
            let mut out: Vec<String> = Vec::new();
            for p in parts {
                match p.get("type").and_then(|t| t.as_str()) {
                    Some("text") => {
                        if let Some(s) = p.get("text").and_then(|t| t.as_str()) {
                            out.push(s.to_string());
                        }
                    }
                    Some("image_url") | Some("image") => {
                        let url = p
                            .get("image_url")
                            .and_then(|u| u.get("url"))
                            .and_then(|u| u.as_str())
                            .or_else(|| p.get("image").and_then(|u| u.as_str()))
                            .ok_or_else(|| {
                                ApiError::bad_request("image part needs image_url.url")
                            })?;
                        images.push(
                            crate::gemma::media::resolve_media(url, "img")
                                .map_err(ApiError::bad_request)?,
                        );
                        out.push("<|image|>".to_string());
                    }
                    Some("input_audio") | Some("audio") => {
                        let payload = p
                            .get("input_audio")
                            .and_then(|a| a.get("data"))
                            .and_then(|d| d.as_str())
                            .or_else(|| p.get("audio").and_then(|a| a.as_str()))
                            .ok_or_else(|| {
                                ApiError::bad_request("audio part needs input_audio.data")
                            })?;
                        let format = p
                            .get("input_audio")
                            .and_then(|a| a.get("format"))
                            .and_then(|f| f.as_str())
                            .unwrap_or("wav");
                        // `input_audio.data` is bare base64 (no data: prefix).
                        let uri = if payload.starts_with("data:") {
                            payload.to_string()
                        } else {
                            format!("data:audio/{format};base64,{payload}")
                        };
                        audios.push(
                            crate::gemma::media::resolve_media(&uri, "wav")
                                .map_err(ApiError::bad_request)?,
                        );
                        out.push("<|audio|>".to_string());
                    }
                    Some(other) => {
                        return Err(ApiError::bad_request(format!(
                            "unsupported content part type '{other}'"
                        )));
                    }
                    None => {
                        return Err(ApiError::bad_request("content part without a type"));
                    }
                }
            }
            Ok(out.join(" "))
        }
        _ => Ok(String::new()),
    }
}

pub(crate) async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Result<Json<serde_json::Value>, OpenAiError> {
    if req.stream == Some(true) {
        return Err(ApiError::bad_request(
            "streaming is not supported; omit `stream` or use the Ollama API",
        )
        .into());
    }
    if req.messages.is_empty() {
        return Err(ApiError::bad_request("empty messages").into());
    }
    let mut images: Vec<ResolvedMedia> = Vec::new();
    let mut audios: Vec<ResolvedMedia> = Vec::new();
    let mut messages: Vec<ChatMessage> = Vec::with_capacity(req.messages.len());
    for m in req.messages.iter() {
        let content = content_parts(&m.content, &mut images, &mut audios)?;
        messages.push(ChatMessage {
            role: m.role.clone(),
            content,
        });
    }
    if images.len() > 1 || audios.len() > 1 {
        return Err(
            ApiError::bad_request("at most one image and one audio per request for now").into(),
        );
    }
    let model_name = state.settings.model_name.clone();
    // `images`/`audios` hold the temp-file guards for the duration of the
    // request; only their paths are moved into the turn.
    let turn = ChatTurn {
        messages,
        prompt: None,
        system: None,
        raw: false,
        image: images.first().map(|m| m.path().to_owned()),
        audio: audios.first().map(|m| m.path().to_owned()),
        max_new_tokens: req.max_tokens.unwrap_or(state.settings.max_new_tokens),
        temperature: req.temperature.unwrap_or(state.settings.temperature),
        top_p: req.top_p,
        top_k: req.top_k,
        enable_thinking: req.enable_thinking.unwrap_or(false),
        format_json: false,
        stop_strings: req.stop.map(StopField::into_vec).unwrap_or_default(),
        seed: 0,
        num_ctx: None,
    };

    let (completion, timing) = state.compute(move |engine| engine.chat(turn)).await?;
    log::chat(&state, "chat.completions", &completion, &timing);
    let created = now_secs();
    Ok(Json(serde_json::json!({
        "id": format!("chatcmpl-{}", now_millis()),
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": completion.text},
            "finish_reason": completion.done_reason,
        }],
        "usage": {
            "prompt_tokens": completion.prompt_tokens,
            "completion_tokens": completion.eval_count,
            "total_tokens": completion.prompt_tokens + completion.eval_count,
        },
    })))
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn now_millis() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(json: &str) -> EmbeddingRequest {
        serde_json::from_str(json).expect("parse embedding request")
    }

    fn parts_of(input: Input) -> Result<Vec<EmbedPart>, ApiError> {
        match input {
            Input::Blocks(mut blocks) => {
                assert_eq!(blocks.len(), 1, "expected one block");
                block_parts(blocks.remove(0))
            }
            Input::Block(block) => block_parts(block),
            _ => panic!("expected a content block"),
        }
    }

    #[test]
    fn parses_the_openrouter_image_example() {
        let req = request(
            r#"{
                "model": "nvidia/llama-nemotron-embed-vl-1b-v2",
                "input": [{
                    "content": [
                        {"type": "text", "text": "A scenic boardwalk"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/a.jpg"}}
                    ]
                }],
                "encoding_format": "float",
                "dimensions": 512
            }"#,
        );
        assert_eq!(req.dimensions, Some(512));
        let parts = parts_of(req.input).unwrap();
        assert!(matches!(&parts[0], EmbedPart::Text(t) if t == "A scenic boardwalk"));
        assert!(matches!(&parts[1], EmbedPart::Image(u) if u == "https://example.com/a.jpg"));
    }

    #[test]
    fn accepts_a_single_block_and_a_bare_image_string() {
        let req = request(
            r#"{"input": {"content": [{"type": "image_url", "image_url": "data:image/png;base64,AA=="}]}}"#,
        );
        let parts = parts_of(req.input).unwrap();
        assert!(matches!(&parts[0], EmbedPart::Image(u) if u.starts_with("data:image/png")));
    }

    #[test]
    fn wraps_bare_base64_audio_with_its_format() {
        let req = request(
            r#"{"input": [{"content": [{"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}]}]}"#,
        );
        let parts = parts_of(req.input).unwrap();
        assert!(matches!(&parts[0], EmbedPart::Audio(v) if v == "data:audio/wav;base64,AAAA"));
    }

    #[test]
    fn rejects_input_file_and_unknown_parts() {
        let file = request(
            r#"{"input": [{"content": [{"type": "input_file", "input_file": {"data": "x"}}]}]}"#,
        );
        let err = parts_of(file.input).unwrap_err();
        assert!(err.message.contains("input_file"), "{}", err.message);

        let unknown = request(r#"{"input": [{"content": [{"type": "video", "x": 1}]}]}"#);
        let err = parts_of(unknown.input).unwrap_err();
        assert!(err.message.contains("unsupported"), "{}", err.message);
    }

    #[test]
    fn legacy_input_shapes_still_parse() {
        assert!(matches!(request(r#"{"input": "hi"}"#).input, Input::One(_)));
        assert!(matches!(
            request(r#"{"input": ["a", "b"]}"#).input,
            Input::Many(_)
        ));
        assert!(matches!(
            request(r#"{"input": [1, 2]}"#).input,
            Input::Tokens(_)
        ));
        assert!(matches!(
            request(r#"{"input": [[1, 2], [3]]}"#).input,
            Input::TokenBatches(_)
        ));
    }
}
