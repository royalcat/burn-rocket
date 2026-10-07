//! OpenAI-compatible endpoints: `/v1/embeddings` and `/v1/chat/completions`.

use std::path::PathBuf;

use axum::{Json, extract::State};
use base64::Engine as _;
use serde::{Deserialize, Serialize};

use super::AppState;
use super::engine::{ChatMessage, ChatTurn, EmbedInput, EmbedTextsRequest};
use super::error::{ApiError, OpenAiError, blocking};

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
    };
    let request = EmbedTextsRequest {
        inputs,
        prompt: req.prompt,
        dim: req.dim,
    };

    let state = state.clone();
    let result = blocking(move || state.lock().embed_texts(request)).await?;

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
    images: &mut Vec<PathBuf>,
    audios: &mut Vec<PathBuf>,
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
    let mut images: Vec<PathBuf> = Vec::new();
    let mut audios: Vec<PathBuf> = Vec::new();
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
    let turn = ChatTurn {
        messages,
        prompt: None,
        system: None,
        raw: false,
        image: images.pop(),
        audio: audios.pop(),
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

    let state = state.clone();
    let completion = blocking(move || state.lock().chat(turn)).await?;
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
