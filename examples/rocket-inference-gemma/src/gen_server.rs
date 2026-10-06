//! Non-streaming OpenAI-compatible `/v1/chat/completions` for the Gemma 4
//! generation model (text content only for now; media parts are a later phase).

use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use anyhow::{Context, Result};
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use burn::prelude::*;
use burn::tensor::DType;
use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::chat::{self, GenOptions};
use crate::inputs;
use crate::gen_model::GenRoot;
use crate::server::resolve_media;

pub struct ChatServeOptions {
    pub addr: String,
    pub model_name: String,
    pub model_dir: std::path::PathBuf,
    pub attn_chunk: usize,
    pub max_new_tokens: usize,
}

struct Settings {
    model_name: String,
    model_dir: std::path::PathBuf,
    attn_chunk: usize,
    max_new_tokens: usize,
    image_soft_tokens: usize,
    video_soft_tokens: usize,
    eos: Vec<u32>,
}

struct Inner {
    model: GenRoot,
    lm_head: Tensor<2>,
    tokenizer: Tokenizer,
    device: Device,
    pad_id: u32,
}

#[derive(Clone)]
struct AppState {
    inner: Arc<Mutex<Inner>>,
    settings: Arc<Settings>,
}

fn lock_or_recover<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(g) => g,
        Err(PoisonError { .. }) => mutex.lock().unwrap_or_else(|e| e.into_inner()),
    }
}

pub fn serve(
    model: GenRoot,
    lm_head: Tensor<2>,
    tokenizer: Tokenizer,
    device: Device,
    opts: ChatServeOptions,
) -> Result<()> {
    let (image_soft_tokens, video_soft_tokens) = inputs::media_soft_tokens(&opts.model_dir);
    let pad_id = std::fs::read_to_string(opts.model_dir.join("config.json"))
        .ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .and_then(|j| j["text_config"]["pad_token_id"].as_u64())
        .map(|v| v as u32)
        .unwrap_or(0);
    let settings = Arc::new(Settings {
        model_name: opts.model_name.clone(),
        model_dir: opts.model_dir.clone(),
        attn_chunk: opts.attn_chunk,
        max_new_tokens: opts.max_new_tokens,
        image_soft_tokens,
        video_soft_tokens,
        eos: crate::gen_eos(&opts.model_dir),
    });
    let state = AppState {
        inner: Arc::new(Mutex::new(Inner {
            model,
            lm_head,
            tokenizer,
            device,
            pad_id,
        })),
        settings,
    };
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state);
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("tokio runtime")?;
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(&opts.addr)
            .await
            .with_context(|| format!("bind {}", opts.addr))?;
        println!("listening on http://{}", opts.addr);
        axum::serve(listener, app).await?;
        Ok::<(), anyhow::Error>(())
    })
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "ok"}))
}

async fn list_models(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "object": "list",
        "data": [{"id": state.settings.model_name, "object": "model", "owned_by": "local"}],
    }))
}

#[derive(Deserialize)]
struct ChatRequest {
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
    enable_thinking: Option<bool>,
    #[serde(default)]
    stream: Option<bool>,
}

#[derive(Deserialize)]
struct MessageIn {
    role: String,
    content: serde_json::Value,
}

/// OpenAI content parts: text is kept as-is, an `image_url` / `input_audio`
/// part becomes the `<|image|>` / `<|audio|>` placeholder (its payload is
/// resolved to a local file for the towers).
fn content_parts(
    v: &serde_json::Value,
    images: &mut Vec<PathBuf>,
    audios: &mut Vec<PathBuf>,
) -> Result<String, String> {
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
                            .ok_or("image part needs image_url.url")?;
                        images.push(resolve_media(url, "img")?);
                        out.push("<|image|>".to_string());
                    }
                    Some("input_audio") | Some("audio") => {
                        let payload = p
                            .get("input_audio")
                            .and_then(|a| a.get("data"))
                            .and_then(|d| d.as_str())
                            .or_else(|| p.get("audio").and_then(|a| a.as_str()))
                            .ok_or("audio part needs input_audio.data")?;
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
                        audios.push(resolve_media(&uri, "wav")?);
                        out.push("<|audio|>".to_string());
                    }
                    Some(other) => return Err(format!("unsupported content part type '{other}'")),
                    None => return Err("content part without a type".to_string()),
                }
            }
            Ok(out.join(" "))
        }
        _ => Ok(String::new()),
    }
}

fn bad_request(msg: impl std::fmt::Display) -> Response {
    (StatusCode::BAD_REQUEST, msg.to_string()).into_response()
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Response {
    if req.stream == Some(true) {
        return bad_request("streaming is not supported yet; omit `stream`");
    }
    if req.messages.is_empty() {
        return bad_request("empty messages");
    }
    let mut images: Vec<PathBuf> = Vec::new();
    let mut audios: Vec<PathBuf> = Vec::new();
    let mut messages: Vec<chat::Message> = Vec::with_capacity(req.messages.len());
    for m in req.messages.iter() {
        let content = match content_parts(&m.content, &mut images, &mut audios) {
            Ok(c) => c,
            Err(e) => return bad_request(e),
        };
        messages.push(chat::Message {
            role: m.role.clone(),
            content,
        });
    }
    if images.len() > 1 || audios.len() > 1 {
        return bad_request("at most one image and one audio per request for now");
    }
    let thinking = req.enable_thinking.unwrap_or(false);
    let opts = GenOptions {
        max_new_tokens: req.max_tokens.unwrap_or(state.settings.max_new_tokens),
        greedy: req.temperature.map(|t| t == 0.0).unwrap_or(true),
        temperature: req.temperature.unwrap_or(1.0),
        top_k: req.top_k.unwrap_or(64),
        top_p: req.top_p.unwrap_or(0.95),
        eos: state.settings.eos.clone(),
    };
    let attn_chunk = state.settings.attn_chunk;
    let model_dir = state.settings.model_dir.clone();
    let image_soft = state.settings.image_soft_tokens;
    let video_soft = state.settings.video_soft_tokens;
    let inner = state.inner.clone();
    let computed = tokio::task::spawn_blocking(move || -> Result<(String, usize, usize), String> {
        let mut guard = lock_or_recover(&inner);
        let rendered = chat::render(&messages, thinking);
        let (ids, soft) = if images.is_empty() && audios.is_empty() {
            (
                chat::encode(&guard.tokenizer, &rendered).map_err(|e| format!("{e:#}"))?,
                None,
            )
        } else {
            let req = inputs::MediaInputs {
                text: Some(rendered.clone()),
                image: images.first().cloned(),
                video: None,
                audio: audios.first().cloned(),
                prompt_prefix: String::new(),
                max_tokens: None,
                image_soft_tokens: image_soft,
                video_soft_tokens: video_soft,
                video_fps: 1.0,
                video_max_frames: 32,
                attn_chunk,
            };
            let prepared = inputs::prepare(
                &guard.model,
                &guard.tokenizer,
                &req,
                &inputs::DebugPaths::default(),
            )
            .map_err(|e| format!("{e:#}"))?;
            (prepared.ids, prepared.soft)
        };
        let n_prompt = ids.len();
        let pad_id = guard.pad_id;
        let (out, _stats) = chat::generate_with_media(
            &guard.model,
            &guard.lm_head,
            &ids,
            soft,
            pad_id,
            &opts,
            attn_chunk,
            &guard.device,
        )
        .map_err(|e| format!("{e:#}"))?;
        let text = guard
            .tokenizer
            .decode(&out, true)
            .map_err(|e| format!("decode: {e}"))?;
        Ok((text, n_prompt, out.len()))
    })
    .await;
    match computed {
        Ok(Ok((text, n_prompt, n_out))) => Json(serde_json::json!({
            "id": format!("chatcmpl-{}", now_millis()),
            "object": "chat.completion",
            "created": now_secs(),
            "model": state.settings.model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": n_prompt,
                "completion_tokens": n_out,
                "total_tokens": n_prompt + n_out,
            },
        }))
        .into_response(),
        Ok(Err(e)) => bad_request(e),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("task join error: {e}"),
        )
            .into_response(),
    }
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

/// Unused import guard (DType is used by the loader path in this module's
/// callers; keep the import list honest).
#[allow(dead_code)]
fn _dtype_marker(_: DType) {}
