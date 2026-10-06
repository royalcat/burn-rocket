//! Non-streaming OpenAI-compatible `/v1/chat/completions` for the Gemma 4
//! generation model (text content only for now; media parts are a later phase).

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
use crate::gen_model::GenRoot;

pub struct ChatServeOptions {
    pub addr: String,
    pub model_name: String,
    pub model_dir: std::path::PathBuf,
    pub attn_chunk: usize,
    pub max_new_tokens: usize,
}

struct Settings {
    model_name: String,
    attn_chunk: usize,
    max_new_tokens: usize,
    eos: Vec<u32>,
}

struct Inner {
    model: GenRoot,
    lm_head: Tensor<2>,
    tokenizer: Tokenizer,
    device: Device,
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
    let settings = Arc::new(Settings {
        model_name: opts.model_name.clone(),
        attn_chunk: opts.attn_chunk,
        max_new_tokens: opts.max_new_tokens,
        eos: crate::gen_eos(&opts.model_dir),
    });
    let state = AppState {
        inner: Arc::new(Mutex::new(Inner {
            model,
            lm_head,
            tokenizer,
            device,
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

/// OpenAI content is a string or an array of parts; only text parts are used
/// (media parts arrive in the multimodal phase).
fn content_text(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(parts) => parts
            .iter()
            .filter_map(|p| {
                if p.get("type").and_then(|t| t.as_str()) == Some("text") {
                    p.get("text").and_then(|t| t.as_str()).map(|s| s.to_string())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join(" "),
        _ => String::new(),
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
    let messages: Vec<chat::Message> = req
        .messages
        .iter()
        .map(|m| chat::Message {
            role: m.role.clone(),
            content: content_text(&m.content),
        })
        .collect();
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
    let inner = state.inner.clone();
    let computed = tokio::task::spawn_blocking(move || -> Result<(String, usize, usize), String> {
        let mut guard = lock_or_recover(&inner);
        let rendered = chat::render(&messages, thinking);
        let ids = chat::encode(&guard.tokenizer, &rendered).map_err(|e| format!("{e:#}"))?;
        let n_prompt = ids.len();
        let (out, _stats) =
            chat::generate(&guard.model, &guard.lm_head, &ids, &opts, attn_chunk, &guard.device)
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
