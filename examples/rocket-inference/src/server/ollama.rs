//! Ollama-compatible API for the chat models: `/`, `/api/version`,
//! `/api/tags`, `/api/show`, `/api/chat`, `/api/generate`.
//!
//! Requests are verbatim from the original intent-model server: any
//! `Content-Type` is accepted (litellm posts JSON as `application/octet-stream`)
//! and `stream: true` returns one NDJSON content line plus the final `done`
//! line. The chat template is the model's own (both chat families render it
//! inside the engine).

use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    Json, Router,
    body::Body,
    extract::State,
    http::{HeaderValue, StatusCode, header},
    middleware::map_request,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::Deserialize;

use super::AppState;
use super::engine::{ChatMessage, ChatTurn, Family};
use super::error::{ApiError, OllamaError};
use super::log;

/// ChatML template (intent model); Gemma 4 renders its own template, so
/// `/api/show` reports it without one.
const CHATML_TEMPLATE: &str = "{{ if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}<|im_start|>user\n{{ .Prompt }}<|im_end|>\n<|im_start|>assistant\n";

pub(crate) fn router() -> Router<AppState> {
    Router::new()
        .route("/", get(|| async { "Ollama is running" }))
        .route("/api/version", get(version))
        .route("/api/tags", get(tags))
        .route("/api/show", post(show))
        .route("/api/chat", post(chat))
        .route("/api/generate", post(generate))
        .layer(map_request(force_json_content_type))
}

/// Ollama's Go server never inspects the request `Content-Type`, and litellm's
/// Ollama client sends its JSON bodies as `application/octet-stream`
/// (litellm 1.83: `client.post(url, data=json.dumps(...))`), which axum's `Json`
/// extractor rejects with 415. Coerce the header on the way in so JSON bodies
/// are accepted whatever the client declares — same tolerance as Ollama.
async fn force_json_content_type(mut req: axum::extract::Request) -> axum::extract::Request {
    let is_json = req
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|v| v.to_ascii_lowercase().contains("application/json"))
        .unwrap_or(false);
    if !is_json {
        req.headers_mut().insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );
    }
    req
}

#[derive(Deserialize, Default, Clone)]
struct GenOptions {
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    num_predict: Option<i64>,
    #[serde(default)]
    num_ctx: Option<usize>,
    #[serde(default)]
    stop: Option<Vec<String>>,
    #[serde(default)]
    seed: Option<i64>,
}

#[derive(Deserialize)]
struct OllamaChatMessage {
    role: String,
    #[serde(default)]
    content: String,
}

#[derive(Deserialize)]
struct ChatRequest {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<OllamaChatMessage>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    format: Option<serde_json::Value>,
    #[serde(default)]
    options: GenOptions,
    /// Accepted (and ignored) by the intent model: it was not trained with thinking.
    #[serde(default)]
    #[allow(dead_code)]
    think: Option<bool>,
}

#[derive(Deserialize)]
struct GenerateRequest {
    #[serde(default)]
    model: Option<String>,
    prompt: String,
    #[serde(default)]
    system: Option<String>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    format: Option<serde_json::Value>,
    #[serde(default)]
    raw: Option<bool>,
    #[serde(default)]
    options: GenOptions,
}

async fn version() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "version": "0.1.0-burn-rocket" }))
}

fn parameter_size(family: Family) -> &'static str {
    match family {
        Family::Intent => "0.8B",
        Family::Gemma4 => "5.1B",
        Family::Qwen3Embed => "0.6B",
        Family::Emb2 => "0.4B",
    }
}

fn now_rfc3339() -> String {
    // Seconds-resolution RFC3339-ish timestamp; Ollama clients only parse it loosely.
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("1970-01-01T00:00:{:02}Z", secs % 60)
}

fn model_entry(state: &AppState) -> serde_json::Value {
    let settings = &state.settings;
    let family = settings.family.ollama_architecture();
    serde_json::json!({
        "name": settings.model_name,
        "model": settings.model_name,
        "modified_at": now_rfc3339(),
        "size": 0,
        "digest": "",
        "details": {
            "parent_model": "",
            "format": "safetensors",
            "family": family,
            "families": [family],
            "parameter_size": parameter_size(settings.family),
            "quantization_level": "F32",
        },
        "context_length": settings.max_tokens,
    })
}

async fn tags(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(serde_json::json!({ "models": [model_entry(&state)] }))
}

async fn show(State(state): State<AppState>) -> Json<serde_json::Value> {
    let settings = &state.settings;
    let family = settings.family.ollama_architecture();
    let template = match settings.family {
        Family::Intent => CHATML_TEMPLATE,
        _ => "",
    };
    let mut model_info = serde_json::Map::new();
    model_info.insert(
        "general.architecture".to_string(),
        serde_json::json!(family),
    );
    model_info.insert(
        "general.context_length".to_string(),
        serde_json::json!(settings.max_tokens),
    );
    model_info.insert(
        format!("{family}.context_length"),
        serde_json::json!(settings.max_tokens),
    );
    Json(serde_json::json!({
        "modelfile": "# powered by burn-rocket",
        "parameters": "",
        "template": template,
        "details": model_entry(&state)["details"],
        "model_info": model_info,
    }))
}

// ---------------------------------------------------------------------------
// Chat / generate
// ---------------------------------------------------------------------------

fn resolve_max_new(options: &GenOptions, settings: &super::Settings) -> usize {
    match options.num_predict {
        Some(n) if n > 0 => (n as usize).min(settings.max_new_tokens),
        _ => settings.max_new_tokens, // -1/-2/absent: use the server cap
    }
}

fn resolve_temperature(options: &GenOptions, settings: &super::Settings) -> f32 {
    options
        .temperature
        .map(|t| t as f32)
        .unwrap_or(settings.temperature)
}

fn resolve_num_ctx(options: &GenOptions, settings: &super::Settings) -> Option<usize> {
    options
        .num_ctx
        .filter(|n| *n > 0)
        .map(|n| n.min(settings.max_tokens))
}

fn seed_of(options: &GenOptions) -> u64 {
    options.seed.unwrap_or(0).max(0) as u64
}

async fn chat(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Result<Response, OllamaError> {
    let stream = req.stream.unwrap_or(false);
    let model_name = req
        .model
        .clone()
        .unwrap_or_else(|| state.settings.model_name.clone());
    let turn = ChatTurn {
        messages: req
            .messages
            .iter()
            .map(|m| ChatMessage {
                role: m.role.clone(),
                content: m.content.clone(),
            })
            .collect(),
        prompt: None,
        system: None,
        raw: false,
        image: None,
        audio: None,
        max_new_tokens: resolve_max_new(&req.options, &state.settings),
        temperature: resolve_temperature(&req.options, &state.settings),
        top_p: None,
        top_k: None,
        enable_thinking: false,
        format_json: req.format.is_some(),
        stop_strings: req.options.stop.clone().unwrap_or_default(),
        seed: seed_of(&req.options),
        num_ctx: resolve_num_ctx(&req.options, &state.settings),
    };
    let (completion, timing) = state.compute(move |engine| engine.chat(turn)).await?;
    log::chat(&state, "api.chat", &completion, &timing);

    let created = now_rfc3339();
    if stream {
        let first = serde_json::json!({
            "model": model_name,
            "created_at": created,
            "message": { "role": "assistant", "content": completion.text },
            "done": false,
        });
        let last = serde_json::json!({
            "model": model_name,
            "created_at": now_rfc3339(),
            "message": { "role": "assistant", "content": "" },
            "done": true,
            "done_reason": completion.done_reason,
            "total_duration": ((completion.prompt_eval_ms + completion.eval_ms) * 1e6) as u64,
            "prompt_eval_count": completion.prompt_tokens,
            "prompt_eval_duration": (completion.prompt_eval_ms * 1e6) as u64,
            "eval_count": completion.eval_count,
            "eval_duration": (completion.eval_ms * 1e6) as u64,
        });
        let body = format!("{first}\n{last}\n");
        return Ok(Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "application/x-ndjson")
            .body(Body::from(body))
            .map_err(|e| ApiError::internal(e.to_string()))?);
    }

    Ok(Json(serde_json::json!({
        "model": model_name,
        "created_at": created,
        "message": { "role": "assistant", "content": completion.text },
        "done": true,
        "done_reason": completion.done_reason,
        "total_duration": ((completion.prompt_eval_ms + completion.eval_ms) * 1e6) as u64,
        "load_duration": 0,
        "prompt_eval_count": completion.prompt_tokens,
        "prompt_eval_duration": (completion.prompt_eval_ms * 1e6) as u64,
        "eval_count": completion.eval_count,
        "eval_duration": (completion.eval_ms * 1e6) as u64,
    }))
    .into_response())
}

async fn generate(
    State(state): State<AppState>,
    Json(req): Json<GenerateRequest>,
) -> Result<Response, OllamaError> {
    let stream = req.stream.unwrap_or(false);
    let model_name = req
        .model
        .clone()
        .unwrap_or_else(|| state.settings.model_name.clone());
    let turn = ChatTurn {
        messages: Vec::new(),
        prompt: Some(req.prompt.clone()),
        system: req.system.clone(),
        raw: req.raw.unwrap_or(false),
        image: None,
        audio: None,
        max_new_tokens: resolve_max_new(&req.options, &state.settings),
        temperature: resolve_temperature(&req.options, &state.settings),
        top_p: None,
        top_k: None,
        enable_thinking: false,
        format_json: req.format.is_some(),
        stop_strings: req.options.stop.clone().unwrap_or_default(),
        seed: seed_of(&req.options),
        num_ctx: resolve_num_ctx(&req.options, &state.settings),
    };
    let (completion, timing) = state.compute(move |engine| engine.chat(turn)).await?;
    log::chat(&state, "api.generate", &completion, &timing);

    let created = now_rfc3339();
    if stream {
        let first = serde_json::json!({
            "model": model_name,
            "created_at": created,
            "response": completion.text,
            "done": false,
        });
        let last = serde_json::json!({
            "model": model_name,
            "created_at": now_rfc3339(),
            "response": "",
            "done": true,
            "done_reason": completion.done_reason,
            "context": completion.ids,
            "total_duration": ((completion.prompt_eval_ms + completion.eval_ms) * 1e6) as u64,
            "prompt_eval_count": completion.prompt_tokens,
            "prompt_eval_duration": (completion.prompt_eval_ms * 1e6) as u64,
            "eval_count": completion.eval_count,
            "eval_duration": (completion.eval_ms * 1e6) as u64,
        });
        let body = format!("{first}\n{last}\n");
        return Ok(Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "application/x-ndjson")
            .body(Body::from(body))
            .map_err(|e| ApiError::internal(e.to_string()))?);
    }

    Ok(Json(serde_json::json!({
        "model": model_name,
        "created_at": created,
        "response": completion.text,
        "done": true,
        "done_reason": completion.done_reason,
        "context": completion.ids,
        "total_duration": ((completion.prompt_eval_ms + completion.eval_ms) * 1e6) as u64,
        "load_duration": 0,
        "prompt_eval_count": completion.prompt_tokens,
        "prompt_eval_duration": (completion.prompt_eval_ms * 1e6) as u64,
        "eval_count": completion.eval_count,
        "eval_duration": (completion.eval_ms * 1e6) as u64,
    }))
    .into_response())
}
