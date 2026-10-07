//! Ollama-compatible API for the intent model (`/api/chat`, `/api/generate`, ...).
//!
//! OpenViking's recommended query-planner config points a LiteLLM `ollama/...`
//! route at this server; the chat template mirrors the model's Ollama template
//! exactly:
//!
//! ```text
//! {{ if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}<|im_start|>user\n{{ .Prompt }}<|im_end|>\n<|im_start|>assistant\n
//! ```

use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use axum::{
    Json, Router,
    body::Body,
    extract::State,
    http::{HeaderValue, StatusCode, header},
    middleware::map_request,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use crate::qwen35_intent::model::IntentModel;
use crate::util::http::lock_or_recover;

pub struct OllamaOptions {
    pub addr: String,
    pub model_name: String,
    /// Context cap (prompt tokens).
    pub max_tokens: usize,
    /// Default cap on generated tokens.
    pub max_new_tokens: usize,
    pub temperature: f32,
    /// Token ids that end generation (EOS and stop specials).
    pub stop_ids: Vec<u32>,
}

/// Immutable server settings, kept outside the model lock.
struct Settings {
    model_name: String,
    max_tokens: usize,
    max_new_tokens: usize,
    temperature: f32,
    stop_ids: Vec<u32>,
}

struct Inner {
    model: IntentModel,
    tokenizer: Tokenizer,
}

#[derive(Clone)]
struct AppState {
    inner: Arc<Mutex<Inner>>,
    settings: Arc<Settings>,
}

// ---------------------------------------------------------------------------
// Request/response types
// ---------------------------------------------------------------------------

#[derive(Deserialize, Default)]
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
struct ChatMessage {
    role: String,
    #[serde(default)]
    content: String,
}

#[derive(Deserialize)]
struct ChatRequest {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    format: Option<serde_json::Value>,
    #[serde(default)]
    options: GenOptions,
    /// Accepted (and ignored): the SFT model was not trained with thinking.
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

#[derive(Serialize)]
struct ErrorBody {
    error: String,
}

struct ApiError(StatusCode, String);

impl ApiError {
    fn bad_request(msg: impl Into<String>) -> Self {
        Self(StatusCode::BAD_REQUEST, msg.into())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(ErrorBody { error: self.1 })).into_response()
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

pub fn serve(model: IntentModel, tokenizer: Tokenizer, opts: OllamaOptions) -> Result<()> {
    let OllamaOptions {
        addr,
        model_name,
        max_tokens,
        max_new_tokens,
        temperature,
        stop_ids,
    } = opts;

    let state = AppState {
        inner: Arc::new(Mutex::new(Inner { model, tokenizer })),
        settings: Arc::new(Settings {
            model_name,
            max_tokens,
            max_new_tokens,
            temperature,
            stop_ids,
        }),
    };
    let app = Router::new()
        .route("/", get(|| async { "Ollama is running" }))
        .route("/health", get(|| async { "ok" }))
        .route("/api/version", get(version))
        .route("/api/tags", get(tags))
        .route("/api/show", post(show))
        .route("/api/chat", post(chat))
        .route("/api/generate", post(generate))
        .layer(map_request(force_json_content_type))
        .with_state(state);

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?;
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        println!("listening on http://{}", addr);
        axum::serve(listener, app).await?;
        Ok::<(), anyhow::Error>(())
    })
}

async fn version() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "version": "0.1.0-ov-intent" }))
}

fn model_entry(name: &str, max_tokens: usize) -> serde_json::Value {
    serde_json::json!({
        "name": name,
        "model": name,
        "modified_at": now_rfc3339(),
        "size": 0,
        "digest": "",
        "details": {
            "parent_model": "",
            "format": "safetensors",
            "family": "qwen3_5",
            "families": ["qwen3_5"],
            "parameter_size": "0.8B",
            "quantization_level": "F32",
        },
        "context_length": max_tokens,
    })
}

async fn tags(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "models": [model_entry(&state.settings.model_name, state.settings.max_tokens)]
    }))
}

async fn show(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "modelfile": "# powered by burn-rocket",
        "parameters": "",
        "template": "{{ if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}<|im_start|>user\n{{ .Prompt }}<|im_end|>\n<|im_start|>assistant\n",
        "details": model_entry(&state.settings.model_name, state.settings.max_tokens)["details"],
        "model_info": {
            "general.architecture": "qwen3_5",
            "general.context_length": state.settings.max_tokens,
            "qwen3_5.context_length": state.settings.max_tokens,
        },
    }))
}

// ---------------------------------------------------------------------------
// Chat / generate
// ---------------------------------------------------------------------------

fn render_chat(messages: &[ChatMessage]) -> Result<String, ApiError> {
    let mut out = String::new();
    for m in messages {
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
    out.push_str("<|im_start|>assistant\n");
    Ok(out)
}

impl GenerateRequest {
    fn render(&self) -> String {
        if self.raw.unwrap_or(false) {
            return self.prompt.clone();
        }
        let mut out = String::new();
        if let Some(sys) = &self.system {
            if !sys.is_empty() {
                out.push_str("<|im_start|>system\n");
                out.push_str(sys);
                out.push_str("<|im_end|>\n");
            }
        }
        out.push_str("<|im_start|>user\n");
        out.push_str(&self.prompt);
        out.push_str("<|im_end|>\n<|im_start|>assistant\n");
        out
    }
}

struct Completion {
    text: String,
    ids: Vec<u32>,
    prompt_tokens: usize,
    prompt_eval_ms: f64,
    eval_ms: f64,
    eval_count: usize,
    done_reason: &'static str,
}

fn run_completion(
    state: &AppState,
    prompt: &str,
    options: &GenOptions,
    format_json: bool,
) -> Result<Completion, ApiError> {
    let settings = &state.settings;
    let mut inner = lock_or_recover(&state.inner);
    let Inner { model, tokenizer } = &mut *inner;

    let enc = tokenizer
        .encode(prompt, false)
        .map_err(|e| ApiError::bad_request(format!("tokenize: {e}")))?;
    let mut ids: Vec<u32> = enc.get_ids().to_vec();
    let num_ctx = options
        .num_ctx
        .filter(|n| *n > 0)
        .unwrap_or(settings.max_tokens)
        .min(settings.max_tokens);
    if ids.len() > num_ctx {
        // Keep the tail of the conversation (Ollama truncates the context too).
        ids.drain(0..ids.len() - num_ctx);
    }
    if ids.is_empty() {
        return Err(ApiError::bad_request("empty prompt"));
    }

    let max_new = match options.num_predict {
        Some(n) if n > 0 => (n as usize).min(settings.max_new_tokens),
        Some(_) => settings.max_new_tokens, // -1/-2: use the server cap
        None => settings.max_new_tokens,
    };
    let temperature = options
        .temperature
        .map(|t| t as f32)
        .unwrap_or(settings.temperature);
    let seed = options.seed.unwrap_or(0) as u64;

    let t0 = Instant::now();
    // Stop on the EOS set; also stop on the caller's string stops by trimming below.
    let (new_ids, stats) = model.generate(&ids, max_new, &settings.stop_ids, temperature, seed);
    let total = t0.elapsed().as_secs_f64();
    let _ = total;

    let mut text = tokenizer
        .decode(&new_ids, false)
        .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, format!("decode: {e}")))?;
    let mut done_reason = if stats.stopped { "stop" } else { "length" };
    if let Some(stops) = &options.stop {
        let mut cut = None;
        for s in stops {
            if s.is_empty() {
                continue;
            }
            if let Some(pos) = text.find(s.as_str()) {
                cut = Some(cut.map_or(pos, |c: usize| c.min(pos)));
            }
        }
        if let Some(c) = cut {
            text.truncate(c);
            done_reason = "stop";
        }
    }
    if format_json {
        // The model is SFT'd to emit a JSON object; trim any stray wrapper.
        if let Some(c) = extract_json(&text) {
            text = c;
        }
    }

    Ok(Completion {
        text,
        ids: new_ids,
        prompt_tokens: ids.len(),
        prompt_eval_ms: stats.prefill_s * 1000.0,
        eval_ms: stats.decode_s * 1000.0,
        eval_count: stats.steps,
        done_reason,
    })
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

async fn chat(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Result<Response, ApiError> {
    let prompt = render_chat(&req.messages)?;
    let stream = req.stream.unwrap_or(false);
    let format = req.format.clone();
    let options = req.options;
    let model_name = req
        .model
        .unwrap_or_else(|| state.settings.model_name.clone());
    let state2 = state.clone();
    let completion = tokio::task::spawn_blocking(move || {
        catch_compute(move || run_completion(&state2, &prompt, &options, format.is_some()))
    })
    .await
    .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, format!("worker: {e}")))??;

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
            .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?);
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
) -> Result<Response, ApiError> {
    let prompt = req.render();
    let stream = req.stream.unwrap_or(false);
    let format = req.format.clone();
    let options = req.options;
    let model_name = req
        .model
        .unwrap_or_else(|| state.settings.model_name.clone());
    let state2 = state.clone();
    let completion = tokio::task::spawn_blocking(move || {
        catch_compute(move || run_completion(&state2, &prompt, &options, format.is_some()))
    })
    .await
    .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, format!("worker: {e}")))??;

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
            .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?);
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

/// Run the blocking compute, converting a panic into an HTTP error. Hard NPU
/// device failures are unrecoverable: log and exit so a supervisor restarts a
/// clean engine.
fn catch_compute<T>(f: impl FnOnce() -> Result<T, ApiError>) -> Result<T, ApiError> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(result) => result,
        Err(payload) => Err(panic_to_api(payload)),
    }
}

fn panic_to_api(payload: Box<dyn Any + Send>) -> ApiError {
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    if let Some(failure) = payload.downcast_ref::<burn_rocket::OpFailure>() {
        let rc = failure.error.rc;
        let detail = format!(
            "{} failed (rc={rc}, m={} k={} n={})",
            failure.error.op, failure.m, failure.k, failure.n
        );
        if rc == burn_rocket::ffi::ROCKET_E_NOMEM {
            return ApiError(
                StatusCode::SERVICE_UNAVAILABLE,
                format!("NPU out of memory: {detail}; retry the request"),
            );
        }
        if rc == burn_rocket::ffi::ROCKET_E_SHAPE || rc == burn_rocket::ffi::ROCKET_E_TILING {
            return ApiError(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("NPU error: {detail}"),
            );
        }
        eprintln!("fatal NPU failure: {detail}; exiting for a clean restart");
        std::process::exit(1);
    }
    ApiError(
        StatusCode::INTERNAL_SERVER_ERROR,
        format!(
            "internal error: {}",
            payload
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "see server log".to_string())
        ),
    )
}
