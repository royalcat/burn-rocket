//! OpenAI-compatible `/v1/embeddings` server.

use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use anyhow::Result;
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use base64::Engine;
use burn::prelude::*;
use burn::tensor::{DType, Int, TensorData};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use crate::model::{Qwen3Config, Qwen3Embedding, RopeCache};

pub struct ServeOptions {
    pub addr: String,
    pub max_tokens: usize,
    pub model_name: String,
    pub chunk: usize,
    pub key_block: usize,
    pub attn_fused: bool,
}

/// Immutable server settings, kept outside the model lock so `/v1/models` and
/// request routing never block behind a running forward.
struct Settings {
    max_tokens: usize,
    model_name: String,
    chunk: usize,
    key_block: usize,
    attn_fused: bool,
}

struct Inner {
    model: Qwen3Embedding,
    cfg: Qwen3Config,
    tokenizer: Tokenizer,
    device: Device,
}

#[derive(Clone)]
struct AppState {
    inner: Arc<Mutex<Inner>>,
    settings: Arc<Settings>,
}

/// Lock the model, recovering from poisoning. A panic inside a forward is
/// contained (see [`catch_compute`]) and the model holds no per-request state
/// (RoPE tables, activations and device scratch are rebuilt per call), so a
/// poisoned lock must not wedge the server until restart.
fn lock_inner(state: &AppState) -> MutexGuard<'_, Inner> {
    lock_or_recover(&state.inner)
}

fn lock_or_recover<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

#[derive(Deserialize)]
struct EmbeddingRequest {
    input: Input,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    encoding_format: Option<String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Input {
    One(String),
    Many(Vec<String>),
    Tokens(Vec<i64>),
    TokenBatches(Vec<Vec<i64>>),
}

#[derive(Serialize)]
struct EmbeddingResponse {
    object: &'static str,
    data: Vec<EmbeddingData>,
    model: String,
    usage: Usage,
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

#[derive(Serialize)]
struct ErrorBody {
    error: ErrorDetail,
}

#[derive(Serialize)]
struct ErrorDetail {
    message: String,
    r#type: &'static str,
}

#[derive(Debug)]
struct ApiError(StatusCode, String);

impl ApiError {
    fn bad_request(msg: impl Into<String>) -> Self {
        Self(StatusCode::BAD_REQUEST, msg.into())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = ErrorBody {
            error: ErrorDetail {
                message: self.1,
                r#type: if self.0.is_server_error() {
                    "server_error"
                } else {
                    "invalid_request_error"
                },
            },
        };
        (self.0, Json(body)).into_response()
    }
}

pub fn serve(
    model: Qwen3Embedding,
    cfg: Qwen3Config,
    tokenizer: Tokenizer,
    device: Device,
    opts: ServeOptions,
) -> Result<()> {
    let ServeOptions {
        addr,
        max_tokens,
        model_name,
        chunk,
        key_block,
        attn_fused,
    } = opts;
    let state = AppState {
        inner: Arc::new(Mutex::new(Inner {
            model,
            cfg,
            tokenizer,
            device,
        })),
        settings: Arc::new(Settings {
            max_tokens,
            model_name,
            chunk,
            key_block,
            attn_fused,
        }),
    };
    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/v1/models", get(list_models))
        .route("/v1/embeddings", post(embeddings))
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

async fn list_models(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "object": "list",
        "data": [{ "id": state.settings.model_name, "object": "model", "owned_by": "local" }],
    }))
}

async fn embeddings(
    State(state): State<AppState>,
    Json(req): Json<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>, ApiError> {
    let EmbeddingRequest {
        input,
        model,
        encoding_format,
    } = req;
    let encoding_format = encoding_format.unwrap_or_else(|| "float".to_string());
    if encoding_format != "float" && encoding_format != "base64" {
        return Err(ApiError::bad_request(format!(
            "unsupported encoding_format '{encoding_format}'"
        )));
    }
    let model_name = model.unwrap_or_else(|| state.settings.model_name.clone());

    let state = state.clone();
    let computed = tokio::task::spawn_blocking(move || {
        // The forward runs on a blocking thread; a panic inside it (e.g. an NPU
        // allocation failure) becomes an HTTP error instead of poisoning the
        // lock and killing every later request.
        catch_compute(move || run_embeddings(&state, input))
    })
    .await
    .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, format!("worker: {e}")))?;
    let (vectors, total) = computed?;

    let data = vectors
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
    Ok(Json(EmbeddingResponse {
        object: "list",
        data,
        model: model_name,
        usage: Usage {
            prompt_tokens: total,
            total_tokens: total,
        },
    }))
}

/// Run the blocking compute, converting a panic into an HTTP error. Hard NPU
/// device failures are unrecoverable: they are logged and the process exits so
/// a supervisor restarts a clean engine.
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
        format!("internal error: {}", panic_message(&*payload)),
    )
}

fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "see server log".to_string()
    }
}

/// One request's tokenization + forward, with the model lock held.
fn run_embeddings(state: &AppState, input: Input) -> Result<(Vec<Vec<f32>>, usize), ApiError> {
    let settings = &state.settings;
    let mut inner = lock_inner(state);
    let Inner {
        model,
        cfg,
        tokenizer,
        device,
    } = &mut *inner;

    let texts: Vec<String> = match input {
        Input::One(text) => vec![text],
        Input::Many(texts) => texts,
        Input::Tokens(tokens) => {
            if tokens.is_empty() {
                return Err(ApiError::bad_request("input must not be empty"));
            }
            if tokens.len() > settings.max_tokens {
                return Err(ApiError::bad_request(format!(
                    "raw token input has {} tokens, over the server limit of {}",
                    tokens.len(),
                    settings.max_tokens
                )));
            }
            let total = tokens.len();
            return Ok((
                vec![embed_tokens(
                    model,
                    cfg,
                    device.clone(),
                    settings.chunk,
                    settings.key_block,
                    settings.attn_fused,
                    tokens,
                )],
                total,
            ));
        }
        Input::TokenBatches(batches) => {
            if batches.is_empty() {
                return Err(ApiError::bad_request("input must not be empty"));
            }
            let mut data = Vec::new();
            let mut total = 0;
            for tokens in batches {
                if tokens.is_empty() {
                    return Err(ApiError::bad_request("input must not be empty"));
                }
                if tokens.len() > settings.max_tokens {
                    return Err(ApiError::bad_request(format!(
                        "raw token input has {} tokens, over the server limit of {}",
                        tokens.len(),
                        settings.max_tokens
                    )));
                }
                total += tokens.len();
                data.push(embed_tokens(
                    model,
                    cfg,
                    device.clone(),
                    settings.chunk,
                    settings.key_block,
                    settings.attn_fused,
                    tokens,
                ));
            }
            return Ok((data, total));
        }
    };
    if texts.is_empty() {
        return Err(ApiError::bad_request("input must not be empty"));
    }
    let mut data = Vec::new();
    let mut total = 0;
    for text in &texts {
        let enc = tokenizer
            .encode(text.as_str(), true)
            .map_err(|e| ApiError::bad_request(format!("tokenize: {e}")))?;
        let mut ids = enc.get_ids().to_vec();
        if ids.len() > settings.max_tokens {
            ids.truncate(settings.max_tokens);
        }
        if ids.is_empty() {
            return Err(ApiError::bad_request("input must not be empty"));
        }
        total += ids.len();
        let ids_i64: Vec<i64> = ids.iter().map(|&t| t as i64).collect();
        data.push(embed_tokens(
            model,
            cfg,
            device.clone(),
            settings.chunk,
            settings.key_block,
            settings.attn_fused,
            ids_i64,
        ));
    }
    Ok((data, total))
}

/// Runs the embedding model on one token sequence, returning the 1024-dim vector.
fn embed_tokens(
    model: &Qwen3Embedding,
    cfg: &Qwen3Config,
    device: Device,
    chunk: usize,
    key_block: usize,
    attn_fused: bool,
    ids: Vec<i64>,
) -> Vec<f32> {
    let n = ids.len();
    let input = Tensor::<2, Int>::from_data(TensorData::new(ids, [1, n]), &device);
    let rope = RopeCache::new(n, cfg.head_dim, cfg.rope_theta, DType::F32, &device);
    let out = model.forward(input, &rope, chunk, key_block, attn_fused);
    out.cast(DType::F32)
        .to_data()
        .try_to_vec()
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lock_or_recover_survives_a_poisoned_mutex() {
        let mutex = Arc::new(Mutex::new(7u32));
        let poisoned = mutex.clone();
        let joined = std::thread::spawn(move || {
            let _guard = poisoned.lock().unwrap();
            panic!("poison the mutex");
        })
        .join();
        assert!(joined.is_err(), "the helper thread must have panicked");
        assert_eq!(*lock_or_recover(&mutex), 7);
    }

    #[test]
    fn catch_compute_turns_panics_into_server_errors() {
        let err = catch_compute::<()>(|| panic!("boom")).unwrap_err();
        assert_eq!(err.0, StatusCode::INTERNAL_SERVER_ERROR);
        assert!(err.1.contains("boom"), "message was {:?}", err.1);

        let ok = catch_compute(|| Ok::<_, ApiError>(3)).unwrap();
        assert_eq!(ok, 3);
    }

    #[test]
    fn catch_compute_keeps_handler_errors() {
        let err = catch_compute::<()>(|| Err(ApiError::bad_request("bad input"))).unwrap_err();
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
        assert_eq!(err.1, "bad input");
    }
}
