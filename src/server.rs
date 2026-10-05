//! OpenAI-compatible `/v1/embeddings` server.

use std::sync::{Arc, Mutex};

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

struct Inner {
    model: Qwen3Embedding,
    cfg: Qwen3Config,
    tokenizer: Tokenizer,
    max_tokens: usize,
    model_name: String,
    chunk: usize,
    key_block: usize,
    attn_fused: bool,
    device: Device,
}

#[derive(Clone)]
struct AppState {
    inner: Arc<Mutex<Inner>>,
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
                r#type: "invalid_request_error",
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
    let state = AppState {
        inner: Arc::new(Mutex::new(Inner {
            model,
            cfg,
            tokenizer,
            max_tokens: opts.max_tokens,
            model_name: opts.model_name,
            chunk: opts.chunk,
            key_block: opts.key_block,
            attn_fused: opts.attn_fused,
            device,
        })),
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
        let listener = tokio::net::TcpListener::bind(&opts.addr).await?;
        println!("listening on http://{}", opts.addr);
        axum::serve(listener, app).await?;
        Ok::<(), anyhow::Error>(())
    })
}

async fn list_models(State(state): State<AppState>) -> Json<serde_json::Value> {
    let name = state.inner.lock().unwrap().model_name.clone();
    Json(serde_json::json!({
        "object": "list",
        "data": [{ "id": name, "object": "model", "owned_by": "local" }],
    }))
}

async fn embeddings(
    State(state): State<AppState>,
    Json(req): Json<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>, ApiError> {
    let encoding_format = req.encoding_format.unwrap_or_else(|| "float".to_string());
    if encoding_format != "float" && encoding_format != "base64" {
        return Err(ApiError::bad_request(format!(
            "unsupported encoding_format '{encoding_format}'"
        )));
    }
    let requested_model = req.model;
    let default_model_name = state.inner.lock().unwrap().model_name.clone();
    let state = state.clone();
    let result = tokio::task::spawn_blocking(move || {
        let mut inner = state.inner.lock().unwrap();
        let Inner {
            model,
            cfg,
            tokenizer,
            max_tokens,
            model_name: _,
            chunk,
            key_block,
            attn_fused,
            device,
        } = &mut *inner;
        let texts: Vec<String> = match req.input {
            Input::One(text) => vec![text],
            Input::Many(texts) => texts,
            Input::Tokens(tokens) => {
                let total = tokens.len();
                return Ok((
                    vec![embed_tokens(model, cfg, device.clone(), *chunk, *key_block, *attn_fused, tokens)],
                    total,
                ));
            }
            Input::TokenBatches(batches) => {
                let mut data = Vec::new();
                let mut total = 0;
                for tokens in batches {
                    total += tokens.len();
                    data.push(embed_tokens(model, cfg, device.clone(), *chunk, *key_block, *attn_fused, tokens));
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
            if ids.len() > *max_tokens {
                ids.truncate(*max_tokens);
            }
            if ids.is_empty() {
                return Err(ApiError::bad_request("input must not be empty"));
            }
            total += ids.len();
            let ids_i64: Vec<i64> = ids.iter().map(|&t| t as i64).collect();
            data.push(embed_tokens(model, cfg, device.clone(), *chunk, *key_block, *attn_fused, ids_i64));
        }
        Ok((data, total))
    })
    .await
    .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, format!("worker: {e}")))??;

    let (vectors, total) = result;
    let model_name = requested_model.unwrap_or(default_model_name);
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
