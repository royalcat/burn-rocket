//! OpenAI-compatible `/v1/embeddings` plus a native multimodal `/embed`
//! endpoint.
//!
//! The model is behind one mutex (a forward is CPU-bound and single-flight);
//! panics inside a forward are contained so the server survives them.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

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

use crate::gemma::config::Emb2Config;
use crate::gemma::embeddinggemma::model::Emb2Model;
use crate::gemma::inputs::{self, DebugPaths, MediaInputs};
use crate::gemma::media::resolve_media;
use crate::util::http::{lock_or_recover, panic_message};

pub struct ServeOptions {
    pub addr: String,
    pub model_name: String,
    pub model_dir: PathBuf,
    pub attn_chunk: usize,
    pub max_tokens: usize,
    pub video_fps: f64,
    pub video_max_frames: usize,
}

/// Immutable settings, kept outside the model lock.
struct Settings {
    model_name: String,
    attn_chunk: usize,
    max_tokens: usize,
    image_soft_tokens: usize,
    video_soft_tokens: usize,
    video_fps: f64,
    video_max_frames: usize,
    prompts: std::collections::HashMap<String, String>,
}

struct Inner {
    model: Emb2Model,
    tokenizer: Tokenizer,
    device: Device,
    dtype: DType,
}

#[derive(Clone)]
struct AppState {
    inner: Arc<Mutex<Inner>>,
    settings: Arc<Settings>,
}

pub fn serve(
    model: Emb2Model,
    _cfg: Emb2Config,
    tokenizer: Tokenizer,
    device: Device,
    dtype: DType,
    opts: ServeOptions,
) -> Result<()> {
    let (image_soft_tokens, video_soft_tokens) = inputs::media_soft_tokens(&opts.model_dir);
    let settings = Arc::new(Settings {
        model_name: opts.model_name.clone(),
        attn_chunk: opts.attn_chunk,
        max_tokens: opts.max_tokens,
        image_soft_tokens,
        video_soft_tokens,
        video_fps: opts.video_fps,
        video_max_frames: opts.video_max_frames,
        prompts: inputs::task_prompts(&opts.model_dir),
    });
    let state = AppState {
        inner: Arc::new(Mutex::new(Inner {
            model,
            tokenizer,
            device,
            dtype,
        })),
        settings,
    };
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/embeddings", post(embeddings))
        .route("/embed", post(embed_native))
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
struct OpenAiRequest {
    input: Input,
    #[serde(default)]
    #[allow(dead_code)]
    model: Option<String>,
    #[serde(default)]
    dim: Option<usize>,
    #[serde(default)]
    prompt: Option<String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Input {
    Text(String),
    Texts(Vec<String>),
}

#[derive(Deserialize, Default)]
struct NativeRequest {
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    image: Option<String>,
    #[serde(default)]
    video: Option<String>,
    #[serde(default)]
    audio: Option<String>,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    dim: Option<usize>,
}

fn bad_request(msg: impl std::fmt::Display) -> Response {
    (StatusCode::BAD_REQUEST, msg.to_string()).into_response()
}

fn internal_error(msg: impl std::fmt::Display) -> Response {
    (StatusCode::INTERNAL_SERVER_ERROR, msg.to_string()).into_response()
}

async fn embeddings(State(state): State<AppState>, Json(req): Json<OpenAiRequest>) -> Response {
    let texts = match req.input {
        Input::Text(t) => vec![t],
        Input::Texts(v) => v,
    };
    if texts.is_empty() {
        return bad_request("empty input");
    }
    let dim = req.dim;
    let prompt = req.prompt.clone();
    let inner = state.inner.clone();
    let settings = state.settings.clone();
    let computed = tokio::task::spawn_blocking(move || -> Result<Vec<(Vec<f32>, usize)>, String> {
        let mut guard = lock_or_recover(&inner);
        let mut out = Vec::with_capacity(texts.len());
        for text in texts {
            out.push(embed_one(
                &mut guard,
                &settings,
                Some(text),
                None,
                None,
                None,
                prompt.as_deref(),
                dim,
            )?);
        }
        Ok(out)
    })
    .await;
    match computed {
        Ok(Ok(vectors)) => {
            let tokens: usize = vectors.iter().map(|(_, n)| n).sum();
            let data: Vec<serde_json::Value> = vectors
                .iter()
                .enumerate()
                .map(|(i, (v, _))| {
                    serde_json::json!({"object": "embedding", "embedding": v, "index": i})
                })
                .collect();
            Json(serde_json::json!({
                "object": "list",
                "data": data,
                "model": state.settings.model_name,
                "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
            }))
            .into_response()
        }
        Ok(Err(e)) => bad_request(e),
        Err(e) => internal_error(format!("task join error: {e}")),
    }
}

async fn embed_native(State(state): State<AppState>, Json(req): Json<NativeRequest>) -> Response {
    let inner = state.inner.clone();
    let settings = state.settings.clone();
    let computed = tokio::task::spawn_blocking(move || -> Result<Vec<f32>, String> {
        let mut guard = lock_or_recover(&inner);
        let image = req
            .image
            .as_deref()
            .map(|v| resolve_media(v, "img"))
            .transpose()?;
        let video = req
            .video
            .as_deref()
            .map(|v| resolve_media(v, "mp4"))
            .transpose()?;
        let audio = req
            .audio
            .as_deref()
            .map(|v| resolve_media(v, "wav"))
            .transpose()?;
        embed_one(
            &mut guard,
            &settings,
            req.text.clone(),
            image,
            video,
            audio,
            req.prompt.as_deref(),
            req.dim,
        )
        .map(|(v, _)| v)
    })
    .await;
    match computed {
        Ok(Ok(v)) => Json(serde_json::json!({
            "model": state.settings.model_name,
            "dim": v.len(),
            "embedding": v,
        }))
        .into_response(),
        Ok(Err(e)) => bad_request(e),
        Err(e) => internal_error(format!("task join error: {e}")),
    }
}

fn embed_one(
    inner: &mut Inner,
    settings: &Settings,
    text: Option<String>,
    image: Option<PathBuf>,
    video: Option<PathBuf>,
    audio: Option<PathBuf>,
    prompt: Option<&str>,
    dim: Option<usize>,
) -> Result<(Vec<f32>, usize), String> {
    let prefix = match prompt {
        Some(name) => settings
            .prompts
            .get(&name.to_lowercase())
            .cloned()
            .ok_or_else(|| format!("unknown prompt '{name}'"))?,
        None => String::new(),
    };
    let has_media = image.is_some() || video.is_some() || audio.is_some();
    let req = MediaInputs {
        text,
        image,
        video,
        audio,
        prompt_prefix: prefix,
        max_tokens: if has_media { None } else { Some(settings.max_tokens) },
        image_soft_tokens: settings.image_soft_tokens,
        video_soft_tokens: settings.video_soft_tokens,
        video_fps: settings.video_fps,
        video_max_frames: settings.video_max_frames,
        attn_chunk: settings.attn_chunk,
    };
    let prepared = inputs::prepare(&inner.model, &inner.tokenizer, &req, &DebugPaths::default())
        .map_err(|e| format!("{e:#}"))?;
    let n = prepared.ids.len();
    let input = inputs::make_input(&prepared.ids, &inner.device);
    let rope = inner
        .model
        .text()
        .rope_tables(n, inner.dtype, &inner.device);
    let out = catch_unwind(AssertUnwindSafe(|| {
        inner
            .model
            .embed_ids(input, &rope, settings.attn_chunk, prepared.soft)
    }))
    .map_err(|payload| format!("forward panicked: {}", panic_message(&payload)))?;
    let mut v: Vec<f32> = out
        .cast(DType::F32)
        .into_data()
        .try_to_vec()
        .map_err(|e| format!("read embedding: {e}"))?;
    if let Some(d) = dim {
        v.truncate(d);
    }
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for x in &mut v {
        *x /= norm;
    }
    Ok((v, n))
}
