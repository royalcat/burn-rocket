//! One inference-speed log line per completed request.
//!
//! `/health` and `/v1/models` never touch the model and are not logged. The
//! level follows `RUST_LOG` (default `info`), so `RUST_LOG=warn` silences these
//! lines. Queue wait is the time from request arrival to acquiring the model
//! lock (the server is single-flight); compute time is the model call itself.

use tracing::info;

use super::AppState;
use super::engine::Completion;
use crate::util::mem::rss_mib;

/// Timings captured around one model call by [`AppState::compute`].
pub(crate) struct RequestTiming {
    /// Request arrival -> model lock acquired (queueing + worker scheduling).
    pub queue_s: f64,
    /// Time inside the model lock (tokenize/prefill/decode/readback).
    pub compute_s: f64,
}

fn round(v: f64, scale: f64) -> f64 {
    (v * scale).round() / scale
}

/// `/v1/embeddings` and `/embed`: `tokens` is the prompt-token count across all
/// inputs; media preparation is included in `compute_s`.
pub(crate) fn embedding(
    state: &AppState,
    endpoint: &'static str,
    inputs: usize,
    tokens: usize,
    timing: &RequestTiming,
) {
    info!(
        endpoint = %endpoint,
        model = %state.settings.model_name,
        inputs,
        tokens,
        queue_s = round(timing.queue_s, 1000.0),
        compute_s = round(timing.compute_s, 1000.0),
        tok_s = round(tokens as f64 / timing.compute_s.max(1e-9), 10.0),
        rss_mib = round(rss_mib(), 10.0),
        "embedding request"
    );
}

/// OpenAI/Ollama chat endpoints: the prefill/decode split comes from the model's
/// own timings; `compute_s` also covers rendering/tokenization/readback.
pub(crate) fn chat(
    state: &AppState,
    endpoint: &'static str,
    completion: &Completion,
    timing: &RequestTiming,
) {
    let prefill_s = completion.prompt_eval_ms / 1000.0;
    let decode_s = completion.eval_ms / 1000.0;
    info!(
        endpoint = %endpoint,
        model = %state.settings.model_name,
        prompt_tokens = completion.prompt_tokens,
        prefill_s = round(prefill_s, 1000.0),
        prefill_tok_s = round(completion.prompt_tokens as f64 / prefill_s.max(1e-9), 10.0),
        gen_tokens = completion.eval_count,
        decode_s = round(decode_s, 1000.0),
        decode_tok_s = round(completion.eval_count as f64 / decode_s.max(1e-9), 10.0),
        queue_s = round(timing.queue_s, 1000.0),
        compute_s = round(timing.compute_s, 1000.0),
        rss_mib = round(rss_mib(), 10.0),
        "chat request"
    );
}
