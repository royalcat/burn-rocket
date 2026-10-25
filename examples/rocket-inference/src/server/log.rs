//! One inference-speed log line per completed request.
//!
//! `/health` and `/v1/models` never touch the model and are not logged. The
//! level follows `RUST_LOG` (default `info`), so `RUST_LOG=warn` silences these
//! lines. Queue wait is the time from request arrival to acquiring the model
//! lock (the server is single-flight); compute time is the model call itself.

use tracing::info;
use tracing_opentelemetry::OpenTelemetrySpanExt;

use super::AppState;
use super::engine::Completion;
use crate::util::mem::rss_mib;

/// Timings captured around one model call by [`AppState::compute`].
pub(crate) struct RequestTiming {
    /// Request arrival -> model lock acquired (queueing + worker scheduling).
    pub queue_s: f64,
    /// Time inside the model lock (tokenize/prefill/decode/readback).
    pub compute_s: f64,
    /// Vision-tower forward time inside `compute_s` (image/video inputs only;
    /// 0 for text-only and audio-only requests).
    pub encode_s: f64,
}

fn round(v: f64, scale: f64) -> f64 {
    (v * scale).round() / scale
}

/// `/v1/embeddings` and `/embed`: `tokens` is the prompt-token count across all
/// inputs; media preparation is included in `compute_s`, `encode_s` splits out
/// the vision-tower forwards and `vision_attn` names the tower's attention mode
/// (both absent for text-only requests).
pub(crate) fn embedding(
    state: &AppState,
    endpoint: &'static str,
    inputs: usize,
    tokens: usize,
    timing: &RequestTiming,
) {
    // The span attributes keep the raw timings; the log line keeps its
    // rounded presentation (and the exact field order).
    let tok_s = round(tokens as f64 / timing.compute_s.max(1e-9), 10.0);
    let encode_s = (timing.encode_s > 0.0).then(|| round(timing.encode_s, 1000.0));
    let vision_attn =
        (timing.encode_s > 0.0).then(|| crate::gemma::vision::vision_stats().attn_mode.name());
    let rss = rss_mib();

    let span = tracing::Span::current();
    span.set_attribute("rocket.endpoint", endpoint);
    span.set_attribute("inputs", as_i64(inputs));
    span.set_attribute("tokens", as_i64(tokens));
    span.set_attribute("queue_s", timing.queue_s);
    span.set_attribute("compute_s", timing.compute_s);
    span.set_attribute("tok_s", tok_s);
    if timing.encode_s > 0.0 {
        span.set_attribute("encode_s", timing.encode_s);
    }
    if let Some(vision_attn) = vision_attn {
        span.set_attribute("vision_attn", vision_attn);
    }
    span.set_attribute("rss_mib", rss);

    info!(
        endpoint = %endpoint,
        model = %state.settings.model_name,
        inputs,
        tokens,
        queue_s = round(timing.queue_s, 1000.0),
        compute_s = round(timing.compute_s, 1000.0),
        tok_s,
        encode_s,
        vision_attn,
        rss_mib = round(rss, 10.0),
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
    let prefill_raw = completion.prompt_eval_ms / 1000.0;
    let decode_raw = completion.eval_ms / 1000.0;
    let prefill_tok_s = round(
        completion.prompt_tokens as f64 / prefill_raw.max(1e-9),
        10.0,
    );
    let decode_tok_s = round(completion.eval_count as f64 / decode_raw.max(1e-9), 10.0);
    let rss = rss_mib();

    let span = tracing::Span::current();
    span.set_attribute("rocket.endpoint", endpoint);
    span.set_attribute("prompt_tokens", as_i64(completion.prompt_tokens));
    span.set_attribute("prefill_s", prefill_raw);
    span.set_attribute("prefill_tok_s", prefill_tok_s);
    span.set_attribute("gen_tokens", as_i64(completion.eval_count));
    span.set_attribute("decode_s", decode_raw);
    span.set_attribute("decode_tok_s", decode_tok_s);
    span.set_attribute("queue_s", timing.queue_s);
    span.set_attribute("compute_s", timing.compute_s);
    span.set_attribute("rss_mib", rss);

    info!(
        endpoint = %endpoint,
        model = %state.settings.model_name,
        prompt_tokens = completion.prompt_tokens,
        prefill_s = round(prefill_raw, 1000.0),
        prefill_tok_s,
        gen_tokens = completion.eval_count,
        decode_s = round(decode_raw, 1000.0),
        decode_tok_s,
        queue_s = round(timing.queue_s, 1000.0),
        compute_s = round(timing.compute_s, 1000.0),
        rss_mib = round(rss, 10.0),
        "chat request"
    );
}

/// `tracing` has no `Value` impl for `usize`; saturate instead of wrapping.
fn as_i64(value: usize) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}
