//! Gemma 4 chat template rendering and the generation loop.

use std::path::Path;

use anyhow::Result;
use burn::prelude::*;
use burn::tensor::DType;
use tokenizers::Tokenizer;

use crate::gemma::gemma4::model::{GenKv, GenRoot, GenRopes, gen_stage_stats};
use crate::server::telemetry;

#[derive(Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
        }
    }
}

/// Renders the canonical Gemma 4 chat template (tools omitted; thinking is
/// emitted only when `enable_thinking` is set). Matches the HF tokenizer's
/// rendering for system/user/model turns.
pub fn render(messages: &[Message], enable_thinking: bool) -> String {
    let mut out = String::new();
    out.push_str("<bos>");
    let first_is_system = messages
        .first()
        .map(|m| m.role == "system" || m.role == "developer")
        .unwrap_or(false);
    let mut rest = messages;
    if enable_thinking || first_is_system {
        out.push_str("<|turn>system\n");
        if enable_thinking {
            out.push_str("<|think|>\n");
        }
        if first_is_system {
            out.push_str(messages[0].content.trim());
            rest = &messages[1..];
        }
        out.push_str("<turn|>\n");
    }
    for (i, m) in rest.iter().enumerate() {
        let role = if m.role == "assistant" {
            "model"
        } else {
            m.role.as_str()
        };
        let next_is_assistant = rest
            .get(i + 1)
            .map(|n| n.role == "assistant")
            .unwrap_or(false);
        out.push_str("<|turn>");
        out.push_str(role);
        out.push('\n');
        out.push_str(m.content.trim());
        // A model turn followed by another model turn is a continuation (the
        // closing token is deferred).
        if !(role == "model" && next_is_assistant) {
            out.push_str("<turn|>\n");
        }
    }
    out.push_str("<|turn>model\n");
    out
}

/// Tokenize a rendered conversation (`add_special_tokens = false`: the template
/// already emits `<bos>`).
pub fn encode(tokenizer: &Tokenizer, rendered: &str) -> Result<Vec<u32>> {
    let enc = tokenizer
        .encode(rendered, false)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
    Ok(enc.get_ids().to_vec())
}

#[derive(Debug, Clone)]
pub struct GenOptions {
    pub max_new_tokens: usize,
    pub greedy: bool,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub eos: Vec<u32>,
    /// Collect per-step top-8 logits into `GenStats::top8` (debug/parity dumps).
    pub collect_top8: bool,
}

impl Default for GenOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 64,
            greedy: true,
            temperature: 1.0,
            top_k: 64,
            top_p: 0.95,
            eos: vec![1, 106, 50],
            collect_top8: false,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct GenStats {
    pub prefill_s: f64,
    pub decode_s: f64,
    pub prompt_tokens: usize,
    pub generated: usize,
    /// True when generation ended on an EOS token (not the length cap).
    pub stopped: bool,
    /// Per-step top-8 `(id, logit)` (cheap; the CLI can dump it for parity checks).
    pub top8: Vec<Vec<(u32, f32)>>,
}

/// Top-`k` ids and logits of a logit tensor (any rank, flattened).
pub fn top_k<const D: usize>(logits: &Tensor<D>, k: usize) -> Vec<(u32, f32)> {
    let n: usize = logits.dims().iter().product();
    let values: Vec<f32> = logits
        .clone()
        .reshape([n])
        .into_data()
        .try_to_vec()
        .expect("logits f32");
    let mut idx: Vec<usize> = (0..values.len()).collect();
    idx.sort_unstable_by(|a, b| values[*b].total_cmp(&values[*a]));
    idx.into_iter()
        .take(k)
        .map(|i| (i as u32, values[i]))
        .collect()
}

/// Deterministic SplitMix64 for the sampled path (reproducible runs).
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Greedy or top-k/top-p/temperature sampling from `[1, vocab]` logits.
fn sample(logits: &Tensor<2>, opts: &GenOptions, rng: &mut SplitMix64) -> Result<u32> {
    let mut v: Vec<f32> = logits.clone().cast(DType::F32).into_data().try_to_vec()?;
    if opts.greedy {
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &x) in v.iter().enumerate() {
            if x > best_v {
                best_v = x;
                best = i;
            }
        }
        return Ok(best as u32);
    }
    let temp = opts.temperature.max(1e-6);
    for x in &mut v {
        *x /= temp;
    }
    // Softmax in f32.
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = v.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in &mut probs {
        *p /= sum;
    }
    // Top-k: zero everything below the k-th largest.
    if opts.top_k > 0 && opts.top_k < probs.len() {
        let mut sorted = probs.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = sorted[opts.top_k - 1];
        for p in &mut probs {
            if *p < threshold {
                *p = 0.0;
            }
        }
    }
    // Top-p: keep the smallest prefix (by probability) whose cumulative mass
    // reaches top_p, always keeping the argmax.
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut cum = 0.0f32;
    let mut keep = vec![false; probs.len()];
    for (rank, &idx) in order.iter().enumerate() {
        keep[idx] = true;
        cum += probs[idx];
        if cum >= opts.top_p && rank > 0 {
            break;
        }
    }
    let mut mass = 0.0f32;
    for (i, p) in probs.iter_mut().enumerate() {
        if !keep[i] {
            *p = 0.0;
        }
        mass += *p;
    }
    if mass <= 0.0 {
        return Ok(order[0] as u32);
    }
    let target = rng.next_f32() * mass;
    let mut acc = 0.0f32;
    for (i, p) in probs.iter().enumerate() {
        acc += *p;
        if acc >= target {
            return Ok(i as u32);
        }
    }
    Ok(order[0] as u32)
}

/// Prefill + decode loop with optional media soft tokens scattered into the
/// placeholder positions of `ids` (the ids are used for the PLE token lookup,
/// with the media slots replaced by `pad_id`, exactly like the reference).
#[allow(clippy::too_many_arguments)]
pub fn generate_with_media(
    model: &GenRoot,
    lm_head: &Tensor<2>,
    ids: &[u32],
    soft: Option<(Vec<usize>, Tensor<2>)>,
    pad_id: u32,
    opts: &GenOptions,
    chunk: usize,
    device: &Device,
) -> Result<(Vec<u32>, GenStats)> {
    let spec = model.text().spec();
    let max_seq = ids.len() + opts.max_new_tokens + 1;
    let ropes = GenRopes::new(spec, max_seq, DType::F32, device);
    let mut kv = GenKv::new(spec.num_layers);
    let mut stats = GenStats {
        prompt_tokens: ids.len(),
        ..Default::default()
    };
    let mut rng = SplitMix64(0x5EED_1234_ABCD_0001);

    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    {
        burn_rocket::i8_phase_reset();
        burn_rocket::stats_reset();
    }
    let t0 = std::time::Instant::now();
    // NPU prefill: the packed weights are used only inside this call.
    crate::gemma::layers::set_prefill_mode(true);
    let logits = model.prefill(ids, soft, &ropes, &mut kv, lm_head, chunk, pad_id);
    crate::gemma::layers::set_prefill_mode(false);
    if opts.collect_top8 {
        stats.top8.push(top_k(&logits, 8));
    }
    let next = sample(&logits, opts, &mut rng)?;
    stats.prefill_s = t0.elapsed().as_secs_f64();
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    if std::env::var_os("ROCKET_PHASE_LOG").is_some() {
        let st = burn_rocket::stats();
        let p = burn_rocket::i8_phase_stats();
        eprintln!(
            "prefill {:.1}s: in-driver matmul {:.1}s, host convert {:.1}s, calls {}; int8 (worker-summed s): pack {:.1} emit {:.1} submit {:.1} wait {:.1} detile {:.1} fanout {:.1}, jobs {}",
            stats.prefill_s,
            st.npu_s,
            st.convert_s,
            st.calls,
            p.pack_s,
            p.emit_s,
            p.submit_s,
            p.wait_s,
            p.detile_s,
            p.fanout_s,
            p.jobs
        );
    }

    let t1 = std::time::Instant::now();
    let (out, stopped) = decode_loop(
        model,
        lm_head,
        &ropes,
        &mut kv,
        opts,
        next,
        ids,
        chunk,
        device,
        &mut rng,
        &mut stats.top8,
    )?;
    stats.decode_s = t1.elapsed().as_secs_f64();
    stats.generated = out.len();
    stats.stopped = stopped;
    Ok((out, stats))
}

/// Draft length of the prompt-lookup speculation (`ROCKET_SPEC`, 0 = off). Only
/// greedy runs speculate: a sampled run would need rejection sampling, and the
/// per-step top-8 dump wants one logits row per forward.
fn spec_draft_len(opts: &GenOptions) -> usize {
    if !opts.greedy || opts.collect_top8 {
        return 0;
    }
    std::env::var("ROCKET_SPEC")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(3)
        .min(7)
}

/// The tokens that followed the most recent earlier occurrence of the history's
/// trailing 3- (then 2-) gram, at most `k` of them; empty when there is none.
fn draft_lookup(hist: &[u32], k: usize) -> Vec<u32> {
    let len = hist.len();
    for n in [3usize, 2] {
        if len <= n {
            continue;
        }
        let pat = &hist[len - n..];
        for i in (0..len - n).rev() {
            if &hist[i..i + n] == pat {
                let from = i + n;
                let to = (from + k).min(len);
                if to > from {
                    return hist[from..to].to_vec();
                }
            }
        }
    }
    Vec::new()
}

/// The token-by-token decode loop: one cached forward, logits and a sample per
/// generated token. Returns the generated ids and whether an EOS id stopped it.
#[telemetry::instrument(
    name = "decode",
    skip_all,
    fields(
        gen_tokens = tracing::field::Empty,
        tok_s = tracing::field::Empty,
        attn_s = tracing::field::Empty,
        mlp_s = tracing::field::Empty,
        ple_s = tracing::field::Empty,
        norms_s = tracing::field::Empty,
    )
)]
#[allow(clippy::too_many_arguments)]
fn decode_loop(
    model: &GenRoot,
    lm_head: &Tensor<2>,
    ropes: &GenRopes,
    kv: &mut GenKv,
    opts: &GenOptions,
    first: u32,
    context: &[u32],
    chunk: usize,
    device: &Device,
    rng: &mut SplitMix64,
    top8: &mut Vec<Vec<(u32, f32)>>,
) -> Result<(Vec<u32>, bool)> {
    let stages_before = gen_stage_stats();
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    {
        burn_rocket::i8_phase_reset();
        burn_rocket::stats_reset();
    }
    let (mut t_forward, mut t_logits, mut t_sample) = (0.0f64, 0.0f64, 0.0f64);
    let started = std::time::Instant::now();
    let mut out: Vec<u32> = Vec::new();
    let mut stopped = false;
    let mut next = first;
    // Greedy prompt-lookup speculation: draft the continuation of the latest
    // repeated n-gram and verify it in one forward (the NPU pads M to 4 anyway).
    let spec_k = spec_draft_len(opts);
    let mut hist: Vec<u32> = context.to_vec();
    let (mut drafted, mut accepted_n, mut verifies) = (0usize, 0usize, 0usize);
    while out.len() < opts.max_new_tokens {
        out.push(next);
        hist.push(next);
        if opts.eos.contains(&next) {
            stopped = true;
            break;
        }
        if out.len() >= opts.max_new_tokens {
            break;
        }
        let seq_start = kv.len;
        let draft = if spec_k > 0 {
            draft_lookup(&hist, spec_k)
        } else {
            Vec::new()
        };
        if !draft.is_empty() {
            let mut toks = vec![next];
            toks.extend_from_slice(&draft);
            let input = crate::gemma::inputs::make_input(&toks, device);
            let t = std::time::Instant::now();
            let h = model.text().forward(input, ropes, kv, seq_start, chunk);
            t_forward += t.elapsed().as_secs_f64();
            let t = std::time::Instant::now();
            let logits = model.text().logits_rows(h, lm_head);
            let vocab = logits.dims()[1];
            let all: Vec<f32> = logits.cast(DType::F32).into_data().try_to_vec()?;
            t_logits += t.elapsed().as_secs_f64();
            let pred = |r: usize| -> u32 {
                let row = &all[r * vocab..(r + 1) * vocab];
                let mut best = 0usize;
                let mut best_v = f32::NEG_INFINITY;
                for (i, &x) in row.iter().enumerate() {
                    if x > best_v {
                        best_v = x;
                        best = i;
                    }
                }
                best as u32
            };
            let mut a = 0;
            while a < draft.len() && pred(a) == draft[a] {
                a += 1;
            }
            drafted += draft.len();
            accepted_n += a;
            verifies += 1;
            kv.truncate(seq_start + 1 + a);
            // The accepted drafts are real output; the row after them is the next token.
            let mut done = false;
            for &d in &draft[..a] {
                out.push(d);
                hist.push(d);
                if opts.eos.contains(&d) {
                    stopped = true;
                    done = true;
                    break;
                }
                if out.len() >= opts.max_new_tokens {
                    done = true;
                    break;
                }
            }
            if done {
                break;
            }
            next = pred(a);
            continue;
        }
        let input = crate::gemma::inputs::make_input(&[next], device);
        let t = std::time::Instant::now();
        let h = model.text().forward(input, ropes, kv, seq_start, chunk);
        let _ = h.clone().slice(s![.., 0..1, 0..1]).into_data(); // sync the lazy graph
        t_forward += t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        let logits = model.text().logits(h, lm_head);
        let _ = logits.clone().slice(s![0..1, 0..1]).into_data();
        t_logits += t.elapsed().as_secs_f64();
        if opts.collect_top8 {
            top8.push(top_k(&logits, 8));
        }
        let t = std::time::Instant::now();
        next = sample(&logits, opts, rng)?;
        t_sample += t.elapsed().as_secs_f64();
    }
    if verifies > 0 && std::env::var_os("ROCKET_PHASE_LOG").is_some() {
        eprintln!(
            "speculation: {verifies} verify passes, {accepted_n}/{drafted} drafted tokens accepted"
        );
    }
    let decode_s = started.elapsed().as_secs_f64();
    if std::env::var_os("ROCKET_PHASE_LOG").is_some() && !out.is_empty() {
        let n = out.len() as f64;
        eprintln!(
            "decode/token: forward {:.0} ms, logits {:.0} ms, sample {:.0} ms ({} tokens)",
            t_forward / n * 1e3,
            t_logits / n * 1e3,
            t_sample / n * 1e3,
            out.len()
        );
        #[cfg(all(feature = "npu", target_arch = "aarch64"))]
        {
            let st = burn_rocket::stats();
            eprintln!(
                "ext per token: in-driver matmul {:.0} ms, host convert (quantize+epilogue) {:.0} ms, calls {:.0}",
                st.npu_s / n * 1e3,
                st.convert_s / n * 1e3,
                st.calls as f64 / n
            );
            let p = burn_rocket::i8_phase_stats();
            let c = p.calls.max(1) as f64;
            eprintln!(
                "int8 npu calls/token {:.0}, jobs/token {:.0}; per call (ms, worker-summed): \
                 pack {:.3} emit {:.3} submit {:.3} wait {:.3} detile {:.3} | fanout {:.3}",
                p.calls as f64 / n,
                p.jobs as f64 / n,
                p.pack_s / c * 1e3,
                p.emit_s / c * 1e3,
                p.submit_s / c * 1e3,
                p.wait_s / c * 1e3,
                p.detile_s / c * 1e3,
                p.fanout_s / c * 1e3,
            );
        }
    }
    let (attn_s, mlp_s, ple_s, norms_s) = gen_stage_stats();
    let span = tracing::Span::current();
    span.record("gen_tokens", out.len());
    span.record("tok_s", out.len() as f64 / decode_s.max(1e-9));
    span.record("attn_s", attn_s - stages_before.0);
    span.record("mlp_s", mlp_s - stages_before.1);
    span.record("ple_s", ple_s - stages_before.2);
    span.record("norms_s", norms_s - stages_before.3);
    Ok((out, stopped))
}

/// EOS ids from the checkpoint's `generation_config.json` (fallback: 1, 106, 50).
pub(crate) fn gen_eos(model_dir: &Path) -> Vec<u32> {
    let mut eos = vec![1u32, 106, 50];
    if let Ok(text) = std::fs::read_to_string(model_dir.join("generation_config.json"))
        && let Ok(json) = serde_json::from_str::<serde_json::Value>(&text)
    {
        match json.get("eos_token_id") {
            Some(serde_json::Value::Number(n)) => {
                if let Some(v) = n.as_u64() {
                    eos = vec![v as u32];
                }
            }
            Some(serde_json::Value::Array(a)) => {
                eos = a
                    .iter()
                    .filter_map(|v| v.as_u64())
                    .map(|v| v as u32)
                    .collect();
            }
            _ => {}
        }
    }
    eos
}

#[cfg(test)]
mod spec_tests {
    use super::draft_lookup;

    #[test]
    fn lookup_drafts_the_continuation_of_the_latest_repeat() {
        // ... 5 6 7 8 9 ... 5 6 7  -> draft 8 9 ...
        let h = [1, 5, 6, 7, 8, 9, 2, 5, 6, 7];
        assert_eq!(draft_lookup(&h, 3), vec![8, 9, 2]);
        assert_eq!(draft_lookup(&h, 2), vec![8, 9]);
    }

    #[test]
    fn lookup_is_empty_without_a_repeat() {
        assert!(draft_lookup(&[1, 2, 3, 4, 5], 3).is_empty());
        assert!(draft_lookup(&[1, 2], 3).is_empty());
    }
}
