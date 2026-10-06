//! Gemma 4 chat template rendering and the generation loop.

use anyhow::Result;
use burn::prelude::*;
use burn::tensor::DType;
use tokenizers::Tokenizer;

use crate::gen_model::{GenKv, GenRopes, GenRoot};

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

    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
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
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct GenStats {
    pub prefill_s: f64,
    pub decode_s: f64,
    pub prompt_tokens: usize,
    pub generated: usize,
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
    order.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal));
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

/// Prefill + decode loop. `ids` are the rendered-conversation token ids.
pub fn generate(
    model: &GenRoot,
    lm_head: &Tensor<2>,
    ids: &[u32],
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

    let t0 = std::time::Instant::now();
    let input = crate::inputs::make_input(ids, device);
    let h = model.text().forward(input, &ropes, &mut kv, 0, chunk);
    let logits = model.text().logits(h, lm_head);
    let mut next = sample(&logits, opts, &mut rng)?;
    stats.prefill_s = t0.elapsed().as_secs_f64();

    let mut out: Vec<u32> = Vec::new();
    let t1 = std::time::Instant::now();
    for _ in 0..opts.max_new_tokens {
        out.push(next);
        if opts.eos.contains(&next) {
            break;
        }
        let input = crate::inputs::make_input(&[next], device);
        let seq_start = kv.len;
        let h = model.text().forward(input, &ropes, &mut kv, seq_start, chunk);
        let logits = model.text().logits(h, lm_head);
        next = sample(&logits, opts, &mut rng)?;
    }
    stats.decode_s = t1.elapsed().as_secs_f64();
    stats.generated = out.len();
    Ok((out, stats))
}
