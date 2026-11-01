//! The loaded d3 model: one noncausal backbone pass per question, then the
//! 255-code readout.
//!
//! `system_one` reproduces `d3_runtime.D3.system_one` for the text path: the
//! questions run in request order, each in its own forward pass, and a question
//! that cannot be answered (invalid shape, over the length cap) is reported in
//! `answers` while the rest of the request still runs.

use anyhow::{Result, anyhow};
use burn::prelude::*;
use serde_json::{Map, Value};
use tokenizers::Tokenizer;

use super::config::{DecisionConfig, MAX_OPTIONS};
use super::format::{self, Question};
use super::request::DecisionRequest;
use crate::qwen35_intent::model::IntentModel;

/// A loaded decision checkpoint, ready to answer System One requests.
#[derive(Debug)]
pub struct DecisionModel {
    pub backbone: IntentModel,
    /// `[255, hidden]` f32 readout rows, one per answer code.
    pub readout: Tensor<2>,
    pub config: DecisionConfig,
    pub tokenizer: Tokenizer,
}

impl DecisionModel {
    /// The per-question prompt-token cap for a request (`--max-state-tokens` /
    /// the server's `--max-tokens`), narrowed by the checkpoint's own
    /// `max_length` when it declares one.
    pub fn prompt_cap(&self, max_tokens: usize) -> usize {
        match self.config.max_length {
            Some(limit) => max_tokens.min(limit),
            None => max_tokens,
        }
    }

    /// Answer one request. Returns `(response, input_tokens, answered_questions)`;
    /// only envelope problems (empty/malformed `questions`, media) are errors.
    pub fn system_one(
        &mut self,
        model: &str,
        req: DecisionRequest,
        max_tokens: usize,
    ) -> Result<(Value, usize, usize)> {
        let questions = req.questions()?;
        let cap = self.prompt_cap(max_tokens);
        let mut answers = Map::new();
        let mut tokens = 0usize;
        let mut answered = 0usize;
        for (id, value) in questions {
            let raw_kind = value.get("type").cloned().unwrap_or(Value::Null);
            let question = match format::normalize_question(value) {
                Ok(question) => question,
                Err(message) => {
                    answers.insert(
                        id.clone(),
                        error_answer(raw_kind, "invalid_question", message),
                    );
                    continue;
                }
            };
            let user = match format::user_prompt(&req.state, &question, &self.config.codes) {
                Ok(user) => user,
                Err(message) => {
                    answers.insert(
                        id.clone(),
                        error_answer(
                            Value::String(question.kind.as_str().to_string()),
                            "invalid_question",
                            message,
                        ),
                    );
                    continue;
                }
            };
            let prompt = format::chatml(&user);
            let encoding = self
                .tokenizer
                .encode(prompt.as_str(), false)
                .map_err(|e| anyhow!("tokenize the '{id}' prompt: {e}"))?;
            let ids = encoding.get_ids();
            if ids.len() > cap {
                answers.insert(
                    id.clone(),
                    error_answer(
                        Value::String(question.kind.as_str().to_string()),
                        "max_length_exceeded",
                        format!(
                            "the question prompt has {} tokens, over the maximum context length \
                             of {cap} tokens; nothing was truncated",
                            ids.len()
                        ),
                    ),
                );
                continue;
            }
            let probabilities = self.probabilities(&question, ids)?;
            let answer = match format::product_answer(&question, &probabilities) {
                Ok(answer) => answer,
                Err(message) => {
                    answers.insert(
                        id.clone(),
                        error_answer(
                            Value::String(question.kind.as_str().to_string()),
                            "invalid_model_output",
                            message,
                        ),
                    );
                    continue;
                }
            };
            answers.insert(id.clone(), answer);
            tokens += ids.len();
            answered += 1;
        }
        let response = serde_json::json!({
            "model": model,
            "answers": Value::Object(answers),
            "usage": {"input_tokens": tokens, "output_tokens": 0},
        });
        Ok((response, tokens, answered))
    }

    /// One forward pass: the final-normed last hidden through the readout, the
    /// invalid codes masked to `-inf`, the temperature and the softmax (the
    /// reference's `logits` + `probabilities`).
    fn probabilities(&mut self, question: &Question, ids: &[u32]) -> Result<Vec<f64>> {
        self.backbone.cache.reset();
        let hidden = self.backbone.forward_last_normed(ids);
        let logits = hidden.matmul(self.readout.clone().transpose());
        let values: Vec<f32> = logits
            .to_data()
            .try_to_vec()
            .map_err(|e| anyhow!("read the readout logits: {e}"))?;
        if values.len() < MAX_OPTIONS {
            return Err(anyhow!(
                "the readout returned {} logits, expected {MAX_OPTIONS}",
                values.len()
            ));
        }
        Ok(code_probabilities(
            &values,
            question.keys.len(),
            self.config.temperature,
        ))
    }
}

/// The softmax over the first `n` codes of one masked readout row.
///
/// The reference divides the f32 logits by the temperature, softmaxes (with the
/// row max subtracted, over the valid codes only) and reads the first `n`
/// entries back; invalid codes are `-inf` and contribute nothing.
pub fn code_probabilities(logits: &[f32], n: usize, temperature: f64) -> Vec<f64> {
    let n = n.min(logits.len());
    let mut max = f32::NEG_INFINITY;
    for &value in &logits[..n] {
        if value > max {
            max = value;
        }
    }
    let mut exps = Vec::with_capacity(n);
    let mut sum = 0f32;
    for &value in &logits[..n] {
        let scaled = ((value - max) as f64 / temperature).exp() as f32;
        exps.push(scaled);
        sum += scaled;
    }
    if sum <= 0.0 || sum.is_nan() {
        // Every code was masked (no valid option); the caller reports the answer
        // as invalid output rather than dividing by zero.
        return vec![f32::NAN as f64; n];
    }
    exps.iter().map(|value| (*value / sum) as f64).collect()
}

/// One per-question error record (`{"type", "error", "message"}`).
fn error_answer(kind: Value, error: &'static str, message: String) -> Value {
    serde_json::json!({"type": kind, "error": error, "message": message})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_readout_softmax_masks_later_codes() {
        let mut logits = vec![f32::NEG_INFINITY; MAX_OPTIONS];
        logits[0] = 2.0;
        logits[1] = 1.0;
        logits[2] = -1.0;
        let probabilities = code_probabilities(&logits, 3, 1.0);
        assert_eq!(probabilities.len(), 3);
        let total: f64 = probabilities.iter().sum();
        assert!((total - 1.0).abs() < 1e-6, "{probabilities:?}");
        assert!(probabilities[0] > probabilities[1]);
        assert!(probabilities[1] > probabilities[2]);
        // The same row cut to two options renormalizes over them.
        let two = code_probabilities(&logits, 2, 1.0);
        assert!((two.iter().sum::<f64>() - 1.0).abs() < 1e-6);
        assert!((two[0] - probabilities[0] / (probabilities[0] + probabilities[1])).abs() < 1e-6);
    }

    #[test]
    fn the_temperature_flattens_the_distribution() {
        let mut logits = vec![f32::NEG_INFINITY; MAX_OPTIONS];
        logits[0] = 2.0;
        logits[1] = 0.0;
        let hot = code_probabilities(&logits, 2, 1.0);
        let warm = code_probabilities(&logits, 2, 4.0);
        assert!(warm[0] < hot[0], "{hot:?} vs {warm:?}");
        assert!(warm[1] > hot[1]);
        assert!((warm.iter().sum::<f64>() - 1.0).abs() < 1e-6);
    }
}
