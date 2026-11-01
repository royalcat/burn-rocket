//! The loaded d3 model: one noncausal backbone pass per question, then the
//! 255-code readout.
//!
//! The flow mirrors `d3_runtime.D3`: [`DecisionModel::prepare`] validates and
//! tokenizes every question (`invalid_question` / `max_length_exceeded` land in
//! the per-question errors, nothing runs yet), [`DecisionModel::run`] forwards
//! the runnable ones and [`DecisionModel::respond`] builds the official answer
//! object. The HTTP handler refuses a request whose `prepare` reported an
//! `invalid_question` (422, nothing runs); the CLI keeps the runtime's per-question
//! errors in a 200-shaped answer.

use std::collections::HashMap;

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

/// One validated request: the questions that will run and the errors of those
/// that will not (`d3_runtime.Prepared`).
#[derive(Debug)]
pub struct Prepared {
    /// Question ids in request order.
    pub keys: Vec<String>,
    /// The validated question of every entry that got past normalization.
    pub questions: HashMap<String, Question>,
    /// Question id -> its answer error record (`invalid_question` or
    /// `max_length_exceeded`).
    pub errors: Map<String, Value>,
    /// `(question id, prompt token ids)` of the runnable questions, request order.
    pub sequences: Vec<(String, Vec<u32>)>,
}

impl Prepared {
    /// Question id -> message of every question that failed to normalize, in
    /// request order. The HTTP server refuses the request (422) when this is not
    /// empty; `max_length_exceeded` is not part of it (that stays a 200 answer).
    pub fn invalid_questions(&self) -> Map<String, Value> {
        let mut invalid = Map::new();
        for key in &self.keys {
            if let Some(error) = self.errors.get(key)
                && error.get("error").and_then(Value::as_str) == Some("invalid_question")
                && let Some(message) = error.get("message")
            {
                invalid.insert(key.clone(), message.clone());
            }
        }
        invalid
    }
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

    /// Validate, render and tokenize every question of one request.
    ///
    /// Envelope problems are already refused by [`DecisionRequest::from_value`];
    /// a question that cannot be answered ends up in [`Prepared::errors`] and the
    /// rest still runs.
    pub fn prepare(&self, req: &DecisionRequest, max_tokens: usize) -> Result<Prepared> {
        let cap = self.prompt_cap(max_tokens);
        let mut prepared = Prepared {
            keys: req.questions.keys().cloned().collect(),
            questions: HashMap::new(),
            errors: Map::new(),
            sequences: Vec::new(),
        };
        for (id, value) in &req.questions {
            let raw_kind = value.get("type").cloned().unwrap_or(Value::Null);
            let question = match format::normalize_question(value) {
                Ok(question) => question,
                Err(message) => {
                    prepared.errors.insert(
                        id.clone(),
                        error_answer(raw_kind, "invalid_question", message),
                    );
                    continue;
                }
            };
            let user = match format::user_prompt(&req.state, &question, &self.config.codes) {
                Ok(user) => user,
                Err(message) => {
                    prepared.errors.insert(
                        id.clone(),
                        error_answer(
                            Value::String(question.kind.as_str().to_string()),
                            "invalid_question",
                            message,
                        ),
                    );
                    prepared.questions.insert(id.clone(), question);
                    continue;
                }
            };
            let prompt = format::chatml(&user);
            let encoding = self
                .tokenizer
                .encode(prompt.as_str(), false)
                .map_err(|e| anyhow!("tokenize the '{id}' prompt: {e}"))?;
            let ids = encoding.get_ids();
            prepared.questions.insert(id.clone(), question);
            if ids.len() > cap {
                prepared.errors.insert(
                    id.clone(),
                    error_answer(
                        Value::String(
                            prepared
                                .questions
                                .get(id)
                                .expect("just inserted")
                                .kind
                                .as_str()
                                .to_string(),
                        ),
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
            prepared.sequences.push((id.clone(), ids.to_vec()));
        }
        Ok(prepared)
    }

    /// Forward every runnable question (request order) and return its option
    /// probabilities plus the answered questions' prompt-token sum.
    pub fn run(&mut self, prepared: &Prepared) -> Result<(HashMap<String, Vec<f64>>, usize)> {
        let mut probabilities = HashMap::with_capacity(prepared.sequences.len());
        let mut tokens = 0usize;
        for (id, ids) in &prepared.sequences {
            let question = prepared
                .questions
                .get(id)
                .ok_or_else(|| anyhow!("internal error: '{id}' has no validated question"))?;
            probabilities.insert(id.clone(), self.probabilities(question, ids)?);
            tokens += ids.len();
        }
        Ok((probabilities, tokens))
    }

    /// The official System One response for one prepared request
    /// (`d3_runtime.D3.respond`): every question id in request order, either its
    /// answer or its error record.
    pub fn respond(
        &self,
        model: &str,
        prepared: &Prepared,
        probabilities: &HashMap<String, Vec<f64>>,
        tokens: usize,
    ) -> Value {
        let mut answers = Map::new();
        for id in &prepared.keys {
            if let Some(error) = prepared.errors.get(id) {
                answers.insert(id.clone(), error.clone());
                continue;
            }
            let Some(question) = prepared.questions.get(id) else {
                continue;
            };
            let Some(values) = probabilities.get(id) else {
                continue;
            };
            let answer = match format::product_answer(question, values) {
                Ok(answer) => answer,
                Err(message) => error_answer(
                    Value::String(question.kind.as_str().to_string()),
                    "invalid_model_output",
                    message,
                ),
            };
            answers.insert(id.clone(), answer);
        }
        serde_json::json!({
            "model": model,
            "answers": Value::Object(answers),
            "usage": {"input_tokens": tokens, "output_tokens": 0},
        })
    }

    /// Answer one request with the runtime's own semantics (per-question errors
    /// stay in the answer). Returns `(response, input_tokens, answered_questions)`;
    /// only envelope problems are errors.
    pub fn system_one(
        &mut self,
        model: &str,
        req: DecisionRequest,
        max_tokens: usize,
    ) -> Result<(Value, usize, usize)> {
        let prepared = self.prepare(&req, max_tokens)?;
        let (probabilities, tokens) = self.run(&prepared)?;
        let answered = prepared.sequences.len();
        Ok((
            self.respond(model, &prepared, &probabilities, tokens),
            tokens,
            answered,
        ))
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

    fn prepared_with(errors: Vec<(&str, &'static str)>) -> Prepared {
        let mut prepared = Prepared {
            keys: errors.iter().map(|(id, _)| id.to_string()).collect(),
            questions: HashMap::new(),
            errors: Map::new(),
            sequences: Vec::new(),
        };
        for (id, error) in errors {
            prepared.errors.insert(
                id.to_string(),
                error_answer(
                    Value::String("choice".to_string()),
                    error,
                    format!("{error} for {id}"),
                ),
            );
        }
        prepared
    }

    #[test]
    fn only_invalid_questions_refuse_a_request() {
        let prepared = prepared_with(vec![
            ("a", "invalid_question"),
            ("b", "max_length_exceeded"),
        ]);
        let invalid = prepared.invalid_questions();
        assert_eq!(invalid.len(), 1);
        assert_eq!(invalid["a"], "invalid_question for a");
        assert!(
            prepared.sequences.is_empty(),
            "nothing runs for a refused question"
        );
        assert!(prepared_with(vec![]).invalid_questions().is_empty());
    }
}
