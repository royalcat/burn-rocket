//! `ClefModel`: tokenizer + layer-streamed thinker + `JointSchemaHead`, answering one
//! Jev/SystemOne request (`systemone` in the release's `joint_schema_model.py`).

use super::answer::{probabilities, systemone_answer};
use super::encode::{EncodedRecord, encode_record};
use super::head::{ClefHead, LmHeadFile};
use super::model::{Backbone, ForwardOpts};
use super::pack::{Manifest, PackDir};
use super::request::ClefRequest;
use anyhow::{Context, Result};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;
use std::path::Path;
use tokenizers::Tokenizer;

pub struct ClefModel {
    pub backbone: Backbone,
    pub head: ClefHead,
    pub lm_head: LmHeadFile,
    tokenizer: Tokenizer,
    opts: ForwardOpts,
    pool: rayon::ThreadPool,
}

impl std::fmt::Debug for ClefModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClefModel")
            .field("layers", &self.backbone.cfg.layers)
            .field("threads", &self.opts.threads)
            .finish_non_exhaustive()
    }
}

/// One answered request.
#[derive(Debug)]
pub struct Answered {
    /// `{"model", "answers", "usage"}`.
    pub response: Value,
    /// `usage.input_tokens`.
    pub tokens: usize,
    pub questions: usize,
    /// Per question (request order): id and the softmax probability of each option.
    pub probabilities: Vec<(String, BTreeMap<String, f64>)>,
}

/// Worker threads of the private pools: the machine's cores, at most 8.
pub fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .clamp(1, 8)
}

impl ClefModel {
    /// Open a Clef-Omni checkpoint directory (`config.json`, the 15 BF16 shards,
    /// `joint_head.safetensors`, `tokenizer.json`). The thinker's weights stay on disk
    /// and stream per layer; only the head (~436 MB f32) is resident.
    pub fn load(dir: &Path) -> Result<Self> {
        Self::load_with(dir, default_threads())
    }

    pub fn load_with(dir: &Path, threads: usize) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("tokenizer.json: {e}"))?;
        // A `clef-pack` directory (manifest.json) streams int8 experts; otherwise the
        // BF16 checkpoint shards.
        let (backbone, lm_head) = if PackDir::is_pack(dir) {
            let backbone = Backbone::open_pack(dir).context("open the clef pack")?;
            let lm = Manifest::read(dir)?.lm_head;
            let lm_head = LmHeadFile::from_file(&dir.join(&lm.file), &lm.tensor)
                .context("open the pack's lm_head")?;
            (backbone, lm_head)
        } else {
            (
                Backbone::open(dir).context("open the thinker shards")?,
                LmHeadFile::open(dir).context("open thinker.lm_head.weight")?,
            )
        };
        let head = ClefHead::load(dir).context("load joint_head.safetensors")?;
        let opts = ForwardOpts {
            threads: threads.max(1),
            ..ForwardOpts::default()
        };
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(opts.threads)
            .build()?;
        Ok(Self {
            backbone,
            head,
            lm_head,
            tokenizer,
            opts,
            pool,
        })
    }

    /// Answer one request. `max_length` caps the whole record in tokens (the release's
    /// default is [`DEFAULT_MAX_LENGTH`]); a text-only build refuses media. A client
    /// error is an [`Invalid`](super::request::Invalid) in the error chain.
    pub fn answer(&self, req: &ClefRequest, max_length: usize) -> Result<Answered> {
        if let Some(refusal) = req.media_refusal() {
            return Err(refusal.into());
        }
        let encoded = encode_record(&self.tokenizer, req, max_length, None)?;
        let ids = &encoded.input_ids;
        let out = self.backbone.forward(ids, &self.opts)?;
        let spans: Vec<_> = encoded.questions.iter().map(|q| q.spans()).collect();
        let logits = self.pool.install(|| {
            self.head
                .forward(&out.final_normed, ids, &spans, &self.lm_head)
        })?;
        assemble(req, &encoded, &logits)
    }
}

/// softmax per question -> `systemone_answer` -> `{"model", "answers", "usage"}`
/// (answers keyed by question id in request order).
pub fn assemble(
    req: &ClefRequest,
    encoded: &EncodedRecord,
    logits: &[Vec<f32>],
) -> Result<Answered> {
    let mut answers = Map::new();
    let mut all = Vec::new();
    for (q, l) in encoded.questions.iter().zip(logits) {
        let probs = probabilities(&q.option_ids, l);
        let question = req
            .questions
            .get(&q.id)
            .with_context(|| format!("no question `{}` in the request", q.id))?;
        answers.insert(q.id.clone(), systemone_answer(question, &probs)?);
        all.push((q.id.clone(), probs));
    }
    let tokens = encoded.input_ids.len();
    Ok(Answered {
        response: json!({
            "model": req.model,
            "answers": answers,
            "usage": {"input_tokens": tokens, "output_tokens": 0},
        }),
        tokens,
        questions: encoded.questions.len(),
        probabilities: all,
    })
}

#[cfg(test)]
mod tests;
