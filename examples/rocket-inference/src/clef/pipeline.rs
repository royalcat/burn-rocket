//! `ClefModel`: tokenizer + layer-streamed thinker + `JointSchemaHead`, answering one
//! Jev/SystemOne request (`systemone` in the release's `joint_schema_model.py`).

use super::answer::{probabilities, systemone_answer};
use super::encode::{EncodedRecord, encode_record};
use super::engine::{self, LayerEngine};
use super::head::{ClefHead, LmHeadFile};
use super::model::{Backbone, ForwardOpts};
use super::pack::{Manifest, PackDir};
use super::request::ClefRequest;
use anyhow::{Context, Result};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;

/// The thinker engine actually built (`ROCKET_CLEF_EMU=1` turns any placement into `Emu`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineKind {
    Cpu,
    Emu,
    Npu,
}

/// Where the thinker's heavy math runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClefDevice {
    /// f32 on the CPU (the phase 3-5 path).
    Cpu,
    /// int8 g128 W8A8 on the NPU with streamed experts (a pack, aarch64 `npu` builds).
    Npu,
}

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
        Self::load_placed(dir, threads, ClefDevice::Cpu)
    }

    /// Like [`ClefModel::load_with`] with the thinker placed on `device`.
    /// `ROCKET_CLEF_EMU=1` selects the host emulation of the NPU path on any host
    /// (a test/diagnostic knob; it needs a pack and overrides `device`).
    pub fn load_placed(dir: &Path, threads: usize, device: ClefDevice) -> Result<Self> {
        let kind = if engine::emu_from_env() {
            EngineKind::Emu
        } else if device == ClefDevice::Npu {
            EngineKind::Npu
        } else {
            EngineKind::Cpu
        };
        Self::load_kind(dir, threads, kind)
    }

    /// Load with an explicit thinker engine.
    pub fn load_kind(dir: &Path, threads: usize, kind: EngineKind) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("tokenizer.json: {e}"))?;
        // A `clef-pack` directory (manifest.json) streams int8 experts; otherwise the
        // BF16 checkpoint shards.
        if kind != EngineKind::Cpu && !PackDir::is_pack(dir) {
            anyhow::bail!(
                "the NPU/emulated Clef path needs a clef-pack directory (run `clef-pack` on \
                 the int8 checkpoint, docs/recipes/clef-pack.md); {} has no manifest.json",
                dir.display()
            );
        }
        let (backbone, lm_head) = if PackDir::is_pack(dir) {
            let mut backbone = Backbone::open_pack(dir).context("open the clef pack")?;
            if kind != EngineKind::Cpu {
                let engine = build_engine(dir, &backbone.cfg, threads, kind == EngineKind::Emu)?;
                eprintln!("clef: thinker engine: {}", engine.label());
                backbone.set_engine(engine);
            }
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

    /// The tokenizer (a clone is how the server encodes a request before queueing it).
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Validate and encode one request (the client-error half of [`ClefModel::answer`]).
    pub fn encode(&self, req: &ClefRequest, max_length: usize) -> Result<EncodedRecord> {
        if let Some(refusal) = req.media_refusal() {
            return Err(refusal.into());
        }
        encode_record(&self.tokenizer, req, max_length, None)
    }

    /// Answer several requests with their thinker passes shared (see
    /// [`Backbone::forward_batch`]): results come back in input order, a request that
    /// fails validation/encoding (or the head) fails alone, a failure of a shared pass
    /// fails that pass's members. Each answer is bit-identical to [`ClefModel::answer`]
    /// on the request alone (engine permitting, see the batching notes).
    /// `max_batch_tokens` caps the rows of one shared pass (`0` = no cap).
    pub fn answer_batch(
        &self,
        reqs: &[&ClefRequest],
        max_length: usize,
        max_batch_tokens: usize,
    ) -> BatchAnswers {
        let encoded: Vec<Result<EncodedRecord>> =
            reqs.iter().map(|r| self.encode(r, max_length)).collect();
        self.answer_encoded(reqs, encoded, max_batch_tokens)
    }

    /// [`ClefModel::answer_batch`] for records the caller already validated and
    /// encoded (the server does that before queueing); an `Err` entry fails that
    /// request alone and keeps its error (so an `Invalid` stays a client error).
    pub fn answer_encoded(
        &self,
        reqs: &[&ClefRequest],
        encoded: Vec<Result<EncodedRecord>>,
        max_batch_tokens: usize,
    ) -> BatchAnswers {
        assert_eq!(reqs.len(), encoded.len());
        let mut results: Vec<Option<Result<Answered>>> = (0..reqs.len()).map(|_| None).collect();
        let (mut idx, mut recs) = (Vec::new(), Vec::new());
        for (i, e) in encoded.into_iter().enumerate() {
            match e {
                Ok(r) => {
                    idx.push(i);
                    recs.push(r);
                }
                Err(e) => results[i] = Some(Err(e)),
            }
        }
        let ids: Vec<&[u32]> = recs.iter().map(|r| r.input_ids.as_slice()).collect();
        let mut chunks = Vec::new();
        match self
            .backbone
            .forward_batch(&ids, &self.opts, max_batch_tokens)
        {
            Err(e) => {
                let msg = format!("{e:#}");
                for &i in &idx {
                    results[i] = Some(Err(anyhow::anyhow!("{msg}")));
                }
            }
            Ok(fwd) => {
                chunks = fwd.chunks;
                for ((&i, enc), out) in idx.iter().zip(&recs).zip(fwd.outs) {
                    results[i] = Some(out.and_then(|out| {
                        let spans: Vec<_> = enc.questions.iter().map(|q| q.spans()).collect();
                        let logits = self.pool.install(|| {
                            self.head.forward(
                                &out.final_normed,
                                &enc.input_ids,
                                &spans,
                                &self.lm_head,
                            )
                        })?;
                        assemble(reqs[i], enc, &logits)
                    }));
                }
            }
        }
        BatchAnswers {
            results: results
                .into_iter()
                .map(|r| r.expect("every request has a result"))
                .collect(),
            chunks,
        }
    }
}

/// The results of [`ClefModel::answer_batch`], in request order, and the shared passes.
pub struct BatchAnswers {
    pub results: Vec<Result<Answered>>,
    pub chunks: Vec<super::model::BatchChunk>,
}

impl BatchAnswers {
    /// Expert records read over all passes (0 on the plain CPU path, which has no phases).
    pub fn experts_read(&self) -> usize {
        self.chunks
            .iter()
            .filter_map(|c| c.phases.as_ref())
            .map(|p| p.experts)
            .sum()
    }
}

/// The NPU engine of an aarch64 `npu` build, or the host emulation when `emu`.
fn build_engine(
    dir: &Path,
    cfg: &super::model::TextConfig,
    threads: usize,
    emu: bool,
) -> Result<Box<dyn LayerEngine>> {
    let pack = Arc::new(PackDir::open(dir)?);
    let attn = engine::attn_device_from_env()?;
    if emu {
        return Ok(Box::new(engine::emu_engine(
            pack,
            cfg.clone(),
            threads,
            attn,
        )));
    }
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    {
        Ok(Box::new(super::engine_npu::npu_engine(
            pack,
            cfg.clone(),
            threads,
            attn,
        )?))
    }
    #[cfg(not(all(feature = "npu", target_arch = "aarch64")))]
    {
        let _ = (pack, attn);
        anyhow::bail!("this build has no NPU support (aarch64 + the `npu` feature)")
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
