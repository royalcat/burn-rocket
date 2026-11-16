//! `ClefModel`: tokenizer + layer-streamed thinker + `JointSchemaHead`, answering one
//! Jev/SystemOne request (`systemone` in the release's `joint_schema_model.py`).

use super::answer::{probabilities, systemone_answer};
use super::audio::AudioTower;
use super::autoround_store::ShardStore;
use super::encode::{EncodedRecord, encode_request};
use super::engine::{self, ClefStore, LayerEngine};
use super::head::{ClefHead, LmHeadFile};
use super::media::{MediaFeatures, TowerOutputs, rope_index};
use super::model::{Backbone, ForwardOpts, RecordRef};
use super::pack::{Manifest, PackDir};
use super::request::ClefRequest;
use super::vision::{TowerDevice, VisionTower};
use anyhow::{Context, Result};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
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
    /// The checkpoint directory (the vision tower opens its shards from it, lazily).
    dir: PathBuf,
    tower_device: TowerDevice,
    /// The vision tower, opened by the first request with images.
    vision: Mutex<Option<Arc<VisionTower>>>,
    /// The directory holds `thinker.visual.*` tensors (a clef-pack does not).
    images_supported: bool,
    /// The audio tower, opened by the first request with audio (or a soundtrack in use);
    /// lazy and independent of the vision tower.
    audio: Mutex<Option<Arc<AudioTower>>>,
    /// The directory holds `thinker.audio_tower.*` tensors (a clef-pack does not).
    audio_supported: bool,
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

/// NPU workers of the vision tower's contexts (`ROCKET_CLEF_TOWER_THREADS`, default 3 = the
/// part's cores).
fn tower_threads() -> usize {
    std::env::var("ROCKET_CLEF_TOWER_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n: &usize| (1..=8).contains(&n))
        .unwrap_or(3)
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
        // A `clef-pack` directory (manifest.json) streams repacked int8 experts; the
        // AutoRound checkpoint itself (quantized expert tensors in the shard index) is
        // read as stored; the BF16 checkpoint is the CPU-only oracle path.
        let is_pack = PackDir::is_pack(dir);
        let is_autoround = !is_pack && ShardStore::is_autoround(dir);
        if kind != EngineKind::Cpu && !is_pack && !is_autoround {
            anyhow::bail!(
                "the NPU/emulated Clef path needs the int8 AutoRound checkpoint \
                 (moeshawky/clef-omni-int8-autoround) or a clef-pack directory; {} is neither",
                dir.display()
            );
        }
        let (backbone, lm_head) = if is_pack || is_autoround {
            let mut backbone = if is_pack {
                Backbone::open_pack(dir).context("open the clef pack")?
            } else {
                Backbone::open_autoround(dir).context("open the AutoRound shards")?
            };
            if kind != EngineKind::Cpu {
                let engine = build_engine(dir, &backbone.cfg, threads, kind == EngineKind::Emu)?;
                eprintln!("clef: thinker engine: {}", engine.label());
                backbone.set_engine(engine);
            }
            let lm_head = if is_pack {
                let lm = Manifest::read(dir)?.lm_head;
                LmHeadFile::from_file(&dir.join(&lm.file), &lm.tensor)
                    .context("open the pack's lm_head")?
            } else {
                LmHeadFile::open(dir).context("open thinker.lm_head.weight")?
            };
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
        let tower_device = TowerDevice::resolve(match kind {
            EngineKind::Npu => TowerDevice::Npu,
            EngineKind::Cpu | EngineKind::Emu => TowerDevice::Cpu,
        })?;
        Ok(Self {
            backbone,
            head,
            lm_head,
            tokenizer,
            opts,
            pool,
            dir: dir.to_path_buf(),
            tower_device,
            vision: Mutex::new(None),
            images_supported: VisionTower::available(dir),
            audio: Mutex::new(None),
            audio_supported: AudioTower::available(dir),
        })
    }

    /// Whether this checkpoint directory can serve `images` and `videos` (it holds the vision
    /// tower).
    pub fn images_supported(&self) -> bool {
        self.images_supported
    }

    /// Whether this checkpoint directory can serve `audio` and soundtracks (it holds the audio
    /// tower).
    pub fn audio_supported(&self) -> bool {
        self.audio_supported
    }

    /// The audio tower, opened on first use (shards indexed, and on the NPU every linear
    /// packed as a resident fp16 weight: ~1.2 GiB of BOs).
    fn audio_tower(&self) -> Result<Arc<AudioTower>> {
        let mut slot = self.audio.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(t) = &*slot {
            return Ok(t.clone());
        }
        anyhow::ensure!(
            self.audio_supported,
            "{} holds no audio tower (a clef-pack skips `thinker.audio_tower.*`): serve the BF16 \
             checkpoint or the AutoRound shards to answer audio requests",
            self.dir.display()
        );
        let t0 = std::time::Instant::now();
        let tower = AudioTower::open(&self.dir, self.tower_device, tower_threads())
            .context("open the audio tower")?;
        eprintln!(
            "clef: audio tower opened in {:.1}s ({})",
            t0.elapsed().as_secs_f64(),
            tower.label()
        );
        let tower = Arc::new(tower);
        *slot = Some(tower.clone());
        Ok(tower)
    }

    /// The vision tower, opened on first use (shards indexed, and on the NPU every linear
    /// packed as a resident fp16 weight: ~1 GiB of BOs).
    fn vision_tower(&self) -> Result<Arc<VisionTower>> {
        let mut slot = self.vision.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(t) = &*slot {
            return Ok(t.clone());
        }
        anyhow::ensure!(
            self.images_supported,
            "{} holds no vision tower (a clef-pack skips `thinker.visual.*`): serve the BF16 \
             checkpoint or the AutoRound shards to answer image requests",
            self.dir.display()
        );
        let t0 = std::time::Instant::now();
        let tower = VisionTower::open(&self.dir, self.tower_device, tower_threads())
            .context("open the vision tower")?;
        eprintln!(
            "clef: vision tower opened in {:.1}s ({})",
            t0.elapsed().as_secs_f64(),
            tower.label()
        );
        let tower = Arc::new(tower);
        *slot = Some(tower.clone());
        Ok(tower)
    }

    /// Run the vision tower over a record's images and line the rows up with its placeholders
    /// (`None` for a text-only record). A device failure unwinds as the typed panic the server
    /// maps; anything else fails this record alone.
    fn media_features(&self, enc: &EncodedRecord) -> Result<Option<MediaFeatures>> {
        let Some(media) = &enc.media else {
            return Ok(None);
        };
        let check =
            std::env::var("ROCKET_CLEF_TOWER_CHECK").is_ok_and(|v| !v.is_empty() && v != "0");
        let mut out = TowerOutputs::default();
        if !media.images.is_empty() || !media.videos.is_empty() {
            let tower = self.vision_tower()?;
            let t0 = std::time::Instant::now();
            out = self
                .pool
                .install(|| tower.encode_media(&media.images, &media.videos, self.opts.threads))
                .inspect_err(engine::raise_if_device_failure)
                .context("vision tower")?;
            let patches: usize = media.images.iter().map(|i| i.patches()).sum::<usize>()
                + media.videos.iter().map(|v| v.patches()).sum::<usize>();
            eprintln!(
                "clef: vision tower: {} image(s), {} video(s), {patches} patches -> {} tokens in {:.1}s",
                media.images.len(),
                media.videos.len(),
                patches / 4,
                t0.elapsed().as_secs_f64()
            );
            if tower.device() != TowerDevice::Cpu && check {
                for (i, img) in media.images.iter().enumerate() {
                    match self
                        .pool
                        .install(|| tower.check_against_cpu(img, self.opts.threads))
                    {
                        Ok(report) => {
                            eprintln!("clef: tower check images[{i}] vs f32 CPU: {report}")
                        }
                        Err(e) => eprintln!("clef: tower check images[{i}] failed: {e:#}"),
                    }
                }
                for (i, v) in media.videos.iter().enumerate() {
                    match self
                        .pool
                        .install(|| tower.check_pixels(&v.pixels, v.grid, self.opts.threads))
                    {
                        Ok(report) => {
                            eprintln!("clef: tower check videos[{i}] vs f32 CPU: {report}")
                        }
                        Err(e) => eprintln!("clef: tower check videos[{i}] failed: {e:#}"),
                    }
                }
            }
            tower.trim(crate::util::mem::keep_scratch_mib() * 1024 * 1024);
        }
        if let Some(mel) = &media.audio {
            let tower = self.audio_tower()?;
            let t0 = std::time::Instant::now();
            let a = self
                .pool
                .install(|| tower.forward(mel, self.opts.threads, &[]))
                .inspect_err(engine::raise_if_device_failure)
                .context("audio tower")?;
            eprintln!(
                "clef: audio tower: {} clip(s), {} mel frames -> {} tokens in {:.1}s",
                mel.clips(),
                mel.valid.iter().sum::<usize>(),
                a.features.len() / self.backbone.cfg.hidden,
                t0.elapsed().as_secs_f64()
            );
            out.audio = a.features;
            if tower.device() != TowerDevice::Cpu && check {
                match self
                    .pool
                    .install(|| tower.check_against_cpu(mel, self.opts.threads))
                {
                    Ok(report) => eprintln!("clef: tower check audio vs f32 CPU: {report}"),
                    Err(e) => eprintln!("clef: tower check audio failed: {e:#}"),
                }
            }
            tower.trim(crate::util::mem::keep_scratch_mib() * 1024 * 1024);
        }
        let positions = rope_index(&enc.input_ids, &media.layout)?;
        MediaFeatures::assemble(
            &enc.input_ids,
            Some(positions),
            &out,
            self.backbone.cfg.hidden,
        )
        .map(Some)
    }

    /// Answer one request. `max_length` caps the whole record in tokens (the release's
    /// default is [`DEFAULT_MAX_LENGTH`]); images count toward it, audio/video are refused. A
    /// client error is an [`Invalid`](super::request::Invalid) in the error chain.
    pub fn answer(&self, req: &ClefRequest, max_length: usize) -> Result<Answered> {
        let encoded = encode_request(&self.tokenizer, req, max_length)?;
        let ids = &encoded.input_ids;
        let features = self.media_features(&encoded)?;
        let out = self
            .backbone
            .forward_media(ids, features.as_ref(), &self.opts)?;
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
        encode_request(&self.tokenizer, req, max_length)
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
        // Tower runs first (one record's failure fails that record alone); records whose
        // tower failed leave the shared thinker pass.
        let mut features: Vec<Option<MediaFeatures>> = Vec::with_capacity(recs.len());
        let (mut idx2, mut recs2) = (Vec::new(), Vec::new());
        for (&i, r) in idx.iter().zip(recs) {
            match self.media_features(&r) {
                Ok(f) => {
                    features.push(f);
                    idx2.push(i);
                    recs2.push(r);
                }
                Err(e) => results[i] = Some(Err(e)),
            }
        }
        let (idx, recs) = (idx2, recs2);
        let refs: Vec<RecordRef<'_>> = recs
            .iter()
            .zip(&features)
            .map(|(r, f)| RecordRef {
                ids: &r.input_ids,
                media: f.as_ref(),
            })
            .collect();
        let mut chunks = Vec::new();
        match self
            .backbone
            .forward_batch_media(&refs, &self.opts, max_batch_tokens)
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
    let pack: Arc<dyn ClefStore> = if PackDir::is_pack(dir) {
        Arc::new(PackDir::open(dir)?)
    } else {
        Arc::new(ShardStore::open(dir)?)
    };
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
