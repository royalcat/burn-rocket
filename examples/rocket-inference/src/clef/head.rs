//! `JointSchemaHead` (Clef-Omni): the typed-decision head over the thinker's final
//! normed hidden states. Reference: `tools/clef_joint_schema_model.py`
//! (`EvidenceRoutingLayer`, `JointSchemaHead`). Device-free f32 on the CPU; the
//! BF16 checkpoint is widened once at load (~109M params, ~436 MB).

use super::math::{
    Mha, cosine, dot, gelu_in_place, layer_norm, linear, normalize, sigmoid, softmax_row,
};
use super::st::{SafeFile, TensorInfo};
use anyhow::{Context, Result, bail};
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::BTreeSet;
use std::path::Path;

const LN_EPS: f32 = 1e-5;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeadConfig {
    pub hidden_size: usize,
    pub width: usize,
    pub routing_layers: usize,
    pub layers: usize,
    pub heads: usize,
    pub feedforward: usize,
}

/// Token/option layout of one question inside the encoded record (half-open token
/// index ranges, as in the oracle's `.json`).
#[derive(Debug, Clone)]
pub struct QuestionSpans {
    /// 0 noul, 1 choice, 2 score.
    pub question_type: usize,
    pub question_span: (usize, usize),
    pub option_spans: Vec<(usize, usize)>,
}

/// Source of the output-embedding (`lm_head`) rows used as lexical option vectors.
pub trait LmHeadRows {
    /// Write row `id` (length `hidden_size`) into `out`.
    fn row(&self, id: u32, out: &mut [f32]) -> Result<()>;
}

struct Norm {
    w: Vec<f32>,
    b: Vec<f32>,
}

impl Norm {
    fn apply(&self, x: &[f32], rows: usize) -> Vec<f32> {
        layer_norm(x, rows, self.w.len(), &self.w, &self.b, LN_EPS)
    }
}

struct Lin {
    w: Vec<f32>,
    b: Option<Vec<f32>>,
    out: usize,
    inp: usize,
}

impl Lin {
    fn apply(&self, x: &[f32], rows: usize) -> Vec<f32> {
        linear(x, rows, self.inp, &self.w, self.out, self.b.as_deref())
    }
}

struct Evidence {
    query_norm: Norm,
    memory_norm: Norm,
    attention: Mha,
    ff_norm: Norm,
    ff1: Lin,
    ff2: Lin,
}

struct DecoderLayer {
    norm1: Norm,
    norm2: Norm,
    norm3: Norm,
    self_attn: Mha,
    cross_attn: Mha,
    linear1: Lin,
    linear2: Lin,
}

pub struct ClefHead {
    pub config: HeadConfig,
    hidden_norm: Norm,
    memory_projection: Lin,
    question_projection: Lin,
    option_question_projection: Lin,
    global_projection: Lin,
    option_context_projection: Lin,
    option_lexical_projection: Lin,
    type_embedding: Vec<f32>,
    evidence: Vec<Evidence>,
    option_summary_norm: Norm,
    layers: Vec<DecoderLayer>,
    field_norm: Norm,
    option_norm: Norm,
    scorer1: Lin,
    scorer2: Lin,
    prior_logit_scale: f32,
    joint_logit_scale: f32,
    residual_gate: f32,
}

/// Strict tensor taker: every key must be taken with the right shape and none may be
/// left over (like `load_state_dict(strict=True)`).
struct Take {
    file: SafeFile,
    left: BTreeSet<String>,
}

impl Take {
    fn info(&mut self, name: &str, shape: &[usize]) -> Result<TensorInfo> {
        let info = self
            .file
            .tensors
            .get(name)
            .with_context(|| format!("joint head: missing tensor `{name}`"))?
            .clone();
        if info.shape != shape {
            bail!(
                "joint head: `{name}` has shape {:?}, expected {shape:?}",
                info.shape
            );
        }
        self.left.remove(name);
        Ok(info)
    }

    fn vec(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let info = self.info(name, shape)?;
        Ok(self.file.to_f32(&info))
    }

    fn scalar(&mut self, name: &str) -> Result<f32> {
        Ok(self.vec(name, &[])?[0])
    }

    fn norm(&mut self, prefix: &str, n: usize) -> Result<Norm> {
        Ok(Norm {
            w: self.vec(&format!("{prefix}.weight"), &[n])?,
            b: self.vec(&format!("{prefix}.bias"), &[n])?,
        })
    }

    fn lin(&mut self, prefix: &str, inp: usize, out: usize, bias: bool) -> Result<Lin> {
        Ok(Lin {
            w: self.vec(&format!("{prefix}.weight"), &[out, inp])?,
            b: if bias {
                Some(self.vec(&format!("{prefix}.bias"), &[out])?)
            } else {
                None
            },
            out,
            inp,
        })
    }

    fn mha(&mut self, prefix: &str, d: usize, heads: usize) -> Result<Mha> {
        Ok(Mha {
            in_w: self.vec(&format!("{prefix}.in_proj_weight"), &[3 * d, d])?,
            in_b: self.vec(&format!("{prefix}.in_proj_bias"), &[3 * d])?,
            out_w: self.vec(&format!("{prefix}.out_proj.weight"), &[d, d])?,
            out_b: self.vec(&format!("{prefix}.out_proj.bias"), &[d])?,
            heads,
        })
    }
}

fn add_in_place(a: &mut [f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    a.par_iter_mut()
        .zip(b.par_iter())
        .for_each(|(x, y)| *x += y);
}

fn mean_span(values: &[f32], d: usize, (s, e): (usize, usize), what: &str) -> Result<Vec<f32>> {
    let rows = values.len() / d;
    if s >= e || e > rows {
        bail!("{what}: span ({s}, {e}) is empty or outside the {rows}-token sequence");
    }
    let mut out = vec![0f32; d];
    for r in s..e {
        for (o, v) in out.iter_mut().zip(&values[r * d..(r + 1) * d]) {
            *o += v;
        }
    }
    let n = (e - s) as f32;
    out.iter_mut().for_each(|o| *o /= n);
    Ok(out)
}

impl ClefHead {
    /// Load `joint_head_config.json` + `joint_head.safetensors` from a checkpoint dir.
    pub fn load(dir: &Path) -> Result<Self> {
        let cfg_path = dir.join("joint_head_config.json");
        let config: HeadConfig = serde_json::from_slice(
            &std::fs::read(&cfg_path).with_context(|| format!("read {}", cfg_path.display()))?,
        )
        .with_context(|| format!("parse {}", cfg_path.display()))?;
        let file = SafeFile::open(&dir.join("joint_head.safetensors"))?;
        Self::from_file(config, file)
    }

    fn from_file(config: HeadConfig, file: SafeFile) -> Result<Self> {
        let (h, w, ff, heads) = (
            config.hidden_size,
            config.width,
            config.feedforward,
            config.heads,
        );
        if heads == 0 || !w.is_multiple_of(heads) {
            bail!("joint head: width {w} is not divisible by {heads} heads");
        }
        let left = file.tensors.keys().cloned().collect();
        let mut t = Take { file, left };
        let hidden_norm = t.norm("hidden_norm", h)?;
        let memory_projection = t.lin("memory_projection", h, w, false)?;
        let question_projection = t.lin("question_projection", h, w, false)?;
        let option_question_projection = t.lin("option_question_projection", h, w, false)?;
        let global_projection = t.lin("global_projection", h, w, false)?;
        let option_context_projection = t.lin("option_context_projection", h, w, false)?;
        let option_lexical_projection = t.lin("option_lexical_projection", h, w, false)?;
        let type_embedding = t.vec("type_embedding.weight", &[3, w])?;
        let mut evidence = Vec::new();
        for i in 0..config.routing_layers {
            let p = format!("evidence_layers.{i}");
            evidence.push(Evidence {
                query_norm: t.norm(&format!("{p}.query_norm"), w)?,
                memory_norm: t.norm(&format!("{p}.memory_norm"), w)?,
                attention: t.mha(&format!("{p}.attention"), w, heads)?,
                ff_norm: t.norm(&format!("{p}.feedforward_norm"), w)?,
                ff1: t.lin(&format!("{p}.feedforward.0"), w, ff, true)?,
                ff2: t.lin(&format!("{p}.feedforward.3"), ff, w, true)?,
            });
        }
        let option_summary_norm = t.norm("option_summary_norm", w)?;
        let mut layers = Vec::new();
        for i in 0..config.layers {
            let p = format!("layers.{i}");
            layers.push(DecoderLayer {
                norm1: t.norm(&format!("{p}.norm1"), w)?,
                norm2: t.norm(&format!("{p}.norm2"), w)?,
                norm3: t.norm(&format!("{p}.norm3"), w)?,
                self_attn: t.mha(&format!("{p}.self_attn"), w, heads)?,
                cross_attn: t.mha(&format!("{p}.multihead_attn"), w, heads)?,
                linear1: t.lin(&format!("{p}.linear1"), w, ff, true)?,
                linear2: t.lin(&format!("{p}.linear2"), ff, w, true)?,
            });
        }
        let field_norm = t.norm("field_norm", w)?;
        let option_norm = t.norm("option_norm", w)?;
        let scorer1 = t.lin("residual_scorer.0", 4 * w, w, true)?;
        let scorer2 = t.lin("residual_scorer.3", w, 1, true)?;
        let prior_logit_scale = t.scalar("prior_logit_scale")?;
        let joint_logit_scale = t.scalar("joint_logit_scale")?;
        let residual_gate = t.scalar("residual_gate")?;
        if !t.left.is_empty() {
            bail!("joint head: unexpected tensors {:?}", t.left);
        }
        Ok(Self {
            config,
            hidden_norm,
            memory_projection,
            question_projection,
            option_question_projection,
            global_projection,
            option_context_projection,
            option_lexical_projection,
            type_embedding,
            evidence,
            option_summary_norm,
            layers,
            field_norm,
            option_norm,
            scorer1,
            scorer2,
            prior_logit_scale,
            joint_logit_scale,
            residual_gate,
        })
    }

    fn evidence_layer(&self, l: &Evidence, q: &mut [f32], n: usize, memory: &[f32], t: usize) {
        let w = self.config.width;
        let nq = l.query_norm.apply(q, n);
        let nm = l.memory_norm.apply(memory, t);
        let routed = l.attention.forward(&nq, n, &nm, t, w);
        add_in_place(q, &routed);
        let mut hdn = l.ff1.apply(&l.ff_norm.apply(q, n), n);
        gelu_in_place(&mut hdn);
        let out = l.ff2.apply(&hdn, n);
        add_in_place(q, &out);
    }

    fn decoder_layer(&self, l: &DecoderLayer, x: &mut [f32], n: usize, memory: &[f32], t: usize) {
        let w = self.config.width;
        // x = x + sa(norm1(x)); x = x + mha(norm2(x), memory); x = x + ff(norm3(x))
        let n1 = l.norm1.apply(x, n);
        let sa = l.self_attn.forward(&n1, n, &n1, n, w);
        add_in_place(x, &sa);
        let n2 = l.norm2.apply(x, n);
        let ca = l.cross_attn.forward(&n2, n, memory, t, w);
        add_in_place(x, &ca);
        let mut hdn = l.linear1.apply(&l.norm3.apply(x, n), n);
        gelu_in_place(&mut hdn);
        let out = l.linear2.apply(&hdn, n);
        add_in_place(x, &out);
    }

    /// Per-question logits (one per option, in the record's `option_ids` order).
    ///
    /// `hidden` is the thinker's final normed output `[t, hidden_size]`, `input_ids`
    /// the record's token ids (`[t]`), `lm` supplies the output-embedding rows.
    pub fn forward(
        &self,
        hidden: &[f32],
        input_ids: &[u32],
        questions: &[QuestionSpans],
        lm: &dyn LmHeadRows,
    ) -> Result<Vec<Vec<f32>>> {
        let (h, w) = (self.config.hidden_size, self.config.width);
        if hidden.is_empty() || !hidden.len().is_multiple_of(h) {
            bail!("hidden has {} values, not a multiple of {h}", hidden.len());
        }
        let t = hidden.len() / h;
        if input_ids.len() != t {
            bail!("{} input ids for {t} hidden rows", input_ids.len());
        }
        if questions.is_empty() {
            bail!("no questions");
        }
        let nh = self.hidden_norm.apply(hidden, t);
        let memory = self.memory_projection.apply(&nh, t);
        let global_vec = &nh[(t - 1) * h..];

        let mut qvecs = Vec::new(); // [Q, h]
        let mut ctx = Vec::new(); // [N, h]
        let mut lex = Vec::new(); // [N, h]
        let mut counts = Vec::new();
        let mut row = vec![0f32; h];
        for (qi, q) in questions.iter().enumerate() {
            if q.question_type > 2 {
                bail!("question {qi}: type {} is not 0..=2", q.question_type);
            }
            if q.option_spans.is_empty() {
                bail!("question {qi} has no options");
            }
            qvecs.extend(mean_span(&nh, h, q.question_span, "question span")?);
            for &(s, e) in &q.option_spans {
                ctx.extend(mean_span(&nh, h, (s, e), "option span")?);
                let mut acc = vec![0f32; h];
                for &id in &input_ids[s..e] {
                    lm.row(id, &mut row)?;
                    for (a, r) in acc.iter_mut().zip(&row) {
                        *a += r;
                    }
                }
                let n = (e - s) as f32;
                acc.iter_mut().for_each(|a| *a /= n);
                lex.extend(acc);
            }
            counts.push(q.option_spans.len());
        }
        let nq = questions.len();
        let n: usize = counts.iter().sum();

        // Option queries: context + lexical + (broadcast) question projections.
        let mut opts = self.option_context_projection.apply(&ctx, n);
        add_in_place(&mut opts, &self.option_lexical_projection.apply(&lex, n));
        let qopt = self.option_question_projection.apply(&qvecs, nq);
        let mut r = 0;
        for (qi, &c) in counts.iter().enumerate() {
            for _ in 0..c {
                for (o, v) in opts[r * w..(r + 1) * w]
                    .iter_mut()
                    .zip(&qopt[qi * w..(qi + 1) * w])
                {
                    *o += v;
                }
                r += 1;
            }
        }
        for l in &self.evidence {
            self.evidence_layer(l, &mut opts, n, &memory, t);
        }

        // Option summaries by routing softmax against the base field.
        let base_fields = self.question_projection.apply(&qvecs, nq);
        let mut summaries = vec![0f32; nq * w];
        let mut r = 0;
        for (qi, &c) in counts.iter().enumerate() {
            let field = &base_fields[qi * w..(qi + 1) * w];
            let o = &opts[r * w..(r + c) * w];
            let scale = (w as f32).sqrt();
            let mut weights: Vec<f32> = (0..c)
                .map(|j| dot(&o[j * w..(j + 1) * w], field) / scale)
                .collect();
            softmax_row(&mut weights);
            let out = &mut summaries[qi * w..(qi + 1) * w];
            for (j, p) in weights.iter().enumerate() {
                for (a, v) in out.iter_mut().zip(&o[j * w..(j + 1) * w]) {
                    *a += p * v;
                }
            }
            r += c;
        }
        let mut fields = base_fields;
        add_in_place(&mut fields, &self.option_summary_norm.apply(&summaries, nq));
        let global_row = self.global_projection.apply(global_vec, 1);
        for qi in 0..nq {
            let ty = questions[qi].question_type;
            for i in 0..w {
                fields[qi * w + i] += global_row[i];
                fields[qi * w + i] += self.type_embedding[ty * w + i];
            }
        }
        for l in &self.layers {
            self.decoder_layer(l, &mut fields, nq, &memory, t);
        }
        let fields = self.field_norm.apply(&fields, nq);

        // Scoring.
        let onorm = self.option_norm.apply(&opts, n);
        let mut feats = vec![0f32; n * 4 * w];
        let mut cos = vec![0f32; n];
        let mut r = 0;
        for (qi, &c) in counts.iter().enumerate() {
            let f = &fields[qi * w..(qi + 1) * w];
            for _ in 0..c {
                let o = &onorm[r * w..(r + 1) * w];
                cos[r] = cosine(f, o);
                let dst = &mut feats[r * 4 * w..(r + 1) * 4 * w];
                for i in 0..w {
                    dst[i] = f[i];
                    dst[w + i] = o[i];
                    dst[2 * w + i] = f[i] * o[i];
                    dst[3 * w + i] = (f[i] - o[i]).abs();
                }
                r += 1;
            }
        }
        let mut hid = self.scorer1.apply(&feats, n);
        gelu_in_place(&mut hid);
        let residual = self.scorer2.apply(&hid, n);

        let cap = 100f32.ln();
        let prior_scale = self.prior_logit_scale.min(cap).exp();
        let joint_scale = self.joint_logit_scale.min(cap).exp();
        let gate = sigmoid(self.residual_gate);
        let mut out = Vec::with_capacity(nq);
        let mut r = 0;
        for (qi, &c) in counts.iter().enumerate() {
            let qv = &qvecs[qi * h..(qi + 1) * h];
            let sum: Vec<f32> = qv.iter().zip(global_vec).map(|(a, b)| a + b).collect();
            let anchor = normalize(&sum);
            let logits = (0..c)
                .map(|j| {
                    let lexical_anchor = normalize(&lex[(r + j) * h..(r + j + 1) * h]);
                    let prior = prior_scale * dot(&lexical_anchor, &anchor);
                    let joint = joint_scale * cos[r + j] + residual[r + j];
                    prior + gate * joint
                })
                .collect();
            out.push(logits);
            r += c;
        }
        Ok(out)
    }
}

/// `thinker.lm_head.weight` rows read lazily from the checkpoint shard named by
/// `model.safetensors.index.json` (only the needed rows are touched).
pub struct LmHeadFile {
    file: SafeFile,
    info: TensorInfo,
}

impl LmHeadFile {
    pub fn open(dir: &Path) -> Result<Self> {
        const NAME: &str = "thinker.lm_head.weight";
        let idx_path = dir.join("model.safetensors.index.json");
        let idx: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&idx_path).with_context(|| format!("read {}", idx_path.display()))?,
        )?;
        let shard = idx["weight_map"][NAME]
            .as_str()
            .with_context(|| format!("{NAME} is not in the shard index"))?;
        let file = SafeFile::open(&dir.join(shard))?;
        let info = file
            .tensors
            .get(NAME)
            .with_context(|| format!("{shard} has no {NAME}"))?
            .clone();
        Ok(Self { file, info })
    }
}

impl LmHeadRows for LmHeadFile {
    fn row(&self, id: u32, out: &mut [f32]) -> Result<()> {
        self.file.row_f32(&self.info, id as usize, out)
    }
}

#[cfg(test)]
mod tests;
