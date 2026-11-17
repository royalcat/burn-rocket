//! `rocket-inference clef-pack`: repack the AutoRound int8 checkpoint into a Clef pack.
//!
//! ```text
//! rocket-inference clef-pack --src <autoround dir> --out <pack dir>
//!     [--tokenizer-from <dir>] [--threads 8] [--layers A..B] [--verify | --verify-only]
//! ```
//!
//! Streaming: one layer at a time (one ~0.6 GB expert buffer being built while the previous
//! one is hashed and written by a writer thread), multi-threaded unpack on a private
//! rayon pool, progress on stderr.

use super::autoround::{self, Source};
use super::format::{
    ALIGN, Dims, ExpertRecordMut, HEADER_BYTES, LM_HEAD, LM_HEAD_TENSOR, LayerHeader, MANIFEST,
    RESIDENT, layer_file,
};
use super::reader::PackDir;
use super::{
    FORMAT_NAME, FileInfo, LmHeadInfo, Manifest, ModelInfo, RecordInfo, ResidentInfo, SkippedInfo,
    SourceInfo, TensorEntry, hex,
};
use crate::clef::model::TextConfig;
use crate::clef::st::Dtype;
use anyhow::{Context, Result, bail};
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub const USAGE: &str = "\
usage: rocket-inference clef-pack --src <autoround dir> --out <pack dir> [flags]

Repack the moeshawky/clef-omni-int8-autoround checkpoint (AutoRound int8, group 128)
into the Clef pack: one 4 KiB-aligned layer_NN.experts file per layer, a resident
safetensors file and a manifest (docs/experiments/clef-pack.md).

  --src <dir>             the AutoRound checkpoint (index + shards + config)
  --out <dir>             the pack directory (created; existing pack files are replaced)
  --tokenizer-from <dir>  where to take tokenizer.json / tokenizer_config.json when the
                          source lacks them (the BF16 clef-omni checkpoint)
  --threads <n>           unpack workers (default 8, max 8)
  --layers A..B           only pack/verify this layer range (a partial pack has no manifest)
  --verify                after packing, re-read every expert + the resident file from the
                          pack and compare with a fresh unpack of the source; re-hash files
  --verify-only           verify an existing pack (--out) against --src without packing
  --compare-bf16 <dir>    no packing: dequantize experts 0 and E-1 and the four attention
                          projections of every layer in --layers (default 0..1) and report
                          the error against the BF16 checkpoint in <dir>";

struct Args {
    src: PathBuf,
    out: PathBuf,
    tokenizer_from: Option<PathBuf>,
    threads: usize,
    layers: Option<Range<usize>>,
    verify: bool,
    verify_only: bool,
    compare_bf16: Option<PathBuf>,
}

fn parse(mut it: impl Iterator<Item = String>) -> Result<Args> {
    let (mut src, mut out, mut tok, mut threads) = (None, None, None, 8usize);
    let (mut layers, mut verify, mut verify_only) = (None, false, false);
    let mut compare = None;
    while let Some(a) = it.next() {
        let mut val = |name: &str| it.next().with_context(|| format!("{name} needs a value"));
        match a.as_str() {
            "--src" => src = Some(PathBuf::from(val("--src")?)),
            "--out" => out = Some(PathBuf::from(val("--out")?)),
            "--tokenizer-from" => tok = Some(PathBuf::from(val("--tokenizer-from")?)),
            "--threads" => threads = val("--threads")?.parse().context("--threads")?,
            "--layers" => {
                let v = val("--layers")?;
                let (a, b) = v.split_once("..").context("--layers wants A..B")?;
                layers = Some(a.parse().context("--layers")?..b.parse().context("--layers")?);
            }
            "--verify" => verify = true,
            "--verify-only" => verify_only = true,
            "--compare-bf16" => compare = Some(PathBuf::from(val("--compare-bf16")?)),
            "--help" | "-h" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other => bail!("unknown flag {other}\n\n{USAGE}"),
        }
    }
    Ok(Args {
        src: src.context("--src is required")?,
        out: out.unwrap_or_default(),
        compare_bf16: compare,
        tokenizer_from: tok,
        threads: threads.clamp(1, 8),
        layers,
        verify,
        verify_only,
    })
}

pub fn run(it: impl Iterator<Item = String>) -> Result<()> {
    let a = parse(it)?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(a.threads)
        .build()?;
    pool.install(|| {
        let src =
            Source::open(&a.src).with_context(|| format!("open source {}", a.src.display()))?;
        let plan = Plan::from_source(&src, &a.src)?;
        let layers = a.layers.clone().unwrap_or(0..plan.cfg.layers);
        if layers.end > plan.cfg.layers || layers.start >= layers.end {
            bail!("--layers {layers:?} outside 0..{}", plan.cfg.layers);
        }
        if let Some(bf16) = &a.compare_bf16 {
            let layers = a.layers.clone().unwrap_or(0..1);
            return compare_bf16(&src, &plan, bf16, layers);
        }
        if a.out.as_os_str().is_empty() {
            bail!("--out is required");
        }
        if !a.verify_only {
            pack(&a, &src, &plan, layers.clone())?;
        }
        if a.verify || a.verify_only {
            verify(&a.out, &src, &plan, layers)?;
        }
        Ok(())
    })
}

// ---------------------------------------------------------------- plan

/// Model facts needed to pack, read from the source `config.json`.
struct Plan {
    cfg: TextConfig,
    vocab: usize,
    dims: Dims,
    quant: serde_json::Value,
}

impl Plan {
    fn from_source(src: &Source, dir: &Path) -> Result<Self> {
        let cfg = TextConfig::from_model_dir(dir)?;
        let raw: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
        let q = raw["quantization_config"].clone();
        let group = q["group_size"]
            .as_u64()
            .context("quantization_config.group_size")? as usize;
        if q["bits"] != 8 || q["sym"] != true {
            bail!(
                "only symmetric 8-bit AutoRound is supported (bits={}, sym={})",
                q["bits"],
                q["sym"]
            );
        }
        if !q["packing_format"]
            .as_str()
            .is_some_and(|p| p.contains("auto_gptq"))
        {
            bail!("unsupported packing_format {}", q["packing_format"]);
        }
        if q["block_name_to_quantize"] != "thinker.model.layers" {
            bail!("unexpected quantized block {}", q["block_name_to_quantize"]);
        }
        let vocab = raw["thinker_config"]["text_config"]["vocab_size"]
            .as_u64()
            .context("text_config.vocab_size")? as usize;
        let dims = Dims {
            hidden: cfg.hidden,
            inter: cfg.inter,
            group,
        };
        dims.validate()?;
        if cfg.heads * cfg.head_dim % group != 0 {
            bail!("group {group} does not divide the attention width");
        }
        for t in [
            "thinker.model.embed_tokens.weight",
            "thinker.model.norm.weight",
            LM_HEAD_TENSOR,
        ] {
            if !src.has(t) {
                bail!("source lacks {t}");
            }
        }
        Ok(Self {
            cfg,
            vocab,
            dims,
            quant: q,
        })
    }
}

/// Dense (non-expert) tensors of a layer: (name, kept bf16) and the attention linears
/// `(name, n, k)` for a [`Plan`].
fn layer_dense_names(l: usize) -> Vec<String> {
    let p = format!("thinker.model.layers.{l}.");
    [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "mlp.gate.weight",
    ]
    .iter()
    .map(|n| format!("{p}{n}"))
    .collect()
}

fn layer_attn(cfg: &TextConfig, l: usize) -> Vec<(String, usize, usize)> {
    let (h, d) = (cfg.hidden, cfg.head_dim);
    let p = format!("thinker.model.layers.{l}.self_attn.");
    vec![
        (format!("{p}q_proj"), cfg.heads * d, h),
        (format!("{p}k_proj"), cfg.kv_heads * d, h),
        (format!("{p}v_proj"), cfg.kv_heads * d, h),
        (format!("{p}o_proj"), h, cfg.heads * d),
    ]
}

/// Fill one expert's `stride`-byte region from the source (gate | up fused on N).
fn fill_expert(src: &Source, plan: &Plan, layer: usize, e: usize, region: &mut [u8]) -> Result<()> {
    let d = plan.dims;
    let rec = ExpertRecordMut::split(d, region);
    let p = format!("thinker.model.layers.{layer}.mlp.experts.{e}.");
    let gu_n = d.inter * d.hidden;
    let gu_s = d.inter * (d.hidden / d.group) * 2;
    let (g_codes, u_codes) = rec.gu_codes.split_at_mut(gu_n);
    let (g_sc, u_sc) = rec.gu_scales.split_at_mut(gu_s);
    for (name, codes, sc) in [("gate_proj", g_codes, g_sc), ("up_proj", u_codes, u_sc)] {
        let q = src.qlinear(&format!("{p}{name}"), d.inter, d.hidden, d.group)?;
        autoround::unpack(&q, codes, sc).with_context(|| format!("{p}{name}"))?;
    }
    let q = src.qlinear(&format!("{p}down_proj"), d.hidden, d.inter, d.group)?;
    autoround::unpack(&q, rec.down_codes, rec.down_scales)
        .with_context(|| format!("{p}down_proj"))?;
    Ok(())
}

// ---------------------------------------------------------------- writing

struct HashFile {
    f: std::fs::File,
    h: Sha256,
    n: u64,
}

impl HashFile {
    fn create(path: &Path) -> Result<Self> {
        Ok(Self {
            f: std::fs::File::create(path).with_context(|| format!("create {}", path.display()))?,
            h: Sha256::new(),
            n: 0,
        })
    }

    fn write(&mut self, b: &[u8]) -> Result<()> {
        self.h.update(b);
        self.f.write_all(b)?;
        self.n += b.len() as u64;
        Ok(())
    }

    fn finish(self, name: &str) -> Result<FileInfo> {
        self.f.sync_all()?;
        Ok(FileInfo {
            name: name.to_string(),
            bytes: self.n,
            sha256: hex(&self.h.finalize()),
        })
    }
}

/// One tensor of a safetensors file being written.
struct OutTensor {
    name: String,
    dtype: Dtype,
    shape: Vec<usize>,
    start: u64,
    end: u64,
}

fn tensor_bytes(dtype: Dtype, shape: &[usize]) -> u64 {
    (dtype.size() * shape.iter().product::<usize>()) as u64
}

/// The safetensors header for `tensors` (offsets already assigned), padded to 8 bytes.
fn st_header(tensors: &[OutTensor], format: &str) -> Vec<u8> {
    let mut m = serde_json::Map::new();
    m.insert(
        "__metadata__".into(),
        serde_json::json!({ "format": format }),
    );
    for t in tensors {
        m.insert(
            t.name.clone(),
            serde_json::json!({
                "dtype": t.dtype.name(),
                "shape": t.shape,
                "data_offsets": [t.start, t.end],
            }),
        );
    }
    let mut j = serde_json::to_vec(&serde_json::Value::Object(m)).unwrap();
    while !j.len().is_multiple_of(8) {
        j.push(b' ');
    }
    let mut out = (j.len() as u64).to_le_bytes().to_vec();
    out.extend(j);
    out
}

fn assign(list: &mut Vec<OutTensor>, name: String, dtype: Dtype, shape: Vec<usize>) {
    let start = list.last().map_or(0, |t| t.end);
    let end = start + tensor_bytes(dtype, &shape);
    list.push(OutTensor {
        name,
        dtype,
        shape,
        start,
        end,
    });
}

fn pack(a: &Args, src: &Source, plan: &Plan, layers: Range<usize>) -> Result<()> {
    let t0 = Instant::now();
    std::fs::create_dir_all(&a.out)?;
    let cfg = &plan.cfg;
    let partial = layers != (0..cfg.layers);
    // The old manifest goes first: a half-written pack must not look valid.
    let _ = std::fs::remove_file(a.out.join(MANIFEST));
    let mut files: Vec<FileInfo> = Vec::new();

    // ---- expert files (writer thread overlaps hash + write with the next layer's unpack)
    let (tx, rx) = std::sync::mpsc::sync_channel::<(usize, Vec<u8>)>(0);
    let out_dir = a.out.clone();
    let writer = std::thread::spawn(move || -> Result<Vec<FileInfo>> {
        let mut done = Vec::new();
        for (l, buf) in rx {
            let name = layer_file(l);
            let mut f = HashFile::create(&out_dir.join(&name))?;
            f.write(&buf)?;
            done.push(f.finish(&name)?);
        }
        Ok(done)
    });
    let header = |l: usize| LayerHeader::new(l, cfg.experts, plan.dims);
    let stride = plan.dims.layout().stride;
    let build: Result<()> = (|| {
        for l in layers.clone() {
            let lt = Instant::now();
            let h = header(l);
            let mut buf = vec![0u8; h.file_bytes() as usize];
            buf[..HEADER_BYTES].copy_from_slice(&h.encode());
            buf[HEADER_BYTES..]
                .par_chunks_mut(stride)
                .enumerate()
                .try_for_each(|(e, region)| fill_expert(src, plan, l, e, region))?;
            src.release();
            let mib = buf.len() as f64 / 1048576.0;
            eprintln!(
                "layer {l:2}/{}: {mib:.0} MiB unpacked in {:.1}s  (elapsed {:.0}s)",
                cfg.layers,
                lt.elapsed().as_secs_f64(),
                t0.elapsed().as_secs_f64()
            );
            if tx.send((l, buf)).is_err() {
                bail!("the writer thread stopped");
            }
        }
        Ok(())
    })();
    drop(tx);
    let written = writer
        .join()
        .map_err(|_| anyhow::anyhow!("writer thread panicked"))?;
    build?;
    files.extend(written?);

    // ---- resident file: embeddings, norms, router gates, int8 attention
    let mut tl: Vec<OutTensor> = Vec::new();
    // Plan: bf16 copies first (embed, norm), then per layer in order.
    let g = plan.dims.group;
    let mut order: Vec<(String, Option<(usize, usize)>)> = Vec::new(); // (name, Some((n,k)) = quantized module)
    for n in [
        "thinker.model.embed_tokens.weight",
        "thinker.model.norm.weight",
    ] {
        order.push((n.to_string(), None));
    }
    for l in layers.clone() {
        for n in layer_dense_names(l) {
            order.push((n, None));
        }
        for (m, n, k) in layer_attn(cfg, l) {
            order.push((m, Some((n, k))));
        }
    }
    for (name, q) in &order {
        match q {
            None => {
                let t = src.info(name)?;
                assign(&mut tl, name.clone(), t.dtype, t.shape.clone());
            }
            Some((n, k)) => {
                assign(&mut tl, format!("{name}.codes"), Dtype::I8, vec![*n, *k]);
                assign(
                    &mut tl,
                    format!("{name}.scales"),
                    Dtype::F16,
                    vec![*n, *k / g],
                );
            }
        }
    }
    let header_bytes = st_header(&tl, "clef-pack-resident");
    let data_start = header_bytes.len() as u64;
    let mut rf = HashFile::create(&a.out.join(RESIDENT))?;
    rf.write(&header_bytes)?;
    let rt = Instant::now();
    for (name, q) in &order {
        match q {
            None => {
                let t = src.info(name)?;
                rf.write(src.bytes(name, t.dtype, &t.shape)?)?;
            }
            Some((n, k)) => {
                let ql = src.qlinear(name, *n, *k, g)?;
                let mut codes = vec![0i8; n * k];
                let mut scales = vec![0u8; n * (k / g) * 2];
                autoround::unpack(&ql, &mut codes, &mut scales).with_context(|| name.clone())?;
                // SAFETY: i8 and u8 share a layout.
                rf.write(unsafe {
                    std::slice::from_raw_parts(codes.as_ptr().cast::<u8>(), codes.len())
                })?;
                rf.write(&scales)?;
            }
        }
    }
    src.release();
    let rinfo = rf.finish(RESIDENT)?;
    eprintln!(
        "resident: {:.2} GiB, {} tensors in {:.1}s",
        rinfo.bytes as f64 / 1073741824.0,
        tl.len(),
        rt.elapsed().as_secs_f64()
    );
    files.push(rinfo);

    // ---- lm_head as its own mmap-able file
    {
        let t = src.info(LM_HEAD_TENSOR)?;
        let one = [OutTensor {
            name: LM_HEAD_TENSOR.to_string(),
            dtype: t.dtype,
            shape: t.shape.clone(),
            start: 0,
            end: tensor_bytes(t.dtype, &t.shape),
        }];
        let mut f = HashFile::create(&a.out.join(LM_HEAD))?;
        f.write(&st_header(&one, "clef-pack-lm-head"))?;
        f.write(src.bytes(LM_HEAD_TENSOR, t.dtype, &t.shape)?)?;
        files.push(f.finish(LM_HEAD)?);
        src.release();
    }

    // ---- verbatim copies
    let mut copied = Vec::new();
    for name in [
        "config.json",
        "joint_head.safetensors",
        "joint_head_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ] {
        let from = [Some(&a.src), a.tokenizer_from.as_ref()]
            .into_iter()
            .flatten()
            .map(|d| d.join(name))
            .find(|p| p.exists());
        let Some(from) = from else {
            bail!(
                "{name} is in neither --src nor --tokenizer-from (pass --tokenizer-from <clef-omni dir>)"
            );
        };
        let mut r = std::fs::File::open(&from)?;
        let mut f = HashFile::create(&a.out.join(name))?;
        let mut chunk = vec![0u8; 8 << 20];
        loop {
            let n = r.read(&mut chunk)?;
            if n == 0 {
                break;
            }
            f.write(&chunk[..n])?;
        }
        files.push(f.finish(name)?);
        copied.push(name.to_string());
    }

    // ---- skipped source tensors, accounted by 2-component prefix
    let skipped = skipped_summary(src);

    files.sort_by(|x, y| x.name.cmp(&y.name));
    let table: Vec<TensorEntry> = tl
        .iter()
        .map(|t| TensorEntry {
            name: t.name.clone(),
            dtype: t.dtype.name().to_string(),
            shape: t.shape.clone(),
            offset: data_start + t.start,
            bytes: t.end - t.start,
        })
        .collect();
    if partial {
        eprintln!("partial pack ({layers:?}): no manifest written");
        return Ok(());
    }
    let l = plan.dims.layout();
    let manifest = Manifest {
        format: FORMAT_NAME.to_string(),
        version: super::format::FORMAT_VERSION,
        model: ModelInfo {
            layers: cfg.layers,
            hidden: cfg.hidden,
            experts: cfg.experts,
            moe_intermediate: cfg.inter,
            group: g,
            heads: cfg.heads,
            kv_heads: cfg.kv_heads,
            head_dim: cfg.head_dim,
            vocab: plan.vocab,
        },
        source: SourceInfo {
            repo: "moeshawky/clef-omni-int8-autoround".to_string(),
            index_sha256: src.index_sha256.clone(),
            bits: 8,
            group: g,
            sym: true,
            packing_format: plan.quant["packing_format"]
                .as_str()
                .unwrap_or("")
                .to_string(),
        },
        expert_record: RecordInfo {
            align: ALIGN,
            header_bytes: HEADER_BYTES,
            record_bytes: l.record_bytes,
            stride: l.stride,
            gu_codes: l.gu_codes,
            down_codes: l.down_codes,
            gu_scales: l.gu_scales,
            down_scales: l.down_scales,
        },
        files,
        resident: ResidentInfo {
            file: RESIDENT.to_string(),
            tensors: table,
        },
        lm_head: LmHeadInfo {
            file: LM_HEAD.to_string(),
            tensor: LM_HEAD_TENSOR.to_string(),
        },
        copied,
        skipped,
    };
    manifest.validate()?;
    let tmp = a.out.join("manifest.json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(&manifest)?)?;
    std::fs::rename(&tmp, a.out.join(MANIFEST))?;
    let total: u64 = manifest.files.iter().map(|f| f.bytes).sum();
    eprintln!(
        "pack complete: {:.2} GiB in {} files, {:.0}s",
        total as f64 / 1073741824.0,
        manifest.files.len(),
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

/// Source tensors the pack does not carry, grouped by their first two name components.
fn skipped_summary(src: &Source) -> Vec<SkippedInfo> {
    let mut by: std::collections::BTreeMap<String, (usize, u64)> = Default::default();
    for name in src.names() {
        let kept = name.starts_with("thinker.model.layers.")
            || name == "thinker.model.embed_tokens.weight"
            || name == "thinker.model.norm.weight"
            || name == LM_HEAD_TENSOR;
        if kept {
            continue;
        }
        let prefix: String = name.split('.').take(2).collect::<Vec<_>>().join(".");
        let bytes = src.info(name).map_or(0, |t| (t.end - t.start) as u64);
        let e = by.entry(prefix).or_default();
        e.0 += 1;
        e.1 += bytes;
    }
    by.into_iter()
        .map(|(prefix, (tensors, bytes))| SkippedInfo {
            prefix,
            tensors,
            bytes,
        })
        .collect()
}

// ---------------------------------------------------------------- BF16 comparison

/// Dequantized AutoRound weights vs the BF16 originals: error relative to the weight scale.
fn compare_bf16(src: &Source, plan: &Plan, bf16: &Path, layers: Range<usize>) -> Result<()> {
    let shards = crate::clef::model::Shards::open(bf16)?;
    let (cfg, g) = (&plan.cfg, plan.dims.group);
    let mut jobs: Vec<(String, usize, usize)> = Vec::new();
    for l in layers {
        jobs.extend(layer_attn(cfg, l));
        for e in [0, cfg.experts - 1] {
            let p = format!("thinker.model.layers.{l}.mlp.experts.{e}.");
            jobs.push((format!("{p}gate_proj"), cfg.inter, cfg.hidden));
            jobs.push((format!("{p}up_proj"), cfg.inter, cfg.hidden));
            jobs.push((format!("{p}down_proj"), cfg.hidden, cfg.inter));
        }
    }
    println!(
        "{:<62} {:>10} {:>11} {:>10} {:>10}",
        "module", "max|e|/max|w|", "mean|e|/mean|w|", "rms rel", "cosine"
    );
    let (mut worst_max, mut sum_mean) = (0f64, 0f64);
    for (module, n, k) in &jobs {
        let q = src.qlinear(module, *n, *k, g)?;
        let mut codes = vec![0i8; n * k];
        let mut sc = vec![0u8; n * (k / g) * 2];
        autoround::unpack(&q, &mut codes, &mut sc)?;
        let mut dq = vec![0f32; n * k];
        for (row, o) in dq.chunks_mut(*k).enumerate() {
            // SAFETY: i8/u8 share a layout.
            let c =
                unsafe { std::slice::from_raw_parts(codes[row * k..].as_ptr().cast::<u8>(), *k) };
            super::reader::dequant_row(c, &sc[row * (k / g) * 2..(row + 1) * (k / g) * 2], g, o);
        }
        // independent per-element reference straight from the packed words
        let refw = autoround::dequant_reference(&q);
        if refw != dq {
            bail!("{module}: unpack disagrees with the per-element reference");
        }
        let w = shards.f32(&format!("{module}.weight"), &[*n, *k])?;
        let (mut maxw, mut maxe, mut sume, mut sumw, mut sq_e, mut sq_w, mut dotp, mut nq) =
            (0f64, 0f64, 0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
        for (a, b) in dq.iter().zip(&w) {
            let (a, b) = (f64::from(*a), f64::from(*b));
            let e = (a - b).abs();
            maxw = maxw.max(b.abs());
            maxe = maxe.max(e);
            sume += e;
            sumw += b.abs();
            sq_e += e * e;
            sq_w += b * b;
            dotp += a * b;
            nq += a * a;
        }
        let (rel_max, rel_mean) = (maxe / maxw, sume / sumw);
        worst_max = worst_max.max(rel_max);
        sum_mean += rel_mean;
        println!(
            "{:<62} {:>10.4e} {:>11.4e} {:>10.4e} {:>10.8}",
            module.trim_start_matches("thinker.model.layers."),
            rel_max,
            rel_mean,
            (sq_e / sq_w).sqrt(),
            dotp / (nq.sqrt() * sq_w.sqrt())
        );
    }
    println!(
        "{} modules: worst max|e|/max|w| {:.4e}, mean of mean|e|/mean|w| {:.4e}",
        jobs.len(),
        worst_max,
        sum_mean / jobs.len() as f64
    );
    Ok(())
}

// ---------------------------------------------------------------- verify

fn hash_file(path: &Path) -> Result<(u64, String)> {
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 8 << 20];
    let mut n = 0u64;
    loop {
        let k = f.read(&mut buf)?;
        if k == 0 {
            break;
        }
        h.update(&buf[..k]);
        n += k as u64;
    }
    Ok((n, hex(&h.finalize())))
}

fn verify(out: &Path, src: &Source, plan: &Plan, layers: Range<usize>) -> Result<()> {
    let t0 = Instant::now();
    let pack = PackDir::open(out).with_context(|| format!("open pack {}", out.display()))?;
    let m = &pack.manifest;
    if m.source.index_sha256 != src.index_sha256 {
        bail!(
            "pack was built from a different source index ({} vs {})",
            m.source.index_sha256,
            src.index_sha256
        );
    }
    // 1. file checksums
    let bad: Vec<String> = m
        .files
        .par_iter()
        .filter_map(|f| match hash_file(&out.join(&f.name)) {
            Ok((n, h)) if n == f.bytes && h == f.sha256 => None,
            Ok((n, h)) => Some(format!(
                "{}: {n} bytes sha {h}, manifest {} bytes sha {}",
                f.name, f.bytes, f.sha256
            )),
            Err(e) => Some(format!("{}: {e:#}", f.name)),
        })
        .collect();
    if !bad.is_empty() {
        bail!("checksum mismatch:\n  {}", bad.join("\n  "));
    }
    eprintln!(
        "verify: {} file checksums ok ({:.0}s)",
        m.files.len(),
        t0.elapsed().as_secs_f64()
    );

    // 2. every expert vs a fresh unpack of the source
    let cfg = &plan.cfg;
    let stride = pack.stride();
    let mut experts = 0usize;
    for l in layers.clone() {
        let lt = Instant::now();
        let mism: Vec<usize> = (0..cfg.experts)
            .into_par_iter()
            .map_init(
                || (pack.new_buffer(), vec![0u8; stride]),
                |(buf, want), e| {
                    want.fill(0);
                    let ok = fill_expert(src, plan, l, e, want).is_ok()
                        && pack.read_expert(l, e, buf).is_ok()
                        && buf.as_slice()[..stride] == want[..];
                    (!ok).then_some(e)
                },
            )
            .flatten()
            .collect();
        src.release();
        if !mism.is_empty() {
            bail!("layer {l}: experts differ from the source unpack: {mism:?}");
        }
        experts += cfg.experts;
        eprintln!(
            "verify: layer {l:2} experts ok ({:.1}s)",
            lt.elapsed().as_secs_f64()
        );
    }

    // 3. resident tensors
    let g = plan.dims.group;
    let res = pack.resident();
    let mut names = 0usize;
    let check_copy = |name: &str| -> Result<()> {
        let t = src.info(name)?;
        let r = pack.tensor(name)?;
        if res.bytes(r) != src.bytes(name, t.dtype, &t.shape)? {
            bail!("resident {name} differs from the source");
        }
        Ok(())
    };
    check_copy("thinker.model.embed_tokens.weight")?;
    check_copy("thinker.model.norm.weight")?;
    names += 2;
    for l in layers.clone() {
        for n in layer_dense_names(l) {
            check_copy(&n)?;
            names += 1;
        }
        for (module, n, k) in layer_attn(cfg, l) {
            let q = src.qlinear(&module, n, k, g)?;
            let mut codes = vec![0i8; n * k];
            let mut scales = vec![0u8; n * (k / g) * 2];
            autoround::unpack(&q, &mut codes, &mut scales)?;
            let c = res.bytes(pack.tensor(&format!("{module}.codes"))?);
            let s = res.bytes(pack.tensor(&format!("{module}.scales"))?);
            // SAFETY: i8/u8 share a layout.
            let cb =
                unsafe { std::slice::from_raw_parts(codes.as_ptr().cast::<u8>(), codes.len()) };
            if c != cb || s != &scales[..] {
                bail!("resident {module} differs from the source unpack");
            }
            names += 2;
        }
        src.release();
    }
    // lm_head
    let t = src.info(LM_HEAD_TENSOR)?;
    pack.lm_head().context("open the pack's lm_head")?;
    let lmf = crate::clef::st::SafeFile::open(&out.join(&m.lm_head.file))?;
    if lmf.bytes(&lmf.tensors[LM_HEAD_TENSOR]) != src.bytes(LM_HEAD_TENSOR, t.dtype, &t.shape)? {
        bail!("lm_head differs from the source");
    }
    eprintln!(
        "verify OK: {experts} experts ({} reads), {names} resident tensors, lm_head, {} checksums in {:.0}s",
        if pack.is_direct() {
            "O_DIRECT"
        } else {
            "buffered"
        },
        m.files.len(),
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

// ---------------------------------------------------------------- tests support

/// Write a safetensors file from `(name, dtype, shape, bytes)` (test fixtures and tools).
pub fn write_safetensors(
    path: &Path,
    tensors: &[(String, Dtype, Vec<usize>, Vec<u8>)],
) -> Result<()> {
    let mut list = Vec::new();
    for (n, d, s, b) in tensors {
        assert_eq!(b.len() as u64, tensor_bytes(*d, s), "{n}: byte length");
        assign(&mut list, n.clone(), *d, s.clone());
    }
    let mut f = std::fs::File::create(path)?;
    f.write_all(&st_header(&list, "pt"))?;
    for (_, _, _, b) in tensors {
        f.write_all(b)?;
    }
    Ok(())
}
