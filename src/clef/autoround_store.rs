//! Serving straight from the AutoRound checkpoint (`moeshawky/clef-omni-int8-autoround`,
//! safetensors shards) with no repack: [`ShardStore`].
//!
//! Layout of an expert in the shards (measured, `docs/experiments/clef-omni.md` "Step 0"): for one
//! expert the six `qweight`/`qzeros` tensors (down, gate, up) are **one contiguous run**
//! (4.76 MB, consecutive experts follow each other) and the three f16 `scales` tensors
//! are a **second contiguous run** (72 KB) in a separate region at the end of the shard,
//! also ordered by expert. So an expert costs **two aligned reads** (the extents are
//! computed from the safetensors headers, not assumed, so another layout only costs more
//! reads). Both are rounded out to 4 KiB for `O_DIRECT`.
//!
//! The reader converts the GPTQ words in RAM, on the reader thread, into exactly the
//! record the pack stores ([`super::pack::format`]): `[N, K]` signed int8 codes
//! (`q ^ 0x80`) and `[N, K/group]` f16 scales, gate|up fused. The conversion is
//! [`autoround::unpack`], the packer's own, so a record read here is **bit-identical** to
//! the same expert read from a `clef-pack` (tested against a real pack when present).
//!
//! Resident tensors (embeddings, norms, router, the int8 attention projections, the lm
//! head) are used as stored: BF16 tensors are mapped and widened on use; the attention
//! `qweight`s are unpacked per use (`resident_linear`), or once at load for the NPU.

use super::engine::{ClefStore, ExpertReader, fatal_read};
use super::model::{
    ExpertSource, ExpertWeights, LayerSmall, LayerWeights, TextConfig, WeightSource,
};
use super::pack::autoround::{self, QLinear, Source};
use super::pack::format::{ALIGN, Dims, ExpertRecord, ExpertRecordMut, RecordLayout, round_up};
use super::pack::reader::{AlignedBuf, DiskFile, dequant_expert, dequant_row};
use super::st::{Dtype, f16_to_f32};
use anyhow::{Context, Result, bail};
use std::borrow::Cow;
use std::cell::RefCell;
use std::path::{Path, PathBuf};

const INDEX: &str = "model.safetensors.index.json";
/// Two extents closer than this are read as one (the gap is cheaper than a second I/O).
const MERGE_GAP: u64 = 64 * 1024;
const MODULES: [&str; 3] = ["gate_proj", "up_proj", "down_proj"];
const KINDS: [&str; 3] = ["qweight", "qzeros", "scales"];

thread_local! {
    /// Record buffer of [`AutoRoundWeights::expert`] (the CPU path).
    static EXPERT_BUF: RefCell<Option<AlignedBuf>> = const { RefCell::new(None) };
}

/// One planned aligned read: the tensor bytes `[start, end)` of shard `shard`, rounded
/// out to `[a0, a1)` for `O_DIRECT`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Extent {
    shard: usize,
    start: u64,
    end: u64,
}

impl Extent {
    fn a0(&self) -> u64 {
        self.start & !(ALIGN as u64 - 1)
    }
    fn a1(&self) -> u64 {
        round_up(self.end as usize, ALIGN) as u64
    }
}

/// Merge `(shard, start, end)` ranges (any order) into read extents.
fn plan_extents(mut locs: Vec<(usize, u64, u64)>) -> Vec<Extent> {
    locs.sort_unstable();
    let mut out: Vec<Extent> = Vec::new();
    for (shard, start, end) in locs {
        match out.last_mut() {
            Some(e) if e.shard == shard && start <= e.end + MERGE_GAP => e.end = e.end.max(end),
            _ => out.push(Extent { shard, start, end }),
        }
    }
    out
}

/// The AutoRound shards opened for streaming.
pub struct ShardStore {
    src: Source,
    dims: Dims,
    layout: RecordLayout,
    layers: usize,
    experts: usize,
    disk: Vec<DiskFile>,
}

impl ShardStore {
    /// Whether `dir` is an AutoRound checkpoint this store reads (its index has the
    /// quantized expert tensors).
    pub fn is_autoround(dir: &Path) -> bool {
        std::fs::read(dir.join(INDEX))
            .ok()
            .and_then(|b| String::from_utf8(b).ok())
            .is_some_and(|s| s.contains("experts.0.gate_proj.qweight"))
    }

    /// Open and validate: the index, every shard file (existence and exact size), and
    /// the presence, dtype and shape of every tensor the text path reads. A failure
    /// names the problem; callers exit non-zero.
    pub fn open(dir: &Path) -> Result<Self> {
        let cfg = TextConfig::from_model_dir(dir)?;
        let group = group_size(dir)?;
        let dims = Dims {
            hidden: cfg.hidden,
            inter: cfg.inter,
            group,
        };
        dims.validate()?;
        let src = Source::open_filtered(dir, |n| n.starts_with("thinker."))
            .with_context(|| format!("open the AutoRound index/shards in {}", dir.display()))?;
        let mut store = Self {
            dims,
            layout: dims.layout(),
            layers: cfg.layers,
            experts: cfg.experts,
            disk: Vec::new(),
            src,
        };
        store.validate(&cfg, dir)?;
        store.disk = store
            .src
            .shard_names()
            .iter()
            .map(|n| DiskFile::open(&dir.join(n)))
            .collect::<Result<_>>()?;
        Ok(store)
    }

    fn validate(&self, cfg: &TextConfig, dir: &Path) -> Result<()> {
        for (name, f) in self.src.shard_names().iter().zip(self.src.shard_files()) {
            let path: PathBuf = dir.join(name);
            let len = std::fs::metadata(&path)
                .with_context(|| format!("shard {}: missing", path.display()))?
                .len() as usize;
            if len != f.expected_len() {
                bail!(
                    "shard {} has {len} bytes but its header describes {} (truncated or \
                     incomplete download?)",
                    path.display(),
                    f.expected_len()
                );
            }
        }
        let (h, d, g) = (cfg.hidden, cfg.head_dim, self.dims.group);
        let check = |name: &str, dtype: Dtype, shape: &[usize]| -> Result<()> {
            let t = self.src.info(name)?;
            if t.dtype != dtype || t.shape != shape {
                bail!(
                    "{name}: {:?} {:?} in the shards, expected {dtype:?} {shape:?}",
                    t.dtype,
                    t.shape
                );
            }
            Ok(())
        };
        let qlin = |m: &str, n: usize, k: usize| -> Result<()> {
            check(&format!("{m}.qweight"), Dtype::I32, &[k / 4, n])?;
            check(&format!("{m}.qzeros"), Dtype::I32, &[k / g, n / 4])?;
            check(&format!("{m}.scales"), Dtype::F16, &[k / g, n])
        };
        for l in 0..cfg.layers {
            let p = format!("thinker.model.layers.{l}.");
            check(&format!("{p}input_layernorm.weight"), Dtype::Bf16, &[h])?;
            check(
                &format!("{p}post_attention_layernorm.weight"),
                Dtype::Bf16,
                &[h],
            )?;
            check(&format!("{p}self_attn.q_norm.weight"), Dtype::Bf16, &[d])?;
            check(&format!("{p}self_attn.k_norm.weight"), Dtype::Bf16, &[d])?;
            check(
                &format!("{p}mlp.gate.weight"),
                Dtype::Bf16,
                &[cfg.experts, h],
            )?;
            qlin(&format!("{p}self_attn.q_proj"), cfg.heads * d, h)?;
            qlin(&format!("{p}self_attn.k_proj"), cfg.kv_heads * d, h)?;
            qlin(&format!("{p}self_attn.v_proj"), cfg.kv_heads * d, h)?;
            qlin(&format!("{p}self_attn.o_proj"), h, cfg.heads * d)?;
            for e in 0..cfg.experts {
                let m = format!("{p}mlp.experts.{e}.");
                qlin(&format!("{m}gate_proj"), cfg.inter, h)?;
                qlin(&format!("{m}up_proj"), cfg.inter, h)?;
                qlin(&format!("{m}down_proj"), h, cfg.inter)?;
            }
        }
        for n in [
            "thinker.model.embed_tokens.weight",
            "thinker.lm_head.weight",
        ] {
            self.src.info(n)?;
        }
        check("thinker.model.norm.weight", Dtype::Bf16, &[h])
    }

    pub fn dims(&self) -> Dims {
        self.dims
    }

    pub fn is_direct(&self) -> bool {
        self.disk.first().is_some_and(DiskFile::is_direct)
    }

    pub fn source(&self) -> &Source {
        &self.src
    }

    /// The read extents of expert `e` of `layer` (2 on the real checkpoint).
    fn plan(&self, layer: usize, e: usize) -> Result<Vec<Extent>> {
        let p = format!("thinker.model.layers.{layer}.mlp.experts.{e}.");
        let mut locs = Vec::with_capacity(9);
        for m in MODULES {
            for k in KINDS {
                let l = self.src.locate(&format!("{p}{m}.{k}"))?;
                locs.push((l.shard, l.start, l.end));
            }
        }
        Ok(plan_extents(locs))
    }

    /// Number of aligned reads expert `e` of `layer` takes.
    pub fn reads_per_expert(&self, layer: usize, e: usize) -> Result<usize> {
        Ok(self.plan(layer, e)?.len())
    }

    /// Bytes a buffer needs: the record plus the raw extents of any expert.
    pub fn buffer_bytes(&self) -> usize {
        let mut raw = 0usize;
        for l in 0..self.layers {
            for e in [0, self.experts - 1] {
                if let Ok(x) = self.plan(l, e) {
                    raw = raw.max(x.iter().map(|x| (x.a1() - x.a0()) as usize).sum());
                }
            }
        }
        self.layout.stride + raw + 4 * ALIGN
    }

    /// Phase 1 (I/O only): read the aligned extents of expert `e` of `layer` into the
    /// tail of `buf` (after the `stride` bytes the record takes).
    pub fn read_raw(&self, layer: usize, e: usize, buf: &mut AlignedBuf) -> Result<()> {
        if layer >= self.layers || e >= self.experts {
            bail!("layer {layer} expert {e} out of range");
        }
        let stride = self.layout.stride;
        let extents = self.plan(layer, e)?;
        let mut off = stride;
        for x in &extents {
            let len = (x.a1() - x.a0()) as usize;
            if buf.len() < off + len {
                bail!("buffer of {} bytes < {} needed", buf.len(), off + len);
            }
            let name = &self.src.shard_names()[x.shard];
            let got = self.disk[x.shard]
                .read_at(&mut buf.as_mut_slice()[off..off + len], x.a0())
                .with_context(|| format!("read shard {name}"))?;
            // The last extent of a shard may run past EOF (rounded to 4 KiB).
            if (got as u64) < x.end - x.a0() {
                bail!(
                    "short read of shard {name} at {}: {got} of {} bytes",
                    x.a0(),
                    x.end - x.a0()
                );
            }
            off += len;
        }
        Ok(())
    }

    /// Phase 2 (CPU only): unpack the GPTQ words read by [`Self::read_raw`] into the
    /// pack record layout at the head of `buf`.
    pub fn convert(&self, layer: usize, e: usize, buf: &mut AlignedBuf) -> Result<()> {
        let stride = self.layout.stride;
        let d = self.dims;
        let extents = self.plan(layer, e)?;
        let mut offs = Vec::with_capacity(extents.len());
        let mut off = stride;
        for x in &extents {
            offs.push(off);
            off += (x.a1() - x.a0()) as usize;
        }
        let (head, raw) = buf.as_mut_slice().split_at_mut(stride);
        let p = format!("thinker.model.layers.{layer}.mlp.experts.{e}.");
        let tensor = |m: &str, k: &str| -> Result<&[u8]> {
            let l = self.src.locate(&format!("{p}{m}.{k}"))?;
            let (i, x) = extents
                .iter()
                .enumerate()
                .find(|(_, x)| x.shard == l.shard && x.start <= l.start && l.end <= x.end)
                .context("tensor outside every planned extent")?;
            let o = offs[i] - stride + (l.start - x.a0()) as usize;
            Ok(&raw[o..o + (l.end - l.start) as usize])
        };
        let ql = |m: &str, n: usize, k: usize| -> Result<QLinear<'_>> {
            Ok(QLinear {
                qweight: tensor(m, "qweight")?,
                qzeros: tensor(m, "qzeros")?,
                scales: tensor(m, "scales")?,
                n,
                k,
                group: d.group,
            })
        };
        let rec = ExpertRecordMut::split(d, head);
        let gu_n = d.inter * d.hidden;
        let gu_s = d.inter * (d.hidden / d.group) * 2;
        let (g_codes, u_codes) = rec.gu_codes.split_at_mut(gu_n);
        let (g_sc, u_sc) = rec.gu_scales.split_at_mut(gu_s);
        for (m, codes, sc) in [("gate_proj", g_codes, g_sc), ("up_proj", u_codes, u_sc)] {
            autoround::unpack(&ql(m, d.inter, d.hidden)?, codes, sc)
                .with_context(|| format!("{p}{m}"))?;
        }
        autoround::unpack(
            &ql("down_proj", d.hidden, d.inter)?,
            rec.down_codes,
            rec.down_scales,
        )
        .with_context(|| format!("{p}down_proj"))
    }

    /// Read expert `e` of `layer` into `buf` as a pack record (`stride` bytes): the
    /// aligned extents are read, then the GPTQ words are unpacked into the record layout.
    pub fn read_expert(&self, layer: usize, e: usize, buf: &mut AlignedBuf) -> Result<()> {
        self.read_raw(layer, e, buf)?;
        self.convert(layer, e, buf)
    }

    /// Widen a small BF16/F16/F32 tensor to f32 (shape checked).
    pub fn f32(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let (f, t) = self.src.get(name)?;
        if t.shape != shape {
            bail!("{name}: shape {:?}, expected {shape:?}", t.shape);
        }
        Ok(f.to_f32(t))
    }

    /// `<module>` unpacked to `[n, k]` int8 codes and `[n, k/group]` f16 scale bytes.
    fn unpack_linear(&self, module: &str, n: usize, k: usize) -> Result<(Vec<i8>, Vec<u8>)> {
        let g = self.dims.group;
        let q = self.src.qlinear(module, n, k, g)?;
        let mut codes = vec![0i8; n * k];
        let mut scales = vec![0u8; n * (k / g) * 2];
        autoround::unpack(&q, &mut codes, &mut scales).with_context(|| module.to_string())?;
        Ok((codes, scales))
    }
}

/// `quantization_config.group_size` of `config.json` (8-bit symmetric only).
fn group_size(dir: &Path) -> Result<usize> {
    let p = dir.join("config.json");
    let v: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&p).with_context(|| format!("read {}", p.display()))?,
    )?;
    let q = &v["quantization_config"];
    if q["bits"].as_u64() != Some(8) || q["sym"].as_bool() != Some(true) {
        bail!(
            "{}: quantization_config is not 8-bit symmetric AutoRound",
            p.display()
        );
    }
    Ok(q["group_size"]
        .as_u64()
        .context("quantization_config.group_size")? as usize)
}

impl ExpertReader for ShardStore {
    fn dims(&self) -> Dims {
        self.dims
    }
    fn layers(&self) -> usize {
        self.layers
    }
    fn experts(&self) -> usize {
        self.experts
    }
    fn new_buffer(&self) -> AlignedBuf {
        AlignedBuf::new(self.buffer_bytes())
    }
    fn read(&self, layer: usize, expert: usize, buf: &mut AlignedBuf) -> Result<()> {
        if let Err(e) = self.read_raw(layer, expert, buf) {
            fatal_read(layer, expert, &e);
        }
        Ok(())
    }
    fn needs_finish(&self) -> bool {
        true
    }
    fn finish(&self, layer: usize, expert: usize, buf: &mut AlignedBuf) -> Result<()> {
        if let Err(e) = self.convert(layer, expert, buf) {
            fatal_read(layer, expert, &e);
        }
        Ok(())
    }
    fn release_resident(&self) {
        self.src.release();
    }
}

impl ClefStore for ShardStore {
    fn resident_linear(
        &self,
        module: &str,
        n: usize,
        k: usize,
    ) -> Result<(Cow<'_, [i8]>, Vec<f32>)> {
        let (codes, sb) = self.unpack_linear(module, n, k)?;
        let scales = sb
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f16_to_f32(u16::from_le_bytes(*c)))
            .collect();
        Ok((Cow::Owned(codes), scales))
    }

    fn dequant_linear(&self, module: &str, n: usize, k: usize) -> Result<Vec<f32>> {
        use rayon::prelude::*;
        let g = self.dims.group;
        let (codes, scales) = self.unpack_linear(module, n, k)?;
        // SAFETY: i8 and u8 share a layout.
        let cb = unsafe { std::slice::from_raw_parts(codes.as_ptr().cast::<u8>(), codes.len()) };
        let mut w = vec![0f32; n * k];
        w.par_chunks_mut(k).enumerate().for_each(|(row, out)| {
            let s = &scales[row * (k / g) * 2..(row + 1) * (k / g) * 2];
            dequant_row(&cb[row * k..(row + 1) * k], s, g, out);
        });
        Ok(w)
    }
}

/// The backbone's whole weight source on the AutoRound shards (the f32 CPU path and the
/// small tensors of the engine paths): experts are read as pack records and widened.
pub struct AutoRoundWeights {
    pub store: ShardStore,
    cfg: TextConfig,
}

impl AutoRoundWeights {
    pub fn open(dir: &Path, cfg: TextConfig) -> Result<Self> {
        let store = ShardStore::open(dir)?;
        Ok(Self { store, cfg })
    }
}

impl ExpertSource for AutoRoundWeights {
    fn expert(&self, layer: usize, e: usize) -> Result<ExpertWeights> {
        // Take the buffer out of the thread-local instead of holding a `RefCell` borrow:
        // `dequant_expert` runs nested rayon work, and a worker waiting on it may steal
        // another expert task of this same loop, which re-enters this function on the same
        // thread (that re-entry used to panic with "RefCell already borrowed"; it now simply
        // allocates its own buffer).
        let stride = self.store.layout.stride;
        let mut buf = EXPERT_BUF
            .with(|cell| cell.borrow_mut().take())
            .filter(|b| b.len() >= stride)
            .unwrap_or_else(|| self.store.new_buffer());
        let out = (|| {
            self.store.read_expert(layer, e, &mut buf)?;
            let rec = ExpertRecord::parse(self.store.dims, buf.as_slice())?;
            Ok(dequant_expert(&rec))
        })();
        EXPERT_BUF.with(|cell| *cell.borrow_mut() = Some(buf));
        out
    }
}

impl WeightSource for AutoRoundWeights {
    fn layer_weights(&self, l: usize) -> Result<LayerWeights> {
        let (c, s) = (&self.cfg, &self.store);
        let (h, d) = (c.hidden, c.head_dim);
        let pre = format!("thinker.model.layers.{l}.");
        let f = |n: &str, shape: &[usize]| s.f32(&format!("{pre}{n}"), shape);
        let q = |n: &str, o: usize, i: usize| {
            ClefStore::dequant_linear(s, &format!("{pre}self_attn.{n}"), o, i)
        };
        Ok(LayerWeights {
            input_norm: f("input_layernorm.weight", &[h])?,
            post_norm: f("post_attention_layernorm.weight", &[h])?,
            q: q("q_proj", c.heads * d, h)?,
            k: q("k_proj", c.kv_heads * d, h)?,
            v: q("v_proj", c.kv_heads * d, h)?,
            o: q("o_proj", h, c.heads * d)?,
            q_norm: f("self_attn.q_norm.weight", &[d])?,
            k_norm: f("self_attn.k_norm.weight", &[d])?,
            router: f("mlp.gate.weight", &[c.experts, h])?,
        })
    }

    fn layer_small(&self, l: usize) -> Result<LayerSmall> {
        let (c, s) = (&self.cfg, &self.store);
        let (h, d) = (c.hidden, c.head_dim);
        let pre = format!("thinker.model.layers.{l}.");
        let f = |n: &str, shape: &[usize]| s.f32(&format!("{pre}{n}"), shape);
        Ok(LayerSmall {
            input_norm: f("input_layernorm.weight", &[h])?,
            post_norm: f("post_attention_layernorm.weight", &[h])?,
            q_norm: f("self_attn.q_norm.weight", &[d])?,
            k_norm: f("self_attn.k_norm.weight", &[d])?,
            router: f("mlp.gate.weight", &[c.experts, h])?,
        })
    }

    fn experts(&self) -> &dyn ExpertSource {
        self
    }

    fn embed_row(&self, id: u32, out: &mut [f32]) -> Result<()> {
        let (f, t) = self.store.src.get("thinker.model.embed_tokens.weight")?;
        f.row_f32(t, id as usize, out)
    }

    fn final_norm(&self) -> Result<Vec<f32>> {
        self.store
            .f32("thinker.model.norm.weight", &[self.cfg.hidden])
    }

    fn release(&self) {
        self.store.src.release();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adjacent_tensors_merge_and_distant_ones_stay_apart() {
        // the real expert: six words/zeros tensors back to back, scales far away
        let (a, z) = (1_572_864u64, 12_288u64);
        let base = 1_264_488u64;
        let sc = 5_287_779_688u64;
        let mut locs = Vec::new();
        for i in 0..3u64 {
            locs.push((0usize, base + i * (a + z), base + i * (a + z) + a));
            locs.push((0usize, base + i * (a + z) + a, base + (i + 1) * (a + z)));
            locs.push((0usize, sc + i * 24_576, sc + (i + 1) * 24_576));
        }
        let x = plan_extents(locs);
        assert_eq!(x.len(), 2);
        assert_eq!((x[0].start, x[0].end), (base, base + 3 * (a + z)));
        assert_eq!((x[1].start, x[1].end), (sc, sc + 73_728));
        for e in &x {
            assert_eq!(e.a0() % ALIGN as u64, 0);
            assert_eq!(e.a1() % ALIGN as u64, 0);
            assert!(e.a0() <= e.start && e.end <= e.a1());
            assert!(e.a1() - e.end < ALIGN as u64 && e.start - e.a0() < ALIGN as u64);
        }
        // different shards never merge
        assert_eq!(plan_extents(vec![(0, 0, 10), (1, 10, 20)]).len(), 2);
    }
}
