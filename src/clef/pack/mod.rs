//! The Clef pack: the AutoRound int8 checkpoint repacked for streaming (phase 5).
//!
//! * [`format`]: layer-file header, expert record layout, alignment (documented in
//!   `docs/experiments/clef-pack.md`).
//! * [`autoround`]: reading the `moeshawky/clef-omni-int8-autoround` shards and unpacking
//!   GPTQ words into `[N, K]` int8 codes + `[N, K/128]` f16 scales.
//! * [`build`]: the `clef-pack` command (checkpoint dir -> pack dir, `--verify`).
//! * [`reader`]: `PackDir` (O_DIRECT expert reads, resident tensors), `PackedExperts`
//!   (an [`ExpertSource`](super::model::ExpertSource)) and the backbone's pack-backed
//!   weight source.

pub mod autoround;
pub mod build;
pub mod format;
pub mod reader;

#[allow(unused_imports)]
pub use reader::{PackDir, PackedExperts, PackedWeights};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const FORMAT_NAME: &str = "clef-pack";

pub fn sha256_hex(b: &[u8]) -> String {
    hex(&Sha256::digest(b))
}

pub fn hex(b: &[u8]) -> String {
    use std::fmt::Write;
    b.iter()
        .fold(String::with_capacity(b.len() * 2), |mut s, x| {
            let _ = write!(s, "{x:02x}");
            s
        })
}

/// `manifest.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub format: String,
    pub version: u32,
    pub model: ModelInfo,
    pub source: SourceInfo,
    pub expert_record: RecordInfo,
    pub files: Vec<FileInfo>,
    pub resident: ResidentInfo,
    /// The BF16 `thinker.lm_head.weight` (only option-token rows are ever read).
    pub lm_head: LmHeadInfo,
    /// Files copied verbatim from the source (config, tokenizer, joint head).
    pub copied: Vec<String>,
    /// Source tensors deliberately left out (media towers, talker, code2wav).
    pub skipped: Vec<SkippedInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub layers: usize,
    pub hidden: usize,
    pub experts: usize,
    pub moe_intermediate: usize,
    pub group: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub vocab: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceInfo {
    pub repo: String,
    pub index_sha256: String,
    pub bits: u32,
    pub group: usize,
    pub sym: bool,
    pub packing_format: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordInfo {
    pub align: usize,
    pub header_bytes: usize,
    pub record_bytes: usize,
    pub stride: usize,
    pub gu_codes: usize,
    pub down_codes: usize,
    pub gu_scales: usize,
    pub down_scales: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInfo {
    pub name: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorEntry {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    /// Absolute byte offset in the file.
    pub offset: u64,
    pub bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidentInfo {
    pub file: String,
    pub tensors: Vec<TensorEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LmHeadInfo {
    pub file: String,
    pub tensor: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkippedInfo {
    pub prefix: String,
    pub tensors: usize,
    pub bytes: u64,
}

impl Manifest {
    pub fn read(dir: &std::path::Path) -> anyhow::Result<Self> {
        use anyhow::Context;
        let p = dir.join(format::MANIFEST);
        serde_json::from_slice(&std::fs::read(&p).with_context(|| format!("read {}", p.display()))?)
            .with_context(|| format!("parse {}", p.display()))
    }

    /// Structural checks a reader relies on; returns the expert geometry.
    pub fn validate(&self) -> anyhow::Result<format::Dims> {
        use anyhow::bail;
        if self.format != FORMAT_NAME {
            bail!("manifest format `{}` is not `{FORMAT_NAME}`", self.format);
        }
        if self.version != format::FORMAT_VERSION {
            bail!(
                "manifest version {} unsupported (this build reads {})",
                self.version,
                format::FORMAT_VERSION
            );
        }
        let m = &self.model;
        let dims = format::Dims {
            hidden: m.hidden,
            inter: m.moe_intermediate,
            group: m.group,
        };
        dims.validate()?;
        let l = dims.layout();
        let r = &self.expert_record;
        if (r.align, r.header_bytes, r.record_bytes, r.stride)
            != (
                format::ALIGN,
                format::HEADER_BYTES,
                l.record_bytes,
                l.stride,
            )
            || (r.gu_codes, r.down_codes, r.gu_scales, r.down_scales)
                != (l.gu_codes, l.down_codes, l.gu_scales, l.down_scales)
        {
            bail!("manifest expert_record disagrees with the model geometry");
        }
        for layer in 0..m.layers {
            let name = format::layer_file(layer);
            let Some(f) = self.files.iter().find(|f| f.name == name) else {
                bail!("manifest lists no {name}");
            };
            let want = format::LayerHeader::new(layer, m.experts, dims).file_bytes();
            if f.bytes != want {
                bail!("{name}: {} bytes in the manifest, expected {want}", f.bytes);
            }
        }
        let Some(res) = self.files.iter().find(|f| f.name == self.resident.file) else {
            bail!("manifest lists no {}", self.resident.file);
        };
        for t in &self.resident.tensors {
            if t.offset + t.bytes > res.bytes {
                bail!("resident tensor {} lies beyond the file", t.name);
            }
        }
        Ok(dims)
    }
}

/// f32 -> f16 bits, round to nearest even (test helper and the packer's own checks).
#[cfg(test)]
pub(crate) fn f32_to_f16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let man = b & 0x7f_ffff;
    if v == 0.0 {
        return sign;
    }
    assert!(
        (1..31).contains(&exp),
        "test helper handles normal halves only"
    );
    let mut h = (exp as u32) << 10 | (man >> 13);
    let rest = man & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && h & 1 == 1) {
        h += 1;
    }
    sign | h as u16
}

#[cfg(test)]
mod tests;
