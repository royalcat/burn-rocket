//! Minimal read-only safetensors access (mmap + header), enough for the joint head
//! and for lazily reading single rows of a huge tensor (`thinker.lm_head.weight`).

use anyhow::{Context, Result, bail};
use memmap2::Mmap;
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32,
    Bf16,
    F16,
    /// Signed 8-bit codes (the pack's quantized weights).
    I8,
    /// Packed GPTQ/AutoRound words (four 8-bit values per word).
    I32,
}

impl Dtype {
    pub fn size(self) -> usize {
        match self {
            Dtype::F32 | Dtype::I32 => 4,
            Dtype::Bf16 | Dtype::F16 => 2,
            Dtype::I8 => 1,
        }
    }

    /// The safetensors header spelling.
    pub fn name(self) -> &'static str {
        match self {
            Dtype::F32 => "F32",
            Dtype::Bf16 => "BF16",
            Dtype::F16 => "F16",
            Dtype::I8 => "I8",
            Dtype::I32 => "I32",
        }
    }

    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "F32" => Dtype::F32,
            "BF16" => Dtype::Bf16,
            "F16" => Dtype::F16,
            "I8" => Dtype::I8,
            "I32" => Dtype::I32,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub start: usize,
    pub end: usize,
}

impl TensorInfo {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

pub struct SafeFile {
    mmap: Mmap,
    data_start: usize,
    pub tensors: BTreeMap<String, TensorInfo>,
}

impl SafeFile {
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
        // SAFETY: read-only mapping of a checkpoint that is not modified while loaded.
        let mmap = unsafe { Mmap::map(&file)? };
        Self::from_mmap(mmap).with_context(|| path.display().to_string())
    }

    fn from_mmap(mmap: Mmap) -> Result<Self> {
        if mmap.len() < 8 {
            bail!("safetensors file shorter than its length prefix");
        }
        let n = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
        if 8 + n > mmap.len() {
            bail!("safetensors header length {n} exceeds the file");
        }
        let header: serde_json::Value = serde_json::from_slice(&mmap[8..8 + n])?;
        let obj = header.as_object().context("header is not an object")?;
        let data_start = 8 + n;
        let data_len = mmap.len() - data_start;
        let mut tensors = BTreeMap::new();
        for (name, v) in obj {
            if name == "__metadata__" {
                continue;
            }
            let dtype_name = v["dtype"].as_str().unwrap_or("");
            let Some(dtype) = Dtype::parse(dtype_name) else {
                bail!("{name}: unsupported dtype {dtype_name}");
            };
            let shape: Vec<usize> = v["shape"]
                .as_array()
                .with_context(|| format!("{name}: no shape"))?
                .iter()
                .map(|d| d.as_u64().map(|d| d as usize))
                .collect::<Option<_>>()
                .with_context(|| format!("{name}: bad shape"))?;
            let off = v["data_offsets"]
                .as_array()
                .with_context(|| format!("{name}: no data_offsets"))?;
            let (start, end) = (
                off[0].as_u64().context("offset")? as usize,
                off[1].as_u64().context("offset")? as usize,
            );
            let info = TensorInfo {
                dtype,
                shape,
                start,
                end,
            };
            if end > data_len || start > end || end - start != info.numel() * dtype.size() {
                bail!("{name}: inconsistent data offsets");
            }
            tensors.insert(name.clone(), info);
        }
        Ok(Self {
            mmap,
            data_start,
            tensors,
        })
    }

    /// Absolute file offset of the first data byte (`8 + header length`).
    pub fn data_start(&self) -> usize {
        self.data_start
    }

    /// The file length this header describes (`data_start` + the end of the last tensor);
    /// a different on-disk length means a truncated or foreign file.
    pub fn expected_len(&self) -> usize {
        self.data_start + self.tensors.values().map(|t| t.end).max().unwrap_or(0)
    }

    pub fn bytes(&self, info: &TensorInfo) -> &[u8] {
        &self.mmap[self.data_start + info.start..self.data_start + info.end]
    }

    /// Tell the kernel the mapped pages are no longer needed (drops them from this
    /// process's RSS; the page cache keeps them). Safe to call at any time: the mapping
    /// is a read-only file map, so the pages are simply re-faulted on the next access.
    pub fn release(&self) {
        // SAFETY: `DontNeed` on a read-only file mapping only discards resident pages.
        unsafe {
            let _ = self
                .mmap
                .unchecked_advise(memmap2::UncheckedAdvice::DontNeed);
        }
    }

    /// Widen a whole tensor to f32.
    pub fn to_f32(&self, info: &TensorInfo) -> Vec<f32> {
        widen(info.dtype, self.bytes(info))
    }

    /// Widen row `row` of a 2-D tensor into `out` (length = columns).
    pub fn row_f32(&self, info: &TensorInfo, row: usize, out: &mut [f32]) -> Result<()> {
        if info.shape.len() != 2 || out.len() != info.shape[1] || row >= info.shape[0] {
            bail!("row {row} out of range for shape {:?}", info.shape);
        }
        let cols = info.shape[1];
        let sz = info.dtype.size();
        let b = &self.bytes(info)[row * cols * sz..(row + 1) * cols * sz];
        out.copy_from_slice(&widen(info.dtype, b));
        Ok(())
    }
}

pub fn widen(dtype: Dtype, b: &[u8]) -> Vec<f32> {
    match dtype {
        Dtype::F32 => b
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        Dtype::Bf16 => b
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16))
            .collect(),
        Dtype::F16 => b
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f16_to_f32(u16::from_le_bytes(*c)))
            .collect(),
        Dtype::I8 => b.iter().map(|&c| f32::from(c as i8)).collect(),
        Dtype::I32 => b
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_le_bytes(*c) as f32)
            .collect(),
    }
}

pub(crate) fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = match (exp, man) {
        (0, 0) => sign,
        (0, m) => {
            // subnormal: normalize
            let mut e = 127 - 15 + 1;
            let mut m = m;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
        (0x1f, 0) => sign | 0x7f80_0000,
        (0x1f, m) => sign | 0x7f80_0000 | (m << 13),
        (e, m) => sign | ((e + 127 - 15) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_and_f16_widen_exactly() {
        assert_eq!(widen(Dtype::Bf16, &[0x80, 0x3f]), vec![1.0]);
        assert_eq!(widen(Dtype::Bf16, &[0x20, 0xc0]), vec![-2.5]);
        assert_eq!(widen(Dtype::F16, &[0x00, 0x3c]), vec![1.0]);
        assert_eq!(widen(Dtype::F16, &[0x00, 0xc1]), vec![-2.5]);
        assert_eq!(widen(Dtype::F16, &[0x01, 0x00]), vec![2f32.powi(-24)]);
        assert_eq!(widen(Dtype::F32, &1.5f32.to_le_bytes()), vec![1.5]);
    }
}
