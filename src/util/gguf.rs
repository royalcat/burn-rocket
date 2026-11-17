//! Minimal GGUF access: parse the metadata/tensor table with `gguf-rs`, memory-map
//! the file ourselves, and dequantize tensor bytes on demand.
//!
//! GGUF stores every tensor with its dimensions in `ne[0..]` order, where `ne[0]`
//! is the fastest-varying axis — the reverse of the HF/PyTorch row-major
//! convention (`[out, in]` for a linear, `[vocab, hidden]` for an embedding). This
//! module normalizes tensors to the HF/PyTorch convention so the rest of the
//! loader stack (the `PyTorchToBurnAdapter` and the per-family loaders) is
//! unchanged between safetensors and GGUF checkpoints.
//!
//! `gguf-rs`'s own `MmapGGUF` copies the whole file to parse it (see its source),
//! so we only use its streaming parser for the header and mmap the file ourselves
//! for zero-copy access to tensor bytes.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use burn::tensor::{TensorData, bf16, f16};
use memmap2::Mmap;

/// A parsed GGUF file: metadata, tensor table, and a memory map of the bytes.
pub struct GgufFile {
    model: gguf_rs::GGUFModel,
    mmap: Mmap,
}

/// GGML tensor type ids we understand (see `gguf_rs::GGMLType`).
pub mod kind {
    pub const F32: u32 = 0;
    pub const F16: u32 = 1;
    pub const Q8_0: u32 = 8;
    pub const I32: u32 = 26;
    pub const F64: u32 = 28;
    pub const BF16: u32 = 30;
}

impl GgufFile {
    /// Parse `path` (metadata + tensor table) and map its bytes.
    pub fn open(path: &Path) -> Result<Self> {
        let path_str = path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("non-UTF-8 GGUF path: {}", path.display()))?;
        // u64::MAX keeps the full tokenizer arrays (the plain helper truncates to 3).
        let mut container = gguf_rs::get_gguf_container_array_size(path_str, u64::MAX)
            .with_context(|| format!("open GGUF {}", path.display()))?;
        let model = container
            .decode()
            .with_context(|| format!("decode GGUF {}", path.display()))?;
        let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
        // SAFETY: the map is read-only and the file is not mutated while mapped.
        let mmap =
            unsafe { Mmap::map(&file) }.with_context(|| format!("mmap {}", path.display()))?;
        Ok(Self { model, mmap })
    }

    /// All metadata key/value pairs (`general.*`, `tokenizer.ggml.*`, per-arch keys).
    pub fn metadata(&self) -> &BTreeMap<String, serde_json::Value> {
        self.model.metadata()
    }

    /// `general.architecture`, when present.
    pub fn arch(&self) -> Option<&str> {
        self.metadata()
            .get("general.architecture")
            .and_then(serde_json::Value::as_str)
    }

    /// The GGUF tensor names, in file order.
    pub fn names(&self) -> Vec<String> {
        self.model
            .tensors()
            .iter()
            .map(|t| t.name.clone())
            .collect()
    }

    fn tensor(&self, name: &str) -> Option<&gguf_rs::Tensor> {
        self.model.tensors().iter().find(|t| t.name == name)
    }

    /// Raw little-endian tensor bytes (a slice of the memory map).
    pub fn raw(&self, name: &str) -> Result<Option<&[u8]>> {
        let Some(t) = self.tensor(name) else {
            return Ok(None);
        };
        let n: usize = t.shape.iter().map(|&d| d as usize).product();
        // NB: `gguf-rs`'s `Tensor::size` is wrong for unquantized types above
        // kind 9 (it divides by a 256 block size), so compute the length here.
        let len = byte_len(t.kind, n).with_context(|| format!("tensor '{name}'"))?;
        let start = (self.model.data_offset() + t.offset) as usize;
        let end = start + len;
        self.mmap
            .get(start..end)
            .map(Some)
            .with_context(|| format!("tensor '{name}' slice {start}..{end} out of range"))
    }

    /// The tensor's on-disk shape (GGUF `ne[]` order) and element type id.
    pub fn shape_and_kind(&self, name: &str) -> Result<Option<(Vec<usize>, u32)>> {
        let Some(t) = self.tensor(name) else {
            return Ok(None);
        };
        let shape = t.shape.iter().map(|&d| d as usize).collect();
        Ok(Some((shape, t.kind)))
    }

    /// Dequantize one tensor into f32 values (GGUF element order).
    pub fn values_f32(&self, name: &str) -> Result<Option<Vec<f32>>> {
        let Some((shape, k)) = self.shape_and_kind(name)? else {
            return Ok(None);
        };
        let n: usize = shape.iter().product();
        let raw = self.raw(name)?.expect("tensor exists");
        Ok(Some(
            dequantize(raw, k, n).with_context(|| format!("dequantize '{name}'"))?,
        ))
    }
}

/// The on-disk byte length of `n` values of the given GGML type id.
///
/// Computed here because `gguf-rs`'s `Tensor::size` divides unquantized types
/// above kind 9 by a 256 block size (wrong for BF16/F64/I32/I64).
pub fn byte_len(kind: u32, n: usize) -> Result<usize> {
    Ok(match kind {
        kind::F32 | kind::I32 => n * 4,
        kind::F16 | kind::BF16 => n * 2,
        kind::F64 => n * 8,
        kind::Q8_0 => {
            if !n.is_multiple_of(32) {
                bail!("Q8_0 tensor with {n} values (not a multiple of 32)");
            }
            (n / 32) * 34
        }
        other => {
            bail!("unsupported GGML tensor type id {other} (supported: F32/F16/BF16/F64/I32/Q8_0)")
        }
    })
}

/// Dequantize `raw` into `n` f32 values for the given GGML type id.
pub fn dequantize(raw: &[u8], kind: u32, n: usize) -> Result<Vec<f32>> {
    match kind {
        kind::F32 => {
            let need = n * 4;
            if raw.len() < need {
                bail!("F32 tensor: {} bytes < {} needed", raw.len(), need);
            }
            let (chunks, _) = raw[..need].as_chunks::<4>();
            Ok(chunks.iter().map(|b| f32::from_le_bytes(*b)).collect())
        }
        kind::F16 => {
            let need = n * 2;
            if raw.len() < need {
                bail!("F16 tensor: {} bytes < {} needed", raw.len(), need);
            }
            let (chunks, _) = raw[..need].as_chunks::<2>();
            Ok(chunks
                .iter()
                .map(|b| f16::from_bits(u16::from_le_bytes(*b)).to_f32())
                .collect())
        }
        kind::BF16 => {
            let need = n * 2;
            if raw.len() < need {
                bail!("BF16 tensor: {} bytes < {} needed", raw.len(), need);
            }
            let (chunks, _) = raw[..need].as_chunks::<2>();
            Ok(chunks
                .iter()
                .map(|b| bf16::from_bits(u16::from_le_bytes(*b)).to_f32())
                .collect())
        }
        kind::F64 => {
            let need = n * 8;
            if raw.len() < need {
                bail!("F64 tensor: {} bytes < {} needed", raw.len(), need);
            }
            let (chunks, _) = raw[..need].as_chunks::<8>();
            Ok(chunks
                .iter()
                .map(|b| f64::from_le_bytes(*b) as f32)
                .collect())
        }
        kind::I32 => {
            let need = n * 4;
            if raw.len() < need {
                bail!("I32 tensor: {} bytes < {} needed", raw.len(), need);
            }
            let (chunks, _) = raw[..need].as_chunks::<4>();
            Ok(chunks
                .iter()
                .map(|b| i32::from_le_bytes(*b) as f32)
                .collect())
        }
        kind::Q8_0 => {
            // llama.cpp Q8_0: 32-value blocks of {f16 scale, int8 q[32]} = 34 bytes.
            if !n.is_multiple_of(32) {
                bail!("Q8_0 tensor with {n} values (not a multiple of 32)");
            }
            let blocks = n / 32;
            let need = blocks * 34;
            if raw.len() < need {
                bail!("Q8_0 tensor: {} bytes < {} needed", raw.len(), need);
            }
            let (blocks, _) = raw[..need].as_chunks::<34>();
            let mut out = Vec::with_capacity(n);
            for blk in blocks {
                let d = f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])).to_f32();
                for &q in &blk[2..34] {
                    out.push(d * (q as i8) as f32);
                }
            }
            Ok(out)
        }
        other => {
            bail!("unsupported GGML tensor type id {other} (supported: F32/F16/BF16/F64/I32/Q8_0)")
        }
    }
}

/// Reverse a GGUF `ne[]` shape to the HF/PyTorch order.
///
/// GGUF stores dimensions with `ne[0]` fastest, and HF/PyTorch row-major stores
/// the last dimension fastest — the *same* memory order. So converting a GGUF
/// tensor to the HF convention reverses the shape labels but leaves the element
/// bytes untouched (no permutation).
pub fn hf_shape(dims: &[usize]) -> Vec<usize> {
    dims.iter().rev().copied().collect()
}

/// Build a `TensorData` from f32 `values` with `dims` (rank 0..=4).
pub fn tensor_data_f32(values: Vec<f32>, dims: &[usize]) -> Result<TensorData> {
    Ok(match dims {
        [a] => TensorData::new(values, [*a]),
        [a, b] => TensorData::new(values, [*a, *b]),
        [a, b, c] => TensorData::new(values, [*a, *b, *c]),
        [a, b, c, d] => TensorData::new(values, [*a, *b, *c, *d]),
        [] => TensorData::new(values, [1]),
        other => bail!("unsupported tensor rank {}", other.len()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_0_block_dequant() {
        // One block: scale 0.5, q = [-1, 0, 1, ...].
        let mut raw = vec![0u8; 34];
        raw[0..2].copy_from_slice(&f16::from_f32(0.5).to_bits().to_le_bytes());
        raw[2] = 0xff; // -1
        raw[3] = 0x00; // 0
        raw[4] = 0x01; // 1
        let v = dequantize(&raw, kind::Q8_0, 32).unwrap();
        assert_eq!(v[0], -0.5);
        assert_eq!(v[1], 0.0);
        assert_eq!(v[2], 0.5);
    }

    #[test]
    fn hf_shape_reverses_labels() {
        assert_eq!(hf_shape(&[2, 3]), vec![3, 2]);
        assert_eq!(hf_shape(&[4, 5, 6]), vec![6, 5, 4]);
    }
}
