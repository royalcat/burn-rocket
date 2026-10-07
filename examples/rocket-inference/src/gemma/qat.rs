//! Native support for the pre-quantized Gemma checkpoints
//! (`google/gemma-4-E2B-it-qat-mobile-transformers`).
//!
//! Reference: `transformers/integrations/gemma_quant.py` (`QuantizedLinear`,
//! `QuantizedEmbedding`, `apply_srq`) and `transformers/quantizers/quantizer_gemma.py`.
//!
//! - Weights are packed INT2 (4 values/byte, bits [1:0]/[3:2]/[5:4]/[7:6]) or
//!   INT4 (2 values/byte, low nibble first), stored unsigned and shifted to
//!   signed; INT8 is stored directly. Dequantization multiplies by a scale:
//!   per-output-channel for linears (`weight_scale` `[N, 1]`), per-row block
//!   scales for embeddings (`embedding_scale` `[rows, blocks]`).
//! - SRQ (static range quantization) rounds activations to an int8 grid around
//!   the affected linears when the checkpoint's scale is calibrated (non-zero).

use std::collections::HashMap;

use anyhow::{Context, Result};
use fancy_regex::Regex;

/// Per-module bit-width selection from `quantization_config`.
#[derive(Debug, Clone)]
pub struct QatConfig {
    default_bits: u8,
    /// `(pattern, bits)` in file order; the first match wins, as in transformers.
    overrides: Vec<(Regex, u8)>,
    not_convert: Vec<Regex>,
}

impl QatConfig {
    /// Parse `config.json`; `None` when the checkpoint is not a pre-quantized
    /// Gemma one.
    pub fn from_json(cfg: &serde_json::Value) -> Result<Option<Self>> {
        let Some(q) = cfg.get("quantization_config") else {
            return Ok(None);
        };
        if q.get("quant_method").and_then(|v| v.as_str()) != Some("gemma") {
            return Ok(None);
        }
        let default_bits = q.get("num_bits").and_then(|v| v.as_u64()).unwrap_or(4) as u8;
        let mut overrides = Vec::new();
        if let Some(map) = q.get("module_quant_configs").and_then(|v| v.as_object()) {
            for (pattern, v) in map {
                let bits = v
                    .get("num_bits")
                    .and_then(|b| b.as_u64())
                    .unwrap_or(default_bits as u64) as u8;
                overrides.push((
                    Regex::new(pattern).with_context(|| format!("bad quant pattern {pattern}"))?,
                    bits,
                ));
            }
        }
        let mut not_convert = Vec::new();
        if let Some(list) = q.get("modules_to_not_convert").and_then(|v| v.as_array()) {
            for v in list {
                if let Some(s) = v.as_str() {
                    not_convert.push(Regex::new(s).with_context(|| format!("bad skip {s}"))?);
                }
            }
        }
        Ok(Some(Self {
            default_bits,
            overrides,
            not_convert,
        }))
    }

    /// `None` when the module is not quantized in this checkpoint.
    pub fn bits_for(&self, module_path: &str) -> Option<u8> {
        // `should_convert_module`: prefix match (optionally followed by a dot),
        // or the path ends with the pattern.
        for p in &self.not_convert {
            let prefix = format!("^(?:{})", p.as_str());
            let with_dot = Regex::new(&format!("{prefix}\\.")).expect("compiled");
            let bare = Regex::new(&prefix).expect("compiled");
            if with_dot.is_match(module_path).unwrap_or(false)
                || bare.is_match(module_path).unwrap_or(false)
                || module_path.ends_with(p.as_str())
            {
                return None;
            }
        }
        for (re, bits) in &self.overrides {
            if re.is_match(module_path).unwrap_or(false) {
                return Some(*bits);
            }
        }
        Some(self.default_bits)
    }
}

/// One quantized module's scales (keyed by module path, without `.weight`).
#[derive(Debug, Clone, Default)]
pub struct ModuleScales {
    /// Row-major `[rows, blocks]`; `blocks == 1` for linears.
    pub weight_scale: Vec<f32>,
    pub blocks: usize,
    pub in_scale: f32,
    pub out_scale: f32,
}

/// Values packed into one byte for the given bit width.
pub fn values_per_byte(bits: u8) -> usize {
    match bits {
        2 => 4,
        4 => 2,
        _ => 1,
    }
}

/// Dequantize a `[rows, packed_k]` integer weight into row-major f32 `[rows, k]`.
pub fn dequantize_weight(
    packed: &[u8],
    bits: u8,
    rows: usize,
    k: usize,
    scales: &ModuleScales,
) -> Vec<f32> {
    let vpb = values_per_byte(bits);
    let packed_k = k.div_ceil(vpb);
    let blocks = scales.blocks.max(1);
    let block = k.div_ceil(blocks).max(1);
    let mut out = vec![0.0f32; rows * k];
    for r in 0..rows {
        let prow = &packed[r * packed_k..(r + 1) * packed_k];
        let srow = &scales.weight_scale[r * blocks..(r + 1) * blocks];
        let orow = &mut out[r * k..(r + 1) * k];
        for (c, o) in orow.iter_mut().enumerate() {
            let q: i8 = match bits {
                2 => {
                    let byte = prow[c / 4];
                    let v = (byte >> (2 * (c % 4))) & 0x03;
                    v as i8 - 2
                }
                4 => {
                    let byte = prow[c / 2];
                    let v = if c % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                    v as i8 - 8
                }
                _ => prow[c] as i8,
            };
            *o = q as f32 * srow[(c / block).min(blocks - 1)];
        }
    }
    out
}

/// Read all `*.weight_scale` / `*_activation_scale` tensors into per-module
/// scale records.
pub fn read_scales(
    store: &mut burn_store::SafetensorsStore,
) -> Result<HashMap<String, ModuleScales>> {
    use burn::tensor::DType;
    use burn_store::ModuleStore;

    let keys = store.keys()?;
    let mut out: HashMap<String, ModuleScales> = HashMap::new();
    let read_f32 = |store: &mut burn_store::SafetensorsStore, key: &str| -> Result<Vec<f32>> {
        let t = store
            .get_tensor(key)?
            .ok_or_else(|| anyhow::anyhow!("missing {key}"))?;
        let data = burn_store::bridge::to_data(t)?.convert_dtype(DType::F32);
        Ok(data.try_to_vec()?)
    };
    for key in &keys {
        if let Some(path) = key.strip_suffix(".weight_scale") {
            let values = read_f32(store, key)?;
            let entry = out.entry(path.to_string()).or_default();
            entry.weight_scale = values;
        } else if let Some(path) = key.strip_suffix(".embedding_scale") {
            let values = read_f32(store, key)?;
            let entry = out.entry(path.to_string()).or_default();
            entry.weight_scale = values;
        }
    }
    // Blocks per row need the row count: derive it from the packed weight shape.
    for key in &keys {
        let path = if let Some(p) = key.strip_suffix(".weight") {
            p
        } else if let Some(p) = key.strip_suffix(".embedding_quantized") {
            p
        } else {
            continue;
        };
        let Some(entry) = out.get_mut(path) else { continue };
        if entry.blocks != 0 {
            continue;
        }
        if let Some(t) = store.get_tensor(key)? {
            let dims: Vec<usize> = t.shape.clone().into();
            let rows = dims.first().copied().unwrap_or(1).max(1);
            entry.blocks = (entry.weight_scale.len() / rows).max(1);
        }
    }
    for key in &keys {
        let (path, which) = if let Some(p) = key.strip_suffix(".input_activation_scale") {
            (p, 0)
        } else if let Some(p) = key.strip_suffix(".output_activation_scale") {
            (p, 1)
        } else {
            continue;
        };
        let values = read_f32(store, key)?;
        let v = values.first().copied().unwrap_or(0.0);
        let entry = out.entry(path.to_string()).or_default();
        if which == 0 {
            entry.in_scale = v;
        } else {
            entry.out_scale = v;
        }
    }
    Ok(out)
}

/// A token table kept in its packed form: rows are dequantized on lookup, which
/// is bit-identical to dequantizing the whole table (the PLE table alone is
/// 4.7 GiB in f16 and 9.4 GiB in f32, versus 1.2 GiB packed).
#[derive(Debug, Clone)]
pub struct PackedTable {
    pub bits: u8,
    pub rows: usize,
    pub k: usize,
    pub blocks: usize,
    pub data: Vec<u8>,
    pub scales: Vec<f32>,
}

impl PackedTable {
    /// Dequantize the requested rows into row-major f32 `[ids.len(), k]`.
    pub fn gather_f32(&self, ids: &[u32]) -> Vec<f32> {
        let vpb = values_per_byte(self.bits);
        let packed_k = self.k.div_ceil(vpb);
        let blocks = self.blocks.max(1);
        let block = self.k.div_ceil(blocks).max(1);
        let mut out = vec![0.0f32; ids.len() * self.k];
        for (i, &id) in ids.iter().enumerate() {
            let r = (id as usize).min(self.rows - 1);
            let prow = &self.data[r * packed_k..(r + 1) * packed_k];
            let srow = &self.scales[r * blocks..(r + 1) * blocks];
            let orow = &mut out[i * self.k..(i + 1) * self.k];
            for (c, o) in orow.iter_mut().enumerate() {
                let q: i8 = match self.bits {
                    2 => {
                        let byte = prow[c / 4];
                        ((byte >> (2 * (c % 4))) & 0x03) as i8 - 2
                    }
                    4 => {
                        let byte = prow[c / 2];
                        let v = if c % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                        v as i8 - 8
                    }
                    _ => prow[c] as i8,
                };
                *o = q as f32 * srow[(c / block).min(blocks - 1)];
            }
        }
        out
    }
}

/// Read a packed token table (`*.embedding_quantized` + `*.embedding_scale`).
pub fn read_packed_table(
    store: &mut burn_store::SafetensorsStore,
    base: &str,
    bits: u8,
) -> Result<PackedTable> {
    use burn::tensor::DType;
    use burn_store::ModuleStore;

    let quant_key = format!("{base}.embedding_quantized");
    let scale_key = format!("{base}.embedding_scale");
    let t = store
        .get_tensor(&quant_key)?
        .ok_or_else(|| anyhow::anyhow!("missing {quant_key}"))?;
    let dims: Vec<usize> = t.shape.clone().into();
    let (rows, packed_k) = (dims[0], dims[1]);
    let data: Vec<u8> = if t.dtype == DType::U8 {
        burn_store::bridge::to_data(t)?.try_to_vec()?
    } else {
        let v: Vec<i8> = burn_store::bridge::to_data(t)?.try_to_vec()?;
        v.into_iter().map(|x| x as u8).collect()
    };
    let scales: Vec<f32> = burn_store::bridge::to_data(
        store
            .get_tensor(&scale_key)?
            .ok_or_else(|| anyhow::anyhow!("missing {scale_key}"))?,
    )?
    .convert_dtype(DType::F32)
    .try_to_vec()?;
    let k = packed_k * values_per_byte(bits);
    let blocks = (scales.len() / rows).max(1);
    Ok(PackedTable {
        bits,
        rows,
        k,
        blocks,
        data,
        scales,
    })
}
