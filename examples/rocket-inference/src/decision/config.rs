//! `decision_config.json` and the readout head of a d3 checkpoint.
//!
//! The config carries the format contract the runtime must honour (prompt
//! family, answer codes and their token ids, attention mode, pooling,
//! temperature) and the readout head is one `[255, hidden]` f32 matrix whose row
//! order matches [`DecisionConfig::codes`].

use std::path::Path;

use anyhow::{Context, Result, bail};
use burn::prelude::*;
use burn::tensor::{DType, Tensor};
use tokenizers::Tokenizer;

use crate::util::checkpoint::SafetensorsCheckpoint;

/// `d3_format.FORMAT_ID`.
pub const FORMAT_ID: &str = "d3-code-readout-v1";
/// `d3_format.MAX_OPTIONS`: the readout head has one row per answer code.
pub const MAX_OPTIONS: usize = 255;
/// `d3_runtime.FORMAT_VERSION`.
pub const FORMAT_VERSION: u64 = 1;
/// `d3_format._CODE_PROBE`: the user message the answer-code prefix is built from.
pub const CODE_PROBE: &str = "Choose an option.";

/// The validated parts of `decision_config.json`.
#[derive(Debug, Clone)]
pub struct DecisionConfig {
    /// The 255 answer codes (`A`..`JT`) in readout-row order.
    pub codes: Vec<String>,
    /// One token id per code (a code is a single token of its own).
    pub token_ids: Vec<u32>,
    /// Readout temperature, applied before the softmax.
    pub temperature: f64,
    /// The checkpoint's own input cap (`null` = none); the CLI/server flag also
    /// applies.
    pub max_length: Option<usize>,
}

/// The raw file, validated into [`DecisionConfig`].
#[derive(Debug, serde::Deserialize)]
struct RawConfig {
    #[serde(default)]
    format_version: Option<u64>,
    #[serde(default)]
    format_id: Option<String>,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    codes: Vec<String>,
    #[serde(default)]
    token_ids: Vec<u32>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    attention_mode: Option<String>,
    #[serde(default)]
    pooling: Option<String>,
    #[serde(default)]
    max_length: Option<usize>,
    #[serde(default)]
    readout_dtype: Option<String>,
}

impl DecisionConfig {
    /// Load and validate `<dir>/decision_config.json`.
    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join("decision_config.json");
        let text =
            std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        let raw: RawConfig =
            serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
        raw.validate()
            .with_context(|| format!("{}", path.display()))
    }
}

impl RawConfig {
    fn validate(self) -> Result<DecisionConfig> {
        match self.format_version {
            Some(FORMAT_VERSION) => {}
            Some(version) => {
                bail!("format_version is {version}; this build reads version {FORMAT_VERSION}")
            }
            None => bail!("format_version is missing; this build reads version {FORMAT_VERSION}"),
        }
        match self.format_id.as_deref() {
            Some(FORMAT_ID) => {}
            Some(other) => bail!("format_id is '{other}', expected '{FORMAT_ID}'"),
            None => bail!("format_id is missing (expected '{FORMAT_ID}')"),
        }
        match self.prompt.as_deref() {
            Some("d3") => {}
            Some(other) => bail!("prompt is '{other}', expected 'd3'"),
            None => bail!("prompt is missing (expected 'd3')"),
        }
        if self.codes.len() != MAX_OPTIONS {
            bail!(
                "codes has {} entries, expected {MAX_OPTIONS}",
                self.codes.len()
            );
        }
        if self.codes.iter().any(|c| c.is_empty()) {
            bail!("codes must be nonempty strings");
        }
        if self.token_ids.len() != MAX_OPTIONS {
            bail!(
                "token_ids has {} entries, expected {MAX_OPTIONS}",
                self.token_ids.len()
            );
        }
        let temperature = match self.temperature {
            Some(t) if t.is_finite() && t > 0.0 => t,
            Some(t) => bail!("temperature must be finite and positive, got {t}"),
            None => bail!("temperature is missing (the d3 runtime defaults to 1.0)"),
        };
        match self.attention_mode.as_deref() {
            Some("noncausal_full_attention") => {}
            Some(other) => bail!(
                "attention_mode is '{other}'; this build implements \
                 'noncausal_full_attention' only"
            ),
            None => bail!("attention_mode is missing (expected 'noncausal_full_attention')"),
        }
        match self.pooling.as_deref() {
            Some("last") => {}
            Some(other) => bail!("pooling is '{other}', expected 'last'"),
            None => bail!("pooling is missing (expected 'last')"),
        }
        match self.readout_dtype.as_deref() {
            Some("float32") => {}
            Some(other) => bail!("readout_dtype is '{other}', expected 'float32'"),
            None => bail!("readout_dtype is missing (expected 'float32')"),
        }
        Ok(DecisionConfig {
            codes: self.codes,
            token_ids: self.token_ids,
            temperature,
            max_length: self.max_length,
        })
    }
}

/// Load `<dir>/readout.safetensors` (`weight`, `[255, hidden]`, f32).
pub fn load_readout(dir: &Path, hidden: usize, device: &Device) -> Result<Tensor<2>> {
    let path = dir.join("readout.safetensors");
    if !path.exists() {
        bail!(
            "{} is missing (a d3 checkpoint stores its 255-code readout head there)",
            path.display()
        );
    }
    let mut store = SafetensorsCheckpoint::open(&path)?;
    let data = crate::util::checkpoint::Checkpoint::get(&mut store, "weight", DType::F32)?
        .ok_or_else(|| anyhow::anyhow!("{} has no `weight` tensor", path.display()))?;
    let rank = data.shape().num_dims();
    if rank != 2 {
        bail!(
            "{} weight has {rank} dimensions, expected 2 ([{MAX_OPTIONS}, {hidden}])",
            path.display()
        );
    }
    let shape = data.shape().dims::<2>();
    if shape != [MAX_OPTIONS, hidden] {
        bail!(
            "{} weight has shape {shape:?}, expected [{MAX_OPTIONS}, {hidden}]",
            path.display()
        );
    }
    Ok(Tensor::<2>::from_data(data, device))
}

/// Every code the model may use, in preference order (`d3_format.candidate_codes`):
/// the single letters `A`..`Z`, then the two-letter codes `AA`..`ZZ`.
pub fn candidate_codes() -> Vec<String> {
    let letters: Vec<char> = ('A'..='Z').collect();
    let mut codes: Vec<String> = letters.iter().map(|c| c.to_string()).collect();
    for first in &letters {
        for second in &letters {
            codes.push(format!("{first}{second}"));
        }
    }
    codes
}

/// The assistant-turn prefix `d3_format.answer_codes` probes with: the chat
/// template of one user turn (`enable_thinking=False`), up to where the answer
/// code is read.
pub fn code_probe_prefix() -> String {
    format!(
        "<|im_start|>user\n{CODE_PROBE}<|im_end|>\n\
         <|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
}

fn encode(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    tokenizer
        .encode(text, false)
        .map(|encoding| encoding.get_ids().to_vec())
        .map_err(|e| anyhow::anyhow!("tokenize the answer-code probe: {e}"))
}

/// `d3_format.answer_codes`: the canonical 255 answer codes and their token ids.
///
/// A code qualifies when it is a single token of its own, that token is not
/// already taken by an earlier code, and it stays that single token right after
/// the assistant turn opens.
pub fn answer_codes(tokenizer: &Tokenizer) -> Result<(Vec<String>, Vec<u32>)> {
    let prefix = code_probe_prefix();
    let prefix_ids = encode(tokenizer, &prefix)?;
    let mut chosen: Vec<String> = Vec::with_capacity(MAX_OPTIONS);
    let mut chosen_ids: Vec<u32> = Vec::with_capacity(MAX_OPTIONS);
    let mut taken = std::collections::HashSet::new();
    for code in candidate_codes() {
        let alone = encode(tokenizer, &code)?;
        let [id] = alone.as_slice() else {
            continue;
        };
        if taken.contains(id) {
            continue;
        }
        let mut after_prefix = prefix_ids.clone();
        after_prefix.push(*id);
        if encode(tokenizer, &format!("{prefix}{code}"))? != after_prefix {
            continue;
        }
        chosen.push(code);
        chosen_ids.push(*id);
        taken.insert(*id);
        if chosen.len() == MAX_OPTIONS {
            return Ok((chosen, chosen_ids));
        }
    }
    bail!(
        "the tokenizer yields only {} usable answer codes; the model needs {MAX_OPTIONS}",
        chosen.len()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw(json: serde_json::Value) -> RawConfig {
        serde_json::from_value(json).unwrap()
    }

    fn good() -> serde_json::Value {
        serde_json::json!({
            "format_version": 1,
            "format_id": FORMAT_ID,
            "prompt": "d3",
            "codes": (0..MAX_OPTIONS).map(|i| format!("C{i}")).collect::<Vec<_>>(),
            "token_ids": (0..MAX_OPTIONS).collect::<Vec<_>>(),
            "temperature": 1.0,
            "attention_mode": "noncausal_full_attention",
            "pooling": "last",
            "max_length": null,
            "readout_dtype": "float32",
        })
    }

    #[test]
    fn a_full_config_is_accepted() {
        let cfg = raw(good()).validate().unwrap();
        assert_eq!(cfg.codes.len(), MAX_OPTIONS);
        assert_eq!(cfg.temperature, 1.0);
        assert!(cfg.max_length.is_none());
    }

    #[test]
    fn every_contract_field_is_checked() {
        let cases: [(&str, serde_json::Value, &str); 8] = [
            ("format_id", serde_json::json!("other"), "format_id"),
            ("prompt", serde_json::json!("other"), "prompt"),
            (
                "attention_mode",
                serde_json::json!("causal"),
                "noncausal_full_attention",
            ),
            ("pooling", serde_json::json!("mean"), "pooling"),
            (
                "readout_dtype",
                serde_json::json!("bfloat16"),
                "readout_dtype",
            ),
            ("temperature", serde_json::json!(0.0), "temperature"),
            ("codes", serde_json::json!(["A"]), "codes"),
            ("token_ids", serde_json::json!([32]), "token_ids"),
        ];
        for (key, value, needle) in cases {
            let mut json = good();
            json[key] = value;
            let err = format!("{}", raw(json).validate().unwrap_err());
            assert!(err.contains(needle), "{key}: {err}");
        }
        // The reference refuses another version and a missing one.
        for version in [serde_json::json!(2), serde_json::Value::Null] {
            let mut json = good();
            json["format_version"] = version.clone();
            if version.is_null() {
                json.as_object_mut().unwrap().remove("format_version");
            }
            let err = format!("{}", raw(json).validate().unwrap_err());
            assert!(err.contains("format_version"), "{err}");
        }
    }

    #[test]
    fn the_candidate_codes_are_the_reference_sequence() {
        let codes = candidate_codes();
        assert_eq!(codes.len(), 26 + 26 * 26);
        assert_eq!(&codes[..3], ["A", "B", "C"]);
        assert_eq!(codes[25], "Z");
        assert_eq!(codes[26], "AA");
        assert_eq!(codes[27], "AB");
        assert_eq!(codes[26 + 25], "AZ");
        assert_eq!(codes[26 + 26], "BA");
        assert_eq!(codes.last().unwrap(), "ZZ");
        let mut unique = codes.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), codes.len(), "the sequence must be unique");
    }

    #[test]
    fn the_code_probe_prefix_is_the_chat_template_opening() {
        assert_eq!(
            code_probe_prefix(),
            "<|im_start|>user\nChoose an option.<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }
}
