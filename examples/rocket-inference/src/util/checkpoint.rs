//! Format-agnostic checkpoint access.
//!
//! Both storages (`safetensors` and `GGUF`) implement [`Checkpoint`], so the
//! family loaders and the server no longer name a storage format. Tensors are
//! exposed in the HF/PyTorch convention (`[out, in]` for a linear, `[vocab,
//! hidden]` for an embedding) — the convention `PyTorchToBurnAdapter` consumes —
//! so the module-tree and explicit-key paths are identical for both backends.
//!
//! The module-tree entry point ([`Checkpoint::load_into`]) is generic over the
//! module, so it cannot live in a `dyn Checkpoint`. [`AnyCheckpoint`] is the
//! erased handle the loaders use: it implements `Checkpoint` and forwards
//! `load_into` to the concrete backend.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use burn::tensor::{DType, TensorData};
use burn_store::burn_pack::Error as PackError;
use burn_store::{
    ApplyResult, FloatCastAdapter, ModuleAdapter, ModuleSnapshot, ModuleStore,
    PyTorchToBurnAdapter, SafetensorsStore, bridge,
};
use tokenizers::Tokenizer;

use crate::util::gguf::{GgufFile, hf_shape, tensor_data_f32};

/// Generation/special token ids a checkpoint declares.
#[allow(dead_code)] // consumed by the chat families as they are wired to the trait
#[derive(Debug, Clone, Default)]
pub struct SpecialIds {
    pub eos: Vec<u32>,
    pub bos: Option<u32>,
    pub pad: Option<u32>,
}

/// A resolved, opened model checkpoint independent of its on-disk format.
#[allow(dead_code)] // the read API is consumed family-by-family as they move to the trait
pub trait Checkpoint {
    /// "safetensors" or "gguf".
    fn format(&self) -> &'static str;
    /// The checkpoint's native float dtype (the `--dtype` default).
    fn native_dtype(&self) -> DType;
    /// Model config as HF-style JSON (`config.json`, or synthesized from GGUF).
    fn config_json(&self) -> Result<serde_json::Value>;
    /// A ready tokenizer.
    fn tokenizer(&self) -> Result<Tokenizer>;
    /// Special token ids (eos/bos/pad).
    fn special_ids(&self) -> SpecialIds;
    /// All tensor names in HF convention.
    fn keys(&self) -> Vec<String>;
    /// Whether `hf_key` exists.
    fn has(&self, hf_key: &str) -> bool {
        self.keys().iter().any(|k| k == hf_key)
    }
    /// One tensor, dequantized, cast to `dtype`, in the HF/PyTorch convention.
    fn get(&mut self, hf_key: &str, dtype: DType) -> Result<Option<TensorData>>;
    /// All tensors as burn-store entries (names in HF convention), for a custom
    /// adapter chain.
    fn pack_tensors(&mut self) -> Result<Vec<burn_store::burn_pack::Tensor>>;
    /// Apply every tensor to a module tree through the format's adapter.
    fn load_into<M: ModuleSnapshot>(&mut self, model: &mut M, dtype: DType) -> Result<ApplyResult>
    where
        Self: Sized;
}

impl AnyCheckpoint {
    /// Load a module tree with a caller-supplied adapter chain (the tensors are
    /// already in the HF/PyTorch convention, so `PyTorchToBurnAdapter`-based
    /// chains are correct for both backends).
    pub fn load_into_with<M: ModuleSnapshot>(
        &mut self,
        model: &mut M,
        adapter: Box<dyn ModuleAdapter>,
    ) -> Result<ApplyResult> {
        let tensors = self.pack_tensors()?;
        Ok(model.apply(tensors, None, Some(adapter), false))
    }
}

/// The format-erased checkpoint handle the loaders carry.
#[allow(clippy::large_enum_variant)]
pub enum AnyCheckpoint {
    Safetensors(SafetensorsCheckpoint),
    Gguf(GgufCheckpoint),
}

impl AnyCheckpoint {
    /// The GGUF architecture (`general.architecture`), if this is a GGUF checkpoint.
    pub fn arch(&self) -> Option<&str> {
        match self {
            Self::Gguf(c) => Some(c.arch()),
            Self::Safetensors(_) => None,
        }
    }

    /// One tensor by its raw GGUF name (mmproj clip scalars). `None` for safetensors.
    pub fn get_raw(&mut self, gguf_name: &str, dtype: DType) -> Result<Option<TensorData>> {
        match self {
            Self::Gguf(c) => c.get_raw(gguf_name, dtype),
            Self::Safetensors(_) => Ok(None),
        }
    }
}

impl Checkpoint for AnyCheckpoint {
    fn format(&self) -> &'static str {
        match self {
            Self::Safetensors(c) => c.format(),
            Self::Gguf(c) => c.format(),
        }
    }
    fn native_dtype(&self) -> DType {
        match self {
            Self::Safetensors(c) => c.native_dtype(),
            Self::Gguf(c) => c.native_dtype(),
        }
    }
    fn config_json(&self) -> Result<serde_json::Value> {
        match self {
            Self::Safetensors(c) => c.config_json(),
            Self::Gguf(c) => c.config_json(),
        }
    }
    fn tokenizer(&self) -> Result<Tokenizer> {
        match self {
            Self::Safetensors(c) => c.tokenizer(),
            Self::Gguf(c) => c.tokenizer(),
        }
    }
    fn special_ids(&self) -> SpecialIds {
        match self {
            Self::Safetensors(c) => c.special_ids(),
            Self::Gguf(c) => c.special_ids(),
        }
    }
    fn keys(&self) -> Vec<String> {
        match self {
            Self::Safetensors(c) => c.keys(),
            Self::Gguf(c) => c.keys(),
        }
    }
    fn get(&mut self, hf_key: &str, dtype: DType) -> Result<Option<TensorData>> {
        match self {
            Self::Safetensors(c) => c.get(hf_key, dtype),
            Self::Gguf(c) => c.get(hf_key, dtype),
        }
    }
    fn pack_tensors(&mut self) -> Result<Vec<burn_store::burn_pack::Tensor>> {
        match self {
            Self::Safetensors(c) => c.pack_tensors(),
            Self::Gguf(c) => c.pack_tensors(),
        }
    }
    fn load_into<M: ModuleSnapshot>(&mut self, model: &mut M, dtype: DType) -> Result<ApplyResult> {
        match self {
            Self::Safetensors(c) => c.load_into(model, dtype),
            Self::Gguf(c) => c.load_into(model, dtype),
        }
    }
}

/// Open a checkpoint, choosing the backend by the file's magic.
pub fn open(path: &Path) -> Result<AnyCheckpoint> {
    if looks_like_gguf(path)? {
        Ok(AnyCheckpoint::Gguf(GgufCheckpoint::open(path)?))
    } else {
        Ok(AnyCheckpoint::Safetensors(SafetensorsCheckpoint::open(
            path,
        )?))
    }
}

fn looks_like_gguf(path: &Path) -> Result<bool> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut magic = [0u8; 4];
    if f.read_exact(&mut magic).is_err() {
        return Ok(false);
    }
    Ok(&magic == b"GGUF")
}

/// Locate the weight file in a model directory: `model.safetensors`, else the
/// sole `*.gguf` (mmproj files excluded).
pub fn find_in_dir(dir: &Path) -> Result<PathBuf> {
    let st = dir.join("model.safetensors");
    if st.exists() {
        return Ok(st);
    }
    let mut cands: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("read_dir {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
        })
        .filter(|p| {
            !p.file_name()
                .map(|n| n.to_string_lossy().starts_with("mmproj"))
                .unwrap_or(false)
        })
        .collect();
    cands.sort();
    cands
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("no model.safetensors or *.gguf found in {}", dir.display()))
}

/// The `--dtype` default for a model directory (F32 for GGUF).
pub fn default_dtype_in_dir(dir: &Path) -> DType {
    match find_in_dir(dir) {
        Ok(p)
            if p.extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("gguf")) =>
        {
            DType::F32
        }
        Ok(p) => crate::util::store::native_dtype(&p),
        Err(_) => DType::BF16,
    }
}

/// Open a checkpoint chosen by [`find_in_dir`].
pub fn open_in_dir(dir: &Path) -> Result<AnyCheckpoint> {
    open(&find_in_dir(dir)?)
}

// ---------------------------------------------------------------------------
// safetensors
// ---------------------------------------------------------------------------

pub struct SafetensorsCheckpoint {
    path: PathBuf,
    dir: PathBuf,
    #[allow(dead_code)] // read by `get`, wired per family
    store: SafetensorsStore,
}

impl SafetensorsCheckpoint {
    pub fn open(path: &Path) -> Result<Self> {
        let dir = path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        let store = SafetensorsStore::from_file(path);
        Ok(Self {
            path: path.to_path_buf(),
            dir,
            store,
        })
    }

    fn load_into<M: ModuleSnapshot>(&mut self, model: &mut M, dtype: DType) -> Result<ApplyResult> {
        let mut store = SafetensorsStore::from_file(&self.path)
            .allow_partial(true)
            .with_from_adapter(PyTorchToBurnAdapter.chain(FloatCastAdapter::to(dtype)));
        Ok(model.load_from(&mut store)?)
    }
}

impl Checkpoint for SafetensorsCheckpoint {
    fn format(&self) -> &'static str {
        "safetensors"
    }

    fn native_dtype(&self) -> DType {
        crate::util::store::native_dtype(&self.path)
    }

    fn config_json(&self) -> Result<serde_json::Value> {
        let p = self.dir.join("config.json");
        let text = std::fs::read_to_string(&p).with_context(|| format!("read {}", p.display()))?;
        serde_json::from_str(&text).with_context(|| format!("parse {}", p.display()))
    }

    fn tokenizer(&self) -> Result<Tokenizer> {
        let p = self.dir.join("tokenizer.json");
        Tokenizer::from_file(&p).map_err(|e| anyhow::anyhow!("tokenizer {}: {e}", p.display()))
    }

    fn special_ids(&self) -> SpecialIds {
        let mut ids = SpecialIds::default();
        if let Ok(text) = std::fs::read_to_string(self.dir.join("generation_config.json"))
            && let Ok(json) = serde_json::from_str::<serde_json::Value>(&text)
        {
            ids.eos = eos_from_value(json.get("eos_token_id"));
        }
        ids
    }

    fn keys(&self) -> Vec<String> {
        let mut store = SafetensorsStore::from_file(&self.path);
        store.keys().unwrap_or_default()
    }

    fn get(&mut self, hf_key: &str, dtype: DType) -> Result<Option<TensorData>> {
        let Some(tensor) = self.store.get_tensor(hf_key)? else {
            return Ok(None);
        };
        let data = bridge::to_data(tensor)?;
        Ok(Some(data.convert_dtype(dtype)))
    }

    fn load_into<M: ModuleSnapshot>(&mut self, model: &mut M, dtype: DType) -> Result<ApplyResult> {
        SafetensorsCheckpoint::load_into(self, model, dtype)
    }

    fn pack_tensors(&mut self) -> Result<Vec<burn_store::burn_pack::Tensor>> {
        let mut store = SafetensorsStore::from_file(&self.path);
        Ok(store.get_all_tensors()?.values().cloned().collect())
    }
}

/// Parse `eos_token_id` (scalar or array) into a sorted, deduped id list.
#[allow(dead_code)] // used by the chat families as they move to the trait
pub fn eos_from_value(v: Option<&serde_json::Value>) -> Vec<u32> {
    let mut ids: Vec<u32> = match v {
        Some(serde_json::Value::Number(n)) => n.as_u64().map(|x| x as u32).into_iter().collect(),
        Some(serde_json::Value::Array(a)) => a
            .iter()
            .filter_map(|x| x.as_u64())
            .map(|x| x as u32)
            .collect(),
        _ => Vec::new(),
    };
    ids.sort_unstable();
    ids.dedup();
    ids
}

// ---------------------------------------------------------------------------
// GGUF
// ---------------------------------------------------------------------------

pub struct GgufCheckpoint {
    file: Arc<GgufFile>,
    dir: PathBuf,
    arch: String,
    /// HF key -> GGUF tensor name.
    map: BTreeMap<String, String>,
    /// A `clip` multimodal projector (`mmproj-*.gguf`): apply tower shape fixes.
    mmproj: bool,
}

impl GgufCheckpoint {
    pub fn open(path: &Path) -> Result<Self> {
        Self::open_with(path, None)
    }

    /// Open an EmbeddingGemma 2 `mmproj-*.gguf` (clip vision/audio towers).
    pub fn open_mmproj(path: &Path) -> Result<Self> {
        Self::open_with(path, Some(""))
    }

    /// Open a Gemma 4 `mmproj-*.gguf` (tower tree under `model.`).
    pub fn open_mmproj_gemma4(path: &Path) -> Result<Self> {
        Self::open_with(path, Some("model."))
    }

    fn open_with(path: &Path, mmproj: Option<&str>) -> Result<Self> {
        let file = Arc::new(GgufFile::open(path)?);
        let dir = path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        let arch = file.arch().unwrap_or("clip").to_string();
        let mut map = BTreeMap::new();
        for name in file.names() {
            let hf = match mmproj {
                Some(prefix) => mmproj_key_to_hf(&name, prefix),
                None => gguf_key_to_hf(&arch, &name),
            };
            if let Some(hf) = hf {
                map.insert(hf, name);
            }
        }
        Ok(Self {
            file,
            dir,
            arch,
            map,
            mmproj: mmproj.is_some(),
        })
    }

    pub fn arch(&self) -> &str {
        &self.arch
    }

    /// The GGUF metadata key/value map.
    pub fn metadata(&self) -> &BTreeMap<String, serde_json::Value> {
        self.file.metadata()
    }

    fn meta(&self) -> &BTreeMap<String, serde_json::Value> {
        self.file.metadata()
    }

    fn meta_u64(&self, key: &str) -> Option<u64> {
        self.meta().get(key).and_then(serde_json::Value::as_u64)
    }

    fn meta_f64(&self, key: &str) -> Option<f64> {
        self.meta().get(key).and_then(serde_json::Value::as_f64)
    }

    fn vocab_size(&self) -> Option<u64> {
        self.meta()
            .get("tokenizer.ggml.tokens")
            .and_then(serde_json::Value::as_array)
            .map(|a| a.len() as u64)
    }

    /// Fetch the base model's `config.json` (via hf-hub) so dense Gemma configs
    /// keep their per-layer overrides. `None` when offline or unspecified.
    fn base_config_json(&self) -> Option<serde_json::Value> {
        let from_meta = self
            .meta()
            .get("general.base_model.0.repo_url")
            .and_then(serde_json::Value::as_str)
            .and_then(|url| url.split("huggingface.co/").nth(1))
            .map(|rest| rest.trim_end_matches('/').to_string());
        let repo = from_meta.or_else(|| default_base_repo(&self.arch).map(str::to_string))?;
        let (owner, name) = repo.split_once('/')?;
        crate::util::hub::fetch_config_json(owner, name).ok()
    }

    /// One tensor by its raw GGUF name (used for mmproj clip scalars, which are
    /// not part of the tower module tree).
    pub fn get_raw(&mut self, gguf_name: &str, dtype: DType) -> Result<Option<TensorData>> {
        let Some((gguf_dims, _)) = self.file.shape_and_kind(gguf_name)? else {
            return Ok(None);
        };
        let Some(values) = self.file.values_f32(gguf_name)? else {
            return Ok(None);
        };
        let dims = hf_shape(&gguf_dims);
        let data = tensor_data_f32(values, &dims)?;
        Ok(Some(data.convert_dtype(dtype)))
    }

    /// All tensors as burn-store entries for a custom adapter chain.
    pub fn pack_tensors(&mut self) -> Result<Vec<burn_store::burn_pack::Tensor>> {
        let mut tensors = Vec::with_capacity(self.map.len());
        for (hf, gguf_name) in &self.map {
            let Some((gdims, _)) = self.file.shape_and_kind(gguf_name)? else {
                continue;
            };
            let shape = burn::tensor::Shape::from(final_dims(self.mmproj, hf, &gdims));
            let name = hf.clone();
            let hf_key = hf.clone();
            let mmproj = self.mmproj;
            let gguf_name = gguf_name.clone();
            let f = self.file.clone();
            let provider = move || -> Result<TensorData, PackError> {
                let (gdims, _) = f
                    .shape_and_kind(&gguf_name)
                    .map_err(|e| PackError::ValidationError(format!("{e:#}")))?
                    .ok_or_else(|| PackError::ValidationError(format!("missing {gguf_name}")))?;
                let values = f
                    .values_f32(&gguf_name)
                    .map_err(|e| PackError::ValidationError(format!("{e:#}")))?
                    .ok_or_else(|| PackError::ValidationError(format!("missing {gguf_name}")))?;
                let (hf_dims, values) = finalize(mmproj, &hf_key, &gdims, values)
                    .map_err(|e| PackError::ValidationError(format!("{e:#}")))?;
                tensor_data_f32(values, &hf_dims)
                    .map_err(|e| PackError::ValidationError(format!("{e:#}")))
            };
            tensors.push(bridge::deferred(name, DType::F32, shape, None, provider));
        }
        Ok(tensors)
    }

    pub fn load_into<M: ModuleSnapshot>(
        &mut self,
        model: &mut M,
        dtype: DType,
    ) -> Result<ApplyResult> {
        let tensors = self.pack_tensors()?;
        let adapter: Box<dyn ModuleAdapter> =
            Box::new(PyTorchToBurnAdapter.chain(FloatCastAdapter::to(dtype)));
        Ok(model.apply(tensors, None, Some(adapter), false))
    }
}

impl Checkpoint for GgufCheckpoint {
    fn format(&self) -> &'static str {
        "gguf"
    }

    fn native_dtype(&self) -> DType {
        DType::F32
    }

    fn config_json(&self) -> Result<serde_json::Value> {
        // Prefer a real config.json: a sibling file, else the base model repo's
        // (the dense Gemma configs carry per-layer overrides GGUF metadata does
        // not reproduce). Fall back to synthesis from GGUF metadata.
        let sibling = self.dir.join("config.json");
        if sibling.exists() {
            let text = std::fs::read_to_string(&sibling)
                .with_context(|| format!("read {}", sibling.display()))?;
            return serde_json::from_str(&text)
                .with_context(|| format!("parse {}", sibling.display()));
        }
        if matches!(self.arch.as_str(), "gemma-embedding2" | "gemma4" | "qwen35")
            && let Some(value) = self.base_config_json()
        {
            return Ok(value);
        }
        config_from_gguf(self).with_context(|| format!("synthesize config for {}", self.arch))
    }

    fn tokenizer(&self) -> Result<Tokenizer> {
        crate::util::gguf_tokenizer::build(self).context("build tokenizer from GGUF metadata")
    }

    fn special_ids(&self) -> SpecialIds {
        let mut ids = SpecialIds::default();
        for key in ["tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id"] {
            if let Some(v) = self.meta_u64(key) {
                ids.eos.push(v as u32);
            }
        }
        ids.eos.sort_unstable();
        ids.eos.dedup();
        ids.bos = self
            .meta_u64("tokenizer.ggml.bos_token_id")
            .map(|v| v as u32);
        ids.pad = self
            .meta_u64("tokenizer.ggml.padding_token_id")
            .map(|v| v as u32);
        ids
    }

    fn keys(&self) -> Vec<String> {
        self.map.keys().cloned().collect()
    }

    fn get(&mut self, hf_key: &str, dtype: DType) -> Result<Option<TensorData>> {
        let Some(gguf_name) = self.map.get(hf_key) else {
            return Ok(None);
        };
        let (gguf_dims, _) = self
            .file
            .shape_and_kind(gguf_name)?
            .ok_or_else(|| anyhow::anyhow!("missing GGUF tensor {gguf_name}"))?;
        let values = self
            .file
            .values_f32(gguf_name)?
            .ok_or_else(|| anyhow::anyhow!("missing GGUF tensor {gguf_name}"))?;
        let (hf_dims, values) = finalize(self.mmproj, hf_key, &gguf_dims, values)?;
        let data = tensor_data_f32(values, &hf_dims)?;
        Ok(Some(data.convert_dtype(dtype)))
    }

    fn load_into<M: ModuleSnapshot>(&mut self, model: &mut M, dtype: DType) -> Result<ApplyResult> {
        GgufCheckpoint::load_into(self, model, dtype)
    }

    fn pack_tensors(&mut self) -> Result<Vec<burn_store::burn_pack::Tensor>> {
        GgufCheckpoint::pack_tensors(self)
    }
}

// ---------------------------------------------------------------------------
// GGUF name mapping and config synthesis
// ---------------------------------------------------------------------------

/// Map a GGUF tensor name to its HF key for `arch`, or `None` for unused tensors.
pub fn gguf_key_to_hf(arch: &str, name: &str) -> Option<String> {
    match arch {
        "qwen3" | "qwen35" => qwen_map(arch, name),
        "gemma-embedding2" => gemma_emb2_map(name),
        "gemma4" => gemma4_map(name),
        _ => None,
    }
}

/// HF-convention dims for a GGUF tensor, applying mmproj per-tensor shape fixes.
fn final_dims(mmproj: bool, hf_key: &str, gguf_dims: &[usize]) -> Vec<usize> {
    let hf = hf_shape(gguf_dims);
    if !mmproj {
        return hf;
    }
    if hf_key.ends_with("patch_embedder.input_proj.weight") && hf.len() == 4 {
        // GGUF [16,16,3,768] -> HF Linear [out=768, in=768].
        return vec![hf[0], hf[1] * hf[2] * hf[3]];
    }
    if hf_key.ends_with("lconv1d.depthwise_conv1d.weight") && hf.len() == 2 {
        // GGUF [5,1024] -> HF Conv1d [1024, 1, 5] (depthwise).
        return vec![hf[0], 1, hf[1]];
    }
    hf
}

/// HF-convention dims + element order. GGUF and HF share the element memory
/// layout, so only the mmproj patch embed (channels-last flatten) permutes.
fn finalize(
    mmproj: bool,
    hf_key: &str,
    gguf_dims: &[usize],
    values: Vec<f32>,
) -> Result<(Vec<usize>, Vec<f32>)> {
    let dims = final_dims(mmproj, hf_key, gguf_dims);
    if mmproj && hf_key.ends_with("self_attn.per_dim_scale") {
        // GGUF stores `softplus(per_dim_scale)`; the model applies softplus again.
        let out = values.iter().map(|&v| v.exp_m1().ln()).collect();
        return Ok((dims, out));
    }
    if mmproj && hf_key.ends_with("patch_embedder.input_proj.weight") && gguf_dims.len() == 4 {
        // (out, c, h, w) -> (out, h, w, c): the patch flattens channels-last.
        let hf = hf_shape(gguf_dims);
        let (o, c, h, w) = (hf[0], hf[1], hf[2], hf[3]);
        let mut out = vec![0f32; values.len()];
        for oi in 0..o {
            for hi in 0..h {
                for wi in 0..w {
                    for ci in 0..c {
                        let src = ((oi * c + ci) * h + hi) * w + wi;
                        let dst = ((oi * h + hi) * w + wi) * c + ci;
                        out[dst] = values[src];
                    }
                }
            }
        }
        return Ok((dims, out));
    }
    Ok((dims, values))
}

/// Map a `clip` `mmproj-*.gguf` tensor name to the EmbeddingGemma 2 tower module
/// path. Clip scalars (`input_min`/`output_max`, ...) are deliberately unmapped:
/// they feed the `ClipBounds` map, not the module tree.
fn mmproj_key_to_hf(name: &str, prefix: &str) -> Option<String> {
    if let Some(rest) = name.strip_prefix("v.blk.") {
        let (layer, suffix) = rest.split_once('.')?;
        let hf = match suffix {
            "ln1.weight" => "input_layernorm.weight",
            "ln2.weight" => "pre_feedforward_layernorm.weight",
            "attn_post_norm.weight" => "post_attention_layernorm.weight",
            "ffn_post_norm.weight" => "post_feedforward_layernorm.weight",
            "attn_q.weight" => "self_attn.q_proj.linear.weight",
            "attn_k.weight" => "self_attn.k_proj.linear.weight",
            "attn_v.weight" => "self_attn.v_proj.linear.weight",
            "attn_out.weight" => "self_attn.o_proj.linear.weight",
            "attn_q_norm.weight" => "self_attn.q_norm.weight",
            "attn_k_norm.weight" => "self_attn.k_norm.weight",
            "ffn_gate.weight" => "mlp.gate_proj.linear.weight",
            "ffn_up.weight" => "mlp.up_proj.linear.weight",
            "ffn_down.weight" => "mlp.down_proj.linear.weight",
            _ => return None,
        };
        return Some(format!("{prefix}vision_tower.encoder.layers.{layer}.{hf}"));
    }
    if let Some(rest) = name.strip_prefix("a.blk.") {
        let (layer, suffix) = rest.split_once('.')?;
        let hf = match suffix {
            "ln2.weight" => "norm_out.weight",
            "attn_pre_norm.weight" => "norm_pre_attn.weight",
            "attn_post_norm.weight" => "norm_post_attn.weight",
            "ffn_norm.weight" => "feed_forward1.pre_layer_norm.weight",
            "ffn_post_norm.weight" => "feed_forward1.post_layer_norm.weight",
            "ffn_norm_1.weight" => "feed_forward2.pre_layer_norm.weight",
            "ffn_post_norm_1.weight" => "feed_forward2.post_layer_norm.weight",
            "attn_q.weight" => "self_attn.q_proj.linear.weight",
            "attn_k.weight" => "self_attn.k_proj.linear.weight",
            "attn_v.weight" => "self_attn.v_proj.linear.weight",
            "attn_out.weight" => "self_attn.post.linear.weight",
            "attn_k_rel.weight" => "self_attn.relative_k_proj.weight",
            "per_dim_scale.weight" => "self_attn.per_dim_scale",
            "ffn_up.weight" => "feed_forward1.ffw_layer_1.linear.weight",
            "ffn_down.weight" => "feed_forward1.ffw_layer_2.linear.weight",
            "ffn_up_1.weight" => "feed_forward2.ffw_layer_1.linear.weight",
            "ffn_down_1.weight" => "feed_forward2.ffw_layer_2.linear.weight",
            "conv_dw.weight" => "lconv1d.depthwise_conv1d.weight",
            "conv_pw1.weight" => "lconv1d.linear_start.linear.weight",
            "conv_pw2.weight" => "lconv1d.linear_end.linear.weight",
            "conv_norm.weight" => "lconv1d.pre_layer_norm.weight",
            "norm_conv.weight" => "lconv1d.conv_norm.weight",
            _ => return None,
        };
        return Some(format!("{prefix}audio_tower.layers.{layer}.{hf}"));
    }
    if let Some(rest) = name.strip_prefix("a.conv1d.") {
        let (layer, suffix) = rest.split_once('.')?;
        let suffix = if suffix == "weight" {
            "conv.weight"
        } else {
            suffix
        };
        return Some(format!(
            "{prefix}audio_tower.subsample_conv_projection.layer{layer}.{suffix}"
        ));
    }
    let hf = match name {
        "v.patch_embd.weight" => "vision_tower.patch_embedder.input_proj.weight",
        "v.position_embd.weight" => "vision_tower.patch_embedder.position_embedding_table",
        "mm.input_projection.weight" => "embed_vision.embedding_projection.weight",
        "mm.a.input_projection.weight" => "embed_audio.embedding_projection.weight",
        "a.input_projection.weight" => {
            "audio_tower.subsample_conv_projection.input_proj_linear.weight"
        }
        "a.pre_encode.out.weight" => "audio_tower.output_proj.weight",
        "a.pre_encode.out.bias" => "audio_tower.output_proj.bias",
        _ => return None,
    };
    Some(format!("{prefix}{hf}"))
}

/// Qwen-family mapping (`layers.N.*`, `embed_tokens`, `norm`, `lm_head`).
///
/// Qwen3-Embedding's module tree is top-level; Qwen3.5's checkpoint is nested
/// under `model.language_model.*` (the intent loader reads those keys).
fn qwen_map(arch: &str, name: &str) -> Option<String> {
    let base = if arch == "qwen35" {
        "model.language_model."
    } else {
        ""
    };
    if let Some(rest) = name.strip_prefix("blk.") {
        let (layer, suffix) = rest.split_once('.')?;
        let hf = match suffix {
            "attn_norm.weight" => "input_layernorm.weight",
            "attn_q.weight" => "self_attn.q_proj.weight",
            "attn_k.weight" => "self_attn.k_proj.weight",
            "attn_v.weight" => "self_attn.v_proj.weight",
            "attn_output.weight" => "self_attn.o_proj.weight",
            "attn_q_norm.weight" => "self_attn.q_norm.weight",
            "attn_k_norm.weight" => "self_attn.k_norm.weight",
            "ffn_norm.weight" => "post_attention_layernorm.weight",
            // qwen3.5 names the post-attention norm differently.
            "post_attention_norm.weight" => "post_attention_layernorm.weight",
            "ffn_gate.weight" => "mlp.gate_proj.weight",
            "ffn_up.weight" => "mlp.up_proj.weight",
            "ffn_down.weight" => "mlp.down_proj.weight",
            // qwen3.5 Gated DeltaNet (linear-attention) layers.
            "ssm_a" => "linear_attn.A_log",
            "ssm_alpha.weight" => "linear_attn.in_proj_a.weight",
            "ssm_beta.weight" => "linear_attn.in_proj_b.weight",
            "ssm_conv1d.weight" => "linear_attn.conv1d.weight",
            "ssm_dt.bias" => "linear_attn.dt_bias",
            "ssm_norm.weight" => "linear_attn.norm.weight",
            "ssm_out.weight" => "linear_attn.out_proj.weight",
            "attn_gate.weight" => "linear_attn.in_proj_z.weight",
            "attn_qkv.weight" => "linear_attn.in_proj_qkv.weight",
            _ => return None,
        };
        return Some(format!("{base}layers.{layer}.{hf}"));
    }
    match name {
        "token_embd.weight" => Some(format!("{base}embed_tokens.weight")),
        "output_norm.weight" => Some(format!("{base}norm.weight")),
        "output.weight" => Some(format!("{base}lm_head.weight")),
        _ => None,
    }
}

/// Gemma-family mapping (module tree under `language_model.*`).
fn gemma_emb2_map(name: &str) -> Option<String> {
    if let Some(rest) = name.strip_prefix("blk.") {
        let (layer, suffix) = rest.split_once('.')?;
        // NB: for Gemma, `ffn_norm` is the *pre*-feedforward norm and
        // `post_attention_norm` is the post-attention norm (unlike Qwen, where
        // `ffn_norm` is the post-attention norm).
        let hf = match suffix {
            "attn_norm.weight" => "input_layernorm.weight",
            "post_attention_norm.weight" => "post_attention_layernorm.weight",
            "ffn_norm.weight" => "pre_feedforward_layernorm.weight",
            "post_ffw_norm.weight" => "post_feedforward_layernorm.weight",
            "attn_q.weight" => "self_attn.q_proj.weight",
            "attn_k.weight" => "self_attn.k_proj.weight",
            "attn_v.weight" => "self_attn.v_proj.weight",
            "attn_output.weight" => "self_attn.o_proj.weight",
            "attn_q_norm.weight" => "self_attn.q_norm.weight",
            "attn_k_norm.weight" => "self_attn.k_norm.weight",
            "ffn_gate.weight" => "mlp.gate_proj.weight",
            "ffn_up.weight" => "mlp.up_proj.weight",
            "ffn_down.weight" => "mlp.down_proj.weight",
            "inp_gate.weight" => "ple_block.per_layer_input_gate.weight",
            "proj.weight" => "ple_block.per_layer_projection.weight",
            "post_norm.weight" => "ple_block.post_per_layer_input_norm.weight",
            "layer_output_scale.weight" => "layer_scalar",
            _ => return None,
        };
        return Some(format!("language_model.layers.{layer}.{hf}"));
    }
    let hf = match name {
        "token_embd.weight" => "language_model.embed_tokens.weight",
        "output.weight" => "language_model.embedding_projection.weight",
        "output_norm.weight" => "language_model.norm.weight",
        "per_layer_model_proj.weight" => "language_model.ple.per_layer_model_projection.weight",
        "per_layer_proj_norm.weight" => "language_model.ple.per_layer_projection_norm.weight",
        "per_layer_token_embd.weight" => "language_model.per_layer_token_embd.weight",
        "rope_freqs.weight" => "language_model.rope_freqs.weight",
        _ => return None,
    };
    Some(hf.to_string())
}

/// Gemma 4 mapping: the text tree is under `model.language_model.*` with flat
/// per-layer PLE names (unlike EmbeddingGemma 2's `ple_block.*`).
fn gemma4_map(name: &str) -> Option<String> {
    const P: &str = "model.language_model.";
    if let Some(rest) = name.strip_prefix("blk.") {
        let (layer, suffix) = rest.split_once('.')?;
        let hf = match suffix {
            "attn_norm.weight" => "input_layernorm.weight",
            "post_attention_norm.weight" => "post_attention_layernorm.weight",
            "ffn_norm.weight" => "pre_feedforward_layernorm.weight",
            "post_ffw_norm.weight" => "post_feedforward_layernorm.weight",
            "attn_q.weight" => "self_attn.q_proj.weight",
            "attn_k.weight" => "self_attn.k_proj.weight",
            "attn_v.weight" => "self_attn.v_proj.weight",
            "attn_output.weight" => "self_attn.o_proj.weight",
            "attn_q_norm.weight" => "self_attn.q_norm.weight",
            "attn_k_norm.weight" => "self_attn.k_norm.weight",
            "ffn_gate.weight" => "mlp.gate_proj.weight",
            "ffn_up.weight" => "mlp.up_proj.weight",
            "ffn_down.weight" => "mlp.down_proj.weight",
            "inp_gate.weight" => "per_layer_input_gate.weight",
            "proj.weight" => "per_layer_projection.weight",
            "post_norm.weight" => "post_per_layer_input_norm.weight",
            "layer_output_scale.weight" => "layer_scalar",
            _ => return None,
        };
        return Some(format!("{P}layers.{layer}.{hf}"));
    }
    let hf = match name {
        "token_embd.weight" => "embed_tokens.weight",
        "per_layer_token_embd.weight" => "embed_tokens_per_layer.weight",
        "output_norm.weight" => "norm.weight",
        "per_layer_model_proj.weight" => "per_layer_model_projection.weight",
        "per_layer_proj_norm.weight" => "per_layer_projection_norm.weight",
        _ => return None,
    };
    Some(format!("{P}{hf}"))
}

/// The canonical HF base repo for an architecture when the GGUF metadata does not
/// name one (EmbeddingGemma 2's does not).
fn default_base_repo(arch: &str) -> Option<&'static str> {
    match arch {
        "gemma-embedding2" => Some("google/embeddinggemma-2"),
        "gemma4" => Some("google/gemma-4-E2B-it"),
        _ => None,
    }
}

/// Synthesize an HF-style config JSON from GGUF metadata.
fn config_from_gguf(c: &GgufCheckpoint) -> Result<serde_json::Value> {
    use serde_json::json;
    let arch = c.arch();
    let p = |s: &str| format!("{arch}.{s}");
    let need = |k: &str| -> Result<u64> {
        c.meta_u64(&p(k))
            .ok_or_else(|| anyhow::anyhow!("GGUF metadata missing {k}"))
    };
    let model_type = match arch {
        "qwen3" => "qwen3",
        "gemma-embedding2" => "embedding_gemma2",
        "qwen35" => "qwen3_5",
        "gemma4" => "gemma4",
        other => bail!("unknown GGUF architecture '{other}'"),
    };
    let head_dim = c
        .meta_u64(&p("attention.key_length"))
        .or_else(|| c.meta_u64(&p("attention.value_length")))
        .unwrap_or(0);
    let cfg = json!({
        "model_type": model_type,
        "vocab_size": c.vocab_size().unwrap_or(0),
        "hidden_size": need("embedding_length")?,
        "intermediate_size": c.meta_u64(&p("feed_forward_length")).unwrap_or(0),
        "num_hidden_layers": need("block_count")?,
        "num_attention_heads": need("attention.head_count")?,
        "num_key_value_heads": c.meta_u64(&p("attention.head_count_kv")).unwrap_or(1),
        "head_dim": head_dim,
        "rms_norm_eps": c.meta_f64(&p("attention.layer_norm_rms_epsilon")).unwrap_or(1e-6),
        "rope_theta": c.meta_f64(&p("rope.freq_base")).unwrap_or(10000.0),
        "max_position_embeddings": c.meta_u64(&p("context_length")).unwrap_or(0),
    });
    Ok(cfg)
}
