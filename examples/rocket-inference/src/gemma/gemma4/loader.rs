//! Loading for the Gemma 4 generation checkpoint (E2B-it).
//!
//! The 10.25 GB checkpoint holds 5.12B parameters, half of them in two huge
//! token tables. Loading it as f32 and casting afterwards would need ~30 GB of
//! RAM, so a store adapter converts each tensor to its final dtype *while it is
//! applied*:
//!
//! - the two `embed_tokens*` tables become f16 (exact: the source is bf16, and
//!   f16 has more mantissa bits), which halves the dominant 9.4 GB table;
//! - the projection weights (`Struct:Linear`) stay f32 by default (parity), or
//!   become Q8_0 when `--quant q8` is set;
//! - everything else (norms, scalars) stays f32.
//!
//! The LM head is tied to `embed_tokens` (the checkpoint has no `lm_head`
//! tensor); the loader materializes the transposed table once for logits.

use std::path::Path;

use anyhow::{Context, Result};
use burn::prelude::*;
use burn::tensor::DType;
use burn_store::burn_pack::Tensor as PackTensor;
use burn_store::{
    ModuleAdapter, ModuleContext, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore, bridge,
};

use crate::gemma::config::Emb2Config;
use crate::gemma::gemma4::model::{GenRoot, GenTextModel};
use crate::gemma::layers::ClipBounds;

/// Pack the generation model's text projections into resident fp16 NPU weights
/// for the prefill pass. `keep_cpu` retains the f32 copies (decode runs on the
/// CPU); with `keep_cpu == false` the CPU copies are dropped.
#[cfg(all(feature = "npu", target_arch = "aarch64"))]
pub fn pack_text_for_prefill(
    model: &mut GenRoot,
    threads: usize,
    device: &Device,
    keep_cpu: bool,
) -> Result<()> {
    use crate::gemma::layers;
    burn_rocket::init(threads).map_err(|e| anyhow::anyhow!("NPU context creation failed: {e}"))?;
    layers::set_npu_prefill_only(keep_cpu);
    let t0 = std::time::Instant::now();
    let mut count = 0usize;
    let mut bytes = 0usize;
    model.text_mut().for_each_projection_mut(|lin| {
        let (n, k) = layers::pack_linear_into_npu(lin, device, keep_cpu);
        count += 1;
        bytes += n * k * 2;
    });
    if !keep_cpu {
        #[cfg(all(target_os = "linux", target_env = "gnu"))]
        unsafe {
            libc::malloc_trim(0);
        }
    }
    println!(
        "NPU: packed {count} text projections ({:.2} GiB f16) in {:.2}s; \
         prefill on the NPU, decode on the CPU{} (resident {:.0} MiB anon)",
        bytes as f64 / (1u64 << 30) as f64,
        t0.elapsed().as_secs_f64(),
        if keep_cpu { " (CPU copies kept)" } else { "" },
        crate::util::rss_mib()
    );
    Ok(())
}

/// Per-tensor dtype selection, applied during loading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadDtype {
    /// f32 projections and body (parity).
    F32,
    /// f16 projections: half the memory, f16 GEMM; the body stays f32.
    F16,
    /// f16 during load, then Q8_0-resident (memory-only: decode dequantizes).
    Q8,
    /// The checkpoint's native bf16 throughout: projections, tables and body.
    Bf16,
}

impl LoadDtype {
    /// The dtype linear projections (and the tied LM head) load in.
    pub fn tensor_dtype(self) -> DType {
        match self {
            LoadDtype::F32 => DType::F32,
            LoadDtype::F16 | LoadDtype::Q8 => DType::F16,
            LoadDtype::Bf16 => DType::BF16,
        }
    }

    /// Map the checkpoint's native float dtype to a `LoadDtype` (bf16 for the
    /// current checkpoints; f32/f16 checkpoints stay in their own dtype).
    pub fn from_native(model_dir: &Path) -> Self {
        match crate::util::store::native_dtype(&model_dir.join("model.safetensors")) {
            DType::F16 => LoadDtype::F16,
            DType::F32 => LoadDtype::F32,
            _ => LoadDtype::Bf16,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LoadDtypeAdapter {
    /// Linear projections.
    linear: DType,
    /// Token tables (`embed_tokens*`).
    table: DType,
    /// Every other float tensor in the module tree (norms, scalars, ...).
    rest: DType,
}

impl LoadDtypeAdapter {
    pub fn new(_device: &Device, dtype: LoadDtype) -> Self {
        match dtype {
            LoadDtype::F32 => Self {
                linear: DType::F32,
                // f16 tables halve the dominant table while staying exact for
                // bf16 sources; the gathered rows are cast back per forward.
                table: DType::F16,
                rest: DType::F32,
            },
            LoadDtype::F16 | LoadDtype::Q8 => Self {
                linear: DType::F16,
                table: DType::F16,
                rest: DType::F32,
            },
            LoadDtype::Bf16 => Self {
                linear: DType::BF16,
                table: DType::BF16,
                // The generation model computes in f32 (f32 softmax/PLE/logits);
                // norms and scalars must match the body.
                rest: DType::F32,
            },
        }
    }
}

impl LoadDtypeAdapter {
    /// Keep the token tables in f32 (QAT checkpoints dequantize to f32).
    pub fn with_f32_tables(mut self, on: bool) -> Self {
        if on {
            self.table = DType::F32;
        }
        self
    }
}

impl ModuleAdapter for LoadDtypeAdapter {
    fn adapt(&self, tensor: PackTensor, ctx: ModuleContext<'_>) -> PackTensor {
        let name = tensor.name.clone();
        let shape = tensor.shape.clone();
        let is_table = name.ends_with("embed_tokens.weight")
            || name.ends_with("embed_tokens_per_layer.weight");
        let is_linear = ctx.module_type() == Some("Struct:Linear");

        let target = if is_table {
            self.table
        } else if is_linear {
            // f16 during load (halves the f32 footprint); the Q8 pass runs after
            // the model is fully loaded, because the store applies tensors in the
            // backend's packed layout, which cannot be produced from the adapter.
            self.linear
        } else {
            self.rest
        };
        bridge::map_data(tensor, name, target, shape, move |data| {
            data.convert_dtype(target)
        })
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(self.clone())
    }
}

/// Q8-resident projections (Q8_0: symmetric int8, 32-value blocks, f16 scales),
/// applied after loading; `lin` dequantizes each weight per call.
pub fn quantize_gen(model: GenRoot) -> Result<GenRoot> {
    use crate::util::quant::{LowRam, param_group};
    let t0 = std::time::Instant::now();
    let proj_group = param_group(
        r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|per_layer_model_projection|per_layer_input_gate|per_layer_projection)\.weight$",
    )?;
    let mut mapper = LowRam::new(vec![proj_group], Vec::new(), None);
    let model = model.map(&mut mapper);
    crate::gemma::layers::set_quantized(true);
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    unsafe {
        libc::malloc_trim(0);
    }
    println!(
        "Q8-resident projections in {:.2}s (resident {:.0} MiB anon)",
        t0.elapsed().as_secs_f64(),
        rss_mib()
    );
    Ok(model)
}

/// Vision-tower `Gemma4ClippableLinear` paths (E2B ships clip bounds for these;
/// EmbeddingGemma 2 does not).
pub const VISION_LINEAR_NAMES: [&str; 7] = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
];

/// Audio-tower `Gemma4ClippableLinear` paths (same as the EmbeddingGemma 2 round).
pub const AUDIO_LINEAR_NAMES: [&str; 10] = [
    "feed_forward1.ffw_layer_1",
    "feed_forward1.ffw_layer_2",
    "feed_forward2.ffw_layer_1",
    "feed_forward2.ffw_layer_2",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.post",
    "lconv1d.linear_start",
    "lconv1d.linear_end",
];

/// Read `input_min/max`, `output_min/max` scalars for every clippable linear of
/// `layers` layers, keyed `layers.{i}.{name}` (relative to the tower root).
pub fn read_clip_bounds(
    model_dir: &std::path::Path,
    key_prefix: &str,
    names: &[&str],
    layers: usize,
) -> Result<std::collections::HashMap<String, ClipBounds>> {
    use burn_store::ModuleStore;
    let mut store = SafetensorsStore::from_file(model_dir.join("model.safetensors"));
    let mut read = |key: &str| -> Result<f32> {
        let tensor = store
            .get_tensor(key)?
            .ok_or_else(|| anyhow::anyhow!("missing clip scalar {key}"))?;
        let data = burn_store::bridge::to_data(tensor)?.convert_dtype(DType::F32);
        let v: Vec<f32> = data.try_to_vec()?;
        v.first()
            .copied()
            .ok_or_else(|| anyhow::anyhow!("empty clip scalar {key}"))
    };
    let mut bounds = std::collections::HashMap::new();
    for layer in 0..layers {
        for name in names {
            let base = format!("{key_prefix}.{layer}.{name}");
            bounds.insert(
                format!("layers.{layer}.{name}"),
                ClipBounds {
                    input_min: read(&format!("{base}.input_min"))?,
                    input_max: read(&format!("{base}.input_max"))?,
                    output_min: read(&format!("{base}.output_min"))?,
                    output_max: read(&format!("{base}.output_max"))?,
                },
            );
        }
    }
    Ok(bounds)
}

/// Load the E2B model: text backbone (`model.language_model.*`) plus the Gemma 4
/// vision/audio towers (`model.vision_tower.*`, `model.audio_tower.*`).
pub fn load_gen_model(
    model_dir: &std::path::Path,
    dtype: LoadDtype,
    device: &Device,
) -> Result<(GenRoot, Emb2Config)> {
    let cfg = Emb2Config::from_file(&model_dir.join("config.json"))?;
    let vision_cfg = cfg
        .vision_config
        .as_ref()
        .context("checkpoint has no vision_config")?;
    let audio_cfg = cfg
        .audio_config
        .as_ref()
        .context("checkpoint has no audio_config")?;
    let t0 = std::time::Instant::now();
    // Only checkpoints trained with clipped linears carry the bound scalars
    // (the QAT export sets `use_clipped_linears: false` and has none).
    let vision_bounds = if vision_cfg.use_clipped_linears {
        read_clip_bounds(
            model_dir,
            "model.vision_tower.encoder.layers",
            &VISION_LINEAR_NAMES,
            vision_cfg.num_hidden_layers,
        )?
    } else {
        Default::default()
    };
    let audio_bounds = if audio_cfg.use_clipped_linears {
        read_clip_bounds(
            model_dir,
            "model.audio_tower.layers",
            &AUDIO_LINEAR_NAMES,
            audio_cfg.num_hidden_layers,
        )?
    } else {
        Default::default()
    };
    let mut model = GenRoot::new(
        &cfg.text_config,
        vision_cfg,
        audio_cfg,
        &vision_bounds,
        &audio_bounds,
        device,
    );
    // Pre-quantized (QAT) checkpoints carry a `quantization_config`; their
    // weights are packed INT2/4/8 with scales and SRQ activation rounding.
    let cfg_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(model_dir.join("config.json"))?)?;
    let qat = crate::gemma::qat::QatConfig::from_json(&cfg_json)?;
    if qat.is_some() {
        // The PLE table is read packed; stub the parameter first so the
        // dequantized table is never allocated.
        model.text_mut().shrink_ple_table();
    }
    let mut store =
        SafetensorsStore::from_file(model_dir.join("model.safetensors")).allow_partial(true);
    let scales = match &qat {
        Some(_) => {
            let scales = crate::gemma::qat::read_scales(&mut store)?;
            println!("QAT checkpoint: {} quantized modules", scales.len());
            Some(std::sync::Arc::new(scales))
        }
        None => None,
    };
    let mut store = match &qat {
        Some(qat) => store.with_from_adapter(
            QatAdapter {
                qat: std::sync::Arc::new(qat.clone()),
                scales: scales.clone().expect("scales"),
                dtype,
            }
            .chain(LoadDtypeAdapter::new(device, dtype).with_f32_tables(dtype == LoadDtype::F32))
            .chain(PyTorchToBurnAdapter),
        ),
        None => store
            .with_from_adapter(PyTorchToBurnAdapter.chain(LoadDtypeAdapter::new(device, dtype))),
    };
    let result = model
        .load_from(&mut store)
        .with_context(|| format!("load {}", model_dir.display()))?;
    // KV-shared layers never compute K/V, so checkpoints may omit their
    // weights (the QAT export does; the bf16 one ships them anyway).
    let first_shared = cfg
        .text_config
        .num_hidden_layers
        .saturating_sub(cfg.text_config.num_kv_shared_layers);
    crate::util::store::check_load_report(&result, |name| is_shared_kv_param(name, first_shared))?;
    let tolerated = result.missing.len();
    if qat.is_some() {
        let bits = qat
            .as_ref()
            .and_then(|q| q.bits_for("model.language_model.embed_tokens_per_layer"))
            .unwrap_or(4);
        let table = crate::gemma::qat::read_packed_table(
            &mut store,
            "model.language_model.embed_tokens_per_layer",
            bits,
        )?;
        println!(
            "QAT: PLE table kept packed ({} bits, {:.2} GiB vs {:.2} GiB dequantized)",
            table.bits,
            (table.data.len() + table.scales.len() * 4) as f64 / (1u64 << 30) as f64,
            (table.rows * table.k * 4) as f64 / (1u64 << 30) as f64
        );
        model.text_mut().set_packed_ple(table);
    }
    if let Ok(want) = std::env::var("DUMP_PARAM") {
        dump_param(&model, &want)?;
    }
    if let Some(scales) = &scales {
        if std::env::var("NO_SRQ").is_ok() {
            println!("QAT: SRQ disabled (NO_SRQ)");
        } else {
            let n = register_srq_scales(&model, scales);
            println!("QAT: SRQ activation rounding on {n} linears");
        }
    }
    if tolerated > 0 {
        println!("note: {tolerated} KV-shared layer weights absent (never used)");
    }
    println!(
        "loaded {} tensors{} in {:.2}s (resident {:.0} MiB anon)",
        result.applied.len(),
        match dtype {
            LoadDtype::F32 => " (f32 projections, f16 tables)",
            LoadDtype::F16 => " (f16 projections, f16 tables)",
            LoadDtype::Q8 => " (f16 load -> Q8 projections, f16 tables)",
            LoadDtype::Bf16 => " (native bf16)",
        },
        t0.elapsed().as_secs_f64(),
        rss_mib()
    );
    drop(store);
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    unsafe {
        libc::malloc_trim(0);
    }
    let model = match dtype {
        LoadDtype::Q8 => quantize_gen(model)?,
        LoadDtype::F16 => {
            crate::gemma::layers::set_f16_weights(true);
            model
        }
        LoadDtype::Bf16 => {
            crate::gemma::layers::set_bf16_weights(true);
            model
        }
        LoadDtype::F32 => model,
    };
    crate::gemma::layers::set_model_dtype(dtype.tensor_dtype());
    Ok((model, cfg))
}

/// Transposed, tied LM head `[hidden, vocab]` for the logits path, in the
/// projections' dtype (f32 parity; f16 in q8/f16 mode; native bf16).
pub fn build_lm_head(text: &GenTextModel, dtype: DType, device: &Device) -> Tensor<2> {
    text.embed_table()
        .val()
        .cast(dtype)
        .swap_dims(0, 1)
        .to_device(device)
}

fn rss_mib() -> f64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in s.lines() {
        if let Some(v) = line.strip_prefix("RssAnon:") {
            return v
                .trim()
                .trim_end_matches(" kB")
                .parse::<f64>()
                .unwrap_or(0.0)
                / 1024.0;
        }
    }
    0.0
}

// ---------------------------------------------------------------------------
// QAT (pre-quantized) checkpoints
// ---------------------------------------------------------------------------

/// Unpacks packed INT2/INT4/INT8 weights (and `embedding_quantized` tables) into
/// ordinary float tensors while the store applies them, so the model tree sees
/// the same `Linear`/`Embedding` fields as for a bf16 checkpoint.
pub struct QatAdapter {
    qat: std::sync::Arc<crate::gemma::qat::QatConfig>,
    scales: std::sync::Arc<std::collections::HashMap<String, crate::gemma::qat::ModuleScales>>,
    dtype: LoadDtype,
}

impl ModuleAdapter for QatAdapter {
    fn adapt(&self, tensor: PackTensor, _ctx: ModuleContext<'_>) -> PackTensor {
        use crate::gemma::qat::{dequantize_weight, values_per_byte};
        use burn_store::burn_pack::Shape;

        let name = tensor.name.clone();
        let (target_name, module_path) = if let Some(p) = name.strip_suffix(".embedding_quantized")
        {
            (format!("{p}.weight"), p.to_string())
        } else if let Some(p) = name.strip_suffix(".weight") {
            (name.clone(), p.to_string())
        } else {
            // scale tensors and other auxiliaries: no target parameter
            return tensor;
        };
        // The PLE table stays packed (rows are dequantized on lookup).
        if module_path.ends_with("embed_tokens_per_layer") {
            use burn_store::burn_pack::Shape;
            return bridge::map_data(tensor, target_name, DType::F32, Shape::new([1, 1]), |_| {
                TensorData::new(vec![0.0f32], [1, 1])
            });
        }
        let Some(bits) = self.qat.bits_for(&module_path) else {
            return tensor;
        };
        let Some(scales) = self.scales.get(&module_path) else {
            return tensor;
        };
        let dims: Vec<usize> = tensor.shape.clone().into();
        let (rows, packed_k) = match dims.as_slice() {
            [r, k] => (*r, *k),
            _ => return tensor,
        };
        let k = packed_k * values_per_byte(bits);
        let target = match self.dtype {
            LoadDtype::F32 => DType::F32,
            LoadDtype::F16 | LoadDtype::Q8 => DType::F16,
            LoadDtype::Bf16 => DType::BF16,
        };
        let scales = scales.clone();
        let shape = Shape::new([rows, k]);
        bridge::map_data(tensor, target_name, target, shape, move |data| {
            let bytes: Vec<u8> = if data.dtype() == DType::U8 {
                data.try_to_vec().expect("packed u8")
            } else {
                data.try_to_vec::<i8>()
                    .expect("packed i8")
                    .into_iter()
                    .map(|v| v as u8)
                    .collect()
            };
            let w = dequantize_weight(&bytes, bits, rows, k, &scales);
            TensorData::new(w, [rows, k]).convert_dtype(target)
        })
    }

    fn get_alternative_param_name(&self, param_name: &str, container_type: &str) -> Option<String> {
        // Quantized embedding tables are stored as `embedding_quantized`.
        if param_name == "weight" && container_type == "Struct:Embedding" {
            return Some("embedding_quantized".to_string());
        }
        None
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(Self {
            qat: self.qat.clone(),
            scales: self.scales.clone(),
            dtype: self.dtype,
        })
    }
}

/// Register the SRQ scales of every quantized linear, keyed by the weight's
/// `ParamId` (the visitor path matches the checkpoint's module paths).
fn register_srq_scales(
    model: &GenRoot,
    scales: &std::collections::HashMap<String, crate::gemma::qat::ModuleScales>,
) -> usize {
    use burn::module::{ModuleVisitor, Param};

    #[derive(Default)]
    struct Paths {
        stack: Vec<String>,
        out: Vec<(String, burn::module::ParamId)>,
    }
    impl ModuleVisitor for Paths {
        fn enter_module(&mut self, name: &str, _container_type: &str) {
            self.stack.push(name.to_string());
        }
        fn exit_module(&mut self, _name: &str, _container_type: &str) {
            self.stack.pop();
        }
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
            self.out.push((self.stack.join("."), param.id));
        }
    }
    let mut paths = Paths::default();
    model.visit(&mut paths);
    if std::env::var("DUMP_SRQ").is_ok() {
        eprintln!(
            "debug: {} param paths, {} scale keys",
            paths.out.len(),
            scales.len()
        );
        for (p, _) in paths.out.iter().take(6) {
            eprintln!("  param: {p}");
        }
        for k in scales.keys().take(6) {
            eprintln!("  scale: {k}");
        }
    }
    let mut n = 0;
    for (path, id) in paths.out {
        // The visitor's stack ends at the parameter itself (`...q_proj.weight`).
        let key = path.strip_suffix(".weight").unwrap_or(&path);
        if let Some(s) = scales.get(key)
            && (s.in_scale != 0.0 || s.out_scale != 0.0)
        {
            crate::gemma::layers::register_srq(id, s.in_scale, s.out_scale);
            n += 1;
        }
    }
    n
}

/// `model.language_model.layers.{i}.self_attn.{k_proj,k_norm,v_proj,v_norm}` for
/// a KV-shared layer (never used by the forward).
fn is_shared_kv_param(name: &str, first_shared: usize) -> bool {
    let Some(rest) = name.strip_prefix("model.language_model.layers.") else {
        return false;
    };
    let Some((idx, tail)) = rest.split_once('.') else {
        return false;
    };
    let Ok(i) = idx.parse::<usize>() else {
        return false;
    };
    i >= first_shared
        && [
            "self_attn.k_proj",
            "self_attn.k_norm",
            "self_attn.v_proj",
            "self_attn.v_norm",
        ]
        .iter()
        .any(|p| tail.starts_with(p))
}

/// Debug: dump one loaded parameter as raw f32 + dims (compare with the
/// reference's dequantized weights).
fn dump_param(model: &GenRoot, want: &str) -> Result<()> {
    use burn::module::{ModuleVisitor, Param};

    #[derive(Default)]
    struct Find<'a> {
        stack: Vec<String>,
        want: &'a str,
        found: Option<(String, Vec<usize>, Vec<f32>)>,
    }
    impl ModuleVisitor for Find<'_> {
        fn enter_module(&mut self, name: &str, _ct: &str) {
            self.stack.push(name.to_string());
        }
        fn exit_module(&mut self, _name: &str, _ct: &str) {
            self.stack.pop();
        }
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
            let path = self.stack.join(".");
            if self.found.is_none() && path.contains(self.want) {
                let dims: Vec<usize> = param.val().dims().to_vec();
                let values: Vec<f32> = param
                    .val()
                    .cast(DType::F32)
                    .into_data()
                    .try_to_vec()
                    .expect("f32");
                self.found = Some((path, dims, values));
            }
        }
    }
    let mut find = Find {
        want,
        ..Default::default()
    };
    model.visit(&mut find);
    let (path, dims, values) = find
        .found
        .ok_or_else(|| anyhow::anyhow!("no param matching {want}"))?;
    let out = std::env::var("DUMP_PARAM_OUT").unwrap_or_else(|_| "/tmp/param.bin".into());
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in &values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(&out, &bytes)?;
    println!("dumped {path} {dims:?} ({} values) to {out}", values.len());
    Ok(())
}
