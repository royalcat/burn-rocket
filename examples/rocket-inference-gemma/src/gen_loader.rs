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

use anyhow::{Context, Result};
use burn::prelude::*;
use burn::tensor::DType;
use burn::tensor::quantization::{
    Calibration, QuantScheme, QuantValue, ScaleDtype, compute_q_params, compute_range,
};
use burn_store::burn_pack::Tensor as PackTensor;
use burn_store::{
    FloatCastAdapter, ModuleAdapter, ModuleContext, ModuleSnapshot, PyTorchToBurnAdapter,
    SafetensorsStore, bridge,
};

use crate::config::Emb2Config;
use crate::gen_model::{GenRoot, GenTextModel};

/// Q8_0 (llama.cpp layout: symmetric int8, 32-value blocks, f16 scales).
pub fn q8_scheme() -> QuantScheme {
    QuantScheme::default()
        .with_value(QuantValue::Q8S)
        .per_block([32], ScaleDtype::F16)
}

/// Per-tensor dtype selection, applied during loading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadDtype {
    /// f32 projections (parity).
    F32,
    /// f16 projections: half the memory, f16 GEMM.
    F16,
    /// f16 during load, then Q8_0-resident (memory-only: decode dequantizes).
    Q8,
}

#[derive(Debug, Clone)]
pub struct LoadDtypeAdapter {
    device: Device,
    q8: bool,
    f16: bool,
}

impl LoadDtypeAdapter {
    pub fn new(device: &Device, dtype: LoadDtype) -> Self {
        Self {
            device: device.clone(),
            q8: dtype == LoadDtype::Q8,
            f16: matches!(dtype, LoadDtype::F16 | LoadDtype::Q8),
        }
    }
}

impl ModuleAdapter for LoadDtypeAdapter {
    fn adapt(&self, tensor: PackTensor, ctx: ModuleContext<'_>) -> PackTensor {
        let name = tensor.name.clone();
        let shape = tensor.shape.clone();
        let is_table = name.ends_with("embed_tokens.weight")
            || name.ends_with("embed_tokens_per_layer.weight");
        let is_linear = ctx.module_type() == Some("Struct:Linear");

        if is_table {
            // f16 storage; the gathered rows are cast back to f32.
            bridge::map_data(tensor, name, DType::F16, shape, |data| {
                data.convert_dtype(DType::F16)
            })
        } else if is_linear && self.f16 {
            // f16 during load (halves the f32 footprint); the Q8 pass runs after
            // the model is fully loaded, because the store applies tensors in the
            // backend's packed layout, which cannot be produced from the adapter.
            bridge::map_data(tensor, name, DType::F16, shape, |data| {
                data.convert_dtype(DType::F16)
            })
        } else {
            bridge::map_data(tensor, name, DType::F32, shape, |data| {
                data.convert_dtype(DType::F32)
            })
        }
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(self.clone())
    }
}

/// Q8-resident projections (Q8_0: symmetric int8, 32-value blocks, f16 scales),
/// applied after loading; `lin` dequantizes each weight per call.
pub fn quantize_gen(model: GenRoot) -> Result<GenRoot> {
    use burn::module::{ModuleMapper, Param, ParamGroup};
    let t0 = std::time::Instant::now();
    let scheme = q8_scheme();
    let group = ParamGroup::from_regex(
        r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|per_layer_model_projection|per_layer_input_gate|per_layer_projection)\.weight$",
    )
    .map_err(|e| anyhow::anyhow!("bad param group regex: {e:?}"))?;

    struct LowRam {
        scheme: QuantScheme,
        group: ParamGroup,
        path: Vec<String>,
    }
    impl ModuleMapper for LowRam {
        fn enter_module(&mut self, name: &str, _container_type: &str) {
            self.path.push(name.to_string());
        }
        fn exit_module(&mut self, _name: &str, _container_type: &str) {
            self.path.pop();
        }
        fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
            let path = self.path.join(".");
            if self.group.matches(&param.id, Some(&path)) {
                param.map(|tensor| {
                    let tensor = tensor.cast(DType::F32);
                    let range = compute_range(&self.scheme, &tensor, &Calibration::MinMax);
                    let qparams = compute_q_params(&self.scheme, range);
                    tensor.quantize(&self.scheme, qparams)
                })
            } else {
                param
            }
        }
    }
    let mut mapper = LowRam {
        scheme,
        group,
        path: Vec::new(),
    };
    let model = model.map(&mut mapper);
    crate::layers::set_quantized(true);
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

/// Load the E2B-it text model (`model.language_model.*`); vision/audio keys are
/// left unused until the multimodal phase.
pub fn load_gen_model(
    model_dir: &std::path::Path,
    dtype: LoadDtype,
    device: &Device,
) -> Result<(GenRoot, Emb2Config)> {
    let cfg = Emb2Config::from_file(&model_dir.join("config.json"))?;
    let t0 = std::time::Instant::now();
    let mut model = GenRoot::new(&cfg.text_config, device);
    let mut store = SafetensorsStore::from_file(model_dir.join("model.safetensors"))
        .allow_partial(true)
        .with_from_adapter(
            PyTorchToBurnAdapter.chain(LoadDtypeAdapter::new(device, dtype)),
        );
    let result = model
        .load_from(&mut store)
        .with_context(|| format!("load {}", model_dir.display()))?;
    if !result.errors.is_empty() {
        anyhow::bail!("load errors: {:?}", result.errors);
    }
    if !result.missing.is_empty() {
        anyhow::bail!(
            "{} model parameters missing from file (first: {:?})",
            result.missing.len(),
            result.missing.first()
        );
    }
    println!(
        "loaded {} tensors{} in {:.2}s (resident {:.0} MiB anon)",
        result.applied.len(),
        match dtype {
            LoadDtype::F32 => " (f32 projections, f16 tables)",
            LoadDtype::F16 => " (f16 projections, f16 tables)",
            LoadDtype::Q8 => " (f16 load -> Q8 projections, f16 tables)",
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
            crate::layers::set_f16_weights(true);
            model
        }
        LoadDtype::F32 => model,
    };
    Ok((model, cfg))
}

/// Transposed, tied LM head `[hidden, vocab]` for the logits path. f32 by
/// default (parity); f16 in q8 mode, where logits are computed in f16 like the
/// bf16 reference.
pub fn build_lm_head(text: &GenTextModel, low_precision: bool, device: &Device) -> Tensor<2> {
    let dtype = if low_precision { DType::F16 } else { DType::F32 };
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
            return v.trim().trim_end_matches(" kB").parse::<f64>().unwrap_or(0.0) / 1024.0;
        }
    }
    0.0
}
