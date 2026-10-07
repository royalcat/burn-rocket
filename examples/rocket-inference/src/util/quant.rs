//! Low-RAM Q8_0 quantization (`--quant q8`): projection weights stay quantized
//! in memory and are dequantized per call, so only the current layer's f32
//! weights are materialized.

use anyhow::Result;
use burn::module::{ModuleMapper, Param, ParamGroup};
use burn::prelude::*;
use burn::tensor::DType;
use burn::tensor::quantization::{
    Calibration, QuantScheme, QuantValue, ScaleDtype, compute_q_params, compute_range,
};

/// llama.cpp's Q8_0 layout: symmetric int8, 32-value blocks, f16 block scales.
pub fn q8_scheme() -> QuantScheme {
    QuantScheme::default()
        .with_value(QuantValue::Q8S)
        .per_block([32], ScaleDtype::F16)
}

/// Compile a param-group regex with a readable error.
pub fn param_group(regex: &str) -> Result<ParamGroup> {
    ParamGroup::from_regex(regex).map_err(|e| anyhow::anyhow!("bad param group regex: {e:?}"))
}

/// Map every float param matching one of `quantize` to [`q8_scheme`] and every
/// param matching one of `to_f16` to f16; everything else is left untouched.
///
/// Params are cast to f32 before quantization (a no-op in the modes that allow
/// `--quant q8`, and the safe behavior for checkpoints loaded in another dtype).
pub struct LowRam {
    scheme: QuantScheme,
    quantize: Vec<ParamGroup>,
    to_f16: Vec<ParamGroup>,
    path: Vec<String>,
}

impl LowRam {
    pub fn new(quantize: Vec<ParamGroup>, to_f16: Vec<ParamGroup>) -> Self {
        Self {
            scheme: q8_scheme(),
            quantize,
            to_f16,
            path: Vec::new(),
        }
    }
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
        if self
            .quantize
            .iter()
            .any(|g| g.matches(&param.id, Some(&path)))
        {
            param.map(|tensor| {
                let tensor = tensor.cast(DType::F32);
                let range = compute_range(&self.scheme, &tensor, &Calibration::MinMax);
                let qparams = compute_q_params(&self.scheme, range);
                tensor.quantize(&self.scheme, qparams)
            })
        } else if self
            .to_f16
            .iter()
            .any(|g| g.matches(&param.id, Some(&path)))
        {
            param.map(|tensor| tensor.cast(DType::F16))
        } else {
            param
        }
    }
}
