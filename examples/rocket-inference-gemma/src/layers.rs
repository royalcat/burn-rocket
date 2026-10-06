//! Shared layer wrappers for the Gemma 4 towers.

use std::sync::atomic::{AtomicBool, Ordering};

use burn::module::Param;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::DType;

/// Process-wide low-RAM flag (`--quant q8`): projection weights stay
/// Q8-resident and are dequantized per call, so only the current layer's f32
/// weights are materialized.
static QUANTIZED: AtomicBool = AtomicBool::new(false);

pub fn set_quantized(on: bool) {
    QUANTIZED.store(on, Ordering::Relaxed);
}

pub fn is_quantized() -> bool {
    QUANTIZED.load(Ordering::Relaxed)
}

/// Process-wide f16-weight flag (`gen --f16`): the projection weights are f16
/// and activations are cast per call (f32 compute stays in the norms/attention).
static F16_WEIGHTS: AtomicBool = AtomicBool::new(false);

pub fn set_f16_weights(on: bool) {
    F16_WEIGHTS.store(on, Ordering::Relaxed);
}

pub fn f16_weights() -> bool {
    F16_WEIGHTS.load(Ordering::Relaxed)
}

/// The activation dtype matching a (possibly Q8-resident) weight: activations
/// stay f32 in low-RAM mode because `lin` dequantizes weights to f32.
pub(crate) fn weight_dtype<const D: usize>(w: &Param<Tensor<D>>) -> DType {
    if is_quantized() {
        DType::F32
    } else {
        w.val().dtype()
    }
}

/// Linear forward that supports Q8-resident weights and NPU-resident weights.
///
/// The NPU registry is keyed by the weight's `ParamId`, so no model code has to
/// know which projections were packed: the loader packs a weight, registers its
/// id and drops the CPU copy.
pub(crate) fn lin<const D: usize>(l: &Linear, x: Tensor<D>) -> Tensor<D> {
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    if let Some(id) = npu_lookup(&l.weight) {
        return burn_rocket::matmul(x, &id);
    }
    if is_quantized() {
        let weight = l.weight.val().dequantize();
        return burn::tensor::module::linear(x, weight, l.bias.as_ref().map(|b| b.val()));
    }
    if f16_weights() {
        // f16-resident weights: compute in f16, return in the activation dtype.
        let dt = x.dtype();
        let out = burn::tensor::module::linear(x.cast(DType::F16), l.weight.val(), None);
        return out.cast(dt);
    }
    l.forward(x)
}

// ---------------------------------------------------------------------------
// NPU-resident weights (aarch64 + `npu` feature)
// ---------------------------------------------------------------------------

/// Packed NPU weights by `ParamId` value.
#[cfg(all(feature = "npu", target_arch = "aarch64"))]
static NPU_WEIGHTS: std::sync::Mutex<Option<std::collections::HashMap<u64, burn_rocket::WeightId>>> =
    std::sync::Mutex::new(None);

/// Register a packed weight for `param` (the caller drops the CPU copy).
#[cfg(all(feature = "npu", target_arch = "aarch64"))]
pub fn register_npu_weight(param: &Param<Tensor<2>>, id: burn_rocket::WeightId) {
    let mut guard = NPU_WEIGHTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    guard
        .get_or_insert_with(std::collections::HashMap::new)
        .insert(param.id.val(), id);
}

#[cfg(all(feature = "npu", target_arch = "aarch64"))]
fn npu_lookup(param: &Param<Tensor<2>>) -> Option<burn_rocket::WeightId> {
    let guard = NPU_WEIGHTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    guard.as_ref().and_then(|m| m.get(&param.id.val()).copied())
}

/// Number of registered NPU weights (for loader summaries).
#[cfg(all(feature = "npu", target_arch = "aarch64"))]
pub fn npu_weight_count() -> usize {
    let guard = NPU_WEIGHTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    guard.as_ref().map(|m| m.len()).unwrap_or(0)
}

/// Clipping bounds for `Gemma4ClippableLinear` (audio tower only; the vision
/// tower ships `use_clipped_linears = false` and has no bound tensors).
#[derive(Debug, Clone, Copy)]
pub struct ClipBounds {
    pub input_min: f32,
    pub input_max: f32,
    pub output_min: f32,
    pub output_max: f32,
}

/// A linear wrapped in the checkpoint's `..._proj.linear.weight` naming, with
/// optional input/output clipping (`torch.clamp` on both sides).
#[derive(Module, Debug)]
pub struct ClippableLinear {
    linear: Linear,
    #[module(skip)]
    clip: Option<ClipBounds>,
}

impl ClippableLinear {
    pub fn new(in_features: usize, out_features: usize, device: &Device) -> Self {
        Self {
            linear: LinearConfig::new(in_features, out_features)
                .with_bias(false)
                .init(device),
            clip: None,
        }
    }

    pub fn with_clip(
        in_features: usize,
        out_features: usize,
        clip: Option<ClipBounds>,
        device: &Device,
    ) -> Self {
        Self {
            linear: LinearConfig::new(in_features, out_features)
                .with_bias(false)
                .init(device),
            clip,
        }
    }

    pub fn weight(&self) -> &Param<Tensor<2>> {
        &self.linear.weight
    }

    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        let x = match self.clip {
            Some(c) => x.clamp(c.input_min, c.input_max),
            None => x,
        };
        let y = lin(&self.linear, x);
        match self.clip {
            Some(c) => y.clamp(c.output_min, c.output_max),
            None => y,
        }
    }
}
