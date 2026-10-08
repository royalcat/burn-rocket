//! Shared layer wrappers for the Gemma 4 towers.

use std::sync::atomic::{AtomicBool, Ordering};

use burn::module::Param;
use burn::nn::{Linear, LinearConfig, RmsNorm};
use burn::prelude::*;
use burn::tensor::DType;
use burn::tensor::activation::gelu_approximate;
use burn::tensor::s;

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

/// Linear forward that supports Q8-resident weights, NPU-resident weights, f16
/// weights and SRQ activation rounding (QAT checkpoints).
///
/// The NPU registry is keyed by the weight's `ParamId`, so no model code has to
/// know which projections were packed: the loader packs a weight, registers its
/// id and drops the CPU copy. The SRQ registry works the same way.
pub(crate) fn lin<const D: usize>(l: &Linear, x: Tensor<D>) -> Tensor<D> {
    let srq = srq_lookup(&l.weight);
    let x = match srq {
        Some((in_scale, _)) => apply_srq(x, in_scale),
        None => x,
    };
    let y = linear_body(l, x);
    match srq {
        Some((_, out_scale)) => apply_srq(y, out_scale),
        None => y,
    }
}

fn linear_body<const D: usize>(l: &Linear, x: Tensor<D>) -> Tensor<D> {
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    if let Some(id) = npu_lookup(&l.weight) {
        // Generation packs weights for prefill only: decode keeps the CPU
        // copies (NPU matmuls pad M to 256, so M=1 would be wasted work).
        if !npu_prefill_only() || prefill_mode() {
            return burn_rocket::matmul(x, &id);
        }
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
// Fused CPU glue (NPU build, `--npu`): single-pass host kernels for the
// elementwise chains flex runs as several single-threaded scalar passes.
// ---------------------------------------------------------------------------

/// Set with `--npu`: the CPU glue routes `gelu*up`, RMSNorm and RoPE through
/// `burn-rocket`'s fused kernels. Off in the pure-CPU modes, whose f32 results
/// are bit-comparable with the HF reference.
#[cfg(all(feature = "npu", target_arch = "aarch64"))]
static NPU_GLUE: AtomicBool = AtomicBool::new(false);

#[cfg(all(feature = "npu", target_arch = "aarch64"))]
pub fn set_npu_glue(on: bool) {
    NPU_GLUE.store(on, Ordering::Relaxed);
}

/// Whether the fused glue kernels are active. `--npu` turns them on; the
/// `ROCKET_GLUE` env var overrides both ways (`ROCKET_GLUE=0` for the composite
/// ops — the A/B that validates the kernels against them; `ROCKET_GLUE=1` to
/// test the kernels in a CPU mode).
#[cfg(all(feature = "npu", target_arch = "aarch64"))]
pub fn npu_glue() -> bool {
    static ONCE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ONCE.get_or_init(|| match std::env::var("ROCKET_GLUE") {
        Ok(v) => v != "0",
        Err(_) => NPU_GLUE.load(Ordering::Relaxed),
    })
}

/// `gelu_approximate(gate) * up` (fused single pass with the NPU glue on).
pub(crate) fn gelu_mul<const D: usize>(gate: Tensor<D>, up: Tensor<D>) -> Tensor<D> {
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    if npu_glue() {
        return burn_rocket::gelu_mul(gate, up);
    }
    gelu_approximate(gate) * up
}

/// `RmsNorm::forward` (fused single pass with the NPU glue on).
pub(crate) fn rms_norm<const D: usize>(norm: &RmsNorm, x: Tensor<D>) -> Tensor<D> {
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    if npu_glue() {
        return burn_rocket::rms_norm(x, norm.gamma.val(), norm.epsilon);
    }
    norm.forward(x)
}

/// `rms_norm_noscale` (fused single pass with the NPU glue on).
pub(crate) fn rms_norm_noscale<const D: usize>(x: Tensor<D>, eps: f64) -> Tensor<D> {
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    if npu_glue() {
        return burn_rocket::rms_norm_noscale(x, eps);
    }
    crate::gemma::rms_norm_noscale(x, eps)
}

/// Rotate-half RoPE of `x` `[B, S, H, D]` with `cos`/`sin` `[S, D/2]`
/// (fused single pass with the NPU glue on).
pub(crate) fn rope_apply(x: Tensor<4>, cos: Tensor<2>, sin: Tensor<2>) -> Tensor<4> {
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    if npu_glue() {
        return burn_rocket::rope_apply(x, cos, sin);
    }
    let [_, s, _, d] = x.dims();
    let half = d / 2;
    let cos = cos.reshape([1, s, 1, half]);
    let sin = sin.reshape([1, s, 1, half]);
    let x1 = x.clone().slice(s![.., .., .., 0..half]);
    let x2 = x.slice(s![.., .., .., half..d]);
    let o1 = x1.clone() * cos.clone() - x2.clone() * sin.clone();
    let o2 = x2 * cos + x1 * sin;
    Tensor::cat(vec![o1, o2], 3)
}

// ---------------------------------------------------------------------------
// SRQ (static range quantization) activation rounding, for QAT checkpoints
// ---------------------------------------------------------------------------

/// Calibrated `(input, output)` SRQ scales by weight `ParamId`.
static SRQ: std::sync::Mutex<Option<std::collections::HashMap<u64, (f32, f32)>>> =
    std::sync::Mutex::new(None);

/// Register the SRQ scales of a quantized linear (a scale of 0 is a no-op).
pub fn register_srq(id: burn::module::ParamId, in_scale: f32, out_scale: f32) {
    let mut guard = SRQ
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    guard
        .get_or_insert_with(std::collections::HashMap::new)
        .insert(id.val(), (in_scale, out_scale));
}

fn srq_lookup<const D: usize>(w: &Param<Tensor<D>>) -> Option<(f32, f32)> {
    let guard = SRQ
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    guard.as_ref().and_then(|m| m.get(&w.id.val()).copied())
}

/// `clamp(round(x / scale), -128, 127) * scale` (the int8 SRQ grid, in f32).
///
/// Rounding is ties-to-even, like `torch.round`; the difference matters because
/// a long sequence makes exact `.5` quotients likely somewhere in the model.
pub(crate) fn apply_srq<const D: usize>(x: Tensor<D>, scale: f32) -> Tensor<D> {
    if scale == 0.0 {
        return x;
    }
    let r = x / scale;
    let f = r.clone().floor();
    let frac = r - f.clone();
    let half = frac.clone().equal_elem(0.5);
    let f_mod2 = f.clone() - (f.clone() / 2.0).floor() * 2.0;
    let f_odd = f_mod2.equal_elem(1.0);
    let up = frac.greater_elem(0.5).bool_or(half.bool_and(f_odd));
    let rounded = f + up.float();
    rounded.clamp(-128.0, 127.0) * scale
}

// ---------------------------------------------------------------------------
// NPU-resident weights (aarch64 + `npu` feature)
// ---------------------------------------------------------------------------

/// Packed NPU weights by `ParamId` value.
#[cfg(all(feature = "npu", target_arch = "aarch64"))]
static NPU_WEIGHTS: std::sync::Mutex<
    Option<std::collections::HashMap<u64, burn_rocket::WeightId>>,
> = std::sync::Mutex::new(None);

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

/// Generation mode: registered NPU weights are used only inside prefill.
#[cfg(all(feature = "npu", target_arch = "aarch64"))]
static NPU_PREFILL_ONLY: AtomicBool = AtomicBool::new(false);
static PREFILL_MODE: AtomicBool = AtomicBool::new(false);

#[cfg(all(feature = "npu", target_arch = "aarch64"))]
pub fn set_npu_prefill_only(on: bool) {
    NPU_PREFILL_ONLY.store(on, Ordering::Relaxed);
}

#[cfg(all(feature = "npu", target_arch = "aarch64"))]
pub(crate) fn npu_prefill_only() -> bool {
    NPU_PREFILL_ONLY.load(Ordering::Relaxed)
}

/// Set around the prefill pass of a generation request.
pub fn set_prefill_mode(on: bool) {
    PREFILL_MODE.store(on, Ordering::Relaxed);
}

#[cfg(all(feature = "npu", target_arch = "aarch64"))]
pub fn prefill_mode() -> bool {
    PREFILL_MODE.load(Ordering::Relaxed)
}

/// Pack one `Linear` into resident NPU memory and register it. With
/// `keep_cpu` the f32 copy stays (prefill-only mode), otherwise it is shrunk to
/// a `[1, 1]` placeholder (the id is preserved so `lin` still finds the weight).
#[cfg(all(feature = "npu", target_arch = "aarch64"))]
pub fn pack_linear_into_npu(lin: &mut Linear, device: &Device, keep_cpu: bool) -> (usize, usize) {
    let w = lin.weight.val();
    let [k, n] = w.shape().dims::<2>();
    let values: Vec<f32> = w
        .cast(DType::F32)
        .into_data()
        .try_to_vec()
        .expect("f32 projection weights");
    // Burn `Linear` is `[in, out]`; the NPU packs HF `[out, in]` = `[N, K]`.
    let mut t = vec![0f32; n * k];
    for i in 0..k {
        let src = &values[i * n..(i + 1) * n];
        for j in 0..n {
            t[j * k + i] = src[j];
        }
    }
    let tensor = Tensor::<2>::from_data(TensorData::new(t, [n, k]), device);
    let id = burn_rocket::pack(tensor);
    register_npu_weight(&lin.weight, id);
    if !keep_cpu {
        let dev = device.clone();
        lin.weight = lin.weight.clone().map(|_| Tensor::zeros([1, 1], &dev));
    }
    (n, k)
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
