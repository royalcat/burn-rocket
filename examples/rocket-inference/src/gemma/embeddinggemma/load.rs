//! EmbeddingGemma 2 model loading: f32 weights, Q8 low-RAM (`--quant q8`) and
//! NPU text-backbone packing (`--npu`).

use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use burn::prelude::*;
use burn::tensor::DType;
#[cfg(all(feature = "npu", target_arch = "aarch64"))]
use burn::tensor::TensorData;
use burn_store::{
    FloatCastAdapter, ModuleAdapter, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore,
};

use crate::gemma::config::Emb2Config;
use crate::gemma::embeddinggemma::model::Emb2Model;
use crate::util::rss_mib;

#[allow(clippy::too_many_arguments)]
pub fn load_model(
    model_dir: &Path,
    dtype: DType,
    quant_q8: bool,
    npu: bool,
    npu_threads: usize,
    npu_attn: bool,
    device: &Device,
) -> Result<(Emb2Model, Emb2Config)> {
    let cfg = Emb2Config::from_file(&model_dir.join("config.json"))?;
    let _ = (npu_threads, npu_attn); // only used by the aarch64+npu build
    if dtype != DType::F32 {
        bail!(
            "--dtype f16 is not numerically supported (RMSNorm/softmax/PLE precision): \
             cosine drops to 0.98 (text) / 0.70 (image) vs the f32 reference; use f32"
        );
    }
    let vision_cfg = cfg
        .vision_config
        .as_ref()
        .context("checkpoint has no vision_config (text-only checkpoints are not supported yet)")?;
    let audio_cfg = cfg
        .audio_config
        .as_ref()
        .context("checkpoint has no audio_config (text-only checkpoints are not supported yet)")?;
    let t0 = Instant::now();
    let audio_bounds = read_audio_clip_bounds(model_dir, audio_cfg)?;
    let mut model = Emb2Model::new(
        &cfg.text_config,
        vision_cfg,
        audio_cfg,
        &audio_bounds,
        device,
    );
    let mut store = SafetensorsStore::from_file(model_dir.join("model.safetensors"))
        .allow_partial(true)
        .with_from_adapter(PyTorchToBurnAdapter.chain(FloatCastAdapter::to(dtype)));
    let result = model
        .load_from(&mut store)
        .with_context(|| format!("load {}", model_dir.display()))?;
    crate::util::store::check_load_report(&result, |_| false)?;
    println!(
        "loaded {} tensors as {:?} in {:.2}s (resident {:.0} MiB anon)",
        result.applied.len(),
        dtype,
        t0.elapsed().as_secs_f64(),
        rss_mib()
    );
    drop(store);
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    if npu {
        if dtype != DType::F32 {
            bail!("--npu needs --dtype f32 (NPU activations are f32 on the CPU side)");
        }
        // With both flags the projections are quantized first and the text ones
        // are then packed from their dequantized values (lowest-memory mode).
        if quant_q8 {
            model = quantize_low_ram(model)?;
        }
        return load_npu_text(model, cfg, npu_threads, npu_attn, device);
    }
    #[cfg(not(all(feature = "npu", target_arch = "aarch64")))]
    if npu {
        bail!("--npu requires an aarch64 build with --features npu");
    }
    if quant_q8 {
        model = quantize_low_ram(model)?;
    }
    Ok((model, cfg))
}

/// Pack the text backbone's projections into resident fp16 NPU buffers
/// (HF `[N, K]` layout, one pack per weight) and drop the f32 copies.
#[cfg(all(feature = "npu", target_arch = "aarch64"))]
fn load_npu_text(
    mut model: Emb2Model,
    cfg: Emb2Config,
    threads: usize,
    npu_attn: bool,
    device: &Device,
) -> Result<(Emb2Model, Emb2Config)> {
    burn_rocket::init(threads)
        .map_err(|e| anyhow::anyhow!("NPU context creation failed: {e}"))?;
    let t0 = Instant::now();
    let mut count = 0usize;
    let mut bytes = 0usize;
    model.text_mut().for_each_projection_mut(|lin| {
        let w = lin.weight.val();
        let w = if crate::gemma::layers::is_quantized() {
            w.dequantize()
        } else {
            w
        };
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
        crate::gemma::layers::register_npu_weight(&lin.weight, id);
        // Shrink the (now redundant) f32 copy in place: `Param::map` keeps the
        // parameter id, so the registry lookup in `lin` still hits.
        let dev = device.clone();
        lin.weight = lin.weight.clone().map(|_| Tensor::zeros([1, 1], &dev));
        count += 1;
        bytes += n * k * 2;
    });
    if npu_attn {
        model.text_mut().set_npu_attn(true);
    }
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    unsafe {
        libc::malloc_trim(0);
    }
    println!(
        "NPU: packed {count} text projections into resident fp16 weights ({:.2} GiB, {threads} threads) \
         in {:.2}s; attention on the {} (resident {:.0} MiB anon)",
        bytes as f64 / (1u64 << 30) as f64,
        t0.elapsed().as_secs_f64(),
        if npu_attn { "NPU" } else { "CPU" },
        rss_mib()
    );
    Ok((model, cfg))
}

/// Low-RAM mode (`--quant q8`): projection weights stay Q8_0-quantized
/// (symmetric int8, 32-value blocks, f16 block scales = llama.cpp's Q8_0) and
/// are dequantized per call, so only the current layer's f32 weights are
/// materialized. The embedding table, norms, scalars and position tables stay
/// f32.
fn quantize_low_ram(model: Emb2Model) -> Result<Emb2Model> {
    use crate::util::quant::{LowRam, param_group};

    let t0 = Instant::now();
    let proj_group = param_group(
        r"(q_proj|k_proj|v_proj|o_proj|post|relative_k_proj|gate_proj|up_proj|down_proj|per_layer_model_projection|per_layer_input_gate|per_layer_projection|embedding_projection|input_proj|input_proj_linear|ffw_layer_1|ffw_layer_2|linear_start|linear_end|output_proj)\.(linear\.)?weight$",
    )?;
    let mut mapper = LowRam::new(vec![proj_group], Vec::new());
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

/// Read the audio tower's `Gemma4ClippableLinear` clip bounds (`input_min` etc.
/// scalars) straight from the checkpoint, keyed by module path relative to
/// `audio_tower` (e.g. `layers.0.feed_forward1.ffw_layer_1`).
fn read_audio_clip_bounds(
    model_dir: &Path,
    cfg: &crate::gemma::config::AudioConfig,
) -> Result<std::collections::HashMap<String, crate::gemma::layers::ClipBounds>> {
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
    let names = [
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
    let mut bounds = std::collections::HashMap::new();
    for layer in 0..cfg.num_hidden_layers {
        for name in names {
            let base = format!("audio_tower.layers.{layer}.{name}");
            let clip = crate::gemma::layers::ClipBounds {
                input_min: read(&format!("{base}.input_min"))?,
                input_max: read(&format!("{base}.input_max"))?,
                output_min: read(&format!("{base}.output_min"))?,
                output_max: read(&format!("{base}.output_max"))?,
            };
            bounds.insert(format!("layers.{layer}.{name}"), clip);
        }
    }
    Ok(bounds)
}
