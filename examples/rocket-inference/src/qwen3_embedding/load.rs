//! Qwen3-Embedding model loading: f32/f16 weights, Q8 low-RAM (`--quant q8`)
//! and pack-and-drop NPU loading (`--npu`).

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

use crate::qwen3_embedding::model::{Qwen3Config, Qwen3Embedding};
use crate::util::rss_mib;

pub fn load_model(
    model_dir: &Path,
    dtype: DType,
    quant_q8: bool,
    npu: bool,
    npu_threads: usize,
    npu_attn: bool,
    device: &Device,
) -> Result<(Qwen3Embedding, Qwen3Config)> {
    let cfg = Qwen3Config::from_file(&model_dir.join("config.json"))?;
    let _ = (npu_threads, npu_attn); // only used by the aarch64+npu build
    let mut model = Qwen3Embedding::new(&cfg, device);
    let t0 = Instant::now();
    let mut store = SafetensorsStore::from_file(model_dir.join("model.safetensors"))
        .with_from_adapter(PyTorchToBurnAdapter.chain(FloatCastAdapter::to(dtype)));
    let result = model
        .load_from(&mut store)
        .with_context(|| format!("load {}", model_dir.display()))?;
    crate::util::store::check_load_report(&result, |_| false)?;
    println!(
        "loaded {} tensors as {:?} in {:.2}s",
        result.applied.len(),
        dtype,
        t0.elapsed().as_secs_f64()
    );
    drop(store);
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    if npu {
        if quant_q8 {
            bail!("--npu packs its own weights; --quant q8 is not applicable");
        }
        return load_npu_projections(
            model,
            cfg,
            model_dir,
            dtype,
            npu_threads,
            npu_attn,
            device,
            t0,
        );
    }
    #[cfg(not(all(feature = "npu", target_arch = "aarch64")))]
    if npu {
        bail!("--npu requires an aarch64 build with --features npu");
    }
    // NPU builds skip the projection fields in the module tree (the NPU loader
    // packs them straight from the store); load them here for the CPU paths.
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    load_cpu_projections(&mut model, &cfg, model_dir, dtype, quant_q8, device)?;
    if quant_q8 {
        use crate::util::quant::{LowRam, param_group};
        let t0 = Instant::now();
        // Low-RAM mode. Projection weights stay Q8_0-quantized (symmetric int8,
        // 32-value blocks, f16 block scales = llama.cpp's Q8_0); the forward dequantizes
        // each weight on the fly, so only the current layer's weights are materialized
        // (~62 MB) instead of all of them. The token-embedding table stays in the
        // model's native dtype (16-bit for bf16/f16 models); with an f32 body it is
        // kept in f16 (exact for bf16-sourced values in range, and half the bytes)
        // and gathered rows are cast back per forward.
        let proj_group = param_group(r"\.(q|k|v|o|gate|up|down)_proj\.weight$")?;
        let embed_group = param_group(r"embed_tokens\.weight$")?;
        let table_dtype = (dtype == DType::F32).then_some(DType::F16);
        let mut mapper = LowRam::new(vec![proj_group], vec![embed_group], table_dtype);
        model = model.map(&mut mapper);
        model.set_quantized(true);
        model.set_body_dtype(dtype);
        // Return freed pages to the OS: glibc keeps them in its arenas,
        // which would hide the memory saving (measured: ~1.3 GB retained).
        #[cfg(all(target_os = "linux", target_env = "gnu"))]
        unsafe {
            libc::malloc_trim(0);
        }
        println!(
            "Q8-resident projections + {:?} embedding table in {:.2}s (resident {:.0} MiB anon)",
            table_dtype.unwrap_or(dtype),
            t0.elapsed().as_secs_f64(),
            rss_mib()
        );
    }
    Ok((model, cfg))
}

/// Read one projection weight from the store as `[n, k]` data in `dtype`.
#[cfg(all(feature = "npu", target_arch = "aarch64"))]
fn read_projection(
    store: &mut SafetensorsStore,
    layer: usize,
    kind: crate::util::proj::ProjKind,
    dtype: DType,
) -> Result<(usize, usize, TensorData)> {
    use burn_store::ModuleStore;
    let key = format!("layers.{layer}.{}", kind.key());
    let tensor = store
        .get_tensor(&key)?
        .ok_or_else(|| anyhow::anyhow!("missing tensor {key}"))?;
    let data = burn_store::bridge::to_data(tensor)?.convert_dtype(dtype);
    let [n, k] = data.shape().dims::<2>();
    Ok((k, n, data))
}

/// In NPU builds the projection fields are `#[module(skip)]` (the NPU loader packs
/// them straight from the store), so CPU-mode runs must load them explicitly.
/// With `quant_q8` the weights are quantized on the fly, matching the low-RAM mode
/// of the non-NPU build.
#[cfg(all(feature = "npu", target_arch = "aarch64"))]
fn load_cpu_projections(
    model: &mut Qwen3Embedding,
    cfg: &Qwen3Config,
    model_dir: &Path,
    dtype: DType,
    quant_q8: bool,
    device: &Device,
) -> Result<()> {
    use crate::util::proj::{Proj, ProjKind};
    use burn::module::Param;
    use burn::tensor::quantization::{
        Calibration, QuantScheme, QuantValue, ScaleDtype, compute_q_params, compute_range,
    };

    let t0 = Instant::now();
    let mut store = SafetensorsStore::from_file(model_dir.join("model.safetensors"));
    let scheme = QuantScheme::default()
        .with_value(QuantValue::Q8S)
        .per_block([32], ScaleDtype::F16);
    for layer in 0..cfg.num_hidden_layers {
        for kind in ProjKind::ALL {
            let (_, _, data) = read_projection(&mut store, layer, kind, dtype)?;
            // The store holds PyTorch `[out, in]`; Burn's `Linear` wants `[in, out]`.
            let t = Tensor::<2>::from_data(data, (&*device, dtype)).swap_dims(0, 1);
            let t = if quant_q8 {
                let t = t.cast(DType::F32);
                let range = compute_range(&scheme, &t, &Calibration::MinMax);
                let qparams = compute_q_params(&scheme, range);
                t.quantize(&scheme, qparams)
            } else {
                t.cast(dtype)
            };
            match model.projection_mut(layer, kind) {
                Proj::Cpu(lin) => lin.weight = Param::from_tensor(t),
                Proj::Npu(_) => unreachable!("projections are CPU-resident at load time"),
            }
        }
    }
    println!(
        "loaded {} projection weights{} in {:.2}s",
        cfg.num_hidden_layers * ProjKind::ALL.len(),
        if quant_q8 { " (Q8-resident)" } else { "" },
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

/// Pack every projection weight straight into NPU buffers (reusing the model's
/// native dtype — f32 packs a resident fp16 weight, bf16 stays native and uses
/// the bf16 stream; HF `[out, in]` = `[N, K]` layout, no transpose). Pack-and-drop:
/// the CPU never holds projection weights. The token-embedding table keeps the
/// model dtype (f32 builds cast it to f16 to save the bytes).
#[cfg(all(feature = "npu", target_arch = "aarch64"))]
fn load_npu_projections(
    mut model: Qwen3Embedding,
    cfg: Qwen3Config,
    model_dir: &Path,
    dtype: DType,
    npu_threads: usize,
    npu_attn: bool,
    device: &Device,
    t0: Instant,
) -> Result<(Qwen3Embedding, Qwen3Config)> {
    use crate::util::proj::{FusedGroup, Proj, ProjKind};

    burn_rocket::init(npu_threads)
        .map_err(|e| anyhow::anyhow!("NPU context creation failed: {e}"))?;
    crate::util::glue::set_glue(true);

    let mut store = SafetensorsStore::from_file(model_dir.join("model.safetensors"));
    let t_npu = Instant::now();
    let mut bytes = 0usize;

    /// One `pack2`/`pack3` call for the group's parts (output-column order).
    fn pack_group(parts: Vec<Tensor<2>>) -> burn_rocket::WeightId {
        let mut it = parts.into_iter();
        match (it.next(), it.next(), it.next(), it.next()) {
            (Some(a), Some(b), Some(c), None) => burn_rocket::pack3(a, b, c),
            (Some(a), Some(b), None, None) => burn_rocket::pack2(a, b),
            _ => unreachable!("fused groups have 2 or 3 members"),
        }
    }

    for layer in 0..cfg.num_hidden_layers {
        // o and down keep individual weights.
        for kind in [ProjKind::O, ProjKind::Down] {
            let (k, n, data) = read_projection(&mut store, layer, kind, dtype)?;
            bytes += k * n * 2;
            let id = burn_rocket::pack(Tensor::<2>::from_data(data, (&*device, dtype)));
            *model.projection_mut(layer, kind) = Proj::Npu(id);
        }
        // q|k|v and gate|up are packed as one weight each (one matmul,
        // one A-pack per group).
        let groups: [(FusedGroup, &[ProjKind]); 2] = [
            (FusedGroup::Qkv, &[ProjKind::Q, ProjKind::K, ProjKind::V]),
            (FusedGroup::GateUp, &[ProjKind::Gate, ProjKind::Up]),
        ];
        for (group, kinds) in groups {
            let mut parts = Vec::new();
            let mut k = 0usize;
            for &kind in kinds {
                let (kk, n, data) = read_projection(&mut store, layer, kind, dtype)?;
                if k == 0 {
                    k = kk;
                }
                assert_eq!(k, kk, "fused group members must share K");
                bytes += kk * n * 2;
                parts.push(Tensor::<2>::from_data(data, (&*device, dtype)));
            }
            model.set_fused_handle(layer, group, pack_group(parts));
        }
    }
    if npu_attn {
        model.set_npu_attn(true);
        println!("NPU: attention offload enabled (n_head=16, n_kv=8, head_dim=128)");
    }
    if dtype == DType::F32 {
        model.embed_table_to_f16();
    } else {
        model.set_quantized(true);
        model.set_body_dtype(dtype);
    }
    println!(
        "NPU: offloaded {} {} projections ({:.2} GiB, {} threads) in {:.2}s; \
         model ready in {:.2}s (resident {:.0} MiB anon)",
        cfg.num_hidden_layers * ProjKind::ALL.len(),
        if dtype == DType::F32 {
            "resident fp16".to_string()
        } else if burn_rocket::bf16_stream_mode() {
            "native bf16 via the bf16 stream".to_string()
        } else {
            "native bf16 packed resident fp16".to_string()
        },
        bytes as f64 / (1u64 << 30) as f64,
        npu_threads,
        t_npu.elapsed().as_secs_f64(),
        t0.elapsed().as_secs_f64(),
        rss_mib()
    );
    Ok((model, cfg))
}
