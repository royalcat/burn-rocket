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
    if dtype == DType::BF16 {
        bail!(
            "--dtype bf16 is broken in burn-flex 0.22.0-pre.4 (bf16 embedding gather panics); use f32 (or f16 for a smaller model)"
        );
    }
    let mut model = Qwen3Embedding::new(&cfg, device);
    let t0 = Instant::now();
    let mut store = SafetensorsStore::from_file(model_dir.join("model.safetensors"))
        .with_from_adapter(PyTorchToBurnAdapter.chain(FloatCastAdapter::to(dtype)));
    let result = model
        .load_from(&mut store)
        .with_context(|| format!("load {}", model_dir.display()))?;
    if !result.errors.is_empty() {
        bail!("load errors: {:?}", result.errors);
    }
    if !result.missing.is_empty() {
        bail!(
            "{} model parameters missing from file",
            result.missing.len()
        );
    }
    if !result.unused.is_empty() {
        println!(
            "warning: {} file tensors unused by the model (first: {:?})",
            result.unused.len(),
            result.unused.first()
        );
    }
    println!(
        "loaded {} tensors as {:?} in {:.2}s",
        result.applied.len(),
        dtype,
        t0.elapsed().as_secs_f64()
    );
    drop(store);
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    if npu {
        if dtype != DType::F32 {
            bail!("--npu needs --dtype f32 (NPU activations are f32 on the CPU side)");
        }
        if quant_q8 {
            bail!("--npu packs its own fp16 weights; --quant q8 is not applicable");
        }
        return load_npu_projections(model, cfg, model_dir, npu_threads, npu_attn, device, t0);
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
        if dtype != DType::F32 {
            bail!(
                "--quant q8 needs --dtype f32 (Q8-resident weights are dequantized to f32 on the fly)"
            );
        }
        use crate::util::quant::{LowRam, param_group};
        let t0 = Instant::now();
        // Low-RAM mode. Projection weights stay Q8_0-quantized (symmetric int8,
        // 32-value blocks, f16 block scales = llama.cpp's Q8_0); the forward dequantizes
        // each weight on the fly, so only the current layer's f32 weights are
        // materialized (~62 MB) instead of all 1.75 GB. The token-embedding table is
        // kept in f16 (exact for bf16-sourced values in range) and gathered rows are
        // cast back to f32 per forward. flex's bf16 gather is broken (dtype panic).
        let proj_group = param_group(r"\.(q|k|v|o|gate|up|down)_proj\.weight$")?;
        let embed_group = param_group(r"embed_tokens\.weight$")?;
        let mut mapper = LowRam::new(vec![proj_group], vec![embed_group]);
        model = model.map(&mut mapper);
        model.set_quantized(true);
        // Return freed f32 weight pages to the OS: glibc keeps them in its arenas,
        // which would hide the memory saving (measured: ~1.3 GB retained).
        #[cfg(all(target_os = "linux", target_env = "gnu"))]
        unsafe {
            libc::malloc_trim(0);
        }
        println!(
            "Q8-resident projections + f16 embedding table in {:.2}s (resident {:.0} MiB anon)",
            t0.elapsed().as_secs_f64(),
            rss_mib()
        );
    }
    Ok((model, cfg))
}

/// Read one projection weight from the store as f32 `[n, k]` data.
#[cfg(all(feature = "npu", target_arch = "aarch64"))]
fn read_projection(
    store: &mut SafetensorsStore,
    layer: usize,
    kind: crate::util::proj::ProjKind,
) -> Result<(usize, usize, TensorData)> {
    use burn_store::ModuleStore;
    let key = format!("layers.{layer}.{}", kind.key());
    let tensor = store
        .get_tensor(&key)?
        .ok_or_else(|| anyhow::anyhow!("missing tensor {key}"))?;
    let data = burn_store::bridge::to_data(tensor)?.convert_dtype(DType::F32);
    let [n, k] = data.shape.dims::<2>();
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
            let (_, _, data) = read_projection(&mut store, layer, kind)?;
            // The store holds PyTorch `[out, in]`; Burn's `Linear` wants `[in, out]`.
            let t = Tensor::<2>::from_data(data, device).swap_dims(0, 1);
            let t = if quant_q8 {
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

/// Pack every projection weight straight into resident NPU buffers (fp16, HF
/// `[out, in]` = `[N, K]` layout, no transpose) and keep only an f16 embedding
/// table on the CPU. Pack-and-drop: the CPU never holds projection weights.
#[cfg(all(feature = "npu", target_arch = "aarch64"))]
fn load_npu_projections(
    mut model: Qwen3Embedding,
    cfg: Qwen3Config,
    model_dir: &Path,
    npu_threads: usize,
    npu_attn: bool,
    device: &Device,
    t0: Instant,
) -> Result<(Qwen3Embedding, Qwen3Config)> {
    use crate::util::proj::{FusedGroup, Proj, ProjKind};

    burn_rocket::init(npu_threads)
        .map_err(|e| anyhow::anyhow!("NPU context creation failed: {e}"))?;

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
        // o and down keep individual resident weights.
        for kind in [ProjKind::O, ProjKind::Down] {
            let (k, n, data) = read_projection(&mut store, layer, kind)?;
            bytes += k * n * 2;
            let id = burn_rocket::pack(Tensor::<2>::from_data(data, device));
            *model.projection_mut(layer, kind) = Proj::Npu(id);
        }
        // q|k|v and gate|up are packed as one resident weight each (one matmul,
        // one A-pack per group).
        let groups: [(FusedGroup, &[ProjKind]); 2] = [
            (FusedGroup::Qkv, &[ProjKind::Q, ProjKind::K, ProjKind::V]),
            (FusedGroup::GateUp, &[ProjKind::Gate, ProjKind::Up]),
        ];
        for (group, kinds) in groups {
            let mut parts = Vec::new();
            let mut k = 0usize;
            for &kind in kinds {
                let (kk, n, data) = read_projection(&mut store, layer, kind)?;
                if k == 0 {
                    k = kk;
                }
                assert_eq!(k, kk, "fused group members must share K");
                bytes += kk * n * 2;
                parts.push(Tensor::<2>::from_data(data, device));
            }
            model.set_fused_handle(layer, group, pack_group(parts));
        }
    }
    if npu_attn {
        model.set_npu_attn(true);
        println!("NPU: attention offload enabled (n_head=16, n_kv=8, head_dim=128)");
    }
    model.embed_table_to_f16();
    println!(
        "NPU: packed {} projections into resident fp16 weights ({:.2} GiB, {} threads) in {:.2}s; \
         model ready in {:.2}s (resident {:.0} MiB anon)",
        cfg.num_hidden_layers * ProjKind::ALL.len(),
        bytes as f64 / (1u64 << 30) as f64,
        npu_threads,
        t_npu.elapsed().as_secs_f64(),
        t0.elapsed().as_secs_f64(),
        rss_mib()
    );
    Ok((model, cfg))
}
