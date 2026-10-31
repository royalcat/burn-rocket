//! Qwen3-Embedding model loading: f32/f16 weights, Q8 low-RAM (`--quant q8`)
//! and pack-and-drop NPU loading (`--npu`). The weight source is a
//! [`AnyCheckpoint`] (safetensors or GGUF); tensors arrive in the HF/PyTorch
//! convention, so both formats share this code.

use std::time::Instant;

use anyhow::{Context, Result};
use burn::prelude::*;
use burn::tensor::DType;
#[cfg(all(feature = "npu", target_arch = "aarch64"))]
use burn::tensor::TensorData;

use crate::qwen3_embedding::model::{Qwen3Config, Qwen3Embedding};
use crate::util::checkpoint::{AnyCheckpoint, Checkpoint};
use crate::util::rss_mib;

/// NPU stages for qwen3: the projections and the attention are independent.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(not(all(feature = "npu", target_arch = "aarch64")), allow(dead_code))]
pub struct NpuOpts {
    pub threads: usize,
    pub text: bool,
    pub attn: bool,
}

pub fn load_model(
    ckpt: &mut AnyCheckpoint,
    dtype: DType,
    quant_q8: bool,
    npu: Option<NpuOpts>,
    device: &Device,
) -> Result<(Qwen3Embedding, Qwen3Config)> {
    let cfg = Qwen3Config::from_value(ckpt.config_json()?)?;
    let mut model = Qwen3Embedding::new(&cfg, device);
    let t0 = Instant::now();
    let result = ckpt
        .load_into(&mut model, dtype)
        .with_context(|| format!("load {} checkpoint", ckpt.format()))?;
    crate::util::store::check_load_report(&result, |_| false)?;
    println!(
        "loaded {} tensors as {:?} from {} in {:.2}s",
        result.applied.len(),
        dtype,
        ckpt.format(),
        t0.elapsed().as_secs_f64()
    );
    #[allow(unused_mut)]
    let mut npu_attn = false;
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    if let Some(opts) = npu {
        if opts.text {
            if quant_q8 {
                eprintln!(
                    "note: --quant q8 applies to CPU-resident weights only; the qwen3 \
                     text projections are packed from {dtype:?}"
                );
            }
            return load_npu_projections(
                ckpt,
                model,
                cfg,
                dtype,
                opts.threads,
                opts.attn,
                device,
                t0,
            );
        }
        // Attention only (or an empty set): the projections stay on the CPU.
        burn_rocket::init(opts.threads)
            .map_err(|e| anyhow::anyhow!("NPU context creation failed: {e}"))?;
        crate::util::glue::set_glue(true);
        npu_attn = opts.attn;
    }
    #[cfg(not(all(feature = "npu", target_arch = "aarch64")))]
    if npu.is_some() {
        anyhow::bail!("an NPU placement requires an aarch64 build with `--features npu`");
    }
    let _ = &npu;
    let _ = npu_attn;
    // NPU builds skip the projection fields in the module tree (the NPU loader
    // packs them straight from the checkpoint); load them here for the CPU paths.
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    load_cpu_projections(&mut model, ckpt, &cfg, dtype, quant_q8, device)?;
    if quant_q8 {
        use crate::util::quant::{LowRam, param_group};
        let t0 = Instant::now();
        let proj_group = param_group(r"\.(q|k|v|o|gate|up|down)_proj\.weight$")?;
        let embed_group = param_group(r"embed_tokens\.weight$")?;
        let table_dtype = (dtype == DType::F32).then_some(DType::F16);
        let mut mapper = LowRam::new(vec![proj_group], vec![embed_group], table_dtype);
        model = model.map(&mut mapper);
        model.set_quantized(true);
        model.set_body_dtype(dtype);
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
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    if npu_attn {
        model.set_npu_attn(true);
        println!("NPU: attention offload enabled (n_head=16, n_kv=8, head_dim=128)");
    }
    Ok((model, cfg))
}

/// Read one projection weight as `[n, k]` HF-convention data in `dtype`.
#[cfg(all(feature = "npu", target_arch = "aarch64"))]
fn read_projection(
    ckpt: &mut AnyCheckpoint,
    layer: usize,
    kind: crate::util::proj::ProjKind,
    dtype: DType,
) -> Result<(usize, usize, TensorData)> {
    let key = format!("layers.{layer}.{}", kind.key());
    let data = ckpt
        .get(&key, dtype)?
        .ok_or_else(|| anyhow::anyhow!("missing tensor {key}"))?;
    let [n, k] = data.shape().dims::<2>();
    Ok((k, n, data))
}

/// In NPU builds the projection fields are `#[module(skip)]` (the NPU loader packs
/// them straight from the checkpoint), so CPU-mode runs must load them explicitly.
#[cfg(all(feature = "npu", target_arch = "aarch64"))]
fn load_cpu_projections(
    model: &mut Qwen3Embedding,
    ckpt: &mut AnyCheckpoint,
    cfg: &Qwen3Config,
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
    let scheme = QuantScheme::default()
        .with_value(QuantValue::Q8S)
        .per_block([32], ScaleDtype::F16);
    for layer in 0..cfg.num_hidden_layers {
        for kind in ProjKind::ALL {
            let (_, _, data) = read_projection(ckpt, layer, kind, dtype)?;
            // The checkpoint holds HF `[out, in]`; Burn's `Linear` wants `[in, out]`.
            let t = Tensor::<2>::from_data(data, (device, dtype)).swap_dims(0, 1);
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

/// Pack every projection weight straight into NPU buffers (HF `[out, in]` =
/// `[N, K]` layout, no transpose). Pack-and-drop: the CPU never holds projection
/// weights. The token-embedding table keeps the model dtype (f32 builds cast it
/// to f16 to save the bytes).
#[cfg(all(feature = "npu", target_arch = "aarch64"))]
#[allow(clippy::too_many_arguments)]
fn load_npu_projections(
    ckpt: &mut AnyCheckpoint,
    mut model: Qwen3Embedding,
    cfg: Qwen3Config,
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
        for kind in [ProjKind::O, ProjKind::Down] {
            let (k, n, data) = read_projection(ckpt, layer, kind, dtype)?;
            bytes += k * n * 2;
            let id = burn_rocket::pack(Tensor::<2>::from_data(data, (device, dtype)));
            *model.projection_mut(layer, kind) = Proj::Npu(id);
        }
        let groups: [(FusedGroup, &[ProjKind]); 2] = [
            (FusedGroup::Qkv, &[ProjKind::Q, ProjKind::K, ProjKind::V]),
            (FusedGroup::GateUp, &[ProjKind::Gate, ProjKind::Up]),
        ];
        for (group, kinds) in groups {
            let mut parts = Vec::new();
            let mut k = 0usize;
            for &kind in kinds {
                let (kk, n, data) = read_projection(ckpt, layer, kind, dtype)?;
                if k == 0 {
                    k = kk;
                }
                assert_eq!(k, kk, "fused group members must share K");
                bytes += kk * n * 2;
                parts.push(Tensor::<2>::from_data(data, (device, dtype)));
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
