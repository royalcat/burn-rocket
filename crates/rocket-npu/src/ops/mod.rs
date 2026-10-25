//! The elementwise / reduction / normalization / composite operation families.
//!
//! [`activation`] is the port of `rocket_activation.c`: the DPU LUT activation family
//! (the standalone flying-mode DPU job), the elementwise binary ops that ride the
//! conv-main EW datapath, and the composite entries on top of them (clip,
//! LeakyReLU/ELU/SELU/PReLU, divide), plus their host references and the LUT model.
//!
//! [`reduce`] is the port of `rocket_reduce.c`: the PPU spatial pools (the telescoping
//! global average/max/min), the feature-axis ones-matmul reduce, and the triangular-ones
//! cumsum, with their host references and the cube-layout helper.
//!
//! [`softmax`] is the port of `rocket_softmax.c`: row softmax / log-softmax /
//! cross-entropy, each a composition of the EXP LUT, the feature reduce and either the
//! per-row scale ([`norm`]) or an EW subtract.
//!
//! [`norm`] is the port of `rocket_norm.c`: the per-row broadcast scale (softmax's tail),
//! RMSNorm and LayerNorm, plus the fp16-square overflow guard they share.
//!
//! [`normvision`] is the port of `rocket_normvision.c`: the `[N,C,P]` vision norms
//! (BatchNorm from running stats, GroupNorm, InstanceNorm) and L2-Normalize.
//!
//! [`ffn`] is the port of `rocket_ffn.c`: the gated activation core (GeGLU/SwiGLU), the
//! three-matmul FFN block, the cross-op fused FFN and the non-gated encoder MLP.
//!
//! [`encoder`] is the port of `rocket_encoder.c`: one pre-norm Whisper/transformer
//! encoder block assembled from the primitives above and [`crate::attn::mha_self`].
//!
//! [`conv`] is the port of `rocket_conv.c` (both halves: the fp16 convolution engine —
//! direct and depthwise, 1D, and the conv→activation fusion with its tiling and
//! resident-BO context — and the native int8 direct conv with its multicore pool plus
//! the int8-out depthwise requant), [`conv_transpose`] is
//! `rocket_conv_transpose.c`'s transposed conv (the hardware deconvolution route and the
//! dilated-input lowering), [`pool`] is `rocket_pool.c` (the PPU MaxPool/AveragePool
//! with the int8/uint8 entries) and [`resize`] is `rocket_resize.c` (the integer-factor
//! upsample realised as a depthwise transposed conv).
//!
//! [`sam`] and [`siglip`] are the two model-level encoders (the M9 capstone):
//! `rocket_sam_encoder.c` — the windowed, relative-position-biased SAM ViT-Det image
//! encoder with its conv neck — and `rocket_siglip_encoder.c` — the plain-ViT SigLIP
//! vision tower. Each is a graph of the primitives above plus its blob loader, a
//! device-free host reference and a resident (prepacked) context.

use half::f16;

pub mod activation;
pub mod conv;
pub mod conv_transpose;
pub mod encoder;
pub mod ffn;
pub mod norm;
pub mod normvision;
pub mod pool;
pub mod reduce;
pub mod resize;
pub mod sam;
pub mod siglip;
pub mod softmax;

pub use activation::{
    Activation, EwOp, LUT_ENTRIES, LutEpilogue, LutTable, SELU_ALPHA, SELU_LAMBDA, activation,
    activation_model, activation_ref, clip, clip_ref, div, div_ref, elu, elu_model_alpha, elu_ref,
    eval, ew_add, ew_add_ref, ew_max, ew_max_ref, ew_min, ew_min_ref, ew_mul, ew_mul_ref, ew_ref,
    ew_sub, ew_sub_ref, leaky_relu, leaky_relu_model, leaky_relu_ref, lut_apply,
    lut_epilogue_build, prelu, prelu_ref, selu, selu_model, selu_ref, table_domain,
};
pub use conv::{
    CONV_MIN_DATAIN_H, CONV_POOL_MAX, Conv2dDesc, ConvContext, ConvInt8Pool, ConvInt8Tile,
    ConvInt8Tiles, DW_INT8_G, balance_bands, conv_dw_group, conv_int8_tile_plan, conv_out_dim,
    conv1d, conv2d, conv2d_act, conv2d_act_code, conv2d_act_ctx, conv2d_ctx, conv2d_deconv,
    conv2d_deconv_fits, conv2d_dw_int8, conv2d_dw_int8_ctx, conv2d_dw_int8_model,
    conv2d_dw_int8_plan_code, conv2d_int8, conv2d_int8_ctx, conv2d_int8_mt, conv2d_int8_plan_code,
    conv2d_job_extent, conv2d_oh, conv2d_ow, conv2d_plan, conv2d_plan_code, conv2d_ref,
    conv2d_ref_int8, conv2d_trail_x, conv2d_trail_y, datain_h, dw_chunk_fits, dw_int8_chunk,
    feature_cube_index, feature_cube_index_i8, gather_output, gather_output_i8, gather_output_i32,
    requant_apply, requant_apply_zp, requant_params, requant_round_shift, scatter_feature,
    scatter_feature_i8, scatter_weights, scatter_weights_dw, scatter_weights_dw_i8,
    scatter_weights_i8, weight_cube_dw_index, weight_cube_index, weight_cube_index_i8,
};
pub use conv_transpose::{
    ConvTranspose2dDesc, ConvTransposeRoute, ROCKET_CONV_TRANSPOSE_DECONV,
    ROCKET_CONV_TRANSPOSE_LOWERED, conv_transpose2d, conv_transpose2d_ctx,
    conv_transpose2d_deconv_eligible, conv_transpose2d_oh, conv_transpose2d_ow,
    conv_transpose2d_plan, conv_transpose2d_plan_code, conv_transpose2d_ref,
    conv_transpose2d_route, conv_transpose2d_route_code, lower_desc,
};
pub use encoder::{EncoderParams, encoder_block, encoder_block_ref};
pub use ffn::{ffn, ffn_fused, ffn_ref, geglu, geglu_ref, mlp, mlp_fused, mlp_ref};
pub use norm::{
    layernorm, layernorm_ref, rmsnorm, rmsnorm_ref, scale_rows, scale_rows_ref, square_prescale_k,
};
pub use normvision::{
    batchnorm, batchnorm_ref, groupnorm, groupnorm_ref, instancenorm, instancenorm_ref, l2norm,
    l2norm_ref,
};
pub use pool::{
    POOL_ERR_AVG_K, POOL_ERR_CUBE, POOL_ERR_EXCLUDE_PAD, POOL_ERR_EXTENT, POOL_ERR_KERNEL,
    POOL_ERR_METHOD, POOL_ERR_PAD, PoolDesc, pool_fp16, pool_fp16_plan, pool_fp16_plan_code,
    pool_int8, pool_out_dim, pool_ref_fp16, pool_ref_int8, pool_ref_uint8, pool_uint8,
};
pub use reduce::{
    REDUCE_MAX_DIM, REDUCE_MAX_PASS, cumsum, cumsum_ref, feature_data, global_avgpool,
    global_avgpool_plan, global_avgpool_plan_code, global_avgpool_ref, global_maxpool,
    global_maxpool_ref, global_minpool, global_minpool_ref, recip_pair, reduce_factor_axis,
    reduce_feature, reduce_feature_ref,
};
pub use resize::{
    bilinear_ksize, build_kernel, upsample_bilinear, upsample_bilinear_plan, upsample_bilinear_ref,
    upsample_desc, upsample_nearest, upsample_nearest_plan, upsample_nearest_ref,
};
pub use sam::{
    SAM_MAGIC, SAM_MAX_LAYERS, SAM_VERSION, SamContext, SamGeometry, SamLayerWeights, SamModel,
    SamNeckWeights, sam_encode, sam_encode_ctx, sam_encode_ref, sam_load,
};
pub use siglip::{
    SIGLIP_MAGIC, SIGLIP_MAX_TAPS, SIGLIP_VERSION, SiglipAct, SiglipContext, SiglipGeometry,
    SiglipLayerWeights, SiglipModel, SiglipVariant, siglip_encode, siglip_encode_ctx,
    siglip_encode_ref, siglip_load,
};
pub use softmax::{
    cross_entropy, cross_entropy_ref, logsoftmax, logsoftmax_ref, softmax, softmax_ref,
};

/// A correctly rounded (round-to-nearest, ties-to-even) `f64 -> f16` conversion.
///
/// The C library's reference paths all narrow a `double` with a plain `(_Float16)` cast,
/// which on aarch64 compiles to the single-rounding `fcvt h, d`. `half`'s
/// [`f16::from_f64`] does **not** guarantee that: with F16C it goes through `f32`
/// (`f as f32` then `f32 -> f16`), so a value just past an fp16 tie rounds *down* by one
/// ULP, and the crate's own `aarch64` path uses the hardware conversion. A reference
/// that must be bit-identical to the C on every host therefore cannot use it.
///
/// This is the same function everywhere: decompose the f64, keep 11 significand bits
/// with a sticky-based ties-to-even round, and re-encode as a normal / subnormal fp16.
pub(crate) fn f16_from_f64(v: f64) -> f16 {
    let bits = v.to_bits();
    let sgn = ((bits >> 48) & 0x8000) as u16;
    let exp = ((bits >> 52) & 0x7ff) as i32;
    let frac = bits & ((1u64 << 52) - 1);
    if exp == 0x7ff {
        // Infinities stay infinities; a NaN becomes a quiet NaN (the payload is never
        // observable in this crate's arithmetic).
        return f16::from_bits(if frac == 0 {
            sgn | 0x7c00
        } else {
            sgn | 0x7e00
        });
    }
    if exp == 0 {
        // An f64 subnormal is below 2^-1022, i.e. far under the smallest fp16 subnormal.
        return f16::from_bits(sgn);
    }
    let e = exp - 1023;
    if e > 15 {
        // |v| >= 2^16 > the 65520 overflow threshold.
        return f16::from_bits(sgn | 0x7c00);
    }
    let sig = (1u64 << 52) | frac;
    // A normal fp16 keeps 11 significand bits (implicit + 10 stored); below 2^-14 the
    // significand shifts into the subnormal range (fp16 min subnormal 2^-24).
    let shift = (-14 - e).max(0) as u32;
    let drop = 42 + shift;
    if drop >= 64 {
        return f16::from_bits(sgn);
    }
    let keep = sig >> drop;
    let rest = sig & ((1u64 << drop) - 1);
    let half = 1u64 << (drop - 1);
    let mut k = keep;
    if rest > half || (rest == half && (keep & 1) == 1) {
        k += 1;
    }
    if e >= -14 {
        let mut et = e;
        if k == 2048 {
            k = 1024;
            et += 1;
        }
        if et > 15 {
            return f16::from_bits(sgn | 0x7c00);
        }
        return f16::from_bits(sgn | (((et + 15) as u16) << 10) | (k as u16 & 0x3ff));
    }
    // Subnormal — or the smallest normal when `k` rounded up to 1024, whose encoding is
    // exactly that bit pattern.
    f16::from_bits(sgn | k as u16)
}

#[cfg(test)]
mod f16_convert_tests {
    use super::f16_from_f64;
    use half::f16;

    #[test]
    fn every_fp16_value_round_trips() {
        for u in 0..=0xffffu16 {
            let v = f16::from_bits(u);
            if v.is_nan() {
                continue;
            }
            assert_eq!(f16_from_f64(v.to_f64()), v, "bits {u:#06x}");
        }
    }

    #[test]
    fn it_rounds_like_a_single_rounding_cast() {
        // The case that exposed the f32 detour: just past an fp16 tie at 2.0, so the
        // correctly rounded result is the next value up.
        let v = -2.012_695_386_074f64;
        assert_eq!(f16_from_f64(v), f16::from_f32(-2.013_671_9));
        // Exactly on the tie: ties-to-even keeps the even significand.
        assert_eq!(
            f16_from_f64(-2.012_695_312_5f64),
            f16::from_f32(-2.011_718_8)
        );
        // The normal/subnormal boundary and the overflow threshold.
        assert_eq!(f16_from_f64(2f64.powi(-24)), f16::from_bits(0x0001));
        assert_eq!(f16_from_f64(2f64.powi(-25)), f16::ZERO);
        assert_eq!(f16_from_f64(2f64.powi(-14)), f16::from_bits(0x0400));
        assert_eq!(f16_from_f64(65504.0), f16::MAX);
        assert_eq!(f16_from_f64(65519.0), f16::MAX);
        assert_eq!(f16_from_f64(65520.0), f16::INFINITY);
        assert_eq!(f16_from_f64(f64::INFINITY), f16::INFINITY);
        assert_eq!(f16_from_f64(-0.0).to_bits(), 0x8000);
        // ... and it agrees with the hardware path on plain values.
        for x in [0.5f64, 1.0, -1.5, 1e-5, -3e4, 3.0e-8, 12345.678] {
            assert_eq!(f16_from_f64(x), f16::from_f64(x), "x = {x}");
        }
    }
}
