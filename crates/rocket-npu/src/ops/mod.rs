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
pub use softmax::{
    cross_entropy, cross_entropy_ref, logsoftmax, logsoftmax_ref, softmax, softmax_ref,
};
