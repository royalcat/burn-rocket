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
//! [`conv`] is the port of the RK3588 fp16 half of `rocket_conv.c`: the native
//! convolution engine (direct and depthwise, 1D, and the conv→activation fusion with
//! its tiling and resident-BO context), and [`conv_transpose`] is
//! `rocket_conv_transpose.c`'s transposed conv (the hardware deconvolution route and the
//! dilated-input lowering).

pub mod activation;
pub mod conv;
pub mod conv_transpose;
pub mod encoder;
pub mod ffn;
pub mod norm;
pub mod normvision;
pub mod reduce;
pub mod softmax;

pub use activation::{
    Activation, EwOp, LUT_ENTRIES, LutEpilogue, LutTable, SELU_ALPHA, SELU_LAMBDA, activation,
    activation_model, activation_ref, clip, clip_ref, div, div_ref, elu, elu_model_alpha, elu_ref,
    eval, ew_add, ew_add_ref, ew_max, ew_max_ref, ew_min, ew_min_ref, ew_mul, ew_mul_ref, ew_ref,
    ew_sub, ew_sub_ref, leaky_relu, leaky_relu_model, leaky_relu_ref, lut_apply,
    lut_epilogue_build, prelu, prelu_ref, selu, selu_model, selu_ref, table_domain,
};
pub use conv::{
    CONV_MIN_DATAIN_H, Conv2dDesc, ConvContext, balance_bands, conv_dw_group, conv_out_dim, conv1d,
    conv2d, conv2d_act, conv2d_act_code, conv2d_act_ctx, conv2d_ctx, conv2d_deconv,
    conv2d_deconv_fits, conv2d_job_extent, conv2d_oh, conv2d_ow, conv2d_plan, conv2d_plan_code,
    conv2d_ref, conv2d_trail_x, conv2d_trail_y, datain_h, dw_chunk_fits, feature_cube_index,
    gather_output, scatter_feature, scatter_weights, scatter_weights_dw, weight_cube_dw_index,
    weight_cube_index,
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
pub use reduce::{
    REDUCE_MAX_DIM, REDUCE_MAX_PASS, cumsum, cumsum_ref, feature_data, global_avgpool,
    global_avgpool_plan, global_avgpool_plan_code, global_avgpool_ref, global_maxpool,
    global_maxpool_ref, global_minpool, global_minpool_ref, recip_pair, reduce_factor_axis,
    reduce_feature, reduce_feature_ref,
};
pub use softmax::{
    cross_entropy, cross_entropy_ref, logsoftmax, logsoftmax_ref, softmax, softmax_ref,
};
