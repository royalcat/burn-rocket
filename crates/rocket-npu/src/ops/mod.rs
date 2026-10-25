//! The elementwise / reduction operation families.
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
//! [`norm`] holds the per-row broadcast scale (softmax's tail). The rest of
//! `rocket_norm.h` lands in M7.

pub mod activation;
pub mod norm;
pub mod reduce;
pub mod softmax;

pub use activation::{
    Activation, EwOp, LUT_ENTRIES, LutEpilogue, LutTable, SELU_ALPHA, SELU_LAMBDA, activation,
    activation_model, activation_ref, clip, clip_ref, div, div_ref, elu, elu_model_alpha, elu_ref,
    eval, ew_add, ew_add_ref, ew_max, ew_max_ref, ew_min, ew_min_ref, ew_mul, ew_mul_ref, ew_ref,
    ew_sub, ew_sub_ref, leaky_relu, leaky_relu_model, leaky_relu_ref, lut_apply,
    lut_epilogue_build, prelu, prelu_ref, selu, selu_model, selu_ref, table_domain,
};
pub use norm::{scale_rows, scale_rows_ref};
pub use reduce::{
    REDUCE_MAX_DIM, REDUCE_MAX_PASS, cumsum, cumsum_ref, feature_data, global_avgpool,
    global_avgpool_plan, global_avgpool_plan_code, global_avgpool_ref, global_maxpool,
    global_maxpool_ref, global_minpool, global_minpool_ref, recip_pair, reduce_factor_axis,
    reduce_feature, reduce_feature_ref,
};
pub use softmax::{
    cross_entropy, cross_entropy_ref, logsoftmax, logsoftmax_ref, softmax, softmax_ref,
};
