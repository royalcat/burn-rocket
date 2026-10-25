//! The elementwise operation families.
//!
//! [`activation`] is the port of `rocket_activation.c`: the DPU LUT activation
//! family (the standalone flying-mode DPU job), the elementwise binary ops that ride
//! the conv-main EW datapath, and the composite entries on top of them (clip,
//! LeakyReLU/ELU/SELU/PReLU, divide), plus their host references and the LUT model.

pub mod activation;

pub use activation::{
    Activation, EwOp, LUT_ENTRIES, LutEpilogue, LutTable, SELU_ALPHA, SELU_LAMBDA, activation,
    activation_model, activation_ref, clip, clip_ref, div, div_ref, elu, elu_model_alpha, elu_ref,
    eval, ew_add, ew_add_ref, ew_max, ew_max_ref, ew_min, ew_min_ref, ew_mul, ew_mul_ref, ew_ref,
    ew_sub, ew_sub_ref, leaky_relu, leaky_relu_model, leaky_relu_ref, lut_apply,
    lut_epilogue_build, prelu, prelu_ref, selu, selu_model, selu_ref, table_domain,
};
