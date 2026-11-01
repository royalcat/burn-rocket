//! vllm-sr Decision 3.0 (`d3-code-readout-v1`): typed System One decisions.
//!
//! A d3 checkpoint is a Qwen3.5 text backbone (`language_model.*`) plus a
//! 255-code readout head (`readout.safetensors`, `[255, hidden]`). One question
//! is decided per forward pass: the prompt lists every option under a token code
//! (`A`, `B`, ... `JT`), the final-normed hidden at the last prompt token goes
//! through the readout, and the softmax over the valid codes gives a probability
//! for every option. The prompt's full-attention layers are bidirectional
//! (`noncausal_full_attention`); the Gated DeltaNet layers stay causal.
//!
//! The format contract lives in [`format`] and is a byte-faithful port of the
//! checkpoint's `d3_format.py`; the checkpoint metadata and readout head are in
//! [`config`]. `tools/decision.md` and `tools/decision_parity.py` are the
//! reference and the parity gate.

pub mod config;
pub mod format;
pub mod load;
pub mod model;
pub mod request;

pub use load::load_decision_model;
pub use model::DecisionModel;
pub use request::DecisionRequest;
