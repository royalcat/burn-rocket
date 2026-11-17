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
//!
//! Media: [`vision`] holds the Qwen3.5 vision tower, the Qwen2-VL image processor
//! and the d3 video sampling/processor (2 frames per second, 4-32 frames, temporal
//! pairs), [`mrope`] the multimodal rotary positions of an image or video prompt,
//! and [`model`] splices the tower's features into the text token rows (a video in
//! one run per temporal pair, with the timestamps between the runs). The transport
//! decides how an `images`/`videos` entry is read ([`MediaMode`]): the CLI takes
//! local paths (parity fixtures), the server takes `data:image/...;base64` and
//! `data:video/...;base64` URLs only, like the checkpoint's `d3_server.py`.

pub mod config;
pub mod format;
pub mod load;
pub mod model;
pub mod mrope;
pub mod request;
pub mod vision;

pub use load::load_decision_model;
pub use model::DecisionModel;
pub use request::DecisionRequest;
