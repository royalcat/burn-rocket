//! EmbeddingGemma 2: multimodal (text/image/video/audio) embeddings, the
//! OpenAI-compatible `/v1/embeddings` + native `/embed` server, and its CLI.

pub mod cli;
pub mod load;
pub mod model;
pub mod server;
