//! Query-chunked causal attention for the NPU path.
//!
//! The dense causal op ([`burn_rocket::attention`] with `is_causal = true`, and
//! its sliding-window sibling [`burn_rocket::attention_causal_window`]) builds
//! the additive mask host-side at the model's full `[n, n]` size, so a 30k-token
//! request needs ~1.8 GiB of f16 mask alone (plus the driver's score matrices)
//! and OOMs the board. This module walks the queries in chunks and calls
//! [`burn_rocket::attention_causal_block`] (or its windowed form) per chunk with
//! only the key prefix that chunk can see:
//!
//! * pure causal: keys `[0, q1)` (the whole prefix up to the chunk's last query);
//! * sliding window `w`: keys `[q0 - (w - 1), q1)`, i.e. the band the chunk's
//!   first query can still reach (the same convention as the CPU
//!   `causal_attention` path and [`burn_rocket::attention_causal_window`]).
//!
//! The per-chunk mask/score matrix is then `[C, q1]` (at most `[C, C + w - 1]`
//! for a sliding layer) instead of `[n, n]`: at 30k tokens and the default
//! 1024-wide chunk that is 60 MB instead of 1.8 GiB.
//!
//! Below the threshold the original dense call is used unchanged, so short
//! requests stay bit-identical to a build without this module.

#![cfg_attr(not(all(feature = "npu", target_arch = "aarch64")), allow(dead_code))]

#[cfg(all(feature = "npu", target_arch = "aarch64"))]
use burn::prelude::*;

/// Default query-chunk width (`ROCKET_CAUSAL_CHUNK`).
pub const DEFAULT_CAUSAL_CHUNK: usize = 1024;

/// Query-chunk width for the NPU causal path: `ROCKET_CAUSAL_CHUNK` (default
/// [`DEFAULT_CAUSAL_CHUNK`]; `0` keeps the dense masked path, `1` is legal but
/// pointless). Read once per process.
pub fn causal_chunk() -> usize {
    static OVERRIDE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *OVERRIDE.get_or_init(|| {
        std::env::var("ROCKET_CAUSAL_CHUNK")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_CAUSAL_CHUNK)
    })
}

/// One query chunk of a causal attention run.
///
/// `q`/`k` are absolute token positions (the caller's query 0 sits at
/// `q_offset`, keys are `[0, kn)`); `q0..q1` / `k0..k1` index the tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CausalChunk {
    /// Query rows `q0..q1` (local indices).
    pub q0: usize,
    pub q1: usize,
    /// Keys `k0..k1` (local indices).
    pub k0: usize,
    pub k1: usize,
}

impl CausalChunk {
    /// Absolute position of the first query row.
    pub fn q_pos(&self, q_offset: usize) -> usize {
        q_offset + self.q0
    }
}

/// Plan a causal attention run over `s` queries and `kn` keys.
///
/// `chunk == 0` (or a run that fits one chunk, or a dense-compatible run) yields
/// a single chunk covering everything; otherwise each chunk is
/// `(q0..q1, k0..q1)` with `k0 = q0 - (window - 1)` for a sliding layer (clamped
/// at 0) and `k1 = min(q1, kn)` (queries are causal, so no later key is ever
/// visible). The tail chunk is ragged: `q1 = s`.
pub fn causal_chunk_plan(
    s: usize,
    kn: usize,
    q_offset: usize,
    window: Option<usize>,
    chunk: usize,
) -> Vec<CausalChunk> {
    assert!(s > 0, "causal attention needs at least one query");
    assert!(kn > 0, "causal attention needs at least one key");
    let whole = CausalChunk {
        q0: 0,
        q1: s,
        k0: window.map_or(0, |w| q_offset.saturating_sub(w.saturating_sub(1))),
        k1: kn,
    };
    if chunk == 0 || s <= chunk {
        return vec![whole];
    }
    let mut plan = Vec::with_capacity(s.div_ceil(chunk));
    let mut q0 = 0;
    while q0 < s {
        let q1 = (q0 + chunk).min(s);
        let p0 = q_offset + q0;
        let p1 = q_offset + q1 - 1;
        let k0 = window.map_or(0, |w| p0.saturating_sub(w.saturating_sub(1)));
        let k1 = (p1 + 1).min(kn);
        plan.push(CausalChunk { q0, q1, k0, k1 });
        q0 = q1;
    }
    plan
}

/// Causal (optionally sliding-window) attention over `[1, s, H*D]` queries and
/// `[1, kn, KV*D]` keys/values, chunked over queries when `chunk != 0` and
/// `s > chunk`. `q_offset` is the absolute position of query row 0 (0 for a
/// prefill from the start); keys are the prefix `[0, kn)`.
///
/// A single-chunk run with `q_offset == 0 && kn == s` takes the dense op
/// (`attention_causal_window`, i.e. the same `MaskMode` the model used before
/// this helper existed), so short requests are bit-identical; everything else
/// goes through [`burn_rocket::attention_causal_block`] with a per-chunk mask.
#[cfg(all(feature = "npu", target_arch = "aarch64"))]
#[allow(clippy::too_many_arguments)]
pub fn causal_attention(
    q: Tensor<3>,
    k: Tensor<3>,
    v: Tensor<3>,
    n_head: usize,
    n_kv_heads: usize,
    head_dim: usize,
    scale: f64,
    softcap: Option<f32>,
    window: Option<usize>,
    q_offset: usize,
    chunk: usize,
) -> Tensor<3> {
    use burn::tensor::s;

    let [_, s, _] = q.dims();
    let kn = k.dims()[1];
    let plan = causal_chunk_plan(s, kn, q_offset, window, chunk);
    if plan.len() == 1 && q_offset == 0 && kn == s {
        let win = window.map_or(-1, |w| w as i64);
        return burn_rocket::attention_causal_window(
            q, k, v, n_head, n_kv_heads, head_dim, scale, softcap, win,
        );
    }
    let win = window.map_or(-1, |w| w as i64);
    let mut outs: Vec<Tensor<3>> = Vec::with_capacity(plan.len());
    for c in plan {
        let qc = q.clone().slice(s![.., c.q0..c.q1, ..]);
        let kc = k.clone().slice(s![.., c.k0..c.k1, ..]);
        let vc = v.clone().slice(s![.., c.k0..c.k1, ..]);
        outs.push(burn_rocket::attention_causal_block_window(
            qc,
            kc,
            vc,
            n_head,
            n_kv_heads,
            head_dim,
            scale,
            softcap,
            win,
            c.q_pos(q_offset) as i64,
            c.k0 as i64,
        ));
    }
    if outs.len() == 1 {
        return outs.pop().unwrap();
    }
    Tensor::cat(outs, 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_chunk_zero_is_the_dense_run() {
        let p = causal_chunk_plan(5000, 5000, 0, None, 0);
        assert_eq!(
            p,
            vec![CausalChunk {
                q0: 0,
                q1: 5000,
                k0: 0,
                k1: 5000
            }]
        );
        // Short sequences are one chunk even with a chunk width.
        let p = causal_chunk_plan(64, 64, 0, None, 1024);
        assert_eq!(p.len(), 1);
        assert_eq!((p[0].q0, p[0].q1, p[0].k0, p[0].k1), (0, 64, 0, 64));
    }

    #[test]
    fn plan_pure_causal_prefixes() {
        let p = causal_chunk_plan(2500, 2500, 0, None, 1000);
        assert_eq!(
            p,
            vec![
                CausalChunk {
                    q0: 0,
                    q1: 1000,
                    k0: 0,
                    k1: 1000
                },
                CausalChunk {
                    q0: 1000,
                    q1: 2000,
                    k0: 0,
                    k1: 2000
                },
                CausalChunk {
                    q0: 2000,
                    q1: 2500,
                    k0: 0,
                    k1: 2500
                },
            ]
        );
        // Every chunk covers each query exactly once and never asks for a key
        // above the chunk's last query.
        for c in &p {
            assert_eq!(c.k0, 0);
            assert!(c.k1 <= c.q1);
            assert_eq!(c.k1, c.q1);
        }
    }

    #[test]
    fn plan_sliding_window_bands() {
        let (w, chunk) = (256usize, 1024usize);
        let p = causal_chunk_plan(3000, 3000, 0, Some(w), chunk);
        // First chunk starts the band at 0 (clamped); later chunks open the band
        // exactly `w - 1` below the first query.
        assert_eq!(p[0].k0, 0);
        for c in &p[1..] {
            assert_eq!(c.k0, c.q0.saturating_sub(w - 1), "chunk {:?}", c);
            assert_eq!(c.k1, c.q1);
            let keys = c.k1 - c.k0;
            assert!(keys <= chunk + w - 1, "{:?}", c);
        }
        // The last chunk is ragged and its band still reaches back `w - 1`.
        let last = p.last().unwrap();
        assert_eq!(last.q1, 3000);
        assert_eq!(last.k0, last.q0 - (w - 1));
    }

    #[test]
    fn plan_respects_a_shorter_key_axis() {
        // A prefill appended to a shorter cache: `kn` bounds every chunk's tail.
        let p = causal_chunk_plan(1500, 2000, 0, None, 512);
        for c in &p {
            assert_eq!(c.k1, c.q1.min(2000));
        }
        // Queries shifted to an offset (decode-adjacent call): keys stop at the
        // last query and a window looks back from the chunk's first query.
        let p = causal_chunk_plan(600, 4000, 3000, Some(128), 256);
        assert!(p.iter().all(|c| c.k1 == 3000 + c.q1));
        assert_eq!(p[0].k0, 3000 - 127);
        assert_eq!(p[1].k0, 3000 + 256 - 127);
    }

    #[test]
    fn plan_ragged_tail_counts_every_query_once() {
        for s in [1usize, 7, 1024, 1025, 3001] {
            for chunk in [1usize, 3, 64, 1024, 4096] {
                let p = causal_chunk_plan(s, s, 0, None, chunk);
                let mut next = 0;
                for c in &p {
                    assert_eq!(c.q0, next, "s={s} chunk={chunk}");
                    assert!(c.q1 > c.q0);
                    next = c.q1;
                }
                assert_eq!(next, s, "s={s} chunk={chunk}");
            }
        }
    }
}
