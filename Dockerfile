# syntax=docker/dockerfile:1
#
# embeddings-fast — Qwen3-Embedding-0.6B OpenAI-compatible embedding server
# (Rust/Burn, flex backend) for the openviking stack on rock-5b-plus.
#
# aarch64 only: the `npu` feature links the aarch64-only static librocketnpu,
# and `serve --npu` offloads the projections to the RK3588 NPU through the
# mainline `rocket` driver (needs /dev/accel/accel0 at run time). Without
# `--npu` the same image serves the flex CPU path.
#
# Build context: this source tree plus vendor/rocketnpu/librocketnpu.a
# (aarch64 archive, gitignored — copy it from the board's
# /root/npu-poc/rocket-userspace/build, or build gregordinary/rocket-userspace).
# docker/build.sh assembles the context, builds on the board, and pushes.

FROM rust:1.98-trixie AS build
WORKDIR /src

COPY . .

# Fail early instead of at link time (build.rs only warns when the archive is
# missing, and the error surfaces much later).
RUN test -s vendor/rocketnpu/librocketnpu.a \
    || { echo "missing vendor/rocketnpu/librocketnpu.a in the build context" >&2; exit 1; }

# Cap parallelism: this builds on the 8-core RK3588 board.
ARG CARGO_BUILD_JOBS=4
RUN cargo build --release --locked \
        --no-default-features --features npu \
        --target aarch64-unknown-linux-gnu \
        --jobs "${CARGO_BUILD_JOBS}"

FROM debian:trixie-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY --from=build /src/target/aarch64-unknown-linux-gnu/release/embeddings-fast /usr/local/bin/embeddings-fast

EXPOSE 8383
ENTRYPOINT ["embeddings-fast"]
# Production defaults (openviking stack on rock-5b-plus): NPU projections with
# CPU flash attention (NPU attention is memory-hungry and slower on this board —
# deployment finding, 2026-10-05) and an 8192-token cap (OpenViking's client
# times out at 600 s). Mount the weights at /models/qwen3-embedding-0.6b and
# pass --device /dev/accel/accel0.
CMD ["serve", "--backend", "flex", "--dtype", "f32", "--npu", "--npu-attn", "cpu", "--port", "8383", \
     "--max-tokens", "8192", "--model-dir", "/models/qwen3-embedding-0.6b", \
     "--model-name", "qwen3-embedding"]
