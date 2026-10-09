# syntax=docker/dockerfile:1
#
# rocket-inference — the merged inference app (Qwen3-Embedding-0.6B, Qwen3.5-0.8B
# intent, EmbeddingGemma 2, Gemma 4 E2B-it; Rust/Burn, flex backend) for the
# openviking stack on rock-5b-plus. Built on the burn-rocket library at the repo
# root; see examples/rocket-inference. The default command is the Qwen3
# embedding server.
#
# One Dockerfile, two architectures; the build platform picks the branch:
#   arm64/aarch64 — the `npu` feature links the aarch64-only static librocketnpu
#                   and `serve` offloads to the RK3588 NPU through the mainline
#                   `rocket` driver (needs /dev/accel/accel0 at run time).
#                   Requires vendor/rocketnpu/librocketnpu.a in the build
#                   context (scripts/build-rocketnpu.sh stages it).
#   amd64/x86_64  — flex CPU build for x86-64-v3 (AVX2), no NPU support.
#
# The build context is the repo root:
#   docker build -f examples/rocket-inference/Dockerfile .
# examples/rocket-inference/docker/build.sh assembles the same context (with the
# archive) for the board. The default command is identical on both
# architectures: `serve` picks the NPU on aarch64 builds that have it (pass
# `--no-npu` for the flex CPU path) and stays on flex elsewhere.

FROM rust:1.98-trixie AS build
WORKDIR /src

COPY . .

# Cap parallelism: this also builds on the 8-core RK3588 board.
ARG CARGO_BUILD_JOBS=4
# Docker sets TARGETARCH for the build platform (arm64 on the board and other
# aarch64 hosts, amd64 elsewhere); fall back to the machine's own architecture
# for builders that do not provide it.
ARG TARGETARCH

RUN set -eux; \
    arch="${TARGETARCH:-$(uname -m)}"; \
    case "$arch" in \
        arm64|aarch64) \
            test -s vendor/rocketnpu/librocketnpu.a \
                || { echo "missing vendor/rocketnpu/librocketnpu.a in the build context (aarch64 builds link the NPU archive; run scripts/build-rocketnpu.sh)" >&2; exit 1; }; \
            cargo build --release --locked \
                -p rocket-inference \
                --no-default-features --features npu \
                --target aarch64-unknown-linux-gnu \
                --jobs "${CARGO_BUILD_JOBS}" \
            ;; \
        amd64|x86_64) \
            RUSTFLAGS="-C target-cpu=x86-64-v3" \
            cargo build --release --locked \
                -p rocket-inference \
                --no-default-features \
                --jobs "${CARGO_BUILD_JOBS}" \
            ;; \
        *) echo "unsupported build architecture: $arch" >&2; exit 1 ;; \
    esac; \
    bin="$(find target -path '*/release/rocket-inference' -type f -print -quit)"; \
    test -n "$bin"; \
    install -Dm755 "$bin" /src/out/rocket-inference

FROM debian:trixie-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY --from=build /src/out/rocket-inference /usr/local/bin/rocket-inference

EXPOSE 8383
ENTRYPOINT ["rocket-inference"]
# Production defaults (openviking stack on rock-5b-plus): NPU projections are
# `serve`'s default on aarch64 builds that have them (`--no-npu` forces the flex
# CPU path), with CPU flash attention (NPU attention is memory-hungry on this
# board — deployment finding, 2026-10-05) and an 8192-token cap (OpenViking's
# client times out at 600 s). Mount the weights at /models/qwen3-embedding-0.6b
# and pass --device /dev/accel/accel0. On amd64 the NPU flags are inert and the
# same command runs the flex CPU path.
CMD ["serve", "--backend", "flex", "--dtype", "f32", "--npu-attn", "cpu", "--port", "8383", \
     "--max-tokens", "8192", "--model-dir", "/models/qwen3-embedding-0.6b", \
     "--model-name", "qwen3-embedding"]
