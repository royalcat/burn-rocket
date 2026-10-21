#!/usr/bin/env bash
# Build the aarch64 container image for rocket-inference and push it to the
# Forgejo registry (git.kmsign.org/royalcat/rocket-inference).
#
# The image is aarch64-only (the NPU driver is aarch64), so it is built natively
# on the rock-5b-plus board, which has the NPU. The control host performs the
# push because it holds the registry credentials.
#
# Usage:
#   examples/rocket-inference/docker/build.sh                 # HEAD of this checkout
#   examples/rocket-inference/docker/build.sh <git-ref>       # any commit/tag in this checkout
#   COMMIT=<sha> TAG=<tag> examples/rocket-inference/docker/build.sh
#
# Env overrides:
#   BOARD=root@rock-5b-plus.lan      ssh target for the build host
#   REMOTE_DIR=/root/rocket-inference-image  staging dir on the board
#   CARGO_BUILD_JOBS=4               cargo parallelism inside the image build
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"

COMMIT="${COMMIT:-${1:-HEAD}}"
SHA="$(git -C "$REPO" rev-parse --short "$COMMIT")"
TAG="${TAG:-$SHA}"
IMAGE="git.kmsign.org/royalcat/rocket-inference:${TAG}"
BOARD="${BOARD:-root@rock-5b-plus.lan}"
STAGE="${STAGE:-/tmp/rocket-inference-build-$TAG}"
REMOTE_DIR="${REMOTE_DIR:-/root/rocket-inference-image}"
JOBS="${CARGO_BUILD_JOBS:-4}"

echo "==> staging source ${SHA} (${TAG}) -> ${STAGE}"
rm -rf "$STAGE"
mkdir -p "$STAGE"
git -C "$REPO" archive "$COMMIT" | tar -C "$STAGE" -xf -
# New refs keep the Dockerfile under examples/; stage it at the context root so
# plain `docker build .` works. Old refs (root Dockerfile) pass through.
[ -f "$STAGE/Dockerfile" ] || cp "$REPO/examples/rocket-inference/Dockerfile" "$STAGE/Dockerfile"
[ -f "$STAGE/.dockerignore" ] || cp "$REPO/.dockerignore" "$STAGE/.dockerignore"

echo "==> shipping context -> ${BOARD}:${REMOTE_DIR}"
ssh "$BOARD" "rm -rf '$REMOTE_DIR' && mkdir -p '$REMOTE_DIR'"
tar -C "$STAGE" -cf - . | ssh "$BOARD" "tar -C '$REMOTE_DIR' -xf -"

echo "==> building on the board (CARGO_BUILD_JOBS=${JOBS})"
ssh "$BOARD" "cd '$REMOTE_DIR' && docker build --progress=plain --build-arg CARGO_BUILD_JOBS=$JOBS -t '$IMAGE' ."

echo "==> transferring image to the control host and pushing"
ssh "$BOARD" "docker save '$IMAGE'" | docker load
docker push "$IMAGE"

echo "==> done: $IMAGE"
