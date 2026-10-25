#!/usr/bin/env bash
# Build the host-side C oracle (planners + regcmd emitters for every matmul dtype).
#
#   tools/c-oracle/build.sh <rocket-userspace-src> <host-install-dir> [out]
#
# <host-install-dir> is a host build install (librocketnpu.a + headers), e.g.
# `scripts/build-rocketnpu.sh --target host --out DIR` from the historical tree.
# The source checkout supplies the internal headers (npu_matmul.h, npu_hw.h, ...).
set -euo pipefail

SRC="${1:?rocket-userspace source checkout}"
INST="${2:?host install dir with librocketnpu.a}"
OUT="${3:-/tmp/opencode/c-oracle}"
CC="${CC:-cc}"

mkdir -p "$OUT"
"$CC" -O2 -o "$OUT/oracle" \
    "$(dirname "$0")/oracle.c" \
    -I"$SRC/include" -I"$SRC/src" \
    "$INST/librocketnpu.a" -lpthread -lm -ldl
echo "built $OUT/oracle"
