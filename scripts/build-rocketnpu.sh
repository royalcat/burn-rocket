#!/usr/bin/env bash
# Build librocketnpu.a (gregordinary/rocket-userspace) on this host and install
# it into vendor/rocketnpu/ (or --out <dir>).
#
# The upstream source is pinned to a known-good commit; the install dir records
# what was built (COMMIT + ARCH) and the script is a no-op when it already
# matches the request. Nothing is copied from the board.
#
# Source cache: ${XDG_CACHE_HOME:-$HOME/.cache}/rocket-userspace (clone + one
# cmake tree per target/commit); it survives cargo clean.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_COMMIT=f86cf52c666b4eddadc17d80d6c558b4067c0d6b
UPSTREAM=https://github.com/gregordinary/rocket-userspace
CACHE_BASE="${XDG_CACHE_HOME:-$HOME/.cache}/rocket-userspace"
HEADERS=(rocket_npu.h rocket_matmul.h)

TARGET=aarch64
COMMIT="$DEFAULT_COMMIT"
OUT="$REPO/vendor/rocketnpu"
SRC=""
JOBS="$(nproc)"
FORCE=0

usage() {
    cat <<'EOF'
Build librocketnpu.a (gregordinary/rocket-userspace) on this host and install
it into vendor/rocketnpu/ (or --out <dir>).

Usage: scripts/build-rocketnpu.sh [options]

  --target aarch64|host   aarch64 = cross-build for the board (default)
                          host    = native build for link checks on this machine
  --commit <sha>          upstream commit to build (default: the pinned commit)
  --out <dir>             install dir (default: <repo>/vendor/rocketnpu)
  --src <checkout>        use an existing rocket-userspace checkout (offline)
  --jobs <n>              build parallelism (default: nproc)
  --force                 rebuild even when --out already matches
  -h, --help              show this help

The install dir gets librocketnpu.a, librocketgraph.a, rocket_npu.h,
rocket_matmul.h and the COMMIT/ARCH provenance pair; when those already match
the requested commit and target the script only prints a summary.
EOF
}

die() { echo "error: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --target)  TARGET="${2:?--target needs a value}"; shift 2 ;;
        --commit)  COMMIT="${2:?--commit needs a value}"; shift 2 ;;
        --out)     OUT="${2:?--out needs a value}"; shift 2 ;;
        --src)     SRC="${2:?--src needs a value}"; shift 2 ;;
        --jobs)    JOBS="${2:?--jobs needs a value}"; shift 2 ;;
        --force)   FORCE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1 (try --help)" ;;
    esac
done

case "$TARGET" in
    aarch64) ARCH=aarch64 ;;
    host)
        case "$(uname -m)" in
            x86_64)        ARCH=x86_64 ;;
            aarch64|arm64) ARCH=aarch64 ;;
            armv7l|armv6l) ARCH=arm ;;
            riscv64)       ARCH=riscv64 ;;
            *)             ARCH="$(uname -m)" ;;
        esac
        ;;
    *) die "unknown --target: $TARGET (want aarch64 or host)" ;;
esac

# The install dir records what was built; skip when it already matches. Short
# and full forms of the same sha count as equal (prefix in either direction).
stored_commit="$(cat "$OUT/COMMIT" 2>/dev/null || true)"
commit_matches=0
if [ -n "$stored_commit" ]; then
    case "$stored_commit" in "$COMMIT"*) commit_matches=1 ;; esac
    case "$COMMIT" in "$stored_commit"*) commit_matches=1 ;; esac
fi
if [ "$FORCE" -eq 0 ] \
    && [ -s "$OUT/librocketnpu.a" ] \
    && [ "$commit_matches" -eq 1 ] \
    && [ "$(cat "$OUT/ARCH" 2>/dev/null || true)" = "$ARCH" ]; then
    echo "==> up to date: $OUT already holds $ARCH @ ${COMMIT:0:12} (use --force to rebuild)"
    echo "    sha256 $(sha256sum "$OUT/librocketnpu.a" | cut -d' ' -f1)"
    exit 0
fi

command -v cmake >/dev/null || die "cmake not found"
command -v pkg-config >/dev/null || die "pkg-config not found"
pkg-config --exists libdrm \
    || die "libdrm development files not found (pkg-config --exists libdrm)"
command -v ar >/dev/null || die "ar not found (binutils)"
command -v file >/dev/null || die "file not found"
if [ "$TARGET" = aarch64 ]; then
    command -v aarch64-linux-gnu-gcc >/dev/null \
        || die "aarch64-linux-gnu-gcc not found (install the aarch64 GNU cross toolchain)"
fi

# --- source -----------------------------------------------------------------
if [ -n "$SRC" ]; then
    [ -d "$SRC" ] || die "--src directory not found: $SRC"
    SRC_ABS="$(cd "$SRC" && pwd)"
else
    SRC_ABS="$CACHE_BASE/src"
    if [ ! -d "$SRC_ABS/.git" ]; then
        echo "==> cloning $UPSTREAM -> $SRC_ABS"
        mkdir -p "$(dirname "$SRC_ABS")"
        git clone --quiet "$UPSTREAM" "$SRC_ABS" \
            || die "git clone failed (no network?); pass --src <existing checkout> for an offline build"
    fi
    if ! git -C "$SRC_ABS" cat-file -e "$COMMIT^{commit}" 2>/dev/null; then
        echo "==> fetching $UPSTREAM"
        git -C "$SRC_ABS" fetch --quiet origin \
            || die "git fetch failed (no network?) and $COMMIT is not in the cache; pass --src <existing checkout>"
    fi
fi
git -C "$SRC_ABS" cat-file -e "$COMMIT^{commit}" 2>/dev/null \
    || die "commit $COMMIT not found in $SRC_ABS"
git -C "$SRC_ABS" checkout --quiet "$COMMIT"
# Canonicalize (short sha / tag -> full sha) so provenance and build-dir keys
# are stable across input forms.
COMMIT="$(git -C "$SRC_ABS" rev-parse --verify "${COMMIT}^{commit}")"
[ "$(git -C "$SRC_ABS" rev-parse HEAD)" = "$COMMIT" ] \
    || die "checkout in $SRC_ABS did not land on ${COMMIT:0:12}"

# --- build ------------------------------------------------------------------
# The build tree is keyed by target, commit and source path: cmake caches the
# source directory, so `--src` must not reuse a tree configured for the clone.
SRC_KEY="$(printf '%s' "$SRC_ABS" | cksum | cut -d' ' -f1)"
BUILD="$CACHE_BASE/build-$TARGET-${COMMIT:0:12}-$SRC_KEY"
if [ ! -f "$BUILD/CMakeCache.txt" ]; then
    echo "==> configuring ($TARGET) in $BUILD"
    cmake_args=(-S "$SRC_ABS" -B "$BUILD" -DROCKETNPU_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release)
    if [ "$TARGET" = aarch64 ]; then
        cmake_args+=(-DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64
                    -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc)
    fi
    cmake "${cmake_args[@]}"
fi
echo "==> building (jobs=$JOBS)"
cmake --build "$BUILD" -j "$JOBS"

LIB="$BUILD/librocketnpu.a"
[ -s "$LIB" ] || die "build did not produce $LIB"

# --- sanity + install -------------------------------------------------------
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT
ar p "$LIB" "$(ar t "$LIB" | head -n1)" > "$tmp"
desc="$(file -b "$tmp")"
case "$ARCH" in
    aarch64) want=aarch64 ;;
    x86_64)  want=x86-64 ;;
    arm)     want=ARM ;;
    riscv64) want=RISC-V ;;
    *)       want="" ;;
esac
if [ -n "$want" ]; then
    case "$desc" in
        *"$want"*) ;;
        *) die "archive architecture mismatch: want $want, got: $desc" ;;
    esac
else
    echo "warning: cannot verify archive architecture for $ARCH" >&2
fi
echo "==> archive check: $desc"

mkdir -p "$OUT"
cp -f "$LIB" "$OUT/librocketnpu.a"
if [ -f "$BUILD/librocketgraph.a" ]; then
    cp -f "$BUILD/librocketgraph.a" "$OUT/librocketgraph.a"
fi
for h in "${HEADERS[@]}"; do
    if [ -f "$SRC_ABS/include/$h" ]; then
        cp -f "$SRC_ABS/include/$h" "$OUT/$h"
    fi
done
printf '%s\n' "$COMMIT" > "$OUT/COMMIT"
printf '%s\n' "$ARCH" > "$OUT/ARCH"

echo
echo "==> installed $ARCH @ ${COMMIT:0:12} in $OUT"
echo "    sha256 $(sha256sum "$OUT/librocketnpu.a" | cut -d' ' -f1)"
case "$TARGET" in
    aarch64) echo "    next: cargo build --release -p rocket-inference --no-default-features --features npu --target aarch64-unknown-linux-gnu" ;;
    host)    echo "    next: cargo build --release -p burn-rocket --features npu --example probe" ;;
esac
