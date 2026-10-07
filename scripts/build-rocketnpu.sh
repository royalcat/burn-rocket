#!/usr/bin/env bash
# Build librocketnpu.a (gregordinary/rocket-userspace) on this host and install
# it into vendor/rocketnpu/ (or --out <dir>).
#
# The upstream source is pinned to a known-good commit; the install dir records
# what was built (COMMIT + ARCH) and the script is a no-op when it already
# matches the request. Nothing is copied from the board.
#
# Source/build cache: $OUT_DIR/rocket-userspace when invoked from a build
# script (build.rs), else ${CARGO_TARGET_DIR:-<repo>/target}/<profile>/build/
# burn-rocket (--profile, default release). Holds the clone plus one cmake tree
# per target/commit; removed by cargo clean.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_COMMIT=f86cf52c666b4eddadc17d80d6c558b4067c0d6b
UPSTREAM=https://github.com/gregordinary/rocket-userspace
HEADERS=(rocket_npu.h rocket_matmul.h)

TARGET=aarch64
COMMIT="$DEFAULT_COMMIT"
OUT="$REPO/vendor/rocketnpu"
SRC="${ROCKETNPU_SRC:-}"
JOBS="$(nproc)"
FORCE=0
CACHE_ARG=""
PROFILE=release

usage() {
    cat <<'EOF'
Build librocketnpu.a (gregordinary/rocket-userspace) on this host and install
it into vendor/rocketnpu/ (or --out <dir>).

Usage: scripts/build-rocketnpu.sh [options]

  --target aarch64|host   aarch64 = cross-build for the board (default)
                          host    = native build for link checks on this machine
  --commit <sha>          upstream commit to build (default: the pinned commit)
  --out <dir>             install dir (default: <repo>/vendor/rocketnpu)
  --cache <dir>           source/build cache (default: $OUT_DIR/rocket-userspace
                          from build.rs, else
                          <target-dir>/<profile>/build/burn-rocket)
  --profile <name>        profile in the manual cache default (default: release)
  --src <checkout>        use an existing rocket-userspace checkout (offline;
                          also read from $ROCKETNPU_SRC)
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
        --cache)   CACHE_ARG="${2:?--cache needs a value}"; shift 2 ;;
        --profile) PROFILE="${2:?--profile needs a value}"; shift 2 ;;
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

# --- cache location ---------------------------------------------------------
# From a build script the cache lives inside OUT_DIR (per target/profile/hash;
# `cargo clean` removes it). A manual run uses the cargo target dir, mirroring
# the layout of build-script outputs.
if [ -n "$CACHE_ARG" ]; then
    CACHE_BASE="$CACHE_ARG"
elif [ -n "${OUT_DIR:-}" ]; then
    CACHE_BASE="$OUT_DIR/rocket-userspace"
else
    CACHE_BASE="${CARGO_TARGET_DIR:-$REPO/target}/$PROFILE/build/burn-rocket"
fi

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
DRM_INCLUDEDIR="$(pkg-config --variable=includedir libdrm 2>/dev/null || true)"
[ -n "$DRM_INCLUDEDIR" ] || DRM_INCLUDEDIR=/usr/include
LIBDRM_INC="$DRM_INCLUDEDIR/libdrm"
[ -d "$LIBDRM_INC" ] || die "libdrm headers not found at $LIBDRM_INC"
command -v ar >/dev/null || die "ar not found (binutils)"
command -v file >/dev/null || die "file not found"
if [ "$TARGET" = aarch64 ]; then
    command -v aarch64-linux-gnu-gcc >/dev/null \
        || die "aarch64-linux-gnu-gcc not found (install the aarch64 GNU cross toolchain)"
fi

# --- source -----------------------------------------------------------------
mkdir -p "$CACHE_BASE"
if command -v flock >/dev/null; then
    # Serialize runs sharing the cache: two cargo invocations on one target
    # dir, or a manual run next to a build script.
    exec 9>"$CACHE_BASE/.lock"
    flock -w 600 9 || die "timed out waiting for the build lock in $CACHE_BASE"
else
    echo "warning: flock not found; runs sharing $CACHE_BASE are not serialized" >&2
fi
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

# The aarch64 sysroot carries the target uapi headers (linux/, asm/,
# drm/rocket_accel.h) but no libdrm. A plain pkg-config build adds the host's
# -I/usr/include first, where asm/posix_types.h takes its non-x86 branch and
# makes __kernel_size_t 32-bit: struct drm_version becomes 56 bytes instead of
# 64, DRM_IOCTL_VERSION encodes the wrong size, the kernel never fills the name
# and rocket_open fails with ENODEV on the board. Build against a shim that
# exposes only libdrm/ and let <linux/*>, <asm/*> and <drm/rocket_accel.h>
# resolve from the cross sysroot. (The native host build needs no shim: its
# compiler defines __x86_64__, so the host uapi headers are consistent.)
if [ "$TARGET" = aarch64 ]; then
    SHIM="$BUILD/shim"
    DRM_VERSION="$(pkg-config --modversion libdrm)"
    mkdir -p "$SHIM/include" "$SHIM/pkgconfig"
    ln -sfn "$LIBDRM_INC" "$SHIM/include/libdrm"
    cat > "$SHIM/pkgconfig/libdrm.pc" <<EOF
prefix=/usr
includedir=$SHIM/include
Name: libdrm
Description: userspace interface to kernel DRM services (burn-rocket cross shim)
Version: $DRM_VERSION
Libs: -ldrm
Cflags: -I\${includedir}
EOF
    export PKG_CONFIG_LIBDIR="$SHIM/pkgconfig"
fi

# Configure on every run: the flags depend on the shim and on pkg-config, and
# an existing tree must pick them up after a script change. -U drops the cached
# pkg-config results (pkg_check_modules would otherwise reuse them on a
# reconfigure of an existing build dir).
echo "==> configuring ($TARGET) in $BUILD"
cmake_args=(-S "$SRC_ABS" -B "$BUILD" -DROCKETNPU_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
            -U 'DRM_*')
if [ "$TARGET" = aarch64 ]; then
    cmake_args+=(-DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64
                -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc)
fi
cmake "${cmake_args[@]}"
echo "==> building (jobs=$JOBS)"
cmake --build "$BUILD" -j "$JOBS"

LIB="$BUILD/librocketnpu.a"
[ -s "$LIB" ] || die "build did not produce $LIB"

# --- ABI guard --------------------------------------------------------------
# The DRM ioctl ABI must match the board kernel; the guard catches a header
# leak (see the shim comment above) before anything is installed.
printf '#include <libdrm/drm.h>\n_Static_assert(sizeof(struct drm_version) == 64, "drm_version ABI");\n' > "$BUILD/abi_guard.c"
if [ "$TARGET" = aarch64 ]; then
    aarch64-linux-gnu-gcc -I"$SHIM/include" -c "$BUILD/abi_guard.c" -o /dev/null \
        || die "DRM ABI guard failed: struct drm_version is not 64 bytes with the cross headers"
else
    cc -c "$BUILD/abi_guard.c" -o /dev/null \
        || die "DRM ABI guard failed: struct drm_version is not 64 bytes with the host headers"
fi

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
