#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Dtype parity harness: the Rust rocket-npu planner/emitters vs the C oracle.

Every RK3588 matmul dtype gets a shape matrix with that dtype's alignments. For
each shape the C oracle and `regcmd_dump` are run with identical arguments and
their stdout is compared byte for byte:

  plan  -> "Mt Kt Nt jobs", or "-1"
  emit  -> task_count, then one %016llx word per NPUOP (at the planned tile dims)

A final `pool` section covers the spatial-reduce machinery: the decomposition
(`pool C H W`: the `rocket_global_avgpool_plan` reason code and the per-axis
`rocket_reduce_factor_axis` factors) over a shape matrix, the PPU pool generator
(`emit-pool`, a pure function of `pool_params_t`) over a shape/method/pad/stride
matrix, and the fp16 average-pool reciprocal (`recip`,
`ppu_recip_kernel_fp16`) for every kernel extent.

An `int8conv` section covers the native int8 CONV_2D generators over a shape /
zero-point / scale matrix: `emit-conv-int8` (the direct int32-raw program) and
`emit-conv-dw-int8` (the depthwise int8-out requant writer, and its int32-raw
diagnostic form), including the C's own refusals. int8/uint8 *pooling* has no program
of its own — the C routes it through `gen_pool_fp16` with an int8<->fp16 cube
boundary — so those geometries appear as ordinary `emit-pool` rows.

A `conv` section covers the fp16 CONV_2D family: `conv` (the `rocket_conv2d_plan`
reason code) over the direct/depthwise/dilation/stride/explicit-extent matrix,
`emit-conv` (`gen_conv2d_fp16` / `gen_conv2d_dw_fp16`, including the CNA
deconvolution strides and the refusals) and `emit-conv-act` (the fused-activation
program with its two 513-word LUT uploads).

Dtypes whose Rust planner/emitter is not ported yet report a visible PENDING
line instead of failing. Any mismatch exits nonzero and prints the first
differing line with both values.

Usage:
  python3 tools/c-oracle/parity.py
  python3 tools/c-oracle/parity.py --dump /path/to/regcmd_dump --verbose
  python3 tools/c-oracle/parity.py --kind fp16,int8
  python3 tools/c-oracle/parity.py --kind pool
  python3 tools/c-oracle/parity.py --kind conv
  python3 tools/c-oracle/parity.py --kind int8conv
"""

from __future__ import annotations

import argparse
import math
import os
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ORACLE = "/tmp/opencode/c-oracle/oracle"

# Fixed DMA addresses so the emitted programs are stable across runs.
IN_DMA = "0x1000"
WT_DMA = "0x2000"
OUT_DMA = "0x3000"
ACC_DMA = "0x4000"


@dataclass(frozen=True)
class Dtype:
    """One matmul dtype: its oracle/Rust subcommands and shape alignment."""

    name: str
    plan_mode: str
    emit_mode: str
    n_align: int
    k_align: int
    milestone: str = ""
    groups: tuple[int | None, ...] = (None,)
    # Extra emit flag sets (FP32TOFP16, ACCUMULATE, ADD_DMA); () means defaults.
    emit_variants: tuple[tuple[str, ...], ...] = ((),)


DTYPES: tuple[Dtype, ...] = (
    Dtype("fp16", "plan", "emit", n_align=16, k_align=32,
          emit_variants=((), ("0",), ("1", "1", ACC_DMA))),
    Dtype("int8", "plan-int8", "emit-int8", n_align=32, k_align=32),
    Dtype("int8-gw", "plan-int8-gw", "emit-int8", n_align=32, k_align=32,
          groups=(None, 32, 64, 128, 256, 512, 2048)),
    Dtype("int4", "plan-int4", "emit-int4", n_align=64, k_align=32),
    Dtype("tf32", "plan-tf32", "emit-tf32", n_align=16, k_align=16),
    Dtype("bf16", "plan-bf16", "emit-bf16", n_align=16, k_align=32),
)

# (M, K, N) before per-dtype alignment: tiny, small, medium, tile-boundary and
# large shapes. K > Kt, N > Nt and M > Mt all occur at the large end.
BASE_SHAPES: tuple[tuple[int, int, int], ...] = (
    (4, 32, 16),        # one tiny tile
    (8, 48, 32),        # K%16 but not K%32: exercises tf32's K alignment
    (8, 96, 64),        # small, non-power-of-two K
    (64, 256, 128),     # medium
    (128, 512, 256),    # medium, N at the tile cap
    (256, 1024, 512),   # M at the tile cap, N > Nt
    (260, 2080, 320),   # tails in every dimension
    (512, 2560, 4096),  # M, K and N all tile
    (1024, 8192, 2048), # very large
    (4, 16384, 16),     # deep K
    (512, 32, 16),      # K at the alignment floor
)

# Shapes that must be refused identically (plan prints -1 on both sides).
INVALID_SHAPES: tuple[tuple[int, int, int], ...] = (
    (6, 64, 16),   # M % 4
    (4, 48, 16),   # K % 32
    (4, 64, 8),    # N % 16
    (0, 64, 16),   # non-positive
)


# ── PPU pool matrix ───────────────────────────────────────────────────────────
#
# (name, c, ih, iw, oh, ow, kh, kw, sy, sx, pt, pl, pb, pr, method, expect_ok)
#
# The geometry follows the reduce decomposition: exact tiling (stride == kernel,
# no pad) for every pass of a global pool, plus the padded/overlapping shapes the
# generator must also encode. `expect_ok=False` pins an identical refusal.
POOL_CASES: tuple[tuple, ...] = (
    # Global-pool passes: 7x7 (one pass), 14x14 (one pass), 56x56 = 14 then 4,
    # 49x49 = 7 then 7, and the asymmetric 28x20 = (14x10) then (2x2).
    ("7x7",     8,  7,  7, 1,  1,  7,  7,  7,  7, 0, 0, 0, 0, "avg", True),
    ("14x14",  16, 14, 14, 1,  1, 14, 14, 14, 14, 0, 0, 0, 0, "avg", True),
    ("56x56/1", 3, 56, 56, 4,  4, 14, 14, 14, 14, 0, 0, 0, 0, "avg", True),
    ("56x56/2", 3,  4,  4, 1,  1,  4,  4,  4,  4, 0, 0, 0, 0, "avg", True),
    ("49x49/1", 4, 49, 49, 7,  7,  7,  7,  7,  7, 0, 0, 0, 0, "avg", True),
    ("49x49/2", 4,  7,  7, 1,  1,  7,  7,  7,  7, 0, 0, 0, 0, "avg", True),
    ("28x20/1", 16, 28, 20, 2, 2, 14, 10, 14, 10, 0, 0, 0, 0, "avg", True),
    ("28x20/2", 16,  2,  2, 1, 1,  2,  2,  2,  2, 0, 0, 0, 0, "avg", True),
    # Extrema: idempotent, so no reciprocal; padded MAX fills -inf, MIN fills 0.
    ("max pad", 8,  5,  5, 3,  3,  3,  3,  3,  3, 1, 1, 1, 1, "max", True),
    ("min pad", 8,  5,  5, 3,  3,  3,  3,  3,  3, 1, 1, 1, 1, "min", True),
    ("max",     4, 49, 49, 7,  7,  7,  7,  7,  7, 0, 0, 0, 0, "max", True),
    ("min",    64, 14, 14, 1,  1, 14, 14, 14, 14, 0, 0, 0, 0, "min", True),
    # A padded, overlapping average (stride 1, kernel 3, symmetric pad 1).
    ("avg pad", 8,  5,  5, 5,  5,  3,  3,  1,  1, 1, 1, 1, 1, "avg", True),
    # Last encodable cube channel (field = 8191) and a 13-bit width.
    ("field max", 8192, 4, 8192, 1, 1024, 4, 8, 4, 8, 0, 0, 0, 0, "avg", True),
    # Kernel 1: legal for the generator (avg would need an infinite reciprocal).
    ("k1 max",  2,  4,  4, 4,  4,  1,  1,  1,  1, 0, 0, 0, 0, "max", True),
    # Refusals: 4-bit kernel field, 13-bit channel field, zero extent.
    ("kw17",    8,  4,  4, 2,  2,  2, 17,  2,  2, 0, 0, 0, 0, "avg", False),
    ("c8193", 8193, 4,  4, 2,  2,  2,  2,  2,  2, 0, 0, 0, 0, "avg", False),
    ("iw0",     8,  4,  0, 2,  2,  2,  2,  2,  2, 0, 0, 0, 0, "avg", False),
    # The int8/uint8 pool geometries: those entries route through this same fp16
    # program with an int8<->fp16 cube boundary, so there is no second encoding to
    # compare — the 2x2 detection max and the padded 3x3 average are the shapes the
    # delegated networks use.
    ("i8 max",  32,  8,  8, 4,  4,  2,  2,  2,  2, 0, 0, 0, 0, "max", True),
    ("i8 avg",  64,  7,  7, 7,  7,  3,  3,  1,  1, 1, 1, 1, 1, "avg", True),
    ("u8 max",  16,  5,  5, 3,  3,  3,  3,  1,  1, 1, 1, 1, 1, "max", True),
    ("u8 avg",   8, 14, 14, 1,  1, 14, 14, 14, 14, 0, 0, 0, 0, "avg", True),
)

POOL_METHODS = {"avg": 0, "max": 1, "min": 2}

# The int8 conv harness's bias-cube IOVA (the feature/weight/output IOVAs are shared
# with the fp16 conv rows above; they land in the program words verbatim).
BIAS_DMA = "0x400000"

# ── Conv matrix ───────────────────────────────────────────────────────────────
#
# `IM2COL = 0` in every row: this family never materialises one. The 13 scalars after
# the mode are `IC IH IW OC KH KW SY SX PT PL DY DX DEPTHWISE`; the plan adds `OH OW`
# (0 = derive) and the emitters add `FP32TOFP16|KIND IN WT OUT [DSY DSX]`.
#
# (name, ic, ih, iw, oc, kh, kw, sy, sx, pt, pl, dy, dx, depthwise, oh, ow, expect_code)
# `expect_code` pins the C's reason class: 0 runnable, -1..-4 the negated refusals.
CONV_PLANS: tuple[tuple, ...] = (
    # Direct: the plain 3x3, the RGB stem (IC=3 padded to 32), an SSD OC=24 (padded to
    # 32 by the weight group), 1x1, stride 2, pad 2, dilation 2, and a 7x7 k3 pad 3.
    ("3x3 s1 p1",   32,   8,   8,  16, 3, 3, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0),
    ("rgb stem",     3,  56,  56,  64, 3, 3, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0),
    ("rgb s2",       3, 224, 224,  32, 3, 3, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0),
    ("oc24 head",   32,   8,   8,  24, 5, 5, 1, 1, 2, 2, 1, 1, 0, 0, 0, 0),
    ("1x1",         64,  14,  14, 128, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0),
    ("k3 s2 p1",   128,  28,  28, 128, 3, 3, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0),
    ("k5 dil2",     32,  16,  16,  64, 5, 5, 1, 1, 4, 4, 2, 2, 0, 0, 0, 0),
    ("k7 p3",       32,  16,  16,  32, 7, 7, 1, 1, 3, 3, 1, 1, 0, 0, 0, 0),
    ("conv1d k3",   80, 3000,   1, 384, 3, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0),
    ("conv1d s2",   80, 3000,   1, 384, 3, 1, 2, 1, 1, 0, 1, 1, 0, 0, 0, 0),
    # Depthwise (OC == IC, IC % 32 == 0).
    ("dw 3x3",      32,  56,  56,  32, 3, 3, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0),
    ("dw s2",       64,  56,  56,  64, 3, 3, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0),
    ("dw 1x1",      96,  14,  14,  96, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0),
    # An explicit extent: one that matches is accepted, one that does not is -1.
    ("explicit ok", 32,   8,   8,  16, 3, 3, 1, 1, 1, 1, 1, 1, 0, 8, 8, 0),
    ("explicit bad", 32,  8,   8,  16, 3, 3, 1, 1, 1, 1, 1, 1, 0, 9, 8, -1),
    # Refusals: depthwise OC != IC (-2), depthwise IC % G (-2), a kernel that cannot fit
    # one CBUF bank (-4), a collapsed extent (-3), and malformed fields (-1).
    ("dw oc!=ic",   32,   8,   8,  16, 3, 3, 1, 1, 1, 1, 1, 1, 1, 0, 0, -2),
    ("dw ic%32",    24,   8,   8,  24, 3, 3, 1, 1, 1, 1, 1, 1, 1, 0, 0, -2),
    ("dw k25",      32,  32,  32,  32, 25, 25, 1, 1, 0, 0, 1, 1, 1, 0, 0, -4),
    ("k3 ic6000", 6000,   4,   4,  16, 3, 3, 1, 1, 0, 0, 1, 1, 0, 0, 0, -4),
    ("k5 ih4",      32,   4,   4,  16, 5, 5, 1, 1, 0, 0, 1, 1, 0, 0, 0, -3),
    ("ic0",          0,   8,   8,  16, 3, 3, 1, 1, 1, 1, 1, 1, 0, 0, 0, -1),
    ("sy0",         32,   8,   8,  16, 3, 3, 0, 1, 1, 1, 1, 1, 0, 0, 0, -1),
    ("dy0",         32,   8,   8,  16, 3, 3, 1, 1, 1, 1, 0, 1, 0, 0, 0, -1),
)

# The emitters: the same geometries plus fp32-out, deconvolution and refusals.
# (name, ..., depthwise, fp32tofp16, dsy, dsx)
CONV_EMITS: tuple[tuple, ...] = (
    ("3x3 s1 p1",   32,   8,   8,  16, 3, 3, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1),
    ("3x3 fp32 out", 32,  8,   8,  16, 3, 3, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1),
    ("rgb stem",     3,  56,  56,  64, 3, 3, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1),
    ("oc24 head",   32,   8,   8,  24, 5, 5, 1, 1, 2, 2, 1, 1, 0, 1, 1, 1),
    ("1x1",         64,  14,  14, 128, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1),
    ("k3 s2 p1",   128,  28,  28, 128, 3, 3, 2, 2, 1, 1, 1, 1, 0, 1, 1, 1),
    ("k5 dil2",     32,  16,  16,  64, 5, 5, 1, 1, 4, 4, 2, 2, 0, 1, 1, 1),
    ("k7 p3",       32,  16,  16,  32, 7, 7, 1, 1, 3, 3, 1, 1, 0, 1, 1, 1),
    # A conv1d *tile*: the full 3000-step sequence is the tiler's job (the row bands),
    # so the single-job generator is asked for one band's worth of rows.
    ("conv1d tile", 80,  256,   1, 384, 3, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1),
    ("dw 3x3",      32,  56,  56,  32, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
    ("dw 1x1",      96,  14,  14,  96, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1),
    # The CNA deconvolution mode: the interior-dilation strides land in CONV_CON1/3.
    ("deconv 2x2",  32,  24,  24,  32, 3, 3, 1, 1, 4, 4, 1, 1, 0, 1, 2, 2),
    ("deconv 4x2",  32,  24,  24,  32, 3, 3, 1, 1, 4, 4, 1, 1, 0, 1, 4, 2),
    ("deconv 8x8",  32,   8,   8,  16, 3, 3, 1, 1, 5, 5, 1, 1, 0, 1, 8, 8),
    # Refusals: the feature alone cannot exceed 11 banks (-1), a kernel must fit a bank
    # (-2), and the deconvolution mode takes a direct stride-1 conv only (-1).
    ("feat banks",  32, 400, 400,  16, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1),
    ("k3 ic6000", 6000,   4,   4,  16, 3, 3, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1),
    ("deconv dw",   32,  24,  24,  32, 3, 3, 1, 1, 4, 4, 1, 1, 1, 1, 2, 2),
    ("deconv s2",   32,  24,  24,  32, 3, 3, 2, 2, 4, 4, 1, 1, 0, 1, 2, 2),
)

# The fused-activation programs: (name, ..., depthwise, kind). KIND is a smooth
# single-pass ROCKET_ACTIVATION_* value (3 SiLU, 4 tanh, 5 GELU); the generator accepts
# these on either path (the entry point, not the generator, is what refuses depthwise).
CONV_ACTS: tuple[tuple, ...] = (
    ("act silu",    32,   8,   8,  16, 3, 3, 1, 1, 1, 1, 1, 1, 0, 3),
    ("act tanh",    32,   8,   8,  16, 3, 3, 1, 1, 1, 1, 1, 1, 0, 4),
    ("act gelu",    32,   8,   8,  24, 3, 3, 1, 1, 1, 1, 1, 1, 0, 5),
    ("act k3 s2",  128,  28,  28, 128, 3, 3, 2, 2, 1, 1, 1, 1, 0, 5),
    ("act rgb",      3,  56,  56,  64, 3, 3, 1, 1, 1, 1, 1, 1, 0, 5),
    ("act dw",      32,  56,  56,  32, 3, 3, 1, 1, 1, 1, 1, 1, 1, 3),
)

# Emit cases the generator must refuse (an identical "-1" on both sides).
CONV_EMIT_REFUSALS = {"feat banks", "k3 ic6000", "deconv dw", "deconv s2"}

# ── int8 conv matrix ──────────────────────────────────────────────────────────
#
# The int8 generators take the same 12 geometry scalars and the three IOVAs; the DW
# rows add the quant tail (the int8 direct path ignores it). IC/OC are the group-padded
# extents the runtime passes (32 and 32 for the int8 weight cube) except where a row is
# deliberately unaligned to pin the generator's own arithmetic.
#
# (name, ic, ih, iw, oc, kh, kw, sy, sx, pt, pl, dy, dx, dw_group, expect_ok)
CONV_INT8_EMITS: tuple[tuple, ...] = (
    # Direct: the plain 3x3, the RGB stem tile (IC padded to 32 by the caller), 1x1,
    # stride 2, pad 0, a 7x7, and the spatial-tiled band shapes the tiler emits.
    ("i8 3x3 p1",    32,   8,   8,  32, 3, 3, 1, 1, 1, 1, 1, 1, 0, True),
    ("i8 3x3 p0",    64,  14,  14,  64, 3, 3, 1, 1, 0, 0, 1, 1, 0, True),
    ("i8 1x1",       64,   8,   8,  64, 1, 1, 1, 1, 0, 0, 1, 1, 0, True),
    ("i8 s2",        32,  16,  16,  32, 3, 3, 2, 2, 0, 0, 1, 1, 0, True),
    ("i8 k7 p3",     32,  16,  16,  32, 7, 7, 1, 1, 3, 3, 1, 1, 0, True),
    ("i8 dil2",      32,  16,  16,  32, 3, 3, 1, 1, 2, 2, 2, 2, 0, True),
    # A 4-bank weight tile: 96 kernels at IC 128 3x3 = 110592 B.
    ("i8 oc96",     128,  28,  28,  96, 3, 3, 1, 1, 1, 1, 1, 1, 0, True),
    # A row-band tile: the same shape at a shallower height (the tiler's sub-job).
    ("i8 band",     128,  14,  28,  96, 3, 3, 1, 1, 0, 0, 1, 1, 0, True),
    # Unaligned extents the generator still encodes (the runtime pads them away).
    ("i8 ic3",        3,  56,  56,  32, 3, 3, 1, 1, 1, 1, 1, 1, 0, True),
    # Refusals: a feature above 11 banks (-1) and a kernel above one bank (-2).
    ("i8 feat banks", 32, 400, 400, 32, 1, 1, 1, 1, 0, 0, 1, 1, 0, False),
    ("i8 kernel bank", 12000, 4, 4, 32, 3, 3, 1, 1, 0, 0, 1, 1, 0, False),
    # Depthwise, int8 out: the requant writer. The scale triple feeds the OUT_CVT
    # derivation; the zero points are model-domain (the harness adds 0x80).
    # (..., dw_group, int8_out, in_scale, w_scale, out_scale, in_zp, w_zp, out_zp)
    ("i8dw 3x3",     64,   8,   8,  64, 3, 3, 1, 1, 1, 1, 1, 1, 64, True,
     1, 0.0213, 0.00347, 0.0925, -2, 0, 5),
    ("i8dw s2",      64,  16,  16,  64, 3, 3, 2, 2, 1, 1, 1, 1, 64, True,
     1, 0.0213, 0.00347, 0.0925, 37, 0, -20),
    ("i8dw zp ends", 64,   9,   7,  64, 3, 3, 1, 2, 1, 1, 1, 1, 64, True,
     1, 0.0213, 0.00347, 0.0925, -128, 0, 127),
    ("i8dw 5x5",     64,   8,  11,  64, 5, 5, 1, 1, 2, 2, 1, 1, 64, True,
     1, 0.0156, 0.0078, 0.125, 127, 0, -128),
    ("i8dw unit",    32,   6,   6,  32, 3, 3, 1, 1, 0, 0, 1, 1, 64, True,
     1, 1.0, 1.0, 1.0, 0, 0, 0),
    ("i8dw raw",     64,   8,   8,  64, 3, 3, 1, 1, 1, 1, 1, 1, 64, True,
     0, 0.0213, 0.00347, 0.0925, -2, 0, 5),
    ("i8dw g0",      64,   8,   8,  64, 3, 3, 1, 1, 1, 1, 1, 1, 0, True,
     1, 0.0213, 0.00347, 0.0925, 0, 0, 0),
    # The generator is more permissive than the runtime plan: a 25x25 G-chunk above one
    # bank still *emits* (the DW branch only checks the whole weight cube against the
    # banks the feature leaves; `conv2d_dw_int8_plan_code` is what refuses it), and a
    # feature above 11 banks is the generator's own -1.
    ("i8dw k25",     64,  64,  64,  64, 25, 25, 1, 1, 0, 0, 1, 1, 64, True,
     1, 0.0213, 0.00347, 0.0925, 0, 0, 0),
    ("i8dw feat banks", 64, 400, 400, 64, 3, 3, 1, 1, 0, 0, 1, 1, 64, False,
     1, 0.0213, 0.00347, 0.0925, 0, 0, 0),
)

CONV_INT8_REFUSALS = {"i8 feat banks", "i8 kernel bank", "i8dw feat banks"}

SCALE_BITS = lambda v: f"0x{struct.unpack('<I', struct.pack('<f', v))[0]:08x}"


def check_conv(
    oracle: Path,
    dump: Path,
    failures: list[Failure],
    verbose: bool,
) -> tuple[int, int, int]:
    """Returns (plan checks, direct/dw program checks, fused-act program checks)."""
    kind = Dtype("conv", "conv", "emit-conv", n_align=1, k_align=1)
    plan_checks = 0
    for row in CONV_PLANS:
        (name, ic, ih, iw, oc, kh, kw, sy, sx, pt, pl, dy, dx, dw, oh, ow,
         expect) = row
        args = ["conv", str(ic), str(ih), str(iw), str(oc), str(kh), str(kw),
                str(sy), str(sx), str(pt), str(pl), str(dy), str(dx), str(dw)]
        if oh or ow:
            args += [str(oh), str(ow)]
        c_proc = run(oracle, args)
        r_proc = run(dump, args)
        plan_checks += 1
        if not compare(failures, kind, f"{name}: {' '.join(args[1:])}", c_proc, r_proc):
            continue
        got = c_proc.stdout.strip()
        if got != str(expect):
            failures.append(
                Failure(kind.name, name, f"expected plan code {expect}, got {got}")
            )
        elif verbose:
            print(f"  {name:<12} {' '.join(args[1:])[:62]:<62} -> {got}")

    prog_checks = 0
    for row in CONV_EMITS:
        (name, ic, ih, iw, oc, kh, kw, sy, sx, pt, pl, dy, dx, dw, fp32,
         dsy, dsx) = row
        args = ["emit-conv", str(ic), str(ih), str(iw), str(oc), str(kh), str(kw),
                str(sy), str(sx), str(pt), str(pl), str(dy), str(dx), str(dw),
                str(fp32), IN_DMA, WT_DMA, OUT_DMA, str(dsy), str(dsx)]
        c_proc = run(oracle, args)
        r_proc = run(dump, args)
        prog_checks += 1
        if not compare(failures, kind, f"{name}: {' '.join(args[1:])}", c_proc, r_proc):
            continue
        refused = c_proc.stdout.strip() == "-1"
        if refused != (name in CONV_EMIT_REFUSALS):
            failures.append(
                Failure(
                    kind.name,
                    name,
                    f"expected {'a refusal' if name in CONV_EMIT_REFUSALS else 'a program'}, "
                    f"got {c_proc.stdout.strip()!r}",
                )
            )
        elif verbose:
            words = c_proc.stdout.splitlines()
            what = "refused" if refused else f"{words[0]} words"
            print(f"  {name:<12} {' '.join(args[1:])[:58]:<58} -> {what}")

    act_checks = 0
    for row in CONV_ACTS:
        name, ic, ih, iw, oc, kh, kw, sy, sx, pt, pl, dy, dx, dw, k = row
        args = ["emit-conv-act", str(ic), str(ih), str(iw), str(oc), str(kh), str(kw),
                str(sy), str(sx), str(pt), str(pl), str(dy), str(dx), str(dw),
                str(k), IN_DMA, WT_DMA, OUT_DMA]
        c_proc = run(oracle, args)
        r_proc = run(dump, args)
        act_checks += 1
        if not compare(failures, kind, f"{name}: {' '.join(args[1:])}", c_proc, r_proc):
            continue
        if c_proc.stdout.strip() == "-1":
            failures.append(Failure(kind.name, name, "the fused program was refused"))
        elif verbose:
            words = c_proc.stdout.splitlines()
            print(f"  {name:<12} {' '.join(args[1:])[:58]:<58} -> {words[0]} words")
    return plan_checks, prog_checks, act_checks

# (C, H, W) for the decomposition (`pool C H W`): the plan reason code plus the per-axis
# factor lists. Covers every branch of `rocket_global_avgpool_plan` — square, equal-count
# rectangle, unequal count (-6), not 16-smooth on H (-4) and on W (-5), the 1x1 identity,
# the dim bound (-3), the channel-field bound (-5) and a non-positive dim (-1).
POOL_PLAN_SHAPES: tuple[tuple[int, int, int], ...] = (
    (8, 7, 7),
    (8, 14, 14),
    (64, 28, 28),
    (16, 28, 20),
    (8, 16, 8),
    (16, 56, 8),
    (512, 7, 7),
    (130, 7, 7),
    (12, 98, 98),
    (4, 224, 224),
    (8, 17, 17),
    (12, 19, 19),
    (8, 7, 19),
    (8, 1, 1),
    (1, 4096, 4096),
    (8, 8192, 8192),
    (8, 8193, 8),
    (8192, 8, 8),
    (8193, 8, 8),
    (0, 8, 8),
)


def fp16_bits(value: float) -> int:
    """The fp16 bit pattern of `value` (RNE), as the C `_Float16` cast."""
    return int.from_bytes(struct.pack("<e", value), "little")


def pool_recip(kh: int, kw: int) -> int:
    """The reduce `recip_pair` convention: exact per-axis, sqrt split when asymmetric."""
    if kh == kw:
        return fp16_bits(65536.0 / kh)
    return fp16_bits(65536.0 / math.sqrt(kh * kw))


def round_up(value: int, align: int) -> int:
    return max(align, (value + align - 1) // align * align)


def shapes_for(dtype: Dtype) -> list[tuple[int, int, int]]:
    """The valid shape matrix, aligned to this dtype's geometry."""
    return [
        (m, round_up(k, dtype.k_align), round_up(n, dtype.n_align))
        for m, k, n in BASE_SHAPES
    ]


def default_dump(profile: str) -> Path:
    """`$CARGO_TARGET_DIR` (or repo `target`) + the active profile."""
    base = Path(os.environ.get("CARGO_TARGET_DIR") or REPO_ROOT / "target")
    candidates = {
        "debug": base / "debug" / "regcmd_dump",
        "release": base / "release" / "regcmd_dump",
    }
    if profile != "auto":
        return candidates[profile]
    existing = [p for p in candidates.values() if p.is_file()]
    if not existing:
        return candidates["debug"]
    # The most recently built binary is the active profile.
    return max(existing, key=lambda p: p.stat().st_mtime)


@dataclass
class Failure:
    dtype: str
    case: str
    detail: str


def run(binary: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(binary), *args], capture_output=True, text=True, check=False
    )


def is_not_implemented(proc: subprocess.CompletedProcess[str]) -> bool:
    if proc.returncode == 0:
        return False
    text = (proc.stderr or "").lower()
    return "not implemented in the rust port" in text or "unknown mode" in text


def differing_lines(a: str, b: str) -> int:
    a_lines, b_lines = a.splitlines(), b.splitlines()
    diffs = sum(1 for x, y in zip(a_lines, b_lines) if x != y)
    return diffs + abs(len(a_lines) - len(b_lines))


def first_difference(a: str, b: str) -> str:
    """A compact diff: the first differing line, both values, and the count."""
    a_lines, b_lines = a.splitlines(), b.splitlines()
    for i, (x, y) in enumerate(zip(a_lines, b_lines), start=1):
        if x != y:
            return (
                f"line {i}: oracle={x!r} rust={y!r} "
                f"({differing_lines(a, b)} differing line(s))"
            )
    if len(a_lines) != len(b_lines):
        return (
            f"line count: oracle={len(a_lines)} rust={len(b_lines)} "
            f"(first extra: oracle={a_lines[len(b_lines):]!r} "
            f"rust={b_lines[len(a_lines):]!r})"
        )
    return "outputs differ only in trailing bytes"


def compare(
    failures: list[Failure],
    dtype: Dtype,
    case: str,
    oracle_proc: subprocess.CompletedProcess[str],
    rust_proc: subprocess.CompletedProcess[str],
) -> bool:
    if oracle_proc.returncode != 0:
        failures.append(
            Failure(
                dtype.name,
                case,
                f"oracle exited {oracle_proc.returncode}: "
                f"{(oracle_proc.stderr or '').strip()}",
            )
        )
        return False
    if rust_proc.returncode != 0:
        failures.append(
            Failure(
                dtype.name,
                case,
                f"regcmd_dump exited {rust_proc.returncode}: "
                f"{(rust_proc.stderr or '').strip()}",
            )
        )
        return False
    if oracle_proc.stdout != rust_proc.stdout:
        failures.append(
            Failure(dtype.name, case, first_difference(oracle_proc.stdout, rust_proc.stdout))
        )
        return False
    return True


def parse_plan(stdout: str) -> tuple[int, int, int] | None:
    line = stdout.strip()
    if line == "-1":
        return None
    parts = line.split()
    if len(parts) != 4:
        return None
    try:
        mt, kt, nt = (int(p) for p in parts[:3])
    except ValueError:
        return None
    return mt, kt, nt


def probe(
    dtype: Dtype,
    dump: Path,
    failures: list[Failure],
    verbose: bool,
) -> tuple[bool, bool]:
    """(plan_pending, emit_pending) by probing the smallest valid shape."""
    m, k = 4, round_up(32, dtype.k_align)
    n = round_up(32, dtype.n_align)
    plan_pending = emit_pending = False

    plan_args = [dtype.plan_mode, str(m), str(k), str(n)]
    if dtype.groups != (None,):
        plan_args.append("32")
    proc = run(dump, plan_args)
    if is_not_implemented(proc):
        plan_pending = True
        if verbose:
            print(f"  probe {dtype.plan_mode}: {(proc.stderr or '').strip()}")
    elif proc.returncode != 0:
        failures.append(
            Failure(
                dtype.name,
                f"{dtype.plan_mode} probe",
                f"regcmd_dump exited {proc.returncode}: {(proc.stderr or '').strip()}",
            )
        )

    proc = run(dump, [dtype.emit_mode, str(m), str(k), str(n), IN_DMA, WT_DMA, OUT_DMA])
    if is_not_implemented(proc):
        emit_pending = True
        if verbose:
            print(f"  probe {dtype.emit_mode}: {(proc.stderr or '').strip()}")
    elif proc.returncode != 0:
        failures.append(
            Failure(
                dtype.name,
                f"{dtype.emit_mode} probe",
                f"regcmd_dump exited {proc.returncode}: {(proc.stderr or '').strip()}",
            )
        )
    return plan_pending, emit_pending


def check_dtype(
    dtype: Dtype,
    oracle: Path,
    dump: Path,
    failures: list[Failure],
    verbose: bool,
) -> tuple[str, int, int]:
    """Returns (status, plan_checks, emit_checks)."""
    plan_pending, emit_pending = probe(dtype, dump, failures, verbose)
    if plan_pending or emit_pending:
        pending = []
        if plan_pending:
            pending.append(dtype.plan_mode)
        if emit_pending:
            pending.append(dtype.emit_mode)
        detail = "not implemented in the Rust port yet"
        if dtype.milestone:
            detail += f" ({dtype.milestone})"
        print(f"PENDING  {dtype.name}: {', '.join(pending)} — {detail}")
        return "pending", 0, 0

    plan_checks = 0
    emit_programs: dict[tuple[int, int, int, tuple[str, ...]], str] = {}

    def check_plan(m: int, k: int, n: int, group: int | None) -> tuple[int, int, int] | None:
        nonlocal plan_checks
        args = [dtype.plan_mode, str(m), str(k), str(n)]
        if group is not None:
            args.append(str(group))
        case = " ".join(args)
        c_proc = run(oracle, args)
        r_proc = run(dump, args)
        plan_checks += 1
        if not compare(failures, dtype, case, c_proc, r_proc):
            return None
        tile = parse_plan(c_proc.stdout)
        if verbose:
            print(f"  {case:<42} -> {c_proc.stdout.strip()}")
        return tile

    def check_emit(tile: tuple[int, int, int], variant: tuple[str, ...]) -> None:
        mt, kt, nt = tile
        key = (mt, kt, nt, variant)
        if key in emit_programs:
            return
        args = [dtype.emit_mode, str(mt), str(kt), str(nt), IN_DMA, WT_DMA, OUT_DMA, *variant]
        case = " ".join(args)
        c_proc = run(oracle, args)
        r_proc = run(dump, args)
        emit_programs[key] = case
        if compare(failures, dtype, case, c_proc, r_proc) and verbose:
            words = c_proc.stdout.splitlines()
            print(f"  {case:<42} -> {words[0] if words else '?'} words")

    for m, k, n in shapes_for(dtype):
        for group in dtype.groups:
            tile = check_plan(m, k, n, group)
            if tile is not None:
                for variant in dtype.emit_variants:
                    check_emit(tile, variant)

    for m, k, n in INVALID_SHAPES:
        # An "invalid" shape only is one for this dtype's geometry (e.g. K=48 is
        # legal for tf32 but not for fp16/int8/bf16).
        if m <= 0 or k <= 0 or n <= 0 or m % 4 or k % dtype.k_align or n % dtype.n_align:
            check_plan(m, k, n, None)

    status = "pass" if not any(f.dtype == dtype.name for f in failures) else "fail"
    return status, plan_checks, len(emit_programs)


def check_pool(
    oracle: Path,
    dump: Path,
    failures: list[Failure],
    verbose: bool,
) -> tuple[int, int]:
    """Returns (recip checks, plan checks, program checks) over the pool matrix."""
    kind = Dtype("pool", "recip", "emit-pool", n_align=1, k_align=1)
    recip_checks = 0
    for k in list(range(1, 17)) + [32, 100, 1000]:
        args = ["recip", str(k)]
        c_proc = run(oracle, args)
        r_proc = run(dump, args)
        recip_checks += 1
        ok = compare(failures, kind, " ".join(args), c_proc, r_proc)
        if verbose and ok:
            print(f"  {' '.join(args):<24} -> {c_proc.stdout.strip()}")

    plan_checks = 0
    for (c, h, w) in POOL_PLAN_SHAPES:
        args = ["pool", str(c), str(h), str(w)]
        c_proc = run(oracle, args)
        r_proc = run(dump, args)
        plan_checks += 1
        if compare(failures, kind, " ".join(args), c_proc, r_proc) and verbose:
            print(f"  {' '.join(args):<24} -> {c_proc.stdout.split()!r}")

    prog_checks = 0
    for (name, c, ih, iw, oh, ow, kh, kw, sy, sx, pt, pl, pb, pr, method,
         expect_ok) in POOL_CASES:
        rw = rh = 0
        if method == "avg" and kh >= 2 and kw >= 2:
            r = pool_recip(kh, kw)
            rw = rh = r
        args = [
            "emit-pool",
            str(c), str(ih), str(iw), str(oh), str(ow),
            str(kh), str(kw), str(sy), str(sx),
            str(pt), str(pl), str(pb), str(pr),
            str(POOL_METHODS[method]), str(rw), str(rh), IN_DMA, OUT_DMA,
        ]
        c_proc = run(oracle, args)
        r_proc = run(dump, args)
        prog_checks += 1
        ok = compare(failures, kind, f"{name}: {' '.join(args[1:])}", c_proc, r_proc)
        if not ok:
            continue
        refused = c_proc.stdout.strip() == "-1"
        if refused == expect_ok:
            failures.append(
                Failure(
                    kind.name,
                    name,
                    f"expected {'a program' if expect_ok else 'a refusal'}, "
                    f"got {c_proc.stdout.strip()!r}",
                )
            )
        elif verbose:
            words = c_proc.stdout.splitlines()
            what = "refused" if refused else f"{words[0]} words"
            print(f"  {name:<10} {' '.join(args[1:])[:60]:<60} -> {what}")
    return recip_checks, plan_checks, prog_checks


def check_conv_int8(
    oracle: Path,
    dump: Path,
    failures: list[Failure],
    verbose: bool,
) -> tuple[int, int]:
    """The native int8 conv programs: (direct checks, depthwise checks)."""
    kind = Dtype("int8conv", "emit-conv-int8", "emit-conv-dw-int8", n_align=1, k_align=1)
    direct = 0
    dwise = 0
    for row in CONV_INT8_EMITS:
        name = row[0]
        (ic, ih, iw, oc, kh, kw, sy, sx, pt, pl, dy, dx, dw_group) = row[1:14]
        expect_ok = row[14]
        if len(row) == 15:
            args = ["emit-conv-int8", str(ic), str(ih), str(iw), str(oc), str(kh), str(kw),
                    str(sy), str(sx), str(pt), str(pl), str(dy), str(dx),
                    IN_DMA, WT_DMA, OUT_DMA, str(dw_group)]
            direct += 1
        else:
            (int8_out, in_s, w_s, out_s, in_zp, w_zp, out_zp) = row[15:22]
            args = ["emit-conv-dw-int8", str(ic), str(ih), str(iw), str(oc), str(kh),
                    str(kw), str(sy), str(sx), str(pt), str(pl), str(dy), str(dx),
                    IN_DMA, WT_DMA, OUT_DMA, BIAS_DMA,
                    SCALE_BITS(in_s), SCALE_BITS(w_s), SCALE_BITS(out_s),
                    str(in_zp), str(w_zp), str(out_zp), str(int8_out), str(dw_group)]
            dwise += 1
        c_proc = run(oracle, args)
        r_proc = run(dump, args)
        if not compare(failures, kind, f"{name}: {' '.join(args[1:])}", c_proc, r_proc):
            continue
        refused = c_proc.stdout.strip() == "-1"
        if refused != (not expect_ok):
            failures.append(
                Failure(
                    kind.name,
                    name,
                    f"expected {'a refusal' if not expect_ok else 'a program'}, "
                    f"got {c_proc.stdout.strip()!r}",
                )
            )
        elif verbose:
            words = c_proc.stdout.splitlines()
            what = "refused" if refused else f"{words[0]} words"
            print(f"  {name:<14} {' '.join(args[1:])[:56]:<56} -> {what}")
    return direct, dwise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare the Rust rocket-npu planner/emitters against the C oracle.",
    )
    parser.add_argument("--oracle", default=DEFAULT_ORACLE,
                        help=f"C oracle binary (default: {DEFAULT_ORACLE})")
    parser.add_argument("--dump", default=None,
                        help="regcmd_dump binary (default: $CARGO_TARGET_DIR or target/<profile>)")
    parser.add_argument("--profile", choices=("auto", "debug", "release"), default="auto",
                        help="cargo profile used when --dump is omitted (default: auto = newest)")
    parser.add_argument("--kind", action="append", default=None,
                        help="dtype filter, repeatable or comma-separated "
                             f"({', '.join(d.name for d in DTYPES)})")
    parser.add_argument("--verbose", action="store_true", help="print every case and its output")
    args = parser.parse_args()

    oracle = Path(args.oracle)
    dump = Path(args.dump) if args.dump else default_dump(args.profile)
    for label, path in (("oracle", oracle), ("regcmd_dump", dump)):
        if not path.is_file():
            print(f"error: {label} binary not found at {path}", file=sys.stderr)
            if label == "regcmd_dump":
                print("build it with: cargo build -p rocket-npu --bin regcmd_dump",
                      file=sys.stderr)
            return 2

    if args.kind:
        wanted: list[str] = []
        for chunk in args.kind:
            wanted.extend(part.strip() for part in chunk.split(",") if part.strip())
        known = {d.name for d in DTYPES} | {"pool", "conv", "int8conv"}
        unknown = [k for k in wanted if k not in known]
        if unknown:
            print(f"error: unknown --kind {', '.join(unknown)}; known: {', '.join(sorted(known))}",
                  file=sys.stderr)
            return 2
        if not wanted:
            print("error: --kind selected no dtypes", file=sys.stderr)
            return 2
        selected = tuple(d for d in DTYPES if d.name in wanted)
        pool_selected = "pool" in wanted
        conv_selected = "conv" in wanted
        int8conv_selected = "int8conv" in wanted
    else:
        selected = DTYPES
        pool_selected = True
        conv_selected = True
        int8conv_selected = True

    print(f"oracle: {oracle}")
    print(f"dump:   {dump}")
    print()

    failures: list[Failure] = []
    results: list[tuple[Dtype, str, int, int]] = []
    for dtype in selected:
        status, plan_checks, emit_checks = check_dtype(dtype, oracle, dump, failures, args.verbose)
        results.append((dtype, status, plan_checks, emit_checks))

    pool_counts: tuple[int, int, int] | None = None
    if pool_selected:
        pool_counts = check_pool(oracle, dump, failures, args.verbose)

    conv_counts: tuple[int, int, int] | None = None
    if conv_selected:
        conv_counts = check_conv(oracle, dump, failures, args.verbose)

    int8conv_counts: tuple[int, int] | None = None
    if int8conv_selected:
        int8conv_counts = check_conv_int8(oracle, dump, failures, args.verbose)

    print()
    names = [len(d.name) for d in selected]
    if pool_counts is not None:
        names.append(len("pool"))
    if conv_counts is not None:
        names.append(len("conv"))
    if int8conv_counts is not None:
        names.append(len("int8conv"))
    width = max(names)
    for dtype, status, plan_checks, emit_checks in results:
        if status == "pending":
            continue
        counts = f"plan {plan_checks}, emit {emit_checks}"
        print(f"{dtype.name:<{width}}  {status.upper():<5}  {counts}")
    if pool_counts is not None:
        recip_checks, plan_checks, prog_checks = pool_counts
        status = "fail" if any(f.dtype == "pool" for f in failures) else "pass"
        print(f"{'pool':<{width}}  {status.upper():<5}  "
              f"recip {recip_checks}, plan {plan_checks}, program {prog_checks}")
    if conv_counts is not None:
        plan_checks, prog_checks, act_checks = conv_counts
        status = "fail" if any(f.dtype == "conv" for f in failures) else "pass"
        print(f"{'conv':<{width}}  {status.upper():<5}  "
              f"plan {plan_checks}, emit {prog_checks}, act {act_checks}")
    if int8conv_counts is not None:
        direct, dwise = int8conv_counts
        status = "fail" if any(f.dtype == "int8conv" for f in failures) else "pass"
        print(f"{'int8conv':<{width}}  {status.upper():<5}  "
              f"direct {direct}, depthwise {dwise}")

    if failures:
        print()
        print(f"{len(failures)} mismatch(es):")
        for failure in failures:
            print(f"  FAIL {failure.dtype}: {failure.case}")
            print(f"       {failure.detail}")
        print()
        print("RESULT: FAIL")
        return 1

    passed = sum(1 for _, status, _, _ in results if status == "pass")
    pending = sum(1 for _, status, _, _ in results if status == "pending")
    print()
    if results:
        print(f"{len(results)} dtype(s): {passed} pass, {pending} pending, 0 fail")
    if pool_counts is not None:
        print(
            f"pool: {pool_counts[0]} reciprocal(s) + {pool_counts[1]} plan(s) + "
            f"{pool_counts[2]} program(s) compared, all byte-identical"
        )
    if conv_counts is not None:
        print(
            f"conv: {conv_counts[0]} plan(s) + {conv_counts[1]} program(s) + "
            f"{conv_counts[2]} fused-activation program(s) compared, all byte-identical"
        )
    if int8conv_counts is not None:
        print(
            f"int8conv: {int8conv_counts[0]} direct + {int8conv_counts[1]} depthwise "
            f"program(s) compared, all byte-identical"
        )
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
