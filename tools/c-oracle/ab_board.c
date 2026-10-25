// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * ab_board.c — raw-file A/B harness for the rocket-npu port.
 *
 * Links the C librocketnpu natively on the board and runs ONE op on a raw input file,
 * writing the raw output, using exactly the argument layout of `reduce_selftest --io`
 * (M6b) and `ops_selftest --io` (M7) in the Rust port:
 *
 *   ab_board avg|max|min C H W      IN OUT     in f16 [C][H][W], out f16 [C]
 *   ab_board feature M H MEAN       IN OUT     in f16 [M][H],    out f32 [M]
 *   ab_board cumsum M N EXCL REV    IN OUT     in/out f16 [M][N]
 *   ab_board softmax|logsoftmax M N IN OUT     in/out f16 [M][N]
 *   ab_board ce M N                 IN OUT     in f16 [M][N],    out f32 [M]
 *   ab_board scale_rows M N RFILE   IN OUT     in/out f16 [M][N], r f32 [M]
 *
 * M7 (norm / vision norm / FFN / encoder). `-` means "no tensor" (the C NULL):
 *
 *   ab_board rmsnorm    M H EPS W               IN OUT
 *   ab_board layernorm  M H EPS GAMMA BETA      IN OUT
 *   ab_board batchnorm  N C P EPS GAMMA BETA MEAN VAR IN OUT
 *   ab_board groupnorm  N C G P EPS GAMMA BETA  IN OUT
 *   ab_board instancenorm N C P EPS GAMMA BETA   IN OUT
 *   ab_board l2norm     M H EPS                 IN OUT
 *   ab_board geglu      N KIND                  GATE UP OUT
 *   ab_board ffn|ffn_fused M H I KIND           X WG WU WD OUT
 *   ab_board mlp|mlp_fused M DIN DFF DOUT KIND B1 X W1 W2 OUT
 *   ab_board encoder T D H DFF EPS LN1G LN1B WQ BQ WK BK WV BV WO BO \
 *                    LN2G LN2B WF1 BF1 WF2 BF2 X OUT
 *
 * M8a (fp16 convolution). Same argument layout as `conv_selftest --io`:
 *
 *   ab_board conv     IC IH IW OC KH KW SY SX PT PL DY DX DW  IN W OUT
 *   ab_board conv_act IC IH IW OC KH KW SY SX PT PL DY DX DW KIND IN W OUT
 *   ab_board conv1d   IC IT OC KW STRIDE PAD  IN W OUT
 *   ab_board conv_t   IC IH IW OC KH KW SY SX PT PL DY DX OPADY OPADX DW  IN W OUT
 *
 * M8b (int8 convolution, pool, resize). Same argument layout as `conv_selftest --io`:
 *
 *   ab_board conv_int8    IC IH IW OC KH KW SY SX PT PL DY DX  IN W OUT
 *   ab_board conv_dw_int8 IC IH IW KH KW SY SX PT PL DY DX  BIAS I_SCALE_BITS \
 *         W_SCALE_BITS O_SCALE_BITS IN_ZP W_ZP OUT_ZP  IN W OUT
 *   ab_board pool|pool_int8|pool_uint8 METHOD C IH IW KH KW SY SX PT PL PB PR  IN OUT
 *   ab_board resize_nearest|resize_bilinear C IH IW SY SX  IN OUT
 *
 * (`W` is `[OC][IC][KH][KW]`/`[C][KH][KW]` for a conv, `[OC][IC][KW]` for conv1d and
 * `[IC][OC][KH][KW]`/`[C][KH][KW]` for a transposed conv. The int8 conv's IN/W and the
 * int8/uint8 pools are one byte per element, `conv_int8`'s OUT is int32, and the scales
 * are f32 bit patterns in hex.)
 *
 * M9 (the model encoders). `BLOB` is a `tools/sam_extract.py` / `tools/siglip_extract.py`
 * blob, `PIXELS` the raw f16 `[ic][image_size][image_size]` input, `OUT` receives the
 * neck / final output and `OUT.hidden` the `(n_layers+1)` intermediate token blocks (the
 * same layout as `encoders_selftest --io`). `*_ref` runs the exact host reference
 * (`fd < 0`) and needs no NPU at all:
 *
 *   ab_board sam|sam_ref       BLOB PIXELS OUT
 *   ab_board siglip|siglip_ref BLOB PIXELS OUT
 *
 * (In the encoder line any of LN1B/BQ/BK/BV/BO/LN2B/BF1/BF2 may be `-`. `mlp` is the
 * C's internal host-handoff MLP, which has no public entry point: the fused entry is
 * what this harness can call, so `mlp` is rejected with a message.)
 *
 * M10 (attention, the softmax placement A/B). Same argument layout as
 * `attn_selftest --io`; `-` is a NULL tensor and the two floats travel as f32 bit
 * patterns in hex, so neither side parses a decimal:
 *
 *   ab_board flash T KV DK DV HEADS HKV SCALE_BITS SOFTCAP_BITS Q K V MASK OUT
 *   ab_board mha   T D HEADS X WQ BQ WK BK WV BV WO BO OUT
 *
 * `ROCKET_ATTN_HOST_SOFTMAX` is read by the C library itself, with its own per-entry
 * defaults: `flash` parses the value (`0` = on-NPU) while `mha` only tests presence
 * (`=1` host, unset on-NPU). `ab_compare.py --attn` sets each side accordingly.
 *
 * The CE target rule matches the port: target[row] = (row*7919 + 13) % N, with row 0
 * set to -1 so the out-of-range guard is exercised on both sides.
 *
 * Build (on the board, from a native librocketnpu build):
 *   gcc -O2 -I ref/include -I ref ab_board.c ref/build/librocketnpu.a -lm -lpthread -o ab_board
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "rocket_npu.h"
#include "rocket_attn.h"
#include "rocket_conv.h"
#include "rocket_pool.h"
#include "rocket_reduce.h"
#include "rocket_resize.h"
#include "rocket_softmax.h"
#include "rocket_norm.h"
#include "rocket_normvision.h"
#include "rocket_ffn.h"
#include "rocket_encoder.h"
#include "rocket_sam.h"
#include "rocket_siglip.h"

static void die(const char *what)
{
    fprintf(stderr, "ab_board: %s\n", what);
    exit(1);
}


static void *slurp(const char *path, size_t *bytes)
{
    FILE *f = fopen(path, "rb");
    if (!f) die("open input");
    if (fseek(f, 0, SEEK_END) != 0) die("seek");
    long n = ftell(f);
    if (n < 0) die("tell");
    rewind(f);
    void *p = malloc((size_t)n ? (size_t)n : 1);
    if (!p) die("malloc");
    if (fread(p, 1, (size_t)n, f) != (size_t)n) die("read");
    fclose(f);
    *bytes = (size_t)n;
    return p;
}

static void spit(const char *path, const void *p, size_t bytes)
{
    FILE *f = fopen(path, "wb");
    if (!f) die("open output");
    if (fwrite(p, 1, bytes, f) != bytes) die("write");
    fclose(f);
}

/* An fp16 tensor of exactly `n` elements. */
static _Float16 *slurp_f16(const char *path, size_t n)
{
    size_t bytes;
    _Float16 *p = slurp(path, &bytes);
    if (bytes != n * sizeof(_Float16)) die("input size");
    return p;
}

/* An optional fp16 tensor: `-` is the C NULL. */
static _Float16 *slurp_f16_opt(const char *path, size_t n)
{
    return !strcmp(path, "-") ? NULL : slurp_f16(path, n);
}

/* An f32 argument given as a hex bit pattern (`0x3db504f3`), so the two sides cannot
 * disagree about a decimal's rounding. */
static float arg_f32(const char *s)
{
    uint32_t bits = (uint32_t)strtoul(s, NULL, 0);
    float v;
    memcpy(&v, &bits, sizeof v);
    return v;
}

int main(int argc, char **argv)
{
    if (argc < 3) die("usage: ab_board FAMILY ... IN OUT");
    const char *family = argv[1];
    /* The `*_ref` encoder families run the C `fd < 0` host reference, so they work on a
     * host with no NPU (the M9 self-check on x86). Every other family needs the device. */
    const int host_only = !strcmp(family, "sam_ref") || !strcmp(family, "siglip_ref");
    int fd = host_only ? -1 : rocket_open();
    if (!host_only && fd < 0) die("cannot open the NPU");

    int rc = 0;
    if (!strcmp(family, "avg") || !strcmp(family, "max") || !strcmp(family, "min")) {
        if (argc < 7) die("avg/max/min C H W IN OUT");
        int C = atoi(argv[2]), H = atoi(argv[3]), W = atoi(argv[4]);
        size_t bytes;
        _Float16 *in = slurp(argv[5], &bytes);
        if (bytes != (size_t)C * H * W * sizeof(_Float16)) die("input size");
        _Float16 *out = calloc((size_t)C, sizeof(_Float16));
        if (!out) die("calloc");
        if (!strcmp(family, "avg"))
            rc = rocket_global_avgpool_fp16(fd, C, H, W, in, out);
        else if (!strcmp(family, "max"))
            rc = rocket_global_maxpool_fp16(fd, C, H, W, in, out);
        else
            rc = rocket_global_minpool_fp16(fd, C, H, W, in, out);
        if (rc) die("pool op failed");
        spit(argv[6], out, (size_t)C * sizeof(_Float16));
        free(in);
        free(out);
    } else if (!strcmp(family, "feature")) {
        if (argc < 7) die("feature M H MEAN IN OUT");
        int M = atoi(argv[2]), H = atoi(argv[3]), mean = atoi(argv[4]);
        size_t bytes;
        _Float16 *in = slurp(argv[5], &bytes);
        if (bytes != (size_t)M * H * sizeof(_Float16)) die("input size");
        float *out = calloc((size_t)M, sizeof(float));
        if (!out) die("calloc");
        rc = rocket_reduce_feature_fp16(fd, M, H, in, out, mean);
        if (rc) die("feature op failed");
        spit(argv[6], out, (size_t)M * sizeof(float));
        free(in);
        free(out);
    } else if (!strcmp(family, "cumsum")) {
        if (argc < 8) die("cumsum M N EXCL REV IN OUT");
        int M = atoi(argv[2]), N = atoi(argv[3]);
        int excl = atoi(argv[4]), rev = atoi(argv[5]);
        size_t bytes;
        _Float16 *in = slurp(argv[6], &bytes);
        if (bytes != (size_t)M * N * sizeof(_Float16)) die("input size");
        _Float16 *out = calloc((size_t)M * N, sizeof(_Float16));
        if (!out) die("calloc");
        rc = rocket_cumsum_fp16(fd, M, N, in, out, excl, rev);
        if (rc) die("cumsum op failed");
        spit(argv[7], out, (size_t)M * N * sizeof(_Float16));
        free(in);
        free(out);
    } else if (!strcmp(family, "softmax") || !strcmp(family, "logsoftmax")) {
        if (argc < 6) die("softmax M N IN OUT");
        int M = atoi(argv[2]), N = atoi(argv[3]);
        size_t bytes;
        _Float16 *in = slurp(argv[4], &bytes);
        if (bytes != (size_t)M * N * sizeof(_Float16)) die("input size");
        _Float16 *out = calloc((size_t)M * N, sizeof(_Float16));
        if (!out) die("calloc");
        rc = !strcmp(family, "softmax")
                 ? rocket_softmax_fp16(fd, M, N, in, out)
                 : rocket_logsoftmax_fp16(fd, M, N, in, out);
        if (rc) die("softmax op failed");
        spit(argv[5], out, (size_t)M * N * sizeof(_Float16));
        free(in);
        free(out);
    } else if (!strcmp(family, "ce")) {
        if (argc < 6) die("ce M N IN OUT");
        int M = atoi(argv[2]), N = atoi(argv[3]);
        size_t bytes;
        _Float16 *in = slurp(argv[4], &bytes);
        if (bytes != (size_t)M * N * sizeof(_Float16)) die("input size");
        int *target = calloc((size_t)M, sizeof(int));
        float *out = calloc((size_t)M, sizeof(float));
        if (!target || !out) die("calloc");
        for (int m = 0; m < M; m++) {
            target[m] = (m * 7919 + 13) % N;
            if (m == 0) target[m] = -1;
        }
        rc = rocket_cross_entropy_fp16(fd, M, N, in, target, out);
        if (rc) die("cross-entropy op failed");
        spit(argv[5], out, (size_t)M * sizeof(float));
        free(in);
        free(target);
        free(out);
    } else if (!strcmp(family, "scale_rows")) {
        if (argc < 7) die("scale_rows M N RFILE IN OUT");
        int M = atoi(argv[2]), N = atoi(argv[3]);
        size_t nbytes, rbytes;
        _Float16 *in = slurp(argv[5], &nbytes);
        float *r = slurp(argv[4], &rbytes);
        if (nbytes != (size_t)M * N * sizeof(_Float16)) die("input size");
        if (rbytes != (size_t)M * sizeof(float)) die("r size");
        _Float16 *out = calloc((size_t)M * N, sizeof(_Float16));
        if (!out) die("calloc");
        rc = rocket_scale_rows_fp16(fd, M, N, in, r, out);
        if (rc) die("scale_rows op failed");
        spit(argv[6], out, (size_t)M * N * sizeof(_Float16));
        free(in);
        free(r);
        free(out);
    } else if (!strcmp(family, "rmsnorm")) {
        if (argc < 8) die("rmsnorm M H EPS W IN OUT");
        int M = atoi(argv[2]), H = atoi(argv[3]);
        float eps = (float)atof(argv[4]);
        _Float16 *w = slurp_f16(argv[5], (size_t)H);
        _Float16 *in = slurp_f16(argv[6], (size_t)M * H);
        _Float16 *out = calloc((size_t)M * H, sizeof(_Float16));
        if (!out) die("calloc");
        rc = rocket_rmsnorm_fp16(fd, M, H, in, w, eps, out);
        if (rc) die("rmsnorm op failed");
        spit(argv[7], out, (size_t)M * H * sizeof(_Float16));
        free(in); free(w); free(out);
    } else if (!strcmp(family, "layernorm")) {
        if (argc < 8) die("layernorm M H EPS GAMMA BETA IN OUT");
        int M = atoi(argv[2]), H = atoi(argv[3]);
        float eps = (float)atof(argv[4]);
        _Float16 *gamma = slurp_f16(argv[5], (size_t)H);
        _Float16 *beta = slurp_f16_opt(argv[6], (size_t)H);
        _Float16 *in = slurp_f16(argv[7], (size_t)M * H);
        _Float16 *out = calloc((size_t)M * H, sizeof(_Float16));
        if (!out) die("calloc");
        rc = rocket_layernorm_fp16(fd, M, H, in, gamma, beta, eps, out);
        if (rc) die("layernorm op failed");
        spit(argv[8], out, (size_t)M * H * sizeof(_Float16));
        free(in); free(gamma); free(beta); free(out);
    } else if (!strcmp(family, "batchnorm")) {
        if (argc < 11) die("batchnorm N C P EPS GAMMA BETA MEAN VAR IN OUT");
        int N = atoi(argv[2]), C = atoi(argv[3]), P = atoi(argv[4]);
        float eps = (float)atof(argv[5]);
        _Float16 *gamma = slurp_f16_opt(argv[6], (size_t)C);
        _Float16 *beta = slurp_f16_opt(argv[7], (size_t)C);
        _Float16 *mean = slurp_f16(argv[8], (size_t)C);
        _Float16 *var = slurp_f16(argv[9], (size_t)C);
        size_t tot = (size_t)N * C * P;
        _Float16 *in = slurp_f16(argv[10], tot);
        _Float16 *out = calloc(tot, sizeof(_Float16));
        if (!out) die("calloc");
        rc = rocket_batchnorm_fp16(fd, N, C, P, in, gamma, beta, mean, var, eps, out);
        if (rc) die("batchnorm op failed");
        spit(argv[11], out, tot * sizeof(_Float16));
        free(in); free(gamma); free(beta); free(mean); free(var); free(out);
    } else if (!strcmp(family, "groupnorm")) {
        if (argc < 10) die("groupnorm N C G P EPS GAMMA BETA IN OUT");
        int N = atoi(argv[2]), C = atoi(argv[3]), G = atoi(argv[4]), P = atoi(argv[5]);
        float eps = (float)atof(argv[6]);
        _Float16 *gamma = slurp_f16_opt(argv[7], (size_t)C);
        _Float16 *beta = slurp_f16_opt(argv[8], (size_t)C);
        size_t tot = (size_t)N * C * P;
        _Float16 *in = slurp_f16(argv[9], tot);
        _Float16 *out = calloc(tot, sizeof(_Float16));
        if (!out) die("calloc");
        rc = rocket_groupnorm_fp16(fd, N, C, G, P, in, gamma, beta, eps, out);
        if (rc) die("groupnorm op failed");
        spit(argv[10], out, tot * sizeof(_Float16));
        free(in); free(gamma); free(beta); free(out);
    } else if (!strcmp(family, "instancenorm")) {
        if (argc < 9) die("instancenorm N C P EPS GAMMA BETA IN OUT");
        int N = atoi(argv[2]), C = atoi(argv[3]), P = atoi(argv[4]);
        float eps = (float)atof(argv[5]);
        _Float16 *gamma = slurp_f16_opt(argv[6], (size_t)C);
        _Float16 *beta = slurp_f16_opt(argv[7], (size_t)C);
        size_t tot = (size_t)N * C * P;
        _Float16 *in = slurp_f16(argv[8], tot);
        _Float16 *out = calloc(tot, sizeof(_Float16));
        if (!out) die("calloc");
        rc = rocket_instancenorm_fp16(fd, N, C, P, in, gamma, beta, eps, out);
        if (rc) die("instancenorm op failed");
        spit(argv[9], out, tot * sizeof(_Float16));
        free(in); free(gamma); free(beta); free(out);
    } else if (!strcmp(family, "l2norm")) {
        if (argc < 7) die("l2norm M H EPS IN OUT");
        int M = atoi(argv[2]), H = atoi(argv[3]);
        float eps = (float)atof(argv[4]);
        _Float16 *in = slurp_f16(argv[5], (size_t)M * H);
        _Float16 *out = calloc((size_t)M * H, sizeof(_Float16));
        if (!out) die("calloc");
        rc = rocket_l2norm_fp16(fd, M, H, in, eps, out);
        if (rc) die("l2norm op failed");
        spit(argv[6], out, (size_t)M * H * sizeof(_Float16));
        free(in); free(out);
    } else if (!strcmp(family, "geglu")) {
        if (argc < 6) die("geglu N KIND GATE UP OUT");
        int n = atoi(argv[2]), kind = atoi(argv[3]);
        _Float16 *gate = slurp_f16(argv[4], (size_t)n);
        _Float16 *up = slurp_f16(argv[5], (size_t)n);
        _Float16 *out = calloc((size_t)n, sizeof(_Float16));
        if (!out) die("calloc");
        rc = rocket_geglu_fp16(fd, gate, up, kind, out, n);
        if (rc) die("geglu op failed");
        spit(argv[6], out, (size_t)n * sizeof(_Float16));
        free(gate); free(up); free(out);
    } else if (!strcmp(family, "ffn") || !strcmp(family, "ffn_fused")) {
        if (argc < 10) die("ffn M H I KIND X WG WU WD OUT");
        int M = atoi(argv[2]), H = atoi(argv[3]), I = atoi(argv[4]), kind = atoi(argv[5]);
        _Float16 *x  = slurp_f16(argv[6], (size_t)M * H);
        _Float16 *wg = slurp_f16(argv[7], (size_t)I * H);
        _Float16 *wu = slurp_f16(argv[8], (size_t)I * H);
        _Float16 *wd = slurp_f16(argv[9], (size_t)H * I);
        _Float16 *out = calloc((size_t)M * H, sizeof(_Float16));
        if (!out) die("calloc");
        rc = !strcmp(family, "ffn")
                 ? rocket_ffn_fp16(fd, M, H, I, x, wg, wu, wd, kind, out)
                 : rocket_ffn_fp16_fused(fd, M, H, I, x, wg, wu, wd, kind, out);
        if (rc) die("ffn op failed");
        spit(argv[10], out, (size_t)M * H * sizeof(_Float16));
        free(x); free(wg); free(wu); free(wd); free(out);
    } else if (!strcmp(family, "mlp_fused")) {
        if (argc < 11) die("mlp_fused M DIN DFF DOUT KIND B1 X W1 W2 OUT");
        int M = atoi(argv[2]), Din = atoi(argv[3]), Dff = atoi(argv[4]);
        int Dout = atoi(argv[5]), kind = atoi(argv[6]);
        _Float16 *b1 = slurp_f16_opt(argv[7], (size_t)Dff);
        _Float16 *x  = slurp_f16(argv[8], (size_t)M * Din);
        _Float16 *w1 = slurp_f16(argv[9], (size_t)Dff * Din);
        _Float16 *w2 = slurp_f16(argv[10], (size_t)Dout * Dff);
        _Float16 *out = calloc((size_t)M * Dout, sizeof(_Float16));
        if (!out) die("calloc");
        rc = rocket_mlp_fp16_fused(fd, M, Din, Dff, Dout, x, w1, b1, w2, kind, out);
        if (rc) die("mlp op failed");
        spit(argv[11], out, (size_t)M * Dout * sizeof(_Float16));
        free(x); free(b1); free(w1); free(w2); free(out);
    } else if (!strcmp(family, "encoder")) {
        if (argc < 25) die("encoder T D H DFF EPS LN1G LN1B WQ BQ WK BK WV BV WO BO "
                           "LN2G LN2B WF1 BF1 WF2 BF2 X OUT");
        int T = atoi(argv[2]), d = atoi(argv[3]), nh = atoi(argv[4]), dff = atoi(argv[5]);
        float eps = (float)atof(argv[6]);
        char **a = &argv[7];
        _Float16 *ln1_g = slurp_f16(a[0], (size_t)d);
        _Float16 *ln1_b = slurp_f16_opt(a[1], (size_t)d);
        _Float16 *wq    = slurp_f16(a[2], (size_t)d * d);
        _Float16 *bq    = slurp_f16_opt(a[3], (size_t)d);
        _Float16 *wk    = slurp_f16(a[4], (size_t)d * d);
        _Float16 *bk    = slurp_f16_opt(a[5], (size_t)d);
        _Float16 *wv    = slurp_f16(a[6], (size_t)d * d);
        _Float16 *bv    = slurp_f16_opt(a[7], (size_t)d);
        _Float16 *wo    = slurp_f16(a[8], (size_t)d * d);
        _Float16 *bo    = slurp_f16_opt(a[9], (size_t)d);
        _Float16 *ln2_g = slurp_f16(a[10], (size_t)d);
        _Float16 *ln2_b = slurp_f16_opt(a[11], (size_t)d);
        _Float16 *wf1   = slurp_f16(a[12], (size_t)dff * d);
        _Float16 *bf1   = slurp_f16_opt(a[13], (size_t)dff);
        _Float16 *wf2   = slurp_f16(a[14], (size_t)d * dff);
        _Float16 *bf2   = slurp_f16_opt(a[15], (size_t)d);
        _Float16 *x     = slurp_f16(a[16], (size_t)T * d);
        _Float16 *out   = calloc((size_t)T * d, sizeof(_Float16));
        if (!out) die("calloc");
        rc = rocket_encoder_block_fp16(fd, T, d, nh, dff, x, ln1_g, ln1_b, wq, bq, wk, bk,
                                       wv, bv, wo, bo, ln2_g, ln2_b, wf1, bf1, wf2, bf2,
                                       eps, out);
        if (rc) die("encoder op failed");
        spit(a[17], out, (size_t)T * d * sizeof(_Float16));
        free(x); free(ln1_g); free(ln1_b); free(wq); free(bq); free(wk); free(bk);
        free(wv); free(bv); free(wo); free(bo); free(ln2_g); free(ln2_b); free(wf1);
        free(bf1); free(wf2); free(bf2); free(out);
    } else if (!strcmp(family, "conv") || !strcmp(family, "conv_act")) {
        int act = !strcmp(family, "conv_act");
        /* conv IC IH IW OC KH KW SY SX PT PL DY DX DW [KIND] IN W OUT */
        if (argc < (act ? 19 : 18)) die("conv IC IH IW OC KH KW SY SX PT PL DY DX DW [KIND] IN W OUT");
        rocket_conv2d_desc d = {0};
        d.ic = atoi(argv[2]);  d.ih = atoi(argv[3]);  d.iw = atoi(argv[4]);
        d.oc = atoi(argv[5]);  d.kh = atoi(argv[6]);  d.kw = atoi(argv[7]);
        d.stride_y = atoi(argv[8]);  d.stride_x = atoi(argv[9]);
        d.pad_top = atoi(argv[10]);  d.pad_left = atoi(argv[11]);
        d.dil_y = atoi(argv[12]);    d.dil_x = atoi(argv[13]);
        d.depthwise = atoi(argv[14]);
        int kind = 0;
        char **a = &argv[15];
        if (act) { kind = atoi(a[0]); a = &argv[16]; }
        _Float16 *in = slurp_f16(a[0], (size_t)d.ic * d.ih * d.iw);
        size_t wn = d.depthwise ? (size_t)d.ic * d.kh * d.kw
                                : (size_t)d.oc * d.ic * d.kh * d.kw;
        _Float16 *w = slurp_f16(a[1], wn);
        size_t on = (size_t)d.oc * rocket_conv2d_oh(&d) * rocket_conv2d_ow(&d);
        _Float16 *out = calloc(on, sizeof(_Float16));
        if (!out) die("calloc");
        rc = act ? rocket_conv2d_act_fp16(fd, &d, kind, in, w, out)
                 : rocket_conv2d_fp16(fd, &d, in, w, out);
        if (rc) die("conv op failed");
        spit(a[2], out, on * sizeof(_Float16));
        free(in); free(w); free(out);
    } else if (!strcmp(family, "conv1d")) {
        /* conv1d IC IT OC KW STRIDE PAD IN W OUT */
        if (argc < 11) die("conv1d IC IT OC KW STRIDE PAD IN W OUT");
        int ic = atoi(argv[2]), it = atoi(argv[3]), oc = atoi(argv[4]), kw = atoi(argv[5]);
        int stride = atoi(argv[6]), pad = atoi(argv[7]);
        _Float16 *in = slurp_f16(argv[8], (size_t)ic * it);
        _Float16 *w = slurp_f16(argv[9], (size_t)oc * ic * kw);
        int ot = rocket_conv_out_dim(it, kw, stride, pad, 1);
        _Float16 *out = calloc((size_t)oc * ot, sizeof(_Float16));
        if (!out) die("calloc");
        rc = rocket_conv1d_fp16(fd, ic, it, oc, kw, stride, pad, in, w, out);
        if (rc) die("conv1d op failed");
        spit(argv[10], out, (size_t)oc * ot * sizeof(_Float16));
        free(in); free(w); free(out);
    } else if (!strcmp(family, "conv_t")) {
        /* conv_t IC IH IW OC KH KW SY SX PT PL DY DX OPADY OPADX DW IN W OUT */
        if (argc < 20) die("conv_t IC IH IW OC KH KW SY SX PT PL DY DX OPADY OPADX DW IN W OUT");
        rocket_conv_transpose2d_desc d = {0};
        d.ic = atoi(argv[2]);  d.ih = atoi(argv[3]);  d.iw = atoi(argv[4]);
        d.oc = atoi(argv[5]);  d.kh = atoi(argv[6]);  d.kw = atoi(argv[7]);
        d.stride_y = atoi(argv[8]);  d.stride_x = atoi(argv[9]);
        d.pad_top = atoi(argv[10]);  d.pad_left = atoi(argv[11]);
        d.dil_y = atoi(argv[12]);    d.dil_x = atoi(argv[13]);
        d.opad_y = atoi(argv[14]);   d.opad_x = atoi(argv[15]);
        d.depthwise = atoi(argv[16]);
        _Float16 *in = slurp_f16(argv[17], (size_t)d.ic * d.ih * d.iw);
        size_t wn = d.depthwise ? (size_t)d.ic * d.kh * d.kw
                                : (size_t)d.ic * d.oc * d.kh * d.kw;
        _Float16 *w = slurp_f16(argv[18], wn);
        size_t on = (size_t)d.oc * rocket_conv_transpose2d_oh(&d) *
                    rocket_conv_transpose2d_ow(&d);
        _Float16 *out = calloc(on, sizeof(_Float16));
        if (!out) die("calloc");
        rc = rocket_conv_transpose2d_fp16(fd, &d, in, w, out);
        if (rc) die("conv_t op failed");
        spit(argv[19], out, on * sizeof(_Float16));
        free(in); free(w); free(out);
    } else if (!strcmp(family, "conv_int8")) {
        /* conv_int8 IC IH IW OC KH KW SY SX PT PL DY DX IN W OUT */
        if (argc < 16) die("conv_int8 IC IH IW OC KH KW SY SX PT PL DY DX IN W OUT");
        rocket_conv2d_desc d = {0};
        d.ic = atoi(argv[2]); d.ih = atoi(argv[3]); d.iw = atoi(argv[4]);
        d.oc = atoi(argv[5]); d.kh = atoi(argv[6]); d.kw = atoi(argv[7]);
        d.stride_y = atoi(argv[8]); d.stride_x = atoi(argv[9]);
        d.pad_top = atoi(argv[10]); d.pad_left = atoi(argv[11]);
        d.dil_y = atoi(argv[12]);   d.dil_x = atoi(argv[13]);
        size_t ib, wb;
        int8_t *in = slurp(argv[14], &ib);
        int8_t *w = slurp(argv[15], &wb);
        if (ib != (size_t)d.ic * d.ih * d.iw) die("input size");
        if (wb != (size_t)d.oc * d.ic * d.kh * d.kw) die("weight size");
        size_t on = (size_t)d.oc * rocket_conv2d_oh(&d) * rocket_conv2d_ow(&d);
        int32_t *out = calloc(on, sizeof(int32_t));
        if (!out) die("calloc");
        rc = rocket_conv2d_int8(fd, &d, in, w, out);
        if (rc) die("conv_int8 op failed");
        spit(argv[16], out, on * sizeof(int32_t));
        free(in); free(w); free(out);
    } else if (!strcmp(family, "conv_dw_int8")) {
        /* conv_dw_int8 IC IH IW KH KW SY SX PT PL DY DX BIAS I_SCALE W_SCALE
         *               O_SCALE IN_ZP W_ZP OUT_ZP IN W OUT */
        if (argc < 22) die("conv_dw_int8 " "IC IH IW KH KW SY SX PT PL DY DX BIAS "
                           "I_SCALE W_SCALE O_SCALE IN_ZP W_ZP OUT_ZP IN W OUT");
        rocket_conv2d_desc d = {0};
        d.ic = atoi(argv[2]); d.ih = atoi(argv[3]); d.iw = atoi(argv[4]);
        d.oc = d.ic;
        d.kh = atoi(argv[5]); d.kw = atoi(argv[6]);
        d.stride_y = atoi(argv[7]); d.stride_x = atoi(argv[8]);
        d.pad_top = atoi(argv[9]); d.pad_left = atoi(argv[10]);
        d.dil_y = atoi(argv[11]);   d.dil_x = atoi(argv[12]);
        d.depthwise = 1;
        char **a = &argv[13];
        int32_t *bias = NULL;
        if (strcmp(a[0], "-")) {
            size_t bb;
            bias = slurp(a[0], &bb);
            if (bb != (size_t)d.ic * sizeof(int32_t)) die("bias size");
        }
        union { uint32_t u; float f; } cv;
        cv.u = (uint32_t)strtoul(a[1], NULL, 16); float in_s = cv.f;
        cv.u = (uint32_t)strtoul(a[2], NULL, 16); float w_s = cv.f;
        cv.u = (uint32_t)strtoul(a[3], NULL, 16); float out_s = cv.f;
        int in_zp = atoi(a[4]), w_zp = atoi(a[5]), out_zp = atoi(a[6]);
        size_t ib, wb;
        int8_t *in = slurp(a[7], &ib);
        int8_t *w = slurp(a[8], &wb);
        if (ib != (size_t)d.ic * d.ih * d.iw) die("input size");
        if (wb != (size_t)d.ic * d.kh * d.kw) die("weight size");
        size_t on = (size_t)d.ic * rocket_conv2d_oh(&d) * rocket_conv2d_ow(&d);
        int8_t *out = calloc(on, 1);
        if (!out) die("calloc");
        rc = rocket_conv2d_dw_int8(fd, &d, in, w, bias, in_s, w_s, out_s,
                                   in_zp, w_zp, out_zp, out);
        if (rc) die("conv_dw_int8 op failed");
        spit(a[9], out, on);
        free(in); free(w); free(out); free(bias);
    } else if (!strcmp(family, "pool") || !strcmp(family, "pool_int8") ||
               !strcmp(family, "pool_uint8")) {
        /* pool* METHOD C IH IW KH KW SY SX PT PL PB PR IN OUT */
        if (argc < 14) die("pool METHOD C IH IW KH KW SY SX PT PL PB PR IN OUT");
        rocket_pool_desc d = {0};
        d.method = atoi(argv[2]);
        d.c = atoi(argv[3]);  d.ih = atoi(argv[4]);  d.iw = atoi(argv[5]);
        d.kh = atoi(argv[6]); d.kw = atoi(argv[7]);
        d.stride_y = atoi(argv[8]); d.stride_x = atoi(argv[9]);
        d.pad_top = atoi(argv[10]); d.pad_left = atoi(argv[11]);
        d.pad_bottom = atoi(argv[12]); d.pad_right = atoi(argv[13]);
        size_t ib;
        void *in = slurp(argv[14], &ib);
        size_t ni = (size_t)d.c * d.ih * d.iw;
        size_t no = (size_t)d.c * rocket_pool_oh(&d) * rocket_pool_ow(&d);
        if (!strcmp(family, "pool")) {
            _Float16 *out = calloc(no, sizeof(_Float16));
            if (!out) die("calloc");
            rc = rocket_pool_fp16(fd, &d, in, out);
            if (rc) die("pool op failed");
            spit(argv[15], out, no * sizeof(_Float16));
            free(out);
        } else if (!strcmp(family, "pool_int8")) {
            if (ib != ni) die("input size");
            int8_t *out = calloc(no, 1);
            if (!out) die("calloc");
            rc = rocket_pool_int8(fd, &d, in, out);
            if (rc) die("pool_int8 op failed");
            spit(argv[15], out, no);
            free(out);
        } else {
            uint8_t *out = calloc(no, 1);
            if (!out) die("calloc");
            rc = rocket_pool_uint8(fd, &d, in, out);
            if (rc) die("pool_uint8 op failed");
            spit(argv[15], out, no);
            free(out);
        }
        free(in);
    } else if (!strcmp(family, "resize_nearest") || !strcmp(family, "resize_bilinear")) {
        /* resize_* C IH IW SY SX IN OUT */
        if (argc < 7) die("resize_* C IH IW SY SX IN OUT");
        int C = atoi(argv[2]), IH = atoi(argv[3]), IW = atoi(argv[4]);
        int sy = atoi(argv[5]), sx = atoi(argv[6]);
        size_t ib;
        _Float16 *in = slurp(argv[7], &ib);
        if (ib != (size_t)C * IH * IW * sizeof(_Float16)) die("input size");
        size_t on = (size_t)C * IH * sy * IW * sx;
        _Float16 *out = calloc(on, sizeof(_Float16));
        if (!out) die("calloc");
        rc = !strcmp(family, "resize_bilinear")
             ? rocket_upsample_bilinear_fp16(fd, in, out, C, IH, IW, sy, sx)
             : rocket_upsample_nearest_fp16(fd, in, out, C, IH, IW, sy, sx);
        if (rc) die("resize op failed");
        spit(argv[8], out, on * sizeof(_Float16));
        free(in); free(out);
    } else if (!strcmp(family, "flash")) {
        /* flash T KV DK DV HEADS HKV SCALE_BITS SOFTCAP_BITS Q K V MASK OUT */
        if (argc < 15) die("flash T KV DK DV HEADS HKV SCALE_BITS SOFTCAP_BITS Q K V MASK OUT");
        int T = atoi(argv[2]), KV = atoi(argv[3]), DK = atoi(argv[4]), DV = atoi(argv[5]);
        int NH = atoi(argv[6]), NKVH = atoi(argv[7]);
        float scale = arg_f32(argv[8]), softcap = arg_f32(argv[9]);
        size_t qn = (size_t)NH * T * DK, kn = (size_t)NKVH * KV * DK;
        size_t vn = (size_t)NKVH * DV * KV, on = (size_t)NH * T * DV;
        _Float16 *Q = slurp_f16(argv[10], qn);
        _Float16 *K = slurp_f16(argv[11], kn);
        _Float16 *V = slurp_f16(argv[12], vn);
        _Float16 *M = slurp_f16_opt(argv[13], (size_t)T * KV);
        _Float16 *out = calloc(on, sizeof(_Float16));
        if (!out) die("calloc");
        rc = rocket_flash_attn_fp16(fd, T, KV, DK, DV, NH, NKVH, scale, softcap, Q, K, V, M, out);
        if (rc) die("flash attention op failed");
        spit(argv[14], out, on * sizeof(_Float16));
        free(Q); free(K); free(V); free(M); free(out);
    } else if (!strcmp(family, "mha")) {
        /* mha T D HEADS X WQ BQ WK BK WV BV WO BO OUT */
        if (argc < 15) die("mha T D HEADS X WQ BQ WK BK WV BV WO BO OUT");
        int T = atoi(argv[2]), D = atoi(argv[3]), NH = atoi(argv[4]);
        size_t dd = (size_t)D * D;
        _Float16 *x = slurp_f16(argv[5], (size_t)T * D);
        _Float16 *wq = slurp_f16(argv[6], dd);
        _Float16 *bq = slurp_f16_opt(argv[7], (size_t)D);
        _Float16 *wk = slurp_f16(argv[8], dd);
        _Float16 *bk = slurp_f16_opt(argv[9], (size_t)D);
        _Float16 *wv = slurp_f16(argv[10], dd);
        _Float16 *bv = slurp_f16_opt(argv[11], (size_t)D);
        _Float16 *wo = slurp_f16(argv[12], dd);
        _Float16 *bo = slurp_f16_opt(argv[13], (size_t)D);
        _Float16 *out = calloc((size_t)T * D, sizeof(_Float16));
        if (!out) die("calloc");
        rc = rocket_mha_self_fp16(fd, T, D, NH, x, wq, bq, wk, bk, wv, bv, wo, bo, out);
        if (rc) die("mha_self op failed");
        spit(argv[14], out, (size_t)T * D * sizeof(_Float16));
        free(x); free(wq); free(bq); free(wk); free(bk);
        free(wv); free(bv); free(wo); free(bo); free(out);
    } else if (!strcmp(family, "sam") || !strcmp(family, "sam_ref")) {
        /* sam|sam_ref BLOB PIXELS OUT   (OUT.hidden gets the (nL+1) token blocks) */
        if (argc < 5) die("sam BLOB PIXELS OUT");
        rocket_sam_model m;
        int lrc = rocket_sam_load(argv[2], &m);
        if (lrc) die("rocket_sam_load");
        size_t pb;
        _Float16 *pix = slurp(argv[3], &pb);
        const size_t pix_n = (size_t)m.ic * m.image_size * m.image_size;
        if (pb != pix_n * sizeof(_Float16)) die("pixels size");
        const size_t Gd = (size_t)m.grid * m.grid * m.d;
        const size_t Nn = (size_t)m.neck_out * m.grid * m.grid;
        _Float16 *out = calloc(Nn, sizeof(_Float16));
        _Float16 *hid = calloc((size_t)(m.n_layers + 1) * Gd, sizeof(_Float16));
        if (!out || !hid) die("calloc");
        rc = rocket_sam_encode(fd, &m, pix, out, hid);
        if (rc) die("rocket_sam_encode");
        char hpath[4096];
        snprintf(hpath, sizeof hpath, "%s.hidden", argv[4]);
        spit(argv[4], out, Nn * sizeof(_Float16));
        spit(hpath, hid, (size_t)(m.n_layers + 1) * Gd * sizeof(_Float16));
        free(pix); free(out); free(hid);
        rocket_sam_free(&m);
    } else if (!strcmp(family, "siglip") || !strcmp(family, "siglip_ref")) {
        /* siglip|siglip_ref BLOB PIXELS OUT   (OUT.hidden gets the (nL+1) token blocks) */
        if (argc < 5) die("siglip BLOB PIXELS OUT");
        rocket_siglip_model m;
        int lrc = rocket_siglip_load(argv[2], &m);
        if (lrc) die("rocket_siglip_load");
        size_t pb;
        _Float16 *pix = slurp(argv[3], &pb);
        const size_t pix_n = (size_t)m.ic * m.image_size * m.image_size;
        if (pb != pix_n * sizeof(_Float16)) die("pixels size");
        const size_t ntok = (size_t)m.L + m.n_prefix;
        const size_t Td = ntok * (size_t)m.d;
        _Float16 *out = calloc(m.n_taps > 0 ? (size_t)m.n_taps * Td : Td, sizeof(_Float16));
        _Float16 *hid = calloc((size_t)(m.n_layers + 1) * Td, sizeof(_Float16));
        if (!out || !hid) die("calloc");
        rc = rocket_siglip_encode(fd, &m, pix, out, hid);
        if (rc) die("rocket_siglip_encode");
        char hpath[4096];
        snprintf(hpath, sizeof hpath, "%s.hidden", argv[4]);
        spit(argv[4], out, (m.n_taps > 0 ? (size_t)m.n_taps * Td : Td) * sizeof(_Float16));
        spit(hpath, hid, (size_t)(m.n_layers + 1) * Td * sizeof(_Float16));
        free(pix); free(out); free(hid);
        rocket_siglip_free(&m);
    } else {
        die("unknown family");
    }

    if (fd >= 0) rocket_close(fd);
    return 0;
}
