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
 * (In the encoder line any of LN1B/BQ/BK/BV/BO/LN2B/BF1/BF2 may be `-`. `mlp` is the
 * C's internal host-handoff MLP, which has no public entry point: the fused entry is
 * what this harness can call, so `mlp` is rejected with a message.)
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
#include "rocket_reduce.h"
#include "rocket_softmax.h"
#include "rocket_norm.h"
#include "rocket_normvision.h"
#include "rocket_ffn.h"
#include "rocket_encoder.h"

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

int main(int argc, char **argv)
{
    if (argc < 3) die("usage: ab_board FAMILY ... IN OUT");
    const char *family = argv[1];
    int fd = rocket_open();
    if (fd < 0) die("cannot open the NPU");

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
    } else {
        die("unknown family");
    }

    rocket_close(fd);
    return 0;
}
