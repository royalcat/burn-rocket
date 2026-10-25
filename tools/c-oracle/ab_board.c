// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * ab_board.c — raw-file A/B harness for the M6b reduce/softmax port.
 *
 * Links the C librocketnpu natively on the board and runs ONE op on a raw input file,
 * writing the raw output, using exactly the argument layout of
 * `reduce_selftest --io` in the Rust port:
 *
 *   ab_board avg|max|min C H W      IN OUT     in f16 [C][H][W], out f16 [C]
 *   ab_board feature M H MEAN       IN OUT     in f16 [M][H],    out f32 [M]
 *   ab_board cumsum M N EXCL REV    IN OUT     in/out f16 [M][N]
 *   ab_board softmax|logsoftmax M N IN OUT     in/out f16 [M][N]
 *   ab_board ce M N                 IN OUT     in f16 [M][N],    out f32 [M]
 *   ab_board scale_rows M N RFILE   IN OUT     in/out f16 [M][N], r f32 [M]
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
    } else {
        die("unknown family");
    }

    rocket_close(fd);
    return 0;
}
