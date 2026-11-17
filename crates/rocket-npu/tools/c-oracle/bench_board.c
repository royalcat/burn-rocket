// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * bench_board.c — A/B the C library's fp16 one-shot and resident matmul on the
 * board, with the same shape/dims the Rust selftest uses.
 *
 *   bench_board M K N THREADS REPS
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "rocket_npu.h"
#include "rocket_matmul.h"

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static _Float16 rnd(uint64_t *s)
{
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    float v = (float)((*s >> 40) / (double)(1u << 24)) - 0.5f;
    return (_Float16)v;
}

int main(int argc, char **argv)
{
    int M = argc > 1 ? atoi(argv[1]) : 512;
    int K = argc > 2 ? atoi(argv[2]) : 1024;
    int N = argc > 3 ? atoi(argv[3]) : 768;
    int threads = argc > 4 ? atoi(argv[4]) : 4;
    int reps = argc > 5 ? atoi(argv[5]) : 3;

    _Float16 *A = malloc((size_t)M * K * sizeof(*A));
    _Float16 *B = malloc((size_t)N * K * sizeof(*B));
    _Float16 *C = calloc((size_t)M * N, sizeof(*C));
    if (!A || !B || !C) return 2;
    uint64_t seed = 0x123456789abcdef1ull;
    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = rnd(&seed);
    for (size_t i = 0; i < (size_t)N * K; i++) B[i] = rnd(&seed);

    int fd = rocket_open();
    if (fd < 0) { fprintf(stderr, "rocket_open %d\n", fd); return 1; }
    rocket_matmul_fp16(fd, M, K, N, A, B, C);
    double t = now_ms();
    for (int i = 0; i < reps; i++) rocket_matmul_fp16(fd, M, K, N, A, B, C);
    double one = (now_ms() - t) / reps;

    rocket_ctx *ctx = rocket_ctx_create_ex(threads, ROCKET_CTX_TILING_CANONICAL);
    if (!ctx) { fprintf(stderr, "ctx create failed\n"); return 1; }
    double tp = now_ms();
    rocket_weights *w = rocket_weights_pack(ctx, 256, K, N, B);
    double pack = now_ms() - tp;
    if (!w) { fprintf(stderr, "weight pack failed\n"); return 1; }
    rocket_matmul_fp16_prepacked(ctx, M, K, N, A, C, w);
    t = now_ms();
    for (int i = 0; i < reps; i++)
        rocket_matmul_fp16_prepacked(ctx, M, K, N, A, C, w);
    double pre = (now_ms() - t) / reps;

    printf("C threads=%d one_shot %.2f ms prepacked %.2f ms (pack %.2f) M=%d K=%d N=%d\n",
           threads, one, pre, pack, M, K, N);
    rocket_weights_free(ctx, w);
    rocket_ctx_free(ctx);
    rocket_close(fd);
    free(A); free(B); free(C);
    return 0;
}
