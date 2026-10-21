// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * bench_attn.c — time the C library's flash attention on the board.
 *
 *   bench_attn T KV HEADS KV_HEADS THREADS REPS
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "rocket_attn.h"

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
    return (_Float16)((float)((*s >> 40) / (double)(1u << 24)) - 0.5f);
}

int main(int argc, char **argv)
{
    int T = argc > 1 ? atoi(argv[1]) : 2587;
    int KV = argc > 2 ? atoi(argv[2]) : T;
    int H = argc > 3 ? atoi(argv[3]) : 8;
    int HKV = argc > 4 ? atoi(argv[4]) : H;
    int threads = argc > 5 ? atoi(argv[5]) : 4;
    int reps = argc > 6 ? atoi(argv[6]) : 3;
    const int dk = 128, dv = 128;
    float scale = 0.08838834764831845f; /* 1/sqrt(128) */

    _Float16 *Q = malloc((size_t)H * T * dk * sizeof(*Q));
    _Float16 *K = malloc((size_t)HKV * KV * dk * sizeof(*K));
    _Float16 *V = malloc((size_t)HKV * dv * KV * sizeof(*V));
    _Float16 *mask = malloc((size_t)T * KV * sizeof(*mask));
    _Float16 *out = calloc((size_t)H * T * dv, sizeof(*out));
    if (!Q || !K || !V || !mask || !out) return 2;
    uint64_t seed = 0x123456789abcdef1ull;
    for (size_t i = 0; i < (size_t)H * T * dk; i++) Q[i] = rnd(&seed);
    for (size_t i = 0; i < (size_t)HKV * KV * dk; i++) K[i] = rnd(&seed);
    for (size_t i = 0; i < (size_t)HKV * dv * KV; i++) V[i] = rnd(&seed);
    for (int i = 0; i < T; i++)
        for (int j = 0; j < KV; j++)
            mask[(size_t)i * KV + j] = j > i ? (_Float16)-INFINITY : (_Float16)0;

    rocket_fa_ctx *ctx = rocket_fa_ctx_create(threads);
    if (!ctx) { fprintf(stderr, "ctx create failed\n"); return 1; }
    int rc = rocket_flash_attn_fp16_ctx(ctx, T, KV, dk, dv, H, HKV, scale, 0.0f,
                                        Q, K, V, mask, out);
    if (rc) { fprintf(stderr, "attn failed %d\n", rc); return 1; }
    double t = now_ms();
    for (int i = 0; i < reps; i++)
        rc = rocket_flash_attn_fp16_ctx(ctx, T, KV, dk, dv, H, HKV, scale, 0.0f,
                                        Q, K, V, mask, out);
    double ms = (now_ms() - t) / reps;
    printf("C attn T=%d KV=%d H=%d HKV=%d threads=%d: %.2f ms rc=%d\n",
           T, KV, H, HKV, threads, ms, rc);
    rocket_fa_ctx_free(ctx);
    free(Q); free(K); free(V); free(mask); free(out);
    return 0;
}
