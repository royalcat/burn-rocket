// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * oracle.c — the C library's planners and regcmd emitters for every RK3588
 * matmul dtype, for parity checks against the Rust port. Host-only: it never
 * opens the NPU device.
 *
 * Build:
 *   tools/c-oracle/build.sh <rocket-userspace-src> <host-install-dir> <out>
 *
 * Usage:
 *   oracle plan         M K N
 *   oracle plan-int8    M K N
 *   oracle plan-int8-gw M K N [GROUP]
 *   oracle plan-int4    M K N
 *   oracle plan-tf32    M K N
 *   oracle plan-bf16    M K N
 *   oracle emit         M K N IN_DMA WT_DMA OUT_DMA [FP32TOFP16] [ACCUMULATE] [ADD_DMA]
 *   oracle emit-int8    M K N IN_DMA WT_DMA OUT_DMA [FP32TOFP16] [ACCUMULATE] [ADD_DMA]
 *   oracle emit-int4    M K N IN_DMA WT_DMA OUT_DMA [FP32TOFP16] [ACCUMULATE] [ADD_DMA]
 *   oracle emit-tf32    M K N IN_DMA WT_DMA OUT_DMA [FP32TOFP16] [ACCUMULATE] [ADD_DMA]
 *   oracle emit-bf16    M K N IN_DMA WT_DMA OUT_DMA [FP32TOFP16] [ACCUMULATE] [ADD_DMA]
 *
 * A plan prints "Mt Kt Nt jobs", or "-1" for an unsupported shape.
 * An emit prints the task count, then one %016llx per NPUOP word.
 * The emitters take the TILE dims (Mt Kt Nt), not the full problem dims.
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "npu_matmul.h"
#include "rocket_matmul.h"

/* One regcmd program is <= RC_STRIDE (128) words; 512 leaves headroom. */
#define ORACLE_MAX_OPS 512

static void usage(const char *argv0)
{
    fprintf(stderr,
            "usage: %s plan|plan-int8|plan-int8-gw|plan-int4|plan-tf32|plan-bf16 M K N [GROUP]\n"
            "       %s emit|emit-int8|emit-int4|emit-tf32|emit-bf16 M K N IN WT OUT "
            "[FP32TOFP16] [ACCUMULATE] [ADD_DMA]\n",
            argv0, argv0);
}

static int is_plan_mode(const char *mode)
{
    return !strcmp(mode, "plan") || !strcmp(mode, "plan-int8") ||
           !strcmp(mode, "plan-int8-gw") || !strcmp(mode, "plan-int4") ||
           !strcmp(mode, "plan-tf32") || !strcmp(mode, "plan-bf16");
}

int main(int argc, char **argv)
{
    if (argc < 5) {
        usage(argv[0]);
        return 2;
    }
    const char *mode = argv[1];
    int M = atoi(argv[2]), K = atoi(argv[3]), N = atoi(argv[4]);

    if (is_plan_mode(mode)) {
        int Mt = 0, Kt = 0, Nt = 0;
        int jobs;
        if (!strcmp(mode, "plan-int8"))
            jobs = rocket_matmul_plan_int8(M, K, N, &Mt, &Kt, &Nt);
        else if (!strcmp(mode, "plan-int8-gw"))
            jobs = rocket_matmul_plan_int8_gw(M, K, N, argc > 5 ? atoi(argv[5]) : 0,
                                              &Mt, &Kt, &Nt);
        else if (!strcmp(mode, "plan-int4"))
            jobs = rocket_matmul_plan_int4(M, K, N, &Mt, &Kt, &Nt);
        else if (!strcmp(mode, "plan-tf32"))
            jobs = rocket_matmul_plan_tf32(M, K, N, &Mt, &Kt, &Nt);
        else if (!strcmp(mode, "plan-bf16"))
            jobs = rocket_matmul_plan_bf16(M, K, N, &Mt, &Kt, &Nt);
        else
            jobs = rocket_matmul_plan(M, K, N, &Mt, &Kt, &Nt);
        if (jobs < 0) {
            printf("-1\n");
        } else {
            printf("%d %d %d %d\n", Mt, Kt, Nt, jobs);
        }
        return 0;
    }

    if (argc < 8) {
        fprintf(stderr, "emit needs M K N IN WT OUT\n");
        return 2;
    }
    matmul_params_t p = {0};
    p.m = (uint16_t)M;
    p.k = (uint16_t)K;
    p.n = (uint16_t)N;
    p.input_dma = (uint32_t)strtoul(argv[5], NULL, 0);
    p.weights_dma = (uint32_t)strtoul(argv[6], NULL, 0);
    p.output_dma = (uint32_t)strtoul(argv[7], NULL, 0);
    p.fp32tofp16 = argc > 8 ? (uint8_t)(atoi(argv[8]) != 0) : 1;
    p.accumulate = argc > 9 ? (uint8_t)(atoi(argv[9]) != 0) : 0;
    p.add_dma = argc > 10 ? (uint32_t)strtoul(argv[10], NULL, 0) : 0;

    uint64_t ops[ORACLE_MAX_OPS] = {0};
    p.tasks = ops;
    int rc;
    if (!strcmp(mode, "emit-int8"))
        rc = gen_matmul_int8(&p);
    else if (!strcmp(mode, "emit-int4"))
        rc = gen_matmul_int4(&p);
    else if (!strcmp(mode, "emit-tf32"))
        rc = gen_matmul_tf32(&p);
    else if (!strcmp(mode, "emit-bf16"))
        rc = gen_matmul_bf16(&p);
    else if (!strcmp(mode, "emit"))
        rc = gen_matmul_fp16(&p);
    else {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }
    if (rc < 0) {
        printf("-1\n");
        return 0;
    }
    printf("%u\n", p.task_count);
    for (uint32_t i = 0; i < p.task_count; i++)
        printf("%016llx\n", (unsigned long long)ops[i]);
    return 0;
}
