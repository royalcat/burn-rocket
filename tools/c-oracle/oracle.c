// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * oracle.c — the C library's planner and fp16 regcmd emitter, for parity checks
 * against the Rust port. Host-only: it never opens the NPU device.
 *
 * Build:
 *   tools/c-oracle/build.sh <rocket-userspace-src> <host-install-dir> <out>
 *
 * Usage:
 *   oracle plan M K N
 *   oracle emit M K N IN_DMA WT_DMA OUT_DMA [FP32TOFP16] [ACCUMULATE] [ADD_DMA]
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "npu_matmul.h"
#include "rocket_matmul.h"

int main(int argc, char **argv)
{
    if (argc < 5) {
        fprintf(stderr, "usage: %s plan|emit M K N ...\n", argv[0]);
        return 2;
    }
    int M = atoi(argv[2]), K = atoi(argv[3]), N = atoi(argv[4]);
    if (argv[1][0] == 'p') {
        int Mt = 0, Kt = 0, Nt = 0;
        int jobs;
        if (!strcmp(argv[1], "plan-int8"))
            jobs = rocket_matmul_plan_int8_gw(M, K, N, argc > 5 ? atoi(argv[5]) : 0, &Mt, &Kt, &Nt);
        else if (!strcmp(argv[1], "plan-bf16"))
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
    const int is_i8 = !strcmp(argv[1], "emit-int8");
    const int is_bf = !strcmp(argv[1], "emit-bf16");
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
    uint64_t ops[512] = {0};
    p.tasks = ops;
    int rc = is_i8 ? gen_matmul_int8(&p) : (is_bf ? gen_matmul_bf16(&p) : gen_matmul_fp16(&p));
    if (rc < 0) {
        printf("-1\n");
        return 0;
    }
    printf("%u\n", p.task_count);
    for (uint32_t i = 0; i < p.task_count; i++)
        printf("%016llx\n", (unsigned long long)ops[i]);
    return 0;
}
