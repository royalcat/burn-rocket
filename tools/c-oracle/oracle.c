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
 *   oracle recip        K
 *   oracle pool         C H W
 *   oracle emit-pool    C IH IW OH OW KH KW SY SX PT PL PB PR METHOD RW RH IN_DMA OUT_DMA
 *
 * A plan prints "Mt Kt Nt jobs", or "-1" for an unsupported shape.
 * An emit prints the task count, then one %016llx per NPUOP word.
 * The emitters take the TILE dims (Mt Kt Nt), not the full problem dims.
 *
 * `recip` prints the fp16 average-pool reciprocal bits of K (the C
 * `ppu_recip_kernel_fp16`, the PPU's only division). `pool` prints the spatial-reduce
 * decomposition for one (C, H, W): the `rocket_global_avgpool_plan` reason code, then
 * one `H <n> f...` / `W <n> f...` line with `rocket_reduce_factor_axis`'s factors (an
 * axis that is not 16-smooth prints `-1` for its count). `emit-pool` prints the word
 * count and the program of `gen_pool_fp16`, or "-1" if the generator refuses the params
 * — METHOD is 0 avg / 1 max / 2 min, and RW/RH are the raw reciprocal fields (0 for
 * max/min; see `recip`).
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "npu_matmul.h"
#include "npu_pool.h"
#include "rocket_matmul.h"
#include "rocket_reduce.h"

/* One regcmd program is <= RC_STRIDE (128) words; 512 leaves headroom. */
#define ORACLE_MAX_OPS 512

static void usage(const char *argv0)
{
    fprintf(stderr,
            "usage: %s plan|plan-int8|plan-int8-gw|plan-int4|plan-tf32|plan-bf16 M K N [GROUP]\n"
            "       %s emit|emit-int8|emit-int4|emit-tf32|emit-bf16 M K N IN WT OUT "
            "[FP32TOFP16] [ACCUMULATE] [ADD_DMA]\n"
            "       %s recip K\n"
            "       %s pool C H W\n"
            "       %s emit-pool C IH IW OH OW KH KW SY SX PT PL PB PR METHOD RW RH IN OUT\n",
            argv0, argv0, argv0, argv0, argv0);
}

static int is_plan_mode(const char *mode)
{
    return !strcmp(mode, "plan") || !strcmp(mode, "plan-int8") ||
           !strcmp(mode, "plan-int8-gw") || !strcmp(mode, "plan-int4") ||
           !strcmp(mode, "plan-tf32") || !strcmp(mode, "plan-bf16");
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        usage(argv[0]);
        return 2;
    }
    const char *mode = argv[1];
    int M = atoi(argv[2]);

    if (!strcmp(mode, "recip")) {
        printf("%08x\n", ppu_recip_kernel_fp16(M));
        return 0;
    }

    if (!strcmp(mode, "pool")) {
        if (argc < 5) {
            fprintf(stderr, "pool needs C H W\n");
            return 2;
        }
        int C = atoi(argv[2]), H = atoi(argv[3]), W = atoi(argv[4]);
        int fh[ROCKET_REDUCE_MAX_DIM > 64 ? 64 : ROCKET_REDUCE_MAX_DIM];
        int fw[ROCKET_REDUCE_MAX_DIM > 64 ? 64 : ROCKET_REDUCE_MAX_DIM];
        printf("%d\n", rocket_global_avgpool_plan(C, H, W));
        int nh = rocket_reduce_factor_axis(H, fh, 64);
        printf("H %d", nh);
        for (int i = 0; i < nh; i++) printf(" %d", fh[i]);
        printf("\n");
        int nw = rocket_reduce_factor_axis(W, fw, 64);
        printf("W %d", nw);
        for (int i = 0; i < nw; i++) printf(" %d", fw[i]);
        printf("\n");
        return 0;
    }

    if (!strcmp(mode, "emit-pool")) {
        if (argc < 20) {
            fprintf(stderr,
                    "emit-pool needs C IH IW OH OW KH KW SY SX PT PL PB PR METHOD RW RH IN OUT\n");
            return 2;
        }
        pool_params_t p = {0};
        p.c = (uint16_t)atoi(argv[2]);
        p.ih = (uint16_t)atoi(argv[3]);
        p.iw = (uint16_t)atoi(argv[4]);
        p.oh = (uint16_t)atoi(argv[5]);
        p.ow = (uint16_t)atoi(argv[6]);
        p.kh = (uint8_t)atoi(argv[7]);
        p.kw = (uint8_t)atoi(argv[8]);
        p.stride_y = (uint8_t)atoi(argv[9]);
        p.stride_x = (uint8_t)atoi(argv[10]);
        p.pad_top = (uint8_t)atoi(argv[11]);
        p.pad_left = (uint8_t)atoi(argv[12]);
        p.pad_bottom = (uint8_t)atoi(argv[13]);
        p.pad_right = (uint8_t)atoi(argv[14]);
        p.method = (uint8_t)atoi(argv[15]);
        p.recip_w = (uint32_t)strtoul(argv[16], NULL, 0);
        p.recip_h = (uint32_t)strtoul(argv[17], NULL, 0);
        p.input_dma = (uint32_t)strtoul(argv[18], NULL, 0);
        p.output_dma = (uint32_t)strtoul(argv[19], NULL, 0);

        uint64_t ops[ORACLE_MAX_OPS] = {0};
        p.tasks = ops;
        int rc = gen_pool_fp16(&p);
        if (rc < 0) {
            /* The C's local rejection class (-1/-2/-3) does not line up with the port's
             * Status enum, so the harness compares the portable "-1", as the matmul
             * emits do. Valid programs are compared word for word. */
            printf("-1\n");
            return 0;
        }
        printf("%u\n", p.task_count);
        for (uint32_t i = 0; i < p.task_count; i++)
            printf("%016llx\n", (unsigned long long)ops[i]);
        return 0;
    }

    if (argc < 5) {
        usage(argv[0]);
        return 2;
    }
    int K = atoi(argv[3]), N = atoi(argv[4]);

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
