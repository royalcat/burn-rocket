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
 *   oracle conv         IC IH IW OC KH KW SY SX PT PL DY DX DEPTHWISE [OH OW]
 *   oracle emit-conv    IC IH IW OC KH KW SY SX PT PL DY DX DEPTHWISE FP32TOFP16 IN WT OUT [DSY DSX]
 *   oracle emit-conv-act IC IH IW OC KH KW SY SX PT PL DY DX DEPTHWISE KIND IN WT OUT
 *   oracle emit-conv-int8 IC IH IW OC KH KW SY SX PT PL DY DX IN WT OUT [DW_GROUP]
 *   oracle emit-conv-dw-int8 IC IH IW OC KH KW SY SX PT PL DY DX IN WT OUT BIAS \
 *        I_SCALE_BITS W_SCALE_BITS O_SCALE_BITS IN_ZP W_ZP OUT_ZP INT8_OUT
 *
 * A plan prints "Mt Kt Nt jobs", or "-1" for an unsupported shape.
 * An emit prints the task count, then one %016llx per NPUOP word.
 * The emitters take the TILE dims (Mt Kt Nt), not the full problem dims.
 *
 * `conv` prints `rocket_conv2d_plan`'s raw code (0 or the negated reason). `emit-conv`
 * runs `gen_conv2d_fp16` / `gen_conv2d_dw_fp16` (DEPTHWISE selects the latter) with the
 * optional deconvolution interior-dilation strides DSY/DSX (1 = off) and prints the word
 * count + program, or "-1" if the generator refuses. `emit-conv-act` fuses a smooth
 * activation (KIND is the ROCKET_ACTIVATION_* value) through
 * `rocket_lut_epilogue_build`, which is what makes that program ~1200 words, not ~130.
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
#include "rocket_activation.h"
#include "rocket_conv.h"
#include "rocket_matmul.h"
#include "rocket_reduce.h"

/* One regcmd program is <= RC_STRIDE (128) words for a matmul; the conv->activation
 * fusion appends the two 513-word table uploads and lands near 1200. 2048 covers it. */
#define ORACLE_MAX_OPS 2048

static void usage(const char *argv0)
{
    fprintf(stderr,
            "usage: %s plan|plan-int8|plan-int8-gw|plan-int4|plan-tf32|plan-bf16 M K N [GROUP]\n"
            "       %s emit|emit-int8|emit-int4|emit-tf32|emit-bf16 M K N IN WT OUT "
            "[FP32TOFP16] [ACCUMULATE] [ADD_DMA]\n"
            "       %s recip K\n"
            "       %s pool C H W\n"
            "       %s emit-pool C IH IW OH OW KH KW SY SX PT PL PB PR METHOD RW RH IN OUT\n"
            "       %s conv IC IH IW OC KH KW SY SX PT PL DY DX DEPTHWISE [OH OW]\n"
            "       %s emit-conv IC IH IW OC KH KW SY SX PT PL DY DX DEPTHWISE FP32TOFP16 "
            "IN WT OUT [DSY DSX]\n"
            "       %s emit-conv-act IC IH IW OC KH KW SY SX PT PL DY DX DEPTHWISE KIND "
            "IN WT OUT\n"
            "       %s emit-conv-int8 IC IH IW OC KH KW SY SX PT PL DY DX IN WT OUT "
            "[DW_GROUP]\n"
            "       %s emit-conv-dw-int8 IC IH IW OC KH KW SY SX PT PL DY DX IN WT OUT BIAS "
            "I_SCALE_BITS W_SCALE_BITS O_SCALE_BITS IN_ZP W_ZP OUT_ZP INT8_OUT\n",
            argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0);
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

    if (!strcmp(mode, "emit-conv-int8") || !strcmp(mode, "emit-conv-dw-int8")) {
        /* The int8 family: the emitters take the job's tile dims, as emit-conv does.
         * Scale arguments are raw f32 BIT PATTERNS (hex) so the two harnesses read the
         * identical value rather than reparsing a decimal. The zero points are
         * MODEL-domain signed int8; the library adds 0x80 before calling the generator,
         * and so does this oracle. */
        if (argc < 17) {
            fprintf(stderr, "emit-conv-int8 needs IC IH IW OC KH KW SY SX PT PL DY DX IN WT OUT\n");
            return 2;
        }
        int dw = !strcmp(mode, "emit-conv-dw-int8");
        conv_params_t p = {0};
        /* The depthwise invariant OC == IC; a 0 in that argument means "the entry's
         * own C" (the library passes p.oc = C). */
        uint64_t ops[ORACLE_MAX_OPS] = {0};
        p.tasks = ops;
        p.ic = (uint16_t)atoi(argv[2]);
        p.ih = (uint16_t)atoi(argv[3]);
        p.iw = (uint16_t)atoi(argv[4]);
        p.oc = (uint16_t)atoi(argv[5]);
        if (dw && p.oc == 0) p.oc = p.ic;
        p.kh = (uint16_t)atoi(argv[6]);
        p.kw = (uint16_t)atoi(argv[7]);
        p.oh = (uint16_t)rocket_conv_out_dim(atoi(argv[3]), atoi(argv[6]), atoi(argv[8]),
                                             atoi(argv[10]), atoi(argv[12]));
        p.ow = (uint16_t)rocket_conv_out_dim(atoi(argv[4]), atoi(argv[7]), atoi(argv[9]),
                                             atoi(argv[11]), atoi(argv[13]));
        p.stride_y = (uint8_t)atoi(argv[8]);
        p.stride_x = (uint8_t)atoi(argv[9]);
        p.pad_top = (uint8_t)atoi(argv[10]);
        p.pad_left = (uint8_t)atoi(argv[11]);
        p.dil_y = (uint8_t)atoi(argv[12]);
        p.dil_x = (uint8_t)atoi(argv[13]);
        p.input_dma = (uint32_t)strtoul(argv[14], NULL, 0);
        p.weights_dma = (uint32_t)strtoul(argv[15], NULL, 0);
        p.output_dma = (uint32_t)strtoul(argv[16], NULL, 0);
        int rc;
        if (!dw) {
            p.dw_group = argc > 17 ? (uint8_t)atoi(argv[17]) : 0;
            rc = gen_conv2d_int8(&p);
        } else {
            if (argc < 24) {
                fprintf(stderr, "emit-conv-dw-int8 needs BIAS I W O SCALE_BITS IN_ZP W_ZP "
                                "OUT_ZP INT8_OUT\n");
                return 2;
            }
            union { uint32_t u; float f; } cv;
            p.bias_dma = (uint32_t)strtoul(argv[17], NULL, 0);
            cv.u = (uint32_t)strtoul(argv[18], NULL, 16); p.in_scale = cv.f;
            cv.u = (uint32_t)strtoul(argv[19], NULL, 16); p.w_scale = cv.f;
            cv.u = (uint32_t)strtoul(argv[20], NULL, 16); p.out_scale = cv.f;
            p.input_zero_point = atoi(argv[21]) + 0x80;
            p.output_zero_point = atoi(argv[22]) + 0x80;
            p.weight_zero_point = atoi(argv[23]) + 0x80;
            p.int8_out = argc > 24 ? (uint8_t)(atoi(argv[24]) != 0) : 1;
            if (argc > 25) p.dw_group = (uint8_t)atoi(argv[25]);
            rc = gen_conv2d_dw_int8(&p);
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

    if (!strcmp(mode, "conv") || !strcmp(mode, "emit-conv") ||
        !strcmp(mode, "emit-conv-act")) {
        if (argc < 15) {
            fprintf(stderr, "conv needs IC IH IW OC KH KW SY SX PT PL DY DX DEPTHWISE\n");
            return 2;
        }
        rocket_conv2d_desc d = {0};
        d.ic = atoi(argv[2]);
        d.ih = atoi(argv[3]);
        d.iw = atoi(argv[4]);
        d.oc = atoi(argv[5]);
        d.kh = atoi(argv[6]);
        d.kw = atoi(argv[7]);
        d.stride_y = atoi(argv[8]);
        d.stride_x = atoi(argv[9]);
        d.pad_top = atoi(argv[10]);
        d.pad_left = atoi(argv[11]);
        d.dil_y = atoi(argv[12]);
        d.dil_x = atoi(argv[13]);
        d.depthwise = atoi(argv[14]);
        if (!strcmp(mode, "conv")) {
            if (argc > 15) d.oh = atoi(argv[15]);
            if (argc > 16) d.ow = atoi(argv[16]);
            printf("%d\n", rocket_conv2d_plan(&d));
            return 0;
        }
        if (argc < 19) {
            fprintf(stderr, "emit-conv needs FP32TOFP16|KIND IN WT OUT\n");
            return 2;
        }
        conv_params_t p = {0};
        p.ic = (uint16_t)d.ic;
        p.ih = (uint16_t)d.ih;
        p.iw = (uint16_t)d.iw;
        p.oc = (uint16_t)d.oc;
        p.oh = (uint16_t)rocket_conv2d_oh(&d);
        p.ow = (uint16_t)rocket_conv2d_ow(&d);
        p.kh = (uint16_t)d.kh;
        p.kw = (uint16_t)d.kw;
        p.stride_y = (uint8_t)d.stride_y;
        p.stride_x = (uint8_t)d.stride_x;
        p.dil_y = (uint8_t)d.dil_y;
        p.dil_x = (uint8_t)d.dil_x;
        p.pad_top = (uint8_t)d.pad_top;
        p.pad_left = (uint8_t)d.pad_left;
        p.input_dma = (uint32_t)strtoul(argv[16], NULL, 0);
        p.weights_dma = (uint32_t)strtoul(argv[17], NULL, 0);
        p.output_dma = (uint32_t)strtoul(argv[18], NULL, 0);
        p.deconv_sy = argc > 19 ? (uint8_t)atoi(argv[19]) : 1;
        p.deconv_sx = argc > 20 ? (uint8_t)atoi(argv[20]) : 1;
        uint64_t ops[ORACLE_MAX_OPS] = {0};
        p.tasks = ops;
        int rc;
        if (!strcmp(mode, "emit-conv-act")) {
            uint16_t lut[1026];
            lut_epilogue_t ep;
            int b = rocket_lut_epilogue_build(atoi(argv[15]), lut, &ep);
            if (b < 0) {
                printf("-1\n");
                return 0;
            }
            p.fp32tofp16 = 1;
            p.act = &ep;
            rc = d.depthwise ? gen_conv2d_dw_fp16(&p) : gen_conv2d_fp16(&p);
        } else {
            p.fp32tofp16 = (uint8_t)(atoi(argv[15]) != 0);
            rc = d.depthwise ? gen_conv2d_dw_fp16(&p) : gen_conv2d_fp16(&p);
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
