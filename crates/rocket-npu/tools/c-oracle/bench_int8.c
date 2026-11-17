// SPDX-License-Identifier: GPL-3.0-or-later
/* bench_int8 M K N GROUP THREADS REPS */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "rocket_npu.h"
#include "rocket_matmul.h"
static double now_ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1000.0+t.tv_nsec/1e6;}
static uint64_t s=0x123456789abcdef1ull;
static int8_t r8(void){s^=s<<13;s^=s>>7;s^=s<<17;return (int8_t)((s>>48)&0xff);}
static float rf(void){s^=s<<13;s^=s>>7;s^=s<<17;return ((s>>40)/(double)(1u<<24))-0.5;}
int main(int argc,char**argv){
 int M=argc>1?atoi(argv[1]):512,K=argc>2?atoi(argv[2]):3072,N=argc>3?atoi(argv[3]):1024;
 int G=argc>4?atoi(argv[4]):64,T=argc>5?atoi(argv[5]):5,R=argc>6?atoi(argv[6]):2;
 int ng=K/G;
 int8_t*A=malloc((size_t)M*K),*B=malloc((size_t)N*K);
 float*as=malloc((size_t)M*ng*4),*bs=malloc((size_t)N*ng*4),*C=malloc((size_t)M*N*4);
 for(size_t i=0;i<(size_t)M*K;i++)A[i]=r8();
 for(size_t i=0;i<(size_t)N*K;i++)B[i]=r8();
 for(size_t i=0;i<(size_t)M*ng;i++)as[i]=rf()*0.01f;
 for(size_t i=0;i<(size_t)N*ng;i++)bs[i]=rf()*0.01f;
 rocket_i8_ctx*ctx=rocket_i8_ctx_create(T);
 if(!ctx){fprintf(stderr,"ctx\n");return 1;}
 rocket_i8_weights*w=rocket_i8_weights_pack_gw(ctx,256,K,N,B,G);
 if(!w){fprintf(stderr,"pack\n");return 1;}
 int rc=rocket_matmul_int8_prepacked_gw(ctx,M,K,N,A,as,bs,C,w);
 double t=now_ms();
 for(int i=0;i<R;i++)rc=rocket_matmul_int8_prepacked_gw(ctx,M,K,N,A,as,bs,C,w);
 double ms=(now_ms()-t)/R;
 printf("C int8 M=%d K=%d N=%d G=%d T=%d: %.2f ms rc=%d\n",M,K,N,G,T,ms,rc);
 rocket_i8_weights_free(ctx,w);rocket_i8_ctx_free(ctx);
 free(A);free(B);free(as);free(bs);free(C);return 0;
}
