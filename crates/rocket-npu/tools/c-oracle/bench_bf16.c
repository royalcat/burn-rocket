// SPDX-License-Identifier: GPL-3.0-or-later
/* bench_bf16 M K N THREADS REPS */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "rocket_npu.h"
#include "rocket_matmul.h"
static double now_ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1000.0+t.tv_nsec/1e6;}
static uint64_t s=0x123456789abcdef1ull;
static float rf(void){s^=s<<13;s^=s>>7;s^=s<<17;return ((s>>40)/(double)(1u<<24))-0.5;}
int main(int argc,char**argv){
 int M=argc>1?atoi(argv[1]):512,K=argc>2?atoi(argv[2]):3072,N=argc>3?atoi(argv[3]):1024;
 int T=argc>4?atoi(argv[4]):5,R=argc>5?atoi(argv[5]):2;
 float*A=malloc((size_t)M*K*4),*B=malloc((size_t)N*K*4),*C=malloc((size_t)M*N*4);
 for(size_t i=0;i<(size_t)M*K;i++)A[i]=rf();
 for(size_t i=0;i<(size_t)N*K;i++)B[i]=rf();
 rocket_bf16_stream*st=rocket_bf16_stream_create(T);
 if(!st){fprintf(stderr,"ctx\n");return 1;}
 int rc=rocket_matmul_bf16_stream(st,M,K,N,A,B,C);
 double t=now_ms();
 for(int i=0;i<R;i++)rc=rocket_matmul_bf16_stream(st,M,K,N,A,B,C);
 double ms=(now_ms()-t)/R;
 printf("C bf16 M=%d K=%d N=%d T=%d: %.2f ms rc=%d\n",M,K,N,T,ms,rc);
 rocket_bf16_stream_free(st);free(A);free(B);free(C);return 0;
}
