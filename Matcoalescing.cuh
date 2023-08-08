

#ifndef _MATRIXMUL_COALESCING_H_
#define _MATRIXMUL_COALESCING_H_

#include <stdio.h>
#include "matrixMul.h"

#define CHECK_BANK_CONFLICTS 0
#if CHECK_BANK_CONFLICTS
#define AS(i, j) cutilBankChecker(((float*)&As[0][0]), (BLOCK_SIZE * i + j))
#define BS(i, j) cutilBankChecker(((float*)&Bs[0][0]), (BLOCK_SIZE * i + j))
#else
#define AS(i, j) As[i][j]
#define BS(i, j) Bs[i][j]
#endif


__global__ void
matrixMul_coalescing( float* C, float* A, float* B, int wA, int wB)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;


    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];


    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int aBegin = wA * BLOCK_SIZE * by;


    int aEnd   = aBegin + wA - 1;

    int aStep  = BLOCK_SIZE;


    int bBegin = BLOCK_SIZE * bx;


    int bStep  = BLOCK_SIZE * wB;

    float Csub = 0;


    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) {


        AS(ty, tx) = A[a + wA * ty + tx];
        BS(tx, ty) = B[b + wB * ty + tx];

        __syncthreads();


        for (int k = 0; k < BLOCK_SIZE; ++k)
	  Csub += AS(ty, k) * BS(tx, k);


        __syncthreads();
    }


    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

#endif
