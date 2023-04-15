#ifndef _CUDA_KERNELS_H
#define _CUDA_KERNELS_H

#include "aspen.h"

#ifdef GPU
void cuda_matmul (const unsigned int M, const unsigned int N, const unsigned int K,
		 const float *A, const unsigned int lda, const float *B, const unsigned int ldb, float *C, const unsigned int ldc,
         const float *Bias, LAYER_ACT activation_type);
#endif

#endif // _CUDA_KERNELS_H