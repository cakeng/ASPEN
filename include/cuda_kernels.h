#ifndef _CUDA_KERNELS_H
#define _CUDA_KERNELS_H

#include "aspen.h"

#ifdef GPU
void cuda_matmul (const unsigned int M, const unsigned int N, const unsigned int K,
		 const float *A, const unsigned int lda, const float *B, const unsigned int ldb, float *C, const unsigned int ldc,
         const float *Bias, LAYER_ACT activation_type, cudaStream_t stream);

void cuda_layernorm (const float *input, const float *kernel, const float *bias, 
    float *output, unsigned int N, unsigned int M, cudaStream_t stream);

void cuda_k_attention (const float *input_1, const float *input_2, float *output, unsigned int batch_size
    , unsigned int num_heads, unsigned int num_hidden, unsigned int num_seq, cudaStream_t stream);

void cuda_v_attention (const float *input_1, const float *input_2, float *output, unsigned int batch_size
    , unsigned int num_heads, unsigned int num_hidden, unsigned int num_seq, cudaStream_t stream);

void cuda_residual (const float *input_1, const float *input_2, float *output, unsigned int num_elements
    , cudaStream_t stream);

#endif

#endif // _CUDA_KERNELS_H