#ifndef _CUDA_KERNELS_H
#define _CUDA_KERNELS_H

#include "aspen.h"

#define _BLOCK_RESIDUAL_SIZE 128
#define _BLOCK_LAYERNORM_SIZE 128
#define _BLOCK_K_SIZE 32
#define _BLOCK_M_SIZE 64
#define _BLOCK_N_SIZE 64
#define _THREAD_M_SIZE 8 // MUST be same to _VEC_SIZE_M
#define _THREAD_N_SIZE 4
#define _THREAD_NUM ((_BLOCK_M_SIZE / _THREAD_M_SIZE) * (_BLOCK_N_SIZE / _THREAD_N_SIZE)) // 128
#define _CACHE_A_K_PER_LOAD (_THREAD_NUM / _BLOCK_M_SIZE) // 2
#define _CACHE_B_K_PER_LOAD (_THREAD_NUM / _BLOCK_N_SIZE) // 2

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