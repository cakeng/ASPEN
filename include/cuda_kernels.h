#ifndef _CUDA_KERNELS_H
#define _CUDA_KERNELS_H

#include "aspen.h"

#define _BLOCK_ATT_K_PROB_SIZE 128
#define _BLOCK_RESIDUAL_SIZE 128
#define _BLOCK_TILED_RESIDUAL_SIZE 16
#define _BLOCK_LAYERNORM_SIZE 128
#define _A_MIN_DIM 8
#define _BLOCK_K_SIZE 8
#define _BLOCK_M_SIZE 64
#define _BLOCK_N_SIZE 128
#define _THREAD_M_SIZE 8 // MUST be same to _VEC_SIZE_M
#define _THREAD_N_SIZE 8
#define _THREAD_NUM ((_BLOCK_M_SIZE / _THREAD_M_SIZE) * (_BLOCK_N_SIZE / _THREAD_N_SIZE)) // 128
#define _CACHE_A_K_PER_LOAD (_THREAD_NUM / _BLOCK_M_SIZE) // 2
#define _CACHE_B_K_PER_LOAD (_THREAD_NUM / _BLOCK_N_SIZE) // 2

#ifdef GPU
__global__ void cuda_conv2d_kernel(
    const unsigned int M, const unsigned int N, 
    const int *col_idx_arr, const unsigned int col_per_n, const unsigned int K_col,
    const float *A, const unsigned int lda, const float *B, const unsigned int ldb, float *C, const unsigned int ldc,
    const float *Bias, LAYER_ACT activation_type);

__global__ void cuda_maxpool_kernel(
    const unsigned int M, const unsigned int N, const unsigned int K_pos,
    const int *col_idx_arr, const unsigned int col_per_n,
    const float *B, const unsigned int ldb, float *C, const unsigned int ldc,
    LAYER_ACT activation_type);

__global__ void cuda_avgpool_kernel(
    const unsigned int M, const unsigned int N, const unsigned int K_pos,
    const int *col_idx_arr, const unsigned int col_per_n,
    const float *B, const unsigned int ldb, float *C, const unsigned int ldc,
    LAYER_ACT activation_type);

__global__ void cuda_matmul_kernel(const unsigned int M, const unsigned int N, const unsigned int K,
    const float *A, const unsigned int lda, const float *B, const unsigned int ldb, float *C, const unsigned int ldc,
    const float *Bias, LAYER_ACT activation_type);

__global__ void cuda_k_attention_kernel(const unsigned int num_heads, const unsigned int num_hidden, const unsigned int num_seq,
    const unsigned int M, const unsigned int N, const unsigned int K,
    const float *key, const unsigned int ldk, const float *B, const unsigned int ldb, float *C, const unsigned int ldc);

__global__ void cuda_k_attention_prob_kernel(const unsigned int num_heads, const unsigned int num_hidden, const unsigned int num_seq,
    const unsigned int M, const unsigned int N, const unsigned int K,
    const float *key, const unsigned int ldk, const float *B, const unsigned int ldb, float *C, const unsigned int ldc);

__global__ void cuda_v_attention_kernel(const unsigned int num_heads, const unsigned int num_hidden, const unsigned int num_seq,
    const unsigned int M, const unsigned int N, const unsigned int K,
    const float *value, const unsigned int ldv, const float *B, const unsigned int ldb, float *C, const unsigned int ldc);

__global__ void cuda_residual_kernel (const unsigned int num_elements, const float *A, const float *B, float *C, LAYER_ACT activation_type);

__global__ void cuda_layernorm_kernel(const float *input, const float *weight, const float *bias, 
    float *output, unsigned int N, unsigned int M, unsigned int ldb, unsigned int ldc);

__global__ void cuda_tiled_conv2d_kernel(
    const unsigned int M, const unsigned int N, 
    const float **col_ptr_arr, const unsigned int col_per_n, const unsigned int K_col,
    const float *A, const unsigned int lda, const float *B, const unsigned int ldb, float *C, const unsigned int ldc,
    const float *Bias, LAYER_ACT activation_type);

__global__ void cuda_tiled_maxpool_kernel(
    const unsigned int M, const unsigned int N, 
    const float **col_ptr_arr, const unsigned int col_per_n,
    float *C, const unsigned int ldc,
    LAYER_ACT activation_type);

__global__ void cuda_tiled_avgpool_kernel(
    const unsigned int M, const unsigned int N, 
    const float **col_ptr_arr, const unsigned int col_per_n,
    float *C, const unsigned int ldc,
    LAYER_ACT activation_type);

__global__ void cuda_tiled_k_matmul_kernel(
    const unsigned int M, const unsigned int N, const unsigned int K,
    const float *key_head, const unsigned int ldk, const float *B_head, const unsigned int ldb, float *C_head, const unsigned int ldc);

__global__ void cuda_tiled_k_prob_kernel(
    const unsigned int M, const unsigned int N, const unsigned int K,
    const float *key_head, const unsigned int ldk, const float *B_head, const unsigned int ldb, float *C_head, const unsigned int ldc);

__global__ void cuda_tiled_v_attention_kernel(const unsigned int M, const unsigned int N, const unsigned int K,
    const float *val_head, const unsigned int ldv, const float *B_head, const unsigned int ldb, float *C_head, const unsigned int ldc);

__global__ void cuda_tiled_residual_kernel (const float *input_1, const float *input_2, float *output, unsigned int N, unsigned int M, const unsigned int ldc, LAYER_ACT activation_type);

__global__ void cuda_tiled_layernorm_kernel(const float *input, const float *kernel, const float *bias, 
    float *output, unsigned int N, unsigned int M);

void cuda_preset_conv2d_ptrs(
    const unsigned int N, const unsigned int Range, float *null_data,
    int *col_idx_arr, float **col_ptr_arr, const unsigned int col_per_n, const unsigned int K_col,
    float *B, const unsigned int ldb, cudaStream_t stream);

void cuda_conv2d (const unsigned int M, const unsigned int N, 
    const int *col_idx_arr, const unsigned int col_per_n, const unsigned int K_col,
    const float *A, const unsigned int lda, const float *B, const unsigned int ldb, float *C, const unsigned int ldc,
    const float *Bias, LAYER_ACT activation_type, cudaStream_t stream);

void cuda_maxpool(
    const unsigned int M, const unsigned int N, 
    const int *col_idx_arr, const unsigned int col_per_n,
    const float *B, const unsigned int ldb, float *C, const unsigned int ldc,
    LAYER_ACT activation_type, cudaStream_t stream);

void cuda_avgpool(
    const unsigned int M, const unsigned int N, 
    const int *col_idx_arr, const unsigned int col_per_n,
    const float *B, const unsigned int ldb, float *C, const unsigned int ldc,
    LAYER_ACT activation_type, cudaStream_t stream);

void cuda_tiled_conv2d (const unsigned int M, const unsigned int N, 
    const float **col_ptr_arr, const unsigned int col_per_n, const unsigned int K_col,
    const float *A, const unsigned int lda, const float *B, const unsigned int ldb, float *C, const unsigned int ldc,
    const float *Bias, LAYER_ACT activation_type, cudaStream_t stream);

void cuda_tiled_maxpool(
    const unsigned int M, const unsigned int N, const unsigned int K_pos,
    const float **col_ptr_arr, const unsigned int col_per_n,
    float *C, const unsigned int ldc,
    LAYER_ACT activation_type, cudaStream_t stream);

void cuda_tiled_avgpool(
    const unsigned int M, const unsigned int N, const unsigned int K_pos,
    const float **col_ptr_arr, const unsigned int col_per_n,
    float *C, const unsigned int ldc,
    LAYER_ACT activation_type, cudaStream_t stream);

void cuda_tiled_k_attention (
    const unsigned int M, const unsigned int N, const unsigned int K,
    const float *key_head, const unsigned int ldk, const float *B_head, const unsigned int ldb, float *C_head, const unsigned int ldc,
    cudaStream_t stream);

void cuda_tiled_v_attention (
    const unsigned int M, const unsigned int N, const unsigned int K,
    const float *val_head, const unsigned int ldv, const float *B_head, const unsigned int ldb, float *C_head, const unsigned int ldc
    , cudaStream_t stream);

void cuda_tiled_residual (const float *input_1, const float *input_2, float *output, unsigned int N, unsigned int M, const unsigned int ldc
    , LAYER_ACT activation_type, cudaStream_t stream);

void cuda_matmul (const unsigned int M, const unsigned int N, const unsigned int K,
		 const float *A, const unsigned int lda, const float *B, const unsigned int ldb, float *C, const unsigned int ldc,
         const float *Bias, LAYER_ACT activation_type, cudaStream_t stream);

void cuda_layernorm (const float *input, const float *weight, const float *bias, 
    float *output, unsigned int N, unsigned int M, unsigned int ldb, unsigned int ldc, cudaStream_t stream);

void cuda_k_attention (const float *input_1, const float *input_2, float *output, unsigned int batch_size
    , unsigned int num_heads, unsigned int num_hidden, unsigned int num_seq, cudaStream_t stream);

void cuda_v_attention (const float *input_1, const float *input_2, float *output, unsigned int batch_size
    , unsigned int num_heads, unsigned int num_hidden, unsigned int num_seq, cudaStream_t stream);

void cuda_residual (const float *input_1, const float *input_2, float *output, unsigned int num_elements
    , LAYER_ACT activation_type, cudaStream_t stream);

#endif

#endif // _CUDA_KERNELS_H