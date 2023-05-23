#ifndef _KERNELS_H_
#define _KERNELS_H_

#include "aspen.h"
#include "dse.h"
#include "util.h"

#define _TILE_SIZE_M 128
#define _TILE_SIZE_N 120
#define _TILE_SIZE_K 368
#define _VEC_SIZE_M 8
#define _VEC_SIZE_N 12
#define _VEC_SIZE_K 8

#if AVX2
#define SGEMM_KERNEL_OMP avx2_sgemm_vectorized_with_omp
#define SGEMM_KERNEL avx2_sgemm_vectorized
#define SGEMM_KERNEL_FULL_TILE avx2_sgemm_full_tile
#define SGEMM_KERNEL_TILE_M avx2_sgemm_tile_M
#define SGEMM_KERNEL_TILE_N avx2_sgemm_tile_N
#elif NEON
#define SGEMM_KERNEL_OMP naive_sgemm_vectorized_with_omp
#define SGEMM_KERNEL neon_sgemm_vectorized
#define SGEMM_KERNEL_FULL_TILE neon_sgemm_full_tile
#define SGEMM_KERNEL_TILE_M neon_sgemm_tile_M
#define SGEMM_KERNEL_TILE_N neon_sgemm_tile_N
#else
#define SGEMM_KERNEL_OMP naive_sgemm_vectorized_with_omp
#define SGEMM_KERNEL naive_sgemm_vectorized
#define SGEMM_KERNEL_FULL_TILE naive_sgemm_vectorized
#define SGEMM_KERNEL_TILE_M naive_sgemm_vectorized
#define SGEMM_KERNEL_TILE_N naive_sgemm_vectorized
#endif

void tiled_conv2d (ninst_t *ninst, dse_t *dse);
void tiled_maxpool2d (ninst_t *ninst, dse_t *dse);
void tiled_avgpool2d (ninst_t *ninst, dse_t *dse);
void tiled_fully_connected (ninst_t *ninst, dse_t *dse);
void tiled_residual (ninst_t *ninst, dse_t *dse);
void tiled_softmax (ninst_t *ninst, dse_t *dse);
void tiled_yolo (ninst_t *ninst, dse_t *dse);
void tiled_append (ninst_t *ninst, dse_t *dse);
void tiled_matmul (ninst_t *ninst, dse_t *dse);
void tiled_layernorm (ninst_t *ninst, dse_t *dse);
void tiled_k_attention (ninst_t *ninst, dse_t *dse);
void tiled_v_attention (ninst_t *ninst, dse_t *dse);

void naive_sigmoid (float *input, float *output, int size);
void naive_activate (float *input, unsigned int num_elements, LAYER_ACT activation_type);

void naive_conv2d
(const float *input, const float *kernel, const float *bias, float *output, 
    unsigned int batch_size, unsigned int input_channels, unsigned int height, unsigned int width,  
        unsigned int output_channels, unsigned int kernel_width , unsigned int kernel_height, 
            unsigned int stride, unsigned int padding);

void naive_conv2d_im2col_mm
(const float *input, const float *kernel, const float *bias, float *output, 
    unsigned int batch_size, unsigned int input_channels, unsigned int height, unsigned int width,  
        unsigned int output_channels, unsigned int kernel_width , unsigned int kernel_height, 
            unsigned int stride, unsigned int padding);

void naive_maxpool2d
(const float *input, float *output, 
    unsigned int batch_size, unsigned int channels, unsigned int height, unsigned int width,  
        unsigned int kernel_height, unsigned int kernel_width, unsigned int stride, unsigned int padding);

void naive_avgpool2d
(const float *input, float *output, 
    unsigned int batch_size, unsigned int channels, unsigned int height, unsigned int width,  
        unsigned int kernel_height, unsigned int kernel_width, unsigned int stride, unsigned int padding);

void naive_fully_connected
(const float *input, const float *kernel, const float *bias, float *output, 
    unsigned int batch_size, unsigned int input_size, unsigned int output_size);

void naive_layernorm (const float *input, const float *kernel, const float *bias, 
    float *output, unsigned int N, unsigned int M);

void naive_yolo (const float *input, const float *anchors, 
    float *output, unsigned int yolo_c, unsigned int h, unsigned int w, unsigned int c, unsigned int stride);

void naive_append (const float *input_1, const float *input_2, float *output,
    const int stride, const int c1, const int c2, const int h2, const int w2);

void naive_k_attention (const float *input_1, const float *input_2, float *output, unsigned int batch_size
    , unsigned int num_heads, unsigned int num_hidden, unsigned int num_seq, unsigned int masked);

void naive_v_attention (const float *input_1, const float *input_2, float *output, unsigned int batch_size
    , unsigned int num_heads, unsigned int num_hidden, unsigned int num_seq);

void naive_residual (const float *input_1, const float *input_2, float *output, unsigned int num_elements);

void naive_softmax (float *input, float *output, unsigned int num_batch, unsigned int num_elements);

void naive_sgemm_with_omp (const unsigned int M, const unsigned int N, const unsigned int K,
		 const float *A, const unsigned int lda, const float *B, const unsigned int ldb, float *C, const unsigned int ldc);

void naive_sgemm (const unsigned int M, const unsigned int N, const unsigned int K,
		 const float *A, const unsigned int lda, const float *B, const unsigned int ldb, float *C, const unsigned int ldc);

void naive_sgemm_vectorized_with_omp (const unsigned int M, const unsigned int N, const unsigned int K,
		 const float *A, const unsigned int lda, const float *B, const unsigned int ldb, float *C, const unsigned int ldc);

void naive_sgemm_vectorized (const unsigned int M, const unsigned int N, const unsigned int K,
		 const float *A, const unsigned int lda, const float *B, const unsigned int ldb, float *C, const unsigned int ldc);

#ifdef AVX2
// avx2 accelerated matmul kernels
void avx2_sgemm_vectorized_with_omp(const unsigned int M, const unsigned int N, const unsigned int K,
    const float *A, const unsigned int lda, const float *B, const unsigned int ldb, float *C, const unsigned int ldc);
void avx2_sgemm_vectorized (const unsigned int M, const unsigned int N, const unsigned int K,
    const float *A, const unsigned int lda, const float *B, const unsigned int ldb, float *C, const unsigned int ldc);
void avx2_sgemm_full_tile(const unsigned int M, const unsigned int N, const unsigned int K,
    const float *A, const unsigned int lda, const float *B, const unsigned int ldb, float *C, const unsigned int ldc);
void avx2_sgemm_tile_M(const unsigned int M, const unsigned int N, const unsigned int K,
    const float *A, const unsigned int lda, const float *B, const unsigned int ldb, float *C, const unsigned int ldc);
void avx2_sgemm_tile_N(const unsigned int M, const unsigned int N, const unsigned int K,
    const float *A, const unsigned int lda, const float *B, const unsigned int ldb, float *C, const unsigned int ldc);
#endif //_AVX2
#ifdef NEON
void neon_sgemm_vectorized (const unsigned int M, const unsigned int N, const unsigned int K,
    const float *A, const unsigned int lda, const float *B, const unsigned int ldb, float *C, const unsigned int ldc);
#endif //_NEON
#endif // _KERNELS_H_