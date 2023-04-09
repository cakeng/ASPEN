#ifndef _KERNELS_H_
#define _KERNELS_H_

#include "aspen.h"
#include "ase.h"
#include "util.h"

#define __L1_CACHE 32768
#define __L2_CACHE 512*1024
#define __SLACK 1024
#define _TILE_SIZE_M 128
#define _TILE_SIZE_N 128
#define _TILE_SIZE_K 256
#define _VEC_SIZE_M 8
#define _VEC_SIZE_N 12
#define _VEC_SIZE_K 8

void tiled_conv2d (ninst_t *ninst, ase_t *ase);
void tiled_maxpool2d (ninst_t *ninst, ase_t *ase);
void tiled_avgpool2d (ninst_t *ninst, ase_t *ase);
void tiled_fully_connected (ninst_t *ninst, ase_t *ase);
void tiled_residual (ninst_t *ninst, ase_t *ase);
void tiled_softmax (ninst_t *ninst, ase_t *ase);

void naive_activate (float *input, unsigned int num_elements, LAYER_ACT activation_type);

void naive_conv2d
(const float *input, const float *kernel, const float *bias, float **output_ptr, 
    unsigned int batch_size, unsigned int input_channels, unsigned int height, unsigned int width,  
        unsigned int output_channels, unsigned int kernel_width , unsigned int kernel_height, 
            unsigned int stride, unsigned int padding);

void naive_conv2d_im2col_mm
(const float *input, const float *kernel, const float *bias, float **output_ptr, 
    unsigned int batch_size, unsigned int input_channels, unsigned int height, unsigned int width,  
        unsigned int output_channels, unsigned int kernel_width , unsigned int kernel_height, 
            unsigned int stride, unsigned int padding);

void naive_maxpool2d
(const float *input, float **output_ptr, 
    unsigned int batch_size, unsigned int channels, unsigned int height, unsigned int width,  
        unsigned int kernel_height, unsigned int kernel_width, unsigned int stride, unsigned int padding);

void naive_avgpool2d
(const float *input, float **output_ptr, 
    unsigned int batch_size, unsigned int channels, unsigned int height, unsigned int width,  
        unsigned int kernel_height, unsigned int kernel_width, unsigned int stride, unsigned int padding);

void naive_fully_connected
(const float *input, const float *kernel, const float *bias, float **output_ptr, 
    unsigned int batch_size, unsigned int input_size, unsigned int output_size);

void naive_residual (const float *input_1, const float *input_2, float **output_ptr, unsigned int num_elements);

void naive_softmax (float *input, float **output_ptr, unsigned int num_batch, unsigned int num_elements);

void naive_sgemm(const unsigned int M, const unsigned int N, const unsigned int K,
		 const float *A, const unsigned int lda, const float *B, const unsigned int ldb, float *C, const unsigned int ldc);

void naive_sgemm_without_omp(const unsigned int M, const unsigned int N, const unsigned int K,
		 const float *A, const unsigned int lda, const float *B, const unsigned int ldb, float *C, const unsigned int ldc);

void naive_sgemm_vectorized_without_omp(const unsigned int M, const unsigned int N, const unsigned int K,
		 const float *A, const unsigned int lda, const float *B, const unsigned int ldb, float *C, const unsigned int ldc);

void naive_sgemm_tiled_without_omp(const unsigned int M, const unsigned int N, const unsigned int K,
		 const float *A, const unsigned int lda, const float *B, const unsigned int ldb, float *C, const unsigned int ldc);

void naive_sgemm_tiled(const unsigned int M, const unsigned int N, const unsigned int K,
		 const float *A, const unsigned int lda, const float *B, const unsigned int ldb, float *C, const unsigned int ldc);

void naive_sgemm_vectorized(const unsigned int M, const unsigned int N, const unsigned int K,
		 const float *A, const unsigned int lda, const float *B, const unsigned int ldb, float *C, const unsigned int ldc);

#ifdef AVX2
#include <immintrin.h>
// avx2 accelerated matmul kernels
void matmul_f32_avx2_8x1(float *A, float *B, float **C, int k);
void matmul_f32_avx2_8x2(float *A, float *B, float **C, int k);
void matmul_f32_avx2_8x3(float *A, float *B, float **C, int k);
void matmul_f32_avx2_8x4(float *A, float *B, float **C, int k);
void matmul_f32_avx2_8x5(float *A, float *B, float **C, int k);
void matmul_f32_avx2_8x6(float *A, float *B, float **C, int k);
void matmul_f32_avx2_8x7(float *A, float *B, float **C, int k);
void matmul_f32_avx2_8x8(float *A, float *B, float **C, int k);
void matmul_f32_avx2_8x9(float *A, float *B, float **C, int k);
void matmul_f32_avx2_8x10(float *A, float *B, float **C, int k);
void matmul_f32_avx2_8x11(float *A, float *B, float **C, int k);
void matmul_f32_avx2_8x12(float *A, float *B, float **C, int k);

void matmul_f32_avx2_16x1(float *A, float *B, float **C, int k);
void matmul_f32_avx2_32x1(float *A, float *B, float **C, int k);
void matmul_f32_avx2_64x1(float *A, float *B, float **C, int k);

void maxpool2d_f32_avx2 (float **input, float *output, int kernel_size, int cin);

#endif //_AVX2

#ifdef NEON
#include <arm_neon.h>
// NEON accelerated matmul kernels
void matmul_f32_NEON_8x1(float *A, float *B, float **C, int k);
void matmul_f32_NEON_8x2(float *A, float *B, float **C, int k);
void matmul_f32_NEON_8x3(float *A, float *B, float **C, int k);
void matmul_f32_NEON_8x4(float *A, float *B, float **C, int k);
void matmul_f32_NEON_8x5(float *A, float *B, float **C, int k);
void matmul_f32_NEON_8x6(float *A, float *B, float **C, int k);
void matmul_f32_NEON_8x7(float *A, float *B, float **C, int k);
void matmul_f32_NEON_8x8(float *A, float *B, float **C, int k);
void matmul_f32_NEON_8x9(float *A, float *B, float **C, int k);
void matmul_f32_NEON_8x10(float *A, float *B, float **C, int k);
void matmul_f32_NEON_8x11(float *A, float *B, float **C, int k);
void matmul_f32_NEON_8x12(float *A, float *B, float **C, int k);

void matmul_f32_NEON_16x1(float *A, float *B, float **C, int k);
void matmul_f32_NEON_32x1(float *A, float *B, float **C, int k);
void matmul_f32_NEON_64x1(float *A, float *B, float **C, int k);

void maxpool2d_f32_NEON (float **input, float *output, int kernel_size, int cin);
#endif //_NEON

#ifdef OPENBLAS
#include <cblas.h>
#endif

#endif // _KERNELS_H_