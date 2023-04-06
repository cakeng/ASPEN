#ifndef _KERNELS_H_
#define _KERNELS_H_

#include "aspen.h"
#include "util.h"

void tiled_conv2d (ninst_t *ninst);
void tiled_maxpool2d (ninst_t *ninst);
void tiled_avgpool2d (ninst_t *ninst);
void tiled_fully_connected (ninst_t *ninst);
void tiled_residual (ninst_t *ninst);
void tiled_softmax (ninst_t *ninst);

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

void matmul_f32_base(float *A, float *B, float **C, int k, int m, int n);
void matmul_f32_base_8x1(float *A, float *B, float **C, int k);
void matmul_f32_base_8x2(float *A, float *B, float **C, int k);
void matmul_f32_base_8x3(float *A, float *B, float **C, int k);
void matmul_f32_base_8x4(float *A, float *B, float **C, int k);
void matmul_f32_base_8x5(float *A, float *B, float **C, int k);
void matmul_f32_base_8x6(float *A, float *B, float **C, int k);
void matmul_f32_base_8x7(float *A, float *B, float **C, int k);
void matmul_f32_base_8x8(float *A, float *B, float **C, int k);
void matmul_f32_base_8x9(float *A, float *B, float **C, int k);
void matmul_f32_base_8x10(float *A, float *B, float **C, int k);
void matmul_f32_base_8x11(float *A, float *B, float **C, int k);
void matmul_f32_base_8x12(float *A, float *B, float **C, int k);

void matmul_f32_base_16x1(float *A, float *B, float **C, int k);
void matmul_f32_base_32x1(float *A, float *B, float **C, int k);
void matmul_f32_base_64x1(float *A, float *B, float **C, int k);

void maxpool2d_f32_base (float **input, float *output, int kernel_size, int cin);
void avgpool2d_f32_base (float **input, float *output, int kernel_size, int cin);
void residual_f32_base (float **input, float *output, int cin);

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
#endif // _KERNELS_H_