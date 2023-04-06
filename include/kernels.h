#ifndef _KERNELS_H_
#define _KERNELS_H_

#include "aspen.h"
#include "util.h"

void naive_activate (float *input, unsigned int num_elements, LAYER_ACT activation_type);

void naive_conv2d
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

#ifdef AVX2
#include <immintrin.h>
#endif

#ifdef GPU
#endif

#endif // _KERNELS_H_