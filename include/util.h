#ifndef _UTIL_H_
#define _UTIL_H_

#include "aspen.h"
#include "nasm.h"

void *aspen_calloc (size_t num, size_t size);
void *aspen_malloc (size_t num, size_t size);
void aspen_free (void *ptr);
void *aspen_gpu_calloc (size_t num, size_t size, int gpu_num);
void *aspen_gpu_malloc (size_t num, size_t size, int gpu_num);

void aspen_host_to_gpu_memcpy (void *dst, void *src, size_t num, int gpu_num);
void aspen_gpu_to_host_memcpy (void *dst, void *src, size_t num, int gpu_num);
void aspen_host_to_gpu_async_memcpy (void *dst, void *src, size_t num, int gpu_num);
void aspen_gpu_to_host_async_memcpy (void *dst, void *src, size_t num, int gpu_num);
void aspen_sync_gpu (int gpu_num);
void aspen_sync_gpu_stream (int gpu_num, int stream_num);
int aspen_get_next_stream (int gpu_num);
void aspen_gpu_free (void *ptr, int gpu_num);

unsigned int get_smallest_dividable (unsigned int num, unsigned int divider);
void *aspen_load_image_bin (char *file_path, unsigned int image_size);

void* load_arr (char *file_path, unsigned int size);
void save_arr (void *input, char *file_path, unsigned int size);
void fold_batchnorm_float (float *bn_var, float *bn_mean, float *bn_weight, 
    float *weight, float *bias, int cout, int cin, int hfil, int wfil);

int compare_float_array (float *input1, float* input2, int num_to_compare, float epsilon_ratio, int skip_val);
int compare_float_tensor (float *input1, float* input2, int n, int c, int h ,int w, int num_to_compare, float epsilon_ratio, int skip_val);

#endif /* _UTIL_H_ */