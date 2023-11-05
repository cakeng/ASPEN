#ifndef _UTIL_H_
#define _UTIL_H_

#include "aspen.h"
#include "apu.h"
#include "nasm.h"

#define DYNAMIC_ALLOC_MIN_SIZE (64UL*1024) // 64KiB
#define DYNAMIC_ALLOC_RANGE_SCALE (1.414213562373)
#define DYNAMIC_ALLOC_RANGE (33) // 64KiB ~ 4GiB
#define DYNAMIC_ALLOC_ARR_INIT_SIZE (8)

void *aspen_calloc (size_t num, size_t size);
void *aspen_malloc (size_t num, size_t size);
void aspen_free (void *ptr);
void *aspen_dynamic_calloc (size_t num, size_t size);
void *aspen_dynamic_malloc (size_t num, size_t size);
void aspen_dynamic_free (void *ptr, size_t num, size_t size);

size_t get_smallest_dividable (size_t num, size_t divider);

void* load_arr (char *file_path, unsigned int size);
void save_arr (void *input, char *file_path, unsigned int size);
void fold_batchnorm_float (float *bn_var, float *bn_mean, float *bn_weight, 
    float *weight, float *bias, int cout, int cin, int hfil, int wfil);

void NHWC_to_NCHW (void *input, void *output, unsigned int n, unsigned int c, unsigned int h, unsigned int w, unsigned int element_size);
void NCHW_to_NHWC (void *input, void *output, unsigned int n, unsigned int c, unsigned int h, unsigned int w, unsigned int element_size);
void set_float_tensor_val (float *output, unsigned int n, unsigned int c, unsigned int h, unsigned int w);

int compare_float_array (float *input1, float* input2, int num_to_compare, float epsilon_ratio, float epsilon_abs, int skip_val);
int compare_float_tensor (float *input1, float* input2, int n, int c, int h ,int w, float epsilon_ratio, float epsilon_abs, int skip_val);

unsigned int get_cpu_count();

void get_probability_results (char *class_data_path, float* probabilities, unsigned int num);
double get_time_secs();
void get_elapsed_time (char *name);

void print_float_array (float *input, int num, int newline_num);
void print_float_tensor (float *input, int n, int c, int h, int w);
#endif /* _UTIL_H_ */