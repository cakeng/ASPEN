#ifndef _UTIL_H_
#define _UTIL_H_

#include "aspen.h"
#include "apu.h"
#include "nasm.h"

void *aspen_calloc (size_t num, size_t size);
void *aspen_malloc (size_t num, size_t size);
void aspen_free (void *ptr);
void *aspen_gpu_calloc (size_t num, size_t size, int gpu_idx);
void aspen_gpu_memset (void *ptr, int val, size_t size, int gpu_idx);
void *aspen_gpu_malloc (size_t num, size_t size, int gpu_idx);
void *aspen_gpu_malloc_minus_one (size_t num, size_t size, int gpu_idx);

void aspen_host_to_gpu_memcpy (void *dst, void *src, size_t num, int gpu_idx);
void aspen_gpu_to_host_memcpy (void *dst, void *src, size_t num, int gpu_idx);
void aspen_host_to_gpu_async_memcpy (void *dst, void *src, size_t num, int gpu_idx);
void aspen_gpu_to_host_async_memcpy (void *dst, void *src, size_t num, int gpu_idx);
void aspen_sync_gpu (int gpu_idx);
void aspen_sync_gpu_stream (int gpu_idx, int stream_num);
int aspen_get_next_stream (int gpu_idx);
void aspen_gpu_free (void *ptr, int gpu_idx);

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

void save_ninst_log(FILE* log_fp, nasm_t* nasm);

ssize_t read_n(int fd, const void *buf, size_t n);
ssize_t write_n(int fd, const void *buf, size_t n);

int create_server_sock(char *rx_ip, int rx_port);
int accept_client_sock(int server_sock);
int connect_server_sock(char *rx_ip, int rx_port);
#endif /* _UTIL_H_ */