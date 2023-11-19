#ifndef _UTIL_H_
#define _UTIL_H_

#include "aspen.h"
#include "apu.h"
#include "nasm.h"

#define GPU_MEM_STREAM_HOST_TO_GPU (35)
#define GPU_MEM_STREAM_GPU_TO_HOST (34)
#define GPU_GRAPH_RUN_STREAM (33)
#define GPU_NAIVE_RUN_STREAM (32)
#define GPU_RUN_STREAM_NUM (32)
#define CUDAGRAPH_MAX_ARG_NUM (16)
#define DYNAMIC_ALLOC_MIN_SIZE (64UL*1024) // 64KiB
#define DYNAMIC_ALLOC_RANGE_SCALE (1.414213562373)
#define DYNAMIC_ALLOC_RANGE (33) // 64KiB ~ 4GiB
#define DYNAMIC_ALLOC_ARR_INIT_SIZE (8)

extern int use_gpu; // Default: 1
extern int aspen_num_gpus;

void *aspen_calloc (size_t num, size_t size);
void *aspen_malloc (size_t num, size_t size);
void aspen_free (void *ptr);
void *aspen_dynamic_calloc (size_t num, size_t size);
void *aspen_dynamic_malloc (size_t num, size_t size);
void aspen_dynamic_free (void *ptr, size_t num, size_t size);
void aspen_flush_dynamic_memory ();
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
double get_time_secs_offset();
void get_elapsed_time (char *name);
void get_elapsed_time_only();
double get_elapsed_time_only_return();
void set_elapsed_time_start();
double get_elapsed_time_start();
float get_max_recv_time(nasm_t* nasm);
float get_min_recv_time(nasm_t* nasm);
float get_max_sent_time(nasm_t* nasm);
float get_min_sent_time(nasm_t* nasm);
float get_max_computed_time(nasm_t* nasm);
float get_min_computed_time(nasm_t* nasm);

void print_float_array (float *input, int num, int newline_num);
void print_float_tensor (float *input, int n, int c, int h, int w);

void save_ninst_log(FILE* log_fp, nasm_t* nasm);

ssize_t read_n(int fd, void *buf, size_t n);
ssize_t write_n(int fd, void *buf, size_t n);

void create_connection(DEVICE_MODE dev_mode, char *server_ip, int server_port, int *server_sock, int *client_sock);
int create_server_sock(char *server_ip, int server_port);
int accept_client_sock(int server_sock);
int connect_server_sock(char *server_ip, int server_port);

double get_sec();
void softmax (float *input, float *output, unsigned int num_batch, unsigned int num_elements);
void get_prob_results (char *class_data_path, float* probabilities, unsigned int num);

#endif /* _UTIL_H_ */