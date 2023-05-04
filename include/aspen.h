#ifndef _ASPEN_H_
#define _ASPEN_H_

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <pthread.h>
#include <stdint.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>
#include <stdlib.h>

#ifdef AVX2
#include <immintrin.h>
#endif //_AVX2
#ifdef NEON
#include <arm_neon.h>
#endif //_NEON

#define MAX_TENSOR_DIMS 8
#define MAX_STRING_LEN 256
#define MAX_PARENT_NINST_NUM (1<<16) // 65536
#define MAX_NUM_GPUS 16
#define NINST_H_MIN (64)
#define NINST_W_MIN (12)
#define MEM_ALIGN 64
#define GPU_MEM_STREAM_HOST_TO_GPU (35)
#define GPU_MEM_STREAM_GPU_TO_HOST (34)
#define GPU_GRAPH_RUN_STREAM (33)
#define GPU_NAIVE_RUN_STREAM (32)
#define GPU_RUN_STREAM_NUM (32)
#define CUDAGRAPH_MAX_ARG_NUM (16)
// #define GPU

#if SUPPRESS_OUTPUT == 0
#define PRT(...) printf(__VA_ARGS__) 
#define FPRT(...) fprintf(__VA_ARGS__) 
#else
#define PRT(...)
#define FPRT(...) fprintf(stderr, "////An error has occurred////\n") 
#endif

#ifdef GPU
#include <cuda_runtime.h> // CUDA
static inline cudaError_t check_CUDA(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) 
  {
    FPRT(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
  }
#endif
  return result;
}
#endif

typedef enum {NINST_NOT_READY, NINST_READY, NINST_COMPLETED, NUM_NINST_STATES} NINST_STATE;
typedef enum {NO_LAYER_TYPE, INPUT_LAYER, CONV_LAYER, FC_LAYER,
 RESIDUAL_LAYER, BATCHNORM_LAYER, YOLO_LAYER, APPEND_LAYER, ACTIVATION_LAYER, MAXPOOL_LAYER, AVGPOOL_LAYER,
 ROUTE_LAYER, SOFTMAX_LAYER,
 MATMUL_LAYER, LAYERNORM_LAYER, K_ATTENTION_LAYER, V_ATTENTION_LAYER,
 NUM_LAYER_ELEMENTS} LAYER_TYPE;
typedef enum {
    OUT_W, OUT_H, OUT_C, BATCH, IN_W, IN_H, IN_C, WEIGHT_W, WEIGHT_H, SUB_C, STRIDE, PADDING, DILATION, GROUPS,
    NUM_HIDDEN, NUM_HEAD, NUM_SEQ, MAT_M, MAT_N, MAT_K, SUB_M,
    FORM_BYTES, NUM_PARAM_ELEMENTS
} LAYER_PARAMS;
typedef enum {NULL_TENSOR, OUTPUT_TENSOR, INPUT_TENSOR, WEIGHT_TENSOR, BIAS_TENSOR, COL_IDX_TENSOR, ANCHOR_TENSOR,
    BN_VAR_TENSOR, BN_MEAN_TENSOR, BN_WEIGHT_TENSOR, NUM_TENSORS} LAYER_TENSORS;
typedef enum {PARENT_NONE, PARENT_0, PARENT_1, PARENT_WEIGHT, NUM_PARENT_ELEMENTS} LAYER_PARENTS;
typedef enum {NO_ACTIVATION, SIGMOID, LINEAR, TANH, RELU, LEAKY_RELU, ELU, SELU, GELU, NUM_ACTIVATIONS} LAYER_ACT;
typedef enum {RPOOL_DNN, RPOOL_LAYER_TYPE, RPOOL_LAYER_IDX, RPOOL_NASM, RPOOL_ASE, NUM_RPOOL_CONDS} RPOOL_CONDS;

extern char *ninst_state_str [NUM_NINST_STATES];
extern char *layer_type_str [NUM_LAYER_ELEMENTS];
extern char *param_type_str[NUM_PARAM_ELEMENTS];
extern char *tensor_type_str[NUM_TENSORS];
extern char *parent_type_str[NUM_PARENT_ELEMENTS];
extern char *activation_type_str [NUM_ACTIVATIONS];
extern char *rpool_cond_str [NUM_RPOOL_CONDS];

extern int use_gpu; // Default: 1
extern int aspen_num_gpus;
#ifdef GPU
extern cudaStream_t aspen_CUDA_streams[MAX_NUM_GPUS][GPU_MEM_STREAM_HOST_TO_GPU+1];
#endif

typedef struct aspen_dnn_t aspen_dnn_t;
typedef struct aspen_layer_t aspen_layer_t;
typedef struct aspen_tensor_t aspen_tensor_t;

typedef struct ninst_t ninst_t; // Neural instruction
typedef struct nasm_t nasm_t;   // Neural assembly
typedef struct nasm_ldata_t nasm_ldata_t; // Dynamic layer data

typedef struct rpool_t rpool_t; // Ready pool
typedef struct rpool_queue_t rpool_queue_t;
typedef struct rpool_queue_group_t rpool_queue_group_t;

typedef struct ase_t ase_t;     // Asynchronous scheduling engine
typedef struct ase_group_t ase_group_t;

void *aspen_load_input(char *input_filename, unsigned int *input_dims, unsigned int element_size);
void *aspen_load_input_NHWC(char *input_filename, unsigned int *input_dims, unsigned int element_size);
void aspen_run_naive (aspen_dnn_t* dnn, unsigned int *input_params, void *input_data, int gpu_idx);
void aspen_init_naive (aspen_dnn_t* dnn, unsigned int *input_params, void *input_data, int gpu_idx);

aspen_dnn_t *apu_create_dnn(char *input_path, char *data_path);
aspen_dnn_t *apu_create_transformer_encoder_dnn (unsigned int num_transformers,
    unsigned int num_hidden, unsigned int num_head, unsigned int ff_scale, char* name, char *weight_path);
void apu_destroy_dnn(aspen_dnn_t *dnn);
void apu_save_dnn_to_file(aspen_dnn_t *dnn, char *filename);
void apu_load_dnn_data_from_file (aspen_dnn_t *dnn, char *input_path);
aspen_dnn_t *apu_load_dnn_from_file(char *filename);
nasm_t *apu_create_nasm(aspen_dnn_t *dnn, unsigned int flop_per_ninst, unsigned int min_ninst_per_ldata, unsigned int batch_size);
nasm_t *apu_create_transformer_encoder_nasm
  (aspen_dnn_t *dnn, unsigned int flop_per_ninst, unsigned int min_ninst_per_ldata, 
  unsigned int batch_size, unsigned int seq_num);
void apu_destroy_nasm(nasm_t *nasm);
nasm_t *apu_load_nasm_from_file(char *filename, aspen_dnn_t *dnn);
void apu_save_nasm_to_file(nasm_t *nasm, char *filename);

rpool_t *rpool_init (int gpu_idx);
void rpool_destroy (rpool_t *rpool);
void rpool_add_nasm_raw_input (rpool_t *rpool, nasm_t* nasm, float weight, void* input_data);
void rpool_add_nasm (rpool_t *rpool, nasm_t* nasm, float weight, char *input_filename);
void rpool_set_nasm_weight (rpool_t *rpool, nasm_t* nasm, float weight);
void rpool_add_queue_group (rpool_t *rpool, char *queue_group_info, unsigned int num_queues, float weight, void **blacklist, void **whitelist);
void rpool_queue_group_set_blacklist (rpool_queue_group_t *rpool_queue_group, void **blacklist);
void rpool_queue_group_set_whitelist (rpool_queue_group_t *rpool_queue_group, void **whitelist);
void rpool_reset_nasm (rpool_t *rpool, nasm_t *nasm, float weight);

ase_group_t *ase_group_init (unsigned int num_ase, int gpu_idx);
void ase_group_set_rpool (ase_group_t *ase_group, rpool_t *rpool);
void ase_group_destroy (ase_group_t *ase_group);
void ase_cudagraph_run (rpool_t *rpool, nasm_t *nasm);
void ase_group_run (ase_group_t *ase_group);
void ase_group_stop (ase_group_t *ase_group);
void ase_group_run_until_nasm_completion (ase_group_t *ase_group, nasm_t *nasm);
void ase_wait_for_nasm_completion (nasm_t *nasm);
unsigned int ase_check_nasm_completion (nasm_t *nasm);
void *ase_get_ldata_result (nasm_t *nasm, unsigned int ldata_idx, LAYER_PARAMS *order);
void *ase_get_nasm_result (nasm_t *nasm, LAYER_PARAMS *order);

void print_aspen_build_info(void);
void print_dnn_info (aspen_dnn_t *dnn, int print_data);
void print_layer_info (aspen_layer_t *layer, int print_data);
void print_tensor_info (aspen_tensor_t *tensor, int print_data);
void print_nasm_info (nasm_t *nasm, int print_ninst, int print_data);
void print_ldata_info (nasm_ldata_t *ldata, int print_ninst, int print_data);
void print_ninst_info (ninst_t *ninst, int print_data);
void print_rpool_queue_info (rpool_queue_t *rpool_queue);
void print_rpool_queue_group_info (rpool_queue_group_t *rpool_queue_group);
void print_rpool_info (rpool_t *rpool);
void print_nasm_cudagraph_info (nasm_t *nasm, char* output_dot_filename);

#endif /* _ASPEN_H_ */