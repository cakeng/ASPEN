#ifndef _ASPEN_H_
#define _ASPEN_H_

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <limits.h>
#include <assert.h>
#include <cuda_runtime.h>

#define MAX_TENSOR_DIMS 8
#define MAX_STRING_LEN 256
#define NINST_H_MIN 64
#define NINST_W_MIN 32
#define MEM_ALIGN 64

#if SUPPRESS_OUTPUT == 0
#define PRT(...) printf(__VA_ARGS__) 
#define FPRT(...) fprintf(__VA_ARGS__) 
#else
#define PRT(...)
#define FPRT(...) fprintf(__VA_ARGS__) 
#endif

static inline cudaError_t check_CUDA(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s, at line %d in file %s\n"
        , cudaGetErrorString(result), __LINE__, __FILE__);
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

typedef enum {NO_LAYER_TYPE, INPUT_LAYER, CONV_LAYER, FC_LAYER,
 RESIDUAL_LAYER, BATCHNORM_LAYER, YOLO_LAYER, ACTIVATION_LAYER, MAXPOOL_LAYER, AVGPOOL_LAYER,
 ROUTE_LAYER, SOFTMAX_LAYER, NUM_LAYER_ELEMENTS} LAYER_TYPE;
typedef enum {NO_ACTIVATION, SIGMOID, LINEAR, TANH, RELU, LEAKY_RELU, ELU, SELU, NUM_ACTIVATION_ELEMENTS} LAYER_ACT;
typedef enum {NO_OPERATION, N_CONV2D, N_FC, NUM_NIST_OP_ELEMENTS} NIST_OP;

typedef struct aspen_dnn_t aspen_dnn_t;
typedef struct aspen_layer_t aspen_layer_t;
typedef struct aspen_tensor_t aspen_tensor_t;
typedef struct aspen_gpu_ldata_t aspen_gpu_ldata_t;

typedef struct ninst_t ninst_t; // Neural instruction
typedef struct nasm_t nasm_t;   // Neural assembly
typedef struct nasm_ldata_t nasm_ldata_t; // Dynamic per layer data
typedef struct nasm_gpu_ldata_t nasm_gpu_ldata_t; // Dynamic per layer data for GPU
typedef struct rpool_t rpool_t; // Ready pool

aspen_dnn_t *apu_create_dnn(char *input_path, char *weight_path);
void aspen_destroy_dnn(aspen_dnn_t *dnn);

nasm_t *apu_create_nasm(aspen_dnn_t *dnn, int flop_per_ninst);
void aspen_destroy_nasm(nasm_t *nasm);
void apu_load_nasm_from_file(char *filename, nasm_t *output_nasm, aspen_dnn_t *output_dnn);
void apu_save_nasm_to_file(char *filename);

void *aspen_calloc (size_t num, size_t size);
void *aspen_malloc (size_t num, size_t size);
void aspen_free (void *ptr);
void *aspen_gpu_calloc (size_t num, size_t size, int gpu_num);
void *aspen_gpu_malloc (size_t num, size_t size, int gpu_num);
void aspen_gpu_free (void *ptr);

void print_build_info(void);
void print_dnn_info (aspen_dnn_t *dnn, int print_data);
void print_layer_info (aspen_layer_t *layer, int print_data);
void print_tensor_info (aspen_tensor_t *tensor, int print_data);
unsigned int get_smallest_dividable (unsigned int num, unsigned int divider);
#endif /* _ASPEN_H_ */