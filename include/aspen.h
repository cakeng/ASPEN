#ifndef _ASPEN_H_
#define _ASPEN_H_

typedef enum {NONE, CONV2D, FC} layer_type;
typedef enum {NONE, SIGMOID, TANH, RELU, LEAKY_RELU, ELU, SELU} layer_act;

typedef enum {NONE, CONV2D, FC} ninst_op;

typedef struct aspen_dnn_t aspen_dnn_t;
typedef struct aspen_layer_t aspen_layer_t;
typedef struct aspen_gpu_ldata_t aspen_gpu_ldata_t;

typedef struct ninst_t ninst_t; // Neural instruction
typedef struct nasm_t nasm_t;   // Neural assembly
typedef struct nasm_ldata_t nasm_ldata_t; // Dynamic per layer data
typedef struct nasm_gpu_ldata_t nasm_gpu_ldata_t; // Dynamic per layer data for GPU
typedef struct rpool_t rpool_t; // Ready pool

aspen_dnn_t *apu_create_dnn(void *input);
void aspen_destroy_dnn(aspen_dnn_t *dnn);

nasm_t *apu_create_nasm(aspen_dnn_t *dnn);
void aspen_destroy_nasm(nasm_t *nasm);

#endif /* _ASPEN_H_ */