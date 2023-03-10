#ifndef _ASPEN_H_
#define _ASPEN_H_
#if SUPPRESS_OUTPUT == 0
#define PRT(...) printf(__VA_ARGS__) 
#define FPRT(...) fprintf(__VA_ARGS__) 
#else
#define PRT(...)
#define FPRT(...) fprintf(__VA_ARGS__) 
#endif

typedef enum {NO_LAYER_TYPE, INPUT_LAYER, CONV_LAYER, FC_LAYER,
 RESIDUAL_LAYER, BATCHNORM_LAYER, YOLO_LAYER, ACTIVATION_LAYER, MAXPOOL_LAYER, AVGPOOL_LAYER,
 ROUTE_LAYER, SOFTMAX_LAYER, NUM_LAYER_ELEMENTS} LAYER_TYPE;
typedef enum {NO_ACTIVATION, SIGMOID, LINEAR, TANH, RELU, LEAKY_RELU, ELU, SELU, NUM_ACTIVATION_ELEMENTS} LAYER_ACT;
typedef enum {NO_OPERATION, N_CONV2D, N_FC, NUM_NIST_OP_ELEMENTS} NIST_OP;

typedef struct aspen_dnn_t aspen_dnn_t;
typedef struct aspen_layer_t aspen_layer_t;
typedef struct aspen_gpu_ldata_t aspen_gpu_ldata_t;

typedef struct ninst_t ninst_t; // Neural instruction
typedef struct nasm_t nasm_t;   // Neural assembly
typedef struct nasm_ldata_t nasm_ldata_t; // Dynamic per layer data
typedef struct nasm_gpu_ldata_t nasm_gpu_ldata_t; // Dynamic per layer data for GPU
typedef struct rpool_t rpool_t; // Ready pool

aspen_dnn_t *apu_create_dnn(char *input_path, char *weight_path);
void aspen_destroy_dnn(aspen_dnn_t *dnn);

nasm_t *apu_create_nasm(aspen_dnn_t *dnn);
void aspen_destroy_nasm(nasm_t *nasm);
void apu_load_nasm_from_file(char *filename, nasm_t *output_nasm, aspen_dnn_t *output_dnn);
void apu_save_nasm_to_file(char *filename);

void print_build_info(void);
void print_dnn_info (aspen_dnn_t *dnn);
void print_layer_info (aspen_layer_t *layer);

#endif /* _ASPEN_H_ */