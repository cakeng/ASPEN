#ifndef _NASM_H_
#define _NASM_H_
#include "aspen.h"

typedef enum {
    OUT_W, OUT_H, IN_W, IN_H, IN_C, OUT_C, F_W, F_H, STRIDE, PADDING, DILATION, GROUPS,
    SEQ_LEN, HEAD_NUM, HIDDEN_PER_HEAD,
    FORM_BYTES, NUM_PARAM_ELEMENTS
} LAYER_PARAMS;

typedef enum {
    NULL_TENSOR, OUTPUT, INPUT, FILTER, BIAS,
    NUM_TENSOR_ELEMENTS
} LAYER_TENSORS;

typedef enum {
    PARENT_NONE, PARENT_0, PARENT_1, PARENT_FILTER,
    NUM_PARENT_ELEMENTS
} LAYER_PARENTS;

extern char *layer_type_str [NUM_LAYER_ELEMENTS];
extern char *param_type_str[NUM_PARAM_ELEMENTS];
extern char *tensor_type_str[NUM_TENSOR_ELEMENTS];
extern char *parent_type_str[NUM_PARENT_ELEMENTS];
extern char *activation_type_str [NUM_ACTIVATION_ELEMENTS];
extern char *nist_op_str [NUM_NIST_OP_ELEMENTS];

struct nasm_t
{
    aspen_dnn_t *dnn;
    nasm_ldata_t *l_data;
    nasm_gpu_ldata_t *gpu_l_data;
    unsigned int num_layers;
};

struct nasm_ldata_t
{
    ninst_t *ninsts;
    unsigned int num_ninst;

    void *output_tensor;
    unsigned int num_ninst_completed;
};

struct nasm_gpu_ldata_t
{
    void *gpu_output_tensor;
};

struct aspen_dnn_t
{
    char name [256];
    aspen_layer_t *layers;
    unsigned int num_layers;
};

struct aspen_layer_t
{
    aspen_dnn_t* dnn;
    unsigned int layer_idx;

    LAYER_TYPE type;
    LAYER_ACT activation;
    unsigned int params [NUM_PARAM_ELEMENTS];

    aspen_layer_t *parent_layer [NUM_PARENT_ELEMENTS];

    void *tensors [NUM_TENSOR_ELEMENTS];
    unsigned int is_tensor_owner [NUM_TENSOR_ELEMENTS];
};

char *get_param_type_str (LAYER_PARAMS param_type);
char *get_tensor_type_str (LAYER_TENSORS tensor_type);
char *get_parent_type_str (LAYER_PARENTS parent_type);

#endif /* _NASM_H_ */