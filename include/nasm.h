#ifndef _NASM_H_
#define _NASM_H_


#include "aspen.h"

typedef enum {
    OUT_W, OUT_H, IN_W, IN_H, IN_C, OUT_C, F_W, F_H, STRIDE, PADDING, DILATION, GROUPS,
    SEQ_LEN, HEAD_NUM, HIDDEN_PER_HEAD,
    FORM_BYTES, NUM_PARAM_ELEMENTS
} layer_params;

typedef enum {
    NULL, OUTPUT, INPUT, FILTER, BIAS,
    NUM_TENSOR_ELEMENTS
} layer_tensors;

typedef enum {
    NONE, PARENT_0, PARENT_1, FILTER,
    NUM_PARENT_ELEMENTS
} layer_parents;

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
    aspen_layer_t *layers;
    unsigned int num_layers;
    unsigned int num_gpus;
};

struct aspen_layer_t
{
    aspen_dnn_t* dnn;
    unsigned int layer_num;

    layer_type type;
    layer_act activation;
    unsigned int layer_params [layer_params::NUM_PARAM_ELEMENTS];

    aspen_layer_t *parent_layer [layer_parents::NUM_PARENT_ELEMENTS];

    void *tensors [layer_tensors::NUM_TENSOR_ELEMENTS];
    unsigned int is_tensor_owner [layer_tensors::NUM_TENSOR_ELEMENTS];

    aspen_gpu_ldata_t *gpu_ldata;
};

struct aspen_gpu_ldata_t
{
    void *gpu_tensors [layer_tensors::NUM_TENSOR_ELEMENTS];
    unsigned int is_gpu_tensor_owner [layer_tensors::NUM_TENSOR_ELEMENTS];
};
