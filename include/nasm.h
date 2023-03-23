#ifndef _NASM_H_
#define _NASM_H_
#include "aspen.h"

typedef enum {
    OUT_W, OUT_H, OUT_C, BATCH, IN_W, IN_H, IN_C, F_W, F_H, STRIDE, PADDING, DILATION, GROUPS,
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

typedef enum {
    NINST_NOT_READY, NINST_READY, NINST_COMPLETED,
} NINST_STATE;

extern char *layer_type_str [NUM_LAYER_ELEMENTS];
extern char *param_type_str[NUM_PARAM_ELEMENTS];
extern char *tensor_type_str[NUM_TENSOR_ELEMENTS];
extern char *parent_type_str[NUM_PARENT_ELEMENTS];
extern char *activation_type_str [NUM_ACTIVATION_ELEMENTS];
extern char *nist_op_str [NUM_NIST_OP_ELEMENTS];

struct nasm_t
{
    aspen_dnn_t *dnn;
    unsigned int batch_size;
    nasm_ldata_t *ldata_arr;
    ninst_t *ninst_arr;
    unsigned int num_ldata;
    unsigned int flop_per_ninst;
};

struct nasm_ldata_t
{
    nasm_t *nasm;
    aspen_layer_t *layer;
    unsigned int parent_ldata_idx_arr [NUM_PARENT_ELEMENTS];
    unsigned int *child_ldata_idx_arr; // Allows duplicate entries.
    unsigned int num_child_ldata;
    unsigned int num_child_ldata_completed;

    unsigned int flop_per_output;
    unsigned int out_mat_dims [2];
    unsigned int out_mat_stride;
    size_t out_mat_size;
    
    unsigned int ninst_tile_dims [2];
    ninst_t *ninst_arr_start;
    
    unsigned int num_ninst;
    unsigned int num_ninst_completed;
    rpool_t *forced_pool;
};

struct ninst_t 
{
    nasm_ldata_t *ldata;
    NINST_STATE state;
    unsigned int ninst_idx;
    unsigned int out_mat_pos [2];

    unsigned int *parent_ninst_idx_arr;
    unsigned int num_parent_ninsts;
    unsigned int num_parent_ninsts_completed;
    
    void *out_mat;
    rpool_t *affinity_pool;
    unsigned int parent_data_offset [NUM_PARENT_ELEMENTS];
};

struct aspen_dnn_t
{
    char name [MAX_STRING_LEN];
    size_t element_size;
    aspen_layer_t *layers;
    unsigned int num_layers;
    unsigned int ref_nasms;
};

struct aspen_tensor_t
{
    unsigned int num_dims;
    unsigned int data_dim_order[MAX_TENSOR_DIMS];
    unsigned int dims[NUM_PARAM_ELEMENTS];
    unsigned int num_elements;
    void *data;
};

struct aspen_layer_t
{
    aspen_dnn_t* dnn;
    unsigned int layer_idx;

    LAYER_TYPE type;
    LAYER_ACT activation;
    aspen_layer_t *parent_layers [NUM_PARENT_ELEMENTS];
    unsigned int params [NUM_PARAM_ELEMENTS];
    aspen_tensor_t *tensors [NUM_TENSOR_ELEMENTS];
};

#endif /* _NASM_H_ */