#ifndef _APU_H_
#define _APU_H_

#include "aspen.h"
#include "nasm.h"
#include "util.h"
#include "kernels.h"
#include "rpool.h"

#define APU_GENERATION_COEFF ((double)0.8)
#define APU_GENERATION_NUM_NINST 512
#define APU_GENERATION_NUM_FLOPS 1e8

struct aspen_dnn_t
{
    char name [MAX_STRING_LEN];
    unsigned int element_size;
    aspen_layer_t *layers;
    unsigned int num_layers;
    _Atomic unsigned int ref_nasms;
};

struct aspen_tensor_t
{
    unsigned int num_dims;
    LAYER_PARAMS data_dim_order[MAX_TENSOR_DIMS];
    unsigned int dims[NUM_PARAM_ELEMENTS];
    unsigned int num_elements;
    unsigned int element_size;
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
    aspen_tensor_t *tensors [NUM_TENSORS];
};

aspen_dnn_t *init_aspen_dnn (unsigned int num_layers, char* name);
void apu_load_dnn_data_from_file (aspen_dnn_t *dnn, char *input_path);
void *aspen_load_input_NHWC(char *input_filename, unsigned int *input_dims, unsigned int element_size);
void *aspen_load_input(char *input_filename, unsigned int *input_dims, unsigned int element_size);

void init_aspen_layer (aspen_layer_t *layer, unsigned int layer_num, aspen_dnn_t *dnn);
void create_layer_tensors (aspen_layer_t *layer);
void create_layer_output_tensor (aspen_layer_t *layer);
void create_layer_col_idx_tensor (aspen_layer_t *layer);
void destroy_aspen_layers (aspen_layer_t* layers, unsigned int num_layers);

aspen_tensor_t *init_aspen_tensor (unsigned int *params_arr, LAYER_PARAMS *order, int num_dims, unsigned int element_size);
void calloc_aspen_tensor (aspen_tensor_t *tensor);
void copy_buffer_to_aspen_tensor  (aspen_tensor_t *tensor, void *buffer);
void copy_aspen_tensor_to_buffer  (aspen_tensor_t *tensor, void *buffer);
void copy_aspen_tensor_to_tensor  (aspen_tensor_t *dst, aspen_tensor_t *src);
void reorder_aspen_tensor (aspen_tensor_t **tensor_ptr, unsigned int *params_arr, LAYER_PARAMS *order, int num_dims);
void *get_aspen_tensor_data (aspen_tensor_t *tensor, LAYER_PARAMS *output_order);
void *get_aspen_tensor_element_ptr (aspen_tensor_t *tensor, unsigned int *pos);
void destroy_aspen_tensor(aspen_tensor_t *tensor);
#endif /* _APU_H_ */