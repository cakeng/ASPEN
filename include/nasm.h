#ifndef _NASM_H_
#define _NASM_H_

#include "aspen.h"
#include "scheduling.h"
#include "profiling.h"
#include <stdatomic.h>
#include <netinet/in.h>

struct nasm_t
{
    unsigned int nasm_id;
    aspen_dnn_t *dnn;
    unsigned int batch_size;
    unsigned int tr_seq_len;
    nasm_ldata_t *ldata_arr;
    ninst_t *ninst_arr;
    unsigned int num_ldata;
    _Atomic unsigned int num_ldata_completed;
    unsigned int num_ninst;
    unsigned int min_ninst_per_ldata;
    unsigned int flop_per_ninst;
    size_t total_flops;
    
    int gpu_idx;
    void *data;

    pthread_mutex_t nasm_mutex;
    pthread_cond_t nasm_cond;

    void *gpu_null_data;
    // #ifdef GPU
    // cudaGraph_t cuda_graph;
    // cudaGraphExec_t cuda_graph_exec; 
    // int cudagraph_instantiated;
    // #endif

    int inference_id;
};

struct nasm_ldata_t
{
    nasm_t *nasm;
    aspen_layer_t *layer;
    unsigned int parent_ldata_idx_arr [NUM_PARENT_ELEMENTS];
    unsigned int *child_ldata_idx_arr; // Allows duplicate entries.
    unsigned int num_child_ldata;
    _Atomic unsigned int num_child_ldata_completed;

    unsigned int flop_per_output;
    unsigned int out_mat_dims [2];
    unsigned int out_mat_stride;
    size_t out_mat_mem_size;
    void *out_mat;
    unsigned int ninst_tile_dims [2];
    ninst_t *ninst_arr_start;
    
    unsigned int num_ninst;
    _Atomic unsigned int num_ninst_completed;
};

struct ninst_t 
{
    nasm_ldata_t *ldata;
    _Atomic NINST_STATE state;
    unsigned int ninst_idx;
    unsigned int out_mat_pos [2];
    unsigned int tile_dims [2];
    
    unsigned int *parent_ninst_idx_arr;
    unsigned int num_parent_ninsts;
    _Atomic unsigned int num_parent_ninsts_completed;

    ninst_t **child_ninst_arr;
    _Atomic unsigned int num_child_ninsts;
    
    int *input_pos_idx_arr;
    void **input_pos_ptr_arr_gpu;
    unsigned int num_input_pos;
    void *out_mat;
    void *network_buf;
    rpool_t *affinity_pool;

    //For logging
    float computed_time;
    float received_time;
    float sent_time;

    // For Scheduling
    int dev_to_compute [SCHEDULE_MAX_DEVICES];       // who will compute this ninst?
    int dev_send_target [SCHEDULE_MAX_DEVICES];    // who wants the result of this ninst?

    float compute_start;
    float compute_end;
    // float send_from_here;
    // float recv_from_other;

    float rank_upward;
    float rank_downward;

    // #ifdef GPU
    // cudaGraphNode_t cudagraph_node;
    // #endif
};

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
    void *data_gpu[MAX_NUM_GPUS];
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

#endif /* _NASM_H_ */