#ifndef _NASM_H_
#define _NASM_H_

#include "aspen.h"
#include "scheduling.h"
#include "profiling.h"
#include <stdatomic.h>
#include <netinet/in.h>

#define APU_GENERATION_COEFF ((double)0.8)
#define APU_GENERATION_NUM_NINST 512
#define APU_GENERATION_COEFF_GPU ((double)0.8)
#define APU_GENERATION_NUM_NINST_GPU 50
#define APU_GENERATION_NUM_FLOPS 5e8
#define INIT_NUM_PARENT_LDATA 2

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
    _Atomic unsigned int completed;
    unsigned int num_ninst;
    unsigned int min_ninst_per_ldata;
    unsigned int flop_per_ninst;
    size_t total_flops;
    
    int gpu_idx;

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
    float eft_edge;
    float eft_server;
    int dse_idx;

    // For Scheduling
    atomic_int dev_to_compute [SCHEDULE_MAX_DEVICES+1];       // who will compute this ninst?
    atomic_int dev_send_target [SCHEDULE_MAX_DEVICES+1];    // who wants the result of this ninst?
    // atomic_int offloaded;

    double compute_start;
    double compute_end;
    // float send_from_here;
    // float recv_from_other;

    float rank_upward;
    float rank_downward;

    // #ifdef GPU
    // cudaGraphNode_t cudagraph_node;
    // #endif
};

nasm_t *apu_create_nasm_without_finding_ninst_parents (aspen_dnn_t *dnn, unsigned int flop_per_ninst, unsigned int batch_size,  unsigned int min_ninst_per_ldata, unsigned int transformer_seq_len);

double test_nasm_time_sec (nasm_t *nasm, unsigned int num_iter, int gpu_idx);

void init_nasm_ldata (nasm_t *nasm, nasm_ldata_t *ldata, aspen_layer_t *layer);
void set_nasm_inference_id (nasm_t *nasm, int inference_id);
void destroy_nasm_ldata_arr (nasm_ldata_t *ldata_arr, int num_ldata);
void set_nasm_to_finished (nasm_t *nasm);

void copy_ldata_out_mat_to_buffer (nasm_ldata_t *ldata, void *buffer)
void copy_buffer_to_ldata_out_mat (nasm_ldata_t *ldata, void *buffer)
void copy_ninst_data_to_buffer (ninst_t *ninst, void *buffer);
void copy_buffer_to_ninst_data (ninst_t *ninst, void *buffer);

void alloc_ldata_out_mat (nasm_ldata_t *ldata);
void free_ldata_out_mat (nasm_ldata_t *ldata);

void set_ldata_out_mat_mem_pos (nasm_ldata_t *ldata);
void set_ninst_out_mat_mem_pos (ninst_t *ninst);

unsigned int get_tensor_idx_from_pos (aspen_tensor_t *tensor, unsigned int *pos);
void get_tensor_pos_from_idx (aspen_tensor_t *tensor, unsigned int idx, unsigned int *pos);
ninst_t *get_ninst_from_tensor_pos (nasm_ldata_t *ldata, unsigned int *tensor_pos);
ninst_t *get_ninst_from_out_mat_pos (nasm_ldata_t *ldata, unsigned int h, unsigned int w);
void get_out_mat_pos_from_nist (nasm_ldata_t *ldata, ninst_t *ninst, unsigned int *out_mat_pos);
void get_out_mat_pos_from_tensor_pos (nasm_ldata_t *ldata, unsigned int *tensor_pos, unsigned int *out_mat_pos);
void get_tensor_pos_from_out_mat_pos (nasm_ldata_t *ldata, unsigned int *out_mat_pos, unsigned int *tensor_pos);
void get_tensor_pos_from_nist (nasm_ldata_t *ldata, ninst_t *ninst, unsigned int *tensor_pos);

void *get_packed_ldata_output_colwise (nasm_ldata_t *ldata);
void *get_packed_ldata_output_rowwise (nasm_ldata_t *ldata);
void *get_ldata_output (nasm_ldata_t *ldata, LAYER_PARAMS *order);

#endif /* _NASM_H_ */