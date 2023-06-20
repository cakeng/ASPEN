#ifndef _dse_H_
#define _dse_H_

#include "aspen.h"
#include "nasm.h"
#include "apu.h"
#include "rpool.h"
#include "util.h"
#include "kernels.h"
#include "cuda_kernels.h"
#include "networking.h"
#include "scheduling.h"

#define dse_NINST_CACHE_BALLANCE 1
#define dse_NINST_CACHE_DIFF 0
#define dse_SCRATCHPAD_SIZE 1024*1024*32 // 32 MB

struct dse_group_t
{
    unsigned int num_ases;
    dse_t *dse_arr;
    int gpu_idx;
};

struct dse_t
{
    _Atomic int run;
    _Atomic int kill;
    unsigned int thread_id;
    void *scratchpad;
    void *gpu_scratchpad;
    pthread_t thread;
    pthread_mutex_t thread_mutex;
    pthread_cond_t thread_cond;
    ninst_t *target;
    rpool_queue_t *ninst_cache;
    rpool_t *rpool;

    networking_engine *net_engine;

    int device_idx;

    int gpu_idx;
};

void dse_init (dse_t *dse, int gpu_idx);
void dse_destroy (dse_t *dse);

void dse_run (dse_t *dse);
void dse_stop (dse_t *dse);

void dse_group_set_net_engine (dse_group_t *dse_group, networking_engine *net_engine);
void dse_group_set_device (dse_group_t *dse_group, int device_idx);

void update_children_to_cache_but_prioritize_dse_target (rpool_queue_t *cache, ninst_t *ninst, ninst_t **dse_target);
void update_children_to_cache (rpool_queue_t *cache, ninst_t *ninst);
void update_children_but_prioritize_dse_target (rpool_t *rpool, ninst_t *ninst, dse_t *dse);
void update_children (rpool_t *rpool, ninst_t *ninst, unsigned int dse_idx);
void push_first_layer_to_rpool (rpool_t *rpool, nasm_t *nasm, void* input_data);

void set_ldata_out_mat_mem_pos (nasm_ldata_t *ldata);
void set_ninst_out_mat_mem_pos (ninst_t *ninst);

void generate_cudagraph (nasm_t *nasm);
void run_cudagraph (nasm_t *nasm);
#ifdef GPU
void add_cudagraph_node_conv2d (cudaGraph_t cuda_graph, ninst_t *ninst, int gpu_idx);
void add_cudagraph_node_matmul (cudaGraph_t cuda_graph, ninst_t *ninst, int gpu_idx);
void add_cudagraph_node_maxpool2d (cudaGraph_t cuda_graph, ninst_t *ninst, int gpu_idx);
void add_cudagraph_node_avgpool2d (cudaGraph_t cuda_graph, ninst_t *ninst, int gpu_idx);
void add_cudagraph_node_fully_connected (cudaGraph_t cuda_graph, ninst_t *ninst, int gpu_idx);
void add_cudagraph_node_residual (cudaGraph_t cuda_graph, ninst_t *ninst, int gpu_idx);
void add_cudagraph_node_softmax (cudaGraph_t cuda_graph, ninst_t *ninst, int gpu_idx);
void add_cudagraph_node_layernorm (cudaGraph_t cuda_graph, ninst_t *ninst, int gpu_idx);
void add_cudagraph_node_k_attention (cudaGraph_t cuda_graph, ninst_t *ninst, int gpu_idx);
void add_cudagraph_node_v_attention (cudaGraph_t cuda_graph, ninst_t *ninst, int gpu_idx);
#endif
#endif