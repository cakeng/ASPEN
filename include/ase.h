#ifndef _ASE_H_
#define _ASE_H_

#include "aspen.h"
#include "nasm.h"
#include "apu.h"
#include "rpool.h"
#include "util.h"
#include "kernels.h"
#include "cuda_kernels.h"

#define ASE_NINST_CACHE_BALLANCE 1
#define ASE_NINST_CACHE_DIFF 0
#define ASE_SCRATCHPAD_SIZE 1024*1024*32 // 32 MB

struct ase_group_t
{
    unsigned int num_ases;
    ase_t *ase_arr;
    int gpu_idx;
};

struct ase_t
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

    int gpu_idx;
};

void ase_init (ase_t *ase, int gpu_idx);
void ase_destroy (ase_t *ase);

void ase_run (ase_t *ase);
void ase_stop (ase_t *ase);

void update_children_to_cache_but_prioritize_ase_target (rpool_queue_t *cache, ninst_t *ninst, ninst_t **ase_target);
void update_children_to_cache (rpool_queue_t *cache, ninst_t *ninst);
void update_children_but_prioritize_ase_target (rpool_t *rpool, ninst_t *ninst, ase_t *ase);
void update_children (rpool_t *rpool, ninst_t *ninst, unsigned int ase_idx);
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