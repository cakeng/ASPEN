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

#define DSE_NINST_CACHE_BALLANCE 1
#define DSE_NINST_CACHE_DIFF 0
#define DSE_SCRATCHPAD_SIZE 1024*1024*2 // 2 MiB

struct dse_group_t
{
    unsigned int num_dess;
    dse_t *dse_arr;
    int gpu_idx;
};

struct dse_t
{
    dse_group_t *dse_group;
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
    
    // for multi-user case
    int is_multiuser_case;
    int prioritize_rpool[SCHEDULE_MAX_DEVICES];
    int enabled_device[SCHEDULE_MAX_DEVICES];
    rpool_t *rpool_arr[SCHEDULE_MAX_DEVICES];
    networking_engine *net_engine_arr[SCHEDULE_MAX_DEVICES];

    int device_idx;
    DEVICE_MODE device_mode;
    int num_edge_devices;

    int target_device;

    int gpu_idx;

    // for dynamic scheduling
    int is_dynamic_scheduling;
    dynamic_scheduler_t* dynamic_scheduler;

    // for profiling stage
    int profile_compute;

    int operating_mode;

    // for fl offloading
    int is_fl_offloading;
};

void dse_init (dse_group_t *dse_group, dse_t *dse, int gpu_idx);
void dse_destroy (dse_t *dse);

void dse_run (dse_t *dse);
void dse_stop (dse_t *dse);

void dse_schedule (dse_t *dse);
void dse_schedule_fl (dse_t *dse);
void dse_set_starting_path (fl_path_t *path);

void dse_group_set_net_engine (dse_group_t *dse_group, networking_engine *net_engine);
void dse_group_set_device (dse_group_t *dse_group, int device_idx);
void dse_group_set_profile (dse_group_t *dse_group, int profile_compute);
void dse_group_set_multiuser (dse_group_t *dse_group, int is_multiuser_case);
void dse_group_set_dynamic_scheduler (dse_group_t *dse_group, dynamic_scheduler_t* dynamic_scheduler);
void dse_group_set_device_mode (dse_group_t *dse_group, DEVICE_MODE device_mode);
void dse_group_set_operating_mode (dse_group_t *dse_group, int operating_mode);
void dse_group_set_num_edge_devices (dse_group_t *dse_group, int num_edge_devices);
void dse_group_add_prioritize_rpool (dse_group_t *dse_group, int device_idx);
void dse_group_init_enable_device(dse_group_t *dse_group, int num_edge_devices);
void dse_group_set_enable_device(dse_group_t *dse_group, int device_idx, int enable);
void dse_group_add_rpool_arr(dse_group_t *dse_group, rpool_t *rpool, int device_idx);
void dse_group_init_netengine_arr (dse_group_t *dse_group);
void dse_group_add_netengine_arr (dse_group_t *dse_group, networking_engine *net_engine, int device_idx);

void update_children_to_cache_but_prioritize_dse_target (rpool_queue_t *cache, ninst_t *ninst, ninst_t **dse_target);
void update_children_to_cache (rpool_queue_t *cache, ninst_t *ninst);
void update_children_but_prioritize_dse_target (rpool_t *rpool, ninst_t *ninst, dse_t *dse);
void update_children (rpool_t *rpool, ninst_t *ninst);
void push_first_layer_to_rpool (rpool_t *rpool, nasm_t *nasm, void* input_data);

// void generate_cudagraph (nasm_t *nasm);
// void run_cudagraph (nasm_t *nasm);
// void dse_cudagraph_run (rpool_t *rpool, nasm_t *nasm);
// #ifdef GPU
// void add_cudagraph_node_conv2d (cudaGraph_t cuda_graph, ninst_t *ninst, int gpu_idx);
// void add_cudagraph_node_matmul (cudaGraph_t cuda_graph, ninst_t *ninst, int gpu_idx);
// void add_cudagraph_node_maxpool2d (cudaGraph_t cuda_graph, ninst_t *ninst, int gpu_idx);
// void add_cudagraph_node_avgpool2d (cudaGraph_t cuda_graph, ninst_t *ninst, int gpu_idx);
// void add_cudagraph_node_fully_connected (cudaGraph_t cuda_graph, ninst_t *ninst, int gpu_idx);
// void add_cudagraph_node_residual (cudaGraph_t cuda_graph, ninst_t *ninst, int gpu_idx);
// void add_cudagraph_node_softmax (cudaGraph_t cuda_graph, ninst_t *ninst, int gpu_idx);
// void add_cudagraph_node_layernorm (cudaGraph_t cuda_graph, ninst_t *ninst, int gpu_idx);
// void add_cudagraph_node_k_attention (cudaGraph_t cuda_graph, ninst_t *ninst, int gpu_idx);
// void add_cudagraph_node_v_attention (cudaGraph_t cuda_graph, ninst_t *ninst, int gpu_idx);
// #endif
#endif