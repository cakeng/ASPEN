#ifndef _ASE_H_
#define _ASE_H_

#include "aspen.h"
#include "nasm.h"
#include "rpool.h"
#include "util.h"

#define ASE_NINST_CACHE_BALLANCE 6
#define ASE_NINST_CACHE_DIFF 4
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

    rpool_queue_t *ninst_cache;
    rpool_t *rpool;

    int gpu_idx;
};

void ase_init (ase_t *ase, int gpu_idx);
void ase_destroy (ase_t *ase);

void ase_run (ase_t *ase);
void ase_stop (ase_t *ase);

void update_children_to_cache (rpool_queue_t *cache, ninst_t *ninst);
void update_children (rpool_t *rpool, ninst_t *ninst);
void push_first_layer_to_rpool (rpool_t *rpool, nasm_t *nasm);

void set_ldata_out_mat_mem_pos (nasm_ldata_t *ldata);
void set_ninst_out_mat_mem_pos (ninst_t *ninst);

#endif