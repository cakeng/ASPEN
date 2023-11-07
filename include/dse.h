#ifndef _dse_H_
#define _dse_H_

#include "aspen.h"
#include "nasm.h"
#include "apu.h"
#include "rpool.h"
#include "util.h"
#include "kernels.h"

#define DSE_NINST_CACHE_BALLANCE 1
#define DSE_NINST_CACHE_DIFF 0
#define DSE_SCRATCHPAD_SIZE 1024*1024*4 // 4 MiB
#define DSE_PROFILE_RUN_NUM 100

struct dse_group_t
{
    unsigned int num_dses;
    dse_t *dse_arr;

    aspen_peer_t *my_peer_data; // Prioritized over dse_t->my_peer_data

    size_t num_profiles;
    size_t max_num_profiles;
    runtime_profile_t **profile_arr;
};

struct dse_t
{
    dse_group_t *dse_group;
    _Atomic int run;
    _Atomic int kill;
    unsigned int thread_id;
    void *scratchpad;
    pthread_t thread;
    pthread_mutex_t thread_mutex;
    pthread_cond_t thread_cond;
    ninst_t *target;
    rpool_queue_t *ninst_cache;
    rpool_t *rpool;
    aspen_peer_t *my_peer_data;
};

struct runtime_profile_t
{
    HASH_t ninst_hash;
    size_t runtime_usec;
};

void dse_init (dse_t *dse);
void dse_destroy (dse_t *dse);

void dse_run (dse_t *dse);
void dse_stop (dse_t *dse);

int dse_check_compute (dse_t *dse, ninst_t *ninst);
int dse_check_send (dse_t *dse, ninst_t *ninst);

void dse_execute (dse_t *dse, ninst_t *ninst);
void dse_schedule (dse_t *dse);

runtime_profile_t *dse_profile_init (HASH_t ninst_hash, size_t runtime_usec);
void dse_profile_destroy (runtime_profile_t *dse_profile);

void update_children_to_cache_but_prioritize_dse_target (rpool_queue_t *cache, ninst_t *ninst, ninst_t **dse_target);
void update_children_to_cache (rpool_queue_t *cache, ninst_t *ninst);
void update_children_but_prioritize_dse_target (rpool_t *rpool, ninst_t *ninst, dse_t *dse);
void update_children (rpool_t *rpool, ninst_t *ninst, unsigned int dse_idx);
void push_first_layer_to_rpool (rpool_t *rpool, nasm_t *nasm, void* input_data);

#endif