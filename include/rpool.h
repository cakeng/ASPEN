#ifndef _RPOOL_H_
#define _RPOOL_H_

#include "aspen.h"
#include "nasm.h"
#include "dse.h"
#include <stdatomic.h>

#define RPOOL_INIT_QUEUE_SIZE 1024
#define MAX_QUEUE_GROUPS 128
#define MAX_NUM_QUEUES (1024*4)
#define NINST_PUSH_BATCH_SIZE 16
#define NUM_QUEUE_PER_LAYER ((float)0.5)
#define NUM_LAYERQUEUE_PER_ASE ((float)0.1)

struct rpool_queue_t
{
    rpool_queue_group_t* queue_group;
    // _Atomic unsigned int occupied;
    pthread_mutex_t occupied_mutex;
    unsigned int idx_start;
    unsigned int idx_end;
    unsigned int num_stored;
    unsigned int max_stored;
    ninst_t **ninst_ptr_arr;
};

struct rpool_queue_group_t
{
    unsigned int idx;
    char queue_group_info[MAX_STRING_LEN];
    void* blacklist_conds [NUM_RPOOL_CONDS];
    void* whitelist_conds [NUM_RPOOL_CONDS];
    rpool_queue_t queue_arr[MAX_NUM_QUEUES];
    _Atomic unsigned int num_queues;
    // _Atomic unsigned int num_ninsts;
    // _Atomic unsigned int num_fetched;
};

struct rpool_t
{
    _Atomic unsigned int num_groups;
    rpool_queue_group_t queue_group_arr[MAX_QUEUE_GROUPS];
    float queue_group_weight_arr[MAX_QUEUE_GROUPS];
    float queue_group_weight_sum;
    rpool_queue_t default_queue;
    _Atomic unsigned int ref_ases;
    int gpu_idx;
};

void rpool_init_queue (rpool_queue_t *rpool_queue);
void rpool_destroy_queue (rpool_queue_t *rpool_queue);

void rpool_init_queue_group (rpool_queue_group_t *rpool_queue_group, char *queue_group_info, unsigned int num_queues);
void rpool_destroy_queue_group (rpool_queue_group_t *rpool_queue_group);

void rpool_pop_all_nasm (rpool_t *rpool, nasm_t *nasm);
void rpool_pop_all (rpool_t *rpool);
void rpool_finish_nasm (rpool_t *rpool, nasm_t *nasm);

rpool_queue_group_t *get_queue_group_from_nasm (rpool_t *rpool, nasm_t *nasm);
int get_queue_group_idx_from_nasm (rpool_t *rpool, nasm_t *nasm);
void rpool_add_queue_group (rpool_t *rpool, char *queue_group_info, unsigned int num_queues, void **blacklist, void **whitelist);
void rpool_queue_group_set_blacklist (rpool_queue_group_t *rpool_queue_group, void **blacklist);
void rpool_queue_group_set_whitelist (rpool_queue_group_t *rpool_queue_group, void **whitelist);
void rpool_set_nasm_weight (rpool_t *rpool, nasm_t* nasm, float weight);

void set_queue_group_weight (rpool_t *rpool, rpool_queue_group_t *rpool_queue_group, float weight);
void queue_group_add_queues (rpool_queue_group_t *rpool_queue_group, unsigned int num_queues);
void add_ref_ases (rpool_t *rpool, unsigned int num_ases);
unsigned int check_blacklist_cond (void **blacklist, void **input_cond);
unsigned int check_whitelist_cond (void **whitelist, void **input_cond);

unsigned int pop_ninsts_from_queue (rpool_queue_t *rpool_queue, ninst_t **ninst_ptr_list, unsigned int max_ninsts_to_get);
unsigned int pop_ninsts_from_queue_enabled (rpool_queue_t *rpool_queue, ninst_t **ninst_ptr_list, unsigned int max_ninsts_to_get, int *enabled_device);
unsigned int pop_ninsts_from_queue_back (rpool_queue_t *rpool_queue, ninst_t **ninst_ptr_list, unsigned int max_ninsts_to_get);
void push_ninsts_to_queue (rpool_queue_t *rpool_queue, ninst_t **ninst_ptr_list, unsigned int num_ninsts);
void push_ninsts_to_queue_front (rpool_queue_t *rpool_queue, ninst_t **ninst_ptr_list, unsigned int num_ninsts);

rpool_queue_t *get_queue_for_fetching (rpool_t *rpool, void **input_cond, unsigned int dse_idx);
rpool_queue_t *get_queue_for_storing (rpool_t *rpool, unsigned int queue_val, void **input_cond);

unsigned int rpool_fetch_ninsts (rpool_t *rpool, ninst_t **ninst_ptr_list, unsigned int max_ninst_to_fetch, unsigned int dse_idx);
void rpool_push_ninsts (rpool_t *rpool, ninst_t **ninst_ptr_list, unsigned int num_ninsts, unsigned int dse_idx);

#endif /* _RPOOL_H_ */