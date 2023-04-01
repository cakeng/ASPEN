#ifndef _RPOOL_H_
#define _RPOOL_H_

#include <stdatomic.h>
#include <stdlib.h>
#include "aspen.h"
#include "nasm.h"

#define INIT_QUEUE_SIZE 512
#define MAX_QUEUE_GROUPS 128
#define MAX_NUM_QUEUES 1024*4
#define NUM_QUEUE_PER_ASE ((float)1/4)
#define NUM_QUEUE_PER_LAYER ((float)3)

struct rpool_queue_t
{
    _Atomic unsigned int occupied;
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
    _Atomic unsigned int num_queues;
    rpool_queue_t queue_arr[MAX_NUM_QUEUES];
};

struct rpool_t
{
    _Atomic unsigned int num_groups;
    rpool_queue_group_t queue_group_arr[MAX_QUEUE_GROUPS];
    float queue_group_weight_arr[MAX_QUEUE_GROUPS];
    float queue_group_weight_sum;
    rpool_queue_t default_queue;
    _Atomic unsigned int ref_ases;
};

void rpool_init_queue (rpool_queue_t *rpool_queue);
void rpool_init_queue_group (rpool_queue_group_t *rpool_queue_group, char *queue_group_info, unsigned int num_queues);
void rpool_destroy_queue (rpool_queue_t *rpool_queue);
void rpool_destroy_queue_group (rpool_queue_group_t *rpool_queue_group);

rpool_queue_group_t *get_queue_group_from_nasm (rpool_t *rpool, nasm_t *nasm);
void set_queue_group_weight (rpool_t *rpool, rpool_queue_group_t *rpool_queue_group, float weight);
void queue_group_add_queues (rpool_queue_group_t *rpool_queue_group, unsigned int num_queues);
void add_ref_ases (rpool_t *rpool, unsigned int num_ases);

unsigned int check_blacklist_cond (void **blacklist, void **input_cond);
unsigned int check_whitelist_cond (void **whitelist, void **input_cond);

unsigned int pop_ninsts_from_queue (rpool_queue_t *rpool_queue, ninst_t **ninst_ptr_list, unsigned int max_ninsts_to_get);
void push_ninsts_to_queue (rpool_queue_t *rpool_queue, ninst_t **ninst_ptr_list, unsigned int num_ninsts);

rpool_queue_t *get_queue_for_fetching (rpool_t *rpool, void **input_cond);
rpool_queue_t *get_queue_for_storing (rpool_t *rpool, unsigned int queue_val, void **input_cond);

unsigned int rpool_fetch_ninsts (rpool_t *rpool, ninst_t **ninst_ptr_list, unsigned int max_ninst_to_fetch);
void rpool_push_ninsts (rpool_t *rpool, ninst_t **ninst_ptr_list, unsigned int num_ninsts);

#endif /* _RPOOL_H_ */