#ifndef _RPOOL_H_
#define _RPOOL_H_

#include <stdatomic.h>
#include <stdlib.h>
#include "aspen.h"
#include "nasm.h"

#define RPOOL_QUEUE_SIZE 1024
#define RAND_SEED 1345908792
#define MAX_QUEUE_GROUPS 1024

struct rpool_queue_t
{
    char queue_info[MAX_STRING_LEN];
    _Atomic unsigned int occupied;
    unsigned int idx_start;
    unsigned int idx_end;
    unsigned int max_stored;
    ninst_t *ninst_idx_data_arr;
};

struct rpool_queue_group_t
{
    char queue_group_info[MAX_STRING_LEN];
    void* blacklist_conds [NUM_RPOOL_CONDS];
    void* whitelist_conds [NUM_RPOOL_CONDS];
    unsigned int num_queues;
    struct rpool_queue_t *rpool_queue_arr;
};

struct rpool_t
{
    _Atomic unsigned int write_lock;
    unsigned int num_groups;
    unsigned int max_num_groups;
    rpool_queue_group_t rpool_queue_group_arr[MAX_QUEUE_GROUPS];
    float queue_group_weight_arr[MAX_QUEUE_GROUPS];
    float queue_group_weight_sum;

    rpool_queue_t default_queue;
};

void rpool_init_queue (rpool_queue_t *rpool_queue, char *queue_info);
void rpool_init_queue_group (rpool_queue_group_t *rpool_queue_group, char *queue_group_info);
void rpool_destroy_queue (rpool_queue_t *rpool_queue);
void rpool_destroy_queue_group (rpool_queue_group_t *rpool_queue_group);

unsigned int check_blacklist_cond (void *blacklist, void *input_cond);
unsigned int check_whitelist_cond (void *whitelist, void *input_cond);

unsigned int get_nists_from_queue (rpool_queue_t *rpool_queue, ninst_t **ninst_list, unsigned int *max_ninsts_to_get);
void push_ninsts_to_queue (rpool_queue_t *rpool_queue, ninst_t *ninst_list, unsigned int num_ninsts);

rpool_queue_t *get_queue_for_fetching (rpool_t *rpool);
rpool_queue_t *get_queue_for_storing (rpool_t *rpool);

unsigned int rpool_fetch_ninsts (rpool_t *rpool, ninst_t **ninst_list, unsigned int *max_ninst_to_fetch);
void rpool_push_ninsts (rpool_t *rpool, ninst_t *ninst_list, unsigned int num_ninsts);

#endif /* _RPOOL_H_ */