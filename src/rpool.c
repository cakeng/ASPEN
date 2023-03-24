#include "rpool.h"

void rpool_init_queue (rpool_queue_t *rpool_queue, char *queue_info)
{

}
void rpool_init_queue_group (rpool_queue_group_t *rpool_queue_group, char *queue_group_info)
{

}
void rpool_destroy_queue (rpool_queue_t *rpool_queue)
{

}
void rpool_destroy_queue_group (rpool_queue_group_t *rpool_queue_group)
{

}

rpool_t *rpool_init ()
{

}
void rpool_destroy (rpool_t *rpool)
{

}
void rpool_add_queue_group (rpool_t *rpool, char *queue_group_info, unsigned int num_queues, float weight)
{

}
void rpool_queue_group_set_blacklist (rpool_queue_group_t *rpool_queue_group, void *blacklist)
{

}
void rpool_queue_group_set_whitelist (rpool_queue_group_t *rpool_queue_group, void *whitelist)
{

}

unsigned int check_blacklist_cond (void *blacklist, void *input_cond)
{

}
unsigned int check_whitelist_cond (void *whitelist, void *input_cond)
{

}

unsigned int get_nists_from_queue (rpool_queue_t *rpool_queue, ninst_t **ninst_list, unsigned int *max_ninsts_to_get)
{

}
void push_ninsts_to_queue (rpool_queue_t *rpool_queue, ninst_t *ninst_list, unsigned int num_ninsts)
{

}

rpool_queue_t *get_queue_for_fetching (rpool_t *rpool)
{

}
rpool_queue_t *get_queue_for_storing (rpool_t *rpool)
{

}

unsigned int rpool_fetch_ninsts (rpool_t *rpool, ninst_t **ninst_list, unsigned int *max_ninst_to_fetch)
{

}
void rpool_push_ninsts (rpool_t *rpool, ninst_t *ninst_list, unsigned int num_ninsts)
{

}

void print_rpool_queue_info (rpool_queue_t *rpool_queue)
{

}
void print_rpool_queue_group_info (rpool_queue_group_t *rpool_queue_group)
{

}
void print_rpool_info (rpool_t *rpool)
{

}
