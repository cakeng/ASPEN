#include "rpool.h"

void rpool_init_queue (rpool_queue_t *rpool_queue, char *queue_info)
{
    strncpy (rpool_queue->queue_info, queue_info, MAX_STRING_LEN-1);
    rpool_queue->occupied = 0;
    rpool_queue->idx_start = 0;
    rpool_queue->idx_end = 0;
    rpool_queue->max_stored = RPOOL_INIT_QUEUE_SIZE;
    rpool_queue->ninst_ptr_arr = calloc (RPOOL_INIT_QUEUE_SIZE, sizeof(ninst_t*));
}
void rpool_init_queue_group (rpool_queue_group_t *rpool_queue_group, char *queue_group_info, unsigned int num_queues)
{
    strncpy (rpool_queue_group->queue_group_info, queue_group_info, MAX_STRING_LEN-1);
    rpool_queue_group->num_queues = num_queues;
    rpool_queue_group->queue_arr = calloc (num_queues, sizeof(rpool_queue_t));
    char info_str[MAX_STRING_LEN*2];
    for (int i = 0; i < num_queues; i++)
    {
        sprintf (info_str, "%s_queue_%d", rpool_queue_group->queue_group_info, i);
        rpool_init_queue (&rpool_queue_group->queue_arr[i], "queue");
    }
}
void rpool_destroy_queue (rpool_queue_t *rpool_queue)
{
    if (rpool_queue == NULL)
        return;
    if (rpool_queue->ninst_ptr_arr != NULL)
        free (rpool_queue->ninst_ptr_arr);
}
void rpool_destroy_queue_group (rpool_queue_group_t *rpool_queue_group)
{
    if (rpool_queue_group == NULL)
        return;
    if (rpool_queue_group->queue_arr != NULL)
    {
        for (int i = 0; i < rpool_queue_group->num_queues; i++)
            rpool_destroy_queue (&rpool_queue_group->queue_arr[i]);
        free (rpool_queue_group->queue_arr);
    }
}

rpool_t *rpool_init ()
{
    rpool_t *rpool = calloc (1, sizeof(rpool_t));
    rpool->write_lock = 0;
    rpool->num_groups = 0;
    rpool->queue_group_weight_sum = 0;
    bzero (rpool->queue_group_weight_arr, sizeof(float)*MAX_QUEUE_GROUPS);
    rpool_init_queue (&rpool->default_queue, "default_queue");
    return rpool;
}
void rpool_destroy (rpool_t *rpool)
{
    if (rpool == NULL)
        return;
    for (int i = 0; i < rpool->num_groups; i++)
        rpool_destroy_queue_group (&rpool->queue_group_arr[i]);
    rpool_destroy_queue (&rpool->default_queue);
    free (rpool);
}
void rpool_add_queue_group 
    (rpool_t *rpool, char *queue_group_info, unsigned int num_queues, float weight, void **blacklist, void **whitelist)
{
    if (rpool == NULL)
    {
        FPRT (stderr, "ERROR: rpool_add_queue_group: rpool is NULL.\n");
        return;
    }
    if (rpool->num_groups >= MAX_QUEUE_GROUPS)
    {
        FPRT (stderr, "ERROR: rpool_add_queue_group: max number of queue groups (%d) reached. Cannot add queue group \"%s\".\n", 
            MAX_QUEUE_GROUPS, queue_group_info);
        return;
    }
    while (atomic_exchange (&rpool->write_lock, 1) == 1)
    {
        // wait
    }
    rpool_init_queue_group (&rpool->queue_group_arr[rpool->num_groups], queue_group_info, num_queues);
    rpool_queue_group_set_blacklist (&rpool->queue_group_arr[rpool->num_groups], blacklist);
    rpool_queue_group_set_whitelist (&rpool->queue_group_arr[rpool->num_groups], whitelist);
    rpool->queue_group_weight_arr[rpool->num_groups] = weight;
    rpool->queue_group_weight_sum += weight;
    rpool->num_groups++;
    atomic_store (&rpool->write_lock, 0);
}
void rpool_queue_group_set_blacklist (rpool_queue_group_t *rpool_queue_group, void **blacklist)
{
    if (rpool_queue_group == NULL)
    {
        FPRT (stderr, "ERROR: rpool_queue_group_set_blacklist: rpool_queue_group is NULL.\n");
        return;
    }
    for (int i = 0; i < NUM_RPOOL_CONDS; i++)
        rpool_queue_group->blacklist_conds[i] = blacklist[i];
}
void rpool_queue_group_set_whitelist (rpool_queue_group_t *rpool_queue_group, void **whitelist)
{
    if (rpool_queue_group == NULL)
    {
        FPRT (stderr, "ERROR: rpool_queue_group_set_whitelist: rpool_queue_group is NULL.\n");
        return;
    }
    for (int i = 0; i < NUM_RPOOL_CONDS; i++)
        rpool_queue_group->whitelist_conds[i] = whitelist[i];
}

unsigned int check_blacklist_cond (void **blacklist, void **input_cond)
{
    #ifdef DEBUG
    if (blacklist == NULL)
    {
        FPRT (stderr, "ERROR: check_blacklist_cond: blacklist is NULL.\n");
        return 0;
    }
    if (input_cond == NULL)
    {
        FPRT (stderr, "ERROR: check_blacklist_cond: input_cond is NULL.\n");
        return 0;
    }
    #endif
    for (int i = 0; i < NUM_RPOOL_CONDS; i++)
    {
        if (blacklist[i] == NULL || input_cond[i] == NULL)
            continue;
        if (blacklist[i] == input_cond[i])
            return 0;
    }
    return 1;
}
unsigned int check_whitelist_cond (void **whitelist, void **input_cond)
{
    #ifdef DEBUG
    if (whitelist == NULL)
    {
        FPRT (stderr, "ERROR: check_whitelist_cond: whitelist is NULL.\n");
        return 0;
    }
    if (input_cond == NULL)
    {
        FPRT (stderr, "ERROR: check_whitelist_cond: input_cond is NULL.\n");
        return 0;
    }
    #endif
    for (int i = 0; i < NUM_RPOOL_CONDS; i++)
    {
        if (whitelist[i] == NULL || input_cond[i] == NULL)
            continue;
        if (whitelist[i] != input_cond[i])
            return 0;
    }
    return 1;
}

unsigned int pop_ninsts_from_queue (rpool_queue_t *rpool_queue, ninst_t **ninst_ptr_list, unsigned int max_ninsts_to_get)
{
    #ifdef DEBUG
    if (rpool_queue == NULL)
    {
        FPRT (stderr, "ERROR: pop_nists_from_queue: rpool_queue is NULL.\n");
        return 0;
    }
    if (ninst_ptr_list == NULL)
    {
        FPRT (stderr, "ERROR: pop_nists_from_queue: ninst_ptr_list is NULL.\n");
        return 0;
    }
    #endif
    unsigned int num_ninsts = 0;
    unsigned int i = rpool_queue->idx_start;
    for (; num_ninsts < rpool_queue->num_stored; num_ninsts++)
    {
        if (num_ninsts >= max_ninsts_to_get)
            break;
        ninst_ptr_list[num_ninsts] = rpool_queue->ninst_ptr_arr[i];
        i++;
        if (i == rpool_queue->max_stored)
            i = 0;
    }
    rpool_queue->idx_start = i;
    rpool_queue->num_stored -= num_ninsts;
    return num_ninsts;
}
void push_ninsts_to_queue (rpool_queue_t *rpool_queue, ninst_t **ninst_ptr_list, unsigned int num_ninsts)
{
    #ifdef DEBUG
    if (rpool_queue == NULL)
    {
        FPRT (stderr, "ERROR: push_ninsts_to_queue: rpool_queue is NULL.\n");
        return;
    }
    if (ninst_ptr_list == NULL)
    {
        FPRT (stderr, "ERROR: push_ninsts_to_queue: ninst_ptr_list is NULL.\n");
        return;
    }
    #endif
    if (rpool_queue->num_stored + num_ninsts > rpool_queue->max_stored)
    {
        ninst_t **new_ninst_ptr_arr = (ninst_t **) calloc 
            (rpool_queue->max_stored*2, sizeof (ninst_t *));
        if (rpool_queue->idx_start < rpool_queue->idx_end)
        {
            memcpy (new_ninst_ptr_arr, rpool_queue->ninst_ptr_arr + rpool_queue->idx_start, 
                    (rpool_queue->idx_end - rpool_queue->idx_start)*sizeof (ninst_t *));
        }
        else
        {
            memcpy (new_ninst_ptr_arr, rpool_queue->ninst_ptr_arr + rpool_queue->idx_start, 
                    (rpool_queue->max_stored - rpool_queue->idx_start)*sizeof (ninst_t *));
            memcpy (new_ninst_ptr_arr + rpool_queue->max_stored - rpool_queue->idx_start, 
                    rpool_queue->ninst_ptr_arr, rpool_queue->idx_end*sizeof (ninst_t *));
        }
        free (rpool_queue->ninst_ptr_arr);
        rpool_queue->ninst_ptr_arr = new_ninst_ptr_arr;
        rpool_queue->idx_start = 0;
        rpool_queue->idx_end = rpool_queue->num_stored;
        rpool_queue->max_stored *= 2;
    }
    unsigned int i = rpool_queue->idx_end;
    for (int j = 0; j < num_ninsts; j++)
    {
        rpool_queue->ninst_ptr_arr[i] = ninst_ptr_list[j];
        i++;
        if (i == rpool_queue->max_stored)
            i = 0;
    }
    rpool_queue->idx_end = i;
    rpool_queue->num_stored += num_ninsts;
}

rpool_queue_t *get_queue_for_fetching (rpool_t *rpool, void **input_cond)
{
    #ifdef DEBUG
    if (rpool == NULL)
    {
        FPRT (stderr, "ERROR: get_queue_for_fetching: rpool is NULL.\n");
        return NULL;
    }
    #endif
    while (atomic_load (&rpool->write_lock) == 1)
    {
        // wait
    }
    if (rpool->default_queue.num_stored > 0)
    {
        if (atomic_exchange (&rpool->default_queue.occupied, 1) == 0)
            return &rpool->default_queue;
    }
    rpool_queue_group_t *rpool_queue_group = NULL;
    unsigned int num_tries = 0;
    while (rpool_queue_group == NULL)
    {
        float rand_weight = ((float) rand () / (float) RAND_MAX) * rpool->queue_group_weight_sum;
        for (int i = 0; i < rpool->num_groups; i++)
        {
            if (rand_weight < rpool->queue_group_weight_arr[i])
            {
                if (input_cond == NULL)
                    rpool_queue_group = &rpool->queue_group_arr[i];
                else
                {
                    if (check_blacklist_cond (rpool->queue_group_arr[i].blacklist_conds, input_cond)
                    && check_whitelist_cond (rpool->queue_group_arr[i].whitelist_conds, input_cond))
                        rpool_queue_group = &rpool->queue_group_arr[i];
                }
                break;
            }
            rand_weight -= rpool->queue_group_weight_arr[i];
        }
        num_tries++;
        if (num_tries > 100)
        {
            FPRT (stderr, "ERROR: get_queue_for_fetching: could not find a queue group.\n");
            return NULL;
        }
    }
    rpool_queue_t *rpool_queue = NULL;
    for (int i = 0; i < rpool_queue_group->num_queues; i++)
    {
        if (rpool_queue_group->queue_arr[i].num_stored > 0)
        {
            if (atomic_exchange (&rpool_queue->occupied, 1) == 0)
            {
                rpool_queue = &rpool_queue_group->queue_arr[i];
                break;
            }
        }
    }
    return rpool_queue;
}
rpool_queue_t *get_queue_for_storing (rpool_t *rpool, unsigned int queue_val, void **input_cond)
{
    #ifdef DEBUG
    if (rpool == NULL)
    {
        FPRT (stderr, "ERROR: get_queue_for_storing: rpool is NULL.\n");
        return NULL;
    }
    #endif
    rpool_queue_group_t *rpool_queue_group = NULL;
    while (atomic_load (&rpool->write_lock) == 1)
    {
        // wait
    }
    for (int i = 0; i < rpool->num_groups; i++)
    {
        if (input_cond == NULL)
            rpool_queue_group = &rpool->queue_group_arr[i];
        else
        {
            if (check_blacklist_cond (rpool->queue_group_arr[i].blacklist_conds, input_cond)
                && check_whitelist_cond (rpool->queue_group_arr[i].whitelist_conds, input_cond))
                    rpool_queue_group = &rpool->queue_group_arr[i];
            break;
        }     
    }
    if (rpool_queue_group == NULL)
    {
        while (atomic_exchange (&rpool->default_queue.occupied, 1) == 1)
        {
            // wait
        }
        return &rpool->default_queue;
    }
    rpool_queue_t *rpool_queue = NULL;
    unsigned int queue_idx = queue_val % rpool_queue_group->num_queues;
    while (rpool_queue == NULL)
    {
        if (atomic_exchange (&rpool_queue_group->queue_arr[queue_idx].occupied, 1) == 0)
            rpool_queue = &rpool_queue_group->queue_arr[queue_idx];
        else
            queue_idx = (queue_idx + 1) % rpool_queue_group->num_queues;
    }
    return rpool_queue;
}

unsigned int rpool_fetch_ninsts (rpool_t *rpool, ninst_t **ninst_ptr_list, unsigned int max_ninst_to_fetch)
{
    #ifdef DEBUG
    if (rpool == NULL)
    {
        FPRT (stderr, "ERROR: rpool_fetch_ninsts: rpool is NULL.\n");
        return 0;
    }
    if (ninst_ptr_list == NULL)
    {
        FPRT (stderr, "ERROR: rpool_fetch_ninsts: ninst_ptr_list is NULL.\n");
        return 0;
    }
    #endif
    if (max_ninst_to_fetch == 0)
        return 0;
    rpool_queue_t *rpool_queue = NULL;
    while (rpool_queue == NULL)
    {
        rpool_queue = get_queue_for_fetching (rpool, NULL);
    }
    unsigned int num_ninsts = pop_ninsts_from_queue (rpool_queue, ninst_ptr_list, max_ninst_to_fetch);
    atomic_store (&rpool_queue->occupied, 0);
    return num_ninsts;
}

void rpool_push_ninsts (rpool_t *rpool, ninst_t **ninst_ptr_list, unsigned int num_ninsts)
{
    #ifdef DEBUG
    if (rpool == NULL)
    {
        FPRT (stderr, "ERROR: rpool_push_ninsts: rpool is NULL.\n");
        return;
    }
    if (ninst_ptr_list == NULL)
    {
        FPRT (stderr, "ERROR: rpool_push_ninsts: ninst_ptr_list is NULL.\n");
        return;
    }
    #endif
    if (num_ninsts == 0)
        return;
    rpool_queue_t *rpool_queue = NULL;
    for (int i = 0; i < num_ninsts; i++)
    {
        ninst_t *ninst = ninst_ptr_list[i];
        aspen_layer_t *layer = ninst->ldata->layer;
        void* input_conds[NUM_RPOOL_CONDS] = {[RPOOL_DNN] = (void*)layer->dnn,
            [RPOOL_LAYER_TYPE] = (void*)layer->type, [RPOOL_LAYER_IDX] = (void*)(NULL + layer->layer_idx),
                [RPOOL_NASM] = (void*)ninst->ldata->nasm, [RPOOL_ASE] = NULL};
        while (rpool_queue == NULL)
        {
            rpool_queue = get_queue_for_storing (rpool, ninst_ptr_list[0]->ldata->layer->layer_idx, input_conds);
        }
        push_ninsts_to_queue (rpool_queue, ninst_ptr_list, num_ninsts);
        atomic_store (&rpool_queue->occupied, 0);
    }
}

void print_rpool_cond_list (void **input_list)
{
    if (input_list == NULL)
    {
        printf ("\t\tError: Ready Pool Condition List is NULL.\n");
        return;
    }
    printf ("\t\t");
    for (int i = 0; i < NUM_RPOOL_CONDS; i++)
    {
        printf("%s:%p ", rpool_cond_str[i], input_list[i]);
    }
    printf("\n");
}

void print_rpool_info (rpool_t *rpool)
{
    if (rpool == NULL)
    {
        printf ("Error: Ready Pool is NULL.\n");
        return;
    }
    while (atomic_exchange (&rpool->write_lock, 1) == 1)
    {
        // wait
    }
    printf("//////// Printing Ready Pool Info ////////\n");
    printf("Number of Queue Groups: %d\n", rpool->num_groups);
    printf("Sum of Queue Weight: %4.4f, Weights:\n", rpool->queue_group_weight_sum);
    for (int i = 0; i < rpool->num_groups; i++)
    {
        printf("%d:%4.4f, ", i, rpool->queue_group_weight_arr[i]);
    }
    printf("\n");
    for (int i = 0; i < rpool->num_groups; i++)
    {
        printf("\tQueue Group %d:\n", i);
        print_rpool_queue_group_info (&rpool->queue_group_arr[i]);
    }
    printf("Default Queue:\n");
    print_rpool_queue_info (&rpool->default_queue);
    printf("/////////////////////////////////////////\n");
    atomic_store (&rpool->write_lock, 0);
}
void print_rpool_queue_group_info (rpool_queue_group_t *rpool_queue_group)
{
    printf("\tQueue Group Info: %s\n", rpool_queue_group->queue_group_info);
    printf("\tBlacklist Conditions\n");
    print_rpool_cond_list (rpool_queue_group->blacklist_conds);
    printf("\tWhitelist Conditions\n");
    print_rpool_cond_list (rpool_queue_group->whitelist_conds);
    printf("\tNumber of Queues: %d\n", rpool_queue_group->num_queues);
    for (int i = 0; i < rpool_queue_group->num_queues; i++)
    {
        printf("\t\tQueue %d:\n", i);
        print_rpool_queue_info (&rpool_queue_group->queue_arr[i]);
    }
}

void print_rpool_queue_info (rpool_queue_t *rpool_queue)
{
    while (atomic_exchange (&rpool_queue->occupied, 1) == 1)
    {
        // wait
    }
    printf("\t\tQueue Info: %s\n", rpool_queue->queue_info);
    printf("\t\tNumber of ninsts stored: %d\n", rpool_queue->num_stored);
    printf("\t\tNumber of maximum ninsts: %d\n", rpool_queue->max_stored);
    printf("\t\tIndex range: %d ~ %d\n", rpool_queue->idx_start, rpool_queue->idx_end);
    printf("\t\tNinsts:\n\t\t\t");
    for (int i = 0; i < rpool_queue->num_stored; i++)
    {
        printf("%d:%p ", i, rpool_queue->ninst_ptr_arr[i]);
    }
    printf("\n");
    atomic_store (&rpool_queue->occupied, 0);
}
