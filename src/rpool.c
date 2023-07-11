#include "rpool.h"
unsigned int _rand_seed = 19960930;
inline unsigned int xors_rand ()
{
    _rand_seed ^= _rand_seed << 13;
    _rand_seed ^= _rand_seed >> 17;
    _rand_seed ^= _rand_seed << 5;
    return _rand_seed;
}

void rpool_init_queue (rpool_queue_t *rpool_queue)
{
    // atomic_store (&rpool_queue->occupied, 0);
    pthread_mutex_init (&rpool_queue->occupied_mutex, NULL);
    rpool_queue->idx_start = 0;
    rpool_queue->idx_end = 0;
    rpool_queue->max_stored = INIT_QUEUE_SIZE;
    rpool_queue->ninst_ptr_arr = calloc (INIT_QUEUE_SIZE, sizeof(ninst_t*));
}
void rpool_init_queue_group (rpool_queue_group_t *rpool_queue_group, char *queue_group_info, unsigned int num_queues)
{
    strncpy (rpool_queue_group->queue_group_info, queue_group_info, MAX_STRING_LEN-1);
    atomic_store (&rpool_queue_group->num_queues, num_queues);
    for (int i = 0; i < num_queues; i++)
    {
        rpool_init_queue (&rpool_queue_group->queue_arr[i]);
        rpool_queue_group->queue_arr[i].queue_group = rpool_queue_group;
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
        for (int i = 0; i < atomic_load (&rpool_queue_group->num_queues); i++)
            rpool_destroy_queue (&rpool_queue_group->queue_arr[i]);
    }
}

rpool_t *rpool_init (int gpu_idx)
{
    if (gpu_idx >= 0 && gpu_idx >= aspen_num_gpus)
    {
        FPRT (stderr, "ERROR: rpool_init: gpu_idx %d is out of range... Falling back to CPU\n", gpu_idx);
        gpu_idx = -1;
    }
    rpool_t *rpool = calloc (1, sizeof(rpool_t));
    rpool->ref_ases = 0;
    rpool->num_groups = 0;
    rpool->queue_group_weight_sum = 0;
    bzero (rpool->queue_group_weight_arr, sizeof(float)*MAX_QUEUE_GROUPS);
    rpool_init_queue (&rpool->default_queue);
    if (gpu_idx < 0)
        rpool->gpu_idx = -1;
    else
        rpool->gpu_idx = gpu_idx;
    return rpool;
}

void rpool_destroy (rpool_t *rpool)
{
    if (rpool == NULL)
        return;
    if (atomic_load(&rpool->ref_ases) > 0)
    {
        FPRT (stderr, "ERROR: rpool_destroy: rpool is still referenced by %d ases.\n", atomic_load(&rpool->ref_ases));
        return;
    }
    for (int i = 0; i < atomic_load (&rpool->num_groups); i++)
        rpool_destroy_queue_group (&rpool->queue_group_arr[i]);
    rpool_destroy_queue (&rpool->default_queue);
    free (rpool);
}

rpool_queue_group_t *get_queue_group_from_nasm (rpool_t *rpool, nasm_t *nasm)
{
    if (rpool == NULL)
    {
        FPRT (stderr, "ERROR: get_queue_group_from_nasm: rpool is NULL.\n");
        return NULL;
    }
    if (nasm == NULL)
    {
        FPRT (stderr, "ERROR: get_queue_group_from_nasm: nasm is NULL.\n");
        return NULL;
    }
    for (int i = 0; i < rpool->num_groups; i++)
    {
        if (rpool->queue_group_arr[i].whitelist_conds[RPOOL_NASM] == nasm)
            return &rpool->queue_group_arr[i];
    }
    return NULL;
}

int get_queue_group_idx_from_nasm (rpool_t *rpool, nasm_t *nasm)
{
    if (rpool == NULL)
    {
        FPRT (stderr, "ERROR: get_queue_group_idx_from_nasm: rpool is NULL.\n");
        return -1;
    }
    if (nasm == NULL)
    {
        FPRT (stderr, "ERROR: get_queue_group_idx_from_nasm: nasm is NULL.\n");
        return -1;
    }
    for (int i = 0; i < rpool->num_groups; i++)
    {
        if (rpool->queue_group_arr[i].whitelist_conds[RPOOL_NASM] == nasm)
            return i;
    }
    return -1;
}

void set_queue_group_weight (rpool_t *rpool, rpool_queue_group_t *rpool_queue_group, float weight)
{
    if (rpool == NULL)
    {
        FPRT (stderr, "ERROR: set_queue_group_weight: rpool is NULL.\n");
        return;
    }
    if (rpool_queue_group == NULL)
    {
        FPRT (stderr, "ERROR: set_queue_group_weight: rpool_queue_group is NULL.\n");
        return;
    }
    if (weight < 0)
    {
        FPRT (stderr, "ERROR: set_queue_group_weight: weight must not be negative. Cannot set weight for queue group \"%s\".\n", 
            rpool_queue_group->queue_group_info);
        return;
    }
    float old_weight = rpool->queue_group_weight_arr[rpool_queue_group->idx];
    rpool->queue_group_weight_arr[rpool_queue_group->idx] = weight;
    rpool->queue_group_weight_sum += weight - old_weight;
}

void queue_group_add_queues (rpool_queue_group_t *rpool_queue_group, unsigned int num_queues)
{
    if (rpool_queue_group == NULL)
    {
        FPRT (stderr, "ERROR: queue_group_add_queues: rpool_queue_group is NULL.\n");
        return;
    }
    if (num_queues == 0)
        return;
    if (atomic_load (&rpool_queue_group->num_queues) + num_queues > MAX_NUM_QUEUES)
        return;
    for (int i = 0; i < num_queues; i++)
    {
        rpool_init_queue (&rpool_queue_group->queue_arr[rpool_queue_group->num_queues + i]);
        rpool_queue_group->queue_arr[rpool_queue_group->num_queues + i].queue_group = rpool_queue_group;
    }
    atomic_fetch_add (&rpool_queue_group->num_queues, num_queues);
}

void add_ref_ases (rpool_t *rpool, unsigned int num_ases)
{
    if (rpool == NULL)
    {
        FPRT (stderr, "ERROR: add_ref_ases: rpool is NULL.\n");
        return;
    }
    atomic_fetch_add (&rpool->ref_ases, num_ases);
    for (int i = 0; i < rpool->num_groups; i++)
    {
        if (rpool->queue_group_arr[i].whitelist_conds[RPOOL_NASM] != NULL)
        {
            nasm_t *nasm = rpool->queue_group_arr[i].whitelist_conds[RPOOL_NASM];
            unsigned int num_queues = rpool->ref_ases * NUM_LAYERQUEUE_PER_ASE * nasm->dnn->num_layers  * NUM_QUEUE_PER_LAYER;
            if (num_queues < 1)
                num_queues = 1;
            if (num_queues > atomic_load (&rpool->queue_group_arr[i].num_queues))
                queue_group_add_queues (&rpool->queue_group_arr[i], num_queues - rpool->queue_group_arr[i].num_queues);
        }
    }
}

void rpool_add_nasm_raw_input (rpool_t *rpool, nasm_t* nasm, float weight, void* input_data)
{
    if (rpool == NULL)
    {
        FPRT (stderr, "ERROR: rpool_add_nasm_raw_input: rpool is NULL.\n");
        return;
    }
    if (nasm == NULL)
    {
        FPRT (stderr, "ERROR: rpool_add_nasm_raw_input: nasm is NULL.\n");
        return;
    }
    if (weight <= 0)
    {
        FPRT (stderr, "ERROR: rpool_add_nasm_raw_input: weight must be positive. Cannot add nasm \"%s_nasm_%d\".\n", nasm->dnn->name, nasm->nasm_id);
        return;
    }
    char info_str[MAX_STRING_LEN*2];
    void *whitelist[NUM_RPOOL_CONDS] = {NULL};
    whitelist [RPOOL_NASM] = nasm;
    sprintf (info_str, "%s_%s_%d", nasm->dnn->name, "nasm", nasm->nasm_id);
    unsigned int num_queues = rpool->ref_ases * NUM_LAYERQUEUE_PER_ASE * nasm->dnn->num_layers*  NUM_QUEUE_PER_LAYER;
    if (num_queues < 1)
        num_queues = 1;
    nasm->gpu_idx = rpool->gpu_idx;
    rpool_add_queue_group (rpool, info_str, num_queues, weight, NULL, whitelist);
    push_first_layer_to_rpool (rpool, nasm, input_data);
}

void rpool_add_nasm (rpool_t *rpool, nasm_t* nasm, float weight, char *input_filename)
{
    aspen_dnn_t *dnn = nasm->dnn;
    aspen_layer_t *first_layer = &dnn->layers[0];
    void *data = NULL;
    unsigned int input_params[NUM_PARAM_ELEMENTS] = {0};
    input_params[BATCH] = nasm->batch_size;
    if (first_layer->params[OUT_C] != 0 && first_layer->params[OUT_H] != 0 && first_layer->params[OUT_W] != 0)
    {
        input_params[OUT_C] = first_layer->params[OUT_C];
        input_params[OUT_H] = first_layer->params[OUT_H];
        input_params[OUT_W] = first_layer->params[OUT_W];
        data = aspen_load_input_NHWC (input_filename, input_params, sizeof(float));
    }
    else if (first_layer->params[MAT_M] != 0)
    {
        input_params[MAT_M] = first_layer->params[MAT_M];
        input_params[MAT_N] = nasm->tr_seq_len;
        data = aspen_load_input (input_filename, input_params, sizeof(float));
    }
    else
    {
        FPRT (stderr, "ERROR: rpool_add_nasm: first layer of dnn \"%s\" does not have output dimensions. Cannot add nasm \"%s_nasm_%d\".\n", 
            dnn->name, dnn->name, nasm->nasm_id);
        return;
    }
    rpool_add_nasm_raw_input (rpool, nasm, weight, data);
    aspen_free (data); 
}

void rpool_pop_all_nasm (rpool_t *rpool, nasm_t *nasm)
{
    int queue_group_idx = get_queue_group_idx_from_nasm (rpool, nasm);
    if (queue_group_idx == -1)
        FPRT (stderr, "ERROR: rpool_pop_all_nasm: nasm \"%s_nasm_%d\" is not in rpool.\n", nasm->dnn->name, nasm->nasm_id);
    rpool_queue_group_t *queue_group = &rpool->queue_group_arr[queue_group_idx];
    unsigned int num_queues = queue_group->num_queues;
    for (int i = 0; i < num_queues; i++)
    {
        pthread_mutex_lock (&queue_group->queue_arr[i].occupied_mutex);
        queue_group->queue_arr[i].idx_start = 0;
        queue_group->queue_arr[i].idx_end = 0;
        queue_group->queue_arr[i].num_stored = 0;
        pthread_mutex_unlock (&queue_group->queue_arr[i].occupied_mutex);
    }
}

void rpool_finish_nasm (rpool_t *rpool, nasm_t *nasm)
{
    set_nasm_to_finished (nasm);
    rpool_pop_all_nasm (rpool, nasm);
}

void rpool_reset_nasm (rpool_t *rpool, nasm_t *nasm, float weight)
{
    reset_nasm (nasm);
    rpool_pop_all_nasm (rpool, nasm);
    int queue_group_idx = get_queue_group_idx_from_nasm (rpool, nasm);
    if (queue_group_idx == -1)
        FPRT (stderr, "ERROR: rpool_reset_nasm: nasm \"%s_nasm_%d\" is not in rpool.\n", nasm->dnn->name, nasm->nasm_id);
    rpool->queue_group_weight_arr[queue_group_idx] = weight;
    rpool->queue_group_weight_sum += weight;
    push_first_layer_to_rpool (rpool, nasm, NULL);
}

void rpool_add_queue_group 
    (rpool_t *rpool, char *queue_group_info, unsigned int num_queues, float weight, void **blacklist, void **whitelist)
{
    if (rpool == NULL)
    {
        FPRT (stderr, "ERROR: rpool_add_queue_group: rpool is NULL.\n");
        return;
    }
    if (weight <= 0)
    {
        FPRT (stderr, "ERROR: rpool_add_queue_group: weight must be positive. Cannot add queue group \"%s\".\n", queue_group_info);
        return;
    }
    if (num_queues > MAX_NUM_QUEUES)
        num_queues = MAX_NUM_QUEUES;
    unsigned int num_groups = rpool->num_groups;
    if (num_groups + 1 >= MAX_QUEUE_GROUPS)
    {
        FPRT (stderr, "ERROR: rpool_add_queue_group: max number of queue groups (%d) reached. Cannot add queue group \"%s\".\n", 
            MAX_QUEUE_GROUPS, queue_group_info);
        return;
    }
    rpool_init_queue_group (&rpool->queue_group_arr[num_groups], queue_group_info, num_queues);
    rpool->queue_group_arr[num_groups].idx = num_groups;
    rpool_queue_group_set_blacklist (&rpool->queue_group_arr[num_groups], blacklist);
    rpool_queue_group_set_whitelist (&rpool->queue_group_arr[num_groups], whitelist);
    rpool->queue_group_weight_arr[num_groups] = weight;
    rpool->queue_group_weight_sum += weight;
    atomic_store (&rpool->num_groups, num_groups + 1);
}
void rpool_queue_group_set_blacklist (rpool_queue_group_t *rpool_queue_group, void **blacklist)
{
    if (rpool_queue_group == NULL)
    {
        FPRT (stderr, "ERROR: rpool_queue_group_set_blacklist: rpool_queue_group is NULL.\n");
        return;
    }
    if (blacklist == NULL)
    {
        for (int i = 0; i < NUM_RPOOL_CONDS; i++)
            rpool_queue_group->blacklist_conds[i] = NULL;
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
    if (whitelist == NULL)
    {
        for (int i = 0; i < NUM_RPOOL_CONDS; i++)
            rpool_queue_group->whitelist_conds[i] = NULL;
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
        else if (whitelist[i] != input_cond[i])
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
    // if (rpool_queue->queue_group != NULL)
    //     atomic_fetch_sub (&rpool_queue->queue_group->num_ninsts, num_ninsts);    
    return num_ninsts;
}
unsigned int pop_ninsts_from_queue_enabled (rpool_queue_t *rpool_queue, ninst_t **ninst_ptr_list, unsigned int max_ninsts_to_get, int *enabled_device)
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
    // if (rpool_queue->queue_group != NULL)
    //     atomic_fetch_sub (&rpool_queue->queue_group->num_ninsts, num_ninsts);    
    return num_ninsts;
}
unsigned int pop_ninsts_from_queue_back (rpool_queue_t *rpool_queue, ninst_t **ninst_ptr_list, unsigned int max_ninsts_to_get)
{
    #ifdef DEBUG
    if (rpool_queue == NULL)
    {
        FPRT (stderr, "ERROR: pop_nists_from_queue_back: rpool_queue is NULL.\n");
        return 0;
    }
    if (ninst_ptr_list == NULL)
    {
        FPRT (stderr, "ERROR: pop_nists_from_queue_back: ninst_ptr_list is NULL.\n");
        return 0;
    }
    #endif
    unsigned int num_ninsts = 0;
    unsigned int i = rpool_queue->idx_end;
    for (; num_ninsts < rpool_queue->num_stored; num_ninsts++)
    {
        if (num_ninsts >= max_ninsts_to_get)
            break;
        i--;
        if (i == -1)
            i = rpool_queue->max_stored - 1;
        ninst_ptr_list[num_ninsts] = rpool_queue->ninst_ptr_arr[i];
    }
    rpool_queue->idx_end = i;
    rpool_queue->num_stored -= num_ninsts;
    // if (rpool_queue->queue_group != NULL)
    //     atomic_fetch_sub (&rpool_queue->queue_group->num_ninsts, num_ninsts);
    return num_ninsts;
}
void check_and_update_queue_size (rpool_queue_t *rpool_queue, unsigned int num_to_add)
{
    if (rpool_queue->num_stored + num_to_add > rpool_queue->max_stored)
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
}
void update_queue_size (rpool_queue_t *rpool_queue, unsigned int num_to_add)
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
        update_queue_size (rpool_queue, num_ninsts);
    unsigned i = rpool_queue->idx_end;
    for (int j = 0; j < num_ninsts; j++)
    {
        rpool_queue->ninst_ptr_arr[i] = ninst_ptr_list[j];
        i++;
        if (i == rpool_queue->max_stored)
            i = 0;
    }
    rpool_queue->idx_end = i;
    rpool_queue->num_stored += num_ninsts;
    // if (rpool_queue->queue_group != NULL)
    //     atomic_fetch_add (&rpool_queue->queue_group->num_ninsts, num_ninsts);
}
void push_ninsts_to_queue_front (rpool_queue_t *rpool_queue, ninst_t **ninst_ptr_list, unsigned int num_ninsts)
{
    #ifdef DEBUG
    if (rpool_queue == NULL)
    {
        FPRT (stderr, "ERROR: push_ninsts_to_queue_back: rpool_queue is NULL.\n");
        return;
    }
    if (ninst_ptr_list == NULL)
    {
        FPRT (stderr, "ERROR: push_ninsts_to_queue_back: ninst_ptr_list is NULL.\n");
        return;
    }
    #endif
    check_and_update_queue_size (rpool_queue, num_ninsts);
    unsigned int i = rpool_queue->idx_start;
    for (int j = num_ninsts - 1; j >= 0; j--)
    {
        i--;
        if (i == -1)
            i = rpool_queue->max_stored - 1;
        rpool_queue->ninst_ptr_arr[i] = ninst_ptr_list[j];
    }
    rpool_queue->idx_start = i;
    rpool_queue->num_stored += num_ninsts;
    // if (rpool_queue->queue_group != NULL)
    //     atomic_fetch_add (&rpool_queue->queue_group->num_ninsts, num_ninsts);
}
rpool_queue_t *get_queue_for_fetching (rpool_t *rpool, void **input_cond, unsigned int dse_idx)
{
    #ifdef DEBUG
    if (rpool == NULL)
    {
        FPRT (stderr, "ERROR: get_queue_for_fetching: rpool is NULL.\n");
        return NULL;
    }
    #endif
    if (rpool->default_queue.num_stored > 0)
    {
        // if (atomic_exchange (&rpool->default_queue.occupied, 1) == 0)
        if (pthread_mutex_trylock (&rpool->default_queue.occupied_mutex) == 0)
            return &rpool->default_queue;
    }
    rpool_queue_group_t *rpool_queue_group = &rpool->queue_group_arr[0];
    // unsigned int num_tries = 0;
    // unsigned int num_groups = rpool->num_groups;
    // while (rpool_queue_group == NULL)
    // {
    //     float rand_weight = ((xors_rand () % RAND_MAX) / (float) RAND_MAX) * rpool->queue_group_weight_sum;
    //     for (int i = 0; i < num_groups; i++)
    //     {
    //         if (rand_weight < rpool->queue_group_weight_arr[i])
    //         {
    //             if (input_cond == NULL)
    //                 rpool_queue_group = &rpool->queue_group_arr[i];
    //             else
    //             {
    //                 if (check_blacklist_cond (rpool->queue_group_arr[i].blacklist_conds, input_cond)
    //                 && check_whitelist_cond (rpool->queue_group_arr[i].whitelist_conds, input_cond))
    //                     rpool_queue_group = &rpool->queue_group_arr[i];
    //             }
    //             rpool_queue_group = &rpool->queue_group_arr[i];
    //             break;
    //         }
    //         rand_weight -= rpool->queue_group_weight_arr[i];
    //     }
    //     // if (rpool_queue_group != NULL && atomic_load (&rpool_queue_group->num_ninsts) == 0)
    //     //     rpool_queue_group = NULL;
    //     num_tries++;
    //     if (num_tries > 10)
    //     {
    //         return NULL;
    //     }
    // }
    // atomic_fetch_add (&rpool_queue_group->num_fetched, 1);
    unsigned int num_queues = atomic_load (&rpool_queue_group->num_queues);
    unsigned int num_ase = rpool->ref_ases > 0 ? rpool->ref_ases : 1;
    unsigned int queue_idx = num_queues * dse_idx / num_ase;
    for (int i = 0; i < num_queues; i++)
    {
        rpool_queue_t *rpool_queue = &rpool_queue_group->queue_arr[queue_idx];
        // printf ("queue %d: num_stored: %d\n", i, rpool_queue->num_stored);
        if (rpool_queue->num_stored > 0)
        {
            // if (atomic_exchange (&rpool_queue->occupied, 1) == 0)
            if (pthread_mutex_trylock (&rpool_queue->occupied_mutex) == 0)
            {
                return rpool_queue;
            }
        }
        queue_idx++;
        if (queue_idx == num_queues)
            queue_idx = 0;
    }
    return NULL;
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
    rpool_queue_group_t *rpool_queue_group = &rpool->queue_group_arr[0];
    // unsigned int num_groups = atomic_load (&rpool->num_groups);
    // for (int i = 0; i < num_groups; i++)
    // {
    //     if (input_cond == NULL)
    //         rpool_queue_group = &rpool->queue_group_arr[i];
    //     else
    //     {
    //         if (check_blacklist_cond (rpool->queue_group_arr[i].blacklist_conds, input_cond)
    //             && check_whitelist_cond (rpool->queue_group_arr[i].whitelist_conds, input_cond))
    //         {
    //             rpool_queue_group = &rpool->queue_group_arr[i];
    //             break;
    //         }
    //     }     
    // }
    // if (rpool_queue_group == NULL)
    // {
    //     // while (atomic_exchange (&rpool->default_queue.occupied, 1) == 1)
    //     // {
    //     //     // wait
    //     // }
    //     pthread_mutex_lock (&rpool->default_queue.occupied_mutex);
    //     return &rpool->default_queue;
    // }
    rpool_queue_t *rpool_queue = NULL;
    unsigned int num_queues = atomic_load (&rpool_queue_group->num_queues);
    #ifdef DEBUG
    if (num_queues == 0)
    {
        FPRT (stderr, "ERROR: get_queue_for_storing: num_queues is 0.\n");
        return NULL;
    }
    #endif
    unsigned int queue_idx = queue_val % num_queues;
    while (rpool_queue == NULL)
    {
        // if (atomic_exchange (&rpool_queue_group->queue_arr[queue_idx].occupied, 1) == 0)
        if (pthread_mutex_trylock (&rpool_queue_group->queue_arr[queue_idx].occupied_mutex) == 0)
            rpool_queue = &rpool_queue_group->queue_arr[queue_idx];
        else
        {
            queue_idx++;
            if (queue_idx == num_queues)
                queue_idx = 0;
        }
    }
    return rpool_queue;
}

unsigned int rpool_fetch_ninsts (rpool_t *rpool, ninst_t **ninst_ptr_list, unsigned int max_ninst_to_fetch, unsigned int dse_idx)
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
    rpool_queue = get_queue_for_fetching (rpool, NULL, dse_idx);
    if (rpool_queue == NULL)
        return 0;
    unsigned int num_ninsts = pop_ninsts_from_queue (rpool_queue, ninst_ptr_list, max_ninst_to_fetch);
    // atomic_store (&rpool_queue->occupied, 0);
    pthread_mutex_unlock (&rpool_queue->occupied_mutex);
    return num_ninsts;
}

unsigned int rpool_fetch_ninsts_enabled (rpool_t *rpool, ninst_t **ninst_ptr_list, unsigned int max_ninst_to_fetch, unsigned int dse_idx, int *enabled_device)
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
    rpool_queue = get_queue_for_fetching (rpool, NULL, dse_idx);
    if (rpool_queue == NULL)
        return 0;
    unsigned int num_ninsts = pop_ninsts_from_queue_enabled (rpool_queue, ninst_ptr_list, max_ninst_to_fetch, enabled_device);
    // atomic_store (&rpool_queue->occupied, 0);
    pthread_mutex_unlock (&rpool_queue->occupied_mutex);
    return num_ninsts;
}

void rpool_push_ninsts (rpool_t *rpool, ninst_t **ninst_ptr_list, unsigned int num_ninsts, unsigned int dse_idx)
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
    unsigned int i = 0;
    if (num_ninsts%NINST_PUSH_BATCH_SIZE != 0)
    {
        ninst_t *ninst = ninst_ptr_list[i];
        aspen_layer_t *layer = ninst->ldata->layer;
        unsigned int queue_val = (dse_idx * layer->dnn->num_layers * NUM_LAYERQUEUE_PER_ASE + (layer->layer_idx - 1)) * NUM_QUEUE_PER_LAYER;
        if (queue_val < 0)
            queue_val = 0;
        void* input_conds[NUM_RPOOL_CONDS] = {[RPOOL_DNN] = (void*)layer->dnn,
            [RPOOL_LAYER_TYPE] = (void*)layer->type, [RPOOL_LAYER_IDX] = (void*)(NULL + layer->layer_idx),
                [RPOOL_NASM] = (void*)ninst->ldata->nasm, [RPOOL_ASE] = NULL};
        
        rpool_queue = get_queue_for_storing (rpool, queue_val, input_conds);
        
        push_ninsts_to_queue (rpool_queue, &ninst_ptr_list[i], num_ninsts%NINST_PUSH_BATCH_SIZE);
        // atomic_store (&rpool_queue->occupied, 0);
        pthread_mutex_unlock (&rpool_queue->occupied_mutex);
        i += num_ninsts%NINST_PUSH_BATCH_SIZE;
    }
    for (; i < num_ninsts; i += NINST_PUSH_BATCH_SIZE)
    {
        ninst_t *ninst = ninst_ptr_list[i];
        aspen_layer_t *layer = ninst->ldata->layer;
        unsigned int queue_val = (dse_idx * layer->dnn->num_layers * NUM_LAYERQUEUE_PER_ASE + (layer->layer_idx - 1)) * NUM_QUEUE_PER_LAYER;
        if (queue_val < 0)
            queue_val = 0;
        void* input_conds[NUM_RPOOL_CONDS] = {[RPOOL_DNN] = (void*)layer->dnn,
            [RPOOL_LAYER_TYPE] = (void*)layer->type, [RPOOL_LAYER_IDX] = (void*)(NULL + layer->layer_idx),
                [RPOOL_NASM] = (void*)ninst->ldata->nasm, [RPOOL_ASE] = NULL};
        rpool_queue = get_queue_for_storing (rpool, queue_val, input_conds);
        push_ninsts_to_queue (rpool_queue, &ninst_ptr_list[i], NINST_PUSH_BATCH_SIZE);
        // atomic_store (&rpool_queue->occupied, 0);
        pthread_mutex_unlock (&rpool_queue->occupied_mutex);
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
        if (i == RPOOL_NASM && input_list[i] != NULL)
            printf("%s:%d ", rpool_cond_str[i], ((nasm_t*)input_list[i])->nasm_id);
        else
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
    unsigned int num_groups = atomic_load (&rpool->num_groups);
    unsigned int ref_ases = atomic_load (&rpool->ref_ases);
    printf("//////// Printing Ready Pool Info ////////\n");
    printf("Number of referencing ASEs: %d\n", ref_ases);
    printf("Number of Queue Groups: %d\n", num_groups);
    printf("GPU index: %d\n", rpool->gpu_idx);
    printf("Sum of Queue Weight: %4.4f\nWeights: ", rpool->queue_group_weight_sum);
    for (int i = 0; i < num_groups; i++)
    {
        printf("%d:%4.4f, ", i, rpool->queue_group_weight_arr[i]);
    }
    printf("\n");
    printf("Queue Groups:\n");
    for (int i = 0; i < num_groups; i++)
    {
        printf("\tQueue Group %d:\n", i);
        print_rpool_queue_group_info (&rpool->queue_group_arr[i]);
    }
    printf("Default Queue:\n");
    print_rpool_queue_info (&rpool->default_queue);
    printf("/////////////////////////////////////////\n");
}
void print_rpool_queue_group_info (rpool_queue_group_t *rpool_queue_group)
{
    printf("\tQueue Group Info: %s", rpool_queue_group->queue_group_info);
    // printf("\t, Stored: %d, Num Fetched: %d", 
    //     atomic_load (&rpool_queue_group->num_ninsts), atomic_load (&rpool_queue_group->num_fetched));
    printf("\n");
    printf("\tBlacklist Conditions\n");
    print_rpool_cond_list (rpool_queue_group->blacklist_conds);
    printf("\tWhitelist Conditions\n");
    print_rpool_cond_list (rpool_queue_group->whitelist_conds);
    unsigned int num_queues = atomic_load (&rpool_queue_group->num_queues);
    printf("\tNumber of Queues: %d\n", num_queues);
    for (int i = 0; i < num_queues; i++)
    {
        printf("\t\tQueue %d:\n", i);
        print_rpool_queue_info (&rpool_queue_group->queue_arr[i]);
    }
}

void print_rpool_queue_info (rpool_queue_t *rpool_queue)
{
    // while (atomic_exchange (&rpool_queue->occupied, 1) == 1)
    // {
    //     // wait
    // }
    pthread_mutex_lock (&rpool_queue->occupied_mutex);
    printf("\t\tNumber of ninsts stored: %d\n", rpool_queue->num_stored);
    printf("\t\tNumber of maximum ninsts: %d\n", rpool_queue->max_stored);
    printf("\t\tIndex range: %d ~ %d\n", rpool_queue->idx_start, rpool_queue->idx_end);
    printf("\t\tNinsts:\n\t\t\t");
    for (int i = 0; i < rpool_queue->num_stored; i++)
    {
        ninst_t *ninst = rpool_queue->ninst_ptr_arr[(i + rpool_queue->idx_start)%rpool_queue->max_stored];
        printf("%d:(N%d:L%ld:%d) ", i, ninst->ldata->nasm->nasm_id,
            ninst->ldata - ninst->ldata->nasm->ldata_arr, ninst->ninst_idx);
    }
    printf("\n");
    // atomic_store (&rpool_queue->occupied, 0);
    pthread_mutex_unlock (&rpool_queue->occupied_mutex);
}
