#include "ase.h"

static unsigned int ase_thread_id_counter = 0;

ase_group_t *ase_group_init (unsigned int num_ase, int gpu_idx)
{
    if (gpu_idx >= 0 && gpu_idx >= aspen_num_gpus)
    {
        FPRT (stderr, "ERROR: rpool_init: gpu_idx %d is out of range... Falling back to CPU\n", gpu_idx);
        gpu_idx = -1;
    }
    ase_group_t *ase_group = (ase_group_t *) calloc (1, sizeof (ase_group_t));
    ase_group->num_ases = num_ase;
    if (gpu_idx < 0)
        ase_group->gpu_idx = -1;
    else
        ase_group->gpu_idx = gpu_idx;
    ase_group->ase_arr = (ase_t *) calloc (num_ase, sizeof (ase_t));
    for (int i = 0; i < num_ase; i++)
    {
        ase_init (&ase_group->ase_arr[i], ase_group->gpu_idx);
    }
    return ase_group;
}

void ase_group_set_rpool (ase_group_t *ase_group, rpool_t *rpool)
{
    if (ase_group == NULL)
    {
        FPRT (stderr, "ERROR: ase_group_set_rpool: ase_group is NULL\n");
        exit (1);
    }
    if (rpool == NULL)
    {
        FPRT (stderr, "ERROR: ase_group_set_rpool: rpool is NULL\n");
        exit (1);
    }
    if (ase_group->gpu_idx != rpool->gpu_idx)
    {
        FPRT (stderr, "ERROR: ase_group_set_rpool: ase_group->gpu_idx %d != rpool->gpu_idx %d\n", ase_group->gpu_idx, rpool->gpu_idx);
        exit (1);
    }
    for (int i = 0; i < ase_group->num_ases; i++)
    {
        ase_group->ase_arr[i].rpool = rpool;
        atomic_fetch_add (&rpool->ref_ases, 1);
    }
}

void ase_group_destroy (ase_group_t *ase_group)
{
    if (ase_group == NULL)
        return;
    for (int i = 0; i < ase_group->num_ases; i++)
    {
        if (ase_group->ase_arr[i].rpool != NULL)
            atomic_fetch_sub (&ase_group->ase_arr[i].rpool->ref_ases, 1);
        ase_destroy (&ase_group->ase_arr[i]);
    }
    free (ase_group->ase_arr);
    free (ase_group);
}

void ase_init (ase_t *ase, int gpu_idx)
{
    if (ase == NULL)
    {
        FPRT (stderr, "ERROR: ase_init: ase is NULL\n");
        exit (1);
    }
    if (gpu_idx >= 0 && gpu_idx >= aspen_num_gpus)
    {
        FPRT (stderr, "ERROR: rpool_init: gpu_idx %d is out of range... Falling back to CPU\n", gpu_idx);
        gpu_idx = -1;
    }
    ase->running = 0;
    ase->thread_id = atomic_fetch_add (&ase_thread_id_counter, 1);
    ase->rpool = NULL;
    ase->gpu_idx = gpu_idx;
    if (gpu_idx < 0)
        ase->scratchpad = aspen_calloc (ASE_SCRATCHPAD_SIZE, 1);
    else
        ase->scratchpad = aspen_gpu_calloc (ASE_SCRATCHPAD_SIZE, 1, gpu_idx);
    ase->thread_mutex = (pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER;
    ase->thread_cond = (pthread_cond_t)PTHREAD_COND_INITIALIZER;
    ase->ninst_cache = calloc (1, sizeof (rpool_queue_t));
    rpool_init_queue (ase->ninst_cache);
}

void ase_destroy (ase_t *ase)
{
    if (ase == NULL)
        return;
    if (ase->gpu_idx < 0)
        aspen_free (ase->scratchpad);
    else
        aspen_gpu_free (ase->scratchpad, ase->gpu_idx);
    rpool_destroy_queue (ase->ninst_cache);
    free (ase->ninst_cache);
}

void update_children (rpool_t *rpool, ninst_t *ninst)
{
    #ifdef DEBUG
    if (rpool == NULL || ninst == NULL)
    {
        FPRT (stderr, "Error: Invalid arguments to ase_update_children()\n");
        exit (1);
    }
    if (ninst->state != NINST_COMPLETED)
    {
        FPRT (stderr, "Error: ninst->state != NINST_STATE_COMPLETED in ase_update_children()\n");
        exit (1);
    }
    #endif
    if (ninst->state != NINST_COMPLETED)
        return;
    for (int i = 0; i < ninst->num_child_ninsts; i++)
    {
        ninst_t *child_ninst = ninst->child_ninst_arr[i];
        unsigned int num_parent_ninsts_completed = atomic_fetch_add (&child_ninst->num_parent_ninsts_completed, 1);
        if (num_parent_ninsts_completed == child_ninst->num_parent_ninsts - 1)
        {
            #ifdef DEBUG
            if (child_ninst->state != NINST_NOT_READY)
            {
                FPRT (stderr, "Error: child_ninst->state != NINST_NOT_READY in ase_update_children()\n");
                exit (1);
            }
            #endif
            child_ninst->state = NINST_READY;
            rpool_push_ninsts (rpool, &child_ninst, 1);
        }
    }
}
void push_first_layer_to_rpool (rpool_t *rpool, nasm_t *nasm)
{
    #ifdef DEBUG
    if (rpool == NULL || nasm == NULL)
    {
        FPRT (stderr, "Error: Invalid arguments to ase_push_first_layer_to_rpool()\n");
        exit (1);
    }
    #endif

    // Static Tensor Memory Allocation code
    // TODO: optimize memory usage by dynamically allocating memory only for the live ninsts.
    // Get sum of all memory requirements of a nasm
    size_t total_mem_req = 0;
    for (int i = 0; i < nasm->num_ldata; i++)
    {
        nasm_ldata_t *ldata = &nasm->ldata_arr[i];
        total_mem_req += ldata->out_mat_mem_size;
    }
    if (rpool->gpu_idx < 0)
    {
        nasm->data = aspen_calloc (total_mem_req, 1);
    }
    else
    {
        nasm->data = aspen_gpu_calloc (total_mem_req, 1, rpool->gpu_idx);
    }
    if (nasm->data == NULL)
    {
        FPRT (stderr, "Error: nasm->data == NULL in ase_push_first_layer_to_rpool()\n");
        exit (1);
    }
    for (int i = 0; i < nasm->num_ldata; i++)
    {
        nasm_ldata_t *ldata = &nasm->ldata_arr[i];
        set_ldata_out_mat_mem_pos (ldata);
    }
    nasm_ldata_t *ldata = &nasm->ldata_arr[0];
    for (int i = 0; i < ldata->num_ninst; i++)
    {
        ninst_t *ninst = &ldata->ninst_arr_start[i];
        if (ninst->state != NINST_NOT_READY)
        {
            FPRT (stderr, "Error: ninst->state != NINST_NOT_READY in ase_push_first_layer_to_rpool()\n");
            exit (1);
        }
        ninst->state = NINST_COMPLETED;
        update_children (rpool, ninst);
    }
}

void set_ldata_out_mat_mem_pos (nasm_ldata_t *ldata)
{
    #ifdef DEBUG
    if (ldata == NULL)
    {
        FPRT (stderr, "Error: Invalid arguments to set_ldata_out_mat_mem_pos()\n");
        exit (1);
    }
    #endif
    nasm_t *nasm = ldata->nasm;
    if (nasm->data == NULL)
    {
        FPRT (stderr, "Error: nasm->data == NULL in set_ldata_out_mat_mem_pos()\n");
        exit (1);
    }
    char *out_mat_mem_pos = nasm->data;
    for (int i = 0; i < ldata - nasm->ldata_arr; i++)
    {
        nasm_ldata_t *prev_ldata = &nasm->ldata_arr[i];
        out_mat_mem_pos += prev_ldata->out_mat_mem_size;
    }
    ldata->out_mat = out_mat_mem_pos;
    for (int i = 0; i < ldata->num_ninst; i++)
    {
        ninst_t *ninst = &ldata->ninst_arr_start[i];
        set_ninst_out_mat_mem_pos (ninst);
    }
}

void set_ninst_out_mat_mem_pos (ninst_t *ninst)
{
    #ifdef DEBUG
    if (ninst == NULL)
    {
        FPRT (stderr, "Error: Invalid arguments to set_ninst_out_mat_mem_pos()\n");
        exit (1);
    }
    #endif
    nasm_ldata_t *ldata = ninst->ldata;
    if (ninst->ldata->out_mat == NULL)
    {
        FPRT (stderr, "Error: ninst->ldata->out_mat == NULL in set_ninst_out_mat_mem_pos()\n");
        exit (1);
    }
    ninst->out_mat = (char*)ninst->ldata->out_mat 
        + (ninst->out_mat_pos[OUT_W]*ldata->out_mat_stride + ninst->out_mat_pos[OUT_H])
            *ldata->nasm->dnn->element_size;
    
}