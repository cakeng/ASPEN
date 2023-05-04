#include "ase.h"

static _Atomic unsigned int ase_thread_id_counter = 0;

void *ase_thread_runtime (void* thread_info)
{
    ase_t *ase = (ase_t*) thread_info;
    pthread_mutex_lock(&ase->thread_mutex);
    while (ase->kill == 0)
    {
        // print_rpool_info (ase->rpool);
        // print_rpool_queue_info (ase->ninst_cache);
        if (ase->run == 0)
        {   
            pthread_cond_wait(&ase->thread_cond, &ase->thread_mutex); 
        }
        // if ((ase->ninst_cache->num_stored < ASE_NINST_CACHE_BALLANCE - ASE_NINST_CACHE_DIFF) || 
        //     ase->ninst_cache->num_stored == 0)
        if (ase->target == NULL)
        {
            rpool_fetch_ninsts (ase->rpool, &ase->target, 1);
            if (ase->target == NULL)
                continue;
            // unsigned int fetch_num = 
            //     rpool_fetch_ninsts (ase->rpool, ase->scratchpad, ASE_NINST_CACHE_BALLANCE - ase->ninst_cache->num_stored);
            // push_ninsts_to_queue (ase->ninst_cache, ase->scratchpad, fetch_num);
            // PRT ("Thread %d fetched %d ninsts from rpool\n", ase->thread_id, fetch_num);
            // #ifdef DEBUG
            // PRT ("Thread %d fetched %d ninsts from rpool\n", ase->thread_id, fetch_num);
            // print_rpool_info (ase->rpool);
            // print_rpool_queue_info (ase->ninst_cache);
            // #endif
        }
        // else if (ase->ninst_cache->num_stored > ASE_NINST_CACHE_BALLANCE + ASE_NINST_CACHE_DIFF)
        // if (ase->ninst_cache->num_stored > 0)
        // {
        //     // unsigned int push_num = 
        //     //     pop_ninsts_from_queue_back (ase->ninst_cache, ase->scratchpad, ase->ninst_cache->num_stored - ASE_NINST_CACHE_BALLANCE);
        //     rpool_push_ninsts (ase->rpool, ase->ninst_cache->ninst_ptr_arr, ase->ninst_cache->num_stored);
        //     ase->ninst_cache->num_stored = 0;
        //     ase->ninst_cache->idx_end = 0;
        //     // PRT ("Thread %d pushed %d ninsts to rpool\n", ase->thread_id, push_num);
        //     // #ifdef DEBUG
        //     // PRT ("Thread %d pushed %d ninsts to rpool\n", ase->thread_id, push_num);
        //     // print_rpool_info (ase->rpool);
        //     // print_rpool_queue_info (ase->ninst_cache);
        //     // #endif
        // }

        // unsigned int num_ninsts = ase->ninst_cache->num_stored;
        // for (int i = 0; i < num_ninsts; i++)
        // {
        //     ninst_t *ninst;
        //     pop_ninsts_from_queue (ase->ninst_cache, &ninst, 1);
            // PRT ("Thread %d running ninst #%d - N%d:L%d:%d\n", ase->thread_id, i,
            //         ninst->ldata->nasm->nasm_id, ninst->ldata->layer->layer_idx, ninst->ninst_idx);
            // #ifdef DEBUG
            // if (ninst == NULL)
            // {
            //     FPRT (stderr, "ERROR: ase_thread_runtime: ninst is NULL\n");
            //     assert (0);
            // }
            // else 
            // {
            //     PRT ("Thread %d running ninst #%d - N%d:L%d:%d\n", ase->thread_id, i,
            //         ninst->ldata->nasm->nasm_id, ninst->ldata->layer->layer_idx, ninst->ninst_idx);
            // }
            // if (ninst->state != NINST_READY)
            // {
            //     FPRT (stderr, "ERROR: ase_thread_runtime: ninst->state != NINST_READY\n");
            //     assert (0);
            // }
            // #endif
            // Execute.
            ninst_t *ninst = ase->target;
            ase->target = NULL;
            #ifdef DEBUG 
            if (ninst->state != NINST_READY)
            {
                FPRT (stderr, "Error: ninst->state != NINST_READY in ase_thread_runtime()\n");
                assert (0);
            }
            #endif
            switch (ninst->ldata->layer->type)
            {
                case CONV_LAYER:
                    tiled_conv2d (ninst, ase);
                    break;
                case MAXPOOL_LAYER:
                    tiled_maxpool2d (ninst, ase);
                    break;
                case AVGPOOL_LAYER:
                    tiled_avgpool2d (ninst, ase);
                    break;
                case FC_LAYER:
                    tiled_fully_connected (ninst, ase);
                    break;
                case RESIDUAL_LAYER:
                    tiled_residual (ninst, ase);
                    break;
                case SOFTMAX_LAYER:
                    tiled_softmax (ninst, ase);
                    break;
                case MATMUL_LAYER:
                    tiled_matmul (ninst, ase);
                    break;
                case LAYERNORM_LAYER:
                    tiled_layernorm (ninst, ase);
                    break;
                case K_ATTENTION_LAYER:
                    tiled_k_attention (ninst, ase);
                    break;
                case V_ATTENTION_LAYER:
                    tiled_v_attention (ninst, ase);
                    break;
                default:
                    // FPRT (stderr, "ERROR: ase_thread_runtime: layer type %s is not supported\n", layer_type_str[ninst->ldata->layer->type]);
                    break;
            }
            ninst->state = NINST_COMPLETED;
            unsigned int num_ninst_completed = atomic_fetch_add (&ninst->ldata->num_ninst_completed, 1);
            if (num_ninst_completed == ninst->ldata->num_ninst - 1)
            {
                // printf ("\t\tThread %d completed layer %d of nasm %d\n", 
                //     ase->thread_id, ninst->ldata->layer->layer_idx, ninst->ldata->nasm->nasm_id);
                nasm_t *nasm = ninst->ldata->nasm;
                unsigned int num_ldata_completed = atomic_fetch_add (&nasm->num_ldata_completed, 1);
                if (num_ldata_completed == nasm->num_ldata - 1)
                {
                    // printf ("\t\tSignaling nasm completion...\n");
                    // All layers of the nasm is completed.
                    rpool_queue_group_t *rpool_queue_group 
                        = get_queue_group_from_nasm (ase->rpool, ninst->ldata->nasm);
                    set_queue_group_weight (ase->rpool, rpool_queue_group, 0);
                    pthread_mutex_lock (&nasm->nasm_mutex);
                    pthread_cond_signal (&nasm->nasm_cond);
                    pthread_mutex_unlock (&nasm->nasm_mutex);
                }
            }
            // update_children_to_cache (ase->ninst_cache, ninst);
            update_children_but_prioritize_ase_target (ase->rpool, ninst, ase);
        // }
    }
    return NULL;
}

ase_group_t *ase_group_init (unsigned int num_ase, int gpu_idx)
{
    if (gpu_idx >= 0 && gpu_idx >= aspen_num_gpus)
    {
        FPRT (stderr, "ERROR: ase_group_init: gpu_idx %d is out of range... Falling back to CPU\n", gpu_idx);
        gpu_idx = -1;
    }
    else if (gpu_idx >= 0 && gpu_idx < aspen_num_gpus)
    {
        num_ase = num_ase > GPU_RUN_STREAM_NUM ? GPU_RUN_STREAM_NUM : num_ase;
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
        assert (0);
    }
    if (rpool == NULL)
    {
        FPRT (stderr, "ERROR: ase_group_set_rpool: rpool is NULL\n");
        assert (0);
    }
    if (ase_group->gpu_idx != rpool->gpu_idx)
    {
        FPRT (stderr, "ERROR: ase_group_set_rpool: ase_group->gpu_idx %d != rpool->gpu_idx %d\n", ase_group->gpu_idx, rpool->gpu_idx);
        assert (0);
    }
    for (int i = 0; i < ase_group->num_ases; i++)
    {
        ase_group->ase_arr[i].rpool = rpool;
    }
    add_ref_ases (rpool, ase_group->num_ases);
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
        assert (0);
    }
    if (gpu_idx >= 0 && gpu_idx >= aspen_num_gpus)
    {
        FPRT (stderr, "ERROR: ase_init: gpu_idx %d is out of range... Falling back to CPU\n", gpu_idx);
        gpu_idx = -1;
    }
    ase->thread_id = atomic_fetch_add (&ase_thread_id_counter, 1);
    ase->rpool = NULL;
    ase->gpu_idx = gpu_idx;
    ase->scratchpad = aspen_calloc (ASE_SCRATCHPAD_SIZE, 1);
    if (gpu_idx >= 0)
        ase->gpu_scratchpad = aspen_gpu_calloc (ASE_SCRATCHPAD_SIZE, 1, gpu_idx);
    ase->thread_mutex = (pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER;
    ase->thread_cond = (pthread_cond_t)PTHREAD_COND_INITIALIZER;
    ase->ninst_cache = calloc (1, sizeof (rpool_queue_t));
    atomic_store (&ase->run, 0);
    atomic_store (&ase->kill, 0);
    rpool_init_queue (ase->ninst_cache);
    pthread_mutex_lock(&ase->thread_mutex);
    pthread_create (&ase->thread, NULL, ase_thread_runtime, (void*)ase);
}

void ase_destroy (ase_t *ase)
{
    if (ase == NULL)
        return;
    if (atomic_load (&ase->run) == 1)
    {
        FPRT (stderr, "ERROR: Tried to destroy ase while it is running.\n");
    }
    atomic_store (&ase->kill, 1);
    if (atomic_load (&ase->run) != 1)
        ase_run (ase);
    pthread_join (ase->thread, NULL);
    pthread_mutex_destroy (&ase->thread_mutex);
    pthread_cond_destroy (&ase->thread_cond);
    if (ase->scratchpad != NULL)
        aspen_free (ase->scratchpad);
    if (ase->gpu_scratchpad != NULL)
        aspen_gpu_free (ase->gpu_scratchpad, ase->gpu_idx);
    rpool_destroy_queue (ase->ninst_cache);
    free (ase->ninst_cache);
}

void ase_group_run (ase_group_t *ase_group)
{
    if (ase_group == NULL)
    {
        FPRT (stderr, "ERROR: ase_group_run: ase_group is NULL\n");
        assert (0);
    }
    for (int i = 0; i < ase_group->num_ases; i++)
    {
        ase_run (&ase_group->ase_arr[i]);
    }
}

void ase_group_stop (ase_group_t *ase_group)
{
    if (ase_group == NULL)
    {
        FPRT (stderr, "ERROR: ase_group_stop: ase_group is NULL\n");
        assert (0);
    }
    for (int i = 0; i < ase_group->num_ases; i++)
    {
        ase_stop (&ase_group->ase_arr[i]);
    }
}

unsigned int ase_check_nasm_completion (nasm_t *nasm)
{
    #ifdef DEBUG
    if (nasm == NULL)
    {
        FPRT (stderr, "ERROR: ase_check_nasm_completion: nasm is NULL\n");
        assert (0);
    }
    #endif
    if (atomic_load(&nasm->num_ldata_completed) == nasm->num_ldata)
        return 1;
    return 0;
}

void ase_group_run_until_nasm_completion (ase_group_t *ase_group, nasm_t *nasm)
{
    pthread_mutex_lock (&nasm->nasm_mutex);
    if (ase_check_nasm_completion (nasm) == 1)
    {
        pthread_mutex_unlock (&nasm->nasm_mutex);
        return;
    }
    ase_group_run (ase_group);
    pthread_cond_wait (&nasm->nasm_cond, &nasm->nasm_mutex);
    ase_group_stop (ase_group);
}

void ase_wait_for_nasm_completion (nasm_t *nasm)
{
    pthread_mutex_lock (&nasm->nasm_mutex);
    if (ase_check_nasm_completion (nasm) == 1)
    {
        pthread_mutex_unlock (&nasm->nasm_mutex);
        return;
    }
    pthread_cond_wait (&nasm->nasm_cond, &nasm->nasm_mutex);
    pthread_mutex_unlock (&nasm->nasm_mutex);
}

void ase_cudagraph_run (rpool_t *rpool, nasm_t *nasm)
{
    if (nasm == NULL)
    {
        FPRT (stderr, "ERROR: ase_cudagraph_run: nasm is NULL\n");
        assert (0);
    }
    if (nasm->gpu_idx < 0)
    {
        FPRT (stderr, "ERROR: ase_cudagraph_run: gpu not initialized.\n");
        assert (0);
    }
    rpool_pop_all_nasm (rpool, nasm);
    run_cudagraph (nasm);
}

void ase_run (ase_t *ase)
{
    if (ase == NULL)
    {
        FPRT (stderr, "ERROR: ase_run: ase is NULL\n");
        return;
    }
    unsigned int state = atomic_exchange (&ase->run, 1);
    if (state == 1)
    {
        return;
    }
    else 
    {
        pthread_cond_signal (&ase->thread_cond);
        pthread_mutex_unlock (&ase->thread_mutex);
    }
}

void ase_stop (ase_t *ase)
{
    if (ase == NULL)
    {
        FPRT (stderr, "ERROR: ase_stop: ase is NULL\n");
        return;
    }
    unsigned int state = atomic_exchange (&ase->run, 0);
    if (state == 0)
    {
        return;
    }
    else 
    {
        pthread_mutex_lock (&ase->thread_mutex);
    }
}

void update_children (rpool_t *rpool, ninst_t *ninst)
{
    #ifdef DEBUG
    if (rpool == NULL || ninst == NULL)
    {
        FPRT (stderr, "Error: Invalid arguments to ase_update_children()\n");
        assert (0);
    }
    if (ninst->state != NINST_COMPLETED)
    {
        FPRT (stderr, "Error: ninst->state != NINST_STATE_COMPLETED in ase_update_children()\n");
        assert (0);
    }
    #endif
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
                assert (0);
            }
            #endif
            child_ninst->state = NINST_READY;
            rpool_push_ninsts (rpool, &child_ninst, 1);
        }
    }
}

void update_children_to_cache (rpool_queue_t *cache, ninst_t *ninst)
{
    #ifdef DEBUG
    if (cache == NULL || ninst == NULL)
    {
        FPRT (stderr, "Error: Invalid arguments to ase_update_children_to_cache()\n");
        assert (0);
    }
    if (ninst->state != NINST_COMPLETED)
    {
        FPRT (stderr, "Error: ninst->state != NINST_STATE_COMPLETED in ase_update_children_to_cache()\n");
        assert (0);
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
                FPRT (stderr, "Error: child_ninst->state != NINST_NOT_READY in ase_update_children_to_cache()\n");
                assert (0);
            }
            #endif
            child_ninst->state = NINST_READY;
            push_ninsts_to_queue (cache, &child_ninst, 1);
        }
    }
}

void update_children_but_prioritize_ase_target (rpool_t *rpool, ninst_t *ninst, ase_t *ase)
{
    #ifdef DEBUG
    if (ninst->state != NINST_COMPLETED)
    {
        FPRT (stderr, "Error: ninst->state != NINST_STATE_COMPLETED in ase_update_children_to_cache()\n");
        assert (0);
    }
    #endif
    if (ninst->state != NINST_COMPLETED)
        return;
    ninst_t **cache = ase->scratchpad;
    unsigned int num_cache = 0;
    for (int i = 0; i < ninst->num_child_ninsts; i++)
    {
        ninst_t *child_ninst = ninst->child_ninst_arr[i];
        unsigned int num_parent_ninsts_completed = atomic_fetch_add (&child_ninst->num_parent_ninsts_completed, 1);
        if (num_parent_ninsts_completed == child_ninst->num_parent_ninsts - 1)
        {
            #ifdef DEBUG 
            if (child_ninst->state != NINST_NOT_READY)
            {
                FPRT (stderr, "Error: child_ninst->state != NINST_NOT_READY in ase_update_children_to_cache()\n");
                assert (0);
            }
            #endif
            child_ninst->state = NINST_READY;
            if (ase->target != NULL)
            {
                cache[num_cache++] = child_ninst;
            }
            else
                ase->target = child_ninst;
        }
    }
    rpool_push_ninsts (rpool, cache, num_cache);
}

void update_children_to_cache_but_prioritize_ase_target (rpool_queue_t *cache, ninst_t *ninst, ninst_t **ase_target)
{
    #ifdef DEBUG
    if (cache == NULL || ninst == NULL)
    {
        FPRT (stderr, "Error: Invalid arguments to ase_update_children_to_cache()\n");
        assert (0);
    }
    if (ninst->state != NINST_COMPLETED)
    {
        FPRT (stderr, "Error: ninst->state != NINST_STATE_COMPLETED in ase_update_children_to_cache()\n");
        assert (0);
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
                FPRT (stderr, "Error: child_ninst->state != NINST_NOT_READY in ase_update_children_to_cache()\n");
                assert (0);
            }
            #endif
            child_ninst->state = NINST_READY;
            if (*ase_target != NULL)
                push_ninsts_to_queue (cache, &child_ninst, 1);
            else
                *ase_target = child_ninst;
        }
    }
}

void push_first_layer_to_rpool (rpool_t *rpool, nasm_t *nasm, void* input_data)
{
    #ifdef DEBUG
    if (rpool == NULL || nasm == NULL)
    {
        FPRT (stderr, "Error: Invalid arguments to ase_push_first_layer_to_rpool()\n");
        assert (0);
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
    if (nasm->data == NULL)
    {
        nasm->data = aspen_calloc (total_mem_req, 1);
        if (input_data != NULL)
        {
            nasm_ldata_t *ldata = &nasm->ldata_arr[0];
            aspen_layer_t *layer = ldata->layer;
            size_t num_cols = 0;
            if (layer->params[OUT_H] != 0 && layer->params[OUT_W] != 0)
                num_cols = nasm->batch_size * layer->params[OUT_H] * layer->params[OUT_W];
            else if (layer->params[MAT_M] != 0)
                num_cols = nasm->batch_size * nasm->tr_seq_len;
            for (int i = 0; i < num_cols; i++)
                memcpy 
                    ((char*)nasm->data + i * ldata->out_mat_stride * nasm->dnn->element_size, 
                    (char*)input_data + i * ldata->out_mat_dims[OUT_H] * nasm->dnn->element_size, 
                    ldata->out_mat_dims[OUT_H] * nasm->dnn->element_size);
        }
        if (rpool->gpu_idx >= 0)
        {
            void *temp_gpu_data = aspen_gpu_calloc (total_mem_req, 1, rpool->gpu_idx);
            nasm->gpu_null_data = aspen_gpu_calloc (ASE_SCRATCHPAD_SIZE, 1, rpool->gpu_idx);
            aspen_host_to_gpu_async_memcpy (temp_gpu_data, nasm->data, nasm->ldata_arr[0].out_mat_mem_size, rpool->gpu_idx);
            aspen_free(nasm->data);
            nasm->data = temp_gpu_data;
        }
        if (nasm->data == NULL)
        {
            FPRT (stderr, "Error: nasm->data == NULL in ase_push_first_layer_to_rpool()\n");
            assert (0);
        }
        for (int i = 0; i < nasm->num_ldata; i++)
        {
            nasm_ldata_t *ldata = &nasm->ldata_arr[i];
            set_ldata_out_mat_mem_pos (ldata);
        }
        for (int i = 0; i < nasm->num_ninst; i++)
        {
            #ifdef GPU
            ninst_t *ninst = &nasm->ninst_arr[i];
            if (ninst->input_pos_idx_arr != NULL && rpool->gpu_idx >= 0)
            {
                nasm_ldata_t *ldata = ninst->ldata;
                aspen_layer_t *layer = ninst->ldata->layer;
                nasm_ldata_t *p_ldata = (ldata->parent_ldata_idx_arr[PARENT_0] + ldata->nasm->ldata_arr);
                const unsigned int input_pos_per_n = ninst->num_input_pos/ninst->tile_dims[OUT_W];
                size_t pos_arr_range = ninst->num_input_pos + input_pos_per_n*_TILE_SIZE_M;
                ninst->input_pos_ptr_arr_gpu = 
                    aspen_gpu_calloc (pos_arr_range, sizeof (void*), rpool->gpu_idx);
                void *idx_arr_temp = 
                    aspen_gpu_calloc (pos_arr_range, sizeof (int), rpool->gpu_idx);
                aspen_host_to_gpu_memcpy 
                    (idx_arr_temp, ninst->input_pos_idx_arr, 
                        ninst->num_input_pos * sizeof (int), rpool->gpu_idx);
                aspen_sync_gpu (rpool->gpu_idx);
                cuda_preset_conv2d_ptrs (ninst->tile_dims[OUT_W], pos_arr_range/input_pos_per_n, nasm->gpu_null_data, 
                    idx_arr_temp, (float**)ninst->input_pos_ptr_arr_gpu, input_pos_per_n, layer->params[IN_C],
                    p_ldata->out_mat, p_ldata->out_mat_stride,
                    aspen_CUDA_streams[rpool->gpu_idx][GPU_NAIVE_RUN_STREAM]);
                aspen_sync_gpu_stream (rpool->gpu_idx, GPU_NAIVE_RUN_STREAM);
                aspen_gpu_free (idx_arr_temp, rpool->gpu_idx);
            }
            #endif
        }
    }
    if (rpool->gpu_idx >= 0)
        generate_cudagraph (nasm);
    nasm_ldata_t *ldata = &nasm->ldata_arr[0];
    for (int i = 0; i < ldata->num_ninst; i++)
    {
        ninst_t *ninst = &ldata->ninst_arr_start[i];
        if (ninst->state != NINST_NOT_READY)
        {
            FPRT (stderr, "Error: ninst->state != NINST_NOT_READY in ase_push_first_layer_to_rpool()\n");
            assert (0);
        }
        ninst->state = NINST_COMPLETED;
        atomic_fetch_add (&ninst->ldata->num_ninst_completed , 1);
        update_children (rpool, ninst);
    }
    atomic_fetch_add (&nasm->num_ldata_completed, 1);
}

void set_ldata_out_mat_mem_pos (nasm_ldata_t *ldata)
{
    #ifdef DEBUG
    if (ldata == NULL)
    {
        FPRT (stderr, "Error: Invalid arguments to set_ldata_out_mat_mem_pos()\n");
        assert (0);
    }
    #endif
    nasm_t *nasm = ldata->nasm;
    if (nasm->data == NULL)
    {
        FPRT (stderr, "Error: nasm->data == NULL in set_ldata_out_mat_mem_pos()\n");
        assert (0);
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
        assert (0);
    }
    #endif
    nasm_ldata_t *ldata = ninst->ldata;
    if (ninst->ldata->out_mat == NULL)
    {
        FPRT (stderr, "Error: ninst->ldata->out_mat == NULL in set_ninst_out_mat_mem_pos()\n");
        assert (0);
    }
    ninst->out_mat = (char*)ninst->ldata->out_mat 
        + (ninst->out_mat_pos[OUT_W]*ldata->out_mat_stride + ninst->out_mat_pos[OUT_H])
            *ldata->nasm->dnn->element_size;
    
}

void *ase_get_ldata_result (nasm_t *nasm, unsigned int ldata_idx, LAYER_PARAMS *order)
{
    nasm_ldata_t *ldata = &nasm->ldata_arr[ldata_idx];
    return get_ldata_output (ldata, order);
}

void *ase_get_nasm_result (nasm_t *nasm, LAYER_PARAMS *order)
{
    return ase_get_ldata_result (nasm, nasm->num_ldata - 1, order);
}