#include "dse.h"

static _Atomic unsigned int dse_thread_id_counter = 0;

void *dse_thread_runtime (void* thread_info)
{
    dse_t *dse = (dse_t*) thread_info;
    pthread_mutex_lock(&dse->thread_mutex);
    while (dse->kill == 0)
    {
        // print_rpool_info (dse->rpool);
        // print_rpool_queue_info (dse->ninst_cache);
        if (dse->run == 0)
        {   
            pthread_cond_wait(&dse->thread_cond, &dse->thread_mutex); 
        }
        // if ((dse->ninst_cache->num_stored < dse_NINST_CACHE_BALLANCE - dse_NINST_CACHE_DIFF) || 
        //     dse->ninst_cache->num_stored == 0)
        if (dse->target == NULL)
        {
            rpool_fetch_ninsts (dse->rpool, &dse->target, 1, 0);
            if (dse->target == NULL)
                continue;
            // unsigned int fetch_num = 
            //     rpool_fetch_ninsts (dse->rpool, dse->scratchpad, dse_NINST_CACHE_BALLANCE - dse->ninst_cache->num_stored);
            // push_ninsts_to_queue (dse->ninst_cache, dse->scratchpad, fetch_num);
            // PRT ("Thread %d fetched %d ninsts from rpool\n", dse->thread_id, fetch_num);
            // #ifdef DEBUG
            // PRT ("Thread %d fetched %d ninsts from rpool\n", dse->thread_id, fetch_num);
            // print_rpool_info (dse->rpool);
            // print_rpool_queue_info (dse->ninst_cache);
            // #endif
        }
        // else if (dse->ninst_cache->num_stored > dse_NINST_CACHE_BALLANCE + dse_NINST_CACHE_DIFF)
        // if (dse->ninst_cache->num_stored > 0)
        // {
        //     // unsigned int push_num = 
        //     //     pop_ninsts_from_queue_back (dse->ninst_cache, dse->scratchpad, dse->ninst_cache->num_stored - dse_NINST_CACHE_BALLANCE);
        //     rpool_push_ninsts (dse->rpool, dse->ninst_cache->ninst_ptr_arr, dse->ninst_cache->num_stored);
        //     dse->ninst_cache->num_stored = 0;
        //     dse->ninst_cache->idx_end = 0;
        //     // PRT ("Thread %d pushed %d ninsts to rpool\n", dse->thread_id, push_num);
        //     // #ifdef DEBUG
        //     // PRT ("Thread %d pushed %d ninsts to rpool\n", dse->thread_id, push_num);
        //     // print_rpool_info (dse->rpool);
        //     // print_rpool_queue_info (dse->ninst_cache);
        //     // #endif
        // }

        // unsigned int num_ninsts = dse->ninst_cache->num_stored;
        // for (int i = 0; i < num_ninsts; i++)
        // {
        //     ninst_t *ninst;
        //     pop_ninsts_from_queue (dse->ninst_cache, &ninst, 1);
            // PRT ("Thread %d running ninst #%d - N%d:L%d:%d\n", dse->thread_id, i,
            //         ninst->ldata->nasm->nasm_id, ninst->ldata->layer->layer_idx, ninst->ninst_idx);
            // #ifdef DEBUG
            // if (ninst == NULL)
            // {
            //     FPRT (stderr, "ERROR: dse_thread_runtime: ninst is NULL\n");
            //     assert (0);
            // }
            // else 
            // {
            //     PRT ("Thread %d running ninst #%d - N%d:L%d:%d\n", dse->thread_id, i,
            //         ninst->ldata->nasm->nasm_id, ninst->ldata->layer->layer_idx, ninst->ninst_idx);
            // }
            // if (ninst->state != NINST_READY)
            // {
            //     FPRT (stderr, "ERROR: dse_thread_runtime: ninst->state != NINST_READY\n");
            //     assert (0);
            // }
            // #endif
            // Execute.
            ninst_t *ninst = dse->target;
            dse->target = NULL;
            #ifdef DEBUG 
            if (ninst->state != NINST_READY && ninst->state != NINST_COMPLETED)
            {
                FPRT (stderr, "Error: ninst->state != NINST_READY in dse_thread_runtime()\n");
                assert (0);
            }
            #endif
            // printf("fetched ninst %d, offload: %d, compute: %d\n", ninst->ninst_idx, ninst->offload, ninst->compute);
            if (is_ninst_mine(ninst, dse->device_idx))    // It's mine, so compute
            {
                // printf("compute ninst %d\n", ninst->ninst_idx);
                if (dse->profile_compute) ninst->compute_start = get_time_secs();
                switch (ninst->ldata->layer->type)
                {
                    case CONV_LAYER:
                        tiled_conv2d (ninst, dse);
                        break;
                    case MAXPOOL_LAYER:
                        tiled_maxpool2d (ninst, dse);
                        break;
                    case AVGPOOL_LAYER:
                        tiled_avgpool2d (ninst, dse);
                        break;
                    case FC_LAYER:
                        tiled_fully_connected (ninst, dse);
                        break;
                    case RESIDUAL_LAYER:
                        tiled_residual (ninst, dse);
                        break;
                    case SOFTMAX_LAYER:
                        tiled_softmax (ninst, dse);
                        break;
                    case YOLO_LAYER:
                        tiled_yolo (ninst, dse);
                        break;
                    case APPEND_LAYER:
                        tiled_append (ninst, dse);
                        break;
                    case MATMUL_LAYER:
                        tiled_matmul (ninst, dse);
                        break;
                    case LAYERNORM_LAYER:
                        tiled_layernorm (ninst, dse);
                        break;
                    case K_ATTENTION_LAYER:
                        tiled_k_attention (ninst, dse);
                        break;
                    case V_ATTENTION_LAYER:
                        tiled_v_attention (ninst, dse);
                        break;
                    default:
                        // FPRT (stderr, "ERROR: dse_thread_runtime: layer type %s is not supported\n", layer_type_str[ninst->ldata->layer->type]);
                        break;
                }

                ninst->computed_time = get_time_secs();
                if (dse->profile_compute) ninst->compute_end = ninst->computed_time;
            
                ninst->state = NINST_COMPLETED;
                unsigned int num_ninst_completed = atomic_fetch_add (&ninst->ldata->num_ninst_completed, 1);
                if (num_ninst_completed == ninst->ldata->num_ninst - 1)
                {
                    // printf ("\t\tThread %d completed layer %d of nasm %d\n", 
                    //     dse->thread_id, ninst->ldata->layer->layer_idx, ninst->ldata->nasm->nasm_id);
                    nasm_t *nasm = ninst->ldata->nasm;
                    unsigned int num_ldata_completed = atomic_fetch_add (&nasm->num_ldata_completed, 1);
                    if (num_ldata_completed == nasm->num_ldata - 2)
                    {
                        printf ("\t\tSignaling nasm completion...\n");
                        // All layers of the nasm is completed.
                        rpool_queue_group_t *rpool_queue_group 
                            = get_queue_group_from_nasm (dse->rpool, ninst->ldata->nasm);
                        set_queue_group_weight (dse->rpool, rpool_queue_group, 0);
                        pthread_mutex_lock (&nasm->nasm_mutex);
                        pthread_cond_signal (&nasm->nasm_cond);
                        pthread_mutex_unlock (&nasm->nasm_mutex);
                    }
                }
            // update_children_to_cache (dse->ninst_cache, ninst);
                update_children_but_prioritize_dse_target (dse->rpool, ninst, dse);

                // check desiring devices for the computation output
                if (ninst->desiring_devices[!dse->device_idx]) // Should be offload
                    {
                        networking_engine *net_engine = dse->net_engine;
                        pthread_mutex_lock(&net_engine->net_engine_mutex);
                        push_ninsts_to_net_queue(net_engine->net_queue, ninst, 1);
                        pthread_mutex_unlock(&net_engine->net_engine_mutex);
                    }
                }
        // }
    }
    return NULL;
}

dse_group_t *dse_group_init (unsigned int num_ase, int gpu_idx)
{
    if (gpu_idx >= 0 && gpu_idx >= aspen_num_gpus)
    {
        FPRT (stderr, "ERROR: dse_group_init: gpu_idx %d is out of range... Falling back to CPU\n", gpu_idx);
        gpu_idx = -1;
    }
    else if (gpu_idx >= 0 && gpu_idx < aspen_num_gpus)
    {
        num_ase = num_ase > GPU_RUN_STREAM_NUM ? GPU_RUN_STREAM_NUM : num_ase;
    }
    dse_group_t *dse_group = (dse_group_t *) calloc (1, sizeof (dse_group_t));
    dse_group->num_ases = num_ase;
    if (gpu_idx < 0)
        dse_group->gpu_idx = -1;
    else
        dse_group->gpu_idx = gpu_idx;
    dse_group->dse_arr = (dse_t *) calloc (num_ase, sizeof (dse_t));
    for (int i = 0; i < num_ase; i++)
    {
        dse_init (&dse_group->dse_arr[i], dse_group->gpu_idx);
    }
    return dse_group;
}

void dse_group_set_rpool (dse_group_t *dse_group, rpool_t *rpool)
{
    if (dse_group == NULL)
    {
        FPRT (stderr, "ERROR: dse_group_set_rpool: dse_group is NULL\n");
        assert (0);
    }
    if (rpool == NULL)
    {
        FPRT (stderr, "ERROR: dse_group_set_rpool: rpool is NULL\n");
        assert (0);
    }
    if (dse_group->gpu_idx != rpool->gpu_idx)
    {
        FPRT (stderr, "ERROR: dse_group_set_rpool: dse_group->gpu_idx %d != rpool->gpu_idx %d\n", dse_group->gpu_idx, rpool->gpu_idx);
        assert (0);
    }
    for (int i = 0; i < dse_group->num_ases; i++)
    {
        dse_group->dse_arr[i].rpool = rpool;
    }
    add_ref_ases (rpool, dse_group->num_ases);
}

void dse_group_set_net_engine (dse_group_t *dse_group, networking_engine *net_engine)
{
    if (dse_group == NULL)
    {
        FPRT (stderr, "ERROR: dse_group_set_net_engine: dse_group is NULL\n");
        assert (0);
    }
    if (net_engine == NULL)
    {
        FPRT (stderr, "ERROR: dse_group_set_net_engine: net_engine is NULL\n");
        assert (0);
    }
    for (int i = 0; i < dse_group->num_ases; i++)
    {
        dse_group->dse_arr[i].net_engine = net_engine;
    }
}

void dse_group_set_device (dse_group_t *dse_group, int device_idx)
{
    if (dse_group == NULL)
    {
        FPRT (stderr, "ERROR: dse_group_set_device: dse_group is NULL\n");
        assert (0);
    }
    for (int i = 0; i < dse_group->num_ases; i++)
    {
        dse_group->dse_arr[i].device_idx = device_idx;
    }
}

void dse_group_set_profile (dse_group_t *dse_group, int profile_compute)
{
    if (dse_group == NULL)
    {
        FPRT (stderr, "ERROR: dse_group_set_device: dse_group is NULL\n");
        assert (0);
    }
    for (int i = 0; i < dse_group->num_ases; i++)
    {
        dse_group->dse_arr[i].profile_compute = profile_compute;
    }
}

void dse_group_destroy (dse_group_t *dse_group)
{
    if (dse_group == NULL)
        return;
    for (int i = 0; i < dse_group->num_ases; i++)
    {
        if (dse_group->dse_arr[i].rpool != NULL)
            atomic_fetch_sub (&dse_group->dse_arr[i].rpool->ref_ases, 1);
        dse_destroy (&dse_group->dse_arr[i]);
    }
    free (dse_group->dse_arr);
    free (dse_group);
}

void dse_init (dse_t *dse, int gpu_idx)
{
    if (dse == NULL)
    {
        FPRT (stderr, "ERROR: dse_init: dse is NULL\n");
        assert (0);
    }
    if (gpu_idx >= 0 && gpu_idx >= aspen_num_gpus)
    {
        FPRT (stderr, "ERROR: dse_init: gpu_idx %d is out of range... Falling back to CPU\n", gpu_idx);
        gpu_idx = -1;
    }
    dse->thread_id = atomic_fetch_add (&dse_thread_id_counter, 1);
    dse->rpool = NULL;
    dse->gpu_idx = gpu_idx;
    dse->scratchpad = aspen_calloc (dse_SCRATCHPAD_SIZE, 1);
    if (gpu_idx >= 0)
        dse->gpu_scratchpad = aspen_gpu_calloc (dse_SCRATCHPAD_SIZE, 1, gpu_idx);
    dse->thread_mutex = (pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER;
    dse->thread_cond = (pthread_cond_t)PTHREAD_COND_INITIALIZER;
    dse->ninst_cache = calloc (1, sizeof (rpool_queue_t));
    atomic_store (&dse->run, 0);
    atomic_store (&dse->kill, 0);
    rpool_init_queue (dse->ninst_cache);
    pthread_mutex_lock(&dse->thread_mutex);
    pthread_create (&dse->thread, NULL, dse_thread_runtime, (void*)dse);
}

void dse_destroy (dse_t *dse)
{
    if (dse == NULL)
        return;
    if (atomic_load (&dse->run) == 1)
    {
        FPRT (stderr, "ERROR: Tried to destroy dse while it is running.\n");
    }
    atomic_store (&dse->kill, 1);
    if (atomic_load (&dse->run) != 1)
        dse_run (dse);
    pthread_join (dse->thread, NULL);
    pthread_mutex_destroy (&dse->thread_mutex);
    pthread_cond_destroy (&dse->thread_cond);
    if (dse->scratchpad != NULL)
        aspen_free (dse->scratchpad);
    if (dse->gpu_scratchpad != NULL)
        aspen_gpu_free (dse->gpu_scratchpad, dse->gpu_idx);
    rpool_destroy_queue (dse->ninst_cache);
    free (dse->ninst_cache);
}

void dse_group_run (dse_group_t *dse_group)
{
    if (dse_group == NULL)
    {
        FPRT (stderr, "ERROR: dse_group_run: dse_group is NULL\n");
        assert (0);
    }
    for (int i = 0; i < dse_group->num_ases; i++)
    {
        dse_run (&dse_group->dse_arr[i]);
    }
}

void dse_group_stop (dse_group_t *dse_group)
{
    if (dse_group == NULL)
    {
        FPRT (stderr, "ERROR: dse_group_stop: dse_group is NULL\n");
        assert (0);
    }
    for (int i = 0; i < dse_group->num_ases; i++)
    {
        dse_stop (&dse_group->dse_arr[i]);
    }
}

unsigned int dse_check_nasm_completion (nasm_t *nasm)
{
    #ifdef DEBUG
    if (nasm == NULL)
    {
        FPRT (stderr, "ERROR: dse_check_nasm_completion: nasm is NULL\n");
        assert (0);
    }
    #endif
    if (atomic_load(&nasm->num_ldata_completed) == nasm->num_ldata)
        return 1;
        
    return 0;
}

void dse_group_run_until_nasm_completion (dse_group_t *dse_group, nasm_t *nasm)
{
    dse_group_run (dse_group);
    dse_wait_for_nasm_completion (nasm);
    dse_group_stop (dse_group);
}

void dse_wait_for_nasm_completion (nasm_t *nasm)
{
    pthread_mutex_lock (&nasm->nasm_mutex);
    if (dse_check_nasm_completion (nasm) == 1)
    {
        pthread_mutex_unlock (&nasm->nasm_mutex);
        return;
    }
    pthread_cond_wait (&nasm->nasm_cond, &nasm->nasm_mutex);
    pthread_mutex_unlock (&nasm->nasm_mutex);
}

void dse_cudagraph_run (rpool_t *rpool, nasm_t *nasm)
{
    if (nasm == NULL)
    {
        FPRT (stderr, "ERROR: dse_cudagraph_run: nasm is NULL\n");
        assert (0);
    }
    if (nasm->gpu_idx < 0)
    {
        FPRT (stderr, "ERROR: dse_cudagraph_run: gpu not initialized.\n");
        assert (0);
    }
    rpool_finish_nasm (rpool, nasm);
    run_cudagraph (nasm);
}

void dse_run (dse_t *dse)
{
    if (dse == NULL)
    {
        FPRT (stderr, "ERROR: dse_run: dse is NULL\n");
        return;
    }
    unsigned int state = atomic_exchange (&dse->run, 1);
    if (state == 1)
    {
        return;
    }
    else 
    {
        pthread_cond_signal (&dse->thread_cond);
        pthread_mutex_unlock (&dse->thread_mutex);
    }
}

void dse_stop (dse_t *dse)
{
    if (dse == NULL)
    {
        FPRT (stderr, "ERROR: dse_stop: dse is NULL\n");
        return;
    }
    unsigned int state = atomic_exchange (&dse->run, 0);
    if (state == 0)
    {
        return;
    }
    else 
    {
        pthread_mutex_lock (&dse->thread_mutex);
    }
}

void update_children (rpool_t *rpool, ninst_t *ninst, unsigned int dse_idx)
{
    #ifdef DEBUG
    if (rpool == NULL || ninst == NULL)
    {
        FPRT (stderr, "Error: Invalid arguments to dse_update_children()\n");
        assert (0);
    }
    if (ninst->state != NINST_COMPLETED)
    {
        FPRT (stderr, "Error: ninst->state != NINST_STATE_COMPLETED in dse_update_children()\n");
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
                FPRT (stderr, "Error: child_ninst->state != NINST_NOT_READY in dse_update_children()\n");
                assert (0);
            }
            #endif
            child_ninst->state = NINST_READY;
            rpool_push_ninsts (rpool, &child_ninst, 1, 0);
        }
    }
}

void update_children_to_cache (rpool_queue_t *cache, ninst_t *ninst)
{
    #ifdef DEBUG
    if (cache == NULL || ninst == NULL)
    {
        FPRT (stderr, "Error: Invalid arguments to dse_update_children_to_cache()\n");
        assert (0);
    }
    if (ninst->state != NINST_COMPLETED)
    {
        FPRT (stderr, "Error: ninst->state != NINST_STATE_COMPLETED in dse_update_children_to_cache()\n");
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
                FPRT (stderr, "Error: child_ninst->state != NINST_NOT_READY in dse_update_children_to_cache()\n");
                assert (0);
            }
            #endif
            child_ninst->state = NINST_READY;
            push_ninsts_to_queue (cache, &child_ninst, 1);
        }
    }
}

void update_children_but_prioritize_dse_target (rpool_t *rpool, ninst_t *ninst, dse_t *dse)
{
    #ifdef DEBUG
    if (ninst->state != NINST_COMPLETED)
    {
        FPRT (stderr, "Error: ninst->state != NINST_STATE_COMPLETED in dse_update_children_to_cache()\n");
        assert (0);
    }
    #endif
    if (ninst->state != NINST_COMPLETED)
        return;
    ninst_t **cache = dse->scratchpad;
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
                FPRT (stderr, "Error: child_ninst->state != NINST_NOT_READY in dse_update_children_to_cache()\n");
                assert (0);
            }
            #endif
            child_ninst->state = NINST_READY;
            if (dse->target != NULL)
            {
                cache[num_cache++] = child_ninst;
            }
            else
                dse->target = child_ninst;
        }
    }
    rpool_push_ninsts (rpool, cache, num_cache, 0);
}

void update_children_to_cache_but_prioritize_dse_target (rpool_queue_t *cache, ninst_t *ninst, ninst_t **dse_target)
{
    #ifdef DEBUG
    if (cache == NULL || ninst == NULL)
    {
        FPRT (stderr, "Error: Invalid arguments to dse_update_children_to_cache()\n");
        assert (0);
    }
    if (ninst->state != NINST_COMPLETED)
    {
        FPRT (stderr, "Error: ninst->state != NINST_STATE_COMPLETED in dse_update_children_to_cache()\n");
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
                FPRT (stderr, "Error: child_ninst->state != NINST_NOT_READY in dse_update_children_to_cache()\n");
                assert (0);
            }
            #endif
            child_ninst->state = NINST_READY;
            if (*dse_target != NULL)
                push_ninsts_to_queue (cache, &child_ninst, 1);
            else
                *dse_target = child_ninst;
        }
    }
}

void push_first_layer_to_rpool (rpool_t *rpool, nasm_t *nasm, void* input_data)
{
    #ifdef DEBUG
    if (rpool == NULL || nasm == NULL)
    {
        FPRT (stderr, "Error: Invalid arguments to dse_push_first_layer_to_rpool()\n");
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
            nasm->gpu_null_data = aspen_gpu_calloc (dse_SCRATCHPAD_SIZE, 1, rpool->gpu_idx);
            aspen_host_to_gpu_async_memcpy (temp_gpu_data, nasm->data, nasm->ldata_arr[0].out_mat_mem_size, rpool->gpu_idx);
            aspen_free(nasm->data);
            nasm->data = temp_gpu_data;
        }
        if (nasm->data == NULL)
        {
            FPRT (stderr, "Error: nasm->data == NULL in dse_push_first_layer_to_rpool()\n");
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
            FPRT (stderr, "Error: ninst->state != NINST_NOT_READY in dse_push_first_layer_to_rpool()\n");
            assert (0);
        }
        ninst->state = NINST_COMPLETED;
        atomic_fetch_add (&ninst->ldata->num_ninst_completed , 1);
        int num_ase = rpool->ref_ases > 0 ? rpool->ref_ases : 1;
        update_children (rpool, ninst, i/(ldata->num_ninst/num_ase));
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

void *dse_get_ldata_result (nasm_t *nasm, unsigned int ldata_idx, LAYER_PARAMS *order)
{
    nasm_ldata_t *ldata = &nasm->ldata_arr[ldata_idx];
    if (ldata->layer->type == YOLO_LAYER)
    {
        void *output = NULL;
        size_t output_size = 0;
        for (int i = 0; i < nasm->num_ldata; i++)
        {
            nasm_ldata_t *targ_ldata = &nasm->ldata_arr[i];
            if (targ_ldata->layer->type == YOLO_LAYER)
            {
                size_t elem_size = targ_ldata->layer->dnn->element_size;
                size_t data_size = targ_ldata->out_mat_dims[OUT_H] * targ_ldata->out_mat_dims[OUT_W] * elem_size;
                output_size += data_size;
            }
        }
        output = calloc (output_size, 1);
        size_t offset = 0;
        size_t batch_num = nasm->batch_size;
        for (int b = 0; b < batch_num; b++)
        {
            for (int i = 0; i < nasm->num_ldata; i++)
            {
                nasm_ldata_t *targ_ldata = &nasm->ldata_arr[i];
                if (targ_ldata->layer->type == YOLO_LAYER)
                {
                    void *data = get_ldata_output (targ_ldata, order);
                    size_t elem_size = targ_ldata->layer->dnn->element_size;
                    size_t data_size = targ_ldata->out_mat_dims[OUT_H] * targ_ldata->out_mat_dims[OUT_W] * elem_size / batch_num;
                    memcpy ((char*)output + offset, (char*)data + data_size*b, data_size);
                    offset += data_size;
                    free (data);
                }
            }
        }
        return output;
    }
    return get_ldata_output (ldata, order);
}

void *dse_get_nasm_result (nasm_t *nasm, LAYER_PARAMS *order)
{
    return dse_get_ldata_result (nasm, nasm->num_ldata - 1, order);
}