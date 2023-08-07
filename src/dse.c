#include "dse.h"

static _Atomic unsigned int dse_thread_id_counter = 0;

void *dse_thread_runtime (void* thread_info)
{
    dse_t *dse = (dse_t*) thread_info;
    pthread_mutex_lock(&dse->thread_mutex);
    pthread_cond_wait(&dse->thread_cond, &dse->thread_mutex); 
    while (dse->kill == 0)
    {
        dse_schedule (dse);
        if (dse->run == 0)
            pthread_cond_wait(&dse->thread_cond, &dse->thread_mutex); 
    }
    return NULL;
}

void dse_schedule (dse_t *dse)
{
    // print_rpool_info (dse->rpool);
    // print_rpool_queue_info (dse->ninst_cache);
    
    // if ((dse->ninst_cache->num_stored < dse_NINST_CACHE_BALLANCE - dse_NINST_CACHE_DIFF) || 
    //     dse->ninst_cache->num_stored == 0)
    int target_device;
    if (dse->target == NULL)
    {
        if (dse->is_multiuser_case)
        {
            if (dse->device_idx != DEV_SERVER) 
            {
                rpool_fetch_ninsts (dse->rpool_arr[0], &dse->target, 1, 0);
                if (dse->target == NULL)
                    return;
            }
            else
            {
                int checked[SCHEDULE_MAX_DEVICES] = {0};
                for (int i = 0; i < SCHEDULE_MAX_DEVICES; i++) 
                {
                    if (dse->prioritize_rpool[i] != -1) 
                    {
                        rpool_fetch_ninsts (dse->rpool_arr[dse->prioritize_rpool[i]], &dse->target, 1, 0);
                        checked[dse->prioritize_rpool[i]] = 1;
                        if (dse->target) 
                        {
                            target_device = dse->prioritize_rpool[i];
                            break;
                        }
                    }
                    else 
                        break;
                }
                for (int i = 1; i < SCHEDULE_MAX_DEVICES; i++) 
                {
                    if (!dse->enabled_device[i]) 
                        continue;
                    if (checked[i]) 
                        continue;

                    rpool_fetch_ninsts (dse->rpool_arr[i], &dse->target, 1, 0);
                    if (dse->target) 
                    {
                        target_device = i;
                        break;
                    }
                }
                if (dse->target == NULL) 
                    return;
            }
        }
        else 
        {
            rpool_fetch_ninsts (dse->rpool, &dse->target, 1, 0);
            if (dse->target == NULL)
                return;
        }
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
        if (atomic_exchange (&ninst->state, NINST_COMPLETED) == NINST_COMPLETED)
            return;

        // printf("fetched ninst %d, offload: %d, compute: %d\n", ninst->ninst_idx, ninst->offload, ninst->compute);
        if (is_device_compute_dev(ninst, dse->device_idx) || dse->profile_compute)    // It's mine, so compute
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

            // For dynamic offloading, kmbin added
            if(dse->is_dynamic_scheduling)
            {        
                if(dse->device_idx != DEV_SERVER)
                {
                    // Check all childs
                    // Update when child is not to offload
                    for(int i = 0; i < ninst->num_child_ninsts; i++)
                    {
                        ninst_t* child_ninst = ninst->child_ninst_arr[i];

                        for(int j = 0; j < child_ninst->num_parent_ninsts; j++)
                        {
                            // If one of parent is allocated to server, send output to server
                            int parent_idx = child_ninst->parent_ninst_idx_arr[j];
                            if(dse->net_engine->nasm->ninst_arr[parent_idx].dev_to_compute[0])
                            {
                                printf("\t ninst %d to server dev_to_compute[%d] -> ", child_ninst->ninst_idx, child_ninst->dev_to_compute[DEV_SERVER]);
                                child_ninst->dev_to_compute[DEV_SERVER] = 1;
                                ninst->dev_send_target[DEV_SERVER] = 1;
                                printf("dev_to_compute[%d]", child_ninst->dev_to_compute[DEV_SERVER]);
                                break;
                            }
                        }
                    }

                    float eft_mobile = 0.0;
                    float eft_server = 1.0;
                    if(eft_mobile < eft_server)
                    {
                        for(int i = 0; i < ninst->num_child_ninsts; i++)
                        {
                            ninst->child_ninst_arr[i]->dev_to_compute[DEV_SERVER] = 0;
                            ninst->child_ninst_arr[i]->dev_to_compute[dse->device_idx] = 1;
                        }
                    }
                    else
                    {
                        for(int i = 0; i < ninst->num_child_ninsts; i++)
                        {
                            ninst->child_ninst_arr[i]->dev_to_compute[DEV_SERVER] = 1;
                            ninst->child_ninst_arr[i]->dev_to_compute[dse->device_idx] = 0;
                        }
                        ninst->dev_send_target[DEV_SERVER] = 1;
                    }
                }
            }
        
            
            unsigned int num_ninst_completed = atomic_fetch_add (&ninst->ldata->num_ninst_completed, 1);
            if (num_ninst_completed == ninst->ldata->num_ninst - 1)
            {
                // printf ("\t\tThread %d completed layer %d of nasm %d\n", 
                //     dse->thread_id, ninst->ldata->layer->layer_idx, ninst->ldata->nasm->nasm_id);
                nasm_t *nasm = ninst->ldata->nasm;
                // unsigned int num_ldata_completed = atomic_fetch_add (&nasm->num_ldata_completed, 1);
                atomic_fetch_add (&nasm->num_ldata_completed, 1);
                // if (num_ldata_completed == nasm->num_ldata - 1)

                if (ninst->ldata == &nasm->ldata_arr[nasm->num_ldata - 1])
                {
                    // printf ("\t\tSignaling nasm completion...\n");
                    // All layers of the nasm is completed.
                    atomic_store (&nasm->completed, 1);
                    rpool_queue_group_t *rpool_queue_group;
                    if (dse->is_multiuser_case) {
                        rpool_queue_group = get_queue_group_from_nasm (dse->rpool_arr[target_device], ninst->ldata->nasm);
                        set_queue_group_weight (dse->rpool_arr[target_device], rpool_queue_group, 0);
                    }
                    else {
                        rpool_queue_group = get_queue_group_from_nasm (dse->rpool, ninst->ldata->nasm);
                        set_queue_group_weight (dse->rpool, rpool_queue_group, 0);
                    }
                    pthread_mutex_lock (&nasm->nasm_mutex);
                    pthread_cond_signal (&nasm->nasm_cond);
                    pthread_mutex_unlock (&nasm->nasm_mutex);
                }
            }
            // update_children_to_cache (dse->ninst_cache, ninst);
            if (dse->is_multiuser_case && dse->device_idx == 0) {
                update_children_but_prioritize_dse_target (dse->rpool_arr[target_device], ninst, dse);
            }
            else if (dse->is_multiuser_case && dse->device_idx != 0) {
                update_children_but_prioritize_dse_target (dse->rpool_arr[0], ninst, dse);
            }
            else if (!dse->is_multiuser_case && dse->is_dynamic_scheduling && ninst->ldata->layer->layer_idx == 0) {
                update_children (dse->rpool, ninst, 0);
            }
            else {
                update_children_but_prioritize_dse_target (dse->rpool, ninst, dse);
            }

            // check devices to send to for the computation output
            if (dse->is_multiuser_case) 
            {
                for (int i = 0; i < SCHEDULE_MAX_DEVICES; i++) 
                {
                    if (i == dse->device_idx) continue;
                    if (ninst->dev_send_target[i]) // Should be offload
                    {
                        networking_engine *net_engine = dse->net_engine_arr[i];
                        create_network_buffer_for_ninst (ninst);   
                        pthread_mutex_lock(&net_engine->tx_queue->queue_mutex);
                        push_ninsts_to_net_queue(net_engine->tx_queue, &ninst, 1);
                        pthread_mutex_unlock(&net_engine->tx_queue->queue_mutex);
                    }
                }
            }
            else 
            {
                for (int i = 0; i < SCHEDULE_MAX_DEVICES; i++) 
                {
                    if (i == dse->device_idx) continue;
                    if (ninst->dev_send_target[i]) 
                    {
                        // printf ("\tninst idx %d (L%d), target device: %d, current device: %d, desired device%d\n", 
                        // ninst->ninst_idx, ninst->ldata->layer->layer_idx, i, dse->device_idx,
                        // ninst->dev_send_target[i]);
                        networking_engine *net_engine = dse->net_engine;
                        create_network_buffer_for_ninst (ninst);
                        pthread_mutex_lock(&net_engine->tx_queue->queue_mutex);
                        push_ninsts_to_net_queue(net_engine->tx_queue, &ninst, 1);
                        pthread_mutex_unlock(&net_engine->tx_queue->queue_mutex);
                    }
                }
            }
        }
    // }
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
        dse_init (dse_group, &dse_group->dse_arr[i], dse_group->gpu_idx);
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
    add_ref_dses (rpool, dse_group->num_ases);
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

void dse_group_set_multiuser (dse_group_t *dse_group, int is_multiuser_case) {
    if (dse_group == NULL)
    {
        FPRT (stderr, "ERROR: dse_group_set_multiuser: dse_group is NULL\n");
        assert (0);
    }
    for (int i = 0; i < dse_group->num_ases; i++)
    {
        dse_group->dse_arr[i].is_multiuser_case = is_multiuser_case;
    }
}

void dse_group_add_prioritize_rpool (dse_group_t *dse_group, int device_idx) {
    for (int i=0; i<dse_group->num_ases; i++) {
        for (int j=0; j<SCHEDULE_MAX_DEVICES; j++) {
            if (dse_group->dse_arr[i].prioritize_rpool[j] == -1)
                dse_group->dse_arr[i].prioritize_rpool[j] = device_idx;
        }
    }
}

void dse_group_init_enable_device(dse_group_t *dse_group) {
    for (int i = 0; i < dse_group->num_ases; i++) {
        for (int j=0; j < SCHEDULE_MAX_DEVICES; j++)
        dse_group->dse_arr[i].enabled_device[j] = 0;
    }
}

void dse_group_set_enable_device(dse_group_t *dse_group, int device_idx, int enable) {
    for (int i = 0; i < dse_group->num_ases; i++) {
        dse_group->dse_arr[i].enabled_device[device_idx] = enable;
    }
}

void dse_group_add_rpool_arr(dse_group_t *dse_group, rpool_t *rpool, int device_idx) {
    if (rpool == NULL)
    {
        FPRT (stderr, "ERROR: dse_group_add_rpool_arr: rpool is NULL\n");
        assert (0);
    }
    for (int i = 0; i < dse_group->num_ases; i++) {
        dse_group->dse_arr[i].rpool_arr[device_idx] = rpool;
    }
}

void dse_group_init_netengine_arr (dse_group_t *dse_group) {
    if (dse_group == NULL)
    {
        FPRT (stderr, "ERROR: dse_group_init_netengine_arr: dse_group is NULL\n");
        assert (0);
    }
    for (int i = 0; i < dse_group->num_ases; i++) {
        for (int j=0; j<SCHEDULE_MAX_DEVICES; j++) {
            dse_group->dse_arr[i].net_engine_arr[j] = NULL;
        }
    }
}

void dse_group_add_netengine_arr (dse_group_t *dse_group, networking_engine *net_engine, int device_idx) {
    if (dse_group == NULL)
    {
        FPRT (stderr, "ERROR: dse_group_add_netengine_arr: dse_group is NULL\n");
        assert (0);
    }
    for (int i = 0; i < dse_group->num_ases; i++) {
        dse_group->dse_arr[i].net_engine_arr[device_idx] = net_engine;
        dse_group->dse_arr[i].rpool_arr[device_idx] = net_engine->rpool;
    }
}

void dse_group_destroy (dse_group_t *dse_group)
{
    if (dse_group == NULL)
        return;
    for (int i = 0; i < dse_group->num_ases; i++)
    {
        if (dse_group->dse_arr[i].rpool != NULL)
            atomic_fetch_sub (&dse_group->dse_arr[i].rpool->ref_dses, 1);
        dse_destroy (&dse_group->dse_arr[i]);
    }
    free (dse_group->dse_arr);
    free (dse_group);
}

void dse_init (dse_group_t *dse_group, dse_t *dse, int gpu_idx)
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
    dse->dse_group = dse_group;
    dse->thread_id = atomic_fetch_add (&dse_thread_id_counter, 1);
    dse->rpool = NULL;
    dse->gpu_idx = gpu_idx;
    dse->scratchpad = aspen_calloc (dse_SCRATCHPAD_SIZE, 1);
    if (gpu_idx >= 0)
        dse->gpu_scratchpad = aspen_gpu_calloc (dse_SCRATCHPAD_SIZE, 1, gpu_idx);
    dse->thread_mutex = (pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER;
    dse->thread_cond = (pthread_cond_t)PTHREAD_COND_INITIALIZER;
    dse->ninst_cache = calloc (1, sizeof (rpool_queue_t));
    for (int i=0; i<SCHEDULE_MAX_DEVICES; i++) dse->prioritize_rpool[i] = -1;
    atomic_store (&dse->run, 0);
    atomic_store (&dse->kill, 0);
    rpool_init_queue (dse->ninst_cache);
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
        pthread_mutex_lock (&dse->thread_mutex);
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
        pthread_mutex_unlock (&dse->thread_mutex);
    }
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
    return atomic_load (&nasm->completed);
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
    // run_cudagraph (nasm);
}

void update_children (rpool_t *rpool, ninst_t *ninst, unsigned int dse_idx)
{
    #ifdef DEBUG
    if (rpool == NULL || ninst == NULL)
    {
        FPRT (stderr, "Error: Invalid arguments to update_children()\n");
        assert (0);
    }
    if (atomic_load (&ninst->state) != NINST_COMPLETED)
    {
        FPRT (stderr, "Error: ninst->state != NINST_STATE_COMPLETED in update_children()\n");
        assert (0);
    }
    #endif
    for (int i = 0; i < ninst->num_child_ninsts; i++)
    {
        ninst_t *child_ninst = ninst->child_ninst_arr[i];
        unsigned int num_parent_ninsts_completed = atomic_fetch_add (&child_ninst->num_parent_ninsts_completed, 1);
        if (num_parent_ninsts_completed == child_ninst->num_parent_ninsts - 1)
        {
            // Pseudo-mutex while ninst state change, using NINST_COMPLETED state as lock
            NINST_STATE old_state = atomic_exchange (&child_ninst->state, NINST_COMPLETED);
            if (old_state == NINST_NOT_READY) 
            {
                atomic_store (&child_ninst->state, NINST_READY);
                rpool_push_ninsts (rpool, &child_ninst, 1, 0);
            }
            else
            {
                #ifdef DEBUG
                if (old_state != NINST_COMPLETED)
                {
                    FPRT (stderr, "Error: update_children_to_cache_but_prioritize_dse_target(), old_state = %d\n", old_state);
                    assert (0);
                }
                #endif
                atomic_store (&child_ninst->state, old_state);
            }
        }
    }
}

void update_children_to_cache (rpool_queue_t *cache, ninst_t *ninst)
{
    #ifdef DEBUG
    if (cache == NULL || ninst == NULL)
    {
        FPRT (stderr, "Error: Invalid arguments to update_children_to_cache()\n");
        assert (0);
    }
    if (ninst->state != NINST_COMPLETED)
    {
        FPRT (stderr, "Error: ninst->state != NINST_STATE_COMPLETED in update_children_to_cache()\n");
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
            // Pseudo-mutex while ninst state change, using NINST_COMPLETED state as lock
            NINST_STATE old_state = atomic_exchange (&child_ninst->state, NINST_COMPLETED);
            if (old_state == NINST_NOT_READY) 
            {
                atomic_store (&child_ninst->state, NINST_READY);
                push_ninsts_to_queue (cache, &child_ninst, 1);
            }
            else
            {
                #ifdef DEBUG
                if (old_state != NINST_COMPLETED)
                {
                    FPRT (stderr, "Error: update_children_to_cache_but_prioritize_dse_target(), old_state = %d\n", old_state);
                    assert (0);
                }
                #endif
                atomic_store (&child_ninst->state, old_state);
            }
        }
    }
}

void update_children_but_prioritize_dse_target (rpool_t *rpool, ninst_t *ninst, dse_t *dse)
{
    #ifdef DEBUG
    if (ninst->state != NINST_COMPLETED)
    {
        FPRT (stderr, "Error: ninst->state != NINST_STATE_COMPLETED in update_children_but_prioritize_dse_target()\n");
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
            // Pseudo-mutex while ninst state change, using NINST_COMPLETED state as lock
            NINST_STATE old_state = atomic_exchange (&child_ninst->state, NINST_COMPLETED);
            if (old_state == NINST_NOT_READY) 
            {
                atomic_store (&child_ninst->state, NINST_READY);
                if (dse->target != NULL)
                {
                    cache[num_cache++] = child_ninst;
                }
                else
                    dse->target = child_ninst;
            }
            else
            {
                #ifdef DEBUG
                if (old_state != NINST_COMPLETED)
                {
                    FPRT (stderr, "Error: update_children_to_cache_but_prioritize_dse_target(), old_state = %d\n", old_state);
                    assert (0);
                }
                #endif
                atomic_store (&child_ninst->state, old_state);
            }
        }

    }
    rpool_push_ninsts (rpool, cache, num_cache, 0);
}

void update_children_to_cache_but_prioritize_dse_target (rpool_queue_t *cache, ninst_t *ninst, ninst_t **dse_target)
{
    #ifdef DEBUG
    if (cache == NULL || ninst == NULL)
    {
        FPRT (stderr, "Error: Invalid arguments to update_children_to_cache_but_prioritize_dse_target()\n");
        assert (0);
    }
    if (ninst->state != NINST_COMPLETED)
    {
        FPRT (stderr, "Error: ninst->state != NINST_STATE_COMPLETED in update_children_to_cache_but_prioritize_dse_target()\n");
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
           // Pseudo-mutex while ninst state change, using NINST_COMPLETED state as lock
            NINST_STATE old_state = atomic_exchange (&child_ninst->state, NINST_COMPLETED);
            if (old_state == NINST_NOT_READY) 
            {
                atomic_store (&child_ninst->state, NINST_READY);
                if (*dse_target != NULL)
                    push_ninsts_to_queue (cache, &child_ninst, 1);
                else
                    *dse_target = child_ninst;
            }
            else
            {
                #ifdef DEBUG
                if (old_state != NINST_COMPLETED)
                {
                    FPRT (stderr, "Error: update_children_to_cache_but_prioritize_dse_target(), old_state = %d\n", old_state);
                    assert (0);
                }
                #endif
                atomic_store (&child_ninst->state, old_state);
            }
        }
    }
}

void prepare_gpu_im2col (ninst_t *ninst, void **input_ptr_arr, size_t num)
{
    nasm_ldata_t *ldata = ninst->ldata;
    aspen_layer_t *layer = ldata->layer;
    nasm_ldata_t *p_ldata = (ldata->parent_ldata_idx_arr[PARENT_0] + ldata->nasm->ldata_arr);
    aspen_layer_t *p_layer = (ldata->parent_ldata_idx_arr[PARENT_0] + ldata->nasm->ldata_arr)->layer;
    unsigned int parent_stride = p_ldata->out_mat_stride;
    unsigned int num_idx = 0;
    if (layer->type == CONV_LAYER || layer->type == MAXPOOL_LAYER || layer->type == AVGPOOL_LAYER)
    {
        unsigned int mat_w = ninst->out_mat_pos[OUT_W];
        for (; mat_w < ninst->out_mat_pos[OUT_W] + ninst->tile_dims[OUT_W]; mat_w++)
        {
            unsigned int out_b = mat_w / (layer->params[OUT_H] * layer->params[OUT_W]); 
            unsigned int out_h = (mat_w % (layer->params[OUT_H] * layer->params[OUT_W])) / layer->params[OUT_W];
            unsigned int out_w = mat_w % layer->params[OUT_W];
            unsigned int in_b = out_b;
            for (int kh = 0; kh < layer->params[WEIGHT_H]; kh++)
            {
                for (int kw = 0; kw < layer->params[WEIGHT_W]; kw++)
                {
                    int in_h = out_h * layer->params[STRIDE] + kh  - layer->params[PADDING];
                    int in_w = out_w * layer->params[STRIDE] + kw  - layer->params[PADDING];
                    if (in_h < 0 || in_h >= p_layer->params[OUT_H] || in_w < 0 || in_w >= p_layer->params[OUT_W])
                    {
                        input_ptr_arr[num_idx++] = ldata->nasm->gpu_null_data;
                        continue;
                    }
                    unsigned int in_idx = in_b * p_layer->params[OUT_H] * p_layer->params[OUT_W] 
                        + in_h * p_layer->params[OUT_W] + in_w;
                    input_ptr_arr[num_idx++] = (char*)p_ldata->out_mat + in_idx*parent_stride * layer->dnn->element_size;
                }
            }
        }
    }
    else
    {
        FPRT(stderr, "ERROR: Unsupported layer type %s, at line %d in file %s\n" , layer_type_str[layer->type], 0, " ");
        assert (0);
    }
    for (; num_idx < num; num_idx++)
        input_ptr_arr[num_idx] = ldata->nasm->gpu_null_data;
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
            if (ninst->num_input_pos != 0 && rpool->gpu_idx >= 0)
            {
                nasm_ldata_t *ldata = ninst->ldata;
                aspen_layer_t *layer = ninst->ldata->layer;
                nasm_ldata_t *p_ldata = (ldata->parent_ldata_idx_arr[PARENT_0] + ldata->nasm->ldata_arr);
                const unsigned int input_pos_per_n = ninst->num_input_pos/ninst->tile_dims[OUT_W];
                size_t pos_arr_range = ninst->num_input_pos + input_pos_per_n*_BLOCK_N_SIZE;
                ninst->input_pos_ptr_arr_gpu = 
                    aspen_gpu_calloc (pos_arr_range, sizeof (void*), rpool->gpu_idx);
                void **idx_ptr_temp = aspen_calloc (pos_arr_range, sizeof (void*));
                prepare_gpu_im2col (ninst, idx_ptr_temp, pos_arr_range);
                aspen_host_to_gpu_memcpy 
                    (ninst->input_pos_ptr_arr_gpu, idx_ptr_temp, 
                        ninst->num_input_pos * sizeof (void*), rpool->gpu_idx);
                aspen_free (idx_ptr_temp);
            }
            #endif
        }
    }
    // if (rpool->gpu_idx >= 0)
    //     generate_cudagraph (nasm);
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
        int num_ase = rpool->ref_dses > 0 ? rpool->ref_dses : 1;
        update_children (rpool, ninst, i/(1 + ldata->num_ninst/num_ase));
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

size_t dse_get_ldata_size (nasm_t *nasm, unsigned int ldata_idx)
{
    if (nasm->data == NULL)
    {
        FPRT (stderr, "Error: nasm->data == NULL in dse_get_ldata_result()\n");
        assert (0);
    }
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
        return output_size;
    }
    size_t elem_size = ldata->layer->dnn->element_size;
    return ldata->out_mat_dims[OUT_H] * ldata->out_mat_dims[OUT_W] * elem_size;
}

void *dse_get_ldata_result (nasm_t *nasm, unsigned int ldata_idx, LAYER_PARAMS *order)
{
    if (nasm->data == NULL)
    {
        FPRT (stderr, "Error: nasm->data == NULL in dse_get_ldata_result()\n");
        assert (0);
    }
    nasm_ldata_t *ldata = &nasm->ldata_arr[ldata_idx];
    if (ldata->layer->type == YOLO_LAYER)
    {
        void *output = NULL;
        size_t output_size = dse_get_ldata_size (nasm, ldata_idx);
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

size_t dse_get_nasm_result_size (nasm_t *nasm)
{
    return dse_get_ldata_size (nasm, nasm->num_ldata - 1);
}