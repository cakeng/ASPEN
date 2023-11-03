#include "dse.h"

static _Atomic unsigned int dse_thread_id_counter = 0;
static _Atomic fl_path_t *dse_now_path = NULL;
static unsigned int fl_post_group_idx;

void *dse_thread_runtime (void* thread_info)
{
    dse_t *dse = (dse_t*) thread_info;
    pthread_mutex_lock(&dse->thread_mutex);
    pthread_cond_wait(&dse->thread_cond, &dse->thread_mutex); 
    while (dse->kill == 0 || dse->target != NULL)
    {
        dse_schedule (dse);
        if (dse->run == 0 && dse->target == NULL)
            pthread_cond_wait(&dse->thread_cond, &dse->thread_mutex); 
    }
    return NULL;
}

void dse_schedule (dse_t *dse)
{
    if (dse->operating_mode == OPER_MODE_FL_PATH) {
        dse_schedule_fl (dse);
        return;
    }
    // print_rpool_info (dse->rpool);
    // print_rpool_queue_info (dse->ninst_cache);
    
    // if ((dse->ninst_cache->num_stored < DSE_NINST_CACHE_BALLANCE - DSE_NINST_CACHE_DIFF) || 
    //     dse->ninst_cache->num_stored == 0)
    int target_device = dse->device_idx;
    if(dse->device_mode == DEV_LOCAL || dse->profile_compute)
        dse->target_device = target_device;
    else if (dse->device_mode == DEV_SERVER)
    {
        int min_value = 0;
        int max_value = dse->num_edge_devices - 1;
        target_device = rand() % (max_value - min_value + 1) + min_value;
        dse->target_device = target_device;
    }
    if (dse->target == NULL && (dse->run != 0 && dse->kill == 0))
    {
        if (dse->is_multiuser_case)
        {
            if (dse->device_mode != DEV_SERVER) 
            {
                dse->target_device = target_device;
                rpool_fetch_ninsts (dse->rpool_arr[target_device], &dse->target, 1, 0);
                if (dse->target == NULL)
                    return;
            }
            else
            {
                rpool_fetch_ninsts (dse->rpool_arr[target_device], &dse->target, 1, 0);
                dse->target_device = target_device;
                
                if (dse->target == NULL) 
                    return;
            }
        }
        else 
        {
            if (dse->rpool->is_core_exclusive) {
                rpool_fetch_ninsts_from_group (dse->rpool, &dse->target, 1, dse->thread_id);
            }
            else {
                rpool_fetch_ninsts (dse->rpool, &dse->target, 1, 0);
            }
            if (dse->target == NULL)
                return;
        }
        // unsigned int fetch_num = 
        //     rpool_fetch_ninsts (dse->rpool, dse->scratchpad, DSE_NINST_CACHE_BALLANCE - dse->ninst_cache->num_stored);
        // push_ninsts_to_queue (dse->ninst_cache, dse->scratchpad, fetch_num);
        // PRTF ("Thread %d fetched %d ninsts from rpool\n", dse->thread_id, fetch_num);
        // #ifdef DEBUG
        // PRTF ("Thread %d fetched %d ninsts from rpool\n", dse->thread_id, fetch_num);
        // print_rpool_info (dse->rpool);
        // print_rpool_queue_info (dse->ninst_cache);
        // #endif
    }
    if (dse->target == NULL)
        return;
    // YELLOW_PRTF ("ninst %d(L%d) fetched by DSE %d.\n", dse->target->ninst_idx, dse->target->ldata->layer->layer_idx, dse->thread_id);
    // else if (dse->ninst_cache->num_stored > DSE_NINST_CACHE_BALLANCE + DSE_NINST_CACHE_DIFF)
    // if (dse->ninst_cache->num_stored > 0)
    // {
    //     // unsigned int push_num = 
    //     //     pop_ninsts_from_queue_back (dse->ninst_cache, dse->scratchpad, dse->ninst_cache->num_stored - DSE_NINST_CACHE_BALLANCE);
    //     rpool_push_ninsts (dse->rpool, dse->ninst_cache->ninst_ptr_arr, dse->ninst_cache->num_stored);
    //     dse->ninst_cache->num_stored = 0;
    //     dse->ninst_cache->idx_end = 0;
    //     // PRTF ("Thread %d pushed %d ninsts to rpool\n", dse->thread_id, push_num);
    //     // #ifdef DEBUG
    //     // PRTF ("Thread %d pushed %d ninsts to rpool\n", dse->thread_id, push_num);
    //     // print_rpool_info (dse->rpool);
    //     // print_rpool_queue_info (dse->ninst_cache);
    //     // #endif
    // }

    // unsigned int num_ninsts = dse->ninst_cache->num_stored;
    // for (int i = 0; i < num_ninsts; i++)
    // {
    //     ninst_t *ninst;
    //     pop_ninsts_from_queue (dse->ninst_cache, &ninst, 1);
        // PRTF ("Thread %d running ninst #%d - N%d:L%d:%d\n", dse->thread_id, i,
        //         ninst->ldata->nasm->nasm_id, ninst->ldata->layer->layer_idx, ninst->ninst_idx);
        // #ifdef DEBUG
        // if (ninst == NULL)
        // {
        //     ERROR_PRTF ("ERROR: dse_thread_runtime: ninst is NULL\n");
        //     assert (0);
        // }
        // else 
        // {
        //     PRTF ("Thread %d running ninst #%d - N%d:L%d:%d\n", dse->thread_id, i,
        //         ninst->ldata->nasm->nasm_id, ninst->ldata->layer->layer_idx, ninst->ninst_idx);
        // }
        // if (ninst->state != NINST_READY)
        // {
        //     ERROR_PRTF ("ERROR: dse_thread_runtime: ninst->state != NINST_READY\n");
        //     assert (0);
        // }
        // #endif
        // Execute.
        ninst_t *ninst = dse->target;
        dse->target = NULL;
        if (atomic_exchange (&ninst->state, NINST_COMPLETED) == NINST_COMPLETED)
            return;
        // GREEN_PRTF ("ninst %d(L%d) completed by DSE %d.\n", ninst->ninst_idx, ninst->ldata->layer->layer_idx, dse->thread_id);

        // printf("[Device: %d] Fetched ninst %d, dev_to_compute[DEV_SERVER]: %d, dev_to_compute[DEV_EDGE]: %d\n", 
        //                                                                                     target_device,
        //                                                                                     ninst->ninst_idx, 
        //                                                                                     ninst->dev_to_compute[dse->num_edge_devices],
        //                                                                                     ninst->dev_to_compute[dse->device_idx]);
        if (dse->rpool->is_core_exclusive && !is_core_compute(ninst, dse->thread_id)) {
            printf("UNALLOWED NINST!!! ninst %d (allowed: %d %d) to dse %d\n", ninst->ninst_idx, ninst->core_allowed[0], ninst->core_allowed[1], dse->thread_id);
        }
        if (is_dev_compute(ninst, dse->device_idx) || dse->profile_compute)    // It's mine, so compute
        {
            printf("\t[Device %d] Compute ninst (N%d L%d)\n", target_device, ninst->ninst_idx, ninst->ldata->layer->layer_idx);
            if (dse->profile_compute) ninst->compute_start = get_time_secs_offset ();
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
                    // ERROR_PRTF ("ERROR: dse_thread_runtime: layer type %s is not supported\n", layer_type_str[ninst->ldata->layer->type]);
                    break;
            }

            //For logging
            ninst->computed_time = get_time_secs_offset ();
            if (dse->profile_compute) ninst->compute_end = ninst->computed_time;
            ninst->dse_idx = dse->thread_id;

            // For dynamic offloading
            if(dse->is_dynamic_scheduling)
            {
                // For edge side offloading decision
                if(dse->device_mode == DEV_EDGE)
                {
                    
                    float eft_edge = get_eft_edge(dse->dynamic_scheduler, dse->rpool_arr[target_device], dse->device_idx, dse->net_engine_arr[target_device]->dse_group->num_dess, ninst->num_child_ninsts);
                    float eft_server = get_eft_server(dse->dynamic_scheduler, dse->net_engine_arr[target_device], dse->device_idx, ninst->tile_dims[OUT_H] * ninst->tile_dims[OUT_W] * sizeof(float));
                    
                    ninst->eft_edge = eft_edge;
                    ninst->eft_server = eft_server;

                    if(eft_server <= eft_edge)
                        ninst_set_send_target_device(ninst, dse->num_edge_devices);
                }
            }
            
            unsigned int num_ninst_completed = atomic_fetch_add (&ninst->ldata->num_ninst_completed, 1);
            if (num_ninst_completed == ninst->ldata->num_ninst - 1)
            {
                // printf ("\t\t ldata %d completed\n", ninst->ldata->layer->layer_idx);
                for (int pidx = 0; pidx < NUM_PARENT_ELEMENTS; pidx++)
                {
                    if (ninst->ldata->parent_ldata_idx_arr[pidx] == -1)
                        continue;
                    nasm_ldata_t *parent_ldata = &ninst->ldata->nasm->ldata_arr[ninst->ldata->parent_ldata_idx_arr[pidx]];
                    unsigned int num_child_ldata_completed = atomic_fetch_add (&parent_ldata->num_child_ldata_completed, 1);
                    if (num_child_ldata_completed + 1 == parent_ldata->num_child_ldata && (parent_ldata != parent_ldata->nasm->ldata_arr))
                    {
                        free_ldata_out_mat (parent_ldata);
                        // YELLOW_PRTF ("ldata %d output freed by DSE %d.\n", parent_ldata->layer->layer_idx, dse->thread_id);
                    }
                }

                nasm_t *nasm = ninst->ldata->nasm;
                atomic_fetch_add (&nasm->num_ldata_completed, 1);
                if (ninst->ldata == &nasm->ldata_arr[nasm->num_ldata - 1])
                {
                    // printf ("\t\tSignaling nasm completion...\n");
                    // All layers of the nasm is completed.
                    atomic_store (&nasm->completed, 1);
                    rpool_queue_group_t *rpool_queue_group;
                    if (dse->is_multiuser_case) {
                        rpool_queue_group = get_queue_group_from_nasm (dse->rpool_arr[dse->target_device], ninst->ldata->nasm);
                        set_queue_group_weight (dse->rpool_arr[dse->target_device], rpool_queue_group, 0);
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
            if (dse->is_multiuser_case && dse->device_mode == DEV_SERVER) {
                update_children_but_prioritize_dse_target (dse->rpool_arr[dse->target_device], ninst, dse);
            }
            else if (dse->is_multiuser_case && dse->device_mode != DEV_SERVER) {
                update_children_but_prioritize_dse_target (dse->rpool_arr[dse->target_device], ninst, dse);
            }
            else if (!dse->is_multiuser_case && dse->is_dynamic_scheduling && ninst->ldata->layer->layer_idx == 0) {
                update_children (dse->rpool, ninst);
            }
            else {
                update_children_but_prioritize_dse_target (dse->rpool, ninst, dse);
            }

            // check devices to send to for the computation output
            if (!dse->profile_compute && dse->is_multiuser_case) 
            {
                for (int i = 0; i <= dse->num_edge_devices; i++)
                {
                    if (i == dse->device_idx) continue;
                    if (atomic_load(&ninst->dev_send_target[i]) == 1) // Should be offload
                    {
                        networking_engine *net_engine;
                        if(dse->device_mode == DEV_SERVER) net_engine = dse->net_engine_arr[i];
                        else if(dse->device_mode == DEV_EDGE) net_engine = dse->net_engine_arr[dse->device_idx];
                        else continue;
                        create_network_buffer_for_ninst (ninst);
                        pthread_mutex_lock(&net_engine->tx_queue->queue_mutex);
                        push_ninsts_to_net_queue(net_engine->tx_queue, &ninst, 1);                            
                        pthread_mutex_unlock(&net_engine->tx_queue->queue_mutex);
                    }
                }
            }
            else if (!dse->profile_compute && !dse->is_multiuser_case)
            {
                for (int i = 0; i <= dse->num_edge_devices; i++) 
                {
                    if (i == dse->device_idx) continue;
                    if (atomic_load(&ninst->dev_send_target[i]) == 1) 
                    {
                        networking_engine *net_engine;
                        if(dse->device_mode == DEV_SERVER) net_engine = dse->net_engine_arr[i];
                        else if(dse->device_mode == DEV_EDGE) net_engine = dse->net_engine_arr[dse->device_idx];
                        else continue;
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

void dse_schedule_fl (dse_t *dse) {
    /******************************************
    
    DSE RUNNING FUNCTION FOR FL_OFFLOADING MODE
    * Commonly,
      - State of ninst only affects computing option: real or dummy.
      - When every path was completed, change operating mode of dse and nasm to default and continue.
      - rpool_group is for each layer.

    * For DEV_LOCAL,
      - Every ninst will be given in ready_pool, so never care ninst updates.
      - If one ninst is completed, check if a path is finished then change path to next one.

    * For DEV_EDGE,
      - Every ninst will be given in ready_pool, so never care ninst updates.
      - If one ninst is completed, check if a path is fully offloaded then change path to next one.
      
    * For DEV_SERVER,
      - When target ninst was updated, check if it is in a right path and a right layer.
        + if not, push the ninst into rpool.
    
    ******************************************/

    /* SET TARGET NINST */
    if (dse->target == NULL && (dse->run != 0 && dse->kill == 0)) {
        fl_path_t *now_path = atomic_load(&dse_now_path);
        if (dse->operating_mode == OPER_MODE_FL_PATH && now_path == NULL) {
            rpool_fetch_ninsts_from_group (dse->rpool, &dse->target, 1, 0);
            if (dse->target != NULL) {
                atomic_store(&dse_now_path, dse->target->ldata->nasm->path_ptr_arr[dse->target->ldata->nasm->path_now_idx]);
            }
        }
        else if (dse->operating_mode == OPER_MODE_FL_PATH) {
            unsigned int nplc = atomic_load(&now_path->num_path_layers_completed);
            if (nplc != now_path->num_path_layers) {
                rpool_fetch_ninsts_from_group (dse->rpool, &dse->target, 1, nplc);
            }
            else return;
        }
        else {
            rpool_fetch_ninsts_from_group (dse->rpool, &dse->target, 1, fl_post_group_idx);
        }
    }
        
    
    if (dse->target == NULL)
        return;

    /* EXECUTE NINST */
    ninst_t *ninst = dse->target;
    dse->target = NULL;

    // Check if the ninst is mine(device)
    if (is_dev_compute(ninst, dse->device_idx) || dse->profile_compute) {
        // Good! it's mine

        // Check the timing to compute
        nasm_t *nasm;
        unsigned int path_now_idx;
        fl_path_t *path;
        unsigned int ninst_layer_idx;
        unsigned int pnplc;
        fl_path_layer_t *path_layer;
        if (dse->operating_mode == OPER_MODE_FL_PATH) {
            nasm = ninst->ldata->nasm;
            path_now_idx = atomic_load(&nasm->path_now_idx);
            path = nasm->path_ptr_arr[path_now_idx];
            if (path == NULL) return;
            ninst_layer_idx = ninst->ldata->layer->layer_idx;
            pnplc = atomic_load(&path->num_path_layers_completed);
            path_layer = &path->path_layers_arr[pnplc];

            if (!fl_is_ninst_in_path_layer(path_layer, ninst)) {
                // If it's not a good time to compute this ninst, return the ninst into rpool
                printf("\tReject ninst (N %d, L %d)\n", ninst->ninst_idx, ninst->ldata->layer->layer_idx);
                rpool_push_ninsts_to_group(dse->rpool, &ninst, 1, ninst->ldata->layer->layer_idx - 1);
                return;
            }
        }
        
        
        printf("\tCompute ninst (N %d, L %d)\n", ninst->ninst_idx, ninst->ldata->layer->layer_idx);
        if (dse->profile_compute) ninst->compute_start = get_time_secs_offset ();
        if (ninst->state == NINST_COMPLETED) ninst->compute_option = NINST_COMPUTE_DUMMY;

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
                // ERROR_PRTF ("ERROR: dse_thread_runtime: layer type %s is not supported\n", layer_type_str[ninst->ldata->layer->type]);
                break;
        }

        // For logging
        ninst->computed_time = get_time_secs_offset ();
        if (dse->profile_compute) ninst->compute_end = ninst->computed_time;
        ninst->dse_idx = dse->thread_id;

        // Check devices to send the computation output
        if (!dse->profile_compute && dse->is_multiuser_case) 
        {
            for (int i = 0; i <= dse->num_edge_devices; i++)
            {
                if (i == dse->device_idx) continue;
                if (atomic_load(&ninst->dev_send_target[i]) == 1) // Should be offload
                {
                    networking_engine *net_engine;
                    if(dse->device_mode == DEV_SERVER) net_engine = dse->net_engine_arr[i];
                    else if(dse->device_mode == DEV_EDGE) net_engine = dse->net_engine_arr[dse->device_idx];
                    else continue;
                    create_network_buffer_for_ninst (ninst);
                    pthread_mutex_lock(&net_engine->tx_queue->queue_mutex);
                    push_ninsts_to_net_queue(net_engine->tx_queue, &ninst, 1);                            
                    pthread_mutex_unlock(&net_engine->tx_queue->queue_mutex);
                }
            }
        }
        else if (!dse->profile_compute && !dse->is_multiuser_case)
        {
            for (int i = 0; i <= dse->num_edge_devices; i++) 
            {
                if (i == dse->device_idx) continue;
                if (atomic_load(&ninst->dev_send_target[i]) == 1) 
                {
                    networking_engine *net_engine;
                    if(dse->device_mode == DEV_SERVER) net_engine = dse->net_engine_arr[i];
                    else if(dse->device_mode == DEV_EDGE) net_engine = dse->net_engine_arr[dse->device_idx];
                    else continue;
                    create_network_buffer_for_ninst (ninst);
                    pthread_mutex_lock(&net_engine->tx_queue->queue_mutex);
                    push_ninsts_to_net_queue(net_engine->tx_queue, &ninst, 1);
                    pthread_mutex_unlock(&net_engine->tx_queue->queue_mutex);
                }
            }
        }

        // Update ninst only for last layer
        if (ninst->ldata->layer->layer_idx == path->path_layers_arr) {
            update_children(dse->rpool, ninst);
        }

        // Update path layer info
        if (dse->operating_mode == OPER_MODE_FL_PATH) {
            unsigned int num_player_ninsts_completed = atomic_fetch_add(&path_layer->num_ninsts_completed, 1);
            if (num_player_ninsts_completed == path_layer->num_ninsts - 1) {
                printf("PATH_LAYER %d COMPLETE!\n", path_layer->ldata->layer->layer_idx);
                unsigned int num_path_layers_completed = atomic_fetch_add(&path->num_path_layers_completed, 1) + 1;
                // atomic_fetch_add(&dse_now_path_layer_idx, 1);
                
                int change_path = 0;
                if (dse->device_mode == DEV_EDGE) {
                    unsigned int last_layer_num_ninst_complete = atomic_load(
                        &path->path_layers_arr[path->edge_final_layer_idx].num_ninsts_completed
                    );
                    if (last_layer_num_ninst_complete == path->path_layers_arr[path->edge_final_layer_idx].num_ninsts) {
                        printf("EDGE STOPPING AT LAYER %d\n", path->edge_final_layer_idx + 1);
                        change_path = 1;
                    }
                }
                else {
                    unsigned int last_layer_num_ninst_complete = atomic_load(
                        &path->path_layers_arr[path->num_path_layers - 1].num_ninsts_completed
                    );
                    if (last_layer_num_ninst_complete == path->path_layers_arr[path->num_path_layers - 1].num_ninsts) {
                        printf("NONEDGE STOPPING AT LAYER %d\n", path->edge_final_layer_idx + 1);
                        change_path = 1;
                    }
                }
                if (change_path) {
                    // Change path!
                    printf("PATH %d FINISHED!\n", path->path_idx);
                    unsigned int next_path_idx = atomic_fetch_add(&nasm->path_now_idx, 1) + 1;

                    // atomic_store(&dse_now_path_layer_idx, 0);
                    atomic_store(&dse_now_path, nasm->path_ptr_arr[next_path_idx]);
                    if (next_path_idx == nasm->num_paths) {
                        printf("FL_PATH MODE FINISHED!\n");
                        // Change mode!
                        fl_post_group_idx = next_path_idx;
                        fl_push_ninsts_only(dse->rpool, nasm, path->num_path_layers + 1, 0);
                        dse_group_set_operating_mode(dse->dse_group, OPER_MODE_DEFAULT);
                        return;
                    }

                    // Push new ninsts into rpool
                    fl_path_t *next_path = nasm->path_ptr_arr[nasm->path_now_idx];
                    fl_push_path_ninsts(dse->rpool, next_path);
                    // fl_push_path_ninsts_until(dse->rpool, next_path, 2);
                    // next_path->edge_final_layer_idx = 2;
                }
            }
        }
    }
}

dse_group_t *dse_group_init (unsigned int num_des, int gpu_idx)
{
    if (gpu_idx >= 0 && gpu_idx >= aspen_num_gpus)
    {
        ERROR_PRTF ("ERROR: dse_group_init: gpu_idx %d is out of range... Falling back to CPU\n", gpu_idx);
        gpu_idx = -1;
    }
    else if (gpu_idx >= 0 && gpu_idx < aspen_num_gpus)
    {
        num_des = num_des > GPU_RUN_STREAM_NUM ? GPU_RUN_STREAM_NUM : num_des;
    }
    dse_group_t *dse_group = (dse_group_t *) calloc (1, sizeof (dse_group_t));
    dse_group->num_dess = num_des;
    if (gpu_idx < 0)
        dse_group->gpu_idx = -1;
    else
        dse_group->gpu_idx = gpu_idx;
    dse_group->dse_arr = (dse_t *) calloc (num_des, sizeof (dse_t));
    for (int i = 0; i < num_des; i++)
    {
        dse_init (dse_group, &dse_group->dse_arr[i], dse_group->gpu_idx);
    }
    return dse_group;
}

void dse_group_set_rpool (dse_group_t *dse_group, rpool_t *rpool)
{
    if (dse_group == NULL)
    {
        ERROR_PRTF ("ERROR: dse_group_set_rpool: dse_group is NULL\n");
        assert (0);
    }
    if (rpool == NULL)
    {
        ERROR_PRTF ("ERROR: dse_group_set_rpool: rpool is NULL\n");
        assert (0);
    }
    if (dse_group->gpu_idx != rpool->gpu_idx)
    {
        ERROR_PRTF ("ERROR: dse_group_set_rpool: dse_group->gpu_idx %d != rpool->gpu_idx %d\n", dse_group->gpu_idx, rpool->gpu_idx);
        assert (0);
    }
    for (int i = 0; i < dse_group->num_dess; i++)
    {
        dse_group->dse_arr[i].rpool = rpool;
    }
    add_ref_dses (rpool, dse_group->num_dess);
}

void dse_group_set_net_engine (dse_group_t *dse_group, networking_engine *net_engine)
{
    if (dse_group == NULL)
    {
        ERROR_PRTF ("ERROR: dse_group_set_net_engine: dse_group is NULL\n");
        assert (0);
    }
    if (net_engine == NULL)
    {
        ERROR_PRTF ("ERROR: dse_group_set_net_engine: net_engine is NULL\n");
        assert (0);
    }
    for (int i = 0; i < dse_group->num_dess; i++)
    {
        dse_group->dse_arr[i].net_engine = net_engine;
    }
}

void dse_group_set_device_mode (dse_group_t *dse_group, DEVICE_MODE device_mode)
{
    if (dse_group == NULL)
    {
        ERROR_PRTF ("ERROR: dse_group_set_device_mode: dse_group is NULL\n");
        assert (0);
    }
    for (int i = 0; i < dse_group->num_dess; i++)
    {
        dse_group->dse_arr[i].device_mode = device_mode;
    }
}

void dse_group_set_device (dse_group_t *dse_group, int device_idx)
{
    if (dse_group == NULL)
    {
        ERROR_PRTF ("ERROR: dse_group_set_device: dse_group is NULL\n");
        assert (0);
    }
    for (int i = 0; i < dse_group->num_dess; i++)
    {
        dse_group->dse_arr[i].device_idx = device_idx;
    }
}

void dse_group_set_operating_mode (dse_group_t *dse_group, int operating_mode) {
    if (dse_group == NULL)
    {
        ERROR_PRTF ("ERROR: dse_group_set_device: dse_group is NULL\n");
        assert (0);
    }
    for (int i = 0; i < dse_group->num_dess; i++)
    {
        dse_group->dse_arr[i].operating_mode = operating_mode;
    }
}

void dse_group_set_num_edge_devices (dse_group_t *dse_group, int num_edge_devices)
{
    if (dse_group == NULL)
    {
        ERROR_PRTF ("ERROR: dse_group_set_num_edge_devices: dse_group is NULL\n");
        assert (0);
    }
    for (int i = 0; i < dse_group->num_dess; i++)
    {
        dse_group->dse_arr[i].num_edge_devices = num_edge_devices;
    }
}

void dse_group_set_profile (dse_group_t *dse_group, int profile_compute)
{
    if (dse_group == NULL)
    {
        ERROR_PRTF ("ERROR: dse_group_set_device: dse_group is NULL\n");
        assert (0);
    }
    for (int i = 0; i < dse_group->num_dess; i++)
    {
        dse_group->dse_arr[i].profile_compute = profile_compute;
    }
}

void dse_group_set_dynamic_scheduler (dse_group_t *dse_group, dynamic_scheduler_t* dynamic_scheduler)
{
    if (dse_group == NULL)
    {
        ERROR_PRTF ("ERROR: dse_group_set_scheduler: dse_group is NULL\n");
        assert (0);
    }
    for (int i = 0; i < dse_group->num_dess; i++)
    {
        dse_group->dse_arr[i].dynamic_scheduler = dynamic_scheduler;
        dse_group->dse_arr[i].is_dynamic_scheduling = 1;
    }
}

void dse_group_set_multiuser (dse_group_t *dse_group, int is_multiuser_case) {
    if (dse_group == NULL)
    {
        ERROR_PRTF ("ERROR: dse_group_set_multiuser: dse_group is NULL\n");
        assert (0);
    }
    for (int i = 0; i < dse_group->num_dess; i++)
    {
        dse_group->dse_arr[i].is_multiuser_case = is_multiuser_case;
    }
}

void dse_group_add_prioritize_rpool (dse_group_t *dse_group, int device_idx) {
    for (int i=0; i<dse_group->num_dess; i++) {
        for (int j=0; j<SCHEDULE_MAX_DEVICES; j++) {
            if (dse_group->dse_arr[i].prioritize_rpool[j] == -1)
                dse_group->dse_arr[i].prioritize_rpool[j] = device_idx;
        }
    }
}

void dse_group_init_enable_device(dse_group_t *dse_group, int num_edge_devices) {
    for (int i = 0; i < dse_group->num_dess; i++) {
        for (int j = 1; j <= num_edge_devices; j++)
        dse_group->dse_arr[i].enabled_device[j] = 0;
    }
}

void dse_group_set_enable_device(dse_group_t *dse_group, int device_idx, int enable) {
    for (int i = 0; i < dse_group->num_dess; i++) {
        dse_group->dse_arr[i].enabled_device[device_idx] = enable;
    }
}

void dse_group_add_rpool_arr(dse_group_t *dse_group, rpool_t *rpool, int device_idx) {
    if (rpool == NULL)
    {
        ERROR_PRTF ("ERROR: dse_group_add_rpool_arr: rpool is NULL\n");
        assert (0);
    }
    for (int i = 0; i < dse_group->num_dess; i++) {
        dse_group->dse_arr[i].rpool_arr[device_idx] = rpool;
    }
}

void dse_group_init_netengine_arr (dse_group_t *dse_group) {
    if (dse_group == NULL)
    {
        ERROR_PRTF ("ERROR: dse_group_init_netengine_arr: dse_group is NULL\n");
        assert (0);
    }
    for (int i = 0; i < dse_group->num_dess; i++) {
        for (int j=0; j<SCHEDULE_MAX_DEVICES; j++) {
            dse_group->dse_arr[i].net_engine_arr[j] = NULL;
        }
    }
}

void dse_group_add_netengine_arr (dse_group_t *dse_group, networking_engine *net_engine, int device_idx) {
    if (dse_group == NULL)
    {
        ERROR_PRTF ("ERROR: dse_group_add_netengine_arr: dse_group is NULL\n");
        assert (0);
    }
    for (int i = 0; i < dse_group->num_dess; i++) {
        dse_group->dse_arr[i].net_engine_arr[device_idx] = net_engine;
        dse_group->dse_arr[i].rpool_arr[device_idx] = net_engine->rpool;
    }
}

void dse_group_destroy (dse_group_t *dse_group)
{
    if (dse_group == NULL)
        return;
    for (int i = 0; i < dse_group->num_dess; i++)
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
        ERROR_PRTF ("ERROR: dse_init: dse is NULL\n");
        assert (0);
    }
    if (gpu_idx >= 0 && gpu_idx >= aspen_num_gpus)
    {
        ERROR_PRTF ("ERROR: dse_init: gpu_idx %d is out of range... Falling back to CPU\n", gpu_idx);
        gpu_idx = -1;
    }
    dse->dse_group = dse_group;
    dse->thread_id = atomic_fetch_add (&dse_thread_id_counter, 1);
    dse->rpool = NULL;
    dse->gpu_idx = gpu_idx;
    dse->scratchpad = aspen_calloc (DSE_SCRATCHPAD_SIZE, 1);
    if (gpu_idx >= 0)
        dse->gpu_scratchpad = aspen_gpu_calloc (DSE_SCRATCHPAD_SIZE, 1, gpu_idx);
    dse->thread_mutex = (pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER;
    dse->thread_cond = (pthread_cond_t)PTHREAD_COND_INITIALIZER;
    dse->ninst_cache = calloc (1, sizeof (rpool_queue_t));
    for (int i=0; i<SCHEDULE_MAX_DEVICES; i++) dse->prioritize_rpool[i] = -1;
    atomic_store (&dse->run, 0);
    atomic_store (&dse->kill, 0);
    rpool_init_queue (dse->ninst_cache);
    pthread_create (&dse->thread, NULL, dse_thread_runtime, (void*)dse);
    dse->operating_mode = OPER_MODE_DEFAULT;
}

void dse_destroy (dse_t *dse)
{
    if (dse == NULL)
        return;
    if (atomic_load (&dse->run) == 1)
    {
        ERROR_PRTF ("ERROR: Tried to destroy dse while it is running.\n");
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
        ERROR_PRTF ("ERROR: dse_run: dse is NULL\n");
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
        ERROR_PRTF ("ERROR: dse_stop: dse is NULL\n");
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
        ERROR_PRTF ("ERROR: dse_group_run: dse_group is NULL\n");
        assert (0);
    }
    for (int i = 0; i < dse_group->num_dess; i++)
    {
        dse_run (&dse_group->dse_arr[i]);
    }
}

void dse_group_stop (dse_group_t *dse_group)
{
    if (dse_group == NULL)
    {
        ERROR_PRTF ("ERROR: dse_group_stop: dse_group is NULL\n");
        assert (0);
    }
    for (int i = 0; i < dse_group->num_dess; i++)
    {
        dse_stop (&dse_group->dse_arr[i]);
    }
}

unsigned int dse_check_nasm_completion (nasm_t *nasm)
{
    #ifdef DEBUG
    if (nasm == NULL)
    {
        ERROR_PRTF ("ERROR: dse_check_nasm_completion: nasm is NULL\n");
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
        ERROR_PRTF ("ERROR: dse_cudagraph_run: nasm is NULL\n");
        assert (0);
    }
    if (nasm->gpu_idx < 0)
    {
        ERROR_PRTF ("ERROR: dse_cudagraph_run: gpu not initialized.\n");
        assert (0);
    }
    rpool_finish_nasm (rpool, nasm);
    // run_cudagraph (nasm);
}

void update_children (rpool_t *rpool, ninst_t *ninst)
{
    #ifdef DEBUG
    if (rpool == NULL || ninst == NULL)
    {
        ERROR_PRTF ("Error: Invalid arguments to update_children()\n");
        assert (0);
    }
    if (atomic_load (&ninst->state) != NINST_COMPLETED)
    {
        ERROR_PRTF ("Error: ninst->state != NINST_STATE_COMPLETED in update_children()\n");
        assert (0);
    }
    #endif
    for (int i = 0; i < ninst->num_child_ninsts; i++)
    {
        ninst_t *child_ninst = ninst->child_ninst_arr[i];
        unsigned int num_parent_ninsts_completed = atomic_fetch_add (&child_ninst->num_parent_ninsts_completed, 1);
        // BLUE_PRTF ("1 Ninst %d(L%d) incrementing child ninst %d(L%d), parent_comp = %d/%d\n", ninst->ninst_idx, ninst->ldata->layer->layer_idx, 
        //     child_ninst->ninst_idx, child_ninst->ldata->layer->layer_idx, num_parent_ninsts_completed, child_ninst->num_parent_ninsts);
        // if (num_parent_ninsts_completed >= child_ninst->num_parent_ninsts)
        //     assert (0);
        if (num_parent_ninsts_completed == child_ninst->num_parent_ninsts - 1)
        {
            // Pseudo-mutex while ninst state change, using NINST_COMPLETED state as lock
            NINST_STATE old_state = atomic_exchange (&child_ninst->state, NINST_COMPLETED);
            if (old_state == NINST_NOT_READY) 
            {
                // BLUE_PRTF ("1 Ninst %d(L%d) ready, pushing to rpool\n", child_ninst->ninst_idx, child_ninst->ldata->layer->layer_idx);
                atomic_store (&child_ninst->state, NINST_READY);
                if (rpool->is_core_exclusive) {
                    int target_dse = get_allowed_core_idx(child_ninst);
                    rpool_push_ninsts_to_group (rpool, &child_ninst, 1, target_dse);
                }
                else {
                    rpool_push_ninsts (rpool, &child_ninst, 1, 0);
                }
            }
            else
            {
                #ifdef DEBUG
                if (old_state != NINST_COMPLETED)
                {
                    ERROR_PRTF ("Error: update_children_to_cache_but_prioritize_dse_target(), old_state = %d\n", old_state);
                    assert (0);
                }
                #endif
                atomic_store (&child_ninst->state, old_state);
            }
            if (child_ninst->ldata->out_mat == NULL)
                alloc_ldata_out_mat (child_ninst->ldata);
        }
    }
}

void update_children_to_cache (rpool_queue_t *cache, ninst_t *ninst)
{
    #ifdef DEBUG
    if (cache == NULL || ninst == NULL)
    {
        ERROR_PRTF ("Error: Invalid arguments to update_children_to_cache()\n");
        assert (0);
    }
    if (ninst->state != NINST_COMPLETED)
    {
        ERROR_PRTF ("Error: ninst->state != NINST_STATE_COMPLETED in update_children_to_cache()\n");
        assert (0);
    }
    #endif
    if (ninst->state != NINST_COMPLETED)
        return;
    for (int i = 0; i < ninst->num_child_ninsts; i++)
    {
        ninst_t *child_ninst = ninst->child_ninst_arr[i];
        unsigned int num_parent_ninsts_completed = atomic_fetch_add (&child_ninst->num_parent_ninsts_completed, 1);
        // BLUE_PRTF ("2 Ninst %d(L%d) incrementing child ninst %d(L%d), parent_comp = %d/%d\n", ninst->ninst_idx, ninst->ldata->layer->layer_idx, 
        //     child_ninst->ninst_idx, child_ninst->ldata->layer->layer_idx, num_parent_ninsts_completed, child_ninst->num_parent_ninsts);
        // if (num_parent_ninsts_completed >= child_ninst->num_parent_ninsts)
        //     assert (0);
        if (num_parent_ninsts_completed == child_ninst->num_parent_ninsts - 1)
        {
            // Pseudo-mutex while ninst state change, using NINST_COMPLETED state as lock
            NINST_STATE old_state = atomic_exchange (&child_ninst->state, NINST_COMPLETED);
            if (old_state == NINST_NOT_READY) 
            {
                // BLUE_PRTF ("2 Ninst %d(L%d) ready, pushing to rpool\n", child_ninst->ninst_idx, child_ninst->ldata->layer->layer_idx);
                atomic_store (&child_ninst->state, NINST_READY);
                push_ninsts_to_queue (cache, &child_ninst, 1);
            }
            else
            {
                #ifdef DEBUG
                if (old_state != NINST_COMPLETED)
                {
                    ERROR_PRTF ("Error: update_children_to_cache_but_prioritize_dse_target(), old_state = %d\n", old_state);
                    assert (0);
                }
                #endif
                atomic_store (&child_ninst->state, old_state);
            }
            if (child_ninst->ldata->out_mat == NULL)
                alloc_ldata_out_mat (child_ninst->ldata);
        }
    }
}

void update_children_but_prioritize_dse_target (rpool_t *rpool, ninst_t *ninst, dse_t *dse)
{
    #ifdef DEBUG
    if (ninst->state != NINST_COMPLETED)
    {
        ERROR_PRTF ("Error: ninst->state != NINST_STATE_COMPLETED in update_children_but_prioritize_dse_target()\n");
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
        // BLUE_PRTF ("3 Ninst %d(L%d) incrementing child ninst %d(L%d), parent_comp = %d/%d\n", ninst->ninst_idx, ninst->ldata->layer->layer_idx, 
        //     child_ninst->ninst_idx, child_ninst->ldata->layer->layer_idx, num_parent_ninsts_completed, child_ninst->num_parent_ninsts);
        // if (num_parent_ninsts_completed >= child_ninst->num_parent_ninsts)
        //     assert (0);
        if (num_parent_ninsts_completed == child_ninst->num_parent_ninsts - 1)
        {
            // Pseudo-mutex while ninst state change, using NINST_COMPLETED state as lock
            NINST_STATE old_state = atomic_exchange (&child_ninst->state, NINST_COMPLETED);
            if (old_state == NINST_NOT_READY) 
            {
                
                atomic_store (&child_ninst->state, NINST_READY);
                int target_dse = get_allowed_core_idx(child_ninst);
                if (rpool->is_core_exclusive) {
                    if (dse->target != NULL || dse->thread_id != target_dse)
                    {
                        cache[num_cache++] = child_ninst;
                        // BLUE_PRTF ("31 Ninst %d(L%d) ready, pushing to rpool\n", child_ninst->ninst_idx, child_ninst->ldata->layer->layer_idx);
                    }
                    else
                    {
                        // BLUE_PRTF ("32 Ninst %d(L%d) ready, pushing to rpool\n", child_ninst->ninst_idx, child_ninst->ldata->layer->layer_idx);
                        dse->target = child_ninst;
                    }
                }
                else {
                    if (dse->target != NULL)
                    {
                        cache[num_cache++] = child_ninst;
                        // BLUE_PRTF ("31 Ninst %d(L%d) ready, pushing to rpool\n", child_ninst->ninst_idx, child_ninst->ldata->layer->layer_idx);
                    }
                    else
                    {
                        // BLUE_PRTF ("32 Ninst %d(L%d) ready, pushing to rpool\n", child_ninst->ninst_idx, child_ninst->ldata->layer->layer_idx);
                        dse->target = child_ninst;
                    }
                }
            }
            else
            {
                #ifdef DEBUG
                if (old_state != NINST_COMPLETED)
                {
                    ERROR_PRTF ("Error: update_children_to_cache_but_prioritize_dse_target(), old_state = %d\n", old_state);
                    assert (0);
                }
                #endif
                atomic_store (&child_ninst->state, old_state);
            }
            if (child_ninst->ldata->out_mat == NULL)
                alloc_ldata_out_mat (child_ninst->ldata);
        }
    }
    if (rpool->is_core_exclusive) {
        rpool_push_ninsts_to_allowed_group (rpool, cache, num_cache);
    }
    else {
        rpool_push_ninsts (rpool, cache, num_cache, 0);
    }
}

void update_children_to_cache_but_prioritize_dse_target (rpool_queue_t *cache, ninst_t *ninst, ninst_t **dse_target)
{
    #ifdef DEBUG
    if (cache == NULL || ninst == NULL)
    {
        ERROR_PRTF ("Error: Invalid arguments to update_children_to_cache_but_prioritize_dse_target()\n");
        assert (0);
    }
    if (ninst->state != NINST_COMPLETED)
    {
        ERROR_PRTF ("Error: ninst->state != NINST_STATE_COMPLETED in update_children_to_cache_but_prioritize_dse_target()\n");
        assert (0);
    }
    #endif
    if (ninst->state != NINST_COMPLETED)
        return;
    for (int i = 0; i < ninst->num_child_ninsts; i++)
    {
        ninst_t *child_ninst = ninst->child_ninst_arr[i];
        unsigned int num_parent_ninsts_completed = atomic_fetch_add (&child_ninst->num_parent_ninsts_completed, 1);
        // BLUE_PRTF ("4 Ninst %d(L%d) incrementing child ninst %d(L%d), parent_comp = %d/%d\n", ninst->ninst_idx, ninst->ldata->layer->layer_idx, 
        //     child_ninst->ninst_idx, child_ninst->ldata->layer->layer_idx, num_parent_ninsts_completed, child_ninst->num_parent_ninsts);
        // if (num_parent_ninsts_completed >= child_ninst->num_parent_ninsts)
        //     assert (0);
        if (num_parent_ninsts_completed == child_ninst->num_parent_ninsts - 1)
        {
           // Pseudo-mutex while ninst state change, using NINST_COMPLETED state as lock
            NINST_STATE old_state = atomic_exchange (&child_ninst->state, NINST_COMPLETED);
            if (old_state == NINST_NOT_READY) 
            {
                // BLUE_PRTF ("4 Ninst %d(L%d) ready, pushing to rpool\n", child_ninst->ninst_idx, child_ninst->ldata->layer->layer_idx);
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
                    ERROR_PRTF ("Error: update_children_to_cache_but_prioritize_dse_target(), old_state = %d\n", old_state);
                    assert (0);
                }
                #endif
                atomic_store (&child_ninst->state, old_state);
            }
            if (child_ninst->ldata->out_mat == NULL)
                alloc_ldata_out_mat (child_ninst->ldata);
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
        ERROR_PRTF ( "ERROR: Unsupported layer type %s, at line %d in file %s\n" , layer_type_str[layer->type], 0, " ");
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
        ERROR_PRTF ("Error: Invalid arguments to dse_push_first_layer_to_rpool()\n");
        assert (0);
    }
    #endif    
    nasm_ldata_t *ldata = &nasm->ldata_arr[0];
    alloc_ldata_out_mat (ldata);
    if (input_data != NULL)
        copy_buffer_to_ldata_out_mat (ldata, input_data);
    for (int i = 0; i < ldata->num_ninst; i++)
    {
        ninst_t *ninst = &ldata->ninst_arr_start[i];
        if (ninst->state != NINST_NOT_READY)
        {
            ERROR_PRTF ("Error: ninst->state != NINST_NOT_READY in dse_push_first_layer_to_rpool()\n");
            assert (0);
        }
        ninst->state = NINST_COMPLETED;
        atomic_fetch_add (&ninst->ldata->num_ninst_completed , 1);
        // int num_des = rpool->ref_dses > 0 ? rpool->ref_dses : 1;
        // update_children (rpool, ninst, i/(1 + ldata->num_ninst/num_des));
        update_children (rpool, ninst);
    }
    atomic_fetch_add (&nasm->num_ldata_completed, 1);
}

size_t dse_get_ldata_size (nasm_t *nasm, unsigned int ldata_idx)
{
    nasm_ldata_t *ldata = &nasm->ldata_arr[ldata_idx];
    if (ldata->layer->type == YOLO_LAYER)
    {
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