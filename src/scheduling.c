#include "scheduling.h"

int is_offloaded(ninst_t *ninst)
{
    return atomic_load(&ninst->offloaded);
}

int is_dev_compute(ninst_t *ninst, int device_idx)
{
    return atomic_load(&ninst->dev_to_compute[device_idx]);
}

int is_dev_send_target(ninst_t *ninst, int device_idx)
{
    return atomic_load(&ninst->dev_send_target[device_idx]);
}

int check_all_parents_target_device(ninst_t *ninst, nasm_t* nasm, int device_idx)
{
    for(int i = 0; i < ninst->num_parent_ninsts; i++)
    {
        int parent_idx = ninst->parent_ninst_idx_arr[i];
        if(!atomic_load(&nasm->ninst_arr[parent_idx].dev_to_compute[device_idx]))
        {
            return 0;
        }
    }
    return 1;
}

void ninst_copy_compute_device(ninst_t* target_ninst, ninst_t* ninst)
{
    // ninst --> target_ninst
    for (int i = 0; i < SCHEDULE_MAX_DEVICES; i++) 
    {
        atomic_exchange(&target_ninst->dev_to_compute[i], &ninst->dev_to_compute[i]);
        // target_ninst->dev_to_compute[i] = ninst->dev_to_compute[i];
    }
}

void ninst_clear_compute_device(ninst_t *ninst) 
{
    for (int i = 0; i < SCHEDULE_MAX_DEVICES; i++) 
    {
        atomic_store(&ninst->dev_to_compute[i], 0);
        // ninst->dev_to_compute[i] = 0;
    }
}

void ninst_set_compute_device(ninst_t *ninst, int device_idx) 
{
    atomic_store(&ninst->dev_to_compute[device_idx], 1);
    // ninst->dev_to_compute[device_idx] = 1;
    
}

void ninst_set_send_target_device(ninst_t *ninst, int device_idx)
{
    atomic_store(&ninst->dev_send_target[device_idx], 1);
    // ninst->dev_send_target[device_idx] = 1;
}

void ninst_clear_send_target_device(ninst_t *ninst) 
{
    for (int i = 0; i<SCHEDULE_MAX_DEVICES; i++) 
    {
        atomic_store(&ninst->dev_send_target[i], 0);
        // ninst->dev_send_target[i] = 0;
    }
}

void nasm_set_ninst_send_target_using_child_compute_device(nasm_t *nasm) 
{
    for (int i = 0; i < nasm->num_ninst; i++) 
    {
        ninst_t *ninst = nasm->ninst_arr + i;
        ninst_clear_send_target_device(ninst);
        for (int j = 0; j < ninst->num_child_ninsts; j++) 
        {
            ninst_t *child_ninst = ninst->child_ninst_arr[j];
            for (int dev = 0; dev < SCHEDULE_MAX_DEVICES; dev++) 
            {
                ninst->dev_send_target[dev] |= child_ninst->dev_to_compute[dev];
            }
        }
    }
}

void nasm_all_ninst_set_compute_device (nasm_t *nasm, int device_idx) 
{
    for (int i = 0; i < nasm->num_ninst; i++) 
    {
        ninst_t *ninst = nasm->ninst_arr + i;
        ninst_clear_send_target_device(ninst);
        atomic_store(&ninst->dev_send_target[device_idx], 1);
    }
}

void nasm_set_last_layer_ninst_send_target_device(nasm_t *nasm, int device_idx) 
{
    nasm_ldata_t *last_ldata = &(nasm->ldata_arr[nasm->num_ldata-1]);
    ninst_t *last_ldata_ninst_arr = last_ldata->ninst_arr_start;
    for (int i = 0; i < last_ldata->num_ninst; i++) 
    {
        // last_ldata_ninst_arr[i].dev_send_target[device_idx] = 1;
        atomic_store(&last_ldata_ninst_arr[i].dev_send_target[device_idx], 1);
    }
}

dynamic_scheduler_t* init_dynamic_scheduler(avg_ninst_profile_t **ninst_profile, network_profile_t **network_profile, DEVICE_MODE device_mode, int device_idx, int num_edge_devices)
{
    dynamic_scheduler_t *dynamic_scheduler = calloc(1, sizeof(dynamic_scheduler_t));

    for(int i = 0; i < num_edge_devices; i++)
    {
        if(device_mode == DEV_SERVER || device_idx == i)
        {
            dynamic_scheduler->avg_server_ninst_compute_time[i] = ninst_profile[i]->avg_server_computation_time;
            dynamic_scheduler->avg_edge_ninst_compute_time[i] = ninst_profile[i]->avg_edge_computation_time;
            dynamic_scheduler->avg_bandwidth[i] = network_profile[i]->transmit_rate;
            dynamic_scheduler->rtt[i] = network_profile[i]->rtt;
            // ** TODO : Implement PF scheduler **
            dynamic_scheduler->scheduling_latency[i] = 0.0;
        }
    }
    
    return dynamic_scheduler;
}

float get_edge_offline_latency_to_split_layer(spinn_scheduler_t* spinn_scheduler, nasm_t* nasm, int device_idx, int split_layer)
{
    int num_ninsts = 0;
    float latency_sum = 0.0;
    for(int i = 0; i < split_layer; i++)
        num_ninsts += nasm->ldata_arr[i].num_ninst;
    
    latency_sum = spinn_scheduler->avg_edge_ninst_compute_time[device_idx] * num_ninsts;

    return latency_sum;
}

float get_server_offline_latency_to_split_layer(spinn_scheduler_t* spinn_scheduler, nasm_t* nasm, int device_idx, int split_layer)
{
    int num_ninsts = 0;
    int data_size = 0;
    float latency_sum = 0.0;
    for(int i = split_layer; i < nasm->num_ldata; i++)
    {
        num_ninsts += nasm->ldata_arr[i].num_ninst;
        for(int j = 0; j < nasm->ldata_arr[i].num_ninst; j++)
            data_size += nasm->ldata_arr[i].ninst_arr_start[j].tile_dims[OUT_W] * nasm->ldata_arr[i].ninst_arr_start[j].tile_dims[OUT_H] * sizeof(float);
    }
        
    latency_sum = spinn_scheduler->avg_server_ninst_compute_time[device_idx] * num_ninsts + 
                spinn_scheduler->rtt[device_idx] * num_ninsts + 
                data_size * 8 / spinn_scheduler->avg_bandwidth[device_idx] / 1000000; // Transmission latency;

    return latency_sum;
}

void spinn_offline_profile(spinn_scheduler_t* spinn_scheduler, nasm_t* nasm, int device_idx)
{
    for(int i = 0; i < spinn_scheduler->num_split_candidates[device_idx]; i++)   
    {
        int split_layer = spinn_scheduler->split_candidates[device_idx][i];
        spinn_scheduler->edge_offline_layer_latency[device_idx][i] = get_edge_offline_latency_to_split_layer(spinn_scheduler, nasm, device_idx, split_layer);
        spinn_scheduler->server_offline_layer_latency[device_idx][i] = get_server_offline_latency_to_split_layer(spinn_scheduler, nasm, device_idx, split_layer);
        printf("%d, %f, %f\n", split_layer, spinn_scheduler->edge_offline_layer_latency[device_idx][i], spinn_scheduler->server_offline_layer_latency[device_idx][i]);
    }
}

// void spinn_update_profile(spinn_scheduler_t* spinn_scheduler, avg_ninst_profile_t **ninst_profile, network_profile_t **network_profile)
// {
//     if(ninst_profile != NULL)
//         spinn_scheduler->ninst_profile = ninst_profile;
//     if(network_profile != NULL)
//         spinn_scheduler->network_profile = network_profile;
// }

spinn_scheduler_t* init_spinn_scheduler(avg_ninst_profile_t **ninst_profile, network_profile_t **network_profile, nasm_t** nasms, DEVICE_MODE device_mode, int device_idx, int num_edge_devices)    
{
    spinn_scheduler_t* spinn_scheduler = calloc(1, sizeof(spinn_scheduler_t));
    
    for(int edge_id = 0; edge_id < num_edge_devices; edge_id++)
    {
        if(device_mode == DEV_SERVER || device_idx == edge_id)
        {
            spinn_scheduler->avg_server_ninst_compute_time[edge_id] = ninst_profile[edge_id]->avg_server_computation_time;
            spinn_scheduler->avg_edge_ninst_compute_time[edge_id] = ninst_profile[edge_id]->avg_edge_computation_time;
            spinn_scheduler->avg_bandwidth[edge_id] = network_profile[edge_id]->transmit_rate;
            spinn_scheduler->rtt[edge_id] = network_profile[edge_id]->rtt;

            // Model Splitter : Find Conv, Maxpool, Residual layers and store indices to split_candidates
            spinn_model_splitter(spinn_scheduler, nasms[edge_id], edge_id);
            spinn_offline_profile(spinn_scheduler, nasms[edge_id], edge_id);
        }
    }

    return spinn_scheduler;
}

int spinn_schedule_layer(spinn_scheduler_t* spinn_scheduler, int device_idx)
{
    printf("[SPINN Scheduler]\n");
    int split_layer = 0;
    float min_latency = 100000000.0;
    for(int i = 0; i < spinn_scheduler->num_split_candidates[device_idx]; i++)
    {
        float latency = spinn_scheduler->edge_offline_layer_latency[device_idx][i] + spinn_scheduler->server_offline_layer_latency[device_idx][i];
        if(latency < min_latency)
        {
            min_latency = latency;
            split_layer = spinn_scheduler->split_candidates[device_idx][i];
        }
    }
    spinn_scheduler->current_split_layer = split_layer;

    return split_layer;
}

void spinn_model_splitter(spinn_scheduler_t* spinn_scheduler, nasm_t* nasm, int device_idx)
{
    printf("[SPINN Model Splitter]\n");
    printf("\tObtained split candidates: ");
    int num_split_candidates = 0;
    spinn_scheduler->split_candidates[device_idx] = calloc(nasm->num_ldata, sizeof(int));
    spinn_scheduler->split_candidates[device_idx][num_split_candidates] = 1;
    printf("%d ", spinn_scheduler->split_candidates[device_idx][num_split_candidates]);
    num_split_candidates++;

    for(int i = 2; i < nasm->num_ldata-1; i++)
    {
        if(nasm->ldata_arr[i].layer->type == CONV_LAYER || nasm->ldata_arr[i].layer->type == MAXPOOL_LAYER || nasm->ldata_arr[i].layer->type == RESIDUAL_LAYER)
        {
            spinn_scheduler->split_candidates[device_idx][num_split_candidates] = i;
            printf("%d ", spinn_scheduler->split_candidates[device_idx][num_split_candidates]);
            num_split_candidates++;
        }
    }
    spinn_scheduler->split_candidates[device_idx][num_split_candidates] = nasm->num_ldata-1;
    printf("%d ", spinn_scheduler->split_candidates[device_idx][num_split_candidates]);
    num_split_candidates++;
    spinn_scheduler->server_offline_layer_latency[device_idx] = calloc(num_split_candidates, sizeof(float));
    spinn_scheduler->server_real_latency[device_idx] = calloc(num_split_candidates, sizeof(float));
    spinn_scheduler->edge_offline_layer_latency[device_idx] = calloc(num_split_candidates, sizeof(float));
    spinn_scheduler->edge_real_latency[device_idx] = calloc(num_split_candidates, sizeof(float));
    spinn_scheduler->num_split_candidates[device_idx] = num_split_candidates;
    
    printf("\n");
    printf("\tTotal num split candidates: (%d/%d)\n", num_split_candidates, nasm->num_ldata);


    
}

float get_eft_edge(dynamic_scheduler_t* dynamic_scheduler, rpool_t* rpool, int device_idx, int num_dse, int num_child_ninsts)
{
    unsigned int rpool_num_stored = atomic_load(&rpool->num_stored);
    // printf("%d, %d, %d\n", rpool_num_stored, num_child_ninsts, num_dse);
    float eft_edge = (float)(rpool_num_stored + num_child_ninsts) * dynamic_scheduler->avg_edge_ninst_compute_time[device_idx] / num_dse;
    return eft_edge;
}

float get_eft_server(dynamic_scheduler_t* dynamic_scheduler, networking_engine* net_engine, int device_idx, int net_tx_queue_bytes)
{
    unsigned int net_tx_queue_num_stored = atomic_load(&net_engine->tx_queue->num_stored);
    float eft_edge = dynamic_scheduler->rtt[device_idx] + // RTT
                    (net_tx_queue_num_stored) * net_tx_queue_bytes * 8 / dynamic_scheduler->avg_bandwidth[device_idx] / 1000000 // Transmission latency
                    + dynamic_scheduler->scheduling_latency[device_idx];
    return eft_edge;
}

void init_full_local(nasm_t *nasm) {
    for (int i = 0; i < nasm->num_ninst; i++) 
    {
        ninst_t *ninst = nasm->ninst_arr + i;
        ninst_clear_compute_device(ninst);
        ninst_set_compute_device(ninst, DEV_EDGE);
    }
    nasm_set_ninst_send_target_using_child_compute_device(nasm);
}

void init_full_offload(nasm_t *nasm) {
    for (int i = 0; i < nasm->num_ninst; i++) 
    {
        ninst_t *ninst = nasm->ninst_arr + i;
        ninst_clear_compute_device(ninst);
        if (ninst->ldata->layer->layer_idx != nasm->num_ldata - 1) 
        {
            ninst_set_compute_device(ninst, DEV_SERVER);
        }
        else 
        {
            ninst_set_compute_device(ninst, DEV_EDGE);
        }
    }
    nasm_set_ninst_send_target_using_child_compute_device(nasm);
}

void init_partial_offload(nasm_t *nasm, float compute_ratio) {
    int layer_start_ninst_idx = nasm->ldata_arr[1].ninst_arr_start[0].ninst_idx;
    int layer_end_ninst_idx = layer_start_ninst_idx + nasm->ldata_arr[1].num_ninst;
    
    int division_idx = layer_start_ninst_idx + (1-compute_ratio) * (layer_end_ninst_idx - layer_start_ninst_idx);
    printf("division idx: %d\n", division_idx);
    for (int i = 0; i < nasm->num_ninst; i++) {
        ninst_t *ninst = nasm->ninst_arr + i;
        ninst_clear_compute_device(ninst);
        if (ninst->ldata->layer->layer_idx == 0) {  // for the input data,
            ninst_set_compute_device(ninst, DEV_EDGE);  // all inputs are generated from TX
        }
        else if (ninst->ldata->layer->layer_idx == 1) { // for the first computation layer,
            if (ninst->ninst_idx < division_idx) {  // front ninsts are for RX
                ninst_set_compute_device(ninst, DEV_SERVER);
            }
            else if (ninst->ninst_idx > division_idx) { // behind ninsts are for TX
                ninst_set_compute_device(ninst, DEV_EDGE);
            }
            else {  // division ninst is for the both
                ninst_set_compute_device(ninst, DEV_SERVER);
                ninst_set_compute_device(ninst, DEV_EDGE);
            }
        }
        else if (ninst->ldata->layer->layer_idx != nasm->num_ldata - 1) {   // intermediate layers are for RX
            ninst_set_compute_device(ninst, DEV_SERVER);
        }
        else {  // final layer is for TX -> main.c has its own logic handling final layer
            // ninst->dev_to_compute[0] = DEV_EDGE;
            // ninst_set_compute_device(ninst, DEV_EDGE);
            ninst_set_compute_device(ninst, DEV_SERVER);
        }
    }
    // nasm_set_ninst_send_target_using_child_compute_device(nasm);
    nasm_set_last_layer_ninst_send_target_device(nasm, DEV_EDGE);
}

void init_sequential_offload(nasm_t *nasm, int split_layer, int from_dev, int to_dev) 
{
    printf("[Init sequential offload] division ninst idx: %d from dev: %d to_dev: %d\n", split_layer, from_dev, to_dev);
    for (int i = 0; i < nasm->num_ldata; i++) 
    {
        if (i < split_layer) 
        {
            for (int j = 0; j < nasm->ldata_arr[i].num_ninst; j++) 
            {
                ninst_clear_compute_device(&(nasm->ldata_arr[i].ninst_arr_start[j]));
                ninst_set_compute_device(&(nasm->ldata_arr[i].ninst_arr_start[j]), from_dev);
                ninst_set_send_target_device(&(nasm->ldata_arr[i].ninst_arr_start[j]), to_dev);
            }
        }
        else 
        {
            for (int j = 0; j < nasm->ldata_arr[i].num_ninst; j++) 
            {
                ninst_clear_compute_device(&(nasm->ldata_arr[i].ninst_arr_start[j]));
                ninst_set_compute_device(&(nasm->ldata_arr[i].ninst_arr_start[j]), to_dev);
            }
        }
    }

    nasm_set_ninst_send_target_using_child_compute_device (nasm);
    nasm_set_last_layer_ninst_send_target_device (nasm, from_dev);
}

void init_dynamic_offload(nasm_t *nasm, DEVICE_MODE device_mode, int device_idx, int server_idx) 
{
    for (int i = 0; i < nasm->num_ldata; i++) 
    {
        for (int j = 0; j<nasm->ldata_arr[i].num_ninst; j++) 
        {
            ninst_clear_compute_device(&(nasm->ldata_arr[i].ninst_arr_start[j]));
            ninst_clear_send_target_device(&(nasm->ldata_arr[i].ninst_arr_start[j]));
            atomic_store(&nasm->ldata_arr[i].ninst_arr_start[j].offloaded, 0);
            if(device_mode == DEV_SERVER)
                ninst_set_compute_device(&(nasm->ldata_arr[i].ninst_arr_start[j]), server_idx);
            else if(device_mode == DEV_EDGE)
                ninst_set_compute_device(&(nasm->ldata_arr[i].ninst_arr_start[j]), device_idx);
            else
                assert(0);
        }
    }
    
    for (int i = 0; i < nasm->ldata_arr[0].num_ninst; i++)
    {
        ninst_set_compute_device(&(nasm->ldata_arr[0].ninst_arr_start[i]), device_idx);
        ninst_set_send_target_device(&(nasm->ldata_arr[0].ninst_arr_start[i]), server_idx);
    }

    // Set last layer to edge as default
    nasm_set_last_layer_ninst_send_target_device(nasm, device_idx);
}

void init_conventional_offload(nasm_t *nasm) 
{
    for (int i = 0; i < nasm->num_ldata; i++) 
    {
        for (int j = 0; j<nasm->ldata_arr[i].num_ninst; j++) 
        {
            ninst_clear_compute_device(&(nasm->ldata_arr[i].ninst_arr_start[j]));
            ninst_set_compute_device(&(nasm->ldata_arr[i].ninst_arr_start[j]), DEV_SERVER);
        }
    }
    nasm_all_ninst_set_compute_device(nasm, DEV_SERVER);
    nasm_set_last_layer_ninst_send_target_device(nasm, DEV_EDGE);
}

sched_processor_t *init_heft(char *target_dnn_dir, char *target_nasm_dir, ninst_profile_t **ninst_profile, network_profile_t *network_profile, int num_device) {
    aspen_dnn_t *target_dnn = apu_load_dnn_from_file(target_dnn_dir);
    nasm_t *nasm = apu_load_nasm_from_file (target_nasm_dir, target_dnn);

    int num_ninst = nasm->num_ninst;

    // dependency: dep[i][j] == 1 means i is parent of j, j is child of i
    int **ninst_dependency = calloc(num_ninst, sizeof(float *));
    for (int i=0; i<num_ninst; i++) ninst_dependency[i] = calloc(num_ninst, sizeof(float));

    float **data = calloc(num_ninst, sizeof(float *));
    for (int i=0; i<num_ninst; i++) data[i] = calloc(num_ninst, sizeof(float));

    float **W = calloc(num_ninst, sizeof(float *));
    float *W_avg = calloc(num_ninst, sizeof(float));
    for (int i=0; i<num_ninst; i++) W[i] = calloc(num_device, sizeof(float));

    float **B = calloc(num_device, sizeof(float *));
    float B_avg;
    for (int i=0; i<num_device; i++) B[i] = calloc(num_device, sizeof(float));

    float *L = calloc(num_device, sizeof(float));
    float L_avg;

    float **C_avg = calloc(num_ninst, sizeof(float *));
    for (int i=0; i<num_ninst; i++) C_avg[i] = calloc(num_ninst, sizeof(float));

    float *rank_upward = calloc(nasm->num_ninst, sizeof(float));

    heft_gen_dependency(nasm, ninst_dependency);
    heft_gen_data(nasm, ninst_profile, ninst_dependency, data);
    heft_gen_W(nasm, ninst_profile, num_device, W, W_avg);
    heft_gen_B(nasm, network_profile, num_device, B, &B_avg);
    heft_gen_L(nasm, network_profile, num_device, L, &L_avg);
    heft_gen_C_avg(nasm, L_avg, data, B_avg, ninst_dependency, C_avg);

    gen_rank_upward(nasm, W_avg, C_avg, ninst_dependency, rank_upward);

    ninst_t **queue_by_rank_upward = calloc(nasm->num_ninst, sizeof(ninst_t *));
    for (int i=0; i<nasm->num_ninst; i++) {
        nasm->ninst_arr[i].rank_upward = rank_upward[i];
        queue_by_rank_upward[i] = nasm->ninst_arr + i;
    }

    qsort(queue_by_rank_upward, num_ninst, sizeof(ninst_t *), compare_by_rank_upward);

    sched_processor_t *sched_processor_arr = heft_init_processor(num_device);
    sched_task_t *sched_task_arr = heft_init_task(num_ninst);

    float *EST = calloc(num_device, sizeof(float));
    float *EFT = calloc(num_device, sizeof(float));
    int *alloc_dev = calloc(num_ninst, sizeof(int));    // TODO: use for convenience!

    for (int i=0; i<num_ninst; i++) {
        ninst_t *target_ninst = queue_by_rank_upward[i];
        
        // calculate EST, EFT of target ninst
        if (target_ninst->ldata->layer->layer_idx == 0) {
            // case of entry task
            const unsigned int total_bytes = target_ninst->tile_dims[OUT_W] * target_ninst->tile_dims[OUT_H] * sizeof(float);


            EST[DEV_EDGE] = heft_earliest_idle(&(sched_processor_arr[DEV_EDGE]), 0, W[i][DEV_EDGE]);
            EFT[DEV_EDGE] = EST[DEV_EDGE] + W[i][DEV_EDGE];
            
            float avail_RX = heft_earliest_idle(&(sched_processor_arr[DEV_SERVER]), 0, W[i][DEV_SERVER]);
            EST[DEV_SERVER] = total_bytes / network_profile->transmit_rate > avail_RX ? total_bytes / network_profile->transmit_rate : avail_RX;
            EFT[DEV_SERVER] = EST[DEV_SERVER] + W[i][DEV_SERVER];

            // find best processor
            float min_EFT = FLT_MAX;
            int min_EFT_proc = -1;
            
            for (int proc=0; proc<num_device; proc++) {
                if (EFT[proc] < min_EFT) {
                    min_EFT = EFT[proc];
                    min_EFT_proc = proc;
                }
            }

            if (min_EFT_proc == -1) {
                PRT("init_heft: min_EFT_proc is -1\n");
                assert(0);
            }

            // push task into processor min_EFT_proc
            // record AFT[i] = min_EFT_proc : recorded at sched_task_arr[i]
            /* TODO */
            sched_task_arr[i].processor = &(sched_processor_arr[min_EFT_proc]);
            sched_task_arr[i].start_time = EST[min_EFT_proc];
            sched_task_arr[i].end_time = min_EFT;

            heft_push_task(&(sched_processor_arr[min_EFT_proc]), &(sched_task_arr[i]));

        }
        else {
            // case of normal task
            float min_EFT = FLT_MAX;
            int min_EFT_proc;

            for (int proc=0; proc<num_device; proc++) {
                float max_dependency_time = 0;
                // when can processor get all the dependency data?
                for (int parent=0; parent<num_ninst; parent++) {
                    if (ninst_dependency[parent][i]) {
                        // check data arrival time from a parent
                        sched_task_t *parent_task = &(sched_task_arr[parent]);
                        float dependency_time = sched_task_arr[parent].end_time + L[parent_task->processor->idx] + data[parent_task->processor->idx][proc] / network_profile->transmit_rate;
                        max_dependency_time = max_dependency_time < dependency_time ? dependency_time : max_dependency_time;
                    }
                }

                // when can processor have big enough idle time?

                EST[proc] = heft_earliest_idle(&(sched_processor_arr[proc]), max_dependency_time, W[i][proc]);
                EFT[proc] = EST[proc] + W[i][proc];

                if (min_EFT > EFT[proc]) {
                    min_EFT = EFT[proc];
                    min_EFT_proc = proc;
                }
            }

            // push task into processor min_EFT_proc at time EST[min_EFT_proc]
            // record AFT[i] = min_EFT_proc : recorded at sched_task_arr[i]
            /* TODO */
            sched_task_arr[i].processor = &(sched_processor_arr[min_EFT_proc]);
            sched_task_arr[i].start_time = EST[min_EFT_proc];
            sched_task_arr[i].end_time = min_EFT;

            heft_push_task(&(sched_processor_arr[min_EFT_proc]), &(sched_task_arr[i]));
        }
    }

    return sched_processor_arr;
}

void heft_gen_dependency(nasm_t *nasm, int **dependency) {
    ninst_t *ninst_arr = nasm->ninst_arr;
    int num_ninst = nasm->num_ninst;

    for (int i=0; i<num_ninst; i++) {
        for (int j=0; j<num_ninst; j++) {
            dependency[i][j] = 0;
        }
    }

    for (int i=0; i<num_ninst; i++) {
        ninst_t *target_ninst = ninst_arr + i;
        for (int j=0; j<target_ninst->num_child_ninsts; j++) {
            int child_idx = target_ninst->child_ninst_arr[j]->ninst_idx;
            dependency[i][child_idx] = 1;
        }
    }
}

void heft_gen_data(nasm_t *nasm, ninst_profile_t **ninst_profile, int **dependency, float **data) {
    int num_ninst = nasm->num_ninst;

    for (int i=0; i<num_ninst; i++) {
        for (int j=0; j<num_ninst; j++) {
            if (dependency[i][j]) data[i][j] = ninst_profile[0][i].transmit_size;
            else data[i][j] = 0;
        }
    }
}

void heft_gen_W(nasm_t *nasm, ninst_profile_t **ninst_profile, int num_device, float **W, float *W_avg) {
    for (int i=0; i<nasm->num_ninst; i++) {
        for (int j=0; j<num_device; j++) {
            W[i][j] = ninst_profile[j][i].computation_time;
            W_avg[i] += ninst_profile[j][i].computation_time;
        }
        W_avg[i] /= num_device;
    }
}

void heft_gen_B(nasm_t *nasm, network_profile_t *network_profile, int num_device, float **B, float *B_avg) {
    *B_avg = 0;
    for (int i=0; i<num_device; i++) {
        for (int j=0; j<num_device; j++) {
            if (i != j) {
                B[i][j] = network_profile->transmit_rate;
                *B_avg += network_profile->transmit_rate;
            }
            else {
                B[i][j] = 0;
            }
        }
    }
    *B_avg /= num_device * (num_device-1);
}

void heft_gen_L(nasm_t *nasm, network_profile_t *network_profile, int num_device, float *L, float *L_avg) {
    for(int i=0; i<num_device; i++) L[i] = 0;
    *L_avg = 0;
}

void heft_gen_C_avg(nasm_t *nasm, float L_avg, float **data, float B_avg, int **dependency, float **C_avg) {
    int num_ninst = nasm->num_ninst;
    for (int i=0; i<num_ninst; i++) {
        for (int j=0; j<num_ninst; j++) {
            C_avg[i][j] = dependency[i][j] ? (L_avg + data[i][j] / B_avg) : FLT_MAX;
        }
    }
}

void gen_rank_upward(nasm_t *nasm, float *W_avg, float **C_avg, int **dependency, float *rank_upward) {
    for (int i=0; i<nasm->num_ninst; i++) rank_upward[i] = -1;
    for (int i=0; i<nasm->num_ninst; i++) calc_rank_upward_rec(nasm, W_avg, C_avg, dependency, rank_upward, i);
}

float calc_rank_upward_rec(nasm_t *nasm, float *W_avg, float **C_avg, int **dependency, float *rank_upward, int target_idx) {
    // already calculated
    if (rank_upward[target_idx] != -1) return rank_upward[target_idx];
    

    int num_ninst = nasm->num_ninst;

    nasm_ldata_t *exit_layer = &(nasm->ldata_arr[nasm->num_ldata-1]);
    ninst_t *exit_ninst_arr = exit_layer->ninst_arr_start;
    int num_exit_ninst = exit_layer->num_ninst;

    // check if exit ninst
    for (int i=0; i<num_exit_ninst; i++) {
        if (exit_ninst_arr[i].ninst_idx == target_idx) {
            rank_upward[target_idx] = W_avg[target_idx];
            return rank_upward[target_idx];
        }
    }

    // normal ninst, not calculated
    float max_critical = 0;
    for (int i=0; i<num_ninst; i++) {
        if (dependency[target_idx][i]) {
            float temp_critical = C_avg[target_idx][i] + calc_rank_upward_rec(nasm, W_avg, C_avg, dependency, rank_upward, i);
            max_critical = max_critical < temp_critical ? temp_critical : max_critical;
        }
    }

    rank_upward[target_idx] = W_avg[target_idx] + max_critical;
    return rank_upward[target_idx];
}

void gen_rank_downward(nasm_t *nasm, float *W_avg, float **C_avg, int **dependency, float *rank_downward) {
    for (int i=0; i<nasm->num_ninst; i++) rank_downward[i] = -1;
    for (int i=0; i<nasm->num_ninst; i++) calc_rank_downward_rec(nasm, W_avg, C_avg, dependency, rank_downward, i);
}

float calc_rank_downward_rec(nasm_t *nasm, float *W_avg, float **C_avg, int **dependency, float *rank_downward, int target_idx) {
    // already calculated
    if (rank_downward[target_idx] != -1) return rank_downward[target_idx];

    int num_ninst = nasm->num_ninst;

    nasm_ldata_t *entry_layer = &(nasm->ldata_arr[0]);
    ninst_t *entry_ninst_arr = entry_layer->ninst_arr_start;
    int num_entry_ninst = entry_layer->num_ninst;

    // check if entry ninst
    for (int i=0; i<num_entry_ninst; i++) {
        if (entry_ninst_arr[i].ninst_idx == target_idx) {
            rank_downward[target_idx] = 0;
            return rank_downward[target_idx];
        }
    }

    // normal ninst, not calculated
    float max_critical = 0;
    for (int i=0; i<num_ninst; i++) {
        if (dependency[i][target_idx]) {
            float temp_critical = C_avg[i][target_idx] + W_avg[i] + calc_rank_downward_rec(nasm, W_avg, C_avg, dependency, rank_downward, i);
            max_critical = max_critical < temp_critical ? temp_critical : max_critical;
        }
    }

    rank_downward[target_idx] = max_critical;
    return rank_downward[target_idx];
}

int compare_by_rank_upward(const void *ninst_1, const void *ninst_2) {
    float a = ((ninst_t *)ninst_1)->rank_upward;
    float b = ((ninst_t *)ninst_2)->rank_upward;

    if (a > b) return -1;
    else if (a < b) return 1;
    else return 0;
}

sched_processor_t *heft_init_processor(int num_processor) {
    sched_processor_t *result_processor_arr = calloc(num_processor, sizeof(sched_processor_t));

    for(int i=0; i<num_processor; i++) {
        result_processor_arr[i].idx = i;
        result_processor_arr[i].num_task = 0;
        result_processor_arr[i].task_list = calloc(1, sizeof(sched_task_t));
        result_processor_arr[i].task_list->processor = result_processor_arr + i;
        result_processor_arr[i].task_list->idx = -1;
        result_processor_arr[i].task_list->prev = NULL;
        result_processor_arr[i].task_list->next = NULL;
        result_processor_arr[i].task_list->start_time = 0;
        result_processor_arr[i].task_list->end_time = 0;
    }

    return result_processor_arr;
}

sched_task_t *heft_init_task(int num_ninst) {
    sched_task_t *result_task_arr = calloc(num_ninst, sizeof(sched_task_t));

    for (int i=0; i<num_ninst; i++) {
        result_task_arr[i].idx = i;
        result_task_arr[i].next = NULL;
        result_task_arr[i].prev = NULL;
        result_task_arr[i].processor = -1;
    }

    return result_task_arr;
}

float heft_earliest_idle(sched_processor_t *sched_processor, float min_limit, float duration) {
    sched_task_t *iter_task = sched_processor->task_list;
    while (1) {
        if (iter_task->next == NULL) return iter_task->end_time;

        if (iter_task->end_time < min_limit) {
            iter_task = iter_task->next;
            continue;
        }

        if (iter_task->next->start_time - iter_task->end_time < duration) {
            iter_task = iter_task->next;
            continue;
        }

        return iter_task->end_time;
    }
}

void heft_push_task(sched_processor_t *sched_processor, sched_task_t *sched_task) {
    sched_task_t *iter_task = sched_processor->task_list;
    sched_processor->num_task++;
    while(1) {
        if (iter_task->next == NULL) {
            // end of schedule - just push
            iter_task->next = sched_task;
            sched_task->prev = iter_task;
            sched_task->next = NULL;
            return;
        }
        else if (iter_task->end_time <= sched_task->start_time && sched_task->end_time < iter_task->next->start_time) {
            // found space - push
            iter_task->next->prev = sched_task;
            sched_task->next = iter_task->next;
            iter_task->next = sched_task;
            sched_task->prev = iter_task;
            return;
        }

        iter_task = iter_task->next;
    }
}

void save_schedule(sched_processor_t *sched_processor_arr, int num_device, char *file_path) {
    // file structure: ${num_device}\n${num_task}\n${tasks...\n}
    FILE *fptr = fopen(file_path, "wb");
    fprintf(fptr, "%d\n", num_device);
    
    for (int i=0; i<num_device; i++) {
        fprintf(fptr, "%d\n", sched_processor_arr[i].num_task);

        sched_task_t *iter_task = sched_processor_arr[i].task_list->next;
        for (int j=0; j<sched_processor_arr[i].num_task; j++) {
            fprintf(fptr, "%d\n", iter_task->idx);
            iter_task = iter_task->next;
        }
    }

    fclose (fptr);
}

sched_processor_t *load_schedule(char *file_path) {

}

void share_schedule(sched_processor_t **sched_processor_arr, int num_device, DEVICE_MODE device_mode, int server_sock, int client_sock) {
    
    if (device_mode == DEV_SERVER) {

        for (int i=0; i<num_device; i++) {
            printf("send %dth device schedule\n", i);
            write_n(client_sock, &((*sched_processor_arr)[i].num_task), sizeof(int));

            sched_task_t *iter_task = (*sched_processor_arr)[i].task_list->next;
            for (int j=0; j<(*sched_processor_arr)[i].num_task; j++) {
                write_n(client_sock, &(iter_task->idx), sizeof(int));
                write_n(client_sock, &(iter_task->start_time), sizeof(float));
                write_n(client_sock, &(iter_task->end_time), sizeof(float));
                iter_task = iter_task->next;
            }
        }
    }
    else if (device_mode == DEV_EDGE) {
        *sched_processor_arr = heft_init_processor(num_device);

        for (int i=0; i<num_device; i++) {
            /* TODO: read integer from server, then create and push task into sched_proccessor_arr */
            printf("receive %dth device schedule\n", i);
            sched_processor_t *processor = *sched_processor_arr + i;
            sched_task_t *iter_task = processor->task_list;
            
            read_n(server_sock, &(processor->num_task), sizeof(int));
            for (int j=0; j<processor->num_task; j++) {
                sched_task_t *new_task = calloc(1, sizeof(sched_task_t));
                iter_task->next = new_task;
                new_task->prev = iter_task;
                new_task->next = NULL;
                new_task->processor = i;

                read_n(server_sock, &(new_task->idx), sizeof(int));
                read_n(server_sock, &(new_task->start_time), sizeof(float));
                read_n(server_sock, &(new_task->end_time), sizeof(float));

                iter_task = new_task;
            }
        }
    }
}

void apply_schedule_to_nasm(nasm_t *nasm, sched_processor_t *sched_processor, int num_device, DEVICE_MODE device_mode) {
    ninst_t *ninst_arr = nasm->ninst_arr;
    int num_ninst = nasm->num_ninst;

    for (int dev=0; dev<num_device; dev++) {
        sched_task_t *iter_task = sched_processor[dev].task_list->next;
        for (int i=0; i<sched_processor[dev].num_task; i++) {
            ninst_arr[iter_task->idx].dev_to_compute[dev] = 1;
            iter_task = iter_task->next;
        }
    }

    // last array is always for RX
    nasm_ldata_t *last_layer = &(nasm->ldata_arr[nasm->num_ldata-1]);
    for (int i=0; i<last_layer->num_ninst; i++) {
        last_layer->ninst_arr_start[i].dev_to_compute[DEV_SERVER] = 1;
    }

    nasm_set_ninst_send_target_using_child_compute_device(nasm);
}