#include "scheduling.h"

// int is_offloaded(ninst_t *ninst)
// {
//     return atomic_load(&ninst->offloaded);
// }

int is_dev_compute(ninst_t *ninst, int device_idx)
{
    return atomic_load(&ninst->dev_to_compute[device_idx]);
}

int is_core_compute(ninst_t *ninst, int core_idx) {
    return atomic_load(&ninst->core_allowed[core_idx]);
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
        atomic_store(&(target_ninst->dev_to_compute[i]), ninst->dev_to_compute[i]);
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
                ninst->dev_to_compute[dev] |= child_ninst->dev_to_compute[dev];
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

void ninst_core_allow_all(ninst_t *ninst)
{
    for (int i = 0; i < ninst->num_cores; i++)
        atomic_store(&ninst->core_allowed[i], 1);
}

void ninst_core_disallow_all(ninst_t *ninst)
{
    for (int i = 0; i < ninst->num_cores; i++)
        atomic_store(&ninst->core_allowed[i], 0);
}

void ninst_core_allow(ninst_t *ninst, int core_idx, int allow)
{
    atomic_store(&ninst->core_allowed[core_idx], allow);
}

void ninst_core_allow_rand(ninst_t *ninst, int num_core) {
    ninst->core_allowed[rand() % num_core] = 1;
}

int get_allowed_core_idx(ninst_t *ninst) {
    for (int i = 0; i < ninst->num_cores; i++) {
        if (atomic_load(&ninst->core_allowed[i])) {
            return i;
        }
    }
    return -1;
}

void core_init_random(nasm_t *nasm, int num_core) {
    for (int i = 0; i < nasm->num_ninst; i++) {
        ninst_t *ninst = nasm->ninst_arr + i;
        ninst_core_disallow_all(ninst);
        ninst_core_allow_rand(ninst, num_core);
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
            dynamic_scheduler->edge_num_dse[i] = ninst_profile[i]->edge_num_dse;
            dynamic_scheduler->server_num_dse[i] = ninst_profile[i]->server_num_dse;
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
    
    latency_sum = spinn_scheduler->avg_edge_ninst_compute_time[device_idx] * num_ninsts / spinn_scheduler->edge_num_dse[device_idx];

    return latency_sum;
}

float get_server_offline_latency_from_split_layer(spinn_scheduler_t* spinn_scheduler, nasm_t* nasm, int device_idx, int split_layer)
{
    int num_ninsts = 0;
    float latency_sum = 0.0;
    for(int i = split_layer; i < nasm->num_ldata; i++)
        num_ninsts += nasm->ldata_arr[i].num_ninst;
        
    latency_sum = spinn_scheduler->avg_server_ninst_compute_time[device_idx] * num_ninsts / spinn_scheduler->server_num_dse[device_idx];

    return latency_sum;
}

float get_data_transmission_latency(spinn_scheduler_t* spinn_scheduler, nasm_t* nasm, int device_idx, int split_layer)
{
    int data_size = 0;
    float latency_sum = 0.0;
    for(int j = 0; j < nasm->ldata_arr[split_layer-1].num_ninst; j++)
        data_size += nasm->ldata_arr[split_layer-1].ninst_arr_start[j].tile_dims[OUT_W] * nasm->ldata_arr[split_layer-1].ninst_arr_start[j].tile_dims[OUT_H] * sizeof(float);
    
    data_size = data_size * 8;
    float bit_per_second = spinn_scheduler->avg_bandwidth[device_idx] * 1000000;
    latency_sum = spinn_scheduler->rtt[device_idx] + // RTT
                data_size / bit_per_second; // Transmission latency;
    return latency_sum;
}

void spinn_offline_profile(spinn_scheduler_t* spinn_scheduler, nasm_t* nasm, int device_idx)
{
    for(int i = 0; i < spinn_scheduler->num_split_candidates[device_idx]; i++)   
    {
        int split_layer = spinn_scheduler->split_candidates[device_idx][i];
        spinn_scheduler->edge_offline_layer_latency[device_idx][i] = get_edge_offline_latency_to_split_layer(spinn_scheduler, nasm, device_idx, split_layer);
        spinn_scheduler->server_offline_layer_latency[device_idx][i] = get_server_offline_latency_from_split_layer(spinn_scheduler, nasm, device_idx, split_layer);
        spinn_scheduler->edge_real_latency[device_idx][i] = spinn_scheduler->edge_offline_layer_latency[device_idx][i];
        spinn_scheduler->server_real_latency[device_idx][i] = spinn_scheduler->server_offline_layer_latency[device_idx][i];
        spinn_scheduler->edge_scaling_factors[device_idx] = 1.0;
        spinn_scheduler->server_scaling_factors[device_idx] = 1.0;

        int data_size = 0;
        if (split_layer > 0)
        {
            for(int j = 0; j < nasm->ldata_arr[split_layer-1].num_ninst; j++)
                data_size += nasm->ldata_arr[split_layer-1].ninst_arr_start[j].tile_dims[OUT_W] * nasm->ldata_arr[split_layer-1].ninst_arr_start[j].tile_dims[OUT_H] * sizeof(float);
        }
        spinn_scheduler->data_size_split_candidates[device_idx][i] = data_size;
    }
}

void spinn_update_profile(spinn_scheduler_t* spinn_scheduler, float rtt, float avg_bandwidth, float avg_edge_latency, float avg_server_latency, int device_idx)
{
    int current_split_layer = spinn_scheduler->current_split_layer[device_idx];
    int idx = -1;
    for (int i = 0; i < spinn_scheduler->num_split_candidates[device_idx]; i++)
    {
        if (spinn_scheduler->split_candidates[device_idx][i] == current_split_layer)
        {
            idx = i;
            break;
        }
    }
    if (idx < 0)
    {
        PRTF("Error: Cannot find split layer %d in split candidates\n", current_split_layer);
        exit(1);
    }
    
    spinn_scheduler->rtt[device_idx] = rtt;
    spinn_scheduler->avg_bandwidth[device_idx] = 0.85 * spinn_scheduler->avg_bandwidth[device_idx] + 0.15 * avg_bandwidth;
    if(avg_edge_latency <= 0) avg_edge_latency = spinn_scheduler->edge_offline_layer_latency[device_idx][idx];
    if(avg_server_latency <= 0) avg_server_latency = spinn_scheduler->server_offline_layer_latency[device_idx][idx];
    
    spinn_scheduler->edge_real_latency[device_idx][idx] = avg_edge_latency;
    spinn_scheduler->server_real_latency[device_idx][idx] = avg_server_latency;
    spinn_scheduler->edge_scaling_factors[device_idx] = spinn_scheduler->edge_real_latency[device_idx][idx] / spinn_scheduler->edge_offline_layer_latency[device_idx][idx];
    spinn_scheduler->server_scaling_factors[device_idx] = spinn_scheduler->server_real_latency[device_idx][idx] / spinn_scheduler->server_offline_layer_latency[device_idx][idx];    
}

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
            spinn_scheduler->edge_num_dse[edge_id] = ninst_profile[edge_id]->edge_num_dse;
            spinn_scheduler->server_num_dse[edge_id] = ninst_profile[edge_id]->server_num_dse;
            
            // Model Splitter : Find Conv, Maxpool, Residual layers and store indices to split_candidates
            spinn_model_splitter(spinn_scheduler, nasms[edge_id], edge_id);
            spinn_offline_profile(spinn_scheduler, nasms[edge_id], edge_id);
        }
    }

    return spinn_scheduler;
}

int spinn_schedule_layer(spinn_scheduler_t* spinn_scheduler, nasm_t* nasm, int device_idx)
{
    PRTF("\t[SPINN Scheduler]\n");
    int split_layer = 1;
    float min_latency = 100000000.0;
    for(int i = 0; i < spinn_scheduler->num_split_candidates[device_idx]; i++)
    {
        float latency = spinn_scheduler->edge_offline_layer_latency[device_idx][i] * spinn_scheduler->edge_scaling_factors[device_idx] + 
            spinn_scheduler->server_offline_layer_latency[device_idx][i] * spinn_scheduler->server_scaling_factors[device_idx] +
            get_data_transmission_latency(spinn_scheduler, nasm, device_idx, spinn_scheduler->split_candidates[device_idx][i]);
        
        // PRTF("\tSplit Layer: %d, Edge: %fms, Server: %fms, Transmission: %fms\n",
        //     spinn_scheduler->split_candidates[device_idx][i],
        //     spinn_scheduler->edge_offline_layer_latency[device_idx][i] * spinn_scheduler->edge_scaling_factors[device_idx] * 1000.0,
        //     spinn_scheduler->server_offline_layer_latency[device_idx][i] * spinn_scheduler->server_scaling_factors[device_idx] * 1000.0,
        //     get_data_transmission_latency(spinn_scheduler, nasm, device_idx, spinn_scheduler->split_candidates[device_idx][i]) * 1000.0
            // );
        // PRTF("BW: %f\n", spinn_scheduler->avg_bandwidth[device_idx]);

        if(latency < min_latency)
        {
            // PRTF("\tSplit Layer: %d, latency: %f, min_latency: %f\n", spinn_scheduler->split_candidates[device_idx][i], latency, min_latency);
            min_latency = latency;
            split_layer = spinn_scheduler->split_candidates[device_idx][i];
        }
    }
    spinn_scheduler->current_split_layer[device_idx] = split_layer;

    return split_layer;
}

int spinn_find_idx_by_split_layer(spinn_scheduler_t* spinn_scheduler, int device_idx)
{
    int current_split_layer = spinn_scheduler->current_split_layer[device_idx];
    int idx = -1;
    for (int i = 0; i < spinn_scheduler->num_split_candidates[device_idx]; i++)
    {
        if (spinn_scheduler->split_candidates[device_idx][i] == current_split_layer)
        {
            idx = i;
            break;
        }
    }
    return idx;
}

void spinn_model_splitter(spinn_scheduler_t* spinn_scheduler, nasm_t* nasm, int device_idx)
{
    PRTF("[SPINN Model Splitter]\n");
    PRTF("\tObtained split candidates: ");
    int num_split_candidates = 0;
    spinn_scheduler->split_candidates[device_idx] = calloc(nasm->num_ldata, sizeof(int));
    spinn_scheduler->split_candidates[device_idx][num_split_candidates] = 1;
    PRTF("%d ", spinn_scheduler->split_candidates[device_idx][num_split_candidates]);
    num_split_candidates++;

    for(int i = 2; i < nasm->num_ldata-1; i++)
    {
        if(nasm->ldata_arr[i].layer->type == CONV_LAYER || nasm->ldata_arr[i].layer->type == MAXPOOL_LAYER || nasm->ldata_arr[i].layer->type == RESIDUAL_LAYER)
        {
            spinn_scheduler->split_candidates[device_idx][num_split_candidates] = i;
            PRTF("%d ", spinn_scheduler->split_candidates[device_idx][num_split_candidates]);
            num_split_candidates++;
        }
    }
    spinn_scheduler->split_candidates[device_idx][num_split_candidates] = nasm->num_ldata-1;
    PRTF("%d ", spinn_scheduler->split_candidates[device_idx][num_split_candidates]);
    num_split_candidates++;
    spinn_scheduler->server_offline_layer_latency[device_idx] = calloc(num_split_candidates, sizeof(float));
    spinn_scheduler->server_real_latency[device_idx] = calloc(num_split_candidates, sizeof(float));
    spinn_scheduler->edge_offline_layer_latency[device_idx] = calloc(num_split_candidates, sizeof(float));
    spinn_scheduler->edge_real_latency[device_idx] = calloc(num_split_candidates, sizeof(float));
    // spinn_scheduler->edge_scaling_factors[device_idx] = calloc(num_split_candidates, sizeof(float));
    // spinn_scheduler->server_scaling_factors[device_idx] = calloc(num_split_candidates, sizeof(float));
    spinn_scheduler->data_size_split_candidates[device_idx] = calloc(num_split_candidates, sizeof(int));
    spinn_scheduler->num_split_candidates[device_idx] = num_split_candidates;
    
    
    PRTF("\n");
    // PRTF("\tTotal num split candidates: (%d/%d)\n", num_split_candidates, nasm->num_ldata);
}

float get_eft_edge(dynamic_scheduler_t* dynamic_scheduler, rpool_t* rpool, int device_idx, int num_dse, int num_child_ninsts)
{
    unsigned int rpool_num_stored = atomic_load(&rpool->num_stored);
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

void init_full_local(nasm_t *nasm, int dev_idx) {
    for (int i = 0; i < nasm->num_ninst; i++) 
    {
        ninst_t *ninst = nasm->ninst_arr + i;
        ninst_clear_compute_device(ninst);
        ninst_set_compute_device(ninst, dev_idx);
    }
    nasm_set_ninst_send_target_using_child_compute_device(nasm);
}

void init_full_offload(nasm_t *nasm, int edge_id, int server_id) {
    for (int i = 0; i < nasm->num_ninst; i++) 
    {
        ninst_t *ninst = nasm->ninst_arr + i;
        ninst_clear_compute_device(ninst);
        if (ninst->ldata->layer->layer_idx != nasm->num_ldata - 1) 
        {
            ninst_set_compute_device(ninst, server_id);
        }
        else 
        {
            ninst_set_compute_device(ninst, edge_id);
        }
    }
    nasm_set_ninst_send_target_using_child_compute_device(nasm);
}

void init_random_offload(nasm_t *nasm, float compute_ratio, int edge_id, int server_id)
{
    srand(time(NULL));
    int total_num_ninst = nasm->num_ninst - nasm->ldata_arr[nasm->num_ldata-1].num_ninst;
    int num_selected = (int)(compute_ratio * total_num_ninst);

    if(num_selected > total_num_ninst)
    {
        PRTF("Error: num_selected > total_num_ninst\n");
        exit(1);
    }

    // PRTF("\t[Random Offload] Selected ninsts: ");

    int selected_ninst_idx[num_selected];
    for(int i = 0; i < num_selected; i++)
    {
        selected_ninst_idx[i] = rand() % total_num_ninst;
        // PRTF("%d ", selected_ninst_idx[i]);
    }
    // PRTF("\n");

    
    for(int i = 0; i < total_num_ninst; i++)
    {
        ninst_t* ninst = nasm->ninst_arr + i;
        ninst_clear_compute_device(ninst);
        ninst_clear_send_target_device(ninst);
        if (ninst->ldata->layer->layer_idx == 0) {  // for the input data,
            ninst_set_compute_device(ninst, edge_id);  // all inputs are generated from TX
            ninst_set_send_target_device(ninst, server_id);
        }
        else
        {
            ninst_set_compute_device(ninst, server_id);
            ninst_set_compute_device(ninst, edge_id);
            for(int count = 0; count < num_selected; count++)
            {
                if(ninst->ninst_idx == selected_ninst_idx[count])
                {
                    ninst_set_send_target_device(ninst, server_id);
                    break;
                }
            }
        }
    }

    // for(int i = 0; i < total_num_ninst; i++)
    // {
    //     ninst_t* ninst = nasm->ninst_arr + i;
    //     PRTF("%d: %d, %d\n", ninst->ninst_idx, ninst->dev_to_compute[server_id], ninst->dev_to_compute[edge_id]);
    // }

    // nasm_set_ninst_send_target_using_child_compute_device(nasm);
    nasm_set_last_layer_ninst_send_target_device(nasm, edge_id);

    // for(int i = 0; i < total_num_ninst; i++)
    // {
    //     ninst_t* ninst = nasm->ninst_arr + i;
    //     PRTF("%d: %d, %d\n", ninst->ninst_idx, ninst->dev_send_target[server_id], ninst->dev_send_target[edge_id]);
    // }
}

void init_partial_offload(nasm_t *nasm, int split_layer, float compute_ratio, int edge_id, int server_id) {
    int layer_start_ninst_idx = nasm->ldata_arr[split_layer].ninst_arr_start[0].ninst_idx;
    int layer_end_ninst_idx = layer_start_ninst_idx + nasm->ldata_arr[split_layer].num_ninst;
    
    int division_idx = layer_start_ninst_idx + (1-compute_ratio) * (layer_end_ninst_idx - layer_start_ninst_idx);
    PRTF("division idx: %d\n", division_idx);
    for (int i = 0; i < nasm->num_ninst; i++) {
        ninst_t *ninst = nasm->ninst_arr + i;
        ninst_clear_compute_device(ninst);
        if (ninst->ldata->layer->layer_idx == 0) {  // for the input data,
            ninst_set_compute_device(ninst, edge_id);  // all inputs are generated from TX
        }
        else if (ninst->ldata->layer->layer_idx < split_layer) { 
            ninst_set_compute_device(ninst, edge_id);
        }
        else if (ninst->ldata->layer->layer_idx == split_layer) {
            if (ninst->ninst_idx < division_idx) {  // front ninsts are for RX
                ninst_set_compute_device(ninst, server_id);
            }
            else if (ninst->ninst_idx > division_idx) { // behind ninsts are for TX
                ninst_set_compute_device(ninst, edge_id);
            }
            else {  // division ninst is for the both
                ninst_set_compute_device(ninst, server_id);
                ninst_set_compute_device(ninst, edge_id);
            }
        }
        else
        {
            ninst_set_compute_device(ninst, server_id);
        }
    }
    nasm_set_ninst_send_target_using_child_compute_device(nasm);
    nasm_set_last_layer_ninst_send_target_device(nasm, edge_id);
}

void init_sequential_offload(nasm_t *nasm, int split_layer, int edge_id, int server_id) 
{
    PRTF("[Init sequential offload] division ninst idx: %d from dev: %d server_id: %d\n", split_layer, edge_id, server_id);
    for (int i = 0; i < nasm->num_ldata; i++) 
    {
        if (i < split_layer) 
        {
            for (int j = 0; j < nasm->ldata_arr[i].num_ninst; j++) 
            {
                ninst_clear_compute_device(&(nasm->ldata_arr[i].ninst_arr_start[j]));
                ninst_set_compute_device(&(nasm->ldata_arr[i].ninst_arr_start[j]), edge_id);
                ninst_set_send_target_device(&(nasm->ldata_arr[i].ninst_arr_start[j]), server_id);
            }
        }
        else 
        {
            for (int j = 0; j < nasm->ldata_arr[i].num_ninst; j++) 
            {
                ninst_clear_compute_device(&(nasm->ldata_arr[i].ninst_arr_start[j]));
                ninst_set_compute_device(&(nasm->ldata_arr[i].ninst_arr_start[j]), server_id);
            }
        }
    }

    nasm_set_ninst_send_target_using_child_compute_device (nasm);
    nasm_set_last_layer_ninst_send_target_device (nasm, edge_id);
}

void init_dynamic_offload(nasm_t *nasm, DEVICE_MODE device_mode, int edge_id, int server_id) 
{
    for (int i = 0; i < nasm->num_ldata; i++) 
    {
        for (int j = 0; j < nasm->ldata_arr[i].num_ninst; j++) 
        {
            ninst_clear_compute_device(&(nasm->ldata_arr[i].ninst_arr_start[j]));
            ninst_clear_send_target_device(&(nasm->ldata_arr[i].ninst_arr_start[j]));
            
            if(device_mode == DEV_SERVER)
                ninst_set_compute_device(&(nasm->ldata_arr[i].ninst_arr_start[j]), server_id);
            else if(device_mode == DEV_EDGE)
                ninst_set_compute_device(&(nasm->ldata_arr[i].ninst_arr_start[j]), edge_id);
            else
                assert(0);
        }
    }
    
    for (int i = 0; i < nasm->ldata_arr[0].num_ninst; i++)
    {
        ninst_set_compute_device(&(nasm->ldata_arr[0].ninst_arr_start[i]), edge_id);
        ninst_set_send_target_device(&(nasm->ldata_arr[0].ninst_arr_start[i]), server_id);
    }

    // Set last layer to edge as default
    nasm_set_last_layer_ninst_send_target_device(nasm, edge_id);
}

void init_conventional_offload(nasm_t *nasm, int edge_id, int server_id)  
{
    for (int i = 0; i < nasm->num_ldata; i++) 
    {
        for (int j = 0; j<nasm->ldata_arr[i].num_ninst; j++) 
        {
            ninst_clear_compute_device(&(nasm->ldata_arr[i].ninst_arr_start[j]));
            ninst_set_compute_device(&(nasm->ldata_arr[i].ninst_arr_start[j]), server_id);
        }
    }
    nasm_all_ninst_set_compute_device(nasm, server_id);
    nasm_set_last_layer_ninst_send_target_device(nasm, edge_id);
}

// void save_schedule(sched_processor_t *sched_processor_arr, int num_device, char *file_path) {
//     // file structure: ${num_device}\n${num_task}\n${tasks...\n}
//     FILE *fptr = fopen(file_path, "wb");
//     fPRTF(fptr, "%d\n", num_device);
    
//     for (int i=0; i<num_device; i++) {
//         fPRTF(fptr, "%d\n", sched_processor_arr[i].num_task);

//         sched_task_t *iter_task = sched_processor_arr[i].task_list->next;
//         for (int j=0; j<sched_processor_arr[i].num_task; j++) {
//             fPRTF(fptr, "%d\n", iter_task->idx);
//             iter_task = iter_task->next;
//         }
//     }

//     fclose (fptr);
// }

// sched_processor_t *load_schedule(char *file_path) 
// {
//     return NULL;
// }

// void share_schedule(sched_processor_t **sched_processor_arr, int num_device, DEVICE_MODE device_mode, int server_sock, int client_sock) {
    
//     if (device_mode == DEV_SERVER) {

//         for (int i=0; i<num_device; i++) {
//             PRTF("send %dth device schedule\n", i);
//             write_n(client_sock, &((*sched_processor_arr)[i].num_task), sizeof(int));

//             sched_task_t *iter_task = (*sched_processor_arr)[i].task_list->next;
//             for (int j=0; j<(*sched_processor_arr)[i].num_task; j++) {
//                 write_n(client_sock, &(iter_task->idx), sizeof(int));
//                 write_n(client_sock, &(iter_task->start_time), sizeof(float));
//                 write_n(client_sock, &(iter_task->end_time), sizeof(float));
//                 iter_task = iter_task->next;
//             }
//         }
//     }
//     else if (device_mode == DEV_EDGE) {
//         *sched_processor_arr = heft_init_processor(num_device);

//         for (int i=0; i<num_device; i++) {
//             /* TODO: read integer from server, then create and push task into sched_proccessor_arr */
//             PRTF("receive %dth device schedule\n", i);
//             sched_processor_t *processor = *sched_processor_arr + i;
//             sched_task_t *iter_task = processor->task_list;
            
//             read_n(server_sock, &(processor->num_task), sizeof(int));
//             for (int j=0; j<processor->num_task; j++) {
//                 sched_task_t *new_task = calloc(1, sizeof(sched_task_t));
//                 iter_task->next = new_task;
//                 new_task->prev = iter_task;
//                 new_task->next = NULL;
//                 new_task->processor = i;

//                 read_n(server_sock, &(new_task->idx), sizeof(int));
//                 read_n(server_sock, &(new_task->start_time), sizeof(float));
//                 read_n(server_sock, &(new_task->end_time), sizeof(float));

//                 iter_task = new_task;
//             }
//         }
//     }
// }

// void apply_schedule_to_nasm(nasm_t *nasm, sched_processor_t *sched_processor, int num_device, DEVICE_MODE device_mode) {
//     ninst_t *ninst_arr = nasm->ninst_arr;
//     int num_ninst = nasm->num_ninst;

//     for (int dev=0; dev<num_device; dev++) {
//         sched_task_t *iter_task = sched_processor[dev].task_list->next;
//         for (int i=0; i<sched_processor[dev].num_task; i++) {
//             ninst_arr[iter_task->idx].dev_to_compute[dev] = 1;
//             iter_task = iter_task->next;
//         }
//     }

//     // last array is always for RX
//     nasm_ldata_t *last_layer = &(nasm->ldata_arr[nasm->num_ldata-1]);
//     for (int i=0; i<last_layer->num_ninst; i++) {
//         last_layer->ninst_arr_start[i].dev_to_compute[DEV_SERVER] = 1;
//     }

//     nasm_set_ninst_send_target_using_child_compute_device(nasm);
// }

// sched_processor_t *init_heft(char *target_dnn_dir, char *target_nasm_dir, ninst_profile_t **ninst_profile, network_profile_t *network_profile, int num_device) {
//     aspen_dnn_t *target_dnn = apu_load_dnn_from_file(target_dnn_dir);
//     nasm_t *nasm = apu_load_nasm_from_file (target_nasm_dir, target_dnn);

//     int num_ninst = nasm->num_ninst;

//     // dependency: dep[i][j] == 1 means i is parent of j, j is child of i
//     int **ninst_dependency = calloc(num_ninst, sizeof(float *));
//     for (int i=0; i<num_ninst; i++) ninst_dependency[i] = calloc(num_ninst, sizeof(float));

//     float **data = calloc(num_ninst, sizeof(float *));
//     for (int i=0; i<num_ninst; i++) data[i] = calloc(num_ninst, sizeof(float));

//     float **W = calloc(num_ninst, sizeof(float *));
//     float *W_avg = calloc(num_ninst, sizeof(float));
//     for (int i=0; i<num_ninst; i++) W[i] = calloc(num_device, sizeof(float));

//     float **B = calloc(num_device, sizeof(float *));
//     float B_avg;
//     for (int i=0; i<num_device; i++) B[i] = calloc(num_device, sizeof(float));

//     float *L = calloc(num_device, sizeof(float));
//     float L_avg;

//     float **C_avg = calloc(num_ninst, sizeof(float *));
//     for (int i=0; i<num_ninst; i++) C_avg[i] = calloc(num_ninst, sizeof(float));

//     float *rank_upward = calloc(nasm->num_ninst, sizeof(float));

//     heft_gen_dependency(nasm, ninst_dependency);
//     heft_gen_data(nasm, ninst_profile, ninst_dependency, data);
//     heft_gen_W(nasm, ninst_profile, num_device, W, W_avg);
//     heft_gen_B(nasm, network_profile, num_device, B, &B_avg);
//     heft_gen_L(nasm, network_profile, num_device, L, &L_avg);
//     heft_gen_C_avg(nasm, L_avg, data, B_avg, ninst_dependency, C_avg);

//     gen_rank_upward(nasm, W_avg, C_avg, ninst_dependency, rank_upward);

//     ninst_t **queue_by_rank_upward = calloc(nasm->num_ninst, sizeof(ninst_t *));
//     for (int i=0; i<nasm->num_ninst; i++) {
//         nasm->ninst_arr[i].rank_upward = rank_upward[i];
//         queue_by_rank_upward[i] = nasm->ninst_arr + i;
//     }

//     qsort(queue_by_rank_upward, num_ninst, sizeof(ninst_t *), compare_by_rank_upward);

//     sched_processor_t *sched_processor_arr = heft_init_processor(num_device);
//     sched_task_t *sched_task_arr = heft_init_task(num_ninst);

//     float *EST = calloc(num_device, sizeof(float));
//     float *EFT = calloc(num_device, sizeof(float));
//     // int *alloc_dev = calloc(num_ninst, sizeof(int));    // TODO: use for convenience!

//     for (int i=0; i<num_ninst; i++) {
//         ninst_t *target_ninst = queue_by_rank_upward[i];
        
//         // calculate EST, EFT of target ninst
//         if (target_ninst->ldata->layer->layer_idx == 0) {
//             // case of entry task
//             const unsigned int total_bytes = target_ninst->tile_dims[OUT_W] * target_ninst->tile_dims[OUT_H] * sizeof(float);


//             EST[DEV_EDGE] = heft_earliest_idle(&(sched_processor_arr[DEV_EDGE]), 0, W[i][DEV_EDGE]);
//             EFT[DEV_EDGE] = EST[DEV_EDGE] + W[i][DEV_EDGE];
            
//             float avail_RX = heft_earliest_idle(&(sched_processor_arr[DEV_SERVER]), 0, W[i][DEV_SERVER]);
//             EST[DEV_SERVER] = total_bytes / network_profile->transmit_rate > avail_RX ? total_bytes / network_profile->transmit_rate : avail_RX;
//             EFT[DEV_SERVER] = EST[DEV_SERVER] + W[i][DEV_SERVER];

//             // find best processor
//             float min_EFT = FLT_MAX;
//             int min_EFT_proc = -1;
            
//             for (int proc=0; proc<num_device; proc++) {
//                 if (EFT[proc] < min_EFT) {
//                     min_EFT = EFT[proc];
//                     min_EFT_proc = proc;
//                 }
//             }

//             if (min_EFT_proc < 0) {
//                 ERROR_PRTF ( "ERROR: init_heft - min_EFT_proc < 0\n");
//                 assert(0);
//             }

//             // push task into processor min_EFT_proc
//             // record AFT[i] = min_EFT_proc : recorded at sched_task_arr[i]
//             /* TODO */
//             sched_task_arr[i].processor = &(sched_processor_arr[min_EFT_proc]);
//             sched_task_arr[i].start_time = EST[min_EFT_proc];
//             sched_task_arr[i].end_time = min_EFT;

//             heft_push_task(&(sched_processor_arr[min_EFT_proc]), &(sched_task_arr[i]));

//         }
//         else {
//             // case of normal task
//             float min_EFT = FLT_MAX;
//             int min_EFT_proc = -1;

//             for (int proc=0; proc<num_device; proc++) {
//                 float max_dependency_time = 0;
//                 // when can processor get all the dependency data?
//                 for (int parent=0; parent<num_ninst; parent++) {
//                     if (ninst_dependency[parent][i]) {
//                         // check data arrival time from a parent
//                         sched_task_t *parent_task = &(sched_task_arr[parent]);
//                         float dependency_time = sched_task_arr[parent].end_time + L[parent_task->processor->idx] + data[parent_task->processor->idx][proc] / network_profile->transmit_rate;
//                         max_dependency_time = max_dependency_time < dependency_time ? dependency_time : max_dependency_time;
//                     }
//                 }

//                 // when can processor have big enough idle time?

//                 EST[proc] = heft_earliest_idle(&(sched_processor_arr[proc]), max_dependency_time, W[i][proc]);
//                 EFT[proc] = EST[proc] + W[i][proc];

//                 if (min_EFT > EFT[proc]) {
//                     min_EFT = EFT[proc];
//                     min_EFT_proc = proc;
//                 }
//             }
//             if (min_EFT_proc < 0)
//             {
//                 ERROR_PRTF ( "ERROR: init_heft - min_EFT_proc < 0\n");
//                 assert(0);
//             }
//             // push task into processor min_EFT_proc at time EST[min_EFT_proc]
//             // record AFT[i] = min_EFT_proc : recorded at sched_task_arr[i]
//             /* TODO */
//             sched_task_arr[i].processor = &(sched_processor_arr[min_EFT_proc]);
//             sched_task_arr[i].start_time = EST[min_EFT_proc];
//             sched_task_arr[i].end_time = min_EFT;

//             heft_push_task(&(sched_processor_arr[min_EFT_proc]), &(sched_task_arr[i]));
//         }
//     }

//     return sched_processor_arr;
// }

// void heft_gen_dependency(nasm_t *nasm, int **dependency) {
//     ninst_t *ninst_arr = nasm->ninst_arr;
//     int num_ninst = nasm->num_ninst;

//     for (int i=0; i<num_ninst; i++) {
//         for (int j=0; j<num_ninst; j++) {
//             dependency[i][j] = 0;
//         }
//     }

//     for (int i=0; i<num_ninst; i++) {
//         ninst_t *target_ninst = ninst_arr + i;
//         for (int j=0; j<target_ninst->num_child_ninsts; j++) {
//             int child_idx = target_ninst->child_ninst_arr[j]->ninst_idx;
//             dependency[i][child_idx] = 1;
//         }
//     }
// }

// void heft_gen_data(nasm_t *nasm, ninst_profile_t **ninst_profile, int **dependency, float **data) {
//     int num_ninst = nasm->num_ninst;

//     for (int i=0; i<num_ninst; i++) {
//         for (int j=0; j<num_ninst; j++) {
//             if (dependency[i][j]) data[i][j] = ninst_profile[0][i].transmit_size;
//             else data[i][j] = 0;
//         }
//     }
// }

// void heft_gen_W(nasm_t *nasm, ninst_profile_t **ninst_profile, int num_device, float **W, float *W_avg) {
//     for (int i=0; i<nasm->num_ninst; i++) {
//         for (int j=0; j<num_device; j++) {
//             W[i][j] = ninst_profile[j][i].computation_time;
//             W_avg[i] += ninst_profile[j][i].computation_time;
//         }
//         W_avg[i] /= num_device;
//     }
// }

// void heft_gen_B(nasm_t *nasm, network_profile_t *network_profile, int num_device, float **B, float *B_avg) {
//     *B_avg = 0;
//     for (int i=0; i<num_device; i++) {
//         for (int j=0; j<num_device; j++) {
//             if (i != j) {
//                 B[i][j] = network_profile->transmit_rate;
//                 *B_avg += network_profile->transmit_rate;
//             }
//             else {
//                 B[i][j] = 0;
//             }
//         }
//     }
//     *B_avg /= num_device * (num_device-1);
// }

// void heft_gen_L(nasm_t *nasm, network_profile_t *network_profile, int num_device, float *L, float *L_avg) {
//     for(int i=0; i<num_device; i++) L[i] = 0;
//     *L_avg = 0;
// }

// void heft_gen_C_avg(nasm_t *nasm, float L_avg, float **data, float B_avg, int **dependency, float **C_avg) {
//     int num_ninst = nasm->num_ninst;
//     for (int i=0; i<num_ninst; i++) {
//         for (int j=0; j<num_ninst; j++) {
//             C_avg[i][j] = dependency[i][j] ? (L_avg + data[i][j] / B_avg) : FLT_MAX;
//         }
//     }
// }

// void gen_rank_upward(nasm_t *nasm, float *W_avg, float **C_avg, int **dependency, float *rank_upward) {
//     for (int i=0; i<nasm->num_ninst; i++) rank_upward[i] = -1;
//     for (int i=0; i<nasm->num_ninst; i++) calc_rank_upward_rec(nasm, W_avg, C_avg, dependency, rank_upward, i);
// }

// float calc_rank_upward_rec(nasm_t *nasm, float *W_avg, float **C_avg, int **dependency, float *rank_upward, int target_idx) {
//     // already calculated
//     if (rank_upward[target_idx] != -1) return rank_upward[target_idx];
    

//     int num_ninst = nasm->num_ninst;

//     nasm_ldata_t *exit_layer = &(nasm->ldata_arr[nasm->num_ldata-1]);
//     ninst_t *exit_ninst_arr = exit_layer->ninst_arr_start;
//     int num_exit_ninst = exit_layer->num_ninst;

//     // check if exit ninst
//     for (int i=0; i<num_exit_ninst; i++) {
//         if (exit_ninst_arr[i].ninst_idx == target_idx) {
//             rank_upward[target_idx] = W_avg[target_idx];
//             return rank_upward[target_idx];
//         }
//     }

//     // normal ninst, not calculated
//     float max_critical = 0;
//     for (int i=0; i<num_ninst; i++) {
//         if (dependency[target_idx][i]) {
//             float temp_critical = C_avg[target_idx][i] + calc_rank_upward_rec(nasm, W_avg, C_avg, dependency, rank_upward, i);
//             max_critical = max_critical < temp_critical ? temp_critical : max_critical;
//         }
//     }

//     rank_upward[target_idx] = W_avg[target_idx] + max_critical;
//     return rank_upward[target_idx];
// }

// void gen_rank_downward(nasm_t *nasm, float *W_avg, float **C_avg, int **dependency, float *rank_downward) {
//     for (int i=0; i<nasm->num_ninst; i++) rank_downward[i] = -1;
//     for (int i=0; i<nasm->num_ninst; i++) calc_rank_downward_rec(nasm, W_avg, C_avg, dependency, rank_downward, i);
// }

// float calc_rank_downward_rec(nasm_t *nasm, float *W_avg, float **C_avg, int **dependency, float *rank_downward, int target_idx) {
//     // already calculated
//     if (rank_downward[target_idx] != -1) return rank_downward[target_idx];

//     int num_ninst = nasm->num_ninst;

//     nasm_ldata_t *entry_layer = &(nasm->ldata_arr[0]);
//     ninst_t *entry_ninst_arr = entry_layer->ninst_arr_start;
//     int num_entry_ninst = entry_layer->num_ninst;

//     // check if entry ninst
//     for (int i=0; i<num_entry_ninst; i++) {
//         if (entry_ninst_arr[i].ninst_idx == target_idx) {
//             rank_downward[target_idx] = 0;
//             return rank_downward[target_idx];
//         }
//     }

//     // normal ninst, not calculated
//     float max_critical = 0;
//     for (int i=0; i<num_ninst; i++) {
//         if (dependency[i][target_idx]) {
//             float temp_critical = C_avg[i][target_idx] + W_avg[i] + calc_rank_downward_rec(nasm, W_avg, C_avg, dependency, rank_downward, i);
//             max_critical = max_critical < temp_critical ? temp_critical : max_critical;
//         }
//     }

//     rank_downward[target_idx] = max_critical;
//     return rank_downward[target_idx];
// }

// int compare_by_rank_upward(const void *ninst_1, const void *ninst_2) {
//     float a = ((ninst_t *)ninst_1)->rank_upward;
//     float b = ((ninst_t *)ninst_2)->rank_upward;

//     if (a > b) return -1;
//     else if (a < b) return 1;
//     else return 0;
// }

// sched_processor_t *heft_init_processor(int num_processor) {
//     sched_processor_t *result_processor_arr = calloc(num_processor, sizeof(sched_processor_t));

//     for(int i=0; i<num_processor; i++) {
//         result_processor_arr[i].idx = i;
//         result_processor_arr[i].num_task = 0;
//         result_processor_arr[i].task_list = calloc(1, sizeof(sched_task_t));
//         result_processor_arr[i].task_list->processor = result_processor_arr + i;
//         result_processor_arr[i].task_list->idx = -1;
//         result_processor_arr[i].task_list->prev = NULL;
//         result_processor_arr[i].task_list->next = NULL;
//         result_processor_arr[i].task_list->start_time = 0;
//         result_processor_arr[i].task_list->end_time = 0;
//     }

//     return result_processor_arr;
// }

// sched_task_t *heft_init_task(int num_ninst) {
//     sched_task_t *result_task_arr = calloc(num_ninst, sizeof(sched_task_t));

//     for (int i=0; i<num_ninst; i++) {
//         result_task_arr[i].idx = i;
//         result_task_arr[i].next = NULL;
//         result_task_arr[i].prev = NULL;
//         result_task_arr[i].processor = NULL;
//     }

//     return result_task_arr;
// }

// float heft_earliest_idle(sched_processor_t *sched_processor, float min_limit, float duration) {
//     sched_task_t *iter_task = sched_processor->task_list;
//     while (1) {
//         if (iter_task->next == NULL) return iter_task->end_time;

//         if (iter_task->end_time < min_limit) {
//             iter_task = iter_task->next;
//             continue;
//         }

//         if (iter_task->next->start_time - iter_task->end_time < duration) {
//             iter_task = iter_task->next;
//             continue;
//         }

//         return iter_task->end_time;
//     }
// }

// void heft_push_task(sched_processor_t *sched_processor, sched_task_t *sched_task) {
//     sched_task_t *iter_task = sched_processor->task_list;
//     sched_processor->num_task++;
//     while(1) {
//         if (iter_task->next == NULL) {
//             // end of schedule - just push
//             iter_task->next = sched_task;
//             sched_task->prev = iter_task;
//             sched_task->next = NULL;
//             return;
//         }
//         else if (iter_task->end_time <= sched_task->start_time && sched_task->end_time < iter_task->next->start_time) {
//             // found space - push
//             iter_task->next->prev = sched_task;
//             sched_task->next = iter_task->next;
//             iter_task->next = sched_task;
//             sched_task->prev = iter_task;
//             return;
//         }

//         iter_task = iter_task->next;
//     }
// }

static unsigned int ninst_find_parents(ninst_t **buffer, unsigned int num_buffer, ninst_t **parent_buffer) {
    nasm_t *nasm = buffer[0]->ldata->nasm;

    unsigned int num_parents = 0;
    for (int i=0; i<num_buffer; i++) {
        ninst_t *ninst_now = buffer[i];

        for (int p_idx=0; p_idx<ninst_now->num_parent_ninsts; p_idx++) {
            ninst_t *parent_now = &nasm->ninst_arr[ninst_now->parent_ninst_idx_arr[p_idx]];

            // check if parent is already in parent_buffer
            int already_exists = 0;
            for (int pb_idx=0; pb_idx<num_parents; pb_idx++) {
                if (parent_buffer[pb_idx] == parent_now) {
                    already_exists = 1;
                    break;
                }
            }

            if (!already_exists) 
                parent_buffer[num_parents++] = parent_now;
        }
    }

    return num_parents;
}

void fl_init(nasm_t *nasm) {
    nasm->operating_mode = OPER_MODE_FL_PATH;
    nasm->num_paths = 0;
    nasm->path_now_idx = 0;
}

fl_path_t *fl_create_path(nasm_t *nasm, ninst_t **last_layer_ninsts, unsigned int num_last_layer_ninsts) {
    fl_path_t *path = (fl_path_t *)malloc(sizeof(fl_path_t));

    nasm->path_ptr_arr[nasm->num_paths] = path;
    path->path_idx = nasm->num_paths++;
    path->num_path_layers = last_layer_ninsts[0]->ldata->layer->layer_idx;
    atomic_store(&path->num_path_layers_completed, 0);
    path->path_layers_arr = (fl_path_layer_t *)malloc(sizeof(fl_path_layer_t) * path->num_path_layers);
    path->edge_final_layer_idx = path->num_path_layers;

    unsigned int num_buffer_parent;
    unsigned int num_buffer;
    
    ninst_t *buffer_parent[256];
    ninst_t *buffer[256];

    num_buffer = num_last_layer_ninsts;
    for (int i=0; i<num_buffer; i++) buffer[i] = last_layer_ninsts[i];

    for (int l=path->num_path_layers-1; l>=0; l--) {
        fl_path_layer_t *path_layer_now = &path->path_layers_arr[l];

        path_layer_now->fl_path = path;
        path_layer_now->ldata = &nasm->ldata_arr[l+1];

        path_layer_now->num_ninsts = num_buffer;
        atomic_store(&path_layer_now->num_ninsts_completed, 0);
        path_layer_now->ninst_ptr_arr = (ninst_t **)malloc(sizeof(ninst_t *) * num_buffer);
        memcpy(path_layer_now->ninst_ptr_arr, buffer, sizeof(ninst_t *) * num_buffer);
        
        num_buffer_parent = ninst_find_parents(buffer, num_buffer, buffer_parent);

        num_buffer = num_buffer_parent;
        memcpy(buffer, buffer_parent, sizeof(ninst_t *) * num_buffer);
    }

    return path;
}

int fl_is_ninst_in_path_layer(fl_path_layer_t *path_layer, ninst_t *ninst) {
    for (int i=0; i<path_layer->num_ninsts; i++) {
        if (path_layer->ninst_ptr_arr[i] == ninst) {
            return 1;
        }
    }
    return 0;
}

void fl_push_path_ninsts(rpool_t *rpool, fl_path_t *path) {
    for (int i=0; i<path->num_path_layers; i++) {
        fl_path_layer_t *pushing_path_layer = &path->path_layers_arr[i];
        for (int j=0; j<pushing_path_layer->num_ninsts; j++) {
            printf("Push ninst in path %d in rpool group %d: N %d, L %d\n", path->path_idx, i, pushing_path_layer->ninst_ptr_arr[j]->ninst_idx, i+1);
            rpool_push_ninsts_to_group(rpool, &pushing_path_layer->ninst_ptr_arr[j], 1, i);
        }
    }
}

void fl_push_path_ninsts_until(rpool_t *rpool, fl_path_t *path, unsigned int last_layer_idx) {
    for (int i=0; i<=last_layer_idx; i++) {
        fl_path_layer_t *pushing_path_layer = &path->path_layers_arr[i];
        for (int j=0; j<pushing_path_layer->num_ninsts; j++) {
            printf("Push ninst in path %d in rpool group %d: N %d, L %d\n", path->path_idx, i, pushing_path_layer->ninst_ptr_arr[j]->ninst_idx, i+1);
            rpool_push_ninsts_to_group(rpool, &pushing_path_layer->ninst_ptr_arr[j], 1, i);
        }
    }
}

