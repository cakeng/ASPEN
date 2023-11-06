// #include "scheduling.h"

// // int is_offloaded(ninst_t *ninst)
// // {
// //     return atomic_load(&ninst->offloaded);
// // }

// int is_dev_compute(ninst_t *ninst, int device_idx)
// {
//     return atomic_load(&ninst->dev_to_compute[device_idx]);
// }

// int is_dev_send_target(ninst_t *ninst, int device_idx)
// {
//     return atomic_load(&ninst->dev_send_target[device_idx]);
// }

// int check_all_parents_target_device(ninst_t *ninst, nasm_t* nasm, int device_idx)
// {
//     for(int i = 0; i < ninst->num_parent_ninsts; i++)
//     {
//         int parent_idx = ninst->parent_ninst_idx_arr[i];
//         if(!atomic_load(&nasm->ninst_arr[parent_idx].dev_to_compute[device_idx]))
//         {
//             return 0;
//         }
//     }
//     return 1;
// }

// void ninst_copy_compute_device(ninst_t* target_ninst, ninst_t* ninst)
// {
//     // ninst --> target_ninst
//     for (int i = 0; i < SCHEDULE_MAX_DEVICES; i++) 
//     {
//         atomic_store(&(target_ninst->dev_to_compute[i]), ninst->dev_to_compute[i]);
//         // target_ninst->dev_to_compute[i] = ninst->dev_to_compute[i];
//     }
// }

// void ninst_clear_compute_device(ninst_t *ninst) 
// {
//     for (int i = 0; i < SCHEDULE_MAX_DEVICES; i++) 
//     {
//         atomic_store(&ninst->dev_to_compute[i], 0);
//         // ninst->dev_to_compute[i] = 0;
//     }
// }

// void ninst_set_compute_device(ninst_t *ninst, int device_idx) 
// {
//     atomic_store(&ninst->dev_to_compute[device_idx], 1);
//     // ninst->dev_to_compute[device_idx] = 1;
    
// }

// void ninst_set_send_target_device(ninst_t *ninst, int device_idx)
// {
//     atomic_store(&ninst->dev_send_target[device_idx], 1);
//     // ninst->dev_send_target[device_idx] = 1;
// }

// void ninst_clear_send_target_device(ninst_t *ninst) 
// {
//     for (int i = 0; i<SCHEDULE_MAX_DEVICES; i++) 
//     {
//         atomic_store(&ninst->dev_send_target[i], 0);
//         // ninst->dev_send_target[i] = 0;
//     }
// }

// void nasm_set_ninst_send_target_using_child_compute_device(nasm_t *nasm) 
// {
//     for (int i = 0; i < nasm->num_ninst; i++) 
//     {
//         ninst_t *ninst = nasm->ninst_arr + i;
//         ninst_clear_send_target_device(ninst);
//         for (int j = 0; j < ninst->num_child_ninsts; j++) 
//         {
//             ninst_t *child_ninst = ninst->child_ninst_arr[j];
//             for (int dev = 0; dev < SCHEDULE_MAX_DEVICES; dev++) 
//             {
//                 ninst->dev_send_target[dev] |= child_ninst->dev_to_compute[dev];
//                 ninst->dev_to_compute[dev] |= child_ninst->dev_to_compute[dev];
//             }
//         }
//     }
// }

// void nasm_all_ninst_set_compute_device (nasm_t *nasm, int device_idx) 
// {
//     for (int i = 0; i < nasm->num_ninst; i++) 
//     {
//         ninst_t *ninst = nasm->ninst_arr + i;
//         ninst_clear_send_target_device(ninst);
//         atomic_store(&ninst->dev_send_target[device_idx], 1);
//     }
// }

// void nasm_set_last_layer_ninst_send_target_device(nasm_t *nasm, int device_idx) 
// {
//     nasm_ldata_t *last_ldata = &(nasm->ldata_arr[nasm->num_ldata-1]);
//     ninst_t *last_ldata_ninst_arr = last_ldata->ninst_arr_start;
//     for (int i = 0; i < last_ldata->num_ninst; i++) 
//     {
//         // last_ldata_ninst_arr[i].dev_send_target[device_idx] = 1;
//         atomic_store(&last_ldata_ninst_arr[i].dev_send_target[device_idx], 1);
//     }
// }

// dynamic_scheduler_t* init_dynamic_scheduler(avg_ninst_profile_t **ninst_profile, network_profile_t **network_profile, DEVICE_MODE device_mode, int device_idx, int num_edge_devices)
// {
//     dynamic_scheduler_t *dynamic_scheduler = calloc(1, sizeof(dynamic_scheduler_t));

//     for(int i = 0; i < num_edge_devices; i++)
//     {
//         if(device_mode == DEV_SERVER || device_idx == i)
//         {
//             dynamic_scheduler->avg_server_ninst_compute_time[i] = ninst_profile[i]->avg_server_computation_time;
//             dynamic_scheduler->avg_edge_ninst_compute_time[i] = ninst_profile[i]->avg_edge_computation_time;
//             dynamic_scheduler->avg_bandwidth[i] = network_profile[i]->transmit_rate;
//             dynamic_scheduler->rtt[i] = network_profile[i]->rtt;
//             dynamic_scheduler->edge_num_dse[i] = ninst_profile[i]->edge_num_dse;
//             dynamic_scheduler->server_num_dse[i] = ninst_profile[i]->server_num_dse;
//             // ** TODO : Implement PF scheduler **
//             dynamic_scheduler->scheduling_latency[i] = 0.0;
//         }
//     }
    
//     return dynamic_scheduler;
// }

// float get_edge_offline_latency_to_split_layer(spinn_scheduler_t* spinn_scheduler, nasm_t* nasm, int device_idx, int split_layer)
// {
//     int num_ninsts = 0;
//     float latency_sum = 0.0;
//     for(int i = 0; i < split_layer; i++)
//         num_ninsts += nasm->ldata_arr[i].num_ninst;
    
//     latency_sum = spinn_scheduler->avg_edge_ninst_compute_time[device_idx] * num_ninsts / spinn_scheduler->edge_num_dse[device_idx];

//     return latency_sum;
// }

// float get_server_offline_latency_from_split_layer(spinn_scheduler_t* spinn_scheduler, nasm_t* nasm, int device_idx, int split_layer)
// {
//     int num_ninsts = 0;
//     float latency_sum = 0.0;
//     for(int i = split_layer; i < nasm->num_ldata; i++)
//         num_ninsts += nasm->ldata_arr[i].num_ninst;
        
//     latency_sum = spinn_scheduler->avg_server_ninst_compute_time[device_idx] * num_ninsts / spinn_scheduler->server_num_dse[device_idx];

//     return latency_sum;
// }

// float get_data_transmission_latency(spinn_scheduler_t* spinn_scheduler, nasm_t* nasm, int device_idx, int split_layer)
// {
//     int data_size = 0;
//     float latency_sum = 0.0;
//     for(int j = 0; j < nasm->ldata_arr[split_layer-1].num_ninst; j++)
//         data_size += nasm->ldata_arr[split_layer-1].ninst_arr_start[j].tile_dims[OUT_W] * nasm->ldata_arr[split_layer-1].ninst_arr_start[j].tile_dims[OUT_H] * sizeof(float);
    
//     data_size = data_size * 8;
//     float bit_per_second = spinn_scheduler->avg_bandwidth[device_idx] * 1000000;
//     latency_sum = spinn_scheduler->rtt[device_idx] + // RTT
//                 data_size / bit_per_second; // Transmission latency;
//     return latency_sum;
// }

// void spinn_offline_profile(spinn_scheduler_t* spinn_scheduler, nasm_t* nasm, int device_idx)
// {
//     for(int i = 0; i < spinn_scheduler->num_split_candidates[device_idx]; i++)   
//     {
//         int split_layer = spinn_scheduler->split_candidates[device_idx][i];
//         spinn_scheduler->edge_offline_layer_latency[device_idx][i] = get_edge_offline_latency_to_split_layer(spinn_scheduler, nasm, device_idx, split_layer);
//         spinn_scheduler->server_offline_layer_latency[device_idx][i] = get_server_offline_latency_from_split_layer(spinn_scheduler, nasm, device_idx, split_layer);
//         spinn_scheduler->edge_real_latency[device_idx][i] = spinn_scheduler->edge_offline_layer_latency[device_idx][i];
//         spinn_scheduler->server_real_latency[device_idx][i] = spinn_scheduler->server_offline_layer_latency[device_idx][i];
//         spinn_scheduler->edge_scaling_factors[device_idx] = 1.0;
//         spinn_scheduler->server_scaling_factors[device_idx] = 1.0;

//         int data_size = 0;
//         if (split_layer > 0)
//         {
//             for(int j = 0; j < nasm->ldata_arr[split_layer-1].num_ninst; j++)
//                 data_size += nasm->ldata_arr[split_layer-1].ninst_arr_start[j].tile_dims[OUT_W] * nasm->ldata_arr[split_layer-1].ninst_arr_start[j].tile_dims[OUT_H] * sizeof(float);
//         }
//         spinn_scheduler->data_size_split_candidates[device_idx][i] = data_size;
//     }
// }

// void spinn_update_profile(spinn_scheduler_t* spinn_scheduler, float rtt, float avg_bandwidth, float avg_edge_latency, float avg_server_latency, int device_idx)
// {
//     int current_split_layer = spinn_scheduler->current_split_layer[device_idx];
//     int idx = -1;
//     for (int i = 0; i < spinn_scheduler->num_split_candidates[device_idx]; i++)
//     {
//         if (spinn_scheduler->split_candidates[device_idx][i] == current_split_layer)
//         {
//             idx = i;
//             break;
//         }
//     }
//     if (idx < 0)
//     {
//         PRTF("Error: Cannot find split layer %d in split candidates\n", current_split_layer);
//         exit(1);
//     }
    
//     spinn_scheduler->rtt[device_idx] = rtt;
//     spinn_scheduler->avg_bandwidth[device_idx] = 0.85 * spinn_scheduler->avg_bandwidth[device_idx] + 0.15 * avg_bandwidth;
//     if(avg_edge_latency <= 0) avg_edge_latency = spinn_scheduler->edge_offline_layer_latency[device_idx][idx];
//     if(avg_server_latency <= 0) avg_server_latency = spinn_scheduler->server_offline_layer_latency[device_idx][idx];
    
//     spinn_scheduler->edge_real_latency[device_idx][idx] = avg_edge_latency;
//     spinn_scheduler->server_real_latency[device_idx][idx] = avg_server_latency;
//     spinn_scheduler->edge_scaling_factors[device_idx] = spinn_scheduler->edge_real_latency[device_idx][idx] / spinn_scheduler->edge_offline_layer_latency[device_idx][idx];
//     spinn_scheduler->server_scaling_factors[device_idx] = spinn_scheduler->server_real_latency[device_idx][idx] / spinn_scheduler->server_offline_layer_latency[device_idx][idx];    
// }

// spinn_scheduler_t* init_spinn_scheduler(avg_ninst_profile_t **ninst_profile, network_profile_t **network_profile, nasm_t** nasms, DEVICE_MODE device_mode, int device_idx, int num_edge_devices)    
// {
//     spinn_scheduler_t* spinn_scheduler = calloc(1, sizeof(spinn_scheduler_t));
    
//     for(int edge_id = 0; edge_id < num_edge_devices; edge_id++)
//     {
//         if(device_mode == DEV_SERVER || device_idx == edge_id)
//         {
//             spinn_scheduler->avg_server_ninst_compute_time[edge_id] = ninst_profile[edge_id]->avg_server_computation_time;
//             spinn_scheduler->avg_edge_ninst_compute_time[edge_id] = ninst_profile[edge_id]->avg_edge_computation_time;
//             spinn_scheduler->avg_bandwidth[edge_id] = network_profile[edge_id]->transmit_rate;
//             spinn_scheduler->rtt[edge_id] = network_profile[edge_id]->rtt;
//             spinn_scheduler->edge_num_dse[edge_id] = ninst_profile[edge_id]->edge_num_dse;
//             spinn_scheduler->server_num_dse[edge_id] = ninst_profile[edge_id]->server_num_dse;
            
//             // Model Splitter : Find Conv, Maxpool, Residual layers and store indices to split_candidates
//             spinn_model_splitter(spinn_scheduler, nasms[edge_id], edge_id);
//             spinn_offline_profile(spinn_scheduler, nasms[edge_id], edge_id);
//         }
//     }

//     return spinn_scheduler;
// }

// int spinn_schedule_layer(spinn_scheduler_t* spinn_scheduler, nasm_t* nasm, int device_idx)
// {
//     PRTF("\t[SPINN Scheduler]\n");
//     int split_layer = 1;
//     float min_latency = 100000000.0;
//     for(int i = 0; i < spinn_scheduler->num_split_candidates[device_idx]; i++)
//     {
//         float latency = spinn_scheduler->edge_offline_layer_latency[device_idx][i] * spinn_scheduler->edge_scaling_factors[device_idx] + 
//             spinn_scheduler->server_offline_layer_latency[device_idx][i] * spinn_scheduler->server_scaling_factors[device_idx] +
//             get_data_transmission_latency(spinn_scheduler, nasm, device_idx, spinn_scheduler->split_candidates[device_idx][i]);
        
//         // PRTF("\tSplit Layer: %d, Edge: %fms, Server: %fms, Transmission: %fms\n",
//         //     spinn_scheduler->split_candidates[device_idx][i],
//         //     spinn_scheduler->edge_offline_layer_latency[device_idx][i] * spinn_scheduler->edge_scaling_factors[device_idx] * 1000.0,
//         //     spinn_scheduler->server_offline_layer_latency[device_idx][i] * spinn_scheduler->server_scaling_factors[device_idx] * 1000.0,
//         //     get_data_transmission_latency(spinn_scheduler, nasm, device_idx, spinn_scheduler->split_candidates[device_idx][i]) * 1000.0
//             // );
//         // PRTF("BW: %f\n", spinn_scheduler->avg_bandwidth[device_idx]);

//         if(latency < min_latency)
//         {
//             // PRTF("\tSplit Layer: %d, latency: %f, min_latency: %f\n", spinn_scheduler->split_candidates[device_idx][i], latency, min_latency);
//             min_latency = latency;
//             split_layer = spinn_scheduler->split_candidates[device_idx][i];
//         }
//     }
//     spinn_scheduler->current_split_layer[device_idx] = split_layer;

//     return split_layer;
// }

// int spinn_find_idx_by_split_layer(spinn_scheduler_t* spinn_scheduler, int device_idx)
// {
//     int current_split_layer = spinn_scheduler->current_split_layer[device_idx];
//     int idx = -1;
//     for (int i = 0; i < spinn_scheduler->num_split_candidates[device_idx]; i++)
//     {
//         if (spinn_scheduler->split_candidates[device_idx][i] == current_split_layer)
//         {
//             idx = i;
//             break;
//         }
//     }
//     return idx;
// }

// void spinn_model_splitter(spinn_scheduler_t* spinn_scheduler, nasm_t* nasm, int device_idx)
// {
//     PRTF("[SPINN Model Splitter]\n");
//     PRTF("\tObtained split candidates: ");
//     int num_split_candidates = 0;
//     spinn_scheduler->split_candidates[device_idx] = calloc(nasm->num_ldata, sizeof(int));
//     spinn_scheduler->split_candidates[device_idx][num_split_candidates] = 1;
//     PRTF("%d ", spinn_scheduler->split_candidates[device_idx][num_split_candidates]);
//     num_split_candidates++;

//     for(int i = 2; i < nasm->num_ldata-1; i++)
//     {
//         if(nasm->ldata_arr[i].layer->type == CONV_LAYER || nasm->ldata_arr[i].layer->type == MAXPOOL_LAYER || nasm->ldata_arr[i].layer->type == RESIDUAL_LAYER)
//         {
//             spinn_scheduler->split_candidates[device_idx][num_split_candidates] = i;
//             PRTF("%d ", spinn_scheduler->split_candidates[device_idx][num_split_candidates]);
//             num_split_candidates++;
//         }
//     }
//     spinn_scheduler->split_candidates[device_idx][num_split_candidates] = nasm->num_ldata-1;
//     PRTF("%d ", spinn_scheduler->split_candidates[device_idx][num_split_candidates]);
//     num_split_candidates++;
//     spinn_scheduler->server_offline_layer_latency[device_idx] = calloc(num_split_candidates, sizeof(float));
//     spinn_scheduler->server_real_latency[device_idx] = calloc(num_split_candidates, sizeof(float));
//     spinn_scheduler->edge_offline_layer_latency[device_idx] = calloc(num_split_candidates, sizeof(float));
//     spinn_scheduler->edge_real_latency[device_idx] = calloc(num_split_candidates, sizeof(float));
//     // spinn_scheduler->edge_scaling_factors[device_idx] = calloc(num_split_candidates, sizeof(float));
//     // spinn_scheduler->server_scaling_factors[device_idx] = calloc(num_split_candidates, sizeof(float));
//     spinn_scheduler->data_size_split_candidates[device_idx] = calloc(num_split_candidates, sizeof(int));
//     spinn_scheduler->num_split_candidates[device_idx] = num_split_candidates;
    
    
//     PRTF("\n");
//     // PRTF("\tTotal num split candidates: (%d/%d)\n", num_split_candidates, nasm->num_ldata);
// }

// float get_eft_edge(dynamic_scheduler_t* dynamic_scheduler, rpool_t* rpool, int device_idx, int num_dse, int num_child_ninsts)
// {
//     unsigned int rpool_num_stored = atomic_load(&rpool->num_stored);
//     float eft_edge = (float)(rpool_num_stored + num_child_ninsts) * dynamic_scheduler->avg_edge_ninst_compute_time[device_idx] / num_dse;
//     return eft_edge;
// }

// float get_eft_server(dynamic_scheduler_t* dynamic_scheduler, networking_engine_t* net_engine, int device_idx, int net_tx_queue_bytes)
// {
//     unsigned int net_tx_queue_num_stored = atomic_load(&net_engine->tx_queue->num_stored);
//     float eft_edge = dynamic_scheduler->rtt[device_idx] + // RTT
//                     (net_tx_queue_num_stored) * net_tx_queue_bytes * 8 / dynamic_scheduler->avg_bandwidth[device_idx] / 1000000 // Transmission latency
//                     + dynamic_scheduler->scheduling_latency[device_idx];
//     return eft_edge;
// }

// void init_full_local(nasm_t *nasm, int dev_idx) {
//     for (int i = 0; i < nasm->num_ninst; i++) 
//     {
//         ninst_t *ninst = nasm->ninst_arr + i;
//         ninst_clear_compute_device(ninst);
//         ninst_set_compute_device(ninst, dev_idx);
//     }
//     nasm_set_ninst_send_target_using_child_compute_device(nasm);
// }

// void init_full_offload(nasm_t *nasm, int edge_id, int server_id) {
//     for (int i = 0; i < nasm->num_ninst; i++) 
//     {
//         ninst_t *ninst = nasm->ninst_arr + i;
//         ninst_clear_compute_device(ninst);
//         if (ninst->ldata->layer->layer_idx != nasm->num_ldata - 1) 
//         {
//             ninst_set_compute_device(ninst, server_id);
//         }
//         else 
//         {
//             ninst_set_compute_device(ninst, edge_id);
//         }
//     }
//     nasm_set_ninst_send_target_using_child_compute_device(nasm);
// }

// void init_random_offload(nasm_t *nasm, float compute_ratio, int edge_id, int server_id)
// {
//     srand(time(NULL));
//     int total_num_ninst = nasm->num_ninst - nasm->ldata_arr[nasm->num_ldata-1].num_ninst;
//     int num_selected = (int)(compute_ratio * total_num_ninst);

//     if(num_selected > total_num_ninst)
//     {
//         PRTF("Error: num_selected > total_num_ninst\n");
//         exit(1);
//     }

//     // PRTF("\t[Random Offload] Selected ninsts: ");

//     int selected_ninst_idx[num_selected];
//     for(int i = 0; i < num_selected; i++)
//     {
//         selected_ninst_idx[i] = rand() % total_num_ninst;
//         // PRTF("%d ", selected_ninst_idx[i]);
//     }
//     // PRTF("\n");

    
//     for(int i = 0; i < total_num_ninst; i++)
//     {
//         ninst_t* ninst = nasm->ninst_arr + i;
//         ninst_clear_compute_device(ninst);
//         ninst_clear_send_target_device(ninst);
//         if (ninst->ldata->layer->layer_idx == 0) {  // for the input data,
//             ninst_set_compute_device(ninst, edge_id);  // all inputs are generated from TX
//             ninst_set_send_target_device(ninst, server_id);
//         }
//         else
//         {
//             ninst_set_compute_device(ninst, server_id);
//             ninst_set_compute_device(ninst, edge_id);
//             for(int count = 0; count < num_selected; count++)
//             {
//                 if(ninst->ninst_idx == selected_ninst_idx[count])
//                 {
//                     ninst_set_send_target_device(ninst, server_id);
//                     break;
//                 }
//             }
//         }
//     }

//     // for(int i = 0; i < total_num_ninst; i++)
//     // {
//     //     ninst_t* ninst = nasm->ninst_arr + i;
//     //     PRTF("%d: %d, %d\n", ninst->ninst_idx, ninst->dev_to_compute[server_id], ninst->dev_to_compute[edge_id]);
//     // }

//     // nasm_set_ninst_send_target_using_child_compute_device(nasm);
//     nasm_set_last_layer_ninst_send_target_device(nasm, edge_id);

//     // for(int i = 0; i < total_num_ninst; i++)
//     // {
//     //     ninst_t* ninst = nasm->ninst_arr + i;
//     //     PRTF("%d: %d, %d\n", ninst->ninst_idx, ninst->dev_send_target[server_id], ninst->dev_send_target[edge_id]);
//     // }
// }

// void init_partial_offload(nasm_t *nasm, int split_layer, float compute_ratio, int edge_id, int server_id) {
//     int layer_start_ninst_idx = nasm->ldata_arr[split_layer].ninst_arr_start[0].ninst_idx;
//     int layer_end_ninst_idx = layer_start_ninst_idx + nasm->ldata_arr[split_layer].num_ninst;
    
//     int division_idx = layer_start_ninst_idx + (1-compute_ratio) * (layer_end_ninst_idx - layer_start_ninst_idx);
//     PRTF("division idx: %d\n", division_idx);
//     for (int i = 0; i < nasm->num_ninst; i++) {
//         ninst_t *ninst = nasm->ninst_arr + i;
//         ninst_clear_compute_device(ninst);
//         if (ninst->ldata->layer->layer_idx == 0) {  // for the input data,
//             ninst_set_compute_device(ninst, edge_id);  // all inputs are generated from TX
//         }
//         else if (ninst->ldata->layer->layer_idx < split_layer) { 
//             ninst_set_compute_device(ninst, edge_id);
//         }
//         else if (ninst->ldata->layer->layer_idx == split_layer) {
//             if (ninst->ninst_idx < division_idx) {  // front ninsts are for RX
//                 ninst_set_compute_device(ninst, server_id);
//             }
//             else if (ninst->ninst_idx > division_idx) { // behind ninsts are for TX
//                 ninst_set_compute_device(ninst, edge_id);
//             }
//             else {  // division ninst is for the both
//                 ninst_set_compute_device(ninst, server_id);
//                 ninst_set_compute_device(ninst, edge_id);
//             }
//         }
//         else
//         {
//             ninst_set_compute_device(ninst, server_id);
//         }
//     }
//     nasm_set_ninst_send_target_using_child_compute_device(nasm);
//     nasm_set_last_layer_ninst_send_target_device(nasm, edge_id);
// }

// void init_sequential_offload(nasm_t *nasm, int split_layer, int edge_id, int server_id) 
// {
//     PRTF("[Init sequential offload] division ninst idx: %d from dev: %d server_id: %d\n", split_layer, edge_id, server_id);
//     for (int i = 0; i < nasm->num_ldata; i++) 
//     {
//         if (i < split_layer) 
//         {
//             for (int j = 0; j < nasm->ldata_arr[i].num_ninst; j++) 
//             {
//                 ninst_clear_compute_device(&(nasm->ldata_arr[i].ninst_arr_start[j]));
//                 ninst_set_compute_device(&(nasm->ldata_arr[i].ninst_arr_start[j]), edge_id);
//                 ninst_set_send_target_device(&(nasm->ldata_arr[i].ninst_arr_start[j]), server_id);
//             }
//         }
//         else 
//         {
//             for (int j = 0; j < nasm->ldata_arr[i].num_ninst; j++) 
//             {
//                 ninst_clear_compute_device(&(nasm->ldata_arr[i].ninst_arr_start[j]));
//                 ninst_set_compute_device(&(nasm->ldata_arr[i].ninst_arr_start[j]), server_id);
//             }
//         }
//     }

//     nasm_set_ninst_send_target_using_child_compute_device (nasm);
//     nasm_set_last_layer_ninst_send_target_device (nasm, edge_id);
// }

// void init_dynamic_offload(nasm_t *nasm, DEVICE_MODE device_mode, int edge_id, int server_id) 
// {
//     for (int i = 0; i < nasm->num_ldata; i++) 
//     {
//         for (int j = 0; j < nasm->ldata_arr[i].num_ninst; j++) 
//         {
//             ninst_clear_compute_device(&(nasm->ldata_arr[i].ninst_arr_start[j]));
//             ninst_clear_send_target_device(&(nasm->ldata_arr[i].ninst_arr_start[j]));
            
//             if(device_mode == DEV_SERVER)
//                 ninst_set_compute_device(&(nasm->ldata_arr[i].ninst_arr_start[j]), server_id);
//             else if(device_mode == DEV_EDGE)
//                 ninst_set_compute_device(&(nasm->ldata_arr[i].ninst_arr_start[j]), edge_id);
//             else
//                 assert(0);
//         }
//     }
    
//     for (int i = 0; i < nasm->ldata_arr[0].num_ninst; i++)
//     {
//         ninst_set_compute_device(&(nasm->ldata_arr[0].ninst_arr_start[i]), edge_id);
//         ninst_set_send_target_device(&(nasm->ldata_arr[0].ninst_arr_start[i]), server_id);
//     }

//     // Set last layer to edge as default
//     nasm_set_last_layer_ninst_send_target_device(nasm, edge_id);
// }

// void init_conventional_offload(nasm_t *nasm, int edge_id, int server_id)  
// {
//     for (int i = 0; i < nasm->num_ldata; i++) 
//     {
//         for (int j = 0; j<nasm->ldata_arr[i].num_ninst; j++) 
//         {
//             ninst_clear_compute_device(&(nasm->ldata_arr[i].ninst_arr_start[j]));
//             ninst_set_compute_device(&(nasm->ldata_arr[i].ninst_arr_start[j]), server_id);
//         }
//     }
//     nasm_all_ninst_set_compute_device(nasm, server_id);
//     nasm_set_last_layer_ninst_send_target_device(nasm, edge_id);
// }
