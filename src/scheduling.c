#include "scheduling.h"

int is_ninst_mine(ninst_t *ninst, int device_idx) {
    for (int i=0; i<SCHEDULE_MAX_DEVICES; i++) {
        if (ninst->alloc_devices[i] == device_idx) return 1;
    }
    return 0;
}

void clear_device_alloc(ninst_t *ninst) {
    for (int i=0; i<SCHEDULE_MAX_DEVICES; i++) {
        ninst->alloc_devices[i] = -1;
    }
}

void alloc_device_to_ninst(ninst_t *ninst, int device_idx) {
    for (int i=0; i<SCHEDULE_MAX_DEVICES; i++) {
        if (ninst->alloc_devices[i] == -1) {
            ninst->alloc_devices[i] = device_idx;
        }
    }
}

void init_full_local(nasm_t *nasm) {
    for (int i = 0; i < nasm->num_ninst; i++) {
        ninst_t *ninst = nasm->ninst_arr + i;
        clear_device_alloc(ninst);
        alloc_device_to_ninst(ninst, SOCK_TX);
    }
}

void init_full_offload(nasm_t *nasm) {
    for (int i = 0; i < nasm->num_ninst; i++) {
        ninst_t *ninst = nasm->ninst_arr + i;
        clear_device_alloc(ninst);
        if (ninst->ldata->layer->layer_idx != nasm->num_ldata - 1) {
            alloc_device_to_ninst(ninst, SOCK_RX);
        }
        else {
            alloc_device_to_ninst(ninst, SOCK_TX);
        }
    }
}

void init_partial_offload(nasm_t *nasm, float compute_ratio) {
    int division_idx = (int)(nasm->ldata_arr[0].num_ninst * (1 - compute_ratio));
    printf("division idx: %d\n", division_idx);
    for (int i = 0; i < nasm->num_ninst; i++) {
        ninst_t *ninst = nasm->ninst_arr + i;
        clear_device_alloc(ninst);
        if (ninst->ldata->layer->layer_idx == 0) {  // for the first layer,
            if (ninst->ninst_idx < division_idx) {  // front ninsts are for RX
                alloc_device_to_ninst(ninst, SOCK_RX);
            }
            else if (ninst->ninst_idx > division_idx) { // behind ninsts are for TX
                alloc_device_to_ninst(ninst, SOCK_TX);
            }
            else {  // division ninst is for the both
                alloc_device_to_ninst(ninst, SOCK_RX);
                alloc_device_to_ninst(ninst, SOCK_TX);
            }
        }
        else if (ninst->ldata->layer->layer_idx != nasm->num_ldata - 1) {   // intermediate layers are for RX
            alloc_device_to_ninst(ninst, SOCK_RX);
        }
        else {  // final layer is for TX -> main.c has its own logic handling final layer
            // ninst->alloc_devices[0] = SOCK_TX;
            alloc_device_to_ninst(ninst, SOCK_RX);
        }
    }
}

void init_heft(char *target_config, char *target_bin, char *target_nasm_dir, ninst_profile_t **ninst_profile, network_profile_t *network_profile, int num_device) {
    aspen_dnn_t *target_dnn = apu_create_dnn(target_config, target_bin);
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
    float *AFT = calloc(num_ninst, sizeof(float));
    int *alloc_dev = calloc(num_ninst, sizeof(int));    // TODO: use for convenience!

    for (int i=0; i<num_ninst; i++) {
        ninst_t *target_ninst = queue_by_rank_upward[i];
        
        // calculate EST, EFT of target ninst
        if (target_ninst->ldata->layer->layer_idx == 0) {
            // case of entry task
            const unsigned int total_bytes = target_ninst->tile_dims[OUT_W] * target_ninst->tile_dims[OUT_H] * sizeof(float);

            EST[SOCK_TX] = 0;
            EST[SOCK_RX] = total_bytes / network_profile->transmit_rate;

            EFT[SOCK_TX] = EST[SOCK_TX] + W[i][SOCK_TX];
            EFT[SOCK_RX] = EST[SOCK_RX] + W[i][SOCK_RX];

            // find best processor
            float min_EFT = FLT_MAX;
            int min_EFT_proc;
            
            for (int proc=0; proc<num_device; proc++) {
                if (EFT[proc] < min_EFT) {
                    min_EFT = EFT[proc];
                    min_EFT_proc = proc;
                }
            }

            // push task into processor min_EFT_proc
            // record AFT[i] = min_EFT_proc
            /* TODO */
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
                        float dependency_time = AFT[parent] + L[parent_task->processor->idx] + data[parent_task->processor->idx][proc] / network_profile->transmit_rate;
                        max_dependency_time = max_dependency_time < dependency_time ? dependency_time : max_dependency_time;
                    }
                }

                // when can processor have big enough idle time?
                float avail = heft_earliest_idle(&(sched_processor_arr[proc]), max_dependency_time, W[i][proc]);

                EST[proc] = max_dependency_time > avail ? max_dependency_time : avail;
                EFT[proc] = EST[proc] + W[i][proc];

                if (min_EFT > EFT[proc]) {
                    min_EFT = EFT[proc];
                    min_EFT_proc = proc;
                }
            }

            // push task into processor min_EFT_proc at time EST[min_EFT_proc]
            // record AFT[i] = min_EFT_proc
            /* TODO */
        }
    }
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