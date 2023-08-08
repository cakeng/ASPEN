#include "profiling.h"

avg_ninst_profile_t *profile_computation(char *target_dnn_dir, char *target_nasm_dir, char *target_input, int gpu, int num_repeat) {
    // ninst_profile_t **ninst_profiles = calloc(num_repeat, sizeof(ninst_profile_t *));
    float avg_computation_time = 0.0;
    int total;

    for (int i=0; i<num_repeat; i++) {
        aspen_dnn_t *target_dnn = apu_load_dnn_from_file(target_dnn_dir);
        nasm_t *target_nasm = apu_load_nasm_from_file (target_nasm_dir, target_dnn);

        rpool_t *rpool = rpool_init (gpu);
        dse_group_t *dse_group = dse_group_init (16, gpu);
        dse_group_set_rpool (dse_group, rpool);
        dse_group_set_profile (dse_group, 1);

        rpool_add_nasm (rpool, target_nasm, target_input); 
        
        dse_group_run (dse_group);
        dse_wait_for_nasm_completion (target_nasm);
        dse_group_stop (dse_group);
        
        LAYER_PARAMS output_order[] = {BATCH, OUT_H, OUT_W, OUT_C};
        float *layer_output = dse_get_nasm_result (target_nasm, output_order);
        float *softmax_output = calloc (1000*target_nasm->batch_size, sizeof(float));
        naive_softmax (layer_output, softmax_output, target_nasm->batch_size, 1000);

        // ninst_profiles[i] = extract_profile_from_ninsts(target_nasm);
        total = target_nasm->num_ninst;
        avg_computation_time += extract_profile_from_ninsts(target_nasm);
        
        free (layer_output);
        free (softmax_output);

        dse_group_destroy (dse_group);
        rpool_destroy (rpool);
        apu_destroy_nasm (target_nasm);
        apu_destroy_dnn (target_dnn);
    }

    avg_ninst_profile_t *result = calloc(1, sizeof(ninst_profile_t));
    result->avg_computation_time = avg_computation_time / num_repeat;
    result->num_ninsts = total;

    // ninst_profile_t *result = merge_computation_profile(ninst_profiles, num_repeat);

    // for (int i=0; i<num_repeat; i++) {
        // free(ninst_profiles[i]);
    // }
    
    // free(ninst_profiles);

    return result;
}

network_profile_t *profile_network(avg_ninst_profile_t **ninst_profile, DEVICE_MODE device_mode, int server_sock, int client_sock) {
    network_profile_t *network_profile = malloc(sizeof(network_profile_t));
    
    const int num_repeat = PROFILE_REPEAT;
    int num_ninst = ninst_profile[device_mode]->num_ninsts;

    float time_offset = 0;
    float t0, t1, t2, t3;
    float rtt = 0.0;

    float long_send_timestamp;
    float long_recv_timestamp;
    float transmit_rate;

    if (device_mode == DEV_SERVER) { 
        printf("\tprofiling as SERVER...\n");

        // Profile RTT
        read_n(client_sock, &t0, sizeof(float));
        t1 = get_time_secs();
        write_n(client_sock, &t1, sizeof(float));
        t2 = get_time_secs();
        write_n(client_sock, &t2, sizeof(float));
        read_n(client_sock, &t3, sizeof(float));

        // Profile Bandwidth
        char* profile_message = malloc(PROFILE_LONG_MESSAGE_SIZE);
        for(int i = 0; i < num_repeat; i++)
        {
            read_n(client_sock, profile_message, PROFILE_LONG_MESSAGE_SIZE);
            long_recv_timestamp = get_time_secs();
            write_n(client_sock, &long_recv_timestamp, sizeof(float));
        }
        free(profile_message);

        // Receive & Send ninst_profile
        ninst_profile[DEV_EDGE] = malloc(sizeof(avg_ninst_profile_t));
        read_n(client_sock, ninst_profile[DEV_EDGE], sizeof(avg_ninst_profile_t));
        write_n(client_sock, ninst_profile[DEV_SERVER], sizeof(avg_ninst_profile_t));
        
        // Receive network_profile from edge
        read_n(client_sock, network_profile, sizeof(network_profile_t));
    }
    else 
    {
        printf("\tprofiling as EDGE...\n");
        // Profile RTT
        t0 = get_time_secs();
        write_n(server_sock, &t0, sizeof(float));
        read_n(server_sock, &t1, sizeof(float));
        t3 = get_time_secs();
        read_n(server_sock, &t2, sizeof(float));
        write_n(server_sock, &t3, sizeof(float));

        // Profile Bandwidth
        char *profile_message = malloc(PROFILE_LONG_MESSAGE_SIZE);
        for(int i = 0; i < num_repeat; i++)
        {
            long_send_timestamp = get_time_secs();
            write_n(server_sock, profile_message, PROFILE_LONG_MESSAGE_SIZE);
            read_n(server_sock, &long_recv_timestamp, sizeof(float));
            long_recv_timestamp = get_time_secs();
            transmit_rate += PROFILE_LONG_MESSAGE_SIZE / ((long_recv_timestamp - long_send_timestamp)) / 125000;
        }
        transmit_rate /= num_repeat;
        free(profile_message);

        // transmit_rate = num_ninst * sizeof(ninst_profile_t) / ((long_recv_timestamp - long_send_timestamp) / 2) / 125000; // Mbps
        rtt = (t3 - t0) - (t2 - t1);
        time_offset = ((t1 - t0) + (t2 - t3))/2;
        
        network_profile->rtt = rtt;
        network_profile->sync = time_offset;
        network_profile->transmit_rate = transmit_rate;

        // Send & receive ninst_profile;
        ninst_profile[DEV_SERVER] = malloc(sizeof(avg_ninst_profile_t));
        write_n(server_sock, ninst_profile[DEV_EDGE], sizeof(avg_ninst_profile_t));
        read_n(server_sock, ninst_profile[DEV_SERVER], sizeof(avg_ninst_profile_t));

        // Send network_profile
        write_n(server_sock, network_profile, sizeof(network_profile_t));
    }

    return network_profile;
}

float profile_network_sync(DEVICE_MODE device_mode, int server_sock, int client_sock) {
    float time_offset = 0;
    float t0, t1, t2, t3;
    float rtt = 0.0;

    if (device_mode == DEV_SERVER) {
        printf("\tprofiling as SERVER...\n");
        read_n(client_sock, &t0, sizeof(float));
        t1 = get_time_secs();
        write_n(client_sock, &t1, sizeof(float));
        t2 = get_time_secs();
        write_n(client_sock, &t2, sizeof(float));
        read_n(client_sock, &t3, sizeof(float));
    }
    else {
        printf("\tprofiling as EDGE...\n");
        t0 = get_time_secs();
        write_n(server_sock, &t0, sizeof(float));
        read_n(server_sock, &t1, sizeof(float));
        t3 = get_time_secs();
        read_n(server_sock, &t2, sizeof(float));
        write_n(server_sock, &t3, sizeof(float));
    }
    rtt = (t3 - t0) - (t2 - t1);
    time_offset = ((t1 - t0) + (t2 - t3))/2;
    return time_offset;
}

float extract_profile_from_ninsts(nasm_t *nasm) {
    float avg_computation_time = 0.0;
    
    for (int i=0; i<nasm->num_ninst; i++) {
        ninst_t *target_ninst = &(nasm->ninst_arr[i]);

        const unsigned int W = target_ninst->tile_dims[OUT_W];
        const unsigned int H = target_ninst->tile_dims[OUT_H];    
        const unsigned int total_bytes = W * H * sizeof(float);

        double elapsed_time = (target_ninst->compute_end - target_ninst->compute_start);
        if (elapsed_time > 0.0) {
            avg_computation_time += elapsed_time;
        } else {
            avg_computation_time += 0;
        }
    }
    avg_computation_time /= nasm->num_ninst;

    return avg_computation_time;
}

ninst_profile_t *merge_computation_profile(ninst_profile_t **ninst_profiles, int num_ninst_profiles) {
    
    int num_ninst = ninst_profiles[0][0].total;
    ninst_profile_t *merged = calloc(num_ninst, sizeof(ninst_profile_t));

    for(int i = 0; i < num_ninst; i++) {
        merged[i].computation_time = 0;
        merged[i].idx = i;
        merged[i].total = num_ninst;
        merged[i].transmit_size = ninst_profiles[0][i].transmit_size;
        for(int j = 0; j < num_ninst_profiles; j++) {
            merged[i].computation_time += ninst_profiles[j][i].computation_time;
        }
        merged[i].computation_time /= num_ninst_profiles;
    }

    return merged;
}

void save_computation_profile(ninst_profile_t *profile, char *file_path) {
    int num_ninst = profile[0].total;
    void *temp = malloc(sizeof(int) + sizeof(ninst_profile_t) * num_ninst);
    *((int *)temp) = num_ninst;
    memcpy((int *)temp + 1, profile, sizeof(ninst_profile_t) * num_ninst);
    save_arr(temp, file_path, sizeof(int) + sizeof(ninst_profile_t) * num_ninst);
    free(temp);
}

ninst_profile_t *load_computation_profile(char *file_path) {

    void *temp = load_arr(file_path, sizeof(int));
    int num_ninst = *(int *)temp;
    free(temp);

    ninst_profile_t *result = malloc(sizeof(ninst_profile_t) * num_ninst);
    temp = load_arr(file_path, sizeof(int) + sizeof(ninst_profile_t) * num_ninst);
    memcpy(result, (int *)temp + 1, sizeof(ninst_profile_t) * num_ninst);
    free(temp);

    return result;
}