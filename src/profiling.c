#include "profiling.h"

ninst_profile_t *profile_computation(char *target_config, char *target_bin, char *target_nasm_dir, char *target_input, int gpu, int num_repeat) {
    ninst_profile_t **ninst_profiles = calloc(num_repeat, sizeof(ninst_profile_t *));

    for (int i=0; i<num_repeat; i++) {
        aspen_dnn_t *target_dnn = apu_create_dnn(target_config, target_bin);
        nasm_t *target_nasm = apu_load_nasm_from_file (target_nasm_dir, target_dnn);

        rpool_t *rpool = rpool_init (gpu);
        dse_group_t *dse_group = dse_group_init (16, gpu);
        dse_group_set_rpool (dse_group, rpool);
        dse_group_set_profile (dse_group, 1);

        rpool_add_nasm (rpool, target_nasm, 1.0, target_input); 
        
        dse_group_run (dse_group);
        dse_wait_for_nasm_completion (target_nasm);
        dse_group_stop (dse_group);
        
        LAYER_PARAMS output_order[] = {BATCH, OUT_H, OUT_W, OUT_C};
        float *layer_output = dse_get_nasm_result (target_nasm, output_order);
        float *softmax_output = calloc (1000*target_nasm->batch_size, sizeof(float));
        naive_softmax (layer_output, softmax_output, target_nasm->batch_size, 1000);

        ninst_profiles[i] = extract_profile_from_ninsts(target_nasm);
        
        free (layer_output);
        free (softmax_output);

        dse_group_destroy (dse_group);
        rpool_destroy (rpool);
        apu_destroy_nasm (target_nasm);
        apu_destroy_dnn (target_dnn);
    }

    ninst_profile_t *result = merge_computation_profile(ninst_profiles, num_repeat);

    for (int i=0; i<num_repeat; i++) {
        free(ninst_profiles[i]);
    }
    
    free(ninst_profiles);

    return result;
}

network_profile_t *profile_network(ninst_profile_t **ninst_profile, int sock_type, int server_sock, int client_sock) {
    network_profile_t *network_profile = malloc(sizeof(network_profile_t));
    
    const int num_repeat = 4;
    int num_ninst = ninst_profile[sock_type]->total;

    if (sock_type == SOCK_RX) { // echo
        printf("\tprofiling as RX...\n");

        // echo shortmessage
        for (int i=0; i<num_repeat; i++) {
            float buf;
            read_n(client_sock, &buf, sizeof(float));
            buf = get_time_secs();
            write_n(client_sock, &buf, sizeof(float));
        }

        // receive & send ninst_profile
        ninst_profile[!sock_type] = malloc(num_ninst * sizeof(ninst_profile_t));
        read_n(client_sock, ninst_profile[!sock_type], num_ninst * sizeof(ninst_profile_t));
        write_n(client_sock, ninst_profile[sock_type], num_ninst * sizeof(ninst_profile_t));
        
        // receive network_profile
        read_n(client_sock, network_profile, sizeof(network_profile_t));
    }
    else {
        printf("\tprofiling as TX...\n");

        // send shortmessage
        float send_timestamp[num_repeat];
        float server_timestamp[num_repeat];
        float recv_timestamp[num_repeat];

        float sync = 0;
        float rtt = 0;

        for (int i=0; i<num_repeat; i++) {
            send_timestamp[i] = get_time_secs();
            write_n(server_sock, &send_timestamp[i], sizeof(float));
            read_n(server_sock, &server_timestamp[i], sizeof(float));
            recv_timestamp[i] = get_time_secs();

            sync += server_timestamp[i] - (recv_timestamp[i] + send_timestamp[i]) / 2;
            rtt += recv_timestamp[i] - send_timestamp[i];

        }

        sync /= num_repeat;
        rtt /= num_repeat;

        // send & receive ninst_profile;
        float long_send_timestamp;
        float long_recv_timestamp;
        float transmit_rate;

        ninst_profile[!sock_type] = malloc(num_ninst * sizeof(ninst_profile_t));
        long_send_timestamp = get_time_secs();
        write_n(server_sock, ninst_profile[sock_type], num_ninst * sizeof(ninst_profile_t));
        read_n(server_sock, ninst_profile[!sock_type], num_ninst * sizeof(ninst_profile_t));
        long_recv_timestamp = get_time_secs();

        transmit_rate = num_ninst * sizeof(ninst_profile_t) / ((long_recv_timestamp - long_send_timestamp) / 2);


        // send network_profile
        network_profile->rtt = rtt;
        network_profile->sync = sync;
        network_profile->transmit_rate = transmit_rate;

        write_n(server_sock, network_profile, sizeof(network_profile_t));
    }

    return network_profile;
}

ninst_profile_t *extract_profile_from_ninsts(nasm_t *nasm) {
    ninst_profile_t *result = calloc(nasm->num_ninst, sizeof(ninst_profile_t));
    for (int i=0; i<nasm->num_ninst; i++) {
        ninst_t *target_ninst = &(nasm->ninst_arr[i]);

        const unsigned int W = target_ninst->tile_dims[OUT_W];
        const unsigned int H = target_ninst->tile_dims[OUT_H];    
        const unsigned int total_bytes = W * H * sizeof(float);

        result[i].idx = i;
        result[i].total = nasm->num_ninst;
        result[i].computation_time = target_ninst->compute_end - target_ninst->compute_start;
        result[i].transmit_size = total_bytes;
    }

    return result;
}

ninst_profile_t *merge_computation_profile(ninst_profile_t **ninst_profiles, int num_ninst_profiles) {
    int num_ninst = ninst_profiles[0][0].total;
    ninst_profile_t *merged = calloc(num_ninst, sizeof(ninst_profile_t));

    for(int i=0; i<num_ninst; i++) {
        merged[i].computation_time = 0;
        merged[i].idx = i;
        merged[i].total = num_ninst;
        merged[i].transmit_size = ninst_profiles[0][i].transmit_size;
        for(int j=0; j<num_ninst_profiles; j++) {
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