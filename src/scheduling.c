#include "scheduling.h"

int is_ninst_mine(ninst_t *ninst, int device_idx) {
    return ninst->alloc_devices[0] == device_idx || ninst->alloc_devices[1] == device_idx;
}

void init_full_offload(nasm_t *nasm) {
    for (int i = 0; i < nasm->num_ninst; i++) {
        ninst_t *ninst = nasm->ninst_arr + i;
        if (ninst->ldata->layer->layer_idx != nasm->num_ldata - 1) {
            ninst->alloc_devices[0] = SOCK_RX;
            ninst->alloc_devices[1] = -1;
        }
        else {
            ninst->alloc_devices[0] = SOCK_TX;
            ninst->alloc_devices[1] = -1;
        }
    }
}

void init_partial_offload(nasm_t *nasm, float compute_ratio) {
    int division_idx = (int)(nasm->ldata_arr[0].num_ninst * (1 - compute_ratio));
    printf("division idx: %d\n", division_idx);
    for (int i = 0; i < nasm->num_ninst; i++) {
        ninst_t *ninst = nasm->ninst_arr + i;
        if (ninst->ldata->layer->layer_idx == 0) {  // for the first layer,
            if (ninst->ninst_idx < division_idx) {  // front ninsts are for RX
                ninst->alloc_devices[0] = SOCK_RX;
                ninst->alloc_devices[1] = -1;
            }
            else if (ninst->ninst_idx > division_idx) { // behind ninsts are for TX
                ninst->alloc_devices[0] = SOCK_TX;
                ninst->alloc_devices[1] = -1;
            }
            else {  // division ninst is for the both
                ninst->alloc_devices[0] = SOCK_RX;
                ninst->alloc_devices[1] = SOCK_TX;
            }
        }
        else if (ninst->ldata->layer->layer_idx != nasm->num_ldata - 1) {   // intermediate layers are for RX
            ninst->alloc_devices[0] = SOCK_RX;
            ninst->alloc_devices[1] = -1;
        }
        else {  // final layer is for TX -> main.c has its own logic handling final layer
            // ninst->alloc_devices[0] = SOCK_TX;
            ninst->alloc_devices[0] = SOCK_RX;
            ninst->alloc_devices[1] = -1;
        }
    }
}

void init_heft_devices(device_t **devices, SOCK_TYPE *types, char **ips, int* ports, int num_devices, int my_dev_idx) {
    device_t *mydev = devices[my_dev_idx];

    mydev->type = types[my_dev_idx];
    mydev->ip = ips[my_dev_idx];
    mydev->port = ports[my_dev_idx];

    // open socket of mine
    if (mydev->type == SOCK_RX) {
        mydev->sock = socket(PF_INET, SOCK_STREAM, 0);
        if (mydev->sock == -1) {
            printf("Error: socket() returned -1\n");
            assert(0);
        }

        memset(&mydev->server_addr, 0, sizeof(mydev->server_addr));
        mydev->server_addr.sin_family = AF_INET;
        mydev->server_addr.sin_addr.s_addr = htonl(INADDR_ANY);
        mydev->server_addr.sin_port = htons(ports[my_dev_idx]);
        
        if (bind(mydev->sock, (struct sockaddr*)&mydev->server_addr, sizeof(mydev->server_addr)) == -1) {
            printf("Error: bind() returned -1\n");
            assert(0);
        }

        if (listen(mydev->sock, 5) == -1) {
            printf("Error: listen() returned -1\n");
            assert(0);
        }
    }
    else if (mydev->type == SOCK_TX) {
        mydev->sock = socket(PF_INET, SOCK_STREAM, 0);
        if(mydev->sock == -1)
            printf("Error: socket() returned -1\n");
            assert(0);
    }

    // save addr of others

    for (int dev=0; dev<num_devices; dev++) {
        if (dev == my_dev_idx) continue;

        device_t *target_dev = devices[dev];
        target_dev->idx = dev;
        target_dev->sock = NULL;

        target_dev->type = types[dev];
        target_dev->ip = ips[dev];
        target_dev->port = ports[dev];

        if (target_dev->type == SOCK_RX) {
            memset(&target_dev->server_addr,0,sizeof(target_dev->server_addr));
            target_dev->server_addr.sin_family=AF_INET;
            target_dev->server_addr.sin_addr.s_addr=inet_addr(target_dev->ip);
            target_dev->server_addr.sin_port=htons(target_dev->port);
        }

    }

}

void init_heft_scheduling(nasm_t *nasm, heft_params_t *heft_params, device_t **devices, int num_devices, int my_dev_idx) {
    
    // variables for convenience
    ninst_t *ninst_arr = nasm->ninst_arr;
    int num_tasks = nasm->num_ninst;
    device_t *my_dev = devices[my_dev_idx];
    device_t *first_rx_device;
    for (int i=0; i<num_devices; i++) {
        if (devices[i]->type == SOCK_TX) {
            first_rx_device = devices[i];
            break;
        }
    }

    // init memeories
    heft_params->entry_tasks = calloc(heft_params->num_entry_tasks, sizeof(ninst_t *));
    heft_params->exit_tasks = calloc(heft_params->num_exit_tasks, sizeof(ninst_t *));

    heft_params->data_mat = calloc(num_tasks, sizeof(int *));
    heft_params->computation_cost = calloc(num_tasks, sizeof(double *));
    heft_params->avg_computation_cost = calloc(num_tasks, sizeof(double));
    heft_params->data_transfer_rate = calloc(num_devices, sizeof(float *));
    heft_params->communication_startup = calloc(num_devices, sizeof(float));
    heft_params->avg_communication_cost = calloc(num_tasks, sizeof(float *));
    heft_params->allocation = calloc(num_tasks, sizeof(char *));
    
    for (int i=0; i<num_tasks; i++) {
        heft_params->data_mat[i] = calloc(num_tasks, sizeof(int));
        for (int j=0; j<num_tasks; j++) heft_params->data_mat[i][j] = INT_MAX;

        heft_params->computation_cost[i] = calloc(num_devices, sizeof(double));
        for (int j=0; j<num_devices; j++) heft_params->computation_cost[i][j] = DBL_MAX;
        
        heft_params->avg_computation_cost[i] = DBL_MAX;

        heft_params->avg_communication_cost[i] = calloc(num_tasks, sizeof(float));
        for (int j=0; j<num_tasks; j++) heft_params->avg_communication_cost[i][j] = FLT_MAX;
    
        heft_params->allocation[i] = calloc(num_devices, sizeof(int));
        for (int j=0; j<num_devices; j++) heft_params->allocation[i][j] = -1;
    }

    for (int i=0; i<num_devices; i++) {
        heft_params->data_transfer_rate[i] = calloc(num_devices, sizeof(float));
        for (int j=0; j<num_devices; j++) heft_params->data_transfer_rate[i][j] = FLT_EPSILON;
    
        heft_params->communication_startup[i] = FLT_MAX;
    }

    // fill params
    // fill params: entry_tasks, exit_tasks
    heft_params->num_entry_tasks = 0;
    heft_params->num_exit_tasks = 0;
    
    int count_entry_tasks = 0;
    int count_exit_tasks = 0;
    for (int i=0; i<num_tasks; i++) {
        if (ninst_arr[i].num_parent_ninsts == 0) heft_params->num_entry_tasks++;
        if (ninst_arr[i].num_child_ninsts == 0) heft_params->num_exit_tasks++;
    }
    for (int i=0; i<num_tasks; i++) {
        if (ninst_arr[i].num_parent_ninsts == 0) heft_params->entry_tasks[count_entry_tasks++] = &ninst_arr[i];
        if (ninst_arr[i].num_child_ninsts == 0) heft_params->exit_tasks[count_exit_tasks++] = &ninst_arr[i];
    }

    // fill params: devices
    heft_params->num_devices = num_devices;
    heft_params->devices = devices;
    
    // profile: data size
    for (int i=0; i<num_tasks; i++) {
        for (int j=0; j<num_tasks; j++) {
            heft_params->data_mat[i][j] = 0;
        }
    }
    
    for (int i=0; i<num_tasks; i++) {
        ninst_t *prev_ninst = &ninst_arr[i];
        for (int j=0; j<prev_ninst->num_parent_ninsts; j++) {
            const unsigned int W = prev_ninst->tile_dims[OUT_W];
            const unsigned int H = prev_ninst->tile_dims[OUT_H];
            heft_params->data_mat[i][j] = W * H * sizeof(float);
        }
    }

    // profile: computation cost

    char *buffer_send = malloc(SCHEDULE_INIT_BUF_SIZE);
    char *buffer_recv = malloc(SCHEDULE_INIT_BUF_SIZE);
    int buffer_send_written = 0;
    int num_repeat = 4;

    for (int i=0; i<num_tasks; i++) {
        heft_params->computation_cost[i][my_dev_idx] = profile_ninst_computation(&ninst_arr[i], num_repeat);
    }

    // profile: transmission
    if (my_dev == first_rx_device) {    // host - echo
        int clnt_addr_size = sizeof(my_dev->client_addr);

        for (int i=0; i<num_devices; i++) {
            
            // accept client connect request
            int client_sock = accept(my_dev->sock, (struct sockaddr*)&my_dev->client_addr, &clnt_addr_size);
            if(client_sock == -1) {
                printf("Error: accept() returned -1\n");
                assert(0);
            }
            
            // echo short message - get timestamp, send my timestamp
            for (int j=0; j<PROFILE_REPEAT; j++) {
                read(client_sock, &buffer_recv, sizeof(double));
                double recv_timestamp = get_time_secs();
                write(client_sock, &recv_timestamp, sizeof(double));
            }
            
            // echo long message - get long message with timestamp, send long message with my timestamp
            for (int j=0; j<PROFILE_REPEAT; j++) {
                read(client_sock, &buffer_recv, sizeof(PROFILE_LONG_MESSAGE_SIZE));
                *((double *)buffer_send) = get_time_secs();
                write(client_sock, &buffer_send, sizeof(PROFILE_LONG_MESSAGE_SIZE));
            }

            // host doesn't need to profile sync and transmission, others will let us know
        }
    }
    else { // non-host: request echo
        
        // connect to first_rx
        if (connect(my_dev->sock, (struct sockaddr *)&first_rx_device->server_addr, sizeof(first_rx_device->server_addr)) == -1) {
            printf("Error: connect() returned -1\n");
            assert(0);
        }

        double send_timestamps[PROFILE_REPEAT];
        double host_recv_timestamps[PROFILE_REPEAT];
        double recv_timestamps[PROFILE_REPEAT];
        double rtts[PROFILE_REPEAT];
        double syncs[PROFILE_REPEAT];
        double transfer_times[PROFILE_REPEAT];
        double rtt = 0;
        double sync = 0;
        double transfer_time = 0;

        // send short message - send timestamp, get host's timestamp
        for (int j=0; j<PROFILE_REPEAT; j++) {
            send_timestamps[j] = get_time_secs();
            write(first_rx_device->sock, &send_timestamps[j], sizeof(double));
            read(first_rx_device->sock, &host_recv_timestamps[j], sizeof(double));
            recv_timestamps[j] = get_time_secs();

            rtts[j] = recv_timestamps[j] - send_timestamps[j];
            syncs[j] = send_timestamps[j] + rtts[j]/2 - host_recv_timestamps[j];
            rtt += rtts[j];
            sync += syncs[j];
        }

        rtt /= PROFILE_REPEAT;
        sync /= PROFILE_REPEAT;

        // send long message - send long message with timestamp, get long message with timestamp
        for (int j=0; j<PROFILE_REPEAT; j++) {
            send_timestamps[j] = get_time_secs();
            write(first_rx_device->sock, &buffer_send[j], sizeof(PROFILE_LONG_MESSAGE_SIZE));
            read(first_rx_device->sock, &buffer_recv[j], sizeof(PROFILE_LONG_MESSAGE_SIZE));
            recv_timestamps[j] = get_time_secs();

            transfer_times[j] = (recv_timestamps[j] - send_timestamps[j]) / 2 - rtt / 2;
        }

        transfer_time /= PROFILE_REPEAT;

        heft_params->data_transfer_rate[my_dev_idx][first_rx_device->idx] = PROFILE_LONG_MESSAGE_SIZE / transfer_time;
        first_rx_device->sync = sync;
    }


    // sync metadata
    if (my_dev == first_rx_device) {    // host - receive from non-hosts
        int clnt_addr_size = sizeof(my_dev->client_addr);
        
        for (int i=0; i<(num_devices-1)*3; i++) {
            // will receive COMPUTE, TRANSMIT, SYNC from devices except self

            int client_sock = accept(my_dev->sock, (struct sockaddr*)&my_dev->client_addr, &clnt_addr_size);
            if(client_sock == -1) {
                printf("Error: accept() returned -1\n");
                assert(0);
            }

            // buffer structure: metadata type(4 Bytes), idx(4 Bytes), data(N Bytes)
            int recv_metadata_type, recv_idx, recv_len_data;
            read(client_sock, &recv_metadata_type, sizeof(int));
            read(client_sock, &recv_idx, sizeof(int));
            if (recv_metadata_type == HEFT_COMPUTATION_COST) {
                read(client_sock, &buffer_recv, num_tasks * sizeof(double));
                for (int i=0; i<num_tasks; i++) {
                    heft_params->computation_cost[i][recv_idx] = *(((int *)buffer_recv) + i);
                }
            }
            else if (recv_metadata_type == HEFT_TRANSMIT_RATE) {
                read(client_sock, &buffer_recv, num_tasks * sizeof(double));
                heft_params->data_transfer_rate[recv_idx][my_dev_idx] = *((int *)buffer_recv);
                heft_params->data_transfer_rate[my_dev_idx][recv_idx] = *((int *)buffer_recv);
            }
            else if (recv_metadata_type == HEFT_SYNC) {
                read(client_sock, &buffer_recv, num_tasks * sizeof(double));
                devices[recv_idx]->sync = -*((int *)buffer_recv);
            }
        }
    }
    else {   // non-host - send to host
        if (connect(my_dev->sock, (struct sockaddr *)&first_rx_device->server_addr, sizeof(first_rx_device->server_addr)) == -1) {
            printf("Error: connect() returned -1\n");
            assert(0);
        }

        // buffer structure: metadata type(4 Bytes), idx(4 Bytes), data(N Bytes)

        // send COMPUTE
        int metadata_type = HEFT_COMPUTATION_COST;
        int len_data = num_tasks * sizeof(double);

        memcpy(buffer_send + buffer_send_written, &metadata_type, sizeof(metadata_type));
        buffer_send_written += sizeof(metadata_type);
        memcpy(buffer_send + buffer_send_written, &my_dev_idx, sizeof(my_dev_idx));
        buffer_send_written += sizeof(my_dev_idx);
        for (int i=0; i<num_tasks; i++) {
            memcpy(buffer_send + buffer_send_written, &heft_params->computation_cost[i][my_dev_idx], sizeof(double));
        }
        buffer_send_written += num_tasks * sizeof(double);

        send(my_dev->sock, buffer_send, buffer_send_written, 0);

        // send TRANSMIT
        metadata_type = HEFT_TRANSMIT_RATE;
        len_data = sizeof(double);

        memcpy(buffer_send + buffer_send_written, &metadata_type, sizeof(metadata_type));
        buffer_send_written += sizeof(metadata_type);
        memcpy(buffer_send + buffer_send_written, &my_dev_idx, sizeof(my_dev_idx));
        buffer_send_written += sizeof(my_dev_idx);
        memcpy(buffer_send + buffer_send_written, &(heft_params->data_transfer_rate[my_dev_idx][first_rx_device->idx]), sizeof(double));
        buffer_send_written += sizeof(double);
        
        send(my_dev->sock, buffer_send, buffer_send_written, 0);

        // send SYNC
        metadata_type = HEFT_SYNC;
        len_data = sizeof(double);

        memcpy(buffer_send + buffer_send_written, &metadata_type, sizeof(metadata_type));
        buffer_send_written += sizeof(metadata_type);
        memcpy(buffer_send + buffer_send_written, &my_dev_idx, sizeof(my_dev_idx));
        buffer_send_written += sizeof(my_dev_idx);
        memcpy(buffer_send + buffer_send_written, &(first_rx_device->sync), sizeof(double));
        buffer_send_written += sizeof(double);
        
        send(my_dev->sock, buffer_send, buffer_send_written, 0);

    }

    // do scheduling
    if (my_dev == first_rx_device) {

    }

    // share scheduling
    if (my_dev == first_rx_device) {
        // send to non-hosts

    }
    else {
        // receive from host

    }
    
    // finish scheduling

    close(my_dev->sock);


}

void init_cpop_scheduling(nasm_t *nasm, cpop_params_t *cpop_params, device_t **devices, int num_devices, int my_dev_idx) {

}

double profile_ninst_computation(ninst_t *ninst, int num_repeat) {
    dse_t *dummy_dse = calloc(1, sizeof(dse_t));
    dse_init(dummy_dse, -1);

    void (*comp_func)(ninst_t *, dse_t *);

    switch (ninst->ldata->layer->type)
    {
        case CONV_LAYER:
            comp_func = tiled_conv2d;
            break;
        case MAXPOOL_LAYER:
            comp_func = tiled_maxpool2d;
            break;
        case AVGPOOL_LAYER:
            comp_func = tiled_avgpool2d;
            break;
        case FC_LAYER:
            comp_func = tiled_fully_connected;
            break;
        case RESIDUAL_LAYER:
            comp_func = tiled_residual;
            break;
        case SOFTMAX_LAYER:
            comp_func = tiled_softmax;
            break;
        case YOLO_LAYER:
            comp_func = tiled_yolo;
            break;
        case APPEND_LAYER:
            comp_func = tiled_append;
            break;
        case MATMUL_LAYER:
            comp_func = tiled_matmul;
            break;
        case LAYERNORM_LAYER:
            comp_func = tiled_layernorm;
            break;
        case K_ATTENTION_LAYER:
            comp_func = tiled_k_attention;
            break;
        case V_ATTENTION_LAYER:
            comp_func = tiled_v_attention;
            break;
        default:
            // FPRT (stderr, "ERROR: dummy_dse_thread_runtime: layer type %s is not supported\n", layer_type_str[ninst->ldata->layer->type]);
            return 0;
            break;
    }

    double start_time = get_time_secs(), end_time;

    for (int i = 0; i < num_repeat; i++) {
        comp_func (ninst, dummy_dse);
    }

    end_time = get_time_secs();

    return end_time - start_time;
}
