#include "networking.h"

void *net_tx_thread_runtime (void* thread_info) 
{
    networking_engine *net_engine = (networking_engine*) thread_info;
    pthread_mutex_lock (&net_engine->tx_thread_mutex);
    pthread_cond_wait (&net_engine->tx_thread_cond, &net_engine->tx_thread_mutex);
    while (!net_engine->tx_kill)
    {
        transmission(net_engine);
        if(!net_engine->tx_run)
            pthread_cond_wait (&net_engine->tx_thread_cond, &net_engine->tx_thread_mutex);
    }
    return NULL;
}

void *net_rx_thread_runtime (void* thread_info) 
{
    networking_engine *net_engine = (networking_engine*) thread_info;
    pthread_mutex_lock (&net_engine->rx_thread_mutex);
    pthread_cond_wait (&net_engine->rx_thread_cond, &net_engine->rx_thread_mutex);
    while (!net_engine->rx_kill)
    {
        receive(net_engine);
        if(!net_engine->rx_run) 
            pthread_cond_wait (&net_engine->rx_thread_cond, &net_engine->rx_thread_mutex);
    }
    return NULL;
}

void init_networking_queue (networking_queue_t *networking_queue)
{
    networking_queue->idx_start = 0;
    networking_queue->idx_end = 0;
    networking_queue->num_stored = 0;
    networking_queue->max_stored = NET_INIT_QUEUE_SIZE;
    
    networking_queue->ninst_ptr_arr = (ninst_t**) calloc (NET_INIT_QUEUE_SIZE, sizeof(ninst_t*));

    networking_queue->queue_cond = (pthread_cond_t) PTHREAD_COND_INITIALIZER;
    networking_queue->queue_mutex = (pthread_mutex_t) PTHREAD_MUTEX_INITIALIZER;
}

void init_server(networking_engine* net_engine, int port, int is_UDP) 
{
    if (is_UDP != 0) 
    {
        net_engine->listen_sock = socket (PF_INET, SOCK_DGRAM, 0);
    }
    else 
    {
        net_engine->listen_sock = socket (PF_INET, SOCK_STREAM, 0);
    }
    if (net_engine->listen_sock == -1) 
    {
        FPRT (stderr, "Error: socket() returned -1\n");
        assert(0);
    }

    bzero (&net_engine->listen_addr, sizeof(net_engine->listen_addr));
    bzero (&net_engine->server_addr, sizeof(net_engine->server_addr));
    bzero (&net_engine->edge_addr, sizeof(net_engine->edge_addr));

    net_engine->device_mode = DEV_SERVER;
    net_engine->comm_sock = 0;
    net_engine->listen_addr.sin_family = AF_INET;
    net_engine->listen_addr.sin_addr.s_addr = htonl (INADDR_ANY);
    net_engine->listen_addr.sin_port = htons (port);
    net_engine->isUDP = is_UDP;

    int option = 1;
    if (setsockopt(net_engine->listen_sock, SOL_SOCKET, SO_REUSEADDR, &option, sizeof(option)) == -1)
    {
        FPRT (stderr, "ERROR! socket setsockopt error\n");
        assert(0);
    }

    if(bind(net_engine->listen_sock, (struct sockaddr*)&net_engine->listen_addr, sizeof(net_engine->listen_addr)) == -1)
    {
        FPRT (stderr, "ERROR! socket bind error\n");
        assert(0);
    }
    if(is_UDP == 0)
    {
        if(listen(net_engine->listen_sock, 5) == -1)
        {
            FPRT (stderr, "ERROR! socket listen error\n");
            assert(0);
        }
        else 
            PRT ("Networking: TCP Server listening on port %d\n", port);
        socklen_t edge_addr_len = sizeof(net_engine->edge_addr);
        net_engine->comm_sock = 
            accept(net_engine->listen_sock, (struct sockaddr*)&net_engine->edge_addr, &edge_addr_len);
        if (net_engine->comm_sock == -1)
        {
            FPRT (stderr, "ERROR! Server socket accept error\n");
            assert(0);
        }
        net_engine->edge_addr.sin_family = AF_INET;
        net_engine->edge_addr.sin_addr.s_addr = inet_addr (inet_ntoa (net_engine->edge_addr.sin_addr));
        net_engine->edge_addr.sin_port = htons (port);
        PRT ("Networking: TCP Server accepted connection from %s\n", inet_ntoa (net_engine->edge_addr.sin_addr));
        char* message = "SERVER ACK";
        write_n(net_engine->comm_sock, message, 10);
        char buf [8] = {0};
        read_n(net_engine->comm_sock, &buf, 8);
        PRT ("Networking: TCP Server received %s\n", buf);
    }
    else 
        PRT ("Networking: UDP Server listening on port %d\n", port);
}

void init_edge(networking_engine* net_engine, char* ip, int port, int is_UDP) 
{
    bzero (&net_engine->listen_addr, sizeof(net_engine->listen_addr));
    bzero (&net_engine->server_addr, sizeof(net_engine->server_addr));
    bzero (&net_engine->edge_addr, sizeof(net_engine->edge_addr));

    net_engine->device_mode = DEV_EDGE;
    net_engine->comm_sock = socket (PF_INET, SOCK_STREAM, 0);
    if (net_engine->comm_sock == -1) 
    { 
        FPRT (stderr, "Error: Edge socket() returned -1\n");
        assert(0);
    }
    net_engine->server_addr.sin_family = AF_INET;
    net_engine->server_addr.sin_addr.s_addr = inet_addr (ip);
    net_engine->server_addr.sin_port = htons (port);
    net_engine->isUDP = 0;
    PRT ("Trying to access server...\n");
    int conn = -1;
    while(conn < 0)
    {
        conn = connect(net_engine->comm_sock, (struct sockaddr*)&net_engine->server_addr, sizeof(net_engine->server_addr));
    }
    PRT ("Networking: TCP Edge connected to server %s port %d\n", inet_ntoa (net_engine->server_addr.sin_addr), port);
    char buf [10];
    read_n(net_engine->comm_sock, &buf, 10);
    PRT ("Networking: TCP Edge received %s\n", buf);
    char* message = "EDGE ACK";
    write_n(net_engine->comm_sock, message, 8);
}


networking_engine* init_networking (nasm_t* nasm, rpool_t* rpool, DEVICE_MODE device_mode, char* ip, int port, int is_UDP, int pipelined) 
{
    PRT("Initializing Networking Engine...\n");
    networking_engine *net_engine = calloc (1, sizeof(networking_engine));
    networking_queue_t *networking_queue = calloc (1, sizeof(networking_queue_t));
    init_networking_queue(networking_queue);

    net_engine->tx_queue = networking_queue;
    atomic_store (&net_engine->tx_run, 0);
    atomic_store (&net_engine->rx_run, 0);
    atomic_store (&net_engine->tx_kill, 0);
    atomic_store (&net_engine->rx_kill, 0);


    net_engine->nasm = nasm;
    net_engine->rpool = rpool;
    net_engine->pipelined = pipelined;
    for (int i=0; i<SCHEDULE_MAX_DEVICES; i++) {
        net_engine->inference_whitelist[i] = -1;
    }

    switch (device_mode)
    {
    case DEV_EDGE:
        init_edge(net_engine, ip, port, is_UDP);
        break;
    case DEV_SERVER:
        init_server(net_engine, port, is_UDP);
        break;
    default:
        FPRT (stderr, "Error - Unsupported socket type. Type: %d\n", net_engine->device_mode);
        assert(0);
        break;
    }

    net_engine->tx_buffer = calloc(NETQUEUE_BUFFER_SIZE, sizeof(char));
    net_engine->rx_buffer = calloc(NETQUEUE_BUFFER_SIZE, sizeof(char));

    char info_str[MAX_STRING_LEN*2];
    void *whitelist[NUM_RPOOL_CONDS] = {NULL};
    whitelist [RPOOL_NASM] = nasm;
    sprintf (info_str, "%s_%s_%d", nasm->dnn->name, "nasm", nasm->nasm_id);
    float queue_per_layer = rpool->ref_dses * NUM_LAYERQUEUE_PER_ASE * NUM_QUEUE_PER_LAYER;
    unsigned int num_queues = nasm->dnn->num_layers*queue_per_layer;
    if (num_queues < 1)
        num_queues = 1;
    nasm->gpu_idx = rpool->gpu_idx;
    rpool_add_queue_group (rpool, info_str, num_queues, NULL, whitelist);

    size_t total_mem_req = 0;
    for (int i = 0; i < nasm->num_ldata; i++)
    {
        nasm_ldata_t *ldata = &nasm->ldata_arr[i];
        total_mem_req += ldata->out_mat_mem_size;
    }
    if (nasm->data == NULL)
    {
        nasm->data = aspen_calloc (total_mem_req, 1);
        if (nasm->data == NULL)
        {
            FPRT (stderr, "Error: nasm->data == NULL\n");
            assert (0);
        }
        for (int i = 0; i < nasm->num_ldata; i++)
        {
            nasm_ldata_t *ldata = &nasm->ldata_arr[i];
            set_ldata_out_mat_mem_pos (ldata);
        }
    }
    

    net_engine->rx_thread_cond = (pthread_cond_t) PTHREAD_COND_INITIALIZER;
    net_engine->rx_thread_mutex = (pthread_mutex_t) PTHREAD_MUTEX_INITIALIZER;
    net_engine->tx_thread_cond = (pthread_cond_t) PTHREAD_COND_INITIALIZER;
    net_engine->tx_thread_mutex = (pthread_mutex_t) PTHREAD_MUTEX_INITIALIZER;
    pthread_create (&net_engine->tx_thread, NULL, net_tx_thread_runtime, (void*)net_engine);
    pthread_create (&net_engine->rx_thread, NULL, net_rx_thread_runtime, (void*)net_engine);
    return net_engine;
}

void transmission(networking_engine *net_engine) 
{
    ninst_t *target_ninst_list[32];
    unsigned int num_ninsts = 0;
    int32_t payload_size = 0;
    char* buffer_ptr = (char*)net_engine->tx_buffer + sizeof(int32_t);

    if(!net_engine->pipelined && net_engine->device_mode == DEV_SERVER)
    {
        if(!atomic_load(&net_engine->nasm->completed))
            return;
    }

    pthread_mutex_lock(&net_engine->tx_queue->queue_mutex);
    num_ninsts = pop_ninsts_from_net_queue(net_engine->tx_queue, target_ninst_list, 4);
    pthread_mutex_unlock(&net_engine->tx_queue->queue_mutex);
    if (num_ninsts == 0)
        return;
    double time_sent = get_time_secs_offset();
    for (int i = 0; i < num_ninsts; i++)
    {
        ninst_t* target_ninst = target_ninst_list[i];
        *(unsigned int*)buffer_ptr = target_ninst_list[i]->ninst_idx;
        buffer_ptr += sizeof(unsigned int);
        *(int*)buffer_ptr = target_ninst_list[i]->ldata->nasm->inference_id;
        buffer_ptr += sizeof(int);
        unsigned int data_size = target_ninst->tile_dims[OUT_W]*target_ninst->tile_dims[OUT_H]*sizeof(float);
        *(unsigned int*)buffer_ptr = data_size;
        buffer_ptr += sizeof(unsigned int);
        memcpy(buffer_ptr, target_ninst->network_buf, data_size);
        free (target_ninst->network_buf);
        target_ninst->network_buf = NULL;
        target_ninst->sent_time = time_sent;
        buffer_ptr += data_size;
        // PRT("Networking: Ninst%d, Sending %d bytes, W%d, H%d, data size %d\n", i, data_size + 3*sizeof(int), 
        //     target_ninst_list[i]->tile_dims[OUT_W], target_ninst_list[i]->tile_dims[OUT_H], 
        //     target_ninst_list[i]->tile_dims[OUT_W]*target_ninst_list[i]->tile_dims[OUT_H]*sizeof(float));
    }
    payload_size = buffer_ptr - (char*)net_engine->tx_buffer - sizeof(int32_t);
    *(int32_t*)net_engine->tx_buffer = payload_size;
    payload_size += sizeof(int32_t);

    #ifdef DEBUG
    PRT("Networking: Sending %d bytes -", payload_size);
    for (int i = 0; i < num_ninsts; i++)
    {
        ninst_t* target_ninst = target_ninst_list[i];
        PRT(" (N%d L%d I%d %ldB)", 
            target_ninst->ninst_idx, target_ninst->ldata->layer->layer_idx, target_ninst->ldata->nasm->inference_id, 
            target_ninst->tile_dims[OUT_W]*target_ninst->tile_dims[OUT_H]*sizeof(float));
    }
    #endif
    int32_t bytes_sent = 0;
    while (bytes_sent < payload_size)
    {
        int ret = write(net_engine->comm_sock
            , (char*)net_engine->tx_buffer + bytes_sent, payload_size - bytes_sent);
        if (ret < 0)
        {
            FPRT(stderr, "Error: send() failed. ret: %d\n", ret);
            assert(0);
        }
        bytes_sent += ret;
    }
    #ifdef DEBUG
    PRT(" - Time taken %fs, %d tx queue remains.\n", (get_time_secs() - time_sent), net_engine->tx_queue->num_stored);
    #endif
}

void receive(networking_engine *net_engine) 
{
    
    int32_t payload_size;
    int ret = read(net_engine->comm_sock, (char*)&payload_size, sizeof(int32_t));
    if (ret == -1)
    {
        return;
    }
    else if (ret < 0)
    {
        FPRT(stderr, "Error: recv() failed. ret: %d\n", ret);
        assert(0);
    }
    else if (ret == 0)
    {
        FPRT(stderr, "Error: RX Connection closed unexpectedly.\n");
        assert(0);
    }
    else
    {
        if (payload_size <= 0)
        {
            // PRT("Networking: RX Command %d received - ", payload_size);
            if (payload_size == RX_STOP_SIGNAL)
            {
                PRT("RX stop signal received.\n");
                atomic_store (&net_engine->rx_run, 0);
                atomic_store (&net_engine->nasm->completed, 1);
                pthread_mutex_lock (&net_engine->nasm->nasm_mutex);
                pthread_cond_signal (&net_engine->nasm->nasm_cond);
                pthread_mutex_unlock (&net_engine->nasm->nasm_mutex);
                return;
            }
            else
            {
                PRT("Unknown command\n");
                assert(0);
            }
        }
        int32_t bytes_received = 0;
        while (bytes_received < payload_size)
        {
            ret = read(net_engine->comm_sock
                , (char*)net_engine->rx_buffer + bytes_received, payload_size - bytes_received);
            if (ret < 0)
            {
                FPRT(stderr, "Error: recv() failed. ret: %d\n", ret);
                assert(0);
            }
            bytes_received += ret;
        }
        #ifdef DEBUG
        // PRT("Networking: Received %d bytes -", bytes_received);
        #endif
        char* buffer_ptr = (char*)net_engine->rx_buffer;
        while (buffer_ptr < (char*)net_engine->rx_buffer + bytes_received)
        {
            unsigned int ninst_idx = *(unsigned int*)buffer_ptr;
            buffer_ptr += sizeof(unsigned int);
            int inference_id = *(int*)buffer_ptr;
            buffer_ptr += sizeof(int);
            unsigned int data_size = *(unsigned int*)buffer_ptr;
            buffer_ptr += sizeof(unsigned int);
            #ifdef DEBUG
            if (net_engine->nasm->inference_id != inference_id)
            {
                PRT("Warning: Received inference_id %d is not matched with ninst_idx %d\n", inference_id, ninst_idx);
                continue;
            }
            if (ninst_idx >= net_engine->nasm->num_ninst)
            {
                PRT("Warning: Received ninst_idx %d is not found in nasm\n", ninst_idx);
                continue;
            }
            #endif
            ninst_t* target_ninst = &net_engine->nasm->ninst_arr[ninst_idx];
            if (!is_inference_whitelist(net_engine, inference_id))
                return;
            if (atomic_exchange (&target_ninst->state, NINST_COMPLETED) == NINST_COMPLETED) 
                return;
                
            #ifdef DEBUG
            if (data_size != target_ninst->tile_dims[OUT_W]*target_ninst->tile_dims[OUT_H]*sizeof(float))
            {
                FPRT(stderr, "Error: Received data size %d is not matched with ninst_idx %d\n", data_size, ninst_idx);
                assert(0);
            }
            #endif
            copy_buffer_to_ninst_data (target_ninst, buffer_ptr);
            buffer_ptr += data_size;
            atomic_store(&target_ninst->state, NINST_COMPLETED);
            target_ninst->received_time = get_time_secs_offset ();
            #ifdef DEBUG
            PRT("\t[Device %d] (N%d L%d I%d %ldB) state: %d",
                net_engine->device_idx,
                target_ninst->ninst_idx, target_ninst->ldata->layer->layer_idx, target_ninst->ldata->nasm->inference_id, 
                target_ninst->tile_dims[OUT_W]*target_ninst->tile_dims[OUT_H]*sizeof(float), atomic_load(&target_ninst->state));
            #endif
            
            unsigned int num_ninst_completed = atomic_fetch_add (&target_ninst->ldata->num_ninst_completed , 1);
            if(net_engine->device_mode == DEV_SERVER)
            {
                for(int i = 0; i < target_ninst->num_child_ninsts; i++)
                {
                    ninst_t* child_ninst = target_ninst->child_ninst_arr[i];
                    ninst_set_compute_device(child_ninst, net_engine->server_idx);
                }
                for(int i = 0; i < target_ninst->num_parent_ninsts; i++)
                {
                    int parent_ninst_idx = target_ninst->parent_ninst_idx_arr[i];
                    ninst_t* parent_ninst = &net_engine->nasm->ninst_arr[parent_ninst_idx];
                    if(atomic_load (&parent_ninst->state) != NINST_COMPLETED)
                        continue;
                    // printf("\t(N%d L%d) Parent: (N%d L%d) Dev_to_compute[%d]: %d\n",target_ninst->ninst_idx, target_ninst->ldata->layer->layer_idx, parent_ninst_idx, parent_ninst->ldata->layer->layer_idx, net_engine->server_idx, parent_ninst->dev_to_compute[net_engine->server_idx]);
                    for(int j = 0; j < parent_ninst->num_child_ninsts; j++)
                    {
                        ninst_t* parent_child_ninst = parent_ninst->child_ninst_arr[j];
                        ninst_set_compute_device(parent_child_ninst, net_engine->server_idx);
                        // int temp = parent_child_ninst->dev_to_compute[net_engine->server_idx];
                        // printf("\t\tParent Child: (N%d L%d) Dev_to_compute[%d]: %d -> %d\n",parent_child_ninst->ninst_idx, parent_child_ninst->ldata->layer->layer_idx, net_engine->server_idx, temp, parent_child_ninst->dev_to_compute[net_engine->server_idx]);
                    }
                    
                    update_children (net_engine->rpool, parent_ninst);
                }
            }

            if(atomic_load(&target_ninst->state) == NINST_COMPLETED)
                update_children (net_engine->rpool, target_ninst);
            
            
            if (num_ninst_completed == target_ninst->ldata->num_ninst - 1)
            {
                // printf ("\t\tNet engine completed layer %d of nasm %d\n", 
                //     target_ninst->ldata->layer->layer_idx, target_ninst->ldata->nasm->nasm_id);
                for (int pidx = 0; pidx < NUM_PARENT_ELEMENTS; pidx++)
                {
                    if (target_ninst->ldata->parent_ldata_idx_arr[pidx] == -1)
                        continue;
                    nasm_ldata_t *parent_ldata = &target_ninst->ldata->nasm->ldata_arr[target_ninst->ldata->parent_ldata_idx_arr[pidx]];
                    unsigned int num_child_ldata_completed = atomic_fetch_add (&parent_ldata->num_child_ldata_completed, 1);
                    if (num_child_ldata_completed == parent_ldata->num_child_ldata && (parent_ldata != parent_ldata->nasm->ldata_arr))
                        free_ldata_out_mat (parent_ldata);
                }

                atomic_fetch_add (&net_engine->nasm->num_ldata_completed, 1);
                // If the last layer of the nasm is completed, signal the nasm thread.
                if(target_ninst->ldata->layer->layer_idx == net_engine->nasm->num_ldata - 1) 
                {
                    atomic_store(&net_engine->nasm->num_ldata_completed, net_engine->nasm->num_ldata);
                    pthread_mutex_lock (&net_engine->nasm->nasm_mutex);
                    pthread_cond_signal (&net_engine->nasm->nasm_cond);
                    pthread_mutex_unlock (&net_engine->nasm->nasm_mutex);
                }
                if(!net_engine->pipelined)
                {
                    // Run SERVER DSEs when all ninst of a layer are downloaded. (Conventional mode)
                    dse_group_set_enable_device(net_engine->dse_group, net_engine->device_idx, 1);
                    dse_group_add_prioritize_rpool(net_engine->dse_group, net_engine->device_idx);
                    dse_group_run(net_engine->dse_group);
                } 
            }
        }
        #ifdef DEBUG
            PRT("\n");
        #endif
    }
}

void net_queue_reset (networking_queue_t *networking_queue)
{
    unsigned int queue_idx = networking_queue->idx_start;
    while (queue_idx != networking_queue->idx_end)
    {
        if (networking_queue->ninst_ptr_arr [queue_idx] != NULL)
        {
            ninst_t *target_ninst = networking_queue->ninst_ptr_arr [queue_idx];
            if (target_ninst->network_buf)
            {
                free (target_ninst->network_buf);
                target_ninst->network_buf = NULL;
            }
        }
        queue_idx++;
        if (queue_idx >= networking_queue->max_stored)
            queue_idx = 0;
    }
    networking_queue->idx_start = 0;
    networking_queue->idx_end = 0;
    networking_queue->num_stored = 0;
}

void net_engine_reset (networking_engine *net_engine)
{
    net_queue_reset(net_engine->tx_queue);
}

void net_engine_run (networking_engine *net_engine)
{
    if (net_engine == NULL)
    {
        FPRT (stderr, "ERROR: net_engine_run: net_engine is NULL\n");
        return;
    }
    unsigned int state = atomic_exchange (&net_engine->rx_run, 1);
    if (state != 1)
    {
        pthread_cond_signal (&net_engine->rx_thread_cond);
        pthread_mutex_unlock (&net_engine->rx_thread_mutex);
    }
    state = atomic_exchange (&net_engine->tx_run, 1);
    if (state != 1)
    {
        pthread_cond_signal (&net_engine->tx_thread_cond);
        pthread_mutex_unlock (&net_engine->tx_thread_mutex);
    }
}


void net_queue_destroy(networking_queue_t* net_queue)
{
    if(net_queue == NULL) return;
    net_queue_reset (net_queue);
    if(net_queue->ninst_ptr_arr != NULL) 
        free(net_queue->ninst_ptr_arr);
    pthread_mutex_destroy(&net_queue->queue_mutex);
    pthread_cond_destroy(&net_queue->queue_cond);
}

void net_engine_stop (networking_engine* net_engine)
{
    if (net_engine == NULL)
    {
        FPRT (stderr, "ERROR: net_engine_stop: net_engine is NULL\n");
        return;
    }
    #ifdef DEBUG
    PRT ("Networking: Stopping network engine tx thread...\n");
    #endif
    int is_running = atomic_exchange (&net_engine->tx_run, 0);
    if (is_running == 0)
        return;
    pthread_mutex_lock (&net_engine->tx_thread_mutex);
    #ifdef DEBUG
    PRT("Networking: Sending network engine rx thread stop signal...\n");
    #endif
    int32_t command = RX_STOP_SIGNAL;
    int ret = write(net_engine->comm_sock, (char*)&command, sizeof(command));
    if (ret < 0)
    {
        FPRT(stderr, "ERROR: net_engine_sto: send() failed.\n");
        assert(0);
    }
    #ifdef DEBUG
    PRT("Networking: Waiting for network engine rx thread to stop...\n");
    #endif
    pthread_mutex_lock (&net_engine->rx_thread_mutex);
    #ifdef DEBUG
    PRT("Networking: Network engine stopped.\n");
    #endif
}

void net_engine_destroy (networking_engine* net_engine)
{
    if(net_engine == NULL) 
        return;
    
    if (atomic_load(&net_engine->tx_run) == 1 && atomic_load(&net_engine->rx_run))
    {
        FPRT (stderr, "WARNING: net_engine_destroy: net_engine is still running\n");
        net_engine_stop(net_engine);
    }
    // Thread cleanup
    atomic_store (&net_engine->tx_kill, 1);
    atomic_store (&net_engine->rx_kill, 1);
    atomic_store (&net_engine->tx_run, 1);
    atomic_store (&net_engine->rx_run, 1);
    pthread_cond_signal (&net_engine->tx_thread_cond);
    pthread_cond_signal (&net_engine->rx_thread_cond);
    pthread_mutex_unlock (&net_engine->tx_thread_mutex);
    pthread_mutex_unlock (&net_engine->rx_thread_mutex);
    pthread_join (net_engine->tx_thread, NULL);
    pthread_join (net_engine->rx_thread, NULL);

    net_queue_destroy(net_engine->tx_queue);
    free (net_engine->tx_queue);
    if (net_engine->is_listen_sock_open)
        close(net_engine->listen_sock);
    if (net_engine->is_comm_sock_open)
        close(net_engine->comm_sock);
    pthread_mutex_destroy(&net_engine->rx_thread_mutex);
    pthread_mutex_destroy(&net_engine->tx_thread_mutex);
    pthread_cond_destroy(&net_engine->tx_thread_cond);
    pthread_cond_destroy(&net_engine->rx_thread_cond);
    if (net_engine->rx_buffer)
        free(net_engine->rx_buffer);
    if (net_engine->tx_buffer)
        free(net_engine->tx_buffer);
    free(net_engine);
}

void net_engine_wait_for_tx_queue_completion (networking_engine *net_engine)
{
    pthread_mutex_lock (&net_engine->tx_queue->queue_mutex);
    if (net_engine->tx_queue->num_stored > 0)
    {
        pthread_cond_wait (&net_engine->tx_queue->queue_cond, &net_engine->tx_queue->queue_mutex);
    }
    pthread_mutex_unlock (&net_engine->tx_queue->queue_mutex);
}

void add_inference_whitelist (networking_engine *net_engine, int inference_id) 
{
    for (int i=0; i<SCHEDULE_MAX_DEVICES; i++) 
    {
        if (net_engine->inference_whitelist[i] == -1) 
        {
            net_engine->inference_whitelist[i] = inference_id;
            return;
        }
    }
}

void remove_inference_whitelist (networking_engine *net_engine, int inference_id) 
{
    for (int i=0; i<SCHEDULE_MAX_DEVICES; i++) 
    {
        if (net_engine->inference_whitelist[i] == inference_id) 
        {
            net_engine->inference_whitelist[i] = -1;
            return;
        }
    }
}

int is_inference_whitelist (networking_engine *net_engine, int inference_id) 
{
    for (int i=0; i<SCHEDULE_MAX_DEVICES; i++) 
    {
        if (net_engine->inference_whitelist[i] == inference_id) 
        {
            return 1;
        }
    }
    return 0;
}

void create_network_buffer_for_ninst (ninst_t *target_ninst)
{
    const unsigned int W = target_ninst->tile_dims[OUT_W];
    const unsigned int H = target_ninst->tile_dims[OUT_H];
    if (target_ninst->network_buf)
        free (target_ninst->network_buf);
    target_ninst->network_buf = calloc (W * H, sizeof (float));
    copy_ninst_data_to_buffer (target_ninst, target_ninst->network_buf);
}

void check_and_update_net_queue_size (networking_queue_t *networking_queue, unsigned int num_to_add)
{
    if (networking_queue->num_stored + num_to_add > networking_queue->max_stored)
    {
        ninst_t **new_ninst_ptr_arr = (ninst_t **) calloc 
            (networking_queue->max_stored*2, sizeof (ninst_t *));
        if (networking_queue->idx_start < networking_queue->idx_end)
        {
            memcpy (new_ninst_ptr_arr, networking_queue->ninst_ptr_arr + networking_queue->idx_start, 
                    (networking_queue->idx_end - networking_queue->idx_start)*sizeof (ninst_t *));
        }
        else
        {
            memcpy (new_ninst_ptr_arr, networking_queue->ninst_ptr_arr + networking_queue->idx_start, 
                    (networking_queue->max_stored - networking_queue->idx_start)*sizeof (ninst_t *));
            memcpy (new_ninst_ptr_arr + networking_queue->max_stored - networking_queue->idx_start, 
                    networking_queue->ninst_ptr_arr, networking_queue->idx_end*sizeof (ninst_t *));
        }
        free (networking_queue->ninst_ptr_arr);
        networking_queue->ninst_ptr_arr = new_ninst_ptr_arr;
        networking_queue->idx_start = 0;
        networking_queue->idx_end = networking_queue->num_stored;
        networking_queue->max_stored *= 2;
    }
}

void update_net_queue_size (networking_queue_t *networking_queue, unsigned int num_to_add)
{
    ninst_t **new_ninst_ptr_arr = (ninst_t **) calloc 
        (networking_queue->max_stored*2, sizeof (ninst_t *));
    if (networking_queue->idx_start < networking_queue->idx_end)
    {
        memcpy (new_ninst_ptr_arr, networking_queue->ninst_ptr_arr + networking_queue->idx_start, 
                (networking_queue->idx_end - networking_queue->idx_start)*sizeof (ninst_t *));
    }
    else
    {
        memcpy (new_ninst_ptr_arr, networking_queue->ninst_ptr_arr + networking_queue->idx_start, 
                (networking_queue->max_stored - networking_queue->idx_start)*sizeof (ninst_t *));
        memcpy (new_ninst_ptr_arr + networking_queue->max_stored - networking_queue->idx_start, 
                networking_queue->ninst_ptr_arr, networking_queue->idx_end*sizeof (ninst_t *));
    }
    free (networking_queue->ninst_ptr_arr);
    networking_queue->ninst_ptr_arr = new_ninst_ptr_arr;
    networking_queue->idx_start = 0;
    networking_queue->idx_end = networking_queue->num_stored;
    networking_queue->max_stored *= 2;
}

unsigned int pop_ninsts_from_net_queue (networking_queue_t *networking_queue, ninst_t **ninst_ptr_list, unsigned int max_ninsts_to_get)
{
    #ifdef DEBUG
    if (networking_queue == NULL)
    {
        FPRT (stderr, "ERROR: pop_nists_from_queue: networking_queue is NULL.\n");
        return 0;
    }
    if (ninst_ptr_list == NULL)
    {
        FPRT (stderr, "ERROR: pop_nists_from_queue: ninst_ptr_list is NULL.\n");
        return 0;
    }
    #endif
    unsigned int num_ninsts = 0;
    unsigned int i = networking_queue->idx_start;
    unsigned int buffer_usage = 0;
    
    if(networking_queue->num_stored > 0) 
    {
        for (; num_ninsts < networking_queue->num_stored; num_ninsts++)
        {
            if (num_ninsts >= max_ninsts_to_get)
                break;
            ninst_ptr_list[num_ninsts] = networking_queue->ninst_ptr_arr[i];
            const unsigned int W = networking_queue->ninst_ptr_arr[i]->tile_dims[OUT_W];
            const unsigned int H = networking_queue->ninst_ptr_arr[i]->tile_dims[OUT_H];
            if (buffer_usage + W * H * sizeof(float) + sizeof (unsigned int) * 2 + sizeof (int) > NETQUEUE_BUFFER_SIZE)
                break;
            buffer_usage += W * H * sizeof(float) + sizeof (unsigned int) * 2 + sizeof (int);
            i++;
            if (i == networking_queue->max_stored)
                i = 0;
        }
        networking_queue->idx_start = i;
        networking_queue->num_stored -= num_ninsts;
        if (networking_queue->num_stored == 0)
        {
            pthread_cond_signal (&networking_queue->queue_cond);
        }
    }
    return num_ninsts;
}

void push_ninsts_to_net_queue (networking_queue_t *networking_queue, ninst_t **ninst_ptr_list, unsigned int num_ninsts)
{
    #ifdef DEBUG
    if (networking_queue == NULL)
    {
        FPRT (stderr, "ERROR: push_ninsts_to_net_queue_back: networking_queue is NULL.\n");
        return;
    }
    if (ninst_ptr_list == NULL)
    {
        FPRT (stderr, "ERROR: push_ninsts_to_net_queue_back: ninst_ptr_list is NULL.\n");
        return;
    }
    #endif

    if (networking_queue->num_stored + num_ninsts > networking_queue->max_stored)
        update_net_queue_size (networking_queue, num_ninsts);
    unsigned i = networking_queue->idx_end;

    for (int j = 0; j < num_ninsts; j++)
    {
        ninst_t *ninst_ptr = ninst_ptr_list[j];
        *(networking_queue->ninst_ptr_arr + i) = ninst_ptr;
        i++;
        if (i == networking_queue->max_stored)
            i = 0;
    }

    networking_queue->idx_end = i;
    networking_queue->num_stored += num_ninsts;
}

void push_ninsts_to_net_queue_front (networking_queue_t *networking_queue, ninst_t **ninst_ptr_list, unsigned int num_ninsts)
{
    #ifdef DEBUG
    if (networking_queue == NULL)
    {
        FPRT (stderr, "ERROR: push_ninsts_to_net_queue_back: networking_queue is NULL.\n");
        return;
    }
    if (ninst_ptr_list == NULL)
    {
        FPRT (stderr, "ERROR: push_ninsts_to_net_queue_back: ninst_ptr_list is NULL.\n");
        return;
    }
    #endif
    check_and_update_net_queue_size (networking_queue, num_ninsts);
    unsigned int i = networking_queue->idx_start;
    for (int j = num_ninsts - 1; j >= 0; j--)
    {
        i--;
        if (i == -1)
            i = networking_queue->max_stored - 1;
        networking_queue->ninst_ptr_arr[i] = ninst_ptr_list[j];
    }
    networking_queue->idx_start = i;
    networking_queue->num_stored += num_ninsts;
    // if (networking_queue->queue_group != NULL)
    //     atomic_fetch_add (&networking_queue->queue_group->num_ninsts, num_ninsts);
}

void add_input_rpool (networking_engine *net_engine, nasm_t* nasm, char *input_filename)
{
    aspen_dnn_t *dnn = nasm->dnn;
    aspen_layer_t *first_layer = &dnn->layers[0];
    void *data = NULL;
    unsigned int input_params[NUM_PARAM_ELEMENTS] = {0};
    input_params[BATCH] = nasm->batch_size;
    if (first_layer->params[OUT_C] != 0 && first_layer->params[OUT_H] != 0 && first_layer->params[OUT_W] != 0)
    {
        input_params[OUT_C] = first_layer->params[OUT_C];
        input_params[OUT_H] = first_layer->params[OUT_H];
        input_params[OUT_W] = first_layer->params[OUT_W];
        data = aspen_load_input_NHWC (input_filename, input_params, sizeof(float));
    }
    else if (first_layer->params[MAT_M] != 0)
    {
        input_params[MAT_M] = first_layer->params[MAT_M];
        input_params[MAT_N] = nasm->tr_seq_len;
        data = aspen_load_input (input_filename, input_params, sizeof(float));
    }
    else
    {
        FPRT (stderr, "ERROR: rpool_add_nasm: first layer of dnn \"%s\" does not have output dimensions. Cannot add nasm \"%s_nasm_%d\".\n", 
            dnn->name, dnn->name, nasm->nasm_id);
        return;
    }

    if (data != NULL)
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
                (char*)data + i * ldata->out_mat_dims[OUT_H] * nasm->dnn->element_size, 
                ldata->out_mat_dims[OUT_H] * nasm->dnn->element_size);
    }
    if (nasm->data == NULL)
    {
        FPRT (stderr, "Error: nasm->data == NULL in add_input_rpool()\n");
        assert (0);
    }

    nasm_ldata_t *ldata = &nasm->ldata_arr[0];
    for (int i = 0; i < ldata->num_ninst; i++)
    {
        ninst_t *ninst = &ldata->ninst_arr_start[i];
        atomic_store (&ninst->state, NINST_COMPLETED);
        update_children (net_engine->rpool, ninst);
        for (int i = 0; i < SCHEDULE_MAX_DEVICES; i++) 
        {
            if (i == net_engine->device_idx) continue;
            if (ninst->dev_send_target[i]) 
            {
                // printf ("\tninst idx %d (L%d), target device: %d, current device: %d, desired device%d\n", 
                // ninst->ninst_idx, ninst->ldata->layer->layer_idx, i, dse->device_idx,
                // ninst->dev_send_target[i]);
                create_network_buffer_for_ninst (ninst);
                pthread_mutex_lock(&net_engine->tx_queue->queue_mutex);
                push_ninsts_to_net_queue(net_engine->tx_queue, &ninst, 1);
                pthread_mutex_unlock(&net_engine->tx_queue->queue_mutex);
            }
        }
    }
    
    aspen_free (data); 
}

void add_input_rpool_reverse (networking_engine *net_engine, nasm_t* nasm, char *input_filename)
{
    aspen_dnn_t *dnn = nasm->dnn;
    aspen_layer_t *first_layer = &dnn->layers[0];
    void *data = NULL;
    unsigned int input_params[NUM_PARAM_ELEMENTS] = {0};
    input_params[BATCH] = nasm->batch_size;
    if (first_layer->params[OUT_C] != 0 && first_layer->params[OUT_H] != 0 && first_layer->params[OUT_W] != 0)
    {
        input_params[OUT_C] = first_layer->params[OUT_C];
        input_params[OUT_H] = first_layer->params[OUT_H];
        input_params[OUT_W] = first_layer->params[OUT_W];
        data = aspen_load_input_NHWC (input_filename, input_params, sizeof(float));
    }
    else if (first_layer->params[MAT_M] != 0)
    {
        input_params[MAT_M] = first_layer->params[MAT_M];
        input_params[MAT_N] = nasm->tr_seq_len;
        data = aspen_load_input (input_filename, input_params, sizeof(float));
    }
    else
    {
        FPRT (stderr, "ERROR: rpool_add_nasm: first layer of dnn \"%s\" does not have output dimensions. Cannot add nasm \"%s_nasm_%d\".\n", 
            dnn->name, dnn->name, nasm->nasm_id);
        return;
    }

    if (data != NULL)
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
                (char*)data + i * ldata->out_mat_dims[OUT_H] * nasm->dnn->element_size, 
                ldata->out_mat_dims[OUT_H] * nasm->dnn->element_size);
    }
    if (nasm->data == NULL)
    {
        FPRT (stderr, "Error: nasm->data == NULL in add_input_rpool_reverse()\n");
        assert (0);
    }

    nasm_ldata_t *ldata = &nasm->ldata_arr[0];
    for (int i = 0; i < ldata->num_ninst; i++)
    {
        ninst_t *ninst = &ldata->ninst_arr_start[ldata->num_ninst - i - 1];
        atomic_store (&ninst->state, NINST_COMPLETED);
        update_children (net_engine->rpool, ninst);
        for (int i = 0; i < SCHEDULE_MAX_DEVICES; i++) 
        {
            if (i == net_engine->device_idx) continue;
            if (ninst->dev_send_target[i]) 
            {
                // printf ("\tninst idx %d (L%d), target device: %d, current device: %d, desired device %d\n", 
                // ninst->ninst_idx, ninst->ldata->layer->layer_idx, i, net_engine->device_idx,
                // ninst->dev_send_target[i]);
                create_network_buffer_for_ninst (ninst);
                pthread_mutex_lock(&net_engine->tx_queue->queue_mutex);
                push_ninsts_to_net_queue(net_engine->tx_queue, &ninst, 1);
                pthread_mutex_unlock(&net_engine->tx_queue->queue_mutex);
            }
        }
    }
    
    aspen_free (data); 
}

