#include "networking.h"

void *net_tx_thread_runtime (void* thread_info) 
{
    networking_engine_t *net_engine = (networking_engine_t*) thread_info;
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
    networking_engine_t *net_engine = (networking_engine_t*) thread_info;
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


networking_engine_t* init_networking (nasm_t* nasm, rpool_t* rpool, DEVICE_MODE device_mode, char* ip, int port, int is_UDP, int pipelined) 
{
    PRTF("Initializing Networking Engine...\n");
    networking_engine_t *net_engine = calloc (1, sizeof(networking_engine_t));
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
        ERROR_PRTF ("Error - Unsupported socket type. Type: %d\n", net_engine->device_mode);
        assert(0);
        break;
    }

    net_engine->tx_buffer = calloc(NETQUEUE_BUFFER_SIZE, sizeof(char));
    net_engine->rx_buffer = calloc(NETQUEUE_BUFFER_SIZE, sizeof(char));

    char info_str[MAX_STRING_LEN*2];
    void *whitelist[NUM_RPOOL_CONDS] = {NULL};
    whitelist [RPOOL_NASM] = nasm;
    sprintf (info_str, "%s_%s_%d", nasm->dnn->name, "nasm", nasm->nasm_hash);
    float queue_per_layer = rpool->ref_dses * NUM_LAYERQUEUE_PER_DSE * NUM_QUEUE_PER_LAYER;
    unsigned int num_queues = nasm->dnn->num_layers*queue_per_layer;
    if (num_queues < 1)
        num_queues = 1;
    nasm->gpu_idx = rpool->gpu_idx;
    rpool_add_queue_group (rpool, info_str, num_queues, NULL, whitelist);

    net_engine->rx_thread_cond = (pthread_cond_t) PTHREAD_COND_INITIALIZER;
    net_engine->rx_thread_mutex = (pthread_mutex_t) PTHREAD_MUTEX_INITIALIZER;
    net_engine->tx_thread_cond = (pthread_cond_t) PTHREAD_COND_INITIALIZER;
    net_engine->tx_thread_mutex = (pthread_mutex_t) PTHREAD_MUTEX_INITIALIZER;
    pthread_create (&net_engine->tx_thread, NULL, net_tx_thread_runtime, (void*)net_engine);
    pthread_create (&net_engine->rx_thread, NULL, net_rx_thread_runtime, (void*)net_engine);
    return net_engine;
}

void send(networking_engine_t *net_engine) 
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
    num_ninsts = pop_ninsts_from_net_queue(net_engine->tx_queue, target_ninst_list, 1);
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
        // PRTF("Networking: Ninst%d, Sending %d bytes, W%d, H%d, data size %d\n", i, data_size + 3*sizeof(int), 
        //     target_ninst_list[i]->tile_dims[OUT_W], target_ninst_list[i]->tile_dims[OUT_H], 
        //     target_ninst_list[i]->tile_dims[OUT_W]*target_ninst_list[i]->tile_dims[OUT_H]*sizeof(float));
    }
    payload_size = buffer_ptr - (char*)net_engine->tx_buffer - sizeof(int32_t);
    *(int32_t*)net_engine->tx_buffer = payload_size;
    payload_size += sizeof(int32_t);

    #ifdef DEBUG
    PRTF("Networking: Sending %d bytes -", payload_size);
    for (int i = 0; i < num_ninsts; i++)
    {
        ninst_t* target_ninst = target_ninst_list[i];
        PRTF(" (N%d L%d I%d %ldB)", 
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
            ERROR_PRTF ( "Error: send() failed. ret: %d\n", ret);
            assert(0);
        }
        bytes_sent += ret;
    }
    #ifdef DEBUG
    PRTF(" - Time taken %fs, %d tx queue remains.\n", (get_time_secs() - time_sent), net_engine->tx_queue->num_stored);
    #endif
}

void receive(networking_engine_t *net_engine) 
{
    
    int32_t payload_size;
    int ret = read(net_engine->comm_sock, (char*)&payload_size, sizeof(int32_t));
    if (ret == -1)
    {
        return;
    }
    else if (ret < 0)
    {
        ERROR_PRTF ( "Error: recv() failed. ret: %d\n", ret);
        assert(0);
    }
    else if (ret == 0)
    {
        ERROR_PRTF ( "Error: RX Connection closed unexpectedly.\n");
        assert(0);
    }
    else
    {
        if (payload_size <= 0)
        {
            // PRTF("Networking: RX Command %d received - ", payload_size);
            if (payload_size == RX_STOP_SIGNAL)
            {
                PRTF("RX stop signal received.\n");
                atomic_store (&net_engine->rx_run, 0);
                atomic_store (&net_engine->nasm->completed, 1);
                pthread_mutex_lock (&net_engine->nasm->nasm_mutex);
                pthread_cond_signal (&net_engine->nasm->nasm_cond);
                pthread_mutex_unlock (&net_engine->nasm->nasm_mutex);
                return;
            }
            else
            {
                PRTF("Unknown command\n");
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
                ERROR_PRTF ("Error: recv() failed. ret: %d\n", ret);
                assert(0);
            }
            bytes_received += ret;
        }
        #ifdef DEBUG
        // PRTF("Networking: Received %d bytes -", bytes_received);
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
                PRTF("Warning: Received inference_id %d is not matched with ninst_idx %d\n", inference_id, ninst_idx);
                continue;
            }
            if (ninst_idx >= net_engine->nasm->num_ninst)
            {
                PRTF("Warning: Received ninst_idx %d is not found in nasm\n", ninst_idx);
                continue;
            }
            #endif
            ninst_t* target_ninst = &net_engine->nasm->ninst_arr[ninst_idx];
            if (!is_inference_whitelist(net_engine, inference_id))
            {
                buffer_ptr += data_size;
                continue;
            }
            if (atomic_exchange (&target_ninst->state, NINST_COMPLETED) == NINST_COMPLETED) 
            {
                buffer_ptr += data_size;
                continue;
            }
            #ifdef DEBUG
            if (data_size != target_ninst->tile_dims[OUT_W]*target_ninst->tile_dims[OUT_H]*sizeof(float))
            {
                ERROR_PRTF ( "Error: Received data size %d is not matched with ninst_idx %d\n", data_size, ninst_idx);
                assert(0);
            }
            #endif
            copy_buffer_to_ninst_data (target_ninst, buffer_ptr);
            buffer_ptr += data_size;
            atomic_store(&target_ninst->state, NINST_COMPLETED);
            target_ninst->received_time = get_time_secs_offset ();
            #ifdef DEBUG
            PRTF("\t[Device %d] (N%d L%d I%d %ldB) state: %d\n",
                net_engine->device_idx,
                target_ninst->ninst_idx, target_ninst->ldata->layer->layer_idx, target_ninst->ldata->nasm->inference_id, 
                target_ninst->tile_dims[OUT_W]*target_ninst->tile_dims[OUT_H]*sizeof(float), atomic_load(&target_ninst->state));
            #endif
            
            unsigned int num_ninst_completed = atomic_fetch_add (&target_ninst->ldata->num_ninst_completed , 1);

            if(atomic_load(&target_ninst->state) == NINST_COMPLETED)
            {
                // RED_PRTF ("2 Ninst %d(L%d) downloaded\n", target_ninst->ninst_idx, target_ninst->ldata->layer->layer_idx);
                update_children (net_engine->rpool, target_ninst);
            }
            
            if (num_ninst_completed == target_ninst->ldata->num_ninst - 1)
            {
                // printf ("\t\tNet engine completed layer %d of nasm %d\n", 
                //     target_ninst->ldata->layer->layer_idx, target_ninst->ldata->nasm->nasm_hash);
                for (int pidx = 0; pidx < NUM_PARENT_ELEMENTS; pidx++)
                {
                    if (target_ninst->ldata->parent_ldata_idx_arr[pidx] == -1)
                        continue;
                    nasm_ldata_t *parent_ldata = &target_ninst->ldata->nasm->ldata_arr[target_ninst->ldata->parent_ldata_idx_arr[pidx]];
                    unsigned int num_child_ldata_completed = atomic_fetch_add (&parent_ldata->num_child_ldata_completed, 1);
                    if (num_child_ldata_completed == parent_ldata->num_child_ldata && (parent_ldata != parent_ldata->nasm->ldata_arr))
                    {
                        free_ldata_out_mat (parent_ldata);
                        // YELLOW_PRTF ("ldata %d output freed by net engine\n", parent_ldata->layer->layer_idx);
                    }
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
        // #ifdef DEBUG
        //     PRTF("\n");
        // #endif
    }
}

void net_engine_run (networking_engine_t *net_engine)
{
    if (net_engine == NULL)
    {
        ERROR_PRTF ("ERROR: net_engine_run: net_engine is NULL\n");
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

void net_engine_stop (networking_engine_t* net_engine)
{
    if (net_engine == NULL)
    {
        ERROR_PRTF ("ERROR: net_engine_stop: net_engine is NULL\n");
        return;
    }
    #ifdef DEBUG
    PRTF ("Networking: Stopping network engine tx thread...\n");
    #endif
    int is_running = atomic_exchange (&net_engine->tx_run, 0);
    if (is_running == 0)
        return;
    pthread_mutex_lock (&net_engine->tx_thread_mutex);
    #ifdef DEBUG
    PRTF("Networking: Sending network engine rx thread stop signal...\n");
    #endif
    int32_t command = RX_STOP_SIGNAL;
    int ret = write(net_engine->comm_sock, (char*)&command, sizeof(command));
    if (ret < 0)
    {
        ERROR_PRTF ( "ERROR: net_engine_sto: send() failed.\n");
        assert(0);
    }
    #ifdef DEBUG
    PRTF("Networking: Waiting for network engine rx thread to stop...\n");
    #endif
    pthread_mutex_lock (&net_engine->rx_thread_mutex);
    #ifdef DEBUG
    PRTF("Networking: Network engine stopped.\n");
    #endif
}

void net_engine_destroy (networking_engine_t* net_engine)
{
    if(net_engine == NULL) 
        return;
    
    if (atomic_load(&net_engine->tx_run) == 1 && atomic_load(&net_engine->rx_run))
    {
        ERROR_PRTF ("WARNING: net_engine_destroy: net_engine is still running\n");
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

    free(net_engine);
}

void net_wait_for_nasm_completion (networking_engine_t *net_engine)
{
    pthread_mutex_lock (&net_engine->tx_queue->queue_mutex);
    if (net_engine->tx_queue->num_stored > 0)
    {
        pthread_cond_wait (&net_engine->tx_queue->queue_cond, &net_engine->tx_queue->queue_mutex);
    }
    pthread_mutex_unlock (&net_engine->tx_queue->queue_mutex);
}
