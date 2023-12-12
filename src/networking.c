#include "networking.h"

void *net_tx_thread_runtime (void* thread_info) 
{
    networking_engine_t *net_engine = (networking_engine_t*) thread_info;
    pthread_mutex_lock (&net_engine->tx_thread_mutex);
    atomic_store (&net_engine->tx_run, 0);
    pthread_cond_wait (&net_engine->tx_thread_cond, &net_engine->tx_thread_mutex);
    while (!net_engine->tx_kill)
    {
        net_engine_send(net_engine);
        if(!net_engine->tx_run)
            pthread_cond_wait (&net_engine->tx_thread_cond, &net_engine->tx_thread_mutex);
    }
    return NULL;
}

void *net_rx_thread_runtime (void* thread_info) 
{
    networking_engine_t *net_engine = (networking_engine_t*) thread_info;
    pthread_mutex_lock (&net_engine->rx_thread_mutex);
    atomic_store (&net_engine->rx_run, 0);
    pthread_cond_wait (&net_engine->rx_thread_cond, &net_engine->rx_thread_mutex);
    while (!net_engine->rx_kill)
    {
        net_engine_receive(net_engine);
        if(!net_engine->rx_run) 
            pthread_cond_wait (&net_engine->rx_thread_cond, &net_engine->rx_thread_mutex);
    }
    return NULL;
}


networking_engine_t* init_net_engine () 
{
    PRTF("Initializing Networking Engine...\n");
    networking_engine_t *net_engine = calloc (1, sizeof(networking_engine_t));

    net_engine->nasm_list = NULL;
    net_engine->rpool = NULL;
    net_engine->num_nasms = 0;
    net_engine->buffer = calloc (NETWORK_BUFFER, 1);

    atomic_store (&net_engine->tx_run, -1);
    atomic_store (&net_engine->rx_run, -1);
    atomic_store (&net_engine->tx_kill, 0);
    atomic_store (&net_engine->rx_kill, 0);

    net_engine->rx_thread_cond = (pthread_cond_t) PTHREAD_COND_INITIALIZER;
    net_engine->rx_thread_mutex = (pthread_mutex_t) PTHREAD_MUTEX_INITIALIZER;
    net_engine->tx_thread_cond = (pthread_cond_t) PTHREAD_COND_INITIALIZER;
    net_engine->tx_thread_mutex = (pthread_mutex_t) PTHREAD_MUTEX_INITIALIZER;
    pthread_create (&net_engine->tx_thread, NULL, net_tx_thread_runtime, (void*)net_engine);
    pthread_create (&net_engine->rx_thread, NULL, net_rx_thread_runtime, (void*)net_engine);
    return net_engine;
}

void destroy_net_engine (networking_engine_t *net_engine)
{
    if (net_engine == NULL)
        return;
    pthread_join (net_engine->tx_thread, NULL);
    pthread_join (net_engine->rx_thread, NULL);
    pthread_cond_destroy (&net_engine->tx_thread_cond);
    pthread_cond_destroy (&net_engine->rx_thread_cond);
    pthread_mutex_destroy (&net_engine->tx_thread_mutex);
    pthread_mutex_destroy (&net_engine->rx_thread_mutex);
    free (net_engine->buffer);
    free (net_engine->nasm_list);
    free (net_engine);
}

void net_engine_set_rpool (networking_engine_t *net_engine, rpool_t *rpool)
{
    if (net_engine == NULL)
    {
        ERROR_PRTF ("ERROR: net_add_rpool: net_engine is NULL\n");
        return;
    }
    if (rpool == NULL)
    {
        ERROR_PRTF ("ERROR: net_add_rpool: rpool is NULL\n");
        return;
    }
    net_engine->rpool = rpool;
}

void net_engine_add_nasm (networking_engine_t *net_engine, nasm_t *nasm)
{
    if (net_engine == NULL)
    {
        ERROR_PRTF ("ERROR: net_add_nasm: net_engine is NULL\n");
        return;
    }
    if (nasm == NULL)
    {
        ERROR_PRTF ("ERROR: net_add_nasm: nasm is NULL\n");
        return;
    }
    net_engine->nasm_list = realloc (net_engine->nasm_list, sizeof(nasm_t*) * (net_engine->num_nasms + 1));
    net_engine->nasm_list[net_engine->num_nasms] = nasm;
    net_engine->num_nasms++;
}

void net_engine_send(networking_engine_t *net_engine) 
{
    #ifdef DEBUG
    if (net_engine == NULL)
    {
        ERROR_PRTF ("ERROR: net_engine_send: net_engine is NULL\n");
        return;
    }
    #endif
    for (int i = 0; i < net_engine->num_nasms; i++)
    {
        
        nasm_t *nasm = net_engine->nasm_list[i];
        if (atomic_load(&nasm->num_ldata_completed) == nasm->num_ldata)
            continue;
        for (int j = 0; j < nasm->num_peers; j++)
        {
            aspen_peer_t *peer = nasm->peer_map[j];
            if (peer->sock == -1)
                continue;
            if (peer->isUDP)
            {
                ERROR_PRTF ("ERROR: UDP is not supported yet.\n");
                assert(0);
            }
            else
            {
                ninst_t **send_arr = net_engine->buffer;
                pthread_mutex_lock (&peer->tx_queue->occupied_mutex);
                int send_num = pop_ninsts_from_queue (peer->tx_queue, send_arr, 1);
                pthread_mutex_unlock (&peer->tx_queue->occupied_mutex);
                if (send_num == 0)
                    continue;
                char *send_buffer = net_engine->buffer + sizeof(ninst_t*) * send_num;
                memcpy (send_buffer, &send_num, sizeof(int));
                size_t send_size = sizeof(int);
                for (int k = 0; k < send_num; k++)
                {
                    ninst_t *send_ninst = send_arr[k];
                    size_t payload_size = sizeof(int) + send_ninst->tile_dims[OUT_H] 
                        * send_ninst->tile_dims[OUT_W] * sizeof(float);
                    if (payload_size + send_size > NETWORK_BUFFER)
                    {
                        ERROR_PRTF ("ERROR: net_engine_send: payload_size + send_size > NETWORK_BUFFER\n");
                        assert(0);
                    }
                    memcpy (send_buffer + send_size, &send_ninst->ninst_idx, sizeof(int));
                    copy_ninst_data_to_buffer (send_ninst, send_buffer + send_size + sizeof(int));
                    send_size += payload_size;
                }
                #ifdef DEBUG
                YELLOW_PRTF ("Networking: Sending %d ninsts to peer %d \\
                    (HASH %08lx, IP %s, Port %d)\n", send_num, j, peer->peer_hash, peer->ip, peer->listen_port);
                #endif
                if (write_bytes (peer->sock, send_buffer, send_size) != send_size)
                {
                    ERROR_PRTF ("ERROR: net_engine_send: write_bytes failed\n");
                    assert(0);
                }
            }
        }
    }
}

void net_engine_receive(networking_engine_t *net_engine) 
{
    #ifdef DEBUG
    if (net_engine == NULL)
    {
        ERROR_PRTF ("ERROR: net_engine_send: net_engine is NULL\n");
        return;
    }
    #endif
    for (int i = 0; i < net_engine->num_nasms; i++)
    {
        nasm_t *nasm = net_engine->nasm_list[i];
        if (atomic_load(&nasm->num_ldata_completed) == nasm->num_ldata)
            continue;
        for (int j = 0; j < nasm->num_peers; j++)
        {
            aspen_peer_t *peer = nasm->peer_map[j];
            if (peer->sock == -1)
                continue;
            if (peer->isUDP)
            {
                ERROR_PRTF ("ERROR: UDP is not supported yet.\n");
                assert(0);
            }
            else
            {
                char *recv_buffer = net_engine->buffer;
                int recv_num = 0;
                if (read_bytes (peer->sock, &recv_num, sizeof(int)) != sizeof(int))
                {
                    ERROR_PRTF ("ERROR: net_engine_receive: read_bytes failed\n");
                    assert(0);
                }
                if (recv_num < 0)
                {
                    close_tcp_connection (peer, recv_num);
                    continue;
                }
                #ifdef DEBUG
                YELLOW_PRTF("Networking: Receiving %d ninsts from peer %d \\
                    (HASH %08lx, IP %s, Port %d)\n", recv_num, j, peer->peer_hash, peer->ip, peer->listen_port);
                #endif
                if (recv_num == 0)
                    continue;
                for (int k = 0; k < recv_num; k++)
                {
                    int ninst_idx = 0;
                    if (read_bytes (peer->sock, &ninst_idx, sizeof(int)) != sizeof(int))
                    {
                        ERROR_PRTF ("ERROR: net_engine_receive: read_bytes failed\n");
                        assert(0);
                    }
                    ninst_t *recv_ninst = &nasm->ninst_arr[ninst_idx];
                    size_t payload_size = recv_ninst->tile_dims[OUT_H] 
                        * recv_ninst->tile_dims[OUT_W] * sizeof(float);
                    if (read_bytes (peer->sock, recv_buffer, payload_size) != payload_size)
                    {
                        ERROR_PRTF ("ERROR: net_engine_receive: read_bytes failed\n");
                        assert(0);
                    }
                    copy_buffer_to_ninst_data (recv_ninst, recv_buffer);
                    recv_ninst->state = NINST_COMPLETED;
                    int num_dse = net_engine->rpool->ref_dses > 0 ? net_engine->rpool->ref_dses : 1;
                    update_children (net_engine->rpool, recv_ninst, 
                        recv_ninst->ldata->num_ninst > num_dse? i/(recv_ninst->ldata->num_ninst/num_dse) : i);
                    unsigned int num_ninst_completed = atomic_fetch_add 
                        (&recv_ninst->ldata->num_ninst_completed , 1);
                    #ifdef DEBUG
                    RED_PRTF("Networking: %d/%d completed for ldata %d\n"
                        , num_ninst_completed + 1, recv_ninst->ldata->num_ninst, recv_ninst->ldata->layer->layer_idx);
                    #endif
                    if (num_ninst_completed == recv_ninst->ldata->num_ninst - 1)
                    {
                        #ifdef DEBUG
                        GREEN_PRTF ("Networking: Layer %d completed.\n", 
                            recv_ninst->ldata->layer->layer_idx);
                        #endif
                        for (int pidx = 0; pidx < NUM_PARENT_ELEMENTS; pidx++)
                        {
                            if (recv_ninst->ldata->parent_ldata_idx_arr[pidx] == -1)
                                continue;
                            nasm_ldata_t *parent_ldata = &recv_ninst->ldata->nasm->ldata_arr[recv_ninst->ldata->parent_ldata_idx_arr[pidx]];
                            unsigned int num_child_ldata_completed = atomic_fetch_add (&parent_ldata->num_child_ldata_completed, 1);
                            if (num_child_ldata_completed + 1 == parent_ldata->num_child_ldata && (parent_ldata != parent_ldata->nasm->ldata_arr))
                                free_ldata_out_mat (parent_ldata);
                        }
                        
                        if (recv_ninst->ldata == &nasm->ldata_arr[nasm->num_ldata - 1])
                        {
                            // All layers of the nasm is completed.
                            atomic_store (&nasm->completed, 1);
                            pthread_mutex_lock (&nasm->nasm_mutex);
                            pthread_cond_signal (&nasm->nasm_cond);
                            pthread_mutex_unlock (&nasm->nasm_mutex);
                        }
                    }
                }
            }       
        }
    }
}

void net_engine_run (networking_engine_t *net_engine)
{
    if (net_engine == NULL)
    {
        ERROR_PRTF ("ERROR: net_engine_run: net_engine is NULL\n");
        return;
    }
    int state = atomic_load (&net_engine->rx_run);
    while (state == -1)
        state = atomic_load (&net_engine->rx_run);
    if (state == 1)
    {
        return;
    }
    else 
    {
        pthread_mutex_lock (&net_engine->rx_thread_mutex);
        atomic_store (&net_engine->rx_run, 1);
        pthread_cond_signal (&net_engine->rx_thread_cond);
        pthread_mutex_unlock (&net_engine->rx_thread_mutex);
    }
    state = atomic_load (&net_engine->tx_run);
    while (state == -1)
        state = atomic_load (&net_engine->tx_run);
    if (state == 1)
    {
        return;
    }
    else 
    {
        pthread_mutex_lock (&net_engine->tx_thread_mutex);
        atomic_store (&net_engine->tx_run, 1);
        pthread_cond_signal (&net_engine->tx_thread_cond);
        pthread_mutex_unlock (&net_engine->tx_thread_mutex);
    }
}

void net_engine_stop (networking_engine_t* net_engine)
{
   if (net_engine == NULL)
    {
        ERROR_PRTF ("ERROR: net_engine_stop: net_engine is NULL");
        return;
    }
    int state = atomic_exchange (&net_engine->tx_run, 0);
    if (state == 0)
    {
        return;
    }
    else 
    {
        pthread_mutex_lock (&net_engine->tx_thread_mutex);
        pthread_mutex_unlock (&net_engine->tx_thread_mutex);
    }
    state = atomic_exchange (&net_engine->rx_run, 0);
    if (state == 0)
    {
        return;
    }
    else 
    {
        pthread_mutex_lock (&net_engine->rx_thread_mutex);
        pthread_mutex_unlock (&net_engine->rx_thread_mutex);
    }
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
