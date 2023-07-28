#include "networking.h"
#include <errno.h>

void* transmission_buffer = NULL;

void *net_tx_thread_runtime (void* thread_info) 
{
    networking_engine *net_engine = (networking_engine*) thread_info;
    while (!net_engine->kill)
    {
        if(net_engine->run) {
            transmission(net_engine);
        }
    }
}

void *net_rx_thread_runtime (void* thread_info) 
{
    networking_engine *net_engine = (networking_engine*) thread_info;
    while (!net_engine->kill)
    {
        if(net_engine->run) {
            receive(net_engine);
        }
    }
}

void init_networking_queue (networking_queue_t *networking_queue)
{
    networking_queue->idx_start = 0;
    networking_queue->idx_end = 0;
    networking_queue->num_stored = 0;
    networking_queue->max_stored = INIT_QUEUE_SIZE;
    
    networking_queue->ninst_ptr_arr = (ninst_t**) calloc (INIT_QUEUE_SIZE, sizeof(ninst_t*));
    networking_queue->ninst_buf_arr = (void**) calloc(INIT_QUEUE_SIZE, sizeof(void*));
}

networking_engine* init_networking (nasm_t* nasm, rpool_t* rpool, SOCK_TYPE sock_type, char* ip, int port, int is_UDP, int sequential) 
{
    printf("Initializing Networking Engine...\n");
    networking_engine *net_engine = calloc (1, sizeof(networking_engine));
    networking_queue_t *networking_queue = calloc (1, sizeof(networking_queue_t));
    init_networking_queue(networking_queue);

    net_engine->net_queue = networking_queue;
    atomic_store (&net_engine->run, 0);
    atomic_store (&net_engine->kill, 0);
    atomic_store (&net_engine->shutdown, 0);
    net_engine->nasm = nasm;
    net_engine->rpool = rpool;
    net_engine->sequential = sequential;
    for (int i=0; i<SCHEDULE_MAX_DEVICES; i++) {
        net_engine->inference_whitelist[i] = -1;
    }

    switch (sock_type)
    {
    case SOCK_TX:
        init_tx(net_engine, ip, port, is_UDP);
        break;
    case SOCK_RX:
        init_rx(net_engine, port, is_UDP);
        break;
    default:
        printf("Error - Unsupported socket type. Type: %d\n", net_engine->sock_type);
        assert(0);
        break;
    }

    char info_str[MAX_STRING_LEN*2];
    void *whitelist[NUM_RPOOL_CONDS] = {NULL};
    whitelist [RPOOL_NASM] = nasm;
    sprintf (info_str, "%s_%s_%d", nasm->dnn->name, "nasm", nasm->nasm_id);
    float queue_per_layer = rpool->ref_ases * NUM_LAYERQUEUE_PER_ASE * NUM_QUEUE_PER_LAYER;
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
        
        nasm_ldata_t *ldata = &nasm->ldata_arr[0];
        aspen_layer_t *layer = ldata->layer;
        size_t num_cols = 0;
        if (layer->params[OUT_H] != 0 && layer->params[OUT_W] != 0)
            num_cols = nasm->batch_size * layer->params[OUT_H] * layer->params[OUT_W];
        else if (layer->params[MAT_M] != 0)
            num_cols = nasm->batch_size * nasm->tr_seq_len;
            
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

    pthread_create (&net_engine->tx_thread, NULL, net_tx_thread_runtime, (void*)net_engine);
    pthread_create (&net_engine->rx_thread, NULL, net_rx_thread_runtime, (void*)net_engine);
    return net_engine;
}

void init_socket (networking_engine* net_engine)
{
    switch (net_engine->sock_type)
    {
    case SOCK_TX:
        if(connect(net_engine->tx_sock,(struct sockaddr*)&net_engine->rx_addr, sizeof(net_engine->rx_addr)) == -1)
        {
            int rx_ip = net_engine->rx_addr.sin_addr.s_addr;
            printf("Error - Socket connection error... Rx ip: %d.%d.%d.%d, Rx Port: %d\n", 
                (rx_ip>>0)&0xff, (rx_ip>>8)&0xff, (rx_ip>>16)&0xff, (rx_ip>>24)&0xff, net_engine->rx_addr.sin_port);
            assert (0);
        }
        break;
    case SOCK_RX:
        if(listen(net_engine->rx_sock, 99) == -1)
        {
            int rx_ip = net_engine->rx_addr.sin_addr.s_addr;
            printf("Error - Socket listen error... Rx ip: %d.%d.%d.%d, Rx Port: %d\n", 
                (rx_ip>>0)&0xff, (rx_ip>>8)&0xff, (rx_ip>>16)&0xff, (rx_ip>>24)&0xff, net_engine->rx_addr.sin_port);
            assert(0);
        }
        socklen_t rx_addr_size = sizeof(net_engine->rx_addr);
        net_engine->tx_sock = accept(net_engine->rx_sock, (struct sockaddr*)&net_engine->rx_addr, &rx_addr_size);
        if(net_engine->tx_sock == -1)
        {
            printf("Error! - Socket accept error.\n");
        } else {
            int rx_ip = net_engine->rx_addr.sin_addr.s_addr;
            printf("Connect to IP: %d.%d.%d.%d\n", (rx_ip>>0)&0xff, (rx_ip>>8)&0xff, (rx_ip>>16)&0xff, (rx_ip>>24)&0xff);
        }
        break;
    default:
        printf("Error - Unsupported socket type. Type: %d\n", net_engine->sock_type);
        assert(0);
        break;
    }
}

void init_rx(networking_engine* net_engine, int port,int is_UDP) {

    if (is_UDP != 0) {
        net_engine->rx_sock = socket (PF_INET, SOCK_DGRAM, 0);
    }
    else {
        net_engine->rx_sock = socket (PF_INET, SOCK_STREAM, 0);
    }

    int option = 1;
    setsockopt(net_engine->rx_sock, SOL_SOCKET, SO_REUSEADDR, &option, sizeof(option));
    net_engine->sock_type = SOCK_RX;
    net_engine->tx_sock = 0;
    net_engine->rx_addr.sin_family = AF_INET;
    net_engine->rx_addr.sin_addr.s_addr = htonl (INADDR_ANY);
    net_engine->rx_addr.sin_port = htons (port);
    net_engine->isUDP = is_UDP;

    if(bind(net_engine->rx_sock,(struct sockaddr*)&net_engine->rx_addr, sizeof(net_engine->rx_addr)) == -1)
        printf("ERROR! socket bind error\n");

    init_socket(net_engine);
}

void init_tx(networking_engine* net_engine, char* ip, int port, int is_UDP) {

    bzero (&net_engine->rx_addr, sizeof(net_engine->rx_addr));
    bzero (&net_engine->tx_addr, sizeof(net_engine->tx_addr));

    net_engine->sock_type = SOCK_TX;
    net_engine->tx_sock = socket (PF_INET, SOCK_STREAM, 0);
    net_engine->rx_addr.sin_family = AF_INET;
    net_engine->rx_addr.sin_addr.s_addr = inet_addr (ip);
    net_engine->rx_addr.sin_port = htons (port);
    net_engine->isUDP = 0;

    init_socket(net_engine);
}

void add_inference_whitelist (networking_engine *net_engine, unsigned int inference_id) {
    for (int i=0; i<SCHEDULE_MAX_DEVICES; i++) {
        if (net_engine->inference_whitelist[i] == -1) {
            net_engine->inference_whitelist[i] = inference_id;
            return;
        }
    }
}

void remove_inference_whitelist (networking_engine *net_engine, unsigned int inference_id) {
    for (int i=0; i<SCHEDULE_MAX_DEVICES; i++) {
        if (net_engine->inference_whitelist[i] == inference_id) {
            net_engine->inference_whitelist[i] = -1;
            return;
        }
    }
}

int is_inference_whitelist (networking_engine *net_engine, unsigned int inference_id) {
    for (int i=0; i<SCHEDULE_MAX_DEVICES; i++) {
        if (net_engine->inference_whitelist[i] == inference_id) {
            return 1;
        }
    }
    return 0;
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

unsigned int pop_ninsts_from_net_queue (networking_queue_t *networking_queue, ninst_t **ninst_ptr_list, char* buffer, unsigned int max_ninsts_to_get)
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
        const unsigned int W = networking_queue->ninst_ptr_arr[i]->tile_dims[OUT_W];
        const unsigned int H = networking_queue->ninst_ptr_arr[i]->tile_dims[OUT_H];
        
        for (; num_ninsts < networking_queue->num_stored; num_ninsts++)
        {
            if (num_ninsts >= max_ninsts_to_get)
                break;
            if (buffer_usage + W * H * sizeof(float) > NETQUEUE_BUFFER_SIZE)
                break;
            ninst_ptr_list[num_ninsts] = networking_queue->ninst_ptr_arr[i];
            memcpy(buffer, networking_queue->ninst_buf_arr[i], W * H * sizeof(float));
            free(networking_queue->ninst_buf_arr[i]);
            i++;
            buffer += W * H * sizeof(float);
            buffer_usage += W * H * sizeof(float);

            if (i == networking_queue->max_stored)
                i = 0;
        }

        networking_queue->idx_start = i;
        networking_queue->num_stored -= num_ninsts;
    }
    
    return num_ninsts;
}

void push_ninsts_to_net_queue (networking_queue_t *networking_queue, ninst_t *ninst_ptr, unsigned int num_ninsts)
{

    if (networking_queue->num_stored + num_ninsts > networking_queue->max_stored)
        update_net_queue_size (networking_queue, num_ninsts);
    float* out_mat = (float*)ninst_ptr->out_mat;
    const unsigned int W = ninst_ptr->tile_dims[OUT_W];
    const unsigned int H = ninst_ptr->tile_dims[OUT_H];
    const unsigned int stride = ninst_ptr->ldata->out_mat_stride;

    unsigned i = networking_queue->idx_end;
    
    for (int j = 0; j < num_ninsts; j++)
    {
        networking_queue->ninst_buf_arr[i] = malloc(W * H * sizeof(float));
        *(networking_queue->ninst_ptr_arr + i) = ninst_ptr;
        for(int w = 0; w < W; w++) {
            memcpy((char*)networking_queue->ninst_buf_arr[i] + w * H * sizeof(float), (char*)out_mat + w * stride * sizeof(float), H * sizeof(float));
        }
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
        FPRT (stderr, "ERROR: push_ninsts_to_queue_back: networking_queue is NULL.\n");
        return;
    }
    if (ninst_ptr_list == NULL)
    {
        FPRT (stderr, "ERROR: push_ninsts_to_queue_back: ninst_ptr_list is NULL.\n");
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

void transmission(networking_engine *net_engine) 
{
    ninst_t *target_ninst_list[4];
    unsigned int num_ninsts = 0;
    if (transmission_buffer == NULL)
        transmission_buffer = malloc(NETQUEUE_BUFFER_SIZE);
    void* buffer_start_ptr = transmission_buffer;

    pthread_mutex_lock(&net_engine->net_engine_mutex);
    num_ninsts = pop_ninsts_from_net_queue(net_engine->net_queue, target_ninst_list, (char*)transmission_buffer, 1);
    pthread_mutex_unlock(&net_engine->net_engine_mutex);

    if(num_ninsts > 0) {
        for(int i = 0; i < num_ninsts; i++)
        {
            ninst_t* target_ninst = target_ninst_list[i];
            const unsigned int W = target_ninst->tile_dims[OUT_W];
            const unsigned int H = target_ninst->tile_dims[OUT_H];
            
            const unsigned int total_bytes = W * H * sizeof(float);
            unsigned int sent_bytes = 0;

            double time_sent = get_time_secs();
            send(net_engine->tx_sock, (char*)&target_ninst->ninst_idx, sizeof(int), 0);
            send(net_engine->tx_sock, (char*)&target_ninst->ldata->nasm->inference_id, sizeof(int), 0);
            while(total_bytes - sent_bytes) {
                sent_bytes += send(net_engine->tx_sock, buffer_start_ptr, total_bytes, 0);
            }
            printf("sent ninst %d (L%d Id%d), time taken %f, netqueue remaining %d,\n", 
                target_ninst->ninst_idx, target_ninst->ldata->layer->layer_idx,
                target_ninst->ldata->nasm->inference_id,
                get_time_secs() - time_sent,
                net_engine->net_queue->num_stored
                );
            
            buffer_start_ptr += W * H * sizeof(float);

            target_ninst->sent_time = get_time_secs();
        }
    }
}

void receive(networking_engine *net_engine) {
    int recv_ninst_idx;
    int inference_id;
    ninst_t* target_ninst;

    if(recv(net_engine->tx_sock, (char*)&recv_ninst_idx, sizeof(int), 0)) {
        // For shutdown process
        if(recv_ninst_idx == -1)
        {
            int close_idx = -1;
            if(net_engine->sock_type == SOCK_RX)
            {
                printf("close_connection recv shutdown in RX\n");
                send(net_engine->tx_sock, (char*)&close_idx, sizeof(int), 0);
                printf("close_connection send shutdown in RX\n");
                atomic_store (&net_engine->shutdown, 1);
            }
            else if(net_engine->sock_type == SOCK_TX)
            {
                printf("close_connection recv shutdown in TX\n");
                atomic_store (&net_engine->shutdown, 1);
            }
        }
        for(int i = 0; i < net_engine->nasm->num_ninst; i++) {
            if(i == recv_ninst_idx) {
                // printf("recv ninst %d\n", recv_ninst_idx);
                target_ninst = &net_engine->nasm->ninst_arr[i];
                char* out_mat = target_ninst->out_mat;
                const unsigned int W = target_ninst->tile_dims[OUT_W];
                const unsigned int H = target_ninst->tile_dims[OUT_H];
                const unsigned int stride = target_ninst->ldata->out_mat_stride;
                int total_bytes = W * H * sizeof(float);
                unsigned int recv_bytes = 0;

                recv(net_engine->tx_sock, (char*)&inference_id, sizeof(int), 0);

                char* buffer = malloc(total_bytes);
                bzero(buffer, total_bytes);

                while(total_bytes - recv_bytes)
                {
                    recv_bytes += recv(net_engine->tx_sock, buffer, total_bytes, MSG_WAITALL);
                }

                target_ninst->recved_time = get_time_secs();
                // printf("recv ninst: %d, %f\n", target_ninst->ninst_idx, target_ninst->recved_time * 1000.0);

                if (atomic_exchange (&target_ninst->state, NINST_COMPLETED) == NINST_COMPLETED || !is_inference_whitelist(net_engine, inference_id)) 
                {
                    free(buffer);
                    return;
                }
                copy_buffer_to_ninst_data (target_ninst, buffer);

                unsigned int num_ninst_completed = atomic_fetch_add (&target_ninst->ldata->num_ninst_completed , 1);
                int num_ase = net_engine->rpool->ref_ases > 0 ? net_engine->rpool->ref_ases : 1;
                update_children (net_engine->rpool, target_ninst, i/(net_engine->nasm->ldata_arr[0].num_ninst/num_ase));
                
                if (num_ninst_completed == target_ninst->ldata->num_ninst - 1)
                {
                    atomic_fetch_add (&net_engine->nasm->num_ldata_completed, 1);
                    if(net_engine->sequential)
                    {
                        dse_group_run(net_engine->dse_group);
                        dse_group_set_enable_device(net_engine->dse_group, net_engine->device_idx, 1);
                        dse_group_add_prioritize_rpool(net_engine->dse_group, net_engine->device_idx);
                    } 
                    if (net_engine->sock_type == SOCK_TX)
                    {
                        if( target_ninst->ldata->layer->layer_idx == net_engine->nasm->num_ldata - 1) {
                            atomic_store(&net_engine->nasm->num_ldata_completed, net_engine->nasm->num_ldata);
                            pthread_mutex_lock (&net_engine->nasm->nasm_mutex);
                            pthread_cond_signal (&net_engine->nasm->nasm_cond);
                            pthread_mutex_unlock (&net_engine->nasm->nasm_mutex);
                        }
                    }
                }

            }
        }
    }
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
        FPRT (stderr, "Error: nasm->data == NULL in add_ninst_net_queue()\n");
        assert (0);
    }

    nasm_ldata_t *ldata = &nasm->ldata_arr[0];
    for (int i = 0; i < ldata->num_ninst; i++)
    {
        ninst_t *ninst = &ldata->ninst_arr_start[i];
        // push_ninsts_to_net_queue(net_engine->net_queue, ninst, 1);
        ninst->state = NINST_READY;
        rpool_push_ninsts(net_engine->rpool, &ninst, 1, 0);
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
        FPRT (stderr, "Error: nasm->data == NULL in add_ninst_net_queue()\n");
        assert (0);
    }

    nasm_ldata_t *ldata = &nasm->ldata_arr[0];
    for (int i = 0; i < ldata->num_ninst; i++)
    {
        ninst_t *ninst = &ldata->ninst_arr_start[ldata->num_ninst - i - 1];
        // push_ninsts_to_net_queue(net_engine->net_queue, ninst, 1);
        ninst->state = NINST_READY;
        rpool_push_ninsts(net_engine->rpool, &ninst, 1, 0);
    }
    
    aspen_free (data); 
}

void net_queue_destory(networking_queue_t* net_queue)
{
    if(net_queue == NULL) return;
    if(net_queue->ninst_ptr_arr != NULL) free(net_queue->ninst_ptr_arr);
    if(net_queue->ninst_buf_arr != NULL) free(net_queue->ninst_buf_arr);
}

void close_connection(networking_engine* net_engine)
{
    if(net_engine == NULL) return;
    int close_idx = -1;
    int shutdown = 0;
    if(net_engine->sock_type == SOCK_TX)
    {
        send(net_engine->tx_sock, (char*)&close_idx, sizeof(int), 0);
        printf("close_connection send shutdown in TX\n");
    } 

    while(!shutdown)
    {
        shutdown = atomic_load(&net_engine->shutdown);
    }
}

void net_engine_destroy(networking_engine* net_engine)
{
    if(net_engine == NULL) return;
    net_engine->kill = 1;
    net_queue_destory(net_engine->net_queue);
    close(net_engine->rx_sock);
    close(net_engine->tx_sock);
    pthread_mutex_destroy(&net_engine->net_engine_mutex);
    free(net_engine);
}
