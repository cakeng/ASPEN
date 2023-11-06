#ifndef _NETWORKING_H_
#define _NETWORKING_H_

#include "aspen.h"
#include "nasm.h"
#include "apu.h"
#include "rpool.h"
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/ioctl.h>

#define RX_TIMEOUT_USEC (1000) // 1ms
#define RX_STOP_SIGNAL (-19960930)
#define NET_INIT_QUEUE_SIZE (1024 * 32)
#define NETQUEUE_BUFFER_SIZE (1024 * 1024 * 2) // 2MiB
#define MAX_NUM_PEERS 32

struct networking_engine_t
{
    rpool_t *rpool;
    int listen_port;
    int isUDP;

    _Atomic int tx_run;
    _Atomic int rx_run;
    _Atomic int tx_kill;
    _Atomic int rx_kill;

    pthread_t rx_thread;
    pthread_t tx_thread;
    pthread_mutex_t rx_thread_mutex;
    pthread_cond_t rx_thread_cond;
    pthread_mutex_t tx_thread_mutex;
    pthread_cond_t tx_thread_cond;
    
    rpool_queue_t *tx_queue;
};

// networking_engine_t* init_networking (nasm_t* nasm, rpool_t* rpool, DEVICE_MODE device_mode, char* ip, int port, int is_UDP, int pipelined);
// void init_networking_queue (networking_queue_t *networking_queue);
// void init_server(networking_engine_t* net_engine, int port,int is_UDP);
// void init_edge(networking_engine_t* net_engine, char* ip, int port, int is_UDP);
// void add_inference_whitelist (networking_engine_t *net_engine, int inference_id);
// void remove_inference_whitelist (networking_engine_t *net_engine, int inference_id);
// int is_inference_whitelist (networking_engine_t *net_engine, int inference_id);
// void net_engine_wait(networking_engine_t* net_engine);
// void transmission(networking_engine_t *net_engine);
// void receive(networking_engine_t *net_engine);

// void net_queue_reset (networking_queue_t *networking_queue);
// void net_engine_reset (networking_engine_t *net_engine);
// void net_engine_stop (networking_engine_t *net_engine);
// void net_engine_run (networking_engine_t *net_engine);
// void net_engine_wait_for_tx_queue_completion (networking_engine_t *net_engine);

// void create_network_buffer_for_ninst (ninst_t *target_ninst);
// unsigned int pop_ninsts_from_net_queue (networking_queue_t *networking_queue, ninst_t **ninst_ptr_list, unsigned int max_ninsts_to_get);
// void push_ninsts_to_net_queue (networking_queue_t *networking_queue, ninst_t **ninst_ptr_list, unsigned int num_ninsts);
// void net_engine_add_input_rpool (networking_engine_t *net_engine, nasm_t* nasm, char *input_filename);
// void net_engine_add_input_rpool_reverse (networking_engine_t *net_engine, nasm_t* nasm, char *input_filename);
// void net_engine_destroy(networking_engine_t* net_engine);

#endif /* _NETWORKING_ */