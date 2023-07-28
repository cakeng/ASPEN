#ifndef _NETWORKING_H_
#define _NETWORKING_H_

#include "aspen.h"
#include "nasm.h"
#include "apu.h"
#include "scheduling.h"
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>


#define INIT_QUEUE_SIZE 1024 * 128 * 4
#define NETQUEUE_BUFFER_SIZE 1024 * 1024 * 32

struct networking_queue_t
{
    unsigned int idx_start;
    unsigned int idx_end;
    _Atomic unsigned int num_stored;
    unsigned int max_stored;

    ninst_t **ninst_ptr_arr;
    void **ninst_buf_arr;

    pthread_mutex_t queue_mutex;
    pthread_cond_t queue_cond;
};

struct networking_engine
{
    struct sockaddr_in rx_addr;
    struct sockaddr_in tx_addr;
    int sock_id;
    int rx_sock, tx_sock;
    int isUDP;
    int sequential;
    SOCK_TYPE sock_type;

    _Atomic int run;
    _Atomic int kill;
    _Atomic int shutdown;

    pthread_t rx_thread;
    pthread_t tx_thread;

    pthread_mutex_t rx_thread_mutex;
    pthread_cond_t rx_thread_cond;
    pthread_mutex_t tx_thread_mutex;
    pthread_cond_t tx_thread_cond;

    nasm_t* nasm;
    rpool_t* rpool;
    networking_queue_t *tx_queue;
    dse_group_t *dse_group;

    // for multiuser case
    int device_idx;
    unsigned int inference_whitelist[SCHEDULE_MAX_DEVICES];
};



networking_engine* init_networking (nasm_t* nasm, rpool_t* rpool, SOCK_TYPE sock_type, char* ip, int port, int is_UDP, int sequential);
void init_networking_queue (networking_queue_t *networking_queue);
void init_rx(networking_engine* net_engine, int port,int is_UDP);
void init_tx(networking_engine* net_engine, char* ip, int port, int is_UDP);
void add_inference_whitelist (networking_engine *net_engine, unsigned int inference_id);
void remove_inference_whitelist (networking_engine *net_engine, unsigned int inference_id);
int is_inference_whitelist (networking_engine *net_engine, unsigned int inference_id);
void net_engine_wait(networking_engine* net_engine);
void transmission(networking_engine *net_engine);
void receive(networking_engine *net_engine);

void net_queue_reset (networking_queue_t *networking_queue);
void net_engine_reset (networking_engine *net_engine);
void net_engine_stop (networking_engine *net_engine);
void net_engine_run (networking_engine *net_engine);
void net_engine_wait_for_tx_queue_completion (networking_engine *net_engine);

unsigned int pop_ninsts_from_net_queue (networking_queue_t *networking_queue, ninst_t **ninst_ptr_list, char* buffer, unsigned int max_ninsts_to_get);
void push_ninsts_to_net_queue (networking_queue_t *networking_queue, ninst_t *ninst_ptr, unsigned int num_ninsts);
void add_input_rpool (networking_engine *net_engine, nasm_t* nasm, char *input_filename);
void add_input_rpool_reverse (networking_engine *net_engine, nasm_t* nasm, char *input_filename);
void net_engine_destroy(networking_engine* net_engine);
void close_connection(networking_engine* net_engine);

#endif /* _NETWORKING_ */