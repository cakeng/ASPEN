#ifndef _NETWORKING_H_
#define _NETWORKING_H_

#include "aspen.h"
#include "nasm.h"
#include "apu.h"
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>


#define INIT_QUEUE_SIZE 1024

struct networking_engine
{
    nasm_t* nasm;
    rpool_t* rpool;
    networking_queue_t *net_queue;

    pthread_mutex_t net_engine_mutex;
    pthread_cond_t net_engine_cond;
    
    struct sockaddr_in rx_addr;
    struct sockaddr_in tx_addr;
    int sock_id;
    int rx_sock, tx_sock;
    int isUDP;
    SOCK_TYPE sock_type;

    _Atomic int run;
    _Atomic int kill;

    pthread_t thread;
};

struct networking_queue_t
{
    // _Atomic unsigned int occupied;
    // pthread_mutex_t occupied_mutex;
    unsigned int idx_start;
    unsigned int idx_end;
    unsigned int num_stored;
    unsigned int max_stored;
    ninst_t **ninst_ptr_arr;
};

networking_engine* init_networking (nasm_t* nasm, rpool_t* rpool, SOCK_TYPE sock_type, char* ip, int port, int is_UDP);
void init_networking_queue (networking_queue_t *networking_queue);
void init_rx(networking_engine* net_engine, int port,int is_UDP);
void init_tx(networking_engine* net_engine, char* ip, int port, int is_UDP);
void net_engine_wait(networking_engine* net_engine);
void transmission(networking_engine *net_engine);
void receive(networking_engine *net_engine);

#endif /* _NETWORKING_ */