#ifndef _NETWORKING_H_
#define _NETWORKING_H_

#include "aspen.h"
#include "nasm.h"
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>


#define INIT_QUEUE_SIZE 1024

struct networking_engine
{
    networking_queue_t *net_queue;
    // For offloading
    struct sockaddr_in rx_addr;
    struct sockaddr_in tx_addr;
    int sock_id;
    int rx_sock, tx_sock;
    int isUDP;
    SOCK_TYPE sock_type;
};

struct networking_queue_t
{
    // _Atomic unsigned int occupied;
    pthread_mutex_t occupied_mutex;
    unsigned int idx_start;
    unsigned int idx_end;
    unsigned int num_stored;
    unsigned int max_stored;
    ninst_t **ninst_ptr_arr;
};

networking_engine* init_networking (SOCK_TYPE sock_type, char* ip, int port, int is_UDP);
void init_networking_queue (networking_queue_t *networking_queue);
void init_rx(networking_engine* net_engine, int port,int is_UDP);
void init_tx(networking_engine* net_engine, char* ip, int port, int is_UDP);

#endif /* _NETWORKING_ */