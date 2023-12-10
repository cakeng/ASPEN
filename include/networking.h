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

struct networking_engine_t
{
    nasm_t **nasm_list;
    int num_nasms;

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

networking_engine_t* init_networking_engine ();

void net_add_nasm (networking_engine_t *net_engine, nasm_t *nasm);
void net_connect_peer_tcp (aspen_peer_t *peer);
void net_connect_nasm_peers_tcp (nasm_t *nasm);

void send(networking_engine_t *net_engine);
void receive(networking_engine_t *net_engine);

void net_reset (networking_engine_t *net_engine);
void net_stop (networking_engine_t *net_engine);
void net_run (networking_engine_t *net_engine);
void net_wait_for_nasm_completion (networking_engine_t *net_engine);

#endif /* _NETWORKING_ */