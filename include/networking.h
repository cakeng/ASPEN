#ifndef _NETWORKING_H_
#define _NETWORKING_H_

#include "aspen.h"
#include "nasm.h"
#include "apu.h"
#include "rpool.h"
#include "server.h"
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/ioctl.h>

#define NETWORK_BUFFER (64*1024*1024) // 64MB

struct networking_engine_t
{
    nasm_t **nasm_list;
    int num_nasms;

    rpool_t *rpool;

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
    
    void* buffer;
};

networking_engine_t* init_net_engine ();
void net_engine_destroy (networking_engine_t *net_engine);

void net_engine_set_rpool (networking_engine_t *net_engine, rpool_t *rpool);
void net_engine_add_nasm (networking_engine_t *net_engine, nasm_t *nasm);

void net_engine_send(networking_engine_t *net_engine);
void net_engine_receive(networking_engine_t *net_engine);

void net_engine_reset (networking_engine_t *net_engine);
void net_engine_stop (networking_engine_t *net_engine);
void net_engine_run (networking_engine_t *net_engine);
void net_engine_wait_for_nasm_completion (networking_engine_t *net_engine);

#endif /* _NETWORKING_ */