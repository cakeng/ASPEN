#ifndef _SCHEDULING_H_
#define _SCHEDULING_H_

#include "nasm.h"
#include "aspen.h"

struct aspen_peer_t
{
    HASH_t peer_hash;
    int sock;  
    int isUDP;
    ssize_t latency_usec;
    ssize_t bandwidth_bps;
    int listen_port;
    char *ip;
    pthread_mutex_t peer_mutex;
    rpool_queue_t *tx_queue;
};

aspen_peer_t *peer_init ();
void peer_copy (aspen_peer_t *new_peer, aspen_peer_t *peer);
void destroy_peer (aspen_peer_t *peer);
void set_ninst_compute_peer_idx (ninst_t *ninst, int peer_idx);
void set_ninst_send_peer_idx (ninst_t *ninst, int peer_idx);
int check_ninst_compute_using_peer_idx (ninst_t *ninst, int peer_idx);
int check_ninst_send_using_peer_idx (ninst_t *ninst, int peer_idx);
int get_peer_idx (nasm_t *nasm, HASH_t peer_hash);
aspen_peer_t *get_peer (nasm_t *nasm, int peer_idx);
void print_peer_info (aspen_peer_t *peer);

void sched_nasm_load_peer_list (nasm_t *nasm, aspen_peer_t **peer_list, int num_peers);
void sched_nasm_send_using_child_compute (nasm_t *nasm);
void sched_ninst_send_using_child_compute (ninst_t *ninst);

// Set local: Computes on local device. Singe peer, no offload.
void sched_set_local (nasm_t *nasm, aspen_peer_t **peer_list, int num_peers);
// Input offload: Peer 0 sends input nodes to peer 1, peer 1 sends the last layer to peer 0.
void sched_set_input_offload (nasm_t *nasm, aspen_peer_t **peer_list, int num_peers);

void sched_set_from_file (nasm_t *nasm, aspen_peer_t **peer_list, int num_peers, char *filename);

#endif