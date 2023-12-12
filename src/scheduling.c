#include "scheduling.h"

aspen_peer_t *peer_init ()
{
    aspen_peer_t *peer = calloc (1, sizeof(aspen_peer_t));
    pthread_mutex_init (&peer->peer_mutex, NULL);
    peer->peer_hash = get_unique_hash ();
    peer->ip = calloc (MAX_STRING_LEN, sizeof(char));
    peer->listen_port = 0;
    peer->sock = -1;
    peer->isUDP = -1;
    peer->latency_usec = -1;
    peer->bandwidth_bps = -1;
    peer->tx_queue = calloc (1, sizeof(rpool_t));
    rpool_init_queue (peer->tx_queue);
    return peer;
}

void peer_copy (aspen_peer_t *new_peer, aspen_peer_t *peer)
{
    if (new_peer == NULL || peer == NULL)
    {
        ERROR_PRTF ("ERROR: new_peer or peer is NULL.");
        assert(0);
    }
    new_peer->peer_hash = peer->peer_hash;
    strcpy (new_peer->ip, peer->ip);
    new_peer->listen_port = peer->listen_port;
    new_peer->sock = peer->sock;
    new_peer->isUDP = peer->isUDP;
    new_peer->latency_usec = peer->latency_usec;
    new_peer->bandwidth_bps = peer->bandwidth_bps;
}

void destroy_peer (aspen_peer_t *peer)
{
    if (peer == NULL)
        return;
    pthread_mutex_destroy (&peer->peer_mutex);
    rpool_destroy_queue (peer->tx_queue);
    if (peer->sock >= 0)
        close (peer->sock);
    if (peer->ip != NULL)
        free (peer->ip);
    free (peer);
}

void set_peer_info (aspen_peer_t *peer, char *ip, int port, int isUDP)
{
    if (peer == NULL)
    {
        ERROR_PRTF ("ERROR: peer is NULL.");
        assert(0);
    }
    if (ip == NULL)
    {
        ERROR_PRTF ("ERROR: ip is NULL.");
        assert(0);
    }
    if (port <= 0)
    {
        ERROR_PRTF ("ERROR: port %d is invalid.", port);
        assert(0);
    }
    if (isUDP != 0 && isUDP != 1)
    {
        ERROR_PRTF ("ERROR: isUDP %d is invalid.", isUDP);
        assert(0);
    }
    strncpy (peer->ip, ip, MAX_STRING_LEN);
    peer->listen_port = port;
    peer->isUDP = isUDP;
}

void set_ninst_compute_peer_idx (ninst_t *ninst, int peer_idx)
{
    if (ninst == NULL)
    {
        ERROR_PRTF ("ERROR: ninst is NULL.");
        assert(0);
    }
    if (peer_idx < 0 || peer_idx >= ninst->ldata->nasm->num_peers)
    {
        ERROR_PRTF ("ERROR: peer_idx %d is out of range [0, %d].", peer_idx, ninst->ldata->nasm->num_peers);
        assert(0);
    }
    ninst->peer_flag[peer_idx] |= PEER_FLAG_COMPUTE;
}

void set_ninst_send_peer_idx (ninst_t *ninst, int peer_idx)
{
    if (ninst == NULL)
    {
        ERROR_PRTF ("ERROR: ninst is NULL.");
        assert(0);
    }
    if (peer_idx < 0 || peer_idx >= ninst->ldata->nasm->num_peers)
    {
        ERROR_PRTF ("ERROR: peer_idx %d is out of range [0, %d].", peer_idx, ninst->ldata->nasm->num_peers);
        assert(0);
    }
    ninst->peer_flag[peer_idx] |= PEER_FLAG_SEND;
}

int check_ninst_compute_using_peer_idx (ninst_t *ninst, int peer_idx)
{
    #ifdef DEBUG
    if (ninst == NULL)
    {
        ERROR_PRTF ("ERROR: ninst is NULL.");
        assert(0);
    }
    if (peer_idx >= ninst->ldata->nasm->num_peers)
    {
        ERROR_PRTF ("ERROR: peer_idx %d is out of range [0, %d].", peer_idx, ninst->ldata->nasm->num_peers);
        assert(0);
    }
    #endif
    if (ninst->peer_flag == NULL || peer_idx < 0)
        return 0;
    return ninst->peer_flag[peer_idx] & PEER_FLAG_COMPUTE;

}

int check_ninst_send_using_peer_idx (ninst_t *ninst, int peer_idx)
{
    #ifdef DEBUG
    if (ninst == NULL)
    {
        ERROR_PRTF ("ERROR: ninst is NULL.");
        assert(0);
    }
    if (peer_idx >= ninst->ldata->nasm->num_peers)
    {
        ERROR_PRTF ("ERROR: peer_idx %d is out of range [0, %d].", peer_idx, ninst->ldata->nasm->num_peers);
        assert(0);
    }
    #endif
    if (ninst->peer_flag == NULL || peer_idx < 0)
        return 0;
    return ninst->peer_flag[peer_idx] & PEER_FLAG_SEND;
}

int get_peer_idx (nasm_t *nasm, HASH_t peer_hash)
{
    #ifdef DEBUG
    if (nasm == NULL)
    {
        ERROR_PRTF ("ERROR: nasm is NULL.");
        assert(0);
    }
    #endif
    for (int i = 0; i < nasm->num_peers; i++)
    {
        if (nasm->peer_map[i]->peer_hash == peer_hash)
            return i;
    }
    return -1;
}

aspen_peer_t *get_peer (nasm_t *nasm, int peer_idx)
{
    #ifdef DEBUG
    if (nasm == NULL)
    {
        ERROR_PRTF ("ERROR: nasm is NULL.");
        assert(0);
    }
    if (peer_idx < 0 || peer_idx >= nasm->num_peers)
    {
        ERROR_PRTF ("ERROR: peer_idx %d is out of range [0, %d].", peer_idx, nasm->num_peers);
        assert(0);
    }
    #endif
    return nasm->peer_map[peer_idx];
}

void print_peer_info (aspen_peer_t *peer)
{
    #ifdef DEBUG
    if (peer == NULL)
    {
        ERROR_PRTF ("ERROR: peer is NULL.");
        assert(0);
    }
    #endif
    printf ("\tPeer Hash: %08lx\n", peer->peer_hash);
    printf ("\tIP: %s\n", peer->ip);
    printf ("\tPort: %d\n", peer->listen_port);
    printf ("\tSocket: %d\n", peer->sock);
    printf ("\tisUDP: %s\n", peer->isUDP? "True": "False");
    printf ("\tLatency: %ld usec\n", peer->latency_usec);
    printf ("\tBandwidth: %ld bps\n", peer->bandwidth_bps);
}

void sched_nasm_send_using_child_compute (nasm_t *nasm)
{
    if (nasm == NULL)
    {
        ERROR_PRTF ("Invalid arguments to sched_nasm_send_using_child_compute: nasm is NULL");
        assert (0);
    }
    for (int i = 0; i < nasm->num_ninst; i++)
    {
        ninst_t *ninst = nasm->ninst_arr + i;
        sched_ninst_send_using_child_compute (ninst);
    }
}

void sched_ninst_send_using_child_compute (ninst_t *ninst)
{
    if (ninst == NULL)
    {
        ERROR_PRTF ("Invalid arguments to sched_ninst_send_set_using_compute: ninst is NULL");
        assert (0);
    }
    int num_peers = ninst->ldata->nasm->num_peers;
    for (int i = 0; i < num_peers; i++)
    {
        for (int j = 0; j < ninst->num_child_ninsts; j++)
        {
            ninst_t *child = ninst->child_ninst_arr[j];
            if (check_ninst_compute_using_peer_idx (child, i))
                set_ninst_send_peer_idx (ninst, i);
        }
    }
}

void sched_ninst_init_peer_data (ninst_t *ninst, int num_peers)
{
    if (ninst == NULL)
    {
        ERROR_PRTF ("Invalid arguments to sched_ninst_init_peer_data: ninst is NULL");
        assert (0);
    }
    if (num_peers <= 0)
    {
        ERROR_PRTF ("Invalid arguments to sched_ninst_init_peer_data: num_peers <= 0");
        assert (0);
    }
    ninst->peer_flag = calloc (num_peers, sizeof(char));
}

void sched_nasm_load_peer_list (nasm_t *nasm, aspen_peer_t **peer_list, int num_peers)
{
    if (num_peers <= 0 || peer_list == NULL)
    {
        ERROR_PRTF ("Invalid arguments to sched_nasm_load_peer_list: num_peers <= 0 || peer_list == NULL");
        assert (0);
    }
    nasm->num_peers = num_peers;
    nasm->peer_map = calloc (num_peers, sizeof(aspen_peer_t*));
    for (int i = 0; i < num_peers; i++)
    {
        nasm->peer_map[i] = peer_init ();
        peer_copy (nasm->peer_map[i], peer_list[i]);
    }
    for (int i = 0; i < nasm->num_ninst; i++)
    {
        ninst_t *ninst = nasm->ninst_arr + i;
        sched_ninst_init_peer_data (ninst, num_peers);
    }
}

void sched_set_local (nasm_t *nasm, aspen_peer_t **peer_list, int num_peers)
{
    if (num_peers != 1 || peer_list == NULL)
    {
        ERROR_PRTF ("Invalid arguments to sched_set_local: num_peers != 1 || peer_list == NULL");
        assert (0);
    }
    sched_nasm_load_peer_list (nasm, peer_list, num_peers);
    for (int i = 0; i < nasm->num_ninst; i++)
    {
        ninst_t *ninst = nasm->ninst_arr + i;
        set_ninst_compute_peer_idx (ninst, 0);
    }
    sched_nasm_send_using_child_compute (nasm);
}

void sched_set_input_offload (nasm_t *nasm, aspen_peer_t **peer_list, int num_peers)
{
    if (num_peers != 2 || peer_list == NULL)
    {
        ERROR_PRTF ("Invalid arguments to sched_set_input_offload: num_peers != 2 || peer_list == NULL");
        assert (0);
    }
    sched_nasm_load_peer_list (nasm, peer_list, num_peers);
    for (int i = 0; i < nasm->num_ninst; i++)
    {
        ninst_t *ninst = nasm->ninst_arr + i;
        set_ninst_compute_peer_idx (ninst, 1);
    }
    sched_nasm_send_using_child_compute (nasm);
    for (int i = 0; i < nasm->ldata_arr[0].num_ninst; i++)
    {
        ninst_t *ninst = nasm->ldata_arr[0].ninst_arr_start + i;
        set_ninst_compute_peer_idx (ninst, 0);
    }
    for (int i = 0; i < nasm->ldata_arr[nasm->num_ldata - 1].num_ninst; i++)
    {
        ninst_t *ninst = nasm->ldata_arr[nasm->num_ldata - 1].ninst_arr_start + i;
        set_ninst_send_peer_idx (ninst, 0);
    }
}

void sched_set_from_file (nasm_t *nasm, aspen_peer_t **peer_list, int num_peers, char *filename);

