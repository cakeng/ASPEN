#ifndef _SERVER_H_
#define _SERVER_H_

#include "aspen.h"
#include "util.h"
#include "scheduling.h"

#define ECHO_CLOSE_SIGNAL (-19960931)
#define PEER_CLOSE_SIGNAL (-19960930)
#define INIT_CLOSE_SIGNAL (-19960929)

typedef struct aspen_server_t
{
    int listen_sock;
    int listen_port;
    int isUDP;

    dse_group_t *dse_group;
    rpool_t *rpool;
    networking_engine_t *networking_engine;

    aspen_dnn_t **dnn_list;
    nasm_t **nasm_list;
    unsigned int num_nasms;

} aspen_server_t;

typedef struct aspen_connection_data_t
{
    HASH_t nasm_hash;
    char dnn_path[MAX_STRING_LEN];
    char nasm_path[MAX_STRING_LEN];
    int num_peers;
    int sender_peer_index;
    int receiver_peer_index;
} aspen_connection_data_t;

void server_add_new_dnn 
    (aspen_server_t *aspen_server, aspen_dnn_t *aspen_dnn, nasm_t *nasm);
nasm_t *get_nasm_from_hash (aspen_server_t *aspen_server, HASH_t nasm_hash);
int get_tcp_listen_sock (int listen_port);
void nasm_close_all_connections (nasm_t *nasm);
void close_tcp_connection (aspen_peer_t *peer, int signal);
int connect_tcp_connection (char *ip, int port);
int accept_tcp_connection (int listen_sock);
void parse_aspen_tcp_connection (int sock);

#endif // _SERVER_H_