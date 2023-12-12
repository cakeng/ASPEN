#include "server.h"

void server_add_new_dnn 
    (aspen_server_t *aspen_server, aspen_dnn_t *aspen_dnn, nasm_t *nasm)
{
    aspen_server->num_nasms++;
    aspen_server->nasm_list = realloc (aspen_server->nasm_list, sizeof(nasm_t *) * aspen_server->num_nasms);
    aspen_server->nasm_list[aspen_server->num_nasms - 1] = nasm;
    aspen_server->dnn_list = realloc (aspen_server->dnn_list, sizeof(aspen_dnn_t *) * aspen_server->num_nasms);
    aspen_server->dnn_list[aspen_server->num_nasms - 1] = aspen_dnn;
}

nasm_t *get_nasm_from_hash (aspen_server_t *aspen_server, HASH_t nasm_hash)
{
    for (int i = 0; i < aspen_server->num_nasms; i++)
    {
        if (aspen_server->nasm_list[i]->nasm_hash == nasm_hash)
            return aspen_server->nasm_list[i];
    }
    return NULL;
}

int get_tcp_listen_sock (int listen_port)
{
    int listen_sock = socket (PF_INET, SOCK_STREAM, 0);
    if (listen_sock == -1) 
    {
        ERRNO_PRTF ("Error: socket() returned -1\n");
        assert(0);
    }
    int optval = 1;
    if (setsockopt (listen_sock, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval)) == -1)
    {
        ERRNO_PRTF ("Error: setsockopt() returned -1\n");
        assert(0);
    }
    struct sockaddr_in listen_addr;
    memset (&listen_addr, 0, sizeof(listen_addr));
    listen_addr.sin_family = AF_INET;
    listen_addr.sin_port = htons (listen_port);
    listen_addr.sin_addr.s_addr = htonl (INADDR_ANY);
    if (bind (listen_sock, (struct sockaddr *)&listen_addr, sizeof(listen_addr)) == -1)
    {
        ERRNO_PRTF ("Error: bind() returned -1\n");
        assert(0);
    }
    if (listen (listen_sock, 5) == -1)
    {
        ERRNO_PRTF ("Error: listen() returned -1\n");
        assert(0);
    }
    return listen_sock;
}

int accept_tcp_connection (int listen_sock)
{
    struct sockaddr_in peer_addr;
    socklen_t peer_addr_len = sizeof(peer_addr);
    int sock = accept (listen_sock, (struct sockaddr *)&peer_addr, &peer_addr_len);
    if (sock == -1)
    {
        ERRNO_PRTF ("Error: accept() returned -1\n");
        assert(0);
    }
    return sock;
}

int connect_tcp_connection (char *ip, int port)
{
    int sock = socket (PF_INET, SOCK_STREAM, 0);
    if (sock == -1) 
    {
        ERRNO_PRTF ("Error: socket() returned -1\n");
        assert(0);
    }
    struct sockaddr_in peer_addr;
    memset (&peer_addr, 0, sizeof(peer_addr));
    peer_addr.sin_family = AF_INET;
    peer_addr.sin_port = htons (port);
    peer_addr.sin_addr.s_addr = inet_addr (ip);
    if (connect (sock, (struct sockaddr *)&peer_addr, sizeof(peer_addr)) == -1)
    {
        int counter = 0;
        while (connect (sock, (struct sockaddr *)&peer_addr, sizeof(peer_addr)) == -1)
        {
            if (counter++ > 60)
            {
                printf ("\n");
                ERRNO_PRTF ("Error: connect() returned -1\n");
                assert(0);
            }
            printf ("\rWaiting for connection to %s:%d", ip, port);
            for (int i = 0; i < counter % 4; i++)
                printf (".");
            for (int i = 0; i < 4 - counter % 4; i++)
                printf (" ");
            fflush (stdout);
            usleep (1000000);
        }
        printf ("\n");
    }
    return sock;
}

void nasm_close_all_connections (nasm_t *nasm)
{
    for (int i = 0; i < nasm->num_peers; i++)
    {
        close_tcp_connection (nasm->peer_map[i], INIT_CLOSE_SIGNAL);
    }
}

void close_tcp_connection (aspen_peer_t *peer, int signal)
{
    if (peer->sock != -1)
    {
        if (signal == INIT_CLOSE_SIGNAL)
        {
            signal = PEER_CLOSE_SIGNAL;
            if (write_bytes (peer->sock, &signal, sizeof(signal)) != sizeof(signal))
            {
                ERRNO_PRTF ("Error: write() returned -1\n");
                assert(0);
            }
        }
        else if (signal == PEER_CLOSE_SIGNAL)
        {
            signal = ECHO_CLOSE_SIGNAL;
            if (write_bytes (peer->sock, &signal, sizeof(signal)) != sizeof(signal))
            {
                ERRNO_PRTF ("Error: write() returned -1\n");
                assert(0);
            }
            close (peer->sock);
            peer->sock = -1;
        }
        else if (signal == ECHO_CLOSE_SIGNAL)
        {
            close (peer->sock);
            peer->sock = -1;
        }
        else
        {
            ERRNO_PRTF ("Error: Invalid signal %d\n", signal);
            assert(0);
        }
    }

}

void send_nasm_schedules (nasm_t *nasm, int sock)
{

}

void send_ninst_schedule (ninst_t *ninst, int sock)
{

}

void receive_nasm_schedules (nasm_t *nasm, int sock)
{

}

void receive_ninst_schedule (ninst_t *ninst, int sock)
{

}

// void parse_aspen_tcp_connection (aspen_server_t *server, int sock)
// {
//     // Read connection data;
//     aspen_connection_data_t conn_data;
//     if (read_bytes (sock, &conn_data, sizeof(conn_data)) != sizeof(conn_data))
//     {
//         ERRNO_PRTF ("Error: read() returned -1\n");
//         assert(0);
//     }

//     nasm_t target_nasm = get_nasm_from_hash (server, conn_data.nasm_hash);
//     if (nasm == NULL)
//     {
//         aspen_dnn_t *target_dnn = apu_load_dnn_from_file (target_aspen);
//         if (target_dnn == NULL)
//         {
//             printf ("Unable to load ASPEN DNN weight file %s\n", target_aspen);
//             exit (1);
//         }
//         target_nasm = apu_load_nasm_from_file (nasm_file_name, target_dnn);
//         if (target_nasm == NULL)
//         {
//             printf ("Unable to load ASPEN graph file %s\n", nasm_file_name);
//             exit (1);
//         }
//         server_add_new_dnn (server, target_nasm, target_dnn);
//     }

// }

// void aspen_server_routine 
//     (dse_group_t *dse_group, rpool_t *rpool, networking_engine_t *networking_engine, int isUDP, int listen_port)
// {
//     if (dse_group == NULL)
//     {
//         ERROR_PRTF ("ERROR: dse_group is NULL.");
//         assert(0);
//     }
//     if (rpool == NULL)
//     {
//         ERROR_PRTF ("ERROR: rpool is NULL.");
//         assert(0);
//     }
//     if (listen_port <= 0)
//     {
//         ERROR_PRTF ("ERROR: listen_port is invalid.");
//         assert(0);
//     }

//     aspen_server_t server;
//     server.listen_port = listen_port;
//     server.isUDP = isUDP;
//     server.listen_sock = -1;
//     server.networking_engine = networking_engine;
//     server.dse_group = dse_group;
//     server.rpool = rpool;

//     if (isUDP)
//     {
//         ERROR_PRTF ("WARNING: UDP is not supported yet.\n");
//         assert(0);
//     }
//     else
//         server.listen_sock = get_tcp_listen_sock (listen_port);
    


//     dse_group_run (dse_group);
//     while (1)
//     {

//     }
//     dse_group_stop (dse_group);
// }