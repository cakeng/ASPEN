#include "networking.h"

void init_networking_queue (networking_queue_t *networking_queue)
{
    // atomic_store (&rpool_queue->occupied, 0);
    pthread_mutex_init (&networking_queue->occupied_mutex, NULL);
    networking_queue->idx_start = 0;
    networking_queue->idx_end = 0;
    networking_queue->max_stored = INIT_QUEUE_SIZE;
    networking_queue->ninst_ptr_arr = calloc (INIT_QUEUE_SIZE, sizeof(ninst_t*));
}

networking_engine* init_networking (SOCK_TYPE sock_type, char* ip, int port, int is_UDP) 
{
    networking_engine *net_engine = calloc (1, sizeof(networking_engine));
    networking_queue_t *networking_queue_t = calloc (1, sizeof(networking_queue_t));
    init_networking_queue(networking_queue_t);

    switch (sock_type)
    {
    case SOCK_TX:
        init_tx(net_engine, ip, port, is_UDP);
        break;
    case SOCK_RX:
        init_rx(net_engine, port, is_UDP);
        break;
    default:
        printf("Error - Unsupported socket type. Type: %d\n", net_engine->sock_type);
        assert(0);
        break;
    }
    return net_engine;
}

void init_socket (networking_engine* net_engine)
{
    printf("SOCK_RX: %d net_engine->type: %d\n", SOCK_RX, net_engine->sock_type);
    switch (net_engine->sock_type)
    {
    case SOCK_TX:
        if(connect(net_engine->tx_sock,(struct sockaddr*)&net_engine->rx_addr, sizeof(net_engine->rx_addr)) == -1)
        {
            int rx_ip = net_engine->rx_addr.sin_addr.s_addr;
            printf("Error - Socket connection error... Rx ip: %d.%d.%d.%d, Rx Port: %d\n", 
                (rx_ip>>0)&0xff, (rx_ip>>8)&0xff, (rx_ip>>16)&0xff, (rx_ip>>24)&0xff, net_engine->rx_addr.sin_port);
            assert (0);
        }
        break;
    case SOCK_RX:
        if(listen(net_engine->rx_sock, 99) == -1)
        {
            int rx_ip = net_engine->rx_addr.sin_addr.s_addr;
            printf("Error - Socket listen error... Rx ip: %d.%d.%d.%d, Rx Port: %d\n", 
                (rx_ip>>0)&0xff, (rx_ip>>8)&0xff, (rx_ip>>16)&0xff, (rx_ip>>24)&0xff, net_engine->rx_addr.sin_port);
            assert(0);
        }
        socklen_t rx_addr_size = sizeof(net_engine->rx_addr);
        net_engine->tx_sock = accept(net_engine->rx_sock, (struct sockaddr*)&net_engine->rx_addr, &rx_addr_size);
        if(net_engine->tx_sock == -1)
        {
            printf("Error! - Socket accept error.\n");
        } else {
            printf("Connect!\n");
        }
        break;
    default:
        printf("Error - Unsupported socket type. Type: %d\n", net_engine->sock_type);
        assert(0);
        break;
    }
}

void init_rx(networking_engine* net_engine, int port,int is_UDP) {

    if (is_UDP != 0) {
        net_engine->rx_sock = socket (PF_INET, SOCK_DGRAM, 0);
    }
    else {
        net_engine->rx_sock = socket (PF_INET, SOCK_STREAM, 0);
    }

    int option = 1;
    setsockopt(net_engine->rx_sock, SOL_SOCKET, SO_REUSEADDR, &option, sizeof(option));
    net_engine->sock_type = SOCK_RX;
    net_engine->tx_sock = 0;
    net_engine->rx_addr.sin_family = AF_INET;
    net_engine->rx_addr.sin_addr.s_addr = htonl (INADDR_ANY);
    net_engine->rx_addr.sin_port = htons (port);
    net_engine->isUDP = is_UDP;

    if(bind(net_engine->rx_sock,(struct sockaddr*)&net_engine->rx_addr, sizeof(net_engine->rx_addr)) == -1)
        printf("ERROR! socket bind error\n");

    init_socket(net_engine);
}

void init_tx(networking_engine* net_engine, char* ip, int port, int is_UDP) {

    bzero (&net_engine->rx_addr, sizeof(net_engine->rx_addr));
    bzero (&net_engine->tx_addr, sizeof(net_engine->tx_addr));

    net_engine->sock_type = SOCK_TX;
    net_engine->tx_sock = socket (PF_INET, SOCK_STREAM, 0);
    net_engine->rx_addr.sin_family = AF_INET;
    net_engine->rx_addr.sin_addr.s_addr = inet_addr (ip);
    net_engine->rx_addr.sin_port = htons (port);
    net_engine->isUDP = 0;

    init_socket(net_engine);
}



void transmission(ninst_t* ninst) {
    if(!ninst->offloaded) {
        return;
    }


}
