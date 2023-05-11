#include "networking.h"

void init_socket (nasm_t* nasm)
{
    switch (nasm->dnn->sock_type)
    {
    case SOCK_TX:
        if(connect(nasm->dnn->tx_sock,(struct sockaddr*)&nasm->dnn->rx_addr, sizeof(nasm->dnn->rx_addr)) == -1)
        {
            int rx_ip = nasm->dnn->rx_addr.sin_addr.s_addr;
            printf("Error - Socket connection error... Rx ip: %d.%d.%d.%d, Rx Port: %d\n", 
                (rx_ip>>0)&0xff, (rx_ip>>8)&0xff, (rx_ip>>16)&0xff, (rx_ip>>24)&0xff, nasm->dnn->rx_addr.sin_port);
            assert (0);
        }
        break;
    case SOCK_RX:
        if(listen(nasm->dnn->rx_sock, 99) == -1)
        {
            int rx_ip = nasm->dnn->rx_addr.sin_addr.s_addr;
            printf("Error - Socket listen error... Rx ip: %d.%d.%d.%d, Rx Port: %d\n", 
                (rx_ip>>0)&0xff, (rx_ip>>8)&0xff, (rx_ip>>16)&0xff, (rx_ip>>24)&0xff, nasm->dnn->rx_addr.sin_port);
            assert(0);
        }
        break;
    default:
        printf("Error - Unsupported socket type. Type: %d\n", nasm->dnn->sock_type);
        assert(0);
        break;
    }
    printf("Connect!\n");
}

void init_rx(nasm_t* nasm, int port,int is_UDP) {

    if (is_UDP != 0) {
        nasm->dnn->rx_sock = socket (PF_INET, SOCK_DGRAM, 0);
    }
    else {
        nasm->dnn->rx_sock = socket (PF_INET, SOCK_STREAM, 0);
    }

    int option = 1;
    setsockopt(nasm->dnn->rx_sock, SOL_SOCKET, SO_REUSEADDR, &option, sizeof(option));
    nasm->dnn->tx_sock = 0;
    nasm->dnn->rx_addr.sin_family = AF_INET;
    nasm->dnn->rx_addr.sin_addr.s_addr = htonl (INADDR_ANY);
    nasm->dnn->rx_addr.sin_port = htons (port);
    nasm->dnn->isUDP = is_UDP;

    if(bind(nasm->dnn->rx_sock,(struct sockaddr*)&nasm->dnn->rx_addr, sizeof(nasm->dnn->rx_addr)) == -1)
        printf("ERROR! socket bind error\n");

    init_socket(nasm);
}

void init_tx(nasm_t* nasm, char* ip, int port, int is_UDP) {

    bzero (&nasm->dnn->rx_addr, sizeof(nasm->dnn->rx_addr));
    bzero (&nasm->dnn->tx_addr, sizeof(nasm->dnn->tx_addr));

    nasm->dnn->tx_sock = socket (PF_INET, SOCK_STREAM, 0);
    nasm->dnn->rx_addr.sin_family = AF_INET;
    nasm->dnn->rx_addr.sin_addr.s_addr = inet_addr (ip);
    nasm->dnn->rx_addr.sin_port = htons (port);
    nasm->dnn->isUDP = 0;

    init_socket(nasm);
}
