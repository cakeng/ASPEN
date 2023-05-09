#include "nasm.h"
#include <sys/socket.h>
#include <arpa/inet.h>

void init_rx(nasm_t* nasm, int port,int is_UDP);
void init_tx(nasm_t* nasm, char* ip, int port, int is_UDP);