#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
 
#define DATASIZE 602112

long int elapsed; 
struct timeval start_time, end_time;
void error_handling(char *message);

int main(int argc, char **argv)
{
    int serv_sock;
    int clnt_sock;
    char message[DATASIZE];
    int str_len;

    struct sockaddr_in serv_addr;
    struct sockaddr_in clnt_addr;
    int clnt_addr_size;

    if(argc != 2) {
        printf("Usage : &s <port>\n", argv[0]);
        exit(1);
    }

    serv_sock = socket(PF_INET, SOCK_STREAM, 0);
    if(serv_sock == -1)
        error_handling("socket() error");

    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    serv_addr.sin_port = htons(atoi(argv[1]));

    
    if( bind(serv_sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr) )==-1)
        error_handling("bind() error");

    if(listen(serv_sock, 5) == -1)
        error_handling("listen() error");

    clnt_addr_size = sizeof(clnt_addr);
        
    
    while(clnt_sock = accept(serv_sock, (struct sockaddr*)&clnt_addr, &clnt_addr_size)) {
        printf("catched!\n");
        if(clnt_sock == -1)
            error_handling("accept() error");

        gettimeofday(&start_time, NULL);
        str_len = recv(clnt_sock, message, DATASIZE, MSG_WAITALL);
        gettimeofday(&end_time, NULL);

        break;
    }
    close(clnt_sock);

    elapsed = (end_time.tv_sec * 1000000 + end_time.tv_usec) - (start_time.tv_sec * 1000000 + start_time.tv_usec);
    printf("time elapsed: %ld\n", elapsed);

    return 0;
}
 
void error_handling(char *message)
{
        fputs(message, stderr);
        fputc('\n', stderr);
        exit(1);
}
