#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <time.h>
 
#define DATASIZE 602112

long int elapsed;
struct timeval start_time, end_time;
 
void error_handling(char *message);
 
int main(int argc, char **argv)
{
    int sock;
    struct sockaddr_in serv_addr;
    char message[DATASIZE] = "hello!\n";
    int str_len;

    if(argc != 3) {
        printf("Usage : %s <IP> <port> \n", argv[0]);
        exit(1);
    }

    sock = socket(PF_INET, SOCK_STREAM, 0);
    if(sock == -1)
        error_handling("socket() error");

    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr(argv[1]);
    serv_addr.sin_port = htons(atoi(argv[2]));

    if( connect(sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) == -1)
        error_handling("connect() error");

    gettimeofday(&start_time, NULL);

    send(sock, message, DATASIZE, 0);
    gettimeofday(&end_time, NULL);
    
    elapsed = (end_time.tv_sec * 1000000 + end_time.tv_usec) - (start_time.tv_sec * 1000000 + start_time.tv_usec);
    printf("time elapsed: %ld\n", elapsed);
    
    close(sock);
    return 0;
}
void error_handling(char *message)
{
    fputs(message, stderr);
    fputc('\n', stderr);
    exit(1);
}
