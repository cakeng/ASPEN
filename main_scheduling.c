#include <stdio.h>
#include "aspen.h"
#include "util.h"
#include "nasm.h"
#include "apu.h"
#include "networking.h"
#include "scheduling.h"

ninst_profile_t *profile_computation(char *target_config, char *target_bin, char *target_nasm_dir, char *target_input, int gpu);
network_profile_t *profile_network(ninst_profile_t **ninst_profile, int sock_type, char *rx_ip, int rx_port);

int main(int argc, char **argv)
{
    int sock_type = 999;

    if(argc > 1) 
    {
        if(!strcmp(argv[1], "RX")) {
            sock_type = SOCK_RX;
        } else if(!strcmp(argv[1], "TX")) {
            sock_type = SOCK_TX;
        }
    }
    else {
        printf("usage: %s [RX/TX]\n", argv[0]);
        exit(0);
    }

    // char *target_config = "data/cfg/resnet50_aspen.cfg";
    // char *target_bin = "data/resnet50/resnet50_data.bin";
    // char *target_nasm_dir = "data/resnet50_B1_aspen.nasm";
    // char *target_nasm_dir = "data/resnet50_B32_fine_aspen.nasm";
    // char* target_input = "data/resnet50/batched_input_64.bin";

    char *target_config = "data/cfg/vgg16_aspen.cfg";
    char *target_bin = "data/vgg16/vgg16_data.bin";
    char *target_nasm_dir = "data/vgg16_B1_aspen.nasm";
    char *target_input = NULL;

    int gpu = -1;

    // nasm_t *resnet50_nasm = apu_load_nasm_from_file ("data/resnet50_B1_aspen.nasm", resnet50_dnn);
    // nasm_t *resnet50_nasm = apu_load_nasm_from_file ("data/resnet50_B32_fine_aspen.nasm", resnet50_dnn);
    // nasm_t *resnet50_nasm = apu_create_nasm(resnet50_dnn, 1e6, 200, 32);
    // nasm_t *vgg16_nasm = apu_create_nasm(vgg16_dnn, 1e6, 8, 1);
    // apu_save_nasm_to_file(resnet50_nasm, "data/resnset50_B32_fine_aspen.nasm");
    // apu_save_nasm_to_file(vgg16_nasm, "data/vgg16_B1_aspen.nasm");

    ninst_profile_t *ninst_profile[SCHEDULE_MAX_DEVICES];
    network_profile_t *network_profile;

    /** STAGE: PROFILING COMPUTATION **/

    printf("STAGE: PROFILING COMPUTATION\n");
    ninst_profile[sock_type] = profile_computation(target_config, target_bin, target_nasm_dir, target_input, gpu);

    
    /** STAGE: PROFILING NETWORK **/

    printf("STAGE: PROFILING NETWORK\n");

    char *rx_ip = "192.168.1.176";
    int rx_port = 3786;

    network_profile = profile_network(ninst_profile, sock_type, rx_ip, rx_port+1);
    

    /** STAGE: INFERENCE **/


    return 0;
}

ninst_profile_t *profile_computation(char *target_config, char *target_bin, char *target_nasm_dir, char *target_input, int gpu) {
    aspen_dnn_t *target_dnn = apu_create_dnn(target_config, target_bin);
    nasm_t *target_nasm = apu_load_nasm_from_file (target_nasm_dir, target_dnn);

    rpool_t *rpool = rpool_init (gpu);
    dse_group_t *dse_group = dse_group_init (16, gpu);
    dse_group_set_rpool (dse_group, rpool);
    dse_group_set_profile (dse_group, 1);
    networking_engine* net_engine = NULL;

    rpool_add_nasm (rpool, target_nasm, 1.0, target_input); 
    
    dse_group_run (dse_group);
    dse_wait_for_nasm_completion (target_nasm);
    dse_group_stop (dse_group);
    
    LAYER_PARAMS output_order[] = {BATCH, OUT_H, OUT_W, OUT_C};
    float *layer_output = dse_get_nasm_result (target_nasm, output_order);
    float *softmax_output = calloc (1000*target_nasm->batch_size, sizeof(float));
    naive_softmax (layer_output, softmax_output, target_nasm->batch_size, 1000);

    ninst_profile_t *my_profile = extract_profile_from_ninsts(target_nasm);
    
    free (layer_output);
    free (softmax_output);

    close_connection (net_engine);
    net_engine_destroy (net_engine);
    dse_group_destroy (dse_group);
    rpool_destroy (rpool);
    apu_destroy_nasm (target_nasm);
    apu_destroy_dnn (target_dnn);

    return my_profile;
}

network_profile_t *profile_network(ninst_profile_t **ninst_profile, int sock_type, char *rx_ip, int rx_port) {
    network_profile_t *network_profile = malloc(sizeof(network_profile_t));
    
    const int num_repeat = 4;
    int num_ninst = ninst_profile[sock_type]->total;

    if (sock_type == SOCK_RX) { // echo
        printf("\tprofiling as RX...\n");
        int server_sock;
        int client_sock;

        struct sockaddr_in server_addr;
        struct sockaddr_in client_addr;
        
        int client_addr_size;
        
        // open server
        server_sock = socket(PF_INET, SOCK_STREAM, 0);
        if (server_sock == -1) {
            printf("Error: socket() returned -1\n");
            assert(0);
        }

        memset(&server_addr, 0, sizeof(server_addr));
        server_addr.sin_family = AF_INET;
        server_addr.sin_addr.s_addr = htonl(INADDR_ANY);
        server_addr.sin_port = htons(rx_port);

        if (bind(server_sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) == -1) {
            printf("Error: bind() returned -1\n");
            assert(0);
        }

        if (listen(server_sock, 5) == -1) {
            printf("Error: listen() returned -1\n");
            assert(0);
        }

        client_addr_size = sizeof(client_addr);
        client_sock = accept(server_sock, (struct sockaddr*)&client_addr, &client_addr_size);
        if (client_sock == -1) {
            printf("Error: accept() returned -1\n");
            assert(0);
        }

        // echo shortmessage
        for (int i=0; i<num_repeat; i++) {
            float buf;
            read_n(client_sock, &buf, sizeof(float));
            buf = get_time_secs();
            write_n(client_sock, &buf, sizeof(float));
        }

        // receive & send ninst_profile
        ninst_profile[!sock_type] = malloc(num_ninst * sizeof(ninst_profile_t));
        read_n(client_sock, ninst_profile[!sock_type], num_ninst * sizeof(ninst_profile_t));
        write_n(client_sock, ninst_profile[sock_type], num_ninst * sizeof(ninst_profile_t));
        
        // receive network_profile
        read_n(client_sock, network_profile, sizeof(network_profile_t));

        close(client_sock);
        close(server_sock);
    }
    else {
        printf("\tprofiling as TX...\n");
        int server_sock;
        struct sockaddr_in server_addr;

        // connect to server
        server_sock = socket(PF_INET, SOCK_STREAM, 0);
        if (server_sock == -1) {
            printf("Error: socket() returned -1\n");
            assert(0);
        }

        memset(&server_addr, 0, sizeof(server_addr));
        server_addr.sin_family = AF_INET;
        server_addr.sin_addr.s_addr = htonl(INADDR_ANY);
        server_addr.sin_port = htons(rx_port);

        if (connect(server_sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) == -1) {
            printf("Error: socket() returned -1\n");
            assert(0);
        }

        // send shortmessage
        float send_timestamp[num_repeat];
        float server_timestamp[num_repeat];
        float recv_timestamp[num_repeat];

        float sync = 0;
        float rtt = 0;

        for (int i=0; i<num_repeat; i++) {
            send_timestamp[i] = get_time_secs();
            write_n(server_sock, &send_timestamp[i], sizeof(float));
            read_n(server_sock, &server_timestamp[i], sizeof(float));
            recv_timestamp[i] = get_time_secs();

            sync += server_timestamp[i] - (recv_timestamp[i] + send_timestamp[i]) / 2;
            rtt += recv_timestamp[i] - send_timestamp[i];

        }

        sync /= num_repeat;
        rtt /= num_repeat;

        // send & receive ninst_profile;
        float long_send_timestamp;
        float long_recv_timestamp;
        float transmit_rate;

        ninst_profile[!sock_type] = malloc(num_ninst * sizeof(ninst_profile_t));
        long_send_timestamp = get_time_secs();
        write_n(server_sock, ninst_profile[sock_type], num_ninst * sizeof(ninst_profile_t));
        read_n(server_sock, ninst_profile[!sock_type], num_ninst * sizeof(ninst_profile_t));
        long_recv_timestamp = get_time_secs();

        transmit_rate = num_ninst * sizeof(ninst_profile_t) / ((long_recv_timestamp - long_send_timestamp) / 2);


        // send network_profile
        network_profile->rtt = rtt;
        network_profile->sync = sync;
        network_profile->transmit_rate = transmit_rate;

        write_n(server_sock, network_profile, sizeof(network_profile_t));

        close(server_sock);
    }

    return network_profile;
}