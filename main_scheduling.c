#include <stdio.h>
#include "aspen.h"
#include "util.h"
#include "nasm.h"
#include "apu.h"
#include "networking.h"
#include "scheduling.h"
#include "profiling.h"

int main(int argc, char **argv)
{
    int sock_type = 999;
    int sequential = 0;

    if (argc > 2) {
        sequential = atoi(argv[2]);
    }
    if (argc > 1) 
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
    // ninst_profile[sock_type] = profile_computation(target_config, target_bin, target_nasm_dir, target_input, gpu, 1);
    ninst_profile[sock_type] = load_computation_profile("./data/vgg16_B1_comp_profile.bin");
    // save_computation_profile(ninst_profile[sock_type], "data/vgg16_B1_comp_profile.bin");

    
    /** STAGE: PROFILING NETWORK **/

    printf("STAGE: PROFILING NETWORK\n");

    char *rx_ip = "192.168.1.176";
    int rx_port = 3786;

    int server_sock;
    int client_sock;
    
    if (sock_type == SOCK_RX) {
        server_sock = create_server_sock(rx_ip, rx_port+1);
        client_sock = accept_client_sock(server_sock);
    }
    else if (sock_type == SOCK_TX) {
        server_sock = connect_server_sock(rx_ip, rx_port+1);
    }

    network_profile = profile_network(ninst_profile, sock_type, server_sock, client_sock);

    int connection_key;
    if (sock_type == SOCK_RX) {
        connection_key = 12534;
        write_n(client_sock, &connection_key, sizeof(int));
        printf("connection key: %d\n", connection_key);
    }
    else if (sock_type == SOCK_TX) {
        connection_key = -1;
        read_n(server_sock, &connection_key, sizeof(int));
        printf("connection key: %d\n", connection_key);
    }

    printf("sync: %f\n", network_profile->sync);
    
    
    /** STAGE: SCHEDULING - HEFT **/

    printf("STAGE: SCHEUDLING - HEFT\n");

    sched_processor_t *schedule;
    if (sock_type == SOCK_RX) {
        schedule = init_heft(target_config, target_bin, target_nasm_dir, ninst_profile, network_profile, 2);
        save_schedule(schedule, 2, "./temp_sched.txt");
    }
    
    share_schedule(&schedule, 2, sock_type, server_sock, client_sock);


    /** STAGE: INFERENCE **/

    char* file_name;
    if(sequential) file_name = sock_type == SOCK_RX ? "./logs/scheduled/sequential_ninst_time_logs_RX.csv" : "./logs/scheduled/sequential_ninst_time_logs_TX.csv";
    else file_name = sock_type == SOCK_RX ? "./logs/scheduled/pipeline_ninst_time_logs_RX.csv" : "./logs/scheduled/pipeline_ninst_time_logs_TX.csv";
    
    FILE *log_fp = fopen(file_name, "w");

    aspen_dnn_t *vgg16_dnn = apu_create_dnn(target_config, target_bin);
    nasm_t *vgg16_nasm = apu_load_nasm_from_file (target_nasm_dir, vgg16_dnn);

    aspen_dnn_t *target_dnn = vgg16_dnn;
    nasm_t *target_nasm = vgg16_nasm;

    apply_schedule_to_nasm(target_nasm, schedule, 2, sock_type);

    rpool_t *rpool = rpool_init (gpu);
    dse_group_t *dse_group = dse_group_init (8, gpu);
    dse_group_set_rpool (dse_group, rpool);
    networking_engine* net_engine = NULL;

    if(sock_type == SOCK_RX || sock_type == SOCK_TX) 
    {
        net_engine = init_networking(target_nasm, rpool, sock_type, rx_ip, rx_port, 0, sequential);
        dse_group_set_net_engine(dse_group, net_engine);
        dse_group_set_device(dse_group, sock_type);
        net_engine->dse_group = dse_group;
        
        if(sock_type == SOCK_TX) {
            add_input_rpool (net_engine, target_nasm, target_input);
        }

        atomic_store (&net_engine->run, 1);
    }
    else { // Local run
        rpool_add_nasm (rpool, target_nasm, 1.0, target_input); 
    }
    
    get_elapsed_time ("init");
    if (!sequential || sock_type == SOCK_TX) dse_group_run (dse_group);
    dse_wait_for_nasm_completion (target_nasm);
    get_elapsed_time ("run_aspen");
    dse_group_stop (dse_group);

    if(sock_type == SOCK_RX) {
        for(int i = 0; i < target_nasm->ldata_arr[target_nasm->num_ldata-1].num_ninst; i++)
        {
            ninst_t* ninst = &target_nasm->ldata_arr[target_nasm->num_ldata-1].ninst_arr_start[i];
            pthread_mutex_lock(&net_engine->net_engine_mutex);
            push_ninsts_to_net_queue(net_engine->net_queue, ninst, 1);
            pthread_mutex_unlock(&net_engine->net_engine_mutex);
        }
    }
    
    LAYER_PARAMS output_order[] = {BATCH, OUT_H, OUT_W, OUT_C};
    float *layer_output = dse_get_nasm_result (target_nasm, output_order);
    float *softmax_output = calloc (1000*target_nasm->batch_size, sizeof(float));
    naive_softmax (layer_output, softmax_output, target_nasm->batch_size, 1000);
    for (int i = 0; i < target_nasm->batch_size; i++)
    {
        get_probability_results ("data/resnet50/imagenet_classes.txt", softmax_output + 1000*i, 1000);
    }
    
    free (layer_output);
    free (softmax_output);

    close_connection (net_engine);
    save_ninst_log(log_fp, target_nasm);
    net_engine_destroy (net_engine);
    dse_group_destroy (dse_group);
    rpool_destroy (rpool);
    apu_destroy_nasm (target_nasm);
    apu_destroy_dnn (target_dnn);



    /** STAGE: FINISH **/
    if (sock_type == SOCK_RX) {
        close(client_sock);
        close(server_sock);
    }
    else if (sock_type == SOCK_TX) {
        close(server_sock);
    }

    

    return 0;
}
