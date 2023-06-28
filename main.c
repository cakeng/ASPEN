#include <stdio.h>
#include "aspen.h"
#include "util.h"
#include "nasm.h"
#include "apu.h"
#include "networking.h"
#include "scheduling.h"

int total_transferred = 0;

int main(int argc, char **argv)
{
    int sequential = 0;
    int sock_type = 999;
    if(argc > 2)
    {
        sequential = atoi(argv[2]);
    }

    if(argc > 1) 
    {
        if(!strcmp(argv[1], "RX")) {
            sock_type = SOCK_RX;
        } else if(!strcmp(argv[1], "TX")) {
            sock_type = SOCK_TX;
        }
        if(sequential > 0) printf("Offloading mode: [Sequential]\n");
        else printf("Offloading mode: [Pipelining]\n");
    }
    
    char* file_name;
    if(sequential) file_name = sock_type == SOCK_RX ? "./logs/sequential_ninst_time_logs_RX.csv" : "./logs/sequential_ninst_time_logs_TX.csv";
    else file_name = sock_type == SOCK_RX ? "./logs/pipeline_ninst_time_logs_RX.csv" : "./logs/pipeline_ninst_time_logs_TX.csv";
    
    
    FILE *log_fp = fopen(file_name, "w");

    aspen_dnn_t *resnet50_dnn = apu_create_dnn("data/cfg/resnet50_aspen.cfg", "data/resnet50/resnet50_data.bin");
    // aspen_dnn_t *vgg16_dnn = apu_create_dnn("data/cfg/vgg16_aspen.cfg", "data/vgg16/vgg16_data.bin");
    int gpu = -1;

    nasm_t *resnet50_nasm = apu_load_nasm_from_file ("data/resnet50_B1_aspen.nasm", resnet50_dnn);
    // nasm_t *resnet50_nasm = apu_load_nasm_from_file ("data/resnet50_B32_fine_aspen.nasm", resnet50_dnn);
    // nasm_t *vgg16_nasm = apu_load_nasm_from_file ("data/vgg16_B1_aspen.nasm", vgg16_dnn);
    // nasm_t *resnet50_nasm = apu_create_nasm(resnet50_dnn, 1e6, 200, 32);
    // nasm_t *vgg16_nasm = apu_create_nasm(vgg16_dnn, 1e6, 8, 1);
    // apu_save_nasm_to_file(resnet50_nasm, "data/resnset50_B32_fine_aspen.nasm");
    // apu_save_nasm_to_file(vgg16_nasm, "data/vgg16_B1_aspen.nasm");
    aspen_dnn_t *target_dnn;
    nasm_t *target_nasm;
    char* target_input = "data/resnet50/batched_input_64.bin";
    // char *target_input = NULL;

    target_dnn = resnet50_dnn;
    target_nasm = resnet50_nasm;

    init_partial_offload(target_nasm, 0.05);
    rpool_t *rpool = rpool_init (gpu);
    dse_group_t *dse_group = dse_group_init (8, gpu);
    dse_group_set_rpool (dse_group, rpool);
    networking_engine* net_engine = NULL;


    if(sock_type == SOCK_RX || sock_type == SOCK_TX) 
    {
        net_engine = init_networking(target_nasm, rpool, sock_type, "192.168.1.176", 3786, 0, sequential);
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

    printf("total transferred: %d\n", total_transferred);
    return 0;
}
