#include <stdio.h>
#include "aspen.h"
#include "util.h"
#include "nasm.h"
#include "apu.h"
#include "networking.h"


int main(int argc, char **argv)
{
    int sock_type = 999;
    if(argc > 1) {
        if(!strcmp(argv[1], "RX")) {
            sock_type = SOCK_RX;
        } else if(!strcmp(argv[1], "TX")) {
            sock_type = SOCK_TX;
        }
    }

    aspen_dnn_t *resnet50_dnn = apu_create_dnn("data/cfg/resnet50_aspen.cfg", "data/resnet50/resnet50_data.bin");
    int gpu = -1;

    nasm_t *resnet50_nasm = apu_load_nasm_from_file ("data/resnet50_B1_aspen.nasm", resnet50_dnn);
    rpool_t *rpool = rpool_init (gpu);
    dse_group_t *dse_group = dse_group_init (16, gpu);
    dse_group_set_rpool (dse_group, rpool);
    networking_engine* net_engine = NULL;


    if(sock_type == SOCK_RX || sock_type == SOCK_TX) 
    {
        net_engine = init_networking(resnet50_nasm, rpool, sock_type, "127.0.0.1", 8080, 0);
        dse_group_set_net_engine(dse_group, net_engine);
        
        if(sock_type == SOCK_TX) {
            add_input_rpool (net_engine, resnet50_nasm, "data/resnet50/batched_input_64.bin");
        }

        atomic_store (&net_engine->run, 1);
    }
    else { // Local run
        rpool_add_nasm (rpool, resnet50_nasm, 1.0, "data/resnet50/batched_input_64.bin"); 
    }

    get_elapsed_time ("init");
    dse_group_run (dse_group);
    dse_wait_for_nasm_completion (resnet50_nasm);
    get_elapsed_time ("run_aspen");
    dse_group_stop (dse_group);

    if(sock_type == SOCK_RX) {
        for(int i = 0; i < resnet50_nasm->ldata_arr[resnet50_nasm->num_ldata-1].num_ninst; i++)
        {
            ninst_t* ninst = &resnet50_nasm->ldata_arr[resnet50_nasm->num_ldata-1].ninst_arr_start[i];
            pthread_mutex_lock(&net_engine->net_engine_mutex);
            push_ninsts_to_net_queue(net_engine->net_queue, ninst, 1);
            pthread_mutex_unlock(&net_engine->net_engine_mutex);
        }
    }
    
    LAYER_PARAMS output_order[] = {BATCH, OUT_H, OUT_W, OUT_C};
    float *layer_output = dse_get_nasm_result (resnet50_nasm, output_order);
    float *softmax_output = calloc (1000*resnet50_nasm->batch_size, sizeof(float));
    naive_softmax (layer_output, softmax_output, resnet50_nasm->batch_size, 1000);
    for (int i = 0; i < resnet50_nasm->batch_size; i++)
    {
        get_probability_results ("data/resnet50/imagenet_classes.txt", softmax_output + 1000*i, 1000);
    }
    
    free (layer_output);
    free (softmax_output);

    close_connection (net_engine);
    net_engine_destroy (net_engine);
    dse_group_destroy (dse_group);
    rpool_destroy (rpool);
    apu_destroy_nasm (resnet50_nasm);
    apu_destroy_dnn (resnet50_dnn);
    return 0;
}