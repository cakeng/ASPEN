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
    // print_aspen_build_info();
    aspen_dnn_t *resnet50_dnn = apu_create_dnn("data/cfg/resnet50_aspen.cfg", "data/resnet50/resnet50_data.bin");
    // aspen_dnn_t *yolov3_dnn = apu_create_dnn("data/cfg/yolov3.cfg", "data/yolov3_data.bin");
    // aspen_dnn_t *bert_dnn = apu_create_dnn ("data/cfg/bert_base_encoder.cfg", "data/bert_base_data.bin");
    // aspen_dnn_t *gpt2_dnn = apu_create_dnn ("data/cfg/gpt2_124M_encoder.cfg", "data/gpt2/gpt2_124M_data.bin");
    // // // if (bert_dnn == NULL) 
    // // // {
    // // //     printf("Error: Failed to create DNN\n");
    // // //     return -1;
    // // // }
    // // print_dnn_info (gpt2_dnn, 0);
    // apu_save_dnn_to_file (gpt2_dnn, "data/gpt2/gpt2_124M_base.aspen");
    // aspen_dnn_t *gpt2_dnn = apu_load_dnn_from_file ("data/gpt2_base.aspen");
    // // if (bert_dnn_2 == NULL) 
    // // {
    // //     printf("Error: Failed to read DNN\n");
    // //     return -1;
    // // }
    // // // print_dnn_info (bert_dnn_2, 0);
    
    // nasm_t *bert_nasm = apu_create_transformer_nasm(bert_dnn, 1e6, 100, 1, 480);
    // nasm_t *gpt2_nasm = apu_create_transformer_nasm(gpt2_dnn, 10e6, 100, 1, 128);
    // nasm_t *resnet50_nasm = apu_create_nasm(resnet50_dnn, 100e6, 100, 32);
    // nasm_t *yolov3_nasm = apu_create_nasm(yolov3_dnn, 100e6, 100, 1);
    // // if (bert_nasm == NULL) 
    // // {
    // //     printf("Error: Failed to create NASM\n");
    // //     return -1;
    // // }
    // print_nasm_info(gpt2_nasm, 1, 0);
    // char nasm_file_name [1024] = "data/resnet50_B1_aspen.nasm";
    // sprintf (nasm_file_name, "data/gpt2_124M_S%d_B%d_M%d_%2.1e.nasm"
    //     , gpt2_nasm->tr_seq_len, gpt2_nasm->batch_size, gpt2_nasm->min_ninst_per_ldata,
    //     (double)gpt2_nasm->flop_per_ninst);
    // sprintf (nasm_file_name, "data/yolov3_B%d_M%d_%2.1e.nasm",
    //     yolov3_nasm->batch_size, yolov3_nasm->min_ninst_per_ldata,
    //     (double)yolov3_nasm->flop_per_ninst);
    // apu_save_nasm_to_file (resnet50_nasm, nasm_file_name);
    int gpu = -1;
    // aspen_dnn_t *bert_dnn = apu_load_dnn_from_file ("data/bert_base.aspen");
    // nasm_t *bert_nasm = apu_load_nasm_from_file ("data/bert_S128_B8.nasm", bert_dnn);
    // aspen_dnn_t *resnet50_dnn = apu_load_dnn_from_file ("data/resnet50_base.aspen");
    // nasm_t *resnet50_nasm = apu_load_nasm_from_file ("data/resnet50_aspen.nasm", resnet50_dnn);
    nasm_t *resnet50_nasm = apu_load_nasm_from_file ("data/resnet50_B1_aspen.nasm", resnet50_dnn);
    // aspen_dnn_t *yolov3_dnn = apu_load_dnn_from_file ("data/yolov3_base.aspen");
    // nasm_t *yolov3_nasm = apu_load_nasm_from_file ("data/yolov3_B1_M100_1.0e+08.nasm", yolov3_dnn);
    // // // // nasm_t *bert_4_nasm = apu_load_nasm_from_file ("data/bert_4.nasm", &bert_dnn);
    // // 
    rpool_t *rpool = rpool_init (gpu);
    dse_group_t *dse_group = dse_group_init (64, gpu);
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
            push_ninsts_to_net_queue(net_engine->net_queue, ninst, 1);
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

    dse_group_destroy (dse_group);
    rpool_destroy (rpool);
    net_engine_destroy (net_engine);    
    apu_destroy_nasm (resnet50_nasm);
    apu_destroy_dnn (resnet50_dnn);

    return 0;
}