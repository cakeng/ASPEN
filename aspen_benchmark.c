#include <stdio.h>
#include "aspen.h"

#define USE_CPU  0
#define USE_GPU  1

#define TFC  0
#define TORCHSCRIPT  1
#define XLA  2
#define TORCH 3

#define RESNET  0
#define YOLO  1
#define VGG  2
#define BERT  3
#define BERTLARGE  4
#define GPT 5
//gettimeofday

int main(int argc,char ** argv) {

    int framework = TFC;
    int hardware = USE_CPU;
    int model = RESNET;

    int batch_size = 4;
    int thread_size = 16;
    int loop_time = 1;
    int warm_up = 1;
    int seq_len = 128;

    if (argc > 1)
    {
        //if fisrt argument is cpu or CPU use cpu
        if (strcmp(argv[1], "cpu") == 0 || strcmp(argv[1], "CPU") == 0)
        {
            hardware = USE_CPU;
        }
        else if (strcmp(argv[1], "gpu") == 0 || strcmp(argv[1], "GPU") == 0)
        {
            hardware = USE_GPU;
        }
        else
        {
            printf("hardware must be cpu or gpu\n");
            return 0;
        }
    }

    //second arg is thread size
    if (argc > 2)
    {
        //if second argument is resnet set model
        if(strcmp(argv[2],"RESNET")==0 || strcmp(argv[2],"resnet")==0)
        {
            model = RESNET;
        }
        if(strcmp(argv[2],"YOLO")==0 || strcmp(argv[2],"yolo")==0)
        {
            model = YOLO;
        }
        if(strcmp(argv[2],"VGG")==0 || strcmp(argv[2],"vgg")==0)
        {
            model = VGG;
        }
        if(strcmp(argv[2],"BERT")==0 || strcmp(argv[2],"bert")==0)
        {
            model = BERT;
        }
        if(strcmp(argv[2],"BERTLARGE")==0 || strcmp(argv[2],"bertlarge")==0)
        {
            model = BERTLARGE;
        }
        if(strcmp(argv[2],"GTPT")==0 || strcmp(argv[2],"gpt")==0)
        {
            model = GPT;
        }
    }
    
    //third is loop time
    if (argc > 3)
    {
       //frame work
       if(strcmp(argv[3],"TFC")==0 || strcmp(argv[3],"tfc")==0)
       {
           framework = TFC;
       }
       //XLA
        if(strcmp(argv[3],"XLA")==0 || strcmp(argv[3],"xla")==0)
        {
            framework = XLA;
        }
        //torch
        if(strcmp(argv[3],"TORCH")==0 || strcmp(argv[3],"torch")==0)
        {
            framework = TORCH;
        }
        //torchscript
        if(strcmp(argv[3],"TORCHSCRIPT")==0 || strcmp(argv[3],"torchscript")==0)
        {
            framework = TORCHSCRIPT;
        }
    
    }

    //forth is whether to warm up
    if (argc > 4)
    {
        batch_size = atoi(argv[4]);
    }

    if( argc>5)
    {
        thread_size = atoi(argv[5]);
    }

    if( argc>6)
    {
        seq_len = atoi(argv[6]);
    }

    int gpu = -1;
    if(hardware == USE_GPU)
        gpu = 0;
    nasm_t *resnet50_nasm;

    aspen_dnn_t *resnet50_dnn; 
    char *resnet50_nasm_file_name = (char *)malloc(sizeof(char) * 100);
    ///home/nxc/benchmark/models/aspen/bert_base_S128_B1_CPU.nasm
        //print model name
    
    if(hardware == USE_CPU){
        if(model == RESNET){
            resnet50_dnn = apu_load_dnn_from_file ("../models/aspen/resnet50_base.aspen");
            sprintf(resnet50_nasm_file_name, "../models/aspen/resnet50_B%d_CPU.nasm", batch_size);

        }
        if(model == YOLO){
            resnet50_dnn = apu_load_dnn_from_file ("../models/aspen/yolov3_base.aspen");
            sprintf(resnet50_nasm_file_name, "../models/aspen/yolov3_B%d_CPU.nasm", batch_size);
        }
        if(model == VGG){
            resnet50_dnn = apu_load_dnn_from_file ("../models/aspen/vgg16_base.aspen");
            sprintf(resnet50_nasm_file_name, "../models/aspen/vgg16_B%d_CPU.nasm", batch_size);
        }
        if(model == BERT){
            resnet50_dnn = apu_load_dnn_from_file ("../models/aspen/bert_base.aspen");
            ///home/nxc/benchmark/models/aspen/bert_base_S128_B1_CPU.nasm
            sprintf(resnet50_nasm_file_name, "../models/aspen/bert_base_S%d_B%d_CPU.nasm", seq_len,batch_size);
        }
        if(model == GPT){
            ///home/nxc/benchmark/models/aspen/gpt2_124M_S128_B1_CPU.nasm
            resnet50_dnn = apu_load_dnn_from_file ("../models/aspen/gpt2_124M_base.aspen");
            sprintf(resnet50_nasm_file_name, "../models/aspen/gpt2_124M_S%d_B%d_CPU.nasm", seq_len,batch_size);
        }
        if(model == BERTLARGE){
            ///home/nxc/benchmark/models/aspen/gpt2_124M_S128_B1_CPU.nasm
            resnet50_dnn = apu_load_dnn_from_file ("../models/aspen/bert_large_base.aspen");
            sprintf(resnet50_nasm_file_name, "../models/aspen/bert_large_S%d_B%d_CPU.nasm", seq_len,batch_size);
        }


    }else{
        if(model == RESNET){
            resnet50_dnn = apu_load_dnn_from_file ("../models/aspen/resnet50_base.aspen");
            sprintf(resnet50_nasm_file_name, "../models/aspen/resnet50_B%d_GPU.nasm", batch_size);
        }
        if(model == YOLO){
            resnet50_dnn = apu_load_dnn_from_file ("../models/aspen/yolov3_base.aspen");
            sprintf(resnet50_nasm_file_name, "../models/aspen/yolov3_B%d_GPU.nasm", batch_size);
        }
        if(model == VGG){
            resnet50_dnn = apu_load_dnn_from_file ("../models/aspen/vgg16_base.aspen");
            sprintf(resnet50_nasm_file_name, "../models/aspen/vgg16_B%d_GPU.nasm", batch_size);
        }
        if(model == BERT){
            resnet50_dnn = apu_load_dnn_from_file ("../models/aspen/bert_base.aspen");
            ///home/nxc/benchmark/models/aspen/bert_base_S128_B1_CPU.nasm
            sprintf(resnet50_nasm_file_name, "../models/aspen/bert_base_S%d_B%d_GPU.nasm", seq_len,batch_size);
        }
        if(model == GPT){
            resnet50_dnn = apu_load_dnn_from_file ("../models/aspen/gpt2_124M_base.aspen");
            ///home/nxc/benchmark/models/aspen/gpt2_124M_S128_B1_CPU.nasm
            sprintf(resnet50_nasm_file_name, "../models/aspen/gpt2_124M_S%d_B%d_GPU.nasm", seq_len,batch_size);
        }
        if(model == BERTLARGE){
            resnet50_dnn = apu_load_dnn_from_file ("../models/aspen/bert_large_base.aspen");
            ///home/nxc/benchmark/models/aspen/gpt2_124M_S128_B1_CPU.nasm
            sprintf(resnet50_nasm_file_name, "../models/aspen/bert_large_S%d_B%d_GPU.nasm", seq_len,batch_size);
        }


    }
    printf("model name: %s\n", resnet50_nasm_file_name);
    resnet50_nasm = apu_load_nasm_from_file (resnet50_nasm_file_name, resnet50_dnn);

   
    rpool_t *rpool = rpool_init (gpu);

    dse_group_t *dse_group = dse_group_init (thread_size, gpu);
    dse_group_set_rpool (dse_group, rpool);

    rpool_add_nasm (rpool, resnet50_nasm, "../models/bert_large_weights/bert_large_data.bin");
   
    get_elapsed_time ("init");



    struct timeval start, end;
    gettimeofday(&start, NULL);
    
    
    dse_group_run (dse_group);

    dse_wait_for_nasm_completion (resnet50_nasm);
    

    gettimeofday(&end, NULL);
    // print time in millisecond
    printf("time: %f ms\n", (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0);
    // float time save time in millisecond
    float time = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0;

    get_elapsed_time ("run_aspen");
    FILE *file;
    if (model == RESNET){
        if(hardware == USE_CPU)
            file = fopen("/home/nxc/benchmark/resnet_bench.txt", "a");
        else
            file = fopen("/home/nxc/benchmark/resnet_gpu_bench.txt", "a");
    }
    if (model == VGG)
        file = fopen("/home/nxc/benchmark/vgg_bench.txt", "a");
    if(model == YOLO)
        file = fopen("/home/nxc/benchmark/yolo_bench.txt", "a");
    if (model == BERT){
        if(hardware == USE_CPU)
            file = fopen("/home/nxc/benchmark/bert_bench.txt", "a");
        else   
            file = fopen("/home/nxc/benchmark/bert_gpu_bench.txt", "a");
    }
    if(model == BERTLARGE)
        file = fopen("/home/nxc/benchmark/bertlarge_bench.txt", "a");
    if (model == GPT)
        file = fopen("/home/nxc/benchmark/gpt_bench.txt", "a");

    //add "aspen batchsize threadsize seq_len time"
   //fprintf(file, "aspen %d %d %d %f\n", batch_size, thread_size, warm_up, time);
    //LAYER_PARAMS output_order[] = {BATCH, OUT_H, OUT_W, OUT_C};
    //LAYER_PARAMS output_order[3];
    LAYER_PARAMS* output_order;

    if (model == BERT||BERTLARGE) {
        output_order = (LAYER_PARAMS*) malloc(3 * sizeof(LAYER_PARAMS));
        output_order[0] = BATCH;
        output_order[1] = MAT_N;
        output_order[2] = MAT_M;
    }
    if (model == RESNET || model == YOLO)
    {
        output_order = (LAYER_PARAMS*) malloc(4 * sizeof(LAYER_PARAMS));
        output_order[0] = BATCH;
        output_order[1] = OUT_H;
        output_order[2] = OUT_W;
        output_order[3] = OUT_C;
    }
    if(model == VGG)
    {
        output_order = (LAYER_PARAMS*) malloc(4 * sizeof(LAYER_PARAMS));
        output_order[0] = BATCH;
        output_order[1] = OUT_C;
        output_order[2] = OUT_H;
        output_order[3] = OUT_W;
    }
    if(model == GPT)
    {
        output_order = (LAYER_PARAMS*) malloc(4 * sizeof(LAYER_PARAMS));
        output_order[0] = BATCH;
        output_order[1] = MAT_N;
        output_order[2] = MAT_M;
        output_order[3] = 0;
    }

    float *layer_output =  dse_get_nasm_result(resnet50_nasm,output_order);
    //save batchsize * 1000 float into ~/benchmark/aspen_resnet50_batch(batchsize).bin
    char *output_file_name = (char *)malloc(sizeof(char) * 100);
    if(model==RESNET)
        sprintf(output_file_name, "../output/aspen_resnet50_batch%d_out.bin", batch_size);
    if(model==YOLO)
        sprintf(output_file_name, "../output/aspen_yolov3_batch%d_out.bin", batch_size);
    if(model==VGG)
        sprintf(output_file_name, "../output/aspen_vgg16_batch%d_out.bin", batch_size);
    if(model==BERT)
        sprintf(output_file_name, "../output/aspen_bert_batch%d_seq%d_out.bin", batch_size,seq_len);
    if(model==BERTLARGE)
        sprintf(output_file_name, "../output/aspen_bertlarge_batch%d_seq%d_out.bin", batch_size,seq_len);
    if(model==GPT)
        sprintf(output_file_name, "../output/aspen_gpt_batch%d_seq%d_out.bin", batch_size,seq_len);
    FILE *output_file = fopen(output_file_name, "wb");
    fwrite(layer_output, sizeof(float), 1000*batch_size, output_file);
    fclose(output_file);

    // Write the line to the file.
    // Close the file.
    fprintf(file, "aspen %d %d %d %d %f\n", batch_size, thread_size, warm_up, seq_len, time);
    fclose(file);   
    return 0;
    
}


