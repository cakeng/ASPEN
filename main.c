#include <stdio.h>
#include "aspen.h"
#include "util.h"
#include "nasm.h"
#include "apu.h"

//gettimeofday
int main(int argc,char ** argv) {

    int batch_size = 1;
    if (argc > 1)
    {
        batch_size = atoi(argv[1]);
    }
    int thread_size = 64;
    if (argc > 2)
    {
        thread_size = atoi(argv[2]);
    }
    

    //third is loop time
    int loop_time = 1;
    if (argc > 3)
    {
        loop_time = atoi(argv[3]);
    }


    //forth is whether to warm up
    int warm_up = 0;
    if (argc > 4)
    {
        warm_up = atoi(argv[4]);
    
    }

    int gpu = -1;
    nasm_t *bert_nasm;
    nasm_t *resnet50_nasm;

    aspen_dnn_t *resnet50_dnn = apu_load_dnn_from_file ("/home/nxc/resnet50_weights/resnet50_bdse.aspen");
    char *resnet50_nasm_file_name = (char *)malloc(sizeof(char) * 100);
    sprintf(resnet50_nasm_file_name, "/home/nxc/resnet50_weights/resnet50_B%d.nasm", batch_size);
        //print model name
    printf("model name: %s\n", resnet50_nasm_file_name);
    resnet50_nasm = apu_load_nasm_from_file (resnet50_nasm_file_name, resnet50_dnn);

   
    rpool_t *rpool = rpool_init (gpu);

    dse_group_t *dse_group = dse_group_init (thread_size, gpu);
    dse_group_set_rpool (dse_group, rpool);

    rpool_add_nasm (rpool, resnet50_nasm, "data/batched_input_128.bin");
   
    get_elapsed_time ("init");

    // dse_cudagraph_run (rpool, bert_nasm);
    //measure time

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

    file = fopen("/home/nxc/benchmark/resnet_bench.txt", "a");
    //add "aspen batchsize threadsize seq_len time"
   //fprintf(file, "aspen %d %d %d %f\n", batch_size, thread_size, warm_up, time);
    //LAYER_PARAMS output_order[] = {BATCH, OUT_H, OUT_W, OUT_C};
    LAYER_PARAMS output_order[] = {BATCH,OUT_H, OUT_W, OUT_C};
    float *layer_output =  dse_get_nasm_result(resnet50_nasm,output_order);

    float *softmax_output = calloc (1000 * batch_size, sizeof(float));
    naive_softmax (layer_output, softmax_output, resnet50_nasm->batch_size, 1000);
    // float *layer_output = get_aspen_tensor_data ((resnet50_dnn->layers + resnet50_dnn->num_layers - 1)->tensors[OUTPUT_TENSOR], output_order);
    // print_float_array (out, 1000*resnet50_nasm->batch_size, 1000);
    // print_float_tensor(out, resnet50_nasm->batch_size, 3, 224, 224);
    for (int i = 0; i < batch_size; i++)
    {
       get_probability_results ("/home/nxc/benchmark/tfc/imagenet_classes.txt", softmax_output + 1000*i, 1000);
    }
    // Write the line to the file.
    // Close the file.
    fclose(file);   
    return 0;
    
}


