#include "aspen.h"
#include "apu.h"
#include "nasm.h"
#include "dse.h"

double get_sec()
{
    struct timeval now;
    gettimeofday (&now, NULL);
    return now.tv_sec + now.tv_usec*1e-6;
}

void softmax (float *input, float *output, unsigned int num_batch, unsigned int num_elements)
{
    for (int i = 0; i < num_batch; i++)
    {
        float max = input[i * num_elements];
        for (int j = 1; j < num_elements; j++)
        {
            if (input[i * num_elements + j] > max)
                max = input[i * num_elements + j];
        }
        float sum = 0;
        for (int j = 0; j < num_elements; j++)
        {
            output[i * num_elements + j] = expf (input[i * num_elements + j] - max);
            sum += output[i * num_elements + j];
        }
        for (int j = 0; j < num_elements; j++)
            output[i * num_elements + j] /= sum;
    }
}

void get_prob_results (char *class_data_path, float* probabilities, unsigned int num)
{
    int buffer_length = 256;
    char buffer[num][buffer_length];
    FILE *fptr = fopen(class_data_path, "r");
    if (fptr == NULL)
        assert (0);
    for (int i = 0; i < num; i++)
    {
        void *tmp = fgets(buffer[i], buffer_length, fptr);
        if (tmp == NULL)
            assert (0);
        for (char *ptr = buffer[i]; *ptr != '\0'; ptr++)
        {
            if (*ptr == '\n')
            {
                *ptr = '\0';
            }
        }
    }
    fclose(fptr);
    printf ("Results:\n");
    for (int i = 0; i < 5; i++)
    {
        float max_val = -INFINITY;
        int max_idx = 0;
        for (int j = 0; j < num; j++)
        {
            if (max_val < *(probabilities + j))
            {
                max_val = *(probabilities + j);
                max_idx = j;
            }
        }
        printf ("%d: %s - %2.2f%%\n", i+1, buffer[max_idx], max_val*100);
        *(probabilities + max_idx) = -INFINITY;
    }
}

int main(void)
{
    print_aspen_build_info();
    
    int batch_size = 1;
    int number_of_iterations = 4;
    int num_cores = 32;
    int gpu_idx = 0;

    aspen_dnn_t *resnet50_dnn = apu_create_dnn("data/cfg/resnet50_aspen.cfg", "data/resnet50_data.bin");
    apu_save_dnn_to_file (resnet50_dnn, "data/resnet50_base.aspen");
    // nasm_t *resnet50_nasm = apu_generate_nasm (resnet50_dnn, batch_size, 10, 0);
    // nasm_t *resnet50_nasm = apu_generate_nasm (resnet50_dnn, 1, 15, 0);
    char nasm_file_name [1024] = {0};
    // sprintf (nasm_file_name, "data/resnet50_B%d_GPU.nasm", batch_size);
    // apu_save_nasm_to_file (resnet50_nasm, nasm_file_name);

    // nasm_t *resnet50_4_nasm = apu_generate_nasm (resnet50_dnn, 4, 15, 0);
    // sprintf (nasm_file_name, "data/resnet50_B%d_GPU.nasm", 4);
    // apu_save_nasm_to_file (resnet50_4_nasm, nasm_file_name);

    // nasm_t *resnet50_32_nasm = apu_generate_nasm (resnet50_dnn, 32, 15, 0);
    // sprintf (nasm_file_name, "data/resnet50_B%d_GPU.nasm", 32);
    // apu_save_nasm_to_file (resnet50_32_nasm, nasm_file_name);

    // nasm_t *resnet50_128_nasm = apu_generate_nasm (resnet50_dnn, 128, 15, 0);
    // nasm_t *resnet50_128_nasm = apu_create_nasm (resnet50_dnn, 50, 128);
    // sprintf (nasm_file_name, "data/resnet50_B%d_GPU.nasm", 128);
    // apu_save_nasm_to_file (resnet50_128_nasm, nasm_file_name);

    // aspen_dnn_t *vgg16_dnn = apu_create_dnn("data/cfg/vgg16_aspen.cfg", "data/vgg16_data.bin");
    // apu_save_dnn_to_file (vgg16_dnn, "data/vgg16_base.aspen");
    // nasm_t *vgg16_nasm = apu_generate_nasm (vgg16_dnn, batch_size, 15, 0);
    // sprintf (nasm_file_name, "data/vgg16_B%d_GPU.nasm", batch_size);
    // apu_save_nasm_to_file (vgg16_nasm, nasm_file_name);

    // aspen_dnn_t *resnet50_dnn = apu_load_dnn_from_file ("data/resnet50_base.aspen");
    nasm_t *resnet50_nasm = apu_load_nasm_from_file ("data/resnet50_B1_GPU.nasm", resnet50_dnn);
    nasm_t *resnet50_4_nasm = apu_load_nasm_from_file ("data/resnet50_B4_GPU.nasm", resnet50_dnn);
    nasm_t *resnet50_128_nasm = apu_load_nasm_from_file ("data/resnet50_B128_GPU.nasm", resnet50_dnn);
    // aspen_dnn_t *vgg16_dnn = apu_load_dnn_from_file ("data/vgg16_base.aspen");
    // nasm_t *vgg16_nasm = apu_load_nasm_from_file ("data/vgg16_B1.nasm", vgg16_dnn);


    rpool_t *rpool = rpool_init (gpu_idx);
    dse_group_t *dse_group = dse_group_init (num_cores, gpu_idx);
    dse_group_set_rpool (dse_group, rpool);
    rpool_add_nasm (rpool, resnet50_128_nasm, "data/batched_input_128.bin");
    // rpool_add_nasm (rpool, resnet50_4_nasm, "data/batched_input_128.bin");
    // rpool_add_nasm (rpool, vgg16_nasm, "data/batched_input_128.bin");

    // print_nasm_info (resnet50_nasm, 0, 0);
    // print_rpool_info (rpool);

    double start_time = get_sec();
    for (int i = 0; i < number_of_iterations; i++)
    {
        rpool_reset (rpool);
        rpool_reset_nasm (rpool, resnet50_128_nasm);
        // rpool_reset_nasm (rpool, resnet50_4_nasm);
        // rpool_reset_nasm (rpool, vgg16_nasm);
        dse_group_run (dse_group);
        dse_wait_for_nasm_completion (resnet50_128_nasm);
        // dse_wait_for_nasm_completion (resnet50_4_nasm);
        // dse_wait_for_nasm_completion (vgg16_nasm);
        dse_group_stop (dse_group);
    }
    double end_time = get_sec();
    printf ("Time taken: %lf seconds\n", (end_time - start_time)/number_of_iterations);
    
    // int i = 72;
    // // gpt2_dnn->layers[7].tensors[WEIGHT_TENSOR]->data = aspen_calloc (1600*1600,4);
    // unsigned int input_params[NUM_PARAM_ELEMENTS] = {0};
    // // input_params[BATCH] = 1; input_params[NUM_SEQ] = 128; input_params[NUM_HIDDEN] = 768;
    // input_params[BATCH] = 1; input_params[OUT_C] = 3; input_params[OUT_H] = 224; input_params[OUT_W] = 224;
    // // void *dog_data = aspen_load_input ("data/batched_input_128.bin", input_params, sizeof(float));
    // void *dog_data = aspen_load_input_NHWC ("data/batched_input_128.bin", input_params, sizeof(float));
    // aspen_init_naive (resnet50_dnn, input_params, dog_data, -1);
    // get_elapsed_time ("init_naive");
    // aspen_run_naive (resnet50_dnn, input_params, dog_data, -1);
    // get_elapsed_time ("run_naive");
    // printf ("\tLayer %d - Type %s\n", i, layer_type_str[resnet50_dnn->layers[i].type]);
    // aspen_layer_t *layer = &resnet50_dnn->layers[i];
    // nasm_ldata_t *ldata = &resnet50_nasm->ldata_arr[i];
    // assert (ldata->layer == layer);
    // LAYER_PARAMS output_order[] = {BATCH, MAT_N, MAT_M, 0};
    // LAYER_PARAMS output_order_nhwc[] = {BATCH, OUT_H, OUT_W, OUT_C};
    // LAYER_PARAMS output_order_nchw[] = {BATCH, OUT_C, OUT_H, OUT_W};
    // if (layer->type == K_ATTENTION_LAYER)
    // {
    //     output_order [1] = NUM_HEAD; 
    //     output_order [2] = MAT_N; 
    //     output_order [3] = MAT_M;
    // }
    // void *layer_output = get_aspen_tensor_data 
    //     (layer->tensors[OUTPUT_TENSOR], output_order_nchw, -1);
    // void *ldata_output = get_ldata_output (ldata, output_order_nchw);

    // compare_float_tensor (layer_output, ldata_output, 
    //     input_params[BATCH], layer->params[OUT_C], layer->params[OUT_H],
    //     layer->params[OUT_W], 1e-2, 1e-4, 20);


    printf ("Resnet50:\n");
    LAYER_PARAMS output_order[] = {BATCH, OUT_C, OUT_H, OUT_W};
    float *layer_output = dse_get_nasm_result (resnet50_nasm, output_order);
    float *softmax_output = calloc (1000*batch_size, sizeof(float));
    softmax (layer_output, softmax_output, batch_size, 1000);
    for (int i = 0; i < batch_size; i++)
    {
        get_prob_results ("data/imagenet_classes.txt", softmax_output + 1000*i, 1000);
    }
    free (layer_output);
    free (softmax_output);

    // printf ("Resnet50_4:\n");
    // layer_output = dse_get_nasm_result (resnet50_4_nasm, output_order);
    // softmax_output = calloc (1000*4, sizeof(float));
    // softmax (layer_output, softmax_output, 4, 1000);
    // for (int i = 0; i < 4; i++)
    // {
    //     get_prob_results ("data/imagenet_classes.txt", softmax_output + 1000*i, 1000);
    // }
    // free (layer_output);
    // free (softmax_output);

    // printf ("VGG16:\n");
    // layer_output = dse_get_nasm_result (vgg16_nasm, output_order);
    // softmax_output = calloc (1000*batch_size, sizeof(float));
    // softmax (layer_output, softmax_output, batch_size, 1000);
    // for (int i = 0; i < batch_size; i++)
    // {
    //     get_prob_results ("data/imagenet_classes.txt", softmax_output + 1000*i, 1000);
    // }
    // free (layer_output);
    // free (softmax_output);

    dse_group_destroy (dse_group);
    rpool_destroy (rpool);
    apu_destroy_nasm (resnet50_nasm);
    // apu_destroy_nasm (resnet50_4_nasm);
    apu_destroy_dnn (resnet50_dnn);
    // apu_destroy_nasm (vgg16_nasm);
    // apu_destroy_dnn (vgg16_dnn);
    return 0;
}