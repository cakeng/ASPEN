#include "aspen.h"

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
    
    int batch_size = 4;
    int number_of_iterations = 1;
    int num_cores = 32;

    aspen_dnn_t *aspen_dnn = apu_create_dnn("data/cfg/resnet50_aspen.cfg", "data/resnet50/resnet50_data.bin");
    apu_save_dnn_to_file (aspen_dnn, "data/resnet50_base.aspen");
    nasm_t *aspen_nasm = apu_generate_nasm (aspen_dnn, batch_size, 20);
    char nasm_file_name [1024] = {0};
    sprintf (nasm_file_name, "data/resnet50_B%d.nasm", batch_size);
    apu_save_nasm_to_file (aspen_nasm, nasm_file_name);

    rpool_t *rpool = rpool_init ();
    dse_group_t *dse_group = dse_group_init (num_cores);
    dse_group_set_rpool (dse_group, rpool);
    rpool_add_nasm (rpool, aspen_nasm, 1.0, "data/batched_input_128.bin");

    double start_time = get_sec();
    for (int i = 0; i < number_of_iterations; i++)
    {
        rpool_reset_nasm (rpool, aspen_nasm, 1.0);
        dse_group_run_until_nasm_completion (dse_group, aspen_nasm);
    }
    double end_time = get_sec();
    printf ("Time taken: %lf seconds\n", (end_time - start_time)/number_of_iterations);
    
    LAYER_PARAMS output_order[] = {BATCH, OUT_C, OUT_H, OUT_W};
    float *layer_output = dse_get_nasm_result (aspen_nasm, output_order);
    float *softmax_output = calloc (1000*batch_size, sizeof(float));
    softmax (layer_output, softmax_output, batch_size, 1000);
    for (int i = 0; i < batch_size; i++)
    {
        get_prob_results ("data/imagenet_classes.txt", softmax_output + 1000*i, 1000);
    }
    free (layer_output);
    free (softmax_output);

    dse_group_destroy (dse_group);
    rpool_destroy (rpool);
    apu_destroy_nasm (aspen_nasm);
    apu_destroy_dnn (aspen_dnn);
    return 0;
}