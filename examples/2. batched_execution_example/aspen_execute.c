#include "../../include/aspen.h"

// Helper functions

// Get current time in seconds
double get_sec();
// Softmax function
void softmax (float *input, float *output, unsigned int num_batch, unsigned int num_elements);
// Get top 5 results from the softmax output
void get_prob_results (char *class_data_path, float* probabilities, unsigned int num);

int main (int argc, char* argv[])
{
    // Print ASPEN build info
    print_aspen_build_info();
    
    int batch_size = 4;
    int number_of_iterations = 100;
    int num_threads = sysconf(_SC_NPROCESSORS_ONLN);
    char input_tensor_dir[1024] = {0};

    if (argc == 2)
    {
        number_of_iterations = atoi(argv[1]);
    }
    else 
    {
        printf("Usage: ./aspen_execute <number_of_iterations>\n");
        return 0;
    }

    // Load the DNN models. (Generated using aspen_generate.c)
    aspen_dnn_t *resnet50_dnn = apu_load_dnn_from_file ("resnet50.aspen");
    aspen_dnn_t *vgg16_dnn = apu_load_dnn_from_file ("vgg16.aspen");

    // Load the NASMs (ASPEN Graphs). (Generated using aspen_generate.c)
    nasm_t *resnet50_B4_nasm = apu_load_nasm_from_file ("resnet50_B4.nasm", resnet50_dnn);
    nasm_t *vgg16_B1_nasm_1 = apu_load_nasm_from_file ("vgg16_B1.nasm", vgg16_dnn);
    nasm_t *vgg16_B1_nasm_2 = apu_load_nasm_from_file ("vgg16_B1.nasm", vgg16_dnn);

    // Initialize Ready Pool.
    rpool_t *rpool = rpool_init ();
    // Initialize Distributed Scheduling Engines.
    dse_group_t *dse_group = dse_group_init (num_threads);
    // Set the Ready Pool as the target for the Distributed Scheduling Engines.
    dse_group_set_rpool (dse_group, rpool);
    // Add the NASM to the Ready Pool, with input tensor data from the specified directory.
    rpool_add_nasm (rpool, resnet50_B4_nasm, "batched_input.tensor");
    rpool_add_nasm (rpool, vgg16_B1_nasm_1, "dog.tensor");
    rpool_add_nasm (rpool, vgg16_B1_nasm_2, "cat.tensor");

    // Run the DSEs.
    double start_time = get_sec();
    for (int i = 0; i < number_of_iterations; i++)
    {
        // Reset the Ready Pool.
        rpool_reset (rpool);
        // Reset ninst (ASPEN graph node) states.
        rpool_reset_nasm (rpool, resnet50_B4_nasm);
        rpool_reset_nasm (rpool, vgg16_B1_nasm_1);
        rpool_reset_nasm (rpool, vgg16_B1_nasm_2);
        // Run the DSEs until the NASM is completed.
        dse_group_run (dse_group);
        dse_wait_for_nasm_completion (resnet50_B4_nasm);
        dse_wait_for_nasm_completion (vgg16_B1_nasm_1);
        dse_wait_for_nasm_completion (vgg16_B1_nasm_2);
        dse_group_stop (dse_group);
    }
    double end_time = get_sec();
    printf ("Average time taken (%d runs): %3.6f s\n", number_of_iterations, (end_time - start_time)/number_of_iterations);
    
    printf ("Resnet50 (Batch of 4):\n");
    // Get the output tensor data from the NASM in a NCWH order.
    LAYER_PARAMS output_order[] = {BATCH, OUT_C, OUT_H, OUT_W};
    float *layer_output = dse_get_nasm_result (resnet50_B4_nasm, output_order);
    // Apply softmax to the output tensor data.
    float *softmax_output = calloc (1000*batch_size, sizeof(float));
    softmax (layer_output, softmax_output, batch_size, 1000);
    // Get the top 5 results from the softmax output.
    for (int i = 0; i < batch_size; i++)
    {
        printf ("Batch %d results:\n", i+1);
        get_prob_results ("../../files/imagenet_classes.txt", softmax_output + 1000*i, 1000);
    }
    free (layer_output);
    free (softmax_output);

    printf ("VGG-16 (Batch of 2):\n");
    float *layer_outputs[2] = {NULL, NULL};
    // Get the output tensor data from the NASM in a NCWH order.
    layer_outputs[0] = dse_get_nasm_result (vgg16_B1_nasm_1, output_order);
    layer_outputs[1] = dse_get_nasm_result (vgg16_B1_nasm_2, output_order);
    // Apply softmax to the output tensor data.
    softmax_output = calloc (1000*2, sizeof(float));
    softmax (layer_outputs[0], softmax_output, 1, 1000);
    softmax (layer_outputs[1], softmax_output + 1000, 1, 1000);
    // Get the top 5 results from the softmax output.
    for (int i = 0; i < 2; i++)
    {
        printf ("Batch %d results:\n", i+1);
        get_prob_results ("../../files/imagenet_classes.txt", softmax_output + 1000*i, 1000);
    }
    free (layer_outputs[0]);
    free (layer_outputs[1]);
    free (softmax_output);

    // Cleanup
    dse_group_destroy (dse_group);
    rpool_destroy (rpool);
    apu_destroy_nasm (resnet50_B4_nasm);
    apu_destroy_dnn (resnet50_dnn);
    apu_destroy_nasm (vgg16_B1_nasm_1);
    apu_destroy_nasm (vgg16_B1_nasm_2);
    apu_destroy_dnn (vgg16_dnn);
    return 0;
}

// Helper functions implementation

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
        printf ("\t%d: %s - %2.2f%%\n", i+1, buffer[max_idx], max_val*100);
        *(probabilities + max_idx) = -INFINITY;
    }
}