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
    
    int batch_size = 1;
    int number_of_iterations = 100;
    int num_threads = sysconf(_SC_NPROCESSORS_ONLN);
    char input_tensor_dir[1024] = {0};

    if (argc == 3)
    {
        strcpy (input_tensor_dir, argv[1]);
        number_of_iterations = atoi(argv[2]);
    }
    else 
    {
        printf("Usage: ./aspen_execute <input_tensor_dir> <number_of_iterations>\n");
        return 0;
    }

    // Load the DNN model info from a file. (Generated using aspen_generate.c)
    aspen_dnn_t *aspen_dnn = apu_load_dnn_from_file ("resnet50.aspen");
    // Load the NASM (ASPEN Graph) for the DNN model from a file. (Generated using aspen_generate.c)
    nasm_t *aspen_nasm = apu_load_nasm_from_file ("resnet50_B1.nasm", aspen_dnn);
    
    // Initialize Ready Pool.
    rpool_t *rpool = rpool_init ();
    // Initialize Distributed Scheduling Engines.
    dse_group_t *dse_group = dse_group_init (num_threads);
    // Set the Ready Pool as the target for the Distributed Scheduling Engines.
    dse_group_set_rpool (dse_group, rpool);
    // Add the NASM to the Ready Pool, with input tensor data from the specified directory.
    rpool_add_nasm (rpool, aspen_nasm, input_tensor_dir);

    // Run the DSEs.
    double start_time = get_sec();
    for (int i = 0; i < number_of_iterations; i++)
    {
        // Reset the Ready Pool.
        rpool_reset (rpool);
        // Reset ninst (ASPEN graph node) states.
        rpool_reset_nasm (rpool, aspen_nasm);
        // Run the DSEs until the NASM is completed.
        dse_group_run_until_nasm_completion (dse_group, aspen_nasm);
    }
    double end_time = get_sec();
    printf ("Average time taken (%d runs): %3.6f s\n", number_of_iterations, (end_time - start_time)/number_of_iterations);
    
    // Get the output tensor data from the NASM in a NCWH order.
    LAYER_PARAMS output_order[] = {BATCH, OUT_C, OUT_H, OUT_W};
    float *layer_output = dse_get_nasm_result (aspen_nasm, output_order);
    // Apply softmax to the output tensor data.
    float *softmax_output = calloc (1000*batch_size, sizeof(float));
    softmax (layer_output, softmax_output, batch_size, 1000);
    // Get the top 5 results from the softmax output.
    for (int i = 0; i < batch_size; i++)
    {
        printf ("Batch %d results:\n", i+1);
        get_prob_results ("../../files/imagenet_classes.txt", softmax_output + 1000*i, 1000);
    }
    // Cleanup
    free (layer_output);
    free (softmax_output);
    dse_group_destroy (dse_group);
    rpool_destroy (rpool);
    apu_destroy_nasm (aspen_nasm);
    apu_destroy_dnn (aspen_dnn);
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