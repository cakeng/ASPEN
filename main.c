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

int main(int argc , char *argv[])
{
    print_aspen_build_info();
    
    int number_of_iterations = 20;
    int gpu_idx = -1;
   
    if (argc < 3 || argc > 4)
    {
        printf ("Usage: %s <dnn_name> <batch_size> [seq_len]\n", argv[0]);
        return 0;
    }

    char *dnn_name = argv[1];
    int batch_size = atoi(argv[2]);
    int seq_len = -1;
    if (argc > 3)
        seq_len = atoi(argv[3]);

    char dnn_cfg_path[1024];
    char dnn_data_path[1024];
    sprintf (dnn_cfg_path, "data/cfg/%s_aspen.cfg", dnn_name);
    sprintf (dnn_data_path, "data/%s_data.bin", dnn_name);
    aspen_dnn_t *dnn = apu_create_dnn(dnn_cfg_path, dnn_data_path);
    char nasm_base_path[1024];
    sprintf (nasm_base_path, "nasms/%s_base.aspen", dnn_name);
    apu_save_dnn_to_file (dnn, nasm_base_path);
    nasm_t *nasm;
    printf ("Generating %s NASM for batch size %d", dnn_name, batch_size);
    if (seq_len > 0)
        printf (" and sequence length %d", seq_len);
    printf ("\n");

    if (seq_len > 0)
        nasm = apu_generate_transformer_nasm (dnn, batch_size, seq_len, number_of_iterations, gpu_idx);
    else
        nasm = apu_generate_nasm (dnn, batch_size, number_of_iterations, gpu_idx);

     char nasm_file_name [1024];
    if (seq_len > 0)
        sprintf (nasm_file_name, "nasms/%s_S%d_B%d.nasm", dnn_name, seq_len, batch_size);
    else
        sprintf (nasm_file_name, "nasms/%s_B%d.nasm", dnn_name, batch_size);
    apu_save_nasm_to_file (nasm, nasm_file_name);

    return 0;
}