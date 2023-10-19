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

int main (int argc, char **argv)
{
    print_aspen_build_info();
    
    char* data_dir = NULL;
    int batch_size = 1;
    int number_of_iterations = 15;
    int num_cores = 64;
    int num_tiles = 100;
    char *target_op = NULL;
    int num_layers = 24;
    int width = 64;
    int channels_M = 64;

    if (argc != 10)
    {
        printf ("Usage: %s <data_dir> <batch_size> <num_tiles> <number_of_iterations> <num_cores> <operator> <num_layers> <width> <channels_M>\n", argv[0]);
        exit (0);
    }
    else 
    {
        data_dir = argv[1];
        batch_size = atoi (argv[2]);
        num_tiles = atoi (argv[3]);
        number_of_iterations = atoi (argv[4]);
        num_cores = atoi (argv[5]);
        target_op = argv[6];
        num_layers = atoi (argv[7]);
        width = atoi (argv[8]);
        channels_M = atoi (argv[9]);
    }


    char target_cfg[1024] = {0};
    sprintf (target_cfg, "%s/%s_W%d_CM%d_L%d_T%d.cfg", data_dir, target_op, width, channels_M, num_layers, num_tiles);
    char target_aspen[1024] = {0};
    sprintf (target_aspen, "%s/%s_W%d_CM%d_L%d_T%d.aspen", data_dir, target_op, width, channels_M, num_layers, num_tiles);
    char nasm_file_name [1024] = {0};
    sprintf (nasm_file_name, "%s/%s_W%d_CM%d_L%d_T%d.nasm", data_dir, target_op, width, channels_M, num_layers, num_tiles);

    aspen_dnn_t *target_dnn = apu_create_dnn(target_cfg, NULL);
    apu_save_dnn_to_file (target_dnn, target_aspen);
    nasm_t *target_nasm = NULL;
    if (strcmp (target_op, "conv") == 0)
        target_nasm = apu_create_nasm (target_dnn, num_tiles, batch_size);
    else
        target_nasm = apu_create_transformer_nasm (target_dnn, num_tiles, batch_size, width);
    apu_save_nasm_to_file (target_nasm, nasm_file_name);

    // aspen_dnn_t *target_dnn = apu_load_dnn_from_file (target_aspen);
    // if (target_dnn == NULL)
    // {
    //     printf ("Unable to load dnn file\n");
    //     exit (0);
    // }
    // nasm_t *target_nasm = apu_load_nasm_from_file (nasm_file_name, target_dnn);
    // if (target_nasm == NULL)
    // {
    //     printf ("Unable to load nasm file\n");
    //     exit (0);
    // }

    // print_nasm_info (target_nasm, 1, 0);
  
    rpool_t *rpool = rpool_init ();
    dse_group_t *dse_group = dse_group_init (num_cores);
    dse_group_set_rpool (dse_group, rpool);

    rpool_add_nasm (rpool, target_nasm, NULL);

    printf ("Running %d iterations\n", number_of_iterations);
    double start_time = get_sec();
    for (int i = 0; i < number_of_iterations; i++)
    {
        rpool_reset_queue (rpool);
        rpool_reset_nasm (rpool, target_nasm);
        dse_group_run (dse_group);
        dse_wait_for_nasm_completion (target_nasm);
        dse_group_stop (dse_group);
    }
    double end_time = get_sec();
    printf ("Time taken: %lf seconds\n", (end_time - start_time)/number_of_iterations);
    aspen_flush_dynamic_memory ();

    // if (strcmp(dnn, "bert_base") != 0)
    // {
    //     LAYER_PARAMS output_order[] = {BATCH, OUT_C, OUT_H, OUT_W};
    //     float *layer_output = dse_get_nasm_result (target_nasm, output_order);
    //     float *softmax_output = calloc (1000*batch_size, sizeof(float));
    //     softmax (layer_output, softmax_output, batch_size, 1000);
    //     for (int i = 0; i < batch_size; i++)
    //     {
    //         get_prob_results ("data/imagenet_classes.txt", softmax_output + 1000*i, 1000);
    //     }
    //     free (layer_output);
    //     free (softmax_output);
    // }
    // dse_group_destroy (dse_group);
    // rpool_destroy (rpool);

    apu_destroy_nasm (target_nasm);
    apu_destroy_dnn (target_dnn);
    return 0;
}