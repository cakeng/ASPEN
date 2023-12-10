#include "aspen.h"
#include "dse.h"

#define RX 0 
#define TX 1

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
            if (*ptr == '\n')
                *ptr = '\0';
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
    
    // 1. Parse command line arguments

    char dnn[256] = {0};
    int batch_size = 4;
    int number_of_iterations = 10;
    int num_cores = 32;
    int num_seq = -1;

    char *rx_ip = "127.0.0.1";
    int rx_port = 12345;
    int mode = RX;

    if (argc == 5)
    {
        strcpy (dnn, argv[1]);
        batch_size = atoi (argv[2]);
        number_of_iterations = atoi (argv[3]);
        num_cores = atoi (argv[4]);
    }
    else if (argc == 6)
    {
        strcpy (dnn, argv[1]);
        batch_size = atoi (argv[2]);
        number_of_iterations = atoi (argv[3]);
        num_cores = atoi (argv[4]);
        num_seq = atoi (argv[5]);
    }
    else
    {
        printf ("Usage: %s <dnn> <batch_size> <number_of_iterations> <num_cores> <num_seq>\n", argv[0]);
        printf ("Only enter <num_seq> for transformer-based DNNs.\n");
        printf ("Example: \"%s resnet50 4 20 4\", \"%s bert_base 4 20 4 128\"\n", argv[0], argv[0]);
        exit (0);
    }

    char target_aspen[1024] = {0};
    sprintf (target_aspen, "data/%s/%s.aspen", dnn, dnn);
    char nasm_file_name [1024] = {0};
    if (num_seq == -1)
        sprintf (nasm_file_name, "data/%s/%s_B%d.nasm", dnn, dnn, batch_size);
    else
        sprintf (nasm_file_name, "data/%s/%s_S%d_B%d.nasm", dnn, dnn, batch_size, num_seq);


    // 2. Load the ASPEN DNN weight file (.aspen file)
    aspen_dnn_t *target_dnn = apu_load_dnn_from_file (target_aspen);
    if (target_dnn == NULL)
    {
        printf ("Unable to load ASPEN DNN weight file %s\n", target_aspen);
        exit (1);
    }

    // 3. Create or load the ASPEN graph (.nasm file)

    // 3-1. Generate the ASPEN graph (nasm)

    // nasm_t *target_nasm = NULL;
    // if (num_seq == -1)
    //     target_nasm = apu_create_nasm (target_dnn, 20, batch_size);
    // else
    //     target_nasm = apu_create_transformer_nasm (target_dnn, 50, batch_size, num_seq);
    // apu_save_nasm_to_file (target_nasm, nasm_file_name);

    // 3-2. Load the ASPEN graph (nasm)

    nasm_t *target_nasm = apu_load_nasm_from_file (nasm_file_name, target_dnn);
    if (target_nasm == NULL)
    {
        printf ("Unable to load ASPEN graph file %s\n", nasm_file_name);
        exit (1);
    }
  
    // 3. Initialize the ASPEN DSEs and Ready Pool

    rpool_t *rpool = rpool_init ();
    dse_group_t *dse_group = dse_group_init (num_cores);
    dse_group_set_rpool (dse_group, rpool);
    rpool_add_nasm (rpool, target_nasm, "data/batched_input_64.bin");

    // 4. Profile the execution

    // print_nasm_info (target_nasm, 1, 0);

    // dse_group_profile_nasm (dse_group, target_nasm);
    // char profile_file_name [1024] = {0};
    // if (num_seq == -1)
    //     sprintf (profile_file_name, "profiles/%s/%s_B%d_profile.csv", dnn, dnn, batch_size);
    // else
    //     sprintf (profile_file_name, "profiles/%s/%s_S%d_B%d_profile.csv", dnn, dnn, num_seq, batch_size);
    // dse_group_profile_nasm (dse_group, target_nasm);
    // dse_group_save_profile_data (dse_group, profile_file_name);
    // char heft_file_name [1024] = {0};
    // if (num_seq == -1)
    //     sprintf (heft_file_name, "profiles/%s/%s_B%d_heft.txt", dnn, dnn, batch_size);
    // else
    //     sprintf (heft_file_name, "profiles/%s/%s_S%d_B%d_heft.txt", dnn, dnn, num_seq, batch_size);
    // dse_group_nasm_export_heft_data (dse_group, target_nasm, heft_file_name);

    // 5. Set scheules

    aspen_peer_t *peer_list[2];
    for (int i = 0; i < 2; i++)
        peer_list[i] = peer_init ();
    peer_copy (peer_list[0], dse_group->my_peer_data);
    sched_set_local (target_nasm, peer_list, 1);
    print_nasm_info (target_nasm, 0, 0);
    for (int i = 0; i < 2; i++)
        destroy_peer (peer_list[i]);

    // 5. Run the ASPEN DSEs

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

    // 6. Print the top-5 results

    LAYER_PARAMS output_order[] = {BATCH, OUT_C, OUT_H, OUT_W};
    float *layer_output = NULL;
    size_t output_size = dse_get_nasm_result (target_nasm, output_order, (void**)&layer_output);
    float *probabilities = (float *) malloc (output_size);
    softmax (layer_output, probabilities, batch_size, 1000);
    for (int i = 0; i < batch_size; i++)
        get_prob_results ("data/imagenet_classes.txt", probabilities + i * 1000, 1000);

    // // 6. Save the DNN output to a file

    // LAYER_PARAMS output_order[] = {BATCH, OUT_C, OUT_H, OUT_W};
    // void *layer_output = NULL;
    // size_t output_size = dse_get_nasm_result (target_nasm, output_order, &layer_output);
    // char output_file_name [1024] = {0};
    // if (num_seq == -1)
    //     sprintf (output_file_name, "ASPEN_%s_B%d.out", dnn, batch_size);
    // else
    //     sprintf (output_file_name, "ASPEN_%s_S%d_B%d.out", dnn, num_seq, batch_size);
    // FILE *output_file = fopen(output_file_name, "wb");
    // printf ("Saving output of size %ld bytes to %s\n", output_size, output_file_name);
    // fwrite (layer_output, output_size, 1, output_file);
    // fclose (output_file);
    // free (layer_output);

    // 7. Cleanup

    dse_group_destroy (dse_group);
    rpool_destroy (rpool);
    apu_destroy_nasm (target_nasm);
    apu_destroy_dnn (target_dnn);
    return 0;
}