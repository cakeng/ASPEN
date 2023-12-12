// This is a generic usage example for the ASPEN APIs.
// Please refer to the included examples for a more detailed step-by-step guides.

#include "aspen.h"

double get_sec()
{
    struct timeval now;
    gettimeofday (&now, NULL);
    return now.tv_sec + now.tv_usec*1e-6;
}

int main (int argc, char **argv)
{
    print_aspen_build_info();
    
    // 1. Parse command line arguments

    char dnn[256] = {0};
    int batch_size = 4;
    int number_of_iterations = 20;
    int num_cores = 4;
    int num_seq = -1;

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
    sprintf (target_aspen, "files/%s/%s.aspen", dnn, dnn);
    char nasm_file_name [1024] = {0};
    if (num_seq == -1)
        sprintf (nasm_file_name, "files/%s/%s_B%d.nasm", dnn, dnn, batch_size);
    else
        sprintf (nasm_file_name, "files/%s/%s_S%d_B%d.nasm", dnn, dnn, batch_size, num_seq);


    // 2. Create or load the ASPEN DNN weight file (.aspen file)

    // 2-1. Generate the ASPEN weight file (.aspen)

    char dnn_cfg[1024] = {0};
    sprintf (dnn_cfg, "files/%s/%s_aspen.cfg", dnn, dnn);
    char weight_bin [1024] = {0};
    sprintf (weight_bin, "files/%s/%s_weight.bin", dnn, dnn);
    aspen_dnn_t *target_dnn = NULL;
    target_dnn = apu_create_dnn (dnn_cfg, weight_bin);
    if (target_dnn == NULL)
    {
        printf ("Unable to load ASPEN DNN weight file %s\n", target_aspen);
        exit (1);
    }
    apu_save_dnn_to_file (target_dnn, target_aspen);

    // 3-2. Load the ASPEN weight file (.aspen)

    // aspen_dnn_t *target_dnn = apu_load_dnn_from_file (target_aspen);
    // if (target_dnn == NULL)
    // {
    //     printf ("Unable to load ASPEN DNN weight file %s\n", target_aspen);
    //     exit (1);
    // }

    // 3. Create or load the ASPEN graph (.nasm file)

    // 3-1. Generate the ASPEN graph (nasm)

    // nasm_t *target_nasm = NULL;
    // if (num_seq == -1)
    //     target_nasm = apu_generate_nasm (target_dnn, batch_size, 20);
    // else
    //     target_nasm = apu_generate_transformer_nasm (target_dnn, batch_size, num_seq, 20);
    // apu_save_nasm_to_file (target_nasm, nasm_file_name);

    // 3-2. Load the ASPEN graph file (nasm)

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
    rpool_add_nasm (rpool, target_nasm, NULL);


    // 4. Run the ASPEN DSEs

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


    // 5. Save the DNN output to a file

    LAYER_PARAMS output_order[] = {BATCH, OUT_C, OUT_H, OUT_W};
    void *layer_output = NULL;
    size_t output_size = dse_get_nasm_result (target_nasm, output_order, &layer_output);
    char output_file_name [1024] = {0};
    if (num_seq == -1)
        sprintf (output_file_name, "ASPEN_%s_B%d.out", dnn, batch_size);
    else
        sprintf (output_file_name, "ASPEN_%s_S%d_B%d.out", dnn, num_seq, batch_size);
    FILE *output_file = fopen(output_file_name, "wb");
    printf ("Saving output of size %ld to %s\n", output_size, output_file_name);
    fwrite (layer_output, output_size, 1, output_file);
    fclose (output_file);
    free (layer_output);


    // 6. Cleanup

    dse_group_destroy (dse_group);
    rpool_destroy (rpool);
    apu_destroy_nasm (target_nasm);
    apu_destroy_dnn (target_dnn);
    return 0;
}