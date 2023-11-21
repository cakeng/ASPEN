#include "aspen.h"
#include "apu.h"
#include "nasm.h"
#include "dse.h"

int main (int argc, char **argv)
{
    print_aspen_build_info();
    
    char dnn[256] = {0};
    int batch_size = 4;
    int number_of_iterations = 1;
    int num_cores = 1;
    int num_tiles = 50;
    int gpu_idx = -1;

    if (argc > 5)
    {
        strcpy (dnn, argv[1]);
        batch_size = atoi (argv[2]);
        num_tiles = atoi (argv[3]);
        number_of_iterations = atoi (argv[4]);
        num_cores = atoi (argv[5]);
    }
    else
    {
        printf ("Usage: %s <dnn> <batch_size> <num_tiles> <number_of_iterations> <num_cores>\n", argv[0]);
        exit (0);
    }

    char target_cfg[1024] = {0};
    if (strcmp(dnn, "bert_base") != 0)
        sprintf (target_cfg, "data/cfg/%s_aspen.cfg", dnn);
    else
        sprintf (target_cfg, "data/cfg/%s_encoder.cfg", dnn);
    char target_data[1024] = {0};
    sprintf (target_data, "data/%s_data.bin", dnn);
    char target_aspen[1024] = {0};
    sprintf (target_aspen, "data/%s_base.aspen", dnn);
    char nasm_file_name [1024] = {0};
    sprintf (nasm_file_name, "data/%s_B%d_T%d.nasm", dnn, batch_size, num_tiles);

    // aspen_dnn_t *target_dnn = apu_create_dnn(target_cfg, target_data);
    // apu_save_dnn_to_file (target_dnn, target_aspen);
    aspen_dnn_t *target_dnn = apu_load_dnn_from_file (target_aspen);
    nasm_t *target_nasm;
    if (strcmp(dnn, "bert_base") == 0)
        target_nasm = apu_create_transformer_nasm (target_dnn, num_tiles, batch_size, 128);
    else
        target_nasm = apu_create_nasm (target_dnn, num_tiles, batch_size);
    apu_save_nasm_to_file (target_nasm, nasm_file_name);

    
    if (target_dnn == NULL)
    {
        printf ("Unable to load dnn file\n");
        exit (0);
    }
    // nasm_t *target_nasm = apu_load_nasm_from_file (nasm_file_name, target_dnn);
    if (target_nasm == NULL)
    {
        printf ("Unable to load nasm file\n");
        exit (0);
    }
    apu_set_nasm_num_cores(target_nasm, num_cores);
  
    // rpool_t *rpool = rpool_init (gpu_idx);
    rpool_t *rpool = rpool_init_multigroup (gpu_idx, num_cores);
    dse_group_t *dse_group = dse_group_init (num_cores, gpu_idx);
    dse_group_set_rpool (dse_group, rpool);
    dse_group_set_device_mode (dse_group, DEV_LOCAL);
    dse_group_set_device (dse_group, 0);
    init_full_local (target_nasm, 0);

    rpool_add_nasm (rpool, target_nasm, "data/batched_input_128.bin");

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

    if (strcmp(dnn, "bert_base") != 0 && strcmp(dnn, "yolov3") != 0)
    {
        LAYER_PARAMS output_order[] = {BATCH, OUT_C, OUT_H, OUT_W};
        float *layer_output = dse_get_nasm_result (target_nasm, output_order);
        float *softmax_output = calloc (1000*batch_size, sizeof(float));
        softmax (layer_output, softmax_output, batch_size, 1000);
        for (int i = 0; i < batch_size; i++)
        {
            get_prob_results ("data/imagenet_classes.txt", softmax_output + 1000*i, 1000);
        }
        free (layer_output);
        free (softmax_output);
    }
    else if (strcmp(dnn, "yolov3") == 0)
    {
        int last_ldata_intsum = get_ldata_intsum(&target_nasm->ldata_arr[target_nasm->num_ldata - 1]);
        printf("last layer intsum: %d\n", last_ldata_intsum);
    }


    dse_group_destroy (dse_group);
    rpool_destroy (rpool);
    apu_destroy_nasm (target_nasm);
    apu_destroy_dnn (target_dnn);
    return 0;
}