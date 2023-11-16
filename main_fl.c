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
    #ifndef SUPPRESS_OUTPUT
    print_aspen_build_info();
    #endif
    
    char dnn[256] = {0};
    int batch_size = 4;
    int number_of_iterations = 1;
    int num_cores = 1;
    int num_tiles = 50;
    int gpu_idx = -1;
    int dev_mode = -1;
    int fl_split_layer_idx = -1;
    int fl_num_path = -1;
    int fl_path_offloading_idx[2048];

    if (argc > 9)
    {
        strcpy (dnn, argv[1]);
        batch_size = atoi (argv[2]);
        num_tiles = atoi (argv[3]);
        number_of_iterations = atoi (argv[4]);
        num_cores = atoi (argv[5]);
        dev_mode = atoi (argv[6]);
        fl_split_layer_idx = atoi (argv[7]);
        fl_num_path = atoi (argv[8]);
        for (int i=0; i<fl_num_path; i++) {
            fl_path_offloading_idx[i] = atoi (argv[9+i]);
        }
    }
    else
    {
        printf ("Usage: %s <dnn> <batch_size> <num_tiles> <number_of_iterations> <num_cores> <dev_mode> <fl_last_layer> <fl_num_path> <fl_path_offloading_idx1> ...\n", argv[0]);
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
    // nasm_t *target_nasm = apu_create_nasm (target_dnn, num_tiles, batch_size);
    // apu_save_nasm_to_file (target_nasm, nasm_file_name);
    
    double start_time, end_time;
    
    /* BASIC MODULES */

    // rpool_t *rpool = rpool_init (gpu_idx);
    rpool_t *rpool = rpool_init_multigroup (gpu_idx, fl_split_layer_idx + 2);
    dse_group_t *dse_group = dse_group_init (num_cores, gpu_idx);
    dse_group_set_rpool (dse_group, rpool);
    dse_group_set_device_mode (dse_group, dev_mode);
    dse_group_set_device (dse_group, 0);
    dse_group_set_num_edge_devices (dse_group, 2);
    networking_engine* net_engine = NULL;

    /* NASM PREPARATION */

    aspen_dnn_t *target_dnn = apu_load_dnn_from_file (target_aspen);
    if (target_dnn == NULL)
    {
        printf ("Unable to load dnn file\n");
        exit (0);
    }
    nasm_t *target_nasm = apu_load_nasm_from_file (nasm_file_name, target_dnn);
    if (target_nasm == NULL)
    {
        printf ("Unable to load nasm file\n");
        exit (0);
    }

    /* FL PATH CREATION */

    unsigned int num_last_layer_ninst = target_nasm->ldata_arr[fl_split_layer_idx].num_ninst;

    PRTF ("FL: last layer of fl path has %d ninsts\n", num_last_layer_ninst);

    fl_init(target_nasm);
    
    for (int i=0; i<fl_num_path; i++) {
        unsigned int intercept_start = num_last_layer_ninst * i / fl_num_path;
        unsigned int intercept_end = num_last_layer_ninst * (i+1) / fl_num_path;

        ninst_t **path_last_ninsts = (ninst_t **)malloc(sizeof(ninst_t *) * (intercept_end - intercept_start));

        for (int j=0; j<(intercept_end - intercept_start); j++) {
            path_last_ninsts[j] = &target_nasm->ldata_arr[fl_split_layer_idx].ninst_arr_start[intercept_start + j];
        }

        fl_path_t *new_path = fl_create_path(target_nasm, path_last_ninsts, intercept_end - intercept_start, fl_path_offloading_idx[i]);

        if (i == 0) dse_set_starting_path(new_path);

        fl_set_dev_compute(target_nasm, new_path, dev_mode);
    }

    for (int i=0; i<num_cores; i++) dse_group->dse_arr[i].is_fl_offloading = 1;

    if (dev_mode == DEV_SERVER) {
        init_allow_all(target_nasm, 2);

        ninst_t *last_layer_ninst_arr_start = target_nasm->ldata_arr[target_nasm->num_ldata - 1].ninst_arr_start;
        unsigned int last_layer_num_ninst = target_nasm->ldata_arr[target_nasm->num_ldata - 1].num_ninst;

        for (int i=0; i<last_layer_num_ninst; i++) {
            last_layer_ninst_arr_start[i].dev_send_target[DEV_EDGE] = 1;
        }
    }
    else if (dev_mode == DEV_LOCAL) init_allow_all(target_nasm, 3);

    // Networking
    if(dev_mode == DEV_SERVER || dev_mode == DEV_EDGE) 
    {
        net_engine = init_networking(target_nasm, rpool, dev_mode, "147.46.130.51", 62000, 0, 1);
        net_engine->is_fl_offloading = 1;
        dse_group_set_net_engine(dse_group, net_engine);
        dse_group_set_device(dse_group, dev_mode);
        net_engine->dse_group = dse_group;
        net_engine_set_operating_mode(net_engine, OPER_MODE_FL_PATH);
    }

    rpool_add_nasm (rpool, target_nasm, "data/batched_input_128.bin");

    PRTF ("Running %d iterations\n", number_of_iterations);
    start_time = get_sec();
    for (int i = 0; i < number_of_iterations; i++)
    {
        net_engine_run(net_engine);
        rpool_reset_queue (rpool);
        apu_reset_nasm(target_nasm);
        dse_group_set_operating_mode(dse_group, OPER_MODE_FL_PATH);
        if (dev_mode == DEV_EDGE) fl_push_path_ninsts_edge(rpool, target_nasm->path_ptr_arr[0]);
        else if (dev_mode == DEV_LOCAL) fl_push_path_ninsts(rpool, target_nasm->path_ptr_arr[0]);
        dse_group_run (dse_group);
        dse_wait_for_nasm_completion (target_nasm);

        if (dev_mode != DEV_LOCAL) {
            unsigned int tx_remaining = atomic_load(&net_engine->rpool->num_stored);
            while (tx_remaining > 0) tx_remaining = atomic_load(&net_engine->rpool->num_stored);
            net_engine_wait_for_tx_queue_completion(net_engine);
            net_engine_reset(net_engine);
            net_engine_set_operating_mode(net_engine, OPER_MODE_FL_PATH);
        }
        dse_group_stop (dse_group);
        if (dev_mode != DEV_LOCAL) net_engine_stop (net_engine);

        fl_reset_nasm_path(target_nasm);

        dse_set_starting_path (target_nasm->path_ptr_arr[0]);

    }
    end_time = get_sec();

    #ifndef SUPRESS_OUTPUT
    printf ("Time taken: %lf seconds\n", (end_time - start_time)/number_of_iterations);
    #else
    printf ("%lf\n", (end_time - start_time)/number_of_iterations);
    #endif

    if (strcmp(dnn, "bert_base") != 0)
    {
        LAYER_PARAMS output_order[] = {BATCH, OUT_C, OUT_H, OUT_W};
        float *layer_output = dse_get_nasm_result (target_nasm, output_order);
        float *softmax_output = calloc (1000*batch_size, sizeof(float));
        softmax (layer_output, softmax_output, batch_size, 1000);
        for (int i = 0; i < batch_size; i++)
        {
            #ifndef SUPPRESS_OUTPUT
            get_prob_results ("data/imagenet_classes.txt", softmax_output + 1000*i, 1000);
            #endif
        }
        free (layer_output);
        free (softmax_output);
    }

    /* WRAP UP */
    fl_destroy_nasm_path(target_nasm);

    aspen_flush_dynamic_memory ();

    net_engine_destroy (net_engine);
    dse_group_destroy (dse_group);
    rpool_destroy (rpool);
    apu_destroy_nasm (target_nasm);
    apu_destroy_dnn (target_dnn);
    return 0;
}