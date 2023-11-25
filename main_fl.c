#include "aspen.h"
#include "apu.h"
#include "nasm.h"
#include "dse.h"

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

    const int server_port = 62000;
    char* server_ip = "147.46.130.51";

    if (argc == 7) {
        strcpy (dnn, argv[1]);
        batch_size = atoi (argv[2]);
        num_tiles = atoi (argv[3]);
        number_of_iterations = atoi (argv[4]);
        num_cores = atoi (argv[5]);
        dev_mode = atoi (argv[6]);

        PRTF("Find FL params automatically\n");
    }
    else if (argc > 9)
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

    /* PROFILING */
profiling:

    PRTF("STAGE: PROFILING\n");

    nasm_t *test_nasm = apu_load_nasm_from_file (nasm_file_name, target_dnn);

    int server_sock = -1, client_sock = -1;
    int control_port = server_port + 1;
    
    create_connection(dev_mode, server_ip, control_port, &server_sock, &client_sock);

    float *server_elapsed_times = (float *)calloc(test_nasm->num_ninst, sizeof(float));
    float *edge_elapsed_times = (float *)calloc(test_nasm->num_ninst, sizeof(float));
    network_profile_t **network_profile = (network_profile_t **)calloc(1, sizeof(network_profile_t *));
    profile_comp_and_net(
        test_nasm, num_cores, dev_mode, server_sock, client_sock,
        server_elapsed_times, edge_elapsed_times, network_profile
    );

    // print_network_profile(*network_profile);

    
    /* FL SCHEDULE */
    PRTF("STAGE: FL SCHEDULE\n");

    // Synchronize pipelining params
    int server_num_dse = 0, edge_num_dse = 0;
    if (dev_mode == DEV_SERVER) {
        write_n(client_sock, &num_cores, sizeof(int));
        read_n(client_sock, &edge_num_dse, sizeof(int));
        server_num_dse = num_cores;
    }
    else if (dev_mode == DEV_EDGE) {
        read_n(server_sock, &server_num_dse, sizeof(int));
        write_n(server_sock, &num_cores, sizeof(int));
        edge_num_dse = num_cores;
    }

    // Schedule FL

    float min_eta;
    if (dev_mode == DEV_SERVER) {
        min_eta = fl_schedule_bruteforce(
            target_nasm, server_num_dse, server_elapsed_times, edge_num_dse, edge_elapsed_times, *network_profile,
            &fl_split_layer_idx, &fl_num_path, fl_path_offloading_idx
        );
    }

    // Synchronize FL params
    if (dev_mode == DEV_SERVER) {
        write_n(client_sock, &fl_split_layer_idx, sizeof(int));
        write_n(client_sock, &fl_num_path, sizeof(int));
        write_n(client_sock, fl_path_offloading_idx, sizeof(int) * fl_num_path);
        write_n(client_sock, &min_eta, sizeof(float));
    }
    else if (dev_mode == DEV_EDGE) {
        read_n(server_sock, &fl_split_layer_idx, sizeof(int));
        read_n(server_sock, &fl_num_path, sizeof(int));
        read_n(server_sock, fl_path_offloading_idx, sizeof(int) * fl_num_path);
        read_n(server_sock, &min_eta, sizeof(float));
    }

    free(server_elapsed_times);
    free(edge_elapsed_times);
    free(*network_profile);
    free(network_profile);
    apu_destroy_nasm(test_nasm);

    if (server_sock != -1) close(server_sock);
    if (client_sock != -1) close(client_sock);

    #ifdef DEBUG
    printf("FL params: split layer %d, num path %d, expected %f\n", fl_split_layer_idx, fl_num_path, min_eta);
    for (int i=0; i<fl_num_path; i++) {
        printf("FL params: path %d: %d\n", i, fl_path_offloading_idx[i]);
    }
    #endif

    if (min_eta < 0) goto profiling;

    /* BASIC MODULES */

    // rpool_t *rpool = rpool_init (gpu_idx);
    rpool_t *rpool = rpool_init_multigroup (gpu_idx, FL_LIMIT_NUM_PATH + 2);
    dse_group_t *dse_group = dse_group_init (num_cores, gpu_idx);
    dse_group_set_rpool (dse_group, rpool);
    dse_group_set_device_mode (dse_group, dev_mode);
    dse_group_set_device (dse_group, 0);
    dse_group_set_num_edge_devices (dse_group, 2);
    networking_engine* net_engine = NULL;

    rpool_add_nasm (rpool, target_nasm, "data/batched_input_128.bin");


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

        free(path_last_ninsts);
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
        net_engine = init_networking(target_nasm, rpool, dev_mode, server_ip, server_port, 0, 1);
        net_engine->is_fl_offloading = 1;
        dse_group_set_net_engine(dse_group, net_engine);
        dse_group_set_device(dse_group, dev_mode);
        net_engine->dse_group = dse_group;
        net_engine_set_operating_mode(net_engine, OPER_MODE_FL_PATH);
    }

    PRTF ("Running %d iterations\n", number_of_iterations);
    

    float prev_edge_latency = 0.0;
    float prev_server_latency = 0.0;
    float prev_bandwidth = 0.0;

    for (int i = 0; i < number_of_iterations; i++)
    {
        double max_recv_time = 0.0;
        double max_sent_time = 0.0;
        double min_recv_time = 0.0;
        double min_sent_time = 0.0;
        double max_computed_time = 0.0;
        double min_computed_time = 0.0;
        double inf_latency = 0.0;
        double start_time = 0.0;

        net_engine_run(net_engine);
        rpool_reset_queue (rpool);
        apu_reset_nasm(target_nasm);
        dse_group_set_operating_mode(dse_group, OPER_MODE_FL_PATH);
        if (dev_mode == DEV_EDGE) fl_push_path_ninsts_edge(rpool, target_nasm->path_ptr_arr[0]);
        else if (dev_mode == DEV_LOCAL) fl_push_path_ninsts(rpool, target_nasm->path_ptr_arr[0]);
        // start_time = get_sec();
        set_elapsed_time_start ();
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
        // end_time = get_sec();
        get_elapsed_time ("run_aspen");
        fl_reset_nasm_path(target_nasm);

        dse_set_starting_path (target_nasm->path_ptr_arr[0]);

        // Communicate profiles
        // PRTF("\t[Communicate profiles]\n");
        max_computed_time = get_max_computed_time(target_nasm);
        min_computed_time = get_min_computed_time(target_nasm);
        
        min_recv_time = get_min_recv_time(target_nasm);
        max_sent_time = get_max_sent_time(target_nasm);
        
        if(dev_mode == DEV_SERVER)
        {
            max_recv_time = get_max_recv_time(target_nasm);
            if((max_computed_time - min_computed_time) > 0)
                prev_server_latency = max_computed_time - min_computed_time;

            write_n(client_sock, &prev_server_latency, sizeof(float));
            write_n(client_sock, &max_recv_time, sizeof(double));
            read_n(client_sock, &min_sent_time, sizeof(double));
        }
        else if (dev_mode == DEV_EDGE)
        {
            min_sent_time = get_min_sent_time(target_nasm);
            if((max_computed_time - min_computed_time) > 0)
                prev_edge_latency = max_computed_time - min_computed_time;
            
            read_n(server_sock, &prev_server_latency, sizeof(float));
            read_n(server_sock, &max_recv_time, sizeof(double));
            write_n(server_sock, &min_sent_time, sizeof(double));
        }
        int total_received = 0;
        for(int j = 0; j < target_nasm->num_ninst; j++)
        {
            if(target_nasm->ninst_arr[j].received_time != 0)
                total_received++;
        }
        PRTF("\t[Edge %d] Total received : (%d/%d)\n", 0, total_received, target_nasm->num_ninst);
        PRTF("\tTransmission latency : %fms\n", (max_recv_time - min_sent_time)*1000);
    }
    

    // #ifndef SUPRESS_OUTPUT
    // printf ("Time taken: %lf seconds\n", (end_time - start_time)/number_of_iterations);
    // #else
    // printf ("%lf\n", (end_time - start_time)/number_of_iterations);
    // #endif

    // if (strcmp(dnn, "bert_base") != 0 && strcmp(dnn, "yolov3") != 0)
    // {
    //     LAYER_PARAMS output_order[] = {BATCH, OUT_C, OUT_H, OUT_W};
    //     float *layer_output = dse_get_nasm_result (target_nasm, output_order);
    //     float *softmax_output = calloc (1000*batch_size, sizeof(float));
    //     softmax (layer_output, softmax_output, batch_size, 1000);
    //     for (int i = 0; i < batch_size; i++)
    //     {
    //         #ifndef SUPPRESS_OUTPUT
    //         get_prob_results ("data/imagenet_classes.txt", softmax_output + 1000*i, 1000);
    //         #endif
    //     }
    //     free (layer_output);
    //     free (softmax_output);
    // }
    // else if (strcmp(dnn, "yolov3") == 0)
    // {
    //     int last_ldata_intsum = get_ldata_intsum(&target_nasm->ldata_arr[target_nasm->num_ldata - 1]);
    //     #ifndef SUPPRESS_OUTPUT
    //     printf("last layer intsum: %d\n", last_ldata_intsum);
    //     #endif
    // }

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