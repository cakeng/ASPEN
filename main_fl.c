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
    int batch_size = 1;
    int number_of_iterations = 1;
    int num_cores = 1;
    int gpu_idx = -1;
    int dev_mode = -1;
    int dev_idx = -1;
    int fl_split_layer_idx = -1;
    int fl_num_path = -1;
    int fl_path_offloading_idx[2048];

    int server_ports[4] = {NULL, 24000, 24001, 24002};
    int control_ports[4] = {NULL, 24003, 24004, 24005};

    char target_aspen1[] = "data/resnet50_base.aspen";
    char target_aspen2[] = "data/bert_base.aspen";
    char target_aspen3[] = "data/vgg16_base.aspen";

    char nasm_file_name1[] = "data/resnet50_B1_T200.nasm";
    char nasm_file_name2[] = "data/bert_base_B1_T20.nasm";
    char nasm_file_name3[] = "data/vgg16_B1_T200.nasm";

    nasm_t *target_nasm1, *target_nasm2, *target_nasm3;
    rpool_t *rpool1, *rpool2, *rpool3;

    char* server_ip = "127.0.0.1";

    if (argc == 5) {
        strcpy (dnn, argv[1]);
        number_of_iterations = atoi (argv[2]);
        num_cores = atoi (argv[3]);
        dev_idx = atoi (argv[4]);
        dev_mode = dev_idx < 1 ? DEV_SERVER : DEV_EDGE;

        PRTF("Find FL params automatically\n");
    }
    else
    {
        printf ("Usage: %s <dnn> <number_of_iterations> <num_cores> <dev_idx>\n", argv[0]);
        exit (0);
    }

    double start_time, end_time;

    if (dev_mode == DEV_SERVER) {

        {
            /* NASM PREPARATION */
            aspen_dnn_t *target_dnn1 = apu_load_dnn_from_file (target_aspen1);
            aspen_dnn_t *target_dnn2 = apu_load_dnn_from_file (target_aspen2);
            aspen_dnn_t *target_dnn3 = apu_load_dnn_from_file (target_aspen3);
            target_nasm1 = apu_load_nasm_from_file (nasm_file_name1, target_dnn1);
            target_nasm2 = apu_load_nasm_from_file (nasm_file_name2, target_dnn2);
            target_nasm3 = apu_load_nasm_from_file (nasm_file_name3, target_dnn3);
            if (target_dnn1 == NULL || target_dnn2 == NULL || target_dnn3 == NULL)
            {
                printf ("Unable to load dnn file\n");
                exit (0);
            }
            if (target_nasm1 == NULL || target_nasm2 == NULL || target_nasm3 == NULL)
            {
                printf ("Unable to load nasm file\n");
                exit (0);
            }
            

            /* PROFILING */
            PRTF("STAGE: PROFILING\n");

            int server_sock1 = -1, client_sock1 = -1;
            nasm_t *test_nasm1 = apu_load_nasm_from_file (nasm_file_name1, target_dnn1);

            
            create_connection(dev_mode, server_ip, control_ports[1], &server_sock1, &client_sock1);

            float *server_elapsed_times1 = (float *)calloc(test_nasm1->num_ninst, sizeof(float));
            float *edge_elapsed_times1 = (float *)calloc(test_nasm1->num_ninst, sizeof(float));
            network_profile_t **network_profile1 = (network_profile_t **)calloc(1, sizeof(network_profile_t *));

            profile_comp_and_net(
                test_nasm1, num_cores, dev_mode, server_sock1, client_sock1,
                server_elapsed_times1, edge_elapsed_times1, network_profile1
            );
        

            // print_network_profile(*network_profile1);

            
            /* FL SCHEDULE */
            PRTF("STAGE: FL SCHEDULE\n");
            float min_eta1 = -1;

            while (min_eta1 < 0) {
                // Synchronize pipelining params
                int server_num_dse1 = 0, edge_num_dse1 = 0;
                if (dev_mode == DEV_SERVER) {
                    write_n(client_sock1, &num_cores, sizeof(int));
                    read_n(client_sock1, &edge_num_dse1, sizeof(int));
                    server_num_dse1 = num_cores;
                }
                else if (dev_mode == DEV_EDGE) {
                    read_n(server_sock1, &server_num_dse1, sizeof(int));
                    write_n(server_sock1, &num_cores, sizeof(int));
                    edge_num_dse1 = num_cores;
                }

                // Schedule FL

                if (dev_mode == DEV_SERVER) {
                    min_eta1 = fl_schedule_bruteforce(
                        target_nasm1, server_num_dse1, server_elapsed_times1, edge_num_dse1, edge_elapsed_times1, *network_profile1,
                        &fl_split_layer_idx, &fl_num_path, fl_path_offloading_idx
                    );
                }

                // Synchronize FL params
                if (dev_mode == DEV_SERVER) {
                    write_n(client_sock1, &fl_split_layer_idx, sizeof(int));
                    write_n(client_sock1, &fl_num_path, sizeof(int));
                    write_n(client_sock1, fl_path_offloading_idx, sizeof(int) * fl_num_path);
                    write_n(client_sock1, &min_eta1, sizeof(float));
                }
                else if (dev_mode == DEV_EDGE) {
                    read_n(server_sock1, &fl_split_layer_idx, sizeof(int));
                    read_n(server_sock1, &fl_num_path, sizeof(int));
                    read_n(server_sock1, fl_path_offloading_idx, sizeof(int) * fl_num_path);
                    read_n(server_sock1, &min_eta1, sizeof(float));
                }
            }

            free(server_elapsed_times1);
            free(edge_elapsed_times1);
            free(*network_profile1);
            free(network_profile1);
            apu_destroy_nasm(test_nasm1);

            if (server_sock1 != -1) close(server_sock1);
            if (client_sock1 != -1) close(client_sock1);
        
            #ifdef DEBUG
            printf("FL params: split layer %d, num path %d, expected %f\n", fl_split_layer_idx, fl_num_path, min_eta1);
            for (int i=0; i<fl_num_path; i++) {
                printf("FL params: path %d: %d\n", i, fl_path_offloading_idx[i]);
            }
            #endif



            /* BASIC MODULES */

            rpool1 = rpool_init_multigroup (gpu_idx, FL_LIMIT_NUM_PATH + 2);
            dse_group_t *dse_group1 = dse_group_init (num_cores, gpu_idx);
            dse_group_set_rpool (dse_group1, rpool1);
            dse_group_set_device_mode (dse_group1, dev_mode);
            dse_group_set_device (dse_group1, 0);
            dse_group_set_num_edge_devices (dse_group1, 2);
            networking_engine* net_engine1 = NULL;

            rpool_add_nasm (rpool1, target_nasm1, "data/batched_input_128.bin");


            rpool2 = rpool_init_multigroup (gpu_idx, FL_LIMIT_NUM_PATH + 2);
            dse_group_t *dse_group2 = dse_group_init (num_cores, gpu_idx);
            dse_group_set_rpool (dse_group2, rpool2);
            dse_group_set_device_mode (dse_group2, dev_mode);
            dse_group_set_device (dse_group2, 0);
            dse_group_set_num_edge_devices (dse_group2, 2);
            networking_engine* net_engine2 = NULL;


            rpool3 = rpool_init_multigroup (gpu_idx, FL_LIMIT_NUM_PATH + 2);
            dse_group_t *dse_group3 = dse_group_init (num_cores, gpu_idx);
            dse_group_set_rpool (dse_group3, rpool3);
            dse_group_set_device_mode (dse_group3, dev_mode);
            dse_group_set_device (dse_group3, 0);
            dse_group_set_num_edge_devices (dse_group3, 2);
            networking_engine* net_engine3 = NULL;

            rpool_add_nasm (rpool3, target_nasm3, "data/batched_input_128.bin");


            /* FL PATH CREATION */

            unsigned int num_last_layer_ninst1 = target_nasm1->ldata_arr[fl_split_layer_idx].num_ninst;

            PRTF ("FL: last layer of fl path has %d ninsts\n", num_last_layer_ninst1);

            fl_init(target_nasm1);
            
            for (int i=0; i<fl_num_path; i++) {
                unsigned int intercept_start = num_last_layer_ninst1 * i / fl_num_path;
                unsigned int intercept_end = num_last_layer_ninst1 * (i+1) / fl_num_path;

                ninst_t **path_last_ninsts = (ninst_t **)malloc(sizeof(ninst_t *) * (intercept_end - intercept_start));

                for (int j=0; j<(intercept_end - intercept_start); j++) {
                    path_last_ninsts[j] = &target_nasm1->ldata_arr[fl_split_layer_idx].ninst_arr_start[intercept_start + j];
                }

                fl_path_t *new_path = fl_create_path(target_nasm1, path_last_ninsts, intercept_end - intercept_start, fl_path_offloading_idx[i]);

                if (i == 0) dse_set_starting_path(new_path);

                fl_set_dev_compute(target_nasm1, new_path, dev_mode);

                free(path_last_ninsts);
            }

            for (int i=0; i<num_cores; i++) dse_group1->dse_arr[i].is_fl_offloading = 1;

            if (dev_mode == DEV_SERVER) {
                init_allow_all(target_nasm1, 2);

                ninst_t *last_layer_ninst_arr_start = target_nasm1->ldata_arr[target_nasm1->num_ldata - 1].ninst_arr_start;
                unsigned int last_layer_num_ninst = target_nasm1->ldata_arr[target_nasm1->num_ldata - 1].num_ninst;

                for (int i=0; i<last_layer_num_ninst; i++) {
                    last_layer_ninst_arr_start[i].dev_send_target[DEV_EDGE] = 1;
                }
            }
            else if (dev_mode == DEV_LOCAL) init_allow_all(target_nasm1, 3);

            // Networking
            if(dev_mode == DEV_SERVER || dev_mode == DEV_EDGE) 
            {
                net_engine1 = init_networking(target_nasm1, rpool1, dev_mode, server_ip, server_ports[1], 0, 1);
                net_engine1->is_fl_offloading = 1;
                dse_group_set_net_engine(dse_group1, net_engine1);
                dse_group_set_device(dse_group1, dev_mode);
                net_engine1->dse_group = dse_group1;
                net_engine_set_operating_mode(net_engine1, OPER_MODE_FL_PATH);
            }

            /* INFERENCE */
            PRTF ("Running %d iterations\n", number_of_iterations);
            start_time = get_sec();
            for (int i = 0; i < number_of_iterations; i++)
            {
                net_engine_run(net_engine1);
                rpool_reset_queue (rpool1);
                apu_reset_nasm(target_nasm1);
                dse_group_set_operating_mode(dse_group1, OPER_MODE_FL_PATH);
                if (dev_mode == DEV_EDGE) fl_push_path_ninsts_edge(rpool1, target_nasm1->path_ptr_arr[0]);
                else if (dev_mode == DEV_LOCAL) fl_push_path_ninsts(rpool1, target_nasm1->path_ptr_arr[0]);
                dse_group_run (dse_group1);
                dse_wait_for_nasm_completion (target_nasm1);

                if (dev_mode != DEV_LOCAL) {
                    unsigned int tx_remaining = atomic_load(&net_engine1->rpool->num_stored);
                    while (tx_remaining > 0) tx_remaining = atomic_load(&net_engine1->rpool->num_stored);
                    net_engine_wait_for_tx_queue_completion(net_engine1);
                    net_engine_reset(net_engine1);
                    net_engine_set_operating_mode(net_engine1, OPER_MODE_FL_PATH);
                }
                dse_group_stop (dse_group1);
                if (dev_mode != DEV_LOCAL) net_engine_stop (net_engine1);

                fl_reset_nasm_path(target_nasm1);

                dse_set_starting_path (target_nasm1->path_ptr_arr[0]);

            }
            end_time = get_sec();

            #ifndef SUPRESS_OUTPUT
            printf ("Time taken: %lf seconds\n", (end_time - start_time)/number_of_iterations);
            #else
            printf ("%lf\n", (end_time - start_time)/number_of_iterations);
            #endif

            if (strcmp(dnn, "bert_base") != 0 && strcmp(dnn, "yolov3") != 0)
            {
                LAYER_PARAMS output_order[] = {BATCH, OUT_C, OUT_H, OUT_W};
                float *layer_output = dse_get_nasm_result (target_nasm1, output_order);
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
            else if (strcmp(dnn, "yolov3") == 0)
            {
                int last_ldata_intsum = get_ldata_intsum(&target_nasm1->ldata_arr[target_nasm1->num_ldata - 1]);
                #ifndef SUPPRESS_OUTPUT
                printf("last layer intsum: %d\n", last_ldata_intsum);
                #endif
            }

            /* WRAP UP */
            fl_destroy_nasm_path(target_nasm1);

            aspen_flush_dynamic_memory ();

            net_engine_destroy (net_engine1);
            dse_group_destroy (dse_group1);
            rpool_destroy (rpool1);
            apu_destroy_nasm (target_nasm1);
            apu_destroy_dnn (target_dnn1);
        
            ////////////////////////////
            //       WORKLOAD 2       //
            ////////////////////////////

            rpool_add_nasm (rpool2, target_nasm2, "data/batched_input_128.bin");


            /* FL PATH CREATION */

            unsigned int num_last_layer_ninst2 = target_nasm2->ldata_arr[fl_split_layer_idx].num_ninst;

            PRTF ("FL: last layer of fl path has %d ninsts\n", num_last_layer_ninst2);

            fl_init(target_nasm2);
            
            for (int i=0; i<fl_num_path; i++) {
                unsigned int intercept_start = num_last_layer_ninst2 * i / fl_num_path;
                unsigned int intercept_end = num_last_layer_ninst2 * (i+1) / fl_num_path;

                ninst_t **path_last_ninsts = (ninst_t **)malloc(sizeof(ninst_t *) * (intercept_end - intercept_start));

                for (int j=0; j<(intercept_end - intercept_start); j++) {
                    path_last_ninsts[j] = &target_nasm2->ldata_arr[fl_split_layer_idx].ninst_arr_start[intercept_start + j];
                }

                fl_path_t *new_path = fl_create_path(target_nasm2, path_last_ninsts, intercept_end - intercept_start, fl_path_offloading_idx[i]);

                if (i == 0) dse_set_starting_path(new_path);

                fl_set_dev_compute(target_nasm2, new_path, dev_mode);

                free(path_last_ninsts);
            }

            for (int i=0; i<num_cores; i++) dse_group2->dse_arr[i].is_fl_offloading = 1;

            if (dev_mode == DEV_SERVER) {
                init_allow_all(target_nasm2, 2);

                ninst_t *last_layer_ninst_arr_start = target_nasm2->ldata_arr[target_nasm2->num_ldata - 1].ninst_arr_start;
                unsigned int last_layer_num_ninst = target_nasm2->ldata_arr[target_nasm2->num_ldata - 1].num_ninst;

                for (int i=0; i<last_layer_num_ninst; i++) {
                    last_layer_ninst_arr_start[i].dev_send_target[DEV_EDGE] = 1;
                }
            }
            else if (dev_mode == DEV_LOCAL) init_allow_all(target_nasm2, 3);

            // Networking
            if(dev_mode == DEV_SERVER || dev_mode == DEV_EDGE) 
            {
                net_engine2 = init_networking(target_nasm2, rpool2, dev_mode, server_ip, server_ports[2], 0, 1);
                dse_group_set_net_engine(dse_group2, net_engine2);
                dse_group_set_device(dse_group2, dev_mode);
                net_engine2->dse_group = dse_group2;
                net_engine_set_operating_mode(net_engine2, OPER_MODE_DEFAULT);
            }


            /* INFERENCE */
            PRTF ("Running %d iterations\n", number_of_iterations);
            start_time = get_sec();
            for (int i = 0; i < number_of_iterations; i++)
            {
                net_engine_run(net_engine2);
                rpool_reset_queue (rpool2);
                apu_reset_nasm(target_nasm2);
                dse_group_set_operating_mode(dse_group2, OPER_MODE_DEFAULT);
                dse_group_run (dse_group2);
                dse_wait_for_nasm_completion (target_nasm2);

                if (dev_mode != DEV_LOCAL) {
                    unsigned int tx_remaining = atomic_load(&net_engine2->rpool->num_stored);
                    while (tx_remaining > 0) tx_remaining = atomic_load(&net_engine2->rpool->num_stored);
                    net_engine_wait_for_tx_queue_completion(net_engine2);
                    net_engine_reset(net_engine2);
                    net_engine_set_operating_mode(net_engine2, OPER_MODE_DEFAULT);
                }
                dse_group_stop (dse_group2);
                if (dev_mode != DEV_LOCAL) net_engine_stop (net_engine2);

                fl_reset_nasm_path(target_nasm2);

                dse_set_starting_path (target_nasm2->path_ptr_arr[0]);

            }
            end_time = get_sec();

            #ifndef SUPRESS_OUTPUT
            printf ("Time taken: %lf seconds\n", (end_time - start_time)/number_of_iterations);
            #else
            printf ("%lf\n", (end_time - start_time)/number_of_iterations);
            #endif


            /* WRAP UP */
            fl_destroy_nasm_path(target_nasm2);

            aspen_flush_dynamic_memory ();

            net_engine_destroy (net_engine2);
            dse_group_destroy (dse_group2);
            rpool_destroy (rpool2);
            apu_destroy_nasm (target_nasm2);
            apu_destroy_dnn (target_dnn2);
        

        
        ////////////////////////////
        //       WORKLOAD 3       //
        ////////////////////////////
        
            

            /* PROFILING */
            PRTF("STAGE: PROFILING\n");
            int server_sock3 = -1, client_sock3 = -1;
            nasm_t *test_nasm3 = apu_load_nasm_from_file (nasm_file_name3, target_dnn3);
            
            create_connection(dev_mode, server_ip, control_ports[3], &server_sock3, &client_sock3);

            float *server_elapsed_times3 = (float *)calloc(test_nasm3->num_ninst, sizeof(float));
            float *edge_elapsed_times3 = (float *)calloc(test_nasm3->num_ninst, sizeof(float));
            network_profile_t **network_profile3 = (network_profile_t **)calloc(1, sizeof(network_profile_t *));
            profile_comp_and_net(
                test_nasm3, num_cores, dev_mode, server_sock3, client_sock3,
                server_elapsed_times3, edge_elapsed_times3, network_profile3
            );
        

            // print_network_profile(*network_profile1);

            
            /* FL SCHEDULE */
            PRTF("STAGE: FL SCHEDULE\n");

            float min_eta3 = -1;
            while (min_eta3 < 0) {
                // Synchronize pipelining params
                int server_num_dse3 = 0, edge_num_dse3 = 0;
                if (dev_mode == DEV_SERVER) {
                    write_n(client_sock3, &num_cores, sizeof(int));
                    read_n(client_sock3, &edge_num_dse3, sizeof(int));
                    server_num_dse3 = num_cores;
                }
                else if (dev_mode == DEV_EDGE) {
                    read_n(server_sock3, &server_num_dse3, sizeof(int));
                    write_n(server_sock3, &num_cores, sizeof(int));
                    edge_num_dse3 = num_cores;
                }

                // Schedule FL

                if (dev_mode == DEV_SERVER) {
                    min_eta3 = fl_schedule_bruteforce(
                        target_nasm3, server_num_dse3, server_elapsed_times3, edge_num_dse3, edge_elapsed_times3, *network_profile3,
                        &fl_split_layer_idx, &fl_num_path, fl_path_offloading_idx
                    );
                }

                // Synchronize FL params
                if (dev_mode == DEV_SERVER) {
                    write_n(client_sock3, &fl_split_layer_idx, sizeof(int));
                    write_n(client_sock3, &fl_num_path, sizeof(int));
                    write_n(client_sock3, fl_path_offloading_idx, sizeof(int) * fl_num_path);
                    write_n(client_sock3, &min_eta3, sizeof(float));
                }
                else if (dev_mode == DEV_EDGE) {
                    read_n(server_sock3, &fl_split_layer_idx, sizeof(int));
                    read_n(server_sock3, &fl_num_path, sizeof(int));
                    read_n(server_sock3, fl_path_offloading_idx, sizeof(int) * fl_num_path);
                    read_n(server_sock3, &min_eta3, sizeof(float));
                }
            }

            free(server_elapsed_times3);
            free(edge_elapsed_times3);
            free(*network_profile3);
            free(network_profile3);
            apu_destroy_nasm(test_nasm3);

            if (server_sock3 != -1) close(server_sock3);
            if (client_sock3 != -1) close(client_sock3);
        
            #ifdef DEBUG
            printf("FL params: split layer %d, num path %d, expected %f\n", fl_split_layer_idx, fl_num_path, min_eta3);
            for (int i=0; i<fl_num_path; i++) {
                printf("FL params: path %d: %d\n", i, fl_path_offloading_idx[i]);
            }
            #endif


            /* FL PATH CREATION */

            unsigned int num_last_layer_ninst3 = target_nasm3->ldata_arr[fl_split_layer_idx].num_ninst;

            PRTF ("FL: last layer of fl path has %d ninsts\n", num_last_layer_ninst3);

            fl_init(target_nasm3);
            
            for (int i=0; i<fl_num_path; i++) {
                unsigned int intercept_start = num_last_layer_ninst3 * i / fl_num_path;
                unsigned int intercept_end = num_last_layer_ninst3 * (i+1) / fl_num_path;

                ninst_t **path_last_ninsts = (ninst_t **)malloc(sizeof(ninst_t *) * (intercept_end - intercept_start));

                for (int j=0; j<(intercept_end - intercept_start); j++) {
                    path_last_ninsts[j] = &target_nasm3->ldata_arr[fl_split_layer_idx].ninst_arr_start[intercept_start + j];
                }

                fl_path_t *new_path = fl_create_path(target_nasm3, path_last_ninsts, intercept_end - intercept_start, fl_path_offloading_idx[i]);

                if (i == 0) dse_set_starting_path(new_path);

                fl_set_dev_compute(target_nasm3, new_path, dev_mode);

                free(path_last_ninsts);
            }

            for (int i=0; i<num_cores; i++) dse_group3->dse_arr[i].is_fl_offloading = 1;

            if (dev_mode == DEV_SERVER) {
                init_allow_all(target_nasm3, 2);

                ninst_t *last_layer_ninst_arr_start = target_nasm3->ldata_arr[target_nasm3->num_ldata - 1].ninst_arr_start;
                unsigned int last_layer_num_ninst = target_nasm3->ldata_arr[target_nasm3->num_ldata - 1].num_ninst;

                for (int i=0; i<last_layer_num_ninst; i++) {
                    last_layer_ninst_arr_start[i].dev_send_target[DEV_EDGE] = 1;
                }
            }
            else if (dev_mode == DEV_LOCAL) init_allow_all(target_nasm3, 3);

            // Networking
            if(dev_mode == DEV_SERVER || dev_mode == DEV_EDGE) 
            {
                net_engine3 = init_networking(target_nasm3, rpool3, dev_mode, server_ip, server_ports[3], 0, 1);
                net_engine3->is_fl_offloading = 1;
                dse_group_set_net_engine(dse_group3, net_engine3);
                dse_group_set_device(dse_group3, dev_mode);
                net_engine3->dse_group = dse_group3;
                net_engine_set_operating_mode(net_engine3, OPER_MODE_FL_PATH);
            }


            /* INFERENCE */
            PRTF ("Running %d iterations\n", number_of_iterations);
            start_time = get_sec();
            for (int i = 0; i < number_of_iterations; i++)
            {
                net_engine_run(net_engine3);
                rpool_reset_queue (rpool3);
                apu_reset_nasm(target_nasm3);
                dse_group_set_operating_mode(dse_group3, OPER_MODE_FL_PATH);
                if (dev_mode == DEV_EDGE) fl_push_path_ninsts_edge(rpool3, target_nasm3->path_ptr_arr[0]);
                else if (dev_mode == DEV_LOCAL) fl_push_path_ninsts(rpool3, target_nasm3->path_ptr_arr[0]);
                dse_group_run (dse_group3);
                dse_wait_for_nasm_completion (target_nasm3);

                if (dev_mode != DEV_LOCAL) {
                    unsigned int tx_remaining = atomic_load(&net_engine3->rpool->num_stored);
                    while (tx_remaining > 0) tx_remaining = atomic_load(&net_engine3->rpool->num_stored);
                    net_engine_wait_for_tx_queue_completion(net_engine3);
                    net_engine_reset(net_engine3);
                    net_engine_set_operating_mode(net_engine3, OPER_MODE_FL_PATH);
                }
                dse_group_stop (dse_group3);
                if (dev_mode != DEV_LOCAL) net_engine_stop (net_engine3);

                fl_reset_nasm_path(target_nasm3);

                dse_set_starting_path (target_nasm3->path_ptr_arr[0]);

            }
            end_time = get_sec();

            #ifndef SUPRESS_OUTPUT
            printf ("Time taken: %lf seconds\n", (end_time - start_time)/number_of_iterations);
            #else
            printf ("%lf\n", (end_time - start_time)/number_of_iterations);
            #endif

            if (strcmp(dnn, "bert_base") != 0 && strcmp(dnn, "yolov3") != 0)
            {
                LAYER_PARAMS output_order[] = {BATCH, OUT_C, OUT_H, OUT_W};
                float *layer_output = dse_get_nasm_result (target_nasm3, output_order);
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
            else if (strcmp(dnn, "yolov3") == 0)
            {
                int last_ldata_intsum = get_ldata_intsum(&target_nasm3->ldata_arr[target_nasm3->num_ldata - 1]);
                #ifndef SUPPRESS_OUTPUT
                printf("last layer intsum: %d\n", last_ldata_intsum);
                #endif
            }

            /* WRAP UP */
            fl_destroy_nasm_path(target_nasm3);

            aspen_flush_dynamic_memory ();

            net_engine_destroy (net_engine3);
            dse_group_destroy (dse_group3);
            rpool_destroy (rpool3);
            apu_destroy_nasm (target_nasm3);
            apu_destroy_dnn (target_dnn3);
        }
    }
    else {
        ////////////////////////////
        //       WORKLOAD 1       //
        ////////////////////////////
        if (dev_idx == 1) {
            const int server_port1 = server_ports[1];
            const int control_port1 = control_ports[1];

            /* NASM PREPARATION */
            aspen_dnn_t *target_dnn1 = apu_load_dnn_from_file (target_aspen1);
            if (target_dnn1 == NULL)
            {
                printf ("Unable to load dnn file\n");
                exit (0);
            }
            target_nasm1 = apu_load_nasm_from_file (nasm_file_name1, target_dnn1);
            if (target_nasm1 == NULL)
            {
                printf ("Unable to load nasm file\n");
                exit (0);
            }

            /* PROFILING */
            PRTF("STAGE: PROFILING\n");

            int server_sock1 = -1, client_sock1 = -1;
            nasm_t *test_nasm1 = apu_load_nasm_from_file (nasm_file_name1, target_dnn1);

            
            create_connection(dev_mode, server_ip, control_port1, &server_sock1, &client_sock1);

            float *server_elapsed_times1 = (float *)calloc(test_nasm1->num_ninst, sizeof(float));
            float *edge_elapsed_times1 = (float *)calloc(test_nasm1->num_ninst, sizeof(float));
            network_profile_t **network_profile1 = (network_profile_t **)calloc(1, sizeof(network_profile_t *));
            profile_comp_and_net(
                test_nasm1, num_cores, dev_mode, server_sock1, client_sock1,
                server_elapsed_times1, edge_elapsed_times1, network_profile1
            );
        

            // print_network_profile(*network_profile1);

            
            /* FL SCHEDULE */
            PRTF("STAGE: FL SCHEDULE\n");
            float min_eta1 = -1;

            while (min_eta1 < 0) {
                // Synchronize pipelining params
                int server_num_dse1 = 0, edge_num_dse1 = 0;
                if (dev_mode == DEV_SERVER) {
                    write_n(client_sock1, &num_cores, sizeof(int));
                    read_n(client_sock1, &edge_num_dse1, sizeof(int));
                    server_num_dse1 = num_cores;
                }
                else if (dev_mode == DEV_EDGE) {
                    read_n(server_sock1, &server_num_dse1, sizeof(int));
                    write_n(server_sock1, &num_cores, sizeof(int));
                    edge_num_dse1 = num_cores;
                }

                // Schedule FL

                if (dev_mode == DEV_SERVER) {
                    min_eta1 = fl_schedule_bruteforce(
                        target_nasm1, server_num_dse1, server_elapsed_times1, edge_num_dse1, edge_elapsed_times1, *network_profile1,
                        &fl_split_layer_idx, &fl_num_path, fl_path_offloading_idx
                    );
                }

                // Synchronize FL params
                if (dev_mode == DEV_SERVER) {
                    write_n(client_sock1, &fl_split_layer_idx, sizeof(int));
                    write_n(client_sock1, &fl_num_path, sizeof(int));
                    write_n(client_sock1, fl_path_offloading_idx, sizeof(int) * fl_num_path);
                    write_n(client_sock1, &min_eta1, sizeof(float));
                }
                else if (dev_mode == DEV_EDGE) {
                    read_n(server_sock1, &fl_split_layer_idx, sizeof(int));
                    read_n(server_sock1, &fl_num_path, sizeof(int));
                    read_n(server_sock1, fl_path_offloading_idx, sizeof(int) * fl_num_path);
                    read_n(server_sock1, &min_eta1, sizeof(float));
                }
            }


            free(server_elapsed_times1);
            free(edge_elapsed_times1);
            free(*network_profile1);
            free(network_profile1);
            apu_destroy_nasm(test_nasm1);

            if (server_sock1 != -1) close(server_sock1);
            if (client_sock1 != -1) close(client_sock1);
        
            #ifdef DEBUG
            printf("FL params: split layer %d, num path %d, expected %f\n", fl_split_layer_idx, fl_num_path, min_eta1);
            for (int i=0; i<fl_num_path; i++) {
                printf("FL params: path %d: %d\n", i, fl_path_offloading_idx[i]);
            }
            #endif


            /* BASIC MODULES */

            rpool1 = rpool_init_multigroup (gpu_idx, FL_LIMIT_NUM_PATH + 2);
            dse_group_t *dse_group1 = dse_group_init (num_cores, gpu_idx);
            dse_group_set_rpool (dse_group1, rpool1);
            dse_group_set_device_mode (dse_group1, dev_mode);
            dse_group_set_device (dse_group1, 0);
            dse_group_set_num_edge_devices (dse_group1, 2);
            networking_engine* net_engine1 = NULL;

            rpool_add_nasm (rpool1, target_nasm1, "data/batched_input_128.bin");


            /* FL PATH CREATION */

            unsigned int num_last_layer_ninst1 = target_nasm1->ldata_arr[fl_split_layer_idx].num_ninst;

            PRTF ("FL: last layer of fl path has %d ninsts\n", num_last_layer_ninst1);

            fl_init(target_nasm1);
            
            for (int i=0; i<fl_num_path; i++) {
                unsigned int intercept_start = num_last_layer_ninst1 * i / fl_num_path;
                unsigned int intercept_end = num_last_layer_ninst1 * (i+1) / fl_num_path;

                ninst_t **path_last_ninsts = (ninst_t **)malloc(sizeof(ninst_t *) * (intercept_end - intercept_start));

                for (int j=0; j<(intercept_end - intercept_start); j++) {
                    path_last_ninsts[j] = &target_nasm1->ldata_arr[fl_split_layer_idx].ninst_arr_start[intercept_start + j];
                }

                fl_path_t *new_path = fl_create_path(target_nasm1, path_last_ninsts, intercept_end - intercept_start, fl_path_offloading_idx[i]);

                if (i == 0) dse_set_starting_path(new_path);

                fl_set_dev_compute(target_nasm1, new_path, dev_mode);

                free(path_last_ninsts);
            }

            for (int i=0; i<num_cores; i++) dse_group1->dse_arr[i].is_fl_offloading = 1;

            if (dev_mode == DEV_SERVER) {
                init_allow_all(target_nasm1, 2);

                ninst_t *last_layer_ninst_arr_start = target_nasm1->ldata_arr[target_nasm1->num_ldata - 1].ninst_arr_start;
                unsigned int last_layer_num_ninst = target_nasm1->ldata_arr[target_nasm1->num_ldata - 1].num_ninst;

                for (int i=0; i<last_layer_num_ninst; i++) {
                    last_layer_ninst_arr_start[i].dev_send_target[DEV_EDGE] = 1;
                }
            }
            else if (dev_mode == DEV_LOCAL) init_allow_all(target_nasm1, 3);

            // Networking
            if(dev_mode == DEV_SERVER || dev_mode == DEV_EDGE) 
            {
                net_engine1 = init_networking(target_nasm1, rpool1, dev_mode, server_ip, server_port1, 0, 1);
                net_engine1->is_fl_offloading = 1;
                dse_group_set_net_engine(dse_group1, net_engine1);
                dse_group_set_device(dse_group1, dev_mode);
                net_engine1->dse_group = dse_group1;
                net_engine_set_operating_mode(net_engine1, OPER_MODE_FL_PATH);
            }

            PRTF ("Running %d iterations\n", number_of_iterations);
            start_time = get_sec();
            for (int i = 0; i < number_of_iterations; i++)
            {
                net_engine_run(net_engine1);
                rpool_reset_queue (rpool1);
                apu_reset_nasm(target_nasm1);
                dse_group_set_operating_mode(dse_group1, OPER_MODE_FL_PATH);
                if (dev_mode == DEV_EDGE) fl_push_path_ninsts_edge(rpool1, target_nasm1->path_ptr_arr[0]);
                else if (dev_mode == DEV_LOCAL) fl_push_path_ninsts(rpool1, target_nasm1->path_ptr_arr[0]);
                dse_group_run (dse_group1);
                dse_wait_for_nasm_completion (target_nasm1);

                if (dev_mode != DEV_LOCAL) {
                    unsigned int tx_remaining = atomic_load(&net_engine1->rpool->num_stored);
                    while (tx_remaining > 0) tx_remaining = atomic_load(&net_engine1->rpool->num_stored);
                    net_engine_wait_for_tx_queue_completion(net_engine1);
                    net_engine_reset(net_engine1);
                    net_engine_set_operating_mode(net_engine1, OPER_MODE_FL_PATH);
                }
                dse_group_stop (dse_group1);
                if (dev_mode != DEV_LOCAL) net_engine_stop (net_engine1);

                fl_reset_nasm_path(target_nasm1);

                dse_set_starting_path (target_nasm1->path_ptr_arr[0]);

            }
            end_time = get_sec();

            #ifndef SUPRESS_OUTPUT
            printf ("Time taken: %lf seconds\n", (end_time - start_time)/number_of_iterations);
            #else
            printf ("%lf\n", (end_time - start_time)/number_of_iterations);
            #endif

            if (strcmp(dnn, "bert_base") != 0 && strcmp(dnn, "yolov3") != 0)
            {
                LAYER_PARAMS output_order[] = {BATCH, OUT_C, OUT_H, OUT_W};
                float *layer_output = dse_get_nasm_result (target_nasm1, output_order);
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
            else if (strcmp(dnn, "yolov3") == 0)
            {
                int last_ldata_intsum = get_ldata_intsum(&target_nasm1->ldata_arr[target_nasm1->num_ldata - 1]);
                #ifndef SUPPRESS_OUTPUT
                printf("last layer intsum: %d\n", last_ldata_intsum);
                #endif
            }

            /* WRAP UP */
            fl_destroy_nasm_path(target_nasm1);

            aspen_flush_dynamic_memory ();

            net_engine_destroy (net_engine1);
            dse_group_destroy (dse_group1);
            rpool_destroy (rpool1);
            apu_destroy_nasm (target_nasm1);
            apu_destroy_dnn (target_dnn1);
        }

        ////////////////////////////
        //       WORKLOAD 2       //
        ////////////////////////////
        else if (dev_idx == 2) {
            const int server_port2 = server_ports[2];
            const int control_port2 = control_ports[2];

            /* NASM PREPARATION */
            aspen_dnn_t *target_dnn2 = apu_load_dnn_from_file (target_aspen2);
            if (target_dnn2 == NULL)
            {
                printf ("Unable to load dnn file\n");
                exit (0);
            }
            target_nasm2 = apu_load_nasm_from_file (nasm_file_name2, target_dnn2);
            if (target_nasm2 == NULL)
            {
                printf ("Unable to load nasm file\n");
                exit (0);
            }

            /* PROFILING */
            fl_split_layer_idx = 1;
            fl_num_path = 1;
            fl_path_offloading_idx[0] = 0;
            fl_path_offloading_idx[1] = 0;


            /* BASIC MODULES */

            rpool2 = rpool_init_multigroup (gpu_idx, FL_LIMIT_NUM_PATH + 2);
            dse_group_t *dse_group2 = dse_group_init (num_cores, gpu_idx);
            dse_group_set_rpool (dse_group2, rpool2);
            dse_group_set_device_mode (dse_group2, dev_mode);
            dse_group_set_device (dse_group2, 0);
            dse_group_set_num_edge_devices (dse_group2, 2);
            networking_engine* net_engine2 = NULL;

            rpool_add_nasm (rpool2, target_nasm2, "data/batched_input_128.bin");


            /* FL PATH CREATION */

            unsigned int num_last_layer_ninst2 = target_nasm2->ldata_arr[fl_split_layer_idx].num_ninst;

            PRTF ("FL: last layer of fl path has %d ninsts\n", num_last_layer_ninst2);

            fl_init(target_nasm2);
            
            for (int i=0; i<fl_num_path; i++) {
                unsigned int intercept_start = num_last_layer_ninst2 * i / fl_num_path;
                unsigned int intercept_end = num_last_layer_ninst2 * (i+1) / fl_num_path;

                ninst_t **path_last_ninsts = (ninst_t **)malloc(sizeof(ninst_t *) * (intercept_end - intercept_start));

                for (int j=0; j<(intercept_end - intercept_start); j++) {
                    path_last_ninsts[j] = &target_nasm2->ldata_arr[fl_split_layer_idx].ninst_arr_start[intercept_start + j];
                }

                fl_path_t *new_path = fl_create_path(target_nasm2, path_last_ninsts, intercept_end - intercept_start, fl_path_offloading_idx[i]);

                if (i == 0) dse_set_starting_path(new_path);

                fl_set_dev_compute(target_nasm2, new_path, dev_mode);

                free(path_last_ninsts);
            }

            for (int i=0; i<num_cores; i++) dse_group2->dse_arr[i].is_fl_offloading = 1;

            if (dev_mode == DEV_SERVER) {
                init_allow_all(target_nasm2, 2);

                ninst_t *last_layer_ninst_arr_start = target_nasm2->ldata_arr[target_nasm2->num_ldata - 1].ninst_arr_start;
                unsigned int last_layer_num_ninst = target_nasm2->ldata_arr[target_nasm2->num_ldata - 1].num_ninst;

                for (int i=0; i<last_layer_num_ninst; i++) {
                    last_layer_ninst_arr_start[i].dev_send_target[DEV_EDGE] = 1;
                }
            }
            else if (dev_mode == DEV_LOCAL) init_allow_all(target_nasm2, 3);

            // Networking
            if(dev_mode == DEV_SERVER || dev_mode == DEV_EDGE) 
            {
                net_engine2 = init_networking(target_nasm2, rpool2, dev_mode, server_ip, server_port2, 0, 1);
                dse_group_set_net_engine(dse_group2, net_engine2);
                dse_group_set_device(dse_group2, dev_mode);
                net_engine2->dse_group = dse_group2;
                net_engine_set_operating_mode(net_engine2, OPER_MODE_DEFAULT);
            }

            init_sequential_offload(target_nasm2, 1, 1, 0);

            PRTF ("Running %d iterations\n", number_of_iterations);
            start_time = get_sec();
            for (int i = 0; i < number_of_iterations; i++)
            {
                net_engine_run(net_engine2);
                rpool_reset_queue (rpool2);
                apu_reset_nasm(target_nasm2);

                push_first_layer_to_net_queue(net_engine2, target_nasm2, NULL);
                dse_group_set_operating_mode(dse_group2, OPER_MODE_DEFAULT);
                dse_group_run (dse_group2);
                dse_wait_for_nasm_completion (target_nasm2);

                if (dev_mode != DEV_LOCAL) {
                    unsigned int tx_remaining = atomic_load(&net_engine2->rpool->num_stored);
                    while (tx_remaining > 0) tx_remaining = atomic_load(&net_engine2->rpool->num_stored);
                    net_engine_wait_for_tx_queue_completion(net_engine2);
                    net_engine_reset(net_engine2);
                    net_engine_set_operating_mode(net_engine2, OPER_MODE_DEFAULT);
                }
                dse_group_stop (dse_group2);
                if (dev_mode != DEV_LOCAL) net_engine_stop (net_engine2);

                fl_reset_nasm_path(target_nasm2);

                dse_set_starting_path (target_nasm2->path_ptr_arr[0]);

            }
            end_time = get_sec();

            #ifndef SUPRESS_OUTPUT
            printf ("Time taken: %lf seconds\n", (end_time - start_time)/number_of_iterations);
            #else
            printf ("%lf\n", (end_time - start_time)/number_of_iterations);
            #endif

            /* WRAP UP */
            fl_destroy_nasm_path(target_nasm2);

            aspen_flush_dynamic_memory ();

            net_engine_destroy (net_engine2);
            dse_group_destroy (dse_group2);
            rpool_destroy (rpool2);
            apu_destroy_nasm (target_nasm2);
            apu_destroy_dnn (target_dnn2);
        }

        
        ////////////////////////////
        //       WORKLOAD 3       //
        ////////////////////////////
        if (dev_idx == 3) {
            const int server_port3 = server_ports[3];
            const int control_port3 = control_ports[3];

            /* NASM PREPARATION */
            aspen_dnn_t *target_dnn3 = apu_load_dnn_from_file (target_aspen3);
            if (target_dnn3 == NULL)
            {
                printf ("Unable to load dnn file\n");
                exit (0);
            }
            target_nasm3 = apu_load_nasm_from_file (nasm_file_name3, target_dnn3);
            if (target_nasm3 == NULL)
            {
                printf ("Unable to load nasm file\n");
                exit (0);
            }

            /* PROFILING */
            PRTF("STAGE: PROFILING\n");
            int server_sock3 = -1, client_sock3 = -1;
            nasm_t *test_nasm3 = apu_load_nasm_from_file (nasm_file_name3, target_dnn3);
            
            create_connection(dev_mode, server_ip, control_port3, &server_sock3, &client_sock3);

            float *server_elapsed_times3 = (float *)calloc(test_nasm3->num_ninst, sizeof(float));
            float *edge_elapsed_times3 = (float *)calloc(test_nasm3->num_ninst, sizeof(float));
            network_profile_t **network_profile3 = (network_profile_t **)calloc(1, sizeof(network_profile_t *));
            profile_comp_and_net(
                test_nasm3, num_cores, dev_mode, server_sock3, client_sock3,
                server_elapsed_times3, edge_elapsed_times3, network_profile3
            );
        

            // print_network_profile(*network_profile1);

            
            /* FL SCHEDULE */
            PRTF("STAGE: FL SCHEDULE\n");

            float min_eta3 = -1;
            while (min_eta3 < 0) {
                // Synchronize pipelining params
                int server_num_dse3 = 0, edge_num_dse3 = 0;
                if (dev_mode == DEV_SERVER) {
                    write_n(client_sock3, &num_cores, sizeof(int));
                    read_n(client_sock3, &edge_num_dse3, sizeof(int));
                    server_num_dse3 = num_cores;
                }
                else if (dev_mode == DEV_EDGE) {
                    read_n(server_sock3, &server_num_dse3, sizeof(int));
                    write_n(server_sock3, &num_cores, sizeof(int));
                    edge_num_dse3 = num_cores;
                }

                // Schedule FL

                if (dev_mode == DEV_SERVER) {
                    min_eta3 = fl_schedule_bruteforce(
                        target_nasm3, server_num_dse3, server_elapsed_times3, edge_num_dse3, edge_elapsed_times3, *network_profile3,
                        &fl_split_layer_idx, &fl_num_path, fl_path_offloading_idx
                    );
                }

                // Synchronize FL params
                if (dev_mode == DEV_SERVER) {
                    write_n(client_sock3, &fl_split_layer_idx, sizeof(int));
                    write_n(client_sock3, &fl_num_path, sizeof(int));
                    write_n(client_sock3, fl_path_offloading_idx, sizeof(int) * fl_num_path);
                    write_n(client_sock3, &min_eta3, sizeof(float));
                }
                else if (dev_mode == DEV_EDGE) {
                    read_n(server_sock3, &fl_split_layer_idx, sizeof(int));
                    read_n(server_sock3, &fl_num_path, sizeof(int));
                    read_n(server_sock3, fl_path_offloading_idx, sizeof(int) * fl_num_path);
                    read_n(server_sock3, &min_eta3, sizeof(float));
                }

            }

            free(server_elapsed_times3);
            free(edge_elapsed_times3);
            free(*network_profile3);
            free(network_profile3);
            apu_destroy_nasm(test_nasm3);

            if (server_sock3 != -1) close(server_sock3);
            if (client_sock3 != -1) close(client_sock3);
        
            #ifdef DEBUG
            printf("FL params: split layer %d, num path %d, expected %f\n", fl_split_layer_idx, fl_num_path, min_eta3);
            for (int i=0; i<fl_num_path; i++) {
                printf("FL params: path %d: %d\n", i, fl_path_offloading_idx[i]);
            }
            #endif


            /* BASIC MODULES */

            rpool3 = rpool_init_multigroup (gpu_idx, FL_LIMIT_NUM_PATH + 2);
            dse_group_t *dse_group3 = dse_group_init (num_cores, gpu_idx);
            dse_group_set_rpool (dse_group3, rpool3);
            dse_group_set_device_mode (dse_group3, dev_mode);
            dse_group_set_device (dse_group3, 0);
            dse_group_set_num_edge_devices (dse_group3, 2);
            networking_engine* net_engine3 = NULL;

            rpool_add_nasm (rpool3, target_nasm3, "data/batched_input_128.bin");


            /* FL PATH CREATION */

            unsigned int num_last_layer_ninst3 = target_nasm3->ldata_arr[fl_split_layer_idx].num_ninst;

            PRTF ("FL: last layer of fl path has %d ninsts\n", num_last_layer_ninst3);

            fl_init(target_nasm3);
            
            for (int i=0; i<fl_num_path; i++) {
                unsigned int intercept_start = num_last_layer_ninst3 * i / fl_num_path;
                unsigned int intercept_end = num_last_layer_ninst3 * (i+1) / fl_num_path;

                ninst_t **path_last_ninsts = (ninst_t **)malloc(sizeof(ninst_t *) * (intercept_end - intercept_start));

                for (int j=0; j<(intercept_end - intercept_start); j++) {
                    path_last_ninsts[j] = &target_nasm3->ldata_arr[fl_split_layer_idx].ninst_arr_start[intercept_start + j];
                }

                fl_path_t *new_path = fl_create_path(target_nasm3, path_last_ninsts, intercept_end - intercept_start, fl_path_offloading_idx[i]);

                if (i == 0) dse_set_starting_path(new_path);

                fl_set_dev_compute(target_nasm3, new_path, dev_mode);

                free(path_last_ninsts);
            }

            for (int i=0; i<num_cores; i++) dse_group3->dse_arr[i].is_fl_offloading = 1;

            if (dev_mode == DEV_SERVER) {
                init_allow_all(target_nasm3, 2);

                ninst_t *last_layer_ninst_arr_start = target_nasm3->ldata_arr[target_nasm3->num_ldata - 1].ninst_arr_start;
                unsigned int last_layer_num_ninst = target_nasm3->ldata_arr[target_nasm3->num_ldata - 1].num_ninst;

                for (int i=0; i<last_layer_num_ninst; i++) {
                    last_layer_ninst_arr_start[i].dev_send_target[DEV_EDGE] = 1;
                }
            }
            else if (dev_mode == DEV_LOCAL) init_allow_all(target_nasm3, 3);

            // Networking
            if(dev_mode == DEV_SERVER || dev_mode == DEV_EDGE) 
            {
                net_engine3 = init_networking(target_nasm3, rpool3, dev_mode, server_ip, server_port3, 0, 1);
                net_engine3->is_fl_offloading = 1;
                dse_group_set_net_engine(dse_group3, net_engine3);
                dse_group_set_device(dse_group3, dev_mode);
                net_engine3->dse_group = dse_group3;
                net_engine_set_operating_mode(net_engine3, OPER_MODE_FL_PATH);
            }

            PRTF ("Running %d iterations\n", number_of_iterations);
            start_time = get_sec();
            for (int i = 0; i < number_of_iterations; i++)
            {
                net_engine_run(net_engine3);
                rpool_reset_queue (rpool3);
                apu_reset_nasm(target_nasm3);
                dse_group_set_operating_mode(dse_group3, OPER_MODE_FL_PATH);
                if (dev_mode == DEV_EDGE) fl_push_path_ninsts_edge(rpool3, target_nasm3->path_ptr_arr[0]);
                else if (dev_mode == DEV_LOCAL) fl_push_path_ninsts(rpool3, target_nasm3->path_ptr_arr[0]);
                dse_group_run (dse_group3);
                dse_wait_for_nasm_completion (target_nasm3);

                if (dev_mode != DEV_LOCAL) {
                    unsigned int tx_remaining = atomic_load(&net_engine3->rpool->num_stored);
                    while (tx_remaining > 0) tx_remaining = atomic_load(&net_engine3->rpool->num_stored);
                    net_engine_wait_for_tx_queue_completion(net_engine3);
                    net_engine_reset(net_engine3);
                    net_engine_set_operating_mode(net_engine3, OPER_MODE_FL_PATH);
                }
                dse_group_stop (dse_group3);
                if (dev_mode != DEV_LOCAL) net_engine_stop (net_engine3);

                fl_reset_nasm_path(target_nasm3);

                dse_set_starting_path (target_nasm3->path_ptr_arr[0]);

            }
            end_time = get_sec();

            #ifndef SUPRESS_OUTPUT
            printf ("Time taken: %lf seconds\n", (end_time - start_time)/number_of_iterations);
            #else
            printf ("%lf\n", (end_time - start_time)/number_of_iterations);
            #endif

            if (strcmp(dnn, "bert_base") != 0 && strcmp(dnn, "yolov3") != 0)
            {
                LAYER_PARAMS output_order[] = {BATCH, OUT_C, OUT_H, OUT_W};
                float *layer_output = dse_get_nasm_result (target_nasm3, output_order);
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
            else if (strcmp(dnn, "yolov3") == 0)
            {
                int last_ldata_intsum = get_ldata_intsum(&target_nasm3->ldata_arr[target_nasm3->num_ldata - 1]);
                #ifndef SUPPRESS_OUTPUT
                printf("last layer intsum: %d\n", last_ldata_intsum);
                #endif
            }

            /* WRAP UP */
            fl_destroy_nasm_path(target_nasm3);

            aspen_flush_dynamic_memory ();

            net_engine_destroy (net_engine3);
            dse_group_destroy (dse_group3);
            rpool_destroy (rpool3);
            apu_destroy_nasm (target_nasm3);
            apu_destroy_dnn (target_dnn3);
        }
    }


    
    return 0;
}