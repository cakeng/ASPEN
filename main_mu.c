#include <stdio.h>
#include "aspen.h"
#include "util.h"
#include "nasm.h"
#include "apu.h"
#include "networking.h"
#include "scheduling.h"
#include "profiling.h"
#include "subcmdline.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

int total_transferred = 0;

#define SCHED_HEFT      0
#define SCHED_PARTIAL   1
#define SCHED_DYNAMIC   2

// OPTIONS
// option "device_mode" - "" int required
// option "dirname" - "" string required
// option "prefix" - "" string optional
// option "log_idx_start" - "" int optional
// option "inference_repeat_num" - "" int optional
// option "target_dnn_dir" - "" string required
// option "target_nasm_dir" - "" string required
// option "target_input" - "" string required
// option "server_ip" - "" string required
// option "server_port" - "" int required
// option "schedule_policy" - "" string required values="partial","2"
// option "sched_partial_ratio" - "" float optional
// option "sched_sequential_idx" - "" int optional
// option "dse_num" - "" int required
// option "output_order" - "" string required values="cnn","transformer"

int main(int argc, char **argv)
{
    print_aspen_build_info();
    struct gengetopt_args_info ai;
    if (cmdline_parser(argc, argv, &ai) != 0) {
        exit(1);
    }

    DEVICE_MODE device_mode = ai.device_mode_arg; // 0: SERVER, 1: EDGE
    char *dirname = ai.dirname_arg;
    char *prefix = ai.prefix_arg ? ai.prefix_arg : "temp";
    int log_idx_start = ai.log_idx_start_arg;
    int inference_repeat_num = ai.inference_repeat_num_arg;
    char *target_dnn_dir = ai.target_dnn_dir_arg;
    char *target_nasm_dir = ai.target_nasm_dir_arg;
    char *target_input = ai.target_input_arg;
    char *server_ip = ai.server_ip_arg;
    int server_port = ai.server_port_arg;
    char *schedule_policy = ai.schedule_policy_arg;
    float sched_partial_ratio = ai.sched_partial_ratio_arg;
    int sched_sequential_idx = ai.sched_sequential_idx_arg;
    int dse_num = ai.dse_num_arg;
    int num_edge_devices = ai.num_edge_devices_arg;
    char *output_order = ai.output_order_arg;
    char nasm_name[256] = {0};
    int device_idx = device_mode;
    
    int gpu = -1;

    avg_ninst_profile_t *ninst_profile[SCHEDULE_MAX_DEVICES];
    network_profile_t *network_profile[SCHEDULE_MAX_DEVICES];

    sched_processor_t *schedule;
    dynamic_scheduler_t *dynamic_scheduler;

    aspen_dnn_t *target_dnn[SCHEDULE_MAX_DEVICES];
    nasm_t *target_nasm[SCHEDULE_MAX_DEVICES];

    int is_conventional = !strcmp(schedule_policy, "conventional");
    int server_sock;
    int client_sock_arr[SCHEDULE_MAX_DEVICES];

    int control_server_port = server_port + SCHEDULE_MAX_DEVICES;
    int server_port_start = server_port;
    int server_ports[SCHEDULE_MAX_DEVICES];
    for (int i = 0; i < SCHEDULE_MAX_DEVICES; i++) {
        server_ports[i] = server_port_start + i;
    }

    if(num_edge_devices > SCHEDULE_MAX_DEVICES-1)
    {
        printf("num_edge_devices should be less than %d\n", SCHEDULE_MAX_DEVICES-1);
        exit(1);
    }

    // Get only the name of the target nasm file without the path and extension
    if (target_nasm_dir && strlen(target_nasm_dir) > 0) 
    {
        char *nasm_name_with_ext = strrchr(target_nasm_dir, '/');
        if (nasm_name_with_ext) 
            nasm_name_with_ext++;
        else 
            nasm_name_with_ext = nasm_name;
        char *nasm_name_ext_end = strrchr(nasm_name_with_ext, '.');
        if (nasm_name_ext_end) 
            strncpy(nasm_name, nasm_name_with_ext, nasm_name_ext_end - nasm_name_with_ext);
        else 
            strcpy(nasm_name, nasm_name_with_ext);
        printf ("nasm_name: %s\n", nasm_name);
    }

    /** STAGE: Initialization **/
    printf("STAGE: Initialization\n");
    if (!strcmp(schedule_policy, "local"))
    {
        device_mode = DEV_LOCAL;
    }
    if (device_mode == DEV_SERVER) 
    {
        server_sock = create_server_sock(server_ip, control_server_port);
        for(int i = 0; i < num_edge_devices; i++)
        {
            client_sock_arr[i] = accept_client_sock(server_sock);
            write_n(client_sock_arr[i], &i, sizeof(int));
            printf("\tEdge %d is connected\n", i);
        }
        device_idx = num_edge_devices;
    }
    else if (device_mode == DEV_EDGE) 
    {
        server_sock = connect_server_sock(server_ip, control_server_port);
        read_n(server_sock, &device_idx, sizeof(int));
    }
    else
    {
        schedule_policy = "local";
    }
    printf("\tInitialized to device idx: %d\n", device_idx);



    // rpool_t *rpool = rpool_init (gpu);
    rpool_t *rpool_arr[SCHEDULE_MAX_DEVICES];
    rpool_arr[device_idx] = rpool_init(gpu);

    if(device_mode == DEV_SERVER)
    {
        for(int edge_id = 0; edge_id < num_edge_devices; edge_id++)
            rpool_arr[edge_id] = rpool_init(gpu);
    }

    dse_group_t *dse_group = dse_group_init (dse_num, gpu);

    dse_group_set_rpool (dse_group, rpool_arr[device_idx]);
    dse_group_set_device_mode (dse_group, device_mode);
    dse_group_set_device (dse_group, device_idx);
    dse_group_set_num_edge_devices (dse_group, num_edge_devices);
    

    if(device_mode == DEV_SERVER)
    {
        for(int edge_id = 0; edge_id < num_edge_devices; edge_id++)
            dse_group_add_rpool_arr(dse_group, rpool_arr[edge_id], edge_id);
    }

    networking_engine* net_engine = NULL;
    networking_engine *net_engine_arr[SCHEDULE_MAX_DEVICES];

    char* target_nasm_dirs[SCHEDULE_MAX_DEVICES];
    char* target_dnn_dirs[SCHEDULE_MAX_DEVICES];
    char* target_inputs[SCHEDULE_MAX_DEVICES];

    target_nasm_dirs[device_idx] = target_nasm_dir;
    target_dnn_dirs[device_idx] = target_dnn_dir;
    target_inputs[device_idx] = target_input;

    /** STAGE: PARTITIONING **/
    printf("STAGE: PARTITIONING\n");
    if(device_mode == DEV_SERVER)
    {
        for(int i = 0; i < num_edge_devices; i++)
        {
            int target_nasm_dir_len, target_dnn_dir_len, target_input_len;
            read_n(client_sock_arr[i], &target_nasm_dir_len, sizeof(int));
            read_n(client_sock_arr[i], &target_dnn_dir_len, sizeof(int));
            read_n(client_sock_arr[i], &target_input_len, sizeof(int));

            target_nasm_dirs[i] = (char*)malloc(target_nasm_dir_len);
            target_dnn_dirs[i] = (char*)malloc(target_dnn_dir_len);
            target_inputs[i] = (char*)malloc(target_input_len);

            read_n(client_sock_arr[i], target_nasm_dirs[i], target_nasm_dir_len);
            read_n(client_sock_arr[i], target_dnn_dirs[i], target_dnn_dir_len);
            read_n(client_sock_arr[i], target_inputs[i], target_input_len);

            printf("\t[EDGE %d] NASM: %s\n", i, target_nasm_dirs[i]);
            printf("\t[EDGE %d] DNN: %s\n", i, target_dnn_dirs[i]);
            printf("\t[EDGE %d] INPUT: %s\n", i, target_inputs[i]);
            
            target_dnn[i] = apu_load_dnn_from_file(target_dnn_dirs[i]);
            target_nasm[i] = apu_load_nasm_from_file(target_nasm_dirs[i], target_dnn[i]);
            
        }
    }
    else if (device_mode == DEV_EDGE)
    {
        int target_nasm_dir_len = strlen(target_nasm_dirs[device_idx]);
        int target_dnn_dir_len = strlen(target_dnn_dirs[device_idx]);
        int target_input_len = strlen(target_inputs[device_idx]);
        write_n(server_sock, &target_nasm_dir_len, sizeof(int));
        write_n(server_sock, &target_dnn_dir_len, sizeof(int));
        write_n(server_sock, &target_input_len, sizeof(int));
        write_n(server_sock, target_nasm_dirs[device_idx], target_nasm_dir_len);
        write_n(server_sock, target_dnn_dirs[device_idx], target_dnn_dir_len);
        write_n(server_sock, target_inputs[device_idx], target_input_len);

        target_dnn[device_idx] = apu_load_dnn_from_file(target_dnn_dirs[device_idx]);
        target_nasm[device_idx] = apu_load_nasm_from_file(target_nasm_dirs[device_idx], target_dnn[device_idx]);
    }

    /** STAGE: PROFILING COMPUTATION FOR DYNAMIC OFFLOADING*/
    printf("STAGE: PROFILING COMPUTATION %d\n", device_idx);
    
    for(int edge_id = 0; edge_id < num_edge_devices; edge_id++)
    {
        if(device_mode == DEV_SERVER || device_idx == edge_id)
        {
            ninst_profile[edge_id] = profile_computation(target_nasm[edge_id], dse_num, device_idx, target_inputs[edge_id], device_mode, gpu, 1);
            float avg_computation_time = device_mode == DEV_SERVER ? ninst_profile[edge_id]->avg_server_computation_time : ninst_profile[edge_id]->avg_edge_computation_time;
            printf("\tTotal: %d\tAvg Computation Time: %fms\n", ninst_profile[edge_id]->num_ninsts, 
                                                avg_computation_time*1000);
        }
        
    }
    
    /** STAGE: PROFILING NETWORK **/
    printf("STAGE: PROFILING NETWORK\n");
    int connection_key;
    if (device_mode == DEV_SERVER) 
    {
        for(int i = 0; i < num_edge_devices; i++)
        { 
            float time_offset = profile_network_sync(device_mode, server_sock, client_sock_arr[i]);
            set_time_offset(time_offset, device_mode);
            connection_key = 12534;
            write_n(client_sock_arr[i], &connection_key, sizeof(int));
            printf("\t[Edge Device %d]connection key: %d\n", i, connection_key);
            // printf("\t[Edge Device %d]time_offset: %f\n", i, time_offset);
            network_profile[i] = profile_network(device_mode, i, server_sock, client_sock_arr[i]);
            // printf("\t[Edge Device %d]RTT: %fms Bandwidth: %fMbps\n", i, network_profile[i]->rtt * 1000.0, network_profile[i]->transmit_rate);
        }
    }
    else if (device_mode == DEV_EDGE) 
    {
        float time_offset = profile_network_sync(device_mode, server_sock, 0);
        set_time_offset(time_offset, device_mode);
        connection_key = -1;
        read_n(server_sock, &connection_key, sizeof(int));
        printf("\t[Edge Device %d]connection key: %d\n", device_idx, connection_key);
        printf("\t[Edge Device %d]time_offset: %f\n", device_idx, time_offset);
        network_profile[device_idx] = profile_network(device_mode, device_idx, server_sock, 0);
        printf("\t[Edge Device %d]RTT: %fms Bandwidth: %fMbps\n", device_idx, network_profile[device_idx]->rtt * 1000.0, network_profile[device_idx]->transmit_rate);
    }
    
    // Communicate profiles
    if(device_mode == DEV_SERVER)
    {
        for(int i = 0 ; i < num_edge_devices; i++)
            communicate_profiles_server(client_sock_arr[i], network_profile[i], ninst_profile[i]);
    }
    else if(device_mode == DEV_EDGE)
        communicate_profiles_edge(server_sock, network_profile[device_idx], ninst_profile[device_idx]);

    /** STAGE: SCHEDULING **/
    printf ("STAGE: SCHEDULING - %s\n", schedule_policy);
    if (!strcmp(schedule_policy, "partial")) 
        init_partial_offload(target_nasm[device_idx], 0.0);
    if (!strcmp(schedule_policy, "conventional"))
    {
        for(int edge_id = 0; edge_id < num_edge_devices; edge_id++)
        {
            if(device_mode == DEV_SERVER || device_idx == edge_id)
                init_sequential_offload(target_nasm[edge_id], sched_sequential_idx, edge_id, num_edge_devices); // server idx == num_edge_devices
        }
    }
    else if (!strcmp(schedule_policy, "sequential")) 
    {
        for(int edge_id = 0; edge_id < num_edge_devices; edge_id++)
        {
            if(device_mode == DEV_SERVER || device_idx == edge_id)
                init_sequential_offload(target_nasm[edge_id], sched_sequential_idx, edge_id, num_edge_devices); // server idx == num_edge_devices
        }
    }
    else if (!strcmp(schedule_policy, "partial")) 
        init_partial_offload(target_nasm[device_idx], 0.0);
    else if (!strcmp(schedule_policy, "dynamic")) 
    {
        // Initialized scheduler
        for(int i = 0; i < num_edge_devices; i++)
        {
            if(device_mode == DEV_SERVER || device_idx == i)
                init_dynamic_offload(target_nasm[i], device_mode, i);
        }
            
        dynamic_scheduler = init_dynamic_scheduler(ninst_profile, network_profile, device_mode, device_idx, num_edge_devices);
        dse_group_set_dynamic_scheduler(dse_group, dynamic_scheduler);
        
        printf("\t[Device %d]Init dynamic scheduler\n", device_idx);
        for(int i = 0; i < num_edge_devices; i++)
        {
            if(device_mode == DEV_SERVER || device_idx == i)
            {
                printf("\t[Device %d]Avg server ninst computation time: %fms\n", i, dynamic_scheduler->avg_server_ninst_compute_time[i]*1000);
                printf("\t[Device %d]Avg edge ninst computation time:   %fms\n", i, dynamic_scheduler->avg_edge_ninst_compute_time[i]*1000);
                printf("\t[Device %d]Avg bandwidth: %fMbps\n", i, dynamic_scheduler->avg_bandwidth[i]);
                printf("\t[Device %d]RTT: %fms\n", i, dynamic_scheduler->rtt[i] * 1000.0);
            }
        }
    }
    else if (!strcmp(schedule_policy, "local")) 
    {
        dse_group_set_device (dse_group, DEV_SERVER);
        init_sequential_offload (target_nasm, 0, DEV_SERVER, DEV_SERVER);
    }
    else
    {
        printf("ERROR: Unknown schedule policy: %s\n", schedule_policy);
        exit(1);
    }

    /** STAGE: INFERENCE **/

    printf("STAGE: INFERENCE\n");
    if (!strcmp(schedule_policy, "local"))
    {
        rpool_add_nasm (rpool_arr[device_idx], target_nasm, target_input);

        double start_time = get_time_secs();
        for (int i = 0; i < inference_repeat_num; i++)
        {
            rpool_reset (rpool_arr[device_idx]);
            rpool_reset_nasm (rpool_arr[device_idx], target_nasm);
            dse_group_run (dse_group);
            dse_wait_for_nasm_completion (target_nasm);
            dse_group_stop (dse_group);
        }

        double end_time = get_time_secs();
        printf ("Time taken: %lf seconds\n", (end_time - start_time)/inference_repeat_num);
    }
    else
    {
        dse_group_set_multiuser (dse_group, 1);
        if (device_mode == DEV_SERVER) {
            if (!strcmp(schedule_policy, "conventional")) 
                dse_group_init_enable_device(dse_group, num_edge_devices);
            else 
            {
                for (int edge_id = 0; edge_id < num_edge_devices; edge_id++) {
                    dse_group_set_enable_device(dse_group, edge_id, 1);
                }
            }
        }

        for(int edge_id = 0; edge_id < num_edge_devices; edge_id++)
        {
            if(device_mode == DEV_SERVER || device_idx == edge_id)
            {
                net_engine_arr[edge_id] = init_networking(target_nasm[edge_id], rpool_arr[edge_id], device_mode, server_ip, server_ports[edge_id], 0, !is_conventional);
                dse_group_add_netengine_arr(dse_group, net_engine_arr[edge_id], edge_id);
                // dse_group_set_device(dse_group, device_idx);
                net_engine_arr[edge_id]->device_idx = edge_id;
                net_engine_arr[edge_id]->dse_group = dse_group;
            }
        }

        for(int inf_num = 0; inf_num < inference_repeat_num; inf_num++)
        {
            // synchronize
            printf("[Inference %d] inference: %d/%d\n", inf_num+1, inf_num+1, inference_repeat_num);

            // if(device_mode == DEV_SERVER)
            // {
            //     for(int edge_id = 0; edge_id < num_edge_devices; edge_id)
            //     {
            //         connection_key = 12534+inf_num;
            //         float time_offset = profile_network_sync(device_mode, server_sock, client_sock_arr[edge_id]);
            //         set_time_offset(time_offset, device_mode);
            //         write_n(client_sock_arr[edge_id], &connection_key, sizeof(int));
            //         printf("\t[Edge Device %d]connection key: %d\n", edge_id, connection_key);
            //         printf("\t[Edge Device %d]time_offset: %f\n", edge_id, time_offset);

            //     }
            // }
            // else
            // {   
            //     float time_offset = profile_network_sync(device_mode, server_sock, 0);
            //     set_time_offset(time_offset, device_mode);
            //     connection_key = -1;
            //     read_n(server_sock, &connection_key, sizeof(int));
            //     printf("\t[Edge Device %d]connection key: %d\n", device_idx, connection_key);
            //     printf("\t[Edge Device %d]time_offset: %f\n", device_idx, time_offset);
            // }

            for(int edge_id = 0; edge_id < num_edge_devices; edge_id++)
            {
                if(device_mode == DEV_SERVER || device_idx == edge_id)
                {
                    remove_inference_whitelist(net_engine_arr[edge_id], target_nasm[edge_id]->inference_id);
                    net_engine_reset(net_engine_arr[edge_id]);
                    rpool_reset(rpool_arr[edge_id]);
                    apu_reset_nasm(target_nasm[edge_id]);

                    if (!strcmp(schedule_policy, "dynamic")) init_dynamic_offload(target_nasm[edge_id], device_mode, edge_id);

                    set_nasm_inference_id(target_nasm[edge_id], connection_key);
                    add_inference_whitelist(net_engine_arr[edge_id], target_nasm[edge_id]->inference_id);
                    target_nasm[edge_id]->nasm_cond = (pthread_cond_t)PTHREAD_COND_INITIALIZER;
                }
            }

            if(device_mode == DEV_EDGE)
                add_input_rpool_reverse (net_engine_arr[device_idx], target_nasm[device_idx], target_inputs[device_idx]);

            set_elapsed_time_start ();
            for(int edge_id = 0; edge_id < num_edge_devices; edge_id++)
            {
                if(device_mode == DEV_SERVER || device_idx == edge_id)
                {
                    net_engine_run (net_engine_arr[edge_id]);
                }
            }

            if (!(device_mode == DEV_SERVER && is_conventional)) 
            {
                printf ("[Inference %d] Running DSEs...\n", inf_num+1);
                dse_group_run (dse_group);
            }

            for(int edge_id = 0; edge_id < num_edge_devices; edge_id++)
            {
                if(device_mode == DEV_SERVER || device_idx == edge_id)
                {
                    dse_wait_for_nasm_completion (target_nasm[edge_id]);
                }
            }
            
            get_elapsed_time ("run_aspen");
            dse_group_stop (dse_group);
            
            for(int edge_id = 0; edge_id < num_edge_devices; edge_id++)
            {
                if (device_mode == DEV_SERVER || edge_id == device_idx)
                {
                    net_engine_wait_for_tx_queue_completion (net_engine_arr[edge_id]);
                    net_engine_stop (net_engine_arr[edge_id]);
                }
            }

            for(int edge_id = 0; edge_id < num_edge_devices; edge_id++)
            {
                if(device_mode == DEV_SERVER || device_idx == edge_id)
                {
                    printf("---------------------[Edge %d] Inference result---------------------\n", edge_id);
                    LAYER_PARAMS output_order_cnn[] = {BATCH, OUT_H, OUT_W, OUT_C};  // for CNN
                    LAYER_PARAMS output_order_transformer[] = {BATCH, MAT_N, MAT_M};    // for Transformer
                    LAYER_PARAMS *output_order_param = !strcmp(output_order, "cnn") ? output_order_cnn : output_order_transformer;
                    float *layer_output = dse_get_nasm_result (target_nasm[edge_id], output_order_param);
                    float *softmax_output = calloc (1000*target_nasm[edge_id]->batch_size, sizeof(float));
                    naive_softmax (layer_output, softmax_output, target_nasm[edge_id]->batch_size, 1000);
                    for (int i = 0; i < target_nasm[edge_id]->batch_size; i++)
                    {
                        get_probability_results ("data/imagenet_classes.txt", softmax_output + 1000*i, 1000);
                    }
                    free (layer_output);
                    free (softmax_output);
                }

                // For logging
                // char file_name[1024];
                // char dir_path[1024];
                // sprintf(dir_path, "./logs/%s", dirname);

                // struct stat st = {0};
                // if (stat("./logs/", &st) == -1) 
                // {
                //     mkdir("./logs/", 0700);
                // }
                // if (stat(dir_path, &st) == -1) 
                // {
                //     mkdir(dir_path, 0700);
                // }

                // sprintf(file_name, "./logs/%s/%s_%s_%s_%s_Iter%d.csv", 
                //     dirname, 
                //     prefix, 
                //     schedule_policy, 
                //     device_mode == DEV_SERVER ? "SERVER" : "EDGE", 
                //     nasm_name, 
                //     log_idx_start+inf_num);
                
                // FILE *log_fp = fopen(file_name, "w");
                // save_ninst_log(log_fp, target_nasm[edge_id]); 
            }
        }
    }

    // // SYNC HERE
    // float sync_key;
    // float sync;
    // int control_server_sock;
    // int client_sock_arr[SCHEDULE_MAX_DEVICES];

    // if (device_idx == 0) {
    //     control_server_sock = create_server_sock(server_ip, server_port_start);
    //     for (int i=1; i<SCHEDULE_MAX_DEVICES; i++) {
    //         client_sock_arr[i] = accept_client_sock(control_server_sock);
    //     }
    //     for (int i=1; i<SCHEDULE_MAX_DEVICES; i++) {
    //         sync_key = get_time_secs();
    //         printf("SYNC KEY SEND %d: %f\n", i, sync_key);
    //         sync += sync_key/2;
    //         write_n(client_sock_arr[i], &sync_key, sizeof(float));
    //         read_n(client_sock_arr[i], &sync_key, sizeof(float));
    //         printf("SYNC KEY RECV %d: %f\n", i, sync_key);
    //         sync -= sync_key;
    //         sync_key = get_time_secs();
    //         printf("SYNC KEY LAST %d: %f\n", i, sync_key);
    //         sync += sync_key/2;
    //         printf("SYNC %d: %f\n", i, sync);
            
    //         close(client_sock_arr[i]);
    //     }
    //     close(control_server_sock);
    // }
    // else {
    //     sleep(5 + device_idx);
    //     control_server_sock = connect_server_sock(server_ip, server_port_start);
    //     read_n(control_server_sock, &sync_key, sizeof(float));
    //     sync_key = get_time_secs();
    //     write_n(control_server_sock, &sync_key, sizeof(float));
    //     close(control_server_sock);
    //     printf("SYNC KEY: %f\n", sync_key);
    // }
    
    // get_elapsed_time ("init");
    // if (!sequential || device_idx != DEV_SERVER) dse_group_run (dse_group);
    // if (device_idx == DEV_SERVER) {
    //     for (int i=1; i<SCHEDULE_MAX_DEVICES; i++) {
    //         dse_wait_for_nasm_completion (target_nasm[i]);
    //     }
    // }
    // else {
    //     dse_wait_for_nasm_completion (target_nasm[device_idx]);
    // }
    
    // get_elapsed_time ("run_aspen");
    // dse_group_stop (dse_group);
    
    // if (device_idx > 0) {
    //     LAYER_PARAMS output_order[] = {BATCH, OUT_H, OUT_W, OUT_C};
    //     float *layer_output = dse_get_nasm_result (target_nasm[device_idx], output_order);
    //     float *softmax_output = calloc (1000*target_nasm[device_idx]->batch_size, sizeof(float));
    //     naive_softmax (layer_output, softmax_output, target_nasm[device_idx]->batch_size, 1000);
    //     for (int i = 0; i < target_nasm[device_idx]->batch_size; i++)
    //     {
    //         get_probability_results ("data/resnet50/imagenet_classes.txt", softmax_output + 1000*i, 1000);
    //     }

    //     free (layer_output);
    //     free (softmax_output);
    // }
    

    // // WRAP UP
    // char file_name[256];
    // if (device_idx == 0) {
    //     FILE *log_fp;
        
    //     for (int i=1; i<SCHEDULE_MAX_DEVICES; i++) {
    //         sprintf(file_name, "./logs/multiuser/%s_dev%d_RX.txt", (sequential ? "seq" : "pip"), i);
    //         log_fp = fopen(file_name, "w");

    //         save_ninst_log(log_fp, target_nasm[i]);
    //         net_engine_destroy (net_engine_arr[i]);
    //         apu_destroy_nasm (target_nasm[i]);
    //         apu_destroy_dnn (target_dnn[i]);
    //     }
    // }
    // else {
    //     sprintf(file_name, "./logs/multiuser/%s_dev%d_TX.txt", (sequential ? "seq" : "pip"), device_idx);
    //     FILE *log_fp = fopen(file_name, "w");

    //     save_ninst_log(log_fp, target_nasm[device_idx]);
    //     net_engine_destroy (net_engine);
    //     apu_destroy_nasm (target_nasm[device_idx]);
    //     apu_destroy_dnn (target_dnn[device_idx]);
    // }
    // dse_group_destroy (dse_group);
    // rpool_destroy (rpool);

    // printf("total transferred: %d\n", total_transferred);
    // return 0;
}
