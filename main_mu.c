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

    if(num_edge_devices > SCHEDULE_MAX_DEVICES)
    {
        printf("num_edge_devices should be less than %d\n", SCHEDULE_MAX_DEVICES);
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

    int is_conventional = !strcmp(schedule_policy, "conventional");
    int server_sock;
    int client_sock_arr[SCHEDULE_MAX_DEVICES];
    
    
    if (!strcmp(schedule_policy, "local"))
    {
        device_mode = DEV_LOCAL;
    }
    if (device_mode == DEV_SERVER) 
    {
        server_sock = create_server_sock(server_ip, server_port+1);
        for(int i = 0; i < num_edge_devices; i++)
        {
            client_sock_arr[i] = accept_client_sock(server_sock);
            write_n(client_sock_arr[i], &i, sizeof(int));
            printf("Edge %d is connected\n", i);
        }
    }
    else if (device_mode == DEV_EDGE) 
    {
        server_sock = connect_server_sock(server_ip, server_port+1);
        read_n(server_sock, &device_idx, sizeof(int));
        printf("Initialized to device idx: %d\n", device_idx);
    }
    else
    {
        schedule_policy = "local";
    }

    avg_ninst_profile_t *ninst_profile[SCHEDULE_MAX_DEVICES];
    network_profile_t *network_profile[SCHEDULE_MAX_DEVICES];

    sched_processor_t *schedule;
    dynamic_scheduler_t *dynamic_scheduler;

    aspen_dnn_t *target_dnn;
    nasm_t *target_nasm;

    rpool_t *rpool = rpool_init (gpu);
    if(device_mode == DEV_SERVER)
    {
        rpool_t *rpool_arr[num_edge_devices];
        for (int i = 0; i < num_edge_devices; i++) rpool_arr[i] = rpool_init (gpu);
    }
    dse_group_t *dse_group = dse_group_init (1, gpu);
    dse_group_set_rpool (dse_group, rpool);
    dse_group_set_multiuser (dse_group, 1);

    networking_engine* net_engine = NULL;
    networking_engine *net_engine_arr[SCHEDULE_MAX_DEVICES];

    // rpool_t *rpool = rpool_init (gpu);
    // dse_group_t *dse_group = dse_group_init (dse_num, gpu);
    // dse_group_set_rpool (dse_group, rpool);
    // networking_engine* net_engine = NULL;

    /** STAGE: PROFILING COMPUTATION FOR DYNAMIC OFFLOADING*/
    printf("STAGE: PROFILING COMPUTATION %d\n", device_idx);
    ninst_profile[device_idx] = profile_computation(target_dnn_dir, target_nasm_dir, target_input, gpu, 1);
    printf("\tTotal: %d\tAvg Computation Time: %fms\n", ninst_profile[device_idx]->num_ninsts, 
                                            ninst_profile[device_idx]->avg_computation_time*1000);

    
    
    /** STAGE: PROFILING NETWORK **/
    int connection_key;
    if (device_mode == DEV_SERVER) 
    {
        printf("STAGE: PROFILING NETWORK\n");
        for(int i = 0; i < num_edge_devices; i++)
        { 
            float time_offset = profile_network_sync(device_mode, server_sock, client_sock_arr[i]);
            set_time_offset(time_offset, device_mode);
            connection_key = 12534;
            write_n(client_sock_arr[i], &connection_key, sizeof(int));
            printf("\t[Edge Device %d]connection key: %d\n", i, connection_key);
            printf("\t[Edge Device %d]time_offset: %f\n", i, time_offset);
            network_profile[i] = profile_network(ninst_profile, device_mode, i, server_sock, client_sock_arr[i]);
            printf("\t[Edge Device %d]RTT: %fms Bandwidth: %fMbps\n", i, network_profile[i]->rtt * 1000.0, network_profile[i]->transmit_rate);
        }
    }
    else if (device_mode == DEV_EDGE) 
    {
        printf("STAGE: PROFILING NETWORK\n");
        float time_offset = profile_network_sync(device_mode, server_sock, 0);
        set_time_offset(time_offset, device_mode);
        connection_key = -1;
        read_n(server_sock, &connection_key, sizeof(int));
        printf("\t[Edge Device %d]connection key: %d\n", device_idx, connection_key);
        printf("\t[Edge Device %d]time_offset: %f\n", device_idx, time_offset);
        network_profile[device_idx] = profile_network(ninst_profile, device_mode, device_idx, server_sock, 0);
        printf("\t[Edge Device %d]RTT: %fms Bandwidth: %fMbps\n", device_idx, network_profile[device_idx]->rtt * 1000.0, network_profile[device_idx]->transmit_rate);
    }

    /** STAGE: SCHEDULING **/
    printf ("STAGE: SCHEDULING - %s\n", schedule_policy);
    if (!strcmp(schedule_policy, "partial")) 
        init_partial_offload(target_nasm, 0.0);
    if (!strcmp(schedule_policy, "conventional")) 
        init_sequential_offload(target_nasm, sched_sequential_idx, device_idx, DEV_SERVER);
    else if (!strcmp(schedule_policy, "sequential")) 
        init_sequential_offload(target_nasm, sched_sequential_idx, device_idx, DEV_SERVER);
    else if (!strcmp(schedule_policy, "partial")) 
        init_partial_offload(target_nasm, 0.0);
    else if (!strcmp(schedule_policy, "dynamic")) 
    {
        /** STAGE: SCHEDULING - DYNAMIC **/
        init_dynamic_offload(target_nasm, device_mode, device_idx);
        dynamic_scheduler = init_dynamic_scheduler(ninst_profile, network_profile, device_idx);
        dse_group_set_dynamic_scheduler(dse_group, dynamic_scheduler);
        printf("\t[Device %d]Init dynamic scheduler\n", device_idx);
        printf("\t[Device %d]Avg server ninst computation time: %fms\n", device_idx, dynamic_scheduler->avg_server_ninst_compute_time*1000);
        printf("\t[Device %d]Avg edge ninst computation time:   %fms\n", device_idx, dynamic_scheduler->avg_edge_ninst_compute_time*1000);
        printf("\t[Device %d]Avg bandwidth: %fMbps\n", device_idx, dynamic_scheduler->avg_bandwidth);
        printf("\t[Device %d]RTT: %fms\n", device_idx, dynamic_scheduler->rtt * 1000.0);
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

    // char* server_ip = "192.168.1.176";
    // int server_port_start = 3786;
    // int server_ports[SCHEDULE_MAX_DEVICES];
    // for (int i=1; i<SCHEDULE_MAX_DEVICES; i++) {
    //     server_ports[i] = server_port_start + i;
    // }

    // if (device_idx == DEV_SERVER) {
    //     for (int i = 1; i < num_edge_devices; i++) {
    //         if (!strcmp(schedule_policy, "conventional")) {
    //             dse_group_init_enable_device(dse_group, num_edge_devices);
    //         }
    //         else {
    //             for (int i = 0; i < num_edge_devices; i++) {
    //                 dse_group_set_enable_device(dse_group, i, 1);
    //             }
    //         }

    //         net_engine_arr[i] = init_networking(target_nasm[i], rpool_arr[i], DEV_SERVER, server_ip, server_ports[i], 0, !is_conventional);
    //         dse_group_add_netengine_arr(dse_group, net_engine_arr[i], i);
    //         dse_group_set_device(dse_group, device_idx);
    //         net_engine_arr[i]->dse_group = dse_group;
    //         net_engine_arr[i]->device_idx = i;
        
    //         atomic_store (&net_engine_arr[i]->run, 1);
    //     }
    // }
    // else {
    //     net_engine = init_networking(target_nasm[device_idx], rpool, DEV_EDGE, server_ip, server_ports[device_idx], 0, !is_conventional);
    //     dse_group_add_netengine_arr(dse_group, net_engine, 0);
    //     dse_group_set_device(dse_group, device_idx);
    //     net_engine->dse_group = dse_group;
    //     add_input_rpool (net_engine, target_nasm[device_idx], target_input);
        
    //     atomic_store (&net_engine->run, 1);
    // }

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
