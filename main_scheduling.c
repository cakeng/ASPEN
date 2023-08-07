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
    char *output_order = ai.output_order_arg;
    char nasm_name[256] = {0};
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

    int gpu = -1;
    int is_conventional = !strcmp(schedule_policy, "conventional");
    int server_sock;
    int client_sock;

    if (!strcmp(schedule_policy, "local"))
    {
        device_mode = DEV_LOCAL;
    }
    if (device_mode == DEV_SERVER) 
    {
        server_sock = create_server_sock(server_ip, server_port+1);
        client_sock = accept_client_sock(server_sock);
    }
    else if (device_mode == DEV_EDGE) 
    {
        server_sock = connect_server_sock(server_ip, server_port+1);
    }
    else
    {
        schedule_policy = "local";
    }
    

    ninst_profile_t *ninst_profile[SCHEDULE_MAX_DEVICES];
    network_profile_t *network_profile;

    sched_processor_t *schedule;
    dynamic_scheduler_t *scheduler;

    aspen_dnn_t *target_dnn;
    nasm_t *target_nasm;

    rpool_t *rpool = rpool_init (gpu);
    dse_group_t *dse_group = dse_group_init (dse_num, gpu);
    dse_group_set_rpool (dse_group, rpool);
    networking_engine* net_engine = NULL;

    /** STAGE: PROFILING COMPUTATION FOR DYNAMIC OFFLOADING*/
    if(!strcmp(schedule_policy, "dynamic"))
    {
        printf("STAGE: PROFILING COMPUTATION %d\n", device_mode);
        ninst_profile[device_mode] = profile_computation(target_dnn_dir, target_nasm_dir, target_input, gpu, 1);
        printf("\tTotal: %d\tTransmit Size: %d\tComputation Time: %f\n", ninst_profile[device_mode][256].total, 
                                            ninst_profile[device_mode][256].transmit_size,
                                            ninst_profile[device_mode][256].computation_time);
    }

    /** STAGE: PROFILING NETWORK **/

    int connection_key;
    if (device_mode == DEV_SERVER) 
    {
        printf("STAGE: PROFILING NETWORK\n");
        float sync = profile_network_sync(device_mode, server_sock, client_sock);
        connection_key = 12534;
        write_n(client_sock, &connection_key, sizeof(int));
        printf("\tconnection key: %d\n", connection_key);
        printf("\tsync: %f\n", sync);
        network_profile = profile_network(ninst_profile, device_mode, server_sock, client_sock);
        printf("\tRTT: %fms Bandwidth: %fMbps\n", network_profile->rtt, network_profile->transmit_rate);
    }
    else if (device_mode == DEV_EDGE) 
    {
        printf("STAGE: PROFILING NETWORK\n");
        float sync = profile_network_sync(device_mode, server_sock, client_sock);
        connection_key = -1;
        read_n(server_sock, &connection_key, sizeof(int));
        printf("\tconnection key: %d\n", connection_key);
        printf("\tsync: %f\n", sync);
        network_profile = profile_network(ninst_profile, device_mode, server_sock, client_sock);
        printf("\tRTT: %fms Bandwidth: %fMbps\n", network_profile->rtt, network_profile->transmit_rate);
    }
    
    target_dnn = apu_load_dnn_from_file(target_dnn_dir);
    target_nasm = apu_load_nasm_from_file(target_nasm_dir, target_dnn);

    /** STAGE: SCHEDULING **/
    printf ("STAGE: SCHEDULING - %s\n", schedule_policy);
    if (!strcmp(schedule_policy, "partial")) 
        init_partial_offload(target_nasm, 0.0);
    if (!strcmp(schedule_policy, "conventional")) 
        init_sequential_offload(target_nasm, sched_sequential_idx, DEV_EDGE, DEV_SERVER);
    else if (!strcmp(schedule_policy, "sequential")) 
        init_sequential_offload(target_nasm, sched_sequential_idx, DEV_EDGE, DEV_SERVER);
    else if (!strcmp(schedule_policy, "partial")) 
        init_partial_offload(target_nasm, 0.0);
    else if (!strcmp(schedule_policy, "dynamic")) 
    {
        /** STAGE: SCHEDULING - DYNAMIC **/
        for (int i=0; i<dse_group->num_ases; i++)
            dse_group->dse_arr[i].is_dynamic_scheduling = 1;
        init_dynamic_offload(target_nasm);
        scheduler = init_dynamic_scheduler(ninst_profile, network_profile);
        printf("\tInit dynamic scheduler\n");
        printf("\tAvg server ninst computation time: %fms\n", scheduler->avg_server_ninst_compute_time);
        printf("\tAvg edge ninst computation time:   %fms\n", scheduler->avg_edge_ninst_compute_time);
        printf("\tAvg bandwidth: %fMbps\n", scheduler->avg_bandwidth);
        printf("\tRTT: %fms\n", scheduler->rtt);
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

    // print_nasm_info (target_nasm, 1, 0);
    
    /** STAGE: INFERENCE **/

    printf("STAGE: INFERENCE\n");
    
    if (!strcmp(schedule_policy, "local"))
    {
        rpool_add_nasm (rpool, target_nasm, target_input);

        double start_time = get_time_secs();
        for (int i = 0; i < inference_repeat_num; i++)
        {
            rpool_reset (rpool);
            rpool_reset_nasm (rpool, target_nasm);
            dse_group_run (dse_group);
            dse_wait_for_nasm_completion (target_nasm);
            dse_group_stop (dse_group);
        }

        double end_time = get_time_secs();
        printf ("Time taken: %lf seconds\n", (end_time - start_time)/inference_repeat_num);
    }
    else
    {
        
        net_engine = init_networking(target_nasm, rpool, device_mode, server_ip, server_port, 0, !is_conventional);
        dse_group_set_net_engine(dse_group, net_engine);
        dse_group_set_device(dse_group, device_mode);
        net_engine->device_idx = device_mode;
        net_engine->dse_group = dse_group;

        for (int i=0; i<inference_repeat_num; i++) 
        {
            // synchronize
            remove_inference_whitelist(net_engine, target_nasm->inference_id);
            printf("Sync between inference...\n");

            float sync = profile_network_sync(device_mode, server_sock, client_sock);

            int connection_key;
            if (device_mode == DEV_SERVER) {
                connection_key = 12534+i;
                write_n(client_sock, &connection_key, sizeof(int));
                printf("connection key: %d\n", connection_key);
            }
            else if (device_mode == DEV_EDGE) {
                connection_key = -1;
                read_n(server_sock, &connection_key, sizeof(int));
                printf("connection key: %d\n", connection_key);
            }

            printf("sync: %f\n", sync);

            net_engine_reset(net_engine);
            rpool_reset(rpool);
            apu_reset_nasm(target_nasm);
            
            if (!strcmp(schedule_policy, "dynamic")) init_dynamic_offload(target_nasm);

            
            set_nasm_inference_id(target_nasm, connection_key);
            add_inference_whitelist(net_engine, target_nasm->inference_id);

            printf("inference: %d/%d\n", i+1, inference_repeat_num);
            printf("inference id: %d\n", target_nasm->inference_id);

            target_nasm->nasm_cond = (pthread_cond_t)PTHREAD_COND_INITIALIZER;

            if(device_mode == DEV_EDGE) 
                add_input_rpool (net_engine, target_nasm, target_input);

            // for (int i = 0; i < 2; i++)
            //     print_ldata_info (&target_nasm->ldata_arr[i], 1, 0);


            set_elapsed_time_start ();
            net_engine_run (net_engine);
            // Do not start SERVER DSEs in conventional mode 
            // (they are started when the offloaded layers are all downloaded.)
            if (!(device_mode == DEV_SERVER && is_conventional)) 
            {
                printf ("Running DSEs...\n");
                dse_group_run (dse_group);
            }
            dse_wait_for_nasm_completion (target_nasm);
            get_elapsed_time ("run_aspen");
            dse_group_stop (dse_group);
            if (device_mode == DEV_SERVER)
                net_engine_wait_for_tx_queue_completion (net_engine);
            net_engine_stop (net_engine);

            
            LAYER_PARAMS output_order_cnn[] = {BATCH, OUT_H, OUT_W, OUT_C};  // for CNN
            LAYER_PARAMS output_order_transformer[] = {BATCH, MAT_N, MAT_M};    // for Transformer
            LAYER_PARAMS *output_order_param = !strcmp(output_order, "cnn") ? output_order_cnn : output_order_transformer;
            float *layer_output = dse_get_nasm_result (target_nasm, output_order_param);
            float *softmax_output = calloc (1000*target_nasm->batch_size, sizeof(float));
            naive_softmax (layer_output, softmax_output, target_nasm->batch_size, 1000);
            for (int i = 0; i < target_nasm->batch_size; i++)
            {
                get_probability_results ("data/imagenet_classes.txt", softmax_output + 1000*i, 1000);
            }
            free (layer_output);
            free (softmax_output);
            
            // For logging
            char file_name[1024];
            char dir_path[1024];
            sprintf(dir_path, "./logs/%s", dirname);

            struct stat st = {0};
            if (stat("./logs/", &st) == -1) 
            {
                mkdir("./logs/", 0700);
            }
            if (stat(dir_path, &st) == -1) 
            {
                mkdir(dir_path, 0700);
            }

            sprintf(file_name, "./logs/%s/%s_%s_%s_%s_Iter%d.csv", dirname, prefix, schedule_policy, device_mode == DEV_SERVER ? "SERVER" : "EDGE", 
                nasm_name, log_idx_start+i);
            
            FILE *log_fp = fopen(file_name, "w");
            save_ninst_log(log_fp, target_nasm);

        }
    }

    // Save output
    LAYER_PARAMS output_order_cnn[] = {BATCH, OUT_H, OUT_W, OUT_C};  // for CNN
    LAYER_PARAMS output_order_transformer[] = {BATCH, MAT_N, MAT_M};    // for Transformer
    LAYER_PARAMS *output_order_param = !strcmp(output_order, "cnn") ? output_order_cnn : output_order_transformer;
    float *layer_output = dse_get_nasm_result (target_nasm, output_order_param);
    float *softmax_output = calloc (1000*target_nasm->batch_size, sizeof(float));
    naive_softmax (layer_output, softmax_output, target_nasm->batch_size, 1000);
    for (int i = 0; i < target_nasm->batch_size; i++)
    {
        get_probability_results ("data/imagenet_classes.txt", softmax_output + 1000*i, 1000);
    }
    free (softmax_output);
    save_arr (layer_output, "aspen_output.tmp", dse_get_nasm_result_size(target_nasm));
    free (layer_output);

    // Wrap up
    net_engine_destroy (net_engine);
    dse_group_destroy (dse_group);
    rpool_destroy (rpool);
    apu_destroy_nasm (target_nasm);
    apu_destroy_dnn (target_dnn);

    if (device_mode == DEV_SERVER) {
        close(client_sock);
        close(server_sock);
    }
    else if (device_mode == DEV_EDGE) {
        close(server_sock);
    }

    /** STAGE: FINISH **/

    return 0;
}
