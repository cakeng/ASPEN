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

double get_sec()
{
    struct timeval now;
    gettimeofday (&now, NULL);
    return now.tv_sec + now.tv_usec*1e-6;
}

// OPTIONS
// option "sock_type" - "" int required
// option "sequential" - "" int required
// option "dirname" - "" string required
// option "prefix" - "" string optional
// option "postfix" - "" string optional
// option "log_idx_start" - "" int optional
// option "inference_repeat_num" - "" int optional
// option "target_dnn_dir" - "" string required
// option "target_nasm_dir" - "" string required
// option "target_input" - "" string optional
// option "rx_ip" - "" string required
// option "rx_port" - "" int required
// option "schedule_policy" - "" string required values="partial","sequential"
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

    int sock_type = ai.sock_type_arg;
    int sequential = !ai.pipelined_arg;
    char *dirname = ai.dirname_arg;
    char *prefix = ai.prefix_arg ? ai.prefix_arg : "temp";
    char *postfix = ai.postfix_arg ? ai.postfix_arg : "0";
    int log_idx_start = ai.log_idx_start_arg;
    int inference_repeat_num = ai.inference_repeat_num_arg;
    char *target_dnn_dir = ai.target_dnn_dir_arg;
    char *target_nasm_dir = ai.target_nasm_dir_arg;
    char *target_input = ai.target_input_arg;
    char *rx_ip = ai.rx_ip_arg;
    int rx_port = ai.rx_port_arg;
    char *schedule_policy = ai.schedule_policy_arg;
    float sched_partial_ratio = ai.sched_partial_ratio_arg;
    int sched_sequential_idx = ai.sched_sequential_idx_arg;
    int dse_num = ai.dse_num_arg;
    char *output_order = ai.output_order_arg;

    int dnn_dir_exist = !target_dnn_dir && strlen(target_dnn_dir) > 0;

    // char *target_config = "data/cfg/resnet50_aspen.cfg";
    // char *target_bin = "data/resnet50/resnet50_data.bin";
    // char *target_nasm_dir = "data/resnet50_B1_aspen.nasm";
    // char *target_nasm_dir = "data/resnet50_B32_fine_aspen.nasm";
    // char* target_input = "data/resnet50/batched_input_64.bin";
    // char *target_config = "data/cfg/bert_base_encoder.cfg";
    // char *target_bin = NULL;

    // char *target_config = "data/cfg/vgg16_aspen.cfg";
    // char *target_bin = "data/vgg16/vgg16_data.bin";
    // char *target_nasm_dir = "data/vgg16_B1_aspen.nasm";
    // char *target_nasm_dir = NULL;
    // char *target_input = NULL;

    int gpu = -1;

    // nasm_t *resnet50_nasm = apu_load_nasm_from_file ("data/resnet50_B1_aspen.nasm", resnet50_dnn);
    // nasm_t *resnet50_nasm = apu_load_nasm_from_file ("data/resnet50_B32_fine_aspen.nasm", resnet50_dnn);
    // nasm_t *resnet50_nasm = apu_create_nasm(resnet50_dnn, 1e6, 200, 32);
    // nasm_t *vgg16_nasm = apu_create_nasm(vgg16_dnn, 1e6, 8, 1);
    // apu_save_nasm_to_file(resnet50_nasm, "data/resnset50_B32_fine_aspen.nasm");
    // apu_save_nasm_to_file(vgg16_nasm, "data/vgg16_B1_aspen.nasm");

    int server_sock;
    int client_sock;

    if (!strcmp(schedule_policy, "local"))
    {
        sock_type = SOCK_LOCAL;
    }
    if (sock_type == SOCK_RX) {
        server_sock = create_server_sock(rx_ip, rx_port+1);
        client_sock = accept_client_sock(server_sock);
    }
    else if (sock_type == SOCK_TX) {
        server_sock = connect_server_sock(rx_ip, rx_port+1);
    }
    else
    {
        schedule_policy = "local";
    }
    

    ninst_profile_t *ninst_profile[SCHEDULE_MAX_DEVICES];
    network_profile_t *network_profile;

    sched_processor_t *schedule;

    aspen_dnn_t *target_dnn;
    nasm_t *target_nasm;

    rpool_t *rpool = rpool_init (gpu);
    dse_group_t *dse_group = dse_group_init (dse_num, gpu);
    dse_group_set_rpool (dse_group, rpool);
    networking_engine* net_engine = NULL;

    if (!strcmp(schedule_policy, "heft")) {
        /** STAGE: PROFILING COMPUTATION **/

        printf("STAGE: PROFILING COMPUTATION\n");
        ninst_profile[sock_type] = profile_computation(target_dnn_dir, target_nasm_dir, target_input, gpu, 1);
        ninst_profile[sock_type] = load_computation_profile("./data/vgg16_B1_comp_profile.bin");
        save_computation_profile(ninst_profile[sock_type], "data/bert_base_comp_profile.bin");

        
        /** STAGE: PROFILING NETWORK **/

        printf("STAGE: PROFILING NETWORK\n");

        network_profile = profile_network(ninst_profile, sock_type, server_sock, client_sock);

        int connection_key;
        if (sock_type == SOCK_RX) {
            connection_key = 12534;
            write_n(client_sock, &connection_key, sizeof(int));
            printf("connection key: %d\n", connection_key);
        }
        else if (sock_type == SOCK_TX) {
            connection_key = -1;
            read_n(server_sock, &connection_key, sizeof(int));
            printf("connection key: %d\n", connection_key);
        }

        printf("sync: %f\n", network_profile->sync);
        
        
        /** STAGE: SCHEDULING - HEFT **/
        printf("STAGE: SCHEUDLING - HEFT\n");

        if (sock_type == SOCK_RX) {
            schedule = init_heft(target_dnn_dir, target_nasm_dir, ninst_profile, network_profile, 2);
            save_schedule(schedule, 2, "./temp_sched.txt");
        }
        
        share_schedule(&schedule, 2, sock_type, server_sock, client_sock);

        target_dnn = apu_load_dnn_from_file(target_dnn_dir);
        target_nasm = apu_load_nasm_from_file (target_nasm_dir, target_dnn);

        apply_schedule_to_nasm(target_nasm, schedule, 2, sock_type);
    }
    else if (!strcmp(schedule_policy, "partial")) {
        /** STAGE: PROFILING NETWORK **/

        printf("STAGE: PROFILING NETWORK\n");

        float sync = profile_network_sync(sock_type, server_sock, client_sock);

        int connection_key;
        if (sock_type == SOCK_RX) {
            connection_key = 12534;
            write_n(client_sock, &connection_key, sizeof(int));
            printf("connection key: %d\n", connection_key);
        }
        else if (sock_type == SOCK_TX) {
            connection_key = -1;
            read_n(server_sock, &connection_key, sizeof(int));
            printf("connection key: %d\n", connection_key);
        }

        printf("sync: %f\n", sync);

        /** STAGE: SCHEDULING - PARTIAL **/
        target_dnn = apu_load_dnn_from_file(target_dnn_dir);
        // target_nasm = apu_create_nasm(target_dnn, 1e4, 8, 32);
        // apu_save_nasm_to_file(target_nasm, "data/bert_base_encoder_B32_S128.nasm");
        target_nasm = apu_load_nasm_from_file(target_nasm_dir, target_dnn);

        init_partial_offload(target_nasm, 0.0);
    }
    else if (!strcmp(schedule_policy, "sequential")) {
        /** STAGE: PROFILING NETWORK **/

        printf("STAGE: PROFILING NETWORK\n");

        float sync = profile_network_sync(sock_type, server_sock, client_sock);

        int connection_key;
        if (sock_type == SOCK_RX) {
            connection_key = 12534;
            write_n(client_sock, &connection_key, sizeof(int));
            printf("connection key: %d\n", connection_key);
        }
        else if (sock_type == SOCK_TX) {
            connection_key = -1;
            read_n(server_sock, &connection_key, sizeof(int));
            printf("connection key: %d\n", connection_key);
        }

        printf("sync: %f\n", sync);

        /** STAGE: SCHEDULING - SEQUENTIAL **/
        target_dnn = apu_load_dnn_from_file(target_dnn_dir);
        target_nasm = apu_load_nasm_from_file(target_nasm_dir, target_dnn);

        init_sequential_offload(target_nasm, sched_sequential_idx, SOCK_TX, SOCK_RX);
    }
    else if (!strcmp(schedule_policy, "dynamic")) {
        /** STAGE: PROFILING NETWORK **/

        printf("STAGE: PROFILING NETWORK\n");

        float sync = profile_network_sync(sock_type, server_sock, client_sock);

        int connection_key;
        if (sock_type == SOCK_RX) {
            connection_key = 12534;
            write_n(client_sock, &connection_key, sizeof(int));
            printf("connection key: %d\n", connection_key);
        }
        else if (sock_type == SOCK_TX) {
            connection_key = -1;
            read_n(server_sock, &connection_key, sizeof(int));
            printf("connection key: %d\n", connection_key);
        }

        printf("sync: %f\n", sync);

        /** STAGE: SCHEDULING - DYNAMIC **/
        target_dnn = apu_load_dnn_from_file(target_dnn_dir);
        target_nasm = apu_load_nasm_from_file(target_nasm_dir, target_dnn);
        for (int i=0; i<dse_group->num_ases; i++) {
            dse_group->dse_arr[i].is_dynamic_scheduling = 1;
        }

        init_dynamic_offload(target_nasm);
    }
    else if (!strcmp(schedule_policy, "local")) {
        target_dnn = apu_load_dnn_from_file(target_dnn_dir);
        target_nasm = apu_load_nasm_from_file(target_nasm_dir, target_dnn);
        dse_group_set_device (dse_group, 0);
        init_sequential_offload (target_nasm, 0, 0, 0);
    }
    
    /** STAGE: INFERENCE **/

    printf("STAGE: INFERENCE\n");
    
    if (!strcmp(schedule_policy, "local"))
    {
        rpool_add_nasm (rpool, target_nasm, target_input);

        double start_time = get_sec();
        for (int i = 0; i < inference_repeat_num; i++)
        {
            rpool_reset (rpool);
            rpool_reset_nasm (rpool, target_nasm);
            dse_group_run (dse_group);
            dse_wait_for_nasm_completion (target_nasm);
            dse_group_stop (dse_group);
        }

        double end_time = get_sec();
        printf ("Time taken: %lf seconds\n", (end_time - start_time)/inference_repeat_num);
    }
    else
    {
        
        net_engine = init_networking(target_nasm, rpool, sock_type, rx_ip, rx_port, 0, sequential);
        dse_group_set_net_engine(dse_group, net_engine);
        dse_group_set_device(dse_group, sock_type);
        net_engine->dse_group = dse_group;

        for (int i=0; i<inference_repeat_num; i++) {

            // synchronize
            remove_inference_whitelist(net_engine, target_nasm->inference_id);
            printf("Sync between inference...\n");

            float sync = profile_network_sync(sock_type, server_sock, client_sock);

            int connection_key;
            if (sock_type == SOCK_RX) {
                connection_key = 12534+i;
                write_n(client_sock, &connection_key, sizeof(int));
                printf("connection key: %d\n", connection_key);
            }
            else if (sock_type == SOCK_TX) {
                connection_key = -1;
                read_n(server_sock, &connection_key, sizeof(int));
                printf("connection key: %d\n", connection_key);
            }

            printf("sync: %f\n", sync);

            rpool_reset(rpool);
            apu_reset_nasm(target_nasm);
            
            if (!strcmp(schedule_policy, "dynamic")) init_dynamic_offload(target_nasm);

            
            set_nasm_inference_id(target_nasm, connection_key);
            add_inference_whitelist(net_engine, target_nasm->inference_id);

            printf("inference: %d/%d\n", i+1, inference_repeat_num);
            printf("inference id: %d\n", target_nasm->inference_id);

            target_nasm->nasm_cond = (pthread_cond_t)PTHREAD_COND_INITIALIZER;

            if(sock_type == SOCK_TX) {
                add_input_rpool (net_engine, target_nasm, target_input);
            }

            atomic_store (&net_engine->run, 1);
            printf("netqueue remaining: %d\n", net_engine->net_queue->num_stored);
            while (net_engine->net_queue->num_stored) {}
            
            
            get_elapsed_time ("init");
            if (!sequential || sock_type == SOCK_TX) dse_group_run (dse_group);
            if (!(!strcmp(schedule_policy, "local") && sock_type == SOCK_RX)) dse_wait_for_nasm_completion (target_nasm);
            get_elapsed_time ("run_aspen");
            dse_group_stop (dse_group);
            
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
            char file_name[256];
            char dir_path[256];
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

            sprintf(file_name, "./logs/%s/%s_%s_%s_%s_%s_%d.csv", dirname, prefix, sequential ? "seq" : "pip", schedule_policy, sock_type == SOCK_RX ? "RX" : "TX", postfix, log_idx_start+i);
            
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
    close_connection (net_engine);
    net_engine_destroy (net_engine);
    dse_group_destroy (dse_group);
    rpool_destroy (rpool);
    apu_destroy_nasm (target_nasm);
    apu_destroy_dnn (target_dnn);

    if (sock_type == SOCK_RX) {
        close(client_sock);
        close(server_sock);
    }
    else if (sock_type == SOCK_TX) {
        close(server_sock);
    }

    /** STAGE: FINISH **/

    return 0;
}
