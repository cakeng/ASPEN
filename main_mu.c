#include <stdio.h>
#include "aspen.h"
#include "util.h"
#include "nasm.h"
#include "apu.h"
#include "networking.h"
#include "scheduling.h"

int total_transferred = 0;

int main(int argc, char **argv)
{
    int device_idx = 0;
    int sequential = 0;

    if (argc > 1) {
        if(strcmp(argv[1], "PIP")) sequential = 0;
        else if (strcmp(argv[1], "SEQ")) sequential = 1;
    }
    else if (argc > 2) {
        device_idx = atoi(argv[2]);
    }
    else {
        printf("Usage: %s [device_idx]\n", argv[0]);
    }

    // aspen_dnn_t *resnet50_dnn = apu_create_dnn("data/cfg/resnet50_aspen.cfg", "data/resnet50/resnet50_data.bin");
    // aspen_dnn_t *vgg16_dnn = apu_create_dnn("data/cfg/vgg16_aspen.cfg", "data/vgg16/vgg16_data.bin");
    int gpu = -1;

    // nasm_t *resnet50_nasm = apu_load_nasm_from_file ("data/resnet50_B1_aspen.nasm", resnet50_dnn);
    // nasm_t *resnet50_nasm = apu_load_nasm_from_file ("data/resnet50_B32_fine_aspen.nasm", resnet50_dnn);
    // nasm_t *vgg16_nasm = apu_load_nasm_from_file ("data/vgg16_B1_aspen.nasm", vgg16_dnn);
    // nasm_t *resnet50_nasm = apu_create_nasm(resnet50_dnn, 1e6, 200, 32);
    // nasm_t *vgg16_nasm = apu_create_nasm(vgg16_dnn, 1e6, 8, 1);
    // apu_save_nasm_to_file(resnet50_nasm, "data/resnset50_B32_fine_aspen.nasm");
    // apu_save_nasm_to_file(vgg16_nasm, "data/vgg16_B1_aspen.nasm");
    aspen_dnn_t *target_dnn[SCHEDULE_MAX_DEVICES];
    nasm_t *target_nasm[SCHEDULE_MAX_DEVICES];
    char* target_input = "data/resnet50/batched_input_64.bin";
    // char *target_input = NULL;

    if (device_idx == 0) {
        for (int i=1; i<SCHEDULE_MAX_DEVICES; i++) {
            target_dnn[i] = apu_create_dnn("data/cfg/resnet50_aspen.cfg", "data/resnet50/resnet50_data.bin");
            target_nasm[i] = apu_load_nasm_from_file ("data/resnet50_B1_aspen.nasm", target_dnn[i]);
            init_sequential_offload(target_nasm[i], 1, i, device_idx);
        }
    }
    else {
        target_dnn[device_idx] = apu_create_dnn("data/cfg/resnet50_aspen.cfg", "data/resnet50/resnet50_data.bin");
        target_nasm[device_idx] = apu_load_nasm_from_file ("data/resnet50_B1_aspen.nasm", target_dnn[device_idx]);
        init_sequential_offload(target_nasm[device_idx], 1, device_idx, 0);
    }

    rpool_t *rpool = rpool_init (gpu);
    dse_group_t *dse_group = dse_group_init (8, gpu);
    dse_group_set_rpool (dse_group, rpool);

    networking_engine* net_engine = NULL;
    networking_engine *net_engine_arr[SCHEDULE_MAX_DEVICES];

    char* rx_ip = "192.168.1.176";
    int rx_port_start = 3786;
    int rx_ports[SCHEDULE_MAX_DEVICES];
    for (int i=1; i<SCHEDULE_MAX_DEVICES; i++) {
        rx_ports[i] = rx_port_start + i;
    }

    if (device_idx == 0) {
        for (int i=1; i<SCHEDULE_MAX_DEVICES; i++) {
            net_engine_arr[i] = init_networking(target_nasm[i], rpool, SOCK_RX, rx_ip, rx_ports[i], 0, sequential);
            dse_group_add_netengine_arr(dse_group, net_engine_arr[i], i);
            dse_group_set_device(dse_group, device_idx);
            net_engine_arr[i]->dse_group = dse_group;
        
            atomic_store (&net_engine_arr[i]->run, 1);
        }
    }
    else {
        net_engine = init_networking(target_nasm[device_idx], rpool, SOCK_TX, rx_ip, rx_ports[device_idx], 0, sequential);
        dse_group_add_netengine_arr(dse_group, net_engine, 0);
        dse_group_set_device(dse_group, device_idx);
        net_engine->dse_group = dse_group;
        add_input_rpool (net_engine, target_nasm[device_idx], target_input);
        
        atomic_store (&net_engine->run, 1);
    }

    // SYNC HERE
    float sync_key;
    int control_server_sock;
    int client_sock_arr[SCHEDULE_MAX_DEVICES];

    if (device_idx == 0) {
        control_server_sock = create_server_sock(rx_ip, rx_port_start);
        for (int i=1; i<SCHEDULE_MAX_DEVICES; i++) {
            client_sock_arr[i] = accept_client_sock(control_server_sock);
        }
        for (int i=1; i<SCHEDULE_MAX_DEVICES; i++) {
            sync_key = get_time_secs();
            printf("SYNC KEY SEND %d: %f\n", i, sync_key);
            write_n(client_sock_arr[i], &sync_key, sizeof(float));
            read_n(client_sock_arr[i], &sync_key, sizeof(float));
            printf("SYNC KEY RECV %d: %f\n", i, sync_key);
            sync_key = get_time_secs();
            printf("SYNC KEY LAST %d: %f\n", i, sync_key);
            close(client_sock_arr[i]);
        }
        close(control_server_sock);
    }
    else {
        sleep(5 + device_idx);
        control_server_sock = connect_server_sock(rx_ip, rx_port_start);
        read_n(control_server_sock, &sync_key, sizeof(float));
        sync_key = get_time_secs();
        write_n(control_server_sock, &sync_key, sizeof(float));
        close(control_server_sock);
        printf("SYNC KEY: %d\n", sync_key);
    }
    
    get_elapsed_time ("init");
    if (!sequential || device_idx != SOCK_RX) dse_group_run (dse_group);
    if (device_idx == SOCK_RX) {
        for (int i=1; i<SCHEDULE_MAX_DEVICES; i++) {
            dse_wait_for_nasm_completion (target_nasm[i]);
        }
    }
    else {
        dse_wait_for_nasm_completion (target_nasm[device_idx]);
    }
    
    get_elapsed_time ("run_aspen");
    dse_group_stop (dse_group);
    
    if (device_idx > 0) {
        LAYER_PARAMS output_order[] = {BATCH, OUT_H, OUT_W, OUT_C};
        float *layer_output = dse_get_nasm_result (target_nasm[device_idx], output_order);
        float *softmax_output = calloc (1000*target_nasm[device_idx]->batch_size, sizeof(float));
        naive_softmax (layer_output, softmax_output, target_nasm[device_idx]->batch_size, 1000);
        for (int i = 0; i < target_nasm[device_idx]->batch_size; i++)
        {
            get_probability_results ("data/resnet50/imagenet_classes.txt", softmax_output + 1000*i, 1000);
        }

        free (layer_output);
        free (softmax_output);
    }
    

    // WRAP UP
    char file_name[256];
    if (device_idx == 0) {
        FILE *log_fp;
        
        for (int i=1; i<SCHEDULE_MAX_DEVICES; i++) {
            sprintf(file_name, "./logs/multiuser/pipeline_dev%d_RX.txt", i);
            log_fp = fopen(file_name, "w");

            close_connection(net_engine_arr[i]);
            save_ninst_log(log_fp, target_nasm[i]);
            net_engine_destroy (net_engine_arr[i]);
            apu_destroy_nasm (target_nasm[i]);
            apu_destroy_dnn (target_dnn[i]);
        }
    }
    else {
        sprintf(file_name, "./logs/multiuser/pipeline_dev%d_TX.txt", device_idx);
        FILE *log_fp = fopen(file_name, "w");

        close_connection (net_engine);
        save_ninst_log(log_fp, target_nasm[device_idx]);
        net_engine_destroy (net_engine);
        apu_destroy_nasm (target_nasm[device_idx]);
        apu_destroy_dnn (target_dnn[device_idx]);
    }
    dse_group_destroy (dse_group);
    rpool_destroy (rpool);

    printf("total transferred: %d\n", total_transferred);
    return 0;
}
