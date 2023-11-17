#ifndef _PROFILING_H_
#define _PROFILING_H_

#include <stdio.h>
#include "util.h"
#include "aspen.h"

struct avg_ninst_profile_t {
    int num_ninsts;
    float avg_server_computation_time;
    float avg_edge_computation_time;
    int edge_num_dse;
    int server_num_dse;
};

struct ninst_profile_t {
    int idx;
    int total;
    int transmit_size;
    float computation_time;
};

struct network_profile_t {
    float sync;     // add to tx_timestamp then becomes rx_timestamp
    float rtt;
    float transmit_rate;
};

void profile_comp_and_net(nasm_t *target_nasm, int dse_num, DEVICE_MODE device_mode, int server_sock, int client_sock, float *server_elapsed_times, float *edge_elapsed_times, network_profile_t **network_profile);
avg_ninst_profile_t *profile_computation(nasm_t *target_nasm, int dse_num, int device_idx, char *target_input, DEVICE_MODE device_mode, int gpu, int num_repeat);
network_profile_t *profile_network(DEVICE_MODE device_mode, int edge_device_idx, int server_sock, int client_sock);
float profile_network_sync(DEVICE_MODE device_mode, int server_sock, int client_sock);
void print_network_profile(network_profile_t *network_profile);

void communicate_profiles_server(int client_sock, network_profile_t *network_profile, avg_ninst_profile_t *ninst_profile);
void communicate_profiles_edge(int server_sock, network_profile_t *network_profile, avg_ninst_profile_t *ninst_profile);

// ninst_profile_t *extract_profile_from_ninsts(nasm_t *nasm);
float extract_profile_from_ninsts(nasm_t *nasm);
ninst_profile_t *merge_computation_profile(ninst_profile_t **ninst_profiles, int num_ninst_profiles);
void save_computation_profile(ninst_profile_t *profile, char *file_path);
ninst_profile_t *load_computation_profile(char *file_path);

#endif