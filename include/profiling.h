#ifndef _PROFILING_H_
#define _PROFILING_H_

#include <stdio.h>
#include "util.h"
#include "aspen.h"

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

ninst_profile_t *profile_computation(char *target_config, char *target_bin, char *target_nasm_dir, char *target_input, int gpu, int num_repeat);
network_profile_t *profile_network(ninst_profile_t **ninst_profile, int sock_type, int server_sock, int client_sock);

ninst_profile_t *extract_profile_from_ninsts(nasm_t *nasm);
ninst_profile_t *merge_computation_profile(ninst_profile_t **ninst_profiles, int num_ninst_profiles);
void save_computation_profile(ninst_profile_t *profile, char *file_path);
ninst_profile_t *load_computation_profile(char *file_path);

#endif