#include <stdio.h>
#include "aspen.h"
#include "util.h"
#include "nasm.h"
#include "apu.h"
#include "networking.h"
#include "scheduling.h"
#include "profiling.h"

int main(int argc, char **argv)
{
    int sock_type = 999;

    if(argc > 1) 
    {
        if(!strcmp(argv[1], "RX")) {
            sock_type = SOCK_RX;
        } else if(!strcmp(argv[1], "TX")) {
            sock_type = SOCK_TX;
        }
    }
    else {
        printf("usage: %s [RX/TX]\n", argv[0]);
        exit(0);
    }

    // char *target_config = "data/cfg/resnet50_aspen.cfg";
    // char *target_bin = "data/resnet50/resnet50_data.bin";
    // char *target_nasm_dir = "data/resnet50_B1_aspen.nasm";
    // char *target_nasm_dir = "data/resnet50_B32_fine_aspen.nasm";
    // char* target_input = "data/resnet50/batched_input_64.bin";

    char *target_config = "data/cfg/vgg16_aspen.cfg";
    char *target_bin = "data/vgg16/vgg16_data.bin";
    char *target_nasm_dir = "data/vgg16_B1_aspen.nasm";
    char *target_input = NULL;

    int gpu = -1;

    // nasm_t *resnet50_nasm = apu_load_nasm_from_file ("data/resnet50_B1_aspen.nasm", resnet50_dnn);
    // nasm_t *resnet50_nasm = apu_load_nasm_from_file ("data/resnet50_B32_fine_aspen.nasm", resnet50_dnn);
    // nasm_t *resnet50_nasm = apu_create_nasm(resnet50_dnn, 1e6, 200, 32);
    // nasm_t *vgg16_nasm = apu_create_nasm(vgg16_dnn, 1e6, 8, 1);
    // apu_save_nasm_to_file(resnet50_nasm, "data/resnset50_B32_fine_aspen.nasm");
    // apu_save_nasm_to_file(vgg16_nasm, "data/vgg16_B1_aspen.nasm");

    ninst_profile_t *ninst_profile[SCHEDULE_MAX_DEVICES];
    network_profile_t *network_profile;

    /** STAGE: PROFILING COMPUTATION **/

    printf("STAGE: PROFILING COMPUTATION\n");
    // ninst_profile[sock_type] = profile_computation(target_config, target_bin, target_nasm_dir, target_input, gpu, 1);
    ninst_profile[sock_type] = load_computation_profile("./data/vgg16_B1_comp_profile.bin");
    // save_computation_profile(ninst_profile[sock_type], "data/vgg16_B1_comp_profile.bin");

    
    /** STAGE: PROFILING NETWORK **/

    printf("STAGE: PROFILING NETWORK\n");

    char *rx_ip = "192.168.1.176";
    int rx_port = 3786;

    network_profile = profile_network(ninst_profile, sock_type, rx_ip, rx_port+1);
    
    
    /** STAGE: SCHEDULING - HEFT **/

    printf("STAGE: SCHEUDLING - HEFT\n");

    init_heft(target_config, target_bin, target_nasm_dir, ninst_profile, network_profile, 2);

    /** STAGE: INFERENCE **/


    return 0;
}
