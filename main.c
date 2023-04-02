#include <stdio.h>
#include "aspen.h"

int main(void)
{
    print_build_info();
    // aspen_dnn_t *resnet50_dnn = apu_create_dnn("data/cfg/resnet50.cfg", NULL);
    // if (resnet50_dnn == NULL) 
    // {
    //     printf("Error: Failed to create DNN\n");
    //     return -1;
    // }
    // print_dnn_info(resnet50_dnn, 0);
    // apu_save_dnn_to_file (resnet50_dnn, "data/resnet50.aspen");
    // aspen_dnn_t *resnet50_dnn_2 = apu_load_dnn_from_file ("data/resnet50.aspen");
    // if (resnet50_dnn_2 == NULL) 
    // {
    //     printf("Error: Failed to read DNN\n");
    //     return -1;
    // }
    // print_dnn_info (resnet50_dnn_2, 0);
    // nasm_t *resnet50_nasm = apu_create_nasm(resnet50_dnn, 5e6, 1);
    // if (resnet50_nasm == NULL) 
    // {
    //     printf("Error: Failed to create NASM\n");
    //     return -1;
    // }
    // print_nasm_info(resnet50_nasm, 0);
    // apu_save_nasm_to_file (resnet50_nasm, "data/resnet50.nasm");

    aspen_dnn_t *resnet50_dnn = NULL;
    nasm_t *resnet50_nasm = apu_load_nasm_from_file ("data/resnet50.nasm", &resnet50_dnn);
    if (resnet50_nasm == NULL) 
    {
        printf("Error: Failed to read NASM\n");
        apu_destroy_nasm (resnet50_nasm);
        apu_destroy_dnn (resnet50_dnn);
        return -1;
    }
    

    rpool_t *rpool = rpool_init (0);
    rpool_add_nasm (rpool, resnet50_nasm, 0.5);
    ase_group_t *ase_group = ase_group_init (4, 0);
    ase_group_set_rpool (ase_group, rpool);
    
    print_nasm_info(resnet50_nasm, 0);
    print_rpool_info (rpool);

    ase_group_destroy (ase_group);
    rpool_destroy (rpool);
    apu_destroy_nasm (resnet50_nasm);
    apu_destroy_dnn (resnet50_dnn);
    return 0;
}