#include <stdio.h>
#include "aspen.h"

int main(void)
{
    // print_build_info();
    // aspen_dnn_t *vgg16 = apu_create_dnn("data/cfg/vgg-16.cfg", NULL);
    // print_dnn_info(vgg16, 0);
    // if (vgg16 == NULL) 
    // {
    //     printf("Error: Failed to create DNN\n");
    //     return -1;
    // }
    // apu_save_dnn_to_file (vgg16, "data/vgg16.aspen");
    aspen_dnn_t *vgg16_2 = apu_load_dnn_from_file ("data/vgg16.aspen");
    if (vgg16_2 == NULL) 
    {
        printf("Error: Failed to read DNN\n");
        return -1;
    }
    print_dnn_info (vgg16_2, 0);
    nasm_t *nasm = apu_create_nasm(vgg16_2, 40e6, 1);
    if (nasm == NULL) 
    {
        printf("Error: Failed to create NASM\n");
        return -1;
    }
    print_nasm_info(nasm, 0);
    aspen_destroy_nasm (nasm);
    aspen_destroy_dnn (vgg16_2);
    return 0;
}