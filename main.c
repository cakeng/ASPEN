#include <stdio.h>
#include "aspen.h"

int main(void)
{
    print_build_info();
    // aspen_dnn_t *vgg16_dnn = apu_create_dnn("data/cfg/vgg-16.cfg", NULL);
    // if (vgg16_dnn == NULL) 
    // {
    //     printf("Error: Failed to create DNN\n");
    //     return -1;
    // }
    // print_dnn_info(vgg16_dnn, 0);
    // apu_save_dnn_to_file (vgg16_dnn, "data/vgg16.aspen");
    // aspen_dnn_t *vgg16_dnn_2 = apu_load_dnn_from_file ("data/vgg16.aspen");
    // if (vgg16_dnn_2 == NULL) 
    // {
    //     printf("Error: Failed to read DNN\n");
    //     return -1;
    // }
    // print_dnn_info (vgg16_dnn_2, 0);
    // nasm_t *vgg16_nasm = apu_create_nasm(vgg16_dnn_2, 40e6, 1);
    // if (vgg16_nasm == NULL) 
    // {
    //     printf("Error: Failed to create NASM\n");
    //     return -1;
    // }
    // print_nasm_info(vgg16_nasm, 0);
    // apu_save_nasm_to_file (vgg16_nasm, "data/vgg16.nasm");

    aspen_dnn_t *vgg16_dnn;
    nasm_t *vgg16_nasm = apu_load_nasm_from_file ("data/vgg16.nasm", &vgg16_dnn);
    if (vgg16_nasm == NULL) 
    {
        printf("Error: Failed to read NASM\n");
        return -1;
    }
    print_nasm_info(vgg16_nasm, 0);

    rpool_t *rpool = rpool_init ();


    
    apu_destroy_nasm (vgg16_nasm);
    apu_destroy_dnn (vgg16_dnn);
    return 0;
}