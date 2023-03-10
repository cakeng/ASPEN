#include <stdio.h>
#include "aspen.h"

int main(void)
{
    printf("Hello World!\n");
    print_build_info();

    aspen_dnn_t *vgg16 = apu_create_dnn("data/cfg/vgg-16.cfg", NULL);
    if (vgg16 == NULL) 
    {
        printf("Error: Failed to create DNN\n");
        return -1;
    }
    aspen_destroy_dnn(vgg16);
    return 0;
}