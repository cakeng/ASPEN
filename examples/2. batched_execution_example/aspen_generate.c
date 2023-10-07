#include "../../include/aspen.h"

int main (int argc, char* argv[])
{
    //Take number of generation iteration from args
    int num_iter = 1;
    if (argc == 2)
    {
        num_iter = atoi(argv[1]);
    }
    else 
    {
        printf("Usage: ./aspen_generate\n");
        return 0;
    }

    print_aspen_build_info();

    // Parse DNN model specification of ResNet50.
    aspen_dnn_t *resnet50_dnn = apu_create_dnn("../../files/resnet50_aspen.cfg", "../../files/resnet50_weight.bin");
    // Save the model to a file
    apu_save_dnn_to_file (resnet50_dnn, "resnet50.aspen");
    // Parse DNN model specification of VGG-16.
    aspen_dnn_t *vgg16_dnn = apu_create_dnn("../../files/vgg16_aspen.cfg", "vgg16_weight.bin");
    // Save the model to a file
    apu_save_dnn_to_file (vgg16_dnn, "vgg16.aspen");

    // Generate NASM (ASPEN Graph) for the ResNet-50 with batch size of 4.
    // May take quite some time depending on your machine.
    // nasm_t *resnet50_B4_nasm = apu_generate_nasm (resnet50_dnn, 4, num_iter);

    // Generate NASM (ASPEN Graph) for the VGG-16 with batch size of 1.
    // May take quite some time depending on your machine.
    // nasm_t *vgg16_B1_nasm = apu_generate_nasm (vgg16_dnn, 1, num_iter);

    // // If apu_generate_nasm() takes too long, use apu_create_nasm() instead,
    // // to create a NASM using a fixed number of nodes per layer.
    // // Usually, 50 ~ 100 nodes per layer is a good number to start with.
    nasm_t *resnet50_B4_nasm = apu_create_nasm (resnet50_dnn, 100, 4);
    nasm_t *vgg16_B1_nasm = apu_create_nasm (vgg16_dnn, 100, 1);

    // Save the generated NASM to a file.
    apu_save_nasm_to_file (resnet50_B4_nasm, "resnet50_B4.nasm");
    apu_save_nasm_to_file (vgg16_B1_nasm, "vgg16_B1.nasm");

    // Cleanup
    apu_destroy_nasm (resnet50_B4_nasm);
    apu_destroy_dnn (resnet50_dnn);
    apu_destroy_nasm (vgg16_B1_nasm);
    apu_destroy_dnn (vgg16_dnn);
    return 0;
}