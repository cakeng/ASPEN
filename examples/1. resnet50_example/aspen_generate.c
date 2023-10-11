#include "../../include/aspen.h"

int main (int argc, char* argv[])
{
    //Take number of generation iteration from args
    int batch_size = 1;
    int num_iter = 1;
    if (argc == 2)
    {
        num_iter = atoi(argv[1]);
    }
    else 
    {
        printf("Usage: ./aspen_generate <num_iter>\n");
        return 0;
    }

    print_aspen_build_info();

    // Parse DNN model specification from .cfg file and model weights from .bin file
    aspen_dnn_t *aspen_dnn = apu_create_dnn("../../files/resnet50_aspen.cfg", "../../files/resnet50_weight.bin");

    // // Print the DNN model specifications. 
    // // Second argument is the verbosity level for model data (weights, bias, etc.)
    // print_dnn_info (aspen_dnn, 0);

    // Save the model to a file
    apu_save_dnn_to_file (aspen_dnn, "resnet50.aspen");

    // Generate NASM (ASPEN Graph) for the DNN model. Searches for the best NASM,
    // by iterating through different number of nodes per layer. 
    // Takes batch size and number of iterations as arguments.
    // May take quite some time depending on your machine.
    nasm_t *aspen_nasm = apu_generate_nasm (aspen_dnn, batch_size, num_iter);

    // // If apu_generate_nasm() takes too long, use apu_create_nasm() instead,
    // // to create a NASM using a fixed number of nodes per layer.
    // // Usually, 50 ~ 100 nodes per layer is a good number to start with.
    // nasm_t *aspen_nasm = apu_create_nasm (aspen_dnn, 100, batch_size); 
    
    // // Print the NASM (ASPEN Graph) for the DNN model.
    // // Second argument is the verbosity level for Ninst (ASPEN node).
    // // Third argument is the verbosity level for data (outputs, etc.)
    // print_nasm_info (aspen_nasm, 0, 0);

    // Save the generated NASM to a file.
    char nasm_file_name [1024] = {0};
    sprintf (nasm_file_name, "resnet50_B%d.nasm", batch_size);
    apu_save_nasm_to_file (aspen_nasm, nasm_file_name);

    // Cleanup
    apu_destroy_nasm (aspen_nasm);
    apu_destroy_dnn (aspen_dnn);
    return 0;
}