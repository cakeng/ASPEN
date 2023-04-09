#include <stdio.h>
#include "aspen.h"
#include "util.h"
#include "nasm.h"
#include "apu.h"


int main(void)
{
    print_aspen_build_info();
    // aspen_dnn_t *resnet50_dnn = apu_create_dnn("data/cfg/resnet50_test.cfg", NULL);
    // if (resnet50_dnn == NULL) 
    // {
    //     printf("Error: Failed to create DNN\n");
    //     return -1;
    // }
    // // print_dnn_info(resnet50_dnn, 0);
    // apu_save_dnn_to_file (resnet50_dnn, "data/resnet50.aspen");
    // aspen_dnn_t *resnet50_dnn_2 = apu_load_dnn_from_file ("data/resnet50.aspen");
    // if (resnet50_dnn_2 == NULL) 
    // {
    //     printf("Error: Failed to read DNN\n");
    //     return -1;
    // }
    // // print_dnn_info (resnet50_dnn_2, 0);
    // nasm_t *resnet50_nasm = apu_create_nasm(resnet50_dnn, 5e5, 4);
    // if (resnet50_nasm == NULL) 
    // {
    //     printf("Error: Failed to create NASM\n");
    //     return -1;
    // }
    // // print_nasm_info(resnet50_nasm, 0);
    // apu_save_nasm_to_file (resnet50_nasm, "data/resnet50.nasm");

    aspen_dnn_t *resnet50_dnn = NULL;
    nasm_t *resnet50_nasm = apu_load_nasm_from_file ("data/resnet50.nasm", &resnet50_dnn);
    // nasm_t *resnet50_4_nasm = apu_load_nasm_from_file ("data/resnet50_4.nasm", &resnet50_dnn);

    apu_load_dnn_data_from_file (resnet50_dnn, "data/resnet50_data.bin");
    unsigned int input_params[NUM_PARAM_ELEMENTS] =
        {[BATCH] = 4, [OUT_C] = 3, [OUT_H] = 224, [OUT_W] = 224};
    void *dog_data = aspen_load_input_from_file ("data/batched_input_64.bin", input_params, sizeof(float));
    rpool_t *rpool = rpool_init (-1);
    ase_group_t *ase_group = ase_group_init (64, -1);
    ase_group_set_rpool (ase_group, rpool);

    // rpool_add_nasm (rpool, resnet50_4_nasm, 0.5, dog_data);
    rpool_add_nasm (rpool, resnet50_nasm, 1.0, dog_data);
    // print_rpool_info (rpool);
    // print_nasm_info(resnet50_nasm, 0);
    // print_dnn_info(resnet50_dnn, 0);

    get_elapsed_time ("init");
    aspen_run_naive (resnet50_dnn, input_params[BATCH], dog_data);
    get_elapsed_time ("run_naive");

    ase_group_run (ase_group);
    ase_wait_for_nasm_completion (resnet50_nasm);
    // ase_wait_for_nasm_completion (resnet50_4_nasm);
    ase_group_stop (ase_group);
    get_elapsed_time ("run_aspen");
    
    // print_rpool_info (rpool);
    
    
    // for (int i = 70; i < resnet50_dnn->num_layers; i++)
    // {
    //     printf ("\tLayer %d - Type %s\n", i, layer_type_str[resnet50_dnn->layers[i].type]);
    //     aspen_layer_t *layer = &resnet50_dnn->layers[i];
    //     nasm_ldata_t *ldata = &resnet50_nasm->ldata_arr[i];
    //     assert (ldata->layer == layer);
    //     LAYER_PARAMS output_order[] = {BATCH, OUT_C, OUT_H, OUT_W};
    //     // void *layer_output = get_aspen_tensor_data 
    //     //     (layer->tensors[OUTPUT_TENSOR], output_order);
    //     void *ldata_output = get_ldata_output (ldata, output_order);
    //     void *ldata_raw_output = get_packed_ldata_output_colwise (ldata);
    //     char filename[256];
    //     sprintf (filename, "resnet50_layer%d.bin", i);
    //     size_t data_size = 0;
    //     if (layer->tensors[OUTPUT_TENSOR] != NULL)
    //     {
    //         size_t data_size = layer->tensors[OUTPUT_TENSOR]->num_elements * sizeof(float);
    //     }
    //     else
    //     {
    //         size_t elem_size = ldata->layer->dnn->element_size;
    //         data_size = ldata->out_mat_dims[OUT_H] * ldata->out_mat_dims[OUT_W] * elem_size;
    //     }
    //     void *expected_output = load_arr (filename, data_size);
        
    //     // printf ("Expected output for layer %d:\n", i);
    //     // print_float_tensor (layer_output, layer->params[BATCH], layer->params[OUT_C], 
    //     //     layer->params[OUT_H], layer->params[OUT_W]);
    //     // printf ("Computed output for layer %d:\n", i);
    //     // print_float_tensor (ldata_output, layer->params[BATCH], layer->params[OUT_C], 
    //     //     layer->params[OUT_H], layer->params[OUT_W]);
    //     // printf ("Raw output for layer %d:\n", i);
    //     // print_float_tensor (ldata_raw_output, layer->params[BATCH], layer->params[OUT_H], 
    //     //     layer->params[OUT_W], layer->params[OUT_C]);

    //     compare_float_tensor (expected_output, ldata_output, 
    //         layer->params[BATCH], layer->params[OUT_C], layer->params[OUT_H], layer->params[OUT_W],
    //         1e-2, 1e-4, 100);
    //     // compare_float_tensor (expected_output, layer_output, 
    //     //     layer->params[BATCH], layer->params[OUT_C], layer->params[OUT_H], layer->params[OUT_W],
    //     //     layer->tensors[OUTPUT_TENSOR]->num_elements, 1e-2, 1e-4, 100);
        
    //     free (expected_output);
    //     free (ldata_output);
    //     free (ldata_raw_output);
    //     // free (layer_output);
    // }

    LAYER_PARAMS output_order[] = {BATCH, OUT_C};
    float *layer_output = ase_get_nasm_result (resnet50_nasm, output_order);
    // print_float_array (layer_output, 1000*input_params[BATCH], 1000);
    for (int i = 0; i < input_params[BATCH]; i++)
    {
        get_probability_results ("data/imagenet_classes.txt", layer_output + 1000*i, 1000);
    }
    free (layer_output);

    // ase_group_destroy (ase_group);
    // rpool_destroy (rpool);
    // apu_destroy_nasm (resnet50_4_nasm);
    apu_destroy_nasm (resnet50_nasm);
    apu_destroy_dnn (resnet50_dnn);
    return 0;
}