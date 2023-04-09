#include <stdio.h>
#include "aspen.h"
#include "util.h"
#include "nasm.h"
#include "apu.h"


int main(void)
{
    print_aspen_build_info();
    // aspen_dnn_t *resnet50_dnn = apu_create_dnn("data/cfg/resnet50_test.cfg", NULL);
    aspen_dnn_t *resnet50_dnn = apu_create_dnn("data/cfg/resnet50_test.cfg", "data/resnet50_data.bin");
    if (resnet50_dnn == NULL) 
    {
        printf("Error: Failed to create DNN\n");
        return -1;
    }
    // print_dnn_info(resnet50_dnn, 0);
    apu_save_dnn_to_file (resnet50_dnn, "data/resnet50.aspen");
    aspen_dnn_t *resnet50_dnn_2 = apu_load_dnn_from_file ("data/resnet50.aspen");
    if (resnet50_dnn_2 == NULL) 
    {
        printf("Error: Failed to read DNN\n");
        return -1;
    }
    // print_dnn_info (resnet50_dnn_2, 0);
    nasm_t *resnet50_nasm = apu_create_nasm(resnet50_dnn, 10e6, 4);
    if (resnet50_nasm == NULL) 
    {
        printf("Error: Failed to create NASM\n");
        return -1;
    }
    // print_nasm_info(resnet50_nasm, 0);
    char nasm_file_name [1024] = {0};
    sprintf (nasm_file_name, "data/resnet50_B%d_%2.1e.nasm", resnet50_nasm->batch_size,
        (double)resnet50_nasm->flop_per_ninst);
    apu_save_nasm_to_file (resnet50_nasm, nasm_file_name);

    // aspen_dnn_t *resnet50_dnn = NULL;
    // nasm_t *resnet50_nasm = apu_load_nasm_from_file ("data/resnet50_B64_1.0e+07.nasm", &resnet50_dnn);
    // nasm_t *resnet50_4_nasm = apu_load_nasm_from_file ("data/resnet50_4.nasm", &resnet50_dnn);
    
    rpool_t *rpool = rpool_init (-1);
    ase_group_t *ase_group = ase_group_init (64, -1);
    ase_group_set_rpool (ase_group, rpool);

    // rpool_add_nasm_raw_input (rpool, resnet50_4_nasm, 0.5, dog_data);
    rpool_add_nasm (rpool, resnet50_nasm, 1.0, "data/batched_input_64.bin");
    // print_rpool_info (rpool);
    // print_nasm_info(resnet50_nasm, 0);
    // print_dnn_info(resnet50_dnn, 0);

    get_elapsed_time ("init");
    ase_group_run (ase_group);
    ase_wait_for_nasm_completion (resnet50_nasm);
    // ase_wait_for_nasm_completion (resnet50_4_nasm);
    ase_group_stop (ase_group);
    get_elapsed_time ("run_aspen");
    
    // print_rpool_info (rpool);

    unsigned int input_params[NUM_PARAM_ELEMENTS] =
        {[BATCH] = resnet50_nasm->batch_size, [OUT_C] = 3, [OUT_H] = 224, [OUT_W] = 224};
    void *dog_data = aspen_load_input_NHWC ("data/batched_input_64.bin", input_params, sizeof(float));
    aspen_run_naive (resnet50_dnn, input_params[BATCH], dog_data);
    get_elapsed_time ("run_naive");
    
    
    for (int i = 70; i < 74; i++)
    {
        printf ("\tLayer %d - Type %s\n", i, layer_type_str[resnet50_dnn->layers[i].type]);
        aspen_layer_t *layer = &resnet50_dnn->layers[i];
        nasm_ldata_t *ldata = &resnet50_nasm->ldata_arr[i];
        assert (ldata->layer == layer);
        LAYER_PARAMS output_order[] = {BATCH, OUT_C, OUT_H, OUT_W};
        void *layer_output = get_aspen_tensor_data 
            (layer->tensors[OUTPUT_TENSOR], output_order);
        void *ldata_output = get_ldata_output (ldata, output_order);
        void *ldata_raw_output = get_packed_ldata_output_colwise (ldata);
        char filename[256];
        sprintf (filename, "resnet50_layer%d.bin", i);
        size_t elem_size = ldata->layer->dnn->element_size;
        size_t data_size = ldata->out_mat_dims[OUT_H] * ldata->out_mat_dims[OUT_W] * elem_size;
        void *expected_output = load_arr (filename, data_size);
        
        // printf ("Expected output for layer %d:\n", i);
        // print_float_tensor (layer_output, layer->params[BATCH], layer->params[OUT_C], 
        //     layer->params[OUT_H], layer->params[OUT_W]);
        // printf ("Computed output for layer %d:\n", i);
        // print_float_tensor (ldata_output, layer->params[BATCH], layer->params[OUT_C], 
        //     layer->params[OUT_H], layer->params[OUT_W]);
        // printf ("Raw output for layer %d:\n", i);
        // print_float_tensor (ldata_raw_output, layer->params[BATCH], layer->params[OUT_H], 
        //     layer->params[OUT_W], layer->params[OUT_C]);

        // if (layer->tensors[WEIGHT_TENSOR] != NULL)
        // {
        //     if (layer->tensors[WEIGHT_TENSOR]->dims[SUB_C] != 0)
        //         print_float_tensor (layer->tensors[WEIGHT_TENSOR]->data, layer->tensors[WEIGHT_TENSOR]->dims[OUT_C], layer->tensors[WEIGHT_TENSOR]->dims[WEIGHT_H], 
        //             layer->tensors[WEIGHT_TENSOR]->dims[WEIGHT_W], layer->tensors[WEIGHT_TENSOR]->dims[IN_C] * layer->tensors[WEIGHT_TENSOR]->dims[SUB_C]);
        //     else
        //         print_float_tensor (layer->tensors[WEIGHT_TENSOR]->data, layer->tensors[WEIGHT_TENSOR]->dims[OUT_C], layer->tensors[WEIGHT_TENSOR]->dims[WEIGHT_H],
        //             layer->tensors[WEIGHT_TENSOR]->dims[WEIGHT_W], layer->tensors[WEIGHT_TENSOR]->dims[IN_C]);
        // }
        // compare_float_tensor (expected_output, ldata_output, 
        //     resnet50_nasm->batch_size, layer->params[OUT_C], layer->params[OUT_H], layer->params[OUT_W],
        //     1e-2, 1e-4, 100);
        compare_float_tensor (expected_output, layer_output, 
            resnet50_nasm->batch_size, layer->params[OUT_C], layer->params[OUT_H], layer->params[OUT_W],
            1e-2, 1e-4, 100);
        
        free (expected_output);
        free (ldata_output);
        free (ldata_raw_output);
        free (layer_output);
    }

    LAYER_PARAMS output_order[] = {BATCH, OUT_C, OUT_H, OUT_W};
    // float *layer_output = ase_get_nasm_result (resnet50_nasm, output_order);
    float *layer_output = get_aspen_tensor_data ((resnet50_dnn->layers + resnet50_dnn->num_layers - 1)->tensors[OUTPUT_TENSOR], output_order);
    // print_float_array (layer_output, 1000*input_params[BATCH], 1000);
    for (int i = 0; i < resnet50_nasm->batch_size; i++)
    {
        get_probability_results ("data/imagenet_classes.txt", layer_output + 1000*i, 1000);
    }
    free (layer_output);

    ase_group_destroy (ase_group);
    rpool_destroy (rpool);
    // apu_destroy_nasm (resnet50_4_nasm);
    apu_destroy_nasm (resnet50_nasm);
    apu_destroy_dnn (resnet50_dnn);
    return 0;
}