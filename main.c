#include <stdio.h>
#include "aspen.h"
#include "util.h"
#include "nasm.h"
#include "apu.h"


int main(void)
{
    print_aspen_build_info();
    // aspen_dnn_t *resnet50_dnn = apu_create_dnn("data/cfg/resnet50_test.cfg", NULL);
    // aspen_dnn_t *resnet50_dnn = apu_create_dnn("data/cfg/resnet50_test.cfg", "data/resnet50_data.bin");
    // // aspen_dnn_t *resnet50_dnn = apu_create_transformer_encoder_dnn (12, 768, 12, 4, "resnet50_base", "data/resnet50_base_data.bin");
    // if (resnet50_dnn == NULL) 
    // {
    //     printf("Error: Failed to create DNN\n");
    //     return -1;
    // }
    // apu_save_dnn_to_file (resnet50_dnn, "data/resnet50_base.aspen");
    // // aspen_dnn_t *resnet50_dnn_2 = apu_load_dnn_from_file ("data/resnet50_base.aspen");
    // // if (resnet50_dnn_2 == NULL) 
    // // {
    // //     printf("Error: Failed to read DNN\n");
    // //     return -1;
    // // }
    // // print_dnn_info (resnet50_dnn_2, 0);
    
    // // nasm_t *resnet50_nasm = apu_create_transformer_encoder_nasm(resnet50_dnn, 10e6, 100, 1, 480);
    // nasm_t *resnet50_nasm = apu_create_nasm(resnet50_dnn, 1e6, 120, 1);
    // if (resnet50_nasm == NULL) 
    // {
    //     printf("Error: Failed to create NASM\n");
    //     return -1;
    // }
    // // print_nasm_info(resnet50_nasm, 1, 0);
    // char nasm_file_name [1024] = {0};
    // sprintf (nasm_file_name, "data/resnet50_B%d_M%d_%2.1e.nasm"
    //     , resnet50_nasm->batch_size, resnet50_nasm->min_ninst_per_ldata,
    //     (double)resnet50_nasm->flop_per_ninst);
    // apu_save_nasm_to_file (resnet50_nasm, nasm_file_name);
    int gpu = 0;
    aspen_dnn_t *resnet50_dnn = apu_load_dnn_from_file ("data/resnet50_base.aspen");
    nasm_t *resnet50_nasm = apu_load_nasm_from_file ("data/resnet50_B1_M100_1.0e+07.nasm", resnet50_dnn);
    // // // nasm_t *resnet50_4_nasm = apu_load_nasm_from_file ("data/resnet50_4.nasm", &resnet50_dnn);
    // 
    rpool_t *rpool = rpool_init (-1);
    ase_group_t *ase_group = ase_group_init (64, -1);
    ase_group_set_rpool (ase_group, rpool);

    // // rpool_add_nasm_raw_input (rpool, resnet50_4_nasm, 0.5, dog_data);
    rpool_add_nasm (rpool, resnet50_nasm, 1.0, "data/batched_input_64.bin");
    // rpool_add_nasm (rpool, resnet50_nasm, 1.0, "data/Text_Len_480_Embedded_input_Batch_32.bin");
    // print_rpool_info (rpool);
    // print_nasm_info(resnet50_nasm, 0, 0);

    get_elapsed_time ("init");

    // ase_cudagraph_run (rpool, resnet50_nasm);
    
    ase_group_run (ase_group);
    ase_wait_for_nasm_completion (resnet50_nasm);
    // ase_wait_for_nasm_completion (resnet50_4_nasm);
    ase_group_stop (ase_group);


    get_elapsed_time ("run_aspen");
    // rpool_reset_nasm (rpool, resnet50_nasm, 1.0);
    // print_nasm_info(resnet50_nasm, 1, 0);
    // // print_rpool_info (rpool);
    // print_nasm_cudagraph_info (resnet50_nasm, "cudagraph_out.txt");

    unsigned int input_params[NUM_PARAM_ELEMENTS] = {0};
    // input_params[BATCH] = 1; input_params[NUM_SEQ] = 480; input_params[NUM_HIDDEN] = 768;
    input_params[BATCH] = 1; input_params[OUT_C] = 3; input_params[OUT_H] = 224; input_params[OUT_W] = 224;
    // void *dog_data = aspen_load_input ("data/Text_Len_480_Embedded_input_Batch_32.bin", input_params, sizeof(float));
    void *dog_data = aspen_load_input_NHWC ("data/batched_input_64.bin", input_params, sizeof(float));
    aspen_init_naive (resnet50_dnn, input_params, dog_data, gpu);
    get_elapsed_time ("init_naive");
    aspen_run_naive (resnet50_dnn, input_params, dog_data, gpu);
    get_elapsed_time ("run_naive");
    // print_dnn_info(resnet50_dnn, 0);

    
    for (int i = 0; i < 2; i++)
    {
        printf ("\tLayer %d - Type %s\n", i, layer_type_str[resnet50_dnn->layers[i].type]);
        aspen_layer_t *layer = &resnet50_dnn->layers[i];
        nasm_ldata_t *ldata = &resnet50_nasm->ldata_arr[i];
        assert (ldata->layer == layer);
        // LAYER_PARAMS output_order[] = {BATCH, MAT_N, MAT_M, 0};
        LAYER_PARAMS output_order_nhwc[] = {BATCH, OUT_H, OUT_W, OUT_C};
        LAYER_PARAMS output_order_nchw[] = {BATCH, OUT_C, OUT_H, OUT_W};
        // if (layer->type == K_ATTENTION_LAYER)
        // {
        //     output_order [1] = NUM_HEAD; 
        //     output_order [2] = MAT_N; 
        //     output_order [3] = MAT_M;
        // }
        void *layer_output = get_aspen_tensor_data 
            (layer->tensors[OUTPUT_TENSOR], output_order_nchw, gpu);
        void *ldata_output = get_ldata_output (ldata, output_order_nchw);
        void *ldata_raw_output = get_packed_ldata_output_colwise (ldata);
        char filename[256];
        sprintf (filename, "data/resnet50_layer1.bin");
        // size_t elem_size = ldata->layer->dnn->element_size;
        // size_t data_size = ldata->out_mat_dims[OUT_H] * ldata->out_mat_dims[OUT_W] * elem_size;
        size_t elem_size = layer->dnn->element_size;
        size_t data_size = layer->tensors[OUTPUT_TENSOR]->num_elements * elem_size;
        void *expected_output = load_arr (filename, data_size);

        // printf ("Expected output for layer %d:\n", i);
        // print_float_tensor (expected_output, input_params[BATCH], 1
        //     , layer->params[MAT_N], layer->params[MAT_M]);
        // printf ("Computed output for layer %d:\n", i);
        // print_float_tensor (layer_output, input_params[BATCH], 1
        //     , layer->params[MAT_N], layer->params[MAT_M]);
        // printf ("Raw output for layer %d:\n", i);
        // print_float_tensor (ldata_raw_output, input_params[BATCH], 1
        //     , layer->params[MAT_N], layer->params[MAT_M]);

        // if (layer->tensors[WEIGHT_TENSOR] != NULL)
        // {
        //     if (layer->tensors[WEIGHT_TENSOR]->dims[SUB_C] != 0)
        //         print_float_tensor (layer->tensors[WEIGHT_TENSOR]->data, layer->tensors[WEIGHT_TENSOR]->dims[OUT_C], layer->tensors[WEIGHT_TENSOR]->dims[WEIGHT_H], 
        //             layer->tensors[WEIGHT_TENSOR]->dims[WEIGHT_W], layer->tensors[WEIGHT_TENSOR]->dims[IN_C] * layer->tensors[WEIGHT_TENSOR]->dims[SUB_C]);
        //     else
        //         print_float_tensor (layer->tensors[WEIGHT_TENSOR]->data, layer->tensors[WEIGHT_TENSOR]->dims[OUT_C], layer->tensors[WEIGHT_TENSOR]->dims[WEIGHT_H],
        //             layer->tensors[WEIGHT_TENSOR]->dims[WEIGHT_W], layer->tensors[WEIGHT_TENSOR]->dims[IN_C]);
        // }
        compare_float_tensor (layer_output, ldata_output, 
            resnet50_nasm->batch_size, layer->params[OUT_C], layer->params[OUT_H], layer->params[OUT_W],
            1e-2, 1e-4, 20);
        // compare_float_tensor (expected_output, ldata_output, 
        //     resnet50_nasm->batch_size, layer->params[OUT_C], layer->params[OUT_H], layer->params[OUT_W],
        //     1e-2, 1e-4, 20);
        // compare_float_tensor (expected_output, layer_output, 
        //     resnet50_nasm->batch_size, layer->params[OUT_C], layer->params[OUT_H], layer->params[OUT_W],
        //     1e-2, 1e-4, 20);
        
        // free (expected_output);
        free (ldata_output);
        // free (ldata_raw_output);
        // free (layer_output);
    }

    LAYER_PARAMS output_order[] = {BATCH, OUT_C, OUT_H, OUT_W};
    float *layer_output = ase_get_nasm_result (resnet50_nasm, output_order);
    float *softmax_output = ase_get_nasm_result (resnet50_nasm, output_order);
    naive_softmax (layer_output, softmax_output, input_params[BATCH], 1000);
    // float *layer_output = get_aspen_tensor_data ((resnet50_dnn->layers + resnet50_dnn->num_layers - 1)->tensors[OUTPUT_TENSOR], output_order);
    // print_float_array (layer_output, 1000*input_params[BATCH], 1000);
    for (int i = 0; i < resnet50_nasm->batch_size; i++)
    {
        get_probability_results ("data/imagenet_classes.txt", softmax_output + 1000*i, 1000);
    }
    free (layer_output);

    // ase_group_destroy (ase_group);
    // rpool_destroy (rpool);
    // // apu_destroy_nasm (resnet50_4_nasm);
    // apu_destroy_nasm (resnet50_nasm);
    apu_destroy_dnn (resnet50_dnn);
    return 0;
}