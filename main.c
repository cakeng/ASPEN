#include <stdio.h>
#include "aspen.h"
#include "util.h"
#include "nasm.h"
#include "apu.h"


int main(void)
{
    // print_aspen_build_info();
    aspen_dnn_t *resnet50_dnn = apu_create_dnn("data/cfg/resnet50.cfg", "data/resnet50/resnet50_data.bin");
    // aspen_dnn_t *yolov3_dnn = apu_create_dnn("data/cfg/yolov3.cfg", "data/yolov3_data.bin");
    // aspen_dnn_t *bert_dnn = apu_create_dnn ("data/cfg/bert_base_encoder.cfg", "data/bert_base_data.bin");
    // aspen_dnn_t *gpt2_dnn = apu_create_dnn ("data/cfg/gpt2_124M_encoder.cfg", "data/gpt2/gpt2_124M_data.bin");
    // // // if (bert_dnn == NULL) 
    // // // {
    // // //     printf("Error: Failed to create DNN\n");
    // // //     return -1;
    // // // }
    // // print_dnn_info (gpt2_dnn, 0);
    // apu_save_dnn_to_file (gpt2_dnn, "data/gpt2/gpt2_124M_base.aspen");
    // aspen_dnn_t *gpt2_dnn = apu_load_dnn_from_file ("data/gpt2_base.aspen");
    // // if (bert_dnn_2 == NULL) 
    // // {
    // //     printf("Error: Failed to read DNN\n");
    // //     return -1;
    // // }
    // // // print_dnn_info (bert_dnn_2, 0);
    
    // nasm_t *bert_nasm = apu_create_transformer_nasm(bert_dnn, 1e6, 100, 1, 480);
    // nasm_t *gpt2_nasm = apu_create_transformer_nasm(gpt2_dnn, 10e6, 100, 1, 128);
    nasm_t *resnet50_nasm = apu_create_nasm(resnet50_dnn, 100e6, 100, 32);
    // nasm_t *yolov3_nasm = apu_create_nasm(yolov3_dnn, 100e6, 100, 1);
    // // if (bert_nasm == NULL) 
    // // {
    // //     printf("Error: Failed to create NASM\n");
    // //     return -1;
    // // }
    // print_nasm_info(gpt2_nasm, 1, 0);
    // char nasm_file_name [1024] = {0};
    // sprintf (nasm_file_name, "data/gpt2_124M_S%d_B%d_M%d_%2.1e.nasm"
    //     , gpt2_nasm->tr_seq_len, gpt2_nasm->batch_size, gpt2_nasm->min_ninst_per_ldata,
    //     (double)gpt2_nasm->flop_per_ninst);
    // // sprintf (nasm_file_name, "data/yolov3_B%d_M%d_%2.1e.nasm",
    // //     yolov3_nasm->batch_size, yolov3_nasm->min_ninst_per_ldata,
    // //     (double)yolov3_nasm->flop_per_ninst);
    // apu_save_nasm_to_file (gpt2_nasm, nasm_file_name);
    int gpu = -1;
    // aspen_dnn_t *bert_dnn = apu_load_dnn_from_file ("data/bert_base.aspen");
    // nasm_t *bert_nasm = apu_load_nasm_from_file ("data/bert_S128_B8.nasm", bert_dnn);
    // aspen_dnn_t *resnet50_dnn = apu_load_dnn_from_file ("data/resnet50_base.aspen");
    // nasm_t *resnet50_nasm = apu_load_nasm_from_file ("data/resnet50_B32.nasm", resnet50_dnn);
    // aspen_dnn_t *yolov3_dnn = apu_load_dnn_from_file ("data/yolov3_base.aspen");
    // nasm_t *yolov3_nasm = apu_load_nasm_from_file ("data/yolov3_B1_M100_1.0e+08.nasm", yolov3_dnn);
    // // // // nasm_t *bert_4_nasm = apu_load_nasm_from_file ("data/bert_4.nasm", &bert_dnn);
    // // 
    rpool_t *rpool = rpool_init (gpu);
    ase_group_t *ase_group = ase_group_init (64, gpu);
    ase_group_set_rpool (ase_group, rpool);

    // // // rpool_add_nasm_raw_input (rpool, bert_4_nasm, 0.5, dog_data);
    rpool_add_nasm (rpool, resnet50_nasm, 1.0, "data/resnet50/batched_input_64.bin");
    // rpool_add_nasm (rpool, yolov3_nasm, 1.0, "data/yolov3_cat_input_128.bin");
    // rpool_add_nasm (rpool, gpt2_nasm, 1.0, "data/gpt2/gpt2_124M_128_layer0_input.bin");
    // // print_rpool_info (rpool);
    // // print_nasm_info(bert_nasm, 0, 0);
    // print_dnn_info(bert_dnn, 0);
    // print_dnn_info(yolov3_dnn, 0);

    get_elapsed_time ("init");

    // // ase_cudagraph_run (rpool, bert_nasm);
    
    ase_group_run (ase_group);
    // ase_wait_for_nasm_completion (bert_nasm);
    ase_wait_for_nasm_completion (resnet50_nasm);
    // ase_wait_for_nasm_completion (gpt2_nasm);
    // ase_wait_for_nasm_completion (bert_4_nasm);
    ase_group_stop (ase_group);


    get_elapsed_time ("run_aspen2");
    // // rpool_reset_nasm (rpool, bert_nasm, 1.0);
    // print_nasm_info(bert_nasm, 1, 0);
    // // print_rpool_info (rpool);
    // print_nasm_cudagraph_info (bert_nasm, "cudagraph_out.txt");

    // gpt2_dnn->layers[7].tensors[WEIGHT_TENSOR]->data = aspen_calloc (1600*1600,4);
    // unsigned int input_params[NUM_PARAM_ELEMENTS] = {0};
    // input_params[BATCH] = 1; input_params[NUM_SEQ] = 128; input_params[NUM_HIDDEN] = 768;
    // // input_params[BATCH] = 1; input_params[OUT_C] = 3; input_params[OUT_H] = 416; input_params[OUT_W] = 416;
    // void *dog_data = aspen_load_input ("data/gpt2/gpt2_124M_128_layer0_input.bin", input_params, sizeof(float));
    // aspen_init_naive (gpt2_dnn, input_params, dog_data, gpu);
    // get_elapsed_time ("init_naive");
    // aspen_run_naive (gpt2_dnn, input_params, dog_data, gpu);
    // get_elapsed_time ("run_naive");
    // // print_dnn_info (gpt2_dnn, 0);
    // for (int i = 144; i < 145; i++)
    // {
    //     printf ("\tLayer %d - Type %s\n", i, layer_type_str[gpt2_dnn->layers[i].type]);
    //     aspen_layer_t *layer = &gpt2_dnn->layers[i];
    //     nasm_ldata_t *ldata = &gpt2_nasm->ldata_arr[i];
    //     assert (ldata->layer == layer);
    //     LAYER_PARAMS output_order[] = {BATCH, MAT_N, MAT_M, 0};
    //     LAYER_PARAMS output_order_nhwc[] = {BATCH, OUT_H, OUT_W, OUT_C};
    //     LAYER_PARAMS output_order_nchw[] = {BATCH, OUT_C, OUT_H, OUT_W};
    //     if (layer->type == K_ATTENTION_LAYER)
    //     {
    //         output_order [1] = NUM_HEAD; 
    //         output_order [2] = MAT_N; 
    //         output_order [3] = MAT_M;
    //     }
    //     void *layer_output = get_aspen_tensor_data 
    //         (layer->tensors[OUTPUT_TENSOR], output_order_nhwc, gpu);
    //     void *ldata_output = get_ldata_output (ldata, output_order);
    //     // void *ldata_raw_output = get_packed_ldata_output_colwise (ldata);
    //     char filename[256];
    //     sprintf (filename, "data/gpt2/gpt2_124M_128_layer11_output.bin");
    //     // size_t elem_size = ldata->layer->dnn->element_size;
    //     // size_t data_size = ldata->out_mat_dims[OUT_H] * ldata->out_mat_dims[OUT_W] * elem_size;
    //     size_t elem_size = layer->dnn->element_size;
    //     size_t data_size = layer->tensors[OUTPUT_TENSOR]->num_elements * elem_size;
    //     void *expected_output = load_arr (filename, data_size);

    //     // printf ("Expected output for layer %d:\n", i);
    //     // print_float_tensor (expected_output, input_params[BATCH], layer->params[NUM_HEAD]
    //     //     , layer->params[MAT_N], layer->params[MAT_M]);
    //     // printf ("Computed output for layer %d:\n", i);
    //     // print_float_tensor (layer_output, input_params[BATCH], layer->params[NUM_HEAD]
    //     //     , layer->params[MAT_N], layer->params[MAT_M]);
    //     // printf ("Raw output for layer %d:\n", i);
    //     // print_float_tensor (ldata_raw_output, input_params[BATCH], 1
    //     //     , layer->params[MAT_N], layer->params[MAT_M]);

    //     // if (layer->tensors[WEIGHT_TENSOR] != NULL)
    //     // {
    //     //     print_tensor_info (layer->tensors[WEIGHT_TENSOR], 1);
    //     // }
    //     compare_float_tensor (expected_output, ldata_output, 
    //         input_params[BATCH],1, layer->params[MAT_N],
    //         layer->params[MAT_M], 1e-2, 1e-4, 20);
    //     // compare_float_tensor (layer_output, ldata_output, 
    //     //     input_params[BATCH],1, layer->params[MAT_N],
    //     //     layer->params[MAT_M], 1e-2, 1e-4, 20);
    //     compare_float_tensor (expected_output, layer_output, 
    //         input_params[BATCH],1, layer->params[MAT_N],
    //         layer->params[MAT_M], 1e-2, 1e-4, 20);
        
        // free (expected_output);
        // free (ldata_output);
        // free (ldata_raw_output);
        // free (layer_output);
    // }

    // LAYER_PARAMS output_order[] = {BATCH, OUT_H, OUT_W, OUT_C};
    // LAYER_PARAMS output_order[] = {BATCH, MAT_N, MAT_M, 0};
    // float *layer_output = ase_get_nasm_result (gpt2_nasm, output_order);
    // void *expected_output = load_arr ("data/gpt2/gpt2_128_layer0_output.bin", 128*1600*sizeof(float));
    // compare_float_array (expected_output, layer_output, 128*1600, 1e-2, 1e-4, 20);
    // float *softmax_output = calloc (1000*resnet50_nasm->batch_size, sizeof(float));
    // naive_softmax (layer_output, softmax_output, resnet50_nasm->batch_size, 1000);
    // // float *layer_output = get_aspen_tensor_data ((resnet50_dnn->layers + resnet50_dnn->num_layers - 1)->tensors[OUTPUT_TENSOR], output_order);
    // // print_float_array (layer_output, 1000*resnet50_nasm->batch_size, 1000);
    // for (int i = 0; i < resnet50_nasm->batch_size; i++)
    // {
    //     get_probability_results ("data/imagenet_classes.txt", softmax_output + 1000*i, 1000);
    // }
    // free (layer_output);
    // free (softmax_output);

    // ase_group_destroy (ase_group);
    // rpool_destroy (rpool);
    // // apu_destroy_nasm (bert_4_nasm);
    // apu_destroy_nasm (bert_nasm);
    // apu_destroy_dnn (bert_dnn);
    return 0;
}