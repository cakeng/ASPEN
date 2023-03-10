#include "aspen.h"
#include "nasm.h"
#include "apu.h"

char *layer_type_str [NUM_LAYER_ELEMENTS] = 
{
    [NO_LAYER_TYPE] = "NO_LAYER_TYPE", [INPUT_LAYER] = "INPUT_LAYER", [CONV_LAYER] = "CONV_LAYER", [FC_LAYER] = "FC_LAYER",
    [RESIDUAL_LAYER] = "RESIDUAL_LAYER", [BATCHNORM_LAYER] = "BATCHNORM_LAYER", [YOLO_LAYER] = "YOLO_LAYER", [ACTIVATION_LAYER] = "ACTIVATION_LAYER", [MAXPOOL_LAYER] = "MAXPOOL_LAYER", [AVGPOOL_LAYER] = "AVGPOOL_LAYER",
    [ROUTE_LAYER] = "ROUTE_LAYER", [SOFTMAX_LAYER] = "SOFTMAX_LAYER"
};

char *param_type_str[NUM_PARAM_ELEMENTS] = 
{
    [OUT_W] = "OUT_W", [OUT_H] = "OUT_H", [IN_W] = "IN_W", [IN_H] = "IN_H", [IN_C] = "IN_C", [OUT_C] = "OUT_C", [F_W] = "F_W", [F_H] = "F_H", [STRIDE] = "STRIDE", [PADDING] = "PADDING", [DILATION] = "DILATION", [GROUPS] = "GROUPS",
    [SEQ_LEN] = "SEQ_LEN", [HEAD_NUM] = "HEAD_NUM", [HIDDEN_PER_HEAD] = "HIDDEN_PER_HEAD",
    [FORM_BYTES] = "FORM_BYTES"
};

char *tensor_type_str[NUM_TENSOR_ELEMENTS] = 
{
    [NULL_TENSOR] = "NULL_TENSOR", [OUTPUT] = "OUTPUT", [INPUT] = "INPUT", [FILTER] = "FILTER", [BIAS] = "BIAS",
};

char *parent_type_str[NUM_PARENT_ELEMENTS] = 
{
    [PARENT_NONE] = "PARENT_NONE", [PARENT_0] = "PARENT_0", [PARENT_1] = "PARENT_1", [PARENT_FILTER] = "PARENT_FILTER",
};

char *activation_type_str [NUM_ACTIVATION_ELEMENTS] = 
{
    [NO_ACTIVATION] = "NO_ACTIVATION", [SIGMOID] = "SIGMOID", [LINEAR] = "LINEAR", [TANH] = "TANH", [RELU] = "RELU", [LEAKY_RELU] = "LEAKY_RELU", [ELU] = "ELU", [SELU] = "SELU"
};
char *nist_op_str [NUM_NIST_OP_ELEMENTS] = 
{
    [NO_OPERATION] = "NO_OPERATION", [N_CONV2D] = "N_CONV2D", [N_FC] = "N_FC"
};

void print_dnn_info (aspen_dnn_t *dnn)
{
    printf("//////// Printing DNN Info ////////\n");
    printf("DNN Name: %s\n", dnn->name);
    printf("Number of Layers: %d\n", dnn->num_layers);
    for (int i = 0; i < dnn->num_layers; i++)
    {
        print_layer_info(&dnn->layers[i]);
    }
    printf("//////// End of DNN Info ////////\n");
}

void print_layer_info (aspen_layer_t *layer)
{
    printf("//////// Printing Layer Info ////////\n");
    printf("Layer Type: %s\n", layer_type_str[layer->type]);
    printf("Layer Index: %d\n", layer->layer_idx);
    printf("Layer Activation: %s\n", activation_type_str[layer->activation]);
    printf("Layer Params:\n");
    for (LAYER_PARAMS i = 0; i < NUM_PARAM_ELEMENTS; i++)
    {
        if (i != NUM_PARAM_ELEMENTS && layer->params[i] != 0)
            printf("\t%s: %d\n", param_type_str[i], layer->params[i]);
    }

}