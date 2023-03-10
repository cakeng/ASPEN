#include "aspen.h"
#include "nasm.h"
#include "apu.h"

char *layer_type_str [NUM_LAYER_ELEMENTS] = {
    "NO_LAYER_TYPE", "INPUT_LAYER", "CONV_LAYER", "FC_LAYER",
    "RESIDUAL_LAYER", "BATCHNORM_LAYER", "YOLO_LAYER", "ACTIVATION_LAYER", "MAXPOOL_LAYER", "AVGPOOL_LAYER",
    "ROUTE_LAYER", "DROPOUT_LAYER", "SOFTMAX_LAYER"
};

char *param_type_str[NUM_PARAM_ELEMENTS] = {
    "OUT_W", "OUT_H", "IN_W", "IN_H", "IN_C", "OUT_C", "F_W", "F_H", "STRIDE", "PADDING", "DILATION", "GROUPS",
    "SEQ_LEN", "HEAD_NUM", "HIDDEN_PER_HEAD",
    "FORM_BYTES"
};

char *tensor_type_str[NUM_TENSOR_ELEMENTS] = {
    "NULL_TENSOR", "OUTPUT", "INPUT", "FILTER", "BIAS"
};

char *parent_type_str[NUM_PARENT_ELEMENTS] = {
    "PARENT_NONE", "PARENT_0", "PARENT_1", "PARENT_FILTER"
};

void print_dnn_info (aspen_dnn_t *dnn)
{
    printf("//////// Printing DNN Info ////////\n");
}

void print_layer_info (aspen_layer_t *layer)
{
    printf("//////// Printing Layer Info ////////\n");
}