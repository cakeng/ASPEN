#include "input_parser.h"

aspen_dnn_t *parse_input (char *filename)
{
    input_type type = get_input_type(filename);
    aspen_dnn_t *dnn = NULL;
    switch(type){
        case darknet_cfg:
            dnn =  parse_darknet_cfg(filename);
            break;
        default:
            FPRT (stderr, "Input file type not supported: %s\n", filename);
            return NULL;
    }
    for (int i = 0; i < dnn->num_layers; i++)
    {
        set_layer_inout_sizes(&dnn->layers[i]);
    }
    for (int i = 0; i < dnn->num_layers; i++)
    {
        create_layer_tensors (&dnn->layers[i]);
    }
    return dnn;
}

input_type get_input_type (char *filename)
{
    char *ext = strrchr(filename, '.');
    if (ext == NULL) return 0;
    if (strcmp(ext, ".cfg") == 0) return darknet_cfg;
    return 0;
}

// Change to add a new layer type
void set_layer_inout_sizes (aspen_layer_t *layer)
{
    if (layer->type == INPUT_LAYER)
    {
        if (layer->params[MAT_M] != 0)
        {
            layer->params[OUT_H] = 1;
            layer->params[OUT_W] = layer->params[MAT_N];
            layer->params[OUT_C] = layer->params[MAT_M];
        }
        else
        {
            layer->params[OUT_C] = layer->params[IN_C];
            layer->params[OUT_H] = layer->params[IN_H];
            layer->params[OUT_W] = layer->params[IN_W];
        }
        return;
    }
    layer->params[IN_H] = layer->parent_layers[PARENT_0]->params[OUT_H];
    layer->params[IN_W] = layer->parent_layers[PARENT_0]->params[OUT_W];
    layer->params[IN_C] = layer->parent_layers[PARENT_0]->params[OUT_C];
    if (layer->type == CONV_LAYER)
    {
        if (layer->params[STRIDE] == 0)
            layer->params[STRIDE] = 1;
        if (layer->params[DILATION] == 0)
            layer->params[DILATION] = 1;
        layer->params[OUT_H] = (layer->params[IN_H] + 2*layer->params[PADDING] - layer->params[DILATION]*(layer->params[WEIGHT_H] - 1) - 1) / layer->params[STRIDE] + 1;
        layer->params[OUT_W] = (layer->params[IN_W] + 2*layer->params[PADDING] - layer->params[DILATION]*(layer->params[WEIGHT_W] - 1) - 1) / layer->params[STRIDE] + 1;
        return;
    }
    else if (layer->type == MAXPOOL_LAYER || layer->type == AVGPOOL_LAYER)
    {
        if (layer->params[STRIDE] == 0)
            layer->params[STRIDE] = 1;
        if (layer->params[DILATION] == 0)
            layer->params[DILATION] = 1;
        layer->params[OUT_H] = (layer->params[IN_H] + 2*layer->params[PADDING] - layer->params[DILATION]*(layer->params[WEIGHT_H] - 1) - 1) / layer->params[STRIDE] + 1;
        layer->params[OUT_W] = (layer->params[IN_W] + 2*layer->params[PADDING] - layer->params[DILATION]*(layer->params[WEIGHT_W] - 1) - 1) / layer->params[STRIDE] + 1;
        layer->params[OUT_C] = layer->params[IN_C];
        return;
    }
    else if (layer->type == SOFTMAX_LAYER || layer->type == FC_LAYER)
    {
        layer->params[OUT_H] = 1;
        layer->params[OUT_W] = 1;
        return;
    }
    else if (layer->type == RESIDUAL_LAYER)
    {
        if (layer->params[MAT_M] != 0)
        {
            layer->params[OUT_H] = 1;
            layer->params[OUT_W] = layer->params[MAT_N];
            layer->params[OUT_C] = layer->params[MAT_M];
        }
        else
        {
            layer->params[OUT_C] = layer->params[IN_C];
            layer->params[OUT_H] = layer->params[IN_H];
            layer->params[OUT_W] = layer->params[IN_W];
        }
        return;
    }
    else if (layer->type == YOLO_LAYER)
    {
        layer->params[OUT_H] = 1;
        layer->params[OUT_W] = layer->params[IN_H] * layer->params[IN_W] * (layer->params[IN_C] / layer->params[OUT_C]);
    }
    else if (layer->type == APPEND_LAYER)
    {
        layer->params[OUT_C] = layer->params[IN_C] + layer->parent_layers[PARENT_1]->params[OUT_C];
        layer->params[OUT_H] = layer->params[IN_H] * layer->params[STRIDE];
        layer->params[OUT_W] = layer->params[IN_W] * layer->params[STRIDE];
    }
    else if (layer->type == MATMUL_LAYER)
    {
        layer->params[MAT_K] = layer->parent_layers[PARENT_0]->params[MAT_M];
        return;
    }
    else if (layer->type == V_ATTENTION_LAYER || layer->type == LAYERNORM_LAYER)
    {
        return;
    }
    else if (layer->type == K_ATTENTION_LAYER)
    {
        layer->params[MAT_K] = layer->params[NUM_HIDDEN] / layer->params[NUM_HEAD];
        return;
    }
    else
    {
        FPRT(stderr, "ERROR: Unsupported layer type %s, at line %d in file %s\n" , layer_type_str[layer->type], __LINE__, __FILE__);
        assert (0);
    }
}

