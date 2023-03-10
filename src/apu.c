#include "aspen.h"
#include "apu.h"
#include "nasm.h"
#include "input_parser.h"

aspen_dnn_t *apu_create_dnn(char *input_path, char *weight_path)
{
    aspen_dnn_t *new_dnn = parse_input (input_path);

    return new_dnn;
}
void aspen_destroy_dnn(aspen_dnn_t *dnn)
{

}

nasm_t *apu_create_nasm(aspen_dnn_t *dnn)
{
    return NULL;
}
void aspen_destroy_nasm(nasm_t *nasm)
{

}

aspen_dnn_t *init_aspen_dnn(unsigned int num_layers, char* name)
{
    aspen_dnn_t *new_dnn = (aspen_dnn_t *) calloc(1, sizeof(aspen_dnn_t));
    strncpy(new_dnn->name, name, 256);
    new_dnn->num_layers = num_layers;
    new_dnn->layers = (aspen_layer_t *) calloc(num_layers, sizeof(aspen_layer_t));
    for (int i = 0; i < num_layers; i++)
    {
        init_aspen_layer(&new_dnn->layers[i], i, new_dnn);
    }
    return new_dnn;
}

void init_aspen_layer (aspen_layer_t *layer, unsigned int layer_idx, aspen_dnn_t *dnn)
{
    layer->layer_idx = layer_idx;
    layer->dnn = dnn;
}
