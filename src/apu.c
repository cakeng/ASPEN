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
    if (dnn == NULL)
        return;
    if (dnn->ref_nasms != 0)
    {
        FPRT (stderr, "Cannot destroy dnn %s with %d nasms still referencing it.\n"
            , dnn->name, dnn->ref_nasms);
        return;
    }
    destroy_aspen_layers(dnn->layers, dnn->num_layers);
    free(dnn);
}

void apu_load_nasm_from_file(char *filename, nasm_t *output_nasm, aspen_dnn_t *output_dnn)
{

}

void apu_save_nasm_to_file(char *filename)
{
    
}

void apu_save_dnn_to_file(char *filename)
{
    
}


int get_nasm_ldata_num_per_layer (aspen_layer_t *layer)
{
    switch (layer->type)
    {
    default:
        return 1;
    }
}

void update_ldata_child_list (nasm_ldata_t *ldata)
{
    if (ldata->num_child_ldata == 0)
        return;
    ldata->child_ldata_idx_arr = calloc(ldata->num_child_ldata, sizeof(unsigned int));
    unsigned int child_idx = 0;
    for (int lidx = 0; lidx < ldata->nasm->num_ldata; lidx++)
    {
        nasm_ldata_t *child = ldata->nasm->ldata_arr + lidx;
        for (LAYER_PARENTS i = 0; i < NUM_PARENT_ELEMENTS; i++)
        {
            if (ldata->nasm->ldata_arr + child->parent_ldata_idx_arr[i] == ldata)
            {
                ldata->child_ldata_idx_arr[child_idx] = lidx;
                child_idx++;
            }
        }
    }
}

nasm_t *apu_create_nasm(aspen_dnn_t *dnn, int flop_per_ninst)
{
    nasm_t *new_nasm = (nasm_t *) calloc(1, sizeof(nasm_t));
    new_nasm->dnn = dnn;
    new_nasm->flop_per_ninst = flop_per_ninst;
    for (int i = 0; i < dnn->num_layers; i++)
    {
        new_nasm->num_ldata += get_nasm_ldata_num_per_layer(&dnn->layers[i]);
    }
    new_nasm->ldata_arr = calloc(new_nasm->num_ldata, sizeof(nasm_ldata_t));
    nasm_ldata_t *ldata_ptr = new_nasm->ldata_arr;
    for (int i = 0; i < dnn->num_layers; i++)
    {
        init_nasm_ldata(new_nasm, ldata_ptr, &dnn->layers[i]);
        ldata_ptr += get_nasm_ldata_num_per_layer(&dnn->layers[i]);
    }
    for (int i = 0; i < new_nasm->num_ldata; i++)
    {
        update_ldata_child_list(&new_nasm->ldata_arr[i]);
    }
    dnn->ref_nasms++;
    return new_nasm;
}
void destroy_nasm_ldata (nasm_ldata_t *ldata)
{
    if (ldata == NULL)
        return;
    if (ldata->ninst_arr != NULL)
        free(ldata->ninst_arr);
    if (ldata->out_mat != NULL)
        free(ldata->out_mat);
    if (ldata->gpu_out_mat != NULL)
        cudaFree(ldata->gpu_out_mat);
    if (ldata->child_ldata_idx_arr != NULL)
        free(ldata->child_ldata_idx_arr);
}

void destroy_nasm_ldata_arr (nasm_ldata_t *ldata_arr, int num_ldata)
{
    if (ldata_arr == NULL)
        return;
    for (int i = 0; i < num_ldata; i++)
    {
        destroy_nasm_ldata(&ldata_arr[i]);
    }
    free(ldata_arr);
}

void aspen_destroy_nasm(nasm_t *nasm)
{
    if (nasm == NULL)
        return;
    destroy_nasm_ldata_arr(nasm->ldata_arr, nasm->num_ldata);
    nasm->dnn->ref_nasms--;
    free(nasm);
}

void get_out_mat_info (nasm_ldata_t *ldata)
{
    aspen_layer_t *layer = ldata->layer;
    switch (layer->type)
    {
    case CONV_LAYER:
        ldata->flop_per_output = 2*layer->params[F_H]*layer->params[F_W]*layer->params[IN_C];
        ldata->out_mat_dims[OUT_H] = layer->params[OUT_C];
        ldata->out_mat_dims[OUT_W] = layer->params[OUT_H]*layer->params[OUT_W];
        break;
    case FC_LAYER:
        ldata->flop_per_output = 2*layer->params[IN_C];
        ldata->out_mat_dims[OUT_H] = layer->params[OUT_C];
        ldata->out_mat_dims[OUT_W] = 1;
        break;
    case MAXPOOL_LAYER:
        ldata->flop_per_output = layer->params[F_H]*layer->params[F_W];
        ldata->out_mat_dims[OUT_H] = layer->params[OUT_C];
        ldata->out_mat_dims[OUT_W] = layer->params[OUT_H]*layer->params[OUT_W];
        break;
    case AVGPOOL_LAYER:
        ldata->flop_per_output = layer->params[F_H]*layer->params[F_W];
        ldata->out_mat_dims[OUT_H] = layer->params[OUT_C];
        ldata->out_mat_dims[OUT_W] = layer->params[OUT_H]*layer->params[OUT_W];
        break;
    default:
        break;
    }
}

void get_ninst_tile_dims (nasm_ldata_t *ldata)
{
    ldata->ninst_tile_dims[OUT_H] = NINST_H_MIN;
    ldata->ninst_tile_dims[OUT_W] = NINST_W_MIN;
    while (ldata->ninst_tile_dims[OUT_H] < ldata->out_mat_dims[OUT_H] && ldata->ninst_tile_dims[OUT_H]*ldata->ninst_tile_dims[OUT_W] < ldata->nasm->flop_per_ninst/ldata->flop_per_output)
    {
        ldata->ninst_tile_dims[OUT_H]++;
    }
    while (ldata->ninst_tile_dims[OUT_W] < ldata->out_mat_dims[OUT_W] && ldata->ninst_tile_dims[OUT_H]*ldata->ninst_tile_dims[OUT_W] < ldata->nasm->flop_per_ninst/ldata->flop_per_output)
    {
        ldata->ninst_tile_dims[OUT_W]++;
    }
}

void init_nasm_ldata (nasm_t *nasm, nasm_ldata_t *ldata_ptr, aspen_layer_t *layer)
{
    ldata_ptr->nasm = nasm;
    ldata_ptr->layer = layer;
    for (LAYER_PARENTS i = 0; i < NUM_PARENT_ELEMENTS; i++)
    {
        ldata_ptr->parent_ldata_idx_arr[i] = -1;
        if (layer->parent_layers[i] != NULL)
        {
            for (int lidx = 0; lidx < nasm->dnn->num_layers; lidx++)
            {
                nasm_ldata_t *parent_ldata = nasm->ldata_arr + lidx;
                if (parent_ldata->layer == layer->parent_layers[i])
                {
                    ldata_ptr->parent_ldata_idx_arr [i] = lidx;
                    parent_ldata->num_child_ldata++;
                }
            }
        }
    }
    ldata_ptr->flop_per_output = 1;
    get_out_mat_info (ldata_ptr);
    get_ninst_tile_dims (ldata_ptr);
    unsigned int out_w = get_smallest_dividable (ldata_ptr->out_mat_dims[OUT_W], ldata_ptr->ninst_tile_dims[OUT_W]);
    unsigned int out_h = get_smallest_dividable (ldata_ptr->out_mat_dims[OUT_H], ldata_ptr->ninst_tile_dims[OUT_H]);
    ldata_ptr->out_mat_stride = out_h;
    ldata_ptr->out_mat_size = (size_t)out_h*out_w;
    ldata_ptr->out_mat = NULL;
    ldata_ptr->gpu_out_mat = NULL;
    ldata_ptr->num_ninst = (out_h/ldata_ptr->ninst_tile_dims[OUT_H])*(out_w/ldata_ptr->ninst_tile_dims[OUT_W]);
    ldata_ptr->ninst_arr = calloc(ldata_ptr->num_ninst, sizeof(ninst_t));
}

aspen_dnn_t *init_aspen_dnn(unsigned int num_layers, char* name)
{
    aspen_dnn_t *new_dnn = (aspen_dnn_t *) calloc(1, sizeof(aspen_dnn_t));
    strncpy(new_dnn->name, name, MAX_STRING_LEN);
    new_dnn->element_size = sizeof(float);
    new_dnn->num_layers = num_layers;
    new_dnn->layers = (aspen_layer_t *) calloc(num_layers, sizeof(aspen_layer_t));
    for (int i = 0; i < num_layers; i++)
    {
        init_aspen_layer(new_dnn->layers + i, i, new_dnn);
    }
    return new_dnn;
}

void init_aspen_layer (aspen_layer_t *layer, unsigned int layer_idx, aspen_dnn_t *dnn)
{
    layer->layer_idx = layer_idx;
    layer->dnn = dnn;
}

void destroy_aspen_layer (aspen_layer_t* layer)
{
    if (layer == NULL)
        return;
    for (int i = 0; i < NUM_TENSOR_ELEMENTS; i++)
    {
        destroy_aspen_tensor(layer->tensors[i]);
    }
}

void destroy_aspen_layers (aspen_layer_t* layers, unsigned int num_layers)
{
    if (layers == NULL)
        return;
    for (int i = 0; i < num_layers; i++)
    {
        destroy_aspen_layer (layers + i);
    }
    free (layers);
}

aspen_tensor_t *init_aspen_tensor (unsigned int *params_arr, LAYER_PARAMS *dim_info_arr, int num_dims, size_t element_size)
{
    aspen_tensor_t *new_tensor = (aspen_tensor_t *) calloc(1, sizeof(aspen_tensor_t));
    new_tensor->num_dims = num_dims;
    new_tensor->num_elements = 1;
    for (int i = 0; i < num_dims; i++)
    {
        new_tensor->dims_info[i] = dim_info_arr[i];
        new_tensor->dims[i] = params_arr[dim_info_arr[i]];
        new_tensor->num_elements *= new_tensor->dims[i];
    }
    new_tensor->data = aspen_calloc(new_tensor->num_elements, element_size);
    return new_tensor;
}

void create_tensors (aspen_layer_t *layer)
{
    if (layer->type == CONV_LAYER)
    {
        LAYER_PARAMS conv_params[] = {IN_C, OUT_C, F_H, F_W};
        layer->tensors [FILTER] = init_aspen_tensor (layer->params, conv_params, 4, layer->dnn->element_size);
        LAYER_PARAMS bias_params[] = {OUT_C};
        layer->tensors [BIAS] = init_aspen_tensor (layer->params, bias_params, 1, layer->dnn->element_size);
        return;
    }
    if (layer->type == FC_LAYER)
    {
        LAYER_PARAMS fc_params[] = {IN_C, OUT_C};
        layer->tensors [FILTER] = init_aspen_tensor (layer->params, fc_params, 2, layer->dnn->element_size);
        LAYER_PARAMS bias_params[] = {OUT_C};
        layer->tensors [BIAS] = init_aspen_tensor (layer->params, bias_params, 1, layer->dnn->element_size);
        return;
    }
}

void destroy_aspen_tensor(aspen_tensor_t *tensor)
{
    if (tensor == NULL)
        return;
    if (tensor->data != NULL)
        aspen_free(tensor->data);
    free(tensor);
}

ninst_t *get_ninst_from_tensor_pos (nasm_ldata_t *ldata, aspen_tensor_t *tensor_pos)
{
    unsigned int out_mat_pos[2] = {0,0};
    get_out_mat_pos_from_tensor_pos (ldata, tensor_pos, out_mat_pos);
    return get_ninst_from_out_mat_pos (ldata, out_mat_pos[0], out_mat_pos[1]);
}
ninst_t *get_ninst_from_out_mat_pos (nasm_ldata_t *ldata, unsigned int h, unsigned int w)
{
    unsigned int out_h = get_smallest_dividable (ldata->out_mat_dims[OUT_H], ldata->ninst_tile_dims[OUT_H]);
    unsigned int ninst_idx = (w/ldata->ninst_tile_dims[OUT_W])*(out_h/ldata->ninst_tile_dims[OUT_H]) + (h/ldata->ninst_tile_dims[OUT_H]);
    return ldata->ninst_arr + ninst_idx;
}
void get_out_mat_pos_from_tensor_pos (nasm_ldata_t *ldata, aspen_tensor_t *tensor_pos, unsigned int *out_mat_pos)
{

}
void get_tensor_pos_from_out_mat_pos (nasm_ldata_t *ldata, unsigned int *out_mat_pos, aspen_tensor_t *tensor_pos)
{

}
void get_parent_tensor_pos_from_child_tensor_pos (nasm_ldata_t *ldata, aspen_tensor_t *child_tensor_pos, aspen_tensor_t *parent_tensor_pos)
{

}