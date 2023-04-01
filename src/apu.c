#include "aspen.h"
#include "apu.h"
#include "nasm.h"
#include "input_parser.h"

static unsigned int nasm_num = 0;

aspen_dnn_t *apu_create_dnn (char *input_path, char *weight_path)
{
    aspen_dnn_t *new_dnn = parse_input (input_path);
    return new_dnn;
}

void apu_destroy_dnn (aspen_dnn_t *dnn)
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

aspen_dnn_t *init_aspen_dnn (unsigned int num_layers, char* name)
{
    aspen_dnn_t *new_dnn = (aspen_dnn_t *) calloc(1, sizeof(aspen_dnn_t));
    strncpy(new_dnn->name, name, MAX_STRING_LEN-1);
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

aspen_tensor_t *init_aspen_tensor (unsigned int *params_arr, LAYER_PARAMS *dim_order_arr, int num_dims)
{
    aspen_tensor_t *new_tensor = (aspen_tensor_t *) calloc(1, sizeof(aspen_tensor_t));
    new_tensor->num_dims = num_dims;
    new_tensor->num_elements = 1;
    for (int i = 0; i < num_dims; i++)
    {
        new_tensor->data_dim_order[i] = dim_order_arr[i];
        new_tensor->dims[dim_order_arr[i]] = params_arr[dim_order_arr[i]];
        new_tensor->num_elements *= new_tensor->dims[dim_order_arr[i]];
    }
    return new_tensor;
}

void fill_tensor_with_nums (aspen_tensor_t *tensor)
{
    if (tensor == NULL || tensor->data == NULL)
        return;
    size_t tensor_dims[MAX_TENSOR_DIMS];
    for (int i = 0; i < MAX_TENSOR_DIMS; i++)
    {
        tensor_dims[i] = 1;
    }
    for (int i = tensor->num_dims - 1; i >= 0; i--)
    {
        for (int j = i; j < tensor->num_dims; j++)
        {
            tensor_dims[i] *= tensor->dims[tensor->data_dim_order[i]];
        }
    }
    for (size_t i = 0; i < tensor->num_elements; i++)
    {
        double out = 0;
        size_t idx = i;
        for (int j = 0; j < tensor->num_dims; j++)
        {
            out *= 100;
            out += (idx / tensor_dims[j+1])*0.01;
            idx = idx % tensor_dims[j+1];
        }
        ((float *)tensor->data)[i] = out;
    }
}

void fill_tensor_with_fixed_num (aspen_tensor_t *tensor, float num)
{
    if (tensor == NULL || tensor->data == NULL)
        return;
    for (size_t i = 0; i < tensor->num_elements; i++)
    {
        ((float *)tensor->data)[i] = num;
    }
}

void create_layer_tensors (aspen_layer_t *layer)
{
    if (layer->type == CONV_LAYER)
    {
        LAYER_PARAMS weight_dim_order[] = {OUT_C, IN_C, F_H, F_W};
        layer->tensors [FILTER] = init_aspen_tensor (layer->params, weight_dim_order, 4);
        layer->tensors [FILTER]->data = aspen_calloc(layer->tensors [FILTER]->num_elements, layer->dnn->element_size);
        LAYER_PARAMS bias_dim_order[] = {OUT_C};
        layer->tensors [BIAS] = init_aspen_tensor (layer->params, bias_dim_order, 1);
        layer->tensors [BIAS]->data = aspen_calloc(layer->tensors [BIAS]->num_elements, layer->dnn->element_size);
    }
    else if (layer->type == FC_LAYER)
    {
        LAYER_PARAMS weight_dim_order[] = {OUT_C, IN_C};
        layer->tensors [FILTER] = init_aspen_tensor (layer->params, weight_dim_order, 2);
        layer->tensors [FILTER]->data = aspen_calloc(layer->tensors [FILTER]->num_elements, layer->dnn->element_size);
        LAYER_PARAMS bias_dim_order[] = {OUT_C};
        layer->tensors [BIAS] = init_aspen_tensor (layer->params, bias_dim_order, 1);
        layer->tensors [BIAS]->data = aspen_calloc(layer->tensors [BIAS]->num_elements, layer->dnn->element_size);
    }
    else if (layer->type == INPUT_LAYER || layer->type == MAXPOOL_LAYER || layer->type == AVGPOOL_LAYER || layer->type == SOFTMAX_LAYER
        || layer->type == RESIDUAL_LAYER)
    {
    }
    else
    {
        FPRT(stderr, "ERROR: Unsupported layer type %s, at line %d in file %s\n" , layer_type_str[layer->type], __LINE__, __FILE__);
        exit(1);
    }
    #ifdef DEBUG
    for (int i = 0; i < NUM_TENSOR_ELEMENTS; i++)
    {
        if (layer->tensors[i] != NULL)
        {
            fill_tensor_with_nums (layer->tensors[i]);
        }
    }
    #endif
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

void ninst_find_parent (ninst_t *ninst)
{
    nasm_ldata_t *ldata = ninst->ldata;
    aspen_layer_t *layer = ldata->layer;
    ninst->num_parent_ninsts = 0;
    unsigned int *parent_arr = calloc (MAX_PARENT_NINST_NUM, sizeof(unsigned int));
    for (unsigned int tile_w = 0; tile_w < ldata->ninst_tile_dims[OUT_W]; tile_w++)
    {
        for (unsigned int tile_h = 0; tile_h < ldata->ninst_tile_dims[OUT_H]; tile_h++)
        {
            unsigned int out_mat_pos[2] = {ninst->out_mat_pos[OUT_W] + tile_w, ninst->out_mat_pos[OUT_H] + tile_h};
            unsigned int out_tensor_pos[NUM_PARAM_ELEMENTS] = {0}, in_tensor_pos[NUM_PARAM_ELEMENTS] = {0}; 
            get_tensor_pos_from_out_mat_pos(ldata, out_mat_pos, out_tensor_pos);
            in_tensor_pos[BATCH] = out_tensor_pos[BATCH];
            nasm_ldata_t *parent_ldata = ldata->nasm->ldata_arr + ldata->parent_ldata_idx_arr[PARENT_0];
            // printf ("Layer %d, ninst %d, tile_w %d, tile_h %d, tensor pos %d,%d,%d,%d -", 
            //     ldata->layer->layer_idx, ninst->ninst_idx, tile_w, tile_h, out_tensor_pos[BATCH], out_tensor_pos[OUT_C],
            //         out_tensor_pos[OUT_H],
            //          out_tensor_pos[OUT_W]);
            if (layer->type == CONV_LAYER || layer->type == MAXPOOL_LAYER || layer->type == AVGPOOL_LAYER)
            {
                for (int i = 0; i < layer->params[IN_C]; i++)
                {
                    in_tensor_pos[OUT_C] = i;
                    for (int j = 0; j < layer->params[F_H]; j++)
                    {
                        in_tensor_pos[OUT_H] = out_tensor_pos[OUT_H]*layer->params[STRIDE]
                            + j*layer->params[DILATION] - layer->params[PADDING];
                        for (int k = 0; k < layer->params[F_W]; k++)
                        {
                            in_tensor_pos[OUT_W] = out_tensor_pos[OUT_W]*layer->params[STRIDE]
                                + k*layer->params[DILATION] - layer->params[PADDING];
                            if (in_tensor_pos[BATCH] >= 0 && in_tensor_pos[BATCH] < ldata->nasm->batch_size && in_tensor_pos[OUT_C] >= 0 && in_tensor_pos[OUT_C] < layer->params[IN_C] &&
                                in_tensor_pos[OUT_H] >= 0 && in_tensor_pos[OUT_H] < layer->params[IN_H] && in_tensor_pos[OUT_W] >= 0 && in_tensor_pos[OUT_W] < layer->params[IN_W])
                            {
                                ninst_t *input_nist = get_ninst_from_tensor_pos(parent_ldata, in_tensor_pos);
                                int duplicate = 0;
                                // printf ("%d,%d,%d,%d,p:%d ", in_tensor_pos[BATCH], in_tensor_pos[OUT_C],
                                //     in_tensor_pos[OUT_H],
                                //     in_tensor_pos[OUT_W], input_nist->ninst_idx);
                                for (int l = 0; l < ninst->num_parent_ninsts; l++)
                                {
                                    if (parent_arr[l] == input_nist->ninst_idx)
                                    {
                                        duplicate = 1;
                                    }
                                }
                                if (!duplicate)
                                {
                                    parent_arr[ninst->num_parent_ninsts] = input_nist->ninst_idx;
                                    ninst->num_parent_ninsts++;  
                                    // printf ("\n\tnum parent:%d\n", ninst->num_parent_ninsts);
                                }
                            }
                        }
                    }
                }
            }
            else if (layer->type == RESIDUAL_LAYER)
            {
                in_tensor_pos[OUT_C] = out_tensor_pos[OUT_C];
                in_tensor_pos[OUT_H] = out_tensor_pos[OUT_H];
                in_tensor_pos[OUT_W] = out_tensor_pos[OUT_W];
                if (in_tensor_pos[BATCH] >= 0 && in_tensor_pos[BATCH] < ldata->nasm->batch_size && in_tensor_pos[OUT_C] >= 0 && in_tensor_pos[OUT_C] < layer->params[IN_C] &&
                    in_tensor_pos[OUT_H] >= 0 && in_tensor_pos[OUT_H] < layer->params[IN_H] && in_tensor_pos[OUT_W] >= 0 && in_tensor_pos[OUT_W] < layer->params[IN_W])
                {
                    ninst_t *input_nist = get_ninst_from_tensor_pos(parent_ldata, in_tensor_pos);
                    int duplicate = 0;
                    // printf ("%d,%d,%d,%d,p:%d ", in_tensor_pos[BATCH], in_tensor_pos[OUT_C],
                    //     in_tensor_pos[OUT_H],
                    //     in_tensor_pos[OUT_W], input_nist->ninst_idx);
                    for (int l = 0; l < ninst->num_parent_ninsts; l++)
                    {
                        if (parent_arr[l] == input_nist->ninst_idx)
                        {
                            duplicate = 1;
                        }
                    }
                    if (!duplicate)
                    {
                        parent_arr[ninst->num_parent_ninsts] = input_nist->ninst_idx;
                        ninst->num_parent_ninsts++;  
                        // printf ("\n\tnum parent:%d\n", ninst->num_parent_ninsts);
                    }
                    input_nist = get_ninst_from_tensor_pos(ldata->nasm->ldata_arr + ldata->parent_ldata_idx_arr[PARENT_1], in_tensor_pos);
                    for (int l = 0; l < ninst->num_parent_ninsts; l++)
                    {
                        if (parent_arr[l] == input_nist->ninst_idx)
                        {
                            duplicate = 1;
                        }
                    }
                    if (!duplicate)
                    {
                        parent_arr[ninst->num_parent_ninsts] = input_nist->ninst_idx;
                        ninst->num_parent_ninsts++;  
                        // printf ("\n\tnum parent:%d\n", ninst->num_parent_ninsts);
                    }
                }
            }
            else if (layer->type == FC_LAYER)
            {
                if (in_tensor_pos[BATCH] >= 0 && in_tensor_pos[BATCH] < ldata->nasm->batch_size)
                {
                    if (parent_ldata->layer->type == CONV_LAYER || parent_ldata->layer->type == MAXPOOL_LAYER || parent_ldata->layer->type == AVGPOOL_LAYER)
                    {
                        for (int i = 0; i < layer->params[IN_C]; i++)
                        {
                            in_tensor_pos[OUT_C] = i;
                            for (int j = 0; j < layer->params[IN_H]; j++)
                            {
                                in_tensor_pos[OUT_H] = j;
                                for (int k = 0; k < layer->params[IN_W]; k++)
                                {
                                    in_tensor_pos[OUT_W] = k;
                                    if (in_tensor_pos[BATCH] >= 0 && in_tensor_pos[BATCH] < ldata->nasm->batch_size && in_tensor_pos[OUT_C] >= 0 && in_tensor_pos[OUT_C] < layer->params[IN_C] &&
                                        in_tensor_pos[OUT_H] >= 0 && in_tensor_pos[OUT_H] < layer->params[IN_H] && in_tensor_pos[OUT_W] >= 0 && in_tensor_pos[OUT_W] < layer->params[IN_W])
                                    {
                                        ninst_t *input_nist = get_ninst_from_tensor_pos(parent_ldata, in_tensor_pos);
                                        int duplicate = 0;
                                        for (int l = 0; l < ninst->num_parent_ninsts; l++)
                                        {
                                            if (parent_arr[l] == input_nist->ninst_idx)
                                            {
                                                duplicate = 1;
                                                break;
                                            }
                                        }
                                        if (!duplicate)
                                        {
                                            parent_arr[ninst->num_parent_ninsts] = input_nist->ninst_idx;
                                            ninst->num_parent_ninsts++;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    else if (parent_ldata->layer->type == FC_LAYER)
                    {
                        for (int i = 0; i < layer->params[IN_C]; i++)
                        {
                            in_tensor_pos[OUT_C] = i;
                            if (in_tensor_pos[BATCH] >= 0 && in_tensor_pos[BATCH] < ldata->nasm->batch_size && in_tensor_pos[OUT_C] >= 0 && in_tensor_pos[OUT_C] < layer->params[IN_C] &&
                                in_tensor_pos[OUT_H] >= 0 && in_tensor_pos[OUT_H] < layer->params[IN_H] && in_tensor_pos[OUT_W] >= 0 && in_tensor_pos[OUT_W] < layer->params[IN_W])
                            {
                                ninst_t *input_nist = get_ninst_from_tensor_pos(parent_ldata, in_tensor_pos);
                                int duplicate = 0;
                                for (int l = 0; l < ninst->num_parent_ninsts; l++)
                                {
                                    if (parent_arr[l] == input_nist->ninst_idx)
                                    {
                                        duplicate = 1;
                                        break;
                                    }
                                }
                                if (!duplicate)
                                {
                                    parent_arr[ninst->num_parent_ninsts] = input_nist->ninst_idx;
                                    ninst->num_parent_ninsts++;
                                }
                            }
                        }
                    }
                    else
                    {
                        FPRT(stderr, "ERROR: Unsupported parent layer type %s, at line %d in file %s.\n" , layer_type_str[parent_ldata->layer->type], __LINE__, __FILE__);
                        exit(1);
                    }
                }
            } 
            else if (layer->type == SOFTMAX_LAYER)
            {
                memcpy (in_tensor_pos, out_tensor_pos, sizeof(int) * NUM_PARAM_ELEMENTS);
                if (in_tensor_pos[BATCH] >= 0 && in_tensor_pos[BATCH] < ldata->nasm->batch_size && in_tensor_pos[OUT_C] >= 0 && in_tensor_pos[OUT_C] < layer->params[IN_C] &&
                    in_tensor_pos[OUT_H] >= 0 && in_tensor_pos[OUT_H] < layer->params[IN_H] && in_tensor_pos[OUT_W] >= 0 && in_tensor_pos[OUT_W] < layer->params[IN_W])
                {
                    ninst_t *input_nist = get_ninst_from_tensor_pos(parent_ldata, in_tensor_pos);
                    int duplicate = 0;
                    for (int l = 0; l < ninst->num_parent_ninsts; l++)
                    {
                        if (parent_arr[l] == input_nist->ninst_idx)
                        {
                            duplicate = 1;
                            break;
                        }
                    }
                    if (!duplicate)
                    {
                        parent_arr[ninst->num_parent_ninsts] = input_nist->ninst_idx;
                        ninst->num_parent_ninsts++;
                    }
                }
            }
            else if (layer->type == INPUT_LAYER)
            {
                // printf ("\n");
                return;
            }
            else
            {
                FPRT(stderr, "ERROR: Unsupported layer type %s, at line %d in file %s\n" , layer_type_str[layer->type], __LINE__, __FILE__);
                exit(1);
            }
            // printf ("\n");
        }
    }
    // fflush (stdout);
    ninst->parent_ninst_idx_arr = calloc(ninst->num_parent_ninsts, sizeof(unsigned int));
    memcpy (ninst->parent_ninst_idx_arr, parent_arr, ninst->num_parent_ninsts*sizeof(unsigned int));
    for (int i = 0; i < ninst->num_parent_ninsts; i++)
    {
        ninst_t *parent = ninst->parent_ninst_idx_arr[i] + ldata->nasm->ninst_arr;
        parent->num_child_ninsts++;
    }
    free (parent_arr);
    // printf ("Layer %d, ninst %d, num_parent_ninsts %d\n", 
    //     ldata->layer->layer_idx, ninst->ninst_idx, ninst->num_parent_ninsts);
    // for (int i = 0; i < ninst->num_parent_ninsts; i++)
    // {
    //     ninst_t *parent = ninst->parent_ninst_idx_arr[i] + ldata->nasm->ninst_arr;
    //     printf ("\tparent_ninst %d, layer %d, ninst %d\n", i, parent->ldata->layer->layer_idx, parent->ninst_idx);
    // }
}

void init_ninst (nasm_ldata_t *ldata, ninst_t *ninst_ptr, int ninst_idx)
{
    ninst_ptr->ldata = ldata;
    ninst_ptr->state = NINST_NOT_READY;
    ninst_ptr->ninst_idx = ninst_idx;
    get_out_mat_pos_from_nist (ldata, ninst_ptr, ninst_ptr->out_mat_pos);
}

void destroy_ninst (ninst_t *ninst)
{
    if (ninst == NULL)
        return;
    if (ninst->parent_ninst_idx_arr != NULL)
        free (ninst->parent_ninst_idx_arr);
}

nasm_t *apu_create_nasm_without_finding_ninst_parents (aspen_dnn_t *dnn, unsigned int flop_per_ninst, unsigned int batch_size)
{
    nasm_t *new_nasm = (nasm_t *) calloc(1, sizeof(nasm_t));
    new_nasm->dnn = dnn;
    new_nasm->flop_per_ninst = flop_per_ninst > 0? flop_per_ninst : 1;
    new_nasm->batch_size = batch_size > 0? batch_size : 1;
    new_nasm->nasm_id = nasm_num;
    nasm_num++;
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
    unsigned int total_ninst = 0;
    for (int i = 0; i < new_nasm->num_ldata; i++)
    {
        total_ninst += new_nasm->ldata_arr[i].num_ninst;
    }
    new_nasm->ninst_arr = calloc(total_ninst, sizeof(ninst_t));
    ninst_t *ninst_ptr = new_nasm->ninst_arr;
    total_ninst = 0;
    for (int i = 0; i < new_nasm->num_ldata; i++)
    {
        new_nasm->ldata_arr[i].ninst_arr_start = ninst_ptr;
        ninst_ptr += new_nasm->ldata_arr[i].num_ninst;
        for (int j = 0; j < new_nasm->ldata_arr[i].num_ninst; j++)
        {
            init_ninst(&new_nasm->ldata_arr[i], &new_nasm->ldata_arr[i].ninst_arr_start[j], total_ninst + j);
        }
        total_ninst += new_nasm->ldata_arr[i].num_ninst;
    }
    for (int i = 0; i < new_nasm->num_ldata; i++)
    {
        update_ldata_child_list(&new_nasm->ldata_arr[i]);
    }
    dnn->ref_nasms++;
    return new_nasm;
}

void set_child_list (ninst_t *ninst)
{
    if (ninst == NULL)
        return;
    if (ninst->num_child_ninsts <= 0)
        return;
    ninst->child_ninst_arr = calloc(ninst->num_child_ninsts, sizeof(ninst_t*));
    nasm_ldata_t *ldata = ninst->ldata;
    nasm_t *nasm = ldata->nasm;
    unsigned int child_idx = 0;
    for (int i = 0; i < nasm->num_ldata; i++)
    {
        for (int j = 0; j < nasm->ldata_arr[i].num_ninst; j++)
        {
            ninst_t *target_ninst = &nasm->ldata_arr[i].ninst_arr_start[j];
            for (int k = 0; k < target_ninst->num_parent_ninsts; k++)
            {
                if (target_ninst->parent_ninst_idx_arr[k] == ninst->ninst_idx)
                {
                    ninst->child_ninst_arr[child_idx] = target_ninst;
                    child_idx++;
                    if (child_idx == ninst->num_child_ninsts)
                    {
                        return;
                    }
                    break;
                }
            }
        }
    }
    FPRT (stderr, "Error: set_child_list failed. Only found %d children for ninst %d, expected %d\n"
        , child_idx, ninst->ninst_idx, ninst->num_child_ninsts);
}

nasm_t *apu_create_nasm(aspen_dnn_t *dnn, unsigned int flop_per_ninst, unsigned int batch_size)
{
    nasm_t *new_nasm = apu_create_nasm_without_finding_ninst_parents(dnn, flop_per_ninst, batch_size);
    for (int i = 0; i < new_nasm->num_ldata; i++)
    {
        #pragma omp parallel for
        for (int j = 0; j < new_nasm->ldata_arr[i].num_ninst; j++)
        {
            ninst_find_parent (&new_nasm->ldata_arr[i].ninst_arr_start[j]);
        }
        PRT ("Layer %d, parents for %d ninsts found.\n", i, new_nasm->ldata_arr[i].num_ninst);
    }
    for (int i = 0; i < new_nasm->num_ldata; i++)
    {
        for (int j = 0; j < new_nasm->ldata_arr[i].num_ninst; j++)
        {
            set_child_list (&new_nasm->ldata_arr[i].ninst_arr_start[j]);
        }
    }
    return new_nasm;
}

void destroy_nasm_ldata (nasm_ldata_t *ldata)
{
    if (ldata == NULL)
        return;
    if (ldata->child_ldata_idx_arr != NULL)
        free(ldata->child_ldata_idx_arr);
    for (int i = 0; i < ldata->num_ninst; i++)
    {
        destroy_ninst(&ldata->ninst_arr_start[i]);
    }
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

void apu_destroy_nasm (nasm_t *nasm)
{
    if (nasm == NULL)
        return;
    destroy_nasm_ldata_arr(nasm->ldata_arr, nasm->num_ldata);
    if (nasm->ninst_arr != NULL)
        free(nasm->ninst_arr);
    nasm->dnn->ref_nasms--;
    free(nasm);
}

void get_out_mat_info (nasm_ldata_t *ldata)
{
    aspen_layer_t *layer = ldata->layer;
    if (layer->type == CONV_LAYER)
    {
        ldata->flop_per_output = 2*layer->params[F_H]*layer->params[F_W]*layer->params[IN_C];
        ldata->out_mat_dims[OUT_H] = layer->params[OUT_C];
        ldata->out_mat_dims[OUT_W] = layer->params[OUT_H]*layer->params[OUT_W]*ldata->nasm->batch_size;
    }
    if (layer->type == FC_LAYER)
    {
        ldata->flop_per_output = 2*layer->params[IN_C];
        ldata->out_mat_dims[OUT_H] = layer->params[OUT_C];
        ldata->out_mat_dims[OUT_W] = ldata->nasm->batch_size;
    }
    if (layer->type == MAXPOOL_LAYER)
    {
        ldata->flop_per_output = layer->params[F_H]*layer->params[F_W];
        ldata->out_mat_dims[OUT_H] = layer->params[OUT_C];
        ldata->out_mat_dims[OUT_W] = layer->params[OUT_H]*layer->params[OUT_W]*ldata->nasm->batch_size;
    }
    if (layer->type == AVGPOOL_LAYER)
    {
        ldata->flop_per_output = layer->params[F_H]*layer->params[F_W];
        ldata->out_mat_dims[OUT_H] = layer->params[OUT_C];
        ldata->out_mat_dims[OUT_W] = layer->params[OUT_H]*layer->params[OUT_W]*ldata->nasm->batch_size;
    }
    if (layer->type == INPUT_LAYER || layer->type == RESIDUAL_LAYER)
    {
        ldata->flop_per_output = 1;
        ldata->out_mat_dims[OUT_H] = layer->params[OUT_C];
        ldata->out_mat_dims[OUT_W] = layer->params[OUT_H]*layer->params[OUT_W]*ldata->nasm->batch_size;
    }
    if (layer->type == SOFTMAX_LAYER)
    {
        ldata->flop_per_output = 1;
        ldata->out_mat_dims[OUT_H] = layer->params[OUT_C];
        ldata->out_mat_dims[OUT_W] = ldata->nasm->batch_size;
    }
    else
    {
        FPRT(stderr, "ERROR) Unsupported layer type %s, at line %d in file %s\n" , layer_type_str[layer->type], __LINE__, __FILE__);
        exit(1);
    }
}

void get_ninst_tile_dims (nasm_ldata_t *ldata)
{
    ldata->ninst_tile_dims[OUT_H] = 1;
    ldata->ninst_tile_dims[OUT_W] = 1;
    while (ldata->ninst_tile_dims[OUT_H]*ldata->ninst_tile_dims[OUT_W] < ldata->nasm->flop_per_ninst/ldata->flop_per_output)
    {
        if (ldata->ninst_tile_dims[OUT_H] < ldata->out_mat_dims[OUT_H])
        {
            ldata->ninst_tile_dims[OUT_H]++;
        }
        else if (ldata->ninst_tile_dims[OUT_W] < ldata->out_mat_dims[OUT_W])
        {
            ldata->ninst_tile_dims[OUT_W]++;
        }
        else
        {
            break;
        }
    }
    while (ldata->ninst_tile_dims[OUT_H]%NINST_H_MIN != 0)
    {
        ldata->ninst_tile_dims[OUT_H]++;
    }
    if (ldata->layer->type != FC_LAYER && ldata->layer->type != SOFTMAX_LAYER)
    {
        while (ldata->ninst_tile_dims[OUT_W]%NINST_W_MIN != 0)
        {
            ldata->ninst_tile_dims[OUT_W]++;
        }
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
    if (layer->type != FC_LAYER && layer->type != SOFTMAX_LAYER)
    {
        while ((out_w/ldata_ptr->ninst_tile_dims[OUT_W])*(out_h/ldata_ptr->ninst_tile_dims[OUT_H]) < 8)
        {
            if (ldata_ptr->ninst_tile_dims[OUT_W] > NINST_W_MIN)
            {
                ldata_ptr->ninst_tile_dims[OUT_W] /= 2;
                while (ldata_ptr->ninst_tile_dims[OUT_W]%NINST_W_MIN != 0)
                {
                    ldata_ptr->ninst_tile_dims[OUT_W]++;
                }
            }
            else if (ldata_ptr->ninst_tile_dims[OUT_H] > NINST_H_MIN)
            {
                ldata_ptr->ninst_tile_dims[OUT_H] /= 2;
                while (ldata_ptr->ninst_tile_dims[OUT_H]%NINST_H_MIN != 0)
                {
                    ldata_ptr->ninst_tile_dims[OUT_H]++;
                }
            }
            out_w = get_smallest_dividable (ldata_ptr->out_mat_dims[OUT_W], ldata_ptr->ninst_tile_dims[OUT_W]);
            out_h = get_smallest_dividable (ldata_ptr->out_mat_dims[OUT_H], ldata_ptr->ninst_tile_dims[OUT_H]);
            if (ldata_ptr->ninst_tile_dims[OUT_W] == NINST_W_MIN && ldata_ptr->ninst_tile_dims[OUT_H] == NINST_H_MIN)
            {
                break;
            }
        }
    }
    ldata_ptr->out_mat_stride = out_h;
    ldata_ptr->out_mat_size = (size_t)out_h*out_w;
    ldata_ptr->num_ninst = (out_h/ldata_ptr->ninst_tile_dims[OUT_H])*(out_w/ldata_ptr->ninst_tile_dims[OUT_W]);
}

void destroy_aspen_tensor(aspen_tensor_t *tensor)
{
    if (tensor == NULL)
        return;
    if (tensor->data != NULL)
        aspen_free(tensor->data);
    free(tensor);
}
ninst_t *get_ninst_from_tensor_pos (nasm_ldata_t *ldata, unsigned int *tensor_pos)
{
    unsigned int out_mat_pos[2] = {0,0};
    get_out_mat_pos_from_tensor_pos (ldata, tensor_pos, out_mat_pos);
    return get_ninst_from_out_mat_pos (ldata, out_mat_pos[OUT_H], out_mat_pos[OUT_W]);
}
ninst_t *get_ninst_from_out_mat_pos (nasm_ldata_t *ldata, unsigned int h, unsigned int w)
{
    unsigned int out_h = get_smallest_dividable (ldata->out_mat_dims[OUT_H], ldata->ninst_tile_dims[OUT_H]);
    unsigned int ninst_idx = (w/ldata->ninst_tile_dims[OUT_W])*(out_h/ldata->ninst_tile_dims[OUT_H]) 
        + (h/ldata->ninst_tile_dims[OUT_H]);
    assert (ninst_idx >= 0);
    assert (ninst_idx < ldata->num_ninst);
    return ldata->ninst_arr_start + ninst_idx;
}
void get_out_mat_pos_from_nist (nasm_ldata_t *ldata, ninst_t *ninst, unsigned int *out_mat_pos)
{
    unsigned int out_h = get_smallest_dividable (ldata->out_mat_dims[OUT_H], ldata->ninst_tile_dims[OUT_H]);
    unsigned int ninst_idx = ninst - ldata->ninst_arr_start;
    out_mat_pos[OUT_H] = (ninst_idx%(out_h/ldata->ninst_tile_dims[OUT_H]))*ldata->ninst_tile_dims[OUT_H];
    out_mat_pos[OUT_W] = (ninst_idx/(out_h/ldata->ninst_tile_dims[OUT_H]))*ldata->ninst_tile_dims[OUT_W];
}
void get_out_mat_pos_from_tensor_pos (nasm_ldata_t *ldata, unsigned int *tensor_pos, unsigned int *out_mat_pos)
{
    aspen_layer_t *layer = ldata->layer;
    if (layer->type == CONV_LAYER || layer->type == MAXPOOL_LAYER || layer->type == AVGPOOL_LAYER || layer->type == INPUT_LAYER
        || layer->type == RESIDUAL_LAYER)
    {
        out_mat_pos[OUT_H] = tensor_pos[OUT_C];
        out_mat_pos[OUT_W] = tensor_pos[BATCH] * layer->params[OUT_H] * layer->params[OUT_W] + 
            tensor_pos[OUT_H] * layer->params[OUT_W] + tensor_pos[OUT_W];
        return;
    }
    else if (layer->type == FC_LAYER || layer->type == SOFTMAX_LAYER)
    {
        out_mat_pos[OUT_H] = tensor_pos[OUT_C];
        out_mat_pos[OUT_W] = tensor_pos[BATCH];
        return;
    }
    else
    {
        FPRT(stderr, "ERROR: Unsupported layer type %s, at line %d in file %s\n" , layer_type_str[layer->type], __LINE__, __FILE__);
        exit(1);
    }
}
void get_tensor_pos_from_out_mat_pos (nasm_ldata_t *ldata, unsigned int *out_mat_pos, unsigned int *tensor_pos)
{
    aspen_layer_t *layer = ldata->layer;
    if (layer->type == CONV_LAYER || layer->type == MAXPOOL_LAYER || layer->type == AVGPOOL_LAYER || layer->type == INPUT_LAYER
        || layer->type == RESIDUAL_LAYER)
    {
        tensor_pos[BATCH] = out_mat_pos[OUT_W] / (layer->params[OUT_H] * layer->params[OUT_W]); 
        tensor_pos[OUT_C] = out_mat_pos[OUT_H];
        tensor_pos[OUT_H] = (out_mat_pos[OUT_W] % (layer->params[OUT_H] * layer->params[OUT_W])) / layer->params[OUT_W];
        tensor_pos[OUT_W] = out_mat_pos[OUT_W] % layer->params[OUT_W];
        return;
    }
    else if (layer->type == FC_LAYER || layer->type == SOFTMAX_LAYER)
    {
        tensor_pos[BATCH] = out_mat_pos[OUT_W];
        tensor_pos[OUT_C] = out_mat_pos[OUT_H];
        return;
    }
    else
    {
        FPRT(stderr, "ERROR: Unsupported layer type %s, at line %d in file %s\n" , layer_type_str[layer->type], __LINE__, __FILE__);
        exit(1);
    }
}
void get_tensor_pos_from_nist (nasm_ldata_t *ldata, ninst_t *ninst, unsigned int *tensor_pos)
{
    unsigned int out_mat_pos[2] = {0,0};
    get_out_mat_pos_from_nist (ldata, ninst, out_mat_pos);
    get_tensor_pos_from_out_mat_pos (ldata, out_mat_pos, tensor_pos);
}

void print_dnn_info (aspen_dnn_t *dnn, int print_data)
{
    if (dnn == NULL)
    {
        printf("Error: DNN is NULL.\n");
        return;
    }
    printf("//////// Printing DNN Info ////////\n");
    printf("DNN Name: %s\n", dnn->name);
    printf("Number of Layers: %d\n", dnn->num_layers);
    for (int i = 0; i < dnn->num_layers; i++)
    {
        print_layer_info(&dnn->layers[i], print_data);
    }
    printf("//////// End of DNN Info ////////\n");
}

void print_layer_info (aspen_layer_t *layer, int print_data)
{
    if (layer == NULL)
    {
        printf("Error: Layer is NULL.\n");
        return;
    }
    printf("//////// Printing Layer Info ////////\n");
    printf("Layer Index: %d\n", layer->layer_idx);
    printf("Layer Type: %s\n", layer_type_str[layer->type]);
    printf("Layer Activation: %s\n", activation_type_str[layer->activation]);
    printf("Layer Parents: ");
    for (int i = 0; i < NUM_PARENT_ELEMENTS; i++)
    {
        if (layer->parent_layers[i] != NULL)
            printf("%s: %d ", parent_type_str[i], layer->parent_layers[i]->layer_idx);
    }
    printf("\nLayer Params:\n");
    for (LAYER_PARAMS i = 0; i < NUM_PARAM_ELEMENTS; i++)
    {
        if (layer->params[i] != 0)
            printf("\t%s: %d\n", param_type_str[i], layer->params[i]);
    }
    printf ("Layer Tensors:\n");
    for (int i = 0; i < NUM_TENSOR_ELEMENTS; i++)
    {
        if (layer->tensors[i] != NULL)
        {
            printf("\t%s\n", tensor_type_str[i]);
            print_tensor_info(layer->tensors[i], print_data);
        }
    }
}

void print_tensor_info (aspen_tensor_t *tensor, int print_data)
{
    if (tensor == NULL)
    {
        printf("Error: Tensor is NULL.\n");
        return;
    }
    printf("\t\tDims: ");
    for (int i = 0; i < tensor->num_dims; i++)
    {
        printf("%s, ", param_type_str[tensor->data_dim_order[i]]);
    }
    printf("\n\t\tSize: ");
    for (int i = 0; i < tensor->num_dims; i++)
    {
        printf("%d, ", tensor->dims[tensor->data_dim_order[i]]);
    }
    printf("\n");
    if (print_data)
    {
        int new_line_num = 0;
        int dims_mult_arr[MAX_TENSOR_DIMS];
        for (int i = 0; i < MAX_TENSOR_DIMS; i++)
        {
            dims_mult_arr[i] = 1;
        }
        for (int i = tensor->num_dims - 1; i >= 0; i--)
        {
            for (int j = i; j < tensor->num_dims; j++)
            {
                dims_mult_arr[i] *= tensor->dims[tensor->data_dim_order[j]];
            }
            if (dims_mult_arr[i] < 20 || new_line_num == 0)
                new_line_num = dims_mult_arr[i];
        }
        printf("\t\tData: ");
        if (tensor->data == NULL)
        {
            printf("Data is NULL.\n");
        }
        else 
        {
            for (int i = 0; i < tensor->num_elements; i++)
            {
                if (i % new_line_num == 0)
                {
                    // printf("\n%d:", i);
                    printf("\n");
                    for (int j = 0; j < tensor->num_dims; j++)
                    {
                        printf("%d,", (i/dims_mult_arr[j+1]) % tensor->dims[tensor->data_dim_order[j]]);
                    }
                    printf(": ");
                }
                printf("%3.2f ", *((float*)tensor->data + i));
            }
            printf("\n");
        }
    }
}

void print_nasm_info (nasm_t *nasm, int print_data)
{
    if (nasm == NULL)
    {
        printf("Error: NASM is NULL.\n");
        return;
    }
    printf("//////////////////////// Printing NASM Info ////////////////////////\n");
    printf("Original DNN name: %s\n", nasm->dnn->name);
    printf("Nasm ID: %d\n", nasm->nasm_id);
    printf("Number of ldata: %d\n", nasm->num_ldata);
    printf("Number of batch: %d\n", nasm->batch_size);
    printf("FLOPs per ninst: %d\n", nasm->flop_per_ninst);
    for (int i = 0; i < nasm->num_ldata; i++)
    {
        print_ldata_info(&nasm->ldata_arr[i], print_data);
    }
    printf("//////////////////////// End of NASM Info ////////////////////////\n");
}

void print_ldata_info (nasm_ldata_t *ldata, int print_data)
{
    if (ldata == NULL)
    {
        printf("Error: ldata is NULL.\n");
        return;
    }
    printf("//////////////////////// Printing ldata Info ////////////////////////\n");
    printf("Ldata Index: %ld\n", ldata - ldata->nasm->ldata_arr);
    printf("Original layer index: %d\n", ldata->layer->layer_idx);
    printf("Original layer type: %s, Params: \n\t", layer_type_str[ldata->layer->type]);
    for (LAYER_PARAMS i = 0; i < NUM_PARAM_ELEMENTS; i++)
    {
        if (i != NUM_PARAM_ELEMENTS && ldata->layer->params[i] != 0)
            printf("%s:%d ", param_type_str[i], ldata->layer->params[i]);
    }
    printf("\n");
    printf("Ldata Parents: ");
    for (int i = 0; i < NUM_PARENT_ELEMENTS; i++)
    {
        if (ldata->parent_ldata_idx_arr[i] != -1)
            printf("%s: %d ", parent_type_str[i], ldata->parent_ldata_idx_arr[i]);
    }
    printf("\n");
    if (ldata->parent_ldata_idx_arr[PARENT_0] != -1)
    {
        aspen_layer_t *p0_layer = ldata->nasm->ldata_arr[ldata->parent_ldata_idx_arr[PARENT_0]].layer;
        printf("Parent 0 idx: %d, type: %s, Params: \n\t", p0_layer->layer_idx, layer_type_str[p0_layer->type]);
        for (LAYER_PARAMS i = 0; i < NUM_PARAM_ELEMENTS; i++)
        {
            if (i != NUM_PARAM_ELEMENTS && p0_layer->params[i] != 0)
                printf("%s:%d ", param_type_str[i], p0_layer->params[i]);
        }
        printf ("\n");
    }
    printf("Ldata Children (Completed: %d/%d): ", ldata->num_child_ldata_completed, ldata->num_child_ldata);
    for (int i = 0; i < ldata->num_child_ldata; i++)
    {
        printf("%d ", ldata->child_ldata_idx_arr[i]);
    }
    printf("\n");
    printf("Ldata Flop per output element: %d\n", ldata->flop_per_output);
    printf("Ldata Output Matrix Dimensions: (H: %d, W: %d), Stride: %d\n"
        , ldata->out_mat_dims[OUT_H], ldata->out_mat_dims[OUT_W], ldata->out_mat_stride);
    printf("Ldata Flop per Ninst: %d\n", ldata->flop_per_output*ldata->ninst_tile_dims[OUT_H]*ldata->ninst_tile_dims[OUT_W]);
    printf("Ldata Ninst Tile Dimensions: (H: %d, W: %d)\n", 
        ldata->ninst_tile_dims[OUT_H], ldata->ninst_tile_dims[OUT_W]);
    printf("Number of ninst: %d, Completed: %d\n", ldata->num_ninst, ldata->num_ninst_completed);
    for (int i = 0; i < ldata->num_ninst; i++)
    {
        printf ("\tNinst %d: ", i);
        print_ninst_info(&ldata->ninst_arr_start[i], print_data);
    }
    printf("////////////////////////  End of ldata Info  ////////////////////////\n");
}

void print_ninst_info (ninst_t *ninst, int print_data)
{
    if (ninst == NULL)
    {
        printf("Error: ninst is NULL.\n");
        return;
    }
    printf ("Ninst Idx: %d\n", ninst->ninst_idx);
    printf ("\t\tNinst tile position: (H: %d, W: %d) ~ (H: %d, W: %d) "
        , ninst->out_mat_pos[OUT_H], ninst->out_mat_pos[OUT_W],
            ninst->out_mat_pos[OUT_H] + ninst->ldata->ninst_tile_dims[OUT_H] - 1
                , ninst->out_mat_pos[OUT_W] + ninst->ldata->ninst_tile_dims[OUT_W] - 1);
    LAYER_TYPE layer_type = ninst->ldata->layer->type;
    if (layer_type == CONV_LAYER || layer_type == MAXPOOL_LAYER || layer_type == AVGPOOL_LAYER || layer_type == INPUT_LAYER 
        || layer_type == RESIDUAL_LAYER)
    {
        unsigned int out_tensor_pos[NUM_PARAM_ELEMENTS]; 
        get_tensor_pos_from_nist (ninst->ldata, ninst, out_tensor_pos);
        printf ("Tensor Pos: (%d,%d,%d,%d)", out_tensor_pos[BATCH], out_tensor_pos[OUT_C],
                    out_tensor_pos[OUT_H],
                     out_tensor_pos[OUT_W]);
    }
    printf ("\n\t\tParent ninst (Completed: %d/%d): "
        , ninst->num_parent_ninsts_completed, ninst->num_parent_ninsts);
    for (int i = 0; i < ninst->num_parent_ninsts; i++)
    {
        if (ninst->parent_ninst_idx_arr == NULL)
        {
            printf("\n\t\t\tError: Parent ninst index array is NULL.\n");
            break;  
        }
        ninst_t *parent_ninst = ninst->parent_ninst_idx_arr[i] + ninst->ldata->nasm->ninst_arr;
        printf("L%ld:%d ", parent_ninst->ldata - parent_ninst->ldata->nasm->ldata_arr,
            parent_ninst->ninst_idx);
    }
    printf("\n\t\tChild ninst (%d): ", ninst->num_child_ninsts);
    for (int i = 0; i < ninst->num_child_ninsts; i++)
    {
        if (ninst->child_ninst_arr == NULL)
        {
            printf("\n\t\t\tError: Child ninst array is NULL.\n");
            break;  
        }
        ninst_t *child_ninst = ninst->child_ninst_arr[i];
        printf("L%ld:%d ", child_ninst->ldata - child_ninst->ldata->nasm->ldata_arr,
            child_ninst->ninst_idx);
    }
    if (print_data)
    {
        printf("\n\t\tData:");
        if (ninst->out_mat == NULL)
        {
            printf("\n\t\t\tError: Output matrix is NULL.\n");
            return;
        }
        for (unsigned int h = 0; h < ninst->ldata->ninst_tile_dims[OUT_H]; h++)
        {
            printf("\n\t\t\t");
            for (unsigned int w = 0; w < ninst->ldata->ninst_tile_dims[OUT_W]; w++)
            {
                unsigned int output_mat_h = ninst->out_mat_pos[OUT_H] + h;
                unsigned int output_mat_w = ninst->out_mat_pos[OUT_W] + w;
                printf("%3.2f ", *((float*)ninst->out_mat 
                    + output_mat_w*ninst->ldata->out_mat_stride + output_mat_h));
            }
        }
    }
    printf("\n");
}