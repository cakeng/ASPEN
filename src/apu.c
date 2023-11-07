#include "aspen.h"
#include "apu.h"
#include "nasm.h"
#include "input_parser.h"

aspen_dnn_t *apu_create_dnn (char *input_path, char *weight_path)
{
    aspen_dnn_t *new_dnn = parse_input (input_path);
    if (weight_path != NULL)
        apu_load_dnn_data_from_file (new_dnn, weight_path);
    for (int i = 0; i < new_dnn->num_layers; i++)
    {
        aspen_layer_t *layer = new_dnn->layers + i;
        if (layer->type == CONV_LAYER)
        {
            LAYER_PARAMS weight_dim_order[] = {OUT_C, WEIGHT_H, WEIGHT_W, IN_C, SUB_C};
            unsigned int params[NUM_PARAM_ELEMENTS] = {0};
            memcpy (params, layer->params, sizeof(unsigned int) * NUM_PARAM_ELEMENTS);
            params[SUB_C] = _VEC_SIZE_M;
            params[OUT_C] = (layer->params[OUT_C] + params[SUB_C] - 1) / params[SUB_C];
            reorder_aspen_tensor (&layer->tensors[WEIGHT_TENSOR], params, weight_dim_order, 5);
        }
        else if (layer->type == FC_LAYER)
        {
            LAYER_PARAMS weight_dim_order[] = {OUT_C, IN_H, IN_W, IN_C, SUB_C};
            unsigned int params[NUM_PARAM_ELEMENTS] = {0};
            memcpy (params, layer->params, sizeof(unsigned int) * NUM_PARAM_ELEMENTS);
            params[SUB_C] = _VEC_SIZE_M;
            params[OUT_C] = (layer->params[OUT_C] + params[SUB_C] - 1) / params[SUB_C];
            reorder_aspen_tensor (&layer->tensors[WEIGHT_TENSOR], params, weight_dim_order, 5);
        }
        else if (layer->type == MATMUL_LAYER)
        {
            LAYER_PARAMS weight_dim_order[] = {MAT_M, MAT_K, SUB_M};
            unsigned int params[NUM_PARAM_ELEMENTS] = {0};
            memcpy (params, layer->params, sizeof(unsigned int) * NUM_PARAM_ELEMENTS);
            params[SUB_M] = _VEC_SIZE_M;
            params[MAT_M] = (layer->params[MAT_M] + params[SUB_M] - 1) / params[SUB_M];
            reorder_aspen_tensor (&layer->tensors[WEIGHT_TENSOR], params, weight_dim_order, 3);
        }
    }
    return new_dnn;
}

void apu_destroy_dnn (aspen_dnn_t *dnn)
{
    if (dnn == NULL)
        return;
    if (dnn->ref_nasms != 0)
    {
        ERROR_PRTF ("Cannot destroy dnn %s with %d nasms still referencing it."
            , dnn->name, dnn->ref_nasms);
        return;
    }
    destroy_aspen_layers(dnn->layers, dnn->num_layers);
    free(dnn);
}

aspen_dnn_t *aspen_dnn_init (unsigned int num_layers, char* name)
{
    aspen_dnn_t *new_dnn = (aspen_dnn_t *) calloc(1, sizeof(aspen_dnn_t));
    strncpy(new_dnn->name, name, MAX_STRING_LEN-1);
    new_dnn->element_size = sizeof(float);
    new_dnn->num_layers = num_layers;
    new_dnn->layers = (aspen_layer_t *) calloc(num_layers, sizeof(aspen_layer_t));
    for (int i = 0; i < num_layers; i++)
    {
        aspen_layer_init(new_dnn->layers + i, i, new_dnn);
    }
    #ifdef AVX2
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    #endif
    return new_dnn;
}

void aspen_layer_init (aspen_layer_t *layer, unsigned int layer_idx, aspen_dnn_t *dnn)
{
    layer->layer_idx = layer_idx;
    layer->dnn = dnn;
}

void destroy_aspen_layer (aspen_layer_t* layer)
{
    if (layer == NULL)
        return;
    for (int i = 0; i < NUM_TENSORS; i++)
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

aspen_tensor_t *aspen_tensor_init (unsigned int *params_arr, LAYER_PARAMS *order, int num_dims, unsigned int element_size)
{
    aspen_tensor_t *new_tensor = (aspen_tensor_t *) calloc(1, sizeof(aspen_tensor_t));
    new_tensor->num_dims = num_dims;
    new_tensor->num_elements = 1;
    new_tensor->element_size = element_size;
    int idx = 0;
    for (int i = 0; i < num_dims; i++)
    {
        if (params_arr[order[i]] <= 0)
            continue;
        new_tensor->data_dim_order[idx] = order[idx];
        new_tensor->dims[order[idx]] = params_arr[order[idx]];
        new_tensor->num_elements *= new_tensor->dims[order[idx]];
        idx++;
    }
    return new_tensor;
}

void calloc_aspen_tensor (aspen_tensor_t *tensor)
{
    if (tensor == NULL)
        return;
    if (tensor->data != NULL)
        aspen_free(tensor->data);
    if (tensor->num_elements == 0 || tensor->element_size == 0)
    {
        ERROR_PRTF ("Cannot calloc tensor with 0 elements or 0 element size.");
        assert (0);
    }
    size_t size = 1;
    for (int i = 0; i < tensor->num_dims; i++)
    {
        if (i == tensor->num_dims - 1)
            size *= get_smallest_dividable (tensor->dims[tensor->data_dim_order[i]], _VEC_SIZE_M);
        else
            size *= tensor->dims[tensor->data_dim_order[i]];
    }
    tensor->data = aspen_calloc(size, tensor->element_size);
}

void copy_buffer_to_aspen_tensor  (aspen_tensor_t *tensor, void *buffer)
{
    if (tensor == NULL || tensor->data == NULL)
        return;
    memcpy (tensor->data, buffer, tensor->num_elements * tensor->element_size);
}

void copy_aspen_tensor_to_buffer  (aspen_tensor_t *tensor, void *buffer)
{
    if (tensor == NULL || tensor->data == NULL)
        return;
    memcpy (buffer, tensor->data, tensor->num_elements * tensor->element_size);
}

void copy_aspen_tensor_to_tensor  (aspen_tensor_t *dst, aspen_tensor_t *src)
{
    if (dst == NULL || dst->data == NULL)
        return;
    if (src == NULL || src->data == NULL)
        return;
    if (dst->element_size != src->element_size)
    {
        ERROR_PRTF ("Error: cannot copy tensor with different element sizes.");
        return;
    }
    if (dst->num_elements != src->num_elements)
    {
        ERROR_PRTF ("Error: cannot copy tensor with different number of elements.");
        return;
    }
    memcpy (dst->data, src->data, dst->num_elements * dst->element_size);
}

void reorder_aspen_tensor (aspen_tensor_t **tensor_ptr, unsigned int *params_arr, LAYER_PARAMS *order, int num_dims)
{
    aspen_tensor_t *tensor = *tensor_ptr;
    aspen_tensor_t *new_tensor = aspen_tensor_init (params_arr, order, num_dims, tensor->element_size);
    if (new_tensor->num_elements < tensor->num_elements)
    {
        ERROR_PRTF ("Error: cannot reorder tensor into smaller number of elements.");
        assert (0);
    }
    calloc_aspen_tensor (new_tensor);
    unsigned int pos[NUM_PARAM_ELEMENTS] = {0};
    for (int idx = 0; idx < tensor->num_elements; idx++)
    {
        get_tensor_pos_from_idx (tensor, idx, pos);
        if (params_arr [SUB_C] > 0)
        {
            pos[SUB_C] = pos[OUT_C] % params_arr [SUB_C];
            pos[OUT_C] = pos[OUT_C] / params_arr [SUB_C];
        }
        else if (params_arr [SUB_M] > 0)
        {
            pos[SUB_M] = pos[MAT_M] % params_arr [SUB_M];
            pos[MAT_M] = pos[MAT_M] / params_arr [SUB_M];
        }
        void *src = (char*) tensor->data + idx * tensor->element_size;
        void *dst = get_aspen_tensor_element_ptr (new_tensor, pos);
        memcpy (dst, src, tensor->element_size);
    }
    *tensor_ptr = new_tensor;
    destroy_aspen_tensor (tensor);
}

void *get_aspen_tensor_data (aspen_tensor_t *tensor, LAYER_PARAMS *output_order)
{
    void *output = calloc (tensor->num_elements, tensor->element_size);
    aspen_tensor_t *new_tensor = aspen_tensor_init (tensor->dims, output_order, tensor->num_dims, tensor->element_size);
    unsigned int pos[NUM_PARAM_ELEMENTS] = {0};
    for (int idx = 0; idx < tensor->num_elements; idx++)
    {
        get_tensor_pos_from_idx (tensor, idx, pos);
        void *src = (char*) tensor->data + idx * tensor->element_size;
        void *dst = (char*) output + get_tensor_idx_from_pos (new_tensor, pos) * tensor->element_size;
        memcpy (dst, src, tensor->element_size);
    }

    destroy_aspen_tensor (new_tensor);
    return output;
}
// Change to add a new layer type
ssize_t get_ldata_output (void **out_ptr, nasm_ldata_t *ldata, LAYER_PARAMS *order)
{
    if (ldata == NULL)
    {
        ERROR_PRTF ("Error in get_ldata_output: ldata is NULL.");
        return -1;
    }
    if (out_ptr == NULL)
    {
        ERROR_PRTF ("Error in get_ldata_output: out_ptr is NULL.");
        return -1;
    }
    size_t elem_size = ldata->layer->dnn->element_size;
    size_t data_size = ldata->out_mat_dims[OUT_H] * ldata->out_mat_dims[OUT_W] * elem_size;
    void *tmp_data = get_packed_ldata_output_colwise (ldata);
    void *packed_data = aspen_calloc (ldata->out_mat_dims[OUT_H] * ldata->out_mat_dims[OUT_W], elem_size);
    memcpy (packed_data, tmp_data, data_size);
    free (tmp_data);
    void *output = NULL;
    aspen_layer_t *layer = ldata->layer;
    aspen_tensor_t *tensor = NULL;
    if ((layer->type == CONV_LAYER || layer->type == INPUT_LAYER || layer->type == MAXPOOL_LAYER || layer->type == AVGPOOL_LAYER 
        || layer->type == RESIDUAL_LAYER || layer->type == APPEND_LAYER || layer->type == YOLO_LAYER) && (layer->params[MAT_M] == 0))
    {
        LAYER_PARAMS org_order[] = {BATCH, OUT_H, OUT_W, OUT_C};
        unsigned int params[NUM_PARAM_ELEMENTS];
        memcpy (params, layer->params, NUM_PARAM_ELEMENTS * sizeof (unsigned int));
        params[BATCH] = ldata->nasm->batch_size;
        tensor = aspen_tensor_init (params, org_order, 4, layer->dnn->element_size);
    }
    else if (layer->type == FC_LAYER || layer->type == SOFTMAX_LAYER)
    {
        LAYER_PARAMS org_order[] = {BATCH, OUT_C};
        unsigned int params[NUM_PARAM_ELEMENTS];
        memcpy (params, layer->params, NUM_PARAM_ELEMENTS * sizeof (unsigned int));
        params[BATCH] = ldata->nasm->batch_size;
        tensor = aspen_tensor_init (params, org_order, 2, layer->dnn->element_size);
    }
    else if (layer->type == MATMUL_LAYER || layer->type == LAYERNORM_LAYER || layer->type == RESIDUAL_LAYER ||
        layer->type == INPUT_LAYER || layer->type == V_ATTENTION_LAYER)
    {
        LAYER_PARAMS org_order[] = {BATCH, MAT_N, MAT_M};
        unsigned int params[NUM_PARAM_ELEMENTS];
        memcpy (params, layer->params, NUM_PARAM_ELEMENTS * sizeof (unsigned int));
        params[BATCH] = ldata->nasm->batch_size;
        params[MAT_N] = ldata->nasm->tr_seq_len;
        tensor = aspen_tensor_init (params, org_order, 3, layer->dnn->element_size);
    }
    else if (layer->type == K_ATTENTION_LAYER)
    {
        LAYER_PARAMS org_order[] = {BATCH, NUM_HEAD, MAT_N, MAT_M};
        unsigned int params[NUM_PARAM_ELEMENTS];
        memcpy (params, layer->params, NUM_PARAM_ELEMENTS * sizeof (unsigned int));
        params[BATCH] = ldata->nasm->batch_size;
        params[MAT_N] = ldata->nasm->tr_seq_len;
        tensor = aspen_tensor_init (params, org_order, 4, layer->dnn->element_size);
    }
    else 
    {
        ERROR_PRTF ("Error in get_ldata_output: unsupported layer type.");
        aspen_free (packed_data);
        assert (0);
    }
    tensor->data = packed_data;
    reorder_aspen_tensor (&tensor, tensor->dims, order, tensor->num_dims);
    output = calloc (ldata->out_mat_dims[OUT_H] * ldata->out_mat_dims[OUT_W], elem_size);
    memcpy (output, tensor->data, data_size);
    destroy_aspen_tensor (tensor);
    *out_ptr = output;
    return data_size;
}

void* get_aspen_tensor_element_ptr (aspen_tensor_t *tensor, unsigned int *pos)
{
    unsigned int idx = get_tensor_idx_from_pos (tensor, pos);
    return (char*)tensor->data + idx*tensor->element_size;
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

void fill_tensor_with_fixed_nums (aspen_tensor_t *tensor, float num)
{
    if (tensor == NULL || tensor->data == NULL)
        return;
    for (size_t i = 0; i < tensor->num_elements; i++)
    {
        ((float *)tensor->data)[i] = num;
    }
}

void fill_tensor_with_rand_nums (aspen_tensor_t *tensor, float range_abs)
{
    if (tensor == NULL || tensor->data == NULL)
        return;
    for (size_t i = 0; i < tensor->num_elements; i++)
    {
        float num = (((float)rand() / (float)RAND_MAX) - 0.5) * 2 * range_abs;
        ((float *)tensor->data)[i] = num;
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

// Change to add a new layer type
void create_layer_tensors (aspen_layer_t *layer)
{
    if (layer->type == CONV_LAYER)
    {
        LAYER_PARAMS weight_dim_order[] = {OUT_C, WEIGHT_H, WEIGHT_W, IN_C};
        layer->tensors [WEIGHT_TENSOR] = aspen_tensor_init (layer->params, weight_dim_order, 4, layer->dnn->element_size);
        calloc_aspen_tensor (layer->tensors [WEIGHT_TENSOR]);

        LAYER_PARAMS bias_dim_order[] = {OUT_C};
        layer->tensors [BIAS_TENSOR] = aspen_tensor_init (layer->params, bias_dim_order, 1, layer->dnn->element_size);
        calloc_aspen_tensor (layer->tensors [BIAS_TENSOR]);
    }
    else if (layer->type == FC_LAYER)
    {
        LAYER_PARAMS weight_dim_order[] = {OUT_C, IN_C, IN_H, IN_W};
        layer->tensors [WEIGHT_TENSOR] = aspen_tensor_init (layer->params, weight_dim_order, 4, layer->dnn->element_size);
        calloc_aspen_tensor (layer->tensors [WEIGHT_TENSOR]);

        LAYER_PARAMS bias_dim_order[] = {OUT_C};
        layer->tensors [BIAS_TENSOR] = aspen_tensor_init (layer->params, bias_dim_order, 1, layer->dnn->element_size);
        calloc_aspen_tensor (layer->tensors [BIAS_TENSOR]);
    }
    else if (layer->type == MATMUL_LAYER)
    {
        LAYER_PARAMS weight_dim_order[] = {MAT_M, MAT_K};
        layer->tensors [WEIGHT_TENSOR] = aspen_tensor_init (layer->params, weight_dim_order, 2, layer->dnn->element_size);
        calloc_aspen_tensor (layer->tensors [WEIGHT_TENSOR]);

        LAYER_PARAMS bias_dim_order[] = {MAT_M};
        layer->tensors [BIAS_TENSOR] = aspen_tensor_init (layer->params, bias_dim_order, 1, layer->dnn->element_size);
        calloc_aspen_tensor (layer->tensors [BIAS_TENSOR]);
    }
    else if (layer->type == LAYERNORM_LAYER)
    {
        LAYER_PARAMS weight_dim_order[] = {MAT_M};
        layer->tensors [WEIGHT_TENSOR] = aspen_tensor_init (layer->params, weight_dim_order, 1, layer->dnn->element_size);
        calloc_aspen_tensor (layer->tensors [WEIGHT_TENSOR]);

        LAYER_PARAMS bias_dim_order[] = {MAT_M};
        layer->tensors [BIAS_TENSOR] = aspen_tensor_init (layer->params, bias_dim_order, 1, layer->dnn->element_size);
        calloc_aspen_tensor (layer->tensors [BIAS_TENSOR]);
    }
    else if (layer->type == INPUT_LAYER || layer->type == MAXPOOL_LAYER || layer->type == AVGPOOL_LAYER || layer->type == SOFTMAX_LAYER || layer->type == YOLO_LAYER || layer->type == APPEND_LAYER
        || layer->type == RESIDUAL_LAYER || layer->type == LAYERNORM_LAYER || layer->type == K_ATTENTION_LAYER || layer->type == V_ATTENTION_LAYER)
    {
    }
    else
    {
        ERROR_PRTF ("ERROR: Unsupported layer type %s, at line %d in file %s" , layer_type_str[layer->type], 0, " ");
        assert (0);
    }

    if (layer->tensors[WEIGHT_TENSOR])
        fill_tensor_with_rand_nums (layer->tensors[WEIGHT_TENSOR], 0.3);
    if (layer->tensors[BIAS_TENSOR])
        fill_tensor_with_rand_nums (layer->tensors[BIAS_TENSOR], 0.3);
}

// Change to add a new layer type
void create_layer_output_tensor (aspen_layer_t *layer)
{
    if (layer->type == CONV_LAYER || layer->type == INPUT_LAYER || layer->type == MAXPOOL_LAYER || layer->type == AVGPOOL_LAYER 
        || layer->type == FC_LAYER || layer->type == RESIDUAL_LAYER || layer->type == YOLO_LAYER || layer->type == APPEND_LAYER)
    {
        if (MAT_M != 0)
        {
            LAYER_PARAMS dim_order[] = {BATCH, OUT_H, OUT_W, OUT_C};
            layer->tensors [OUTPUT_TENSOR] = aspen_tensor_init (layer->params, dim_order, 4, layer->dnn->element_size);
            calloc_aspen_tensor (layer->tensors [OUTPUT_TENSOR]);
        }
        else
        {
            LAYER_PARAMS dim_order[] = {BATCH, MAT_N, MAT_M};
            layer->tensors [OUTPUT_TENSOR] = aspen_tensor_init (layer->params, dim_order, 3, layer->dnn->element_size);
            calloc_aspen_tensor (layer->tensors [OUTPUT_TENSOR]);
        }
    }
    else if (layer->type == FC_LAYER || layer->type == SOFTMAX_LAYER)
    {
        LAYER_PARAMS dim_order[] = {BATCH, OUT_C};
        layer->tensors [OUTPUT_TENSOR] = aspen_tensor_init (layer->params, dim_order, 2, layer->dnn->element_size);
        calloc_aspen_tensor (layer->tensors [OUTPUT_TENSOR]);
    }
    else if (layer->type == LAYERNORM_LAYER
        || layer->type == V_ATTENTION_LAYER || layer->type == MATMUL_LAYER)
    {
        LAYER_PARAMS dim_order[] = {BATCH, MAT_N, MAT_M};
        layer->tensors [OUTPUT_TENSOR] = aspen_tensor_init (layer->params, dim_order, 3, layer->dnn->element_size);
        calloc_aspen_tensor (layer->tensors [OUTPUT_TENSOR]);
    }
    else if (layer->type == K_ATTENTION_LAYER)
    {
        LAYER_PARAMS dim_order[] = {BATCH, NUM_HEAD, MAT_N, MAT_M};
        layer->tensors [OUTPUT_TENSOR] = aspen_tensor_init (layer->params, dim_order, 4, layer->dnn->element_size);
        calloc_aspen_tensor (layer->tensors [OUTPUT_TENSOR]);
    }
    else
    {
        ERROR_PRTF ("ERROR: Unsupported layer type %s, at line %d in file %s" , layer_type_str[layer->type], 0, " ");
        assert (0);
    }
    
    #ifdef DEBUG
    fill_tensor_with_fixed_nums (layer->tensors[OUTPUT_TENSOR], 0);
    #endif
}

void layer_find_input_pos_idx (aspen_layer_t *layer)
{
    aspen_layer_t *p_layer = layer->parent_layers[PARENT_0];
    if (layer->type == CONV_LAYER || layer->type == MAXPOOL_LAYER || layer->type == AVGPOOL_LAYER)
    {
        int *input_idx_arr = layer->tensors [COL_IDX_TENSOR]->data;
        unsigned int num_idx = 0;

        for (int out_b = 0; out_b < layer->params[BATCH]; out_b++)
        {
            unsigned int in_b = out_b;
            for (int out_h = 0; out_h < layer->params[OUT_H]; out_h++)
            {
                for (int out_w = 0; out_w < layer->params[OUT_W]; out_w++)
                {
                    for (int kh = 0; kh < layer->params[WEIGHT_H]; kh++)
                    {
                        for (int kw = 0; kw < layer->params[WEIGHT_W]; kw++)
                        {
                            int in_h = out_h * layer->params[STRIDE] + kh  - layer->params[PADDING];
                            int in_w = out_w * layer->params[STRIDE] + kw  - layer->params[PADDING];
                            if (in_h < 0 || in_h >= p_layer->params[OUT_H] || in_w < 0 || in_w >= p_layer->params[OUT_W])
                            {
                                input_idx_arr[num_idx++] = -1;
                                continue;
                            }
                            int in_idx = in_b * p_layer->params[OUT_H] * p_layer->params[OUT_W] 
                                + in_h * p_layer->params[OUT_W] + in_w;
                            input_idx_arr[num_idx++] = in_idx;
                        }
                    }
                }
            }
        }
    }
}

// Change to add a new layer type
void create_layer_col_idx_tensor (aspen_layer_t *layer)
{
    if (layer->type == CONV_LAYER || layer->type == MAXPOOL_LAYER || layer->type == AVGPOOL_LAYER)
    {
        LAYER_PARAMS dim_order[] = {BATCH, OUT_H, OUT_W, WEIGHT_H, WEIGHT_W};
        layer->tensors [COL_IDX_TENSOR] = aspen_tensor_init (layer->params, dim_order, 5, sizeof(int));
        calloc_aspen_tensor (layer->tensors [COL_IDX_TENSOR]);
        layer_find_input_pos_idx (layer);
    }
    #ifdef DEBUG
    fill_tensor_with_fixed_nums (layer->tensors[OUTPUT_TENSOR], 0);
    #endif
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
    for (int i = 0; i < NUM_TENSORS; i++)
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
    printf("\n\t\tNum Elements: %d\n", tensor->num_elements);
    printf("\t\tElement Size: %d\n", tensor->element_size);
    printf("\t\tIs CPU data allocated? %s (%p)\n", tensor->data ? "Yes" : "No", tensor->data);
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
                    printf("\n\t\t\t");
                    for (int j = 0; j < tensor->num_dims; j++)
                    {
                        printf("%d,", (i/dims_mult_arr[j+1]) % tensor->dims[tensor->data_dim_order[j]]);
                    }
                    printf(": ");
                }
                printf("%3.3e ", *((float*)tensor->data + i));
            }
            printf("\n");
        }
    }
}

void *aspen_load_input_NHWC(char *input_filename, unsigned int *input_dims, unsigned int element_size)
{
    size_t num_elements = 1;
    for (int i = 0; i < NUM_PARAM_ELEMENTS; i++)
    {
        if (input_dims[i] != 0)
            num_elements *= input_dims[i];
    }
    void *file_data = NULL;
    if (input_filename != NULL)
        file_data = load_arr (input_filename, num_elements * element_size);
    else
        file_data = calloc (num_elements, element_size);
    void *output = aspen_calloc (num_elements, element_size);
    if (input_dims [OUT_C] != 0 && input_dims [OUT_H] != 0 && input_dims [OUT_W] != 0)
    {
        // Convert from NCHW to NHWC
        NCHW_to_NHWC (file_data, output, 
            input_dims [BATCH], input_dims [OUT_C], input_dims [OUT_H], input_dims [OUT_W], element_size);
    }
    else
    {
        memcpy (output, file_data, num_elements * element_size);
    }
    free (file_data);
    return output;
}

void *aspen_load_input(char *input_filename, unsigned int *input_dims, unsigned int element_size)
{
    size_t num_elements = 1;
    for (int i = 0; i < NUM_PARAM_ELEMENTS; i++)
    {
        if (input_dims[i] != 0)
            num_elements *= input_dims[i];
    }
    void *file_data = NULL;
    if (input_filename != NULL)
        file_data = load_arr (input_filename, num_elements * element_size);
    else
        file_data = calloc (num_elements, element_size);
    void *output = aspen_calloc (num_elements, element_size);
    memcpy (output, file_data, num_elements * element_size);
    free (file_data);
    return output;
}


void aspen_init_naive (aspen_dnn_t* dnn, unsigned int *input_params, void *input_data)
{
    if (dnn == NULL)
    {
        ERROR_PRTF ("Error: DNN is NULL.");
        assert (0);
    }
    // Create output tensors
    for (int i = 0; i < dnn->num_layers; i++)
    {
        aspen_layer_t *layer = &dnn->layers[i];
        layer->params[BATCH] = input_params[BATCH];
        layer->params[NUM_SEQ] = input_params[NUM_SEQ];
        layer->params[MAT_N] = input_params[NUM_SEQ];
        if (input_params[NUM_SEQ] != 0)
        {
            layer->params[OUT_W] = input_params[NUM_SEQ];
        }
        if (layer->type == K_ATTENTION_LAYER)
        {
            layer->params[MAT_M] = (input_params[NUM_SEQ]);
            layer->params[MAT_K] = (input_params[NUM_HIDDEN] / layer->params[NUM_HEAD]);
        }
        create_layer_output_tensor (layer);
        create_layer_col_idx_tensor (layer);
    }
    memcpy (dnn->layers[0].tensors[OUTPUT_TENSOR]->data, input_data, 
        dnn->layers[0].tensors[OUTPUT_TENSOR]->num_elements * dnn->layers[0].tensors[OUTPUT_TENSOR]->element_size);
}
// Change to add a new layer type
void aspen_run_naive (aspen_dnn_t* dnn, unsigned int *input_params, void *input_data)
{
    if (dnn == NULL)
    {
        ERROR_PRTF ("Error: DNN is NULL.");
        assert (0);
    }
    for (int i = 1; i < dnn->num_layers; i++)
    {
        aspen_layer_t *layer = &dnn->layers[i];
        float *input = (float*)layer->parent_layers[PARENT_0]->tensors[OUTPUT_TENSOR]->data;
        float *input2 = NULL;
        if (layer->parent_layers[PARENT_1] != NULL)
        {
            input2 = (float*)layer->parent_layers[PARENT_1]->tensors[OUTPUT_TENSOR]->data;
        }
        float *output = (float*)layer->tensors[OUTPUT_TENSOR]->data;
        if (layer->type == CONV_LAYER)
        {
            naive_conv2d (input, layer->tensors[WEIGHT_TENSOR]->data, layer->tensors[BIAS_TENSOR]->data, output,
                layer->params[BATCH], layer->params[IN_C], layer->params[IN_H], layer->params[IN_W],
                layer->params[OUT_C], layer->params[WEIGHT_H], layer->params[WEIGHT_W],
                layer->params[STRIDE], layer->params[PADDING]);
        }
        else if (layer->type == MAXPOOL_LAYER)
        {
            naive_maxpool2d (input, output, layer->params[BATCH], layer->params[IN_C], layer->params[IN_H], layer->params[IN_W],
                layer->params[WEIGHT_H], layer->params[WEIGHT_W], layer->params[STRIDE], layer->params[PADDING]);
        }
        else if (layer->type == AVGPOOL_LAYER)
        {
            naive_avgpool2d (input, output, layer->params[BATCH], layer->params[IN_C], layer->params[IN_H], layer->params[IN_W],
                layer->params[WEIGHT_H], layer->params[WEIGHT_W], layer->params[STRIDE], layer->params[PADDING]);
        }
        else if (layer->type == SOFTMAX_LAYER)
        {
            naive_softmax (input, output, layer->params[BATCH], layer->tensors[OUTPUT_TENSOR]->num_elements/ layer->params[BATCH]);
        }
        else if (layer->type == FC_LAYER)
        {
            naive_fully_connected (input, layer->tensors[WEIGHT_TENSOR]->data, layer->tensors[BIAS_TENSOR]->data, output,
                layer->params[BATCH], layer->params[IN_C]*layer->params[IN_H]*layer->params[IN_W], layer->params[OUT_C]);
        }
        else if (layer->type == RESIDUAL_LAYER)
        {
            naive_residual (input, input2, output, layer->tensors[OUTPUT_TENSOR]->num_elements);
        }
        else if (layer->type == MATMUL_LAYER)
        {
            memset (output, 0, layer->tensors[OUTPUT_TENSOR]->num_elements * layer->tensors[OUTPUT_TENSOR]->element_size);
            SGEMM_KERNEL_OMP (layer->params[MAT_M], layer->params[MAT_N]*layer->params[BATCH], layer->params[MAT_K], 
                layer->tensors[WEIGHT_TENSOR]->data, layer->params[MAT_K], input, layer->params[MAT_K], 
                    output, layer->params[MAT_M]);
            for (int j = 0; j < layer->params[MAT_N]*layer->params[BATCH]; j++)
            {
                for (int k = 0; k < layer->params[MAT_M]; k++)
                {
                    output[j*layer->params[MAT_M] + k] += ((float*)layer->tensors[BIAS_TENSOR]->data)[k];
                }
            }
        }
        else if (layer->type == K_ATTENTION_LAYER)
        {
            naive_k_attention (input, input2, output
                , layer->params[BATCH], layer->params[NUM_HEAD], layer->params[NUM_HIDDEN], layer->params[NUM_SEQ], layer->params[MASKED]);
        }
        else if (layer->type == V_ATTENTION_LAYER)
        {
            naive_v_attention (input, input2, output
                , layer->params[BATCH], layer->params[NUM_HEAD], layer->params[NUM_HIDDEN], layer->params[NUM_SEQ]);
        }
        else if (layer->type == LAYERNORM_LAYER)
        {
            naive_layernorm (input, layer->tensors[WEIGHT_TENSOR]->data, layer->tensors[BIAS_TENSOR]->data, output, 
                layer->params[BATCH]*layer->params[MAT_N], layer->params[MAT_M]);
        }
        else if (layer->type == YOLO_LAYER)
        {
            naive_yolo (input, layer->tensors[ANCHOR_TENSOR]->data,
                output, layer->params[OUT_C], layer->params[IN_H], layer->params[IN_W], layer->params[IN_C], layer->params[STRIDE]);
        }
        else if (layer->type == APPEND_LAYER)
        {
            naive_append (input, input2, output,
                layer->params[STRIDE], layer->params[IN_C], layer->params[OUT_C] - layer->params[IN_C], layer->params[OUT_H], layer->params[OUT_W]);
        }
        else if (layer->type == INPUT_LAYER)
        {
            // Do nothing
        }
        else 
        {
            ERROR_PRTF ("Error: Layer type not supported.");
            assert (0);
        }
        naive_activate (output, layer->tensors[OUTPUT_TENSOR]->num_elements, layer->activation);
    }
}