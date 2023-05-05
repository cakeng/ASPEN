#include "aspen.h"
#include "apu.h"
#include "nasm.h"
#include "input_parser.h"

LAYER_TYPE transformer_layer_types[LAYERS_PER_TRANSFORMER] 
= {MATMUL_LAYER, MATMUL_LAYER, MATMUL_LAYER,
    K_ATTENTION_LAYER, V_ATTENTION_LAYER, MATMUL_LAYER, RESIDUAL_LAYER, LAYERNORM_LAYER,
    MATMUL_LAYER, MATMUL_LAYER, RESIDUAL_LAYER, LAYERNORM_LAYER};
int transformer_parents [LAYERS_PER_TRANSFORMER][2]
= { {-1, 1}, {-2, 1}, {-3, 1}, 
    {-2, -3}, {-1, -2}, {-1, 1}, {-1, -7}, {-1, 1},
    {-1, 1}, {-1, 1}, {-1, -3}, {-1, 1}};
// 1. Key MM, 2. Query MM, 3. Value MM, 
// 4. K Attention, 5. V Attention, 6. Attention MM, 7. Residual, 8. LayerNorm, 
// 9. Feedforward MM 1, 10. Feedforward MM 2, 11. Residual, 12. LayerNorm

int use_gpu = 1;
int aspen_num_gpus = 0;

#ifdef GPU
cudaStream_t aspen_CUDA_streams[MAX_NUM_GPUS][GPU_MEM_STREAM_HOST_TO_GPU+1];
#endif
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
            // printf ("Reordering weight tensor for layer %d\n", i);
            LAYER_PARAMS weight_dim_order[] = {OUT_C, WEIGHT_H, WEIGHT_W, IN_C, SUB_C};
            unsigned int params[NUM_PARAM_ELEMENTS] = {0};
            memcpy (params, layer->params, sizeof(unsigned int) * NUM_PARAM_ELEMENTS);
            params[SUB_C] = _VEC_SIZE_M;
            params[OUT_C] = (layer->params[OUT_C] + params[SUB_C] - 1) / params[SUB_C];
            reorder_aspen_tensor (&layer->tensors[WEIGHT_TENSOR], params, weight_dim_order, 5);
        }
    }
    return new_dnn;
}

aspen_dnn_t *apu_create_transformer_encoder_dnn (unsigned int num_transformers,
    unsigned int num_hidden, unsigned int num_head, unsigned int ff_scale, char* name, char *weight_path)
{
    if (num_hidden % num_head != 0)
    {
        printf ("ERROR: num_hidden must be a multiple of num_head\n");
        exit (1);
    }
    aspen_dnn_t *new_dnn = init_aspen_dnn (num_transformers * LAYERS_PER_TRANSFORMER + 1, name);
    new_dnn->layers[0].type = INPUT_LAYER;
    new_dnn->layers[0].params [MAT_M] = num_hidden;
    set_layer_inout_sizes(&new_dnn->layers[0]);
    for (int i = 0; i < num_transformers; i++)
    {
        for (int j = 0; j < LAYERS_PER_TRANSFORMER; j++)
        {
            aspen_layer_t *layer = new_dnn->layers + i * LAYERS_PER_TRANSFORMER + j + 1;
            layer->type = transformer_layer_types[j];
            layer->params [MAT_M] = num_hidden;
            if (j == 8) // Feed-forward MM 1
            {
                layer->params [MAT_M] = ff_scale * num_hidden;
                layer->activation = GELU;
            }
            else if (j <= 4) // Attention layers 
            {
                layer->params [NUM_HIDDEN] = num_hidden;
                layer->params [NUM_HEAD] = num_head;
                layer->params [MAT_M] = num_hidden;
            }
            if (layer->type == K_ATTENTION_LAYER)
            {
                layer->params[MAT_M] = 1;
                layer->params[MAT_K] = (num_hidden / num_head);
            }
            layer->parent_layers [PARENT_0] = layer + transformer_parents[j][0];
            if (transformer_parents[j][1] != 1)
                layer->parent_layers [PARENT_1] = layer + transformer_parents[j][1];
            set_layer_inout_sizes(layer);
            create_layer_tensors (layer);
        }
    }
    // print_dnn_info (new_dnn, 0);
    if (weight_path != NULL)
        apu_load_dnn_data_from_file (new_dnn, weight_path);
    for (int i = 0; i < new_dnn->num_layers; i++)
    {
        aspen_layer_t *layer = new_dnn->layers + i;
        if (layer->type == MATMUL_LAYER)
        {
            // printf ("Reordering weight tensor for layer %d\n", i);
            LAYER_PARAMS weight_dim_order[] = {MAT_M, MAT_K, SUB_M};
            unsigned int params[NUM_PARAM_ELEMENTS] = {0};
            memcpy (params, layer->params, sizeof(unsigned int) * NUM_PARAM_ELEMENTS);
            params[SUB_M] = _VEC_SIZE_M;
            params[MAT_M] = (layer->params[MAT_M] + params[SUB_M] - 1) / params[SUB_M];
            reorder_aspen_tensor (&layer->tensors[WEIGHT_TENSOR], params, weight_dim_order, 3);
        }
        sync_dnn_data_to_gpu_layer (layer);
    }
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
    #ifdef AVX2
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    #endif
    #ifdef GPU
    if (use_gpu == 1 && aspen_num_gpus == 0)
    {
        if (check_CUDA(cudaGetDeviceCount(&aspen_num_gpus)) != 0)
        {
            FPRT (stderr, "Error getting number of CUDA devices.\n");
        }
        #ifdef DEBUG
        PRT ("Found %d CUDA device(s).\n", aspen_num_gpus);
        #endif
        for (int i = 0; i < aspen_num_gpus; i++)
        {
            for (int j = 0; j < GPU_MEM_STREAM_HOST_TO_GPU+1; j++)
            {
                if (check_CUDA(cudaStreamCreateWithFlags(&aspen_CUDA_streams[i][j], cudaStreamNonBlocking)) != 0)
                {
                    FPRT (stderr, "Error creating CUDA stream.\n");
                }
            }
        }
    }
    #endif
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

aspen_tensor_t *init_aspen_tensor (unsigned int *params_arr, LAYER_PARAMS *order, int num_dims, unsigned int element_size)
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
        FPRT (stderr, "Cannot calloc tensor with 0 elements or 0 element size.\n");
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

void calloc_aspen_gpu_tensors (aspen_tensor_t *tensor)
{
    if (tensor == NULL)
        return;
    if (tensor->num_elements == 0 || tensor->element_size == 0)
    {
        FPRT (stderr, "Cannot calloc tensor with 0 elements or 0 element size.\n");
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
    for (int i = 0; i < aspen_num_gpus; i++)
    {
        if (tensor->data_gpu[i] != NULL)
            aspen_gpu_free (tensor->data_gpu[i], i);
        tensor->data_gpu[i] = aspen_gpu_calloc (size, tensor->element_size, i);
    }
}

void copy_ptr_to_aspen_tensor  (aspen_tensor_t *tensor, void *ptr)
{
    if (tensor == NULL || tensor->data == NULL)
        return;
    memcpy (tensor->data, ptr, tensor->num_elements * tensor->element_size);
}

void copy_aspen_tensor_to_ptr  (aspen_tensor_t *tensor, void *ptr)
{
    if (tensor == NULL || tensor->data == NULL)
        return;
    memcpy (ptr, tensor->data, tensor->num_elements * tensor->element_size);
}

void copy_aspen_tensor_to_tensor  (aspen_tensor_t *dst, aspen_tensor_t *src)
{
    if (dst == NULL || dst->data == NULL)
        return;
    if (src == NULL || src->data == NULL)
        return;
    if (dst->element_size != src->element_size)
    {
        FPRT (stderr, "Error: cannot copy tensor with different element sizes.\n");
        return;
    }
    if (dst->num_elements != src->num_elements)
    {
        FPRT (stderr, "Error: cannot copy tensor with different number of elements.\n");
        return;
    }
    memcpy (dst->data, src->data, dst->num_elements * dst->element_size);
}

void copy_aspen_tensor_to_gpu  (aspen_tensor_t *tensor, int gpu_idx)
{
    if (tensor == NULL || tensor->data == NULL)
        return;
    if (gpu_idx < 0 || gpu_idx >= aspen_num_gpus)
        return;
    aspen_host_to_gpu_memcpy (tensor->data_gpu[gpu_idx], tensor->data, tensor->num_elements * tensor->element_size, gpu_idx);
}

void copy_aspen_tensor_to_host  (aspen_tensor_t *tensor, int gpu_idx)
{
    if (tensor == NULL || tensor->data == NULL)
        return;
    if (gpu_idx < 0 || gpu_idx >= aspen_num_gpus)
        return;
    aspen_gpu_to_host_memcpy (tensor->data, tensor->data_gpu[gpu_idx], tensor->num_elements * tensor->element_size, gpu_idx);
}

void reorder_aspen_tensor (aspen_tensor_t **tensor_ptr, unsigned int *params_arr, LAYER_PARAMS *order, int num_dims)
{
    aspen_tensor_t *tensor = *tensor_ptr;
    aspen_tensor_t *new_tensor = init_aspen_tensor (params_arr, order, num_dims, tensor->element_size);
    if (new_tensor->num_elements < tensor->num_elements)
    {
        FPRT (stderr, "Error: cannot reorder tensor into smaller number of elements.\n");
        assert (0);
    }
    calloc_aspen_tensor (new_tensor);
    calloc_aspen_gpu_tensors (new_tensor);
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

void *get_aspen_tensor_data (aspen_tensor_t *tensor, LAYER_PARAMS *output_order, int gpu_idx)
{
    void *output = calloc (tensor->num_elements, tensor->element_size);
    aspen_tensor_t *new_tensor = init_aspen_tensor (tensor->dims, output_order, tensor->num_dims, tensor->element_size);
    unsigned int pos[NUM_PARAM_ELEMENTS] = {0};
    if (gpu_idx >= 0)
    {
        copy_aspen_tensor_to_host (tensor, gpu_idx);
    }
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
void *get_ldata_output (nasm_ldata_t *ldata, LAYER_PARAMS *order)
{
    if (ldata == NULL)
    {
        FPRT (stderr, "Error in get_ldata_output: ldata is NULL.\n");
        return NULL;
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
        tensor = init_aspen_tensor (params, org_order, 4, layer->dnn->element_size);
    }
    else if (layer->type == FC_LAYER || layer->type == SOFTMAX_LAYER)
    {
        LAYER_PARAMS org_order[] = {BATCH, OUT_C};
        unsigned int params[NUM_PARAM_ELEMENTS];
        memcpy (params, layer->params, NUM_PARAM_ELEMENTS * sizeof (unsigned int));
        params[BATCH] = ldata->nasm->batch_size;
        tensor = init_aspen_tensor (params, org_order, 2, layer->dnn->element_size);
    }
    else if (layer->type == MATMUL_LAYER || layer->type == LAYERNORM_LAYER || layer->type == RESIDUAL_LAYER ||
        layer->type == INPUT_LAYER || layer->type == V_ATTENTION_LAYER)
    {
        LAYER_PARAMS org_order[] = {BATCH, MAT_N, MAT_M};
        unsigned int params[NUM_PARAM_ELEMENTS];
        memcpy (params, layer->params, NUM_PARAM_ELEMENTS * sizeof (unsigned int));
        params[BATCH] = ldata->nasm->batch_size;
        params[MAT_N] = ldata->nasm->tr_seq_len;
        tensor = init_aspen_tensor (params, org_order, 3, layer->dnn->element_size);
    }
    else if (layer->type == K_ATTENTION_LAYER)
    {
        LAYER_PARAMS org_order[] = {BATCH, NUM_HEAD, MAT_N, MAT_M};
        unsigned int params[NUM_PARAM_ELEMENTS];
        memcpy (params, layer->params, NUM_PARAM_ELEMENTS * sizeof (unsigned int));
        params[BATCH] = ldata->nasm->batch_size;
        params[MAT_N] = ldata->nasm->tr_seq_len;
        tensor = init_aspen_tensor (params, org_order, 4, layer->dnn->element_size);
    }
    else 
    {
        FPRT (stderr, "Error in get_ldata_output: unsupported layer type.\n");
        aspen_free (packed_data);
        assert (0);
    }
    tensor->data = packed_data;
    reorder_aspen_tensor (&tensor, tensor->dims, order, tensor->num_dims);
    output = calloc (ldata->out_mat_dims[OUT_H] * ldata->out_mat_dims[OUT_W], elem_size);
    memcpy (output, tensor->data, data_size);
    destroy_aspen_tensor (tensor);
    return output;
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
    for (int i = 0; i < aspen_num_gpus; i++)
    {
        if (tensor->data_gpu[i] != NULL)
            aspen_gpu_free (tensor->data_gpu[i], i);
    }
    free(tensor);
}

// Change to add a new layer type
void create_layer_tensors (aspen_layer_t *layer)
{
    if (layer->type == CONV_LAYER)
    {
        LAYER_PARAMS weight_dim_order[] = {OUT_C, WEIGHT_H, WEIGHT_W, IN_C};
        layer->tensors [WEIGHT_TENSOR] = init_aspen_tensor (layer->params, weight_dim_order, 4, layer->dnn->element_size);
        calloc_aspen_tensor (layer->tensors [WEIGHT_TENSOR]);
        calloc_aspen_gpu_tensors (layer->tensors [WEIGHT_TENSOR]);
        
        LAYER_PARAMS bias_dim_order[] = {OUT_C};
        layer->tensors [BIAS_TENSOR] = init_aspen_tensor (layer->params, bias_dim_order, 1, layer->dnn->element_size);
        calloc_aspen_tensor (layer->tensors [BIAS_TENSOR]);
        calloc_aspen_gpu_tensors (layer->tensors [BIAS_TENSOR]);
    }
    else if (layer->type == FC_LAYER)
    {
        LAYER_PARAMS weight_dim_order[] = {OUT_C, IN_C};
        layer->tensors [WEIGHT_TENSOR] = init_aspen_tensor (layer->params, weight_dim_order, 2, layer->dnn->element_size);
        calloc_aspen_tensor (layer->tensors [WEIGHT_TENSOR]);
        calloc_aspen_gpu_tensors (layer->tensors [WEIGHT_TENSOR]);
        LAYER_PARAMS bias_dim_order[] = {OUT_C};
        layer->tensors [BIAS_TENSOR] = init_aspen_tensor (layer->params, bias_dim_order, 1, layer->dnn->element_size);
        calloc_aspen_tensor (layer->tensors [BIAS_TENSOR]);
        calloc_aspen_gpu_tensors (layer->tensors [BIAS_TENSOR]);
    }
    else if (layer->type == MATMUL_LAYER)
    {
        LAYER_PARAMS weight_dim_order[] = {MAT_M, MAT_K};
        layer->tensors [WEIGHT_TENSOR] = init_aspen_tensor (layer->params, weight_dim_order, 2, layer->dnn->element_size);
        calloc_aspen_tensor (layer->tensors [WEIGHT_TENSOR]);
        calloc_aspen_gpu_tensors (layer->tensors [WEIGHT_TENSOR]);
        LAYER_PARAMS bias_dim_order[] = {MAT_M};
        layer->tensors [BIAS_TENSOR] = init_aspen_tensor (layer->params, bias_dim_order, 1, layer->dnn->element_size);
        calloc_aspen_tensor (layer->tensors [BIAS_TENSOR]);
        calloc_aspen_gpu_tensors (layer->tensors [BIAS_TENSOR]);
    }
    else if (layer->type == LAYERNORM_LAYER)
    {
        LAYER_PARAMS weight_dim_order[] = {MAT_M};
        layer->tensors [WEIGHT_TENSOR] = init_aspen_tensor (layer->params, weight_dim_order, 1, layer->dnn->element_size);
        calloc_aspen_tensor (layer->tensors [WEIGHT_TENSOR]);
        calloc_aspen_gpu_tensors (layer->tensors [WEIGHT_TENSOR]);
        LAYER_PARAMS bias_dim_order[] = {MAT_M};
        layer->tensors [BIAS_TENSOR] = init_aspen_tensor (layer->params, bias_dim_order, 1, layer->dnn->element_size);
        calloc_aspen_tensor (layer->tensors [BIAS_TENSOR]);
        calloc_aspen_gpu_tensors (layer->tensors [BIAS_TENSOR]);
    }
    else if (layer->type == INPUT_LAYER || layer->type == MAXPOOL_LAYER || layer->type == AVGPOOL_LAYER || layer->type == SOFTMAX_LAYER || layer->type == YOLO_LAYER || layer->type == APPEND_LAYER
        || layer->type == RESIDUAL_LAYER || layer->type == LAYERNORM_LAYER || layer->type == K_ATTENTION_LAYER || layer->type == V_ATTENTION_LAYER)
    {
    }
    else
    {
        FPRT(stderr, "ERROR: Unsupported layer type %s, at line %d in file %s\n" , layer_type_str[layer->type], __LINE__, __FILE__);
        assert (0);
    }

    if (layer->tensors[WEIGHT_TENSOR])
        fill_tensor_with_rand_nums (layer->tensors[WEIGHT_TENSOR], 0.3);
    if (layer->tensors[BIAS_TENSOR])
        fill_tensor_with_rand_nums (layer->tensors[BIAS_TENSOR], 0.3);
}

// Change to add a new layer type
void create_layer_output_tensor (aspen_layer_t *layer, int gpu_idx)
{
    if (layer->type == CONV_LAYER || layer->type == INPUT_LAYER || layer->type == MAXPOOL_LAYER || layer->type == AVGPOOL_LAYER 
        || layer->type == RESIDUAL_LAYER || layer->type == YOLO_LAYER || layer->type == APPEND_LAYER)
    {
        if (MAT_M != 0)
        {
            LAYER_PARAMS dim_order[] = {BATCH, OUT_H, OUT_W, OUT_C};
            layer->tensors [OUTPUT_TENSOR] = init_aspen_tensor (layer->params, dim_order, 4, layer->dnn->element_size);
            calloc_aspen_tensor (layer->tensors [OUTPUT_TENSOR]);
            if (gpu_idx >= 0)
                calloc_aspen_gpu_tensors (layer->tensors [OUTPUT_TENSOR]);
        }
        else
        {
            LAYER_PARAMS dim_order[] = {BATCH, MAT_N, MAT_M};
            layer->tensors [OUTPUT_TENSOR] = init_aspen_tensor (layer->params, dim_order, 3, layer->dnn->element_size);
            calloc_aspen_tensor (layer->tensors [OUTPUT_TENSOR]);
            if (gpu_idx >= 0)
                calloc_aspen_gpu_tensors (layer->tensors [OUTPUT_TENSOR]);
        }
    }
    else if (layer->type == FC_LAYER || layer->type == SOFTMAX_LAYER)
    {
        LAYER_PARAMS dim_order[] = {BATCH, OUT_C};
        layer->tensors [OUTPUT_TENSOR] = init_aspen_tensor (layer->params, dim_order, 2, layer->dnn->element_size);
        calloc_aspen_tensor (layer->tensors [OUTPUT_TENSOR]);
        if (gpu_idx >= 0)
            calloc_aspen_gpu_tensors (layer->tensors [OUTPUT_TENSOR]);
    }
    else if (layer->type == LAYERNORM_LAYER
        || layer->type == V_ATTENTION_LAYER || layer->type == MATMUL_LAYER)
    {
        LAYER_PARAMS dim_order[] = {BATCH, MAT_N, MAT_M};
        layer->tensors [OUTPUT_TENSOR] = init_aspen_tensor (layer->params, dim_order, 3, layer->dnn->element_size);
        calloc_aspen_tensor (layer->tensors [OUTPUT_TENSOR]);
        if (gpu_idx >= 0)
            calloc_aspen_gpu_tensors (layer->tensors [OUTPUT_TENSOR]);
    }
    else if (layer->type == K_ATTENTION_LAYER)
    {
        LAYER_PARAMS dim_order[] = {BATCH, NUM_HEAD, MAT_N, MAT_M};
        layer->tensors [OUTPUT_TENSOR] = init_aspen_tensor (layer->params, dim_order, 4, layer->dnn->element_size);
        calloc_aspen_tensor (layer->tensors [OUTPUT_TENSOR]);
        if (gpu_idx >= 0)
            calloc_aspen_gpu_tensors (layer->tensors [OUTPUT_TENSOR]);
    }
    else
    {
        FPRT(stderr, "ERROR: Unsupported layer type %s, at line %d in file %s\n" , layer_type_str[layer->type], __LINE__, __FILE__);
        assert (0);
    }
    
    #ifdef DEBUG
    // fill_tensor_with_nums (layer->tensors[OUTPUT_TENSOR]);
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
void create_layer_col_idx_tensor (aspen_layer_t *layer, int gpu_idx)
{
    if (layer->type == CONV_LAYER || layer->type == MAXPOOL_LAYER || layer->type == AVGPOOL_LAYER)
    {
        LAYER_PARAMS dim_order[] = {BATCH, OUT_H, OUT_W, WEIGHT_H, WEIGHT_W};
        layer->tensors [COL_IDX_TENSOR] = init_aspen_tensor (layer->params, dim_order, 5, sizeof(int));
        calloc_aspen_tensor (layer->tensors [COL_IDX_TENSOR]);
        if (gpu_idx >= 0)
            calloc_aspen_gpu_tensors (layer->tensors [COL_IDX_TENSOR]);
        layer_find_input_pos_idx (layer);
        copy_aspen_tensor_to_gpu (layer->tensors [COL_IDX_TENSOR], gpu_idx);
    }
    #ifdef DEBUG
    // fill_tensor_with_nums (layer->tensors[OUTPUT_TENSOR]);
    fill_tensor_with_fixed_nums (layer->tensors[OUTPUT_TENSOR], 0);
    #endif
}


void sync_dnn_data_to_gpu_layer (aspen_layer_t *layer)
{
    if (layer == NULL)
    {
        FPRT (stderr, "ERROR in sync_dnn_data_to_gpu_layer: layer is NULL, at line %d in file %s.\n", __LINE__, __FILE__);
        assert (0);
    }
    for (int i = 0; i < NUM_TENSORS; i++)
    {
        if (layer->tensors[i] != NULL)
        {
            if (i == WEIGHT_TENSOR || i == BIAS_TENSOR)
            {
                for (int j = 0; j < aspen_num_gpus; j++)
                {
                    copy_aspen_tensor_to_gpu (layer->tensors[i], j);
                }
            }
        }
    }
    for (int j = 0; j < aspen_num_gpus; j++)
    {
        aspen_sync_gpu (j);
    }
}
void sync_dnn_data_to_gpu_dnn (aspen_dnn_t *dnn)
{
    if (dnn == NULL)
    {
        FPRT (stderr, "ERROR in sync_dnn_data_to_gpu_dnn: dnn is NULL, at line %d in file %s.\n", __LINE__, __FILE__);
        assert (0);
    }
    for (int i = 0; i < dnn->num_layers; i++)
    {
        sync_dnn_data_to_gpu_layer (&dnn->layers[i]);
    }
    for (int j = 0; j < aspen_num_gpus; j++)
    {
        aspen_sync_gpu (j);
    }
}
void sync_output_data_to_host_layer (aspen_layer_t *layer, int gpu_idx)
{
    if (layer == NULL)
    {
        FPRT (stderr, "ERROR in sync_output_data_to_host_layer: layer is NULL, at line %d in file %s.\n", __LINE__, __FILE__);
        assert (0);
    }
    for (int i = 0; i < NUM_TENSORS; i++)
    {
        if (layer->tensors[i] != NULL)
        {
            if (i == OUTPUT_TENSOR)
            {
                copy_aspen_tensor_to_host (layer->tensors[i], gpu_idx);
            }
        }
    }
    for (int j = 0; j < aspen_num_gpus; j++)
    {
        aspen_sync_gpu (j);
    }
}
void sync_output_data_to_host_dnn (aspen_dnn_t *dnn, int gpu_idx)
{
    if (dnn == NULL)
    {
        FPRT (stderr, "ERROR in sync_output_data_to_host_dnn: dnn is NULL, at line %d in file %s.\n", __LINE__, __FILE__);
        assert (0);
    }
    for (int i = 0; i < dnn->num_layers; i++)
    {
        sync_output_data_to_host_layer (&dnn->layers[i], gpu_idx);
    }
    for (int j = 0; j < aspen_num_gpus; j++)
    {
        aspen_sync_gpu (j);
    }
}
void sync_output_data_to_gpu_layer (aspen_layer_t *layer, int gpu_idx)
{
    if (layer == NULL)
    {
        FPRT (stderr, "ERROR in sync_output_data_to_gpu_layer: layer is NULL, at line %d in file %s.\n", __LINE__, __FILE__);
        assert (0);
    }
    for (int i = 0; i < NUM_TENSORS; i++)
    {
        if (layer->tensors[i] != NULL)
        {
            if (i == OUTPUT_TENSOR)
            {
                copy_aspen_tensor_to_gpu (layer->tensors[i], gpu_idx);
            }
        }
    }
    for (int j = 0; j < aspen_num_gpus; j++)
    {
        aspen_sync_gpu (j);
    }
}
void sync_output_data_to_gpu_dnn (aspen_dnn_t *dnn, int gpu_idx)
{
    if (dnn == NULL)
    {
        FPRT (stderr, "ERROR in sync_output_data_to_gpu_dnn: dnn is NULL, at line %d in file %s.\n", __LINE__, __FILE__);
        assert (0);
    }
    for (int i = 0; i < dnn->num_layers; i++)
    {
        sync_output_data_to_gpu_layer (&dnn->layers[i], gpu_idx);
    }
    for (int j = 0; j < aspen_num_gpus; j++)
    {
        aspen_sync_gpu (j);
    }
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
    printf("\t\tIs GPU data allocated? %s\n", tensor->data_gpu[0] ? "Yes" : "No");
    for (int i = 0; i < aspen_num_gpus; i++)
    {
        if (tensor->data_gpu[i])
            printf("\t\t\tGPU %d: %p\n", i, tensor->data_gpu[i]);
    }
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
    void *file_data = load_arr (input_filename, num_elements * element_size);
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
    void *file_data = load_arr (input_filename, num_elements * element_size);
    void *output = aspen_calloc (num_elements, element_size);
    memcpy (output, file_data, num_elements * element_size);
    free (file_data);
    return output;
}


void aspen_init_naive (aspen_dnn_t* dnn, unsigned int *input_params, void *input_data, int gpu_idx)
{
    if (dnn == NULL)
    {
        FPRT (stderr, "Error: DNN is NULL.\n");
        assert (0);
    }
    if (gpu_idx >= aspen_num_gpus)
    {
        FPRT (stderr, "ERROR: aspen_init_naive: gpu_idx %d is out of range... Falling back to CPU\n", gpu_idx);
        gpu_idx = -1;
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
        create_layer_output_tensor (layer, gpu_idx);
        create_layer_col_idx_tensor (layer, gpu_idx);
    }
    // print_dnn_info (dnn, 0);
    memcpy (dnn->layers[0].tensors[OUTPUT_TENSOR]->data, input_data, 
        dnn->layers[0].tensors[OUTPUT_TENSOR]->num_elements * dnn->layers[0].tensors[OUTPUT_TENSOR]->element_size);
    if (gpu_idx >= 0)
        copy_aspen_tensor_to_gpu (dnn->layers[0].tensors[OUTPUT_TENSOR], gpu_idx);
}
// Change to add a new layer type
void aspen_run_naive (aspen_dnn_t* dnn, unsigned int *input_params, void *input_data, int gpu_idx)
{
    if (dnn == NULL)
    {
        FPRT (stderr, "Error: DNN is NULL.\n");
        assert (0);
    }
    if (gpu_idx >= aspen_num_gpus)
    {
        FPRT (stderr, "ERROR: aspen_run_naive: gpu_idx %d is out of range... Falling back to CPU\n", gpu_idx);
        gpu_idx = -1;
    }
    if (gpu_idx < 0)
    {
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
                    layer->params[BATCH], layer->params[IN_C], layer->params[OUT_C]);
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
                    , layer->params[BATCH], layer->params[NUM_HEAD], layer->params[NUM_HIDDEN], layer->params[NUM_SEQ]);
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
                FPRT (stderr, "Error: Layer type not supported.\n");
                assert (0);
            }
            naive_activate (output, layer->tensors[OUTPUT_TENSOR]->num_elements, layer->activation);
            // PRT ("apu_run_naive: Layer %d done.\n", i);
        }
    }
    else
    {
        for (int i = 1; i < dnn->num_layers; i++)
        {
            #ifdef GPU
            aspen_layer_t *layer = &dnn->layers[i];
            float *input = (float*)layer->parent_layers[PARENT_0]->tensors[OUTPUT_TENSOR]->data_gpu[gpu_idx];
            float *input2 = NULL;
            if (layer->parent_layers[PARENT_1] != NULL)
            {
                input2 = (float*)layer->parent_layers[PARENT_1]->tensors[OUTPUT_TENSOR]->data_gpu[gpu_idx];
            }
            float *output = (float*)layer->tensors[OUTPUT_TENSOR]->data_gpu[gpu_idx];
            if (layer->type == CONV_LAYER)
            {   
                cuda_conv2d (
                    layer->params[OUT_C],
                    layer->params[BATCH]*layer->params[OUT_H]*layer->params[OUT_W], 
                    layer->tensors[COL_IDX_TENSOR]->data_gpu[gpu_idx], 
                    layer->params[WEIGHT_H]*layer->params[WEIGHT_H], 
                    layer->params[IN_C],
                    layer->tensors[WEIGHT_TENSOR]->data_gpu[gpu_idx], 
                    layer->params[IN_C]*layer->params[WEIGHT_H]*layer->params[WEIGHT_W],
                    input,  
                    layer->params[IN_C], 
                    output, 
                    layer->params[OUT_C], 
                    layer->tensors[BIAS_TENSOR]->data_gpu[gpu_idx], 
                    layer->activation, 
                    aspen_CUDA_streams[gpu_idx][GPU_NAIVE_RUN_STREAM]);
            }
            else if (layer->type == MAXPOOL_LAYER)
            {
                cuda_maxpool (
                    layer->params[OUT_C], 
                    layer->params[BATCH]*layer->params[OUT_H]*layer->params[OUT_W], 
                    layer->tensors[COL_IDX_TENSOR]->data_gpu[gpu_idx], 
                    layer->params[WEIGHT_H]*layer->params[WEIGHT_H], 
                    input,  
                    layer->params[IN_C], 
                    output, 
                    layer->params[OUT_C], 
                    layer->activation,
                    aspen_CUDA_streams[gpu_idx][GPU_NAIVE_RUN_STREAM]);
            }
            else if (layer->type == AVGPOOL_LAYER)
            {
                cuda_avgpool (
                    layer->params[OUT_C], 
                    layer->params[BATCH]*layer->params[OUT_H]*layer->params[OUT_W], 
                    layer->tensors[COL_IDX_TENSOR]->data_gpu[gpu_idx], 
                    layer->params[WEIGHT_H]*layer->params[WEIGHT_H], 
                    input,  
                    layer->params[IN_C], 
                    output, 
                    layer->params[OUT_C], 
                    layer->activation,
                    aspen_CUDA_streams[gpu_idx][GPU_NAIVE_RUN_STREAM]);
            }
            else if (layer->type == SOFTMAX_LAYER)
            {
            }
            else if (layer->type == FC_LAYER)
            {

            }
            else if (layer->type == RESIDUAL_LAYER)
            {
                cuda_residual (input, input2, output, layer->tensors[OUTPUT_TENSOR]->num_elements, 
                layer->activation, aspen_CUDA_streams[gpu_idx][GPU_NAIVE_RUN_STREAM]);
            }
            else if (layer->type == MATMUL_LAYER)
            {
                cuda_matmul (layer->params[MAT_M], layer->params[MAT_N]*layer->params[BATCH], layer->params[MAT_K], 
                    layer->tensors[WEIGHT_TENSOR]->data_gpu[gpu_idx], layer->params[MAT_K], input, layer->params[MAT_K], 
                    output, layer->params[MAT_M], layer->tensors[BIAS_TENSOR]->data_gpu[gpu_idx], 
                    layer->activation, aspen_CUDA_streams[gpu_idx][GPU_NAIVE_RUN_STREAM]);
            }
            else if (layer->type == K_ATTENTION_LAYER)
            {
                cuda_k_attention (input, input2, output
                    , layer->params[BATCH], layer->params[NUM_HEAD], layer->params[NUM_HIDDEN], layer->params[NUM_SEQ], 
                    aspen_CUDA_streams[gpu_idx][GPU_NAIVE_RUN_STREAM]);
            }
            else if (layer->type == V_ATTENTION_LAYER)
            {
                cuda_v_attention (input, input2, output
                    , layer->params[BATCH], layer->params[NUM_HEAD], layer->params[NUM_HIDDEN], layer->params[NUM_SEQ], 
                    aspen_CUDA_streams[gpu_idx][GPU_NAIVE_RUN_STREAM]);
            }
            else if (layer->type == LAYERNORM_LAYER)
            {
                cuda_layernorm (input, layer->tensors[WEIGHT_TENSOR]->data_gpu[gpu_idx], layer->tensors[BIAS_TENSOR]->data_gpu[gpu_idx], output, 
                    layer->params[BATCH]*layer->params[MAT_N], layer->params[MAT_M], layer->params[MAT_M], layer->params[MAT_M], 
                    aspen_CUDA_streams[gpu_idx][GPU_NAIVE_RUN_STREAM]);
            }
            else if (layer->type == INPUT_LAYER)
            {
                // Do nothing
            }
            else 
            {
                FPRT (stderr, "Error: Layer type not supported.\n");
                assert (0);
            }
            // PRT ("apu_run_naive: Layer %d done.\n", i);
            #endif
            aspen_sync_gpu_stream (gpu_idx, GPU_NAIVE_RUN_STREAM);
        }
    }
}