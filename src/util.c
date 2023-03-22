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

void *aspen_calloc (size_t num, size_t size)
{
    if (num*size <= 0)
        return NULL;
    void* ptr = NULL;
    ptr = aligned_alloc (MEM_ALIGN, get_smallest_dividable(num * size, MEM_ALIGN));   
    // cudaError_t cuda_err = cudaMallocHost (&ptr, get_smallest_dividable (num * size, MEM_ALIGN));
    // if (ptr == NULL || check_CUDA(cuda_err) != cudaSuccess)
    // {
    //     printf("Error: Failed to allocate Host memory.\n");
    //     exit(1);
    // }
    bzero (ptr, get_smallest_dividable (num * size, MEM_ALIGN));
    return ptr;
}
void *aspen_malloc (size_t num, size_t size)
{
    if (num*size <= 0)
        return NULL;
    void* ptr = NULL;
    ptr = aligned_alloc (MEM_ALIGN, get_smallest_dividable(num * size, MEM_ALIGN));   
    // cudaError_t cuda_err = cudaMallocHost (&ptr, get_smallest_dividable (num * size, MEM_ALIGN));
    // if (ptr == NULL || check_CUDA(cuda_err) != cudaSuccess)
    // {
    //     printf("Error: Failed to allocate Host memory.\n");
    //     exit(1);
    // }
    return ptr;
}
void aspen_free (void *ptr)
{
    if (ptr == NULL)
        return;
    free (ptr);
    // if (check_CUDA(cudaFreeHost(ptr)) != cudaSuccess)
    // {
    //     printf("Error: Failed to free Host memory.\n");
    //     exit(1);
    // }
}
void *aspen_gpu_calloc (size_t num, size_t size, int gpu_num)
{
    if (num*size <= 0)
        return NULL;
    void* ptr = NULL;
    if (check_CUDA(cudaSetDevice(gpu_num)) != cudaSuccess)
    {
        printf("Error: Failed to set GPU device.\n");
        exit(1);
    }
    if (check_CUDA(cudaMalloc(&ptr, get_smallest_dividable (num * size, MEM_ALIGN) )) != cudaSuccess)
    {
        printf("Error: Failed to allocate GPU memory.\n");
        exit(1);
    }
    if (check_CUDA(cudaMemset(ptr, 0, get_smallest_dividable (num * size, MEM_ALIGN) )) != cudaSuccess)
    {
        printf("Error: Failed to set GPU memory to zero.\n");
        exit(1);
    }
    return ptr;
}
void *aspen_gpu_malloc (size_t num, size_t size, int gpu_num)
{
    if (num*size <= 0)
        return NULL;
    void* ptr = NULL;
    if (check_CUDA(cudaSetDevice(gpu_num)) != cudaSuccess)
    {
        printf("Error: Failed to set GPU device.\n");
        exit(1);
    }
    if (check_CUDA(cudaMalloc(&ptr, get_smallest_dividable (num * size, MEM_ALIGN) )) != cudaSuccess)
    {
        printf("Error: Failed to allocate GPU memory.\n");
        exit(1);
    }
    return ptr;
}
void aspen_gpu_free (void *ptr)
{
    if (ptr == NULL)
        return;
    if (check_CUDA(cudaFree(ptr)) != cudaSuccess)
    {
        printf("Error: Failed to free GPU memory.\n");
        exit(1);
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
        printf("Parent 0 type: %s, Params: \n\t", layer_type_str[p0_layer->type]);
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
    if (ninst->ldata->layer->type == CONV_LAYER || ninst->ldata->layer->type == MAXPOOL_LAYER
        || ninst->ldata->layer->type == AVGPOOL_LAYER || ninst->ldata->layer->type == INPUT_LAYER)
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
        printf("%ld:%d ", parent_ninst->ldata - parent_ninst->ldata->nasm->ldata_arr,
            ninst->parent_ninst_idx_arr[i]);
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

unsigned int get_smallest_dividable (unsigned int num, unsigned int divider)
{
    return (num/divider + (num%divider != 0))*divider;
}