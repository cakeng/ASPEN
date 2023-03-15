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
    printf("Layer Type: %s\n", layer_type_str[layer->type]);
    printf("Layer Index: %d\n", layer->layer_idx);
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
        if (i != NUM_PARAM_ELEMENTS && layer->params[i] != 0)
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
        printf("%s, ", param_type_str[tensor->dims_info[i]]);
    }
    printf("\n\t\tSize: ");
    for (int i = 0; i < tensor->num_dims; i++)
    {
        printf("%d, ", tensor->dims[i]);
    }
    printf("\n");
    if (print_data)
    {
        int new_line_num = 0;
        int dims_mult_arr[tensor->num_dims+1];
        for (int i = tensor->num_dims - 1; i >= 0; i--)
        {
            dims_mult_arr[i] = 1;
            for (int j = i; j < tensor->num_dims; j++)
            {
                dims_mult_arr[i] *= tensor->dims[j];
            }
            if (dims_mult_arr[i] < 20 || new_line_num == 0)
                new_line_num = dims_mult_arr[i];
        }
        dims_mult_arr[tensor->num_dims] = 1;
        printf("\t\tData: ");
        for (int i = 0; i < tensor->num_elements; i++)
        {
            if (i % new_line_num == 0)
            {
                // printf("\n%d:", i);
                printf("\n");
                for (int j = 0; j < tensor->num_dims; j++)
                {
                    printf("%d,", (i/dims_mult_arr[j+1]) % tensor->dims[j]);
                }
                printf(": ");
            }
            printf("%3.2f ", *((float*)tensor->data + i));
        }
        printf("\n");
    }
}

unsigned int get_smallest_dividable (unsigned int num, unsigned int divider)
{
    return (num/divider + (num%divider != 0))*divider;
}