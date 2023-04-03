#include "util.h"

char *ninst_state_str[NUM_NINST_STATES] = 
{
    [NINST_NOT_READY] = "NINST_NOT_READY", [NINST_READY] = "NINST_READY", [NINST_COMPLETED] = "NINST_COMPLETED"
};

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

char *rpool_cond_str [NUM_RPOOL_CONDS] = 
{
    [RPOOL_DNN] = "RPOOL_DNN", [RPOOL_LAYER_TYPE] = "RPOOL_LAYER_TYPE", [RPOOL_LAYER_IDX] = "RPOOL_LAYER_IDX", [RPOOL_NASM] = "RPOOL_NASM", [RPOOL_ASE] = "RPOOL_ASE"
};


void *aspen_calloc (size_t num, size_t size)
{
    if (num*size <= 0)
        return NULL;
    void* ptr = NULL;
    if (aspen_num_gpus < 0)
        ptr = aligned_alloc (MEM_ALIGN, get_smallest_dividable(num * size, MEM_ALIGN));   
    else
    {
        #ifdef GPU
        cudaError_t cuda_err = cudaMallocHost (&ptr, get_smallest_dividable (num * size, MEM_ALIGN));
        if (ptr == NULL || check_CUDA(cuda_err) != cudaSuccess)
        {
            printf("Error: Failed to allocate Host memory.\n");
            assert (0);
        }
        #endif
    }
    bzero (ptr, get_smallest_dividable (num * size, MEM_ALIGN));
    return ptr;
}
void *aspen_malloc (size_t num, size_t size)
{
    if (num*size <= 0)
        return NULL;
    void* ptr = NULL;
    if (aspen_num_gpus < 0)
        ptr = aligned_alloc (MEM_ALIGN, get_smallest_dividable(num * size, MEM_ALIGN));   
    else
    {
        #ifdef GPU
        cudaError_t cuda_err = cudaMallocHost (&ptr, get_smallest_dividable (num * size, MEM_ALIGN));
        if (ptr == NULL || check_CUDA(cuda_err) != cudaSuccess)
        {
            printf("Error: Failed to allocate Host memory.\n");
            assert (0);
        }
        #endif
    }
    return ptr;
}
void aspen_free (void *ptr)
{
    if (ptr == NULL)
        return;
    if (aspen_num_gpus < 0)
        free (ptr);
    else
    {
         #ifdef GPU
        if (check_CUDA(cudaFreeHost(ptr)) != cudaSuccess)
        {
            printf("Error: Failed to free Host memory.\n");
            assert (0);
        }
        #endif
    }
}
void *aspen_gpu_calloc (size_t num, size_t size, int gpu_num)
{
    if (num*size <= 0)
        return NULL;
    void* ptr = NULL;
    #ifdef GPU
    if (check_CUDA(cudaSetDevice(gpu_num)) != cudaSuccess)
    {
        printf("Error: Failed to set GPU device.\n");
        assert (0);
    }
    if (check_CUDA(cudaMalloc(&ptr, get_smallest_dividable (num * size, MEM_ALIGN) )) != cudaSuccess)
    {
        printf("Error: Failed to allocate GPU memory.\n");
        assert (0);
    }
    if (check_CUDA(cudaMemset(ptr, 0, get_smallest_dividable (num * size, MEM_ALIGN) )) != cudaSuccess)
    {
        printf("Error: Failed to set GPU memory to zero.\n");
        assert (0);
    }
    #endif
    return ptr;
}
void *aspen_gpu_malloc (size_t num, size_t size, int gpu_num)
{
    if (num*size <= 0)
        return NULL;
    void* ptr = NULL;
    #ifdef GPU
    if (check_CUDA(cudaSetDevice(gpu_num)) != cudaSuccess)
    {
        printf("Error: Failed to set GPU device.\n");
        assert (0);
    }
    if (check_CUDA(cudaMalloc(&ptr, get_smallest_dividable (num * size, MEM_ALIGN) )) != cudaSuccess)
    {
        printf("Error: Failed to allocate GPU memory.\n");
        assert (0);
    }
    #endif
    return ptr;
}
void aspen_gpu_free (void *ptr, int gpu_num)
{
    if (ptr == NULL)
        return;
    #ifdef GPU
    if (check_CUDA(cudaSetDevice(gpu_num)) != cudaSuccess)
    {
        printf("Error: Failed to set GPU device.\n");
        assert (0);
    }
    if (check_CUDA(cudaFree(ptr)) != cudaSuccess)
    {
        printf("Error: Failed to free GPU memory.\n");
        assert (0);
    }
    #endif
}

void aspen_host_to_gpu_memcpy (void *dst, void *src, size_t num, int gpu_num)
{
    #ifdef GPU
    if (check_CUDA(cudaSetDevice(gpu_num)) != cudaSuccess)
    {
        printf("Error: Failed to set GPU device.\n");
        assert (0);
    }
    if (check_CUDA(cudaMemcpy(dst, src, num, cudaMemcpyHostToDevice)) != cudaSuccess)
    {
        printf("Error: Failed to copy Host to GPU memory.\n");
        assert (0);
    }
    #endif
}
void aspen_gpu_to_host_memcpy (void *dst, void *src, size_t num, int gpu_num)
{
    #ifdef GPU
    if (check_CUDA(cudaSetDevice(gpu_num)) != cudaSuccess)
    {
        printf("Error: Failed to set GPU device.\n");
        assert (0);
    }
    if (check_CUDA(cudaMemcpy(dst, src, num, cudaMemcpyDeviceToHost)) != cudaSuccess)
    {
        printf("Error: Failed to copy GPU to Host memory.\n");
        assert (0);
    }
    #endif
}
void aspen_host_to_gpu_async_memcpy (void *dst, void *src, size_t num, int gpu_num, int stream_num)
{
    #ifdef GPU
    if (check_CUDA(cudaSetDevice(gpu_num)) != cudaSuccess)
    {
        printf("Error: Failed to set GPU device.\n");
        assert (0);
    }
    if (check_CUDA(cudaMemcpyAsync(dst, src, num, cudaMemcpyHostToDevice, aspen_CUDA_streams[gpu_num][stream_num])) != cudaSuccess)
    {
        printf("Error: Failed to copy Host to GPU memory.\n");
        assert (0);
    }
    #endif
}
void aspen_gpu_to_host_async_memcpy (void *dst, void *src, size_t num, int gpu_num, int stream_num)
{
    #ifdef GPU
    if (check_CUDA(cudaSetDevice(gpu_num)) != cudaSuccess)
    {
        printf("Error: Failed to set GPU device.\n");
        assert (0);
    }
    if (check_CUDA(cudaMemcpyAsync(dst, src, num, cudaMemcpyDeviceToHost, aspen_CUDA_streams[gpu_num][stream_num])) != cudaSuccess)
    {
        printf("Error: Failed to copy GPU to Host memory.\n");
        assert (0);
    }
    #endif
}

void aspen_sync_gpu (int gpu_num)
{
    #ifdef GPU
    if (check_CUDA(cudaSetDevice(gpu_num)) != cudaSuccess)
    {
        printf("Error: Failed to set GPU device.\n");
        assert (0);
    }
    if (check_CUDA(cudaDeviceSynchronize()) != cudaSuccess)
    {
        printf("Error: Failed to synchronize GPU.\n");
        assert (0);
    }
    #endif
}

void aspen_sync_gpu_stream (int gpu_num, int stream_num)
{
    #ifdef GPU
    if (check_CUDA(cudaSetDevice(gpu_num)) != cudaSuccess)
    {
        printf("Error: Failed to set GPU device.\n");
        assert (0);
    }
    if (check_CUDA(cudaStreamSynchronize(aspen_CUDA_streams[gpu_num][stream_num])) != cudaSuccess)
    {
        printf("Error: Failed to synchronize GPU stream.\n");
        assert (0);
    }
    #endif
}

int aspen_get_next_stream (int gpu_num)
{
    static int stream_num[MAX_NUM_GPUS];
    stream_num[gpu_num] = (stream_num[gpu_num] + 1) % 32;
    return stream_num[gpu_num];
}

unsigned int get_smallest_dividable (unsigned int num, unsigned int divider)
{
    return (num/divider + (num%divider != 0))*divider;
}