#include "dse.h"

void generate_cudagraph (nasm_t *nasm)
{
    #ifdef GPU
    if (nasm->cudagraph_instantiated != 0)
    {   
        FPRT (stderr, "Cudagraph already instantiated.\n");
        return;
    }
    if (check_CUDA(cudaGraphCreate(&nasm->cuda_graph, 0)) != cudaSuccess)
    {
        FPRT (stderr, "Failed to create cudaGraph.\n");
        return;
    }

    for (int i = 0; i < nasm->num_ldata; i++)
    {
        nasm_ldata_t *ldata = nasm->ldata_arr + i;
        for (int j = 0; j < ldata->num_ninst; j++)
        {
            ninst_t *ninst = ldata->ninst_arr_start + j;
            switch (ldata->layer->type)
            {
            case CONV_LAYER:
                add_cudagraph_node_conv2d (nasm->cuda_graph, ninst, nasm->gpu_idx);
                break;
            case MATMUL_LAYER:
                add_cudagraph_node_matmul (nasm->cuda_graph, ninst, nasm->gpu_idx);
                break;
            case MAXPOOL_LAYER:
                add_cudagraph_node_maxpool2d (nasm->cuda_graph, ninst, nasm->gpu_idx);
                break;
            case AVGPOOL_LAYER:
                add_cudagraph_node_avgpool2d (nasm->cuda_graph, ninst, nasm->gpu_idx);
                break;
            case FC_LAYER:
                add_cudagraph_node_fully_connected (nasm->cuda_graph, ninst, nasm->gpu_idx);
                break;
            case RESIDUAL_LAYER:
                add_cudagraph_node_residual (nasm->cuda_graph, ninst, nasm->gpu_idx);
                break;
            case SOFTMAX_LAYER:
                add_cudagraph_node_softmax (nasm->cuda_graph, ninst, nasm->gpu_idx);
                break;
            case LAYERNORM_LAYER:
                add_cudagraph_node_layernorm (nasm->cuda_graph, ninst, nasm->gpu_idx);
                break;
            case K_ATTENTION_LAYER:
                add_cudagraph_node_k_attention (nasm->cuda_graph, ninst, nasm->gpu_idx);
                break;
            case V_ATTENTION_LAYER:
                add_cudagraph_node_v_attention (nasm->cuda_graph, ninst, nasm->gpu_idx);
                break;
            default:
                break;
            }
            
        }
    }
    if (check_CUDA(cudaGraphInstantiate(&nasm->cuda_graph_exec, nasm->cuda_graph, 0)) != cudaSuccess)
    {
        FPRT (stderr, "Failed to instantiate cudaGraph.\n");
        return;
    }
    nasm->cudagraph_instantiated = 1;
    #endif
}
void run_cudagraph (nasm_t *nasm)
{
    #ifdef GPU
    if (nasm->cudagraph_instantiated == 0)
    {
        FPRT (stderr, "Cudagraph not instantiated.\n");
        return;
    }
    if (check_CUDA(cudaGraphLaunch(nasm->cuda_graph_exec, aspen_CUDA_streams[nasm->gpu_idx][GPU_GRAPH_RUN_STREAM])) != cudaSuccess)
    {
        FPRT (stderr, "Failed to launch cudaGraph.\n");
        return;
    }
    set_nasm_to_finished (nasm);
    aspen_sync_gpu_stream (nasm->gpu_idx, GPU_GRAPH_RUN_STREAM);
    #endif
}
#ifdef GPU
void add_cudagraph_node_conv2d (cudaGraph_t cuda_graph, ninst_t *ninst, int gpu_idx)
{
    nasm_ldata_t *ldata = ninst->ldata;
    aspen_layer_t *layer = ninst->ldata->layer;
    nasm_ldata_t *p_ldata = (ldata->parent_ldata_idx_arr[PARENT_0] + ldata->nasm->ldata_arr);
    // void *scratchpad = prepare_input (ninst, dse->scratchpad);
    unsigned int input_col_size = p_ldata->out_mat_dims[OUT_H];
    // void **input_ptr_arr = dse->scratchpad;   
    // char *input = dse->scratchpad; // (char *) scratchpad;
    unsigned int input_pos_per_n = ninst->num_input_pos/ninst->tile_dims[OUT_W];
    unsigned int M = ninst->tile_dims[OUT_H];
    unsigned int N = ninst->tile_dims[OUT_W];
    unsigned int K = layer->params[WEIGHT_H] * layer->params[WEIGHT_W] * layer->params[IN_C];
    unsigned int lda = K;
    unsigned int ldb = K;
    unsigned int ldc = ldata->out_mat_stride;
    void *A = (char*)layer->tensors[WEIGHT_TENSOR]->data + ninst->out_mat_pos[OUT_H] * lda * layer->dnn->element_size;
    // void *B = dse->scratchpad;
    void *C = get_ninst_out_mem (ninst);
    unsigned int rem_n = N % _TILE_SIZE_N;
    unsigned int rem_m = M % _TILE_SIZE_M;
    unsigned int rem_k = K % _TILE_SIZE_K;
    unsigned int n = 0;
    

}

void add_cudagraph_node_matmul (cudaGraph_t cuda_graph, ninst_t *ninst, int gpu_idx)
{
    nasm_ldata_t *ldata = ninst->ldata;
    aspen_layer_t *layer = ninst->ldata->layer;
    nasm_ldata_t *p_ldata = (ldata->parent_ldata_idx_arr[PARENT_0] + ldata->nasm->ldata_arr);
    unsigned int M = ninst->tile_dims[OUT_H];
    unsigned int N = ninst->tile_dims[OUT_W];
    unsigned int K = layer->params[MAT_K];
    unsigned int lda = K;
    unsigned int ldb = p_ldata->out_mat_stride;
    unsigned int ldc = ldata->out_mat_stride;
    void *A = (char*)layer->tensors[WEIGHT_TENSOR]->data_gpu[gpu_idx] 
        + (ninst->out_mat_pos[OUT_H] * lda * layer->dnn->element_size);
    void *B = (char*)p_ldata->out_mat + (ninst->out_mat_pos[OUT_W] * ldb * layer->dnn->element_size);
    void *C = get_ninst_out_mem (ninst);

    float *bias = (float*)layer->tensors[BIAS_TENSOR]->data_gpu[gpu_idx] + ninst->out_mat_pos[OUT_H];

    // dim3 gridDim (M/_BLOCK_M_SIZE + ((M%_BLOCK_M_SIZE) > 0), N/_BLOCK_N_SIZE + ((N%_BLOCK_N_SIZE) > 0), 1);
    // dim3 blockDim ((_BLOCK_M_SIZE / _THREAD_M_SIZE), (_BLOCK_N_SIZE / _THREAD_N_SIZE), 1);
    // cuda_matmul_kernel<<<gridDim, blockDim, 0, stream>>>(M, N, K, A, lda, B, ldb, C, ldc, bias, layer->activation);

    void *cudagraph_kernel_args [11] = {&M, &N, &K, &A, &lda, &B, &ldb, &C, &ldc, &bias, &layer->activation};
    struct cudaKernelNodeParams params = {0};
    params.func = (void*)cuda_matmul_kernel;
    params.gridDim.x = M/_BLOCK_M_SIZE + ((M%_BLOCK_M_SIZE) > 0);
    params.gridDim.y = N/_BLOCK_N_SIZE + ((N%_BLOCK_N_SIZE) > 0);
    params.gridDim.z = 1;
    params.blockDim.x = _BLOCK_M_SIZE / _THREAD_M_SIZE;
    params.blockDim.y = _BLOCK_N_SIZE / _THREAD_N_SIZE;
    params.blockDim.z = 1;
    params.sharedMemBytes = 0;
    params.kernelParams = cudagraph_kernel_args;
    params.extra = NULL;

    unsigned int num_parent_node = 0;
    cudaGraphNode_t *parent_node_arr = NULL;
    if (ninst->num_parent_ninsts != 0)
        parent_node_arr = (cudaGraphNode_t*)malloc (ninst->num_parent_ninsts * sizeof(cudaGraphNode_t));
    for (int i = 0; i < ninst->num_parent_ninsts; i++)
    {
        ninst_t *parent_ninst = ninst->parent_ninst_idx_arr[i] + ldata->nasm->ninst_arr;
        if (parent_ninst->ldata->layer->type != INPUT_LAYER)
        {
            parent_node_arr[num_parent_node] = parent_ninst->cudagraph_node;
            num_parent_node++;
        }
    }
    if (check_CUDA(cudaGraphAddKernelNode(&ninst->cudagraph_node, cuda_graph, 
        parent_node_arr, num_parent_node, &params)) != 0)
        FPRT (stderr, "Error in adding kernel node for matmul. Layer %d (%s), Ninst idx: %d.\n", 
            layer->layer_idx, layer_type_str[layer->type], ninst->ninst_idx);
    if (parent_node_arr != NULL)
        free (parent_node_arr);
}

void add_cudagraph_node_maxpool2d (cudaGraph_t cuda_graph, ninst_t *ninst, int gpu_idx)
{
    nasm_ldata_t *ldata = ninst->ldata;
    aspen_layer_t *layer = ninst->ldata->layer;
    // prepare_input (ninst, dse->scratchpad);
    // void **input_ptr_arr = dse->scratchpad;   
    unsigned int input_pos_per_n = ninst->num_input_pos/ninst->tile_dims[OUT_W];
    unsigned int M = ninst->tile_dims[OUT_H];
    unsigned int N = ninst->tile_dims[OUT_W];
    unsigned int ldc = ldata->out_mat_stride;
    void *C = get_ninst_out_mem (ninst);

    

}
void add_cudagraph_node_avgpool2d (cudaGraph_t cuda_graph, ninst_t *ninst, int gpu_idx)
{
    nasm_ldata_t *ldata = ninst->ldata;
    aspen_layer_t *layer = ninst->ldata->layer;
    // prepare_input (ninst, dse->scratchpad);
    // void **input_ptr_arr = dse->scratchpad;   
    unsigned int input_pos_per_n = ninst->num_input_pos/ninst->tile_dims[OUT_W];
    unsigned int M = ninst->tile_dims[OUT_H];
    unsigned int N = ninst->tile_dims[OUT_W];
    unsigned int ldc = ldata->out_mat_stride;
    void *C = get_ninst_out_mem (ninst);

    

}
void add_cudagraph_node_fully_connected (cudaGraph_t cuda_graph, ninst_t *ninst, int gpu_idx)
{
    

}
void add_cudagraph_node_residual (cudaGraph_t cuda_graph, ninst_t *ninst, int gpu_idx)
{
    nasm_ldata_t *ldata = ninst->ldata;
    aspen_layer_t *layer = ninst->ldata->layer;
    nasm_ldata_t *p0_ldata = (ldata->parent_ldata_idx_arr[PARENT_0] + ldata->nasm->ldata_arr);
    nasm_ldata_t *p1_ldata = (ldata->parent_ldata_idx_arr[PARENT_1] + ldata->nasm->ldata_arr);
    unsigned int M = ninst->tile_dims[OUT_H];
    unsigned int N = ninst->tile_dims[OUT_W];
    unsigned int ldc = ldata->out_mat_stride;
    void *C = get_ninst_out_mem (ninst);
    unsigned int w_pos = ninst->out_mat_pos[OUT_W];
    float *input_0 = (float*)p0_ldata->out_mat + w_pos * p0_ldata->out_mat_stride + ninst->out_mat_pos[OUT_H];
    float *input_1 = (float*)p1_ldata->out_mat + w_pos * p1_ldata->out_mat_stride + ninst->out_mat_pos[OUT_H];
    
    // dim3 gridDim (N/_BLOCK_TILED_RESIDUAL_SIZE + ((N%_BLOCK_TILED_RESIDUAL_SIZE) > 0), M/_BLOCK_TILED_RESIDUAL_SIZE + ((M%_BLOCK_TILED_RESIDUAL_SIZE) > 0), 1);
    // dim3 blockDim (_BLOCK_TILED_RESIDUAL_SIZE, _BLOCK_TILED_RESIDUAL_SIZE, 1);
    // cuda_tiled_residual_kernel<<<gridDim, blockDim, 0, stream>>> (input_0, input_1, C, N, M, ldc);
    
    void *cudagraph_kernel_args [7] = {&input_0, &input_1, &C, &N, &M, &ldc, &layer->activation};
    struct cudaKernelNodeParams params = {0};
    params.func = (void*)cuda_tiled_residual_kernel;
    params.gridDim.x = N/_BLOCK_TILED_RESIDUAL_SIZE + ((N%_BLOCK_TILED_RESIDUAL_SIZE) > 0);
    params.gridDim.y = M/_BLOCK_TILED_RESIDUAL_SIZE + ((M%_BLOCK_TILED_RESIDUAL_SIZE) > 0);
    params.gridDim.z = 1;
    params.blockDim.x = _BLOCK_TILED_RESIDUAL_SIZE;
    params.blockDim.y = _BLOCK_TILED_RESIDUAL_SIZE;
    params.blockDim.z = 1;
    params.sharedMemBytes = 0;
    params.kernelParams = cudagraph_kernel_args;
    params.extra = NULL;

    unsigned int num_parent_node = 0;
    cudaGraphNode_t *parent_node_arr = NULL;
    if (ninst->num_parent_ninsts != 0)
        parent_node_arr = (cudaGraphNode_t*)malloc (ninst->num_parent_ninsts * sizeof(cudaGraphNode_t));
    for (int i = 0; i < ninst->num_parent_ninsts; i++)
    {
        ninst_t *parent_ninst = ninst->parent_ninst_idx_arr[i] + ldata->nasm->ninst_arr;
        if (parent_ninst->ldata->layer->type != INPUT_LAYER)
        {
            parent_node_arr[num_parent_node] = parent_ninst->cudagraph_node;
            num_parent_node++;
        }
    }
    if (check_CUDA(cudaGraphAddKernelNode(&ninst->cudagraph_node, cuda_graph, 
        parent_node_arr, num_parent_node, &params)) != 0)
        FPRT (stderr, "Error in adding kernel node for matmul. Layer %d (%s), Ninst idx: %d.\n", 
            layer->layer_idx, layer_type_str[layer->type], ninst->ninst_idx);
    if (parent_node_arr != NULL)
        free (parent_node_arr);
}

void add_cudagraph_node_softmax (cudaGraph_t cuda_graph, ninst_t *ninst, int gpu_idx)
{
    nasm_ldata_t *ldata = ninst->ldata;
    nasm_ldata_t *p0_ldata = (ldata->parent_ldata_idx_arr[PARENT_0] + ldata->nasm->ldata_arr);
    unsigned int M = ninst->tile_dims[OUT_H];
    unsigned int N = ninst->tile_dims[OUT_W];
    unsigned int ldc = ldata->out_mat_stride;
    void *C = get_ninst_out_mem (ninst);

    
    
}

void add_cudagraph_node_layernorm (cudaGraph_t cuda_graph, ninst_t *ninst, int gpu_idx)
{
    nasm_ldata_t *ldata = ninst->ldata;
    aspen_layer_t *layer = ninst->ldata->layer;
    nasm_ldata_t *p_ldata = (ldata->parent_ldata_idx_arr[PARENT_0] + ldata->nasm->ldata_arr);
    unsigned int M = ninst->tile_dims[OUT_H];
    unsigned int N = ninst->tile_dims[OUT_W];
    unsigned int ldb = p_ldata->out_mat_stride;
    unsigned int ldc = ldata->out_mat_stride;
    float *weight = (float*)layer->tensors[WEIGHT_TENSOR]->data_gpu[gpu_idx] + ninst->out_mat_pos[OUT_H];
    float *bias =(float*)layer->tensors[BIAS_TENSOR]->data_gpu[gpu_idx] + ninst->out_mat_pos[OUT_H];
    void *B = (char*)p_ldata->out_mat + (ninst->out_mat_pos[OUT_W] * ldb * layer->dnn->element_size);
    void *C = get_ninst_out_mem (ninst);

    // dim3 prob_gridDim (N/_BLOCK_LAYERNORM_SIZE + ((N%_BLOCK_LAYERNORM_SIZE) > 0), 1, 1);
    // dim3 prob_blockDim (_BLOCK_LAYERNORM_SIZE, 1, 1);
    // cuda_layernorm_kernel<<<prob_gridDim, prob_blockDim, 0, stream>>> (B, weight, bias, C, N, M, ldb, ldc);

    void *cudagraph_kernel_args [8] = {&B, &weight, &bias, &C, &N, &M, &ldb, &ldc};
    struct cudaKernelNodeParams params = {0};
    params.func = (void*)cuda_layernorm_kernel;
    params.gridDim.x = N/_BLOCK_LAYERNORM_SIZE + ((N%_BLOCK_LAYERNORM_SIZE) > 0);
    params.gridDim.y = 1;
    params.gridDim.z = 1;
    params.blockDim.x = _BLOCK_LAYERNORM_SIZE;
    params.blockDim.y = 1;
    params.blockDim.z = 1;
    params.sharedMemBytes = 0;
    params.kernelParams = cudagraph_kernel_args;
    params.extra = NULL;

    unsigned int num_parent_node = 0;
    cudaGraphNode_t *parent_node_arr = NULL;
    if (ninst->num_parent_ninsts != 0)
        parent_node_arr = (cudaGraphNode_t*)malloc (ninst->num_parent_ninsts * sizeof(cudaGraphNode_t));
    for (int i = 0; i < ninst->num_parent_ninsts; i++)
    {
        ninst_t *parent_ninst = ninst->parent_ninst_idx_arr[i] + ldata->nasm->ninst_arr;
        if (parent_ninst->ldata->layer->type != INPUT_LAYER)
        {
            parent_node_arr[num_parent_node] = parent_ninst->cudagraph_node;
            num_parent_node++;
        }
    }
    if (check_CUDA(cudaGraphAddKernelNode(&ninst->cudagraph_node, cuda_graph, 
        parent_node_arr, num_parent_node, &params)) != 0)
        FPRT (stderr, "Error in adding kernel node for matmul. Layer %d (%s), Ninst idx: %d.\n", 
            layer->layer_idx, layer_type_str[layer->type], ninst->ninst_idx);
    if (parent_node_arr != NULL)
        free (parent_node_arr);
}
void add_cudagraph_node_k_attention (cudaGraph_t cuda_graph, ninst_t *ninst, int gpu_idx)
{
    nasm_ldata_t *ldata = ninst->ldata;
    aspen_layer_t *layer = ninst->ldata->layer;
    nasm_ldata_t *p_ldata = (ldata->parent_ldata_idx_arr[PARENT_0] + ldata->nasm->ldata_arr);
    nasm_ldata_t *pk_ldata = (ldata->parent_ldata_idx_arr[PARENT_1] + ldata->nasm->ldata_arr);
    unsigned int num_hidden = layer->params[NUM_HIDDEN];
    unsigned int num_heads = layer->params[NUM_HEAD];
    unsigned int hidden_per_head = num_hidden / num_heads;
    unsigned int num_seq = ldata->nasm->tr_seq_len;
    unsigned int batch = ninst->out_mat_pos[OUT_W] / (num_seq * num_heads);
    unsigned int head = (ninst->out_mat_pos[OUT_W] % (num_seq * num_heads)) / num_seq;
    unsigned int M = ninst->tile_dims[OUT_H];
    unsigned int N = ninst->tile_dims[OUT_W];
    unsigned int K = layer->params[MAT_K];
    unsigned int ldk = pk_ldata->out_mat_stride;
    unsigned int lda = K;
    unsigned int ldb = p_ldata->out_mat_stride;
    unsigned int ldc = ldata->out_mat_stride;
    // void *A = dse->scratchpad;
    void *key_head = (float*)pk_ldata->out_mat + 
        (batch * ldk * M + head * K + ninst->out_mat_pos[OUT_H]);
    void *B_head = (float*)p_ldata->out_mat + (batch * ldb * num_seq + head * hidden_per_head +
        + (ninst->out_mat_pos[OUT_W] % num_seq) * ldb);
    void *C_head = get_ninst_out_mem (ninst);
    
    unsigned int n = 0;

    // dim3 gridDim (M/_BLOCK_M_SIZE + ((M%_BLOCK_M_SIZE) > 0), N/_BLOCK_N_SIZE + ((N%_BLOCK_N_SIZE) > 0), 1);
    // dim3 blockDim ((_BLOCK_M_SIZE / _THREAD_M_SIZE), (_BLOCK_N_SIZE / _THREAD_N_SIZE), 1);
    // cuda_tiled_k_matmul_kernel<<<gridDim, blockDim, 0, stream>>> (
    //     M, N, K, key_head, ldk, B_head, ldb, C_head, ldc);
    // dim3 prob_gridDim (N/_BLOCK_ATT_K_PROB_SIZE + ((N%_BLOCK_ATT_K_PROB_SIZE) > 0), 1, 1);
    // dim3 prob_blockDim (_BLOCK_ATT_K_PROB_SIZE, 1, 1);
    // cuda_tiled_k_prob_kernel<<<prob_gridDim, prob_blockDim, 0, stream>>> (
    //     M, N, K, key_head, ldk, B_head, ldb, C_head, ldc);

    cudaGraphNode_t matmul_node;
    void *cudagraph_kernel_args [9] = {&M, &N, &K, &key_head, &ldk, &B_head, &ldb, &C_head, &ldc};
    struct cudaKernelNodeParams params = {0};
    params.func = (void*)cuda_tiled_k_matmul_kernel;
    params.gridDim.x = M/_BLOCK_M_SIZE + ((M%_BLOCK_M_SIZE) > 0);
    params.gridDim.y = N/_BLOCK_N_SIZE + ((N%_BLOCK_N_SIZE) > 0);
    params.gridDim.z = 1;
    params.blockDim.x = (_BLOCK_M_SIZE / _THREAD_M_SIZE);
    params.blockDim.y = (_BLOCK_N_SIZE / _THREAD_N_SIZE);
    params.blockDim.z = 1;
    params.sharedMemBytes = 0;
    params.kernelParams = cudagraph_kernel_args;
    params.extra = NULL;

    unsigned int num_parent_node = 0;
    cudaGraphNode_t *parent_node_arr = NULL;
    if (ninst->num_parent_ninsts != 0)
        parent_node_arr = (cudaGraphNode_t*)malloc (ninst->num_parent_ninsts * sizeof(cudaGraphNode_t));
    for (int i = 0; i < ninst->num_parent_ninsts; i++)
    {
        ninst_t *parent_ninst = ninst->parent_ninst_idx_arr[i] + ldata->nasm->ninst_arr;
        if (parent_ninst->ldata->layer->type != INPUT_LAYER)
        {
            parent_node_arr[num_parent_node] = parent_ninst->cudagraph_node;
            num_parent_node++;
        }
    }
    if (check_CUDA(cudaGraphAddKernelNode(&matmul_node, cuda_graph, 
        parent_node_arr, num_parent_node, &params)) != 0)
        FPRT (stderr, "Error in adding kernel node for matmul. Layer %d (%s), Ninst idx: %d.\n", 
            layer->layer_idx, layer_type_str[layer->type], ninst->ninst_idx);
    

    
    params.func = (void*)cuda_tiled_k_prob_kernel;
    params.gridDim.x = N/_BLOCK_ATT_K_PROB_SIZE + ((N%_BLOCK_ATT_K_PROB_SIZE) > 0);
    params.gridDim.y = 1;
    params.gridDim.z = 1;
    params.blockDim.x = _BLOCK_ATT_K_PROB_SIZE;
    params.blockDim.y = 1;
    params.blockDim.z = 1;
    params.sharedMemBytes = 0;
    params.kernelParams = cudagraph_kernel_args;
    params.extra = NULL;
    num_parent_node = 1;
    parent_node_arr[0] = matmul_node;
    if (check_CUDA(cudaGraphAddKernelNode(&ninst->cudagraph_node, cuda_graph, 
        parent_node_arr, num_parent_node, &params)) != 0)
        FPRT (stderr, "Error in adding kernel node for matmul. Layer %d (%s), Ninst idx: %d.\n", 
            layer->layer_idx, layer_type_str[layer->type], ninst->ninst_idx);

    if (parent_node_arr != NULL)
        free (parent_node_arr);
}

void add_cudagraph_node_v_attention (cudaGraph_t cuda_graph, ninst_t *ninst, int gpu_idx)
{
    nasm_ldata_t *ldata = ninst->ldata;
    aspen_layer_t *layer = ninst->ldata->layer;
    nasm_ldata_t *p_ldata = (ldata->parent_ldata_idx_arr[PARENT_0] + ldata->nasm->ldata_arr);
    nasm_ldata_t *pv_ldata = (ldata->parent_ldata_idx_arr[PARENT_1] + ldata->nasm->ldata_arr);
    unsigned int num_hidden = layer->params[NUM_HIDDEN];
    unsigned int num_heads = layer->params[NUM_HEAD];
    unsigned int hidden_per_head = num_hidden / num_heads;
    unsigned int num_seq = ldata->nasm->tr_seq_len;
    unsigned int batch = ninst->out_mat_pos[OUT_W] / num_seq;
    unsigned int head = ninst->out_mat_pos[OUT_H]  / hidden_per_head;
    unsigned int M = ninst->tile_dims[OUT_H];
    unsigned int N = ninst->tile_dims[OUT_W];
    unsigned int K = num_seq;
    unsigned int ldv = pv_ldata->out_mat_stride;
    unsigned int lda = K;
    unsigned int ldb = p_ldata->out_mat_stride;
    unsigned int ldc = ldata->out_mat_stride;
    // void *A = dse->scratchpad;
    float *val_head = (float*)pv_ldata->out_mat + batch * ldv * num_seq + ninst->out_mat_pos[OUT_H];
    void *B_head = (float*)p_ldata->out_mat + (batch * num_heads * num_seq + head * num_seq +
        + (ninst->out_mat_pos[OUT_W] % num_seq)) * ldb;
    void *C_head = get_ninst_out_mem (ninst);

    // dim3 gridDim (M/_BLOCK_M_SIZE + ((M%_BLOCK_M_SIZE) > 0), N/_BLOCK_N_SIZE + ((N%_BLOCK_N_SIZE) > 0), 1);
    // dim3 blockDim ((_BLOCK_M_SIZE / _THREAD_M_SIZE), (_BLOCK_N_SIZE / _THREAD_N_SIZE), 1);
    // cuda_tiled_v_attention_kernel<<<gridDim, blockDim, 0, stream>>> (
    //     M, N, K, val_head, ldv, B_head, ldb, C_head, ldc);

    void *cudagraph_kernel_args [9] = {&M, &N, &K, &val_head, &ldv, &B_head, &ldb, &C_head, &ldc};
    struct cudaKernelNodeParams params = {0};
    params.func = (void*)cuda_tiled_v_attention_kernel;
    params.gridDim.x = M/_BLOCK_M_SIZE + ((M%_BLOCK_M_SIZE) > 0);
    params.gridDim.y = N/_BLOCK_N_SIZE + ((N%_BLOCK_N_SIZE) > 0);
    params.gridDim.z = 1;
    params.blockDim.x = (_BLOCK_M_SIZE / _THREAD_M_SIZE);
    params.blockDim.y = (_BLOCK_N_SIZE / _THREAD_N_SIZE);
    params.blockDim.z = 1;
    params.sharedMemBytes = 0;
    params.kernelParams = cudagraph_kernel_args;
    params.extra = NULL;

    unsigned int num_parent_node = 0;
    cudaGraphNode_t *parent_node_arr = NULL;
    if (ninst->num_parent_ninsts != 0)
        parent_node_arr = (cudaGraphNode_t*)malloc (ninst->num_parent_ninsts * sizeof(cudaGraphNode_t));
    for (int i = 0; i < ninst->num_parent_ninsts; i++)
    {
        ninst_t *parent_ninst = ninst->parent_ninst_idx_arr[i] + ldata->nasm->ninst_arr;
        if (parent_ninst->ldata->layer->type != INPUT_LAYER)
        {
            parent_node_arr[num_parent_node] = parent_ninst->cudagraph_node;
            num_parent_node++;
        }
    }
    if (check_CUDA(cudaGraphAddKernelNode(&ninst->cudagraph_node, cuda_graph, 
        parent_node_arr, num_parent_node, &params)) != 0)
        FPRT (stderr, "Error in adding kernel node for matmul. Layer %d (%s), Ninst idx: %d.\n", 
            layer->layer_idx, layer_type_str[layer->type], ninst->ninst_idx);
    if (parent_node_arr != NULL)
        free (parent_node_arr);
}

void print_nasm_cudagraph_info (nasm_t *nasm, char* output_dot_filename)
{
    cudaGraphDebugDotPrint (nasm->cuda_graph, output_dot_filename, cudaGraphDebugDotFlagsVerbose);

}
#endif