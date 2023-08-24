#include "kernels.h"

#define _SKIP_KERNELS 0

void *prepare_im2col (ninst_t *ninst, void *buffer)
{
    nasm_ldata_t *ldata = ninst->ldata;
    aspen_layer_t *layer = ldata->layer;
    nasm_ldata_t *p_ldata = (ldata->parent_ldata_idx_arr[PARENT_0] + ldata->nasm->ldata_arr);
    aspen_layer_t *p_layer = (ldata->parent_ldata_idx_arr[PARENT_0] + ldata->nasm->ldata_arr)->layer;
    unsigned int parent_stride = p_ldata->out_mat_stride;
    void **input_ptr_arr = buffer;
    unsigned int num_idx = 0;
    if (layer->type == CONV_LAYER || layer->type == MAXPOOL_LAYER || layer->type == AVGPOOL_LAYER)
    {
        unsigned int mat_w = ninst->out_mat_pos[OUT_W];
        for (; mat_w < ninst->out_mat_pos[OUT_W] + ninst->tile_dims[OUT_W]; mat_w++)
        {
            unsigned int out_b = mat_w / (layer->params[OUT_H] * layer->params[OUT_W]); 
            unsigned int out_h = (mat_w % (layer->params[OUT_H] * layer->params[OUT_W])) / layer->params[OUT_W];
            unsigned int out_w = mat_w % layer->params[OUT_W];
            unsigned int in_b = out_b;
            for (int kh = 0; kh < layer->params[WEIGHT_H]; kh++)
            {
                for (int kw = 0; kw < layer->params[WEIGHT_W]; kw++)
                {
                    int in_h = out_h * layer->params[STRIDE] + kh  - layer->params[PADDING];
                    int in_w = out_w * layer->params[STRIDE] + kw  - layer->params[PADDING];
                    if (in_h < 0 || in_h >= p_layer->params[OUT_H] || in_w < 0 || in_w >= p_layer->params[OUT_W])
                    {
                        input_ptr_arr[num_idx++] = NULL;
                        continue;
                    }
                    unsigned int in_idx = in_b * p_layer->params[OUT_H] * p_layer->params[OUT_W] 
                        + in_h * p_layer->params[OUT_W] + in_w;
                    input_ptr_arr[num_idx++] = (char*)p_ldata->out_mat + in_idx*parent_stride * layer->dnn->element_size;
                }
            }
        }
    }
    else
    {
        FPRT(stderr, "ERROR: Unsupported layer type %s, at line %d in file %s\n" , layer_type_str[layer->type], 0, " ");
        assert (0);
    }
    return (void *) (input_ptr_arr + num_idx);
}

void load_im2col (float* output, unsigned int out_stride,
    int *pos_idx_arr, float* input_arr, unsigned int input_stride,
    unsigned int N, unsigned int input_col_num_per_n, unsigned int input_col_size, 
    unsigned int starting_k, unsigned int ending_k)
{
    unsigned int starting_col_idx = starting_k / input_col_size;
    unsigned int ending_col_idx = ending_k / input_col_size;
    for (int n = 0; n < N; n++)
    {
        float *out_ptr = output + n * out_stride;
        for (int col_idx = starting_col_idx; col_idx <= ending_col_idx; col_idx++)
        {
            unsigned int pos_idx = pos_idx_arr[n*input_col_num_per_n + col_idx];
            if (pos_idx == -1)
            {
                if (col_idx == starting_col_idx)
                {
                    unsigned int starting_k_in_col = starting_k % input_col_size;
                    memset (out_ptr, 0, (input_col_size - starting_k_in_col) * sizeof(float));
                    out_ptr += (input_col_size - starting_k_in_col);
                }
                else if (col_idx == ending_col_idx)
                {
                    unsigned int ending_k_in_col = ending_k % input_col_size;
                    memset (out_ptr, 0, ending_k_in_col * sizeof(float));
                }
                else
                {
                    memset (out_ptr, 0, input_col_size * sizeof(float));
                    out_ptr += input_col_size;
                }
                continue;
            }
            else
            {
                float *input = input_arr + pos_idx * input_stride;
                if (col_idx == starting_col_idx)
                {
                    unsigned int starting_k_in_col = starting_k % input_col_size;
                    memcpy (out_ptr, input + starting_k_in_col, (input_col_size - starting_k_in_col) * sizeof(float));
                    out_ptr += (input_col_size - starting_k_in_col);
                }
                else if (col_idx == ending_col_idx)
                {
                    unsigned int ending_k_in_col = ending_k % input_col_size;
                    memcpy (out_ptr, input, ending_k_in_col * sizeof(float));
                }
                else
                {
                    memcpy (out_ptr, input, input_col_size * sizeof(float));
                    out_ptr += input_col_size;
                }
            }
        }
    }
}

void load_im2col_ptr (float* output, unsigned int out_stride,
    void **pos_ptr_arr, float* input_arr, unsigned int input_stride,
    unsigned int N, unsigned int input_col_num_per_n, unsigned int input_col_size, 
    unsigned int starting_k, unsigned int ending_k)
{
    unsigned int starting_col_idx = starting_k / input_col_size;
    unsigned int ending_col_idx = ending_k / input_col_size;
    for (int n = 0; n < N; n++)
    {
        float *out_ptr = output + n * out_stride;
        for (int col_idx = starting_col_idx; col_idx <= ending_col_idx; col_idx++)
        {
            void* pos = pos_ptr_arr[n*input_col_num_per_n + col_idx];
            if (pos == NULL)
            {
                if (col_idx == starting_col_idx)
                {
                    unsigned int starting_k_in_col = starting_k % input_col_size;
                    memset (out_ptr, 0, (input_col_size - starting_k_in_col) * sizeof(float));
                    out_ptr += (input_col_size - starting_k_in_col);
                }
                else if (col_idx == ending_col_idx)
                {
                    unsigned int ending_k_in_col = ending_k % input_col_size;
                    memset (out_ptr, 0, ending_k_in_col * sizeof(float));
                }
                else
                {
                    memset (out_ptr, 0, input_col_size * sizeof(float));
                    out_ptr += input_col_size;
                }
                continue;
            }
            else
            {
                float *input = (float*)pos;
                if (col_idx == starting_col_idx)
                {
                    unsigned int starting_k_in_col = starting_k % input_col_size;
                    memcpy (out_ptr, input + starting_k_in_col, (input_col_size - starting_k_in_col) * sizeof(float));
                    out_ptr += (input_col_size - starting_k_in_col);
                }
                else if (col_idx == ending_col_idx)
                {
                    unsigned int ending_k_in_col = ending_k % input_col_size;
                    memcpy (out_ptr, input, ending_k_in_col * sizeof(float));
                }
                else
                {
                    memcpy (out_ptr, input, input_col_size * sizeof(float));
                    out_ptr += input_col_size;
                }
            }
        }
    }
}

void tiled_conv2d (ninst_t *ninst, dse_t *dse)
{
    #if _SKIP_KERNELS == 0
    nasm_ldata_t *ldata = ninst->ldata;
    aspen_layer_t *layer = ninst->ldata->layer;
    nasm_ldata_t *p_ldata = (ldata->parent_ldata_idx_arr[PARENT_0] + ldata->nasm->ldata_arr);
    void *scratchpad = prepare_im2col (ninst, dse->scratchpad);
    unsigned int input_col_size = p_ldata->out_mat_dims[OUT_H];
    void **input_pos_arr = dse->scratchpad;   
    const unsigned int input_pos_per_n = ninst->num_input_pos/ninst->tile_dims[OUT_W];
    const unsigned int M = ninst->tile_dims[OUT_H];
    const unsigned int N = ninst->tile_dims[OUT_W];
    const unsigned int K = layer->params[WEIGHT_H] * layer->params[WEIGHT_W] * layer->params[IN_C];
    const unsigned int lda = K;
    unsigned int ldb = K;
    const unsigned int ldc = ldata->out_mat_stride;
    const void *A = (float*)layer->tensors[WEIGHT_TENSOR]->data + ninst->out_mat_pos[OUT_H] * lda;
    void *B = (char *) scratchpad;
    void *C = get_ninst_out_mem (ninst);
    const unsigned int rem_n = N % _TILE_SIZE_N;
    const unsigned int rem_m = M % _TILE_SIZE_M;
    const unsigned int rem_k = K % _TILE_SIZE_K;
    unsigned int n = 0;
    if (dse->gpu_idx < 0)
    {
        // <M, N, K> = <M, _TILE_SIZE_N, K>
        for (; n < N - rem_n; n += _TILE_SIZE_N)
        {
            for (int nn = 0; nn < _TILE_SIZE_N; nn++)
            {
                memset ((float*)C + (ldc * (n + nn)), 0, M * sizeof(float));
            }
            unsigned int k = 0;
            // <M, N, K> = <M, _TILE_SIZE_N, _TILE_SIZE_K>
            for (; k < K - rem_k; k += _TILE_SIZE_K)
            {
                ldb = _TILE_SIZE_K;
                load_im2col_ptr (B, ldb, 
                    input_pos_arr + input_pos_per_n * n, p_ldata->out_mat, p_ldata->out_mat_stride, 
                    _TILE_SIZE_N, input_pos_per_n, input_col_size, 
                    k, k + _TILE_SIZE_K);
                unsigned int m = 0;
                
                // <M, N, K> = <_TILE_SIZE_M, _TILE_SIZE_N, _TILE_SIZE_K>
                for (; m < M - rem_m; m += _TILE_SIZE_M)
                {
                    SGEMM_KERNEL_FULL_TILE (_TILE_SIZE_M, _TILE_SIZE_N, _TILE_SIZE_K, 
                        (float*)A + (m * lda + k*_VEC_SIZE_M), lda, (float*)B, ldb, (float*)C + (ldc * n + m), ldc);
                }
                // <M, N, K> = <rem_m, _TILE_SIZE_N, _TILE_SIZE_K>
                if (rem_m != 0)
                    SGEMM_KERNEL_TILE_N (rem_m, _TILE_SIZE_N, _TILE_SIZE_K, 
                            (float*)A + (m * lda + k*_VEC_SIZE_M), lda, (float*)B, ldb, (float*)C + (ldc * n + m), ldc);
            
            }
            // <M, N, K> = <M, _TILE_SIZE_N, rem_k>
            if (rem_k != 0)
            {
                ldb = rem_k;
                load_im2col_ptr (B, ldb, 
                    input_pos_arr + input_pos_per_n * n, p_ldata->out_mat, p_ldata->out_mat_stride, 
                    _TILE_SIZE_N, input_pos_per_n, input_col_size, 
                    k, K);

                unsigned int m = 0;
                // <M, N, K> = <_TILE_SIZE_M, _TILE_SIZE_N, rem_k>
                for (; m < M - rem_m; m += _TILE_SIZE_M)
                {
                    SGEMM_KERNEL_FULL_TILE (_TILE_SIZE_M, _TILE_SIZE_N, rem_k,
                        (float*)A + (m * lda + k*_VEC_SIZE_M), lda, (float*)B, ldb, (float*)C + (ldc * n + m), ldc);
                }
                // <M, N, K> = <rem_m, _TILE_SIZE_N, rem_k>
                SGEMM_KERNEL_TILE_N (rem_m, _TILE_SIZE_N, rem_k, 
                    (float*)A + (m * lda + k*_VEC_SIZE_M), lda, (float*)B, ldb, (float*)C + (ldc * n + m), ldc);
            }
            for (unsigned int nn = n; nn < n + _TILE_SIZE_N; nn++)
            {
                float *out_vec = (float *)C + nn * ldc;
                if (layer->tensors[BIAS_TENSOR] != NULL)
                {
                    float *bias = (float*)layer->tensors[BIAS_TENSOR]->data + ninst->out_mat_pos[OUT_H];
                    for (unsigned int m = 0; m < M; m++)
                    {
                        out_vec[m] += bias[m];
                    }
                }
                naive_activate (out_vec, M, layer->activation);
            }
        }
        // <M, N, K> = <M, rem_n, K>
        if (rem_n != 0)
        {
            for (int nn = 0; nn < rem_n; nn++)
            {
                memset ((float*)C + (ldc * (n + nn)), 0, M * sizeof(float));
            }
            unsigned int k = 0;
            // <M, N, K> = <M, rem_n, _TILE_SIZE_K>
            for (; k < K - rem_k; k += _TILE_SIZE_K)
            {
                ldb = _TILE_SIZE_K;
                load_im2col_ptr (B, ldb, 
                    input_pos_arr + input_pos_per_n * n, p_ldata->out_mat, p_ldata->out_mat_stride, 
                    rem_n, input_pos_per_n, input_col_size, 
                    k, k + _TILE_SIZE_K);
                unsigned int m = 0;
                // <M, N, K> = <_TILE_SIZE_M, rem_n, _TILE_SIZE_K>
                for (; m < M - rem_m; m += _TILE_SIZE_M)
                {
                    SGEMM_KERNEL_TILE_M (_TILE_SIZE_M, rem_n, _TILE_SIZE_K, 
                        (float*)A + (m * lda + k*_VEC_SIZE_M), lda, (float*)B, ldb, (float*)C + (ldc * n + m), ldc);
                }
                // <M, N, K> = <rem_m, rem_n, _TILE_SIZE_K>
                if (rem_m != 0)
                    SGEMM_KERNEL (rem_m, rem_n, _TILE_SIZE_K, 
                            (float*)A + (m * lda + k*_VEC_SIZE_M), lda, (float*)B, ldb, (float*)C + (ldc * n + m), ldc);
            }
            // <M, N, K> = <M, rem_n, rem_k>
            if (rem_k != 0)
            {
                ldb = rem_k;
                load_im2col_ptr (B, ldb, 
                    input_pos_arr + input_pos_per_n * n, p_ldata->out_mat, p_ldata->out_mat_stride, 
                    rem_n, input_pos_per_n, input_col_size, 
                    k, K);
                unsigned int m = 0;
                // <M, N, K> = <_TILE_SIZE_M, rem_n, rem_k>
                for (; m < M - rem_m; m += _TILE_SIZE_M)
                {
                    SGEMM_KERNEL_TILE_M (_TILE_SIZE_M, rem_n, rem_k,
                        (float*)A + (m * lda + k*_VEC_SIZE_M), lda, (float*)B, ldb, (float*)C + (ldc * n + m), ldc);
                }
                // <M, N, K> = <rem_m, rem_n, rem_k>
                SGEMM_KERNEL (rem_m, rem_n, rem_k, 
                    (float*)A + (m * lda + k*_VEC_SIZE_M), lda, (float*)B, ldb, (float*)C + (ldc * n + m), ldc);
            }
            for (unsigned int nn = n; nn < N; nn++)
            {
                float *out_vec = (float *)C + nn * ldc;
                if (layer->tensors[BIAS_TENSOR] != NULL)
                {
                    float *bias = (float*)layer->tensors[BIAS_TENSOR]->data + ninst->out_mat_pos[OUT_H];
                    for (unsigned int m = 0; m < M; m++)
                    {
                        out_vec[m] += bias[m];
                        
                    }
                }
                naive_activate (out_vec, M, layer->activation);
            }
        }
    }
    else
    {
        #ifdef GPU
        A = (float*)layer->tensors[WEIGHT_TENSOR]->data_gpu[dse->gpu_idx] + ninst->out_mat_pos[OUT_H] * lda;
        cuda_tiled_conv2d (M, N, ninst->input_pos_ptr_arr_gpu, input_pos_per_n, layer->params[IN_C],
            A, lda, p_ldata->out_mat, p_ldata->out_mat_stride, C, ldc, 
            (float*)layer->tensors[BIAS_TENSOR]->data_gpu[dse->gpu_idx] + ninst->out_mat_pos[OUT_H], layer->activation,
            aspen_CUDA_streams[dse->gpu_idx][dse->thread_id%GPU_RUN_STREAM_NUM]);
        aspen_sync_gpu_stream (dse->gpu_idx, dse->thread_id%GPU_RUN_STREAM_NUM);
        #endif
    }
    #endif
}

void tiled_matmul (ninst_t *ninst, dse_t *dse)
{
    #if _SKIP_KERNELS == 0
    nasm_ldata_t *ldata = ninst->ldata;
    aspen_layer_t *layer = ninst->ldata->layer;
    nasm_ldata_t *p_ldata = (ldata->parent_ldata_idx_arr[PARENT_0] + ldata->nasm->ldata_arr);
    const unsigned int M = ninst->tile_dims[OUT_H];
    const unsigned int N = ninst->tile_dims[OUT_W];
    const unsigned int K = layer->params[MAT_K];
    const unsigned int lda = K;
    const unsigned int ldb = p_ldata->out_mat_stride;
    const unsigned int ldc = ldata->out_mat_stride;
    const void *A = (char*)layer->tensors[WEIGHT_TENSOR]->data + (ninst->out_mat_pos[OUT_H] * lda * layer->dnn->element_size);
    const void *B = (char*)p_ldata->out_mat + (ninst->out_mat_pos[OUT_W] * ldb * layer->dnn->element_size);
    void *C = get_ninst_out_mem (ninst);
    if (dse->gpu_idx < 0)
    {
        const unsigned int rem_n = N % _TILE_SIZE_N;
        const unsigned int rem_m = M % _TILE_SIZE_M;
        const unsigned int rem_k = K % _TILE_SIZE_K;
        unsigned int n = 0;
        // <M, N, K> = <M, _TILE_SIZE_N, K>
        for (; n < N - rem_n; n += _TILE_SIZE_N)
        {
            for (int nn = 0; nn < _TILE_SIZE_N; nn++)
            {
                memset ((float*)C + (ldc * (n + nn)), 0, M * sizeof(float));
            }
            unsigned int k = 0;
            // <M, N, K> = <M, _TILE_SIZE_N, _TILE_SIZE_K>
            for (; k < K - rem_k; k += _TILE_SIZE_K)
            {
                unsigned int m = 0;
                // <M, N, K> = <_TILE_SIZE_M, _TILE_SIZE_N, _TILE_SIZE_K>
                for (; m < M - rem_m; m += _TILE_SIZE_M)
                {
                    SGEMM_KERNEL_FULL_TILE (_TILE_SIZE_M, _TILE_SIZE_N, _TILE_SIZE_K, 
                        (float*)A + (m * lda + k*_VEC_SIZE_M), lda, (float*)B + (n * ldb + k), ldb, (float*)C + (ldc * n + m), ldc);
                }
                // <M, N, K> = <rem_m, _TILE_SIZE_N, _TILE_SIZE_K>
                if (rem_m != 0)
                    SGEMM_KERNEL_TILE_N (rem_m, _TILE_SIZE_N, _TILE_SIZE_K, 
                            (float*)A + (m * lda + k*_VEC_SIZE_M), lda, (float*)B + (n * ldb + k), ldb, (float*)C + (ldc * n + m), ldc);
            
            }
            // <M, N, K> = <M, _TILE_SIZE_N, rem_k>
            if (rem_k != 0)
            {
                unsigned int m = 0;
                // <M, N, K> = <_TILE_SIZE_M, _TILE_SIZE_N, rem_k>
                for (; m < M - rem_m; m += _TILE_SIZE_M)
                {
                    SGEMM_KERNEL_FULL_TILE (_TILE_SIZE_M, _TILE_SIZE_N, rem_k,
                        (float*)A + (m * lda + k*_VEC_SIZE_M), lda, (float*)B + (n * ldb + k), ldb, (float*)C + (ldc * n + m), ldc);
                }
                // <M, N, K> = <rem_m, _TILE_SIZE_N, rem_k>
                SGEMM_KERNEL_TILE_N (rem_m, _TILE_SIZE_N, rem_k, 
                    (float*)A + (m * lda + k*_VEC_SIZE_M), lda, (float*)B + (n * ldb + k), ldb, (float*)C + (ldc * n + m), ldc);
            }
            for (unsigned int nn = n; nn < n + _TILE_SIZE_N; nn++)
            {
                float *out_vec = (float *)C + nn * ldc;
                if (layer->tensors[BIAS_TENSOR] != NULL)
                {
                    float *bias = (float*)layer->tensors[BIAS_TENSOR]->data + ninst->out_mat_pos[OUT_H];
                    for (unsigned int m = 0; m < M; m++)
                    {
                        out_vec[m] += bias[m];
                    }
                }
                naive_activate (out_vec, M, layer->activation);
            }
        }
        // <M, N, K> = <M, rem_n, K>
        if (rem_n != 0)
        {
            for (int nn = 0; nn < rem_n; nn++)
            {
                memset ((float*)C + (ldc * (n + nn)), 0, M * sizeof(float));
            }
            unsigned int k = 0;
            // <M, N, K> = <M, rem_n, _TILE_SIZE_K>
            for (; k < K - rem_k; k += _TILE_SIZE_K)
            {
                unsigned int m = 0;
                // <M, N, K> = <_TILE_SIZE_M, rem_n, _TILE_SIZE_K>
                for (; m < M - rem_m; m += _TILE_SIZE_M)
                {
                    SGEMM_KERNEL_TILE_M (_TILE_SIZE_M, rem_n, _TILE_SIZE_K, 
                        (float*)A + (m * lda + k*_VEC_SIZE_M), lda, (float*)B + (n * ldb + k), ldb, (float*)C + (ldc * n + m), ldc);
                }
                // <M, N, K> = <rem_m, rem_n, _TILE_SIZE_K>
                if (rem_m != 0)
                    SGEMM_KERNEL (rem_m, rem_n, _TILE_SIZE_K, 
                            (float*)A + (m * lda + k*_VEC_SIZE_M), lda, (float*)B + (n * ldb + k), ldb, (float*)C + (ldc * n + m), ldc);
            }
            // <M, N, K> = <M, rem_n, rem_k>
            if (rem_k != 0)
            {
                unsigned int m = 0;
                // <M, N, K> = <_TILE_SIZE_M, rem_n, rem_k>
                for (; m < M - rem_m; m += _TILE_SIZE_M)
                {
                    SGEMM_KERNEL_TILE_M (_TILE_SIZE_M, rem_n, rem_k,
                        (float*)A + (m * lda + k*_VEC_SIZE_M), lda, (float*)B + (n * ldb + k), ldb, (float*)C + (ldc * n + m), ldc);
                }
                // <M, N, K> = <rem_m, rem_n, rem_k>
                SGEMM_KERNEL (rem_m, rem_n, rem_k, 
                    (float*)A + (m * lda + k*_VEC_SIZE_M), lda, (float*)B + (n * ldb + k), ldb, (float*)C + (ldc * n + m), ldc);
            }
            for (unsigned int nn = n; nn < N; nn++)
            {
                float *out_vec = (float *)C + nn * ldc;
                if (layer->tensors[BIAS_TENSOR] != NULL)
                {
                    float *bias = (float*)layer->tensors[BIAS_TENSOR]->data + ninst->out_mat_pos[OUT_H];
                    for (unsigned int m = 0; m < M; m++)
                    {
                        out_vec[m] += bias[m];
                    }
                }
                naive_activate (out_vec, M, layer->activation);
            }
        }
    }
    else
    {
        #ifdef GPU
        float *bias = (float*)layer->tensors[BIAS_TENSOR]->data_gpu[dse->gpu_idx] + ninst->out_mat_pos[OUT_H];
        A = (char*)layer->tensors[WEIGHT_TENSOR]->data_gpu[dse->gpu_idx] 
            + (ninst->out_mat_pos[OUT_H] * lda * layer->dnn->element_size);
        cuda_matmul (M, N, K, A, lda, B, ldb, C, ldc, bias, layer->activation, 
            aspen_CUDA_streams[dse->gpu_idx][dse->thread_id%GPU_RUN_STREAM_NUM]);
        aspen_sync_gpu_stream (dse->gpu_idx, dse->thread_id%GPU_RUN_STREAM_NUM);
        #endif
    }
    #endif
}

void tiled_maxpool2d (ninst_t *ninst, dse_t *dse)
{
    #if _SKIP_KERNELS == 0
    nasm_ldata_t *ldata = ninst->ldata;
    aspen_layer_t *layer = ninst->ldata->layer;
    prepare_im2col (ninst, dse->scratchpad);
    void **input_ptr_arr = dse->scratchpad;   
    const unsigned int input_pos_per_n = ninst->num_input_pos/ninst->tile_dims[OUT_W];
    const unsigned int M = ninst->tile_dims[OUT_H];
    const unsigned int N = ninst->tile_dims[OUT_W];
    const unsigned int ldc = ldata->out_mat_stride;
    void *C = get_ninst_out_mem (ninst);
    if (dse->gpu_idx < 0)
    {
        for (int n = 0; n < N; n++)
        {
            float *out_vec = (float*)C + n * ldc;
            float *input_vec = (float *)input_ptr_arr[n*input_pos_per_n];
            if (input_vec == NULL)
            {
                for (int m = 0; m < M; m++)
                {
                    out_vec[m] = -INFINITY;
                }
            }
            else
            {
                input_vec += ninst->out_mat_pos[OUT_H];
                memcpy (out_vec, input_vec, M * layer->dnn->element_size);
            }
            for (int i = 1; i < input_pos_per_n; i++)
            {
                input_vec = (float *)input_ptr_arr[n*input_pos_per_n + i];
                if (input_vec != NULL)
                {
                    input_vec += + ninst->out_mat_pos[OUT_H];
                    for (int m = 0; m < M; m++)
                    {
                        out_vec[m] = out_vec[m] >= input_vec[m] ? out_vec[m] : input_vec[m];
                    }
                }
            }
            naive_activate (out_vec, M, layer->activation);
        }
    }
    else
    {
        #ifdef GPU
        cuda_tiled_maxpool (M, N, ninst->out_mat_pos[OUT_H], ninst->input_pos_ptr_arr_gpu, input_pos_per_n,
            C, ldc, layer->activation, 
            aspen_CUDA_streams[dse->gpu_idx][dse->thread_id%GPU_RUN_STREAM_NUM]);
        aspen_sync_gpu_stream (dse->gpu_idx, dse->thread_id%GPU_RUN_STREAM_NUM);
        #endif
    }
    #endif
}
void tiled_avgpool2d (ninst_t *ninst, dse_t *dse)
{
    #if _SKIP_KERNELS == 0
    nasm_ldata_t *ldata = ninst->ldata;
    aspen_layer_t *layer = ninst->ldata->layer;
    prepare_im2col (ninst, dse->scratchpad);
    void **input_ptr_arr = dse->scratchpad;   
    const unsigned int input_pos_per_n = ninst->num_input_pos/ninst->tile_dims[OUT_W];
    const unsigned int M = ninst->tile_dims[OUT_H];
    const unsigned int N = ninst->tile_dims[OUT_W];
    const unsigned int ldc = ldata->out_mat_stride;
    void *C = get_ninst_out_mem (ninst);
    if (dse->gpu_idx < 0)
    {
        for (int n = 0; n < N; n++)
        {
            float *out_vec = (float*)C + n * ldc;
            float *input_vec = (float*)input_ptr_arr[n*input_pos_per_n];
            if (input_vec == NULL)
            {
                memset (out_vec, 0, M * layer->dnn->element_size);
            }
            else
            {
                input_vec += ninst->out_mat_pos[OUT_H];
                memcpy (out_vec, input_vec, M * layer->dnn->element_size);
            }
            for (int i = 1; i < input_pos_per_n; i++)
            {
                input_vec = (float*)input_ptr_arr[n*input_pos_per_n + i];
                if (input_vec != NULL)
                {
                    input_vec += ninst->out_mat_pos[OUT_H];
                    for (int m = 0; m < M; m++)
                    {
                        out_vec[m] += input_vec[m];
                    }
                }
            }
            for (int m = 0; m < M; m++)
            {
                out_vec[m] /= input_pos_per_n;
            }
            naive_activate (out_vec, M, layer->activation);
        }
    }
    else
    {
        #ifdef GPU
        cuda_tiled_avgpool (M, N, ninst->out_mat_pos[OUT_H], ninst->input_pos_ptr_arr_gpu, input_pos_per_n,
            C, ldc, layer->activation, 
            aspen_CUDA_streams[dse->gpu_idx][dse->thread_id%GPU_RUN_STREAM_NUM]);
        aspen_sync_gpu_stream (dse->gpu_idx, dse->thread_id%GPU_RUN_STREAM_NUM);
        #endif
    }
    #endif
}
void tiled_fully_connected (ninst_t *ninst, dse_t *dse)
{
    #if _SKIP_KERNELS == 0
    nasm_ldata_t *ldata = ninst->ldata;
    aspen_layer_t *layer = ninst->ldata->layer;
    nasm_ldata_t *p_ldata = (ldata->parent_ldata_idx_arr[PARENT_0] + ldata->nasm->ldata_arr);
    const unsigned int M = ninst->tile_dims[OUT_H];
    const unsigned int N = ninst->tile_dims[OUT_W];
    const unsigned int K = layer->params[IN_C] * layer->params[IN_H] * layer->params[IN_W];
    const unsigned int lda = K;
    const unsigned int ldb = K;
    const unsigned int ldc = ldata->out_mat_stride;
    const void *A = (char*)layer->tensors[WEIGHT_TENSOR]->data + (ninst->out_mat_pos[OUT_H] * lda * layer->dnn->element_size);
    const void *B = (char*)p_ldata->out_mat + (ninst->out_mat_pos[OUT_W] * ldb * layer->dnn->element_size);
    void *C = get_ninst_out_mem (ninst);
    if (dse->gpu_idx < 0)
    {
        const unsigned int rem_n = N % _TILE_SIZE_N;
        const unsigned int rem_m = M % _TILE_SIZE_M;
        const unsigned int rem_k = K % _TILE_SIZE_K;
        unsigned int n = 0;
        // <M, N, K> = <M, _TILE_SIZE_N, K>
        for (; n < N - rem_n; n += _TILE_SIZE_N)
        {
            for (int nn = 0; nn < _TILE_SIZE_N; nn++)
            {
                memset ((float*)C + (ldc * (n + nn)), 0, M * sizeof(float));
            }
            unsigned int k = 0;
            // <M, N, K> = <M, _TILE_SIZE_N, _TILE_SIZE_K>
            for (; k < K - rem_k; k += _TILE_SIZE_K)
            {
                unsigned int m = 0;
                // <M, N, K> = <_TILE_SIZE_M, _TILE_SIZE_N, _TILE_SIZE_K>
                for (; m < M - rem_m; m += _TILE_SIZE_M)
                {
                    SGEMM_KERNEL_FULL_TILE (_TILE_SIZE_M, _TILE_SIZE_N, _TILE_SIZE_K, 
                        (float*)A + (m * lda + k*_VEC_SIZE_M), lda, (float*)B + (n * ldb + k), ldb, (float*)C + (ldc * n + m), ldc);
                }
                // <M, N, K> = <rem_m, _TILE_SIZE_N, _TILE_SIZE_K>
                if (rem_m != 0)
                    SGEMM_KERNEL_TILE_N (rem_m, _TILE_SIZE_N, _TILE_SIZE_K, 
                            (float*)A + (m * lda + k*_VEC_SIZE_M), lda, (float*)B + (n * ldb + k), ldb, (float*)C + (ldc * n + m), ldc);
            
            }
            // <M, N, K> = <M, _TILE_SIZE_N, rem_k>
            if (rem_k != 0)
            {
                unsigned int m = 0;
                // <M, N, K> = <_TILE_SIZE_M, _TILE_SIZE_N, rem_k>
                for (; m < M - rem_m; m += _TILE_SIZE_M)
                {
                    SGEMM_KERNEL_FULL_TILE (_TILE_SIZE_M, _TILE_SIZE_N, rem_k,
                        (float*)A + (m * lda + k*_VEC_SIZE_M), lda, (float*)B + (n * ldb + k), ldb, (float*)C + (ldc * n + m), ldc);
                }
                // <M, N, K> = <rem_m, _TILE_SIZE_N, rem_k>
                SGEMM_KERNEL_TILE_N (rem_m, _TILE_SIZE_N, rem_k, 
                    (float*)A + (m * lda + k*_VEC_SIZE_M), lda, (float*)B + (n * ldb + k), ldb, (float*)C + (ldc * n + m), ldc);
            }
            for (unsigned int nn = n; nn < n + _TILE_SIZE_N; nn++)
            {
                float *out_vec = (float *)C + nn * ldc;
                if (layer->tensors[BIAS_TENSOR] != NULL)
                {
                    float *bias = (float*)layer->tensors[BIAS_TENSOR]->data + ninst->out_mat_pos[OUT_H];
                    for (unsigned int m = 0; m < M; m++)
                    {
                        out_vec[m] += bias[m];
                    }
                }
                naive_activate (out_vec, M, layer->activation);
            }
        }
        // <M, N, K> = <M, rem_n, K>
        if (rem_n != 0)
        {
            for (int nn = 0; nn < rem_n; nn++)
            {
                memset ((float*)C + (ldc * (n + nn)), 0, M * sizeof(float));
            }
            unsigned int k = 0;
            // <M, N, K> = <M, rem_n, _TILE_SIZE_K>
            for (; k < K - rem_k; k += _TILE_SIZE_K)
            {
                unsigned int m = 0;
                // <M, N, K> = <_TILE_SIZE_M, rem_n, _TILE_SIZE_K>
                for (; m < M - rem_m; m += _TILE_SIZE_M)
                {
                    SGEMM_KERNEL_TILE_M (_TILE_SIZE_M, rem_n, _TILE_SIZE_K, 
                        (float*)A + (m * lda + k*_VEC_SIZE_M), lda, (float*)B + (n * ldb + k), ldb, (float*)C + (ldc * n + m), ldc);
                }
                // <M, N, K> = <rem_m, rem_n, _TILE_SIZE_K>
                if (rem_m != 0)
                    SGEMM_KERNEL (rem_m, rem_n, _TILE_SIZE_K, 
                            (float*)A + (m * lda + k*_VEC_SIZE_M), lda, (float*)B + (n * ldb + k), ldb, (float*)C + (ldc * n + m), ldc);
            }
            // <M, N, K> = <M, rem_n, rem_k>
            if (rem_k != 0)
            {
                unsigned int m = 0;
                // <M, N, K> = <_TILE_SIZE_M, rem_n, rem_k>
                for (; m < M - rem_m; m += _TILE_SIZE_M)
                {
                    SGEMM_KERNEL_TILE_M (_TILE_SIZE_M, rem_n, rem_k,
                        (float*)A + (m * lda + k*_VEC_SIZE_M), lda, (float*)B + (n * ldb + k), ldb, (float*)C + (ldc * n + m), ldc);
                }
                // <M, N, K> = <rem_m, rem_n, rem_k>
                SGEMM_KERNEL (rem_m, rem_n, rem_k, 
                    (float*)A + (m * lda + k*_VEC_SIZE_M), lda, (float*)B + (n * ldb + k), ldb, (float*)C + (ldc * n + m), ldc);
            }
            for (unsigned int nn = n; nn < N; nn++)
            {
                float *out_vec = (float *)C + nn * ldc;
                if (layer->tensors[BIAS_TENSOR] != NULL)
                {
                    float *bias = (float*)layer->tensors[BIAS_TENSOR]->data + ninst->out_mat_pos[OUT_H];
                    for (unsigned int m = 0; m < M; m++)
                    {
                        out_vec[m] += bias[m];
                    }
                }
                naive_activate (out_vec, M, layer->activation);
            }
        }
    }
    else
    {
        #ifdef GPU
        float *bias = (float*)layer->tensors[BIAS_TENSOR]->data_gpu[dse->gpu_idx] + ninst->out_mat_pos[OUT_H];
        A = (char*)layer->tensors[WEIGHT_TENSOR]->data_gpu[dse->gpu_idx] 
            + (ninst->out_mat_pos[OUT_H] * lda * layer->dnn->element_size);
        cuda_matmul (M, N, K, A, lda, B, ldb, C, ldc, bias, layer->activation, 
            aspen_CUDA_streams[dse->gpu_idx][dse->thread_id%GPU_RUN_STREAM_NUM]);
        aspen_sync_gpu_stream (dse->gpu_idx, dse->thread_id%GPU_RUN_STREAM_NUM);
        #endif
    }
    #endif
}
void tiled_residual (ninst_t *ninst, dse_t *dse)
{
    #if _SKIP_KERNELS == 0
    nasm_ldata_t *ldata = ninst->ldata;
    aspen_layer_t *layer = ninst->ldata->layer;
    nasm_ldata_t *p0_ldata = (ldata->parent_ldata_idx_arr[PARENT_0] + ldata->nasm->ldata_arr);
    nasm_ldata_t *p1_ldata = (ldata->parent_ldata_idx_arr[PARENT_1] + ldata->nasm->ldata_arr);
    const unsigned int M = ninst->tile_dims[OUT_H];
    const unsigned int N = ninst->tile_dims[OUT_W];
    const unsigned int ldc = ldata->out_mat_stride;
    void *C = get_ninst_out_mem (ninst);
    if (dse->gpu_idx < 0)
    {
        for (int n = 0; n < N; n++)
        {
            unsigned int w_pos = ninst->out_mat_pos[OUT_W] + n;
            float *input_0 = (float*)p0_ldata->out_mat + w_pos * p0_ldata->out_mat_stride + ninst->out_mat_pos[OUT_H];
            float *input_1 = (float*)p1_ldata->out_mat + w_pos * p1_ldata->out_mat_stride + ninst->out_mat_pos[OUT_H];
            float *out_vec = (float*)C + n * ldc;
            for (int m = 0; m < M; m++)
            {
                out_vec[m] = input_0[m] + input_1[m];
            }
            naive_activate (out_vec, M, layer->activation);
        }
    }
    else
    {
        #ifdef GPU
        unsigned int w_pos = ninst->out_mat_pos[OUT_W];
        float *input_0 = (float*)p0_ldata->out_mat + w_pos * p0_ldata->out_mat_stride + ninst->out_mat_pos[OUT_H];
        float *input_1 = (float*)p1_ldata->out_mat + w_pos * p1_ldata->out_mat_stride + ninst->out_mat_pos[OUT_H];
        cuda_tiled_residual (input_0, input_1, C, N, M, ldc, layer->activation,
            aspen_CUDA_streams[dse->gpu_idx][dse->thread_id%GPU_RUN_STREAM_NUM]);
        aspen_sync_gpu_stream (dse->gpu_idx, dse->thread_id%GPU_RUN_STREAM_NUM);
        #endif
    }
    #endif
}

void tiled_softmax (ninst_t *ninst, dse_t *dse)
{
    #if _SKIP_KERNELS == 0
    nasm_ldata_t *ldata = ninst->ldata;
    nasm_ldata_t *p0_ldata = (ldata->parent_ldata_idx_arr[PARENT_0] + ldata->nasm->ldata_arr);
    const unsigned int M = ninst->tile_dims[OUT_H];
    const unsigned int N = ninst->tile_dims[OUT_W];
    const unsigned int ldc = ldata->out_mat_stride;
    void *C = get_ninst_out_mem (ninst);
    memset (C, 0, M * N * sizeof(float));
    if (dse->gpu_idx < 0)
    {
        for (int n = 0; n < N; n++)
        {
            unsigned int w_pos = ninst->out_mat_pos[OUT_W] + n;
            float *input = (float*)p0_ldata->out_mat + w_pos * p0_ldata->out_mat_stride + ninst->out_mat_pos[OUT_H];
            float *output = (float*)C + n * ldc;
            float max = input[0];
            for (int m = 1; m < M; m++)
            {
                max = max >= input[m] ? max : input[m];
            }
            float sum = 0;
            for (int m = 0; m < M; m++)
            {
                output[m] = expf (input[m] - max);
                sum += output[m];
            }
            for (int m = 0; m < M; m++)
            {
                output[m] /= sum;
            }
        }
    }
    else
    {
        #ifdef GPU
        
        #endif
    }
    #endif
}

void tiled_yolo (ninst_t *ninst, dse_t *dse)
{
    #if _SKIP_KERNELS == 0
    nasm_ldata_t *ldata = ninst->ldata;
    aspen_layer_t *layer = ninst->ldata->layer;
    nasm_ldata_t *p0_ldata = (ldata->parent_ldata_idx_arr[PARENT_0] + ldata->nasm->ldata_arr);
    const unsigned int N = ninst->tile_dims[OUT_W];
    const unsigned int ldc = ldata->out_mat_stride;
    const float *anchors = (float*)layer->tensors[ANCHOR_TENSOR]->data;
    void *C = get_ninst_out_mem (ninst);
    if (dse->gpu_idx < 0)
    {
        for (int n = 0; n < N; n++)
        {
            unsigned int w_pos = ninst->out_mat_pos[OUT_W] + n;
            int a_num = layer->params[IN_C]/layer->params[OUT_C];
            int bidx = w_pos / (layer->params[IN_H] * layer->params[IN_W] * a_num);
            int b_num = w_pos % (layer->params[IN_H] * layer->params[IN_W] * a_num);
            int aidx = b_num / (layer->params[IN_H] * layer->params[IN_W]);
            int wh = b_num % (layer->params[IN_H] * layer->params[IN_W]);
            int hidx  = wh / layer->params[IN_W];
            int widx  = wh % layer->params[IN_W];
            int cidx = ninst->out_mat_pos[OUT_H] + aidx * layer->params[OUT_C];
            float *in = (float*)p0_ldata->out_mat + (bidx*layer->params[IN_H]*layer->params[IN_W] + hidx*layer->params[IN_W] + widx) 
                * p0_ldata->out_mat_stride + cidx;
            float *out = (float*)C + n * ldc;
            out[0] = (1.0f / (1.0f + expf (-in[0])) + widx) * layer->params[STRIDE];
            out[1] = (1.0f / (1.0f + expf (-in[1])) + hidx) * layer->params[STRIDE];
            out[2] = expf (in[2]) * anchors[2 * aidx];
            out[3] = expf (in[3]) * anchors[2 * aidx + 1];
            for (unsigned int j = 4; j < layer->params[OUT_C]; j++)
                out[j] = 1.0f / (1.0f + expf (-in[j]));
        }
    }
    else
    {
        #ifdef GPU
        
        #endif
    }
    #endif
}

void tiled_append (ninst_t *ninst, dse_t *dse)
{
    #if _SKIP_KERNELS == 0
    nasm_ldata_t *ldata = ninst->ldata;
    aspen_layer_t *layer = ninst->ldata->layer;
    nasm_ldata_t *p0_ldata = (ldata->parent_ldata_idx_arr[PARENT_0] + ldata->nasm->ldata_arr);
    nasm_ldata_t *p1_ldata = (ldata->parent_ldata_idx_arr[PARENT_1] + ldata->nasm->ldata_arr);
    const unsigned int N = ninst->tile_dims[OUT_W];
    const unsigned int ldc = ldata->out_mat_stride;
    void *C = get_ninst_out_mem (ninst);
    #ifdef DEBUG
    assert (layer->params[IN_W] == (layer->params[OUT_W] / layer->params[STRIDE]));
    #endif
    if (dse->gpu_idx < 0)
    {
        for (int n = 0; n < N; n++)
        {
            const int c1 = layer->params[IN_C];
            const int c2 = layer->params[OUT_C] - layer->params[IN_C];
            const int w2 = layer->params[OUT_W];
            const int w1 = layer->params[IN_W];
            const int w_pos = ninst->out_mat_pos[OUT_W] + n;
            const int widx_2 = w_pos % w2;
            const int hidx_2 = w_pos / w2;
            const int widx_1 = widx_2 / layer->params[STRIDE];
            const int hidx_1 = hidx_2 / layer->params[STRIDE];
            const float *in1 = ((float*)p0_ldata->out_mat) + (hidx_1*w1 + widx_1)*p0_ldata->out_mat_stride;
            const float *in2 = ((float*)p1_ldata->out_mat) + (hidx_2*w2 + widx_2)*p1_ldata->out_mat_stride;
            float *out = (float*)C + n * ldc;
            memcpy (out, in1, c1*sizeof(float));
            memcpy (out+c1, in2, c2*sizeof(float));
        }
    }
    else
    {
        #ifdef GPU
        
        #endif
    }
    #endif
}

void tiled_layernorm (ninst_t *ninst, dse_t *dse)
{
    #if _SKIP_KERNELS == 0
    nasm_ldata_t *ldata = ninst->ldata;
    aspen_layer_t *layer = ninst->ldata->layer;
    nasm_ldata_t *p_ldata = (ldata->parent_ldata_idx_arr[PARENT_0] + ldata->nasm->ldata_arr);
    const unsigned int M = ninst->tile_dims[OUT_H];
    const unsigned int N = ninst->tile_dims[OUT_W];
    const unsigned int ldb = p_ldata->out_mat_stride;
    const unsigned int ldc = ldata->out_mat_stride;
    const float *weight = (float*)layer->tensors[WEIGHT_TENSOR]->data + ninst->out_mat_pos[OUT_H];
    const float *bias = NULL;
    if (layer->tensors[BIAS_TENSOR] != NULL)
        bias = (float*)layer->tensors[BIAS_TENSOR]->data + ninst->out_mat_pos[OUT_H];
    const void *B = (char*)p_ldata->out_mat + (ninst->out_mat_pos[OUT_W] * ldb * layer->dnn->element_size);
    void *C = get_ninst_out_mem (ninst);
    if (dse->gpu_idx < 0)
    {
        for (int n = 0; n < N; n++)
        {
            const float *input_vec = (float*)B + n * ldb;
            float *out_vec = (float*)C + n * ldc;
            float mean = 0;
            float var = 0;
            for (int m = 0; m < M; m++)
            {
                mean += input_vec[m];
                var += input_vec[m] * input_vec[m];
            }
            mean /= M;
            var /= M;
            var -= mean * mean;
            var = 1 / sqrtf (var + 1e-6f);
            for (int m = 0; m < M; m++)
            {
                out_vec[m] = (input_vec[m] - mean) * var * weight[m];
                if (bias != NULL)
                    out_vec[m] += bias[m];
            }
        }
    }
    else
    {
        #ifdef GPU
        weight = (float*)layer->tensors[WEIGHT_TENSOR]->data_gpu[dse->gpu_idx] + ninst->out_mat_pos[OUT_H];
        bias = (float*)layer->tensors[BIAS_TENSOR]->data_gpu[dse->gpu_idx] + ninst->out_mat_pos[OUT_H];
        cuda_layernorm (B, weight, bias, C, N, M, ldb, ldc, aspen_CUDA_streams[dse->gpu_idx][dse->thread_id%GPU_RUN_STREAM_NUM]);
        aspen_sync_gpu_stream (dse->gpu_idx, dse->thread_id%GPU_RUN_STREAM_NUM);
        #endif
    }
    #endif
}
void tiled_k_attention (ninst_t *ninst, dse_t *dse)
{
    #if _SKIP_KERNELS == 0
    nasm_ldata_t *ldata = ninst->ldata;
    aspen_layer_t *layer = ninst->ldata->layer;
    nasm_ldata_t *p_ldata = (ldata->parent_ldata_idx_arr[PARENT_0] + ldata->nasm->ldata_arr);
    nasm_ldata_t *pk_ldata = (ldata->parent_ldata_idx_arr[PARENT_1] + ldata->nasm->ldata_arr);
    const unsigned int num_hidden = layer->params[NUM_HIDDEN];
    const unsigned int num_heads = layer->params[NUM_HEAD];
    const unsigned int hidden_per_head = num_hidden / num_heads;
    const unsigned int num_seq = ldata->nasm->tr_seq_len;
    const unsigned int batch = ninst->out_mat_pos[OUT_W] / (num_seq * num_heads);
    const unsigned int head = (ninst->out_mat_pos[OUT_W] % (num_seq * num_heads)) / num_seq;
    const unsigned int n_global = ninst->out_mat_pos[OUT_W] % num_seq;
    const unsigned int M = ninst->tile_dims[OUT_H];
    const unsigned int N = ninst->tile_dims[OUT_W];
    const unsigned int K = layer->params[MAT_K];
    const unsigned int ldk = pk_ldata->out_mat_stride;
    const unsigned int lda = K;
    const unsigned int ldb = p_ldata->out_mat_stride;
    const unsigned int ldc = ldata->out_mat_stride;
    void *A = dse->scratchpad;
    const void *key_head = (float*)pk_ldata->out_mat + 
        (batch * ldk * M + head * K + ninst->out_mat_pos[OUT_H]);
    const void *B_head = (float*)p_ldata->out_mat + (batch * ldb * num_seq + head * hidden_per_head +
        + n_global * ldb);
    void *C_head = get_ninst_out_mem (ninst);
    unsigned int n = 0;
    if (dse->gpu_idx < 0)
    {
        const unsigned int rem_n = N % _TILE_SIZE_N;
        const unsigned int rem_m = M % _TILE_SIZE_M;
        const unsigned int rem_k = K % _TILE_SIZE_K;
        // Transpose & Reoder Key data. 
        for (unsigned int m = 0; m < M; m++)
        {
            for (unsigned int k = 0; k < K; k++)
            {
                const float* input_ptr = (float*)key_head + m * ldk + k;
                float* output_ptr = (float*)A + ((m/_VEC_SIZE_M) * lda + k) * _VEC_SIZE_M 
                    + (m % _VEC_SIZE_M);
                *output_ptr = *input_ptr;
            }
        }
        // <M, N, K> = <M, _TILE_SIZE_N, K>
        for (; n < N - rem_n; n += _TILE_SIZE_N)
        {
            for (int nn = 0; nn < _TILE_SIZE_N; nn++)
            {
                memset ((float*)C_head + (ldc * (n + nn)), 0, M * sizeof(float));
            }
            unsigned int k = 0;
            // <M, N, K> = <M, _TILE_SIZE_N, _TILE_SIZE_K>
            for (; k < K - rem_k; k += _TILE_SIZE_K)
            {
                unsigned int m = 0;
                // <M, N, K> = <_TILE_SIZE_M, _TILE_SIZE_N, _TILE_SIZE_K>
                for (; m < M - rem_m; m += _TILE_SIZE_M)
                {
                    SGEMM_KERNEL_FULL_TILE (_TILE_SIZE_M, _TILE_SIZE_N, _TILE_SIZE_K, 
                        (float*)A + (m * lda + k*_VEC_SIZE_M), lda, (float*)B_head + (n * ldb + k), ldb, (float*)C_head + (ldc * n + m), ldc);
                }
                // <M, N, K> = <rem_m, _TILE_SIZE_N, _TILE_SIZE_K>
                if (rem_m != 0)
                    SGEMM_KERNEL_TILE_N (rem_m, _TILE_SIZE_N, _TILE_SIZE_K, 
                            (float*)A + (m * lda + k*_VEC_SIZE_M), lda, (float*)B_head + (n * ldb + k), ldb, (float*)C_head + (ldc * n + m), ldc);
            
            }
            // <M, N, K> = <M, _TILE_SIZE_N, rem_k>
            if (rem_k != 0)
            {
                unsigned int m = 0;
                // <M, N, K> = <_TILE_SIZE_M, _TILE_SIZE_N, rem_k>
                for (; m < M - rem_m; m += _TILE_SIZE_M)
                {
                    SGEMM_KERNEL_FULL_TILE (_TILE_SIZE_M, _TILE_SIZE_N, rem_k,
                        (float*)A + (m * lda + k*_VEC_SIZE_M), lda, (float*)B_head + (n * ldb + k), ldb, (float*)C_head + (ldc * n + m), ldc);
                }
                // <M, N, K> = <rem_m, _TILE_SIZE_N, rem_k>
                SGEMM_KERNEL_TILE_N (rem_m, _TILE_SIZE_N, rem_k, 
                    (float*)A + (m * lda + k*_VEC_SIZE_M), lda, (float*)B_head + (n * ldb + k), ldb, (float*)C_head + (ldc * n + m), ldc);
            }
            for (unsigned int nn = n; nn < n + _TILE_SIZE_N; nn++)
            {
                if (layer->params[MASKED] == 1)
                {
                    for (unsigned int m = 0; m < M; m++)
                    {
                        if (m > nn + n_global)
                            *((float*)C_head + nn*ldc + m) = -1e10;
                    }
                }
                float total = 0;
                for (unsigned int j = 0; j < M; j++)
                {
                    float val = *((float*)C_head + nn*ldc + j);
                    val /= sqrtf (hidden_per_head);
                    *((float*)C_head + nn*ldc + j) = expf (val);
                    total += *((float*)C_head + nn*ldc + j);
                }
                for (unsigned int j = 0; j < M; j++)
                    *((float*)C_head + nn*ldc + j) /= total;
            }
        }
        // <M, N, K> = <M, rem_n, K>
        if (rem_n != 0)
        {
            for (int nn = 0; nn < rem_n; nn++)
            {
                memset ((float*)C_head + (ldc * (n + nn)), 0, M * sizeof(float));
            }
            unsigned int k = 0;
            // <M, N, K> = <M, rem_n, _TILE_SIZE_K>
            for (; k < K - rem_k; k += _TILE_SIZE_K)
            {
                unsigned int m = 0;
                // <M, N, K> = <_TILE_SIZE_M, rem_n, _TILE_SIZE_K>
                for (; m < M - rem_m; m += _TILE_SIZE_M)
                {
                    SGEMM_KERNEL_TILE_M (_TILE_SIZE_M, rem_n, _TILE_SIZE_K, 
                        (float*)A + (m * lda + k*_VEC_SIZE_M), lda, (float*)B_head + (n * ldb + k), ldb, (float*)C_head + (ldc * n + m), ldc);
                }
                // <M, N, K> = <rem_m, rem_n, _TILE_SIZE_K>
                if (rem_m != 0)
                    SGEMM_KERNEL (rem_m, rem_n, _TILE_SIZE_K, 
                            (float*)A + (m * lda + k*_VEC_SIZE_M), lda, (float*)B_head + (n * ldb + k), ldb, (float*)C_head + (ldc * n + m), ldc);
            }
            // <M, N, K> = <M, rem_n, rem_k>
            if (rem_k != 0)
            {
                unsigned int m = 0;
                // <M, N, K> = <_TILE_SIZE_M, rem_n, rem_k>
                for (; m < M - rem_m; m += _TILE_SIZE_M)
                {
                    SGEMM_KERNEL_TILE_M (_TILE_SIZE_M, rem_n, rem_k,
                        (float*)A + (m * lda + k*_VEC_SIZE_M), lda, (float*)B_head + (n * ldb + k), ldb, (float*)C_head + (ldc * n + m), ldc);
                }
                // <M, N, K> = <rem_m, rem_n, rem_k>
                SGEMM_KERNEL (rem_m, rem_n, rem_k, 
                    (float*)A + (m * lda + k*_VEC_SIZE_M), lda, (float*)B_head + (n * ldb + k), ldb, (float*)C_head + (ldc * n + m), ldc);
            }
            for (unsigned int nn = n; nn < N; nn++)
            {
                if (layer->params[MASKED] == 1)
                {
                    for (unsigned int m = 0; m < M; m++)
                    {
                        if (m > nn + n_global)
                            *((float*)C_head + nn*ldc + m) = -1e10;
                    }
                }
                float total = 0;
                for (unsigned int j = 0; j < M; j++)
                {
                    float val = *((float*)C_head + nn*ldc + j);
                    val /= sqrtf (hidden_per_head);
                    *((float*)C_head + nn*ldc + j) = expf (val);
                    total += *((float*)C_head + nn*ldc + j);
                }
                for (unsigned int j = 0; j < M; j++)
                    *((float*)C_head + nn*ldc + j) /= total;
            }
        }
    }
    else
    {
        #ifdef GPU
        cuda_tiled_k_attention (M, N, K, key_head, ldk, B_head, ldb, C_head, ldc, aspen_CUDA_streams[dse->gpu_idx][dse->thread_id%GPU_RUN_STREAM_NUM]);
        aspen_sync_gpu_stream (dse->gpu_idx, dse->thread_id%GPU_RUN_STREAM_NUM);
        #endif
    }
    #endif
}

void tiled_v_attention (ninst_t *ninst, dse_t *dse)
{
    #if _SKIP_KERNELS == 0
    nasm_ldata_t *ldata = ninst->ldata;
    aspen_layer_t *layer = ninst->ldata->layer;
    nasm_ldata_t *p_ldata = (ldata->parent_ldata_idx_arr[PARENT_0] + ldata->nasm->ldata_arr);
    nasm_ldata_t *pv_ldata = (ldata->parent_ldata_idx_arr[PARENT_1] + ldata->nasm->ldata_arr);
    const unsigned int num_hidden = layer->params[NUM_HIDDEN];
    const unsigned int num_heads = layer->params[NUM_HEAD];
    const unsigned int hidden_per_head = num_hidden / num_heads;
    const unsigned int num_seq = ldata->nasm->tr_seq_len;
    unsigned int batch = ninst->out_mat_pos[OUT_W] / num_seq;
    const unsigned int head = ninst->out_mat_pos[OUT_H]  / hidden_per_head;
    const unsigned int M = ninst->tile_dims[OUT_H];
    const unsigned int N = ninst->tile_dims[OUT_W];
    const unsigned int K = num_seq;
    const unsigned int ldv = pv_ldata->out_mat_stride;
    const unsigned int lda = K;
    const unsigned int ldb = p_ldata->out_mat_stride;
    const unsigned int ldc = ldata->out_mat_stride;
    void *A = dse->scratchpad;
    const float *val_head = (float*)pv_ldata->out_mat + batch * ldv * num_seq + ninst->out_mat_pos[OUT_H];
    const void *B_head = (float*)p_ldata->out_mat + (batch * num_heads * num_seq + head * num_seq +
        + (ninst->out_mat_pos[OUT_W] % num_seq)) * ldb;
    void *C_head = get_ninst_out_mem (ninst); 
    if (dse->gpu_idx < 0)
    {
        const unsigned int rem_n = N % _TILE_SIZE_N;
        const unsigned int rem_m = M % _TILE_SIZE_M;
        const unsigned int rem_k = K % _TILE_SIZE_K;
        unsigned int n = 0;
        // Transpose & Reoder Value data.
        for (unsigned int m = 0; m < M; m++)
        {
            for (unsigned int k = 0; k < K; k++)
            {
                const float* input_ptr = val_head + k * ldv + m;
                float* output_ptr = (float*)A + ((m/_VEC_SIZE_M) * lda + k) * _VEC_SIZE_M 
                    + (m % _VEC_SIZE_M);
                *output_ptr = *input_ptr;
            }
        }
        // <M, N, K> = <M, _TILE_SIZE_N, K>
        for (; n < N - rem_n; n += _TILE_SIZE_N)
        {
            for (int nn = 0; nn < _TILE_SIZE_N; nn++)
            {
                memset ((float*)C_head + (ldc * (n + nn)), 0, M * sizeof(float));
            }
            unsigned int k = 0;
            // <M, N, K> = <M, _TILE_SIZE_N, _TILE_SIZE_K>
            for (; k < K - rem_k; k += _TILE_SIZE_K)
            {
                unsigned int m = 0;
                // <M, N, K> = <_TILE_SIZE_M, _TILE_SIZE_N, _TILE_SIZE_K>
                for (; m < M - rem_m; m += _TILE_SIZE_M)
                {
                    SGEMM_KERNEL_FULL_TILE (_TILE_SIZE_M, _TILE_SIZE_N, _TILE_SIZE_K, 
                        (float*)A + (m * lda + k*_VEC_SIZE_M), lda, (float*)B_head + (n * ldb + k), ldb, (float*)C_head + (ldc * n + m), ldc);
                }
                // <M, N, K> = <rem_m, _TILE_SIZE_N, _TILE_SIZE_K>
                if (rem_m != 0)
                    SGEMM_KERNEL_TILE_N (rem_m, _TILE_SIZE_N, _TILE_SIZE_K, 
                            (float*)A + (m * lda + k*_VEC_SIZE_M), lda, (float*)B_head + (n * ldb + k), ldb, (float*)C_head + (ldc * n + m), ldc);
            
            }
            // <M, N, K> = <M, _TILE_SIZE_N, rem_k>
            if (rem_k != 0)
            {
                unsigned int m = 0;
                // <M, N, K> = <_TILE_SIZE_M, _TILE_SIZE_N, rem_k>
                for (; m < M - rem_m; m += _TILE_SIZE_M)
                {
                    SGEMM_KERNEL_FULL_TILE (_TILE_SIZE_M, _TILE_SIZE_N, rem_k,
                        (float*)A + (m * lda + k*_VEC_SIZE_M), lda, (float*)B_head + (n * ldb + k), ldb, (float*)C_head + (ldc * n + m), ldc);
                }
                // <M, N, K> = <rem_m, _TILE_SIZE_N, rem_k>
                SGEMM_KERNEL_TILE_N (rem_m, _TILE_SIZE_N, rem_k, 
                    (float*)A + (m * lda + k*_VEC_SIZE_M), lda, (float*)B_head + (n * ldb + k), ldb, (float*)C_head + (ldc * n + m), ldc);
            }
        }
        // <M, N, K> = <M, rem_n, K>
        if (rem_n != 0)
        {
            for (int nn = 0; nn < rem_n; nn++)
            {
                memset ((float*)C_head + (ldc * (n + nn)), 0, M * sizeof(float));
            }
            unsigned int k = 0;
            // <M, N, K> = <M, rem_n, _TILE_SIZE_K>
            for (; k < K - rem_k; k += _TILE_SIZE_K)
            {
                unsigned int m = 0;
                // <M, N, K> = <_TILE_SIZE_M, rem_n, _TILE_SIZE_K>
                for (; m < M - rem_m; m += _TILE_SIZE_M)
                {
                    SGEMM_KERNEL_TILE_M (_TILE_SIZE_M, rem_n, _TILE_SIZE_K, 
                        (float*)A + (m * lda + k*_VEC_SIZE_M), lda, (float*)B_head + (n * ldb + k), ldb, (float*)C_head + (ldc * n + m), ldc);
                }
                // <M, N, K> = <rem_m, rem_n, _TILE_SIZE_K>
                if (rem_m != 0)
                    SGEMM_KERNEL (rem_m, rem_n, _TILE_SIZE_K, 
                            (float*)A + (m * lda + k*_VEC_SIZE_M), lda, (float*)B_head + (n * ldb + k), ldb, (float*)C_head + (ldc * n + m), ldc);
            }
            // <M, N, K> = <M, rem_n, rem_k>
            if (rem_k != 0)
            {
                unsigned int m = 0;
                // <M, N, K> = <_TILE_SIZE_M, rem_n, rem_k>
                for (; m < M - rem_m; m += _TILE_SIZE_M)
                {
                    SGEMM_KERNEL_TILE_M (_TILE_SIZE_M, rem_n, rem_k,
                        (float*)A + (m * lda + k*_VEC_SIZE_M), lda, (float*)B_head + (n * ldb + k), ldb, (float*)C_head + (ldc * n + m), ldc);
                }
                // <M, N, K> = <rem_m, rem_n, rem_k>
                SGEMM_KERNEL (rem_m, rem_n, rem_k, 
                    (float*)A + (m * lda + k*_VEC_SIZE_M), lda, (float*)B_head + (n * ldb + k), ldb, (float*)C_head + (ldc * n + m), ldc);
            }
        }
    }
    else
    {
        #ifdef GPU
        cuda_tiled_v_attention (M, N, K, val_head, ldv, B_head, ldb, C_head, ldc, aspen_CUDA_streams[dse->gpu_idx][dse->thread_id%GPU_RUN_STREAM_NUM]);
        aspen_sync_gpu_stream (dse->gpu_idx, dse->thread_id%GPU_RUN_STREAM_NUM);
        #endif
    }
    #endif
}