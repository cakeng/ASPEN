extern "C"
{
    #include "cuda_kernels.h"
}

// Custom CUDA GEMM kernel.
__global__ void cuda_tiled_k_matmul_kernel(
    const unsigned int M, const unsigned int N, const unsigned int K,
    const float *key_head, const unsigned int ldk, const float *B_head, const unsigned int ldb, float *C_head, const unsigned int ldc)
{
    const int mLocal = threadIdx.x*_THREAD_M_SIZE;
    const int nLocal = threadIdx.y*_THREAD_N_SIZE; 
    const int mGroup = blockIdx.x*_BLOCK_M_SIZE;
    const int nGroup = blockIdx.y*_BLOCK_N_SIZE;
    const int id = threadIdx.x*(_BLOCK_N_SIZE / _THREAD_N_SIZE) + threadIdx.y;
    __shared__ float ACache [_BLOCK_K_SIZE*_BLOCK_M_SIZE];
    __shared__ float BCache [_BLOCK_K_SIZE*_BLOCK_N_SIZE];
    float cout[_THREAD_N_SIZE][_THREAD_M_SIZE];
    for (int vecN = 0; vecN < _THREAD_N_SIZE; vecN++)
    {
        for (int vecM = 0; vecM < _THREAD_M_SIZE; vecM++)
        {
            cout[vecN][vecM] = 0;
        }   
    }
    int kIdx = 0;  
    if (K%_BLOCK_K_SIZE)
    {
        // Load caches.
        for (int aIdx = 0; aIdx < (_BLOCK_K_SIZE/_CACHE_A_K_PER_LOAD); aIdx++)
        {
            const int cache_idx = id*(_BLOCK_K_SIZE/_CACHE_A_K_PER_LOAD) + aIdx;
            const int m = mGroup + cache_idx%_BLOCK_M_SIZE;
            const int k = kIdx + cache_idx/_BLOCK_M_SIZE;
            ACache[cache_idx] = key_head[m * ldk + k];
        }
        for (int bIdx = 0; bIdx < (_BLOCK_K_SIZE/_CACHE_B_K_PER_LOAD); bIdx++)
        {
            const int cache_idx = id*(_BLOCK_K_SIZE/_CACHE_B_K_PER_LOAD) + bIdx;
            const int n = nGroup + cache_idx%_BLOCK_N_SIZE;
            const int k = kIdx + cache_idx/_BLOCK_N_SIZE;
            BCache[cache_idx] = B_head[ldb*n + k];
        }
        __syncthreads();
        // printf ("Thread %d: %3.3f\n", id, cout[0][0]);
        for (; kIdx < K%_BLOCK_K_SIZE; kIdx++)
        {
            // Calculate.
            for (int vecN = 0; vecN < _THREAD_N_SIZE; vecN++)
            {
                for (int vecM = 0; vecM < _THREAD_M_SIZE; vecM++)
                {
                    // printf ("B%dT%d: (%d, %d) %3.3f, %3.3f\n", blockIdx.x + blockIdx.y
                    //     ,id, vecN, vecM, ACache[kk*_BLOCK_M_SIZE + mLocal + vecM], BCache[kk*_BLOCK_N_SIZE + nLocal + vecN]);
                    cout[vecN][vecM] += ACache[kIdx*_BLOCK_M_SIZE + mLocal + vecM] * BCache[kIdx*_BLOCK_N_SIZE + nLocal + vecN];
                }   
            }
        }
        // Sync threads.
        __syncthreads();
        // printf ("Thread %d: %3.3f\n", id, cout[0][0]);
    }
    for (; kIdx < K; kIdx += _BLOCK_K_SIZE)
    {
        // Load caches.
        for (int aIdx = 0; aIdx < (_BLOCK_K_SIZE/_CACHE_A_K_PER_LOAD); aIdx++)
        {
            const int cache_idx = id*(_BLOCK_K_SIZE/_CACHE_A_K_PER_LOAD) + aIdx;
            const int m = mGroup + cache_idx%_BLOCK_M_SIZE;
            const int k = kIdx + cache_idx/_BLOCK_M_SIZE;
            ACache[cache_idx] = key_head[m * ldk + k];
        }
        for (int bIdx = 0; bIdx < (_BLOCK_K_SIZE/_CACHE_B_K_PER_LOAD); bIdx++)
        {
            const int cache_idx = id*(_BLOCK_K_SIZE/_CACHE_B_K_PER_LOAD) + bIdx;
            const int n = nGroup + cache_idx%_BLOCK_N_SIZE;
            const int k = kIdx + cache_idx/_BLOCK_N_SIZE;
            BCache[cache_idx] = B_head[ldb*n + k];
        }
        __syncthreads();
        for (int kk = 0; kk < _BLOCK_K_SIZE; kk++)
        {
            // Calculate.
            for (int vecN = 0; vecN < _THREAD_N_SIZE; vecN++)
            {
                for (int vecM = 0; vecM < _THREAD_M_SIZE; vecM++)
                {
                    cout[vecN][vecM] += ACache[kk*_BLOCK_M_SIZE + mLocal + vecM] * BCache[kk*_BLOCK_N_SIZE + nLocal + vecN];
                }   
            }
        }
        // Sync threads.
        __syncthreads();
    }
    // Save results
    for (int vecN = 0; vecN < _THREAD_N_SIZE; vecN++)
    {
        for (int vecM = 0; vecM < _THREAD_M_SIZE; vecM++)
        {
            const int m = mGroup + mLocal + vecM;
            const int n = nGroup + nLocal + vecN;
            if (m < M &&  n < N)
            {
                C_head[ldc*n + m] = cout[vecN][vecM];
            }
        }   
    }
    __syncthreads();

}
__global__ void cuda_tiled_k_prob_kernel(
    const unsigned int M, const unsigned int N, const unsigned int K,
    const float *key_head, const unsigned int ldk, const float *B_head, const unsigned int ldb, float *C_head, const unsigned int ldc)
{
    const int n = blockIdx.x * _BLOCK_ATT_K_PROB_SIZE + threadIdx.x;
    if (n >= N)
        return;
    float *C = C_head + n*ldc;
    float total = 0;
    for (unsigned int m = 0; m < M; m++)
    {
        C[m] /= sqrtf (K);
        C[m] = expf (C[m]);
        total += C[m];
    }
    for (unsigned int m = 0; m < M; m++)
    {
        C[m] /= total;
    }
}
__global__ void cuda_tiled_v_attention_kernel(const unsigned int M, const unsigned int N, const unsigned int K,
    const float *val_head, const unsigned int ldv, const float *B_head, const unsigned int ldb, float *C_head, const unsigned int ldc)
{
    const int mLocal = threadIdx.x*_THREAD_M_SIZE;
    const int nLocal = threadIdx.y*_THREAD_N_SIZE; 
    const int mGroup = blockIdx.x*_BLOCK_M_SIZE;
    const int nGroup = blockIdx.y*_BLOCK_N_SIZE;
    const int id = threadIdx.x*(_BLOCK_N_SIZE / _THREAD_N_SIZE) + threadIdx.y;
    __shared__ float ACache [_BLOCK_K_SIZE*_BLOCK_M_SIZE];
    __shared__ float BCache [_BLOCK_K_SIZE*_BLOCK_N_SIZE];
    float cout[_THREAD_N_SIZE][_THREAD_M_SIZE];
    for (int vecN = 0; vecN < _THREAD_N_SIZE; vecN++)
    {
        for (int vecM = 0; vecM < _THREAD_M_SIZE; vecM++)
        {
            cout[vecN][vecM] = 0;
        }   
    }
    int kIdx = 0;  
    if (K%_BLOCK_K_SIZE)
    {
        // Load caches.
        for (int aIdx = 0; aIdx < (_BLOCK_K_SIZE/_CACHE_A_K_PER_LOAD); aIdx++)
        {
            const int cache_idx = id*(_BLOCK_K_SIZE/_CACHE_A_K_PER_LOAD) + aIdx;
            const int m = mGroup + cache_idx%_BLOCK_M_SIZE;
            const int k = kIdx + cache_idx/_BLOCK_M_SIZE;
            ACache[cache_idx] = val_head[k * ldv + m];
        }
        for (int bIdx = 0; bIdx < (_BLOCK_K_SIZE/_CACHE_B_K_PER_LOAD); bIdx++)
        {
            const int cache_idx = id*(_BLOCK_K_SIZE/_CACHE_B_K_PER_LOAD) + bIdx;
            const int n = nGroup + cache_idx%_BLOCK_N_SIZE;
            const int k = kIdx + cache_idx/_BLOCK_N_SIZE;
            BCache[cache_idx] = B_head[ldb*n + k];
        }
        __syncthreads();
        // printf ("Thread %d: %3.3f\n", id, cout[0][0]);
        for (; kIdx < K%_BLOCK_K_SIZE; kIdx++)
        {
            // Calculate.
            for (int vecN = 0; vecN < _THREAD_N_SIZE; vecN++)
            {
                for (int vecM = 0; vecM < _THREAD_M_SIZE; vecM++)
                {
                    // printf ("B%dT%d: (%d, %d) %3.3f, %3.3f\n", blockIdx.x + blockIdx.y
                    //     ,id, vecN, vecM, ACache[kk*_BLOCK_M_SIZE + mLocal + vecM], BCache[kk*_BLOCK_N_SIZE + nLocal + vecN]);
                    cout[vecN][vecM] += ACache[kIdx*_BLOCK_M_SIZE + mLocal + vecM] * BCache[kIdx*_BLOCK_N_SIZE + nLocal + vecN];
                }   
            }
        }
        // Sync threads.
        __syncthreads();
        // printf ("Thread %d: %3.3f\n", id, cout[0][0]);
    }
    for (; kIdx < K; kIdx += _BLOCK_K_SIZE)
    {
        // Load caches.
        for (int aIdx = 0; aIdx < (_BLOCK_K_SIZE/_CACHE_A_K_PER_LOAD); aIdx++)
        {
            const int cache_idx = id*(_BLOCK_K_SIZE/_CACHE_A_K_PER_LOAD) + aIdx;
            const int m = mGroup + cache_idx%_BLOCK_M_SIZE;
            const int k = kIdx + cache_idx/_BLOCK_M_SIZE;
            ACache[cache_idx] = val_head[k * ldv + m];
        }
        for (int bIdx = 0; bIdx < (_BLOCK_K_SIZE/_CACHE_B_K_PER_LOAD); bIdx++)
        {
            const int cache_idx = id*(_BLOCK_K_SIZE/_CACHE_B_K_PER_LOAD) + bIdx;
            const int n = nGroup + cache_idx%_BLOCK_N_SIZE;
            const int k = kIdx + cache_idx/_BLOCK_N_SIZE;
            BCache[cache_idx] = B_head[ldb*n + k];
        }
        __syncthreads();
        for (int kk = 0; kk < _BLOCK_K_SIZE; kk++)
        {
            // Calculate.
            for (int vecN = 0; vecN < _THREAD_N_SIZE; vecN++)
            {
                for (int vecM = 0; vecM < _THREAD_M_SIZE; vecM++)
                {
                    cout[vecN][vecM] += ACache[kk*_BLOCK_M_SIZE + mLocal + vecM] * BCache[kk*_BLOCK_N_SIZE + nLocal + vecN];
                }   
            }
        }
        // Sync threads.
        __syncthreads();
    }
    // Save results
    for (int vecN = 0; vecN < _THREAD_N_SIZE; vecN++)
    {
        for (int vecM = 0; vecM < _THREAD_M_SIZE; vecM++)
        {
            const int m = mGroup + mLocal + vecM;
            const int n = nGroup + nLocal + vecN;
            if (m < M &&  n < N)
            {
                C_head[ldc*n + m] = cout[vecN][vecM];
            }
        }   
    }
    __syncthreads();
}
__global__ void cuda_tiled_residual_kernel (const float *input_1, const float *input_2, float *output, unsigned int N, unsigned int M, const unsigned int ldc, LAYER_ACT activation_type)
{
    const int n = blockIdx.x * _BLOCK_TILED_RESIDUAL_SIZE + threadIdx.x;
    const int m = blockIdx.y * _BLOCK_TILED_RESIDUAL_SIZE + threadIdx.y;
    if (n < N && m < M)
    {
        float val = input_1[n * ldc + m] + input_2[n * ldc + m];
        if (activation_type == RELU)
            val = val > 0 ? val : 0;
        else if (activation_type == GELU)
            val = val * 0.5 * (1 + erff (val*0.7071067811865475f));
        output[n * ldc + m] = val;
    }
}
__global__ void cuda_tiled_layernorm_kernel(const float *input, const float *kernel, const float *bias, 
    float *output, unsigned int N, unsigned int M)
{
    const int id = blockIdx.x * _BLOCK_TILED_RESIDUAL_SIZE + threadIdx.x;
    float mean = 0;
    float var = 0;
    for (unsigned int m = 0; m < M; m++)
    {
        mean += input[id * M + m];
        var += input[id * M + m] * input[id * M + m];
    }
    mean /= M;
    var /= M;
    var -= mean * mean;
    var = 1 / sqrtf (var + 1e-12);
    for (unsigned int m = 0; m < M; m++)
    {
        output[id * M + m] = (input[id * M + m] - mean) * var * kernel[m] + bias[m];
    }
}
// Wrapper function for CUDA kernel.
extern "C"
{
void cuda_tiled_k_attention (
    const unsigned int M, const unsigned int N, const unsigned int K,
    const float *key_head, const unsigned int ldk, const float *B_head, const unsigned int ldb, float *C_head, const unsigned int ldc
    , cudaStream_t stream)
{
    #ifdef DEBUG
    if (input_1 == NULL)
        FPRT (stderr, "Error in naive_k_attention: input_1 is NULL.\n");
    if (input_2 == NULL)
        FPRT (stderr, "Error in naive_k_attention: input_2 is NULL.\n");
    if (output == NULL)
        FPRT (stderr, "Error in naive_k_attention: output is NULL.\n");
    #endif
    #ifdef DEBUG
    if (!(((_BLOCK_K_SIZE*_BLOCK_M_SIZE)%_THREAD_NUM) == 0 && ((_BLOCK_K_SIZE*_BLOCK_N_SIZE)%_THREAD_NUM) == 0))
    {
        printf ("ERROR! - Wrong parameter settings - (_BLOCK_K_SIZE*_BLOCK_M_SIZE)%_THREAD_NUM) = %d, ((_BLOCK_K_SIZE*_BLOCK_N_SIZE)%_THREAD_NUM) = %d\n",
        (_BLOCK_K_SIZE*_BLOCK_M_SIZE)%_THREAD_NUM), ((_BLOCK_K_SIZE*_BLOCK_N_SIZE)%_THREAD_NUM)); 
        exit(0);
    }
    #endif

    dim3 gridDim (M/_BLOCK_M_SIZE + ((M%_BLOCK_M_SIZE) > 0), N/_BLOCK_N_SIZE + ((N%_BLOCK_N_SIZE) > 0), 1);
    dim3 blockDim ((_BLOCK_M_SIZE / _THREAD_M_SIZE), (_BLOCK_N_SIZE / _THREAD_N_SIZE), 1);
    cuda_tiled_k_matmul_kernel<<<gridDim, blockDim, 0, stream>>> (
        M, N, K, key_head, ldk, B_head, ldb, C_head, ldc);
    dim3 prob_gridDim (N/_BLOCK_ATT_K_PROB_SIZE + ((N%_BLOCK_ATT_K_PROB_SIZE) > 0), 1, 1);
    dim3 prob_blockDim (_BLOCK_ATT_K_PROB_SIZE, 1, 1);
    cuda_tiled_k_prob_kernel<<<prob_gridDim, prob_blockDim, 0, stream>>> (
        M, N, K, key_head, ldk, B_head, ldb, C_head, ldc);
    
}
void cuda_tiled_v_attention (
    const unsigned int M, const unsigned int N, const unsigned int K,
    const float *val_head, const unsigned int ldv, const float *B_head, const unsigned int ldb, float *C_head, const unsigned int ldc
    , cudaStream_t stream)
{
    #ifdef DEBUG
    if (!(((_BLOCK_K_SIZE*_BLOCK_M_SIZE)%_THREAD_NUM) == 0 && ((_BLOCK_K_SIZE*_BLOCK_N_SIZE)%_THREAD_NUM) == 0))
    {
        printf ("ERROR! - Wrong parameter settings - (_BLOCK_K_SIZE*_BLOCK_M_SIZE)%_THREAD_NUM) = %d, ((_BLOCK_K_SIZE*_BLOCK_N_SIZE)%_THREAD_NUM) = %d\n",
        (_BLOCK_K_SIZE*_BLOCK_M_SIZE)%_THREAD_NUM), ((_BLOCK_K_SIZE*_BLOCK_N_SIZE)%_THREAD_NUM)); 
        exit(0);
    }
    #endif

    dim3 gridDim (M/_BLOCK_M_SIZE + ((M%_BLOCK_M_SIZE) > 0), N/_BLOCK_N_SIZE + ((N%_BLOCK_N_SIZE) > 0), 1);
    dim3 blockDim ((_BLOCK_M_SIZE / _THREAD_M_SIZE), (_BLOCK_N_SIZE / _THREAD_N_SIZE), 1);
    cuda_tiled_v_attention_kernel<<<gridDim, blockDim, 0, stream>>> (
        M, N, K, val_head, ldv, B_head, ldb, C_head, ldc);
}
void cuda_tiled_residual (const float *input_1, const float *input_2, float *output, unsigned int N, unsigned int M, const unsigned int ldc
    , LAYER_ACT activation_type, cudaStream_t stream)
{
    dim3 gridDim (N/_BLOCK_TILED_RESIDUAL_SIZE + ((N%_BLOCK_TILED_RESIDUAL_SIZE) > 0), M/_BLOCK_TILED_RESIDUAL_SIZE + ((M%_BLOCK_TILED_RESIDUAL_SIZE) > 0), 1);
    dim3 blockDim (_BLOCK_TILED_RESIDUAL_SIZE, _BLOCK_TILED_RESIDUAL_SIZE, 1);
    cuda_tiled_residual_kernel<<<gridDim, blockDim, 0, stream>>> (input_1, input_2, output, N, M, ldc, activation_type);
}
void cuda_tiled_layernorm (const float *input, const float *kernel, const float *bias, 
    float *output, unsigned int N, unsigned int M, cudaStream_t stream)
{
    dim3 prob_gridDim (N/_BLOCK_LAYERNORM_SIZE + ((N%_BLOCK_LAYERNORM_SIZE) > 0), 1, 1);
    dim3 prob_blockDim (_BLOCK_LAYERNORM_SIZE, 1, 1);
    cuda_tiled_layernorm_kernel<<<prob_gridDim, prob_blockDim, 0, stream>>> (input, kernel, bias, output, N, M);
}
}