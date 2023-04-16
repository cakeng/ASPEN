extern "C"
{
    #include "cuda_kernels.h"
}

// Custom CUDA GEMM weight.
__global__ void cuda_matmul_kernel(const unsigned int M, const unsigned int N, const unsigned int K,
    const float *A, const unsigned int lda, const float *B, const unsigned int ldb, float *C, const unsigned int ldc,
    const float *Bias, LAYER_ACT activation_type)
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
            cout[vecN][vecM] = Bias[mGroup + mLocal + vecM];
        }   
    }

    // for (int vecN = 0; vecN < _THREAD_N_SIZE; vecN++)
    // {
    //     for (int vecM = 0; vecM < _THREAD_M_SIZE; vecM++)
    //     {
    //         const int m = mGroup + mLocal + vecM;
    //         const int n = nGroup + nLocal + vecN;
    //         if (m < M &&  n < N)
    //         {
    //             C[n * ldc + m] = 0;
    //             for (int k = 0; k < K; k++)
    //             {
    //                 C[n * ldc + m] += A[((m/_THREAD_M_SIZE)*lda + k) * _THREAD_M_SIZE + m%_THREAD_M_SIZE] * B[n * ldb + k];
    //             }
    //         }
    //     }
    // }
    int kIdx = 0;  
    if (K%_BLOCK_K_SIZE)
    {
        // Load caches.
        for (int aIdx = 0; aIdx < (_BLOCK_K_SIZE/_CACHE_A_K_PER_LOAD); aIdx++)
        {
            const int cache_idx = id*(_BLOCK_K_SIZE/_CACHE_A_K_PER_LOAD) + aIdx;
            const int m = mGroup + cache_idx%_BLOCK_M_SIZE;
            const int k = kIdx + cache_idx/_BLOCK_M_SIZE;
            ACache[cache_idx] = A[((m/_THREAD_M_SIZE)*lda + k) * _THREAD_M_SIZE + m%_THREAD_M_SIZE];
        }
        for (int bIdx = 0; bIdx < (_BLOCK_K_SIZE/_CACHE_B_K_PER_LOAD); bIdx++)
        {
            const int cache_idx = id*(_BLOCK_K_SIZE/_CACHE_B_K_PER_LOAD) + bIdx;
            const int n = nGroup + cache_idx%_BLOCK_N_SIZE;
            const int k = kIdx + cache_idx/_BLOCK_N_SIZE;
            BCache[cache_idx] = B[ldb*n + k];
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
            ACache[cache_idx] = A[((m/_THREAD_M_SIZE)*lda + k) * _THREAD_M_SIZE + m%_THREAD_M_SIZE];
        }
        for (int bIdx = 0; bIdx < (_BLOCK_K_SIZE/_CACHE_B_K_PER_LOAD); bIdx++)
        {
            const int cache_idx = id*(_BLOCK_K_SIZE/_CACHE_B_K_PER_LOAD) + bIdx;
            const int n = nGroup + cache_idx%_BLOCK_N_SIZE;
            const int k = kIdx + cache_idx/_BLOCK_N_SIZE;
            BCache[cache_idx] = B[ldb*n + k];
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
                if (activation_type == RELU)
                    cout[vecN][vecM] = cout[vecN][vecM] > 0 ? cout[vecN][vecM] : 0;
                else if (activation_type == GELU)
                    cout[vecN][vecM] = cout[vecN][vecM] * 0.5 * (1 + erff ((cout[vecN][vecM])*0.7071067811865475f));
                C[ldc*n + m] = cout[vecN][vecM];
            }
        }   
    }
}
__global__ void cuda_k_attention_kernel(const unsigned int num_heads, const unsigned int num_hidden, const unsigned int num_seq,
    const unsigned int M, const unsigned int N, const unsigned int K,
    const float *key, const unsigned int ldk, const float *B, const unsigned int ldb, float *C, const unsigned int ldc)
{
    const int batch = blockIdx.z/num_heads;
    const int head = blockIdx.z%num_heads;
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

    const float *key_head = key + batch * ldk * M + head * K;
    const float *B_head = B + batch * ldb * N + head * K;
    float *C_head = C + batch * num_heads  * ldc * N + head * ldc * N;

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
__global__ void cuda_k_attention_prob_kernel(const unsigned int num_heads, const unsigned int num_hidden, const unsigned int num_seq,
    const unsigned int M, const unsigned int N, const unsigned int K,
    const float *key, const unsigned int ldk, const float *B, const unsigned int ldb, float *C, const unsigned int ldc)
{
    const int batch = blockIdx.x;
    const int head = blockIdx.y;
    const int n = threadIdx.x;
    float *C_head = C + batch * num_heads  * ldc * N + head * ldc * N + n*ldc;
    float total = 0;
    for (unsigned int m = 0; m < M; m++)
    {
        C_head[m] /= sqrtf (K);
        C_head[m] = expf (C_head[m]);
        total += C_head[m];
    }
    for (unsigned int m = 0; m < M; m++)
    {
        C_head[m] /= total;
    }
}
__global__ void cuda_v_attention_kernel(const unsigned int num_heads, const unsigned int num_hidden, const unsigned int num_seq,
    const unsigned int M, const unsigned int N, const unsigned int K,
    const float *value, const unsigned int ldv, const float *B, const unsigned int ldb, float *C, const unsigned int ldc)
{
    const int batch = blockIdx.z/num_heads;
    const int head = blockIdx.z%num_heads;
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

    const float *val_head = value + batch * ldv * K + head * M;
    const float *B_head = B + batch * num_heads * ldb * N + head * ldb * N;
    float *C_head = C + batch * ldc * N + head * M;

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
__global__ void cuda_residual_kernel (const unsigned int num_elements, const float *A, const float *B, float *C)
{
    const int id = blockIdx.x * _BLOCK_RESIDUAL_SIZE + threadIdx.x;
    if (id < num_elements)
    {
        C[id] = A[id] + B[id];
    }
}
__global__ void cuda_layernorm_kernel(const float *input, const float *weight, const float *bias, 
    float *output, unsigned int N, unsigned int M, unsigned int ldb, unsigned int ldc)
{
    const int n = blockIdx.x * _BLOCK_RESIDUAL_SIZE + threadIdx.x;
    if (n >= N)
        return;
    float mean = 0;
    float var = 0;
    for (unsigned int m = 0; m < M; m++)
    {
        mean += input[n * ldb + m];
        var += input[n * ldb + m] * input[n * ldb + m];
    }
    mean /= M;
    var /= M;
    var -= mean * mean;
    var = 1 / sqrtf (var + 1e-12);
    for (unsigned int m = 0; m < M; m++)
    {
        output[n * ldc + m] = (input[n * ldb + m] - mean) * var * weight[m] + bias[m];
    }
}
// Wrapper function for CUDA weight.
extern "C"
{
void cuda_matmul (const unsigned int M, const unsigned int N, const unsigned int K,
		 const float *A, const unsigned int lda, const float *B, const unsigned int ldb, float *C, const unsigned int ldc,
         const float *Bias, LAYER_ACT activation_type, cudaStream_t stream)
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
        cuda_matmul_kernel<<<gridDim, blockDim, 0, stream>>>(M, N, K, A, lda, B, ldb, C, ldc, Bias, activation_type);
}
void cuda_k_attention (const float *input_1, const float *input_2, float *output, unsigned int batch_size
    , unsigned int num_heads, unsigned int num_hidden, unsigned int num_seq, cudaStream_t stream)
{
    #ifdef DEBUG
    if (input_1 == NULL)
        FPRT (stderr, "Error in naive_k_attention: input_1 is NULL.\n");
    if (input_2 == NULL)
        FPRT (stderr, "Error in naive_k_attention: input_2 is NULL.\n");
    if (output == NULL)
        FPRT (stderr, "Error in naive_k_attention: output is NULL.\n");
    #endif
    const unsigned int hidden_per_head = num_hidden / num_heads;
    const unsigned int M = num_seq;
    const unsigned int N = num_seq;
    const unsigned int K = hidden_per_head;
    const unsigned int ldk = num_hidden;
    const unsigned int ldb = num_hidden;
    const unsigned int ldc = num_seq;

    #ifdef DEBUG
    if (!(((_BLOCK_K_SIZE*_BLOCK_M_SIZE)%_THREAD_NUM) == 0 && ((_BLOCK_K_SIZE*_BLOCK_N_SIZE)%_THREAD_NUM) == 0))
    {
        printf ("ERROR! - Wrong parameter settings - (_BLOCK_K_SIZE*_BLOCK_M_SIZE)%_THREAD_NUM) = %d, ((_BLOCK_K_SIZE*_BLOCK_N_SIZE)%_THREAD_NUM) = %d\n",
        (_BLOCK_K_SIZE*_BLOCK_M_SIZE)%_THREAD_NUM), ((_BLOCK_K_SIZE*_BLOCK_N_SIZE)%_THREAD_NUM)); 
        exit(0);
    }
    #endif

    dim3 gridDim (M/_BLOCK_M_SIZE + ((M%_BLOCK_M_SIZE) > 0), N/_BLOCK_N_SIZE + ((N%_BLOCK_N_SIZE) > 0), num_heads*batch_size);
    dim3 blockDim ((_BLOCK_M_SIZE / _THREAD_M_SIZE), (_BLOCK_N_SIZE / _THREAD_N_SIZE), 1);
    cuda_k_attention_kernel<<<gridDim, blockDim, 0, stream>>> (num_heads, num_hidden, num_seq,
        M, N, K, input_2, ldk, input_1, ldb, output, ldc);
    dim3 prob_gridDim (batch_size, num_heads, 1);
    dim3 prob_blockDim (N, 1, 1);
    cuda_k_attention_prob_kernel<<<prob_gridDim, prob_blockDim, 0, stream>>> (num_heads, num_hidden, num_seq,
        M, N, K, input_2, ldk, input_1, ldb, output, ldc);
    
}
void cuda_v_attention (const float *input_1, const float *input_2, float *output, unsigned int batch_size
    , unsigned int num_heads, unsigned int num_hidden, unsigned int num_seq, cudaStream_t stream)
{
    const unsigned int hidden_per_head = num_hidden / num_heads;
    const unsigned int M = hidden_per_head;
    const unsigned int N = num_seq;
    const unsigned int K = num_seq;
    const unsigned int ldv = num_hidden;
    const unsigned int ldb = num_seq;
    const unsigned int ldc = num_hidden;

    #ifdef DEBUG
    if (!(((_BLOCK_K_SIZE*_BLOCK_M_SIZE)%_THREAD_NUM) == 0 && ((_BLOCK_K_SIZE*_BLOCK_N_SIZE)%_THREAD_NUM) == 0))
    {
        printf ("ERROR! - Wrong parameter settings - (_BLOCK_K_SIZE*_BLOCK_M_SIZE)%_THREAD_NUM) = %d, ((_BLOCK_K_SIZE*_BLOCK_N_SIZE)%_THREAD_NUM) = %d\n",
        (_BLOCK_K_SIZE*_BLOCK_M_SIZE)%_THREAD_NUM), ((_BLOCK_K_SIZE*_BLOCK_N_SIZE)%_THREAD_NUM)); 
        exit(0);
    }
    #endif

    dim3 gridDim (M/_BLOCK_M_SIZE + ((M%_BLOCK_M_SIZE) > 0), N/_BLOCK_N_SIZE + ((N%_BLOCK_N_SIZE) > 0), num_heads*batch_size);
    dim3 blockDim ((_BLOCK_M_SIZE / _THREAD_M_SIZE), (_BLOCK_N_SIZE / _THREAD_N_SIZE), 1);
    cuda_v_attention_kernel<<<gridDim, blockDim, 0, stream>>> (num_heads, num_hidden, num_seq,
        M, N, K, input_2, ldv, input_1, ldb, output, ldc);
}
void cuda_residual (const float *input_1, const float *input_2, float *output, unsigned int num_elements
    , cudaStream_t stream)
{
    dim3 gridDim (num_elements/_BLOCK_RESIDUAL_SIZE + ((num_elements%_BLOCK_RESIDUAL_SIZE) > 0), 1, 1);
    dim3 blockDim (_BLOCK_RESIDUAL_SIZE, 1, 1);
    cuda_residual_kernel<<<gridDim, blockDim, 0, stream>>> (num_elements, input_1, input_2, output);
}
void cuda_layernorm (const float *input, const float *weight, const float *bias, 
    float *output, unsigned int N, unsigned int M, unsigned int ldb, unsigned int ldc, cudaStream_t stream)
{
    dim3 prob_gridDim (N/_BLOCK_LAYERNORM_SIZE + ((N%_BLOCK_LAYERNORM_SIZE) > 0), 1, 1);
    dim3 prob_blockDim (_BLOCK_LAYERNORM_SIZE, 1, 1);
    cuda_layernorm_kernel<<<prob_gridDim, prob_blockDim, 0, stream>>> (input, weight, bias, output, N, M, ldb, ldc);
}
}