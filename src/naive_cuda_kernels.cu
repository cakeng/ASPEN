extern "C"
{
    #include "cuda_kernels.h"
}

#define _BLOCK_K_SIZE 32
#define _BLOCK_M_SIZE 64
#define _BLOCK_N_SIZE 64
#define _THREAD_M_SIZE 8
#define _THREAD_N_SIZE 4
#define _THREAD_NUM ((_BLOCK_M_SIZE / _THREAD_M_SIZE) * (_BLOCK_N_SIZE / _THREAD_N_SIZE)) // 128
#define _CACHE_A_K_PER_LOAD (_THREAD_NUM / _BLOCK_M_SIZE) // 2
#define _CACHE_B_K_PER_LOAD (_THREAD_NUM / _BLOCK_N_SIZE) // 2

// Custom CUDA GEMM kernel.
__global__ void sgemm(const unsigned int M, const unsigned int N, const unsigned int K,
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
            BCache[cache_idx] = B[ldb*(nGroup + cache_idx%_BLOCK_N_SIZE) + (kIdx + cache_idx/_BLOCK_N_SIZE)];
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
            BCache[cache_idx] = B[ldb*(nGroup + cache_idx%_BLOCK_N_SIZE) + kIdx + cache_idx/_BLOCK_N_SIZE];
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
    const int m = mGroup + mLocal;
    const int n = nGroup + nLocal;
    for (int vecN = 0; vecN < _THREAD_N_SIZE; vecN++)
    {
        for (int vecM = 0; vecM < _THREAD_M_SIZE; vecM++)
        {
            if ((m + vecM) < M &&  (n + vecN) < N)
            {
                if (activation_type == RELU)
                    cout[vecN][vecM] = cout[vecN][vecM] > 0 ? cout[vecN][vecM] : 0;
                else if (activation_type == GELU)
                    cout[vecN][vecM] = cout[vecN][vecM] * 0.5 * (1 + erff ((cout[vecN][vecM])*0.7071067811865475f));
                C[ldc*(n + vecN) + (m + vecM)] = cout[vecN][vecM];
            }
        }   
    }
}

// Wrapper function for CUDA kernel.
extern "C"
void cuda_matmul (const unsigned int M, const unsigned int N, const unsigned int K,
		 const float *A, const unsigned int lda, const float *B, const unsigned int ldb, float *C, const unsigned int ldc,
         const float *Bias, LAYER_ACT activation_type)
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
        sgemm<<<gridDim, blockDim, 0>>>(M, N, K, A, lda, B, ldb, C, ldc, Bias, activation_type);
}