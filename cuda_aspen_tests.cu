#include <cuda_runtime.h>
#include <cstdio>
#include "cuda_aspen_tests.h"

#define _BLOCK_K_SIZE 16
#define _BLOCK_M_SIZE 128
#define _BLOCK_N_SIZE 128
#define _THREAD_M_SIZE 8
#define _THREAD_N_SIZE 8
#define _OUTC_CHUNK 1
#define _THREAD_NUM ((_BLOCK_M_SIZE / _THREAD_M_SIZE) * (_BLOCK_N_SIZE / _THREAD_N_SIZE))

__global__ void sgemm(const float *A, const float *B, float *C, const int M, const int N, const int K)
{
    const int mLocal = threadIdx.x*_THREAD_M_SIZE;
    const int nLocal = threadIdx.y*_THREAD_N_SIZE; 
    const int mGroup = blockIdx.x*_BLOCK_M_SIZE;
    const int nGroup = blockIdx.y*_BLOCK_N_SIZE;
    const int id = threadIdx.x*(_BLOCK_N_SIZE / _THREAD_N_SIZE) + threadIdx.y;
    const int m = mGroup + mLocal;
    const int n = nGroup + nLocal;
    const int kRem = K%_BLOCK_K_SIZE;
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

    // printf ("Thread %d: %3.3f\n", id, cout[0][0]);

    const int kEnd = K - kRem;
    int kIdx = 0;  
    for (; kIdx < kEnd; kIdx += _BLOCK_K_SIZE)
    {
        // Load caches.
        for (int aIdx = 0; aIdx < ((_BLOCK_K_SIZE*_BLOCK_M_SIZE)/_THREAD_NUM); aIdx++)
        {
            const int cache_idx = id*((_BLOCK_K_SIZE*_BLOCK_M_SIZE)/_THREAD_NUM) + aIdx;
            const int cache_m = cache_idx%_BLOCK_M_SIZE;
            const int cache_k = cache_idx/_BLOCK_M_SIZE;
            ACache[cache_idx] = A[K*(((mGroup + cache_m)/_OUTC_CHUNK)*_OUTC_CHUNK) 
                + (kIdx + cache_k)*_OUTC_CHUNK + (mGroup + cache_m)%_OUTC_CHUNK];
        }
        for (int bIdx = 0; bIdx < ((_BLOCK_K_SIZE*_BLOCK_N_SIZE)/_THREAD_NUM); bIdx++)
        {
            const int cache_idx = id*((_BLOCK_K_SIZE*_BLOCK_N_SIZE)/_THREAD_NUM) + bIdx;
            const int cache_n = cache_idx%_BLOCK_N_SIZE;
            const int cache_k = cache_idx/_BLOCK_N_SIZE;
            BCache[cache_idx] = B[K*(nGroup + cache_n) + (kIdx + cache_k)];
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
    // Remainning K
    if (kRem)
    {
        // Load caches.
        for (int aIdx = 0; aIdx < ((_BLOCK_K_SIZE*_BLOCK_M_SIZE)/_THREAD_NUM); aIdx++)
        {
            const int cache_idx = id*((_BLOCK_K_SIZE*_BLOCK_M_SIZE)/_THREAD_NUM) + aIdx;
            const int cache_m = cache_idx%_BLOCK_M_SIZE;
            const int cache_k = cache_idx/_BLOCK_M_SIZE;
            ACache[cache_idx] = A[K*(((mGroup + cache_m)/_OUTC_CHUNK)*_OUTC_CHUNK) 
                + (kIdx + cache_k)*_OUTC_CHUNK + (mGroup + cache_m)%_OUTC_CHUNK];
        }
        for (int bIdx = 0; bIdx < ((_BLOCK_K_SIZE*_BLOCK_N_SIZE)/_THREAD_NUM); bIdx++)
        {
            const int cache_idx = id*((_BLOCK_K_SIZE*_BLOCK_N_SIZE)/_THREAD_NUM) + bIdx;
            const int cache_n = cache_idx%_BLOCK_N_SIZE;
            const int cache_k = cache_idx/_BLOCK_N_SIZE;
            BCache[cache_idx] = B[K*(nGroup + cache_n) + (kIdx + cache_k)];
        }
        __syncthreads();
        // printf ("Thread %d: %3.3f\n", id, cout[0][0]);
        for (int kk = 0; kk < kRem; kk++)
        {
            // Calculate.
            for (int vecN = 0; vecN < _THREAD_N_SIZE; vecN++)
            {
                for (int vecM = 0; vecM < _THREAD_M_SIZE; vecM++)
                {
                    // printf ("B%dT%d: (%d, %d) %3.3f, %3.3f\n", blockIdx.x + blockIdx.y
                    //     ,id, vecN, vecM, ACache[kk*_BLOCK_M_SIZE + mLocal + vecM], BCache[kk*_BLOCK_N_SIZE + nLocal + vecN]);
                    cout[vecN][vecM] += ACache[kk*_BLOCK_M_SIZE + mLocal + vecM] * BCache[kk*_BLOCK_N_SIZE + nLocal + vecN];
                }   
            }
        }
        // Sync threads.
        __syncthreads();
        // printf ("Thread %d: %3.3f\n", id, cout[0][0]);
    }
    // Save results
    for (int vecN = 0; vecN < _THREAD_N_SIZE; vecN++)
    {
        for (int vecM = 0; vecM < _THREAD_M_SIZE; vecM++)
        {
            if (m + vecM < M &&  n + vecN < N)
                C[M*(n + vecN) + (m + vecM)] = cout[vecN][vecM];
        }   
    }
}

void custom_CUDA_mat_mul(
            int M,
            int N,
            int K,
            const float* alpha,
            const float* A,
            int lda,
            const float* B,
            int ldb,
            const float* beta, 
            float* C,
            int ldc,
            cudaStream_t stream
            )
{
        dim3 gridDim (M/_BLOCK_M_SIZE + ((M%_BLOCK_M_SIZE) > 0), N/_BLOCK_N_SIZE + ((N%_BLOCK_N_SIZE) > 0), 1);
        dim3 blockDim ((_BLOCK_M_SIZE / _THREAD_M_SIZE), (_BLOCK_N_SIZE / _THREAD_N_SIZE), 1);
        sgemm<<<gridDim, blockDim, 0, stream>>>(A, B, C, M, N, K);
}