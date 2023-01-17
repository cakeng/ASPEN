#include "mat_mul.h"

#include <cuda_runtime.h>
#include <cstdio>

#define _BLOCK_K_SIZE 16
#define _BLOCK_M_SIZE 128
#define _BLOCK_N_SIZE 128
#define _THREAD_M_SIZE 8
#define _THREAD_N_SIZE 8
#define _OUTC_CHUNK 8
#define _THREAD_NUM ((_BLOCK_M_SIZE / _THREAD_M_SIZE) * (_BLOCK_N_SIZE / _THREAD_N_SIZE))

int num_devices = 0;

static float *a_d[16], *b_d[16], *c_d[16];

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

void mat_mul_write_to_gpu(float *A, float *B, float *C, int M, int N, int K)
{
    for (int i = 0; i < num_devices; i++)
    {
        cudaSetDevice(i);
        cudaMemcpyAsync(a_d[i], A, (M + (_OUTC_CHUNK - (M%_OUTC_CHUNK))) * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(b_d[i], B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    }

    cudaDeviceSynchronize();
}

void mat_mul_read_from_gpu(float *A, float *B, float *C, int M, int N, int K)
{
    for (int i = 0; i < num_devices; i++)
    {
        cudaSetDevice(i);
        cudaMemcpyAsync(C, c_d[i], M * N * sizeof(float), cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();
}

void mat_mul(float *A, float *B, float *C, int M, int N, int K, int skip_data_movement)
{
    if (!skip_data_movement)
        mat_mul_write_to_gpu (A, B, C, M, N, K);
    
    for (int i = 0; i < num_devices; i++)
    {
        cudaSetDevice(i);
        dim3 gridDim (M/_BLOCK_M_SIZE + ((M%_BLOCK_M_SIZE) > 0), N/_BLOCK_N_SIZE + ((N%_BLOCK_N_SIZE) > 0), 1);
        dim3 blockDim ((_BLOCK_M_SIZE / _THREAD_M_SIZE), (_BLOCK_N_SIZE / _THREAD_N_SIZE), 1);
        sgemm<<<gridDim, blockDim>>>(a_d[i], b_d[i], c_d[i], M, N, K);
    }
    
    if (!skip_data_movement)
        mat_mul_read_from_gpu (A, B, C, M, N, K);
}

void mat_mul_init(float *A, float *B, float *C, int M, int N, int K)
{
    printf ("Block Settings: M: %d, N: %d, K: %d, VecM: %d, VecN: %d, Thread Num: %d, ACache: %d, BCache: %d\n",
        _BLOCK_M_SIZE, _BLOCK_N_SIZE, _BLOCK_K_SIZE, _THREAD_M_SIZE, _THREAD_N_SIZE, _THREAD_NUM, _BLOCK_K_SIZE*_BLOCK_M_SIZE, _BLOCK_K_SIZE*_BLOCK_N_SIZE);
    printf ("Num blocks: %d\n", (_BLOCK_M_SIZE / _THREAD_M_SIZE)*(_BLOCK_N_SIZE / _THREAD_N_SIZE));
    if (!(((_BLOCK_K_SIZE*_BLOCK_M_SIZE)%_THREAD_NUM) == 0 && ((_BLOCK_K_SIZE*_BLOCK_N_SIZE)%_THREAD_NUM) == 0))
    {
        printf ("ERROR! - Wrong parameter settings.\n"); 
        exit(0);
    }
    cudaGetDeviceCount(&num_devices);

    printf("Using %d devices\n", num_devices);
    for (int i = 0; i < num_devices; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("[GPU %d] %s\n", i, prop.name);
    }

    if (num_devices <= 0)
    {
        printf("No CUDA device found. Aborting\n");
        exit(1);
    }

    for (int i = 0; i < num_devices; i++)
    {
        cudaSetDevice(i);
        cudaMalloc(&a_d[i], (M+_BLOCK_M_SIZE) * (K+_BLOCK_K_SIZE) * sizeof(float));
        cudaMemset(a_d[i], 0, (M+_BLOCK_M_SIZE) * (K+_BLOCK_K_SIZE) * sizeof(float));
        cudaMalloc(&b_d[i], (K+_BLOCK_K_SIZE) * (N+_BLOCK_N_SIZE) * sizeof(float));
        cudaMemset(b_d[i], 0, (M+_BLOCK_M_SIZE) * (K+_BLOCK_K_SIZE) * sizeof(float));
        cudaMalloc(&c_d[i], M * N * sizeof(float));
        cudaMemset(c_d[i], 0, M * N * sizeof(float));
        cudaDeviceSynchronize();
    }
}