extern "C"
{
#include "cuda_kernels.h"

__global__ void cuda_tiled_conv2d_kernel(
    const unsigned int M, const unsigned int N, 
    const float **col_ptr_arr, const unsigned int col_per_n, const unsigned int K_col,
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
    // float* A_ptr [(_BLOCK_K_SIZE*_BLOCK_M_SIZE)/_THREAD_NUM];
    // float* B_ptr [(_BLOCK_K_SIZE*_BLOCK_N_SIZE)/_THREAD_NUM];
    for (int vecN = 0; vecN < _THREAD_N_SIZE; vecN++)
    {
        for (int vecM = 0; vecM < _THREAD_M_SIZE; vecM++)
        {
            cout[vecN][vecM] = Bias[mGroup + mLocal + vecM];
        }   
    }

    // for (int col = 0; col < col_per_n; col++)
    // {
    //     for (int vecN = 0; vecN < _THREAD_N_SIZE; vecN++)
    //     {
    //         for (int vecM = 0; vecM < _THREAD_M_SIZE; vecM++)
    //         {
    //             const int m = mGroup + mLocal + vecM;
    //             const int n = nGroup + nLocal + vecN;
    //             if (m < M &&  n < N)
    //             {
    //                 const float *B_col = col_ptr_arr[n*col_per_n + col];
    //                 for (int k = 0; k < K_col; k++)
    //                 {
    //                     int A_k = col*K_col + k;
    //                     cout[vecN][vecM] += A[((m/_A_MIN_DIM)*lda + A_k) * _A_MIN_DIM + m%_A_MIN_DIM] * B_col[k];
    //                 }
    //             }
    //         }
    //     }
    // }

    for (int col = 0; col < col_per_n; col++)
    {
        int kIdx = 0;  
        if (K_col%_BLOCK_K_SIZE)
        {
            // Load caches.
            for (int aIdx = 0; aIdx < (_BLOCK_K_SIZE/_CACHE_A_K_PER_LOAD); aIdx++)
            {
                const int cache_idx = id*(_BLOCK_K_SIZE/_CACHE_A_K_PER_LOAD) + aIdx;
                const int m = mGroup + cache_idx%_BLOCK_M_SIZE;
                const int k = kIdx + cache_idx/_BLOCK_M_SIZE;
                ACache[cache_idx] = A[((m/_A_MIN_DIM)*lda + col*K_col + k) * _A_MIN_DIM + m%_A_MIN_DIM];

            }
            for (int bIdx = 0; bIdx < (_BLOCK_K_SIZE/_CACHE_B_K_PER_LOAD); bIdx++)
            {
                const int cache_idx = id*(_BLOCK_K_SIZE/_CACHE_B_K_PER_LOAD) + bIdx;
                const int n = nGroup + cache_idx%_BLOCK_N_SIZE;
                if (n >= N)
                    continue;
                const int k = kIdx + cache_idx/_BLOCK_N_SIZE;
                const float *B_col = col_ptr_arr[n*col_per_n + col];
                BCache[cache_idx] = B_col[k];
                // if (col_ptr_arr[n*col_per_n + col] != -1)
                // {
                //     const float *B_col = B + col_ptr_arr[n*col_per_n + col] * ldb;
                //     BCache[cache_idx] = B_col[k];
                // }   
                // else 
                //     BCache[cache_idx] = 0;
            }
            __syncthreads();
            // printf ("Thread %d: %3.3f\n", id, cout[0][0]);
            for (; kIdx < K_col%_BLOCK_K_SIZE; kIdx++)
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
        for (; kIdx < K_col; kIdx += _BLOCK_K_SIZE)
        {
            // Load caches.
            for (int aIdx = 0; aIdx < (_BLOCK_K_SIZE/_CACHE_A_K_PER_LOAD); aIdx++)
            {
                const int cache_idx = id*(_BLOCK_K_SIZE/_CACHE_A_K_PER_LOAD) + aIdx;
                const int m = mGroup + cache_idx%_BLOCK_M_SIZE;
                const int k = kIdx + cache_idx/_BLOCK_M_SIZE;
                ACache[cache_idx] = A[((m/_A_MIN_DIM)*lda + col*K_col + k) * _A_MIN_DIM + m%_A_MIN_DIM];
  
            }
            for (int bIdx = 0; bIdx < (_BLOCK_K_SIZE/_CACHE_B_K_PER_LOAD); bIdx++)
            {
                const int cache_idx = id*(_BLOCK_K_SIZE/_CACHE_B_K_PER_LOAD) + bIdx;
                const int n = nGroup + cache_idx%_BLOCK_N_SIZE;
                if (n >= N)
                    continue;
                const int k = kIdx + cache_idx/_BLOCK_N_SIZE;
                const float *B_col = col_ptr_arr[n*col_per_n + col];
                BCache[cache_idx] = B_col[k];
                // if (col_ptr_arr[n*col_per_n + col] != -1)
                // {
                //     const float *B_col = B + col_ptr_arr[n*col_per_n + col] * ldb;
                //     BCache[cache_idx] = B_col[k];
                // }   
                // else 
                //     BCache[cache_idx] = 0;
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
                else if (activation_type == GELU_ACCURATE)
                    cout[vecN][vecM] = cout[vecN][vecM] * 0.5 * (1 + tanhf (0.7978845608028654f * (cout[vecN][vecM] + 0.044715f * powf (cout[vecN][vecM], 3))));
                C[ldc*n + m] = cout[vecN][vecM];
            }
        }   
    }
}
__global__ void cuda_tiled_maxpool_kernel(
    const unsigned int M, const unsigned int N, const unsigned int K_pos, 
    const float **col_ptr_arr, const unsigned int col_per_n,
    float *C, const unsigned int ldc,
    LAYER_ACT activation_type)
{
    const int mLocal = threadIdx.x*_THREAD_M_SIZE;
    const int nLocal = threadIdx.y*_THREAD_N_SIZE; 
    const int mGroup = blockIdx.x*_BLOCK_M_SIZE;
    const int nGroup = blockIdx.y*_BLOCK_N_SIZE;
    float cout[_THREAD_N_SIZE][_THREAD_M_SIZE];
    
    for (int vecN = 0; vecN < _THREAD_N_SIZE; vecN++)
    {
        for (int vecM = 0; vecM < _THREAD_M_SIZE; vecM++)
        {
            cout[vecN][vecM] = -INFINITY;
        }   
    }

    for (int col = 0; col < col_per_n; col++)
    {
        for (int vecN = 0; vecN < _THREAD_N_SIZE; vecN++)
        {
            const int n = nGroup + nLocal + vecN;
            const float *B_col = col_ptr_arr[n*col_per_n + col] + K_pos;
            for (int vecM = 0; vecM < _THREAD_M_SIZE; vecM++)
            {
                const int m = mGroup + mLocal + vecM;
                if (m < M &&  n < N)
                {
                    cout[vecN][vecM] = cout[vecN][vecM] > B_col[m] ? cout[vecN][vecM] : B_col[m];
                }
            }
        }
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
                else if (activation_type == GELU_ACCURATE)
                    cout[vecN][vecM] = cout[vecN][vecM] * 0.5 * (1 + tanhf (0.7978845608028654f * (cout[vecN][vecM] + 0.044715f * powf (cout[vecN][vecM], 3))));
                C[ldc*n + m] = cout[vecN][vecM];
            }
        }   
    }
}
__global__ void cuda_tiled_avgpool_kernel(
    const unsigned int M, const unsigned int N, const unsigned int K_pos,
    const float **col_ptr_arr, const unsigned int col_per_n,
    float *C, const unsigned int ldc,
    LAYER_ACT activation_type)
{
    const int mLocal = threadIdx.x*_THREAD_M_SIZE;
    const int nLocal = threadIdx.y*_THREAD_N_SIZE; 
    const int mGroup = blockIdx.x*_BLOCK_M_SIZE;
    const int nGroup = blockIdx.y*_BLOCK_N_SIZE;
    float cout[_THREAD_N_SIZE][_THREAD_M_SIZE];
    
    for (int vecN = 0; vecN < _THREAD_N_SIZE; vecN++)
    {
        for (int vecM = 0; vecM < _THREAD_M_SIZE; vecM++)
        {
            cout[vecN][vecM] = 0;
        }   
    }

    for (int col = 0; col < col_per_n; col++)
    {
        for (int vecN = 0; vecN < _THREAD_N_SIZE; vecN++)
        {
            const int n = nGroup + nLocal + vecN;
            const float *B_col = col_ptr_arr[n*col_per_n + col] + K_pos;
            for (int vecM = 0; vecM < _THREAD_M_SIZE; vecM++)
            {
                const int m = mGroup + mLocal + vecM;
                if (m < M &&  n < N)
                {
                    cout[vecN][vecM] += B_col[m];
                }
            }
        }
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
                cout[vecN][vecM] = cout[vecN][vecM] / col_per_n;
                if (activation_type == RELU)
                    cout[vecN][vecM] = cout[vecN][vecM] > 0 ? cout[vecN][vecM] : 0;
                else if (activation_type == GELU)
                    cout[vecN][vecM] = cout[vecN][vecM] * 0.5 * (1 + erff ((cout[vecN][vecM])*0.7071067811865475f));
                else if (activation_type == GELU_ACCURATE)
                    cout[vecN][vecM] = cout[vecN][vecM] * 0.5 * (1 + tanhf (0.7978845608028654f * (cout[vecN][vecM] + 0.044715f * powf (cout[vecN][vecM], 3))));
                C[ldc*n + m] = cout[vecN][vecM];
            }
        }   
    }
}
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
        else if (activation_type == GELU_ACCURATE)
            val = val * 0.5 * (1 + tanhf (0.7978845608028654f + (val + 0.044715f * powf (val, 3))));
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

void cuda_tiled_conv2d (const unsigned int M, const unsigned int N, 
    const float **col_ptr_arr, const unsigned int col_per_n, const unsigned int K_col,
    const float *A, const unsigned int lda, const float *B, const unsigned int ldb, float *C, const unsigned int ldc,
    const float *Bias, LAYER_ACT activation_type, cudaStream_t stream)
{
    int p1 = (_BLOCK_K_SIZE*_BLOCK_M_SIZE)%_THREAD_NUM;
    int p2 = (_BLOCK_K_SIZE*_BLOCK_N_SIZE)%_THREAD_NUM;
    if (!(p1 == 0 && p2 == 0))
    {
        printf ("ERROR! - Wrong parameter settings - (_BLOCK_K_SIZE*_BLOCK_M_SIZE)%_THREAD_NUM) = %d, ((_BLOCK_K_SIZE*_BLOCK_N_SIZE)%_THREAD_NUM) = %d\n",
        p1, p2); 
        exit(0);
    }
    dim3 gridDim (M/_BLOCK_M_SIZE + ((M%_BLOCK_M_SIZE) > 0), N/_BLOCK_N_SIZE + ((N%_BLOCK_N_SIZE) > 0), 1);
    dim3 blockDim ((_BLOCK_M_SIZE / _THREAD_M_SIZE), (_BLOCK_N_SIZE / _THREAD_N_SIZE), 1);
    cuda_tiled_conv2d_kernel<<<gridDim, blockDim, 0, stream>>>(M, N, col_ptr_arr, col_per_n, K_col,
        A, lda, B, ldb, C, ldc, Bias, activation_type);
}
void cuda_tiled_maxpool(
    const unsigned int M, const unsigned int N, const unsigned int K_pos, 
    const float **col_ptr_arr, const unsigned int col_per_n,
    float *C, const unsigned int ldc,
    LAYER_ACT activation_type, cudaStream_t stream)
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
        cuda_tiled_maxpool_kernel<<<gridDim, blockDim, 0, stream>>>(M, N, K_pos,
        col_ptr_arr, col_per_n, C, ldc, activation_type);
}
void cuda_tiled_avgpool(
    const unsigned int M, const unsigned int N, const unsigned int K_pos,
    const float **col_ptr_arr, const unsigned int col_per_n,
    float *C, const unsigned int ldc,
    LAYER_ACT activation_type, cudaStream_t stream)
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
        cuda_tiled_avgpool_kernel<<<gridDim, blockDim, 0, stream>>>(M, N, K_pos,
        col_ptr_arr, col_per_n, C, ldc, activation_type);
}
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