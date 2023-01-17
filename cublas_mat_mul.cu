#include "mat_mul.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cublas_v2.h>

#define CUBLAS_CHECK(val)                                                         \
    {                                                                             \
        if (val != CUBLAS_STATUS_SUCCESS)                                         \
        {                                                                         \
            fprintf(stderr, "Error at line %d in file %s\n", __LINE__, __FILE__); \
            exit(1);                                                              \
        }                                                                         \
    }

static cublasHandle_t handle;
static float *a_d, *b_d, *c_d;
static float alpha = 1.0f;
static float beta = 0.0f;

void cublas_mat_mul_write_to_gpu(float *A, float *B, float *C, int M, int N, int K)
{
    CUBLAS_CHECK(cublasSetVector(M * K, sizeof(float), A, 1, a_d, 1));

    CUBLAS_CHECK(cublasSetVector(K * N, sizeof(float), B, 1, b_d, 1));

    cudaDeviceSynchronize();
}

void cublas_mat_mul_read_from_gpu(float *A, float *B, float *C, int M, int N, int K)
{
    CUBLAS_CHECK(cublasGetVector(M * N, sizeof(float), c_d, 1, C, 1));
    cudaDeviceSynchronize();
}

void cublas_mat_mul(float *A, float *B, float *C, int M, int N, int K, int skip_data_movement)
{
    if (!skip_data_movement)
        cublas_mat_mul_write_to_gpu(A, B, C, M, N, K);

    cublasSgemm (handle, CUBLAS_OP_N, CUBLAS_OP_N
        , N, M, K, &alpha, b_d, N, a_d, K, &beta, c_d, N);

    if (!skip_data_movement)
        cublas_mat_mul_read_from_gpu(A, B, C, M, N, K);
}

void cublas_mat_mul_init(float *A, float *B, float *C, int M, int N, int K)
{
    // Allocate the device input matrixs for A, B, C;
    cudaMalloc((void **)&a_d, M * K * sizeof(float));
    cudaMalloc((void **)&b_d, K * N * sizeof(float));
    cudaMalloc((void **)&c_d, M * N * sizeof(float));
    cublasCreate (&handle);
}

void cublas_mat_mul_free(float *A, float *B, float *C, int M, int N, int K)
{
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    cublasDestroy(handle);
}