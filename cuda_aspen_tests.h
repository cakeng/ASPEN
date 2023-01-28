#ifndef CUDA_ASPEN_TESTS_H
#define CUDA_ASPEN_TESTS_H

#include <set>
#include <vector>
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cblas.h>

#include <pthread.h>

static inline cudaError_t check_CUDA(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s, at line %d in file %s\n"
        , cudaGetErrorString(result), __LINE__, __FILE__);
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

static inline cublasStatus_t check_cuBLAS(cublasStatus_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "cuBLAS Runtime Error: %d, at line %d in file %s\n"
        , result, __LINE__, __FILE__);
    assert(result == CUBLAS_STATUS_SUCCESS);
  }
#endif
  return result;
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
            );

class aspen_mat_mul
{
public:
    bool is_calculated();
    void add_child(aspen_mat_mul *child);
    void allocate_cuda_memory_A_C();
    void copy_A_to_cuda();
    void copy_B_to_cuda();
    void copy_C_from_cuda();
    void set_host_memory(float *A, float *B, float *C);
    void set_cuda_memory(float *A_cuda, float *B_cuda, float *c_float);
    void set_cuda_stream(cudaStream_t stream);
    void set_cuda_handle(cublasHandle_t handle);
    void print_handle_and_stream();
    void synchronize();
    void set_mat_C (float val);
    std::vector<aspen_mat_mul *> split_mat_mul_by_num (int M_num, int N_num);
    std::vector<aspen_mat_mul *> split_mat_mul_by_size (int M_size, int N_size);

    aspen_mat_mul(int M, int N, int K, float alpha,
        float *A, int stride_A, float *B, int stride_B, float beta, float *C, int stride_C);
    ~aspen_mat_mul(); 
    void run_cuBLAS();
    void run_cpu();
    void run_custom_CUDA_GEMM();
    void run_cuBLAS_strided_GEMV(int N_stride);
    void run_cuBLAS_strided_GEMM(int N_stride);
    void run_cuBLAS_batched_GEMM(std::vector<aspen_mat_mul *> batch);

private:
    bool is_calculation_done;
    std::vector<aspen_mat_mul *> children;
    int num_parents;
    int calculated_parents;
    int id;

    int C_midx, C_nidx;
    int M, N, K;
    int stride_A, stride_B, stride_C;
    float alpha, beta;
    float *A, *B, *C;
    bool own_A, own_B, own_C;
    
    float *A_cuda, *B_cuda, *C_cuda;
    void *temp_cuda;
    cublasHandle_t handle;
    cudaStream_t stream;
};

typedef enum run_type
{
    RUN_CPU,
    RUN_CUBLAS,
    RUN_CUSTOM_CUDA_GEMM
} run_type;

class aspen_pthread
{
    private:
    int id;
    run_type r_type;

    std::vector<aspen_mat_mul *> mat_muls;
    cublasHandle_t handle;
    cudaStream_t stream;
    
    pthread_t thread;
    pthread_mutex_t mutex;
    pthread_cond_t cond_host_to_child, cond_child_to_host;

    bool running;

    static void *thread_func(void *arg);

    public:
    aspen_pthread();
    ~aspen_pthread();
    void add_mat_mul(aspen_mat_mul *mat_mul);

    void run_cuBLAS();
    void run_cpu();
    void run_custom_CUDA_GEMM();

    void stop();
    void wait();
};

#endif