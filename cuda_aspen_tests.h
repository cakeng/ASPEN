#ifndef CUDA_ASPEN_TESTS_H
#define CUDA_ASPEN_TESTS_H

#include <set>
#include <vector>
#include <cuda_runtime.h>
#include <cstdio>
#include <cublas_v2.h>
#include <cblas.h>

#define CUBLAS_CHECK(val) { \
	if (val != CUBLAS_STATUS_SUCCESS) { \
		fprintf(stderr, "Error at line %d in file %s\n", __LINE__, __FILE__); \
		exit(1); \
	} \
}

class aspen_mat_mul
{
public:
    bool is_calculated();
    void add_child(aspen_mat_mul *child);
    void allocate_cuda_memory();
    void copy_A_B_to_cuda();
    void copy_C_from_cuda();
    void set_host_memory(float *A, float *B, float *C);
    void set_cuda_memory(float *A_cuda, float *B_cuda, float *c_float);
    void set_cuda_stream(cudaStream_t stream);
    void set_cuda_handle(cublasHandle_t handle);
    void synchronize();
    std::vector<aspen_mat_mul *> split_mat_mul (int split_M, int split_N);

    aspen_mat_mul(int M, int N, int K, float alpha,
        float *A, int stride_A, float *B, int stride_B, float beta, float *C, int stride_C);
    ~aspen_mat_mul(); 
    void run_cuBLAS();
    void run_cpu();

private:
    bool is_calculation_done;
    std::vector<aspen_mat_mul *> children;
    int num_parents;
    int calculated_parents;

    int C_midx, C_nidx;
    int M, N, K;
    int stride_A, stride_B, stride_C;
    float alpha, beta;
    float *A, *B, *C;
    bool own_A, own_B, own_C;
    
    float *A_cuda, *B_cuda, *C_cuda;
    bool own_A_cuda, own_B_cuda, own_C_cuda;
    cublasHandle_t handle;
    bool own_handle;
    cudaStream_t stream;
    bool own_stream;
};
#endif