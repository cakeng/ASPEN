#include "cuda_aspen_tests.h"

aspen_mat_mul::aspen_mat_mul(int M, int N, int K, float alpha,
    float *A, int stride_A, float *B, int stride_B, float beta, float *C, int stride_C)
{
    this->A = A;
    this->B = B;
    this->C = C;
    this->M = M;
    this->N = N;
    this->K = K;
    this->stride_A = stride_A;
    this->stride_B = stride_B;
    this->stride_C = stride_C;
    this->alpha = alpha;
    this->beta = beta;

    this->is_calculation_done = false;
    this->num_parents = 0;
    this->calculated_parents = 0;

    this->A_cuda = NULL;
    this->B_cuda = NULL;
    this->C_cuda = NULL;
    this->stream = NULL;
    cublasCreate(&this->handle);
}

aspen_mat_mul::~aspen_mat_mul()
{
}

void aspen_mat_mul::set_host_memory(float *A, float *B, float *C)
{
    this->A = A;
    this->B = B;
    this->C = C;
}

void aspen_mat_mul::allocate_cuda_memory()
{
    cudaMalloc((void **)&this->A_cuda, this->M * this->K * sizeof(float));
    cudaMalloc((void **)&this->B_cuda, this->K * this->N * sizeof(float));
    cudaMalloc((void **)&this->C_cuda, this->M * this->N * sizeof(float));
}

void aspen_mat_mul::copy_A_B_to_cuda()
{
    cudaMemcpyAsync(this->A_cuda, this->A, this->M * this->K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(this->B_cuda, this->B, this->K * this->N * sizeof(float), cudaMemcpyHostToDevice);
}

void aspen_mat_mul::copy_C_from_cuda()
{
    cudaMemcpyAsync(this->C, this->C_cuda, this->M * this->N * sizeof(float), cudaMemcpyDeviceToHost);
}

void aspen_mat_mul::set_cuda_memory(float *A_cuda, float *B_cuda, float *C_cuda)
{
    this->A_cuda = A_cuda;
    this->B_cuda = B_cuda;
    this->C_cuda = C_cuda;
}

void aspen_mat_mul::set_cuda_stream(cudaStream_t stream)
{
    this->stream = stream;
}

void aspen_mat_mul::set_cuda_handle(cublasHandle_t handle)
{
    if (this->handle != NULL)
        cublasDestroy(this->handle);
    this->handle = handle;
}

void aspen_mat_mul::synchronize()
{
    cudaStreamSynchronize(this->stream);
}

void aspen_mat_mul::add_child(aspen_mat_mul *child)
{
    this->children.push_back(child);
    child->num_parents++;
}

bool aspen_mat_mul::is_calculated()
{
    return this->is_calculation_done;
}

void aspen_mat_mul::run_cuBLAS()
{
    if (this->is_calculation_done)
        return;
    cublasSetStream (this->handle, this->stream);
    cublasSgemm (this->handle, CUBLAS_OP_N, CUBLAS_OP_N, this->N, this->M, this->K, 
        &this->alpha, this->B_cuda, this->stride_B, this->A_cuda, this->stride_A, 
             &this->beta, this->C_cuda, this->stride_C);
}

void aspen_mat_mul::run_cpu()
{
    if (this->is_calculation_done)
        return;
    cblas_sgemm (CblasRowMajor, CblasNoTrans, CblasNoTrans, this->M, this->N, this->K, 
        this->alpha, this->A, this->stride_A, this->B, this->stride_B, 
             this->beta, this->C, this->stride_C);
}