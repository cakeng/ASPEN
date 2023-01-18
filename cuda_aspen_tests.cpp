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
    this->handle = NULL;
}

aspen_mat_mul::~aspen_mat_mul()
{
}

void aspen_mat_mul::print_handle_and_stream()
{
    printf("handle: %p, stream: %p\n", this->handle, this->stream);
}

void aspen_mat_mul::set_host_memory(float *A, float *B, float *C)
{
    this->A = A;
    this->B = B;
    this->C = C;
}

void aspen_mat_mul::allocate_cuda_memory_A_C ()
{
    cudaMalloc((void **)&this->A_cuda, this->M * this->K * sizeof(float));
    cudaMalloc((void **)&this->B_cuda, this->K * this->N * sizeof(float));
    cudaMalloc((void **)&this->C_cuda, this->M * this->N * sizeof(float));
}

void aspen_mat_mul::copy_A_to_cuda()
{
    cudaMemcpyAsync(this->A_cuda, this->A, this->M * this->K * sizeof(float), cudaMemcpyHostToDevice, this->stream);
}

void aspen_mat_mul::copy_B_to_cuda()
{
    cudaMemcpyAsync(this->B_cuda, this->B, this->K * this->N * sizeof(float), cudaMemcpyHostToDevice, this->stream);
}

void aspen_mat_mul::copy_C_from_cuda()
{
    cudaMemcpyAsync(this->C, this->C_cuda, this->M * this->N * sizeof(float), cudaMemcpyDeviceToHost, this->stream);
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
    // printf ("Launching kernel on handle %p, stream %p\n", this->handle, this->stream);
    cublasSgemm (this->handle, CUBLAS_OP_T, CUBLAS_OP_N, this->M, this->N, this->K, 
        &this->alpha, this->A_cuda, this->stride_A, this->B_cuda, this->stride_B,
             &this->beta, this->C_cuda, this->stride_C);
}

void aspen_mat_mul::run_cpu()
{
    if (this->is_calculation_done)
        return;
    cblas_sgemm (CblasColMajor, CblasTrans, CblasNoTrans, this->M, this->N, this->K, 
        this->alpha, this->A, this->stride_A, this->B, this->stride_B, 
             this->beta, this->C, this->stride_C);
}

void aspen_mat_mul::run_cuBLAS_strided_GEMV(int N_stride)
{
    if (this->is_calculation_done)
        return;
    cublasSetStream (this->handle, this->stream);
    // printf ("Launching kernel on handle %p, stream %p\n", this->handle, this->stream);
    cublasSgemvStridedBatched (this->handle, CUBLAS_OP_T, this->K, this->M, 
        &this->alpha, this->A_cuda, this->K, 0,
        this->B_cuda, 1, this->stride_B, &this->beta, 
        this->C_cuda, 1, this->stride_C, this->N); 
}

void aspen_mat_mul::run_cuBLAS_strided_GEMM(int N_stride)
{
    if (this->is_calculation_done)
        return;
    cublasSetStream (this->handle, this->stream);
    // printf ("Launching kernel on handle %p, stream %p\n", this->handle, this->stream);
    cublasSgemmStridedBatched (this->handle, CUBLAS_OP_T, CUBLAS_OP_N, this->M, this->N, this->K, 
        &this->alpha, this->A_cuda, this->stride_A, 0,
        this->B_cuda, this->stride_B, N_stride*this->stride_B, &this->beta, 
        this->C_cuda, this->stride_C, N_stride*this->stride_C, this->N/N_stride);
}

std::vector<aspen_mat_mul *> aspen_mat_mul::split_mat_mul (int split_M, int split_N)
{
    std::vector<aspen_mat_mul *> result;
    int M = this->M;
    int N = this->N;
    int K = this->K;
    int stride_A = this->stride_A;
    int stride_B = this->stride_B;
    int stride_C = this->stride_C;
    float alpha = this->alpha;
    float beta = this->beta;
    float *A = this->A;
    float *B = this->B;
    float *C = this->C;

    int M1 = M / split_M;
    int N1 = N / split_N;
    int K1 = K;

    for (int i = 0; i < split_M - 1; i++)
    {
        for (int j = 0; j < split_N - 1; j++)
        {
            aspen_mat_mul *mat_mul_split = new aspen_mat_mul (M1, N1, K1, alpha, 
                A + i * M1 * stride_A, stride_A, 
                B + j * N1 * stride_B, stride_B, beta, 
                C + i * M1 + j * N1 * stride_C, stride_C);
            mat_mul_split->set_cuda_memory (
                this->A_cuda + i * M1 * stride_A, 
                this->B_cuda + j * N1 * stride_B, 
                this->C_cuda + i * M1 + j * N1 * stride_C);
            result.push_back(mat_mul_split);
        }
        aspen_mat_mul *mat_mul_split = new aspen_mat_mul (M1, N - (split_N - 1) * N1, K1, alpha, 
            A + i * M1 * stride_A, stride_A, 
            B + (split_N - 1) * N1 * stride_B, stride_B, beta, 
            C + i * M1 + (split_N - 1) * N1 * stride_C, stride_C);
        mat_mul_split->set_cuda_memory (
            this->A_cuda + i * M1 * stride_A, 
            this->B_cuda + (split_N - 1) * N1 * stride_B, 
            this->C_cuda + i * M1 + (split_N - 1) * N1 * stride_C);
        result.push_back(mat_mul_split);
    }
    for (int j = 0; j < split_N - 1; j++)
    {
        aspen_mat_mul *mat_mul_split = new aspen_mat_mul (M - (split_M - 1) * M1, N1, K1, alpha, 
            A + (split_M - 1) * M1 * stride_A, stride_A, 
            B + j * N1 * stride_B, stride_B, beta, 
            C + (split_M - 1) * M1 + j * N1 * stride_C, stride_C);
        mat_mul_split->set_cuda_memory (
            this->A_cuda + (split_M - 1) * M1 * stride_A, 
            this->B_cuda + j * N1 * stride_B, 
            this->C_cuda + (split_M - 1) * M1 + j * N1 * stride_C);
        result.push_back(mat_mul_split);
    }
    aspen_mat_mul *mat_mul_split = new aspen_mat_mul (M - (split_M - 1) * M1, N - (split_N - 1) * N1, K1, alpha, 
        A + (split_M - 1) * M1 * stride_A, stride_A, 
        B + (split_N - 1) * N1 * stride_B, stride_B, beta, 
        C + (split_M - 1) * M1 + (split_N - 1) * N1 * stride_C, stride_C);
    mat_mul_split->set_cuda_memory (
        this->A_cuda + (split_M - 1) * M1 * stride_A, 
        this->B_cuda + (split_N - 1) * N1 * stride_B, 
        this->C_cuda + (split_M - 1) * M1 + (split_N - 1) * N1 * stride_C);
    result.push_back(mat_mul_split);
            

    return result;
}