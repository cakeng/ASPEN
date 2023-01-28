#include "cuda_aspen_tests.h"

static void *global_cuda_mem = NULL;
static void *global_host_mem = NULL;
static int aspen_mat_mul_id = 0;

aspen_mat_mul::aspen_mat_mul(int M, int N, int K, float alpha,
    float *A, int stride_A, float *B, int stride_B, float beta, float *C, int stride_C)
{
    this->id = aspen_mat_mul_id++;
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

    if (global_cuda_mem == NULL)
    {
        check_CUDA(cudaMalloc(&global_cuda_mem, 16 * 1024 * 1024));
    }
    if (global_host_mem == NULL)
    {
        check_CUDA(cudaMallocHost (&global_host_mem, 16 * 1024 * 1024));
    }
    this->temp_cuda = (char*)global_cuda_mem + this->id * 16 * 1024;
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
    check_CUDA(cudaMalloc((void **)&this->A_cuda, this->M * this->K * sizeof(float)));
    check_CUDA(cudaMalloc((void **)&this->B_cuda, this->K * this->N * sizeof(float)));
    check_CUDA(cudaMalloc((void **)&this->C_cuda, this->M * this->N * sizeof(float)));
    
}

void aspen_mat_mul::copy_A_to_cuda()
{
    check_CUDA(
        cudaMemcpyAsync(this->A_cuda, this->A, 
            this->M * this->K * sizeof(float), cudaMemcpyHostToDevice, this->stream));
}

void aspen_mat_mul::copy_B_to_cuda()
{
    check_CUDA(
        cudaMemcpyAsync(this->B_cuda, this->B, 
            this->K * this->N * sizeof(float), cudaMemcpyHostToDevice, this->stream));
}

void aspen_mat_mul::copy_C_from_cuda()
{
    check_CUDA(
        cudaMemcpyAsync(this->C, this->C_cuda, 
            this->M * this->N * sizeof(float), cudaMemcpyDeviceToHost, this->stream));
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
    check_CUDA(cudaStreamSynchronize(this->stream));
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

void aspen_mat_mul::set_mat_C (float val)
{
    for (int i = 0; i < this->M; i++)
        for (int j = 0; j < this->N; j++)
            this->C[j * this->stride_C + i] = val;
}

void aspen_mat_mul::run_cuBLAS()
{
    if (this->is_calculation_done)
        return;
    check_cuBLAS(cublasSetStream (this->handle, this->stream));
    check_cuBLAS(cublasSgemm (
        this->handle, CUBLAS_OP_T, CUBLAS_OP_N, this->M, this->N, this->K, 
        &this->alpha, this->A_cuda, this->stride_A, 
        this->B_cuda, this->stride_B, &this->beta, 
        this->C_cuda, this->stride_C));
}

void aspen_mat_mul::run_custom_CUDA_GEMM()
{
    if (this->is_calculation_done)
        return;
    custom_CUDA_mat_mul(
        this->M, this->N, this->K, 
        &this->alpha, this->A_cuda, this->stride_A, 
        this->B_cuda, this->stride_B, &this->beta, 
        this->C_cuda, this->stride_C, this->stream);
}

void aspen_mat_mul::run_cpu()
{
    if (this->is_calculation_done)
        return;
    cblas_sgemm (CblasColMajor, CblasTrans, CblasNoTrans, this->M, this->N, this->K, 
        this->alpha, this->A, this->stride_A, 
        this->B, this->stride_B, this->beta, 
        this->C, this->stride_C);
}

void aspen_mat_mul::run_cuBLAS_strided_GEMV(int N_stride)
{
    if (this->is_calculation_done)
        return;
        
    check_cuBLAS(cublasSetStream (this->handle, this->stream));
    check_cuBLAS(cublasSgemvStridedBatched (
        this->handle, CUBLAS_OP_T, this->K, this->M, 
        &this->alpha, this->A_cuda, this->K, 0,
        this->B_cuda, 1, this->stride_B, &this->beta, 
        this->C_cuda, 1, this->stride_C, this->N)); 
}

void aspen_mat_mul::run_cuBLAS_strided_GEMM(int N_stride)
{
    if (this->is_calculation_done)
        return;
    check_cuBLAS(cublasSetStream (this->handle, this->stream));
    check_cuBLAS(cublasSgemmStridedBatched (
        this->handle, CUBLAS_OP_T, CUBLAS_OP_N, this->M, this->N, this->K, 
        &this->alpha, this->A_cuda, this->stride_A, 0,
        this->B_cuda, this->stride_B, N_stride*this->stride_B, &this->beta, 
        this->C_cuda, this->stride_C, N_stride*this->stride_C, this->N/N_stride));
}

void aspen_mat_mul::run_cuBLAS_batched_GEMM (std::vector<aspen_mat_mul *> batch)
{
    if (this->is_calculation_done)
        return;
    int batch_size = batch.size();
    float **arr = (float**) global_host_mem + 16*1024*this->id/sizeof(float**);
    float **A_array = arr;
    float **B_array = arr + batch_size;
    float **C_array = arr + 2*batch_size;
    for (int i = 0; i < batch_size; i++)
    {
        A_array[i] = batch[i]->A_cuda;
        B_array[i] = batch[i]->B_cuda;
        C_array[i] = batch[i]->C_cuda;
    }
    check_cuBLAS(cublasSetStream (this->handle, this->stream));
    check_CUDA(
        cudaMemcpyAsync(this->temp_cuda, arr, 
            batch_size * 3 * sizeof(void*), cudaMemcpyHostToDevice, this->stream));
    check_cuBLAS(cublasSgemmBatched (
        this->handle, CUBLAS_OP_T, CUBLAS_OP_N, 
        batch[0]->M, batch[0]->N, batch[0]->K, 
        &this->alpha, (const float **) this->temp_cuda, batch[0]->stride_A, 
        (const float **) this->temp_cuda + batch_size, batch[0]->stride_B, &this->beta, 
        (float **) this->temp_cuda + batch_size*2, batch[0]->stride_C, batch_size));
}

std::vector<aspen_mat_mul *> aspen_mat_mul::split_mat_mul_by_num (int M_num, int N_num)
{
    int M_size = this->M / M_num + (this->M % M_num == 0 ? 0 : 1);
    int N_size = this->N / N_num + (this->N % N_num == 0 ? 0 : 1);
    return this->split_mat_mul_by_size(M_size, N_size);
}

std::vector<aspen_mat_mul *> aspen_mat_mul::split_mat_mul_by_size (int M_size, int N_size)
{
    assert (M_size > 0);
    assert (N_size > 0);
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
    int M_num = M / M_size + (M % M_size == 0 ? 0 : 1);
    int N_num = N / N_size + (N % N_size == 0 ? 0 : 1);

    for (int i = 0; i < M_num - 1; i++)
    {
        for (int j = 0; j < N_num - 1; j++)
        {
            aspen_mat_mul *mat_mul_split = new aspen_mat_mul (
                M_size, N_size, K, alpha, 
                A + i * M_size * stride_A, stride_A, 
                B + j * N_size * stride_B, stride_B, beta, 
                C + i * M_size + j * N_size * stride_C, stride_C);
            mat_mul_split->set_cuda_memory (
                this->A_cuda + i * M_size * stride_A, 
                this->B_cuda + j * N_size * stride_B, 
                this->C_cuda + i * M_size + j * N_size * stride_C);
            result.push_back(mat_mul_split);
        }
        aspen_mat_mul *mat_mul_split = new aspen_mat_mul (
            M_size, N - (N_num - 1) * N_size, K, alpha, 
            A + i * M_size * stride_A, stride_A, 
            B + (N_num - 1) * N_size * stride_B, stride_B, beta, 
            C + i * M_size + (N_num - 1) * N_size * stride_C, stride_C);
        mat_mul_split->set_cuda_memory (
            this->A_cuda + i * M_size * stride_A, 
            this->B_cuda + (N_num - 1) * N_size * stride_B, 
            this->C_cuda + i * M_size + (N_num - 1) * N_size * stride_C);
        result.push_back(mat_mul_split);
    }
    for (int j = 0; j < N_num - 1; j++)
    {
        aspen_mat_mul *mat_mul_split = new aspen_mat_mul (
            M - (M_num - 1) * M_size, N_size, K, alpha, 
            A + (M_num - 1) * M_size * stride_A, stride_A, 
            B + j * N_size * stride_B, stride_B, beta, 
            C + (M_num - 1) * M_size + j * N_size * stride_C, stride_C);
        mat_mul_split->set_cuda_memory (
            this->A_cuda + (M_num - 1) * M_size * stride_A, 
            this->B_cuda + j * N_size * stride_B, 
            this->C_cuda + (M_num - 1) * M_size + j * N_size * stride_C);
        result.push_back(mat_mul_split);
    }
    aspen_mat_mul *mat_mul_split = new aspen_mat_mul (
        M - (M_num - 1) * M_size, N - (N_num - 1) * N_size, K, alpha, 
        A + (M_num - 1) * M_size * stride_A, stride_A, 
        B + (N_num - 1) * N_size * stride_B, stride_B, beta, 
        C + (M_num - 1) * M_size + (N_num - 1) * N_size * stride_C, stride_C);
    mat_mul_split->set_cuda_memory (
        this->A_cuda + (M_num - 1) * M_size * stride_A, 
        this->B_cuda + (N_num - 1) * N_size * stride_B, 
        this->C_cuda + (M_num - 1) * M_size + (N_num - 1) * N_size * stride_C);
    result.push_back(mat_mul_split);
            

    return result;
}