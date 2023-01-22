#include "cuda_aspen_tests.h"

static int thread_id = 0;

aspen_pthread::aspen_pthread()
{
    this->id = thread_id++;
    this->r_type = RUN_CPU;
    this->mat_muls = std::vector<aspen_mat_mul *>();
    cublasCreate(&this->handle);
    cudaStreamCreate(&this->stream);

    pthread_mutex_init(&this->mutex, NULL);
    pthread_cond_init(&this->cond_host_to_child, NULL);
    pthread_cond_init(&this->cond_child_to_host, NULL);
    pthread_mutex_lock (&this->mutex);
    this->thread = pthread_create(&this->thread, NULL, aspen_pthread::thread_func, this);
    this->running = false;
}

aspen_pthread::~aspen_pthread()
{
}

void aspen_pthread::add_mat_mul(aspen_mat_mul *mat_mul)
{
    this->mat_muls.push_back(mat_mul);
    mat_mul->set_cuda_handle(this->handle);
    mat_mul->set_cuda_stream(this->stream);
}

void aspen_pthread::run_cuBLAS()
{
    this->r_type = RUN_CUBLAS;
    this->running = true;
    pthread_cond_signal(&this->cond_host_to_child);
    pthread_mutex_unlock(&this->mutex);
}

void aspen_pthread::run_cpu()
{
    this->r_type = RUN_CPU;
    this->running = true;
    pthread_cond_signal(&this->cond_host_to_child);
    pthread_mutex_unlock(&this->mutex);
}

void aspen_pthread::run_custom_CUDA_GEMM()
{
    this->r_type = RUN_CUSTOM_CUDA_GEMM;
    this->running = true;
    pthread_cond_signal(&this->cond_host_to_child);
    pthread_mutex_unlock(&this->mutex);
}

void aspen_pthread::stop()
{
    this->running = false;
    pthread_mutex_lock(&this->mutex);
}

void aspen_pthread::wait()
{
    while (this->running);
    pthread_mutex_lock(&this->mutex);
}

void *aspen_pthread::thread_func(void *arg)
{
    aspen_pthread *self = (aspen_pthread *)arg;
    pthread_mutex_lock(&self->mutex);
    while (true)
    {
        self->mat_muls.front()->copy_B_to_cuda();
        for (auto &mat_mul : self->mat_muls)
        {
            switch (self->r_type)
            {
                case RUN_CPU:
                    mat_mul->run_cpu();
                    break;
                case RUN_CUBLAS:
                    mat_mul->run_cuBLAS();
                    break;
                case RUN_CUSTOM_CUDA_GEMM:
                    mat_mul->run_custom_CUDA_GEMM();
                    break;
            }
        }
        self->mat_muls.back()->copy_C_from_cuda();
        self->mat_muls.back()->synchronize();
        self->running = false;
        pthread_cond_wait(&self->cond_host_to_child, &self->mutex);
    }
}
