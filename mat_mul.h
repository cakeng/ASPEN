#pragma once
#define _OUTC_CHUNK 8
void mat_mul(float *A, float *B, float *C, int M, int N, int K, int skip_data_movement);
void mat_mul_init(float *A, float *B, float *C, int M, int N, int K);
void mat_mul_write_to_gpu(float *A, float *B, float *C, int M, int N, int K); 
void mat_mul_read_from_gpu(float *A, float *B, float *C, int M, int N, int K);

void cublas_mat_mul(float *A, float *B, float *C, int M, int N, int K, int skip_data_movement);
void cublas_mat_mul_init(float *A, float *B, float *C, int M, int N, int K);
void cublas_mat_mul_write_to_gpu(float *A, float *B, float *C, int M, int N, int K); 
void cublas_mat_mul_read_from_gpu(float *A, float *B, float *C, int M, int N, int K);
void cublas_mat_mul_free(float *A, float *B, float *C, int M, int N, int K);
