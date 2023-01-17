#include "util.h"
#include "mat_mul.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>

static double start_time[8];

float* mat_to_filter(float *input, int h, int w)
{
    float *output;
    cudaMallocHost (&output, (h + (_OUTC_CHUNK - (h%_OUTC_CHUNK)))*w*sizeof(float));
    for (int hi = 0; hi < h; hi++)
    {
        for (int wi = 0; wi < w; wi++)
        {
            const float val = *(input + hi*w + wi);
            *(output + ((hi/_OUTC_CHUNK)*w + wi)*_OUTC_CHUNK + (hi%_OUTC_CHUNK)) = val;
        }
    }
    return output;
}

float* mat_row_to_col(float *input, int h, int w)
{
    float *output;
    cudaMallocHost (&output, h*w*sizeof(float));
    for (int hi = 0; hi < h; hi++)
    {
        for (int wi = 0; wi < w; wi++)
        {
            const float val = *(input + hi*w + wi);
            *(output + wi*h + hi) = val;
        }
    }
    return output;
}

float* mat_col_to_row(float *input, int h, int w)
{
    float *output;
    cudaMallocHost (&output, h*w*sizeof(float));
    for (int hi = 0; hi < h; hi++)
    {
        for (int wi = 0; wi < w; wi++)
        {
            const float val = *(input + wi*h + hi);
            *(output + hi*w + wi) = val;
        }
    }
    return output;
}

static double get_time()
{
    struct timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void timer_start(int i)
{
    start_time[i] = get_time();
}

double timer_stop(int i)
{
    return get_time() - start_time[i];
}

void check_mat_mul(float *A, float *B, float *C, int M, int N, int K)
{
    printf("Validating...\n");

    float *C_ans;
    alloc_mat(&C_ans, M, N);
    zero_mat(C_ans, M, N);
#pragma omp parallel for
    for (int i = 0; i < M; ++i)
    {
        for (int k = 0; k < K; ++k)
        {
            for (int j = 0; j < N; ++j)
            {
                C_ans[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }

    bool is_valid = true;
    int cnt = 0, thr = 10;
    float eps = 1e-3;
#pragma omp parallel for
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float c = C[i * N + j];
            float c_ans = C_ans[i * N + j];
            if (fabsf(c - c_ans) > eps && (c_ans == 0 || fabsf((c - c_ans) / c_ans) > eps))
            {
                ++cnt;
                if (cnt <= thr)
                    printf("C[%d][%d] : correct_value = %f, your_value = %f\n", i, j, c_ans, c);
                if (cnt == thr + 1)
                    printf("Too many error, only first %d values are printed.\n", thr);
                is_valid = false;
            }
        }
    }

    if (is_valid)
    {
        printf("Result: VALID\n");
    }
    else
    {
        printf("Result: INVALID\n");
    }
}

void print_mat(float *m, int R, int C)
{
    for (int i = 0; i < R; ++i)
    {
        for (int j = 0; j < C; ++j)
        {
            printf("%+.3f ", m[i * C + j]);
        }
        printf("\n");
    }
}

void alloc_mat(float **m, int R, int C)
{
    cudaMallocHost (m, sizeof(float) * R * C);
    if (*m == NULL)
    {
        printf("Failed to allocate memory for matrix.\n");
        exit(0);
    }
    zero_mat (*m, R, C);
}

void rand_mat(float *m, int R, int C)
{
    for (int i = 0; i < R; i++)
    {
        for (int j = 0; j < C; j++)
        {
            m[i * C + j] = (float)rand() / RAND_MAX - 0.5;
        }
    }
}

void set_mat(float *m, int R, int C, float val)
{
    for (int i = 0; i < R; i++)
    {
        for (int j = 0; j < C; j++)
        {
            m[i * C + j] = val;
        }
    }
}

void zero_mat(float *m, int R, int C)
{
    memset(m, 0, sizeof(float) * R * C);
}
