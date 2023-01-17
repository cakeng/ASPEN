#pragma once

float* mat_to_filter(float *input, int h, int w);

float* mat_row_to_col(float *input, int h, int w);
float* mat_col_to_row(float *input, int h, int w);

void timer_start(int i);

double timer_stop(int i);

void check_mat_mul(float *A, float *B, float *C, int M, int N, int K);

void print_mat(float *m, int R, int C);

void alloc_mat(float **m, int R, int C);

void rand_mat(float *m, int R, int C);

void set_mat(float *m, int R, int C, float val);

void zero_mat(float *m, int R, int C);
