#include "kernels.h"
#ifdef AVX2

void matmul_f32_avx2_8x1 (float *A, float *B, float **C, int k)
{
    __m256 o0;
    o0 = _mm256_load_ps(C[0]);
    for (int kidx = 0; kidx < k; kidx++)
    {
        const __m256 A_col = _mm256_load_ps(A);
        A += 8;
        const __m256 B_val0 = _mm256_broadcast_ss(B + k*0 + kidx);
        o0 = _mm256_fmadd_ps(A_col, B_val0, o0);
   }
    _mm256_store_ps(C[0], o0);
}
void matmul_f32_avx2_8x2 (float *A, float *B, float **C, int k)
{
    __m256 o0, o1;
    o0 = _mm256_load_ps(C[0]);
    o1 = _mm256_load_ps(C[1]);
    for (int kidx = 0; kidx < k; kidx++)
    {
        const __m256 A_col = _mm256_load_ps(A);
        A += 8;
        const __m256 B_val0 = _mm256_broadcast_ss(B + k*0 + kidx);
        const __m256 B_val1 = _mm256_broadcast_ss(B + k*1 + kidx);
        o0 = _mm256_fmadd_ps(A_col, B_val0, o0);
        o1 = _mm256_fmadd_ps(A_col, B_val1, o1);
   }
    _mm256_store_ps(C[0], o0);
    _mm256_store_ps(C[1], o1);
}
void matmul_f32_avx2_8x3 (float *A, float *B, float **C, int k)
{
    __m256 o0, o1, o2;
    o0 = _mm256_load_ps(C[0]);
    o1 = _mm256_load_ps(C[1]);
    o2 = _mm256_load_ps(C[2]);
    for (int kidx = 0; kidx < k; kidx++)
    {
        const __m256 A_col = _mm256_load_ps(A);
        A += 8;
        const __m256 B_val0 = _mm256_broadcast_ss(B + k*0 + kidx);
        const __m256 B_val1 = _mm256_broadcast_ss(B + k*1 + kidx);
        const __m256 B_val2 = _mm256_broadcast_ss(B + k*2 + kidx);
        o0 = _mm256_fmadd_ps(A_col, B_val0, o0);
        o1 = _mm256_fmadd_ps(A_col, B_val1, o1);
        o2 = _mm256_fmadd_ps(A_col, B_val2, o2);
   }
    _mm256_store_ps(C[0], o0);
    _mm256_store_ps(C[1], o1);
    _mm256_store_ps(C[2], o2);
}
void matmul_f32_avx2_8x4 (float *A, float *B, float **C, int k)
{
    __m256 o0, o1, o2, o3;
    o0 = _mm256_load_ps(C[0]);
    o1 = _mm256_load_ps(C[1]);
    o2 = _mm256_load_ps(C[2]);
    o3 = _mm256_load_ps(C[3]);
    for (int kidx = 0; kidx < k; kidx++)
    {
        const __m256 A_col = _mm256_load_ps(A);
        A += 8;
        const __m256 B_val0 = _mm256_broadcast_ss(B + k*0 + kidx);
        const __m256 B_val1 = _mm256_broadcast_ss(B + k*1 + kidx);
        const __m256 B_val2 = _mm256_broadcast_ss(B + k*2 + kidx);
        const __m256 B_val3 = _mm256_broadcast_ss(B + k*3 + kidx);
        o0 = _mm256_fmadd_ps(A_col, B_val0, o0);
        o1 = _mm256_fmadd_ps(A_col, B_val1, o1);
        o2 = _mm256_fmadd_ps(A_col, B_val2, o2);
        o3 = _mm256_fmadd_ps(A_col, B_val3, o3);
   }
    _mm256_store_ps(C[0], o0);
    _mm256_store_ps(C[1], o1);
    _mm256_store_ps(C[2], o2);
    _mm256_store_ps(C[3], o3);
}
void matmul_f32_avx2_8x5 (float *A, float *B, float **C, int k)
{
    __m256 o0, o1, o2, o3, o4;
    o0 = _mm256_load_ps(C[0]);
    o1 = _mm256_load_ps(C[1]);
    o2 = _mm256_load_ps(C[2]);
    o3 = _mm256_load_ps(C[3]);
    o4 = _mm256_load_ps(C[4]);
    for (int kidx = 0; kidx < k; kidx++)
    {
        const __m256 A_col = _mm256_load_ps(A);
        A += 8;
        const __m256 B_val0 = _mm256_broadcast_ss(B + k*0 + kidx);
        const __m256 B_val1 = _mm256_broadcast_ss(B + k*1 + kidx);
        const __m256 B_val2 = _mm256_broadcast_ss(B + k*2 + kidx);
        const __m256 B_val3 = _mm256_broadcast_ss(B + k*3 + kidx);
        o0 = _mm256_fmadd_ps(A_col, B_val0, o0);
        o1 = _mm256_fmadd_ps(A_col, B_val1, o1);
        o2 = _mm256_fmadd_ps(A_col, B_val2, o2);
        o3 = _mm256_fmadd_ps(A_col, B_val3, o3);
        const __m256 B_val4 = _mm256_broadcast_ss(B + k*4 + kidx);
        o4 = _mm256_fmadd_ps(A_col, B_val4, o4);
   }
    _mm256_store_ps(C[0], o0);
    _mm256_store_ps(C[1], o1);
    _mm256_store_ps(C[2], o2);
    _mm256_store_ps(C[3], o3);
    _mm256_store_ps(C[4], o4);
}
void matmul_f32_avx2_8x6 (float *A, float *B, float **C, int k)
{
    __m256 o0, o1, o2, o3, o4, o5;
    o0 = _mm256_load_ps(C[0]);
    o1 = _mm256_load_ps(C[1]);
    o2 = _mm256_load_ps(C[2]);
    o3 = _mm256_load_ps(C[3]);
    o4 = _mm256_load_ps(C[4]);
    o5 = _mm256_load_ps(C[5]);
    for (int kidx = 0; kidx < k; kidx++)
    {
        const __m256 A_col = _mm256_load_ps(A);
        A += 8;
        const __m256 B_val0 = _mm256_broadcast_ss(B + k*0 + kidx);
        const __m256 B_val1 = _mm256_broadcast_ss(B + k*1 + kidx);
        const __m256 B_val2 = _mm256_broadcast_ss(B + k*2 + kidx);
        const __m256 B_val3 = _mm256_broadcast_ss(B + k*3 + kidx);
        o0 = _mm256_fmadd_ps(A_col, B_val0, o0);
        o1 = _mm256_fmadd_ps(A_col, B_val1, o1);
        o2 = _mm256_fmadd_ps(A_col, B_val2, o2);
        o3 = _mm256_fmadd_ps(A_col, B_val3, o3);
        const __m256 B_val4 = _mm256_broadcast_ss(B + k*4 + kidx);
        const __m256 B_val5 = _mm256_broadcast_ss(B + k*5 + kidx);
        o4 = _mm256_fmadd_ps(A_col, B_val4, o4);
        o5 = _mm256_fmadd_ps(A_col, B_val5, o5);
   }
    _mm256_store_ps(C[0], o0);
    _mm256_store_ps(C[1], o1);
    _mm256_store_ps(C[2], o2);
    _mm256_store_ps(C[3], o3);
    _mm256_store_ps(C[4], o4);
    _mm256_store_ps(C[5], o5);
}
void matmul_f32_avx2_8x7 (float *A, float *B, float **C, int k)
{
    __m256 o0, o1, o2, o3, o4, o5, o6;
    o0 = _mm256_load_ps(C[0]);
    o1 = _mm256_load_ps(C[1]);
    o2 = _mm256_load_ps(C[2]);
    o3 = _mm256_load_ps(C[3]);
    o4 = _mm256_load_ps(C[4]);
    o5 = _mm256_load_ps(C[5]);
    o6 = _mm256_load_ps(C[6]);
    for (int kidx = 0; kidx < k; kidx++)
    {
        const __m256 A_col = _mm256_load_ps(A);
        A += 8;
        const __m256 B_val0 = _mm256_broadcast_ss(B + k*0 + kidx);
        const __m256 B_val1 = _mm256_broadcast_ss(B + k*1 + kidx);
        const __m256 B_val2 = _mm256_broadcast_ss(B + k*2 + kidx);
        const __m256 B_val3 = _mm256_broadcast_ss(B + k*3 + kidx);
        o0 = _mm256_fmadd_ps(A_col, B_val0, o0);
        o1 = _mm256_fmadd_ps(A_col, B_val1, o1);
        o2 = _mm256_fmadd_ps(A_col, B_val2, o2);
        o3 = _mm256_fmadd_ps(A_col, B_val3, o3);
        const __m256 B_val4 = _mm256_broadcast_ss(B + k*4 + kidx);
        const __m256 B_val5 = _mm256_broadcast_ss(B + k*5 + kidx);
        const __m256 B_val6 = _mm256_broadcast_ss(B + k*6 + kidx);
        o4 = _mm256_fmadd_ps(A_col, B_val4, o4);
        o5 = _mm256_fmadd_ps(A_col, B_val5, o5);
        o6 = _mm256_fmadd_ps(A_col, B_val6, o6);
   }
    _mm256_store_ps(C[0], o0);
    _mm256_store_ps(C[1], o1);
    _mm256_store_ps(C[2], o2);
    _mm256_store_ps(C[3], o3);
    _mm256_store_ps(C[4], o4);
    _mm256_store_ps(C[5], o5);
    _mm256_store_ps(C[6], o6);
}
void matmul_f32_avx2_8x8 (float *A, float *B, float **C, int k)
{
    __m256 o0, o1, o2, o3, o4, o5, o6, o7;
    o0 = _mm256_load_ps(C[0]);
    o1 = _mm256_load_ps(C[1]);
    o2 = _mm256_load_ps(C[2]);
    o3 = _mm256_load_ps(C[3]);
    o4 = _mm256_load_ps(C[4]);
    o5 = _mm256_load_ps(C[5]);
    o6 = _mm256_load_ps(C[6]);
    o7 = _mm256_load_ps(C[7]);
    for (int kidx = 0; kidx < k; kidx++)
    {
        const __m256 A_col = _mm256_load_ps(A);
        A += 8;
        const __m256 B_val0 = _mm256_broadcast_ss(B + k*0 + kidx);
        const __m256 B_val1 = _mm256_broadcast_ss(B + k*1 + kidx);
        const __m256 B_val2 = _mm256_broadcast_ss(B + k*2 + kidx);
        const __m256 B_val3 = _mm256_broadcast_ss(B + k*3 + kidx);
        o0 = _mm256_fmadd_ps(A_col, B_val0, o0);
        o1 = _mm256_fmadd_ps(A_col, B_val1, o1);
        o2 = _mm256_fmadd_ps(A_col, B_val2, o2);
        o3 = _mm256_fmadd_ps(A_col, B_val3, o3);
        const __m256 B_val4 = _mm256_broadcast_ss(B + k*4 + kidx);
        const __m256 B_val5 = _mm256_broadcast_ss(B + k*5 + kidx);
        const __m256 B_val6 = _mm256_broadcast_ss(B + k*6 + kidx);
        const __m256 B_val7 = _mm256_broadcast_ss(B + k*7 + kidx);
        o4 = _mm256_fmadd_ps(A_col, B_val4, o4);
        o5 = _mm256_fmadd_ps(A_col, B_val5, o5);
        o6 = _mm256_fmadd_ps(A_col, B_val6, o6);
        o7 = _mm256_fmadd_ps(A_col, B_val7, o7);
   }
    _mm256_store_ps(C[0], o0);
    _mm256_store_ps(C[1], o1);
    _mm256_store_ps(C[2], o2);
    _mm256_store_ps(C[3], o3);
    _mm256_store_ps(C[4], o4);
    _mm256_store_ps(C[5], o5);
    _mm256_store_ps(C[6], o6);
    _mm256_store_ps(C[7], o7);
}
void matmul_f32_avx2_8x9 (float *A, float *B, float **C, int k)
{
    __m256 o0, o1, o2, o3, o4, o5, o6, o7, o8;
    o0 = _mm256_load_ps(C[0]);
    o1 = _mm256_load_ps(C[1]);
    o2 = _mm256_load_ps(C[2]);
    o3 = _mm256_load_ps(C[3]);
    o4 = _mm256_load_ps(C[4]);
    o5 = _mm256_load_ps(C[5]);
    o6 = _mm256_load_ps(C[6]);
    o7 = _mm256_load_ps(C[7]);
    o8 = _mm256_load_ps(C[8]);
    for (int kidx = 0; kidx < k; kidx++)
    {
        const __m256 A_col = _mm256_load_ps(A);
        A += 8;
        const __m256 B_val0 = _mm256_broadcast_ss(B + k*0 + kidx);
        const __m256 B_val1 = _mm256_broadcast_ss(B + k*1 + kidx);
        const __m256 B_val2 = _mm256_broadcast_ss(B + k*2 + kidx);
        const __m256 B_val3 = _mm256_broadcast_ss(B + k*3 + kidx);
        o0 = _mm256_fmadd_ps(A_col, B_val0, o0);
        o1 = _mm256_fmadd_ps(A_col, B_val1, o1);
        o2 = _mm256_fmadd_ps(A_col, B_val2, o2);
        o3 = _mm256_fmadd_ps(A_col, B_val3, o3);
        const __m256 B_val4 = _mm256_broadcast_ss(B + k*4 + kidx);
        const __m256 B_val5 = _mm256_broadcast_ss(B + k*5 + kidx);
        const __m256 B_val6 = _mm256_broadcast_ss(B + k*6 + kidx);
        const __m256 B_val7 = _mm256_broadcast_ss(B + k*7 + kidx);
        o4 = _mm256_fmadd_ps(A_col, B_val4, o4);
        o5 = _mm256_fmadd_ps(A_col, B_val5, o5);
        o6 = _mm256_fmadd_ps(A_col, B_val6, o6);
        o7 = _mm256_fmadd_ps(A_col, B_val7, o7);
        const __m256 B_val8 = _mm256_broadcast_ss(B + k*8 + kidx);
        o8 = _mm256_fmadd_ps(A_col, B_val8, o8);
   }
    _mm256_store_ps(C[0], o0);
    _mm256_store_ps(C[1], o1);
    _mm256_store_ps(C[2], o2);
    _mm256_store_ps(C[3], o3);
    _mm256_store_ps(C[4], o4);
    _mm256_store_ps(C[5], o5);
    _mm256_store_ps(C[6], o6);
    _mm256_store_ps(C[7], o7);
    _mm256_store_ps(C[8], o8);
}
void matmul_f32_avx2_8x10 (float *A, float *B, float **C, int k)
{
    __m256 o0, o1, o2, o3, o4, o5, o6, o7, o8, o9;
    o0 = _mm256_load_ps(C[0]);
    o1 = _mm256_load_ps(C[1]);
    o2 = _mm256_load_ps(C[2]);
    o3 = _mm256_load_ps(C[3]);
    o4 = _mm256_load_ps(C[4]);
    o5 = _mm256_load_ps(C[5]);
    o6 = _mm256_load_ps(C[6]);
    o7 = _mm256_load_ps(C[7]);
    o8 = _mm256_load_ps(C[8]);
    o9 = _mm256_load_ps(C[9]);
    for (int kidx = 0; kidx < k; kidx++)
    {
        const __m256 A_col = _mm256_load_ps(A);
        A += 8;
        const __m256 B_val0 = _mm256_broadcast_ss(B + k*0 + kidx);
        const __m256 B_val1 = _mm256_broadcast_ss(B + k*1 + kidx);
        const __m256 B_val2 = _mm256_broadcast_ss(B + k*2 + kidx);
        const __m256 B_val3 = _mm256_broadcast_ss(B + k*3 + kidx);
        o0 = _mm256_fmadd_ps(A_col, B_val0, o0);
        o1 = _mm256_fmadd_ps(A_col, B_val1, o1);
        o2 = _mm256_fmadd_ps(A_col, B_val2, o2);
        o3 = _mm256_fmadd_ps(A_col, B_val3, o3);
        const __m256 B_val4 = _mm256_broadcast_ss(B + k*4 + kidx);
        const __m256 B_val5 = _mm256_broadcast_ss(B + k*5 + kidx);
        const __m256 B_val6 = _mm256_broadcast_ss(B + k*6 + kidx);
        const __m256 B_val7 = _mm256_broadcast_ss(B + k*7 + kidx);
        o4 = _mm256_fmadd_ps(A_col, B_val4, o4);
        o5 = _mm256_fmadd_ps(A_col, B_val5, o5);
        o6 = _mm256_fmadd_ps(A_col, B_val6, o6);
        o7 = _mm256_fmadd_ps(A_col, B_val7, o7);
        const __m256 B_val8 = _mm256_broadcast_ss(B + k*8 + kidx);
        const __m256 B_val9 = _mm256_broadcast_ss(B + k*9 + kidx);
        o8 = _mm256_fmadd_ps(A_col, B_val8, o8);
        o9 = _mm256_fmadd_ps(A_col, B_val9, o9);
   }
    _mm256_store_ps(C[0], o0);
    _mm256_store_ps(C[1], o1);
    _mm256_store_ps(C[2], o2);
    _mm256_store_ps(C[3], o3);
    _mm256_store_ps(C[4], o4);
    _mm256_store_ps(C[5], o5);
    _mm256_store_ps(C[6], o6);
    _mm256_store_ps(C[7], o7);
    _mm256_store_ps(C[8], o8);
    _mm256_store_ps(C[9], o9);
}
void matmul_f32_avx2_8x11 (float *A, float *B, float **C, int k)
{
    __m256 o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10;
    o0 = _mm256_load_ps(C[0]);
    o1 = _mm256_load_ps(C[1]);
    o2 = _mm256_load_ps(C[2]);
    o3 = _mm256_load_ps(C[3]);
    o4 = _mm256_load_ps(C[4]);
    o5 = _mm256_load_ps(C[5]);
    o6 = _mm256_load_ps(C[6]);
    o7 = _mm256_load_ps(C[7]);
    o8 = _mm256_load_ps(C[8]);
    o9 = _mm256_load_ps(C[9]);
    o10 = _mm256_load_ps(C[10]);
    for (int kidx = 0; kidx < k; kidx++)
    {
        const __m256 A_col = _mm256_load_ps(A);
        A += 8;
        const __m256 B_val0 = _mm256_broadcast_ss(B + k*0 + kidx);
        const __m256 B_val1 = _mm256_broadcast_ss(B + k*1 + kidx);
        const __m256 B_val2 = _mm256_broadcast_ss(B + k*2 + kidx);
        const __m256 B_val3 = _mm256_broadcast_ss(B + k*3 + kidx);
        o0 = _mm256_fmadd_ps(A_col, B_val0, o0);
        o1 = _mm256_fmadd_ps(A_col, B_val1, o1);
        o2 = _mm256_fmadd_ps(A_col, B_val2, o2);
        o3 = _mm256_fmadd_ps(A_col, B_val3, o3);
        const __m256 B_val4 = _mm256_broadcast_ss(B + k*4 + kidx);
        const __m256 B_val5 = _mm256_broadcast_ss(B + k*5 + kidx);
        const __m256 B_val6 = _mm256_broadcast_ss(B + k*6 + kidx);
        const __m256 B_val7 = _mm256_broadcast_ss(B + k*7 + kidx);
        o4 = _mm256_fmadd_ps(A_col, B_val4, o4);
        o5 = _mm256_fmadd_ps(A_col, B_val5, o5);
        o6 = _mm256_fmadd_ps(A_col, B_val6, o6);
        o7 = _mm256_fmadd_ps(A_col, B_val7, o7);
        const __m256 B_val8 = _mm256_broadcast_ss(B + k*8 + kidx);
        const __m256 B_val9 = _mm256_broadcast_ss(B + k*9 + kidx);
        const __m256 B_val10 = _mm256_broadcast_ss(B + k*10 + kidx);
        o8 = _mm256_fmadd_ps(A_col, B_val8, o8);
        o9 = _mm256_fmadd_ps(A_col, B_val9, o9);
        o10 = _mm256_fmadd_ps(A_col, B_val10, o10);
   }
    _mm256_store_ps(C[0], o0);
    _mm256_store_ps(C[1], o1);
    _mm256_store_ps(C[2], o2);
    _mm256_store_ps(C[3], o3);
    _mm256_store_ps(C[4], o4);
    _mm256_store_ps(C[5], o5);
    _mm256_store_ps(C[6], o6);
    _mm256_store_ps(C[7], o7);
    _mm256_store_ps(C[8], o8);
    _mm256_store_ps(C[9], o9);
    _mm256_store_ps(C[10], o10);
}
void matmul_f32_avx2_8x12 (float *A, float *B, float **C, int k)
{
    __m256 o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11;
    o0 = _mm256_load_ps(C[0]);
    o1 = _mm256_load_ps(C[1]);
    o2 = _mm256_load_ps(C[2]);
    o3 = _mm256_load_ps(C[3]);
    o4 = _mm256_load_ps(C[4]);
    o5 = _mm256_load_ps(C[5]);
    o6 = _mm256_load_ps(C[6]);
    o7 = _mm256_load_ps(C[7]);
    o8 = _mm256_load_ps(C[8]);
    o9 = _mm256_load_ps(C[9]);
    o10 = _mm256_load_ps(C[10]);
    o11 = _mm256_load_ps(C[11]);
    for (int kidx = 0; kidx < k; kidx++)
    {
        const __m256 A_col = _mm256_load_ps(A);
        A += 8;
        const __m256 B_val0 = _mm256_broadcast_ss(B + k*0 + kidx);
        const __m256 B_val1 = _mm256_broadcast_ss(B + k*1 + kidx);
        const __m256 B_val2 = _mm256_broadcast_ss(B + k*2 + kidx);
        o0 = _mm256_fmadd_ps(A_col, B_val0, o0);
        o1 = _mm256_fmadd_ps(A_col, B_val1, o1);
        o2 = _mm256_fmadd_ps(A_col, B_val2, o2);
        const __m256 B_val3 = _mm256_broadcast_ss(B + k*3 + kidx);
        const __m256 B_val4 = _mm256_broadcast_ss(B + k*4 + kidx);
        const __m256 B_val5 = _mm256_broadcast_ss(B + k*5 + kidx);
        o3 = _mm256_fmadd_ps(A_col, B_val3, o3);
        o4 = _mm256_fmadd_ps(A_col, B_val4, o4);
        o5 = _mm256_fmadd_ps(A_col, B_val5, o5);
        const __m256 B_val6 = _mm256_broadcast_ss(B + k*6 + kidx);
        const __m256 B_val7 = _mm256_broadcast_ss(B + k*7 + kidx);
        const __m256 B_val8 = _mm256_broadcast_ss(B + k*8 + kidx);
        o6 = _mm256_fmadd_ps(A_col, B_val6, o6);
        o7 = _mm256_fmadd_ps(A_col, B_val7, o7);
        o8 = _mm256_fmadd_ps(A_col, B_val8, o8);
        const __m256 B_val9 = _mm256_broadcast_ss(B + k*9 + kidx);
        const __m256 B_val10 = _mm256_broadcast_ss(B + k*10 + kidx);
        const __m256 B_val11 = _mm256_broadcast_ss(B + k*11 + kidx);
        o9 = _mm256_fmadd_ps(A_col, B_val9, o9);
        o10 = _mm256_fmadd_ps(A_col, B_val10, o10);
        o11 = _mm256_fmadd_ps(A_col, B_val11, o11);
   }
    _mm256_store_ps(C[0], o0);
    _mm256_store_ps(C[1], o1);
    _mm256_store_ps(C[2], o2);
    _mm256_store_ps(C[3], o3);
    _mm256_store_ps(C[4], o4);
    _mm256_store_ps(C[5], o5);
    _mm256_store_ps(C[6], o6);
    _mm256_store_ps(C[7], o7);
    _mm256_store_ps(C[8], o8);
    _mm256_store_ps(C[9], o9);
    _mm256_store_ps(C[10], o10);
    _mm256_store_ps(C[11], o11);
}

void matmul_f32_avx2_16x1(float *A, float *B, float **C, int k)
{
    __m256 o0, o1;
    o0 = _mm256_load_ps(C[0] + 8*0);
    o1 = _mm256_load_ps(C[0] + 8*1);
    for (int kidx = 0; kidx < k; kidx++)
    {
        const __m256 A_col0 = _mm256_load_ps(A + k*8*0);
        const __m256 A_col1 = _mm256_load_ps(A + k*8*1);
        A += 8;
        const __m256 B_val0 = _mm256_broadcast_ss(B + k*0 + kidx);
        o0 = _mm256_fmadd_ps(A_col0, B_val0, o0);
        o1 = _mm256_fmadd_ps(A_col1, B_val0, o1);
    }
    _mm256_store_ps(C[0] + 8*0, o0);
    _mm256_store_ps(C[0] + 8*1, o1);
}

void matmul_f32_avx2_32x1(float *A, float *B, float **C, int k)
{
    __m256 o0, o1, o2, o3;
    o0 = _mm256_load_ps(C[0] + 8*0);
    o1 = _mm256_load_ps(C[0] + 8*1);
    o2 = _mm256_load_ps(C[0] + 8*2);
    o3 = _mm256_load_ps(C[0] + 8*3);
    for (int kidx = 0; kidx < k; kidx++)
    {
        const __m256 A_col0 = _mm256_load_ps(A + k*8*0);
        const __m256 A_col1 = _mm256_load_ps(A + k*8*1);
        const __m256 A_col2 = _mm256_load_ps(A + k*8*2);
        const __m256 A_col3 = _mm256_load_ps(A + k*8*3);
        A += 8;
        const __m256 B_val0 = _mm256_broadcast_ss(B + k*0 + kidx);
        o0 = _mm256_fmadd_ps(A_col0, B_val0, o0);
        o1 = _mm256_fmadd_ps(A_col1, B_val0, o1);
        o2 = _mm256_fmadd_ps(A_col2, B_val0, o2);
        o3 = _mm256_fmadd_ps(A_col3, B_val0, o3);
    }
    _mm256_store_ps(C[0] + 8*0, o0);
    _mm256_store_ps(C[0] + 8*1, o1);
    _mm256_store_ps(C[0] + 8*2, o2);
    _mm256_store_ps(C[0] + 8*3, o3);
}

void matmul_f32_avx2_64x1(float *A, float *B, float **C, int k)
{
    __m256 o0, o1, o2, o3, o4, o5, o6, o7;
    o0 = _mm256_load_ps(C[0] + 8*0);
    o1 = _mm256_load_ps(C[0] + 8*1);
    o2 = _mm256_load_ps(C[0] + 8*2);
    o3 = _mm256_load_ps(C[0] + 8*3);
    o4 = _mm256_load_ps(C[0] + 8*4);
    o5 = _mm256_load_ps(C[0] + 8*5);
    o6 = _mm256_load_ps(C[0] + 8*6);
    o7 = _mm256_load_ps(C[0] + 8*7);
    for (int kidx = 0; kidx < k; kidx++)
    {
        const __m256 A_col0 = _mm256_load_ps(A + k*8*0);
        const __m256 A_col1 = _mm256_load_ps(A + k*8*1);
        const __m256 A_col2 = _mm256_load_ps(A + k*8*2);
        const __m256 A_col3 = _mm256_load_ps(A + k*8*3);
        const __m256 A_col4 = _mm256_load_ps(A + k*8*4);
        const __m256 A_col5 = _mm256_load_ps(A + k*8*5);
        const __m256 A_col6 = _mm256_load_ps(A + k*8*6);
        const __m256 A_col7 = _mm256_load_ps(A + k*8*7);
        A += 8;
        const __m256 B_val0 = _mm256_broadcast_ss(B + k*0 + kidx);
        o0 = _mm256_fmadd_ps(A_col0, B_val0, o0);
        o1 = _mm256_fmadd_ps(A_col1, B_val0, o1);
        o2 = _mm256_fmadd_ps(A_col2, B_val0, o2);
        o3 = _mm256_fmadd_ps(A_col3, B_val0, o3);
        o4 = _mm256_fmadd_ps(A_col4, B_val0, o4);
        o5 = _mm256_fmadd_ps(A_col5, B_val0, o5);
        o6 = _mm256_fmadd_ps(A_col6, B_val0, o6);
        o7 = _mm256_fmadd_ps(A_col7, B_val0, o7);
    }
    _mm256_store_ps(C[0] + 8*0, o0);
    _mm256_store_ps(C[0] + 8*1, o1);
    _mm256_store_ps(C[0] + 8*2, o2);
    _mm256_store_ps(C[0] + 8*3, o3);
    _mm256_store_ps(C[0] + 8*4, o4);
    _mm256_store_ps(C[0] + 8*5, o5);
    _mm256_store_ps(C[0] + 8*6, o6);
    _mm256_store_ps(C[0] + 8*7, o7);
}

#endif