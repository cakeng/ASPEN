#include "kernels.h"
#ifdef _NEON

// NEON accelerated matmul kernels
void matmul_f32_NEON_8x1 (float *A, float *B, float **C, int k)
{
    float32x4_t o00;
    float32x4_t o01;
    o00 = vld1q_f32(C[0]);
    o01 = vld1q_f32(C[0] + 4);
    for (int kidx = 0; kidx < k; kidx++)
    {
        const float32x4_t A_col0 = vld1q_f32(A);
        const float32x4_t A_col1 = vld1q_f32(A + 4);
        A += 8;
        const float32x4_t B_val0 = vld1q_dup_f32(B + k*0 + kidx);
        o00 = vfmaq_f32(o00, A_col0, B_val0);
        o01 = vfmaq_f32(o01, A_col1, B_val0);
   }
    vst1q_f32(C[0], o00);
    vst1q_f32(C[0] + 4, o01);
}
void matmul_f32_NEON_8x2 (float *A, float *B, float **C, int k)
{
    float32x4_t o00, o10;
    float32x4_t o01, o11;
    o00 = vld1q_f32(C[0]);
    o10 = vld1q_f32(C[1]);
    o01 = vld1q_f32(C[0] + 4);
    o11 = vld1q_f32(C[1] + 4);
    for (int kidx = 0; kidx < k; kidx++)
    {
        const float32x4_t A_col0 = vld1q_f32(A);
        const float32x4_t A_col1 = vld1q_f32(A + 4);
        A += 8;
        const float32x4_t B_val0 = vld1q_dup_f32(B + k*0 + kidx);
        const float32x4_t B_val1 = vld1q_dup_f32(B + k*1 + kidx);
        o00 = vfmaq_f32(o00, A_col0, B_val0);
        o10 = vfmaq_f32(o10, A_col0, B_val1);
        o01 = vfmaq_f32(o01, A_col1, B_val0);
        o11 = vfmaq_f32(o11, A_col1, B_val1);
   }
    vst1q_f32(C[0], o00);
    vst1q_f32(C[1], o10);
    vst1q_f32(C[0] + 4, o01);
    vst1q_f32(C[1] + 4, o11);
}
void matmul_f32_NEON_8x3 (float *A, float *B, float **C, int k)
{
    float32x4_t o00, o10, o20;
    float32x4_t o01, o11, o21;
    o00 = vld1q_f32(C[0]);
    o10 = vld1q_f32(C[1]);
    o20 = vld1q_f32(C[2]);
    o01 = vld1q_f32(C[0] + 4);
    o11 = vld1q_f32(C[1] + 4);
    o21 = vld1q_f32(C[2] + 4);
    for (int kidx = 0; kidx < k; kidx++)
    {
        const float32x4_t A_col0 = vld1q_f32(A);
        const float32x4_t A_col1 = vld1q_f32(A + 4);
        A += 8;
        const float32x4_t B_val0 = vld1q_dup_f32(B + k*0 + kidx);
        const float32x4_t B_val1 = vld1q_dup_f32(B + k*1 + kidx);
        const float32x4_t B_val2 = vld1q_dup_f32(B + k*2 + kidx);
        o00 = vfmaq_f32(o00, A_col0, B_val0);
        o10 = vfmaq_f32(o10, A_col0, B_val1);
        o20 = vfmaq_f32(o20, A_col0, B_val2);
        o01 = vfmaq_f32(o01, A_col1, B_val0);
        o11 = vfmaq_f32(o11, A_col1, B_val1);
        o21 = vfmaq_f32(o21, A_col1, B_val2);
   }
    vst1q_f32(C[0], o00);
    vst1q_f32(C[1], o10);
    vst1q_f32(C[2], o20);
    vst1q_f32(C[0] + 4, o01);
    vst1q_f32(C[1] + 4, o11);
    vst1q_f32(C[2] + 4, o21);
}
void matmul_f32_NEON_8x4 (float *A, float *B, float **C, int k)
{
    float32x4_t o00, o10, o20, o30;
    float32x4_t o01, o11, o21, o31;
    o00 = vld1q_f32(C[0]);
    o10 = vld1q_f32(C[1]);
    o20 = vld1q_f32(C[2]);
    o30 = vld1q_f32(C[3]);
    o01 = vld1q_f32(C[0] + 4);
    o11 = vld1q_f32(C[1] + 4);
    o21 = vld1q_f32(C[2] + 4);
    o31 = vld1q_f32(C[3] + 4);
    for (int kidx = 0; kidx < k; kidx++)
    {
        const float32x4_t A_col0 = vld1q_f32(A);
        const float32x4_t A_col1 = vld1q_f32(A + 4);
        A += 8;
        const float32x4_t B_val0 = vld1q_dup_f32(B + k*0 + kidx);
        const float32x4_t B_val1 = vld1q_dup_f32(B + k*1 + kidx);
        const float32x4_t B_val2 = vld1q_dup_f32(B + k*2 + kidx);
        const float32x4_t B_val3 = vld1q_dup_f32(B + k*3 + kidx);
        o00 = vfmaq_f32(o00, A_col0, B_val0);
        o10 = vfmaq_f32(o10, A_col0, B_val1);
        o20 = vfmaq_f32(o20, A_col0, B_val2);
        o30 = vfmaq_f32(o30, A_col0, B_val3);
        o01 = vfmaq_f32(o01, A_col1, B_val0);
        o11 = vfmaq_f32(o11, A_col1, B_val1);
        o21 = vfmaq_f32(o21, A_col1, B_val2);
        o31 = vfmaq_f32(o31, A_col1, B_val3);
   }
    vst1q_f32(C[0], o00);
    vst1q_f32(C[1], o10);
    vst1q_f32(C[2], o20);
    vst1q_f32(C[3], o30);
    vst1q_f32(C[0] + 4, o01);
    vst1q_f32(C[1] + 4, o11);
    vst1q_f32(C[2] + 4, o21);
    vst1q_f32(C[3] + 4, o31);
}
void matmul_f32_NEON_8x5 (float *A, float *B, float **C, int k)
{
    float32x4_t o00, o10, o20, o30, o40;
    float32x4_t o01, o11, o21, o31, o41;
    o00 = vld1q_f32(C[0]);
    o10 = vld1q_f32(C[1]);
    o20 = vld1q_f32(C[2]);
    o30 = vld1q_f32(C[3]);
    o40 = vld1q_f32(C[4]);
    o01 = vld1q_f32(C[0] + 4);
    o11 = vld1q_f32(C[1] + 4);
    o21 = vld1q_f32(C[2] + 4);
    o31 = vld1q_f32(C[3] + 4);
    o41 = vld1q_f32(C[4] + 4);
    for (int kidx = 0; kidx < k; kidx++)
    {
        const float32x4_t A_col0 = vld1q_f32(A);
        const float32x4_t A_col1 = vld1q_f32(A + 4);
        A += 8;
        const float32x4_t B_val0 = vld1q_dup_f32(B + k*0 + kidx);
        const float32x4_t B_val1 = vld1q_dup_f32(B + k*1 + kidx);
        const float32x4_t B_val2 = vld1q_dup_f32(B + k*2 + kidx);
        const float32x4_t B_val3 = vld1q_dup_f32(B + k*3 + kidx);
        o00 = vfmaq_f32(o00, A_col0, B_val0);
        o10 = vfmaq_f32(o10, A_col0, B_val1);
        o20 = vfmaq_f32(o20, A_col0, B_val2);
        o30 = vfmaq_f32(o30, A_col0, B_val3);
        o01 = vfmaq_f32(o01, A_col1, B_val0);
        o11 = vfmaq_f32(o11, A_col1, B_val1);
        o21 = vfmaq_f32(o21, A_col1, B_val2);
        o31 = vfmaq_f32(o31, A_col1, B_val3);
        const float32x4_t B_val4 = vld1q_dup_f32(B + k*4 + kidx);
        o40 = vfmaq_f32(o40, A_col0, B_val4);
        o41 = vfmaq_f32(o41, A_col1, B_val4);
   }
    vst1q_f32(C[0], o00);
    vst1q_f32(C[1], o10);
    vst1q_f32(C[2], o20);
    vst1q_f32(C[3], o30);
    vst1q_f32(C[4], o40);
    vst1q_f32(C[0] + 4, o01);
    vst1q_f32(C[1] + 4, o11);
    vst1q_f32(C[2] + 4, o21);
    vst1q_f32(C[3] + 4, o31);
    vst1q_f32(C[4] + 4, o41);
}
void matmul_f32_NEON_8x6 (float *A, float *B, float **C, int k)
{
    float32x4_t o00, o10, o20, o30, o40, o50;
    float32x4_t o01, o11, o21, o31, o41, o51;
    o00 = vld1q_f32(C[0]);
    o10 = vld1q_f32(C[1]);
    o20 = vld1q_f32(C[2]);
    o30 = vld1q_f32(C[3]);
    o40 = vld1q_f32(C[4]);
    o50 = vld1q_f32(C[5]);
    o01 = vld1q_f32(C[0] + 4);
    o11 = vld1q_f32(C[1] + 4);
    o21 = vld1q_f32(C[2] + 4);
    o31 = vld1q_f32(C[3] + 4);
    o41 = vld1q_f32(C[4] + 4);
    o51 = vld1q_f32(C[5] + 4);
    for (int kidx = 0; kidx < k; kidx++)
    {
        const float32x4_t A_col0 = vld1q_f32(A);
        const float32x4_t A_col1 = vld1q_f32(A + 4);
        A += 8;
        const float32x4_t B_val0 = vld1q_dup_f32(B + k*0 + kidx);
        const float32x4_t B_val1 = vld1q_dup_f32(B + k*1 + kidx);
        const float32x4_t B_val2 = vld1q_dup_f32(B + k*2 + kidx);
        const float32x4_t B_val3 = vld1q_dup_f32(B + k*3 + kidx);
        o00 = vfmaq_f32(o00, A_col0, B_val0);
        o10 = vfmaq_f32(o10, A_col0, B_val1);
        o20 = vfmaq_f32(o20, A_col0, B_val2);
        o30 = vfmaq_f32(o30, A_col0, B_val3);
        o01 = vfmaq_f32(o01, A_col1, B_val0);
        o11 = vfmaq_f32(o11, A_col1, B_val1);
        o21 = vfmaq_f32(o21, A_col1, B_val2);
        o31 = vfmaq_f32(o31, A_col1, B_val3);
        const float32x4_t B_val4 = vld1q_dup_f32(B + k*4 + kidx);
        const float32x4_t B_val5 = vld1q_dup_f32(B + k*5 + kidx);
        o40 = vfmaq_f32(o40, A_col0, B_val4);
        o50 = vfmaq_f32(o50, A_col0, B_val5);
        o41 = vfmaq_f32(o41, A_col1, B_val4);
        o51 = vfmaq_f32(o51, A_col1, B_val5);
   }
    vst1q_f32(C[0], o00);
    vst1q_f32(C[1], o10);
    vst1q_f32(C[2], o20);
    vst1q_f32(C[3], o30);
    vst1q_f32(C[4], o40);
    vst1q_f32(C[5], o50);
    vst1q_f32(C[0] + 4, o01);
    vst1q_f32(C[1] + 4, o11);
    vst1q_f32(C[2] + 4, o21);
    vst1q_f32(C[3] + 4, o31);
    vst1q_f32(C[4] + 4, o41);
    vst1q_f32(C[5] + 4, o51);
}
void matmul_f32_NEON_8x7 (float *A, float *B, float **C, int k)
{
    float32x4_t o00, o10, o20, o30, o40, o50, o60;
    float32x4_t o01, o11, o21, o31, o41, o51, o61;
    o00 = vld1q_f32(C[0]);
    o10 = vld1q_f32(C[1]);
    o20 = vld1q_f32(C[2]);
    o30 = vld1q_f32(C[3]);
    o40 = vld1q_f32(C[4]);
    o50 = vld1q_f32(C[5]);
    o60 = vld1q_f32(C[6]);
    o01 = vld1q_f32(C[0] + 4);
    o11 = vld1q_f32(C[1] + 4);
    o21 = vld1q_f32(C[2] + 4);
    o31 = vld1q_f32(C[3] + 4);
    o41 = vld1q_f32(C[4] + 4);
    o51 = vld1q_f32(C[5] + 4);
    o61 = vld1q_f32(C[6] + 4);
    for (int kidx = 0; kidx < k; kidx++)
    {
        const float32x4_t A_col0 = vld1q_f32(A);
        const float32x4_t A_col1 = vld1q_f32(A + 4);
        A += 8;
        const float32x4_t B_val0 = vld1q_dup_f32(B + k*0 + kidx);
        const float32x4_t B_val1 = vld1q_dup_f32(B + k*1 + kidx);
        const float32x4_t B_val2 = vld1q_dup_f32(B + k*2 + kidx);
        const float32x4_t B_val3 = vld1q_dup_f32(B + k*3 + kidx);
        o00 = vfmaq_f32(o00, A_col0, B_val0);
        o10 = vfmaq_f32(o10, A_col0, B_val1);
        o20 = vfmaq_f32(o20, A_col0, B_val2);
        o30 = vfmaq_f32(o30, A_col0, B_val3);
        o01 = vfmaq_f32(o01, A_col1, B_val0);
        o11 = vfmaq_f32(o11, A_col1, B_val1);
        o21 = vfmaq_f32(o21, A_col1, B_val2);
        o31 = vfmaq_f32(o31, A_col1, B_val3);
        const float32x4_t B_val4 = vld1q_dup_f32(B + k*4 + kidx);
        const float32x4_t B_val5 = vld1q_dup_f32(B + k*5 + kidx);
        const float32x4_t B_val6 = vld1q_dup_f32(B + k*6 + kidx);
        o40 = vfmaq_f32(o40, A_col0, B_val4);
        o50 = vfmaq_f32(o50, A_col0, B_val5);
        o60 = vfmaq_f32(o60, A_col0, B_val6);
        o41 = vfmaq_f32(o41, A_col1, B_val4);
        o51 = vfmaq_f32(o51, A_col1, B_val5);
        o61 = vfmaq_f32(o61, A_col1, B_val6);
   }
    vst1q_f32(C[0], o00);
    vst1q_f32(C[1], o10);
    vst1q_f32(C[2], o20);
    vst1q_f32(C[3], o30);
    vst1q_f32(C[4], o40);
    vst1q_f32(C[5], o50);
    vst1q_f32(C[6], o60);
    vst1q_f32(C[0] + 4, o01);
    vst1q_f32(C[1] + 4, o11);
    vst1q_f32(C[2] + 4, o21);
    vst1q_f32(C[3] + 4, o31);
    vst1q_f32(C[4] + 4, o41);
    vst1q_f32(C[5] + 4, o51);
    vst1q_f32(C[6] + 4, o61);
}
void matmul_f32_NEON_8x8 (float *A, float *B, float **C, int k)
{
    float32x4_t o00, o10, o20, o30, o40, o50, o60, o70;
    float32x4_t o01, o11, o21, o31, o41, o51, o61, o71;
    o00 = vld1q_f32(C[0]);
    o10 = vld1q_f32(C[1]);
    o20 = vld1q_f32(C[2]);
    o30 = vld1q_f32(C[3]);
    o40 = vld1q_f32(C[4]);
    o50 = vld1q_f32(C[5]);
    o60 = vld1q_f32(C[6]);
    o70 = vld1q_f32(C[7]);
    o01 = vld1q_f32(C[0] + 4);
    o11 = vld1q_f32(C[1] + 4);
    o21 = vld1q_f32(C[2] + 4);
    o31 = vld1q_f32(C[3] + 4);
    o41 = vld1q_f32(C[4] + 4);
    o51 = vld1q_f32(C[5] + 4);
    o61 = vld1q_f32(C[6] + 4);
    o71 = vld1q_f32(C[7] + 4);
    for (int kidx = 0; kidx < k; kidx++)
    {
        const float32x4_t A_col0 = vld1q_f32(A);
        const float32x4_t A_col1 = vld1q_f32(A + 4);
        A += 8;
        const float32x4_t B_val0 = vld1q_dup_f32(B + k*0 + kidx);
        const float32x4_t B_val1 = vld1q_dup_f32(B + k*1 + kidx);
        const float32x4_t B_val2 = vld1q_dup_f32(B + k*2 + kidx);
        const float32x4_t B_val3 = vld1q_dup_f32(B + k*3 + kidx);
        o00 = vfmaq_f32(o00, A_col0, B_val0);
        o10 = vfmaq_f32(o10, A_col0, B_val1);
        o20 = vfmaq_f32(o20, A_col0, B_val2);
        o30 = vfmaq_f32(o30, A_col0, B_val3);
        o01 = vfmaq_f32(o01, A_col1, B_val0);
        o11 = vfmaq_f32(o11, A_col1, B_val1);
        o21 = vfmaq_f32(o21, A_col1, B_val2);
        o31 = vfmaq_f32(o31, A_col1, B_val3);
        const float32x4_t B_val4 = vld1q_dup_f32(B + k*4 + kidx);
        const float32x4_t B_val5 = vld1q_dup_f32(B + k*5 + kidx);
        const float32x4_t B_val6 = vld1q_dup_f32(B + k*6 + kidx);
        const float32x4_t B_val7 = vld1q_dup_f32(B + k*7 + kidx);
        o40 = vfmaq_f32(o40, A_col0, B_val4);
        o50 = vfmaq_f32(o50, A_col0, B_val5);
        o60 = vfmaq_f32(o60, A_col0, B_val6);
        o70 = vfmaq_f32(o70, A_col0, B_val7);
        o41 = vfmaq_f32(o41, A_col1, B_val4);
        o51 = vfmaq_f32(o51, A_col1, B_val5);
        o61 = vfmaq_f32(o61, A_col1, B_val6);
        o71 = vfmaq_f32(o71, A_col1, B_val7);
   }
    vst1q_f32(C[0], o00);
    vst1q_f32(C[1], o10);
    vst1q_f32(C[2], o20);
    vst1q_f32(C[3], o30);
    vst1q_f32(C[4], o40);
    vst1q_f32(C[5], o50);
    vst1q_f32(C[6], o60);
    vst1q_f32(C[7], o70);
    vst1q_f32(C[0] + 4, o01);
    vst1q_f32(C[1] + 4, o11);
    vst1q_f32(C[2] + 4, o21);
    vst1q_f32(C[3] + 4, o31);
    vst1q_f32(C[4] + 4, o41);
    vst1q_f32(C[5] + 4, o51);
    vst1q_f32(C[6] + 4, o61);
    vst1q_f32(C[7] + 4, o71);
}
void matmul_f32_NEON_8x9 (float *A, float *B, float **C, int k)
{
    float32x4_t o00, o10, o20, o30, o40, o50, o60, o70, o80;
    float32x4_t o01, o11, o21, o31, o41, o51, o61, o71, o81;
    o00 = vld1q_f32(C[0]);
    o10 = vld1q_f32(C[1]);
    o20 = vld1q_f32(C[2]);
    o30 = vld1q_f32(C[3]);
    o40 = vld1q_f32(C[4]);
    o50 = vld1q_f32(C[5]);
    o60 = vld1q_f32(C[6]);
    o70 = vld1q_f32(C[7]);
    o80 = vld1q_f32(C[8]);
    o01 = vld1q_f32(C[0] + 4);
    o11 = vld1q_f32(C[1] + 4);
    o21 = vld1q_f32(C[2] + 4);
    o31 = vld1q_f32(C[3] + 4);
    o41 = vld1q_f32(C[4] + 4);
    o51 = vld1q_f32(C[5] + 4);
    o61 = vld1q_f32(C[6] + 4);
    o71 = vld1q_f32(C[7] + 4);
    o81 = vld1q_f32(C[8] + 4);
    for (int kidx = 0; kidx < k; kidx++)
    {
        const float32x4_t A_col0 = vld1q_f32(A);
        const float32x4_t A_col1 = vld1q_f32(A + 4);
        A += 8;
        const float32x4_t B_val0 = vld1q_dup_f32(B + k*0 + kidx);
        const float32x4_t B_val1 = vld1q_dup_f32(B + k*1 + kidx);
        const float32x4_t B_val2 = vld1q_dup_f32(B + k*2 + kidx);
        const float32x4_t B_val3 = vld1q_dup_f32(B + k*3 + kidx);
        o00 = vfmaq_f32(o00, A_col0, B_val0);
        o10 = vfmaq_f32(o10, A_col0, B_val1);
        o20 = vfmaq_f32(o20, A_col0, B_val2);
        o30 = vfmaq_f32(o30, A_col0, B_val3);
        o01 = vfmaq_f32(o01, A_col1, B_val0);
        o11 = vfmaq_f32(o11, A_col1, B_val1);
        o21 = vfmaq_f32(o21, A_col1, B_val2);
        o31 = vfmaq_f32(o31, A_col1, B_val3);
        const float32x4_t B_val4 = vld1q_dup_f32(B + k*4 + kidx);
        const float32x4_t B_val5 = vld1q_dup_f32(B + k*5 + kidx);
        const float32x4_t B_val6 = vld1q_dup_f32(B + k*6 + kidx);
        const float32x4_t B_val7 = vld1q_dup_f32(B + k*7 + kidx);
        o40 = vfmaq_f32(o40, A_col0, B_val4);
        o50 = vfmaq_f32(o50, A_col0, B_val5);
        o60 = vfmaq_f32(o60, A_col0, B_val6);
        o70 = vfmaq_f32(o70, A_col0, B_val7);
        o41 = vfmaq_f32(o41, A_col1, B_val4);
        o51 = vfmaq_f32(o51, A_col1, B_val5);
        o61 = vfmaq_f32(o61, A_col1, B_val6);
        o71 = vfmaq_f32(o71, A_col1, B_val7);
        const float32x4_t B_val8 = vld1q_dup_f32(B + k*8 + kidx);
        o80 = vfmaq_f32(o80, A_col0, B_val8);
        o81 = vfmaq_f32(o81, A_col1, B_val8);
   }
    vst1q_f32(C[0], o00);
    vst1q_f32(C[1], o10);
    vst1q_f32(C[2], o20);
    vst1q_f32(C[3], o30);
    vst1q_f32(C[4], o40);
    vst1q_f32(C[5], o50);
    vst1q_f32(C[6], o60);
    vst1q_f32(C[7], o70);
    vst1q_f32(C[8], o80);
    vst1q_f32(C[0] + 4, o01);
    vst1q_f32(C[1] + 4, o11);
    vst1q_f32(C[2] + 4, o21);
    vst1q_f32(C[3] + 4, o31);
    vst1q_f32(C[4] + 4, o41);
    vst1q_f32(C[5] + 4, o51);
    vst1q_f32(C[6] + 4, o61);
    vst1q_f32(C[7] + 4, o71);
    vst1q_f32(C[8] + 4, o81);
}
void matmul_f32_NEON_8x10 (float *A, float *B, float **C, int k)
{
    float32x4_t o00, o10, o20, o30, o40, o50, o60, o70, o80, o90;
    float32x4_t o01, o11, o21, o31, o41, o51, o61, o71, o81, o91;
    o00 = vld1q_f32(C[0]);
    o10 = vld1q_f32(C[1]);
    o20 = vld1q_f32(C[2]);
    o30 = vld1q_f32(C[3]);
    o40 = vld1q_f32(C[4]);
    o50 = vld1q_f32(C[5]);
    o60 = vld1q_f32(C[6]);
    o70 = vld1q_f32(C[7]);
    o80 = vld1q_f32(C[8]);
    o90 = vld1q_f32(C[9]);
    o01 = vld1q_f32(C[0] + 4);
    o11 = vld1q_f32(C[1] + 4);
    o21 = vld1q_f32(C[2] + 4);
    o31 = vld1q_f32(C[3] + 4);
    o41 = vld1q_f32(C[4] + 4);
    o51 = vld1q_f32(C[5] + 4);
    o61 = vld1q_f32(C[6] + 4);
    o71 = vld1q_f32(C[7] + 4);
    o81 = vld1q_f32(C[8] + 4);
    o91 = vld1q_f32(C[9] + 4);
    for (int kidx = 0; kidx < k; kidx++)
    {
        const float32x4_t A_col0 = vld1q_f32(A);
        const float32x4_t A_col1 = vld1q_f32(A + 4);
        A += 8;
        const float32x4_t B_val0 = vld1q_dup_f32(B + k*0 + kidx);
        const float32x4_t B_val1 = vld1q_dup_f32(B + k*1 + kidx);
        const float32x4_t B_val2 = vld1q_dup_f32(B + k*2 + kidx);
        const float32x4_t B_val3 = vld1q_dup_f32(B + k*3 + kidx);
        o00 = vfmaq_f32(o00, A_col0, B_val0);
        o10 = vfmaq_f32(o10, A_col0, B_val1);
        o20 = vfmaq_f32(o20, A_col0, B_val2);
        o30 = vfmaq_f32(o30, A_col0, B_val3);
        o01 = vfmaq_f32(o01, A_col1, B_val0);
        o11 = vfmaq_f32(o11, A_col1, B_val1);
        o21 = vfmaq_f32(o21, A_col1, B_val2);
        o31 = vfmaq_f32(o31, A_col1, B_val3);
        const float32x4_t B_val4 = vld1q_dup_f32(B + k*4 + kidx);
        const float32x4_t B_val5 = vld1q_dup_f32(B + k*5 + kidx);
        const float32x4_t B_val6 = vld1q_dup_f32(B + k*6 + kidx);
        const float32x4_t B_val7 = vld1q_dup_f32(B + k*7 + kidx);
        o40 = vfmaq_f32(o40, A_col0, B_val4);
        o50 = vfmaq_f32(o50, A_col0, B_val5);
        o60 = vfmaq_f32(o60, A_col0, B_val6);
        o70 = vfmaq_f32(o70, A_col0, B_val7);
        o41 = vfmaq_f32(o41, A_col1, B_val4);
        o51 = vfmaq_f32(o51, A_col1, B_val5);
        o61 = vfmaq_f32(o61, A_col1, B_val6);
        o71 = vfmaq_f32(o71, A_col1, B_val7);
        const float32x4_t B_val8 = vld1q_dup_f32(B + k*8 + kidx);
        const float32x4_t B_val9 = vld1q_dup_f32(B + k*9 + kidx);
        o80 = vfmaq_f32(o80, A_col0, B_val8);
        o90 = vfmaq_f32(o90, A_col0, B_val9);
        o81 = vfmaq_f32(o81, A_col1, B_val8);
        o91 = vfmaq_f32(o91, A_col1, B_val9);
   }
    vst1q_f32(C[0], o00);
    vst1q_f32(C[1], o10);
    vst1q_f32(C[2], o20);
    vst1q_f32(C[3], o30);
    vst1q_f32(C[4], o40);
    vst1q_f32(C[5], o50);
    vst1q_f32(C[6], o60);
    vst1q_f32(C[7], o70);
    vst1q_f32(C[8], o80);
    vst1q_f32(C[9], o90);
    vst1q_f32(C[0] + 4, o01);
    vst1q_f32(C[1] + 4, o11);
    vst1q_f32(C[2] + 4, o21);
    vst1q_f32(C[3] + 4, o31);
    vst1q_f32(C[4] + 4, o41);
    vst1q_f32(C[5] + 4, o51);
    vst1q_f32(C[6] + 4, o61);
    vst1q_f32(C[7] + 4, o71);
    vst1q_f32(C[8] + 4, o81);
    vst1q_f32(C[9] + 4, o91);
}
void matmul_f32_NEON_8x11 (float *A, float *B, float **C, int k)
{
    float32x4_t o00, o10, o20, o30, o40, o50, o60, o70, o80, o90, o100;
    float32x4_t o01, o11, o21, o31, o41, o51, o61, o71, o81, o91, o101;
    o00 = vld1q_f32(C[0]);
    o10 = vld1q_f32(C[1]);
    o20 = vld1q_f32(C[2]);
    o30 = vld1q_f32(C[3]);
    o40 = vld1q_f32(C[4]);
    o50 = vld1q_f32(C[5]);
    o60 = vld1q_f32(C[6]);
    o70 = vld1q_f32(C[7]);
    o80 = vld1q_f32(C[8]);
    o90 = vld1q_f32(C[9]);
    o100 = vld1q_f32(C[10]);
    o01 = vld1q_f32(C[0] + 4);
    o11 = vld1q_f32(C[1] + 4);
    o21 = vld1q_f32(C[2] + 4);
    o31 = vld1q_f32(C[3] + 4);
    o41 = vld1q_f32(C[4] + 4);
    o51 = vld1q_f32(C[5] + 4);
    o61 = vld1q_f32(C[6] + 4);
    o71 = vld1q_f32(C[7] + 4);
    o81 = vld1q_f32(C[8] + 4);
    o91 = vld1q_f32(C[9] + 4);
    o101 = vld1q_f32(C[10] + 4);
    for (int kidx = 0; kidx < k; kidx++)
    {
        const float32x4_t A_col0 = vld1q_f32(A);
        const float32x4_t A_col1 = vld1q_f32(A + 4);
        A += 8;
        const float32x4_t B_val0 = vld1q_dup_f32(B + k*0 + kidx);
        const float32x4_t B_val1 = vld1q_dup_f32(B + k*1 + kidx);
        const float32x4_t B_val2 = vld1q_dup_f32(B + k*2 + kidx);
        const float32x4_t B_val3 = vld1q_dup_f32(B + k*3 + kidx);
        o00 = vfmaq_f32(o00, A_col0, B_val0);
        o10 = vfmaq_f32(o10, A_col0, B_val1);
        o20 = vfmaq_f32(o20, A_col0, B_val2);
        o30 = vfmaq_f32(o30, A_col0, B_val3);
        o01 = vfmaq_f32(o01, A_col1, B_val0);
        o11 = vfmaq_f32(o11, A_col1, B_val1);
        o21 = vfmaq_f32(o21, A_col1, B_val2);
        o31 = vfmaq_f32(o31, A_col1, B_val3);
        const float32x4_t B_val4 = vld1q_dup_f32(B + k*4 + kidx);
        const float32x4_t B_val5 = vld1q_dup_f32(B + k*5 + kidx);
        const float32x4_t B_val6 = vld1q_dup_f32(B + k*6 + kidx);
        const float32x4_t B_val7 = vld1q_dup_f32(B + k*7 + kidx);
        o40 = vfmaq_f32(o40, A_col0, B_val4);
        o50 = vfmaq_f32(o50, A_col0, B_val5);
        o60 = vfmaq_f32(o60, A_col0, B_val6);
        o70 = vfmaq_f32(o70, A_col0, B_val7);
        o41 = vfmaq_f32(o41, A_col1, B_val4);
        o51 = vfmaq_f32(o51, A_col1, B_val5);
        o61 = vfmaq_f32(o61, A_col1, B_val6);
        o71 = vfmaq_f32(o71, A_col1, B_val7);
        const float32x4_t B_val8 = vld1q_dup_f32(B + k*8 + kidx);
        const float32x4_t B_val9 = vld1q_dup_f32(B + k*9 + kidx);
        const float32x4_t B_val10 = vld1q_dup_f32(B + k*10 + kidx);
        o80 = vfmaq_f32(o80, A_col0, B_val8);
        o90 = vfmaq_f32(o90, A_col0, B_val9);
        o100 = vfmaq_f32(o100, A_col0, B_val10);
        o81 = vfmaq_f32(o81, A_col1, B_val8);
        o91 = vfmaq_f32(o91, A_col1, B_val9);
        o101 = vfmaq_f32(o101, A_col1, B_val10);
   }
    vst1q_f32(C[0], o00);
    vst1q_f32(C[1], o10);
    vst1q_f32(C[2], o20);
    vst1q_f32(C[3], o30);
    vst1q_f32(C[4], o40);
    vst1q_f32(C[5], o50);
    vst1q_f32(C[6], o60);
    vst1q_f32(C[7], o70);
    vst1q_f32(C[8], o80);
    vst1q_f32(C[9], o90);
    vst1q_f32(C[10], o100);
    vst1q_f32(C[0] + 4, o01);
    vst1q_f32(C[1] + 4, o11);
    vst1q_f32(C[2] + 4, o21);
    vst1q_f32(C[3] + 4, o31);
    vst1q_f32(C[4] + 4, o41);
    vst1q_f32(C[5] + 4, o51);
    vst1q_f32(C[6] + 4, o61);
    vst1q_f32(C[7] + 4, o71);
    vst1q_f32(C[8] + 4, o81);
    vst1q_f32(C[9] + 4, o91);
    vst1q_f32(C[10] + 4, o101);
}
void matmul_f32_NEON_8x12 (float *A, float *B, float **C, int k)
{
    float32x4_t o00, o10, o20, o30, o40, o50, o60, o70, o80, o90, o100, o110;
    float32x4_t o01, o11, o21, o31, o41, o51, o61, o71, o81, o91, o101, o111;
    o00 = vld1q_f32(C[0]);
    o10 = vld1q_f32(C[1]);
    o20 = vld1q_f32(C[2]);
    o30 = vld1q_f32(C[3]);
    o40 = vld1q_f32(C[4]);
    o50 = vld1q_f32(C[5]);
    o60 = vld1q_f32(C[6]);
    o70 = vld1q_f32(C[7]);
    o80 = vld1q_f32(C[8]);
    o90 = vld1q_f32(C[9]);
    o100 = vld1q_f32(C[10]);
    o110 = vld1q_f32(C[11]);
    o01 = vld1q_f32(C[0] + 4);
    o11 = vld1q_f32(C[1] + 4);
    o21 = vld1q_f32(C[2] + 4);
    o31 = vld1q_f32(C[3] + 4);
    o41 = vld1q_f32(C[4] + 4);
    o51 = vld1q_f32(C[5] + 4);
    o61 = vld1q_f32(C[6] + 4);
    o71 = vld1q_f32(C[7] + 4);
    o81 = vld1q_f32(C[8] + 4);
    o91 = vld1q_f32(C[9] + 4);
    o101 = vld1q_f32(C[10] + 4);
    o111 = vld1q_f32(C[11] + 4);
    for (int kidx = 0; kidx < k; kidx++)
    {
        const float32x4_t A_col0 = vld1q_f32(A);
        const float32x4_t A_col1 = vld1q_f32(A + 4);
        A += 8;
        const float32x4_t B_val0 = vld1q_dup_f32(B + k*0 + kidx);
        const float32x4_t B_val1 = vld1q_dup_f32(B + k*1 + kidx);
        const float32x4_t B_val2 = vld1q_dup_f32(B + k*2 + kidx);
        const float32x4_t B_val3 = vld1q_dup_f32(B + k*3 + kidx);
        o00 = vfmaq_f32(o00, A_col0, B_val0);
        o10 = vfmaq_f32(o10, A_col0, B_val1);
        o20 = vfmaq_f32(o20, A_col0, B_val2);
        o30 = vfmaq_f32(o30, A_col0, B_val3);
        o01 = vfmaq_f32(o01, A_col1, B_val0);
        o11 = vfmaq_f32(o11, A_col1, B_val1);
        o21 = vfmaq_f32(o21, A_col1, B_val2);
        o31 = vfmaq_f32(o31, A_col1, B_val3);
        const float32x4_t B_val4 = vld1q_dup_f32(B + k*4 + kidx);
        const float32x4_t B_val5 = vld1q_dup_f32(B + k*5 + kidx);
        const float32x4_t B_val6 = vld1q_dup_f32(B + k*6 + kidx);
        const float32x4_t B_val7 = vld1q_dup_f32(B + k*7 + kidx);
        o40 = vfmaq_f32(o40, A_col0, B_val4);
        o50 = vfmaq_f32(o50, A_col0, B_val5);
        o60 = vfmaq_f32(o60, A_col0, B_val6);
        o70 = vfmaq_f32(o70, A_col0, B_val7);
        o41 = vfmaq_f32(o41, A_col1, B_val4);
        o51 = vfmaq_f32(o51, A_col1, B_val5);
        o61 = vfmaq_f32(o61, A_col1, B_val6);
        o71 = vfmaq_f32(o71, A_col1, B_val7);
        const float32x4_t B_val8 = vld1q_dup_f32(B + k*8 + kidx);
        const float32x4_t B_val9 = vld1q_dup_f32(B + k*9 + kidx);
        const float32x4_t B_val10 = vld1q_dup_f32(B + k*10 + kidx);
        const float32x4_t B_val11 = vld1q_dup_f32(B + k*11 + kidx);
        o80 = vfmaq_f32(o80, A_col0, B_val8);
        o90 = vfmaq_f32(o90, A_col0, B_val9);
        o100 = vfmaq_f32(o100, A_col0, B_val10);
        o110 = vfmaq_f32(o110, A_col0, B_val11);
        o81 = vfmaq_f32(o81, A_col1, B_val8);
        o91 = vfmaq_f32(o91, A_col1, B_val9);
        o101 = vfmaq_f32(o101, A_col1, B_val10);
        o111 = vfmaq_f32(o111, A_col1, B_val11);
   }
    vst1q_f32(C[0], o00);
    vst1q_f32(C[1], o10);
    vst1q_f32(C[2], o20);
    vst1q_f32(C[3], o30);
    vst1q_f32(C[4], o40);
    vst1q_f32(C[5], o50);
    vst1q_f32(C[6], o60);
    vst1q_f32(C[7], o70);
    vst1q_f32(C[8], o80);
    vst1q_f32(C[9], o90);
    vst1q_f32(C[10], o100);
    vst1q_f32(C[11], o110);
    vst1q_f32(C[0] + 4, o01);
    vst1q_f32(C[1] + 4, o11);
    vst1q_f32(C[2] + 4, o21);
    vst1q_f32(C[3] + 4, o31);
    vst1q_f32(C[4] + 4, o41);
    vst1q_f32(C[5] + 4, o51);
    vst1q_f32(C[6] + 4, o61);
    vst1q_f32(C[7] + 4, o71);
    vst1q_f32(C[8] + 4, o81);
    vst1q_f32(C[9] + 4, o91);
    vst1q_f32(C[10] + 4, o101);
    vst1q_f32(C[11] + 4, o111);
}

void matmul_f32_NEON_16x1(float *A, float *B, float **C, int k)
{
    float32x4_t o0, o1;
    float32x4_t o00, o10;
    o0 = vld1q_f32(C[0] + 8*0);
    o1 = vld1q_f32(C[0] + 8*1);
    o00 = vld1q_f32(C[0] + 8*0 + 4);
    o10 = vld1q_f32(C[0] + 8*1 + 4);
    for (int kidx = 0; kidx < k; kidx++)
    {
        const float32x4_t A_col0 = vld1q_f32(A + k*8*0);
        const float32x4_t A_col1 = vld1q_f32(A + k*8*1);
        const float32x4_t A_col00 = vld1q_f32(A + k*8*0 + 4);
        const float32x4_t A_col10 = vld1q_f32(A + k*8*1 + 4);
        A += 8;
        const float32x4_t B_val0 = vld1q_dup_f32(B + k*0 + kidx);
        o0 = vfmaq_f32(o0, A_col0, B_val0);
        o1 = vfmaq_f32(o1, A_col1, B_val0);
        o00 = vfmaq_f32(o00, A_col00, B_val0);
        o10 = vfmaq_f32(o10, A_col10, B_val0);
    }
    vst1q_f32(C[0] + 8*0, o0);
    vst1q_f32(C[0] + 8*1, o1);
    vst1q_f32(C[0] + 8*0 + 4, o00);
    vst1q_f32(C[0] + 8*1 + 4, o10);
}

void matmul_f32_NEON_32x1(float *A, float *B, float **C, int k)
{
    float32x4_t o0, o1, o2, o3;
    float32x4_t o00, o10, o20, o30;
    o0 = vld1q_f32(C[0] + 8*0);
    o1 = vld1q_f32(C[0] + 8*1);
    o2 = vld1q_f32(C[0] + 8*2);
    o3 = vld1q_f32(C[0] + 8*3);
    o00 = vld1q_f32(C[0] + 8*0 + 4);
    o10 = vld1q_f32(C[0] + 8*1 + 4);
    o20 = vld1q_f32(C[0] + 8*2 + 4);
    o30 = vld1q_f32(C[0] + 8*3 + 4);
    for (int kidx = 0; kidx < k; kidx++)
    {
        const float32x4_t A_col0 = vld1q_f32(A + k*8*0);
        const float32x4_t A_col1 = vld1q_f32(A + k*8*1);
        const float32x4_t A_col2 = vld1q_f32(A + k*8*2);
        const float32x4_t A_col3 = vld1q_f32(A + k*8*3);
        const float32x4_t A_col00 = vld1q_f32(A + k*8*0 + 4);
        const float32x4_t A_col10 = vld1q_f32(A + k*8*1 + 4);
        const float32x4_t A_col20 = vld1q_f32(A + k*8*2 + 4);
        const float32x4_t A_col30 = vld1q_f32(A + k*8*3 + 4);
        A += 8;
        const float32x4_t B_val0 = vld1q_dup_f32(B + k*0 + kidx);
        o0 = vfmaq_f32(o0, A_col0, B_val0);
        o1 = vfmaq_f32(o1, A_col1, B_val0);
        o2 = vfmaq_f32(o2, A_col2, B_val0);
        o3 = vfmaq_f32(o3, A_col3, B_val0);
        o00 = vfmaq_f32(o00, A_col00, B_val0);
        o10 = vfmaq_f32(o10, A_col10, B_val0);
        o20 = vfmaq_f32(o20, A_col20, B_val0);
        o30 = vfmaq_f32(o30, A_col30, B_val0);
    }
    vst1q_f32(C[0] + 8*0, o0);
    vst1q_f32(C[0] + 8*1, o1);
    vst1q_f32(C[0] + 8*2, o2);
    vst1q_f32(C[0] + 8*3, o3);
    vst1q_f32(C[0] + 8*0 + 4, o00);
    vst1q_f32(C[0] + 8*1 + 4, o10);
    vst1q_f32(C[0] + 8*2 + 4, o20);
    vst1q_f32(C[0] + 8*3 + 4, o30);
}

void matmul_f32_NEON_64x1(float *A, float *B, float **C, int k)
{
    float32x4_t o0, o1, o2, o3, o4, o5, o6, o7;
    float32x4_t o00, o10, o20, o30, o40, o50, o60, o70;
    o0 = vld1q_f32(C[0] + 8*0);
    o1 = vld1q_f32(C[0] + 8*1);
    o2 = vld1q_f32(C[0] + 8*2);
    o3 = vld1q_f32(C[0] + 8*3);
    o4 = vld1q_f32(C[0] + 8*4);
    o5 = vld1q_f32(C[0] + 8*5);
    o6 = vld1q_f32(C[0] + 8*6);
    o7 = vld1q_f32(C[0] + 8*7);
    o00 = vld1q_f32(C[0] + 8*0 + 4);
    o10 = vld1q_f32(C[0] + 8*1 + 4);
    o20 = vld1q_f32(C[0] + 8*2 + 4);
    o30 = vld1q_f32(C[0] + 8*3 + 4);
    o40 = vld1q_f32(C[0] + 8*4 + 4);
    o50 = vld1q_f32(C[0] + 8*5 + 4);
    o60 = vld1q_f32(C[0] + 8*6 + 4);
    o70 = vld1q_f32(C[0] + 8*7 + 4);
    for (int kidx = 0; kidx < k; kidx++)
    {
        const float32x4_t A_col0 = vld1q_f32(A + k*8*0);
        const float32x4_t A_col1 = vld1q_f32(A + k*8*1);
        const float32x4_t A_col2 = vld1q_f32(A + k*8*2);
        const float32x4_t A_col3 = vld1q_f32(A + k*8*3);
        const float32x4_t A_col4 = vld1q_f32(A + k*8*4);
        const float32x4_t A_col5 = vld1q_f32(A + k*8*5);
        const float32x4_t A_col6 = vld1q_f32(A + k*8*6);
        const float32x4_t A_col7 = vld1q_f32(A + k*8*7);
        const float32x4_t A_col00 = vld1q_f32(A + k*8*0 + 4);
        const float32x4_t A_col10 = vld1q_f32(A + k*8*1 + 4);
        const float32x4_t A_col20 = vld1q_f32(A + k*8*2 + 4);
        const float32x4_t A_col30 = vld1q_f32(A + k*8*3 + 4);
        const float32x4_t A_col40 = vld1q_f32(A + k*8*4 + 4);
        const float32x4_t A_col50 = vld1q_f32(A + k*8*5 + 4);
        const float32x4_t A_col60 = vld1q_f32(A + k*8*6 + 4);
        const float32x4_t A_col70 = vld1q_f32(A + k*8*7 + 4);
        A += 8;
        const float32x4_t B_val0 = vld1q_dup_f32(B + k*0 + kidx);
        o0 = vfmaq_f32(o0, A_col0, B_val0);
        o1 = vfmaq_f32(o1, A_col1, B_val0);
        o2 = vfmaq_f32(o2, A_col2, B_val0);
        o3 = vfmaq_f32(o3, A_col3, B_val0);
        o4 = vfmaq_f32(o4, A_col4, B_val0);
        o5 = vfmaq_f32(o5, A_col5, B_val0);
        o6 = vfmaq_f32(o6, A_col6, B_val0);
        o7 = vfmaq_f32(o7, A_col7, B_val0);
        o00 = vfmaq_f32(o00, A_col00, B_val0);
        o10 = vfmaq_f32(o10, A_col10, B_val0);
        o20 = vfmaq_f32(o20, A_col20, B_val0);
        o30 = vfmaq_f32(o30, A_col30, B_val0);
        o40 = vfmaq_f32(o40, A_col40, B_val0);
        o50 = vfmaq_f32(o50, A_col50, B_val0);
        o60 = vfmaq_f32(o60, A_col60, B_val0);
        o70 = vfmaq_f32(o70, A_col70, B_val0);
    }
    vst1q_f32(C[0] + 8*0, o0);
    vst1q_f32(C[0] + 8*1, o1);
    vst1q_f32(C[0] + 8*2, o2);
    vst1q_f32(C[0] + 8*3, o3);
    vst1q_f32(C[0] + 8*4, o4);
    vst1q_f32(C[0] + 8*5, o5);
    vst1q_f32(C[0] + 8*6, o6);
    vst1q_f32(C[0] + 8*7, o7);
    vst1q_f32(C[0] + 8*0 + 4, o00);
    vst1q_f32(C[0] + 8*1 + 4, o10);
    vst1q_f32(C[0] + 8*2 + 4, o20);
    vst1q_f32(C[0] + 8*3 + 4, o30);
    vst1q_f32(C[0] + 8*4 + 4, o40);
    vst1q_f32(C[0] + 8*5 + 4, o50);
    vst1q_f32(C[0] + 8*6 + 4, o60);
    vst1q_f32(C[0] + 8*7 + 4, o70);
}

#endif 
