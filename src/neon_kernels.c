#include "kernels.h"
#ifdef NEON
void neon_sgemm_vectorized(const unsigned int M, const unsigned int N, const unsigned int K,
		 const float *A, const unsigned int lda, const float *B, const unsigned int ldb, float *C, const unsigned int ldc)
{
    const unsigned int rem_n = (N%_VEC_SIZE_N);
    const unsigned int rem_m = (M%_VEC_SIZE_M);
    for (unsigned int m = 0; m < M - rem_m; m += _VEC_SIZE_M)
    {
        unsigned int n = 0;
        for (; n < N - rem_n; n += _VEC_SIZE_N)
        {
            #if _VEC_SIZE_N == 12 && _VEC_SIZE_M == 8
            const float *A_ptr = A + m * lda;
            const float *B_ptr = B + n * ldb;
            float *C_ptr = C + n * ldc + m;
            float32x4_t o00, o10, o20, o30, o40, o50, o60, o70, o80, o90, o100, o110;
            float32x4_t o01, o11, o21, o31, o41, o51, o61, o71, o81, o91, o101, o111;
            o00 = vld1q_f32     (C_ptr + 0  * ldc);
            o10 = vld1q_f32     (C_ptr + 1  * ldc);
            o20 = vld1q_f32     (C_ptr + 2  * ldc);
            o30 = vld1q_f32     (C_ptr + 3  * ldc);
            o40 = vld1q_f32     (C_ptr + 4  * ldc);
            o50 = vld1q_f32     (C_ptr + 5  * ldc);
            o60 = vld1q_f32     (C_ptr + 6  * ldc);
            o70 = vld1q_f32     (C_ptr + 7  * ldc);
            o80 = vld1q_f32     (C_ptr + 8  * ldc);
            o90 = vld1q_f32     (C_ptr + 9  * ldc);
            o100 = vld1q_f32    (C_ptr + 10 * ldc);
            o110 = vld1q_f32    (C_ptr + 11 * ldc);
            o01 = vld1q_f32     (C_ptr + 0  * ldc + 4);
            o11 = vld1q_f32     (C_ptr + 1  * ldc + 4);
            o21 = vld1q_f32     (C_ptr + 2  * ldc + 4);
            o31 = vld1q_f32     (C_ptr + 3  * ldc + 4);
            o41 = vld1q_f32     (C_ptr + 4  * ldc + 4);
            o51 = vld1q_f32     (C_ptr + 5  * ldc + 4);
            o61 = vld1q_f32     (C_ptr + 6  * ldc + 4);
            o71 = vld1q_f32     (C_ptr + 7  * ldc + 4);
            o81 = vld1q_f32     (C_ptr + 8  * ldc + 4);
            o91 = vld1q_f32     (C_ptr + 9  * ldc + 4);
            o101 = vld1q_f32    (C_ptr + 10 * ldc + 4);
            o111 = vld1q_f32    (C_ptr + 11 * ldc + 4);
            for (int kidx = 0; kidx < K; kidx++)
            {
                const float32x4_t A_col0 = vld1q_f32(A_ptr);
                const float32x4_t A_col1 = vld1q_f32(A_ptr + 4);
                A_ptr += 8;
                const float32x4_t B_val0 = vld1q_dup_f32(B_ptr + ldb*0 + kidx);
                const float32x4_t B_val1 = vld1q_dup_f32(B_ptr + ldb*1 + kidx);
                const float32x4_t B_val2 = vld1q_dup_f32(B_ptr + ldb*2 + kidx);
                const float32x4_t B_val3 = vld1q_dup_f32(B_ptr + ldb*3 + kidx);
                const float32x4_t B_val4 = vld1q_dup_f32(B_ptr + ldb*4 + kidx);
                const float32x4_t B_val5 = vld1q_dup_f32(B_ptr + ldb*5 + kidx);
                o00 = vfmaq_f32(o00, A_col0, B_val0);
                o01 = vfmaq_f32(o01, A_col1, B_val0);
                o10 = vfmaq_f32(o10, A_col0, B_val1);
                o11 = vfmaq_f32(o11, A_col1, B_val1);
                o20 = vfmaq_f32(o20, A_col0, B_val2);
                o21 = vfmaq_f32(o21, A_col1, B_val2);
                o30 = vfmaq_f32(o30, A_col0, B_val3);
                o31 = vfmaq_f32(o31, A_col1, B_val3);
                o40 = vfmaq_f32(o40, A_col0, B_val4);
                o41 = vfmaq_f32(o41, A_col1, B_val4);
                o50 = vfmaq_f32(o50, A_col0, B_val5);
                o51 = vfmaq_f32(o51, A_col1, B_val5);
                const float32x4_t B_val6 = vld1q_dup_f32(B_ptr + ldb*6 + kidx);
                const float32x4_t B_val7 = vld1q_dup_f32(B_ptr + ldb*7 + kidx);
                const float32x4_t B_val8 = vld1q_dup_f32(B_ptr + ldb*8 + kidx);
                const float32x4_t B_val9 = vld1q_dup_f32(B_ptr + ldb*9 + kidx);
                const float32x4_t B_val10 = vld1q_dup_f32(B_ptr + ldb*10 + kidx);
                const float32x4_t B_val11 = vld1q_dup_f32(B_ptr + ldb*11 + kidx);
                o60 = vfmaq_f32(o60, A_col0, B_val6);
                o61 = vfmaq_f32(o61, A_col1, B_val6);
                o70 = vfmaq_f32(o70, A_col0, B_val7);
                o71 = vfmaq_f32(o71, A_col1, B_val7);
                o80 = vfmaq_f32(o80, A_col0, B_val8);
                o81 = vfmaq_f32(o81, A_col1, B_val8);
                o90 = vfmaq_f32(o90, A_col0, B_val9);
                o91 = vfmaq_f32(o91, A_col1, B_val9);
                o100 = vfmaq_f32(o100, A_col0, B_val10);
                o101 = vfmaq_f32(o101, A_col1, B_val10);
                o110 = vfmaq_f32(o110, A_col0, B_val11);
                o111 = vfmaq_f32(o111, A_col1, B_val11);
            }
            vst1q_f32  (C_ptr + 0  * ldc, o00);
            vst1q_f32  (C_ptr + 1  * ldc, o10);
            vst1q_f32  (C_ptr + 2  * ldc, o20);
            vst1q_f32  (C_ptr + 3  * ldc, o30);
            vst1q_f32  (C_ptr + 4  * ldc, o40);
            vst1q_f32  (C_ptr + 5  * ldc, o50);
            vst1q_f32  (C_ptr + 6  * ldc, o60);
            vst1q_f32  (C_ptr + 7  * ldc, o70);
            vst1q_f32  (C_ptr + 8  * ldc, o80);
            vst1q_f32  (C_ptr + 9  * ldc, o90);
            vst1q_f32  (C_ptr + 10 * ldc, o100);
            vst1q_f32  (C_ptr + 11 * ldc, o110);
            vst1q_f32  (C_ptr + 0  * ldc + 4, o01);
            vst1q_f32  (C_ptr + 1  * ldc + 4, o11);
            vst1q_f32  (C_ptr + 2  * ldc + 4, o21);
            vst1q_f32  (C_ptr + 3  * ldc + 4, o31);
            vst1q_f32  (C_ptr + 4  * ldc + 4, o41);
            vst1q_f32  (C_ptr + 5  * ldc + 4, o51);
            vst1q_f32  (C_ptr + 6  * ldc + 4, o61);
            vst1q_f32  (C_ptr + 7  * ldc + 4, o71);
            vst1q_f32  (C_ptr + 8  * ldc + 4, o81);
            vst1q_f32  (C_ptr + 9  * ldc + 4, o91);
            vst1q_f32  (C_ptr + 10 * ldc + 4, o101);
            vst1q_f32  (C_ptr + 11 * ldc + 4, o111);
            #else 
            for (unsigned int nn = n; nn < n + _VEC_SIZE_N; nn++)
            {
                for (unsigned int mm = m; mm < m + _VEC_SIZE_M; mm++)
                {
                    float c = C[nn * ldc + mm];
                    for (unsigned int k = 0; k < K; k++)
                    {
                        c += A[((mm/_VEC_SIZE_M) * lda + k) * _VEC_SIZE_M + (mm%_VEC_SIZE_M)] * B[nn * ldb + k];
                    }
                    C[nn * ldc + mm] = c;
                }
            }
            #endif
        }
        #if _VEC_SIZE_M == 8
        const float *A_ptr = A + m * lda;
        const float *B_ptr = B + n * ldb;
        float *C_ptr = C + n * ldc + m;
        if (rem_n == 8)
        {
            float32x4_t o00, o10, o20, o30, o40, o50, o60, o70;
            float32x4_t o01, o11, o21, o31, o41, o51, o61, o71;
            o00 = vld1q_f32     (C_ptr + 0  * ldc);
            o10 = vld1q_f32     (C_ptr + 1  * ldc);
            o20 = vld1q_f32     (C_ptr + 2  * ldc);
            o30 = vld1q_f32     (C_ptr + 3  * ldc);
            o40 = vld1q_f32     (C_ptr + 4  * ldc);
            o50 = vld1q_f32     (C_ptr + 5  * ldc);
            o60 = vld1q_f32     (C_ptr + 6  * ldc);
            o70 = vld1q_f32     (C_ptr + 7  * ldc);
            o01 = vld1q_f32     (C_ptr + 0  * ldc + 4);
            o11 = vld1q_f32     (C_ptr + 1  * ldc + 4);
            o21 = vld1q_f32     (C_ptr + 2  * ldc + 4);
            o31 = vld1q_f32     (C_ptr + 3  * ldc + 4);
            o41 = vld1q_f32     (C_ptr + 4  * ldc + 4);
            o51 = vld1q_f32     (C_ptr + 5  * ldc + 4);
            o61 = vld1q_f32     (C_ptr + 6  * ldc + 4);
            o71 = vld1q_f32     (C_ptr + 7  * ldc + 4);
            for (int kidx = 0; kidx < K; kidx++)
            {
                const float32x4_t A_col0 = vld1q_f32(A_ptr);
                const float32x4_t A_col1 = vld1q_f32(A_ptr + 4);
                A_ptr += 8;
                const float32x4_t B_val0 = vld1q_dup_f32(B_ptr + ldb*0 + kidx);
                const float32x4_t B_val1 = vld1q_dup_f32(B_ptr + ldb*1 + kidx);
                const float32x4_t B_val2 = vld1q_dup_f32(B_ptr + ldb*2 + kidx);
                const float32x4_t B_val3 = vld1q_dup_f32(B_ptr + ldb*3 + kidx);   
                const float32x4_t B_val4 = vld1q_dup_f32(B_ptr + ldb*4 + kidx);
                const float32x4_t B_val5 = vld1q_dup_f32(B_ptr + ldb*5 + kidx);
                const float32x4_t B_val6 = vld1q_dup_f32(B_ptr + ldb*6 + kidx);
                const float32x4_t B_val7 = vld1q_dup_f32(B_ptr + ldb*7 + kidx); 
                o00 = vfmaq_f32(o00, A_col0, B_val0);
                o01 = vfmaq_f32(o01, A_col1, B_val0);
                o10 = vfmaq_f32(o10, A_col0, B_val1);
                o11 = vfmaq_f32(o11, A_col1, B_val1);
                o20 = vfmaq_f32(o20, A_col0, B_val2);
                o21 = vfmaq_f32(o21, A_col1, B_val2);
                o30 = vfmaq_f32(o30, A_col0, B_val3);
                o31 = vfmaq_f32(o31, A_col1, B_val3);
                o40 = vfmaq_f32(o40, A_col0, B_val4);
                o41 = vfmaq_f32(o41, A_col1, B_val4);
                o50 = vfmaq_f32(o50, A_col0, B_val5);
                o51 = vfmaq_f32(o51, A_col1, B_val5);
                o60 = vfmaq_f32(o60, A_col0, B_val6);
                o61 = vfmaq_f32(o61, A_col1, B_val6);
                o70 = vfmaq_f32(o70, A_col0, B_val7);
                o71 = vfmaq_f32(o71, A_col1, B_val7);
            }
            vst1q_f32  (C_ptr + 0  * ldc, o00);
            vst1q_f32  (C_ptr + 1  * ldc, o10);
            vst1q_f32  (C_ptr + 2  * ldc, o20);
            vst1q_f32  (C_ptr + 3  * ldc, o30);
            vst1q_f32  (C_ptr + 4  * ldc, o40);
            vst1q_f32  (C_ptr + 5  * ldc, o50);
            vst1q_f32  (C_ptr + 6  * ldc, o60);
            vst1q_f32  (C_ptr + 7  * ldc, o70);
            vst1q_f32  (C_ptr + 0  * ldc + 4, o01);
            vst1q_f32  (C_ptr + 1  * ldc + 4, o11);
            vst1q_f32  (C_ptr + 2  * ldc + 4, o21);
            vst1q_f32  (C_ptr + 3  * ldc + 4, o31);
            vst1q_f32  (C_ptr + 4  * ldc + 4, o41);
            vst1q_f32  (C_ptr + 5  * ldc + 4, o51);
            vst1q_f32  (C_ptr + 6  * ldc + 4, o61);
            vst1q_f32  (C_ptr + 7  * ldc + 4, o71);
        }
        else if (rem_n == 11)
        {
            float32x4_t o00, o10, o20, o30, o40, o50, o60, o70, o80, o90, o100;
            float32x4_t o01, o11, o21, o31, o41, o51, o61, o71, o81, o91, o101;
            o00 = vld1q_f32     (C_ptr + 0  * ldc);
            o10 = vld1q_f32     (C_ptr + 1  * ldc);
            o20 = vld1q_f32     (C_ptr + 2  * ldc);
            o30 = vld1q_f32     (C_ptr + 3  * ldc);
            o40 = vld1q_f32     (C_ptr + 4  * ldc);
            o50 = vld1q_f32     (C_ptr + 5  * ldc);
            o60 = vld1q_f32     (C_ptr + 6  * ldc);
            o70 = vld1q_f32     (C_ptr + 7  * ldc);
            o80 = vld1q_f32     (C_ptr + 8  * ldc);
            o90 = vld1q_f32     (C_ptr + 9  * ldc);
            o100 = vld1q_f32    (C_ptr + 10 * ldc);
            o01 = vld1q_f32     (C_ptr + 0  * ldc + 4);
            o11 = vld1q_f32     (C_ptr + 1  * ldc + 4);
            o21 = vld1q_f32     (C_ptr + 2  * ldc + 4);
            o31 = vld1q_f32     (C_ptr + 3  * ldc + 4);
            o41 = vld1q_f32     (C_ptr + 4  * ldc + 4);
            o51 = vld1q_f32     (C_ptr + 5  * ldc + 4);
            o61 = vld1q_f32     (C_ptr + 6  * ldc + 4);
            o71 = vld1q_f32     (C_ptr + 7  * ldc + 4);
            o81 = vld1q_f32     (C_ptr + 8  * ldc + 4);
            o91 = vld1q_f32     (C_ptr + 9  * ldc + 4);
            o101 = vld1q_f32    (C_ptr + 10 * ldc + 4);
            for (int kidx = 0; kidx < K; kidx++)
            {
                const float32x4_t A_col0 = vld1q_f32(A_ptr);
                const float32x4_t A_col1 = vld1q_f32(A_ptr + 4);
                A_ptr += 8;
                const float32x4_t B_val0 = vld1q_dup_f32(B_ptr + ldb*0 + kidx);
                const float32x4_t B_val1 = vld1q_dup_f32(B_ptr + ldb*1 + kidx);
                const float32x4_t B_val2 = vld1q_dup_f32(B_ptr + ldb*2 + kidx);
                const float32x4_t B_val3 = vld1q_dup_f32(B_ptr + ldb*3 + kidx);
                const float32x4_t B_val4 = vld1q_dup_f32(B_ptr + ldb*4 + kidx);
                const float32x4_t B_val5 = vld1q_dup_f32(B_ptr + ldb*5 + kidx);
                o00 = vfmaq_f32(o00, A_col0, B_val0);
                o01 = vfmaq_f32(o01, A_col1, B_val0);
                o10 = vfmaq_f32(o10, A_col0, B_val1);
                o11 = vfmaq_f32(o11, A_col1, B_val1);
                o20 = vfmaq_f32(o20, A_col0, B_val2);
                o21 = vfmaq_f32(o21, A_col1, B_val2);
                o30 = vfmaq_f32(o30, A_col0, B_val3);
                o31 = vfmaq_f32(o31, A_col1, B_val3);
                o40 = vfmaq_f32(o40, A_col0, B_val4);
                o41 = vfmaq_f32(o41, A_col1, B_val4);
                o50 = vfmaq_f32(o50, A_col0, B_val5);
                o51 = vfmaq_f32(o51, A_col1, B_val5);
                const float32x4_t B_val6 = vld1q_dup_f32(B_ptr + ldb*6 + kidx);
                const float32x4_t B_val7 = vld1q_dup_f32(B_ptr + ldb*7 + kidx);
                const float32x4_t B_val8 = vld1q_dup_f32(B_ptr + ldb*8 + kidx);
                const float32x4_t B_val9 = vld1q_dup_f32(B_ptr + ldb*9 + kidx);
                const float32x4_t B_val10 = vld1q_dup_f32(B_ptr + ldb*10 + kidx);
                o60 = vfmaq_f32(o60, A_col0, B_val6);
                o61 = vfmaq_f32(o61, A_col1, B_val6);
                o70 = vfmaq_f32(o70, A_col0, B_val7);
                o71 = vfmaq_f32(o71, A_col1, B_val7);
                o80 = vfmaq_f32(o80, A_col0, B_val8);
                o81 = vfmaq_f32(o81, A_col1, B_val8);
                o90 = vfmaq_f32(o90, A_col0, B_val9);
                o91 = vfmaq_f32(o91, A_col1, B_val9);
                o100 = vfmaq_f32(o100, A_col0, B_val10);
                o101 = vfmaq_f32(o101, A_col1, B_val10);
            }
            vst1q_f32  (C_ptr + 0  * ldc, o00);
            vst1q_f32  (C_ptr + 1  * ldc, o10);
            vst1q_f32  (C_ptr + 2  * ldc, o20);
            vst1q_f32  (C_ptr + 3  * ldc, o30);
            vst1q_f32  (C_ptr + 4  * ldc, o40);
            vst1q_f32  (C_ptr + 5  * ldc, o50);
            vst1q_f32  (C_ptr + 6  * ldc, o60);
            vst1q_f32  (C_ptr + 7  * ldc, o70);
            vst1q_f32  (C_ptr + 8  * ldc, o80);
            vst1q_f32  (C_ptr + 9  * ldc, o90);
            vst1q_f32  (C_ptr + 10 * ldc, o100);
            vst1q_f32  (C_ptr + 0  * ldc + 4, o01);
            vst1q_f32  (C_ptr + 1  * ldc + 4, o11);
            vst1q_f32  (C_ptr + 2  * ldc + 4, o21);
            vst1q_f32  (C_ptr + 3  * ldc + 4, o31);
            vst1q_f32  (C_ptr + 4  * ldc + 4, o41);
            vst1q_f32  (C_ptr + 5  * ldc + 4, o51);
            vst1q_f32  (C_ptr + 6  * ldc + 4, o61);
            vst1q_f32  (C_ptr + 7  * ldc + 4, o71);
            vst1q_f32  (C_ptr + 8  * ldc + 4, o81);
            vst1q_f32  (C_ptr + 9  * ldc + 4, o91);
            vst1q_f32  (C_ptr + 10 * ldc + 4, o101);
        }
        else if (rem_n == 10)
        {
            float32x4_t o00, o10, o20, o30, o40, o50, o60, o70, o80, o90;
            float32x4_t o01, o11, o21, o31, o41, o51, o61, o71, o81, o91;
            o00 = vld1q_f32     (C_ptr + 0  * ldc);
            o10 = vld1q_f32     (C_ptr + 1  * ldc);
            o20 = vld1q_f32     (C_ptr + 2  * ldc);
            o30 = vld1q_f32     (C_ptr + 3  * ldc);
            o40 = vld1q_f32     (C_ptr + 4  * ldc);
            o50 = vld1q_f32     (C_ptr + 5  * ldc);
            o60 = vld1q_f32     (C_ptr + 6  * ldc);
            o70 = vld1q_f32     (C_ptr + 7  * ldc);
            o80 = vld1q_f32     (C_ptr + 8  * ldc);
            o90 = vld1q_f32     (C_ptr + 9  * ldc);
            o01 = vld1q_f32     (C_ptr + 0  * ldc + 4);
            o11 = vld1q_f32     (C_ptr + 1  * ldc + 4);
            o21 = vld1q_f32     (C_ptr + 2  * ldc + 4);
            o31 = vld1q_f32     (C_ptr + 3  * ldc + 4);
            o41 = vld1q_f32     (C_ptr + 4  * ldc + 4);
            o51 = vld1q_f32     (C_ptr + 5  * ldc + 4);
            o61 = vld1q_f32     (C_ptr + 6  * ldc + 4);
            o71 = vld1q_f32     (C_ptr + 7  * ldc + 4);
            o81 = vld1q_f32     (C_ptr + 8  * ldc + 4);
            o91 = vld1q_f32     (C_ptr + 9  * ldc + 4);
            for (int kidx = 0; kidx < K; kidx++)
            {
                const float32x4_t A_col0 = vld1q_f32(A_ptr);
                const float32x4_t A_col1 = vld1q_f32(A_ptr + 4);
                A_ptr += 8;
                const float32x4_t B_val0 = vld1q_dup_f32(B_ptr + ldb*0 + kidx);
                const float32x4_t B_val1 = vld1q_dup_f32(B_ptr + ldb*1 + kidx);
                const float32x4_t B_val2 = vld1q_dup_f32(B_ptr + ldb*2 + kidx);
                const float32x4_t B_val3 = vld1q_dup_f32(B_ptr + ldb*3 + kidx);
                const float32x4_t B_val4 = vld1q_dup_f32(B_ptr + ldb*4 + kidx);
                o00 = vfmaq_f32(o00, A_col0, B_val0);
                o01 = vfmaq_f32(o01, A_col1, B_val0);
                o10 = vfmaq_f32(o10, A_col0, B_val1);
                o11 = vfmaq_f32(o11, A_col1, B_val1);
                o20 = vfmaq_f32(o20, A_col0, B_val2);
                o21 = vfmaq_f32(o21, A_col1, B_val2);
                o30 = vfmaq_f32(o30, A_col0, B_val3);
                o31 = vfmaq_f32(o31, A_col1, B_val3);
                o40 = vfmaq_f32(o40, A_col0, B_val4);
                o41 = vfmaq_f32(o41, A_col1, B_val4);
                const float32x4_t B_val5 = vld1q_dup_f32(B_ptr + ldb*5 + kidx);
                const float32x4_t B_val6 = vld1q_dup_f32(B_ptr + ldb*6 + kidx);
                const float32x4_t B_val7 = vld1q_dup_f32(B_ptr + ldb*7 + kidx);
                const float32x4_t B_val8 = vld1q_dup_f32(B_ptr + ldb*8 + kidx);
                const float32x4_t B_val9 = vld1q_dup_f32(B_ptr + ldb*9 + kidx);
                o50 = vfmaq_f32(o50, A_col0, B_val5);
                o51 = vfmaq_f32(o51, A_col1, B_val5);
                o60 = vfmaq_f32(o60, A_col0, B_val6);
                o61 = vfmaq_f32(o61, A_col1, B_val6);
                o70 = vfmaq_f32(o70, A_col0, B_val7);
                o71 = vfmaq_f32(o71, A_col1, B_val7);
                o80 = vfmaq_f32(o80, A_col0, B_val8);
                o81 = vfmaq_f32(o81, A_col1, B_val8);
                o90 = vfmaq_f32(o90, A_col0, B_val9);
                o91 = vfmaq_f32(o91, A_col1, B_val9);
            }
            vst1q_f32  (C_ptr + 0  * ldc, o00);
            vst1q_f32  (C_ptr + 1  * ldc, o10);
            vst1q_f32  (C_ptr + 2  * ldc, o20);
            vst1q_f32  (C_ptr + 3  * ldc, o30);
            vst1q_f32  (C_ptr + 4  * ldc, o40);
            vst1q_f32  (C_ptr + 5  * ldc, o50);
            vst1q_f32  (C_ptr + 6  * ldc, o60);
            vst1q_f32  (C_ptr + 7  * ldc, o70);
            vst1q_f32  (C_ptr + 8  * ldc, o80);
            vst1q_f32  (C_ptr + 9  * ldc, o90);
            vst1q_f32  (C_ptr + 0  * ldc + 4, o01);
            vst1q_f32  (C_ptr + 1  * ldc + 4, o11);
            vst1q_f32  (C_ptr + 2  * ldc + 4, o21);
            vst1q_f32  (C_ptr + 3  * ldc + 4, o31);
            vst1q_f32  (C_ptr + 4  * ldc + 4, o41);
            vst1q_f32  (C_ptr + 5  * ldc + 4, o51);
            vst1q_f32  (C_ptr + 6  * ldc + 4, o61);
            vst1q_f32  (C_ptr + 7  * ldc + 4, o71);
            vst1q_f32  (C_ptr + 8  * ldc + 4, o81);
            vst1q_f32  (C_ptr + 9  * ldc + 4, o91);
        }
        else if (rem_n == 9)
        {
            float32x4_t o00, o10, o20, o30, o40, o50, o60, o70, o80;
            float32x4_t o01, o11, o21, o31, o41, o51, o61, o71, o81;
            o00 = vld1q_f32     (C_ptr + 0  * ldc);
            o10 = vld1q_f32     (C_ptr + 1  * ldc);
            o20 = vld1q_f32     (C_ptr + 2  * ldc);
            o30 = vld1q_f32     (C_ptr + 3  * ldc);
            o40 = vld1q_f32     (C_ptr + 4  * ldc);
            o50 = vld1q_f32     (C_ptr + 5  * ldc);
            o60 = vld1q_f32     (C_ptr + 6  * ldc);
            o70 = vld1q_f32     (C_ptr + 7  * ldc);
            o80 = vld1q_f32     (C_ptr + 8  * ldc);
            o01 = vld1q_f32     (C_ptr + 0  * ldc + 4);
            o11 = vld1q_f32     (C_ptr + 1  * ldc + 4);
            o21 = vld1q_f32     (C_ptr + 2  * ldc + 4);
            o31 = vld1q_f32     (C_ptr + 3  * ldc + 4);
            o41 = vld1q_f32     (C_ptr + 4  * ldc + 4);
            o51 = vld1q_f32     (C_ptr + 5  * ldc + 4);
            o61 = vld1q_f32     (C_ptr + 6  * ldc + 4);
            o71 = vld1q_f32     (C_ptr + 7  * ldc + 4);
            o81 = vld1q_f32     (C_ptr + 8  * ldc + 4);
            for (int kidx = 0; kidx < K; kidx++)
            {
                const float32x4_t A_col0 = vld1q_f32(A_ptr);
                const float32x4_t A_col1 = vld1q_f32(A_ptr + 4);
                A_ptr += 8;
                const float32x4_t B_val0 = vld1q_dup_f32(B_ptr + ldb*0 + kidx);
                const float32x4_t B_val1 = vld1q_dup_f32(B_ptr + ldb*1 + kidx);
                const float32x4_t B_val2 = vld1q_dup_f32(B_ptr + ldb*2 + kidx);
                const float32x4_t B_val3 = vld1q_dup_f32(B_ptr + ldb*3 + kidx);
                const float32x4_t B_val4 = vld1q_dup_f32(B_ptr + ldb*4 + kidx);
                const float32x4_t B_val5 = vld1q_dup_f32(B_ptr + ldb*5 + kidx);
                const float32x4_t B_val6 = vld1q_dup_f32(B_ptr + ldb*6 + kidx);
                const float32x4_t B_val7 = vld1q_dup_f32(B_ptr + ldb*7 + kidx);
                const float32x4_t B_val8 = vld1q_dup_f32(B_ptr + ldb*8 + kidx);
                o00 = vfmaq_f32(o00, A_col0, B_val0);
                o01 = vfmaq_f32(o01, A_col1, B_val0);
                o10 = vfmaq_f32(o10, A_col0, B_val1);
                o11 = vfmaq_f32(o11, A_col1, B_val1);
                o20 = vfmaq_f32(o20, A_col0, B_val2);
                o21 = vfmaq_f32(o21, A_col1, B_val2);
                o30 = vfmaq_f32(o30, A_col0, B_val3);
                o31 = vfmaq_f32(o31, A_col1, B_val3);
                o40 = vfmaq_f32(o40, A_col0, B_val4);
                o41 = vfmaq_f32(o41, A_col1, B_val4);
                o50 = vfmaq_f32(o50, A_col0, B_val5);
                o51 = vfmaq_f32(o51, A_col1, B_val5);
                o60 = vfmaq_f32(o60, A_col0, B_val6);
                o61 = vfmaq_f32(o61, A_col1, B_val6);
                o70 = vfmaq_f32(o70, A_col0, B_val7);
                o71 = vfmaq_f32(o71, A_col1, B_val7);
                o80 = vfmaq_f32(o80, A_col0, B_val8);
                o81 = vfmaq_f32(o81, A_col1, B_val8);
            }
            vst1q_f32  (C_ptr + 0  * ldc, o00);
            vst1q_f32  (C_ptr + 1  * ldc, o10);
            vst1q_f32  (C_ptr + 2  * ldc, o20);
            vst1q_f32  (C_ptr + 3  * ldc, o30);
            vst1q_f32  (C_ptr + 4  * ldc, o40);
            vst1q_f32  (C_ptr + 5  * ldc, o50);
            vst1q_f32  (C_ptr + 6  * ldc, o60);
            vst1q_f32  (C_ptr + 7  * ldc, o70);
            vst1q_f32  (C_ptr + 8  * ldc, o80);
            vst1q_f32  (C_ptr + 0  * ldc + 4, o01);
            vst1q_f32  (C_ptr + 1  * ldc + 4, o11);
            vst1q_f32  (C_ptr + 2  * ldc + 4, o21);
            vst1q_f32  (C_ptr + 3  * ldc + 4, o31);
            vst1q_f32  (C_ptr + 4  * ldc + 4, o41);
            vst1q_f32  (C_ptr + 5  * ldc + 4, o51);
            vst1q_f32  (C_ptr + 6  * ldc + 4, o61);
            vst1q_f32  (C_ptr + 7  * ldc + 4, o71);
            vst1q_f32  (C_ptr + 8  * ldc + 4, o81);
        }
        else if (rem_n == 7)
        {
            float32x4_t o00, o10, o20, o30, o40, o50, o60;
            float32x4_t o01, o11, o21, o31, o41, o51, o61;
            o00 = vld1q_f32     (C_ptr + 0  * ldc);
            o10 = vld1q_f32     (C_ptr + 1  * ldc);
            o20 = vld1q_f32     (C_ptr + 2  * ldc);
            o30 = vld1q_f32     (C_ptr + 3  * ldc);
            o40 = vld1q_f32     (C_ptr + 4  * ldc);
            o50 = vld1q_f32     (C_ptr + 5  * ldc);
            o60 = vld1q_f32     (C_ptr + 6  * ldc);
            o01 = vld1q_f32     (C_ptr + 0  * ldc + 4);
            o11 = vld1q_f32     (C_ptr + 1  * ldc + 4);
            o21 = vld1q_f32     (C_ptr + 2  * ldc + 4);
            o31 = vld1q_f32     (C_ptr + 3  * ldc + 4);
            o41 = vld1q_f32     (C_ptr + 4  * ldc + 4);
            o51 = vld1q_f32     (C_ptr + 5  * ldc + 4);
            o61 = vld1q_f32     (C_ptr + 6  * ldc + 4);
            for (int kidx = 0; kidx < K; kidx++)
            {
                const float32x4_t A_col0 = vld1q_f32(A_ptr);
                const float32x4_t A_col1 = vld1q_f32(A_ptr + 4);
                A_ptr += 8;
                const float32x4_t B_val0 = vld1q_dup_f32(B_ptr + ldb*0 + kidx);
                const float32x4_t B_val1 = vld1q_dup_f32(B_ptr + ldb*1 + kidx);
                const float32x4_t B_val2 = vld1q_dup_f32(B_ptr + ldb*2 + kidx);
                const float32x4_t B_val3 = vld1q_dup_f32(B_ptr + ldb*3 + kidx);
                const float32x4_t B_val4 = vld1q_dup_f32(B_ptr + ldb*4 + kidx);
                const float32x4_t B_val5 = vld1q_dup_f32(B_ptr + ldb*5 + kidx);
                const float32x4_t B_val6 = vld1q_dup_f32(B_ptr + ldb*6 + kidx);
                o00 = vfmaq_f32(o00, A_col0, B_val0);
                o01 = vfmaq_f32(o01, A_col1, B_val0);
                o10 = vfmaq_f32(o10, A_col0, B_val1);
                o11 = vfmaq_f32(o11, A_col1, B_val1);
                o20 = vfmaq_f32(o20, A_col0, B_val2);
                o21 = vfmaq_f32(o21, A_col1, B_val2);
                o30 = vfmaq_f32(o30, A_col0, B_val3);
                o31 = vfmaq_f32(o31, A_col1, B_val3);
                o40 = vfmaq_f32(o40, A_col0, B_val4);
                o41 = vfmaq_f32(o41, A_col1, B_val4);
                o50 = vfmaq_f32(o50, A_col0, B_val5);
                o51 = vfmaq_f32(o51, A_col1, B_val5);
                o60 = vfmaq_f32(o60, A_col0, B_val6);
                o61 = vfmaq_f32(o61, A_col1, B_val6);
            }
            vst1q_f32  (C_ptr + 0  * ldc, o00);
            vst1q_f32  (C_ptr + 1  * ldc, o10);
            vst1q_f32  (C_ptr + 2  * ldc, o20);
            vst1q_f32  (C_ptr + 3  * ldc, o30);
            vst1q_f32  (C_ptr + 4  * ldc, o40);
            vst1q_f32  (C_ptr + 5  * ldc, o50);
            vst1q_f32  (C_ptr + 6  * ldc, o60);
            vst1q_f32  (C_ptr + 0  * ldc + 4, o01);
            vst1q_f32  (C_ptr + 1  * ldc + 4, o11);
            vst1q_f32  (C_ptr + 2  * ldc + 4, o21);
            vst1q_f32  (C_ptr + 3  * ldc + 4, o31);
            vst1q_f32  (C_ptr + 4  * ldc + 4, o41);
            vst1q_f32  (C_ptr + 5  * ldc + 4, o51);
            vst1q_f32  (C_ptr + 6  * ldc + 4, o61);
        }
        else if (rem_n == 6)
        {
            float32x4_t o00, o10, o20, o30, o40, o50;
            float32x4_t o01, o11, o21, o31, o41, o51;
            o00 = vld1q_f32     (C_ptr + 0  * ldc);
            o10 = vld1q_f32     (C_ptr + 1  * ldc);
            o20 = vld1q_f32     (C_ptr + 2  * ldc);
            o30 = vld1q_f32     (C_ptr + 3  * ldc);
            o40 = vld1q_f32     (C_ptr + 4  * ldc);
            o50 = vld1q_f32     (C_ptr + 5  * ldc);
            o01 = vld1q_f32     (C_ptr + 0  * ldc + 4);
            o11 = vld1q_f32     (C_ptr + 1  * ldc + 4);
            o21 = vld1q_f32     (C_ptr + 2  * ldc + 4);
            o31 = vld1q_f32     (C_ptr + 3  * ldc + 4);
            o41 = vld1q_f32     (C_ptr + 4  * ldc + 4);
            o51 = vld1q_f32     (C_ptr + 5  * ldc + 4);
            for (int kidx = 0; kidx < K; kidx++)
            {
                const float32x4_t A_col0 = vld1q_f32(A_ptr);
                const float32x4_t A_col1 = vld1q_f32(A_ptr + 4);
                A_ptr += 8;
                const float32x4_t B_val0 = vld1q_dup_f32(B_ptr + ldb*0 + kidx);
                const float32x4_t B_val1 = vld1q_dup_f32(B_ptr + ldb*1 + kidx);
                const float32x4_t B_val2 = vld1q_dup_f32(B_ptr + ldb*2 + kidx);
                const float32x4_t B_val3 = vld1q_dup_f32(B_ptr + ldb*3 + kidx);
                const float32x4_t B_val4 = vld1q_dup_f32(B_ptr + ldb*4 + kidx);
                const float32x4_t B_val5 = vld1q_dup_f32(B_ptr + ldb*5 + kidx);
                o00 = vfmaq_f32(o00, A_col0, B_val0);
                o01 = vfmaq_f32(o01, A_col1, B_val0);
                o10 = vfmaq_f32(o10, A_col0, B_val1);
                o11 = vfmaq_f32(o11, A_col1, B_val1);
                o20 = vfmaq_f32(o20, A_col0, B_val2);
                o21 = vfmaq_f32(o21, A_col1, B_val2);
                o30 = vfmaq_f32(o30, A_col0, B_val3);
                o31 = vfmaq_f32(o31, A_col1, B_val3);
                o40 = vfmaq_f32(o40, A_col0, B_val4);
                o41 = vfmaq_f32(o41, A_col1, B_val4);
                o50 = vfmaq_f32(o50, A_col0, B_val5);
                o51 = vfmaq_f32(o51, A_col1, B_val5);
            }
            vst1q_f32  (C_ptr + 0  * ldc, o00);
            vst1q_f32  (C_ptr + 1  * ldc, o10);
            vst1q_f32  (C_ptr + 2  * ldc, o20);
            vst1q_f32  (C_ptr + 3  * ldc, o30);
            vst1q_f32  (C_ptr + 4  * ldc, o40);
            vst1q_f32  (C_ptr + 5  * ldc, o50);
            vst1q_f32  (C_ptr + 0  * ldc + 4, o01);
            vst1q_f32  (C_ptr + 1  * ldc + 4, o11);
            vst1q_f32  (C_ptr + 2  * ldc + 4, o21);
            vst1q_f32  (C_ptr + 3  * ldc + 4, o31);
            vst1q_f32  (C_ptr + 4  * ldc + 4, o41);
            vst1q_f32  (C_ptr + 5  * ldc + 4, o51);
        }
        else if (rem_n == 5)
        {
            float32x4_t o00, o10, o20, o30, o40;
            float32x4_t o01, o11, o21, o31, o41;
            o00 = vld1q_f32     (C_ptr + 0  * ldc);
            o10 = vld1q_f32     (C_ptr + 1  * ldc);
            o20 = vld1q_f32     (C_ptr + 2  * ldc);
            o30 = vld1q_f32     (C_ptr + 3  * ldc);
            o40 = vld1q_f32     (C_ptr + 4  * ldc);
            o01 = vld1q_f32     (C_ptr + 0  * ldc + 4);
            o11 = vld1q_f32     (C_ptr + 1  * ldc + 4);
            o21 = vld1q_f32     (C_ptr + 2  * ldc + 4);
            o31 = vld1q_f32     (C_ptr + 3  * ldc + 4);
            o41 = vld1q_f32     (C_ptr + 4  * ldc + 4);
            for (int kidx = 0; kidx < K; kidx++)
            {
                const float32x4_t A_col0 = vld1q_f32(A_ptr);
                const float32x4_t A_col1 = vld1q_f32(A_ptr + 4);
                A_ptr += 8;
                const float32x4_t B_val0 = vld1q_dup_f32(B_ptr + ldb*0 + kidx);
                const float32x4_t B_val1 = vld1q_dup_f32(B_ptr + ldb*1 + kidx);
                const float32x4_t B_val2 = vld1q_dup_f32(B_ptr + ldb*2 + kidx);
                const float32x4_t B_val3 = vld1q_dup_f32(B_ptr + ldb*3 + kidx);  
                const float32x4_t B_val4 = vld1q_dup_f32(B_ptr + ldb*4 + kidx);
                o00 = vfmaq_f32(o00, A_col0, B_val0);
                o01 = vfmaq_f32(o01, A_col1, B_val0);
                o10 = vfmaq_f32(o10, A_col0, B_val1);
                o11 = vfmaq_f32(o11, A_col1, B_val1);
                o20 = vfmaq_f32(o20, A_col0, B_val2);
                o21 = vfmaq_f32(o21, A_col1, B_val2);
                o30 = vfmaq_f32(o30, A_col0, B_val3);
                o31 = vfmaq_f32(o31, A_col1, B_val3);
                o40 = vfmaq_f32(o40, A_col0, B_val4);
                o41 = vfmaq_f32(o41, A_col1, B_val4);
            }
            vst1q_f32  (C_ptr + 0  * ldc, o00);
            vst1q_f32  (C_ptr + 1  * ldc, o10);
            vst1q_f32  (C_ptr + 2  * ldc, o20);
            vst1q_f32  (C_ptr + 3  * ldc, o30);
            vst1q_f32  (C_ptr + 4  * ldc, o40);
            vst1q_f32  (C_ptr + 0  * ldc + 4, o01);
            vst1q_f32  (C_ptr + 1  * ldc + 4, o11);
            vst1q_f32  (C_ptr + 2  * ldc + 4, o21);
            vst1q_f32  (C_ptr + 3  * ldc + 4, o31);
            vst1q_f32  (C_ptr + 4  * ldc + 4, o41);
        }
        else if (rem_n == 4)
        {
            float32x4_t o00, o10, o20, o30;
            float32x4_t o01, o11, o21, o31;
            o00 = vld1q_f32     (C_ptr + 0  * ldc);
            o10 = vld1q_f32     (C_ptr + 1  * ldc);
            o20 = vld1q_f32     (C_ptr + 2  * ldc);
            o30 = vld1q_f32     (C_ptr + 3  * ldc);
            o01 = vld1q_f32     (C_ptr + 0  * ldc + 4);
            o11 = vld1q_f32     (C_ptr + 1  * ldc + 4);
            o21 = vld1q_f32     (C_ptr + 2  * ldc + 4);
            o31 = vld1q_f32     (C_ptr + 3  * ldc + 4);
            for (int kidx = 0; kidx < K; kidx++)
            {
                const float32x4_t A_col0 = vld1q_f32(A_ptr);
                const float32x4_t A_col1 = vld1q_f32(A_ptr + 4);
                A_ptr += 8;
                const float32x4_t B_val0 = vld1q_dup_f32(B_ptr + ldb*0 + kidx);
                const float32x4_t B_val1 = vld1q_dup_f32(B_ptr + ldb*1 + kidx);
                const float32x4_t B_val2 = vld1q_dup_f32(B_ptr + ldb*2 + kidx);
                const float32x4_t B_val3 = vld1q_dup_f32(B_ptr + ldb*3 + kidx);  
                o00 = vfmaq_f32(o00, A_col0, B_val0);
                o01 = vfmaq_f32(o01, A_col1, B_val0);
                o10 = vfmaq_f32(o10, A_col0, B_val1);
                o11 = vfmaq_f32(o11, A_col1, B_val1);
                o20 = vfmaq_f32(o20, A_col0, B_val2);
                o21 = vfmaq_f32(o21, A_col1, B_val2);
                o30 = vfmaq_f32(o30, A_col0, B_val3);
                o31 = vfmaq_f32(o31, A_col1, B_val3);
            }
            vst1q_f32  (C_ptr + 0  * ldc, o00);
            vst1q_f32  (C_ptr + 1  * ldc, o10);
            vst1q_f32  (C_ptr + 2  * ldc, o20);
            vst1q_f32  (C_ptr + 3  * ldc, o30);
            vst1q_f32  (C_ptr + 0  * ldc + 4, o01);
            vst1q_f32  (C_ptr + 1  * ldc + 4, o11);
            vst1q_f32  (C_ptr + 2  * ldc + 4, o21);
            vst1q_f32  (C_ptr + 3  * ldc + 4, o31);
        }
        else if (rem_n == 3)
        {
            float32x4_t o00, o10, o20;
            float32x4_t o01, o11, o21;
            o00 = vld1q_f32     (C_ptr + 0  * ldc);
            o10 = vld1q_f32     (C_ptr + 1  * ldc);
            o20 = vld1q_f32     (C_ptr + 2  * ldc);
            o01 = vld1q_f32     (C_ptr + 0  * ldc + 4);
            o11 = vld1q_f32     (C_ptr + 1  * ldc + 4);
            o21 = vld1q_f32     (C_ptr + 2  * ldc + 4);
            for (int kidx = 0; kidx < K; kidx++)
            {
                const float32x4_t A_col0 = vld1q_f32(A_ptr);
                const float32x4_t A_col1 = vld1q_f32(A_ptr + 4);
                A_ptr += 8;
                const float32x4_t B_val0 = vld1q_dup_f32(B_ptr + ldb*0 + kidx);
                const float32x4_t B_val1 = vld1q_dup_f32(B_ptr + ldb*1 + kidx);
                const float32x4_t B_val2 = vld1q_dup_f32(B_ptr + ldb*2 + kidx);
                o00 = vfmaq_f32(o00, A_col0, B_val0);
                o01 = vfmaq_f32(o01, A_col1, B_val0);
                o10 = vfmaq_f32(o10, A_col0, B_val1);
                o11 = vfmaq_f32(o11, A_col1, B_val1);
                o20 = vfmaq_f32(o20, A_col0, B_val2);
                o21 = vfmaq_f32(o21, A_col1, B_val2);
            }
            vst1q_f32  (C_ptr + 0  * ldc, o00);
            vst1q_f32  (C_ptr + 1  * ldc, o10);
            vst1q_f32  (C_ptr + 2  * ldc, o20);
            vst1q_f32  (C_ptr + 0  * ldc + 4, o01);
            vst1q_f32  (C_ptr + 1  * ldc + 4, o11);
            vst1q_f32  (C_ptr + 2  * ldc + 4, o21);
        }
        else if (rem_n == 2)
        {
            float32x4_t o00, o10;
            float32x4_t o01, o11;
            o00 = vld1q_f32     (C_ptr + 0  * ldc);
            o10 = vld1q_f32     (C_ptr + 1  * ldc);
            o01 = vld1q_f32     (C_ptr + 0  * ldc + 4);
            o11 = vld1q_f32     (C_ptr + 1  * ldc + 4);
            for (int kidx = 0; kidx < K; kidx++)
            {
                const float32x4_t A_col0 = vld1q_f32(A_ptr);
                const float32x4_t A_col1 = vld1q_f32(A_ptr + 4);
                A_ptr += 8;
                const float32x4_t B_val0 = vld1q_dup_f32(B_ptr + ldb*0 + kidx);
                const float32x4_t B_val1 = vld1q_dup_f32(B_ptr + ldb*1 + kidx);
                o00 = vfmaq_f32(o00, A_col0, B_val0);
                o01 = vfmaq_f32(o01, A_col1, B_val0);
                o10 = vfmaq_f32(o10, A_col0, B_val1);
                o11 = vfmaq_f32(o11, A_col1, B_val1);
            }
            vst1q_f32  (C_ptr + 0  * ldc, o00);
            vst1q_f32  (C_ptr + 1  * ldc, o10);
            vst1q_f32  (C_ptr + 0  * ldc + 4, o01);
            vst1q_f32  (C_ptr + 1  * ldc + 4, o11);
        }
        else if (rem_n == 1)
        {
            float32x4_t o00;
            float32x4_t o01;
            o00 = vld1q_f32     (C_ptr + 0  * ldc);
            o01 = vld1q_f32     (C_ptr + 0  * ldc + 4);
            for (int kidx = 0; kidx < K; kidx++)
            {
                const float32x4_t A_col0 = vld1q_f32(A_ptr);
                const float32x4_t A_col1 = vld1q_f32(A_ptr + 4);
                A_ptr += 8;
                const float32x4_t B_val0 = vld1q_dup_f32(B_ptr + ldb*0 + kidx);
                o00 = vfmaq_f32(o00, A_col0, B_val0);
                o01 = vfmaq_f32(o01, A_col1, B_val0);
            }
            vst1q_f32  (C_ptr + 0  * ldc, o00);
            vst1q_f32  (C_ptr + 0  * ldc + 4, o01);
        }
        #else
        for (unsigned int n = N - (N%_VEC_SIZE_N); n < N; n++)
        {
            for (unsigned int mm = m; mm < m + _VEC_SIZE_M; mm++)
            {
                float c = C[n * ldc + mm];
                for (unsigned int k = 0; k < K; k++)
                {
                    c += A[((mm/_VEC_SIZE_M) * lda + k) * _VEC_SIZE_M + (mm%_VEC_SIZE_M)] * B[n * ldb + k];
                }
                C[n * ldc + mm] = c;
            }
        }
        #endif
    }
    
    for (unsigned int m = M - rem_m; m < M; m++)
    {
        for (unsigned int n = 0; n < N - rem_n; n += _VEC_SIZE_N)
        {
            for (unsigned int nn = n; nn < n + _VEC_SIZE_N; nn++)
            {
                float c = C[nn * ldc + m];
                for (unsigned int k = 0; k < K; k++)
                {
                    c += A[((m/_VEC_SIZE_M) * lda + k) * _VEC_SIZE_M + rem_m] * B[nn * ldb + k];
                }
                C[nn * ldc + m] = c;
            }
        }
        for (unsigned int n = N - rem_n; n < N; n++)
        {
            float c = C[n * ldc + m];
            for (unsigned int k = 0; k < K; k++)
            {
                c += A[((m/_VEC_SIZE_M) * lda + k) * _VEC_SIZE_M + rem_m] * B[n * ldb + k];
            }
            C[n * ldc + m] = c;
        }
    }
}   
void neon_sgemm_full_tile(const unsigned int M, const unsigned int N, const unsigned int K,
		 const float *A, const unsigned int lda, const float *B, const unsigned int ldb, float *C, const unsigned int ldc)
{
    for (unsigned int m = 0; m < _TILE_SIZE_M; m += _VEC_SIZE_M)
    {
        unsigned int n = 0;
        for (; n < _TILE_SIZE_N; n += _VEC_SIZE_N)
        {
            #if _VEC_SIZE_N == 12 && _VEC_SIZE_M == 8
            const float *A_ptr = A + m * lda;
            const float *B_ptr = B + n * ldb;
            float *C_ptr = C + n * ldc + m;
            float32x4_t o00, o10, o20, o30, o40, o50, o60, o70, o80, o90, o100, o110;
            float32x4_t o01, o11, o21, o31, o41, o51, o61, o71, o81, o91, o101, o111;
            o00 = vld1q_f32     (C_ptr + 0  * ldc);
            o10 = vld1q_f32     (C_ptr + 1  * ldc);
            o20 = vld1q_f32     (C_ptr + 2  * ldc);
            o30 = vld1q_f32     (C_ptr + 3  * ldc);
            o40 = vld1q_f32     (C_ptr + 4  * ldc);
            o50 = vld1q_f32     (C_ptr + 5  * ldc);
            o60 = vld1q_f32     (C_ptr + 6  * ldc);
            o70 = vld1q_f32     (C_ptr + 7  * ldc);
            o80 = vld1q_f32     (C_ptr + 8  * ldc);
            o90 = vld1q_f32     (C_ptr + 9  * ldc);
            o100 = vld1q_f32    (C_ptr + 10 * ldc);
            o110 = vld1q_f32    (C_ptr + 11 * ldc);
            o01 = vld1q_f32     (C_ptr + 0  * ldc + 4);
            o11 = vld1q_f32     (C_ptr + 1  * ldc + 4);
            o21 = vld1q_f32     (C_ptr + 2  * ldc + 4);
            o31 = vld1q_f32     (C_ptr + 3  * ldc + 4);
            o41 = vld1q_f32     (C_ptr + 4  * ldc + 4);
            o51 = vld1q_f32     (C_ptr + 5  * ldc + 4);
            o61 = vld1q_f32     (C_ptr + 6  * ldc + 4);
            o71 = vld1q_f32     (C_ptr + 7  * ldc + 4);
            o81 = vld1q_f32     (C_ptr + 8  * ldc + 4);
            o91 = vld1q_f32     (C_ptr + 9  * ldc + 4);
            o101 = vld1q_f32    (C_ptr + 10 * ldc + 4);
            o111 = vld1q_f32    (C_ptr + 11 * ldc + 4);
            for (int kidx = 0; kidx < K; kidx++)
            {
                const float32x4_t A_col0 = vld1q_f32(A_ptr);
                const float32x4_t A_col1 = vld1q_f32(A_ptr + 4);
                A_ptr += 8;
                const float32x4_t B_val0 = vld1q_dup_f32(B_ptr + ldb*0 + kidx);
                const float32x4_t B_val1 = vld1q_dup_f32(B_ptr + ldb*1 + kidx);
                const float32x4_t B_val2 = vld1q_dup_f32(B_ptr + ldb*2 + kidx);
                const float32x4_t B_val3 = vld1q_dup_f32(B_ptr + ldb*3 + kidx);
                const float32x4_t B_val4 = vld1q_dup_f32(B_ptr + ldb*4 + kidx);
                const float32x4_t B_val5 = vld1q_dup_f32(B_ptr + ldb*5 + kidx);
                o00 = vfmaq_f32(o00, A_col0, B_val0);
                o01 = vfmaq_f32(o01, A_col1, B_val0);
                o10 = vfmaq_f32(o10, A_col0, B_val1);
                o11 = vfmaq_f32(o11, A_col1, B_val1);
                o20 = vfmaq_f32(o20, A_col0, B_val2);
                o21 = vfmaq_f32(o21, A_col1, B_val2);
                o30 = vfmaq_f32(o30, A_col0, B_val3);
                o31 = vfmaq_f32(o31, A_col1, B_val3);
                o40 = vfmaq_f32(o40, A_col0, B_val4);
                o41 = vfmaq_f32(o41, A_col1, B_val4);
                o50 = vfmaq_f32(o50, A_col0, B_val5);
                o51 = vfmaq_f32(o51, A_col1, B_val5);
                const float32x4_t B_val6 = vld1q_dup_f32(B_ptr + ldb*6 + kidx);
                const float32x4_t B_val7 = vld1q_dup_f32(B_ptr + ldb*7 + kidx);
                const float32x4_t B_val8 = vld1q_dup_f32(B_ptr + ldb*8 + kidx);
                const float32x4_t B_val9 = vld1q_dup_f32(B_ptr + ldb*9 + kidx);
                const float32x4_t B_val10 = vld1q_dup_f32(B_ptr + ldb*10 + kidx);
                const float32x4_t B_val11 = vld1q_dup_f32(B_ptr + ldb*11 + kidx);
                o60 = vfmaq_f32(o60, A_col0, B_val6);
                o61 = vfmaq_f32(o61, A_col1, B_val6);
                o70 = vfmaq_f32(o70, A_col0, B_val7);
                o71 = vfmaq_f32(o71, A_col1, B_val7);
                o80 = vfmaq_f32(o80, A_col0, B_val8);
                o81 = vfmaq_f32(o81, A_col1, B_val8);
                o90 = vfmaq_f32(o90, A_col0, B_val9);
                o91 = vfmaq_f32(o91, A_col1, B_val9);
                o100 = vfmaq_f32(o100, A_col0, B_val10);
                o101 = vfmaq_f32(o101, A_col1, B_val10);
                o110 = vfmaq_f32(o110, A_col0, B_val11);
                o111 = vfmaq_f32(o111, A_col1, B_val11);
            }
            vst1q_f32  (C_ptr + 0  * ldc, o00);
            vst1q_f32  (C_ptr + 1  * ldc, o10);
            vst1q_f32  (C_ptr + 2  * ldc, o20);
            vst1q_f32  (C_ptr + 3  * ldc, o30);
            vst1q_f32  (C_ptr + 4  * ldc, o40);
            vst1q_f32  (C_ptr + 5  * ldc, o50);
            vst1q_f32  (C_ptr + 6  * ldc, o60);
            vst1q_f32  (C_ptr + 7  * ldc, o70);
            vst1q_f32  (C_ptr + 8  * ldc, o80);
            vst1q_f32  (C_ptr + 9  * ldc, o90);
            vst1q_f32  (C_ptr + 10 * ldc, o100);
            vst1q_f32  (C_ptr + 11 * ldc, o110);
            vst1q_f32  (C_ptr + 0  * ldc + 4, o01);
            vst1q_f32  (C_ptr + 1  * ldc + 4, o11);
            vst1q_f32  (C_ptr + 2  * ldc + 4, o21);
            vst1q_f32  (C_ptr + 3  * ldc + 4, o31);
            vst1q_f32  (C_ptr + 4  * ldc + 4, o41);
            vst1q_f32  (C_ptr + 5  * ldc + 4, o51);
            vst1q_f32  (C_ptr + 6  * ldc + 4, o61);
            vst1q_f32  (C_ptr + 7  * ldc + 4, o71);
            vst1q_f32  (C_ptr + 8  * ldc + 4, o81);
            vst1q_f32  (C_ptr + 9  * ldc + 4, o91);
            vst1q_f32  (C_ptr + 10 * ldc + 4, o101);
            vst1q_f32  (C_ptr + 11 * ldc + 4, o111);
            #else 
            for (unsigned int nn = n; nn < n + _VEC_SIZE_N; nn++)
            {
                for (unsigned int mm = m; mm < m + _VEC_SIZE_M; mm++)
                {
                    float c = C[nn * ldc + mm];
                    for (unsigned int k = 0; k < K; k++)
                    {
                        c += A[((mm/_VEC_SIZE_M) * lda + k) * _VEC_SIZE_M + (mm%_VEC_SIZE_M)] * B[nn * ldb + k];
                    }
                    C[nn * ldc + mm] = c;
                }
            }
            #endif
        }
    }
}   
void neon_sgemm_tile_M(const unsigned int M, const unsigned int N, const unsigned int K,
		 const float *A, const unsigned int lda, const float *B, const unsigned int ldb, float *C, const unsigned int ldc)
{
    const unsigned int rem_n = (N%_VEC_SIZE_N);
    for (unsigned int m = 0; m < _TILE_SIZE_M; m += _VEC_SIZE_M)
    {
        unsigned int n = 0;
        for (; n < N - rem_n; n += _VEC_SIZE_N)
        {
            #if _VEC_SIZE_N == 12 && _VEC_SIZE_M == 8
            const float *A_ptr = A + m * lda;
            const float *B_ptr = B + n * ldb;
            float *C_ptr = C + n * ldc + m;
            float32x4_t o00, o10, o20, o30, o40, o50, o60, o70, o80, o90, o100, o110;
            float32x4_t o01, o11, o21, o31, o41, o51, o61, o71, o81, o91, o101, o111;
            o00 = vld1q_f32     (C_ptr + 0  * ldc);
            o10 = vld1q_f32     (C_ptr + 1  * ldc);
            o20 = vld1q_f32     (C_ptr + 2  * ldc);
            o30 = vld1q_f32     (C_ptr + 3  * ldc);
            o40 = vld1q_f32     (C_ptr + 4  * ldc);
            o50 = vld1q_f32     (C_ptr + 5  * ldc);
            o60 = vld1q_f32     (C_ptr + 6  * ldc);
            o70 = vld1q_f32     (C_ptr + 7  * ldc);
            o80 = vld1q_f32     (C_ptr + 8  * ldc);
            o90 = vld1q_f32     (C_ptr + 9  * ldc);
            o100 = vld1q_f32    (C_ptr + 10 * ldc);
            o110 = vld1q_f32    (C_ptr + 11 * ldc);
            o01 = vld1q_f32     (C_ptr + 0  * ldc + 4);
            o11 = vld1q_f32     (C_ptr + 1  * ldc + 4);
            o21 = vld1q_f32     (C_ptr + 2  * ldc + 4);
            o31 = vld1q_f32     (C_ptr + 3  * ldc + 4);
            o41 = vld1q_f32     (C_ptr + 4  * ldc + 4);
            o51 = vld1q_f32     (C_ptr + 5  * ldc + 4);
            o61 = vld1q_f32     (C_ptr + 6  * ldc + 4);
            o71 = vld1q_f32     (C_ptr + 7  * ldc + 4);
            o81 = vld1q_f32     (C_ptr + 8  * ldc + 4);
            o91 = vld1q_f32     (C_ptr + 9  * ldc + 4);
            o101 = vld1q_f32    (C_ptr + 10 * ldc + 4);
            o111 = vld1q_f32    (C_ptr + 11 * ldc + 4);
            for (int kidx = 0; kidx < K; kidx++)
            {
                const float32x4_t A_col0 = vld1q_f32(A_ptr);
                const float32x4_t A_col1 = vld1q_f32(A_ptr + 4);
                A_ptr += 8;
                const float32x4_t B_val0 = vld1q_dup_f32(B_ptr + ldb*0 + kidx);
                const float32x4_t B_val1 = vld1q_dup_f32(B_ptr + ldb*1 + kidx);
                const float32x4_t B_val2 = vld1q_dup_f32(B_ptr + ldb*2 + kidx);
                const float32x4_t B_val3 = vld1q_dup_f32(B_ptr + ldb*3 + kidx);
                const float32x4_t B_val4 = vld1q_dup_f32(B_ptr + ldb*4 + kidx);
                const float32x4_t B_val5 = vld1q_dup_f32(B_ptr + ldb*5 + kidx);
                o00 = vfmaq_f32(o00, A_col0, B_val0);
                o01 = vfmaq_f32(o01, A_col1, B_val0);
                o10 = vfmaq_f32(o10, A_col0, B_val1);
                o11 = vfmaq_f32(o11, A_col1, B_val1);
                o20 = vfmaq_f32(o20, A_col0, B_val2);
                o21 = vfmaq_f32(o21, A_col1, B_val2);
                o30 = vfmaq_f32(o30, A_col0, B_val3);
                o31 = vfmaq_f32(o31, A_col1, B_val3);
                o40 = vfmaq_f32(o40, A_col0, B_val4);
                o41 = vfmaq_f32(o41, A_col1, B_val4);
                o50 = vfmaq_f32(o50, A_col0, B_val5);
                o51 = vfmaq_f32(o51, A_col1, B_val5);
                const float32x4_t B_val6 = vld1q_dup_f32(B_ptr + ldb*6 + kidx);
                const float32x4_t B_val7 = vld1q_dup_f32(B_ptr + ldb*7 + kidx);
                const float32x4_t B_val8 = vld1q_dup_f32(B_ptr + ldb*8 + kidx);
                const float32x4_t B_val9 = vld1q_dup_f32(B_ptr + ldb*9 + kidx);
                const float32x4_t B_val10 = vld1q_dup_f32(B_ptr + ldb*10 + kidx);
                const float32x4_t B_val11 = vld1q_dup_f32(B_ptr + ldb*11 + kidx);
                o60 = vfmaq_f32(o60, A_col0, B_val6);
                o61 = vfmaq_f32(o61, A_col1, B_val6);
                o70 = vfmaq_f32(o70, A_col0, B_val7);
                o71 = vfmaq_f32(o71, A_col1, B_val7);
                o80 = vfmaq_f32(o80, A_col0, B_val8);
                o81 = vfmaq_f32(o81, A_col1, B_val8);
                o90 = vfmaq_f32(o90, A_col0, B_val9);
                o91 = vfmaq_f32(o91, A_col1, B_val9);
                o100 = vfmaq_f32(o100, A_col0, B_val10);
                o101 = vfmaq_f32(o101, A_col1, B_val10);
                o110 = vfmaq_f32(o110, A_col0, B_val11);
                o111 = vfmaq_f32(o111, A_col1, B_val11);
            }
            vst1q_f32  (C_ptr + 0  * ldc, o00);
            vst1q_f32  (C_ptr + 1  * ldc, o10);
            vst1q_f32  (C_ptr + 2  * ldc, o20);
            vst1q_f32  (C_ptr + 3  * ldc, o30);
            vst1q_f32  (C_ptr + 4  * ldc, o40);
            vst1q_f32  (C_ptr + 5  * ldc, o50);
            vst1q_f32  (C_ptr + 6  * ldc, o60);
            vst1q_f32  (C_ptr + 7  * ldc, o70);
            vst1q_f32  (C_ptr + 8  * ldc, o80);
            vst1q_f32  (C_ptr + 9  * ldc, o90);
            vst1q_f32  (C_ptr + 10 * ldc, o100);
            vst1q_f32  (C_ptr + 11 * ldc, o110);
            vst1q_f32  (C_ptr + 0  * ldc + 4, o01);
            vst1q_f32  (C_ptr + 1  * ldc + 4, o11);
            vst1q_f32  (C_ptr + 2  * ldc + 4, o21);
            vst1q_f32  (C_ptr + 3  * ldc + 4, o31);
            vst1q_f32  (C_ptr + 4  * ldc + 4, o41);
            vst1q_f32  (C_ptr + 5  * ldc + 4, o51);
            vst1q_f32  (C_ptr + 6  * ldc + 4, o61);
            vst1q_f32  (C_ptr + 7  * ldc + 4, o71);
            vst1q_f32  (C_ptr + 8  * ldc + 4, o81);
            vst1q_f32  (C_ptr + 9  * ldc + 4, o91);
            vst1q_f32  (C_ptr + 10 * ldc + 4, o101);
            vst1q_f32  (C_ptr + 11 * ldc + 4, o111);
            #else 
            for (unsigned int nn = n; nn < n + _VEC_SIZE_N; nn++)
            {
                for (unsigned int mm = m; mm < m + _VEC_SIZE_M; mm++)
                {
                    float c = C[nn * ldc + mm];
                    for (unsigned int k = 0; k < K; k++)
                    {
                        c += A[((mm/_VEC_SIZE_M) * lda + k) * _VEC_SIZE_M + (mm%_VEC_SIZE_M)] * B[nn * ldb + k];
                    }
                    C[nn * ldc + mm] = c;
                }
            }
            #endif
        }
        #if _VEC_SIZE_M == 8
        const float *A_ptr = A + m * lda;
        const float *B_ptr = B + n * ldb;
        float *C_ptr = C + n * ldc + m;
        if (rem_n == 8)
        {
            float32x4_t o00, o10, o20, o30, o40, o50, o60, o70;
            float32x4_t o01, o11, o21, o31, o41, o51, o61, o71;
            o00 = vld1q_f32     (C_ptr + 0  * ldc);
            o10 = vld1q_f32     (C_ptr + 1  * ldc);
            o20 = vld1q_f32     (C_ptr + 2  * ldc);
            o30 = vld1q_f32     (C_ptr + 3  * ldc);
            o40 = vld1q_f32     (C_ptr + 4  * ldc);
            o50 = vld1q_f32     (C_ptr + 5  * ldc);
            o60 = vld1q_f32     (C_ptr + 6  * ldc);
            o70 = vld1q_f32     (C_ptr + 7  * ldc);
            o01 = vld1q_f32     (C_ptr + 0  * ldc + 4);
            o11 = vld1q_f32     (C_ptr + 1  * ldc + 4);
            o21 = vld1q_f32     (C_ptr + 2  * ldc + 4);
            o31 = vld1q_f32     (C_ptr + 3  * ldc + 4);
            o41 = vld1q_f32     (C_ptr + 4  * ldc + 4);
            o51 = vld1q_f32     (C_ptr + 5  * ldc + 4);
            o61 = vld1q_f32     (C_ptr + 6  * ldc + 4);
            o71 = vld1q_f32     (C_ptr + 7  * ldc + 4);
            for (int kidx = 0; kidx < K; kidx++)
            {
                const float32x4_t A_col0 = vld1q_f32(A_ptr);
                const float32x4_t A_col1 = vld1q_f32(A_ptr + 4);
                A_ptr += 8;
                const float32x4_t B_val0 = vld1q_dup_f32(B_ptr + ldb*0 + kidx);
                const float32x4_t B_val1 = vld1q_dup_f32(B_ptr + ldb*1 + kidx);
                const float32x4_t B_val2 = vld1q_dup_f32(B_ptr + ldb*2 + kidx);
                const float32x4_t B_val3 = vld1q_dup_f32(B_ptr + ldb*3 + kidx);   
                const float32x4_t B_val4 = vld1q_dup_f32(B_ptr + ldb*4 + kidx);
                const float32x4_t B_val5 = vld1q_dup_f32(B_ptr + ldb*5 + kidx);
                const float32x4_t B_val6 = vld1q_dup_f32(B_ptr + ldb*6 + kidx);
                const float32x4_t B_val7 = vld1q_dup_f32(B_ptr + ldb*7 + kidx); 
                o00 = vfmaq_f32(o00, A_col0, B_val0);
                o01 = vfmaq_f32(o01, A_col1, B_val0);
                o10 = vfmaq_f32(o10, A_col0, B_val1);
                o11 = vfmaq_f32(o11, A_col1, B_val1);
                o20 = vfmaq_f32(o20, A_col0, B_val2);
                o21 = vfmaq_f32(o21, A_col1, B_val2);
                o30 = vfmaq_f32(o30, A_col0, B_val3);
                o31 = vfmaq_f32(o31, A_col1, B_val3);
                o40 = vfmaq_f32(o40, A_col0, B_val4);
                o41 = vfmaq_f32(o41, A_col1, B_val4);
                o50 = vfmaq_f32(o50, A_col0, B_val5);
                o51 = vfmaq_f32(o51, A_col1, B_val5);
                o60 = vfmaq_f32(o60, A_col0, B_val6);
                o61 = vfmaq_f32(o61, A_col1, B_val6);
                o70 = vfmaq_f32(o70, A_col0, B_val7);
                o71 = vfmaq_f32(o71, A_col1, B_val7);
            }
            vst1q_f32  (C_ptr + 0  * ldc, o00);
            vst1q_f32  (C_ptr + 1  * ldc, o10);
            vst1q_f32  (C_ptr + 2  * ldc, o20);
            vst1q_f32  (C_ptr + 3  * ldc, o30);
            vst1q_f32  (C_ptr + 4  * ldc, o40);
            vst1q_f32  (C_ptr + 5  * ldc, o50);
            vst1q_f32  (C_ptr + 6  * ldc, o60);
            vst1q_f32  (C_ptr + 7  * ldc, o70);
            vst1q_f32  (C_ptr + 0  * ldc + 4, o01);
            vst1q_f32  (C_ptr + 1  * ldc + 4, o11);
            vst1q_f32  (C_ptr + 2  * ldc + 4, o21);
            vst1q_f32  (C_ptr + 3  * ldc + 4, o31);
            vst1q_f32  (C_ptr + 4  * ldc + 4, o41);
            vst1q_f32  (C_ptr + 5  * ldc + 4, o51);
            vst1q_f32  (C_ptr + 6  * ldc + 4, o61);
            vst1q_f32  (C_ptr + 7  * ldc + 4, o71);
        }
        else if (rem_n == 11)
        {
            float32x4_t o00, o10, o20, o30, o40, o50, o60, o70, o80, o90, o100;
            float32x4_t o01, o11, o21, o31, o41, o51, o61, o71, o81, o91, o101;
            o00 = vld1q_f32     (C_ptr + 0  * ldc);
            o10 = vld1q_f32     (C_ptr + 1  * ldc);
            o20 = vld1q_f32     (C_ptr + 2  * ldc);
            o30 = vld1q_f32     (C_ptr + 3  * ldc);
            o40 = vld1q_f32     (C_ptr + 4  * ldc);
            o50 = vld1q_f32     (C_ptr + 5  * ldc);
            o60 = vld1q_f32     (C_ptr + 6  * ldc);
            o70 = vld1q_f32     (C_ptr + 7  * ldc);
            o80 = vld1q_f32     (C_ptr + 8  * ldc);
            o90 = vld1q_f32     (C_ptr + 9  * ldc);
            o100 = vld1q_f32    (C_ptr + 10 * ldc);
            o01 = vld1q_f32     (C_ptr + 0  * ldc + 4);
            o11 = vld1q_f32     (C_ptr + 1  * ldc + 4);
            o21 = vld1q_f32     (C_ptr + 2  * ldc + 4);
            o31 = vld1q_f32     (C_ptr + 3  * ldc + 4);
            o41 = vld1q_f32     (C_ptr + 4  * ldc + 4);
            o51 = vld1q_f32     (C_ptr + 5  * ldc + 4);
            o61 = vld1q_f32     (C_ptr + 6  * ldc + 4);
            o71 = vld1q_f32     (C_ptr + 7  * ldc + 4);
            o81 = vld1q_f32     (C_ptr + 8  * ldc + 4);
            o91 = vld1q_f32     (C_ptr + 9  * ldc + 4);
            o101 = vld1q_f32    (C_ptr + 10 * ldc + 4);
            for (int kidx = 0; kidx < K; kidx++)
            {
                const float32x4_t A_col0 = vld1q_f32(A_ptr);
                const float32x4_t A_col1 = vld1q_f32(A_ptr + 4);
                A_ptr += 8;
                const float32x4_t B_val0 = vld1q_dup_f32(B_ptr + ldb*0 + kidx);
                const float32x4_t B_val1 = vld1q_dup_f32(B_ptr + ldb*1 + kidx);
                const float32x4_t B_val2 = vld1q_dup_f32(B_ptr + ldb*2 + kidx);
                const float32x4_t B_val3 = vld1q_dup_f32(B_ptr + ldb*3 + kidx);
                const float32x4_t B_val4 = vld1q_dup_f32(B_ptr + ldb*4 + kidx);
                const float32x4_t B_val5 = vld1q_dup_f32(B_ptr + ldb*5 + kidx);
                o00 = vfmaq_f32(o00, A_col0, B_val0);
                o01 = vfmaq_f32(o01, A_col1, B_val0);
                o10 = vfmaq_f32(o10, A_col0, B_val1);
                o11 = vfmaq_f32(o11, A_col1, B_val1);
                o20 = vfmaq_f32(o20, A_col0, B_val2);
                o21 = vfmaq_f32(o21, A_col1, B_val2);
                o30 = vfmaq_f32(o30, A_col0, B_val3);
                o31 = vfmaq_f32(o31, A_col1, B_val3);
                o40 = vfmaq_f32(o40, A_col0, B_val4);
                o41 = vfmaq_f32(o41, A_col1, B_val4);
                o50 = vfmaq_f32(o50, A_col0, B_val5);
                o51 = vfmaq_f32(o51, A_col1, B_val5);
                const float32x4_t B_val6 = vld1q_dup_f32(B_ptr + ldb*6 + kidx);
                const float32x4_t B_val7 = vld1q_dup_f32(B_ptr + ldb*7 + kidx);
                const float32x4_t B_val8 = vld1q_dup_f32(B_ptr + ldb*8 + kidx);
                const float32x4_t B_val9 = vld1q_dup_f32(B_ptr + ldb*9 + kidx);
                const float32x4_t B_val10 = vld1q_dup_f32(B_ptr + ldb*10 + kidx);
                o60 = vfmaq_f32(o60, A_col0, B_val6);
                o61 = vfmaq_f32(o61, A_col1, B_val6);
                o70 = vfmaq_f32(o70, A_col0, B_val7);
                o71 = vfmaq_f32(o71, A_col1, B_val7);
                o80 = vfmaq_f32(o80, A_col0, B_val8);
                o81 = vfmaq_f32(o81, A_col1, B_val8);
                o90 = vfmaq_f32(o90, A_col0, B_val9);
                o91 = vfmaq_f32(o91, A_col1, B_val9);
                o100 = vfmaq_f32(o100, A_col0, B_val10);
                o101 = vfmaq_f32(o101, A_col1, B_val10);
            }
            vst1q_f32  (C_ptr + 0  * ldc, o00);
            vst1q_f32  (C_ptr + 1  * ldc, o10);
            vst1q_f32  (C_ptr + 2  * ldc, o20);
            vst1q_f32  (C_ptr + 3  * ldc, o30);
            vst1q_f32  (C_ptr + 4  * ldc, o40);
            vst1q_f32  (C_ptr + 5  * ldc, o50);
            vst1q_f32  (C_ptr + 6  * ldc, o60);
            vst1q_f32  (C_ptr + 7  * ldc, o70);
            vst1q_f32  (C_ptr + 8  * ldc, o80);
            vst1q_f32  (C_ptr + 9  * ldc, o90);
            vst1q_f32  (C_ptr + 10 * ldc, o100);
            vst1q_f32  (C_ptr + 0  * ldc + 4, o01);
            vst1q_f32  (C_ptr + 1  * ldc + 4, o11);
            vst1q_f32  (C_ptr + 2  * ldc + 4, o21);
            vst1q_f32  (C_ptr + 3  * ldc + 4, o31);
            vst1q_f32  (C_ptr + 4  * ldc + 4, o41);
            vst1q_f32  (C_ptr + 5  * ldc + 4, o51);
            vst1q_f32  (C_ptr + 6  * ldc + 4, o61);
            vst1q_f32  (C_ptr + 7  * ldc + 4, o71);
            vst1q_f32  (C_ptr + 8  * ldc + 4, o81);
            vst1q_f32  (C_ptr + 9  * ldc + 4, o91);
            vst1q_f32  (C_ptr + 10 * ldc + 4, o101);
        }
        else if (rem_n == 10)
        {
            float32x4_t o00, o10, o20, o30, o40, o50, o60, o70, o80, o90;
            float32x4_t o01, o11, o21, o31, o41, o51, o61, o71, o81, o91;
            o00 = vld1q_f32     (C_ptr + 0  * ldc);
            o10 = vld1q_f32     (C_ptr + 1  * ldc);
            o20 = vld1q_f32     (C_ptr + 2  * ldc);
            o30 = vld1q_f32     (C_ptr + 3  * ldc);
            o40 = vld1q_f32     (C_ptr + 4  * ldc);
            o50 = vld1q_f32     (C_ptr + 5  * ldc);
            o60 = vld1q_f32     (C_ptr + 6  * ldc);
            o70 = vld1q_f32     (C_ptr + 7  * ldc);
            o80 = vld1q_f32     (C_ptr + 8  * ldc);
            o90 = vld1q_f32     (C_ptr + 9  * ldc);
            o01 = vld1q_f32     (C_ptr + 0  * ldc + 4);
            o11 = vld1q_f32     (C_ptr + 1  * ldc + 4);
            o21 = vld1q_f32     (C_ptr + 2  * ldc + 4);
            o31 = vld1q_f32     (C_ptr + 3  * ldc + 4);
            o41 = vld1q_f32     (C_ptr + 4  * ldc + 4);
            o51 = vld1q_f32     (C_ptr + 5  * ldc + 4);
            o61 = vld1q_f32     (C_ptr + 6  * ldc + 4);
            o71 = vld1q_f32     (C_ptr + 7  * ldc + 4);
            o81 = vld1q_f32     (C_ptr + 8  * ldc + 4);
            o91 = vld1q_f32     (C_ptr + 9  * ldc + 4);
            for (int kidx = 0; kidx < K; kidx++)
            {
                const float32x4_t A_col0 = vld1q_f32(A_ptr);
                const float32x4_t A_col1 = vld1q_f32(A_ptr + 4);
                A_ptr += 8;
                const float32x4_t B_val0 = vld1q_dup_f32(B_ptr + ldb*0 + kidx);
                const float32x4_t B_val1 = vld1q_dup_f32(B_ptr + ldb*1 + kidx);
                const float32x4_t B_val2 = vld1q_dup_f32(B_ptr + ldb*2 + kidx);
                const float32x4_t B_val3 = vld1q_dup_f32(B_ptr + ldb*3 + kidx);
                const float32x4_t B_val4 = vld1q_dup_f32(B_ptr + ldb*4 + kidx);
                o00 = vfmaq_f32(o00, A_col0, B_val0);
                o01 = vfmaq_f32(o01, A_col1, B_val0);
                o10 = vfmaq_f32(o10, A_col0, B_val1);
                o11 = vfmaq_f32(o11, A_col1, B_val1);
                o20 = vfmaq_f32(o20, A_col0, B_val2);
                o21 = vfmaq_f32(o21, A_col1, B_val2);
                o30 = vfmaq_f32(o30, A_col0, B_val3);
                o31 = vfmaq_f32(o31, A_col1, B_val3);
                o40 = vfmaq_f32(o40, A_col0, B_val4);
                o41 = vfmaq_f32(o41, A_col1, B_val4);
                const float32x4_t B_val5 = vld1q_dup_f32(B_ptr + ldb*5 + kidx);
                const float32x4_t B_val6 = vld1q_dup_f32(B_ptr + ldb*6 + kidx);
                const float32x4_t B_val7 = vld1q_dup_f32(B_ptr + ldb*7 + kidx);
                const float32x4_t B_val8 = vld1q_dup_f32(B_ptr + ldb*8 + kidx);
                const float32x4_t B_val9 = vld1q_dup_f32(B_ptr + ldb*9 + kidx);
                o50 = vfmaq_f32(o50, A_col0, B_val5);
                o51 = vfmaq_f32(o51, A_col1, B_val5);
                o60 = vfmaq_f32(o60, A_col0, B_val6);
                o61 = vfmaq_f32(o61, A_col1, B_val6);
                o70 = vfmaq_f32(o70, A_col0, B_val7);
                o71 = vfmaq_f32(o71, A_col1, B_val7);
                o80 = vfmaq_f32(o80, A_col0, B_val8);
                o81 = vfmaq_f32(o81, A_col1, B_val8);
                o90 = vfmaq_f32(o90, A_col0, B_val9);
                o91 = vfmaq_f32(o91, A_col1, B_val9);
            }
            vst1q_f32  (C_ptr + 0  * ldc, o00);
            vst1q_f32  (C_ptr + 1  * ldc, o10);
            vst1q_f32  (C_ptr + 2  * ldc, o20);
            vst1q_f32  (C_ptr + 3  * ldc, o30);
            vst1q_f32  (C_ptr + 4  * ldc, o40);
            vst1q_f32  (C_ptr + 5  * ldc, o50);
            vst1q_f32  (C_ptr + 6  * ldc, o60);
            vst1q_f32  (C_ptr + 7  * ldc, o70);
            vst1q_f32  (C_ptr + 8  * ldc, o80);
            vst1q_f32  (C_ptr + 9  * ldc, o90);
            vst1q_f32  (C_ptr + 0  * ldc + 4, o01);
            vst1q_f32  (C_ptr + 1  * ldc + 4, o11);
            vst1q_f32  (C_ptr + 2  * ldc + 4, o21);
            vst1q_f32  (C_ptr + 3  * ldc + 4, o31);
            vst1q_f32  (C_ptr + 4  * ldc + 4, o41);
            vst1q_f32  (C_ptr + 5  * ldc + 4, o51);
            vst1q_f32  (C_ptr + 6  * ldc + 4, o61);
            vst1q_f32  (C_ptr + 7  * ldc + 4, o71);
            vst1q_f32  (C_ptr + 8  * ldc + 4, o81);
            vst1q_f32  (C_ptr + 9  * ldc + 4, o91);
        }
        else if (rem_n == 9)
        {
            float32x4_t o00, o10, o20, o30, o40, o50, o60, o70, o80;
            float32x4_t o01, o11, o21, o31, o41, o51, o61, o71, o81;
            o00 = vld1q_f32     (C_ptr + 0  * ldc);
            o10 = vld1q_f32     (C_ptr + 1  * ldc);
            o20 = vld1q_f32     (C_ptr + 2  * ldc);
            o30 = vld1q_f32     (C_ptr + 3  * ldc);
            o40 = vld1q_f32     (C_ptr + 4  * ldc);
            o50 = vld1q_f32     (C_ptr + 5  * ldc);
            o60 = vld1q_f32     (C_ptr + 6  * ldc);
            o70 = vld1q_f32     (C_ptr + 7  * ldc);
            o80 = vld1q_f32     (C_ptr + 8  * ldc);
            o01 = vld1q_f32     (C_ptr + 0  * ldc + 4);
            o11 = vld1q_f32     (C_ptr + 1  * ldc + 4);
            o21 = vld1q_f32     (C_ptr + 2  * ldc + 4);
            o31 = vld1q_f32     (C_ptr + 3  * ldc + 4);
            o41 = vld1q_f32     (C_ptr + 4  * ldc + 4);
            o51 = vld1q_f32     (C_ptr + 5  * ldc + 4);
            o61 = vld1q_f32     (C_ptr + 6  * ldc + 4);
            o71 = vld1q_f32     (C_ptr + 7  * ldc + 4);
            o81 = vld1q_f32     (C_ptr + 8  * ldc + 4);
            for (int kidx = 0; kidx < K; kidx++)
            {
                const float32x4_t A_col0 = vld1q_f32(A_ptr);
                const float32x4_t A_col1 = vld1q_f32(A_ptr + 4);
                A_ptr += 8;
                const float32x4_t B_val0 = vld1q_dup_f32(B_ptr + ldb*0 + kidx);
                const float32x4_t B_val1 = vld1q_dup_f32(B_ptr + ldb*1 + kidx);
                const float32x4_t B_val2 = vld1q_dup_f32(B_ptr + ldb*2 + kidx);
                const float32x4_t B_val3 = vld1q_dup_f32(B_ptr + ldb*3 + kidx);
                const float32x4_t B_val4 = vld1q_dup_f32(B_ptr + ldb*4 + kidx);
                const float32x4_t B_val5 = vld1q_dup_f32(B_ptr + ldb*5 + kidx);
                const float32x4_t B_val6 = vld1q_dup_f32(B_ptr + ldb*6 + kidx);
                const float32x4_t B_val7 = vld1q_dup_f32(B_ptr + ldb*7 + kidx);
                const float32x4_t B_val8 = vld1q_dup_f32(B_ptr + ldb*8 + kidx);
                o00 = vfmaq_f32(o00, A_col0, B_val0);
                o01 = vfmaq_f32(o01, A_col1, B_val0);
                o10 = vfmaq_f32(o10, A_col0, B_val1);
                o11 = vfmaq_f32(o11, A_col1, B_val1);
                o20 = vfmaq_f32(o20, A_col0, B_val2);
                o21 = vfmaq_f32(o21, A_col1, B_val2);
                o30 = vfmaq_f32(o30, A_col0, B_val3);
                o31 = vfmaq_f32(o31, A_col1, B_val3);
                o40 = vfmaq_f32(o40, A_col0, B_val4);
                o41 = vfmaq_f32(o41, A_col1, B_val4);
                o50 = vfmaq_f32(o50, A_col0, B_val5);
                o51 = vfmaq_f32(o51, A_col1, B_val5);
                o60 = vfmaq_f32(o60, A_col0, B_val6);
                o61 = vfmaq_f32(o61, A_col1, B_val6);
                o70 = vfmaq_f32(o70, A_col0, B_val7);
                o71 = vfmaq_f32(o71, A_col1, B_val7);
                o80 = vfmaq_f32(o80, A_col0, B_val8);
                o81 = vfmaq_f32(o81, A_col1, B_val8);
            }
            vst1q_f32  (C_ptr + 0  * ldc, o00);
            vst1q_f32  (C_ptr + 1  * ldc, o10);
            vst1q_f32  (C_ptr + 2  * ldc, o20);
            vst1q_f32  (C_ptr + 3  * ldc, o30);
            vst1q_f32  (C_ptr + 4  * ldc, o40);
            vst1q_f32  (C_ptr + 5  * ldc, o50);
            vst1q_f32  (C_ptr + 6  * ldc, o60);
            vst1q_f32  (C_ptr + 7  * ldc, o70);
            vst1q_f32  (C_ptr + 8  * ldc, o80);
            vst1q_f32  (C_ptr + 0  * ldc + 4, o01);
            vst1q_f32  (C_ptr + 1  * ldc + 4, o11);
            vst1q_f32  (C_ptr + 2  * ldc + 4, o21);
            vst1q_f32  (C_ptr + 3  * ldc + 4, o31);
            vst1q_f32  (C_ptr + 4  * ldc + 4, o41);
            vst1q_f32  (C_ptr + 5  * ldc + 4, o51);
            vst1q_f32  (C_ptr + 6  * ldc + 4, o61);
            vst1q_f32  (C_ptr + 7  * ldc + 4, o71);
            vst1q_f32  (C_ptr + 8  * ldc + 4, o81);
        }
        else if (rem_n == 7)
        {
            float32x4_t o00, o10, o20, o30, o40, o50, o60;
            float32x4_t o01, o11, o21, o31, o41, o51, o61;
            o00 = vld1q_f32     (C_ptr + 0  * ldc);
            o10 = vld1q_f32     (C_ptr + 1  * ldc);
            o20 = vld1q_f32     (C_ptr + 2  * ldc);
            o30 = vld1q_f32     (C_ptr + 3  * ldc);
            o40 = vld1q_f32     (C_ptr + 4  * ldc);
            o50 = vld1q_f32     (C_ptr + 5  * ldc);
            o60 = vld1q_f32     (C_ptr + 6  * ldc);
            o01 = vld1q_f32     (C_ptr + 0  * ldc + 4);
            o11 = vld1q_f32     (C_ptr + 1  * ldc + 4);
            o21 = vld1q_f32     (C_ptr + 2  * ldc + 4);
            o31 = vld1q_f32     (C_ptr + 3  * ldc + 4);
            o41 = vld1q_f32     (C_ptr + 4  * ldc + 4);
            o51 = vld1q_f32     (C_ptr + 5  * ldc + 4);
            o61 = vld1q_f32     (C_ptr + 6  * ldc + 4);
            for (int kidx = 0; kidx < K; kidx++)
            {
                const float32x4_t A_col0 = vld1q_f32(A_ptr);
                const float32x4_t A_col1 = vld1q_f32(A_ptr + 4);
                A_ptr += 8;
                const float32x4_t B_val0 = vld1q_dup_f32(B_ptr + ldb*0 + kidx);
                const float32x4_t B_val1 = vld1q_dup_f32(B_ptr + ldb*1 + kidx);
                const float32x4_t B_val2 = vld1q_dup_f32(B_ptr + ldb*2 + kidx);
                const float32x4_t B_val3 = vld1q_dup_f32(B_ptr + ldb*3 + kidx);
                const float32x4_t B_val4 = vld1q_dup_f32(B_ptr + ldb*4 + kidx);
                const float32x4_t B_val5 = vld1q_dup_f32(B_ptr + ldb*5 + kidx);
                const float32x4_t B_val6 = vld1q_dup_f32(B_ptr + ldb*6 + kidx);
                o00 = vfmaq_f32(o00, A_col0, B_val0);
                o01 = vfmaq_f32(o01, A_col1, B_val0);
                o10 = vfmaq_f32(o10, A_col0, B_val1);
                o11 = vfmaq_f32(o11, A_col1, B_val1);
                o20 = vfmaq_f32(o20, A_col0, B_val2);
                o21 = vfmaq_f32(o21, A_col1, B_val2);
                o30 = vfmaq_f32(o30, A_col0, B_val3);
                o31 = vfmaq_f32(o31, A_col1, B_val3);
                o40 = vfmaq_f32(o40, A_col0, B_val4);
                o41 = vfmaq_f32(o41, A_col1, B_val4);
                o50 = vfmaq_f32(o50, A_col0, B_val5);
                o51 = vfmaq_f32(o51, A_col1, B_val5);
                o60 = vfmaq_f32(o60, A_col0, B_val6);
                o61 = vfmaq_f32(o61, A_col1, B_val6);
            }
            vst1q_f32  (C_ptr + 0  * ldc, o00);
            vst1q_f32  (C_ptr + 1  * ldc, o10);
            vst1q_f32  (C_ptr + 2  * ldc, o20);
            vst1q_f32  (C_ptr + 3  * ldc, o30);
            vst1q_f32  (C_ptr + 4  * ldc, o40);
            vst1q_f32  (C_ptr + 5  * ldc, o50);
            vst1q_f32  (C_ptr + 6  * ldc, o60);
            vst1q_f32  (C_ptr + 0  * ldc + 4, o01);
            vst1q_f32  (C_ptr + 1  * ldc + 4, o11);
            vst1q_f32  (C_ptr + 2  * ldc + 4, o21);
            vst1q_f32  (C_ptr + 3  * ldc + 4, o31);
            vst1q_f32  (C_ptr + 4  * ldc + 4, o41);
            vst1q_f32  (C_ptr + 5  * ldc + 4, o51);
            vst1q_f32  (C_ptr + 6  * ldc + 4, o61);
        }
        else if (rem_n == 6)
        {
            float32x4_t o00, o10, o20, o30, o40, o50;
            float32x4_t o01, o11, o21, o31, o41, o51;
            o00 = vld1q_f32     (C_ptr + 0  * ldc);
            o10 = vld1q_f32     (C_ptr + 1  * ldc);
            o20 = vld1q_f32     (C_ptr + 2  * ldc);
            o30 = vld1q_f32     (C_ptr + 3  * ldc);
            o40 = vld1q_f32     (C_ptr + 4  * ldc);
            o50 = vld1q_f32     (C_ptr + 5  * ldc);
            o01 = vld1q_f32     (C_ptr + 0  * ldc + 4);
            o11 = vld1q_f32     (C_ptr + 1  * ldc + 4);
            o21 = vld1q_f32     (C_ptr + 2  * ldc + 4);
            o31 = vld1q_f32     (C_ptr + 3  * ldc + 4);
            o41 = vld1q_f32     (C_ptr + 4  * ldc + 4);
            o51 = vld1q_f32     (C_ptr + 5  * ldc + 4);
            for (int kidx = 0; kidx < K; kidx++)
            {
                const float32x4_t A_col0 = vld1q_f32(A_ptr);
                const float32x4_t A_col1 = vld1q_f32(A_ptr + 4);
                A_ptr += 8;
                const float32x4_t B_val0 = vld1q_dup_f32(B_ptr + ldb*0 + kidx);
                const float32x4_t B_val1 = vld1q_dup_f32(B_ptr + ldb*1 + kidx);
                const float32x4_t B_val2 = vld1q_dup_f32(B_ptr + ldb*2 + kidx);
                const float32x4_t B_val3 = vld1q_dup_f32(B_ptr + ldb*3 + kidx);
                const float32x4_t B_val4 = vld1q_dup_f32(B_ptr + ldb*4 + kidx);
                const float32x4_t B_val5 = vld1q_dup_f32(B_ptr + ldb*5 + kidx);
                o00 = vfmaq_f32(o00, A_col0, B_val0);
                o01 = vfmaq_f32(o01, A_col1, B_val0);
                o10 = vfmaq_f32(o10, A_col0, B_val1);
                o11 = vfmaq_f32(o11, A_col1, B_val1);
                o20 = vfmaq_f32(o20, A_col0, B_val2);
                o21 = vfmaq_f32(o21, A_col1, B_val2);
                o30 = vfmaq_f32(o30, A_col0, B_val3);
                o31 = vfmaq_f32(o31, A_col1, B_val3);
                o40 = vfmaq_f32(o40, A_col0, B_val4);
                o41 = vfmaq_f32(o41, A_col1, B_val4);
                o50 = vfmaq_f32(o50, A_col0, B_val5);
                o51 = vfmaq_f32(o51, A_col1, B_val5);
            }
            vst1q_f32  (C_ptr + 0  * ldc, o00);
            vst1q_f32  (C_ptr + 1  * ldc, o10);
            vst1q_f32  (C_ptr + 2  * ldc, o20);
            vst1q_f32  (C_ptr + 3  * ldc, o30);
            vst1q_f32  (C_ptr + 4  * ldc, o40);
            vst1q_f32  (C_ptr + 5  * ldc, o50);
            vst1q_f32  (C_ptr + 0  * ldc + 4, o01);
            vst1q_f32  (C_ptr + 1  * ldc + 4, o11);
            vst1q_f32  (C_ptr + 2  * ldc + 4, o21);
            vst1q_f32  (C_ptr + 3  * ldc + 4, o31);
            vst1q_f32  (C_ptr + 4  * ldc + 4, o41);
            vst1q_f32  (C_ptr + 5  * ldc + 4, o51);
        }
        else if (rem_n == 5)
        {
            float32x4_t o00, o10, o20, o30, o40;
            float32x4_t o01, o11, o21, o31, o41;
            o00 = vld1q_f32     (C_ptr + 0  * ldc);
            o10 = vld1q_f32     (C_ptr + 1  * ldc);
            o20 = vld1q_f32     (C_ptr + 2  * ldc);
            o30 = vld1q_f32     (C_ptr + 3  * ldc);
            o40 = vld1q_f32     (C_ptr + 4  * ldc);
            o01 = vld1q_f32     (C_ptr + 0  * ldc + 4);
            o11 = vld1q_f32     (C_ptr + 1  * ldc + 4);
            o21 = vld1q_f32     (C_ptr + 2  * ldc + 4);
            o31 = vld1q_f32     (C_ptr + 3  * ldc + 4);
            o41 = vld1q_f32     (C_ptr + 4  * ldc + 4);
            for (int kidx = 0; kidx < K; kidx++)
            {
                const float32x4_t A_col0 = vld1q_f32(A_ptr);
                const float32x4_t A_col1 = vld1q_f32(A_ptr + 4);
                A_ptr += 8;
                const float32x4_t B_val0 = vld1q_dup_f32(B_ptr + ldb*0 + kidx);
                const float32x4_t B_val1 = vld1q_dup_f32(B_ptr + ldb*1 + kidx);
                const float32x4_t B_val2 = vld1q_dup_f32(B_ptr + ldb*2 + kidx);
                const float32x4_t B_val3 = vld1q_dup_f32(B_ptr + ldb*3 + kidx);  
                const float32x4_t B_val4 = vld1q_dup_f32(B_ptr + ldb*4 + kidx);
                o00 = vfmaq_f32(o00, A_col0, B_val0);
                o01 = vfmaq_f32(o01, A_col1, B_val0);
                o10 = vfmaq_f32(o10, A_col0, B_val1);
                o11 = vfmaq_f32(o11, A_col1, B_val1);
                o20 = vfmaq_f32(o20, A_col0, B_val2);
                o21 = vfmaq_f32(o21, A_col1, B_val2);
                o30 = vfmaq_f32(o30, A_col0, B_val3);
                o31 = vfmaq_f32(o31, A_col1, B_val3);
                o40 = vfmaq_f32(o40, A_col0, B_val4);
                o41 = vfmaq_f32(o41, A_col1, B_val4);
            }
            vst1q_f32  (C_ptr + 0  * ldc, o00);
            vst1q_f32  (C_ptr + 1  * ldc, o10);
            vst1q_f32  (C_ptr + 2  * ldc, o20);
            vst1q_f32  (C_ptr + 3  * ldc, o30);
            vst1q_f32  (C_ptr + 4  * ldc, o40);
            vst1q_f32  (C_ptr + 0  * ldc + 4, o01);
            vst1q_f32  (C_ptr + 1  * ldc + 4, o11);
            vst1q_f32  (C_ptr + 2  * ldc + 4, o21);
            vst1q_f32  (C_ptr + 3  * ldc + 4, o31);
            vst1q_f32  (C_ptr + 4  * ldc + 4, o41);
        }
        else if (rem_n == 4)
        {
            float32x4_t o00, o10, o20, o30;
            float32x4_t o01, o11, o21, o31;
            o00 = vld1q_f32     (C_ptr + 0  * ldc);
            o10 = vld1q_f32     (C_ptr + 1  * ldc);
            o20 = vld1q_f32     (C_ptr + 2  * ldc);
            o30 = vld1q_f32     (C_ptr + 3  * ldc);
            o01 = vld1q_f32     (C_ptr + 0  * ldc + 4);
            o11 = vld1q_f32     (C_ptr + 1  * ldc + 4);
            o21 = vld1q_f32     (C_ptr + 2  * ldc + 4);
            o31 = vld1q_f32     (C_ptr + 3  * ldc + 4);
            for (int kidx = 0; kidx < K; kidx++)
            {
                const float32x4_t A_col0 = vld1q_f32(A_ptr);
                const float32x4_t A_col1 = vld1q_f32(A_ptr + 4);
                A_ptr += 8;
                const float32x4_t B_val0 = vld1q_dup_f32(B_ptr + ldb*0 + kidx);
                const float32x4_t B_val1 = vld1q_dup_f32(B_ptr + ldb*1 + kidx);
                const float32x4_t B_val2 = vld1q_dup_f32(B_ptr + ldb*2 + kidx);
                const float32x4_t B_val3 = vld1q_dup_f32(B_ptr + ldb*3 + kidx);  
                o00 = vfmaq_f32(o00, A_col0, B_val0);
                o01 = vfmaq_f32(o01, A_col1, B_val0);
                o10 = vfmaq_f32(o10, A_col0, B_val1);
                o11 = vfmaq_f32(o11, A_col1, B_val1);
                o20 = vfmaq_f32(o20, A_col0, B_val2);
                o21 = vfmaq_f32(o21, A_col1, B_val2);
                o30 = vfmaq_f32(o30, A_col0, B_val3);
                o31 = vfmaq_f32(o31, A_col1, B_val3);
            }
            vst1q_f32  (C_ptr + 0  * ldc, o00);
            vst1q_f32  (C_ptr + 1  * ldc, o10);
            vst1q_f32  (C_ptr + 2  * ldc, o20);
            vst1q_f32  (C_ptr + 3  * ldc, o30);
            vst1q_f32  (C_ptr + 0  * ldc + 4, o01);
            vst1q_f32  (C_ptr + 1  * ldc + 4, o11);
            vst1q_f32  (C_ptr + 2  * ldc + 4, o21);
            vst1q_f32  (C_ptr + 3  * ldc + 4, o31);
        }
        else if (rem_n == 3)
        {
            float32x4_t o00, o10, o20;
            float32x4_t o01, o11, o21;
            o00 = vld1q_f32     (C_ptr + 0  * ldc);
            o10 = vld1q_f32     (C_ptr + 1  * ldc);
            o20 = vld1q_f32     (C_ptr + 2  * ldc);
            o01 = vld1q_f32     (C_ptr + 0  * ldc + 4);
            o11 = vld1q_f32     (C_ptr + 1  * ldc + 4);
            o21 = vld1q_f32     (C_ptr + 2  * ldc + 4);
            for (int kidx = 0; kidx < K; kidx++)
            {
                const float32x4_t A_col0 = vld1q_f32(A_ptr);
                const float32x4_t A_col1 = vld1q_f32(A_ptr + 4);
                A_ptr += 8;
                const float32x4_t B_val0 = vld1q_dup_f32(B_ptr + ldb*0 + kidx);
                const float32x4_t B_val1 = vld1q_dup_f32(B_ptr + ldb*1 + kidx);
                const float32x4_t B_val2 = vld1q_dup_f32(B_ptr + ldb*2 + kidx);
                o00 = vfmaq_f32(o00, A_col0, B_val0);
                o01 = vfmaq_f32(o01, A_col1, B_val0);
                o10 = vfmaq_f32(o10, A_col0, B_val1);
                o11 = vfmaq_f32(o11, A_col1, B_val1);
                o20 = vfmaq_f32(o20, A_col0, B_val2);
                o21 = vfmaq_f32(o21, A_col1, B_val2);
            }
            vst1q_f32  (C_ptr + 0  * ldc, o00);
            vst1q_f32  (C_ptr + 1  * ldc, o10);
            vst1q_f32  (C_ptr + 2  * ldc, o20);
            vst1q_f32  (C_ptr + 0  * ldc + 4, o01);
            vst1q_f32  (C_ptr + 1  * ldc + 4, o11);
            vst1q_f32  (C_ptr + 2  * ldc + 4, o21);
        }
        else if (rem_n == 2)
        {
            float32x4_t o00, o10;
            float32x4_t o01, o11;
            o00 = vld1q_f32     (C_ptr + 0  * ldc);
            o10 = vld1q_f32     (C_ptr + 1  * ldc);
            o01 = vld1q_f32     (C_ptr + 0  * ldc + 4);
            o11 = vld1q_f32     (C_ptr + 1  * ldc + 4);
            for (int kidx = 0; kidx < K; kidx++)
            {
                const float32x4_t A_col0 = vld1q_f32(A_ptr);
                const float32x4_t A_col1 = vld1q_f32(A_ptr + 4);
                A_ptr += 8;
                const float32x4_t B_val0 = vld1q_dup_f32(B_ptr + ldb*0 + kidx);
                const float32x4_t B_val1 = vld1q_dup_f32(B_ptr + ldb*1 + kidx);
                o00 = vfmaq_f32(o00, A_col0, B_val0);
                o01 = vfmaq_f32(o01, A_col1, B_val0);
                o10 = vfmaq_f32(o10, A_col0, B_val1);
                o11 = vfmaq_f32(o11, A_col1, B_val1);
            }
            vst1q_f32  (C_ptr + 0  * ldc, o00);
            vst1q_f32  (C_ptr + 1  * ldc, o10);
            vst1q_f32  (C_ptr + 0  * ldc + 4, o01);
            vst1q_f32  (C_ptr + 1  * ldc + 4, o11);
        }
        else if (rem_n == 1)
        {
            float32x4_t o00;
            float32x4_t o01;
            o00 = vld1q_f32     (C_ptr + 0  * ldc);
            o01 = vld1q_f32     (C_ptr + 0  * ldc + 4);
            for (int kidx = 0; kidx < K; kidx++)
            {
                const float32x4_t A_col0 = vld1q_f32(A_ptr);
                const float32x4_t A_col1 = vld1q_f32(A_ptr + 4);
                A_ptr += 8;
                const float32x4_t B_val0 = vld1q_dup_f32(B_ptr + ldb*0 + kidx);
                o00 = vfmaq_f32(o00, A_col0, B_val0);
                o01 = vfmaq_f32(o01, A_col1, B_val0);
            }
            vst1q_f32  (C_ptr + 0  * ldc, o00);
            vst1q_f32  (C_ptr + 0  * ldc + 4, o01);
        }
        #else
        for (unsigned int n = N - (N%_VEC_SIZE_N); n < N; n++)
        {
            for (unsigned int mm = m; mm < m + _VEC_SIZE_M; mm++)
            {
                float c = C[n * ldc + mm];
                for (unsigned int k = 0; k < K; k++)
                {
                    c += A[((mm/_VEC_SIZE_M) * lda + k) * _VEC_SIZE_M + (mm%_VEC_SIZE_M)] * B[n * ldb + k];
                }
                C[n * ldc + mm] = c;
            }
        }
        #endif
    }
}   
void neon_sgemm_tile_N(const unsigned int M, const unsigned int N, const unsigned int K,
		 const float *A, const unsigned int lda, const float *B, const unsigned int ldb, float *C, const unsigned int ldc)
{
    const unsigned int rem_m = (M%_VEC_SIZE_M);
    for (unsigned int m = 0; m < M - rem_m; m += _VEC_SIZE_M)
    {
        unsigned int n = 0;
        for (; n < _TILE_SIZE_N; n += _VEC_SIZE_N)
        {
            #if _VEC_SIZE_N == 12 && _VEC_SIZE_M == 8
            const float *A_ptr = A + m * lda;
            const float *B_ptr = B + n * ldb;
            float *C_ptr = C + n * ldc + m;
            float32x4_t o00, o10, o20, o30, o40, o50, o60, o70, o80, o90, o100, o110;
            float32x4_t o01, o11, o21, o31, o41, o51, o61, o71, o81, o91, o101, o111;
            o00 = vld1q_f32     (C_ptr + 0  * ldc);
            o10 = vld1q_f32     (C_ptr + 1  * ldc);
            o20 = vld1q_f32     (C_ptr + 2  * ldc);
            o30 = vld1q_f32     (C_ptr + 3  * ldc);
            o40 = vld1q_f32     (C_ptr + 4  * ldc);
            o50 = vld1q_f32     (C_ptr + 5  * ldc);
            o60 = vld1q_f32     (C_ptr + 6  * ldc);
            o70 = vld1q_f32     (C_ptr + 7  * ldc);
            o80 = vld1q_f32     (C_ptr + 8  * ldc);
            o90 = vld1q_f32     (C_ptr + 9  * ldc);
            o100 = vld1q_f32    (C_ptr + 10 * ldc);
            o110 = vld1q_f32    (C_ptr + 11 * ldc);
            o01 = vld1q_f32     (C_ptr + 0  * ldc + 4);
            o11 = vld1q_f32     (C_ptr + 1  * ldc + 4);
            o21 = vld1q_f32     (C_ptr + 2  * ldc + 4);
            o31 = vld1q_f32     (C_ptr + 3  * ldc + 4);
            o41 = vld1q_f32     (C_ptr + 4  * ldc + 4);
            o51 = vld1q_f32     (C_ptr + 5  * ldc + 4);
            o61 = vld1q_f32     (C_ptr + 6  * ldc + 4);
            o71 = vld1q_f32     (C_ptr + 7  * ldc + 4);
            o81 = vld1q_f32     (C_ptr + 8  * ldc + 4);
            o91 = vld1q_f32     (C_ptr + 9  * ldc + 4);
            o101 = vld1q_f32    (C_ptr + 10 * ldc + 4);
            o111 = vld1q_f32    (C_ptr + 11 * ldc + 4);
            for (int kidx = 0; kidx < K; kidx++)
            {
                const float32x4_t A_col0 = vld1q_f32(A_ptr);
                const float32x4_t A_col1 = vld1q_f32(A_ptr + 4);
                A_ptr += 8;
                const float32x4_t B_val0 = vld1q_dup_f32(B_ptr + ldb*0 + kidx);
                const float32x4_t B_val1 = vld1q_dup_f32(B_ptr + ldb*1 + kidx);
                const float32x4_t B_val2 = vld1q_dup_f32(B_ptr + ldb*2 + kidx);
                const float32x4_t B_val3 = vld1q_dup_f32(B_ptr + ldb*3 + kidx);
                const float32x4_t B_val4 = vld1q_dup_f32(B_ptr + ldb*4 + kidx);
                const float32x4_t B_val5 = vld1q_dup_f32(B_ptr + ldb*5 + kidx);
                o00 = vfmaq_f32(o00, A_col0, B_val0);
                o01 = vfmaq_f32(o01, A_col1, B_val0);
                o10 = vfmaq_f32(o10, A_col0, B_val1);
                o11 = vfmaq_f32(o11, A_col1, B_val1);
                o20 = vfmaq_f32(o20, A_col0, B_val2);
                o21 = vfmaq_f32(o21, A_col1, B_val2);
                o30 = vfmaq_f32(o30, A_col0, B_val3);
                o31 = vfmaq_f32(o31, A_col1, B_val3);
                o40 = vfmaq_f32(o40, A_col0, B_val4);
                o41 = vfmaq_f32(o41, A_col1, B_val4);
                o50 = vfmaq_f32(o50, A_col0, B_val5);
                o51 = vfmaq_f32(o51, A_col1, B_val5);
                const float32x4_t B_val6 = vld1q_dup_f32(B_ptr + ldb*6 + kidx);
                const float32x4_t B_val7 = vld1q_dup_f32(B_ptr + ldb*7 + kidx);
                const float32x4_t B_val8 = vld1q_dup_f32(B_ptr + ldb*8 + kidx);
                const float32x4_t B_val9 = vld1q_dup_f32(B_ptr + ldb*9 + kidx);
                const float32x4_t B_val10 = vld1q_dup_f32(B_ptr + ldb*10 + kidx);
                const float32x4_t B_val11 = vld1q_dup_f32(B_ptr + ldb*11 + kidx);
                o60 = vfmaq_f32(o60, A_col0, B_val6);
                o61 = vfmaq_f32(o61, A_col1, B_val6);
                o70 = vfmaq_f32(o70, A_col0, B_val7);
                o71 = vfmaq_f32(o71, A_col1, B_val7);
                o80 = vfmaq_f32(o80, A_col0, B_val8);
                o81 = vfmaq_f32(o81, A_col1, B_val8);
                o90 = vfmaq_f32(o90, A_col0, B_val9);
                o91 = vfmaq_f32(o91, A_col1, B_val9);
                o100 = vfmaq_f32(o100, A_col0, B_val10);
                o101 = vfmaq_f32(o101, A_col1, B_val10);
                o110 = vfmaq_f32(o110, A_col0, B_val11);
                o111 = vfmaq_f32(o111, A_col1, B_val11);
            }
            vst1q_f32  (C_ptr + 0  * ldc, o00);
            vst1q_f32  (C_ptr + 1  * ldc, o10);
            vst1q_f32  (C_ptr + 2  * ldc, o20);
            vst1q_f32  (C_ptr + 3  * ldc, o30);
            vst1q_f32  (C_ptr + 4  * ldc, o40);
            vst1q_f32  (C_ptr + 5  * ldc, o50);
            vst1q_f32  (C_ptr + 6  * ldc, o60);
            vst1q_f32  (C_ptr + 7  * ldc, o70);
            vst1q_f32  (C_ptr + 8  * ldc, o80);
            vst1q_f32  (C_ptr + 9  * ldc, o90);
            vst1q_f32  (C_ptr + 10 * ldc, o100);
            vst1q_f32  (C_ptr + 11 * ldc, o110);
            vst1q_f32  (C_ptr + 0  * ldc + 4, o01);
            vst1q_f32  (C_ptr + 1  * ldc + 4, o11);
            vst1q_f32  (C_ptr + 2  * ldc + 4, o21);
            vst1q_f32  (C_ptr + 3  * ldc + 4, o31);
            vst1q_f32  (C_ptr + 4  * ldc + 4, o41);
            vst1q_f32  (C_ptr + 5  * ldc + 4, o51);
            vst1q_f32  (C_ptr + 6  * ldc + 4, o61);
            vst1q_f32  (C_ptr + 7  * ldc + 4, o71);
            vst1q_f32  (C_ptr + 8  * ldc + 4, o81);
            vst1q_f32  (C_ptr + 9  * ldc + 4, o91);
            vst1q_f32  (C_ptr + 10 * ldc + 4, o101);
            vst1q_f32  (C_ptr + 11 * ldc + 4, o111);
            #else 
            for (unsigned int nn = n; nn < n + _VEC_SIZE_N; nn++)
            {
                for (unsigned int mm = m; mm < m + _VEC_SIZE_M; mm++)
                {
                    float c = C[nn * ldc + mm];
                    for (unsigned int k = 0; k < K; k++)
                    {
                        c += A[((mm/_VEC_SIZE_M) * lda + k) * _VEC_SIZE_M + (mm%_VEC_SIZE_M)] * B[nn * ldb + k];
                    }
                    C[nn * ldc + mm] = c;
                }
            }
            #endif
        }
    }
    
    for (unsigned int m = M - rem_m; m < M; m++)
    {
        for (unsigned int n = 0; n < _TILE_SIZE_N; n += _VEC_SIZE_N)
        {
            for (unsigned int nn = n; nn < n + _VEC_SIZE_N; nn++)
            {
                float c = C[nn * ldc + m];
                for (unsigned int k = 0; k < K; k++)
                {
                    c += A[((m/_VEC_SIZE_M) * lda + k) * _VEC_SIZE_M + rem_m] * B[nn * ldb + k];
                }
                C[nn * ldc + m] = c;
            }
        }
        for (unsigned int n = _TILE_SIZE_N; n < N; n++)
        {
            float c = C[n * ldc + m];
            for (unsigned int k = 0; k < K; k++)
            {
                c += A[((m/_VEC_SIZE_M) * lda + k) * _VEC_SIZE_M + rem_m] * B[n * ldb + k];
            }
            C[n * ldc + m] = c;
        }
    }
}   
#endif 
