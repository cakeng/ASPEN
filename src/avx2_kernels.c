#include "kernels.h"
#ifdef AVX2

void avx2_sgemm_vectorized(const unsigned int M, const unsigned int N, const unsigned int K,
		 const float *A, const unsigned int lda, const float *B, const unsigned int ldb, float *C, const unsigned int ldc)
{
    const unsigned int rem_n = (N%_VEC_SIZE_N);
    const unsigned int rem_m = (M%_VEC_SIZE_M);
    for (unsigned int n = 0; n < N - rem_n; n += _VEC_SIZE_N)
    {
        for (unsigned int m = 0; m < M - rem_m; m += _VEC_SIZE_M)
        {
            #if _VEC_SIZE_N == 12 && _VEC_SIZE_M == 8
            __m256 o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11;
            const float *A_ptr = A + m * lda;
            const float *B_ptr = B + n * ldb;
            float *C_ptr = C + n * ldc + m;
            o0 = _mm256_load_ps     (C_ptr + 0 * ldc);
            o1 = _mm256_load_ps     (C_ptr + 1 * ldc);
            o2 = _mm256_load_ps     (C_ptr + 2 * ldc);
            o3 = _mm256_load_ps     (C_ptr + 3 * ldc);
            o4 = _mm256_load_ps     (C_ptr + 4 * ldc);
            o5 = _mm256_load_ps     (C_ptr + 5 * ldc);
            o6 = _mm256_load_ps     (C_ptr + 6 * ldc);
            o7 = _mm256_load_ps     (C_ptr + 7 * ldc);
            o8 = _mm256_load_ps     (C_ptr + 8 * ldc);
            o9 = _mm256_load_ps     (C_ptr + 9 * ldc);
            o10 = _mm256_load_ps    (C_ptr + 10 * ldc);
            o11 = _mm256_load_ps    (C_ptr + 11 * ldc);
            for (int kidx = 0; kidx < K; kidx++)
            {
                const __m256 A_col = _mm256_load_ps(A_ptr);
                A_ptr += 8;
                const __m256 B_val0 = _mm256_broadcast_ss(B_ptr + ldb*0 + kidx);
                const __m256 B_val1 = _mm256_broadcast_ss(B_ptr + ldb*1 + kidx);
                const __m256 B_val2 = _mm256_broadcast_ss(B_ptr + ldb*2 + kidx);
                o0 = _mm256_fmadd_ps(A_col, B_val0, o0);
                o1 = _mm256_fmadd_ps(A_col, B_val1, o1);
                o2 = _mm256_fmadd_ps(A_col, B_val2, o2);
                const __m256 B_val3 = _mm256_broadcast_ss(B_ptr + ldb*3 + kidx);
                const __m256 B_val4 = _mm256_broadcast_ss(B_ptr + ldb*4 + kidx);
                const __m256 B_val5 = _mm256_broadcast_ss(B_ptr + ldb*5 + kidx);
                o3 = _mm256_fmadd_ps(A_col, B_val3, o3);
                o4 = _mm256_fmadd_ps(A_col, B_val4, o4);
                o5 = _mm256_fmadd_ps(A_col, B_val5, o5);
                const __m256 B_val6 = _mm256_broadcast_ss(B_ptr + ldb*6 + kidx);
                const __m256 B_val7 = _mm256_broadcast_ss(B_ptr + ldb*7 + kidx);
                const __m256 B_val8 = _mm256_broadcast_ss(B_ptr + ldb*8 + kidx);
                o6 = _mm256_fmadd_ps(A_col, B_val6, o6);
                o7 = _mm256_fmadd_ps(A_col, B_val7, o7);
                o8 = _mm256_fmadd_ps(A_col, B_val8, o8);
                const __m256 B_val9 = _mm256_broadcast_ss(B_ptr + ldb*9 + kidx);
                const __m256 B_val10 = _mm256_broadcast_ss(B_ptr + ldb*10 + kidx);
                const __m256 B_val11 = _mm256_broadcast_ss(B_ptr + ldb*11 + kidx);
                o9 = _mm256_fmadd_ps(A_col, B_val9, o9);
                o10 = _mm256_fmadd_ps(A_col, B_val10, o10);
                o11 = _mm256_fmadd_ps(A_col, B_val11, o11);
            }
            _mm256_store_ps(C_ptr + 0 * ldc, o0);
            _mm256_store_ps(C_ptr + 1 * ldc, o1);
            _mm256_store_ps(C_ptr + 2 * ldc, o2);
            _mm256_store_ps(C_ptr + 3 * ldc, o3);
            _mm256_store_ps(C_ptr + 4 * ldc, o4);
            _mm256_store_ps(C_ptr + 5 * ldc, o5);
            _mm256_store_ps(C_ptr + 6 * ldc, o6);
            _mm256_store_ps(C_ptr + 7 * ldc, o7);
            _mm256_store_ps(C_ptr + 8 * ldc, o8);
            _mm256_store_ps(C_ptr + 9 * ldc, o9);
            _mm256_store_ps(C_ptr + 10 * ldc, o10);
            _mm256_store_ps(C_ptr + 11 * ldc, o11);
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
    for (unsigned int m = 0; m < M - rem_m; m += _VEC_SIZE_M)
    {
        #if _VEC_SIZE_M == 8
        const unsigned int n = N - rem_n;
        if (rem_n == 12)
        {
            __m256 o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11;
            const float *A_ptr = A + m * lda;
            const float *B_ptr = B + n * ldb;
            float *C_ptr = C + n * ldc + m;
            o0 = _mm256_load_ps     (C_ptr + 0 * ldc);
            o1 = _mm256_load_ps     (C_ptr + 1 * ldc);
            o2 = _mm256_load_ps     (C_ptr + 2 * ldc);
            o3 = _mm256_load_ps     (C_ptr + 3 * ldc);
            o4 = _mm256_load_ps     (C_ptr + 4 * ldc);
            o5 = _mm256_load_ps     (C_ptr + 5 * ldc);
            o6 = _mm256_load_ps     (C_ptr + 6 * ldc);
            o7 = _mm256_load_ps     (C_ptr + 7 * ldc);
            o8 = _mm256_load_ps     (C_ptr + 8 * ldc);
            o9 = _mm256_load_ps     (C_ptr + 9 * ldc);
            o10 = _mm256_load_ps    (C_ptr + 10 * ldc);
            o11 = _mm256_load_ps    (C_ptr + 11 * ldc);
            for (int kidx = 0; kidx < K; kidx++)
            {
                const __m256 A_col = _mm256_load_ps(A_ptr);
                A_ptr += 8;
                const __m256 B_val0 = _mm256_broadcast_ss(B_ptr + ldb*0 + kidx);
                const __m256 B_val1 = _mm256_broadcast_ss(B_ptr + ldb*1 + kidx);
                const __m256 B_val2 = _mm256_broadcast_ss(B_ptr + ldb*2 + kidx);
                o0 = _mm256_fmadd_ps(A_col, B_val0, o0);
                o1 = _mm256_fmadd_ps(A_col, B_val1, o1);
                o2 = _mm256_fmadd_ps(A_col, B_val2, o2);
                const __m256 B_val3 = _mm256_broadcast_ss(B_ptr + ldb*3 + kidx);
                const __m256 B_val4 = _mm256_broadcast_ss(B_ptr + ldb*4 + kidx);
                const __m256 B_val5 = _mm256_broadcast_ss(B_ptr + ldb*5 + kidx);
                o3 = _mm256_fmadd_ps(A_col, B_val3, o3);
                o4 = _mm256_fmadd_ps(A_col, B_val4, o4);
                o5 = _mm256_fmadd_ps(A_col, B_val5, o5);
                const __m256 B_val6 = _mm256_broadcast_ss(B_ptr + ldb*6 + kidx);
                const __m256 B_val7 = _mm256_broadcast_ss(B_ptr + ldb*7 + kidx);
                const __m256 B_val8 = _mm256_broadcast_ss(B_ptr + ldb*8 + kidx);
                o6 = _mm256_fmadd_ps(A_col, B_val6, o6);
                o7 = _mm256_fmadd_ps(A_col, B_val7, o7);
                o8 = _mm256_fmadd_ps(A_col, B_val8, o8);
                const __m256 B_val9 = _mm256_broadcast_ss(B_ptr + ldb*9 + kidx);
                const __m256 B_val10 = _mm256_broadcast_ss(B_ptr + ldb*10 + kidx);
                const __m256 B_val11 = _mm256_broadcast_ss(B_ptr + ldb*11 + kidx);
                o9 = _mm256_fmadd_ps(A_col, B_val9, o9);
                o10 = _mm256_fmadd_ps(A_col, B_val10, o10);
                o11 = _mm256_fmadd_ps(A_col, B_val11, o11);
            }
            _mm256_store_ps(C_ptr + 0 * ldc, o0);
            _mm256_store_ps(C_ptr + 1 * ldc, o1);
            _mm256_store_ps(C_ptr + 2 * ldc, o2);
            _mm256_store_ps(C_ptr + 3 * ldc, o3);
            _mm256_store_ps(C_ptr + 4 * ldc, o4);
            _mm256_store_ps(C_ptr + 5 * ldc, o5);
            _mm256_store_ps(C_ptr + 6 * ldc, o6);
            _mm256_store_ps(C_ptr + 7 * ldc, o7);
            _mm256_store_ps(C_ptr + 8 * ldc, o8);
            _mm256_store_ps(C_ptr + 9 * ldc, o9);
            _mm256_store_ps(C_ptr + 10 * ldc, o10);
            _mm256_store_ps(C_ptr + 11 * ldc, o11);
        }
        else if (rem_n == 11)
        {
            __m256 o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10;
            const float *A_ptr = A + m * lda;
            const float *B_ptr = B + n * ldb;
            float *C_ptr = C + n * ldc + m;
            o0 = _mm256_load_ps     (C_ptr + 0 * ldc);
            o1 = _mm256_load_ps     (C_ptr + 1 * ldc);
            o2 = _mm256_load_ps     (C_ptr + 2 * ldc);
            o3 = _mm256_load_ps     (C_ptr + 3 * ldc);
            o4 = _mm256_load_ps     (C_ptr + 4 * ldc);
            o5 = _mm256_load_ps     (C_ptr + 5 * ldc);
            o6 = _mm256_load_ps     (C_ptr + 6 * ldc);
            o7 = _mm256_load_ps     (C_ptr + 7 * ldc);
            o8 = _mm256_load_ps     (C_ptr + 8 * ldc);
            o9 = _mm256_load_ps     (C_ptr + 9 * ldc);
            o10 = _mm256_load_ps    (C_ptr + 10 * ldc);
            for (int kidx = 0; kidx < K; kidx++)
            {
                const __m256 A_col = _mm256_load_ps(A_ptr);
                A_ptr += 8;
                const __m256 B_val0 = _mm256_broadcast_ss(B_ptr + ldb*0 + kidx);
                const __m256 B_val1 = _mm256_broadcast_ss(B_ptr + ldb*1 + kidx);
                const __m256 B_val2 = _mm256_broadcast_ss(B_ptr + ldb*2 + kidx);
                const __m256 B_val3 = _mm256_broadcast_ss(B_ptr + ldb*3 + kidx);
                o0 = _mm256_fmadd_ps(A_col, B_val0, o0);
                o1 = _mm256_fmadd_ps(A_col, B_val1, o1);
                o2 = _mm256_fmadd_ps(A_col, B_val2, o2);
                o3 = _mm256_fmadd_ps(A_col, B_val3, o3);
                const __m256 B_val4 = _mm256_broadcast_ss(B_ptr + ldb*4 + kidx);
                const __m256 B_val5 = _mm256_broadcast_ss(B_ptr + ldb*5 + kidx);
                const __m256 B_val6 = _mm256_broadcast_ss(B_ptr + ldb*6 + kidx);
                const __m256 B_val7 = _mm256_broadcast_ss(B_ptr + ldb*7 + kidx);
                o4 = _mm256_fmadd_ps(A_col, B_val4, o4);
                o5 = _mm256_fmadd_ps(A_col, B_val5, o5);  
                o6 = _mm256_fmadd_ps(A_col, B_val6, o6);
                o7 = _mm256_fmadd_ps(A_col, B_val7, o7);
                const __m256 B_val8 = _mm256_broadcast_ss(B_ptr + ldb*8 + kidx);
                const __m256 B_val9 = _mm256_broadcast_ss(B_ptr + ldb*9 + kidx);
                const __m256 B_val10 = _mm256_broadcast_ss(B_ptr + ldb*10 + kidx);
                o8 = _mm256_fmadd_ps(A_col, B_val8, o8);
                o9 = _mm256_fmadd_ps(A_col, B_val9, o9);
                o10 = _mm256_fmadd_ps(A_col, B_val10, o10);
            }
            _mm256_store_ps(C_ptr + 0 * ldc, o0);
            _mm256_store_ps(C_ptr + 1 * ldc, o1);
            _mm256_store_ps(C_ptr + 2 * ldc, o2);
            _mm256_store_ps(C_ptr + 3 * ldc, o3);
            _mm256_store_ps(C_ptr + 4 * ldc, o4);
            _mm256_store_ps(C_ptr + 5 * ldc, o5);
            _mm256_store_ps(C_ptr + 6 * ldc, o6);
            _mm256_store_ps(C_ptr + 7 * ldc, o7);
            _mm256_store_ps(C_ptr + 8 * ldc, o8);
            _mm256_store_ps(C_ptr + 9 * ldc, o9);
            _mm256_store_ps(C_ptr + 10 * ldc, o10);
        }
        else if (rem_n == 10)
        {
            __m256 o0, o1, o2, o3, o4, o5, o6, o7, o8, o9;
            const float *A_ptr = A + m * lda;
            const float *B_ptr = B + n * ldb;
            float *C_ptr = C + n * ldc + m;
            o0 = _mm256_load_ps     (C_ptr + 0 * ldc);
            o1 = _mm256_load_ps     (C_ptr + 1 * ldc);
            o2 = _mm256_load_ps     (C_ptr + 2 * ldc);
            o3 = _mm256_load_ps     (C_ptr + 3 * ldc);
            o4 = _mm256_load_ps     (C_ptr + 4 * ldc);
            o5 = _mm256_load_ps     (C_ptr + 5 * ldc);
            o6 = _mm256_load_ps     (C_ptr + 6 * ldc);
            o7 = _mm256_load_ps     (C_ptr + 7 * ldc);
            o8 = _mm256_load_ps     (C_ptr + 8 * ldc);
            o9 = _mm256_load_ps     (C_ptr + 9 * ldc);
            for (int kidx = 0; kidx < K; kidx++)
            {
                const __m256 A_col = _mm256_load_ps(A_ptr);
                A_ptr += 8;
                const __m256 B_val0 = _mm256_broadcast_ss(B_ptr + ldb*0 + kidx);
                const __m256 B_val1 = _mm256_broadcast_ss(B_ptr + ldb*1 + kidx);
                const __m256 B_val2 = _mm256_broadcast_ss(B_ptr + ldb*2 + kidx);
                const __m256 B_val3 = _mm256_broadcast_ss(B_ptr + ldb*3 + kidx);
                const __m256 B_val4 = _mm256_broadcast_ss(B_ptr + ldb*4 + kidx);
                o0 = _mm256_fmadd_ps(A_col, B_val0, o0);
                o1 = _mm256_fmadd_ps(A_col, B_val1, o1);
                o2 = _mm256_fmadd_ps(A_col, B_val2, o2);
                o3 = _mm256_fmadd_ps(A_col, B_val3, o3);
                o4 = _mm256_fmadd_ps(A_col, B_val4, o4);
                const __m256 B_val5 = _mm256_broadcast_ss(B_ptr + ldb*5 + kidx);
                const __m256 B_val6 = _mm256_broadcast_ss(B_ptr + ldb*6 + kidx);
                const __m256 B_val7 = _mm256_broadcast_ss(B_ptr + ldb*7 + kidx);
                const __m256 B_val8 = _mm256_broadcast_ss(B_ptr + ldb*8 + kidx);
                const __m256 B_val9 = _mm256_broadcast_ss(B_ptr + ldb*9 + kidx);
                o5 = _mm256_fmadd_ps(A_col, B_val5, o5);  
                o6 = _mm256_fmadd_ps(A_col, B_val6, o6);
                o7 = _mm256_fmadd_ps(A_col, B_val7, o7);
                o8 = _mm256_fmadd_ps(A_col, B_val8, o8);
                o9 = _mm256_fmadd_ps(A_col, B_val9, o9);
            }
            _mm256_store_ps(C_ptr + 0 * ldc, o0);
            _mm256_store_ps(C_ptr + 1 * ldc, o1);
            _mm256_store_ps(C_ptr + 2 * ldc, o2);
            _mm256_store_ps(C_ptr + 3 * ldc, o3);
            _mm256_store_ps(C_ptr + 4 * ldc, o4);
            _mm256_store_ps(C_ptr + 5 * ldc, o5);
            _mm256_store_ps(C_ptr + 6 * ldc, o6);
            _mm256_store_ps(C_ptr + 7 * ldc, o7);
            _mm256_store_ps(C_ptr + 8 * ldc, o8);
            _mm256_store_ps(C_ptr + 9 * ldc, o9);
        }
        else if (rem_n == 9)
        {
            __m256 o0, o1, o2, o3, o4, o5, o6, o7, o8;
            const float *A_ptr = A + m * lda;
            const float *B_ptr = B + n * ldb;
            float *C_ptr = C + n * ldc + m;
            o0 = _mm256_load_ps     (C_ptr + 0 * ldc);
            o1 = _mm256_load_ps     (C_ptr + 1 * ldc);
            o2 = _mm256_load_ps     (C_ptr + 2 * ldc);
            o3 = _mm256_load_ps     (C_ptr + 3 * ldc);
            o4 = _mm256_load_ps     (C_ptr + 4 * ldc);
            o5 = _mm256_load_ps     (C_ptr + 5 * ldc);
            o6 = _mm256_load_ps     (C_ptr + 6 * ldc);
            o7 = _mm256_load_ps     (C_ptr + 7 * ldc);
            o8 = _mm256_load_ps     (C_ptr + 8 * ldc);
            for (int kidx = 0; kidx < K; kidx++)
            {
                const __m256 A_col = _mm256_load_ps(A_ptr);
                A_ptr += 8;
                const __m256 B_val0 = _mm256_broadcast_ss(B_ptr + ldb*0 + kidx);
                const __m256 B_val1 = _mm256_broadcast_ss(B_ptr + ldb*1 + kidx);
                const __m256 B_val2 = _mm256_broadcast_ss(B_ptr + ldb*2 + kidx);
                const __m256 B_val3 = _mm256_broadcast_ss(B_ptr + ldb*3 + kidx);
                const __m256 B_val4 = _mm256_broadcast_ss(B_ptr + ldb*4 + kidx);
                o0 = _mm256_fmadd_ps(A_col, B_val0, o0);
                o1 = _mm256_fmadd_ps(A_col, B_val1, o1);
                o2 = _mm256_fmadd_ps(A_col, B_val2, o2);
                o3 = _mm256_fmadd_ps(A_col, B_val3, o3);
                o4 = _mm256_fmadd_ps(A_col, B_val4, o4);
                const __m256 B_val5 = _mm256_broadcast_ss(B_ptr + ldb*5 + kidx);
                const __m256 B_val6 = _mm256_broadcast_ss(B_ptr + ldb*6 + kidx);
                const __m256 B_val7 = _mm256_broadcast_ss(B_ptr + ldb*7 + kidx);
                const __m256 B_val8 = _mm256_broadcast_ss(B_ptr + ldb*8 + kidx);
                o5 = _mm256_fmadd_ps(A_col, B_val5, o5);  
                o6 = _mm256_fmadd_ps(A_col, B_val6, o6);
                o7 = _mm256_fmadd_ps(A_col, B_val7, o7);
                o8 = _mm256_fmadd_ps(A_col, B_val8, o8);
            }
            _mm256_store_ps(C_ptr + 0 * ldc, o0);
            _mm256_store_ps(C_ptr + 1 * ldc, o1);
            _mm256_store_ps(C_ptr + 2 * ldc, o2);
            _mm256_store_ps(C_ptr + 3 * ldc, o3);
            _mm256_store_ps(C_ptr + 4 * ldc, o4);
            _mm256_store_ps(C_ptr + 5 * ldc, o5);
            _mm256_store_ps(C_ptr + 6 * ldc, o6);
            _mm256_store_ps(C_ptr + 7 * ldc, o7);
            _mm256_store_ps(C_ptr + 8 * ldc, o8);
        }
        else if (rem_n == 8)
        {
            __m256 o0, o1, o2, o3, o4, o5, o6, o7;
            const float *A_ptr = A + m * lda;
            const float *B_ptr = B + n * ldb;
            float *C_ptr = C + n * ldc + m;
            o0 = _mm256_load_ps     (C_ptr + 0 * ldc);
            o1 = _mm256_load_ps     (C_ptr + 1 * ldc);
            o2 = _mm256_load_ps     (C_ptr + 2 * ldc);
            o3 = _mm256_load_ps     (C_ptr + 3 * ldc);
            o4 = _mm256_load_ps     (C_ptr + 4 * ldc);
            o5 = _mm256_load_ps     (C_ptr + 5 * ldc);
            o6 = _mm256_load_ps     (C_ptr + 6 * ldc);
            o7 = _mm256_load_ps     (C_ptr + 7 * ldc);
            for (int kidx = 0; kidx < K; kidx++)
            {
                const __m256 A_col = _mm256_load_ps(A_ptr);
                A_ptr += 8;
                const __m256 B_val0 = _mm256_broadcast_ss(B_ptr + ldb*0 + kidx);
                const __m256 B_val1 = _mm256_broadcast_ss(B_ptr + ldb*1 + kidx);
                const __m256 B_val2 = _mm256_broadcast_ss(B_ptr + ldb*2 + kidx);
                const __m256 B_val3 = _mm256_broadcast_ss(B_ptr + ldb*3 + kidx);
                o0 = _mm256_fmadd_ps(A_col, B_val0, o0);
                o1 = _mm256_fmadd_ps(A_col, B_val1, o1);
                o2 = _mm256_fmadd_ps(A_col, B_val2, o2);
                o3 = _mm256_fmadd_ps(A_col, B_val3, o3);
                const __m256 B_val4 = _mm256_broadcast_ss(B_ptr + ldb*4 + kidx);
                const __m256 B_val5 = _mm256_broadcast_ss(B_ptr + ldb*5 + kidx);
                const __m256 B_val6 = _mm256_broadcast_ss(B_ptr + ldb*6 + kidx);
                const __m256 B_val7 = _mm256_broadcast_ss(B_ptr + ldb*7 + kidx);
                o4 = _mm256_fmadd_ps(A_col, B_val4, o4);
                o5 = _mm256_fmadd_ps(A_col, B_val5, o5);  
                o6 = _mm256_fmadd_ps(A_col, B_val6, o6);
                o7 = _mm256_fmadd_ps(A_col, B_val7, o7);
            }
            _mm256_store_ps(C_ptr + 0 * ldc, o0);
            _mm256_store_ps(C_ptr + 1 * ldc, o1);
            _mm256_store_ps(C_ptr + 2 * ldc, o2);
            _mm256_store_ps(C_ptr + 3 * ldc, o3);
            _mm256_store_ps(C_ptr + 4 * ldc, o4);
            _mm256_store_ps(C_ptr + 5 * ldc, o5);
            _mm256_store_ps(C_ptr + 6 * ldc, o6);
            _mm256_store_ps(C_ptr + 7 * ldc, o7);
        }
        else if (rem_n == 7)
        {
            __m256 o0, o1, o2, o3, o4, o5, o6;
            const float *A_ptr = A + m * lda;
            const float *B_ptr = B + n * ldb;
            float *C_ptr = C + n * ldc + m;
            o0 = _mm256_load_ps     (C_ptr + 0 * ldc);
            o1 = _mm256_load_ps     (C_ptr + 1 * ldc);
            o2 = _mm256_load_ps     (C_ptr + 2 * ldc);
            o3 = _mm256_load_ps     (C_ptr + 3 * ldc);
            o4 = _mm256_load_ps     (C_ptr + 4 * ldc);
            o5 = _mm256_load_ps     (C_ptr + 5 * ldc);
            o6 = _mm256_load_ps     (C_ptr + 6 * ldc);
            for (int kidx = 0; kidx < K; kidx++)
            {
                const __m256 A_col = _mm256_load_ps(A_ptr);
                A_ptr += 8;
                const __m256 B_val0 = _mm256_broadcast_ss(B_ptr + ldb*0 + kidx);
                const __m256 B_val1 = _mm256_broadcast_ss(B_ptr + ldb*1 + kidx);
                const __m256 B_val2 = _mm256_broadcast_ss(B_ptr + ldb*2 + kidx);
                const __m256 B_val3 = _mm256_broadcast_ss(B_ptr + ldb*3 + kidx);
                o0 = _mm256_fmadd_ps(A_col, B_val0, o0);
                o1 = _mm256_fmadd_ps(A_col, B_val1, o1);
                o2 = _mm256_fmadd_ps(A_col, B_val2, o2);
                o3 = _mm256_fmadd_ps(A_col, B_val3, o3);
                const __m256 B_val4 = _mm256_broadcast_ss(B_ptr + ldb*4 + kidx);
                const __m256 B_val5 = _mm256_broadcast_ss(B_ptr + ldb*5 + kidx);
                const __m256 B_val6 = _mm256_broadcast_ss(B_ptr + ldb*6 + kidx);
                o4 = _mm256_fmadd_ps(A_col, B_val4, o4);
                o5 = _mm256_fmadd_ps(A_col, B_val5, o5);  
                o6 = _mm256_fmadd_ps(A_col, B_val6, o6);
            }
            _mm256_store_ps(C_ptr + 0 * ldc, o0);
            _mm256_store_ps(C_ptr + 1 * ldc, o1);
            _mm256_store_ps(C_ptr + 2 * ldc, o2);
            _mm256_store_ps(C_ptr + 3 * ldc, o3);
            _mm256_store_ps(C_ptr + 4 * ldc, o4);
            _mm256_store_ps(C_ptr + 5 * ldc, o5);
            _mm256_store_ps(C_ptr + 6 * ldc, o6);
        }
        else if (rem_n == 6)
        {
            __m256 o0, o1, o2, o3, o4, o5;
            const float *A_ptr = A + m * lda;
            const float *B_ptr = B + n * ldb;
            float *C_ptr = C + n * ldc + m;
            o0 = _mm256_load_ps     (C_ptr + 0 * ldc);
            o1 = _mm256_load_ps     (C_ptr + 1 * ldc);
            o2 = _mm256_load_ps     (C_ptr + 2 * ldc);
            o3 = _mm256_load_ps     (C_ptr + 3 * ldc);
            o4 = _mm256_load_ps     (C_ptr + 4 * ldc);
            o5 = _mm256_load_ps     (C_ptr + 5 * ldc);
            for (int kidx = 0; kidx < K; kidx++)
            {
                const __m256 A_col = _mm256_load_ps(A_ptr);
                A_ptr += 8;
                const __m256 B_val0 = _mm256_broadcast_ss(B_ptr + ldb*0 + kidx);
                const __m256 B_val1 = _mm256_broadcast_ss(B_ptr + ldb*1 + kidx);
                const __m256 B_val2 = _mm256_broadcast_ss(B_ptr + ldb*2 + kidx);
                const __m256 B_val3 = _mm256_broadcast_ss(B_ptr + ldb*3 + kidx);
                const __m256 B_val4 = _mm256_broadcast_ss(B_ptr + ldb*4 + kidx);
                const __m256 B_val5 = _mm256_broadcast_ss(B_ptr + ldb*5 + kidx);
                o0 = _mm256_fmadd_ps(A_col, B_val0, o0);
                o1 = _mm256_fmadd_ps(A_col, B_val1, o1);
                o2 = _mm256_fmadd_ps(A_col, B_val2, o2);
                o3 = _mm256_fmadd_ps(A_col, B_val3, o3);
                o4 = _mm256_fmadd_ps(A_col, B_val4, o4);
                o5 = _mm256_fmadd_ps(A_col, B_val5, o5);  
            }
            _mm256_store_ps(C_ptr + 0 * ldc, o0);
            _mm256_store_ps(C_ptr + 1 * ldc, o1);
            _mm256_store_ps(C_ptr + 2 * ldc, o2);
            _mm256_store_ps(C_ptr + 3 * ldc, o3);
            _mm256_store_ps(C_ptr + 4 * ldc, o4);
            _mm256_store_ps(C_ptr + 5 * ldc, o5);
        }
        else if (rem_n == 5)
        {
            __m256 o0, o1, o2, o3, o4;
            const float *A_ptr = A + m * lda;
            const float *B_ptr = B + n * ldb;
            float *C_ptr = C + n * ldc + m;
            o0 = _mm256_load_ps     (C_ptr + 0 * ldc);
            o1 = _mm256_load_ps     (C_ptr + 1 * ldc);
            o2 = _mm256_load_ps     (C_ptr + 2 * ldc);
            o3 = _mm256_load_ps     (C_ptr + 3 * ldc);
            o4 = _mm256_load_ps     (C_ptr + 4 * ldc);
            for (int kidx = 0; kidx < K; kidx++)
            {
                const __m256 A_col = _mm256_load_ps(A_ptr);
                A_ptr += 8;
                const __m256 B_val0 = _mm256_broadcast_ss(B_ptr + ldb*0 + kidx);
                const __m256 B_val1 = _mm256_broadcast_ss(B_ptr + ldb*1 + kidx);
                const __m256 B_val2 = _mm256_broadcast_ss(B_ptr + ldb*2 + kidx);
                const __m256 B_val3 = _mm256_broadcast_ss(B_ptr + ldb*3 + kidx);
                const __m256 B_val4 = _mm256_broadcast_ss(B_ptr + ldb*4 + kidx);
                o0 = _mm256_fmadd_ps(A_col, B_val0, o0);
                o1 = _mm256_fmadd_ps(A_col, B_val1, o1);
                o2 = _mm256_fmadd_ps(A_col, B_val2, o2);
                o3 = _mm256_fmadd_ps(A_col, B_val3, o3);
                o4 = _mm256_fmadd_ps(A_col, B_val4, o4); 
            }
            _mm256_store_ps(C_ptr + 0 * ldc, o0);
            _mm256_store_ps(C_ptr + 1 * ldc, o1);
            _mm256_store_ps(C_ptr + 2 * ldc, o2);
            _mm256_store_ps(C_ptr + 3 * ldc, o3);
            _mm256_store_ps(C_ptr + 4 * ldc, o4);
        }
        else if (rem_n == 4)
        {
            __m256 o0, o1, o2, o3;
            const float *A_ptr = A + m * lda;
            const float *B_ptr = B + n * ldb;
            float *C_ptr = C + n * ldc + m;
            o0 = _mm256_load_ps     (C_ptr + 0 * ldc);
            o1 = _mm256_load_ps     (C_ptr + 1 * ldc);
            o2 = _mm256_load_ps     (C_ptr + 2 * ldc);
            o3 = _mm256_load_ps     (C_ptr + 3 * ldc);
            for (int kidx = 0; kidx < K; kidx++)
            {
                const __m256 A_col = _mm256_load_ps(A_ptr);
                A_ptr += 8;
                const __m256 B_val0 = _mm256_broadcast_ss(B_ptr + ldb*0 + kidx);
                const __m256 B_val1 = _mm256_broadcast_ss(B_ptr + ldb*1 + kidx);
                const __m256 B_val2 = _mm256_broadcast_ss(B_ptr + ldb*2 + kidx);
                const __m256 B_val3 = _mm256_broadcast_ss(B_ptr + ldb*3 + kidx);
                o0 = _mm256_fmadd_ps(A_col, B_val0, o0);
                o1 = _mm256_fmadd_ps(A_col, B_val1, o1);
                o2 = _mm256_fmadd_ps(A_col, B_val2, o2);
                o3 = _mm256_fmadd_ps(A_col, B_val3, o3);
            }
            _mm256_store_ps(C_ptr + 0 * ldc, o0);
            _mm256_store_ps(C_ptr + 1 * ldc, o1);
            _mm256_store_ps(C_ptr + 2 * ldc, o2);
            _mm256_store_ps(C_ptr + 3 * ldc, o3);
        }
        else if (rem_n == 3)
        {
            __m256 o0, o1, o2;
            const float *A_ptr = A + m * lda;
            const float *B_ptr = B + n * ldb;
            float *C_ptr = C + n * ldc + m;
            o0 = _mm256_load_ps     (C_ptr + 0 * ldc);
            o1 = _mm256_load_ps     (C_ptr + 1 * ldc);
            o2 = _mm256_load_ps     (C_ptr + 2 * ldc);
            for (int kidx = 0; kidx < K; kidx++)
            {
                const __m256 A_col = _mm256_load_ps(A_ptr);
                A_ptr += 8;
                const __m256 B_val0 = _mm256_broadcast_ss(B_ptr + ldb*0 + kidx);
                const __m256 B_val1 = _mm256_broadcast_ss(B_ptr + ldb*1 + kidx);
                const __m256 B_val2 = _mm256_broadcast_ss(B_ptr + ldb*2 + kidx);
                o0 = _mm256_fmadd_ps(A_col, B_val0, o0);
                o1 = _mm256_fmadd_ps(A_col, B_val1, o1);
                o2 = _mm256_fmadd_ps(A_col, B_val2, o2);
            }
            _mm256_store_ps(C_ptr + 0 * ldc, o0);
            _mm256_store_ps(C_ptr + 1 * ldc, o1);
            _mm256_store_ps(C_ptr + 2 * ldc, o2);
        }
        else if (rem_n == 2)
        {
            __m256 o0, o1;
            const float *A_ptr = A + m * lda;
            const float *B_ptr = B + n * ldb;
            float *C_ptr = C + n * ldc + m;
            o0 = _mm256_load_ps     (C_ptr + 0 * ldc);
            o1 = _mm256_load_ps     (C_ptr + 1 * ldc);
            for (int kidx = 0; kidx < K; kidx++)
            {
                const __m256 A_col = _mm256_load_ps(A_ptr);
                A_ptr += 8;
                const __m256 B_val0 = _mm256_broadcast_ss(B_ptr + ldb*0 + kidx);
                const __m256 B_val1 = _mm256_broadcast_ss(B_ptr + ldb*1 + kidx);
                o0 = _mm256_fmadd_ps(A_col, B_val0, o0);
                o1 = _mm256_fmadd_ps(A_col, B_val1, o1);
            }
            _mm256_store_ps(C_ptr + 0 * ldc, o0);
            _mm256_store_ps(C_ptr + 1 * ldc, o1);
        }
        else if (rem_n == 1)
        {
            __m256 o0;
            const float *A_ptr = A + m * lda;
            const float *B_ptr = B + n * ldb;
            float *C_ptr = C + n * ldc + m;
            o0 = _mm256_load_ps     (C_ptr + 0 * ldc);
            for (int kidx = 0; kidx < K; kidx++)
            {
                const __m256 A_col = _mm256_load_ps(A_ptr);
                A_ptr += 8;
                const __m256 B_val0 = _mm256_broadcast_ss(B_ptr + ldb*0 + kidx);
                o0 = _mm256_fmadd_ps(A_col, B_val0, o0);
            }
            _mm256_store_ps(C_ptr + 0 * ldc, o0);
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
    for (unsigned int n = 0; n < N - (N%_VEC_SIZE_N); n += _VEC_SIZE_N)
    {
        for (unsigned int m = M - (M%_VEC_SIZE_M); m < M; m++)
        {
            for (unsigned int nn = n; nn < n + _VEC_SIZE_N; nn++)
            {
                float c = C[nn * ldc + m];
                for (unsigned int k = 0; k < K; k++)
                {
                    c += A[((m/_VEC_SIZE_M) * lda + k) * _VEC_SIZE_M + (m%_VEC_SIZE_M)] * B[nn * ldb + k];
                }
                C[nn * ldc + m] = c;
            }
        }
    }
    for (unsigned int m = M - (M%_VEC_SIZE_M); m < M; m++)
    {
        for (unsigned int n = N - (N%_VEC_SIZE_N); n < N; n++)
        {
            float c = C[n * ldc + m];
            for (unsigned int k = 0; k < K; k++)
            {
                c += A[((m/_VEC_SIZE_M) * lda + k) * _VEC_SIZE_M + (m%_VEC_SIZE_M)] * B[n * ldb + k];
            }
            C[n * ldc + m] = c;
        }
    }
}   

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