#include "kernels.h"

void naive_activate (float *input, unsigned int num_elements, LAYER_ACT activation_type)
{
    if (input == NULL)
        FPRT (stderr, "Error in naive_activate: input is NULL.\n");
    if (activation_type == NO_ACTIVATION || activation_type == LINEAR)
        return;
    if (activation_type == RELU)
    {
        for (unsigned int i = 0; i < num_elements; i++)
            input[i] = input[i] > 0 ? input[i] : 0;
    }
    else if (activation_type == LEAKY_RELU)
    {
        for (unsigned int i = 0; i < num_elements; i++)
            input[i] = input[i] > 0 ? input[i] : 0.1 * input[i];
    }
    else if (activation_type == ELU)
    {
        for (unsigned int i = 0; i < num_elements; i++)
            input[i] = input[i] > 0 ? input[i] : 0.1 * (exp (input[i]) - 1);
    }
    else if (activation_type == SELU)
    {
        for (unsigned int i = 0; i < num_elements; i++)
            input[i] = input[i] > 0 ? 1.0507 * input[i] : 1.0507 * 1.67326 * (exp (input[i]) - 1);
    }
    else if (activation_type == SIGMOID)
    {
        for (unsigned int i = 0; i < num_elements; i++)
            input[i] = 1 / (1 + exp (-input[i]));
    }
    else if (activation_type == TANH)
    {
        for (unsigned int i = 0; i < num_elements; i++)
            input[i] = tanh (input[i]);
    }
    else
        FPRT (stderr, "Error in naive_activate: unknown activation type.\n");
}

// Input and output is in NHWC format. Kernel is in OHWI format.
void naive_conv2d
(const float *input, const float *kernel, const float *bias, float **output_ptr, 
    unsigned int batch_size, unsigned int input_channels, unsigned int height, unsigned int width,  
        unsigned int output_channels, unsigned int kernel_width , unsigned int kernel_height, 
            unsigned int stride, unsigned int padding)
{
    if (input == NULL)
        FPRT (stderr, "Error in naive_convolution: input is NULL.\n");
    if (kernel == NULL)
        FPRT (stderr, "Error in naive_convolution: kernel is NULL.\n");
    if (output_ptr == NULL)
        FPRT (stderr, "Error in naive_convolution: output is NULL.\n");
    float *output = *output_ptr;
    unsigned int output_width = (width - kernel_width + 2 * padding) / stride + 1;
    unsigned int output_height = (height - kernel_height + 2 * padding) / stride + 1;
    unsigned int output_size = batch_size * output_width * output_height * output_channels;
    if (output == NULL)
        output = (float *) aspen_calloc (output_size, sizeof (float));
    else
        memset (output, 0, output_size * sizeof (float));

    #pragma omp parallel for collapse(3)
    for (unsigned int b = 0; b < batch_size; b++)
    {
        for (unsigned int oh = 0; oh < output_height; oh++)
        {
            for (unsigned int ow = 0; ow < output_width; ow++)
            {
                for (unsigned int oc = 0; oc < output_channels; oc++)
                {
                    unsigned int output_index = b * output_width * output_height * output_channels + 
                        oh * output_width * output_channels + 
                        ow * output_channels + oc;
                    for (unsigned int kh = 0; kh < kernel_height; kh++)
                    {
                        for (unsigned int kw = 0; kw < kernel_width; kw++)
                        {
                            int ih = oh * stride + kh - padding;
                            int iw = ow * stride + kw - padding;
                            if (ih >= 0 && ih < height && iw >= 0 && iw < width)
                            {
                                unsigned int kernel_index = oc * kernel_width * kernel_height * input_channels + 
                                    kh * kernel_width * input_channels + 
                                    kw * input_channels;
                                unsigned int input_index = b * width * height * input_channels + 
                                    ih * width * input_channels + 
                                    iw * input_channels;
                                for (unsigned int ic = 0; ic < input_channels; ic++)
                                {
                                    output[output_index] += input[input_index + ic] * kernel[kernel_index + ic];
                                }
                            }
                        }
                    }
                    if (bias != NULL)
                        output[output_index] += bias[oc];
                }
            }
        }
    }
}

void naive_conv2d_im2col_mm
(const float *input, const float *kernel, const float *bias, float **output_ptr, 
    unsigned int batch_size, unsigned int input_channels, unsigned int height, unsigned int width,  
        unsigned int output_channels, unsigned int kernel_width , unsigned int kernel_height, 
            unsigned int stride, unsigned int padding)
{
    if (input == NULL)
        FPRT (stderr, "Error in naive_convolution_im2col_mm: input is NULL.\n");
    if (kernel == NULL)
        FPRT (stderr, "Error in naive_convolution_im2col_mm: kernel is NULL.\n");
    if (output_ptr == NULL)
        FPRT (stderr, "Error in naive_convolution_im2col_mm: output is NULL.\n");
    float *output = *output_ptr;
    unsigned int output_width = (width - kernel_width + 2 * padding) / stride + 1;
    unsigned int output_height = (height - kernel_height + 2 * padding) / stride + 1;
    unsigned int output_size = batch_size * output_width * output_height * output_channels;
    if (output == NULL)
        output = (float *) aspen_calloc (output_size, sizeof (float));
    else
        memset (output, 0, output_size * sizeof (float));

    unsigned int im2col_size = batch_size * output_height * output_width * kernel_height * kernel_width * input_channels;
    float *im2col_mat = (float *) aspen_calloc (im2col_size, sizeof (float));
    unsigned M = output_channels, N = batch_size * output_height * output_width, K = kernel_height * kernel_width * input_channels;
    // im2col
    #pragma omp parallel for collapse(3)
    for (unsigned int b = 0; b < batch_size; b++)
    {
        for (unsigned int h = 0; h < output_height; h++)
        {
            for (unsigned int w = 0; w < output_width; w++)
            {
                float* im2col_col = im2col_mat + b * output_height * output_width * kernel_height * kernel_width * input_channels + 
                    h * output_width * kernel_height * kernel_width * input_channels + 
                    w * kernel_height * kernel_width * input_channels;
                for (unsigned int kh = 0; kh < kernel_height; kh++)
                {
                    for (unsigned int kw = 0; kw < kernel_width; kw++)
                    {
                        unsigned int ih = h * stride + kh - padding;
                        unsigned int iw = w * stride + kw - padding;
                        if (ih >= height || iw >= width)
                        {
                            im2col_col += input_channels;
                            continue;
                        }
                        const float *input_col = input + b * height * width * input_channels + 
                            (h * stride + kh - padding) * width * input_channels + 
                            (w * stride + kw - padding) * input_channels;
                        memcpy (im2col_col, input_col, input_channels * sizeof (float));
                        im2col_col += input_channels;
                    }
                }
            }
        }
    }
    for (int n = 0; n < N; n++)
    {
        memcpy (output + n * M, bias, M * sizeof (float));
    }
    // cblas_sgemm (CblasColMajor, CblasTrans, CblasNoTrans, M, N, K, 1.0, kernel, K, im2col_mat, K, 1.0, output, M);
    naive_sgemm_vectorized (M, N, K, kernel, K, im2col_mat, K, output, M);
    aspen_free (im2col_mat);
}

void naive_maxpool2d
(const float *input, float **output_ptr, 
    unsigned int batch_size, unsigned int channels, unsigned int height, unsigned int width,  
        unsigned int kernel_height, unsigned int kernel_width, unsigned int stride, unsigned int padding)
{
    if (input == NULL)
        FPRT (stderr, "Error in naive_maxpool2d: input is NULL.\n");
    if (output_ptr == NULL)
        FPRT (stderr, "Error in naive_maxpool2d: output is NULL.\n");
    float *output = *output_ptr;
    unsigned int output_width = (width - kernel_width + 2 * padding) / stride + 1;
    unsigned int output_height = (height - kernel_height + 2 * padding) / stride + 1;
    unsigned int output_size = batch_size * output_width * output_height * channels;
    if (output == NULL)
        output = (float *) aspen_calloc (output_size, sizeof (float));
    else
        memset (output, 0, output_size * sizeof (float));

    #pragma omp parallel for collapse(3)
    for (unsigned int b = 0; b < batch_size; b++)
    {
        for (unsigned int oh = 0; oh < output_height; oh++)
        {
            for (unsigned int ow = 0; ow < output_width; ow++)
            {
                for (unsigned int oc = 0; oc < channels; oc++)
                {
                    unsigned int output_index = b * output_width * output_height * channels + 
                        oh * output_width * channels + 
                        ow * channels + oc;
                    output[output_index] = -INFINITY;
                    for (unsigned int kh = 0; kh < kernel_height; kh++)
                    {
                        for (unsigned int kw = 0; kw < kernel_width; kw++)
                        {
                            unsigned int ih = oh * stride + kh - padding;
                            unsigned int iw = ow * stride + kw - padding;
                            if (ih < height && iw < width)
                            {
                                unsigned int input_index = b * width * height * channels + 
                                    ih * width * channels + 
                                    iw * channels + oc;
                                output[output_index] = 
                                    output[output_index] > input[input_index] ? 
                                        output[output_index] : input[input_index];
                            }
                        }
                    }
                }
            }
        }
    }
}

void naive_avgpool2d
(const float *input, float **output_ptr, 
    unsigned int batch_size, unsigned int channels, unsigned int height, unsigned int width,  
        unsigned int kernel_height, unsigned int kernel_width, unsigned int stride, unsigned int padding)
{
    if (input == NULL)
        FPRT (stderr, "Error in naive_avgpool2d: input is NULL.\n");
    if (output_ptr == NULL)
        FPRT (stderr, "Error in naive_avgpool2d: output is NULL.\n");
    float *output = *output_ptr;
    unsigned int output_width = (width - kernel_width + 2 * padding) / stride + 1;
    unsigned int output_height = (height - kernel_height + 2 * padding) / stride + 1;
    unsigned int output_size = batch_size * output_width * output_height * channels;
    if (output == NULL)
        output = (float *) aspen_calloc (output_size, sizeof (float));
    else
        memset (output, 0, output_size * sizeof (float));

    #pragma omp parallel for collapse(3)
    for (unsigned int b = 0; b < batch_size; b++)
    {
        for (unsigned int oh = 0; oh < output_height; oh++)
        {
            for (unsigned int ow = 0; ow < output_width; ow++)
            {
                for (unsigned int oc = 0; oc < channels; oc++)
                {
                    unsigned int output_index = b * output_width * output_height * channels + 
                        oh * output_width * channels + 
                        ow * channels + oc;
                    for (unsigned int kh = 0; kh < kernel_height; kh++)
                    {
                        for (unsigned int kw = 0; kw < kernel_width; kw++)
                        {
                            unsigned int ih = oh * stride + kh - padding;
                            unsigned int iw = ow * stride + kw - padding;
                            if (ih < height && iw < width)
                            {
                                unsigned int input_index = b * width * height * channels + 
                                    ih * width * channels + 
                                    iw * channels + oc;
                                output[output_index] += input[input_index];
                            }
                        }
                    }
                    output[output_index] /= (kernel_width * kernel_height);
                }
            }
        }
    }
}

void naive_fully_connected
(const float *input, const float *kernel, const float *bias, float **output_ptr, 
    unsigned int batch_size, unsigned int input_size, unsigned int output_size)
{
    if (input == NULL)
        FPRT (stderr, "Error in naive_fully_connected: input is NULL.\n");
    if (kernel == NULL)
        FPRT (stderr, "Error in naive_fully_connected: kernel is NULL.\n");
    if (output_ptr == NULL)
        FPRT (stderr, "Error in naive_fully_connected: output is NULL.\n");
    float *output = *output_ptr;
    if (output == NULL)
        output = (float *) aspen_calloc (batch_size * output_size, sizeof (float));
    else
        memset (output, 0, batch_size * output_size * sizeof (float));

    #pragma omp parallel for collapse(2)
    for (unsigned int b = 0; b < batch_size; b++)
    {
        for (unsigned int oc = 0; oc < output_size; oc++)
        {
            unsigned int output_index = b * output_size + oc;
            for (unsigned int ic = 0; ic < input_size; ic++)
            {
                unsigned int input_index = b * input_size + ic;
                unsigned int kernel_index = oc * input_size + ic;
                output[output_index] += input[input_index] * kernel[kernel_index];
            }
            if (bias != NULL)
                output[output_index] += bias[oc];
        }
    }
}

void naive_residual
(const float *input_1, const float *input_2, float **output_ptr, unsigned int num_elements)
{
    if (input_1 == NULL)
        FPRT (stderr, "Error in naive_residual: input_1 is NULL.\n");
    if (input_2 == NULL)
        FPRT (stderr, "Error in naive_residual: input_2 is NULL.\n");
    if (output_ptr == NULL)
        FPRT (stderr, "Error in naive_residual: output is NULL.\n");
    float *output = *output_ptr;
    if (output == NULL)
        output = (float *) aspen_calloc (num_elements, sizeof (float));
    else
        memset (output, 0, num_elements * sizeof (float));
    for (unsigned int i = 0; i < num_elements; i++)
        output[i] = input_1[i] + input_2[i];
}

void naive_softmax (float *input, float **output_ptr, unsigned int num_batch, unsigned int num_elements)
{
    if (input == NULL)
        FPRT (stderr, "Error in naive_softmax: input is NULL.\n");
    if (output_ptr == NULL)
        FPRT (stderr, "Error in naive_softmax: output is NULL.\n");
    float *output = *output_ptr;
    if (output == NULL)
        output = (float *) aspen_calloc (num_batch*num_elements, sizeof (float));
    else
        memset (output, 0, num_batch * num_elements * sizeof (float));
    for (int i = 0; i < num_batch; i++)
    {
        float max = input[i * num_elements];
        for (int j = 1; j < num_elements; j++)
        {
            if (input[i * num_elements + j] > max)
                max = input[i * num_elements + j];
        }
        float sum = 0;
        for (int j = 0; j < num_elements; j++)
        {
            output[i * num_elements + j] = exp (input[i * num_elements + j] - max);
            sum += output[i * num_elements + j];
        }
        for (int j = 0; j < num_elements; j++)
            output[i * num_elements + j] /= sum;
    }
}

void naive_sgemm(const unsigned int M, const unsigned int N, const unsigned int K,
		 const float *A, const unsigned int lda, const float *B, const unsigned int ldb, float *C, const unsigned int ldc)
{
    #pragma omp parallel for collapse(2)
    for (unsigned int n = 0; n < N; n++)
    {
        for (unsigned int m = 0; m < M; m++)
        {
            float c = C[n * ldc + m];
            for (unsigned int k = 0; k < K; k++)
            {
                c += A[m * lda + k] * B[n * ldb + k];
            }
            C[n * ldc + m] = c;
        }
    }
}   

void naive_sgemm_without_omp(const unsigned int M, const unsigned int N, const unsigned int K,
		 const float *A, const unsigned int lda, const float *B, const unsigned int ldb, float *C, const unsigned int ldc)
{
    for (unsigned int n = 0; n < N; n++)
    {
        for (unsigned int m = 0; m < M; m++)
        {
            float c = C[n * ldc + m];
            for (unsigned int k = 0; k < K; k++)
            {
                c += A[m * lda + k] * B[n * ldb + k];
            }
            C[n * ldc + m] = c;
        }
    }
}   

void naive_sgemm_vectorized(const unsigned int M, const unsigned int N, const unsigned int K,
		 const float *A, const unsigned int lda, const float *B, const unsigned int ldb, float *C, const unsigned int ldc)
{
    // 8 by 8 tiled matrix multiplication
    #pragma omp parallel for collapse(2)
    for (unsigned int n = 0; n < N - (N%_VEC_SIZE_N); n += _VEC_SIZE_N)
    {
        for (unsigned int m = 0; m < M - (M%_VEC_SIZE_M); m += _VEC_SIZE_M)
        {
            for (unsigned int nn = n; nn < n + _VEC_SIZE_N; nn++)
            {
                for (unsigned int mm = m; mm < m + _VEC_SIZE_M; mm++)
                {
                    float c = C[nn * ldc + mm];
                    for (unsigned int k = 0; k < K; k++)
                    {
                        c += A[mm * lda + k] * B[nn * ldb + k];
                    }
                    C[nn * ldc + mm] = c;
                }
            }
        }
    }
    #pragma omp parallel for
    for (unsigned int m = 0; m < M - (M%_VEC_SIZE_M); m += _VEC_SIZE_M)
    {
        for (unsigned int n = N - (N%_VEC_SIZE_N); n < N; n++)
        {
            for (unsigned int mm = m; mm < m + _VEC_SIZE_M; mm++)
            {
                float c = C[n * ldc + mm];
                for (unsigned int k = 0; k < K; k++)
                {
                    c += A[mm * lda + k] * B[n * ldb + k];
                }
                C[n * ldc + mm] = c;
            }
        }
    }
    #pragma omp parallel for
    for (unsigned int n = 0; n < N - (N%_VEC_SIZE_N); n += _VEC_SIZE_N)
    {
        for (unsigned int m = M - (M%_VEC_SIZE_M); m < M; m++)
        {
            for (unsigned int nn = n; nn < n + _VEC_SIZE_N; nn++)
            {
                float c = C[nn * ldc + m];
                for (unsigned int k = 0; k < K; k++)
                {
                    c += A[m * lda + k] * B[nn * ldb + k];
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
                c += A[m * lda + k] * B[n * ldb + k];
            }
            C[n * ldc + m] = c;
        }
    }
}   

void naive_sgemm_tiled(const unsigned int M, const unsigned int N, const unsigned int K,
		 const float *A, const unsigned int lda, const float *B, const unsigned int ldb, float *C, const unsigned int ldc)
{
    // 8 by 8 tiled matrix multiplication
    #pragma omp parallel for collapse(2)
    for (unsigned int n = 0; n < N/_TILE_SIZE_N; n++)
    {
        for (unsigned int m = 0; m < M/_TILE_SIZE_M; m++)
        {
            for (unsigned int k = 0; k < K/_TILE_SIZE_K; k++)
            {
                for (unsigned int nn = n*_TILE_SIZE_N; nn < (n + 1)*_TILE_SIZE_N; nn += _VEC_SIZE_N)
                {
                    for (unsigned int mm = m*_TILE_SIZE_M; mm < (m + 1)*_TILE_SIZE_M; mm += _VEC_SIZE_M)
                    {
                        for (unsigned int kk = k*_TILE_SIZE_K; kk < (k + 1)*_TILE_SIZE_K; kk += _VEC_SIZE_K)
                        {
                            for (unsigned int i = 0; i < _VEC_SIZE_M; i++)
                            {
                                for (unsigned int j = 0; j < _VEC_SIZE_N; j++)
                                {
                                    float c = C[(nn + j) * ldc + (mm + i)];
                                    for (unsigned int l = 0; l < _VEC_SIZE_K; l++)
                                    {
                                        c += A[(mm + i) * lda + (kk + l)] * B[(nn + j) * ldb + (kk + l)];
                                    }
                                    C[(nn + j) * ldc + (mm + i)] = c;
                                }
                            }
                        }
                    }
                }
            }
            for (unsigned int nn = n*_TILE_SIZE_N; nn < (n + 1)*_TILE_SIZE_N; nn += _VEC_SIZE_N)
            {
                for (unsigned int mm = m*_TILE_SIZE_M; mm < (m + 1)*_TILE_SIZE_M; mm += _VEC_SIZE_M)
                {
                    for (unsigned int i = 0; i < _VEC_SIZE_M; i++)
                    {
                        for (unsigned int j = 0; j < _VEC_SIZE_N; j++)
                        {
                            float c = C[(nn + j) * ldc + (mm + i)];
                            for (unsigned int k = K - K%_TILE_SIZE_K; k < K; k++)
                            {
                                c += A[(mm + i) * lda + k] * B[(nn + j) * ldb + k];
                            }
                            C[(nn + j) * ldc + (mm + i)] = c;
                        }
                    }
                }
            }
        }
    }
    #pragma omp parallel for
    for (unsigned int n = N - N%_TILE_SIZE_N; n < N - (N%_VEC_SIZE_N); n += _VEC_SIZE_N)
    {
        for (unsigned int m = 0; m < M - (M%_VEC_SIZE_M); m += _VEC_SIZE_M)
        {
            for (unsigned int k = 0; k < K/_TILE_SIZE_K; k++)
            {
                for (unsigned int kk = k*_TILE_SIZE_K; kk < (k + 1)*_TILE_SIZE_K; kk += _VEC_SIZE_K)
                {
                    for (unsigned int i = 0; i < _VEC_SIZE_M; i++)
                    {
                        for (unsigned int j = 0; j < _VEC_SIZE_N; j++)
                        {
                            float c = C[(n + j) * ldc + (m + i)];
                            for (unsigned int l = 0; l < _VEC_SIZE_K; l++)
                            {
                                c += A[(m + i) * lda + (kk + l)] * B[(n + j) * ldb + (kk + l)];
                            }
                            C[(n + j) * ldc + (m + i)] = c;
                        }
                    }
                }
            }
            for (unsigned int i = 0; i < _VEC_SIZE_M; i++)
            {
                for (unsigned int j = 0; j < _VEC_SIZE_N; j++)
                {
                    float c = C[(n + j) * ldc + (m + i)];
                    for (unsigned int k = K - K%_TILE_SIZE_K; k < K; k++)
                    {
                        c += A[(m + i) * lda + k] * B[(n + j) * ldb + k];
                    }
                    C[(n + j) * ldc + (m + i)] = c;
                }
            }
        }
    }
    for (unsigned int n = N - (N%_VEC_SIZE_N); n < N; n++)
    {
        for (unsigned int m = 0; m < M; m++)
        {
            float c = C[n * ldc + m];
            for (unsigned int k = 0; k < K; k++)
            {
                c += A[m * lda + k] * B[n * ldb + k];
            }
            C[n * ldc + m] = c;
        }
    }
    #pragma omp parallel for
    for (unsigned int n = 0; n < N/_TILE_SIZE_N; n++)
    {
        for (unsigned int m = M - M%_TILE_SIZE_M; m < M - (M%_VEC_SIZE_M); m += _VEC_SIZE_M)
        {
            for (unsigned int k = 0; k < K/_TILE_SIZE_K; k++)
            {
                for (unsigned int nn = n*_TILE_SIZE_N; nn < (n + 1)*_TILE_SIZE_N; nn += _VEC_SIZE_N)
                {
                    for (unsigned int kk = k*_TILE_SIZE_K; kk < (k + 1)*_TILE_SIZE_K; kk += _VEC_SIZE_K)
                    {
                        for (unsigned int i = 0; i < _VEC_SIZE_M; i++)
                        {
                            for (unsigned int j = 0; j < _VEC_SIZE_N; j++)
                            {
                                float c = C[(nn + j) * ldc + (m + i)];
                                for (unsigned int l = 0; l < _VEC_SIZE_K; l++)
                                {
                                    c += A[(m + i) * lda + (kk + l)] * B[(nn + j) * ldb + (kk + l)];
                                }
                                C[(nn + j) * ldc + (m + i)] = c;
                            }
                        }
                    }
                }
            }
            for (unsigned int nn = n*_TILE_SIZE_N; nn < (n + 1)*_TILE_SIZE_N; nn += _VEC_SIZE_N)
            {
                for (unsigned int i = 0; i < _VEC_SIZE_M; i++)
                {
                    for (unsigned int j = 0; j < _VEC_SIZE_N; j++)
                    {
                        float c = C[(nn + j) * ldc + (m + i)];
                        for (unsigned int k = K - K%_TILE_SIZE_K; k < K; k++)
                        {
                            c += A[(m + i) * lda + k] * B[(nn + j) * ldb + k];
                        }
                        C[(nn + j) * ldc + (m + i)] = c;
                    }
                }
            }
        }
    }
    for (unsigned int n = 0; n < N; n++)
    {
        for (unsigned int m = M - M%_VEC_SIZE_M; m < M; m++)
        {
            float c = C[n * ldc + m];
            for (unsigned int k = 0; k < K; k++)
            {
                c += A[m * lda + k] * B[n * ldb + k];
            }
            C[n * ldc + m] = c;
        }
    }
}   

void naive_sgemm_vectorized_without_omp(const unsigned int M, const unsigned int N, const unsigned int K,
		 const float *A, const unsigned int lda, const float *B, const unsigned int ldb, float *C, const unsigned int ldc)
{
    for (unsigned int n = 0; n < N; n += _VEC_SIZE_N)
    {
        for (unsigned int m = 0; m < - (M%_VEC_SIZE_M); m += _VEC_SIZE_M)
        {
            for (unsigned int k = 0; k < K - (K%_VEC_SIZE_K); k += _VEC_SIZE_K)
            {
                for (unsigned int j = 0; j < _VEC_SIZE_N; j++)
                {
                    if (n + j >= N)
                        break;
                    for (unsigned int i = 0; i < _VEC_SIZE_M; i++)
                    {
                        float c = C[(n + j) * ldc + (m + i)];
                        for (unsigned int l = 0; l < _VEC_SIZE_K; l++)
                        {
                            c += A[(m + i) * lda + (k + l)] * B[(n + j) * ldb + (k + l)];
                        }
                        C[(n + j) * ldc + (m + i)] = c;
                    }
                }
            }
            for (unsigned int i = 0; i < _VEC_SIZE_M; i++)
            {
                if (m + i >= M)
                    break;
                for (unsigned int j = 0; j < _VEC_SIZE_N; j++)
                {
                    float c = C[(n + j) * ldc + (m + i)];
                    for (unsigned int k = K - K%_VEC_SIZE_K; k < K; k++)
                    {
                        c += A[(m + i) * lda + k] * B[(n + j) * ldb + k];
                    }
                    C[(n + j) * ldc + (m + i)] = c;
                }
            }
        }
        for (unsigned int m = M - M%_VEC_SIZE_M; m < M; m++)
        {
            for (unsigned int j = 0; j < _VEC_SIZE_N; j++)
            {
                if (n + j >= N)
                    break;
                float c = C[(n + j) * ldc + m];
                for (unsigned int k = 0; k < K; k++)
                {
                    c += A[m * lda + k] * B[(n + j) * ldb + k];
                }
                C[(n + j) * ldc + m] = c;
            }
        }
    }
}   

void matmul_f32_base(float *A, float *B, float **C, int k, int m, int n)
{
    for (int kidx = 0; kidx < k; kidx++)
    {
        for (int nidx = 0; nidx < n; nidx++)
        {
            for (int midx = 0; midx < m; midx++)
            {
                *(*(C + nidx) + midx) += A[kidx*m + midx] * (*(B + k*nidx + kidx));
            }
        }
    }
}
void matmul_f32_base_8x1(float *A, float *B, float **C, int k)
{
    for (int kidx = 0; kidx < k; kidx++)
    {
        for (int nidx = 0; nidx < 1; nidx++)
        {
            for (int midx = 0; midx < 8; midx++)
            {
                *(*(C + nidx) + midx) += A[kidx*8 + midx] * (*(B + k*nidx + kidx));
            }
        }
    }
}
void matmul_f32_base_8x2(float *A, float *B, float **C, int k)
{
    for (int kidx = 0; kidx < k; kidx++)
    {
        for (int nidx = 0; nidx < 2; nidx++)
        {
            for (int midx = 0; midx < 8; midx++)
            {
                *(*(C + nidx) + midx) += A[kidx*8 + midx] * (*(B + k*nidx + kidx));
            }
        }
    }
}
void matmul_f32_base_8x3(float *A, float *B, float **C, int k)
{
    for (int kidx = 0; kidx < k; kidx++)
    {
        for (int nidx = 0; nidx < 3; nidx++)
        {
            for (int midx = 0; midx < 8; midx++)
            {
                *(*(C + nidx) + midx) += A[kidx*8 + midx] * (*(B + k*nidx + kidx));
            }
        }
    }
}
void matmul_f32_base_8x4(float *A, float *B, float **C, int k)
{
    for (int kidx = 0; kidx < k; kidx++)
    {
        for (int nidx = 0; nidx < 4; nidx++)
        {
            for (int midx = 0; midx < 8; midx++)
            {
                *(*(C + nidx) + midx) += A[kidx*8 + midx] * (*(B + k*nidx + kidx));
            }
        }
    }
}
void matmul_f32_base_8x5(float *A, float *B, float **C, int k)
{
    for (int kidx = 0; kidx < k; kidx++)
    {
        for (int nidx = 0; nidx < 5; nidx++)
        {
            for (int midx = 0; midx < 8; midx++)
            {
                *(*(C + nidx) + midx) += A[kidx*8 + midx] * (*(B + k*nidx + kidx));
            }
        }
    }
}
void matmul_f32_base_8x6(float *A, float *B, float **C, int k)
{
    for (int kidx = 0; kidx < k; kidx++)
    {
        for (int nidx = 0; nidx < 6; nidx++)
        {
            for (int midx = 0; midx < 8; midx++)
            {
                *(*(C + nidx) + midx) += A[kidx*8 + midx] * (*(B + k*nidx + kidx));
            }
        }
    }
}
void matmul_f32_base_8x7(float *A, float *B, float **C, int k)
{
    for (int kidx = 0; kidx < k; kidx++)
    {
        for (int nidx = 0; nidx < 7; nidx++)
        {
            for (int midx = 0; midx < 8; midx++)
            {
                *(*(C + nidx) + midx) += A[kidx*8 + midx] * (*(B + k*nidx + kidx));
            }
        }
    }
}
void matmul_f32_base_8x8(float *A, float *B, float **C, int k)
{
    for (int kidx = 0; kidx < k; kidx++)
    {
        for (int nidx = 0; nidx < 8; nidx++)
        {
            for (int midx = 0; midx < 8; midx++)
            {
                *(*(C + nidx) + midx) += A[kidx*8 + midx] * (*(B + k*nidx + kidx));
            }
        }
    }
}
void matmul_f32_base_8x9(float *A, float *B, float **C, int k)
{
    for (int kidx = 0; kidx < k; kidx++)
    {
        for (int nidx = 0; nidx < 9; nidx++)
        {
            for (int midx = 0; midx < 8; midx++)
            {
                *(*(C + nidx) + midx) += A[kidx*8 + midx] * (*(B + k*nidx + kidx));
            }
        }
    }
}
void matmul_f32_base_8x10(float *A, float *B, float **C, int k)
{
    for (int kidx = 0; kidx < k; kidx++)
    {
        for (int nidx = 0; nidx < 10; nidx++)
        {
            for (int midx = 0; midx < 8; midx++)
            {
                *(*(C + nidx) + midx) += A[kidx*8 + midx] * (*(B + k*nidx + kidx));
            }
        }
    }
}
void matmul_f32_base_8x11(float *A, float *B, float **C, int k)
{
    for (int kidx = 0; kidx < k; kidx++)
    {
        for (int nidx = 0; nidx < 11; nidx++)
        {
            for (int midx = 0; midx < 8; midx++)
            {
                *(*(C + nidx) + midx) += A[kidx*8 + midx] * (*(B + k*nidx + kidx));
            }
        }
    }
}
void matmul_f32_base_8x12(float *A, float *B, float **C, int k)
{
    for (int kidx = 0; kidx < k; kidx++)
    {
        for (int nidx = 0; nidx < 12; nidx++)
        {
            for (int midx = 0; midx < 8; midx++)
            {
                *(*(C + nidx) + midx) += A[kidx*8 + midx] * (*(B + k*nidx + kidx));
            }
        }
    }
}

void matmul_f32_base_16x1(float *A, float *B, float **C, int k)
{
    for (int kidx = 0; kidx < k; kidx++)
    {
        for (int nidx = 0; nidx < 1; nidx++)
        {
            for (int midx = 0; midx < 16; midx++)
            {
                *(*(C + nidx) + midx) += A[kidx*8 + (midx/8)*k*8 + (midx%8)] * (*(B + k*nidx + kidx));
            }
        }
    }
}
void matmul_f32_base_32x1(float *A, float *B, float **C, int k)
{
    for (int kidx = 0; kidx < k; kidx++)
    {
        for (int nidx = 0; nidx < 1; nidx++)
        {
            for (int midx = 0; midx < 32; midx++)
            {
                *(*(C + nidx) + midx) += A[kidx*8 + (midx/8)*k*8 + (midx%8)] * (*(B + k*nidx + kidx));
            }
        }
    }
}
void matmul_f32_base_64x1(float *A, float *B, float **C, int k)
{
    for (int kidx = 0; kidx < k; kidx++)
    {
        for (int nidx = 0; nidx < 1; nidx++)
        {
            for (int midx = 0; midx < 64; midx++)
            {
                *(*(C + nidx) + midx) += A[kidx*8 + (midx/8)*k*8 + (midx%8)] * (*(B + k*nidx + kidx));
            }
        }
    }
}

void maxpool2d_f32_base (float **input, float *output, int kernel_size, int cin)
{
    for (int kidx = 0; kidx < cin; kidx++)
    {
        float val = -INFINITY;
        for (int i = 0; i < kernel_size; i++)
        {
            float* input_ptr = *(input + i);
            if (val < *(input_ptr + kidx))
            {
                val = *(input_ptr + kidx);
            }
        }
        *(output + kidx) = val;
    }
}

void avgpool2d_f32_base (float **input, float *output, int kernel_size, int cin)
{
    for (int kidx = 0; kidx < cin; kidx++)
    {
        float val = 0.0f;
        for (int i = 0; i < kernel_size; i++)
        {
            val += *(*(input + i) + kidx);
        }
        *(output + kidx) = val/kernel_size;
    }
}

void residual_f32_base (float **input, float *output, int cin)
{
    for (int kidx = 0; kidx < cin; kidx++)
    {
        *(output + kidx) = *(*(input + 0) + kidx) + *(*(input + 1) + kidx);
    }
}
