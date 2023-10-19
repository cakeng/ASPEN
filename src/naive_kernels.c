#include "kernels.h"

void naive_sigmoid (float *input, float *output, int size)
{
    for (int kidx = 0; kidx < size; kidx++)
    {
        *(output + kidx) = 1.0f/(1.0f + exp(-(*(input + kidx))));
    }
}

void naive_activate (float *input, unsigned int num_elements, LAYER_ACT activation_type)
{
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
    else if (activation_type == GELU)
    {
        for (unsigned int i = 0; i < num_elements; i++)
            input[i] = 0.5 * input[i] * (1 + erff ((input[i])*0.7071067811865475f));
    }
    else if (activation_type == GELU_ACCURATE)
    {
        for (unsigned int i = 0; i < num_elements; i++)
            input[i] = 0.5 * input[i] * (1 + tanhf (0.7978845608028654f * (input[i] + 0.044715 * input[i] * input[i] * input[i])));
    }
    else
        ERROR_PRTF ("Error in naive_activate: unknown activation type.\n");
}

// Input and output is in NHWC format. Kernel is in (O/8)HWI8 format.
void naive_conv2d
(const float *input, const float *kernel, const float *bias, float *output, 
    unsigned int batch_size, unsigned int input_channels, unsigned int height, unsigned int width,  
        unsigned int output_channels, unsigned int kernel_width , unsigned int kernel_height, 
            unsigned int stride, unsigned int padding)
{
    #ifdef DEBUG
    if (input == NULL)
        ERROR_PRTF ("Error in naive_convolution: input is NULL.\n");
    if (kernel == NULL)
        ERROR_PRTF ("Error in naive_convolution: kernel is NULL.\n");
    if (output == NULL)
        ERROR_PRTF ("Error in naive_convolution: output is NULL.\n");
    #endif
    unsigned int output_width = (width - kernel_width + 2 * padding) / stride + 1;
    unsigned int output_height = (height - kernel_height + 2 * padding) / stride + 1;

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
                                unsigned int kernel_index = ((oc/_VEC_SIZE_M) * kernel_width * kernel_height * input_channels + 
                                    kh * kernel_width * input_channels + 
                                    kw * input_channels) * _VEC_SIZE_M + (oc%_VEC_SIZE_M);
                                unsigned int input_index = b * width * height * input_channels + 
                                    ih * width * input_channels + 
                                    iw * input_channels;
                                for (unsigned int ic = 0; ic < input_channels; ic++)
                                {
                                    output[output_index] += input[input_index + ic] * kernel[kernel_index + ic* _VEC_SIZE_M];
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
(const float *input, const float *kernel, const float *bias, float *output, 
    unsigned int batch_size, unsigned int input_channels, unsigned int height, unsigned int width,  
        unsigned int output_channels, unsigned int kernel_width , unsigned int kernel_height, 
            unsigned int stride, unsigned int padding)
{
    #ifdef DEBUG
    if (input == NULL)
        ERROR_PRTF ("Error in naive_convolution_im2col_mm: input is NULL.\n");
    if (kernel == NULL)
        ERROR_PRTF ("Error in naive_convolution_im2col_mm: kernel is NULL.\n");
    if (output == NULL)
        ERROR_PRTF ("Error in naive_convolution_im2col_mm: output is NULL.\n");
    #endif
    unsigned int output_width = (width - kernel_width + 2 * padding) / stride + 1;
    unsigned int output_height = (height - kernel_height + 2 * padding) / stride + 1;

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
    if (bias != NULL)
    {
        for (int n = 0; n < N; n++)
        {
            memcpy (output + n * M, bias, M * sizeof (float));
        }
    }
    SGEMM_KERNEL_OMP (M, N, K, kernel, K, im2col_mat, K, output, M);
    aspen_free (im2col_mat);
}

void naive_maxpool2d
(const float *input, float *output, 
    unsigned int batch_size, unsigned int channels, unsigned int height, unsigned int width,  
        unsigned int kernel_height, unsigned int kernel_width, unsigned int stride, unsigned int padding)
{
    #ifdef DEBUG
    if (input == NULL)
        ERROR_PRTF ("Error in naive_maxpool2d: input is NULL.\n");
    if (output == NULL)
        ERROR_PRTF ("Error in naive_maxpool2d: output is NULL.\n");
    #endif
    unsigned int output_width = (width - kernel_width + 2 * padding) / stride + 1;
    unsigned int output_height = (height - kernel_height + 2 * padding) / stride + 1;

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
(const float *input, float *output, 
    unsigned int batch_size, unsigned int channels, unsigned int height, unsigned int width,  
        unsigned int kernel_height, unsigned int kernel_width, unsigned int stride, unsigned int padding)
{
    #ifdef DEBUG
    if (input == NULL)
        ERROR_PRTF ("Error in naive_avgpool2d: input is NULL.\n");
    if (output == NULL)
        ERROR_PRTF ("Error in naive_avgpool2d: output is NULL.\n");
    #endif
    unsigned int output_width = (width - kernel_width + 2 * padding) / stride + 1;
    unsigned int output_height = (height - kernel_height + 2 * padding) / stride + 1;

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
(const float *input, const float *kernel, const float *bias, float *output, 
    unsigned int batch_size, unsigned int input_size, unsigned int output_size)
{
    #ifdef DEBUG
    if (input == NULL)
        ERROR_PRTF ("Error in naive_fully_connected: input is NULL.\n");
    if (kernel == NULL)
        ERROR_PRTF ("Error in naive_fully_connected: kernel is NULL.\n");
    if (output == NULL)
        ERROR_PRTF ("Error in naive_fully_connected: output is NULL.\n");
    #endif
    #pragma omp parallel for collapse(2)
    for (unsigned int b = 0; b < batch_size; b++)
    {
        for (unsigned int oc = 0; oc < output_size; oc++)
        {
            unsigned int output_index = b * output_size + oc;
            for (unsigned int ic = 0; ic < input_size; ic++)
            {
                unsigned int input_index = b * input_size + ic;
                unsigned int kernel_index = 
                    ((oc/_VEC_SIZE_M) * input_size + ic)*_VEC_SIZE_M + (oc%_VEC_SIZE_M);
                output[output_index] += input[input_index] * kernel[kernel_index];
            }
            if (bias != NULL)
                output[output_index] += bias[oc];
        }
    }
}

void naive_residual
(const float *input_1, const float *input_2, float *output, unsigned int num_elements)
{
    #ifdef DEBUG
    if (input_1 == NULL)
        ERROR_PRTF ("Error in naive_residual: input_1 is NULL.\n");
    if (input_2 == NULL)
        ERROR_PRTF ("Error in naive_residual: input_2 is NULL.\n");
    if (output == NULL)
        ERROR_PRTF ("Error in naive_residual: output is NULL.\n");
    #endif
    for (unsigned int i = 0; i < num_elements; i++)
        output[i] = input_1[i] + input_2[i];
}

void naive_softmax (float *input, float *output, unsigned int num_batch, unsigned int num_elements)
{
    #ifdef DEBUG
    if (input == NULL)
        ERROR_PRTF ("Error in naive_softmax: input is NULL.\n");
    if (output == NULL)
        ERROR_PRTF ("Error in naive_softmax: output is NULL.\n");
    #endif
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
            output[i * num_elements + j] = expf (input[i * num_elements + j] - max);
            sum += output[i * num_elements + j];
        }
        for (int j = 0; j < num_elements; j++)
            output[i * num_elements + j] /= sum;
    }
}

void naive_layernorm (const float *input, const float *kernel, const float *bias, 
    float *output, unsigned int N, unsigned int M)
{
    #ifdef DEBUG
    if (input == NULL)
        ERROR_PRTF ("Error in naive_convolution: input is NULL.\n");
    if (kernel == NULL)
        ERROR_PRTF ("Error in naive_convolution: kernel is NULL.\n");
    if (output == NULL)
        ERROR_PRTF ("Error in naive_convolution: output is NULL.\n");
    #endif
    #pragma omp parallel for
    for (unsigned int i = 0; i < N; i++)
    {
        float mean = 0;
        float var = 0;
        for (unsigned int j = 0; j < M; j++)
        {
            mean += input[i * M + j];
            var += input[i * M + j] * input[i * M + j];
        }
        mean /= M;
        var /= M;
        var -= mean * mean;
        var = 1 / sqrtf (var + 1e-12);
        for (unsigned int j = 0; j < M; j++)
        {
            output[i * M + j] = (input[i * M + j] - mean) * var * kernel[j];
            if (bias != NULL)
                output[i * M + j] = output[i * M + j] + bias[j];
        }
    }
}

void naive_yolo (const float *input, const float *anchors, 
    float *output, unsigned int yolo_c, unsigned int h, unsigned int w, unsigned int c, unsigned int stride)
{
    #ifdef DEBUG
    if (input == NULL)
        ERROR_PRTF ("Error in naive_convolution: input is NULL.\n");
    if (anchors == NULL)
        ERROR_PRTF ("Error in naive_convolution: anchor is NULL.\n");
    if (output == NULL)
        ERROR_PRTF ("Error in naive_convolution: output is NULL.\n");
    if (c % yolo_c != 0)
        ERROR_PRTF ("Error in naive_convolution: c is not divisible by yolo_c.\n");
    #endif
    #pragma omp parallel for
    for (unsigned int i = 0; i < h*w*(c/yolo_c); i++)
    {
        int widx = (i/(c/yolo_c)) % w;
        int hidx = (i/(c/yolo_c)) / w;
        int aidx = i % (c/yolo_c);
        const float *in = input + i*yolo_c;
        float *out = output + (aidx*h*w + hidx*w + widx)*yolo_c;
        out[0] = (1.0f / (1.0f + expf (-in[0])) + widx) * stride;
        out[1] = (1.0f / (1.0f + expf (-in[1])) + hidx) * stride;
        out[2] = expf (in[2]) * anchors[2 * aidx];
        out[3] = expf (in[3]) * anchors[2 * aidx + 1];
        for (unsigned int j = 4; j < yolo_c; j++)
            out[j] = 1.0f / (1.0f + expf (-in[j]));
    }
}

void naive_append (const float *input_1, const float *input_2, float *output,
    const int stride, const int c1, const int c2, const int h2, const int w2)
{
    #ifdef DEBUG
    if (input_1 == NULL)
        ERROR_PRTF ("Error in naive_convolution: input_1 is NULL.\n");
    if (input_2 == NULL)
        ERROR_PRTF ("Error in naive_convolution: input_2 is NULL.\n");
    if (output == NULL)
        ERROR_PRTF ("Error in naive_convolution: output is NULL.\n");
    #endif
    #pragma omp parallel for
    for (unsigned int i = 0; i < h2*w2; i++)
    {
        const int widx_2 = i % w2;
        const int hidx_2 = i / w2;
        const int w1 = w2 / stride;
        const int widx_1 = widx_2 / stride;
        const int hidx_1 = hidx_2 / stride;
        const float *in1 = input_1 + (hidx_1*w1 + widx_1)*c1;
        const float *in2 = input_2 + (hidx_2*w2 + widx_2)*c2;
        float *out = output + (hidx_2*w2 + widx_2)*(c1+c2);
        memcpy (out, in1, c1*sizeof(float));
        memcpy (out+c1, in2, c2*sizeof(float));
    }
}

void naive_k_attention (const float *input_1, const float *input_2, float *output, unsigned int batch_size
    , unsigned int num_heads, unsigned int num_hidden, unsigned int num_seq, unsigned int masked)
{
    #ifdef DEBUG
    if (input_1 == NULL)
        ERROR_PRTF ("Error in naive_k_attention: input_1 is NULL.\n");
    if (input_2 == NULL)
        ERROR_PRTF ("Error in naive_k_attention: input_2 is NULL.\n");
    if (output == NULL)
        ERROR_PRTF ("Error in naive_k_attention: output is NULL.\n");
    #endif
    const unsigned int seq_padded = get_smallest_dividable (num_seq, _VEC_SIZE_M);
    const unsigned int hidden_per_head = num_hidden / num_heads;
    const unsigned int M = num_seq;
    const unsigned int N = num_seq;
    const unsigned int K = hidden_per_head;
    const unsigned int ldk = num_hidden;
    const unsigned int lda = K;
    const unsigned int ldb = num_hidden;
    const unsigned int ldc = seq_padded;
    float *key_temp = (float *) aspen_calloc (batch_size * num_heads * seq_padded * K, sizeof(float));

    #pragma omp parallel for collapse(2)
    for (unsigned int b = 0; b < batch_size; b++)
    {
        for (unsigned int h = 0; h < num_heads; h++)
        {
            const float *key_head = input_2 + b * num_hidden * num_seq + h * hidden_per_head;
            float *out_key_ptr = key_temp + b * num_heads * seq_padded * K + h * seq_padded * K;
            
            for (unsigned int m = 0; m < M; m++)
            {
                for (unsigned int k = 0; k < K; k++)
                {
                    const float* input_ptr = key_head + m * ldk + k;
                    float* output_ptr = out_key_ptr + ((m/_VEC_SIZE_M) * lda + k) * _VEC_SIZE_M 
                        + (m % _VEC_SIZE_M);
                    *output_ptr = *input_ptr;
                }
            }
            const float *B = input_1 + b * num_hidden * num_seq + h * hidden_per_head;
            const float *A = key_temp + b * num_heads * seq_padded * K + h * seq_padded * K;
            float *C = output + b * num_heads * ldc * N + h * ldc * N;
            SGEMM_KERNEL_OMP (M, N, K, A, lda, B, ldb, C, ldc);
            if (masked)
            {
                for (unsigned int n = 0; n < N; n++)
                {
                    for (unsigned int m = 0; m < M; m++)
                    {
                        if (m > n)
                            C[n*ldc + m] = -1e10;
                    }
                }
            }
            for (unsigned int i = 0; i < N; i++)
            {
                float total = 0;
                for (unsigned int j = 0; j < M; j++)
                {
                    C[i*ldc + j] /= sqrtf (hidden_per_head);
                    C[i*ldc + j] = expf (C[i*ldc + j]);
                    total += C[i*ldc + j];
                }
                for (unsigned int j = 0; j < M; j++)
                    C[i*ldc + j] /= total;
            }
        }
    }
    aspen_free (key_temp);
}

void naive_v_attention (const float *input_1, const float *input_2, float *output, unsigned int batch_size
    , unsigned int num_heads, unsigned int num_hidden, unsigned int num_seq)
{
    #ifdef DEBUG
    if (input_1 == NULL)
        ERROR_PRTF ("Error in naive_v_attention: input_1 is NULL.\n");
    if (input_2 == NULL)
        ERROR_PRTF ("Error in naive_v_attention: input_2 is NULL.\n");
    if (output == NULL)
        ERROR_PRTF ("Error in naive_v_attention: output is NULL.\n");
    #endif
    const unsigned int hidden_per_head = num_hidden / num_heads;
    const unsigned int seq_padded = get_smallest_dividable (num_seq, _VEC_SIZE_M);
    const unsigned int hph_padded = get_smallest_dividable (hidden_per_head, _VEC_SIZE_M);
    const unsigned int M = hidden_per_head;
    const unsigned int N = num_seq;
    const unsigned int K = num_seq;
    const unsigned int ldv = num_hidden;
    const unsigned int lda = K;
    const unsigned int ldb = seq_padded;
    const unsigned int ldc = num_hidden;
    float *val_temp = (float *) aspen_calloc (batch_size * num_heads * hph_padded * K, sizeof(float));

    #pragma omp parallel for collapse(2)
    for (unsigned int b = 0; b < batch_size; b++)
    {
        for (unsigned int h = 0; h < num_heads; h++)
        {
            const float *in_val_ptr = input_2 + b * num_hidden * num_seq + h * hidden_per_head;
            float *out_val_ptr = val_temp + b * num_heads * hph_padded * K + h * hph_padded * K;
            for (unsigned int m = 0; m < M; m++)
            {
                for (unsigned int k = 0; k < K; k++)
                {
                    const float* input_ptr = in_val_ptr + k * ldv + m;
                    float* output_ptr = out_val_ptr + ((m/_VEC_SIZE_M) * lda + k) * _VEC_SIZE_M 
                        + (m % _VEC_SIZE_M);
                    *output_ptr = *input_ptr;
                }
            }
            const float *B = input_1 + b * num_heads * ldb * N + h * ldb * N;
            const float *A = val_temp + b * num_heads * hph_padded * K + h * hph_padded * K;
            float *C = output + b * num_hidden * num_seq + h * hidden_per_head;
            SGEMM_KERNEL_OMP (M, N, K, A, lda, B, ldb, C, ldc);
        }
    }
    aspen_free (val_temp);
}

void naive_sgemm_with_omp(const unsigned int M, const unsigned int N, const unsigned int K,
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
                c += A[((m/_VEC_SIZE_M) * lda + k)*_VEC_SIZE_M + (m%_VEC_SIZE_M)] * B[n * ldb + k];
            }
            C[n * ldc + m] = c;
        }
    }
}   

void naive_sgemm(const unsigned int M, const unsigned int N, const unsigned int K,
		 const float *A, const unsigned int lda, const float *B, const unsigned int ldb, float *C, const unsigned int ldc)
{
    for (unsigned int n = 0; n < N; n++)
    {
        for (unsigned int m = 0; m < M; m++)
        {
            float c = C[n * ldc + m];
            for (unsigned int k = 0; k < K; k++)
            {
                c += A[((m/_VEC_SIZE_M) * lda + k)*_VEC_SIZE_M + (m%_VEC_SIZE_M)] * B[n * ldb + k];
            }
            C[n * ldc + m] = c;
        }
    }
}   

void naive_sgemm_vectorized_with_omp (const unsigned int M, const unsigned int N, const unsigned int K,
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
                        c += A[((mm/_VEC_SIZE_M) * lda + k) * _VEC_SIZE_M + (mm%_VEC_SIZE_M)] * B[nn * ldb + k];
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
                    c += A[((mm/_VEC_SIZE_M) * lda + k) * _VEC_SIZE_M + (mm%_VEC_SIZE_M)] * B[n * ldb + k];
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

void naive_sgemm_vectorized(const unsigned int M, const unsigned int N, const unsigned int K,
		 const float *A, const unsigned int lda, const float *B, const unsigned int ldb, float *C, const unsigned int ldc)
{
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
                        c += A[((mm/_VEC_SIZE_M) * lda + k) * _VEC_SIZE_M + (mm%_VEC_SIZE_M)] * B[nn * ldb + k];
                    }
                    C[nn * ldc + mm] = c;
                }
            }
        }
    }
    for (unsigned int m = 0; m < M - (M%_VEC_SIZE_M); m += _VEC_SIZE_M)
    {
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
