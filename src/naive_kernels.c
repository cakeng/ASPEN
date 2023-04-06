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
                            for (unsigned int ic = 0; ic < input_channels; ic++)
                            {
                                unsigned int kernel_index = oc * kernel_width * kernel_height * input_channels + 
                                    kh * kernel_width * input_channels + 
                                    kw * input_channels + ic;
                                unsigned int ih = oh * stride + kh - padding;
                                unsigned int iw = ow * stride + kw - padding;
                                if (ih < height && iw < width)
                                {
                                    unsigned int input_index = b * width * height * input_channels + 
                                        ih * width * input_channels + 
                                        iw * input_channels + ic;
                                    output[output_index] += input[input_index] * kernel[kernel_index];
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