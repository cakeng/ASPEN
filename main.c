#include "aspen.h"
#include "apu.h"
#include "nasm.h"
#include "dse.h"

double get_sec()
{
    struct timeval now;
    gettimeofday (&now, NULL);
    return now.tv_sec + now.tv_usec*1e-6;
}

void softmax (float *input, float *output, unsigned int num_batch, unsigned int num_elements)
{
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

void get_prob_results (char *class_data_path, float* probabilities, unsigned int num)
{
    int buffer_length = 256;
    char buffer[num][buffer_length];
    FILE *fptr = fopen(class_data_path, "r");
    if (fptr == NULL)
        assert (0);
    for (int i = 0; i < num; i++)
    {
        void *tmp = fgets(buffer[i], buffer_length, fptr);
        if (tmp == NULL)
            assert (0);
        for (char *ptr = buffer[i]; *ptr != '\0'; ptr++)
        {
            if (*ptr == '\n')
            {
                *ptr = '\0';
            }
        }
    }
    fclose(fptr);
    printf ("Results:\n");
    for (int i = 0; i < 5; i++)
    {
        float max_val = -INFINITY;
        int max_idx = 0;
        for (int j = 0; j < num; j++)
        {
            if (max_val < *(probabilities + j))
            {
                max_val = *(probabilities + j);
                max_idx = j;
            }
        }
        printf ("%d: %s - %2.2f%%\n", i+1, buffer[max_idx], max_val*100);
        *(probabilities + max_idx) = -INFINITY;
    }
}

int main(void)
{
    print_aspen_build_info();
    
    int number_of_iterations = 20;
    int num_cores = 32;
    int gpu_idx = -1;
    char nasm_file_name [1024] = {0};

    aspen_dnn_t *resnet50_dnn = apu_create_dnn("data/cfg/resnet50_aspen.cfg", "data/resnet50_data.bin");
    apu_save_dnn_to_file (resnet50_dnn, "nasms/resnet50_base.aspen");
    nasm_t *resnet50_nasm;
    for (int batch_size = 1; batch_size < 33; batch_size++)
    {
        printf ("Generating Resnet 50 NASM for batch size %d\n", batch_size);
        resnet50_nasm = apu_generate_nasm (resnet50_dnn, batch_size, number_of_iterations, gpu_idx);
        sprintf (nasm_file_name, "nasms/resnet50_B%d_CPU.nasm", batch_size);
        apu_save_nasm_to_file (resnet50_nasm, nasm_file_name);
    }

    aspen_dnn_t *vgg16_dnn = apu_create_dnn("data/cfg/vgg16_aspen.cfg", "data/vgg16_data.bin");
    apu_save_dnn_to_file (vgg16_dnn, "nasms/vgg16_base.aspen");
    nasm_t *vgg16_nasm;
    for (int batch_size = 1; batch_size < 17; batch_size++)
    {
        printf ("Generating VGG16 NASM for batch size %d\n", batch_size);
        vgg16_nasm = apu_generate_nasm (vgg16_dnn, batch_size, number_of_iterations, gpu_idx);
        sprintf (nasm_file_name, "nasms/vgg16_B%d_CPU.nasm", batch_size);
        apu_save_nasm_to_file (vgg16_nasm, nasm_file_name);
    }

    aspen_dnn_t *yolov3_dnn = apu_create_dnn("data/cfg/yolov3_aspen.cfg", "data/yolov3_data.bin");
    apu_save_dnn_to_file (yolov3_dnn, "nasms/yolov3_base.aspen");
    nasm_t *yolov3_nasm;
    for (int batch_size = 1; batch_size < 17; batch_size++)
    {
        printf ("Generating YOLOv3 NASM for batch size %d\n", batch_size);
        yolov3_nasm = apu_generate_nasm (yolov3_dnn, batch_size, number_of_iterations, gpu_idx);
        sprintf (nasm_file_name, "nasms/yolov3_B%d_CPU.nasm", batch_size);
        apu_save_nasm_to_file (yolov3_nasm, nasm_file_name);
    }

    aspen_dnn_t *bert_dnn = apu_create_dnn("data/cfg/bert_base_encoder.cfg", "data/bert_base_data.bin");
    apu_save_dnn_to_file (bert_dnn, "nasms/bert_base.aspen");
    nasm_t *bert_nasm;
    for (int batch_size = 1; batch_size < 5; batch_size++)
    {
        for (int seq_len = 1; seq_len < 33; seq_len++)
        {
            printf ("Generating BERT NASM for batch size %d and sequence length %d\n", batch_size, seq_len);
            bert_nasm = apu_generate_transformer_nasm (bert_dnn, batch_size, seq_len, number_of_iterations, gpu_idx);
            sprintf (nasm_file_name, "nasms/bert_S%d_B%d_CPU.nasm", seq_len, batch_size);
            apu_save_nasm_to_file (bert_nasm, nasm_file_name);
        }
    }
    for (int batch_size = 1; batch_size < 2; batch_size++)
    {
        for (int seq_len = 32; seq_len < 128; seq_len += 2)
        {
            printf ("Generating BERT NASM for batch size %d and sequence length %d\n", batch_size, seq_len);
            bert_nasm = apu_generate_transformer_nasm (bert_dnn, batch_size, seq_len, number_of_iterations, gpu_idx);
            sprintf (nasm_file_name, "nasms/bert_S%d_B%d_CPU.nasm", seq_len, batch_size);
            apu_save_nasm_to_file (bert_nasm, nasm_file_name);
        }
        for (int seq_len = 128; seq_len < 513; seq_len += 4)
        {
            printf ("Generating BERT NASM for batch size %d and sequence length %d\n", batch_size, seq_len);
            bert_nasm = apu_generate_transformer_nasm (bert_dnn, batch_size, seq_len, number_of_iterations, gpu_idx);
            sprintf (nasm_file_name, "nasms/bert_S%d_B%d_CPU.nasm", seq_len, batch_size);
            apu_save_nasm_to_file (bert_nasm, nasm_file_name);
        }
    }

    aspen_dnn_t *bert_large_dnn = apu_create_dnn("data/cfg/bert_large_encoder.cfg", "data/bert_large_data.bin");
    apu_save_dnn_to_file (bert_large_dnn, "nasms/bert_large_base.aspen");
    nasm_t *bert_large_nasm;
    for (int batch_size = 1; batch_size < 5; batch_size++)
    {
        for (int seq_len = 1; seq_len < 33; seq_len++)
        {
            printf ("Generating BERT LARGE NASM for batch size %d and sequence length %d\n", batch_size, seq_len);
            bert_large_nasm = apu_generate_transformer_nasm (bert_large_dnn, batch_size, seq_len, number_of_iterations, gpu_idx);
            sprintf (nasm_file_name, "nasms/bert_large_S%d_B%d_CPU.nasm", seq_len, batch_size);
            apu_save_nasm_to_file (bert_large_nasm, nasm_file_name);
        }
    }
    for (int batch_size = 1; batch_size < 2; batch_size++)
    {
        for (int seq_len = 32; seq_len < 128; seq_len += 2)
        {
            printf ("Generating BERT LARGE NASM for batch size %d and sequence length %d\n", batch_size, seq_len);
            bert_large_nasm = apu_generate_transformer_nasm (bert_large_dnn, batch_size, seq_len, number_of_iterations, gpu_idx);
            sprintf (nasm_file_name, "nasms/bert_large_S%d_B%d_CPU.nasm", seq_len, batch_size);
            apu_save_nasm_to_file (bert_large_nasm, nasm_file_name);
        }
        for (int seq_len = 128; seq_len < 513; seq_len += 4)
        {
            printf ("Generating BERT LARGE NASM for batch size %d and sequence length %d\n", batch_size, seq_len);
            bert_large_nasm = apu_generate_transformer_nasm (bert_large_dnn, batch_size, seq_len, number_of_iterations, gpu_idx);
            sprintf (nasm_file_name, "nasms/bert_large_S%d_B%d_CPU.nasm", seq_len, batch_size);
            apu_save_nasm_to_file (bert_large_nasm, nasm_file_name);
        }
    }

    aspen_dnn_t *gpt2_124M_dnn = apu_create_dnn("data/cfg/gpt2_124M_encoder.cfg", "data/gpt2_124M_data.bin");
    apu_save_dnn_to_file (gpt2_124M_dnn, "nasms/gpt2_124M_base.aspen");
    nasm_t *gpt2_124M_nasm;
    for (int batch_size = 1; batch_size < 5; batch_size++)
    {
        for (int seq_len = 1; seq_len < 32; seq_len++)
        {
            printf ("Generating GPT2 124M NASM for batch size %d and sequence length %d\n", batch_size, seq_len);
            gpt2_124M_nasm = apu_generate_transformer_nasm (gpt2_124M_dnn, batch_size, seq_len, number_of_iterations, gpu_idx);
            sprintf (nasm_file_name, "nasms/gpt2_124M_S%d_B%d_CPU.nasm", seq_len, batch_size);
            apu_save_nasm_to_file (gpt2_124M_nasm, nasm_file_name);
        }
    }
    for (int batch_size = 1; batch_size < 2; batch_size++)
    {
        for (int seq_len = 32; seq_len < 128; seq_len += 2)
        {
            printf ("Generating GPT2 124M NASM for batch size %d and sequence length %d\n", batch_size, seq_len);
            gpt2_124M_nasm = apu_generate_transformer_nasm (gpt2_124M_dnn, batch_size, seq_len, number_of_iterations, gpu_idx);
            sprintf (nasm_file_name, "nasms/gpt2_124M_S%d_B%d_CPU.nasm", seq_len, batch_size);
            apu_save_nasm_to_file (gpt2_124M_nasm, nasm_file_name);
        }
        for (int seq_len = 128; seq_len < 513; seq_len += 4)
        {
            printf ("Generating GPT2 124M NASM for batch size %d and sequence length %d\n", batch_size, seq_len);
            gpt2_124M_nasm = apu_generate_transformer_nasm (gpt2_124M_dnn, batch_size, seq_len, number_of_iterations, gpu_idx);
            sprintf (nasm_file_name, "nasms/gpt2_124M_S%d_B%d_CPU.nasm", seq_len, batch_size);
            apu_save_nasm_to_file (gpt2_124M_nasm, nasm_file_name);
        }
    }
    return 0;
}