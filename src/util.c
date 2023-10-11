#include "util.h"

char *ninst_state_str[NUM_NINST_STATES] = 
{
    [NINST_NOT_READY] = "NINST_NOT_READY", [NINST_READY] = "NINST_READY", [NINST_COMPLETED] = "NINST_COMPLETED"
};

char *layer_type_str [NUM_LAYER_ELEMENTS] = 
{
    [NO_LAYER_TYPE] = "NO_LAYER_TYPE", [INPUT_LAYER] = "INPUT_LAYER", [CONV_LAYER] = "CONV_LAYER", [FC_LAYER] = "FC_LAYER",
    [RESIDUAL_LAYER] = "RESIDUAL_LAYER", [BATCHNORM_LAYER] = "BATCHNORM_LAYER", [YOLO_LAYER] = "YOLO_LAYER", [APPEND_LAYER] = "APPEND_LAYER", [ACTIVATION_LAYER] = "ACTIVATION_LAYER", [MAXPOOL_LAYER] = "MAXPOOL_LAYER", [AVGPOOL_LAYER] = "AVGPOOL_LAYER",
    [ROUTE_LAYER] = "ROUTE_LAYER", [SOFTMAX_LAYER] = "SOFTMAX_LAYER",
    [MATMUL_LAYER] = "MATMUL_LAYER", [LAYERNORM_LAYER] = "LAYERNORM_LAYER", [K_ATTENTION_LAYER] = "K_ATTENTION_LAYER", [V_ATTENTION_LAYER] = "V_ATTENTION_LAYER"
};

char *param_type_str[NUM_PARAM_ELEMENTS] = 
{
    [OUT_W] = "OUT_W", [OUT_H] = "OUT_H", [IN_W] = "IN_W", [IN_H] = "IN_H", [IN_C] = "IN_C", [BATCH] = "BATCH", [OUT_C] = "OUT_C", [SUB_C] = "SUB_C", [WEIGHT_W] = "WEIGHT_W", [WEIGHT_H] = "WEIGHT_H", [STRIDE] = "STRIDE", [PADDING] = "PADDING", [DILATION] = "DILATION", [GROUPS] = "GROUPS",
    [NUM_HIDDEN] = "NUM_HIDDEN", [NUM_HEAD] = "NUM_HEAD", [NUM_SEQ] = "NUM_SEQ", [MAT_M] = "MAT_M", [MAT_N] = "MAT_N", [MAT_K] = "MAT_K", [SUB_M] = "SUB_M", [MASKED] = "MASKED",
    [FORM_BYTES] = "FORM_BYTES"
};

char *tensor_type_str[NUM_TENSORS] = 
{
    [NULL_TENSOR] = "NULL_TENSOR", [OUTPUT_TENSOR] = "OUTPUT_TENSOR", [INPUT_TENSOR] = "INPUT_TENSOR", [WEIGHT_TENSOR] = "WEIGHT_TENSOR", [BIAS_TENSOR] = "BIAS_TENSOR", [ANCHOR_TENSOR] = "ANCHOR_TENSOR", [BN_VAR_TENSOR] = "BN_VAR_TENSOR", [BN_MEAN_TENSOR] = "BN_MEAN_TENSOR", [BN_WEIGHT_TENSOR] = "BN_WEIGHT_TENSOR"
};

char *parent_type_str[NUM_PARENT_ELEMENTS] = 
{
    [PARENT_NONE] = "PARENT_NONE", [PARENT_0] = "PARENT_0", [PARENT_1] = "PARENT_1", [PARENT_WEIGHT] = "PARENT_WEIGHT",
};

char *activation_type_str [NUM_ACTIVATIONS] = 
{
    [NO_ACTIVATION] = "NO_ACTIVATION", [SIGMOID] = "SIGMOID", [LINEAR] = "LINEAR", [TANH] = "TANH", [RELU] = "RELU", [LEAKY_RELU] = "LEAKY_RELU", [ELU] = "ELU", [SELU] = "SELU", [GELU] = "GELU"
};

char *rpool_cond_str [NUM_RPOOL_CONDS] = 
{
    [RPOOL_DNN] = "RPOOL_DNN", [RPOOL_LAYER_TYPE] = "RPOOL_LAYER_TYPE", [RPOOL_LAYER_IDX] = "RPOOL_LAYER_IDX", [RPOOL_NASM] = "RPOOL_NASM", [RPOOL_DSE] = "RPOOL_DSE"
};

unsigned int dynamic_mem_init = 0;
void **dynamic_arr[DYNAMIC_ALLOC_RANGE] = {NULL};
size_t dynamic_arr_mem_size[DYNAMIC_ALLOC_RANGE] = {0};
unsigned int dynamic_arr_max_num[DYNAMIC_ALLOC_RANGE] = {0};
pthread_mutex_t dynamic_mutex = (pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER;

void init_dynamic_mem ()
{
    pthread_mutex_lock (&dynamic_mutex);
    if (dynamic_mem_init == 1)
    {
        pthread_mutex_unlock (&dynamic_mutex);
        return;
    }
    for (int i = 0; i < DYNAMIC_ALLOC_RANGE; i++)
    {
        dynamic_arr[i] = calloc (DYNAMIC_ALLOC_ARR_INIT_SIZE, sizeof(void*));
        dynamic_arr_max_num[i] = DYNAMIC_ALLOC_ARR_INIT_SIZE;
        dynamic_arr_mem_size[i] = (DYNAMIC_ALLOC_MIN_SIZE) * (pow (DYNAMIC_ALLOC_RANGE_SCALE, i));
    }
    dynamic_mem_init = 1;
    pthread_mutex_unlock (&dynamic_mutex);
}

void increase_dynamic_mem_arr (int idx)
{
    void **temp = calloc (dynamic_arr_max_num[idx] + DYNAMIC_ALLOC_ARR_INIT_SIZE, sizeof(void*));
    memcpy (temp, dynamic_arr[idx], dynamic_arr_max_num[idx]*sizeof(void*));
    free (dynamic_arr[idx]);
    dynamic_arr[idx] = temp;
    dynamic_arr_max_num[idx] += DYNAMIC_ALLOC_ARR_INIT_SIZE;
}

void print_dynamic_mem_arr()
{
    if (dynamic_mem_init == 0)
        init_dynamic_mem ();
    for (size_t i = 0; i < DYNAMIC_ALLOC_RANGE; i++)
    {
        printf ("\t[%ld KiB]\t\t", dynamic_arr_mem_size[i]/1024);
        int count = 0;
        for (int j = 0; j < dynamic_arr_max_num[i]; j++)
        {
            if (dynamic_arr[i][j] != NULL)
                count++;
        }
        printf ("%d/%d\n", count, dynamic_arr_max_num[i]);
    }
}

void *aspen_dynamic_calloc (size_t num, size_t size)
{
    void *ptr = aspen_dynamic_malloc (num, size);
    if (ptr == NULL)
        bzero (ptr, num * size);
    return ptr;
}

void *aspen_dynamic_malloc (size_t num, size_t size)
{
    if (dynamic_mem_init == 0)
        init_dynamic_mem ();
    if (num*size <= 0)
        return NULL;
    #ifdef DEBUG
    else if (num*size > dynamic_arr_mem_size[DYNAMIC_ALLOC_RANGE-1])
    {
        ERROR_PRTF ("Error: Requested memory size %ld KiB is too large.\n", num*size/1024);
        assert (0);
    }
    #endif
    void* ptr = NULL;
    size_t i = 0;
    for (; i < DYNAMIC_ALLOC_RANGE; i++)
    {
        if (dynamic_arr_mem_size[i] >= num*size)
            break;
    }
    pthread_mutex_lock (&dynamic_mutex);
    for (int j = 0; j < dynamic_arr_max_num[i]; j++)
    {
        if (dynamic_arr[i][j] != NULL)
        {
            ptr = dynamic_arr[i][j];
            dynamic_arr[i][j] = NULL;
            break;
        }
    }
    pthread_mutex_unlock (&dynamic_mutex);
    if (ptr == NULL)
    {
        ptr = aspen_malloc (1, dynamic_arr_mem_size[i]);
    }
    return ptr;
}

void aspen_dynamic_free (void *ptr, size_t num, size_t size)
{
    if (dynamic_mem_init == 0)
        init_dynamic_mem ();
    if (num*size <= 0)
        return;
    #ifdef DEBUG
    else if (num*size > dynamic_arr_mem_size[DYNAMIC_ALLOC_RANGE-1])
    {
        ERROR_PRTF ("Error: Requested memory size %ld KiB is too large.\n", num*size/1024);
        assert (0);
    }
    #endif
    size_t i = 0, j = 0;
    for (; i < DYNAMIC_ALLOC_RANGE; i++)
    {
        if (dynamic_arr_mem_size[i] >= num*size)
            break;
    }
    pthread_mutex_lock (&dynamic_mutex);
    for (; j < dynamic_arr_max_num[i]; j++)
    {
        if (dynamic_arr[i][j] == NULL)
        {
            dynamic_arr[i][j] = ptr;
            break;
        }
    }
    if (j == dynamic_arr_max_num[i])
    {
        increase_dynamic_mem_arr (i);
        dynamic_arr[i][j] = ptr;
    }
    pthread_mutex_unlock (&dynamic_mutex);
}

void aspen_flush_dynamic_memory ()
{
    if (dynamic_mem_init == 0)
        return;
    pthread_mutex_lock (&dynamic_mutex);
    for (int i = 0; i < DYNAMIC_ALLOC_RANGE; i++)
    {
        for (int j = 0; j < dynamic_arr_max_num[i]; j++)
        {
            if (dynamic_arr[i][j] != NULL)
            {
                aspen_free (dynamic_arr[i][j]);
                dynamic_arr[i][j] = NULL;
            }
        }
    }
    pthread_mutex_unlock (&dynamic_mutex);
}

void *aspen_calloc (size_t num, size_t size)
{
    if (num*size <= 0)
        return NULL;
    void* ptr = NULL;
    ptr = aligned_alloc (MEM_ALIGN, get_smallest_dividable(num * size, MEM_ALIGN));   
    bzero (ptr, get_smallest_dividable (num * size, MEM_ALIGN));
    return ptr;
}

void *aspen_malloc (size_t num, size_t size)
{
    if (num*size <= 0)
        return NULL;
    void* ptr = NULL;
    ptr = aligned_alloc (MEM_ALIGN, get_smallest_dividable(num * size, MEM_ALIGN));   
    return ptr;
}

void aspen_free (void *ptr)
{
    if (ptr == NULL)
        return;
    free (ptr);
}

size_t get_smallest_dividable (size_t num, size_t divider)
{
    return (num/divider + (num%divider != 0))*divider;
}

void* load_arr (char *file_path, unsigned int size)
{
    void *input = calloc (size, 1);
    FILE *fptr = fopen(file_path, "rb");
    if (fptr != NULL)
    {
        size_t num = fread (input, sizeof(char), size, fptr);
        if (num < size)
        {
            ERROR_PRTF ("Error: Failed to read file %s - Size mismatch. File size: %ld, Req. size: %d. Exiting.\n", 
                file_path, num, size);
            free (input);
            fclose(fptr);
            assert(0);
            return NULL;
        }
        fclose(fptr);
        return input;
    }
    else
    {
        ERROR_PRTF ("Error: Failed to open file %s. Exiting.\n", file_path);
        free (input);
        return NULL;
    }
}

void save_arr (void *input, char *file_path, unsigned int size)
{   
    FILE *fptr = fopen(file_path, "wb");
    fwrite (input, sizeof(char), size, fptr);
    fclose (fptr);
}

void fold_batchnorm_float (float *bn_var, float *bn_mean, float *bn_weight, 
    float *weight, float *bias, int cout, int cin, int hfil, int wfil)
{
    const double epsilon = 1e-5;
    
    for (int i = 0; i < cout; i++)
    {
        for (int j = 0; j < cin*hfil*wfil; j++)
        {
            float weight_val = *(weight + i*cin*hfil*wfil + j);
            weight_val = weight_val*(*(bn_weight + i))/sqrtf(*(bn_var + i) + epsilon);
            *(weight + i*cin*hfil*wfil + j) = weight_val;
        }
        float bias_val = *(bias + i);
        bias_val = bias_val - *(bn_weight + i)*(*(bn_mean + i))/sqrtf(*(bn_var + i) + epsilon);
        *(bias + i) = bias_val;
    }
}

void NHWC_to_NCHW (void *input, void *output, unsigned int n, unsigned int c, unsigned int h, unsigned int w, unsigned int element_size)
{
    if (input == NULL)
    {
        printf ("Error: Input is NULL.\n");
        return;
    }
    if (output == NULL)
    {
        printf ("Error: Output is NULL.\n");
        return;
    }
    if (input == output)
    {
        printf ("Error: Input and output are the same.\n");
        return;
    }
    for (int ni = 0; ni < n; ni++)
    {
        for (int ci = 0; ci < c; ci++)
        {
            for (int hi = 0; hi < h; hi++)
            {
                for (int wi = 0; wi < w; wi++)
                {
                    char* input_ptr = (char*)input + (ni*h*w*c + hi*w*c + wi*c + ci)*element_size;
                    char* output_ptr = (char*)output + (ni*c*h*w + ci*h*w + hi*w + wi)*element_size;
                    memcpy (output_ptr, input_ptr, element_size);
                }
            }
        }
    }
}
void NCHW_to_NHWC (void *input, void *output, unsigned int n, unsigned int c, unsigned int h, unsigned int w, unsigned int element_size)
{
    if (input == NULL)
    {
        printf ("Error: Input is NULL.\n");
        return;
    }
    if (output == NULL)
    {
        printf ("Error: Output is NULL.\n");
        return;
    }
    if (input == output)
    {
        printf ("Error: Input and output are the same.\n");
        return;
    }
    for (int ni = 0; ni < n; ni++)
    {
        for (int ci = 0; ci < c; ci++)
        {
            for (int hi = 0; hi < h; hi++)
            {
                for (int wi = 0; wi < w; wi++)
                {
                    char* input_ptr = (char*)input + (ni*c*h*w + ci*h*w + hi*w + wi)*element_size;
                    char* output_ptr = (char*)output + (ni*h*w*c + hi*w*c + wi*c + ci)*element_size;
                    memcpy (output_ptr, input_ptr, element_size);
                }
            }
        }
    }
}

void set_float_tensor_val (float *output, unsigned int n, unsigned int c, unsigned int h, unsigned int w)
{
    if (output == NULL)
    {
        printf ("Error: Output is NULL.\n");
        return;
    }
    for (int ni = 0; ni < n; ni++)
    {
        for (int ci = 0; ci < c; ci++)
        {
            for (int hi = 0; hi < h; hi++)
            {
                for (int wi = 0; wi < w; wi++)
                {
                    float* output_ptr = output + (ni*h*w*c + hi*w*c + wi*c + ci);
                    *output_ptr = (float)(ni*h*w*c + hi*w*c + wi*c + ci);
                }
            }
        }
    }
}

int compare_float_array (float *input1, float* input2, int num_to_compare, float epsilon_ratio, float epsilon_abs, int skip_val)
{
    int num = 0;
    printf ("Compare_array_f32 running...\n");
    for (int i = 0; i < num_to_compare; i++)
    {
        float delta = fabsf(*(input1 + i) - *(input2 + i));
        if ((delta / fabsf(*(input1 + i))) >= epsilon_ratio && delta >= epsilon_abs)
        {
            num++;
            if (num < skip_val)
            {
                printf ("\tCompare failed at index %d. Value1: %3.3e, Value2: %3.3e, Diff: %1.2e (%2.2e%%)\n"
                    , i, *(input1 + i), *(input2 + i), delta, delta*100.0/(*(input1 + i)<0? -*(input1 + i):*(input1 + i)));
            }
            else if (num == skip_val)
            {
                printf ("\tToo many errors... (More than %d)\n", skip_val);
            }
        }
    }
    printf ("Compare_array_f32 complete.\nTotal of %d errors detected out of %d SP floats, with epsilon ratio of %1.1e.\n", num, num_to_compare,epsilon_ratio);
    return num;
}

int compare_float_tensor (float *input1, float* input2, int n, int c, int h ,int w, float epsilon_ratio, float epsilon_abs, int skip_val)
{
    int num = 0;
    printf ("Compare_tensor_f32 running...\n");
    for (int ni = 0; ni < n; ni++)
    {
        for (int ci = 0; ci < c; ci++)
        {
            for (int hi = 0; hi < h; hi++)
            {
                for (int wi = 0; wi < w; wi++)
                {
                    int i = ni*c*h*w + ci*h*w + hi*w + wi;
                    float delta = fabs(*(input1 + i) - *(input2 + i));
                    if ((delta / fabs(*(input1 + i))) >= epsilon_ratio && delta >= epsilon_abs)
                    {
                        num++;
                        if (num < skip_val)
                        {
                            printf ("\tCompare failed at index (%d, %d, %d, %d). Value1: %3.3e, Value2: %3.3e, Diff: %1.2e (%2.2e%%)\n"
                                , ni, ci, hi, wi, *(input1 + i), *(input2 + i), delta, delta*100.0/(*(input1 + i)<0? -*(input1 + i):*(input1 + i)));
                        }
                        else if (num == skip_val)
                        {
                            printf ("\tToo many errors... (More than %d)\n", skip_val);
                        }
                    }
                }
            }
        }
    }
    printf ("Compare_tensor_f32 complete.\nTotal of %d errors detected out of %d SP floats, with epsilon ratio of %1.1e.\n", num, n*c*h*w, epsilon_ratio);
    return num;
}

unsigned int get_cpu_count()
{
    cpu_set_t cs;
    CPU_ZERO(&cs);
    sched_getaffinity(0, sizeof(cs), &cs);

    unsigned int count = 0;
    for (int i = 0; i < 256; i++)
    {
        if (CPU_ISSET(i, &cs))
            count++;
        else
            break;
    }
    return count;
}

void get_probability_results (char *class_data_path, float* probabilities, unsigned int num)
{
    int buffer_length = 256;
    char buffer[num][buffer_length];
    FILE *fptr = fopen(class_data_path, "r");
    if (fptr == NULL)
    {
        printf ("Error in get_probability_results: Cannot open file %s.\n", class_data_path);
        return;
    }
    for (int i = 0; i < num; i++)
    {
        void *tmp = fgets(buffer[i], buffer_length, fptr);
        if (tmp == NULL)
        {
            printf ("Error in get_probability_results: Cannot read file %s.\n", class_data_path);
            return;
        }
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

double get_time_secs ()
{
    static int initiallized = 0;
    static struct timeval zero_time;
    if (initiallized == 0)
    {
        initiallized = 1;
        gettimeofday (&zero_time, NULL);
    }
    
    struct timeval now;
    gettimeofday (&now, NULL);
    long sec = now.tv_sec - zero_time.tv_sec;
    long usec = now.tv_usec - zero_time.tv_usec;
    return sec + usec*1e-6;
}

double get_time_secs_suppressed ()
{
    static int initiallized = 0;
    static struct timeval zero_time;
    if (initiallized == 0)
    {
        initiallized = 1;
        gettimeofday (&zero_time, NULL);
    }
    struct timeval now;
    gettimeofday (&now, NULL);
    long sec = now.tv_sec - zero_time.tv_sec;
    long usec = now.tv_usec - zero_time.tv_usec;
    return sec + usec*1e-6;
}

void get_elapsed_time (char *name)
{
    static int call_num = 0;
    static double last = -1;
    if (last < 0)
    {
        last = get_time_secs();
    }
    double now = get_time_secs();
    double elapsed = now - last;
    if (call_num > 0)
    {
        printf ("Time measurement %s (%d): %6.6f - %6.6f secs elapsed since last measurement.\n", name, call_num, now, elapsed);
    }
    call_num++;
    last = now;
}

void get_elapsed_time_only()
{
    static int call_num = 0;
    static double last = 0;
    double now = get_time_secs_suppressed ();
    double elapsed = now - last;
    if (call_num > 0)
    {
        printf ("%6.6f", elapsed);
    }
    call_num++;
    last = now;
}

void print_float_array (float *input, int num, int newline_num)
{
    int i;
    printf ("Printing Array of size %d...\n", num);
    for (i = 0; i < num; i++)
    {
        const float val = *(input + i);
        if (val < 0.0)
        {
            printf ("\t%3.3ef", val);
        }
        else
        {
            printf ("\t %3.3ef", val);
        }
        if (i%newline_num == newline_num-1)
        {
            printf("\n");
        }
    }
    if (i%newline_num != newline_num-1)
    {
        printf("\n");
    }
}

void print_float_tensor (float *input, int n, int c, int h, int w)
{
    printf ("\t");
    int size_arr[] = {n, c, h};
    int newline = w;
    for (int i = 2; i >= 0; i--)
    {
        newline *= size_arr[i];
        if (newline > 20)
        {
            newline /= size_arr[i];
            break;
        }
    }
    size_t idx = 0;
    for (int ni = 0; ni < n; ni++)
    {
        for (int ci = 0; ci < c; ci++)
        {
            for (int hi = 0; hi < h; hi++)
            {
                for (int wi = 0; wi < w; wi++)
                {
                    if (idx%newline == 0)
                    {
                        printf("\n(%d, %d, %d, %d):\t", ni, ci, hi, wi);
                    }
                    const float val = *(input + ni*c*h*w + ci*h*w + hi*w + wi);
                    if (val < 0.0)
                    {
                        printf ("\t%3.3ef", val);
                    }
                    else
                    {
                        printf ("\t %3.3ef", val);
                    }
                    idx++;
                    
                }
            }
        }
    }
    printf ("\n");
}