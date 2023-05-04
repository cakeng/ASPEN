#include "aspen.h"
#include "apu.h"
#include "nasm.h"
#include "input_parser.h"


extern char* branch_info;

char *read_check_and_return (FILE *fp, char *buffer, char *check_str, unsigned int *line_num)
{
    if (fgets(buffer, MAX_STRING_LEN, fp) == NULL)
    {
        FPRT (stderr,"ASPEN DNN file parse error: Unexpected EOF.\n");
        return NULL;
    }
    (*line_num)++;
    buffer[strcspn(buffer, "\n")] = 0;
    char *line = buffer;
    while (*line == ' ' || *line == '\t')
        line++;
    if (strncmp(line, check_str, strlen(check_str)) != 0)
    {
        FPRT(stderr,"Wrong ASPEN DNN file format at line %d, expected \"%s\", got \"%s\"\n", *line_num, check_str, line);
        return NULL;
    }
    return line + strlen(check_str);
}

char *read_and_return_if_EOF (FILE *fp, char *buffer, char *check_str, unsigned int *line_num)
{
    if (fgets(buffer, MAX_STRING_LEN, fp) == NULL)
    {
        return NULL;
    }
    (*line_num)++;
    buffer[strcspn(buffer, "\n")] = 0;
    char *line = buffer;
    while (*line == ' ' || *line == '\t')
        line++;
    if (strncmp(line, check_str, strlen(check_str)) != 0)
    {
        FPRT(stderr,"Wrong ASPEN DNN file format at line %d, expected \"%s\", got \"%s\"\n", *line_num, check_str, line);
        return NULL;
    }
    return line + strlen(check_str);
}

void apu_load_dnn_data_from_file (aspen_dnn_t *dnn, char *input_path)
{
    FILE *fp = fopen(input_path, "rb");
    if (fp == NULL)
    {
        FPRT(stderr,"ASPEN DNN file %s not found.\n", input_path);
        return;
    }
    
    float *bn_var = NULL, *bn_mean = NULL, *bn_weight = NULL;
    char line[MAX_STRING_LEN] = {0};
    char* ptr;
    unsigned int line_num = 0;
    int file_layer_num = 0;
    size_t data_size = 0;
    LAYER_TENSORS tensor_type = NULL_TENSOR;
    if ((ptr = read_check_and_return (fp, line, "ASPEN_DATA", &line_num)) == NULL)
    {
        FPRT(stderr,"ASPEN DNN file %s parse error: Missing ASPEN_DATA.\n", input_path);
        fclose (fp);
        return;
    }
    if ((ptr = read_check_and_return (fp, line, "LAYER:", &line_num)) == NULL)
    {
        FPRT(stderr,"ASPEN DNN file %s parse error: Missing LAYER.\n", input_path);
        if (bn_mean != NULL)
            free (bn_mean);
        if (bn_weight != NULL)
            free (bn_weight);
        if (bn_var != NULL)
            free (bn_var);
        fclose (fp);
        return;
    }
    while (feof(fp) == 0)
    {
        file_layer_num = atoi(ptr);
        int layer_num = 0;
        unsigned int weighted_layer = 0;
        for (int i = 0; i < dnn->num_layers; i++)
        {
            if (dnn->layers[i].type == CONV_LAYER 
                || dnn->layers[i].type == FC_LAYER
                || dnn->layers[i].type == MATMUL_LAYER
                || dnn->layers[i].type == LAYERNORM_LAYER)
            {
                weighted_layer++;
                if (weighted_layer == file_layer_num)
                {
                    layer_num = i;
                    PRT ("File Layer %d loading into DNN layer %d: ", file_layer_num, layer_num);
                    break;
                }
            }
            if (i == dnn->num_layers - 1)
            {
                FPRT(stderr,"ASPEN DNN file %s parse error: Invalid file LAYER %d.\n", input_path, file_layer_num);
                if (bn_mean != NULL)
                    free (bn_mean);
                if (bn_weight != NULL)
                    free (bn_weight);
                if (bn_var != NULL)
                    free (bn_var);
                fclose (fp);
                return;
            }
        }
        aspen_layer_t *layer = &dnn->layers[layer_num];
        if ((ptr = read_check_and_return (fp, line, "TENSOR_TYPE:", &line_num)) == NULL)
        {
            FPRT(stderr,"ASPEN DNN file %s parse error: Missing TENSOR_TYPE.\n", input_path);
            if (bn_mean != NULL)
                free (bn_mean);
            if (bn_weight != NULL)
                free (bn_weight);
            if (bn_var != NULL)
                free (bn_var);
            fclose (fp);
            return;
        }
        if (strcmp(ptr, "INPUT") == 0)
            tensor_type = INPUT_TENSOR;
        else if (strcmp(ptr, "OUTPUT") == 0)
            tensor_type = OUTPUT_TENSOR;
        else if (strcmp(ptr, "WEIGHT") == 0)
            tensor_type = WEIGHT_TENSOR;
        else if (strcmp(ptr, "BIAS") == 0)
            tensor_type = BIAS_TENSOR;
        else if (strcmp(ptr, "BN_VAR") == 0)
            tensor_type = BN_VAR_TENSOR;
        else if (strcmp(ptr, "BN_MEAN") == 0)
            tensor_type = BN_MEAN_TENSOR;
        else if (strcmp(ptr, "BN_WEIGHT") == 0)
            tensor_type = BN_WEIGHT_TENSOR;
        else
        {
            ptr[20] = '\0';
            FPRT(stderr,"ASPEN DNN file %s file layer %d parse error: Invalid TENSOR_TYPE %s.\n", 
                input_path, file_layer_num, ptr);
            if (bn_mean != NULL)
                free (bn_mean);
            if (bn_weight != NULL)
                free (bn_weight);
            if (bn_var != NULL)
                free (bn_var);
            fclose (fp);
            return;
        }
        // PRT ("Tensor type %s.\n", tensor_type_str[tensor_type]);
        if ((ptr = read_check_and_return (fp, line, "DATA_SIZE:", &line_num)) == NULL)
        {
            FPRT(stderr,"ASPEN DNN file %s parse error: Missing DATA_SIZE.\n", input_path);
            if (bn_mean != NULL)
                free (bn_mean);
            if (bn_weight != NULL)
                free (bn_weight);
            if (bn_var != NULL)
                free (bn_var);
            fclose (fp);
            return;
        }
        data_size = atoi(ptr);
        if ((ptr = read_check_and_return (fp, line, "DATA_START:", &line_num)) == NULL)
        {
            FPRT(stderr,"ASPEN DNN file %s parse error: Missing DATA_START.\n", input_path);
            if (bn_mean != NULL)
                free (bn_mean);
            if (bn_weight != NULL)
                free (bn_weight);
            if (bn_var != NULL)
                free (bn_var);
            fclose (fp);
            return;
        }
        void *buffer = malloc(data_size);
        if (buffer == NULL)
        {
            FPRT(stderr,"ASPEN DNN file %s parse error: Failed to allocate memory.\n", input_path);
            free (buffer);
            if (bn_mean != NULL)
                free (bn_mean);
            if (bn_weight != NULL)
                free (bn_weight);
            if (bn_var != NULL)
                free (bn_var);
            fclose (fp);
            return;
        }
        if (fread(buffer, data_size, 1, fp) != 1)
        {
            FPRT(stderr,"ASPEN DNN file %s parse error: Failed to read data.\n", input_path);
            free (buffer);
            if (bn_mean != NULL)
                free (bn_mean);
            if (bn_weight != NULL)
                free (bn_weight);
            if (bn_var != NULL)
                free (bn_var);
            fclose (fp);
            return;
        }
        if ((ptr = read_check_and_return (fp, line, "DATA_END", &line_num)) == NULL)
        {
            FPRT(stderr,"ASPEN DNN file %s parse error: Missing DATA_END.\n", input_path);
            free (buffer);
            if (bn_mean != NULL)
                free (bn_mean);
            if (bn_weight != NULL)
                free (bn_weight);
            if (bn_var != NULL)
                free (bn_var);
            fclose (fp);    
            return;
        }
        if (tensor_type == WEIGHT_TENSOR)
        {
            if (layer->tensors[WEIGHT_TENSOR] != NULL)
            {
                if (layer->type == MATMUL_LAYER)
                {
                    LAYER_PARAMS weight_dim_order[] = {MAT_M, MAT_K};
                    for (int i = 0; i < layer->tensors[WEIGHT_TENSOR]->num_dims; i++)
                    {
                        layer->tensors[WEIGHT_TENSOR]->data_dim_order[i] = weight_dim_order[i];
                    }
                }
                else if (layer->type == LAYERNORM_LAYER)
                {
                    LAYER_PARAMS weight_dim_order[] = {MAT_M};
                    for (int i = 0; i < layer->tensors[WEIGHT_TENSOR]->num_dims; i++)
                    {
                        layer->tensors[WEIGHT_TENSOR]->data_dim_order[i] = weight_dim_order[i];
                    }
                }
                else
                {
                    LAYER_PARAMS weight_dim_order[] = {OUT_C, IN_C, WEIGHT_H, WEIGHT_W};
                    for (int i = 0; i < layer->tensors[WEIGHT_TENSOR]->num_dims; i++)
                    {
                        layer->tensors[WEIGHT_TENSOR]->data_dim_order[i] = weight_dim_order[i];
                    }
                }
                if (layer->tensors[WEIGHT_TENSOR]->num_elements*layer->tensors[WEIGHT_TENSOR]->element_size == data_size)
                    copy_ptr_to_aspen_tensor (layer->tensors[WEIGHT_TENSOR], buffer);
                else
                {
                    FPRT(stderr,"ASPEN DNN file %s parse error: Layer %d WEIGHT_TENSOR size mismatch:\
                        Tensor: %d, File: %ld.\n", input_path, layer_num,
                            layer->tensors[WEIGHT_TENSOR]->num_elements*layer->tensors[WEIGHT_TENSOR]->element_size,
                            data_size);
                    assert(0);
                    free (buffer);
                    if (bn_mean != NULL)
                        free (bn_mean);
                    if (bn_weight != NULL)
                        free (bn_weight);
                    if (bn_var != NULL)
                        free (bn_var);
                    fclose (fp);
                    return;
                }
            }
            else 
            {
                FPRT(stderr,"ASPEN DNN file %s parse error: Layer %d missing WEIGHT_TENSOR.\n",
                    input_path, layer_num);
                free (buffer);
                if (bn_mean != NULL)
                    free (bn_mean);
                if (bn_weight != NULL)
                    free (bn_weight);
                if (bn_var != NULL)
                    free (bn_var);
                fclose (fp);
                return;
            }
            free (buffer);
        }
        else if (tensor_type == BIAS_TENSOR)
        {
            if (layer->tensors[BIAS_TENSOR] != NULL)
            {
                if (layer->type == MATMUL_LAYER || layer->type == LAYERNORM_LAYER)
                {
                    LAYER_PARAMS weight_dim_order[] = {MAT_M};
                    for (int i = 0; i < layer->tensors[WEIGHT_TENSOR]->num_dims; i++)
                    {
                        layer->tensors[WEIGHT_TENSOR]->data_dim_order[i] = weight_dim_order[i];
                    }
                }
                else
                {
                    LAYER_PARAMS bias_dim_order[] = {OUT_C, IN_C};
                    for (int i = 0; i < layer->tensors[BIAS_TENSOR]->num_dims; i++)
                    {
                        layer->tensors[BIAS_TENSOR]->data_dim_order[i] = bias_dim_order[i];
                    }
                }
                
                if (layer->tensors[BIAS_TENSOR]->num_elements*layer->tensors[BIAS_TENSOR]->element_size == data_size)
                    copy_ptr_to_aspen_tensor (layer->tensors[BIAS_TENSOR], buffer);
                else
                {
                    FPRT(stderr,"ASPEN DNN file %s parse error: Layer %d BIAS_TENSOR size mismatch:\
                        Tensor: %d, File: %ld.\n", input_path, layer_num,
                            layer->tensors[BIAS_TENSOR]->num_elements*layer->tensors[BIAS_TENSOR]->element_size,
                            data_size);
                    free (buffer);
                    if (bn_mean != NULL)
                        free (bn_mean);
                    if (bn_weight != NULL)
                        free (bn_weight);
                    if (bn_var != NULL)
                        free (bn_var);
                    fclose (fp);
                    return;
                }
            }
            else 
            {
                FPRT(stderr,"ASPEN DNN file %s parse error: Layer %d missing BIAS_TENSOR.\n", input_path, layer_num);
                free (buffer);
                if (bn_mean != NULL)
                    free (bn_mean);
                if (bn_weight != NULL)
                    free (bn_weight);
                if (bn_var != NULL)
                    free (bn_var);
                fclose (fp);
                return;
            }
            free (buffer);
        }
        else if (tensor_type == BN_VAR_TENSOR)
        {
            if (bn_var == NULL)
                bn_var = (float *)buffer;
            else
            {
                FPRT(stderr,"ASPEN DNN file %s parse error: Duplicate BN_VAR.\n", input_path);
                free (buffer);
                if (bn_mean != NULL)
                    free (bn_mean);
                if (bn_weight != NULL)
                    free (bn_weight);
                if (bn_var != NULL)
                    free (bn_var);
                fclose (fp);
                return;
            }
        }
        else if (tensor_type == BN_MEAN_TENSOR)
        {
            if (bn_mean == NULL)
                bn_mean = (float *)buffer;
            else
            {
                FPRT(stderr,"ASPEN DNN file %s parse error: Duplicate BN_MEAN.\n", input_path);
                free (buffer);
                if (bn_mean != NULL)
                    free (bn_mean);
                if (bn_weight != NULL)
                    free (bn_weight);
                if (bn_var != NULL)
                    free (bn_var);
                fclose (fp);
                return;
            }
        }
        else if (tensor_type == BN_WEIGHT_TENSOR)
        {
            if (bn_weight == NULL)
                bn_weight = (float *)buffer;
            else
            {
                FPRT(stderr,"ASPEN DNN file %s parse error: Duplicate BN_WEIGHT.\n", input_path);
                free (buffer);
                if (bn_mean != NULL)
                    free (bn_mean);
                if (bn_weight != NULL)
                    free (bn_weight);
                if (bn_var != NULL)
                    free (bn_var);
                fclose (fp);
                return;
            }
        }
        if (bn_var != NULL && bn_mean != NULL && bn_weight != NULL)
        {
            fold_batchnorm_float (bn_var, bn_mean, bn_weight,
                                  layer->tensors[WEIGHT_TENSOR]->data,
                                  layer->tensors[BIAS_TENSOR]->data,
                                  layer->params[OUT_C], layer->params[IN_C],
                                  layer->params[WEIGHT_H], layer->params[WEIGHT_W]);
            free(bn_var);
            free(bn_mean);
            free(bn_weight);
            bn_var = NULL;
            bn_mean = NULL;
            bn_weight = NULL;
        }
        printf ("Layer %d data loaded from file info string LAYER: %d, TENSOR_TYPE: %s, DATA_SIZE: %ld\n",
             layer_num, file_layer_num, tensor_type_str[tensor_type], data_size);
        if ((ptr = read_check_and_return (fp, line, "LAYER_END", &line_num)) == NULL)
        {
            FPRT(stderr,"ASPEN DNN file %s parse error: Missing LAYER_END.\n", input_path);
            if (bn_mean != NULL)
                free (bn_mean);
            if (bn_weight != NULL)
                free (bn_weight);
            if (bn_var != NULL)
                free (bn_var);
            fclose (fp);
            return;
        }
        if ((ptr = read_and_return_if_EOF (fp, line, "LAYER:", &line_num)) == NULL)
        {
            break;
        }
    }
    fclose (fp);
}

void apu_save_dnn_to_file(aspen_dnn_t *dnn, char *filename)
{
    FILE *fp = fopen(filename, "wb");
    if (fp == NULL)
    {
        FPRT(stderr, "Error: apu_save_dnn_to_file Failed to open file %s for writing\n", filename);
        return;
    }
    fprintf(fp, "ASPEN_DNN\n");
    fprintf(fp, "ASPEN_BUILD:%s\n", branch_info);
    fprintf(fp, "DNN_NAME:%s\n", dnn->name);
    fprintf(fp, "DNN_ELEMENT_SIZE:%d\n", dnn->element_size);
    fprintf(fp, "NUM_LAYERS:%d\n", dnn->num_layers);
    for (unsigned int i = 0; i < dnn->num_layers; i++)
    {
        aspen_layer_t *layer = dnn->layers + i;
        fprintf(fp, "\tLAYER_IDX:%d\n", layer->layer_idx);
        fprintf(fp, "\tLAYER_TYPE:%d\n", layer->type);
        fprintf(fp, "\tLAYER_ACTIVATION:%d\n", layer->activation);
        fprintf(fp, "\tLAYER_PARENTS:\n");
        for (unsigned int j = 0; j < NUM_PARENT_ELEMENTS; j++)
        {
            if (layer->parent_layers[j] != NULL)
                fprintf(fp, "\t\t%d %d\n", j, layer->parent_layers[j]->layer_idx);
            else
                fprintf(fp, "\t\t%d -1\n", j);
        }
        fprintf(fp, "\tLAYER_PARENTS_END\n");
        fprintf(fp, "\tLAYER_PARAMS:\n");
        for (unsigned int j = 0; j < NUM_PARAM_ELEMENTS; j++)
        {
            fprintf(fp, "\t\t%d %d\n", j, layer->params[j]);
        }
        fprintf(fp, "\tLAYER_PARAMS_END\n");
        fprintf(fp, "\tLAYER_TENSORS:\n");
        for (unsigned int j = 0; j < NUM_TENSORS; j++)
        {
            if (layer->tensors[j] != NULL)
            {
                aspen_tensor_t *tensor = layer->tensors[j];
                fprintf(fp, "\t\t%d 1\n", j);
                fprintf(fp, "\t\tNUM_DIMS:%d\n", tensor->num_dims);
                fprintf(fp, "\t\tDATA_DIM_ORDER:\n");
                for (unsigned int k = 0; k < MAX_TENSOR_DIMS; k++)
                {
                    fprintf(fp, "\t\t\t%d %d\n", k, tensor->data_dim_order[k]);
                }
                fprintf(fp, "\t\tDATA_DIM_ORDER_END\n");
                fprintf(fp, "\t\tTENSOR_DIMS:\n");
                for (unsigned int k = 0; k < NUM_PARAM_ELEMENTS; k++)
                {
                    fprintf(fp, "\t\t\t%d %d\n", k, tensor->dims[k]);
                }
                fprintf(fp, "\t\tTENSOR_DIMS_END\n");
                fprintf(fp, "\t\tTENSOR_DATA:\n");
                fwrite (tensor->data, dnn->element_size, tensor->num_elements, fp);
                fprintf(fp, "\t\tTENSOR_DATA_END\n");
            }
            else
            {
                fprintf(fp, "\t\t%d -1\n", j);
            }
        }
        fprintf(fp, "\tLAYER_TENSORS_END\n");
    }
    fprintf(fp, "ASPEN_DNN_END\n");
    fclose(fp);
}

aspen_dnn_t *apu_parse_dnn_from_file(char *filename, FILE **fp_t, unsigned int *line_num, unsigned int skip_alloc)
{
    char dnn_name[MAX_STRING_LEN] = {0};
    unsigned int num_layers = 0;
    char line[MAX_STRING_LEN] = {0};
    char* ptr;
    if ((ptr = read_check_and_return (*fp_t, line, "ASPEN_DNN", line_num)) == NULL)
    {
        FPRT(stderr,"ASPEN DNN file %s parse error: Not an ASPEN DNN file.\n", filename);
        return NULL;
    }
    if ((ptr = read_check_and_return (*fp_t, line, "ASPEN_BUILD:", line_num)) == NULL)
    {
        FPRT(stderr,"ASPEN DNN file %s parse error: Missing ASPEN_BUILD.\n", filename);
        return NULL;
    }
    if ((ptr = read_check_and_return (*fp_t, line, "DNN_NAME:", line_num)) == NULL)
    {
        FPRT(stderr,"ASPEN DNN file %s parse error: Missing DNN_NAME.\n", filename);
        return NULL;
    }
    sscanf(ptr, "%s", dnn_name);
    if ((ptr = read_check_and_return (*fp_t, line, "DNN_ELEMENT_SIZE:", line_num)) == NULL)
    {
        FPRT(stderr,"ASPEN DNN file %s parse error: Missing DNN_ELEMENT_SIZE.\n", filename);
        return NULL;
    }
    size_t element_size = atol(ptr);
    if ((ptr = read_check_and_return (*fp_t, line, "NUM_LAYERS:", line_num)) == NULL)
    {
        FPRT(stderr,"ASPEN DNN file %s parse error: Missing NUM_LAYERS.\n", filename);
        return NULL;
    }
    num_layers = atoi(ptr);
    aspen_dnn_t *dnn = init_aspen_dnn(num_layers, dnn_name);
    if (dnn == NULL)
    {
        FPRT(stderr,"ASPEN DNN file %s parse error: Failed to create DNN.\n", filename);
        return NULL;
    }
    dnn->element_size = element_size;
    for (unsigned int i = 0; i < num_layers; i++)
    {
        aspen_layer_t *layer = dnn->layers + i;
        if ((ptr = read_check_and_return (*fp_t, line, "LAYER_IDX:", line_num)) == NULL)
        {
            FPRT(stderr,"ASPEN DNN file %s parse error: Missing LAYER_IDX.\n", filename);
            apu_destroy_dnn(dnn);
            return NULL;
        }
        layer->layer_idx = atoi(ptr);
        if ((ptr = read_check_and_return (*fp_t, line, "LAYER_TYPE:", line_num)) == NULL)
        {
            FPRT(stderr,"ASPEN DNN file %s parse error: Missing LAYER_TYPE.\n", filename);
            apu_destroy_dnn(dnn);
            return NULL;
        }
        layer->type = atoi(ptr);
        if ((ptr = read_check_and_return (*fp_t, line, "LAYER_ACTIVATION:", line_num)) == NULL)
        {
            FPRT(stderr,"ASPEN DNN file %s parse error: Missing LAYER_ACTIVATION.\n", filename);
            apu_destroy_dnn(dnn);
            return NULL;
        }
        layer->activation = atoi(ptr);
        if ((ptr = read_check_and_return (*fp_t, line, "LAYER_PARENTS:", line_num)) == NULL)
        {
            FPRT(stderr,"ASPEN DNN file %s parse error: Missing LAYER_PARENTS.\n", filename);
            apu_destroy_dnn(dnn);
            return NULL;
        }
        for (unsigned int j = 0; j < NUM_PARENT_ELEMENTS; j++)
        {
            void *tmp = fgets (line, MAX_STRING_LEN, *fp_t);
            if (tmp == NULL)
            {
                FPRT(stderr,"ASPEN DNN file %s parse error at source line %d, soruce file %s\n", 
                    filename, __LINE__, __FILE__);
                apu_destroy_dnn(dnn);
                return NULL;
            }
            *line_num += 1;
            ptr = line;
            while (*ptr == ' ' || *ptr == '\t')
                ptr++;
            int parent_idx = 0;
            int parent_layer_idx = 0;
            sscanf(ptr, "%d %d", &parent_idx, &parent_layer_idx);
            if (parent_layer_idx != -1)
                layer->parent_layers[parent_idx] = dnn->layers + parent_layer_idx;
            else
                layer->parent_layers[parent_idx] = NULL;
        }
        if ((ptr = read_check_and_return (*fp_t, line, "LAYER_PARENTS_END", line_num)) == NULL)
        {
            FPRT(stderr,"ASPEN DNN file %s parse error: Missing LAYER_PARENTS_END.\n", filename);
            apu_destroy_dnn(dnn);
            return NULL;
        }
        if ((ptr = read_check_and_return (*fp_t, line, "LAYER_PARAMS:", line_num)) == NULL)
        {
            FPRT(stderr,"ASPEN DNN file %s parse error: Missing LAYER_PARAMS.\n", filename);
            apu_destroy_dnn(dnn);
            return NULL;
        }
        for (unsigned int j = 0; j < NUM_PARAM_ELEMENTS; j++)
        {
            void *tmp = fgets (line, MAX_STRING_LEN, *fp_t);
            if (tmp == NULL)
            {
                FPRT(stderr,"ASPEN DNN file %s parse error at source line %d, soruce file %s\n", 
                    filename, __LINE__, __FILE__);
                apu_destroy_dnn(dnn);
                return NULL;
            }
            *line_num += 1;
            ptr = line;
            while (*ptr == ' ' || *ptr == '\t')
                ptr++;
            int param_idx = 0;
            int param_val = 0;
            sscanf(ptr, "%d %d", &param_idx, &param_val);
            layer->params[param_idx] = param_val;
        }
        if ((ptr = read_check_and_return (*fp_t, line, "LAYER_PARAMS_END", line_num)) == NULL)
        {
            FPRT(stderr,"ASPEN DNN file %s parse error: Missing LAYER_PARAMS_END.\n", filename);
            apu_destroy_dnn(dnn);
            return NULL;
        }
        if ((ptr = read_check_and_return (*fp_t, line, "LAYER_TENSORS:", line_num)) == NULL)
        {
            FPRT(stderr,"ASPEN DNN file %s parse error: Missing LAYER_TENSORS.\n", filename);
            apu_destroy_dnn(dnn);
            return NULL;
        }
        for (unsigned int j = 0; j < NUM_TENSORS; j++)
        {
            void * tmp = fgets (line, MAX_STRING_LEN, *fp_t);
            if (tmp == NULL)
            {
                FPRT(stderr,"ASPEN DNN file %s parse error at source line %d, soruce file %s\n", 
                    filename, __LINE__, __FILE__);
                apu_destroy_dnn(dnn);
                return NULL;
            }
            *line_num += 1;
            ptr = line;
            while (*ptr == ' ' || *ptr == '\t')
                ptr++;
            int tensor_idx = 0;
            int tensor_val = 0;
            sscanf(ptr, "%d %d", &tensor_idx, &tensor_val);
            if (tensor_val == -1)
                layer->tensors[tensor_idx] = NULL;
            else
            {
                layer->tensors[tensor_idx] = calloc(1, sizeof(aspen_tensor_t));
                aspen_tensor_t *tensor = layer->tensors[tensor_idx];
                if (tensor == NULL)
                {
                    FPRT(stderr,"ASPEN DNN file %s parse error: Failed to create tensor.\n", filename);
                    apu_destroy_dnn(dnn);
                    return NULL;
                }
                if ((ptr = read_check_and_return (*fp_t, line, "NUM_DIMS:", line_num)) == NULL)
                {
                    FPRT(stderr,"ASPEN DNN file %s parse error: Missing NUM_DIMS.\n", filename);
                    apu_destroy_dnn(dnn);
                    return NULL;
                }
                tensor->num_dims = atoi(ptr);
                if ((ptr = read_check_and_return (*fp_t, line, "DATA_DIM_ORDER:", line_num)) == NULL)
                {
                    FPRT(stderr,"ASPEN DNN file %s parse error: Missing DATA_DIM_ORDER.\n", filename);
                    apu_destroy_dnn(dnn);
                    return NULL;
                }
                for (unsigned int k = 0; k < MAX_TENSOR_DIMS; k++)
                {
                    void * tmp = fgets (line, MAX_STRING_LEN, *fp_t);
                    if (tmp == NULL)
                    {
                        FPRT(stderr,"ASPEN DNN file %s parse error at source line %d, soruce file %s\n", 
                            filename, __LINE__, __FILE__);
                        apu_destroy_dnn(dnn);
                        return NULL;
                    }
                    *line_num += 1;
                    ptr = line;
                    while (*ptr == ' ' || *ptr == '\t')
                        ptr++;
                    int dim_idx = 0;
                    int dim_type = 0;
                    sscanf(ptr, "%d %d", &dim_idx, &dim_type);
                    tensor->data_dim_order[dim_idx] = dim_type;
                }
                if ((ptr = read_check_and_return (*fp_t, line, "DATA_DIM_ORDER_END", line_num)) == NULL)
                {
                    FPRT(stderr,"ASPEN DNN file %s parse error: Missing DATA_DIM_ORDER_END.\n", filename);
                    apu_destroy_dnn(dnn);
                    return NULL;
                }
                if ((ptr = read_check_and_return (*fp_t, line, "TENSOR_DIMS:", line_num)) == NULL)
                {
                    FPRT(stderr,"ASPEN DNN file %s parse error: Missing TENSOR_DIMS.\n", filename);
                    apu_destroy_dnn(dnn);
                    return NULL;
                }
                for (unsigned int k = 0; k < NUM_PARAM_ELEMENTS; k++)
                {
                    void * tmp = fgets (line, MAX_STRING_LEN, *fp_t);
                    if (tmp == NULL)
                    {
                        FPRT(stderr,"ASPEN DNN file %s parse error at source line %d, soruce file %s\n", 
                            filename, __LINE__, __FILE__);
                        apu_destroy_dnn(dnn);
                        return NULL;
                    }
                    *line_num += 1;
                    ptr = line;
                    while (*ptr == ' ' || *ptr == '\t')
                        ptr++;
                    int dim_idx = 0;
                    int dim_val = 0;
                    sscanf(ptr, "%d %d", &dim_idx, &dim_val);
                    tensor->dims[dim_idx] = dim_val;
                }
                if ((ptr = read_check_and_return (*fp_t, line, "TENSOR_DIMS_END", line_num)) == NULL)
                {
                    FPRT(stderr,"ASPEN DNN file %s parse error: Missing TENSOR_DIMS_END.\n", filename);
                    apu_destroy_dnn(dnn);
                    return NULL;
                }
                if ((ptr = read_check_and_return (*fp_t, line, "TENSOR_DATA:", line_num)) == NULL)
                {
                    FPRT(stderr,"ASPEN DNN file %s parse error: Missing TENSOR_DATA.\n", filename);
                    apu_destroy_dnn(dnn);
                    return NULL;
                }
                int num_elements = 1;
                for (unsigned int k = 0; k < tensor->num_dims; k++)
                {
                    num_elements *= tensor->dims[tensor->data_dim_order[k]];
                }
                tensor->num_elements = num_elements;
                tensor->element_size = layer->dnn->element_size;
                if (skip_alloc == 0)
                {
                    tensor->data = aspen_calloc (num_elements, layer->dnn->element_size);
                    if (tensor->data == NULL)
                    {
                        FPRT(stderr,"ASPEN DNN file %s parse error: Failed to allocate tensor data.\n", filename);
                        apu_destroy_dnn(dnn);
                        return NULL;
                    }
                    size_t val = fread (tensor->data, layer->dnn->element_size, num_elements, *fp_t);
                    if (val != num_elements)
                    {
                        FPRT(stderr,"ASPEN DNN file %s parse error: Failed to read tensor data.\n", filename);
                        apu_destroy_dnn(dnn);
                        return NULL;
                    }
                    if (aspen_num_gpus > 0)
                    {
                        calloc_aspen_gpu_tensors (tensor);
                        for (unsigned int k = 0; k < aspen_num_gpus; k++)
                        {
                            copy_aspen_tensor_to_gpu (tensor, k);
                        }
                    }
                }
                else
                {
                    fseek (*fp_t, num_elements*layer->dnn->element_size, SEEK_CUR);
                }
                if ((ptr = read_check_and_return (*fp_t, line, "TENSOR_DATA_END", line_num)) == NULL)
                {
                    FPRT(stderr,"ASPEN DNN file %s parse error: Missing TENSOR_DATA_END.\n", filename);
                    apu_destroy_dnn(dnn);
                    return NULL;
                }
            }
        }
        if ((ptr = read_check_and_return (*fp_t, line, "LAYER_TENSORS_END", line_num)) == NULL)
        {
            FPRT(stderr,"ASPEN DNN file %s parse error: Missing LAYER_TENSORS_END.\n", filename);
            apu_destroy_dnn(dnn);
            return NULL;
        }
    }
    if ((ptr = read_check_and_return (*fp_t, line, "ASPEN_DNN_END", line_num)) == NULL)
    {
        FPRT(stderr,"ASPEN DNN file %s parse error: Missing ASPEN_DNN_END.\n", filename);
        apu_destroy_dnn(dnn);
        return NULL;
    }


    return dnn;
}

aspen_dnn_t *apu_load_dnn_from_file(char *filename)
{
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL)
    {
        FPRT(stderr,"ASPEN DNN file %s not found.\n", filename);
        return NULL;
    }
    unsigned int line_num = 0;
    aspen_dnn_t *dnn = apu_parse_dnn_from_file (filename, &fp, &line_num, 0);
    fclose (fp);
    return dnn;
}

void apu_save_nasm_to_file(nasm_t *nasm, char *filename)
{
    if (nasm == NULL)
    {
        FPRT(stderr,"ASPEN NASM to save is null.\n");
        return;
    }
    if (filename == NULL)
    {
        FPRT(stderr,"ASPEN NASM file name not specified.\n");
        return;
    }
    FILE *fp = fopen (filename, "wb");
    if (fp == NULL)
    {
        FPRT(stderr, "Error: apu_save_dnn_to_file Failed to open file %s for writing\n", filename);
        return;
    }
    fprintf (fp, "ASPEN_NASM\n");
    fprintf (fp, "DNN_NAME:%s\n", nasm->dnn->name);
    fprintf (fp, "NUM_BATCH:%d\n", nasm->batch_size);
    fprintf (fp, "MIN_NINST_PER_LDATA:%d\n", nasm->min_ninst_per_ldata);
    fprintf (fp, "TOTAL_FLOPS:%ld\n", nasm->total_flops);
    fprintf (fp, "FLOP_PER_NINST:%d\n", nasm->flop_per_ninst);
    fprintf (fp, "SEQ_LEN:%d\n", nasm->tr_seq_len);
    fprintf (fp, "NASM_NINSTS:\n");
    for (unsigned int i = 0; i < nasm->num_ldata; i++)
    {
        for (unsigned int j = 0; j < nasm->ldata_arr[i].num_ninst; j++)
        {
            ninst_t *ninst = nasm->ldata_arr[i].ninst_arr_start + j;
            fprintf (fp, "\tNINST_IDX:%ld\n", ninst - nasm->ninst_arr);
            fprintf (fp, "\tNUM_CHILD_NINSTS:%d\n", ninst->num_child_ninsts);
            fprintf (fp, "\tNUM_PARENT_NINSTS:%d\n", ninst->num_parent_ninsts);
            fprintf (fp, "\tPARENT_NINSTS:\n");
            for (unsigned int k = 0; k < ninst->num_parent_ninsts; k++)
            {
                fprintf (fp, "\t\t%d %d\n", k, ninst->parent_ninst_idx_arr[k]);
            }
            fprintf (fp, "\tPARENT_NINSTS_END\n");
            fprintf (fp, "\tNUM_CHILD_NINSTS:%d\n", ninst->num_child_ninsts);
            fprintf (fp, "\tCHILD_NINST_IDXES:\n");
            for (unsigned int k = 0; k < ninst->num_child_ninsts; k++)
            {
                fprintf (fp, "\t\t%d %ld\n", k, ninst->child_ninst_arr[k] - nasm->ninst_arr);
            }
            fprintf (fp, "\tCHILD_NINST_IDXES_END\n");
            fprintf (fp, "\tNUM_INPUT_POS:%d\n", ninst->num_input_pos);
            fprintf (fp, "\tINPUT_POS:\n");
            fwrite (ninst->input_pos_idx_arr, sizeof(int), ninst->num_input_pos, fp);
            fprintf (fp, "\tINPUT_POS_END\n");
        }
    }
    fprintf (fp, "NASM_NINSTS_END\n");
    fprintf (fp, "ASPEN_NASM_END\n");
    fclose (fp);
}

nasm_t *apu_load_nasm_from_file(char *filename, aspen_dnn_t *dnn)
{
    if (dnn == NULL)
    {
        FPRT(stderr,"ASPEN DNN NASM load error: dnn is null.\n");
        return NULL;
    }
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL)
    {
        FPRT(stderr,"ASPEN DNN NASM load error: file %s not found.\n", filename);
        return NULL;
    }
    char line[MAX_STRING_LEN] = {0};
    char* ptr;
    unsigned int line_num = 0;
    unsigned int flop_per_ninst = 0, batch_size = 0, min_ninst_per_ldata = 0, tr_seq_len = 0;

    nasm_t *nasm = NULL;
    if ((ptr = read_check_and_return (fp, line, "ASPEN_NASM", &line_num)) == NULL)
    {
        FPRT(stderr,"ASPEN DNN file %s parse error: Not an ASPEN NASM file.\n", filename);
        fclose (fp);
        return NULL;
    }
    if ((ptr = read_check_and_return (fp, line, "DNN_NAME:", &line_num)) == NULL)
    {
        FPRT(stderr,"ASPEN DNN file %s parse error: Missing DNN_NAME.\n", filename);
        fclose (fp);
        return NULL;
    }
    if (strcmp(ptr, dnn->name) != 0)
    {
        FPRT(stderr,"ASPEN DNN file %s parse error: DNN_NAME %s does not match dnn name %s.\n", filename, ptr, dnn->name);
        fclose (fp);
        return NULL;
    }
    if ((ptr = read_check_and_return (fp, line, "NUM_BATCH:", &line_num)) == NULL)
    {
        FPRT(stderr,"ASPEN DNN file %s parse error: Missing NUM_BATCH.\n", filename);
        fclose (fp);
        return NULL;
    }
    batch_size = atoi(ptr);
    if ((ptr = read_check_and_return (fp, line, "MIN_NINST_PER_LDATA:", &line_num)) == NULL)
    {
        FPRT(stderr,"ASPEN DNN file %s parse error: Missing MIN_NINST_PER_LDATA.\n", filename);
        fclose (fp);
        return NULL;
    }
    min_ninst_per_ldata = atoi(ptr);
    if ((ptr = read_check_and_return (fp, line, "TOTAL_FLOPS:", &line_num)) == NULL)
    {
        FPRT(stderr,"ASPEN DNN file %s parse error: Missing TOTAL_FLOPS.\n", filename);
        fclose (fp);
        return NULL;
    }
    unsigned long total_flops = atol(ptr);
    if ((ptr = read_check_and_return (fp, line, "FLOP_PER_NINST:", &line_num)) == NULL)
    {
        FPRT(stderr,"ASPEN DNN file %s parse error: Missing FLOP_PER_NINST.\n", filename);
        fclose (fp);
        return NULL;
    }
    flop_per_ninst = atoi(ptr);
    if ((ptr = read_check_and_return (fp, line, "SEQ_LEN:", &line_num)) == NULL)
    {
        FPRT(stderr,"ASPEN DNN file %s parse error: Missing SEQ_LEN.\n", filename);
        fclose (fp);
        return NULL;
    }
    tr_seq_len = atoi(ptr);
    nasm = apu_create_nasm_without_finding_ninst_parents 
        (dnn, flop_per_ninst, batch_size, min_ninst_per_ldata, tr_seq_len);
    if (nasm == NULL)
    {
        FPRT(stderr,"ASPEN DNN file %s parse error: Failed to create NASM.\n", filename);
        fclose (fp);
        return NULL;
    }
    nasm->total_flops = total_flops;
    if ((ptr = read_check_and_return (fp, line, "NASM_NINSTS:", &line_num)) == NULL)
    {
        FPRT(stderr,"ASPEN DNN file %s parse error: Missing NASM_NINSTS.\n", filename);
        apu_destroy_nasm (nasm);
        fclose (fp);
        return NULL;
    }
    for (unsigned int i = 0; i < nasm->num_ldata; i++)
    {
        for (unsigned int j = 0; j < nasm->ldata_arr[i].num_ninst; j++)
        {
            ninst_t *ninst = nasm->ldata_arr[i].ninst_arr_start + j;
            if ((ptr = read_check_and_return (fp, line, "NINST_IDX:", &line_num)) == NULL)
            {
                FPRT(stderr,"ASPEN DNN file %s parse error: Missing NINST_IDX.\n", filename);
                apu_destroy_nasm (nasm);
                fclose (fp);
                return NULL;
            }
            unsigned int ninst_idx = atoi(ptr);
            if (ninst_idx != (ninst - nasm->ninst_arr))
            {
                FPRT(stderr,"ASPEN DNN file %s parse error: NINST_IDX mismatch.\n", filename);
                apu_destroy_nasm (nasm);
                fclose (fp);
                return NULL;
            }
            if ((ptr = read_check_and_return (fp, line, "NUM_CHILD_NINSTS:", &line_num)) == NULL)
            {
                FPRT(stderr,"ASPEN DNN file %s parse error: Missing NUM_CHILD_NINSTS.\n", filename);
                apu_destroy_nasm (nasm);
                fclose (fp);
                return NULL;
            }
            ninst->num_child_ninsts = atoi(ptr);
            if ((ptr = read_check_and_return (fp, line, "NUM_PARENT_NINSTS:", &line_num)) == NULL)
            {
                FPRT(stderr,"ASPEN DNN file %s parse error: Missing NUM_PARENT_NINSTS.\n", filename);
                apu_destroy_nasm (nasm);
                fclose (fp);
                return NULL;
            }
            ninst->num_parent_ninsts = atoi(ptr);
            ninst->parent_ninst_idx_arr = calloc (ninst->num_parent_ninsts, sizeof(unsigned int));
            if ((ptr = read_check_and_return (fp, line, "PARENT_NINSTS:", &line_num)) == NULL)
            {
                FPRT(stderr,"ASPEN DNN file %s parse error: Missing PARENT_NINSTS.\n", filename);
                apu_destroy_nasm (nasm);
                fclose (fp);
                return NULL;
            }
            for (unsigned int k = 0; k < ninst->num_parent_ninsts; k++)
            {
                void * tmp = fgets (line, MAX_STRING_LEN, fp);
                if (tmp == NULL)
                {
                    FPRT(stderr,"ASPEN DNN file %s parse error at source line %d, soruce file %s\n", 
                        filename, __LINE__, __FILE__);
                    apu_destroy_nasm (nasm);
                    fclose (fp);
                    return NULL;
                }
                line_num += 1;
                ptr = line;
                while (*ptr == ' ' || *ptr == '\t')
                    ptr++;
                int parent_num = 0;
                int parent_ninst_idx = 0;
                sscanf(ptr, "%d %d", &parent_num, &parent_ninst_idx);
                ninst->parent_ninst_idx_arr[parent_num] = parent_ninst_idx;
            }
            if ((ptr = read_check_and_return (fp, line, "PARENT_NINSTS_END", &line_num)) == NULL)
            {
                FPRT(stderr,"ASPEN DNN file %s parse error: Missing PARENT_NINSTS_END.\n", filename);
                apu_destroy_nasm (nasm);
                fclose (fp);
                return NULL;
            }
            if ((ptr = read_check_and_return (fp, line, "NUM_CHILD_NINSTS:", &line_num)) == NULL)
            {
                FPRT(stderr,"ASPEN DNN file %s parse error: Missing NUM_CHILD_NINSTS.\n", filename);
                apu_destroy_nasm (nasm);
                fclose (fp);
                return NULL;
            }
            ninst->num_child_ninsts = atoi(ptr);
            if ((ptr = read_check_and_return (fp, line, "CHILD_NINST_IDXES:", &line_num)) == NULL)
            {
                FPRT(stderr,"ASPEN DNN file %s parse error: Missing CHILD_NINST_IDXES.\n", filename);
                apu_destroy_nasm (nasm);
                fclose (fp);
                return NULL;
            }
            ninst->child_ninst_arr = calloc (ninst->num_child_ninsts, sizeof(ninst_t *));
            for (unsigned int k = 0; k < ninst->num_child_ninsts; k++)
            {
                void * tmp = fgets (line, MAX_STRING_LEN, fp);
                if (tmp == NULL)
                {
                    FPRT(stderr,"ASPEN DNN file %s parse error at source line %d, soruce file %s\n", 
                        filename, __LINE__, __FILE__);
                    apu_destroy_nasm (nasm);
                    fclose (fp);
                    return NULL;
                }
                line_num += 1;
                ptr = line;
                while (*ptr == ' ' || *ptr == '\t')
                    ptr++;
                int child_num = 0;
                size_t child_ninst_idx = 0;
                sscanf(ptr, "%d %ld", &child_num, &child_ninst_idx);
                ninst->child_ninst_arr[child_num] = child_ninst_idx + nasm->ninst_arr;
            }
            if ((ptr = read_check_and_return (fp, line, "CHILD_NINST_IDXES_END", &line_num)) == NULL)
            {
                FPRT(stderr,"ASPEN DNN file %s parse error: Missing CHILD_NINST_IDXES_END.\n", filename);
                apu_destroy_nasm (nasm);
                fclose (fp);
                return NULL;
            }
            if ((ptr = read_check_and_return (fp, line, "NUM_INPUT_POS:", &line_num)) == NULL)
            {
                FPRT(stderr,"ASPEN DNN file %s parse error: Missing NUM_INPUT_POS.\n", filename);
                apu_destroy_nasm (nasm);
                fclose (fp);
                return NULL;
            }
            ninst->num_input_pos = atoi(ptr);
            if (ninst->num_input_pos > 0)
                ninst->input_pos_idx_arr = calloc (ninst->num_input_pos, sizeof(int));
            if ((ptr = read_check_and_return (fp, line, "INPUT_POS:", &line_num)) == NULL)
            {
                FPRT(stderr,"ASPEN DNN file %s parse error: Missing INPUT_POS.\n", filename);
                apu_destroy_nasm (nasm);
                fclose (fp);
                return NULL;
            }
            size_t val = fread (ninst->input_pos_idx_arr, sizeof(int), ninst->num_input_pos, fp);
            if (val != ninst->num_input_pos)
            {
                FPRT(stderr,"ASPEN DNN file %s parse error at source line %d, soruce file %s\n", 
                    filename, __LINE__, __FILE__);
                apu_destroy_nasm (nasm);
                fclose (fp);
                return NULL;
            }
            if ((ptr = read_check_and_return (fp, line, "INPUT_POS_END", &line_num)) == NULL)
            {
                FPRT(stderr,"ASPEN DNN file %s parse error: Missing INPUT_POS_END.\n", filename);
                apu_destroy_nasm (nasm);
                fclose (fp);
                return NULL;
            }
        }
    }
    if ((ptr = read_check_and_return (fp, line, "NASM_NINSTS_END", &line_num)) == NULL)
    {
        FPRT(stderr,"ASPEN DNN file %s parse error: Missing NASM_NINSTS_END.\n", filename);
        apu_destroy_nasm (nasm);
        fclose (fp);
        return NULL;
    }
    if ((ptr = read_check_and_return (fp, line, "ASPEN_NASM_END", &line_num)) == NULL)
    {
        FPRT(stderr,"ASPEN DNN file %s parse error: Missing ASPEN_NASM_END.\n", filename);
        apu_destroy_nasm (nasm);
        fclose (fp);
        return NULL;
    }
    fclose (fp);
    return nasm;
}