#include "aspen.h"
#include "apu.h"
#include "nasm.h"
#include "input_parser.h"

char* branch_info = BUILD_INFO_BRANCH;

void apu_save_dnn_to_file(aspen_dnn_t *dnn, char *filename)
{
    FILE *fp = fopen(filename, "wb");
    fprintf(fp, "ASPEN_DNN\n");
    fprintf(fp, "ASPEN_BUILD:%s\n", branch_info);
    fprintf(fp, "DNN_NAME:%s\n", dnn->name);
    fprintf(fp, "DNN_ELEMENT_SIZE:%ld\n", dnn->element_size);
    fprintf(fp, "NUM_LAYERS:%d\n", dnn->num_layers);
    for (int i = 0; i < dnn->num_layers; i++)
    {
        aspen_layer_t *layer = dnn->layers + i;
        fprintf(fp, "\tLAYER_IDX:%d\n", layer->layer_idx);
        fprintf(fp, "\tLAYER_TYPE:%d\n", layer->type);
        fprintf(fp, "\tLAYER_ACTIVATION:%d\n", layer->activation);
        fprintf(fp, "\tLAYER_PARENTS:\n");
        for (int j = 0; j < NUM_PARENT_ELEMENTS; j++)
        {
            if (layer->parent_layers[j] != NULL)
                fprintf(fp, "\t\t%d %d\n", j, layer->parent_layers[j]->layer_idx);
            else
                fprintf(fp, "\t\t%d -1\n", j);
        }
        fprintf(fp, "\tLAYER_PARENTS_END\n");
        fprintf(fp, "\tLAYER_PARAMS:\n");
        for (int j = 0; j < NUM_PARAM_ELEMENTS; j++)
        {
            fprintf(fp, "\t\t%d %d\n", j, layer->params[j]);
        }
        fprintf(fp, "\tLAYER_PARAMS_END\n");
        fprintf(fp, "\tLAYER_TENSORS:\n");
        for (int j = 0; j < NUM_TENSOR_ELEMENTS; j++)
        {
            if (layer->tensors[j] != NULL)
            {
                aspen_tensor_t *tensor = layer->tensors[j];
                fprintf(fp, "\t\t%d 1\n", j);
                fprintf(fp, "\t\tNUM_DIMS:%d\n", tensor->num_dims);
                fprintf(fp, "\t\tDATA_DIM_ORDER:\n");
                for (int k = 0; k < MAX_TENSOR_DIMS; k++)
                {
                    fprintf(fp, "\t\t\t%d %d\n", k, tensor->data_dim_order[k]);
                }
                fprintf(fp, "\t\tDATA_DIM_ORDER_END\n");
                fprintf(fp, "\t\tTENSOR_DIMS:\n");
                for (int k = 0; k < NUM_PARAM_ELEMENTS; k++)
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

char *read_check_and_return (FILE *fp, char *buffer, char *check_str, unsigned int *line_num)
{
    if (fgets(buffer, MAX_STRING_LEN, fp) == NULL)
    {
        FPRT(stderr,"ASPEN DNN file parse error: Unexpected EOF.\n");
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

aspen_dnn_t *apu_load_dnn_from_file(char *filename)
{
    FILE *fp = fopen(filename, "rb");
    char dnn_name[MAX_STRING_LEN] = {0};
    unsigned int num_layers = 0;
    char line[MAX_STRING_LEN] = {0};
    char* ptr;
    unsigned int line_num = 0;
    if ((ptr = read_check_and_return (fp, line, "ASPEN_DNN", &line_num)) == NULL)
    {
        FPRT(stderr,"ASPEN DNN file %s parse error: Not an ASPEN DNN file.\n", filename);
        fclose (fp);
        return NULL;
    }
    if ((ptr = read_check_and_return (fp, line, "ASPEN_BUILD:", &line_num)) == NULL)
    {
        FPRT(stderr,"ASPEN DNN file %s parse error: Missing ASPEN_BUILD.\n", filename);
        fclose (fp);
        return NULL;
    }
    if ((ptr = read_check_and_return (fp, line, "DNN_NAME:", &line_num)) == NULL)
    {
        FPRT(stderr,"ASPEN DNN file %s parse error: Missing DNN_NAME.\n", filename);
        fclose (fp);
        return NULL;
    }
    sscanf(ptr, "%s", dnn_name);
    if ((ptr = read_check_and_return (fp, line, "DNN_ELEMENT_SIZE:", &line_num)) == NULL)
    {
        FPRT(stderr,"ASPEN DNN file %s parse error: Missing DNN_ELEMENT_SIZE.\n", filename);
        fclose (fp);
        return NULL;
    }
    size_t element_size = atol(ptr);
    if ((ptr = read_check_and_return (fp, line, "NUM_LAYERS:", &line_num)) == NULL)
    {
        FPRT(stderr,"ASPEN DNN file %s parse error: Missing NUM_LAYERS.\n", filename);
        fclose (fp);
        return NULL;
    }
    num_layers = atoi(ptr);
    aspen_dnn_t *dnn = init_aspen_dnn(num_layers, dnn_name);
    if (dnn == NULL)
    {
        FPRT(stderr,"ASPEN DNN file %s parse error: Failed to create DNN.\n", filename);
        fclose (fp);
        return NULL;
    }
    dnn->element_size = element_size;
    for (int i = 0; i < num_layers; i++)
    {
        aspen_layer_t *layer = dnn->layers + i;
        if ((ptr = read_check_and_return (fp, line, "LAYER_IDX:", &line_num)) == NULL)
        {
            FPRT(stderr,"ASPEN DNN file %s parse error: Missing LAYER_IDX.\n", filename);
            fclose (fp);
            aspen_destroy_dnn(dnn);
            return NULL;
        }
        layer->layer_idx = atoi(ptr);
        if ((ptr = read_check_and_return (fp, line, "LAYER_TYPE:", &line_num)) == NULL)
        {
            FPRT(stderr,"ASPEN DNN file %s parse error: Missing LAYER_TYPE.\n", filename);
            fclose (fp);
            aspen_destroy_dnn(dnn);
            return NULL;
        }
        layer->type = atoi(ptr);
        if ((ptr = read_check_and_return (fp, line, "LAYER_ACTIVATION:", &line_num)) == NULL)
        {
            FPRT(stderr,"ASPEN DNN file %s parse error: Missing LAYER_ACTIVATION.\n", filename);
            fclose (fp);
            aspen_destroy_dnn(dnn);
            return NULL;
        }
        layer->activation = atoi(ptr);
        if ((ptr = read_check_and_return (fp, line, "LAYER_PARENTS:", &line_num)) == NULL)
        {
            FPRT(stderr,"ASPEN DNN file %s parse error: Missing LAYER_PARENTS.\n", filename);
            fclose (fp);
            aspen_destroy_dnn(dnn);
            return NULL;
        }
        for (int j = 0; j < NUM_PARENT_ELEMENTS; j++)
        {
            fgets (line, MAX_STRING_LEN, fp);
            line_num++;
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
        if ((ptr = read_check_and_return (fp, line, "LAYER_PARENTS_END", &line_num)) == NULL)
        {
            FPRT(stderr,"ASPEN DNN file %s parse error: Missing LAYER_PARENTS_END.\n", filename);
            fclose (fp);
            aspen_destroy_dnn(dnn);
            return NULL;
        }
        if ((ptr = read_check_and_return (fp, line, "LAYER_PARAMS:", &line_num)) == NULL)
        {
            FPRT(stderr,"ASPEN DNN file %s parse error: Missing LAYER_PARAMS.\n", filename);
            fclose (fp);
            aspen_destroy_dnn(dnn);
            return NULL;
        }
        for (int j = 0; j < NUM_PARAM_ELEMENTS; j++)
        {
            fgets (line, MAX_STRING_LEN, fp);
            line_num++;
            ptr = line;
            while (*ptr == ' ' || *ptr == '\t')
                ptr++;
            int param_idx = 0;
            int param_val = 0;
            sscanf(ptr, "%d %d", &param_idx, &param_val);
            layer->params[param_idx] = param_val;
        }
        if ((ptr = read_check_and_return (fp, line, "LAYER_PARAMS_END", &line_num)) == NULL)
        {
            FPRT(stderr,"ASPEN DNN file %s parse error: Missing LAYER_PARAMS_END.\n", filename);
            fclose (fp);
            aspen_destroy_dnn(dnn);
            return NULL;
        }
        if ((ptr = read_check_and_return (fp, line, "LAYER_TENSORS:", &line_num)) == NULL)
        {
            FPRT(stderr,"ASPEN DNN file %s parse error: Missing LAYER_TENSORS.\n", filename);
            fclose (fp);
            aspen_destroy_dnn(dnn);
            return NULL;
        }
        for (int j = 0; j < NUM_TENSOR_ELEMENTS; j++)
        {
            fgets (line, MAX_STRING_LEN, fp);
            line_num++;
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
                    fclose (fp);
                    aspen_destroy_dnn(dnn);
                    return NULL;
                }
                if ((ptr = read_check_and_return (fp, line, "NUM_DIMS:", &line_num)) == NULL)
                {
                    FPRT(stderr,"ASPEN DNN file %s parse error: Missing NUM_DIMS.\n", filename);
                    fclose (fp);
                    aspen_destroy_dnn(dnn);
                    return NULL;
                }
                tensor->num_dims = atoi(ptr);
                if ((ptr = read_check_and_return (fp, line, "DATA_DIM_ORDER:", &line_num)) == NULL)
                {
                    FPRT(stderr,"ASPEN DNN file %s parse error: Missing DATA_DIM_ORDER.\n", filename);
                    fclose (fp);
                    aspen_destroy_dnn(dnn);
                    return NULL;
                }
                for (int k = 0; k < MAX_TENSOR_DIMS; k++)
                {
                    fgets (line, MAX_STRING_LEN, fp);
                    line_num++;
                    ptr = line;
                    while (*ptr == ' ' || *ptr == '\t')
                        ptr++;
                    int dim_idx = 0;
                    int dim_type = 0;
                    sscanf(ptr, "%d %d", &dim_idx, &dim_type);
                    tensor->data_dim_order[dim_idx] = dim_type;
                }
                if ((ptr = read_check_and_return (fp, line, "DATA_DIM_ORDER_END", &line_num)) == NULL)
                {
                    FPRT(stderr,"ASPEN DNN file %s parse error: Missing DATA_DIM_ORDER_END.\n", filename);
                    fclose (fp);
                    aspen_destroy_dnn(dnn);
                    return NULL;
                }
                if ((ptr = read_check_and_return (fp, line, "TENSOR_DIMS:", &line_num)) == NULL)
                {
                    FPRT(stderr,"ASPEN DNN file %s parse error: Missing TENSOR_DIMS.\n", filename);
                    fclose (fp);
                    aspen_destroy_dnn(dnn);
                    return NULL;
                }
                for (int k = 0; k < NUM_PARAM_ELEMENTS; k++)
                {
                    fgets (line, MAX_STRING_LEN, fp);
                    line_num++;
                    ptr = line;
                    while (*ptr == ' ' || *ptr == '\t')
                        ptr++;
                    int dim_idx = 0;
                    int dim_val = 0;
                    sscanf(ptr, "%d %d", &dim_idx, &dim_val);
                    tensor->dims[dim_idx] = dim_val;
                }
                if ((ptr = read_check_and_return (fp, line, "TENSOR_DIMS_END", &line_num)) == NULL)
                {
                    FPRT(stderr,"ASPEN DNN file %s parse error: Missing TENSOR_DIMS_END.\n", filename);
                    fclose (fp);
                    aspen_destroy_dnn(dnn);
                    return NULL;
                }
                if ((ptr = read_check_and_return (fp, line, "TENSOR_DATA:", &line_num)) == NULL)
                {
                    FPRT(stderr,"ASPEN DNN file %s parse error: Missing TENSOR_DATA.\n", filename);
                    fclose (fp);
                    aspen_destroy_dnn(dnn);
                    return NULL;
                }
                int num_elements = 1;
                for (int k = 0; k < tensor->num_dims; k++)
                {
                    num_elements *= tensor->dims[tensor->data_dim_order[k]];
                }
                tensor->num_elements = num_elements;
                tensor->data = aspen_calloc (num_elements, layer->dnn->element_size);
                if (tensor->data == NULL)
                {
                    FPRT(stderr,"ASPEN DNN file %s parse error: Failed to allocate tensor data.\n", filename);
                    fclose (fp);
                    aspen_destroy_dnn(dnn);
                    return NULL;
                }
                fread (tensor->data, layer->dnn->element_size, num_elements, fp);
                if ((ptr = read_check_and_return (fp, line, "TENSOR_DATA_END", &line_num)) == NULL)
                {
                    FPRT(stderr,"ASPEN DNN file %s parse error: Missing TENSOR_DATA_END.\n", filename);
                    fclose (fp);
                    aspen_destroy_dnn(dnn);
                    return NULL;
                }
            }
        }
        if ((ptr = read_check_and_return (fp, line, "LAYER_TENSORS_END", &line_num)) == NULL)
        {
            FPRT(stderr,"ASPEN DNN file %s parse error: Missing LAYER_TENSORS_END.\n", filename);
            fclose (fp);
            aspen_destroy_dnn(dnn);
            return NULL;
        }
    }
    if ((ptr = read_check_and_return (fp, line, "ASPEN_DNN_END", &line_num)) == NULL)
    {
        FPRT(stderr,"ASPEN DNN file %s parse error: Missing ASPEN_DNN_END.\n", filename);
        fclose (fp);
        aspen_destroy_dnn(dnn);
        return NULL;
    }
    fclose (fp);
    return dnn;
}

void apu_load_nasm_from_file(char *filename, nasm_t *output_nasm, aspen_dnn_t *output_dnn)
{

}

void apu_save_nasm_to_file(nasm_t *nasm, char *filename)
{
    
}