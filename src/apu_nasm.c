#include "aspen.h"
#include "apu.h"
#include "nasm.h"
#include "input_parser.h"

int get_nasm_ldata_num_per_layer (aspen_layer_t *layer)
{
    switch (layer->type)
    {
    default:
        return 1;
    }
}

void update_ldata_child_list (nasm_ldata_t *ldata)
{
    if (ldata->num_child_ldata == 0)
        return;
    ldata->child_ldata_idx_arr = calloc(ldata->num_child_ldata, sizeof(unsigned int));
    unsigned int child_idx = 0;
    for (int lidx = 0; lidx < ldata->nasm->num_ldata; lidx++)
    {
        nasm_ldata_t *child = ldata->nasm->ldata_arr + lidx;
        for (LAYER_PARENTS i = 0; i < NUM_PARENT_ELEMENTS; i++)
        {
            if (ldata->nasm->ldata_arr + child->parent_ldata_idx_arr[i] == ldata)
            {
                ldata->child_ldata_idx_arr[child_idx] = lidx;
                child_idx++;
            }
        }
    }
}

void ninst_find_input_pos_idx (ninst_t *ninst)
{
    nasm_ldata_t *ldata = ninst->ldata;
    aspen_layer_t *layer = ldata->layer;
    if (layer->type == CONV_LAYER || layer->type == MAXPOOL_LAYER || layer->type == AVGPOOL_LAYER)
    {
        unsigned int num_input_pos = ninst->tile_dims[OUT_W]*layer->params[WEIGHT_H]*layer->params[WEIGHT_W];
        ninst->num_input_pos = num_input_pos;
    }
}

// Change to add a new layer type
void ninst_find_parent (ninst_t *ninst)
{
    nasm_ldata_t *ldata = ninst->ldata;
    aspen_layer_t *layer = ldata->layer;
    ninst->num_parent_ninsts = 0;
    nasm_ldata_t *parent_ldata = ldata->nasm->ldata_arr + ldata->parent_ldata_idx_arr[PARENT_0];
    unsigned int *parent_arr = calloc (MAX_PARENT_NINST_NUM, sizeof(unsigned int));
    for (unsigned int tile_w = 0; tile_w < ninst->tile_dims[OUT_W]; tile_w++)
    {
        for (unsigned int tile_h = 0; tile_h < ninst->tile_dims[OUT_H]; tile_h++)
        {
            unsigned int out_mat_pos[2] = {ninst->out_mat_pos[OUT_W] + tile_w, ninst->out_mat_pos[OUT_H] + tile_h};
            unsigned int out_tensor_pos[NUM_PARAM_ELEMENTS] = {0}, in_tensor_pos[NUM_PARAM_ELEMENTS] = {0}; 
            get_tensor_pos_from_out_mat_pos(ldata, out_mat_pos, out_tensor_pos);
            in_tensor_pos[BATCH] = out_tensor_pos[BATCH];
            if (layer->type == CONV_LAYER || layer->type == MAXPOOL_LAYER || layer->type == AVGPOOL_LAYER)
            {
                for (int i = 0; i < layer->params[IN_C]; i++)
                {
                    in_tensor_pos[OUT_C] = i;
                    for (int j = 0; j < layer->params[WEIGHT_H]; j++)
                    {
                        in_tensor_pos[OUT_H] = out_tensor_pos[OUT_H]*layer->params[STRIDE]
                            + j*layer->params[DILATION] - layer->params[PADDING];
                        for (int k = 0; k < layer->params[WEIGHT_W]; k++)
                        {
                            in_tensor_pos[OUT_W] = out_tensor_pos[OUT_W]*layer->params[STRIDE]
                                + k*layer->params[DILATION] - layer->params[PADDING];
                            if (in_tensor_pos[BATCH] >= 0 && in_tensor_pos[BATCH] < ldata->nasm->batch_size && in_tensor_pos[OUT_C] >= 0 && in_tensor_pos[OUT_C] < layer->params[IN_C] &&
                                in_tensor_pos[OUT_H] >= 0 && in_tensor_pos[OUT_H] < layer->params[IN_H] && in_tensor_pos[OUT_W] >= 0 && in_tensor_pos[OUT_W] < layer->params[IN_W])
                            {
                                ninst_t *input_nist = get_ninst_from_tensor_pos(parent_ldata, in_tensor_pos);
                                int duplicate = 0;
                                for (int l = 0; l < ninst->num_parent_ninsts; l++)
                                {
                                    if (parent_arr[l] == input_nist->ninst_idx)
                                        duplicate = 1;
                                }
                                if (!duplicate)
                                {
                                    parent_arr[ninst->num_parent_ninsts] = input_nist->ninst_idx;
                                    ninst->num_parent_ninsts++;  
                                }
                            }
                        }
                    }
                }
            }
            else if (layer->type == RESIDUAL_LAYER)
            {
                if (layer->params[MAT_M] == 0)
                {
                    in_tensor_pos[OUT_C] = out_tensor_pos[OUT_C];
                    in_tensor_pos[OUT_H] = out_tensor_pos[OUT_H];
                    in_tensor_pos[OUT_W] = out_tensor_pos[OUT_W];
                }
                else
                {
                    in_tensor_pos[MAT_M] = out_tensor_pos[MAT_M];
                    in_tensor_pos[MAT_N] = out_tensor_pos[MAT_N];
                    in_tensor_pos[OUT_C] = out_tensor_pos[OUT_C];
                    in_tensor_pos[OUT_H] = out_tensor_pos[OUT_H];
                    in_tensor_pos[OUT_W] = out_tensor_pos[OUT_W];
                }
                if ((in_tensor_pos[BATCH] >= 0 && in_tensor_pos[BATCH] < ldata->nasm->batch_size && in_tensor_pos[OUT_C] >= 0 && in_tensor_pos[OUT_C] < layer->params[IN_C] &&
                    in_tensor_pos[OUT_H] >= 0 && in_tensor_pos[OUT_H] < layer->params[IN_H] && in_tensor_pos[OUT_W] >= 0 && in_tensor_pos[OUT_W] < layer->params[IN_W]) ||
                    (in_tensor_pos[BATCH] >= 0 && in_tensor_pos[BATCH] < ldata->nasm->batch_size &&
                    in_tensor_pos[MAT_M] >= 0 && in_tensor_pos[MAT_M] < layer->params[MAT_M] && in_tensor_pos[MAT_N] >= 0 && in_tensor_pos[MAT_N] < ldata->nasm->tr_seq_len))
                {
                    ninst_t *input_nist = get_ninst_from_tensor_pos(parent_ldata, in_tensor_pos);
                    int duplicate = 0;
                    for (int l = 0; l < ninst->num_parent_ninsts; l++)
                    {
                        if (parent_arr[l] == input_nist->ninst_idx)
                            duplicate = 1;
                    }
                    if (!duplicate)
                    {
                        parent_arr[ninst->num_parent_ninsts] = input_nist->ninst_idx;
                        ninst->num_parent_ninsts++;  
                    }
                    input_nist = get_ninst_from_tensor_pos(ldata->nasm->ldata_arr + ldata->parent_ldata_idx_arr[PARENT_1], in_tensor_pos);
                    for (int l = 0; l < ninst->num_parent_ninsts; l++)
                    {
                        if (parent_arr[l] == input_nist->ninst_idx)
                            duplicate = 1;
                    }
                    if (!duplicate)
                    {
                        parent_arr[ninst->num_parent_ninsts] = input_nist->ninst_idx;
                        ninst->num_parent_ninsts++;  
                    }
                }
            }
            else if (layer->type == FC_LAYER)
            {
                if (parent_ldata->out_mat_stride != parent_ldata->out_mat_dims[OUT_H])
                {
                    ERROR_PRTF ("ERROR: FC layer parent stride mismatatch - %d, %d", 
                        parent_ldata->out_mat_stride, parent_ldata->out_mat_dims[OUT_H]);
                    assert (0);
                }
                if (in_tensor_pos[BATCH] >= 0 && in_tensor_pos[BATCH] < ldata->nasm->batch_size)
                {
                    for (int i = 0; i < layer->params[IN_C]; i++)
                    {
                        in_tensor_pos[OUT_C] = i;
                        for (int j = 0; j < layer->params[IN_H]; j++)
                        {
                            in_tensor_pos[OUT_H] = j;
                            for (int k = 0; k < layer->params[IN_W]; k++)
                            {
                                in_tensor_pos[OUT_W] = k;
                                if (in_tensor_pos[BATCH] >= 0 && in_tensor_pos[BATCH] < ldata->nasm->batch_size && in_tensor_pos[OUT_C] >= 0 && in_tensor_pos[OUT_C] < layer->params[IN_C] &&
                                    in_tensor_pos[OUT_H] >= 0 && in_tensor_pos[OUT_H] < layer->params[IN_H] && in_tensor_pos[OUT_W] >= 0 && in_tensor_pos[OUT_W] < layer->params[IN_W])
                                {
                                    ninst_t *input_nist = get_ninst_from_tensor_pos(parent_ldata, in_tensor_pos);
                                    int duplicate = 0;
                                    for (int l = 0; l < ninst->num_parent_ninsts; l++)
                                    {
                                        if (parent_arr[l] == input_nist->ninst_idx)
                                        {
                                            duplicate = 1;
                                            break;
                                        }
                                    }
                                    if (!duplicate)
                                    {
                                        parent_arr[ninst->num_parent_ninsts] = input_nist->ninst_idx;
                                        ninst->num_parent_ninsts++;
                                    }
                                }
                            }
                        }
                    }
                }
            } 
            else if (layer->type == SOFTMAX_LAYER)
            {
                memcpy (in_tensor_pos, out_tensor_pos, sizeof(int) * NUM_PARAM_ELEMENTS);
                if (in_tensor_pos[BATCH] >= 0 && in_tensor_pos[BATCH] < ldata->nasm->batch_size && in_tensor_pos[OUT_C] >= 0 && in_tensor_pos[OUT_C] < layer->params[IN_C] &&
                    in_tensor_pos[OUT_H] >= 0 && in_tensor_pos[OUT_H] < layer->params[IN_H] && in_tensor_pos[OUT_W] >= 0 && in_tensor_pos[OUT_W] < layer->params[IN_W])
                {
                    ninst_t *input_nist = get_ninst_from_tensor_pos(parent_ldata, in_tensor_pos);
                    int duplicate = 0;
                    for (int l = 0; l < ninst->num_parent_ninsts; l++)
                    {
                        if (parent_arr[l] == input_nist->ninst_idx)
                        {
                            duplicate = 1;
                            break;
                        }
                    }
                    if (!duplicate)
                    {
                        parent_arr[ninst->num_parent_ninsts] = input_nist->ninst_idx;
                        ninst->num_parent_ninsts++;
                    }
                }
            }
            else if (layer->type == YOLO_LAYER)
            {
                int aidx = out_tensor_pos[OUT_W] / (layer->params[IN_H] * layer->params[IN_W]);
                int wh = out_tensor_pos[OUT_W] % (layer->params[IN_H] * layer->params[IN_W]);
                in_tensor_pos[OUT_H] = wh / layer->params[IN_W];
                in_tensor_pos[OUT_W] = wh % layer->params[IN_W];
                in_tensor_pos[OUT_C] = out_tensor_pos[OUT_C] + aidx * layer->params[OUT_C];
                if (in_tensor_pos[BATCH] >= 0 && in_tensor_pos[BATCH] < ldata->nasm->batch_size && in_tensor_pos[OUT_C] >= 0 && in_tensor_pos[OUT_C] < layer->params[IN_C] &&
                    in_tensor_pos[OUT_H] >= 0 && in_tensor_pos[OUT_H] < layer->params[IN_H] && in_tensor_pos[OUT_W] >= 0 && in_tensor_pos[OUT_W] < layer->params[IN_W])
                {
                    ninst_t *input_nist = get_ninst_from_tensor_pos(parent_ldata, in_tensor_pos);
                    int duplicate = 0;
                    for (int l = 0; l < ninst->num_parent_ninsts; l++)
                    {
                        if (parent_arr[l] == input_nist->ninst_idx)
                        {
                            duplicate = 1;
                            break;
                        }
                    }
                    if (!duplicate)
                    {
                        parent_arr[ninst->num_parent_ninsts] = input_nist->ninst_idx;
                        ninst->num_parent_ninsts++;
                    }
                }
            }
            else if (layer->type == APPEND_LAYER)
            {
                nasm_ldata_t *target_ldata = NULL;
                if (out_tensor_pos[OUT_C] < layer->params[IN_C])
                {
                    target_ldata = ldata->nasm->ldata_arr + ldata->parent_ldata_idx_arr[PARENT_0];
                    in_tensor_pos[OUT_C] = out_tensor_pos[OUT_C];
                    in_tensor_pos[OUT_W] = out_tensor_pos[OUT_W] / layer->params[STRIDE];
                    in_tensor_pos[OUT_H] = out_tensor_pos[OUT_H] / layer->params[STRIDE];
                }
                else
                {
                    target_ldata = ldata->nasm->ldata_arr + ldata->parent_ldata_idx_arr[PARENT_1];
                    in_tensor_pos[OUT_C] = out_tensor_pos[OUT_C] - layer->params[IN_C];
                    in_tensor_pos[OUT_W] = out_tensor_pos[OUT_W];
                    in_tensor_pos[OUT_H] = out_tensor_pos[OUT_H];
                }
                if (in_tensor_pos[BATCH] >= 0 && in_tensor_pos[BATCH] < ldata->nasm->batch_size && in_tensor_pos[OUT_C] >= 0 && in_tensor_pos[OUT_C] < layer->params[IN_C] &&
                    in_tensor_pos[OUT_H] >= 0 && in_tensor_pos[OUT_H] < layer->params[IN_H] && in_tensor_pos[OUT_W] >= 0 && in_tensor_pos[OUT_W] < layer->params[IN_W])
                {
                    ninst_t *input_nist = get_ninst_from_tensor_pos(target_ldata, in_tensor_pos);
                    int duplicate = 0;
                    for (int l = 0; l < ninst->num_parent_ninsts; l++)
                    {
                        if (parent_arr[l] == input_nist->ninst_idx)
                        {
                            duplicate = 1;
                            break;
                        }
                    }
                    if (!duplicate)
                    {
                        parent_arr[ninst->num_parent_ninsts] = input_nist->ninst_idx;
                        ninst->num_parent_ninsts++;
                    }
                }
            }
            else if (layer->type == LAYERNORM_LAYER)
            {
                aspen_layer_t *p_layer = parent_ldata->layer;
                in_tensor_pos[MAT_M] = out_tensor_pos[MAT_M];
                in_tensor_pos[MAT_N] = out_tensor_pos[MAT_N];
                if (in_tensor_pos[BATCH] >= 0 && in_tensor_pos[BATCH] < ldata->nasm->batch_size && in_tensor_pos[MAT_M] >= 0 
                    && in_tensor_pos[MAT_M] < p_layer->params[MAT_M] && in_tensor_pos[MAT_N] >= 0 && in_tensor_pos[MAT_N] < ldata->nasm->tr_seq_len)
                {
                    ninst_t *input_nist = get_ninst_from_tensor_pos(parent_ldata, in_tensor_pos);
                    int duplicate = 0;
                    for (int l = 0; l < ninst->num_parent_ninsts; l++)
                    {
                        if (parent_arr[l] == input_nist->ninst_idx)
                        {
                            duplicate = 1;
                            break;
                        }
                    }
                    if (!duplicate)
                    {
                        parent_arr[ninst->num_parent_ninsts] = input_nist->ninst_idx;
                        ninst->num_parent_ninsts++;
                    }
                }
            }
            else if (layer->type == MATMUL_LAYER)
            {
                aspen_layer_t *p_layer = parent_ldata->layer;
                for (int i = 0; i < layer->params[MAT_K]; i++)
                {
                    in_tensor_pos[MAT_M] = i;
                    in_tensor_pos[MAT_N] = out_tensor_pos[MAT_N];
                    if (in_tensor_pos[BATCH] >= 0 && in_tensor_pos[BATCH] < ldata->nasm->batch_size && in_tensor_pos[MAT_M] >= 0 
                        && in_tensor_pos[MAT_M] < p_layer->params[MAT_M] && in_tensor_pos[MAT_N] >= 0 && in_tensor_pos[MAT_N] < ldata->nasm->tr_seq_len)
                    {
                        ninst_t *input_nist = get_ninst_from_tensor_pos(parent_ldata, in_tensor_pos);
                        int duplicate = 0;
                        for (int l = 0; l < ninst->num_parent_ninsts; l++)
                        {
                            if (parent_arr[l] == input_nist->ninst_idx)
                            {
                                duplicate = 1;
                                break;
                            }
                        }
                        if (!duplicate)
                        {
                            parent_arr[ninst->num_parent_ninsts] = input_nist->ninst_idx;
                            ninst->num_parent_ninsts++;
                        }
                    }
                }
            }
            else if (layer->type == K_ATTENTION_LAYER)
            {
                aspen_layer_t *p_layer = parent_ldata->layer;
                unsigned int hidden_per_head = layer->params[NUM_HIDDEN] / layer->params[NUM_HEAD];
                for (int k = 0; k < layer->params[MAT_K]; k++)
                {
                    in_tensor_pos[MAT_M] = k + out_tensor_pos[NUM_HEAD] * hidden_per_head;
                    in_tensor_pos[MAT_N] = out_tensor_pos[MAT_N];
                    if (in_tensor_pos[BATCH] >= 0 && in_tensor_pos[BATCH] < ldata->nasm->batch_size && in_tensor_pos[MAT_M] >= 0 
                        && in_tensor_pos[MAT_M] < p_layer->params[MAT_M] && in_tensor_pos[MAT_N] >= 0 && in_tensor_pos[MAT_N] < ldata->nasm->tr_seq_len)
                    {
                        ninst_t *input_nist = get_ninst_from_tensor_pos(parent_ldata, in_tensor_pos);
                        int duplicate = 0;
                        for (int l = 0; l < ninst->num_parent_ninsts; l++)
                        {
                            if (parent_arr[l] == input_nist->ninst_idx)
                            {
                                duplicate = 1;
                                break;
                            }
                        }
                        if (!duplicate)
                        {
                            parent_arr[ninst->num_parent_ninsts] = input_nist->ninst_idx;
                            ninst->num_parent_ninsts++;
                        }
                    }
                }
                nasm_ldata_t *parent_k_ldata = ldata->nasm->ldata_arr + ldata->parent_ldata_idx_arr[PARENT_1];
                aspen_layer_t *pk_layer = parent_k_ldata->layer;
                for (int k = 0; k < layer->params[MAT_K]; k++)
                {
                    in_tensor_pos[MAT_N] = out_tensor_pos[MAT_M];
                    in_tensor_pos[MAT_M] = k + out_tensor_pos[NUM_HEAD] * hidden_per_head;
                    if (in_tensor_pos[BATCH] >= 0 && in_tensor_pos[BATCH] < ldata->nasm->batch_size && in_tensor_pos[MAT_M] >= 0 
                    && in_tensor_pos[MAT_M] < pk_layer->params[MAT_M] && in_tensor_pos[MAT_N] >= 0 && in_tensor_pos[MAT_N] < ldata->nasm->tr_seq_len)
                    {
                        ninst_t *input_nist = get_ninst_from_tensor_pos(parent_k_ldata, in_tensor_pos);
                        int duplicate = 0;
                        for (int l = 0; l < ninst->num_parent_ninsts; l++)
                        {
                            if (parent_arr[l] == input_nist->ninst_idx)
                            {
                                duplicate = 1;
                                break;
                            }
                        }
                        if (!duplicate)
                        {
                            parent_arr[ninst->num_parent_ninsts] = input_nist->ninst_idx;
                            ninst->num_parent_ninsts++;
                        }
                    }
                }
            }
            else if (layer->type == V_ATTENTION_LAYER)
            {
                aspen_layer_t *p_layer = parent_ldata->layer;
                unsigned int hidden_per_head = layer->params[NUM_HIDDEN] / layer->params[NUM_HEAD];
                unsigned int head_pos = out_tensor_pos[MAT_M] / hidden_per_head;
                unsigned int seq_num = ldata->nasm->tr_seq_len;
                for (int k = 0; k < seq_num; k++)
                {
                    in_tensor_pos[NUM_HEAD] = head_pos;
                    in_tensor_pos[MAT_M] = k;
                    in_tensor_pos[MAT_N] = out_tensor_pos[MAT_N];
                    if (in_tensor_pos[BATCH] >= 0 && in_tensor_pos[BATCH] < ldata->nasm->batch_size && in_tensor_pos[MAT_M] >= 0 
                        && in_tensor_pos[MAT_M] < ldata->nasm->tr_seq_len && in_tensor_pos[MAT_N] >= 0 && in_tensor_pos[MAT_N] < ldata->nasm->tr_seq_len
                        && in_tensor_pos[NUM_HEAD] >= 0 && in_tensor_pos[NUM_HEAD] < p_layer->params[NUM_HEAD])
                    {
                        ninst_t *input_nist = get_ninst_from_tensor_pos(parent_ldata, in_tensor_pos);
                        int duplicate = 0;
                        for (int l = 0; l < ninst->num_parent_ninsts; l++)
                        {
                            if (parent_arr[l] == input_nist->ninst_idx)
                            {
                                duplicate = 1;
                                break;
                            }
                        }
                        if (!duplicate)
                        {
                            parent_arr[ninst->num_parent_ninsts] = input_nist->ninst_idx;
                            ninst->num_parent_ninsts++;
                        }
                    }
                }
                nasm_ldata_t *parent_v_ldata = ldata->nasm->ldata_arr + ldata->parent_ldata_idx_arr[PARENT_1];
                aspen_layer_t *pv_layer = parent_v_ldata->layer;
                for (int k = 0; k < seq_num; k++)
                {
                    in_tensor_pos[MAT_N] = k;
                    in_tensor_pos[MAT_M] = out_tensor_pos[MAT_M];
                    if (in_tensor_pos[BATCH] >= 0 && in_tensor_pos[BATCH] < ldata->nasm->batch_size && in_tensor_pos[MAT_M] >= 0 
                    && in_tensor_pos[MAT_M] < pv_layer->params[MAT_M] && in_tensor_pos[MAT_N] >= 0 && in_tensor_pos[MAT_N] < ldata->nasm->tr_seq_len)
                    {
                        ninst_t *input_nist = get_ninst_from_tensor_pos(parent_v_ldata, in_tensor_pos);
                        int duplicate = 0;
                        for (int l = 0; l < ninst->num_parent_ninsts; l++)
                        {
                            if (parent_arr[l] == input_nist->ninst_idx)
                            {
                                duplicate = 1;
                                break;
                            }
                        }
                        if (!duplicate)
                        {
                            parent_arr[ninst->num_parent_ninsts] = input_nist->ninst_idx;
                            ninst->num_parent_ninsts++;
                        }
                    }
                }
            }
            else if (layer->type == INPUT_LAYER)
            {
                return;
            }
            else
            {
                ERROR_PRTF ("ERROR: Unsupported layer type %s, at line %d in file %s" , layer_type_str[layer->type], 0, " ");
                assert (0);
            }
        }
    }
    ninst->parent_ninst_idx_arr = calloc(ninst->num_parent_ninsts, sizeof(unsigned int));
    memcpy (ninst->parent_ninst_idx_arr, parent_arr, ninst->num_parent_ninsts*sizeof(unsigned int));
    for (int i = 0; i < ninst->num_parent_ninsts; i++)
    {
        ninst_t *parent = ninst->parent_ninst_idx_arr[i] + ldata->nasm->ninst_arr;
        atomic_fetch_add (&parent->num_child_ninsts, 1);
    }
    free (parent_arr);
    ninst_find_input_pos_idx (ninst);
}

void ninst_init (nasm_ldata_t *ldata, ninst_t *ninst_ptr, int ninst_idx)
{
    ninst_ptr->ldata = ldata;
    ninst_ptr->state = NINST_NOT_READY;
    ninst_ptr->ninst_idx = ninst_idx;
    ninst_ptr->rank = 0;
    get_out_mat_pos_from_nist (ldata, ninst_ptr, ninst_ptr->out_mat_pos);
    ninst_ptr->tile_dims[OUT_W] = ninst_ptr->out_mat_pos[OUT_W] + ldata->ninst_tile_dims[OUT_W] > ldata->out_mat_dims[OUT_W]?
        ldata->out_mat_dims[OUT_W] - ninst_ptr->out_mat_pos[OUT_W]: ldata->ninst_tile_dims[OUT_W];
    ninst_ptr->tile_dims[OUT_H] = ninst_ptr->out_mat_pos[OUT_H] + ldata->ninst_tile_dims[OUT_H] > ldata->out_mat_dims[OUT_H]? 
        ldata->out_mat_dims[OUT_H] - ninst_ptr->out_mat_pos[OUT_H]: ldata->ninst_tile_dims[OUT_H];
    ninst_ptr->type_hash = get_ninst_hash (ninst_ptr);
}

HASH_t get_ninst_hash (ninst_t *ninst)
{
    struct val
    {
        unsigned int operator;
        unsigned int tile_dims[2];
        unsigned int ldata_dims[2];
    };
    struct val v = {.operator=ninst->ldata->layer->type, 
    .tile_dims[0]=ninst->tile_dims[0], .tile_dims[1]=ninst->tile_dims[1]
    , .ldata_dims[0]=ninst->ldata->out_mat_dims[0], .ldata_dims[1]=ninst->ldata->out_mat_dims[1]};
    return get_hash (&v, sizeof(struct val));
}

void destroy_ninst (ninst_t *ninst)
{
    if (ninst == NULL)
        return;
    if (ninst->parent_ninst_idx_arr != NULL)
        free (ninst->parent_ninst_idx_arr);
    if (ninst->child_ninst_arr != NULL)
        free (ninst->child_ninst_arr);
}

HASH_t get_nasm_type_hash (nasm_t *nasm)
{
    struct apu_nasm_type_hash_val
    {
        char dnn_name[MAX_STRING_LEN];
        unsigned int flop_per_ninst;
        unsigned int batch_size;
        unsigned int min_ninst_per_ldata;
        unsigned int transformer_seq_len;
    };
    struct apu_nasm_type_hash_val nasm_type_hash_val = 
        {.flop_per_ninst=nasm->flop_per_ninst, .batch_size=nasm->batch_size,
            .min_ninst_per_ldata=nasm->min_ninst_per_ldata, .transformer_seq_len=nasm->tr_seq_len};
    memset (nasm_type_hash_val.dnn_name, 0, MAX_STRING_LEN);
    strcpy (nasm_type_hash_val.dnn_name, nasm->dnn->name);
    return get_hash (&nasm_type_hash_val, sizeof(struct apu_nasm_type_hash_val));
}

nasm_t *apu_create_nasm_without_finding_ninst_parents (aspen_dnn_t *dnn, unsigned int flop_per_ninst, unsigned int batch_size,  unsigned int min_ninst_per_ldata, unsigned int transformer_seq_len)
{
    if (min_ninst_per_ldata < 1)
    {
        ERROR_PRTF ("ERROR: min_ninst_per_ldata should be at least 1, at line %d in file %s" , 0, " ");
        assert (0);
    }
    nasm_t *new_nasm = (nasm_t *) calloc(1, sizeof(nasm_t));
    new_nasm->dnn = dnn;
    new_nasm->tr_seq_len = transformer_seq_len;
    new_nasm->flop_per_ninst = flop_per_ninst > 0? flop_per_ninst : 1;
    new_nasm->batch_size = batch_size > 0? batch_size : 1;
    new_nasm->nasm_hash = get_unique_hash ();
    new_nasm->type_hash = get_nasm_type_hash (new_nasm);
    new_nasm->min_ninst_per_ldata = min_ninst_per_ldata;
    atomic_store (&new_nasm->completed, 0);
    for (int i = 0; i < dnn->num_layers; i++)
    {
        new_nasm->num_ldata += get_nasm_ldata_num_per_layer(&dnn->layers[i]);
    }
    new_nasm->ldata_arr = calloc(new_nasm->num_ldata, sizeof(nasm_ldata_t));
    nasm_ldata_t *ldata_ptr = new_nasm->ldata_arr;
    for (int i = 0; i < dnn->num_layers; i++)
    {
        nasm_ldata_init(new_nasm, ldata_ptr, &dnn->layers[i]);
        ldata_ptr += get_nasm_ldata_num_per_layer(&dnn->layers[i]);
    }
    unsigned int total_ninst = 0;
    for (int i = 0; i < new_nasm->num_ldata; i++)
    {
        total_ninst += new_nasm->ldata_arr[i].num_ninst;
    }
    new_nasm->num_ninst = total_ninst;
    new_nasm->ninst_arr = calloc(total_ninst, sizeof(ninst_t));
    ninst_t *ninst_ptr = new_nasm->ninst_arr;
    total_ninst = 0;
    for (int i = 0; i < new_nasm->num_ldata; i++)
    {
        new_nasm->ldata_arr[i].ninst_arr_start = ninst_ptr;
        ninst_ptr += new_nasm->ldata_arr[i].num_ninst;
        for (int j = 0; j < new_nasm->ldata_arr[i].num_ninst; j++)
        {
            ninst_init(&new_nasm->ldata_arr[i], &new_nasm->ldata_arr[i].ninst_arr_start[j], total_ninst + j);
        }
        total_ninst += new_nasm->ldata_arr[i].num_ninst;
    }
    for (int i = 0; i < new_nasm->num_ldata; i++)
    {
        update_ldata_child_list(&new_nasm->ldata_arr[i]);
    }

    pthread_mutex_init (&new_nasm->nasm_mutex, NULL);
    pthread_cond_init (&new_nasm->nasm_cond, NULL);

    new_nasm->num_peers = 0;
    new_nasm->peer_map = NULL;
    dnn->ref_nasms++;
    return new_nasm;
}

void set_child_list (ninst_t *ninst)
{
    if (ninst == NULL)
        return;
    if (ninst->num_child_ninsts <= 0)
        return;
    ninst->child_ninst_arr = calloc(ninst->num_child_ninsts, sizeof(ninst_t*));
    nasm_ldata_t *ldata = ninst->ldata;
    nasm_t *nasm = ldata->nasm;
    unsigned int child_idx = 0;
    for (size_t i = ldata - nasm->ldata_arr; i < nasm->num_ldata; i++) // May cause a bug if child ninst is in previous ldata.
    {
        for (int j = 0; j < nasm->ldata_arr[i].num_ninst; j++)
        {
            ninst_t *target_ninst = &nasm->ldata_arr[i].ninst_arr_start[j];
            for (int k = 0; k < target_ninst->num_parent_ninsts; k++)
            {
                if (target_ninst->parent_ninst_idx_arr[k] == ninst->ninst_idx)
                {
                    ninst->child_ninst_arr[child_idx] = target_ninst;
                    child_idx++;
                    if (child_idx == ninst->num_child_ninsts)
                    {
                        return;
                    }
                    break;
                }
            }
        }
    }
    ERROR_PRTF ("Error: set_child_list failed. Only found %d children for ninst %d, expected %d"
        , child_idx, ninst->ninst_idx, ninst->num_child_ninsts);
}

double test_nasm_time_sec (nasm_t *nasm, unsigned int num_iter)
{
    double total_time = 0;
    rpool_t *rpool = rpool_init (-1);
    unsigned int num_cpu = get_cpu_count() / 2;
    if (num_cpu < 1)
        num_cpu = 1;
    dse_group_t *dse_group = dse_group_init (num_cpu);
    dse_group_set_rpool (dse_group, rpool);
    rpool_add_nasm (rpool, nasm, NULL);
    double start = get_elapsed_secs();
    for (int i = 0; i < num_iter; i++)
    {
        dse_group_run_until_nasm_completion (dse_group, nasm);
        rpool_reset_queue (rpool);
        rpool_reset_nasm (rpool, nasm);
    }
    total_time = (double)get_elapsed_secs() - start;
    dse_group_destroy (dse_group);
    rpool_destroy (rpool);
    return total_time / num_iter;
}

nasm_t *apu_generate_nasm(aspen_dnn_t *dnn, unsigned int batch_size, unsigned int num_iter)
{
    size_t num_ninst = APU_GENERATION_NUM_NINST;
    nasm_t *new_nasm = apu_create_nasm(dnn, num_ninst, batch_size);
    double time = test_nasm_time_sec(new_nasm, 50);
    double min_time = time;
    size_t min_num_ninst = num_ninst;
    for (int i = 0; i < num_iter; i++)
    {
        PRTF ("APU: Iteration %d, time = %f sec at ninst num %ld\n", i, time, num_ninst);
        size_t num_old = num_ninst;
        if (time <= min_time*1.05)
        {
            num_ninst = num_ninst * APU_GENERATION_COEFF;
            apu_destroy_nasm (new_nasm);
            new_nasm = apu_create_nasm(dnn, num_ninst, batch_size);
            time = test_nasm_time_sec(new_nasm, 50);
        }
        else
        {
            num_ninst = num_ninst*APU_GENERATION_COEFF + min_num_ninst*(1-APU_GENERATION_COEFF);
            apu_destroy_nasm (new_nasm);
            new_nasm = apu_create_nasm(dnn, num_ninst, batch_size);
            time = test_nasm_time_sec(new_nasm, 50);
        }
        if (time < min_time)
        {
            min_time = time;
            min_num_ninst = num_ninst;
        }
        PRTF ("\tMin time = %f sec, min num ninst = %ld\n", min_time, min_num_ninst);
        if (num_ninst <= num_old*1.10 && num_ninst >= num_old*0.90)
            break;
    }
    apu_destroy_nasm (new_nasm);
    PRTF ("APU: NASM with num %ld, time = %f sec selected.\n", min_num_ninst, min_time);
    new_nasm = apu_create_nasm(dnn, min_num_ninst, batch_size);
    return new_nasm;
}

nasm_t *apu_generate_transformer_nasm(aspen_dnn_t *dnn, unsigned int batch_size, unsigned int seq_num, unsigned int num_iter)
{
    size_t num_ninst = APU_GENERATION_NUM_NINST;
    nasm_t *new_nasm = apu_create_transformer_nasm(dnn, num_ninst, batch_size, seq_num);
    double time = test_nasm_time_sec(new_nasm, 50);
    double min_time = time;
    size_t min_num_ninst = num_ninst;
    for (int i = 0; i < num_iter; i++)
    {
        PRTF ("APU: Iteration %d, time = %f sec at ninst num %ld\n", i, time, num_ninst);
        size_t num_old = num_ninst;
        if (time <= min_time*1.05)
        {
            num_ninst = num_ninst * APU_GENERATION_COEFF;
            apu_destroy_nasm (new_nasm);
            new_nasm = apu_create_transformer_nasm(dnn, num_ninst, batch_size, seq_num);
            time = test_nasm_time_sec(new_nasm, 50);
        }
        else
        {
            num_ninst = num_ninst*APU_GENERATION_COEFF + min_num_ninst*(1-APU_GENERATION_COEFF);
            apu_destroy_nasm (new_nasm);
            new_nasm = apu_create_transformer_nasm(dnn, num_ninst, batch_size, seq_num);
            time = test_nasm_time_sec(new_nasm, 50);
        }
        if (time < min_time)
        {
            min_time = time;
            min_num_ninst = num_ninst;
        }
        PRTF ("\tMin time = %f sec, min num ninst = %ld\n", min_time, min_num_ninst);
        if (num_ninst <= num_old*1.10 && num_ninst >= num_old*0.90)
            break;
    }
    apu_destroy_nasm (new_nasm);
    PRTF ("APU: NASM with num %ld, time = %f sec selected.\n", min_num_ninst, min_time);
    new_nasm = apu_create_transformer_nasm(dnn, num_ninst, batch_size, seq_num);
    return new_nasm;
}


nasm_t *apu_create_nasm(aspen_dnn_t *dnn, unsigned int min_ninst_per_ldata, unsigned int batch_size)
{
    nasm_t *new_nasm = apu_create_nasm_without_finding_ninst_parents(dnn, APU_GENERATION_NUM_FLOPS, batch_size, min_ninst_per_ldata, 0);
    for (int i = 0; i < new_nasm->num_ldata; i++)
    {
        #pragma omp parallel for
        for (int j = 0; j < new_nasm->ldata_arr[i].num_ninst; j++)
        {
            ninst_find_parent (&new_nasm->ldata_arr[i].ninst_arr_start[j]);
        }
        PRTF ("\tAPU: Graphing layer %d/%d (%2.2f%%).\r", i+1, new_nasm->num_ldata, ((float)(i+1)/(float)new_nasm->num_ldata*100));
        fflush (stdout);
    }
    PRTF ("\n");
    for (int i = 0; i < new_nasm->num_ldata; i++)
    {
        #pragma omp parallel for
        for (int j = 0; j < new_nasm->ldata_arr[i].num_ninst; j++)
        {
            set_child_list (&new_nasm->ldata_arr[i].ninst_arr_start[j]);
        }
    }
    // Calculate total flops
    new_nasm->total_flops = 0;
    for (int i = 0; i < new_nasm->num_ldata; i++)
    {
        nasm_ldata_t *ldata = &new_nasm->ldata_arr[i];
        new_nasm->total_flops += 
            ldata->flop_per_output*ldata->out_mat_dims[OUT_H]*ldata->out_mat_dims[OUT_W];
    }
    return new_nasm;
}

nasm_t *apu_create_transformer_nasm
    (aspen_dnn_t *dnn, unsigned int min_ninst_per_ldata,
    unsigned int batch_size, unsigned int seq_num)
{
    nasm_t *new_nasm = apu_create_nasm_without_finding_ninst_parents(dnn, APU_GENERATION_NUM_FLOPS, batch_size, min_ninst_per_ldata, seq_num);
    PRTF ("APU: Graphing ninsts...\n");
    for (int i = 0; i < new_nasm->num_ldata; i++)
    {
        #pragma omp parallel for
        for (int j = 0; j < new_nasm->ldata_arr[i].num_ninst; j++)
        {
            ninst_find_parent (&new_nasm->ldata_arr[i].ninst_arr_start[j]);
        }
        PRTF ("\tAPU: Graphing layer %d/%d (%2.2f%%).\r", i+1, new_nasm->num_ldata, ((float)(i+1)/(float)new_nasm->num_ldata*100));
        fflush (stdout);
    }
    PRTF ("\n");
    for (int i = 0; i < new_nasm->num_ldata; i++)
    {
        #pragma omp parallel for
        for (int j = 0; j < new_nasm->ldata_arr[i].num_ninst; j++)
        {
            set_child_list (&new_nasm->ldata_arr[i].ninst_arr_start[j]);
        }
    }
    PRTF ("\n");
    // Calculate total flops
    new_nasm->total_flops = 0;
    for (int i = 0; i < new_nasm->num_ldata; i++)
    {
        nasm_ldata_t *ldata = &new_nasm->ldata_arr[i];
        new_nasm->total_flops += 
            ldata->flop_per_output*ldata->out_mat_dims[OUT_H]*ldata->out_mat_dims[OUT_W];
    }
    return new_nasm;
}

void apu_reset_nasm (nasm_t *nasm)
{
    atomic_store (&nasm->num_ldata_completed, 0);
    atomic_store (&nasm->completed, 0);
    for (int i = 0; i < nasm->num_ldata; i++)
    {
        nasm_ldata_t *ldata = &nasm->ldata_arr[i];
        atomic_store (&ldata->num_ninst_completed, 0);
        atomic_store (&ldata->num_child_ldata_completed, 0);
        if (i != 0)
            free_ldata_out_mat (ldata);
        for (int j = 0; j < ldata->num_ninst; j++)
        {
            ninst_t *ninst = &ldata->ninst_arr_start[j];
            atomic_store (&ninst->state, NINST_NOT_READY);
            atomic_store (&ninst->num_parent_ninsts_completed, 0);
        }
    }
}

void set_nasm_to_finished (nasm_t *nasm)
{
    atomic_store (&nasm->num_ldata_completed, 1);
    for (int i = 0; i < nasm->num_ldata; i++)
    {
        nasm_ldata_t *ldata = &nasm->ldata_arr[i];
        atomic_store (&ldata->num_ninst_completed, ldata->num_ninst);
        atomic_store (&ldata->num_child_ldata_completed, ldata->num_child_ldata);
        for (int j = 0; j < ldata->num_ninst; j++)
        {
            ninst_t *ninst = &ldata->ninst_arr_start[j];
            ninst->state = NINST_COMPLETED;
            atomic_store (&ninst->num_parent_ninsts_completed, ninst->num_parent_ninsts);
        }
    }
}
aspen_peer_t *peer_init ()
{
    aspen_peer_t *peer = calloc (1, sizeof(aspen_peer_t));
    pthread_mutex_init (&peer->peer_mutex, NULL);
    peer->peer_hash = get_unique_hash ();
    peer->ip = calloc (MAX_STRING_LEN, sizeof(char));
    peer->listen_port = 0;
    peer->sock = 0;
    peer->isUDP = -1;
    peer->latency_usec = -1;
    peer->bandwidth_bps = -1;
    peer->tx_queue = calloc (1, sizeof(rpool_t));
    rpool_init_queue (peer->tx_queue);
    return peer;

}
aspen_peer_t *peer_copy (aspen_peer_t *new_peer, aspen_peer_t *peer)
{
    if (new_peer == NULL || peer == NULL)
    {
        ERROR_PRTF ("ERROR: new_peer or peer is NULL.");
        assert(0);
    }
    new_peer->peer_hash = peer->peer_hash;
    strcpy (new_peer->ip, peer->ip);
    new_peer->listen_port = peer->listen_port;
    new_peer->sock = peer->sock;
    new_peer->isUDP = peer->isUDP;
    new_peer->latency_usec = peer->latency_usec;
    new_peer->bandwidth_bps = peer->bandwidth_bps;
    return new_peer;
}
void destroy_peer (aspen_peer_t *peer)
{
    if (peer == NULL)
        return;
    pthread_mutex_destroy (&peer->peer_mutex);
    rpool_destroy_queue (peer->tx_queue);
    if (peer->ip != NULL)
        free (peer->ip);
    free (peer);
}
void set_ninst_compute_peer_idx (ninst_t *ninst, int peer_idx)
{
    if (ninst == NULL)
    {
        ERROR_PRTF ("ERROR: ninst is NULL.");
        assert(0);
    }
    if (peer_idx < 0 || peer_idx >= ninst->ldata->nasm->num_peers)
    {
        ERROR_PRTF ("ERROR: peer_idx %d is out of range [0, %d].", peer_idx, ninst->ldata->nasm->num_peers);
        assert(0);
    }
    ninst->peer_flag[peer_idx] |= PEER_FLAG_COMPUTE;
}
void set_ninst_send_peer_idx (ninst_t *ninst, int peer_idx)
{
    if (ninst == NULL)
    {
        ERROR_PRTF ("ERROR: ninst is NULL.");
        assert(0);
    }
    if (peer_idx < 0 || peer_idx >= ninst->ldata->nasm->num_peers)
    {
        ERROR_PRTF ("ERROR: peer_idx %d is out of range [0, %d].", peer_idx, ninst->ldata->nasm->num_peers);
        assert(0);
    }
    ninst->peer_flag[peer_idx] |= PEER_FLAG_SEND;
}
int check_ninst_compute_using_peer_idx (ninst_t *ninst, int peer_idx)
{
    #ifdef DEBUG
    if (ninst == NULL)
    {
        ERROR_PRTF ("ERROR: ninst is NULL.");
        assert(0);
    }
    if (peer_idx < 0 || peer_idx >= ninst->ldata->nasm->num_peers)
    {
        ERROR_PRTF ("ERROR: peer_idx %d is out of range [0, %d].", peer_idx, ninst->ldata->nasm->num_peers);
        assert(0);
    }
    #endif
    return ninst->peer_flag[peer_idx] & PEER_FLAG_COMPUTE;

}
int check_ninst_send_using_peer_idx (ninst_t *ninst, int peer_idx)
{
    #ifdef DEBUG
    if (ninst == NULL)
    {
        ERROR_PRTF ("ERROR: ninst is NULL.");
        assert(0);
    }
    if (peer_idx < 0 || peer_idx >= ninst->ldata->nasm->num_peers)
    {
        ERROR_PRTF ("ERROR: peer_idx %d is out of range [0, %d].", peer_idx, ninst->ldata->nasm->num_peers);
        assert(0);
    }
    #endif
    return ninst->peer_flag[peer_idx] & PEER_FLAG_SEND;
}
int get_peer_idx (nasm_t *nasm, HASH_t peer_hash)
{
    #ifdef DEBUG
    if (nasm == NULL)
    {
        ERROR_PRTF ("ERROR: nasm is NULL.");
        assert(0);
    }
    #endif
    for (int i = 0; i < nasm->num_peers; i++)
    {
        if (nasm->peer_map[i].peer_hash == peer_hash)
            return i;
    }
    return -1;
}
aspen_peer_t *get_peer (nasm_t *nasm, int peer_idx)
{
    #ifdef DEBUG
    if (nasm == NULL)
    {
        ERROR_PRTF ("ERROR: nasm is NULL.");
        assert(0);
    }
    if (peer_idx < 0 || peer_idx >= nasm->num_peers)
    {
        ERROR_PRTF ("ERROR: peer_idx %d is out of range [0, %d].", peer_idx, nasm->num_peers);
        assert(0);
    }
    #endif
    return nasm->peer_map + peer_idx;
}


nasm_t *apu_copy_nasm (nasm_t *nasm)
{
    nasm_t *new_nasm = apu_create_nasm_without_finding_ninst_parents 
        (nasm->dnn, nasm->flop_per_ninst, nasm->batch_size, nasm->min_ninst_per_ldata, nasm->tr_seq_len);
    if (new_nasm->type_hash != nasm->type_hash)
    {
        ERROR_PRTF ("ERROR: new_nasm type hash %016lx is not equal to nasm type hash %016lx.", new_nasm->type_hash, nasm->type_hash);
        assert(0);
    }
    new_nasm->nasm_hash = nasm->nasm_hash;
    new_nasm->total_flops = nasm->total_flops;
    new_nasm->num_peers = nasm->num_peers;
    new_nasm->peer_map = calloc (new_nasm->num_peers, sizeof(HASH_t));
    for (int i = 0; i < new_nasm->num_peers; i++)
        peer_copy (new_nasm->peer_map + i, nasm->peer_map + i);
    for (int i = 0; i < nasm->num_ninst; i++)
    {
        copy_ninst (new_nasm->ninst_arr + i, nasm->ninst_arr + i);
    }
    return new_nasm;
}

void copy_ninst (ninst_t *new_ninst, ninst_t *ninst)
{
    if (new_ninst == NULL || ninst == NULL)
    {
        ERROR_PRTF ("ERROR: new_ninst or ninst is NULL.");
        assert(0);
    }
    if (ninst->type_hash != new_ninst->type_hash)
    {
        ERROR_PRTF ("ERROR: ninst type hash %016lx is not equal to new_ninst type hash %016lx.", ninst->type_hash, new_ninst->type_hash);
        assert(0);
    }
    new_ninst->num_parent_ninsts = ninst->num_parent_ninsts;
    new_ninst->num_child_ninsts = ninst->num_child_ninsts;
    new_ninst->num_input_pos = ninst->num_input_pos;
    new_ninst->parent_ninst_idx_arr = calloc (new_ninst->num_parent_ninsts, sizeof(unsigned int));
    new_ninst->child_ninst_arr = calloc (new_ninst->num_child_ninsts, sizeof(ninst_t*));
    memcpy (new_ninst->parent_ninst_idx_arr, ninst->parent_ninst_idx_arr, new_ninst->num_parent_ninsts*sizeof(unsigned int));
    for (int i = 0; i < new_ninst->num_child_ninsts; i++)
    {
        new_ninst->child_ninst_arr[i] = (ninst->child_ninst_arr[i] - ninst->ldata->nasm->ninst_arr) 
            + new_ninst->ldata->nasm->ninst_arr;
    }
    new_ninst->peer_flag = calloc (ninst->ldata->nasm->num_peers, sizeof(char));
    memcpy (new_ninst->peer_flag, ninst->peer_flag, sizeof(char)*ninst->ldata->nasm->num_peers);
}

void copy_ldata_out_mat_to_buffer (nasm_ldata_t *ldata, void *buffer)
{
    #ifdef DEBUG
    if (ldata == NULL || buffer == NULL)
    {
        ERROR_PRTF ("ERROR: ldata or buffer is NULL.");
        assert(0);
    }
    #endif
    char* out_mat = ldata->out_mat;
    const unsigned int W = ldata->out_mat_dims[OUT_W];
    const unsigned int H = ldata->out_mat_dims[OUT_H];
    const unsigned int stride = ldata->out_mat_stride;
    for(int w = 0; w < W; w++) 
    {
        memcpy(buffer + w * H * sizeof(float), out_mat + w * stride * sizeof(float), H * sizeof(float));
    }
}

void copy_buffer_to_ldata_out_mat (nasm_ldata_t *ldata, void *buffer)
{
    #ifdef DEBUG
    if (ldata == NULL || buffer == NULL)
    {
        ERROR_PRTF ("ERROR: ldata or buffer is NULL.");
        assert(0);
    }
    #endif
    char* out_mat = ldata->out_mat;
    const unsigned int W = ldata->out_mat_dims[OUT_W];
    const unsigned int H = ldata->out_mat_dims[OUT_H];
    const unsigned int stride = ldata->out_mat_stride;
    for(int w = 0; w < W; w++) 
    {
        memcpy(out_mat + w * stride * sizeof(float), buffer + w * H * sizeof(float), H * sizeof(float));
    }
}

void copy_ninst_data_to_buffer (ninst_t *ninst, void *buffer)
{
    char* out_mat = get_ninst_out_mem_without_alloc  (ninst);
    #ifdef DEBUG
    if (ninst == NULL || buffer == NULL)
    {
        ERROR_PRTF ("ERROR: ninst or buffer is NULL.");
        assert(0);
    }
    if (out_mat == NULL)
    {
        ERROR_PRTF ("ERROR: ninst out_mat is NULL.");
        assert(0);
    }
    #endif
    const unsigned int W = ninst->tile_dims[OUT_W];
    const unsigned int H = ninst->tile_dims[OUT_H];
    const unsigned int stride = ninst->ldata->out_mat_stride;
    for(int w = 0; w < W; w++) 
    {
        memcpy(buffer + w * H * sizeof(float), out_mat + w * stride * sizeof(float), H * sizeof(float));
    }
}

void copy_buffer_to_ninst_data (ninst_t *ninst, void *buffer)
{
    #ifdef DEBUG
    if (ninst == NULL || buffer == NULL)
    {
        ERROR_PRTF ("ERROR: ninst or buffer is NULL.");
        assert(0);
    }
    #endif
    char* out_mat = get_ninst_out_mem (ninst);
    const unsigned int W = ninst->tile_dims[OUT_W];
    const unsigned int H = ninst->tile_dims[OUT_H];
    const unsigned int stride = ninst->ldata->out_mat_stride;
    for(int w = 0; w < W; w++) 
    {
        memcpy(out_mat + w * stride * sizeof(float), buffer + w * H * sizeof(float), H * sizeof(float));
    }
}

void alloc_ldata_out_mat (nasm_ldata_t *ldata)
{
    pthread_mutex_lock (&ldata->out_mat_mutex);
    if (ldata->out_mat != NULL)
    {
        pthread_mutex_unlock (&ldata->out_mat_mutex);
        return;
    }
    ldata->out_mat = aspen_dynamic_malloc (1, ldata->out_mat_mem_size);
    pthread_mutex_unlock (&ldata->out_mat_mutex);
}

void free_ldata_out_mat (nasm_ldata_t *ldata)
{
    if (ldata->out_mat != NULL)
    {
        aspen_dynamic_free (ldata->out_mat, 1, ldata->out_mat_mem_size);
        ldata->out_mat = NULL;
    }
}

void *get_ninst_out_mem (ninst_t *ninst)
{
    if (ninst->ldata->out_mat == NULL)
    {
        alloc_ldata_out_mat (ninst->ldata);
    }
    return (char*)ninst->ldata->out_mat
        + (ninst->out_mat_pos[OUT_W]*ninst->ldata->out_mat_stride + ninst->out_mat_pos[OUT_H])
            *ninst->ldata->nasm->dnn->element_size;
}

void *get_ninst_out_mem_without_alloc (ninst_t *ninst)
{
    if (ninst->ldata->out_mat == NULL)
        return NULL;
    return (char*)ninst->ldata->out_mat
        + (ninst->out_mat_pos[OUT_W]*ninst->ldata->out_mat_stride + ninst->out_mat_pos[OUT_H])
            *ninst->ldata->nasm->dnn->element_size;
}

void destroy_nasm_ldata (nasm_ldata_t *ldata)
{
    if (ldata == NULL)
        return;
    if (ldata->child_ldata_idx_arr != NULL)
        free(ldata->child_ldata_idx_arr);
    free_ldata_out_mat (ldata);
    for (int i = 0; i < ldata->num_ninst; i++)
    {
        destroy_ninst(&ldata->ninst_arr_start[i]);
    }
}

void destroy_nasm_ldata_arr (nasm_ldata_t *ldata_arr, int num_ldata)
{
    if (ldata_arr == NULL)
        return;
    for (int i = 0; i < num_ldata; i++)
    {
        destroy_nasm_ldata(&ldata_arr[i]);
    }
    free(ldata_arr);
}

void apu_destroy_nasm (nasm_t *nasm)
{
    if (nasm == NULL)
        return;
    destroy_nasm_ldata_arr(nasm->ldata_arr, nasm->num_ldata);
    if (nasm->ninst_arr != NULL)
        free(nasm->ninst_arr);
    if (nasm->data != NULL)
        aspen_free (nasm->data);
    if (nasm->peer_map != NULL)
    {
        for (int i = 0; i < nasm->num_peers; i++)
            destroy_peer (nasm->peer_map + i);
        free (nasm->peer_map);
    }
    pthread_mutex_destroy (&nasm->nasm_mutex);
    pthread_cond_destroy (&nasm->nasm_cond);
    nasm->dnn->ref_nasms--;
    free(nasm);
}

// Change to add a new layer type
void get_out_mat_info (nasm_ldata_t *ldata)
{
    aspen_layer_t *layer = ldata->layer;
    if (layer->type == CONV_LAYER)
    {
        ldata->flop_per_output = 2*layer->params[WEIGHT_H]*layer->params[WEIGHT_W]*layer->params[IN_C];
        ldata->out_mat_dims[OUT_H] = layer->params[OUT_C];
        ldata->out_mat_dims[OUT_W] = layer->params[OUT_H]*layer->params[OUT_W]*ldata->nasm->batch_size;
    }
    else if (layer->type == FC_LAYER)
    {
        ldata->flop_per_output = 2*layer->params[IN_C]*layer->params[IN_H]*layer->params[IN_W] ;
        ldata->out_mat_dims[OUT_H] = layer->params[OUT_C];
        ldata->out_mat_dims[OUT_W] = ldata->nasm->batch_size;
    }
    else if (layer->type == MAXPOOL_LAYER)
    {
        ldata->flop_per_output = layer->params[WEIGHT_H]*layer->params[WEIGHT_W];
        ldata->out_mat_dims[OUT_H] = layer->params[OUT_C];
        ldata->out_mat_dims[OUT_W] = layer->params[OUT_H]*layer->params[OUT_W]*ldata->nasm->batch_size;
    }
    else if (layer->type == AVGPOOL_LAYER)
    {
        ldata->flop_per_output = layer->params[WEIGHT_H]*layer->params[WEIGHT_W];
        ldata->out_mat_dims[OUT_H] = layer->params[OUT_C];
        ldata->out_mat_dims[OUT_W] = layer->params[OUT_H]*layer->params[OUT_W]*ldata->nasm->batch_size;
    }
    else if (layer->type == INPUT_LAYER || layer->type == RESIDUAL_LAYER || layer->type == YOLO_LAYER || layer->type == APPEND_LAYER)
    {
        ldata->flop_per_output = 1;
        ldata->out_mat_dims[OUT_H] = layer->params[OUT_C];
        ldata->out_mat_dims[OUT_W] = layer->params[OUT_H]*layer->params[OUT_W]*ldata->nasm->batch_size;
        if (ldata->nasm->tr_seq_len != 0)
        {
            ldata->out_mat_dims[OUT_H] = layer->params[MAT_M];
            ldata->out_mat_dims[OUT_W] = ldata->nasm->tr_seq_len*ldata->nasm->batch_size;
        }
    }
    else if (layer->type == SOFTMAX_LAYER)
    {
        ldata->flop_per_output = 1;
        ldata->out_mat_dims[OUT_H] = layer->params[OUT_C];
        ldata->out_mat_dims[OUT_W] = ldata->nasm->batch_size;
    }
    else if (layer->type == MATMUL_LAYER)
    {
        ldata->flop_per_output = 2*layer->params[MAT_K];
        ldata->out_mat_dims[OUT_H] = layer->params[MAT_M];
        ldata->out_mat_dims[OUT_W] = ldata->nasm->tr_seq_len*ldata->nasm->batch_size;
    }
    else if (layer->type == K_ATTENTION_LAYER)
    {
        ldata->flop_per_output = 2*layer->params[NUM_HIDDEN]/layer->params[NUM_HEAD];
        ldata->out_mat_dims[OUT_H] = ldata->nasm->tr_seq_len;
        ldata->out_mat_dims[OUT_W] = ldata->nasm->tr_seq_len*layer->params[NUM_HEAD]*ldata->nasm->batch_size;
    }
    else if (layer->type == V_ATTENTION_LAYER)
    {
        ldata->flop_per_output = 2*ldata->nasm->tr_seq_len;
        ldata->out_mat_dims[OUT_H] = layer->params[MAT_M];
        ldata->out_mat_dims[OUT_W] = ldata->nasm->tr_seq_len*ldata->nasm->batch_size;
    }
    else if (layer->type == LAYERNORM_LAYER)
    {
        ldata->flop_per_output = 1;
        ldata->out_mat_dims[OUT_H] = layer->params[MAT_M];
        ldata->out_mat_dims[OUT_W] = ldata->nasm->tr_seq_len*ldata->nasm->batch_size;
    }
    else
    {
        ERROR_PRTF ("ERROR) Unsupported layer type %s, at line %d in file %s" , layer_type_str[layer->type], 0, " ");
        assert (0);
    }
}

void get_ninst_tile_dims (nasm_ldata_t *ldata)
{
    // ldata->ninst_tile_dims[OUT_H] = NINST_H_MIN < ldata->out_mat_dims[OUT_H] ? NINST_H_MIN : ldata->out_mat_dims[OUT_H];
    // ldata->ninst_tile_dims[OUT_W] = NINST_W_MIN < ldata->out_mat_dims[OUT_W] ? NINST_W_MIN : ldata->out_mat_dims[OUT_W];
    // if (ldata->ninst_tile_dims[OUT_H] <= 0)
    //     ldata->ninst_tile_dims[OUT_H] = 1;
    // if (ldata->ninst_tile_dims[OUT_W] <= 0)
    //     ldata->ninst_tile_dims[OUT_W] = 1;
    ldata->ninst_tile_dims[OUT_H] = ldata->out_mat_dims[OUT_H];
    ldata->ninst_tile_dims[OUT_W] = ldata->out_mat_dims[OUT_W];
    
    while (ldata->ninst_tile_dims[OUT_H]*ldata->ninst_tile_dims[OUT_W] < ldata->nasm->flop_per_ninst/ldata->flop_per_output)
    {
        if (ldata->ninst_tile_dims[OUT_H] < ldata->out_mat_dims[OUT_H])
        {
            ldata->ninst_tile_dims[OUT_H]++;
        }
        else if (ldata->ninst_tile_dims[OUT_W] < ldata->out_mat_dims[OUT_W])
        {
            ldata->ninst_tile_dims[OUT_W]++;
        }
        else
        {
            break;
        }
    }
    while (ldata->ninst_tile_dims[OUT_H]%NINST_H_MIN != 0)
    {
        ldata->ninst_tile_dims[OUT_H]++;
    }
    if (ldata->layer->type != FC_LAYER && ldata->layer->type != SOFTMAX_LAYER)
    {
        while (ldata->ninst_tile_dims[OUT_W]%NINST_W_MIN != 0)
        {
            ldata->ninst_tile_dims[OUT_W]++;
        }
    }
}

void nasm_ldata_init (nasm_t *nasm, nasm_ldata_t *ldata_ptr, aspen_layer_t *layer)
{
    ldata_ptr->nasm = nasm;
    ldata_ptr->layer = layer;
    ldata_ptr->out_mat_mutex = (pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER;
    for (LAYER_PARENTS i = 0; i < NUM_PARENT_ELEMENTS; i++)
    {
        ldata_ptr->parent_ldata_idx_arr[i] = -1;
        if (layer->parent_layers[i] != NULL)
        {
            for (int lidx = 0; lidx < nasm->dnn->num_layers; lidx++)
            {
                nasm_ldata_t *parent_ldata = nasm->ldata_arr + lidx;
                if (parent_ldata->layer == layer->parent_layers[i])
                {
                    ldata_ptr->parent_ldata_idx_arr [i] = lidx;
                    parent_ldata->num_child_ldata++;
                }
            }
        }
    }
    ldata_ptr->flop_per_output = 1;
    get_out_mat_info (ldata_ptr);
    get_ninst_tile_dims (ldata_ptr);
    unsigned int out_w = 0;
    unsigned int out_h = 0;
    unsigned int unit_h = NINST_H_MIN;
    unsigned int unit_w = NINST_W_MIN;
    unsigned int hidden_per_head = 0;
    if (layer->type == SOFTMAX_LAYER || layer->type == APPEND_LAYER || layer->type == YOLO_LAYER)
    {
        ldata_ptr->ninst_tile_dims[OUT_H] = ldata_ptr->out_mat_dims[OUT_H];
        while (ldata_ptr->ninst_tile_dims[OUT_H]%NINST_H_MIN != 0)
        {
            ldata_ptr->ninst_tile_dims[OUT_H]++;
        }
    }
    if (layer->type == SOFTMAX_LAYER)
    {
        ldata_ptr->ninst_tile_dims[OUT_W] = 1;
    }
    if (layer->type == FC_LAYER)
    {
        if (ldata_ptr->out_mat_dims[OUT_W] < unit_w)
            unit_w = ldata_ptr->out_mat_dims[OUT_W];
    }
    if (layer->params[NUM_HEAD] > 0)
    {
        hidden_per_head = layer->params[NUM_HIDDEN] / layer->params[NUM_HEAD];
        unit_w = 8;
        unit_h = layer->params[NUM_HIDDEN] / layer->params[NUM_HEAD] / 4;
    }
    out_w = get_smallest_dividable (ldata_ptr->out_mat_dims[OUT_W], ldata_ptr->ninst_tile_dims[OUT_W]);
    out_h = get_smallest_dividable (ldata_ptr->out_mat_dims[OUT_H], ldata_ptr->ninst_tile_dims[OUT_H]);
    if (layer->type != SOFTMAX_LAYER)
    {
        while ((out_w/ldata_ptr->ninst_tile_dims[OUT_W])*(out_h/ldata_ptr->ninst_tile_dims[OUT_H]) < nasm->min_ninst_per_ldata)
        {
            if (ldata_ptr->ninst_tile_dims[OUT_W] > unit_w)
            {
                ldata_ptr->ninst_tile_dims[OUT_W] /= 2;
                while (ldata_ptr->ninst_tile_dims[OUT_W]%unit_w != 0)
                {
                    ldata_ptr->ninst_tile_dims[OUT_W]++;
                }
            }
            else if (ldata_ptr->ninst_tile_dims[OUT_H] > unit_h)
            {
                if (layer->type != APPEND_LAYER && layer->type != YOLO_LAYER)
                {
                    ldata_ptr->ninst_tile_dims[OUT_H] /= 2;
                    while (ldata_ptr->ninst_tile_dims[OUT_H]%unit_h != 0)
                    {
                        ldata_ptr->ninst_tile_dims[OUT_H]++;
                    }
                }
            }
            out_w = get_smallest_dividable (ldata_ptr->out_mat_dims[OUT_W], ldata_ptr->ninst_tile_dims[OUT_W]);
            out_h = get_smallest_dividable (ldata_ptr->out_mat_dims[OUT_H], ldata_ptr->ninst_tile_dims[OUT_H]);
            if (ldata_ptr->ninst_tile_dims[OUT_W] == unit_w 
                && (ldata_ptr->ninst_tile_dims[OUT_H] == unit_h || layer->type == APPEND_LAYER || layer->type == YOLO_LAYER))
            {
                break;
            }
        }
    }
    if (layer->params[NUM_HEAD] > 0 || layer->type == LAYERNORM_LAYER)
    {
        unsigned int old_h = ldata_ptr->ninst_tile_dims[OUT_H];
        if (ldata_ptr->ninst_tile_dims[OUT_H] > hidden_per_head)
            ldata_ptr->ninst_tile_dims[OUT_H] = hidden_per_head;
        if (layer->type == LAYERNORM_LAYER)
            ldata_ptr->ninst_tile_dims[OUT_H] = layer->params[MAT_M];
        else if (layer->type == K_ATTENTION_LAYER)
            ldata_ptr->ninst_tile_dims[OUT_H] = get_smallest_dividable (nasm->tr_seq_len, _VEC_SIZE_M);
        else if (layer->type == V_ATTENTION_LAYER)
        {
            if (hidden_per_head < ldata_ptr->ninst_tile_dims[OUT_H])
                ldata_ptr->ninst_tile_dims[OUT_H] = hidden_per_head;
            while (hidden_per_head % ldata_ptr->ninst_tile_dims[OUT_H] != 0)
            {
                ldata_ptr->ninst_tile_dims[OUT_H]++;
            }
        }
        ldata_ptr->ninst_tile_dims[OUT_W] = (float)ldata_ptr->ninst_tile_dims[OUT_W] * old_h / ldata_ptr->ninst_tile_dims[OUT_H];
        if (ldata_ptr->ninst_tile_dims[OUT_W] > nasm->tr_seq_len)
            ldata_ptr->ninst_tile_dims[OUT_W] = nasm->tr_seq_len;
        if (ldata_ptr->ninst_tile_dims[OUT_W] <= 0)
            ldata_ptr->ninst_tile_dims[OUT_W] = 1;
        while (nasm->tr_seq_len % ldata_ptr->ninst_tile_dims[OUT_W] != 0 && layer->type != LAYERNORM_LAYER)
        {
            ldata_ptr->ninst_tile_dims[OUT_W]++;
        }
        out_w = get_smallest_dividable (ldata_ptr->out_mat_dims[OUT_W], ldata_ptr->ninst_tile_dims[OUT_W]);
        out_h = get_smallest_dividable (ldata_ptr->out_mat_dims[OUT_H], ldata_ptr->ninst_tile_dims[OUT_H]);
    }
    ldata_ptr->out_mat_stride = out_h;
    ldata_ptr->out_mat_mem_size = get_smallest_dividable 
        (ldata_ptr->out_mat_stride*out_w*ldata_ptr->layer->dnn->element_size, MEM_ALIGN);
    ldata_ptr->num_ninst = (out_h/ldata_ptr->ninst_tile_dims[OUT_H])*(out_w/ldata_ptr->ninst_tile_dims[OUT_W]);
}

unsigned int get_tensor_idx_from_pos (aspen_tensor_t *tensor, unsigned int *pos)
{
    unsigned int idx = 0;
    for (int i = 0; i < tensor->num_dims; i++)
    {
        LAYER_PARAMS dim = tensor->data_dim_order[i];
        idx = idx*tensor->dims[dim] + pos[dim];
    }
    return idx;
}
void get_tensor_pos_from_idx (aspen_tensor_t *tensor, unsigned int idx, unsigned int *pos)
{
    for (int i = tensor->num_dims - 1; i >= 0; i--)
    {
        LAYER_PARAMS dim = tensor->data_dim_order[i];
        pos[dim] = idx%tensor->dims[dim];
        idx /= tensor->dims[dim];
    }
}
ninst_t *get_ninst_from_tensor_pos (nasm_ldata_t *ldata, unsigned int *tensor_pos)
{
    unsigned int out_mat_pos[2] = {0,0};
    get_out_mat_pos_from_tensor_pos (ldata, tensor_pos, out_mat_pos);
    return get_ninst_from_out_mat_pos (ldata, out_mat_pos[OUT_H], out_mat_pos[OUT_W]);
}
ninst_t *get_ninst_from_out_mat_pos (nasm_ldata_t *ldata, unsigned int h, unsigned int w)
{
    unsigned int out_h = get_smallest_dividable (ldata->out_mat_dims[OUT_H], ldata->ninst_tile_dims[OUT_H]);
    unsigned int ninst_idx = (w/ldata->ninst_tile_dims[OUT_W])*(out_h/ldata->ninst_tile_dims[OUT_H]) 
        + (h/ldata->ninst_tile_dims[OUT_H]);
    assert (ninst_idx >= 0);
    assert (ninst_idx < ldata->num_ninst);
    return ldata->ninst_arr_start + ninst_idx;
}
void get_out_mat_pos_from_nist (nasm_ldata_t *ldata, ninst_t *ninst, unsigned int *out_mat_pos)
{
    unsigned int out_h = get_smallest_dividable (ldata->out_mat_dims[OUT_H], ldata->ninst_tile_dims[OUT_H]);
    unsigned int ninst_idx = ninst - ldata->ninst_arr_start;
    out_mat_pos[OUT_H] = (ninst_idx%(out_h/ldata->ninst_tile_dims[OUT_H]))*ldata->ninst_tile_dims[OUT_H];
    out_mat_pos[OUT_W] = (ninst_idx/(out_h/ldata->ninst_tile_dims[OUT_H]))*ldata->ninst_tile_dims[OUT_W];
}
// Change to add a new layer type
void get_out_mat_pos_from_tensor_pos (nasm_ldata_t *ldata, unsigned int *tensor_pos, unsigned int *out_mat_pos)
{
    aspen_layer_t *layer = ldata->layer;
    if (layer->type == CONV_LAYER || layer->type == MAXPOOL_LAYER || layer->type == AVGPOOL_LAYER || layer->type == INPUT_LAYER
        || layer->type == RESIDUAL_LAYER || layer->type == YOLO_LAYER || layer->type == APPEND_LAYER)
    {
        if (layer->params[MAT_M] == 0)
        {
            out_mat_pos[OUT_H] = tensor_pos[OUT_C];
            out_mat_pos[OUT_W] = tensor_pos[BATCH] * layer->params[OUT_H] * layer->params[OUT_W] + 
                tensor_pos[OUT_H] * layer->params[OUT_W] + tensor_pos[OUT_W];
        }
        else
        {
            out_mat_pos[OUT_H] = tensor_pos[MAT_M];
            out_mat_pos[OUT_W] = tensor_pos[BATCH] * ldata->nasm->tr_seq_len + tensor_pos[MAT_N];
        }
        return;
    }
    else if (layer->type == FC_LAYER || layer->type == SOFTMAX_LAYER)
    {
        out_mat_pos[OUT_H] = tensor_pos[OUT_C];
        out_mat_pos[OUT_W] = tensor_pos[BATCH];
        return;
    }
    else if (layer->type == MATMUL_LAYER || layer->type == LAYERNORM_LAYER || layer->type == V_ATTENTION_LAYER)
    {
        out_mat_pos[OUT_H] = tensor_pos[MAT_M];
        out_mat_pos[OUT_W] = tensor_pos[BATCH] * ldata->nasm->tr_seq_len + tensor_pos[MAT_N];
        return;
    }
    else if (layer->type == K_ATTENTION_LAYER)
    {
        out_mat_pos[OUT_H] = tensor_pos[MAT_M];
        out_mat_pos[OUT_W] = tensor_pos[BATCH] * ldata->nasm->tr_seq_len * layer->params[NUM_HEAD] +
            tensor_pos[NUM_HEAD] * ldata->nasm->tr_seq_len + tensor_pos[MAT_N];
    }
    else
    {
        ERROR_PRTF ("ERROR: Unsupported layer type %s, at line %d in file %s" , layer_type_str[layer->type], 0, " ");
        assert (0);
    }
}
// Change to add a new layer type
void get_tensor_pos_from_out_mat_pos (nasm_ldata_t *ldata, unsigned int *out_mat_pos, unsigned int *tensor_pos)
{
    aspen_layer_t *layer = ldata->layer;
    if (layer->type == CONV_LAYER || layer->type == MAXPOOL_LAYER || layer->type == AVGPOOL_LAYER || layer->type == INPUT_LAYER
        || layer->type == RESIDUAL_LAYER || layer->type == YOLO_LAYER || layer->type == APPEND_LAYER)
    {
        if (layer->params[MAT_M] == 0)
        {
            tensor_pos[BATCH] = out_mat_pos[OUT_W] / (layer->params[OUT_H] * layer->params[OUT_W]); 
            tensor_pos[OUT_C] = out_mat_pos[OUT_H];
            tensor_pos[OUT_H] = (out_mat_pos[OUT_W] % (layer->params[OUT_H] * layer->params[OUT_W])) / layer->params[OUT_W];
            tensor_pos[OUT_W] = out_mat_pos[OUT_W] % layer->params[OUT_W];
        }
        else 
        {
            tensor_pos[BATCH] = out_mat_pos[OUT_W] / ldata->nasm->tr_seq_len;
            tensor_pos[MAT_N] = out_mat_pos[OUT_W] % ldata->nasm->tr_seq_len;
            tensor_pos[MAT_M] = out_mat_pos[OUT_H];
            tensor_pos[OUT_C] = tensor_pos[MAT_M];
            tensor_pos[OUT_H] = 1;
            tensor_pos[OUT_W] = tensor_pos[MAT_N];
        }
        return;
    }
    else if (layer->type == FC_LAYER || layer->type == SOFTMAX_LAYER)
    {
        tensor_pos[BATCH] = out_mat_pos[OUT_W];
        tensor_pos[OUT_C] = out_mat_pos[OUT_H];
        tensor_pos[OUT_H] = 1;
        tensor_pos[OUT_W] = 1;
        return;
    }
    else if (layer->type == MATMUL_LAYER || layer->type == LAYERNORM_LAYER || layer->type == V_ATTENTION_LAYER)
    {
        tensor_pos[BATCH] = out_mat_pos[OUT_W] / ldata->nasm->tr_seq_len;
        tensor_pos[MAT_N] = out_mat_pos[OUT_W] % ldata->nasm->tr_seq_len;
        tensor_pos[MAT_M] = out_mat_pos[OUT_H];
    }
    else if (layer->type == K_ATTENTION_LAYER)
    {
        tensor_pos[BATCH] = out_mat_pos[OUT_W] / (layer->params[NUM_HEAD] * ldata->nasm->tr_seq_len);
        tensor_pos[NUM_HEAD] = (out_mat_pos[OUT_W] % (layer->params[NUM_HEAD] * ldata->nasm->tr_seq_len)) 
            / ldata->nasm->tr_seq_len;
        tensor_pos[MAT_N] = out_mat_pos[OUT_W] % ldata->nasm->tr_seq_len;
        tensor_pos[MAT_M] = out_mat_pos[OUT_H];
    }
    else
    {
        ERROR_PRTF ("ERROR: Unsupported layer type %s, at line %d in file %s" , layer_type_str[layer->type], 0, " ");
        assert (0);
    }
}
void get_tensor_pos_from_nist (nasm_ldata_t *ldata, ninst_t *ninst, unsigned int *tensor_pos)
{
    unsigned int out_mat_pos[2] = {0,0};
    get_out_mat_pos_from_nist (ldata, ninst, out_mat_pos);
    get_tensor_pos_from_out_mat_pos (ldata, out_mat_pos, tensor_pos);
}

void *get_packed_ldata_output_colwise (nasm_ldata_t *ldata)
{
    if (ldata == NULL)
    {
        ERROR_PRTF ("Error in get_packed_ldata_output_colwise: ldata is NULL.");
        return NULL;
    }
    void *packed_data = NULL;
    size_t elem_size = ldata->layer->dnn->element_size;
    size_t data_size = ldata->out_mat_dims[OUT_H] * ldata->out_mat_dims[OUT_W] * elem_size;
    packed_data = calloc (1, data_size);
    if (packed_data == NULL)
    {
        ERRNO_PRTF ("Error in get_packed_ldata_output_colwise: calloc failed.");
        return NULL;
    }
    void *out_mat = ldata->out_mat;
    for (unsigned int w = 0; w < ldata->out_mat_dims[OUT_W]; w++)
    {
        void *packed_ptr = packed_data + w * ldata->out_mat_dims[OUT_H] * elem_size;
        void *input_ptr = out_mat + w * ldata->out_mat_stride * elem_size;
        memcpy (packed_ptr, input_ptr, ldata->out_mat_dims[OUT_H] * elem_size);
    }
    return packed_data;
}

void *get_packed_ldata_output_rowwise (nasm_ldata_t *ldata)
{
    if (ldata == NULL)
    {
        ERROR_PRTF ("Error in get_packed_ldata_output_rolwise: ldata is NULL.");
        return NULL;
    }
    void *packed_data = NULL;
    size_t elem_size = ldata->layer->dnn->element_size;
    size_t data_size = ldata->out_mat_dims[OUT_H] * ldata->out_mat_dims[OUT_W] * elem_size;
    packed_data = calloc (1, data_size);
    if (packed_data == NULL)
    {
        ERRNO_PRTF ("Error in get_packed_ldata_output_rolwise: calloc failed.");
        return NULL;
    }
    for (unsigned int w = 0; w < ldata->out_mat_dims[OUT_W]; w++)
    {
        for (unsigned int h = 0; h < ldata->out_mat_dims[OUT_H]; h++)
        {
            void *packed_ptr = packed_data + (h * ldata->out_mat_dims[OUT_W] + w) * elem_size;
            void *input_ptr = ldata->out_mat + (w * ldata->out_mat_stride + h) * elem_size;
            memcpy (packed_ptr, input_ptr, elem_size);
        }
    }
    return packed_data;
}

void print_nasm_info (nasm_t *nasm, int print_ninst, int print_data)
{
    if (nasm == NULL)
    {
        printf("Error: NASM is NULL.\n");
        return;
    }
    printf("//////////////////////// Printing NASM Info ////////////////////////\n");
    printf("Original DNN name: %s\n", nasm->dnn->name);
    printf("Nasm hash: %016lx, Type hash: %016lx\n", nasm->nasm_hash, nasm->type_hash);
    printf("Number of ldata: %d\n", nasm->num_ldata);
    printf("Number of batch: %d\n", nasm->batch_size);
    printf("Number of ninst: %d\n", nasm->num_ninst);
    printf("FLOPs per ninst: %d\n", nasm->flop_per_ninst);
    printf("Total FLOPs: %ld\n", nasm->total_flops);
    for (int i = 0; i < nasm->num_ldata; i++)
    {
        print_ldata_info(&nasm->ldata_arr[i], print_ninst, print_data);
    }
    printf("//////////////////////// End of NASM Info ////////////////////////\n");
}

void print_ldata_info (nasm_ldata_t *ldata, int print_ninst, int print_data)
{
    if (ldata == NULL)
    {
        printf("Error: ldata is NULL.\n");
        return;
    }
    printf("//////////////////////// Printing ldata Info ////////////////////////\n");
    printf("Ldata Index: %ld\n", ldata - ldata->nasm->ldata_arr);
    printf("Original layer index: %d\n", ldata->layer->layer_idx);
    printf("Original layer type: %s, Params: \n\t", layer_type_str[ldata->layer->type]);
    for (LAYER_PARAMS i = 0; i < NUM_PARAM_ELEMENTS; i++)
    {
        if (i != NUM_PARAM_ELEMENTS && ldata->layer->params[i] != 0)
            printf("%s:%d ", param_type_str[i], ldata->layer->params[i]);
    }
    printf("\n");
    printf("Ldata Parents: ");
    for (int i = 0; i < NUM_PARENT_ELEMENTS; i++)
    {
        if (ldata->parent_ldata_idx_arr[i] != -1)
            printf("%s: %d ", parent_type_str[i], ldata->parent_ldata_idx_arr[i]);
    }
    printf("\n");
    for (int i = 0; i < NUM_PARENT_ELEMENTS; i++)
    {
        if (ldata->parent_ldata_idx_arr[i] != -1)
        {
            aspen_layer_t *p0_layer = ldata->nasm->ldata_arr[ldata->parent_ldata_idx_arr[i]].layer;
            printf("\t%s idx: %d, type: %s, Params: \n\t\t", parent_type_str[i]
                , p0_layer->layer_idx, layer_type_str[p0_layer->type]);
            for (LAYER_PARAMS i = 0; i < NUM_PARAM_ELEMENTS; i++)
            {
                if (i != NUM_PARAM_ELEMENTS && p0_layer->params[i] != 0)
                    printf("%s:%d ", param_type_str[i], p0_layer->params[i]);
            }
            printf ("\n");
        }
    }
    if (ldata->out_mat != NULL)
    {
        printf("Ldata Output Matrix: %p\n", ldata->out_mat);
    }
    else
    {
        printf ("Ldata Output Matrix: NULL\n");
    }
    printf("Ldata Children (Completed: %d/%d): ", ldata->num_child_ldata_completed, ldata->num_child_ldata);
    for (int i = 0; i < ldata->num_child_ldata; i++)
    {
        printf("%d ", ldata->child_ldata_idx_arr[i]);
    }
    printf("\n");
    printf("Ldata Flop per output element: %d\n", ldata->flop_per_output);
    printf("Ldata Output Matrix Dimensions: (H: %d, W: %d), Stride: %d\n"
        , ldata->out_mat_dims[OUT_H], ldata->out_mat_dims[OUT_W], ldata->out_mat_stride);
    printf("Ldata Output Matrix Memory Size: %ld (bytes)\n", ldata->out_mat_mem_size);
    printf("Ldata Flop per Ninst: %d\n", ldata->flop_per_output*ldata->ninst_tile_dims[OUT_H]*ldata->ninst_tile_dims[OUT_W]);
    printf("Ldata Ninst Tile Dimensions: (H: %d, W: %d)\n", 
        ldata->ninst_tile_dims[OUT_H], ldata->ninst_tile_dims[OUT_W]);
    printf("Number of ninst: %d, Completed: %d\n", ldata->num_ninst, ldata->num_ninst_completed);
    if (print_ninst)
    {
        for (int i = 0; i < ldata->num_ninst; i++)
        {
            printf ("\tNinst %d: ", i);
            print_ninst_info(&ldata->ninst_arr_start[i], print_data);
        }
    }
    printf("////////////////////////  End of ldata Info  ////////////////////////\n");
}

void print_ninst_info (ninst_t *ninst, int print_data)
{
    if (ninst == NULL)
    {
        printf("Error: ninst is NULL.\n");
        return;
    }
    printf ("Ninst Idx: %d, State: %s, Type: %016lx, Operator: %s", 
        ninst->ninst_idx, ninst_state_str[ninst->state], ninst->type_hash,
            layer_type_str[ninst->ldata->layer->type]);
    if (get_ninst_out_mem_without_alloc (ninst) != NULL)
    {
        printf (", Output Matrix: %p\n", get_ninst_out_mem_without_alloc (ninst));
    }
    else
    {
        printf (", Output Matrix: NULL\n");
    }
    printf ("\t\tNinst tile size: (H: %d, W: %d)\n", ninst->tile_dims[OUT_H], ninst->tile_dims[OUT_W]);
    printf ("\t\tNinst tile position: (H: %d, W: %d) ~ (H: %d, W: %d) "
        , ninst->out_mat_pos[OUT_H], ninst->out_mat_pos[OUT_W],
            ninst->out_mat_pos[OUT_H] + ninst->tile_dims[OUT_H] - 1
                , ninst->out_mat_pos[OUT_W] + ninst->tile_dims[OUT_W] - 1);
    LAYER_TYPE layer_type = ninst->ldata->layer->type;
    if ((layer_type == CONV_LAYER || layer_type == FC_LAYER || layer_type == MAXPOOL_LAYER || layer_type == AVGPOOL_LAYER || layer_type == INPUT_LAYER 
        || layer_type == RESIDUAL_LAYER || layer_type == YOLO_LAYER || layer_type == APPEND_LAYER) && ninst->ldata->layer->params[MAT_M] == 0)
    {
        unsigned int out_tensor_pos[NUM_PARAM_ELEMENTS]; 
        get_tensor_pos_from_nist (ninst->ldata, ninst, out_tensor_pos);
        printf ("Tensor Pos (N,C,H,W): (%d,%d,%d,%d)", out_tensor_pos[BATCH], out_tensor_pos[OUT_C],
                    out_tensor_pos[OUT_H],
                     out_tensor_pos[OUT_W]);
    }
    else
    {
        unsigned int out_tensor_pos[NUM_PARAM_ELEMENTS]; 
        get_tensor_pos_from_nist (ninst->ldata, ninst, out_tensor_pos);
        if (layer_type == K_ATTENTION_LAYER)
            printf ("Tensor Pos (B,H,M,N): (%d,%d,%d,%d)", out_tensor_pos[BATCH], out_tensor_pos[NUM_HEAD],
                        out_tensor_pos[MAT_M],
                        out_tensor_pos[MAT_N]);
        else
            printf ("Tensor Pos (B,M,N): (%d,%d,%d)", out_tensor_pos[BATCH],
                    out_tensor_pos[MAT_M],
                     out_tensor_pos[MAT_N]);
    }
    printf ("\n\t\tParent ninst (Completed: %d/%d): "
        , ninst->num_parent_ninsts_completed, ninst->num_parent_ninsts);
    for (int i = 0; i < ninst->num_parent_ninsts; i++)
    {
        if (ninst->parent_ninst_idx_arr == NULL)
        {
            printf("\n\t\t\tError: Parent ninst index array is NULL.\n");
            break;  
        }
        ninst_t *parent_ninst = ninst->parent_ninst_idx_arr[i] + ninst->ldata->nasm->ninst_arr;
        printf("L%ld:%d ", parent_ninst->ldata - parent_ninst->ldata->nasm->ldata_arr,
            parent_ninst->ninst_idx);
    }
    printf("\n\t\tChild ninst (%d): ", ninst->num_child_ninsts);
    for (int i = 0; i < ninst->num_child_ninsts; i++)
    {
        if (ninst->child_ninst_arr == NULL)
        {
            printf("\n\t\t\tError: Child ninst array is NULL.\n");
            break;  
        }
        ninst_t *child_ninst = ninst->child_ninst_arr[i];
        printf("L%ld:%d ", child_ninst->ldata - child_ninst->ldata->nasm->ldata_arr,
            child_ninst->ninst_idx);
    }
    printf("\n\t\tInput pos indexes (%d): ", ninst->num_input_pos);
    printf ("\n");
    if (print_data)
    {
        printf("\n\t\tData:");
        void* out_mat = get_ninst_out_mem_without_alloc (ninst);
        if (out_mat == NULL)
        {
            printf("\n\t\t\tError: Output matrix is NULL.\n");
        }
        for (unsigned int h = 0; h < ninst->tile_dims[OUT_H]; h++)
        {
            printf("\n\t\t\t");
            for (unsigned int w = 0; w < ninst->tile_dims[OUT_W]; w++)
            {
                unsigned int output_mat_h = ninst->out_mat_pos[OUT_H] + h;
                unsigned int output_mat_w = ninst->out_mat_pos[OUT_W] + w;
                printf("%3.2f ", *((float*)out_mat 
                    + output_mat_w*ninst->ldata->out_mat_stride + output_mat_h));
            }
        }
    }
    printf("\n");
}
