#ifndef _APU_H_
#define _APU_H_

#include "aspen.h"
#include "nasm.h"


aspen_dnn_t *init_aspen_dnn (unsigned int num_layers, char* name);
void init_aspen_layer (aspen_layer_t *layer, unsigned int layer_num, aspen_dnn_t *dnn);
void destroy_aspen_layers (aspen_layer_t* layers, unsigned int num_layers);

aspen_tensor_t *init_aspen_tensor (unsigned int *params_arr, LAYER_PARAMS *dim_info_arr, int num_dims);

void destroy_aspen_tensor(aspen_tensor_t *tensor);

void init_nasm_ldata (nasm_t *nasm, nasm_ldata_t *ldata, aspen_layer_t *layer);
void destroy_nasm_ldata_arr (nasm_ldata_t *ldata_arr, int num_ldata);
void create_layer_tensors (aspen_layer_t *layer);

ninst_t *get_ninst_from_tensor_pos (nasm_ldata_t *ldata, unsigned int *tensor_pos);
ninst_t *get_ninst_from_out_mat_pos (nasm_ldata_t *ldata, unsigned int h, unsigned int w);
void get_out_mat_pos_from_nist (nasm_ldata_t *ldata, ninst_t *ninst, unsigned int *out_mat_pos);
void get_out_mat_pos_from_tensor_pos (nasm_ldata_t *ldata, unsigned int *tensor_pos, unsigned int *out_mat_pos);
void get_tensor_pos_from_out_mat_pos (nasm_ldata_t *ldata, unsigned int *out_mat_pos, unsigned int *tensor_pos);
void get_tensor_pos_from_nist (nasm_ldata_t *ldata, ninst_t *ninst, unsigned int *tensor_pos);
#endif /* _APU_H_ */