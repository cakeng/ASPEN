#ifndef _APU_H_
#define _APU_H_

#include "aspen.h"
#include "nasm.h"
#include "util.h"
#include "kernels.h"
#include "cuda_kernels.h"

#define MIN_NINST_TILE_PER_LAYER (10)
#define LAYERS_PER_TRANSFORMER (12)
extern LAYER_TYPE transformer_layer_types[LAYERS_PER_TRANSFORMER];
extern int transformer_parents [LAYERS_PER_TRANSFORMER][2];
// 1. Key MM, 2. Query MM, 3. Value MM,
// 4. K Attention, 5. V Attention, 6. Attention MM, 7. Residual, 8. LayerNorm, 
// 9. Feedforward MM 1, 10. Feedforward MM 2, 11. Residual, 12. LayerNorm

aspen_dnn_t *init_aspen_dnn (unsigned int num_layers, char* name);

void init_aspen_layer (aspen_layer_t *layer, unsigned int layer_num, aspen_dnn_t *dnn);
void create_layer_tensors (aspen_layer_t *layer);
void create_layer_output_tensor (aspen_layer_t *layer, int gpu_idx);
void destroy_aspen_layers (aspen_layer_t* layers, unsigned int num_layers);

aspen_tensor_t *init_aspen_tensor (unsigned int *params_arr, LAYER_PARAMS *order, int num_dims, unsigned int element_size);
void calloc_aspen_tensor (aspen_tensor_t *tensor);
void calloc_aspen_gpu_tensors (aspen_tensor_t *tensor);
void copy_ptr_to_aspen_tensor  (aspen_tensor_t *tensor, void *ptr);
void copy_aspen_tensor_to_ptr  (aspen_tensor_t *tensor, void *ptr);
void copy_aspen_tensor_to_tensor  (aspen_tensor_t *dst, aspen_tensor_t *src);
void copy_aspen_tensor_to_gpu  (aspen_tensor_t *tensor, int gpu_idx);
void copy_aspen_tensor_to_host  (aspen_tensor_t *tensor, int gpu_idx);
void reorder_aspen_tensor (aspen_tensor_t **tensor_ptr, unsigned int *params_arr, LAYER_PARAMS *order, int num_dims);
void *get_aspen_tensor_data (aspen_tensor_t *tensor, LAYER_PARAMS *output_order, int gpu_idx);
void *get_aspen_tensor_element_ptr (aspen_tensor_t *tensor, unsigned int *pos);
void destroy_aspen_tensor(aspen_tensor_t *tensor);
void sync_dnn_data_to_gpu_layer (aspen_layer_t *layer);
void sync_dnn_data_to_gpu_dnn (aspen_dnn_t *dnn);
void sync_output_data_to_host_layer (aspen_layer_t *layer, int gpu_idx);
void sync_output_data_to_host_dnn (aspen_dnn_t *dnn, int gpu_idx);
void sync_output_data_to_gpu_layer (aspen_layer_t *layer, int gpu_idx);
void sync_output_data_to_gpu_dnn (aspen_dnn_t *dnn, int gpu_idx);

nasm_t *apu_create_nasm_without_finding_ninst_parents (aspen_dnn_t *dnn, unsigned int flop_per_ninst, unsigned int batch_size, unsigned int transformer_seq_len);

void init_nasm_ldata (nasm_t *nasm, nasm_ldata_t *ldata, aspen_layer_t *layer);
void destroy_nasm_ldata_arr (nasm_ldata_t *ldata_arr, int num_ldata);

void copy_tensor_data_to_nasm_data (aspen_tensor_t *tensor, nasm_ldata_t *ldata);
void copy_nasm_data_to_tensor_data (nasm_ldata_t *ldata, aspen_tensor_t *tensor);

unsigned int get_tensor_idx_from_pos (aspen_tensor_t *tensor, unsigned int *pos);
void get_tensor_pos_from_idx (aspen_tensor_t *tensor, unsigned int idx, unsigned int *pos);
ninst_t *get_ninst_from_tensor_pos (nasm_ldata_t *ldata, unsigned int *tensor_pos);
ninst_t *get_ninst_from_out_mat_pos (nasm_ldata_t *ldata, unsigned int h, unsigned int w);
void get_out_mat_pos_from_nist (nasm_ldata_t *ldata, ninst_t *ninst, unsigned int *out_mat_pos);
void get_out_mat_pos_from_tensor_pos (nasm_ldata_t *ldata, unsigned int *tensor_pos, unsigned int *out_mat_pos);
void get_tensor_pos_from_out_mat_pos (nasm_ldata_t *ldata, unsigned int *out_mat_pos, unsigned int *tensor_pos);
void get_tensor_pos_from_nist (nasm_ldata_t *ldata, ninst_t *ninst, unsigned int *tensor_pos);

void *get_packed_ldata_output_colwise (nasm_ldata_t *ldata);
void *get_packed_ldata_output_rowwise (nasm_ldata_t *ldata);
void *get_ldata_output (nasm_ldata_t *ldata, LAYER_PARAMS *order);
#endif /* _APU_H_ */