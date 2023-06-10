#ifndef _APU_H_
#define _APU_H_

#include "aspen.h"
#include "nasm.h"
#include "util.h"
#include "kernels.h"
#include "rpool.h"
#include "cuda_kernels.h"

#define APU_GENERATION_COEFF ((double)0.8)
#define APU_GENERATION_NUM_NINST 512
#define APU_GENERATION_COEFF_GPU ((double)0.8)
#define APU_GENERATION_NUM_NINST_GPU 35
#define APU_GENERATION_NUM_FLOPS 1e8

void aspen_init_naive (aspen_dnn_t* dnn, unsigned int *input_params, void *input_data, int gpu_idx);
void aspen_run_naive (aspen_dnn_t* dnn, unsigned int *input_params, void *input_data, int gpu_idx);

aspen_dnn_t *init_aspen_dnn (unsigned int num_layers, char* name);
void apu_load_dnn_data_from_file (aspen_dnn_t *dnn, char *input_path);
void *aspen_load_input_NHWC(char *input_filename, unsigned int *input_dims, unsigned int element_size);
void *aspen_load_input(char *input_filename, unsigned int *input_dims, unsigned int element_size);

void init_aspen_layer (aspen_layer_t *layer, unsigned int layer_num, aspen_dnn_t *dnn);
void create_layer_tensors (aspen_layer_t *layer);
void create_layer_output_tensor (aspen_layer_t *layer, int gpu_idx);
void create_layer_col_idx_tensor (aspen_layer_t *layer, int gpu_idx);
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

double test_nasm_time_sec (nasm_t *nasm, unsigned int num_iter, int gpu_idx);

nasm_t *apu_create_nasm_without_finding_ninst_parents (aspen_dnn_t *dnn, unsigned int flop_per_ninst, unsigned int batch_size,  unsigned int min_ninst_per_ldata, unsigned int transformer_seq_len);

void init_nasm_ldata (nasm_t *nasm, nasm_ldata_t *ldata, aspen_layer_t *layer);
void destroy_nasm_ldata_arr (nasm_ldata_t *ldata_arr, int num_ldata);
void set_nasm_to_finished (nasm_t *nasm);

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