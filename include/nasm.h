#ifndef _NASM_H_
#define _NASM_H_

#include "aspen.h"
#include "rpool.h"

#define PEER_FLAG_SEND 0b01
#define PEER_FLAG_COMPUTE 0b10

struct aspen_peer_t
{
    HASH_t peer_hash;
    int sock;  
    int isUDP;
    ssize_t latency_usec;
    ssize_t bandwidth_bps;
    int listen_port;
    char *ip;
    pthread_mutex_t peer_mutex;
    rpool_queue_t *tx_queue;
};

struct nasm_t
{
    HASH_t nasm_hash;
    HASH_t type_hash;
    aspen_dnn_t *dnn;
    unsigned int batch_size;
    unsigned int tr_seq_len;
    nasm_ldata_t *ldata_arr;
    ninst_t *ninst_arr;
    unsigned int num_ldata;
    _Atomic unsigned int num_ldata_completed;
    _Atomic unsigned int completed;
    unsigned int num_ninst;
    unsigned int min_ninst_per_ldata;
    unsigned int flop_per_ninst;
    size_t total_flops;
    
    void *data;
    pthread_mutex_t nasm_mutex;
    pthread_cond_t nasm_cond;

    int num_peers;
    aspen_peer_t *peer_map;
};

struct nasm_ldata_t
{
    nasm_t *nasm;
    aspen_layer_t *layer;
    unsigned int parent_ldata_idx_arr [NUM_PARENT_ELEMENTS];
    unsigned int *child_ldata_idx_arr; // Allows duplicate entries.
    unsigned int num_child_ldata;
    _Atomic unsigned int num_child_ldata_completed;

    unsigned int flop_per_output;
    unsigned int out_mat_dims [2];
    unsigned int out_mat_stride;
    size_t out_mat_mem_size;
    void *out_mat;
    pthread_mutex_t out_mat_mutex;
    unsigned int ninst_tile_dims [2];
    ninst_t *ninst_arr_start;
    
    unsigned int num_ninst;
    _Atomic unsigned int num_ninst_completed;
};

struct ninst_log_t
{
    HASH_t compute_peer;
    size_t compute_start_usec;
    size_t compute_end_usec;

    HASH_t send_peer;
    size_t send_start_usec;
    size_t send_end_usec;
};

struct ninst_t 
{
    nasm_ldata_t *ldata;
    NINST_STATE state;
    unsigned int ninst_idx;
    unsigned int out_mat_pos [2];
    unsigned int tile_dims [2];
    HASH_t type_hash;

    unsigned int *parent_ninst_idx_arr;
    unsigned int num_parent_ninsts;
    _Atomic unsigned int num_parent_ninsts_completed;

    ninst_t **child_ninst_arr;
    _Atomic unsigned int num_child_ninsts;

    unsigned int num_input_pos;
    
    char *peer_flag; // First bit : Compute, Second bit: Send
};

nasm_t *apu_create_nasm_without_finding_ninst_parents (aspen_dnn_t *dnn, unsigned int flop_per_ninst, unsigned int batch_size,  unsigned int min_ninst_per_ldata, unsigned int transformer_seq_len);

double test_nasm_time_sec (nasm_t *nasm, unsigned int num_iter);

void nasm_ldata_init (nasm_t *nasm, nasm_ldata_t *ldata, aspen_layer_t *layer);
void destroy_nasm_ldata_arr (nasm_ldata_t *ldata_arr, int num_ldata);
void set_nasm_to_finished (nasm_t *nasm);

void ninst_init (nasm_ldata_t *ldata, ninst_t *ninst_ptr, int ninst_idx);
HASH_t get_ninst_hash (ninst_t *ninst);
void destroy_ninst (ninst_t *ninst);

aspen_peer_t *peer_init ();
aspen_peer_t *peer_copy (aspen_peer_t *new_peer, aspen_peer_t *peer);
void destroy_peer (aspen_peer_t *peer);
void set_ninst_compute_peer_idx (ninst_t *ninst, int peer_idx);
void set_ninst_send_peer_idx (ninst_t *ninst, int peer_idx);
int check_ninst_compute_using_peer_idx (ninst_t *ninst, int peer_idx);
int check_ninst_send_using_peer_idx (ninst_t *ninst, int peer_idx);
int get_peer_idx (nasm_t *nasm, HASH_t peer_hash);
aspen_peer_t *get_peer (nasm_t *nasm, int peer_idx);

void copy_ninst (ninst_t *new_ninst, ninst_t *ninst);
void copy_ldata_out_mat_to_buffer (nasm_ldata_t *ldata, void *buffer);
void copy_buffer_to_ldata_out_mat (nasm_ldata_t *ldata, void *buffer);
void copy_ninst_data_to_buffer (ninst_t *ninst, void *buffer);
void copy_buffer_to_ninst_data (ninst_t *ninst, void *buffer);

void alloc_ldata_out_mat (nasm_ldata_t *ldata);
void free_ldata_out_mat (nasm_ldata_t *ldata);

void *get_ninst_out_mem (ninst_t *ninst);
void *get_ninst_out_mem_without_alloc (ninst_t *ninst);

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
ssize_t get_ldata_output (void** out_ptr, nasm_ldata_t *ldata, LAYER_PARAMS *order);

#endif /* _NASM_H_ */