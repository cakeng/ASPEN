#ifndef _SCHEDULING_H_
#define _SCHEDULING_H_

#define SCHEDULE_INIT_BUF_SIZE      (1024 * 1024)
#define PROFILE_REPEAT              4
#define PROFILE_LONG_MESSAGE_SIZE   (1024 * 64)
#define SCHEDULE_MAX_DEVICES        4

#include "nasm.h"
#include "profiling.h"
#include "aspen.h"

#include <float.h>
#include <limits.h>


typedef enum {
    FULL_OFFLOAD,
    PARTIAL_OFFLOAD,
    HEFT,
    CPOP
} schedule_policy_t;

typedef enum {
    SYNCHRONIZE,
    HEFT_COMPUTATION_COST,
    HEFT_TRANSMIT_RATE,
    HEFT_SYNC
} schedule_meta_type_t;

struct sched_task_t {
    int idx;
    sched_processor_t *processor;
    float start_time;
    float end_time;
    sched_task_t *next;
    sched_task_t *prev;
};

struct sched_processor_t {
    int idx;
    int num_task;
    sched_task_t *task_list;
};

int is_ninst_mine(ninst_t *ninst, int device_idx);

void init_full_local(nasm_t *nasm);
void init_full_offload(nasm_t *nasm);
void init_partial_offload(nasm_t *nasm, float compute_ratio);
void init_heft(char *target_config, char *target_bin, char *target_nasm_dir, ninst_profile_t **ninst_profile, network_profile_t *network_profile, int num_device);

void heft_gen_dependency(nasm_t *nasm, int **dependency);
void heft_gen_data(nasm_t *nasm, ninst_profile_t **ninst_profile, int **dependency, float **data);
void heft_gen_W(nasm_t *nasm, ninst_profile_t **ninst_profile, int num_device, float **W, float *W_avg);
void heft_gen_B(nasm_t *nasm, network_profile_t *network_profile, int num_device, float **B, float *B_avg);
void heft_gen_L(nasm_t *nasm, network_profile_t *network_profile, int num_device, float *L, float *L_avg);
void heft_gen_C_avg(nasm_t *nasm, float L_avg, float **data, float B_avg, int **dependency, float **C_avg);
void gen_rank_upward(nasm_t *nasm, float *W_avg, float **C_avg, int **dependency, float *rank_upward);
void gen_rank_downward(nasm_t *nasm, float *W_avg, float **C_avg, int **dependency, float *rank_downward);
float calc_rank_upward_rec(nasm_t *nasm, float *W_avg, float **C_avg, int **dependency, float *rank_upward, int target_idx);
float calc_rank_downward_rec(nasm_t *nasm, float *W_avg, float **C_avg, int **dependency, float *rank_downward, int target_idx);

sched_processor_t *heft_init_processor(int num_processor);
sched_task_t *heft_init_task(int num_ninst);
float heft_earliest_idle(sched_processor_t *sched_processor, float min_limit, float duration);
void heft_push_task(sched_processor_t *sched_processor, sched_task_t *sched_task);

int compare_by_rank_upward(const void *ninst_1, const void *ninst_2);

#endif