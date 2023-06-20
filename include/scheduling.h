#ifndef _SCHEDULING_H_
#define _SCHEDULING_H_

#include "nasm.h"
#include "aspen.h"
#include "dse.h"

#include <float.h>
#include <limits.h>

#define SCHEDULE_INIT_BUF_SIZE      (1024 * 1024)
#define PROFILE_REPEAT              4
#define PROFILE_LONG_MESSAGE_SIZE   (1024 * 64)

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

typedef struct device_t {
    int idx;
    SOCK_TYPE type;
    int sock;
    
    char* ip;
    int port;
    struct sockaddr_in server_addr;
    struct sockaddr_in client_addr;
    
    double sync;    // for the same point of time, device timestamp + sync = my timestamp
} device_t;

typedef struct heft_params_t {
    device_t **devices;               // list of devices
    int num_devices;                // number of devices
    
    ninst_t **entry_tasks;           // they have no parent
    ninst_t **exit_tasks;            // they have no child
    int num_entry_tasks;            // number of devices
    int num_exit_tasks;             // number of devices
    

    int **data_mat;                  // Data: matrix of shape (task_num, task_num), data[i][j] means amount of data required to be transmitted from task n_i to n_j.
    double **computation_cost;        // W: matrix of shape (task_num, device_num), w[i][j] means estimated execution time to complete task n_i on processor p_j.
    double *avg_computation_cost;     // \bar{W}: matrix of shape (task_num), sum up compuation cost of every device, then devide by number of devices
    float **data_transfer_rate;      // B: matrix of shape (device_num, device_num), B[i][j] means transfer rate from device[i] to device[j].
    float *communication_startup;    // L: matrix of shape (device_num)
    // float **communication_cost      // C: matrix of shape (token_num, token_num), c[i][j] means communication cost between task i and j. requires tasks to be scheduled.
    float avg_data_transfer_rate;    // \bar{B}
    float avg_communication_startup; // \bar{L}
    float **avg_communication_cost;  // \bar{C}: matrix of shape (token_num, token_num), \bar{c_{i, j}} = \bar{L} + data_{i, j} / \bar{B}

    int **allocation;                   // matrix of shape (task_num, device_num), marks owners of the tasks

} heft_params_t;

typedef struct cpop_params_t {

} cpop_params_t;

typedef struct schedule_meta_t {
    schedule_meta_type_t type;
    int idx;
    int len_data;
    char *data;

    double send_time_sec;
} schedule_meta_t;

int is_ninst_mine(ninst_t *ninst, int device_idx);

void init_full_offload(nasm_t *nasm);
void init_partial_offload(nasm_t *nasm, float compute_ratio);

void init_heft_devices(device_t **devices, SOCK_TYPE *types, char **ips, int* ports, int num_devices, int my_dev_idx);
void init_heft_scheduling(nasm_t *nasm, heft_params_t *heft_params, device_t **devices, int num_devices, int my_dev_idx);
void init_cpop_scheduling(nasm_t *nasm, cpop_params_t *cpop_params, device_t **devices, int num_devices, int my_dev_idx);

double profile_ninst_computation(ninst_t *ninst, int num_repeat);

#endif