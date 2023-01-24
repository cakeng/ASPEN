#include <stdio.h>
#include <getopt.h>
#include <stdbool.h>
#include <stdlib.h>

#include "util.h"
#include "mat_mul.h"
#include "cuda_aspen_tests.h"

static void print_help(const char *prog_name)
{
    printf("Usage: %s [-pvh] [-n num_iterations] M N K\n", prog_name);
    printf("Options:\n");
    printf("  -p : print matrix data. (default: off)\n");
    printf("  -v : validate matrix multiplication. (default: off)\n");
    printf("  -h : print this page.\n");
    printf("  -c : number of gemm chains (default: 1)\n");
    printf("  -f : number of partitions to the N dimension (default: 1)\n");
    printf("  -n : number of iterations (default: 1)\n");
    printf("   M : number of rows of matrix A and C. (default: 8)\n");
    printf("   N : number of columns of matrix B and C. (default: 8)\n");
    printf("   K : number of columns of matrix A and rows of B. (default: 8)\n");
}

static bool print_matrix = false;
static bool validation = false;
static int num_gemm_chains = 1;
static int M = 128, N = 128, K = 128;
static int num_iterations = 1;
static float *C_ans, *C_out;
static int num_partition = 1;
static int partition_size = -1;

static void parse_opt(int argc, char **argv)
{
    int c;
    while ((c = getopt(argc, argv, "pvhc:f:n:s:")) != -1)
    {
        switch (c)
        {
        case 'p':
            print_matrix = true;
            break;
        case 'v':
            validation = true;
            break;
        case 'n':
            num_iterations = atoi(optarg);
            break;
        case 'c':
            num_gemm_chains = atoi(optarg);
            break;
        case 'f':
            num_partition = atoi(optarg);
            break;
        case 's':
            partition_size = atoi(optarg);
            break;
        case 'h':
        default:
            print_help(argv[0]);
            exit(0);
        }
    }
    for (int i = optind, j = 0; i < argc; ++i, ++j)
    {
        switch (j)
        {
        case 0:
            M = atoi(argv[i]);
            break;
        case 1:
            N = atoi(argv[i]);
            break;
        case 2:
            K = atoi(argv[i]);
            break;
        default:
            break;
        }
    }
    if (partition_size != -1)
        num_partition = N / partition_size + (N % partition_size != 0);
    else
        partition_size = N / num_partition + (N % num_partition != 0);
    printf("Options:\n");
    printf("  Problem size: M = %d, N = %d, K = %d\n", M, N, K);
    printf("  Number of iterations: %d\n", num_iterations);
    printf("  Number of gemm chains: %d\n", num_gemm_chains);
    printf("  Number of partitions: %d\n", num_partition);
    printf("  Partition size: %d\n", partition_size);
    printf("  Print matrix: %s\n", print_matrix ? "on" : "off");
    printf("  Validation: %s\n", validation ? "on" : "off");
    printf("\n");
}

void run_test (void *obj, void (*test_func)(void *obj))
{
    double elapsed_time_sum = 0;
    for (int i = -3; i < num_iterations; ++i)
    {
        if (i >= 0)
        {
            printf("Calculating...(iter=%d) ", i);
            fflush(stdout);
        }
        else
        {
            printf("Warmup...\n");
            fflush(stdout);
        }
        
        timer_start(0);
        test_func(obj);
        double elapsed_time = timer_stop(0);

        if (i >= 0)
        {
            printf("%f sec\n", elapsed_time);
            fflush(stdout);
            elapsed_time_sum += elapsed_time;
        }
    }
    double elapsed_time_avg = elapsed_time_sum / num_iterations;
    printf("Avg. time: %f sec\n", elapsed_time_avg);
    printf("Avg. throughput: %f GFLOPS\n", 2.0 * M * N * K * num_gemm_chains / elapsed_time_avg / 1e9);

    if (print_matrix)
    {
        printf("C\n");
        print_mat(C_out, M, N, false);
    }

    if (validation)
        check_mat_mul(C_ans, C_out, M, N, K);

    rand_mat (C_out, M, N);
}

void aspen_run_cpu (void *obj)
{
    std::vector<aspen_mat_mul> *aspen_objs = (std::vector<aspen_mat_mul>*)obj;
    for (auto &aspen_obj : *aspen_objs)
    {
        aspen_obj.run_cpu();
    }
}

void aspen_run_cuBLAS (void *obj)
{
    std::vector<aspen_mat_mul> *aspen_objs = (std::vector<aspen_mat_mul>*)obj;
    aspen_objs->front().copy_B_to_cuda();
    for (auto &aspen_obj : *aspen_objs)
    {
        aspen_obj.run_cuBLAS();
    }
    aspen_objs->back().copy_C_from_cuda();
    aspen_objs->back().synchronize();
}

void aspen_run_custom_GEMM (void *obj)
{
    std::vector<aspen_mat_mul> *aspen_objs = (std::vector<aspen_mat_mul>*)obj;
    aspen_objs->front().copy_B_to_cuda();
    for (auto &aspen_obj : *aspen_objs)
    {
        aspen_obj.run_custom_CUDA_GEMM();
    }
    aspen_objs->back().copy_C_from_cuda();
    aspen_objs->back().synchronize();
}

void aspen_run_cuBLAS_split (void *obj)
{
    std::vector<aspen_mat_mul*> *aspen_objs = (std::vector<aspen_mat_mul*>*)obj;
    for (int i = 0; i < num_partition; ++i)
    {
        (*aspen_objs)[i]->copy_B_to_cuda();
    }
    for (auto &aspen_obj : *aspen_objs)
    {
        aspen_obj->run_cuBLAS();
    }
    for (int i = 0; i < num_partition; ++i)
    {
        (*aspen_objs)[i]->copy_C_from_cuda();
    }
    for (int i = 0; i < num_partition; ++i)
    {
        (*aspen_objs)[i]->synchronize();
    }
}

void aspen_run_custom_split (void *obj)
{
    std::vector<aspen_mat_mul*> *aspen_objs = (std::vector<aspen_mat_mul*>*)obj;
    for (int i = 0; i < num_partition; ++i)
    {
        (*aspen_objs)[i]->copy_B_to_cuda();
    }
    for (auto &aspen_obj : *aspen_objs)
    {
        aspen_obj->run_custom_CUDA_GEMM();
    }
    for (int i = 0; i < num_partition; ++i)
    {
        (*aspen_objs)[i]->copy_C_from_cuda();
    }
    for (int i = 0; i < num_partition; ++i)
    {
        (*aspen_objs)[i]->synchronize();
    }
}

int main(int argc, char **argv)
{
    parse_opt(argc, argv);
    if (K != M)
    {
        printf("K must be equal to M.\n");
        exit(1);
    }
    printf("Initializing matrix... ");
    float *A_arr [num_gemm_chains+1];
    float *A_cuda_arr [num_gemm_chains+1];
    float *C_out_arr [num_gemm_chains+1];
    float *C_cuda_arr [num_gemm_chains+1];
    float *C_ans_arr [num_gemm_chains+1];
    for (int i = 0; i < num_gemm_chains+1; ++i)
    {
        alloc_mat(&A_arr[i], K, M);
        alloc_mat(&C_ans_arr[i], M, N);
        if (i != 0)
            alloc_mat(&C_out_arr[i], M, N);
        if (i == num_gemm_chains)
        {
            C_out = C_out_arr[i];
            C_ans = C_ans_arr[i];
        }
        rand_mat(A_arr[i], K, M);

        cudaMalloc(&A_cuda_arr[i], M * K * sizeof(float));
        cudaMalloc(&C_cuda_arr[i], M * N * sizeof(float));
    }
    rand_mat(C_ans_arr[0], M, N);
    C_out_arr[0] = C_ans_arr[0];
    for (int i = 1; i < num_gemm_chains+1; ++i)
    {
        compute_mat_mul (A_arr[i-1], C_ans_arr[i-1], C_ans_arr[i], M, N, K);
    }
    cublasHandle_t cublas_handles[num_partition];
    cudaStream_t cuda_streams[num_partition];
    for (int i = 0; i < num_partition; ++i)
    {
        cublasCreate(&cublas_handles[i]);
        cudaStreamCreate(&cuda_streams[i]);
    }
    std::vector<aspen_mat_mul> aspen_mat_mul_chain;

    for (int i = 1; i < num_gemm_chains+1; ++i)
    {
        aspen_mat_mul_chain.emplace_back 
            (M, N, K, 1.0, A_arr[i-1], K, C_out_arr[i-1], K, 0.0, C_out_arr[i], M);
        aspen_mat_mul_chain.back().set_cuda_handle (cublas_handles[0]);
        aspen_mat_mul_chain.back().set_cuda_stream (cuda_streams[0]);
        aspen_mat_mul_chain.back().set_cuda_memory 
            (A_cuda_arr[i-1], C_cuda_arr[i-1], C_cuda_arr[i]);
        cudaMemcpy (A_cuda_arr[i-1], A_arr[i-1], M * K * sizeof(float), cudaMemcpyHostToDevice);
    }

    printf("done!\n");

    printf("Testing CPU GEMM (openBLAS)...\n");
    run_test (&aspen_mat_mul_chain, aspen_run_cpu);

    printf("Testing GPU GEMM (cuBLAS)...\n");
    run_test (&aspen_mat_mul_chain, aspen_run_cuBLAS);

    printf("Testing GPU GEMM (custom)...\n");
    run_test (&aspen_mat_mul_chain, aspen_run_custom_GEMM);

    printf("Testing Split GPU GEMM (cuBLAS)...\n");
    std::vector<aspen_mat_mul*> aspen_mat_mul_chain_split;
    for (auto &aspen_obj : aspen_mat_mul_chain)
    {
        auto aspen_mat_mul_split = aspen_obj.split_mat_mul_by_num (1, num_partition);
        int i = 0;
        for (auto &aspen_split_obj : aspen_mat_mul_split)
        {
            aspen_split_obj->set_cuda_handle (cublas_handles[i%num_partition]);
            aspen_split_obj->set_cuda_stream (cuda_streams[i%num_partition]);
            aspen_mat_mul_chain_split.push_back (aspen_split_obj);
            i++;
        }
    }
    run_test (&aspen_mat_mul_chain_split, aspen_run_cuBLAS_split);

    printf("Testing Split GPU GEMM (custom)...\n");
    run_test (&aspen_mat_mul_chain_split, aspen_run_custom_split);

    return 0;
}
