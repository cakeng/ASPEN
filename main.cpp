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
    printf("  -n : number of iterations (default: 1)\n");
    printf("   M : number of rows of matrix A and C. (default: 8)\n");
    printf("   N : number of columns of matrix B and C. (default: 8)\n");
    printf("   K : number of columns of matrix A and rows of B. (default: 8)\n");
}

static bool print_matrix = false;
static bool validation = false;
static bool skip_data_movement = false;
static int M = 128, N = 128, K = 128;
static int num_iterations = 1;

static void parse_opt(int argc, char **argv)
{
    int c;
    while ((c = getopt(argc, argv, "pvhst:n:")) != -1)
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
        case 's':
            skip_data_movement = true;
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
    printf("Options:\n");
    printf("  Problem size: M = %d, N = %d, K = %d\n", M, N, K);
    printf("  Number of iterations: %d\n", num_iterations);
    printf("  Skip data movement: %s\n", skip_data_movement ? "on" : "off");
    printf("  Print matrix: %s\n", print_matrix ? "on" : "off");
    printf("  Validation: %s\n", validation ? "on" : "off");
    printf("\n");
}

int main(int argc, char **argv)
{
    parse_opt(argc, argv);

    printf("Initializing matrix... ");
    float *A, *B, *C;
    double elapsed_time_sum, elapsed_time_avg;
    alloc_mat(&A, M, K);
    alloc_mat(&B, K, N);
    alloc_mat(&C, M, N);
    rand_mat(A, M, K);
    rand_mat(B, K, N);
    aspen_mat_mul aspen_mat_mul (M, N, K, 1.0, A, K, B, N, 0.0, C, N);
    printf("done!\n");

    // printf("Initializing CPU GEMM (openBLAS)...\n");
    // elapsed_time_sum = 0;
    // for (int i = 0; i < num_iterations; ++i)
    // {
    //     printf("Calculating...(iter=%d) ", i);
    //     fflush(stdout);

    //     timer_start(0);
        
    //     aspen_mat_mul.run_cpu();
    //     double elapsed_time = timer_stop(0);

    //     printf("%f sec\n", elapsed_time);
    //     fflush(stdout);
    //     elapsed_time_sum += elapsed_time;
    // }

    // elapsed_time_avg = elapsed_time_sum / num_iterations;
    // printf("Avg. time: %f sec\n", elapsed_time_avg);
    // printf("Avg. throughput: %f GFLOPS\n", 2.0 * M * N * K / elapsed_time_avg / 1e9);

    // if (print_matrix)
    // {
    //     printf("A\n");
    //     print_mat(A, M, K);
    //     printf("B\n");
    //     print_mat(B, K, N);
    //     printf("C\n");
    //     print_mat(C, M, N);
    // }

    // if (validation)
    //     check_mat_mul(A, B, C, M, N, K);

    printf("Initializing GPU GEMM (cuBLAS)...\n");
    aspen_mat_mul.allocate_cuda_memory();
    elapsed_time_sum = 0;
    for (int i = 0; i < num_iterations; ++i)
    {
        printf("Calculating...(iter=%d) ", i);
        fflush(stdout);
        timer_start(0);
        
        aspen_mat_mul.copy_A_B_to_cuda();
        aspen_mat_mul.run_cuBLAS();
        aspen_mat_mul.copy_C_from_cuda();
        aspen_mat_mul.synchronize();
        double elapsed_time = timer_stop(0);

        printf("%f sec\n", elapsed_time);
        fflush(stdout);
        elapsed_time_sum += elapsed_time;
    }

    elapsed_time_avg = elapsed_time_sum / num_iterations;
    printf("Avg. time: %f sec\n", elapsed_time_avg);
    printf("Avg. throughput: %f GFLOPS\n", 2.0 * M * N * K / elapsed_time_avg / 1e9);

    if (print_matrix)
    {
        printf("A\n");
        print_mat(A, M, K);
        printf("B\n");
        print_mat(B, K, N);
        printf("C\n");
        print_mat(C, M, N);
    }

    if (validation)
        check_mat_mul(A, B, C, M, N, K);

    printf("Initializing Split GPU GEMM (cuBLAS)...\n");
    auto aspen_mat_mul_split = aspen_mat_mul.split_mat_mul(1, 4);
    elapsed_time_sum = 0;
    for (int i = 0; i < num_iterations; ++i)
    {
        printf("Calculating...(iter=%d) ", i);
        fflush(stdout);
        timer_start(0);
        
        for (auto &aspen_mat_mul_itr : aspen_mat_mul_split)
        {
            aspen_mat_mul_itr->copy_A_B_to_cuda();
            aspen_mat_mul_itr->run_cuBLAS();
            aspen_mat_mul_itr->copy_C_from_cuda();
        }
        for (auto &aspen_mat_mul_itr : aspen_mat_mul_split)
        {
            aspen_mat_mul_itr->synchronize();
        }
        double elapsed_time = timer_stop(0);

        printf("%f sec\n", elapsed_time);
        fflush(stdout);
        elapsed_time_sum += elapsed_time;
    }

    elapsed_time_avg = elapsed_time_sum / num_iterations;
    printf("Avg. time: %f sec\n", elapsed_time_avg);
    printf("Avg. throughput: %f GFLOPS\n", 2.0 * M * N * K / elapsed_time_avg / 1e9);

    if (print_matrix)
    {
        printf("A\n");
        print_mat(A, M, K);
        printf("B\n");
        print_mat(B, K, N);
        printf("C\n");
        print_mat(C, M, N);
    }

    if (validation)
        check_mat_mul(A, B, C, M, N, K);

    return 0;
}
