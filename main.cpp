#include <stdio.h>
#include <getopt.h>
#include <stdbool.h>
#include <stdlib.h>

#include "util.h"
#include "mat_mul.h"
#include "cuda_aspen_tests.h"

// This program tests chained matrix multiplication performance of various libraries.
// Chained matrix multiplication is a sequence of matrix multiplications, where the
// output of one matrix multiplication is the input of the next matrix multiplication.
// The output of the last matrix multiplication is the final result.

// The program tests the following 6 cases:
// Three full sized chained matmuls using OpenBLAS(CPU), cuBLAS, and a custom CUDA GEMM kernel.
// Three partitioned chained matmuls, where each matrix on the output chain is partitioned on the 
// N dimension, either using the number of partitions specified by the user (using option -f), 
// or the partition size specified by the user (using option -s).

// There are 6 tests performed, and is executed with the following order.
// 1. Full-sized chained matmul using OpenBLAS on CPU.
// 2. Full-sized chained matmul using cuBLAS.
// 3. Full-sized chained matmul using a custom CUDA GEMM kernel defined in cuda_aspen_tests.cu.
// 4. Partitioned chained matmul using cuBLAS.
// 5. Partitioned chained matmul using a custom CUDA GEMM kernel defined in cuda_aspen_tests.cu.
// 6. Partitioned chained matmul using cuBLAS, but using the "batched GEMM" function of cuBLAS.

// Besides option -f and -s, the program also accepts the following options:
// -p : print matrix elements. (default: off)
// -v : validate matrix multiplication. (default: off)
// -h : print help page.
// -c : number of matrices on the GEMM chain (Number of matrix on the chain) (default: 1)
// -n : number of executions (default: 1)
// M : number of rows of matrix A and C. (default: 8)
// N : number of columns of matrix B and C. (default: 8)
// K : number of columns of matrix A and rows of B. (default: 8)
// Option M, N, K are just specified as an integer, without option flags.
// Example execution:
// ./main -v -c 3 -n 10 -s 256 768 1024 768
// This will execute 10 times of chained matmul with 3 matrices on the chain, and each output matrix
// is partitioned on the N dimension with size 256. It will also validate the result of the computation.
// ASCII art of the computation:
//
//                          1024 (dim N)
//                       256  256  256  256
//                      ___________________
//                     |    |    |    |    |
//                     |    |    |    |    |
//                  768|    |    |    |    |
//                     |    |    |    |    |
//          768        |____|____|____|____| C_out_arr[0]
//     ______________   ___________________
//    |              | |    |    |    |    |
//    |              | |    |    |    |    |
// 768|              | |    |    |    |    |
//    |    A_arr[0]  | |    |    |    |    |
//    |______________| |____|____|____|____| C_out_arr[1]
//     ______________   ___________________
//    |              | |    |    |    |    |
//    |              | |    |    |    |    |
// 768|              | |    |    |    |    |
//    |    A_arr[1]  | |    |    |    |    |
//    |______________| |____|____|____|____| C_out_arr[2]
//     ______________   ___________________
//    |              | |    |    |    |    |
//    |              | |    |    |    |    |
// 768|              | |    |    |    |    |
//    |    A_arr[2]  | |    |    |    |    |
//    |______________| |____|____|____|____| C_out_arr[3] 
//       (Computed 10 times, and validated)

// Full sized chained matmul will perform 3 matrix multiplications of
// A_arr[0] * C_out_arr[0] = C_out_arr[1]
// A_arr[1] * C_out_arr[1] = C_out_arr[2]
// A_arr[2] * C_out_arr[2] = C_out_arr[3].
// Partitioned chained matmul will perform 12 matrix multiplications of
// A_arr[0] * C_out_arr[0][0...3] = C_out_arr[1][0...3]
// A_arr[1] * C_out_arr[1][0...3] = C_out_arr[2][0...3]
// A_arr[2] * C_out_arr[2][0...3] = C_out_arr[3][0...3]
// , where the computation  A_arr[i] * C_out_arr[i][j] on a given j value is performed independently 
// of other computations on the different j values. (i.e. the computation is parallelized on the j)

// The computations are structured using a class, named aspen_mat_mul, defined in cuda_aspen_tests.h.
// The class has member variables that are used to define and execute the given matrix multiplication,
// such as M, N, K, stride, matrix pointers, various CUDA API objects, etc.
// The class member functions contain various utility functions and run_xxx() functions that are used
// to execute the computation using the specified library (OpenBLAS, cuBLAS, or custom CUDA GEMM kernel).

// In the main function, the program first parses the command line arguments, and then creates an
// array of memory pointers for the input matrices and output matrices, in both host and GPU.
// The program then creates an vector of aspen_mat_mul objects, which specifies the chain of 
// matrix multiplications that need to be performed, and initializes them with the
// specified parameters. The program then executes the computation using the specified library,
// using the run_test() function, which takes the vector of aspen_mat_mul objects and the test function
// as input. The test function is a function pointer that points to one of the 6 test cases. 
// run_test performs the execution, validates the results, and returns the execution time. 

///////////////////////////// TODO //////////////////////////////
// The program does not have dynamic dependency tracking & scheduling, where aspen_mat_mul objects are 
// dynamically scheduled to be executed when all of their parent aspen_mat_mul objects are executed. 
// Instead, the program simply puts dependent aspen_mat_mul objects in the same CUDA stream, and executes them sequentially.
// As ASPEN execution requires dynamic dependency tracking & scheduling, please implement a CUDA event based asynchronous
// tracking of executed aspen_mat_mul objects, and use it to dynamically schedule dependent 
// child aspen_mat_mul objects. There are some basic variables and functions in the aspen_mat_mul class
// that can be used to implement this functionality, but the detailed functionality are yet to be implemented. (Please refer to cuda_aspen_tests.h)

// Thank you for your time and effort helping us to implement ASPEN execution in CUDA!
// Jongseok Park.


// Prints the help page.
static void print_help(const char *prog_name)
{
    printf("Usage: %s [-pvh] [-n num_iterations] M N K\n", prog_name);
    printf("Options:\n");
    printf("  -p : print matrix elements. (default: off)\n");
    printf("  -v : validate matrix multiplication. (default: off)\n");
    printf("  -h : print this page.\n");
    printf("  -c : number of matrices on the chain (default: 1)\n");
    printf("  -f : number of partitions to the N dimension (default: 1)\n");
    printf("  -n : number of executions (default: 1)\n");
    printf("   M : number of rows of matrix A and C. (default: 8)\n");
    printf("   N : number of columns of matrix B and C. (default: 8)\n");
    printf("   K : number of columns of matrix A and rows of B. (default: 8)\n");
}

// Default input values.
static bool print_matrix = false;
static bool validation = false;
static int num_gemm_chains = 1;
static int M = 128, N = 128, K = 128;
static int num_iterations = 1;
static float *C_ans, *C_out;
static int num_partition = 1;
static int partition_size = -1; 
// num_partition determines the partition size when partition_size is -1.
// If partition_size is not -1, num_partition is ignored.

// Parses Input options.
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

// Runs the matmul test on given test function and vector of aspen_mat_mul objects.
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

    zero_mat (C_out, M, N);
}

// The 6 test functions, which are passed to run_test.
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
        (*aspen_objs).rbegin()[i]->copy_C_from_cuda();
    }
    for (int i = 0; i < num_partition; ++i)
    {
        (*aspen_objs).rbegin()[i]->synchronize();
    }
}

void aspen_run_cuBLAS_batched (void *obj)
{
    std::vector<aspen_mat_mul*> *aspen_objs = (std::vector<aspen_mat_mul*>*)obj;
    for (int i = 0; i < num_partition; ++i)
    {
        (*aspen_objs)[i]->copy_B_to_cuda();
    }
    for (int i = 0; i < (int)aspen_objs->size(); i += num_partition)
    {
        std::vector<aspen_mat_mul*>::const_iterator first = (*aspen_objs).begin() + i;
        std::vector<aspen_mat_mul*>::const_iterator last = (*aspen_objs).begin() + i + num_partition;
        std::vector<aspen_mat_mul*> newVec(first, last);
        newVec.front()->run_cuBLAS_batched_GEMM (newVec);
    }
    for (int i = 0; i < num_partition; ++i)
    {
        (*aspen_objs).rbegin()[i]->copy_C_from_cuda();
    }
    for (int i = 0; i < num_partition; ++i)
    {
        (*aspen_objs).rbegin()[i]->synchronize();
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
        (*aspen_objs).rbegin()[i]->copy_C_from_cuda();
    }
    for (int i = 0; i < num_partition; ++i)
    {
        (*aspen_objs).rbegin()[i]->synchronize();
    }

}

int main(int argc, char **argv)
{

    parse_opt(argc, argv);
    if (K != M)
    {
        // For the chain of GEMMs to be valid, K must be equal to M.
        printf("K must be equal to M.\n");
        exit(1);
    }

    // Initialize the matrix memory.
    printf("Initializing matrix... ");
    float *A_arr [num_gemm_chains+1];      // Holds the A matrices.
    float *A_cuda_arr [num_gemm_chains+1]; // Holds the A matrices on the GPU.
    float *C_out_arr [num_gemm_chains+1];  // Holds the C matrices.
    float *C_cuda_arr [num_gemm_chains+1]; // Holds the C matrices on the GPU.
    float *C_ans_arr [num_gemm_chains+1];  // Holds the C matrix answers, for validation.
    // Memory allocation loop.
    for (int i = 0; i < num_gemm_chains+1; ++i)
    {
        alloc_mat(&A_arr[i], K, M);
        alloc_mat(&C_ans_arr[i], M, N);
        if (i != 0)
            alloc_mat(&C_out_arr[i], M, N);
        if (i == num_gemm_chains)
        {
            // Set the last C_out and C_ans to a seperate variable,
            // for easier validation.
            C_out = C_out_arr[i];
            C_ans = C_ans_arr[i];
        }
        // Randomize the A matrices.
        rand_mat(A_arr[i], K, M);
        cudaMalloc(&A_cuda_arr[i], M * K * sizeof(float));
        cudaMalloc(&C_cuda_arr[i], M * N * sizeof(float));
    }
    // Randomize the first C matrix (input to the GEMM chain, therefore a fixed value.)
    rand_mat(C_ans_arr[0], M, N);
    C_out_arr[0] = C_ans_arr[0];

    // Compute the GEMM answers using a simple CPU based GEMM.
    for (int i = 1; i < num_gemm_chains+1; ++i)
    {
        compute_mat_mul (A_arr[i-1], C_ans_arr[i-1], C_ans_arr[i], M, N, K);
    }

    // Create GPU handles and streams.
    cublasHandle_t cublas_handles[num_partition];
    cudaStream_t cuda_streams[num_partition];
    for (int i = 0; i < num_partition; ++i)
    {
        cublasCreate(&cublas_handles[i]);
        cudaStreamCreate(&cuda_streams[i]);
    }

    // Create the aspen_mat_mul objects, and set the right variables for the given GEMM,
    // including the memory locations, CUDA handles, and streams.
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
    // Initialization complete.
    printf("done!\n");

    // Run the full chained matrix multiplication tests using run_test function.
    printf("Testing CPU GEMM (openBLAS)...\n");
    run_test (&aspen_mat_mul_chain, aspen_run_cpu);

    printf("Testing GPU GEMM (cuBLAS)...\n");
    run_test (&aspen_mat_mul_chain, aspen_run_cuBLAS);

    printf("Testing GPU GEMM (custom)...\n");
    run_test (&aspen_mat_mul_chain, aspen_run_custom_GEMM);

    printf("Testing Split GPU GEMM (cuBLAS)...\n");
    // For the split tests, we need to split the aspen_mat_mul objects into multiple
    // aspen_mat_mul objects, each with a single GEMM.
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
    // Run the partitioned GEMM tests, using the newly split aspen_mat_mul objects.
    run_test (&aspen_mat_mul_chain_split, aspen_run_cuBLAS_split);

    printf("Testing Split GPU GEMM (custom)...\n");
    run_test (&aspen_mat_mul_chain_split, aspen_run_custom_split);

    printf("Testing Split GPU GEMM (cuBLAS Batched)...\n");
    run_test (&aspen_mat_mul_chain_split, aspen_run_cuBLAS_batched);

    return 0;
}
