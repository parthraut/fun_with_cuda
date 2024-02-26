/*

Problem 3: Parallel Reduction
Objective: Implement a parallel reduction kernel to sum an array of numbers.

Details:

Given an array A of size N, use parallel reduction to find the sum of all elements in A.
Implement a kernel that uses a tree-based approach for reduction. Start by having each thread load one element and then pair-wise add elements across threads, halving the number of active threads in each step.
Make sure to handle the case where N is not a power of two.
Experiment with using shared memory to minimize memory latency.
*/

#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <cuda_runtime.h>
#include <math.h>
#include <assert.h>

__global__ void add(int* A, int N){
    extern __shared__ int sdata[];

    int thx = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // initialize sdata[]
    if(index < N){
        sdata[thx] = A[index];
    }
    else{
        sdata[thx] = 0;
    }

    __syncthreads();

    // reduction step
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (thx < s) {
            sdata[thx] += sdata[thx + s];
        }
        __syncthreads();
    }

    if (thx == 0) A[blockIdx.x] = sdata[0];
}

int main(int argc, char **argv){
    // Initialize array from argv
    assert(argc == 2); // check if N is provided (N is the number of elements in the array A)
    int N = atoi(argv[1]);

    int* A = (int*) malloc(sizeof(int) * N);
    for (int i = 0; i < N; i++){
        // fill A with random numbers
        A[i] = rand() % 100;
    }

    // pointer to device memory
    int* A_d;

    // allocate memory on device
    cudaMalloc((void**)&A_d, sizeof(int) * N);

    // copy array to device memory
    cudaMemcpy(A_d, A, sizeof(int) * N, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int sharedMemSize = threadsPerBlock * sizeof(int);
    int numBlocks = (int)ceil((double)N / (double)threadsPerBlock);

    // add gpu timing code
    clock_t start, stop;
    start = clock();
    add<<<numBlocks, threadsPerBlock, sharedMemSize>>>(A_d, N);

    // now, sum is placed in A_d[0...numBlocks]. Keep invoking kernel until result in A_d[0]
    while(numBlocks > 1){
        N = numBlocks;
        numBlocks = (int)ceil((double)N / (double)threadsPerBlock);
        add<<<numBlocks, threadsPerBlock, sharedMemSize>>>(A_d, N);
    }
    stop = clock();
    double gpu_time = (double)(stop - start) / CLOCKS_PER_SEC;
    printf("GPU time: %f\n", gpu_time);

    // transfer result
    int sum = 0;

    cudaMemcpy(&sum, A_d, sizeof(int), cudaMemcpyDeviceToHost);

    // check sum with CPU
    start = clock();
    int sum_cpu = 0;
    for (int i = 0; i < N; i++){
        sum_cpu += A[i];
    }
    stop = clock();
    double cpu_time = (double)(stop - start) / CLOCKS_PER_SEC;
    printf("CPU time: %f\n", cpu_time);
    
    if(sum == sum_cpu){
        printf("CPU and GPU sum match\n");
    }
    else{
        printf("CPU and GPU sum do not match\n: CPU sum: %d != GPU sum: %d\n", sum_cpu, sum);
    }


    cudaFree(A_d);
    free(A);

}