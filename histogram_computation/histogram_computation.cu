/*

Problem 4: Histogram Computation
Objective: Compute a histogram of values in an array.

Details:

Given an array A of integers (ranging from 0 to M-1) and its size N, compute the histogram of A. The histogram array H of size M should contain the counts of each integer in A (i.e., H[i] is the number of times i appears in A).
Write a CUDA kernel to compute the histogram in parallel. Consider atomic operations to avoid race conditions when updating the histogram counts.
Discuss and handle the potential performance implications of using atomic operations in global memory.

*/

#include <stdio.h>
#include <malloc.h>
#include <cuda_runtime.h>

__global__ void make_histogram(int* A, int* H, int N, int M){
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < N){

        // must be done atomically to avoid race conditions
        atomicAdd(&H[A[index]], 1);
    }
}

int main(){

    // array A of integers - size: N, vals: 0 to M-1

    int N = 100;
    int M = 100;

    int* A = (int*) malloc(N * sizeof(int));

    // generate random values for A
    for(int i = 0; i < N; i++){
        A[i] = rand() % M;
    }

    // create array H of size M
    int* H = (int*) malloc(M * sizeof(int));

    // create device arrays
    int* d_A;
    int* d_H;

    cudaMalloc((void**)&d_A, N * sizeof(int));
    cudaMalloc((void**)&d_H, M * sizeof(int));

    cudaMemset(d_H, 0, M * sizeof(int));

    // copy A to device
    cudaMemcpy(d_A, A, N * sizeof(int), cudaMemcpyHostToDevice);

    // kernel call
    make_histogram<<<(N + 255) / 256, 256>>>(d_A, d_H, N, M);

    // copy H from device
    cudaMemcpy(H, d_H, M * sizeof(int), cudaMemcpyDeviceToHost);

    // run on cpu to verify
    int* H_cpu = (int*) malloc(M * sizeof(int));
    for(int i = 0; i < M; i++){
        H_cpu[i] = 0;
    }

    for(int i = 0; i < N; i++){
        H_cpu[A[i]]++;
    }

    // compare results
    bool isValid = true;
    for(int i = 0; i < M; i++){
        if(H[i] != H_cpu[i]){
            isValid = false;
            printf("Error: Histogram mismatch found.\n");
            break;
        }
    }
    if(isValid){
        printf("Success: GPU and CPU histograms match.\n");
    }

    // free memory
    free(A);
    free(H);
    free(H_cpu);
    cudaFree(d_A);
    cudaFree(d_H);

}