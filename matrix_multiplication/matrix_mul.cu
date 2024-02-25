
/*
Problem 2: Matrix Multiplication
Objective: Implement matrix multiplication in CUDA (C = A * B), where A, B, and C are square matrices of size N x N.

Details:

Initialize matrices A and B with some values (e.g., A[i][j] = i + j; B[i][j] = i - j;).
Write a CUDA kernel to compute the matrix multiplication. Remember, the value of each element C[i][j] is the dot product of the ith row of A and the jth column of B.
Optimize your kernel to make use of shared memory to reduce global memory accesses.

*/
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void matrix_mul(int* A, int* B, int* C, int N){
    // do matrix multiplication
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < N && col < N){
        int sum = 0;

        for(int k = 0; k < N; ++k){
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main(){

    const int matrixSize = 10000;
    int* A = (int*) malloc(matrixSize * matrixSize * sizeof(int));
    int* B = (int*) malloc(matrixSize * matrixSize * sizeof(int));
    int* C = (int*) malloc(matrixSize * matrixSize * sizeof(int));

    // initialize A and B
    for(int i = 0; i < matrixSize; i++){
        for(int j = 0; j < matrixSize; j++){
            A[i * matrixSize + j] = i + j;
            B[i * matrixSize + j] = i - j;
        }
    }

    // allocate A, B, C on device
    int* A_d;
    int* B_d;
    int* C_d;
    
    cudaMalloc((void**)&A_d, sizeof(int) * matrixSize * matrixSize);
    cudaMalloc((void**)&B_d, sizeof(int) * matrixSize * matrixSize);
    cudaMalloc((void**)&C_d, sizeof(int) * matrixSize * matrixSize);


    // transfer to device
    cudaMemcpy(A_d, A, sizeof(int) * matrixSize * matrixSize,  cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, sizeof(int) * matrixSize * matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C, sizeof(int) * matrixSize * matrixSize, cudaMemcpyHostToDevice);

    // calculate parallelization parameters
    dim3 threadsPerBlock(16,16);
    dim3 numBlocks((matrixSize + threadsPerBlock.x - 1) / threadsPerBlock.x,
               (matrixSize + threadsPerBlock.y - 1) / threadsPerBlock.y);

    clock_t start = clock();

    matrix_mul<<<numBlocks, threadsPerBlock>>>(A_d, B_d, C_d, matrixSize);

    clock_t end = clock();
    double gpu_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time on GPU: %f\n", gpu_time);

    // transfer result back to host

    cudaMemcpy(C, C_d, sizeof(int) * matrixSize * matrixSize, cudaMemcpyDeviceToHost);


    // calculate matrix product on CPU
    int* C_CPU = (int*) malloc(matrixSize * matrixSize * sizeof(int));
    start = clock();
    for(int i = 0; i < matrixSize; i++){
        for(int j = 0; j < matrixSize; j++){
            int sum = 0;
            for(int k = 0; k < matrixSize; k++){
                sum += A[i * matrixSize + k] * B[k * matrixSize + j];
            }
            C_CPU[i * matrixSize + j] = sum;
        }
    }
    end = clock();
    double cpu_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time on CPU: %f\n", cpu_time);

    // verify result
    bool correct = true;
    for(int i = 0; i < matrixSize; i++){
        for(int j = 0; j < matrixSize; j++){
            if(C[i * matrixSize + j] != C_CPU[i * matrixSize + j]){
                printf("Error: C[%d][%d] = %d, C_CPU[%d][%d] = %d\n", i, j, C[i * matrixSize + j], i, j, C_CPU[i * matrixSize + j]);
                correct = false;
            }
        }
    }
    if(correct){
        printf("Result is correct!\n");
    }

    // calculate speedup
    printf("Speedup: %f\n", cpu_time / gpu_time);


    free(A);
    free(B);
    free(C);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

}