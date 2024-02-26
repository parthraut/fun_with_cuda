/*

Problem 5: 2D Convolution
Objective: Implement a simple 2D convolution kernel.

Details:

Given a 2D input matrix (image) I of size NxN and a convolution filter (kernel) F of size 3x3, compute the output matrix (filtered image) O using 2D convolution.
For simplicity, assume the convolution is applied without padding (resulting in an output matrix of size (N-2)x(N-2)) and with a stride of 1.
Write a CUDA kernel to apply the convolution filter to the input matrix. Each thread should compute one element of the output matrix.
Consider how shared memory could be used to optimize memory accesses for this problem.


*/


#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void convolution(int *I, int *F, int *O, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N-2 && j < N-2){
        int sum = 0;
        for (int k = 0; k < 3; k++){
            for (int l = 0; l < 3; l++){
                sum += I[(i+k) * N + (j+l)] * F[k * 3 + l];
            }
        }
        O[i * (N-2) + j] = sum;
    }
}

int main(int argc, char **argv){
    
    if (argc != 2){
        printf("Usage: %s N\n", argv[0]);
        exit(1);
    }
    int N = atoi(argv[1]);

    // generate I and F
    int *I = (int *)malloc(N * N * sizeof(int));
    int *F = (int *)malloc(3 * 3 * sizeof(int));

    // initialize I with random values
    for (int i = 0; i < N * N; i++){
        I[i] = rand() % 256;
    }
    // initialize F with random binary values
    for (int i = 0; i < 3 * 3; i++){
        F[i] = rand() % 2;
    }

    // initialize O
    int *O = (int *)malloc((N-2) * (N-2) * sizeof(int));

    // allocate device memory
    int *d_I, *d_F, *d_O;
    cudaMalloc((void **)&d_I, N * N * sizeof(int));
    cudaMalloc((void **)&d_F, 3 * 3 * sizeof(int));
    cudaMalloc((void **)&d_O, (N-2) * (N-2) * sizeof(int));

    // copy data to device
    cudaMemcpy(d_I, I, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_F, F, 3 * 3 * sizeof(int), cudaMemcpyHostToDevice);

    // launch kernel
    dim3 block(32, 32);
    dim3 grid((int)ceil(((double)N-2)/32), (int)ceil(((double)N-2)/32));

    // start timer
    clock_t start = clock();
    convolution<<<grid, block>>>(d_I, d_F, d_O, N);
    cudaDeviceSynchronize();
    // stop timer
    clock_t end = clock();

    double gpu_time = ((double)(end - start)) / CLOCKS_PER_SEC;

    // copy data back to host
    cudaMemcpy(O, d_O, (N-2) * (N-2) * sizeof(int), cudaMemcpyDeviceToHost);

    // execute convolution on CPU to verify results
    int *O_CPU = (int *)malloc((N-2) * (N-2) * sizeof(int));

    start = clock();
    for (int i = 0; i < N-2; i++){
        for (int j = 0; j < N-2; j++){
            int sum = 0;
            for (int k = 0; k < 3; k++){
                for (int l = 0; l < 3; l++){
                    sum += I[(i+k) * N + (j+l)] * F[k * 3 + l];
                }
            }
            O_CPU[i * (N-2) + j] = sum;
        }
    }
    end = clock();

    double cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC;

    // compare results with a flag
    int flag = 1;
    for (int i = 0; i < (N-2) * (N-2); i++){
        if (O[i] != O_CPU[i]){
            flag = 0;
            break;
        }
    }
    if (flag){
        printf("Results match!\n");
    } else {
        printf("Results do not match!\n");
    }

    double speedup = cpu_time / gpu_time;
    printf("speedup: %f\n", speedup);

    FILE* file = fopen("convolution.csv", "a");
    fprintf(file, "%d, %f, %f, %f\n", atoi(argv[1]), gpu_time, cpu_time, cpu_time / gpu_time);
    


    // free device memory
    cudaFree(d_I);
    cudaFree(d_F);
    cudaFree(d_O);

    // free host memory
    free(I);
    free(F);
    free(O);



}