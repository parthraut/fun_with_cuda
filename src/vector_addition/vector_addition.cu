#include <stdio.h>
#include <malloc.h>
#include <cuda_runtime.h>
#include <math.h>

// CUDA Kernel for adding two numbers


__global__ void add_array(int* a, int* b, int* c, int N){
	
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < N) c[index] = a[index] + b[index];

}


int main(int argc, char** argv){
	int size = atoi(argv[1]);

	// host memory pointers
	int* array_a = (int*)malloc(sizeof(int) * size);
	int* array_b = (int*)malloc(sizeof(int) * size);
	int* array_c = (int*)malloc(sizeof(int) * size);

	for(int i = 0; i < size; ++i){
		array_a[i] = array_b[i] = i;
		
	}

	// Time on CPU
	clock_t start = clock();
	for(int i = 0; i < size; ++i){
		array_c[i] = array_a[i] + array_b[i];
	}
	clock_t end = clock();
	double cpu_time = (double)(end - start) / CLOCKS_PER_SEC;
	

	// device memory pointers
	int* array_a_d;
	int* array_b_d;
	int* array_c_d;
	
	cudaMalloc((void**)&array_a_d, sizeof(int) * size);
	cudaMalloc((void**)&array_b_d, sizeof(int) * size);
	cudaMalloc((void**)&array_c_d, sizeof(int) * size);

	// copy
	const int n = size;
	cudaMemcpy(array_a_d, array_a, n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(array_b_d, array_b, n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(array_c_d, array_c, n * sizeof(int), cudaMemcpyHostToDevice);

	
	int threadsPerBlock = 256;
	int numBlocks = (int)ceil((double)size / (double)threadsPerBlock);

	// Time on GPU
	start = clock();
	add_array<<<numBlocks, threadsPerBlock>>>(array_a_d, array_b_d, array_c_d, size);
	cudaDeviceSynchronize();
	end = clock();
	double gpu_time = (double)(end - start) / CLOCKS_PER_SEC;

	cudaMemcpy(array_c, array_c_d, size, cudaMemcpyDeviceToHost);

	// calculate speedup
    printf("Speedup: %f\n", cpu_time / gpu_time);

	FILE* file = fopen("vector_addition.csv", "a");
    fprintf(file, "%d, %f, %f, %f\n", atoi(argv[1]), gpu_time, cpu_time, cpu_time / gpu_time);

	cudaFree(array_a_d);
	cudaFree(array_b_d);
	cudaFree(array_c_d);


	free(array_a);
	free(array_b);
	free(array_c);

}
