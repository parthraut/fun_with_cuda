#include <stdio.h>
#include <malloc.h>
#include <cuda_runtime.h>
#include <math.h>

// CUDA Kernel for adding two numbers
__global__ void add(int a, int b, int *c){
	*c = a + b;
}


__global__ void add_array(int* a, int* b, int* c, int N){
	
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < N) c[index] = a[index] + b[index];

}

int gpu_run();


int main(){
	int a, b, c;
	int *d_c;

	int size = sizeof(int);

	cudaMalloc((void **)&d_c, size);
	
	a = 2;
	b = 7;
	
	add<<<1,1>>>(a, b, d_c);

	cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);
	
	cudaFree(d_c);
	
	printf("2 + 7 = %d\n", c);
	
	gpu_run();
	return 0;

}



int gpu_run(){

	const int size = 1000;
	int* array_a = (int*)malloc(sizeof(int) * size);
	int* array_b = (int*)malloc(sizeof(int) * size);
	int* array_c = (int*)malloc(sizeof(int) * size);

	for(int i = 0; i < size; ++i){
		array_a[i] = array_b[i] = i;
		
	}

	

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

	add_array<<<numBlocks, threadsPerBlock>>>(array_a_d, array_b_d, array_c_d, size);

	cudaMemcpy(array_c, array_c_d, size, cudaMemcpyDeviceToHost);

	printf("success!\n");

	cudaFree(array_a_d);
	cudaFree(array_b_d);
	cudaFree(array_c_d);


	free(array_a);
	free(array_b);
	free(array_c);

}
