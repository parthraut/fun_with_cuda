# Problem 2: Matrix Multiplication
## Objective: Implement matrix multiplication in CUDA (C = A * B), where A, B, and C are square matrices of size N x N.

### Details:

Initialize matrices A and B with some values (e.g., A[i][j] = i + j; B[i][j] = i - j;).
Write a CUDA kernel to compute the matrix multiplication. Remember, the value of each element C[i][j] is the dot product of the ith row of A and the jth column of B.
Optimize your kernel to make use of shared memory to reduce global memory accesses.