# bin/bash

# compile using nvcc
nvcc -o matrix_multiplication matrix_multiplication.cu

# run the program with values [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
for i in 1 2 4 8 16 32 64 128 256 512 1024
do
    ./matrix_multiplication $i
done
