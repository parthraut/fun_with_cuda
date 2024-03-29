#!/bin/bash

# Compile CUDA program using nvcc
if ! nvcc -o matrix_multiplication matrix_mul.cu; then
    echo "Error: Compilation failed"
    exit 1
fi

# Run the program with values [1, 2, 4, ..., 1024]
for ((i=1; i<=2048; i+=32))
do
    echo "Running with value: $i"
    if ! ./matrix_multiplication $i; then
        echo "Error: Execution failed for value $i"
        exit 1
    fi
done
