#!/bin/bash

# Compile CUDA program using nvcc
if ! nvcc -o matrix_multiplication matrix_multiplication.cu; then
    echo "Error: Compilation failed"
    exit 1
fi

# Run the program with values [1, 2, 4, ..., 1024]
for i in {1..10} 1024
do
    echo "Running with value: $i"
    if ! ./matrix_multiplication $i; then
        echo "Error: Execution failed for value $i"
        exit 1
    fi
done
