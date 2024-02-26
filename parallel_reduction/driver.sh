#!/bin/bash

# Compile CUDA program using nvcc
if ! nvcc -o parallel_reduction parallel_reduction.cu; then
    echo "Error: Compilation failed"
    exit 1
fi

# Run the program with values [1, 2, 4, ..., 1024]
for ((i=1; i<=10000000000; i*=10))
do
    echo "Running with value: $i"
    if ! ./parallel_reduction $i; then
        echo "Error: Execution failed for value $i"
        exit 1
    fi
done
