#!/bin/bash

# Compile CUDA program using nvcc
if ! nvcc -o convolution convolution.cu; then
    echo "Error: Compilation failed"
    exit 1
fi

# Run the program with values
for ((i=1; i<=100; i*=2))
do
    echo "Running with value: $i"
    if ! ./convolution $i; then
        echo "Error: Execution failed for value $i"
        exit 1
    fi
done
