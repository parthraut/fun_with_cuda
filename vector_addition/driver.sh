#!/bin/bash

# Compile CUDA program using nvcc
if ! nvcc -o vector_addition vector_addition.cu; then
    echo "Error: Compilation failed"
    exit 1
fi

# Run the program with values
for ((i=1; i<=1000000000; i*=2))
do
    echo "Running with value: $i"
    if ! ./vector_addition $i; then
        echo "Error: Execution failed for value $i"
        exit 1
    fi
done
