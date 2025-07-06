#!/bin/bash
# File: run.sh

set -e
echo "=== Running Hybrid Inference Engine ==="

if [ ! -f "build/hybrid_inference" ]; then
    echo "Error: Executable not found. Please run ./build.sh first"
    exit 1
fi

export PYTHONPATH="${PYTHONPATH}:$(pwd)/models:$(pwd)"
echo "Starting hybrid inference engine..."
./build/hybrid_inference
echo "✓ Execution completed!"

