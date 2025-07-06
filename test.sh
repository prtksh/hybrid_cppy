#!/bin/bash
# File: test.sh

set -e
echo "=== Testing Hybrid Inference Engine ==="

echo "Test 1: Python models standalone..."
cd models
python3 simple_models.py
cd ..

echo "Test 2: Testing build process..."
if [ ! -f "build/hybrid_inference" ]; then
    echo "Building project..."
    ./build.sh
fi

echo "Test 3: Testing full integration..."
./run.sh
echo "✓ All tests completed!"

