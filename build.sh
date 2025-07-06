#!/bin/bash
# File: build.sh

set -e
echo "=== Building Hybrid Inference Engine ==="

if [ -d "build" ]; then
    echo "Cleaning previous build..."
    rm -rf build
fi

mkdir -p build
cd build
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release
echo "Building..."
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
echo "✓ Build completed successfully!"
echo "Executable: build/hybrid_inference"

