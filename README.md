# Hybrid C++/Python Neural Network Inference Engine

## prerequisites

- **C++17** compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- **Python 3.6+** with NumPy
- **CMake 3.12+**
- **pybind11** (automatically fetched during build)

##  installation

### 1. clone this repo
```bash
git clone <repository-url>
cd pycpp-eng
```
### 3. steps to build the project
```bash
# Make scripts executable
chmod +x build.sh run.sh setup.sh

# builds the engine
./build.sh
```

## usage

### run script
```bash
# Run the inference engine
./run.sh
```

## configuration

### Build Configuration
The project uses CMake with the following default settings:
- **C++ Standard**: C++17
- **Build Type**: Release (with -O3 optimization)
- **Python**: Auto-detected Python 3.x with NumPy
- **pybind11**: Automatically fetched from GitHub

### env values
```bash
# Set Python path for model loading
export PYTHONPATH="${PYTHONPATH}:$(pwd)/models:$(pwd)"

# Optional: Set specific Python interpreter
export PYTHON_EXECUTABLE=/path/to/python
```

## testing
```bash
# Run the main test suite
./run.sh

# Run with debug information
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make
./hybrid_inference
```

### test cases
1. **MNIST-like Classification**: Tests SimpleNN with 784-dimensional input
2. **Binary Classification**: Tests BinaryClassifier with 100-dimensional input
3. **Performance Comparison**: Compares C++ vs Python operations
4. **Direct C++ Operations**: Tests standalone C++ tensor operations


## contributing

1. fork the repository
2. create a feature branch 
3. commit your changes 
4. push to the branch 
5. open a PR

### development Setup
```bash
# Clone and setup
git clone <repository-url>
cd pycpp-eng
./setup.sh

# Build in debug mode
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make

# Run tests
./hybrid_inference
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
