// File: src/tensor_engine.h
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/embed.h>
#include <vector>
#include <memory>
#include <iostream>
#include <algorithm>
#include <chrono>

namespace py = pybind11;

class TensorEngine {
private:
    std::unique_ptr<py::scoped_interpreter> interpreter;
    py::object python_model;
    bool is_initialized = false;

public:
    TensorEngine();
    ~TensorEngine();
    
    // Initialize with Python model
    bool initialize(const std::string& model_file, const std::string& model_class);
    
    // Core tensor operations (optimized C++)
    std::vector<float> matmul(const std::vector<float>& a, const std::vector<float>& b,
                             int rows_a, int cols_a, int cols_b);
    
    std::vector<float> relu(const std::vector<float>& input);
    std::vector<float> softmax(const std::vector<float>& input);
    std::vector<float> add_bias(const std::vector<float>& input, const std::vector<float>& bias);
    
    // Main inference function
    std::vector<float> infer(const std::vector<float>& input, const std::vector<int>& input_shape);
    
    // Utility functions
    void print_tensor(const std::vector<float>& tensor, const std::string& name);
    bool is_ready() const { return is_initialized; }
};
