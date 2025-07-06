// File: src/tensor_engine.cpp
#include "tensor_engine.h"
#include <Python.h>

TensorEngine::TensorEngine() {
    try {
        // Initialize Python interpreter with proper error handling
        if (!Py_IsInitialized()) {
            interpreter = std::make_unique<py::scoped_interpreter>();
            std::cout << "Python interpreter initialized successfully" << std::endl;
        } else {
            std::cout << "Python interpreter already initialized" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize Python interpreter: " << e.what() << std::endl;
        interpreter.reset();
    }
}

TensorEngine::~TensorEngine() {
    if (interpreter) {
        std::cout << "Shutting down Python interpreter" << std::endl;
    }
}

bool TensorEngine::initialize(const std::string& model_file, const std::string& model_class) {
    try {
        // Add current directory to Python path
        py::module_ sys = py::module_::import("sys");
        py::list path = sys.attr("path");
        path.append("./models");
        path.append(".");
        
        // Import the model module
        py::module_ model_module = py::module_::import(model_file.c_str());
        
        // Create model instance
        py::object ModelClass = model_module.attr(model_class.c_str());
        python_model = ModelClass();
        
        // For now, skip passing C++ engine to Python model
        // This will use NumPy fallback in Python
        std::cout << "Note: Using NumPy fallback (C++ engine not passed to Python)" << std::endl;
        
        is_initialized = true;
        std::cout << "Model '" << model_class << "' loaded successfully" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize model: " << e.what() << std::endl;
        is_initialized = false;
        return false;
    }
}

std::vector<float> TensorEngine::matmul(const std::vector<float>& a, const std::vector<float>& b,
                                       int rows_a, int cols_a, int cols_b) {
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<float> result(rows_a * cols_b, 0.0f);
    
    // Optimized matrix multiplication
    for (int i = 0; i < rows_a; ++i) {
        for (int j = 0; j < cols_b; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < cols_a; ++k) {
                sum += a[i * cols_a + k] * b[k * cols_b + j];
            }
            result[i * cols_b + j] = sum;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "C++ MatMul (" << rows_a << "x" << cols_a << " * " << cols_a << "x" << cols_b 
              << ") took: " << duration.count() << " μs" << std::endl;
    
    return result;
}

std::vector<float> TensorEngine::relu(const std::vector<float>& input) {
    std::vector<float> result;
    result.reserve(input.size());
    
    for (float val : input) {
        result.push_back(std::max(0.0f, val));
    }
    
    return result;
}

std::vector<float> TensorEngine::softmax(const std::vector<float>& input) {
    std::vector<float> result(input.size());
    
    // Find max for numerical stability
    float max_val = *std::max_element(input.begin(), input.end());
    
    // Compute exp and sum
    float sum = 0.0f;
    for (size_t i = 0; i < input.size(); ++i) {
        result[i] = std::exp(input[i] - max_val);
        sum += result[i];
    }
    
    // Normalize
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] /= sum;
    }
    
    return result;
}

std::vector<float> TensorEngine::add_bias(const std::vector<float>& input, const std::vector<float>& bias) {
    std::vector<float> result(input.size());
    
    for (size_t i = 0; i < input.size(); ++i) {
        result[i] = input[i] + bias[i % bias.size()];
    }
    
    return result;
}

std::vector<float> TensorEngine::infer(const std::vector<float>& input, const std::vector<int>& input_shape) {
    if (!is_initialized) {
        std::cerr << "Engine not initialized!" << std::endl;
        return {};
    }
    
    try {
        // Convert input to numpy array
        py::array_t<float> np_input = py::cast(input);
        np_input = np_input.reshape(input_shape);
        
        // Call Python model
        auto start = std::chrono::high_resolution_clock::now();
        py::array_t<float> result = python_model.attr("forward")(np_input);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Total inference time: " << duration.count() << " ms" << std::endl;
        
        // Convert back to C++ vector
        std::vector<float> cpp_result(result.data(), result.data() + result.size());
        return cpp_result;
    } catch (const std::exception& e) {
        std::cerr << "Inference failed: " << e.what() << std::endl;
        return {};
    }
}

void TensorEngine::print_tensor(const std::vector<float>& tensor, const std::string& name) {
    std::cout << name << ": [";
    for (size_t i = 0; i < std::min(tensor.size(), size_t(10)); ++i) {
        std::cout << tensor[i];
        if (i < tensor.size() - 1) std::cout << ", ";
    }
    if (tensor.size() > 10) std::cout << "...";
    std::cout << "] (size: " << tensor.size() << ")" << std::endl;
}



