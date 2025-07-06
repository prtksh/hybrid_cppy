// File: src/main.cpp
#include "tensor_engine.h"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

std::vector<float> generate_random_input(int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    std::vector<float> input(size);
    for (int i = 0; i < size; ++i) {
        input[i] = dis(gen);
    }
    return input;
}

void print_predictions(const std::vector<float>& predictions, const std::vector<std::string>& labels) {
    std::cout << "\n=== PREDICTIONS ===" << std::endl;
    for (size_t i = 0; i < predictions.size() && i < labels.size(); ++i) {
        std::cout << std::setw(12) << labels[i] << ": " 
                  << std::setw(8) << std::fixed << std::setprecision(4) 
                  << predictions[i] * 100 << "%" << std::endl;
    }
    
    // Find best prediction
    auto max_it = std::max_element(predictions.begin(), predictions.end());
    int best_idx = std::distance(predictions.begin(), max_it);
    std::cout << "\nBest prediction: " << labels[best_idx] 
              << " (" << (*max_it * 100) << "%)" << std::endl;
}

int main() {
    std::cout << "=== Hybrid C++/Python Inference Engine ===" << std::endl;
    
    // Create engine
    TensorEngine engine;
    
    // Initialize with Python model
    if (!engine.initialize("simple_models", "SimpleNN")) {
        std::cerr << "Failed to initialize engine" << std::endl;
        return -1;
    }
    
    std::cout << "\n=== Running Test Cases ===" << std::endl;
    
    // Test Case 1: Simple Neural Network (784 -> 128 -> 10)
    std::cout << "\nTest 1: Simple Neural Network (MNIST-like)" << std::endl;
    std::vector<float> mnist_input = generate_random_input(784);
    std::vector<int> mnist_shape = {1, 784};
    
    std::vector<float> mnist_result = engine.infer(mnist_input, mnist_shape);
    
    if (!mnist_result.empty()) {
        std::vector<std::string> mnist_labels = {
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
        };
        print_predictions(mnist_result, mnist_labels);
    }
    
    // Test Case 2: Different model
    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << "Test 2: Binary Classifier" << std::endl;
    
    if (engine.initialize("simple_models", "BinaryClassifier")) {
        std::vector<float> binary_input = generate_random_input(100);
        std::vector<int> binary_shape = {1, 100};
        
        std::vector<float> binary_result = engine.infer(binary_input, binary_shape);
        
        if (!binary_result.empty()) {
            std::vector<std::string> binary_labels = {"Negative", "Positive"};
            print_predictions(binary_result, binary_labels);
        }
    }
    
    // Test Case 3: Performance comparison
    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << "Test 3: Performance Comparison" << std::endl;
    
    // Test pure C++ operations
    std::cout << "\nTesting C++ operations directly:" << std::endl;
    std::vector<float> a = generate_random_input(100);
    std::vector<float> b = generate_random_input(100);
    
    auto cpp_result = engine.matmul(a, b, 10, 10, 10);
    auto relu_result = engine.relu(cpp_result);
    auto softmax_result = engine.softmax(relu_result);
    
    engine.print_tensor(softmax_result, "Direct C++ result");
    
    std::cout << "\n=== Engine Tests Complete ===" << std::endl;
    
    return 0;
}
