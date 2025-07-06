import numpy as np
import json
import os

class SimpleNN:
    """Simple Neural Network for MNIST-like classification"""
    
    def __init__(self):
        self.cpp_engine = None
        
        # Network architecture
        self.input_size = 784
        self.hidden_size = 128
        self.output_size = 10
        
        # Initialize weights and biases
        self._initialize_weights()
        
        print(f"SimpleNN initialized: {self.input_size} -> {self.hidden_size} -> {self.output_size}")
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization"""
        # Weight matrices
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2.0 / self.input_size)
        self.b1 = np.zeros(self.hidden_size)
        
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2.0 / self.hidden_size)
        self.b2 = np.zeros(self.output_size)
        
        print("Weights initialized with Xavier initialization")
    
    def set_cpp_engine(self, engine):
        """Set reference to C++ engine for fast operations"""
        self.cpp_engine = engine
        print("C++ engine linked to Python model")
    
    def forward(self, x):
        """Forward pass through the network"""
        print(f"Python: Forward pass started, input shape: {x.shape}")
        
        # Flatten input if needed
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)
        
        batch_size = x.shape[0]
        
        if self.cpp_engine:
            print("Python: Using C++ engine for computations")
            
            # First layer: input -> hidden
            # Convert to lists for C++ processing
            x_flat = x.flatten().tolist()
            w1_flat = self.W1.flatten().tolist()
            
            # C++ matrix multiplication
            hidden = self.cpp_engine.matmul(x_flat, w1_flat, batch_size, self.input_size, self.hidden_size)
            
            # Add bias
            hidden = self.cpp_engine.add_bias(hidden, self.b1.tolist())
            
            # ReLU activation
            hidden = self.cpp_engine.relu(hidden)
            
            # Second layer: hidden -> output
            w2_flat = self.W2.flatten().tolist()
            output = self.cpp_engine.matmul(hidden, w2_flat, batch_size, self.hidden_size, self.output_size)
            
            # Add bias
            output = self.cpp_engine.add_bias(output, self.b2.tolist())
            
            # Softmax activation
            output = self.cpp_engine.softmax(output)
            
            print("Python: C++ computations completed")
            return np.array(output).reshape(batch_size, self.output_size)
        
        else:
            print("Python: Using NumPy for computations")
            # Fallback to NumPy
            hidden = np.dot(x, self.W1) + self.b1
            hidden = np.maximum(0, hidden)  # ReLU
            
            output = np.dot(hidden, self.W2) + self.b2
            output = self._softmax(output)
            
            return output
    
    def _softmax(self, x):
        """Softmax activation function"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def get_model_info(self):
        """Return model information"""
        return {
            'name': 'SimpleNN',
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'parameters': self.input_size * self.hidden_size + self.hidden_size * self.output_size
        }


class BinaryClassifier:
    """Simple binary classifier"""
    
    def __init__(self):
        self.cpp_engine = None
        
        # Network architecture
        self.input_size = 100
        self.hidden_size = 50
        self.output_size = 2
        
        # Initialize weights
        self._initialize_weights()
        
        print(f"BinaryClassifier initialized: {self.input_size} -> {self.hidden_size} -> {self.output_size}")
    
    def _initialize_weights(self):
        """Initialize weights"""
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.1
        self.b1 = np.zeros(self.hidden_size)
        
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.1
        self.b2 = np.zeros(self.output_size)
    
    def set_cpp_engine(self, engine):
        """Set reference to C++ engine"""
        self.cpp_engine = engine
        print("C++ engine linked to BinaryClassifier")
    
    def forward(self, x):
        """Forward pass"""
        print(f"Python: BinaryClassifier forward pass, input shape: {x.shape}")
        
        # Flatten input if needed
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)
        
        batch_size = x.shape[0]
        
        if self.cpp_engine:
            # Use C++ engine
            x_flat = x.flatten().tolist()
            w1_flat = self.W1.flatten().tolist()
            
            # First layer
            hidden = self.cpp_engine.matmul(x_flat, w1_flat, batch_size, self.input_size, self.hidden_size)
            hidden = self.cpp_engine.add_bias(hidden, self.b1.tolist())
            hidden = self.cpp_engine.relu(hidden)
            
            # Second layer
            w2_flat = self.W2.flatten().tolist()
            output = self.cpp_engine.matmul(hidden, w2_flat, batch_size, self.hidden_size, self.output_size)
            output = self.cpp_engine.add_bias(output, self.b2.tolist())
            
            # Sigmoid activation for binary classification
            output = [1.0 / (1.0 + np.exp(-x)) for x in output]
            
            return np.array(output).reshape(batch_size, self.output_size)
        
        else:
            # NumPy fallback
            hidden = np.dot(x, self.W1) + self.b1
            hidden = np.maximum(0, hidden)
            
            output = np.dot(hidden, self.W2) + self.b2
            output = 1.0 / (1.0 + np.exp(-output))  # Sigmoid
            
            return output


class ConvolutionalNet:
    """More complex convolutional network"""
    
    def __init__(self):
        self.cpp_engine = None
        
        # For simplicity, we'll simulate conv operations with dense layers
        self.input_size = 28 * 28  # 784 for MNIST
        self.conv_output_size = 32 * 13 * 13  # After conv and pooling
        self.hidden_size = 128
        self.output_size = 10
        
        self._initialize_weights()
        
        print(f"ConvNet initialized (simulated with dense layers)")
    
    def _initialize_weights(self):
        """Initialize weights for simulated conv net"""
        # Simulate conv layer as dense layer
        self.W_conv = np.random.randn(self.input_size, self.conv_output_size) * 0.01
        self.b_conv = np.zeros(self.conv_output_size)
        
        # Dense layers
        self.W_dense = np.random.randn(self.conv_output_size, self.hidden_size) * 0.01
        self.b_dense = np.zeros(self.hidden_size)
        
        self.W_out = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.b_out = np.zeros(self.output_size)
    
    def set_cpp_engine(self, engine):
        """Set reference to C++ engine"""
        self.cpp_engine = engine
        print("C++ engine linked to ConvNet")
    
    def forward(self, x):
        """Forward pass through simulated conv net"""
        print(f"Python: ConvNet forward pass, input shape: {x.shape}")
        
        # Flatten input
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)
        
        batch_size = x.shape[0]
        
        if self.cpp_engine:
            # Simulated conv layer
            x_flat = x.flatten().tolist()
            w_conv_flat = self.W_conv.flatten().tolist()
            
            conv_out = self.cpp_engine.matmul(x_flat, w_conv_flat, batch_size, self.input_size, self.conv_output_size)
            conv_out = self.cpp_engine.add_bias(conv_out, self.b_conv.tolist())
            conv_out = self.cpp_engine.relu(conv_out)
            
            # Dense layer
            w_dense_flat = self.W_dense.flatten().tolist()
            dense_out = self.cpp_engine.matmul(conv_out, w_dense_flat, batch_size, self.conv_output_size, self.hidden_size)
            dense_out = self.cpp_engine.add_bias(dense_out, self.b_dense.tolist())
            dense_out = self.cpp_engine.relu(dense_out)
            
            # Output layer
            w_out_flat = self.W_out.flatten().tolist()
            output = self.cpp_engine.matmul(dense_out, w_out_flat, batch_size, self.hidden_size, self.output_size)
            output = self.cpp_engine.add_bias(output, self.b_out.tolist())
            output = self.cpp_engine.softmax(output)
            
            return np.array(output).reshape(batch_size, self.output_size)
        
        else:
            # NumPy fallback
            conv_out = np.dot(x, self.W_conv) + self.b_conv
            conv_out = np.maximum(0, conv_out)
            
            dense_out = np.dot(conv_out, self.W_dense) + self.b_dense
            dense_out = np.maximum(0, dense_out)
            
            output = np.dot(dense_out, self.W_out) + self.b_out
            output = self._softmax(output)
            
            return output
    
    def _softmax(self, x):
        """Softmax activation function"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# Test function for standalone Python testing
def test_models():
    """Test models without C++ engine"""
    print("Testing models in pure Python mode...")
    
    # Test SimpleNN
    print("\n=== Testing SimpleNN ===")
    model = SimpleNN()
    test_input = np.random.randn(1, 784)
    result = model.forward(test_input)
    print(f"SimpleNN output shape: {result.shape}")
    print(f"SimpleNN output: {result}")
    
    # Test BinaryClassifier
    print("\n=== Testing BinaryClassifier ===")
    binary_model = BinaryClassifier()
    test_input = np.random.randn(1, 100)
    result = binary_model.forward(test_input)
    print(f"BinaryClassifier output shape: {result.shape}")
    print(f"BinaryClassifier output: {result}")
    
    print("\nPython model tests completed!")


if __name__ == "__main__":
    test_models()

