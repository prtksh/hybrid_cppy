#!/usr/bin/env python3
# File: generate_data.py

import numpy as np
import json
import os

def generate_mnist_like_data(num_samples=100):
    images = np.random.randn(num_samples, 28, 28)
    labels = np.random.randint(0, 10, num_samples)
    return images, labels

def generate_binary_data(num_samples=100):
    features = np.random.randn(num_samples, 100)
    labels = np.random.randint(0, 2, num_samples)
    return features, labels

def save_data():
    os.makedirs('data', exist_ok=True)
    mnist_images, mnist_labels = generate_mnist_like_data(1000)
    np.savez('data/mnist_like.npz', images=mnist_images, labels=mnist_labels)
    print("✓ Generated MNIST-like data: data/mnist_like.npz")

    binary_features, binary_labels = generate_binary_data(1000)
    np.savez('data/binary_data.npz', features=binary_features, labels=binary_labels)
    print("✓ Generated binary data: data/binary_data.npz")

    metadata = {
        'mnist_like': {
            'samples': 1000,
            'image_shape': [28, 28],
            'num_classes': 10,
            'description': 'Synthetic MNIST-like data'
        },
        'binary_data': {
            'samples': 1000,
            'feature_dim': 100,
            'num_classes': 2,
            'description': 'Synthetic binary classification data'
        }
    }

    with open('data/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("✓ Generated metadata: data/metadata.json")

if __name__ == "__main__":
    save_data()
    print("\nData generation completed!")

