"""
Dataset loading utilities for the MULAN demo app.
Handles loading of various datasets including image datasets.
"""

import numpy as np
import requests
from io import BytesIO
from PIL import Image
from sklearn import datasets

# Try to import TensorFlow, if available
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class DatasetWrapper:
    """Wrapper class to provide consistent interface for all datasets."""
    
    def __init__(self, data, target, target_names=None, feature_names=None):
        self.data = data
        self.target = target
        self.target_names = (target_names if target_names is not None else [str(i) for i in range(len(np.unique(target)))])
        self.feature_names = (feature_names if feature_names is not None else [f'Feature_{i}' for i in range(data.shape[1])])


def load_fashion_mnist():
    """Load Fashion MNIST dataset."""
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is required to load Fashion MNIST dataset")
    
    try:
        # Try different import patterns for different TensorFlow versions
        try:
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()  # type: ignore
        except AttributeError:
            # Fallback for older TensorFlow versions
            from tensorflow.keras.datasets import fashion_mnist  # type: ignore
            (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    except Exception as e:
        raise ImportError(f"Failed to load Fashion MNIST: {e}")
    
    # Combine train and test sets
    X = np.concatenate([x_train, x_test])
    y = np.concatenate([y_train, y_test])
    
    # Reshape to 2D array and convert to float
    X = X.reshape(X.shape[0], -1).astype(np.float32)
    
    target_names = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]
    feature_names = [f'pixel_{i}' for i in range(X.shape[1])]
    
    return DatasetWrapper(X, y, target_names, feature_names)


def load_mnist():
    """Load MNIST digits dataset."""
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is required to load MNIST dataset")
    
    try:
        # Try different import patterns for different TensorFlow versions
        try:
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()  # type: ignore
        except AttributeError:
            # Fallback for older TensorFlow versions
            from tensorflow.keras.datasets import mnist  # type: ignore
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
    except Exception as e:
        raise ImportError(f"Failed to load MNIST: {e}")
    
    # Combine train and test sets
    X = np.concatenate([x_train, x_test])
    y = np.concatenate([y_train, y_test])
    
    # Reshape to 2D array and convert to float
    X = X.reshape(X.shape[0], -1).astype(np.float32)
    
    target_names = [str(i) for i in range(10)]
    feature_names = [f'pixel_{i}' for i in range(X.shape[1])]
    
    return DatasetWrapper(X, y, target_names, feature_names)


def load_elephant():
    """Load elephant dataset from a sample image with variations."""
    url = "https://upload.wikimedia.org/wikipedia/commons/3/37/African_Bush_Elephant.jpg"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        
        # Convert to grayscale and resize
        img = img.convert('L')
        img = img.resize((28, 28))
        
        # Convert to numpy array
        base_img = np.array(img)
        
        # Create variations of the image
        n_samples = 10
        X = np.zeros((n_samples, 784))  # 28x28 = 784 pixels
        
        # Original image
        X[0] = base_img.reshape(-1)
        
        # Create variations with different transformations
        for i in range(1, n_samples):
            # Random rotation
            angle = np.random.uniform(-15, 15)
            rotated = img.rotate(angle)
            
            # Random brightness adjustment
            brightness = np.random.uniform(0.8, 1.2)
            adjusted = np.clip(np.array(rotated) * brightness, 0, 255).astype(np.uint8)
            
            # Add some noise
            noise = np.random.normal(0, 10, adjusted.shape).astype(np.uint8)
            noisy = np.clip(adjusted + noise, 0, 255).astype(np.uint8)
            X[i] = noisy.reshape(-1)
        
        y = np.zeros(n_samples)  # All samples are class 0 (elephant)
        
        return DatasetWrapper(X, y, ['Elephant'], [f'pixel_{i}' for i in range(X.shape[1])])
        
    except Exception as e:
        print(f"Error loading elephant image: {str(e)}")
        # Return a fallback dataset with random data
        n_samples = 10
        X = np.random.rand(n_samples, 784)
        y = np.zeros(n_samples)
        return DatasetWrapper(X, y, ['Elephant'], [f'pixel_{i}' for i in range(X.shape[1])])


def load_sklearn_dataset(loader_func):
    """Load a sklearn dataset using the provided loader function."""
    dataset = loader_func()
    return DatasetWrapper(dataset.data, dataset.target, dataset.target_names, dataset.feature_names)


def get_dataset(name):
    """Get dataset by name."""
    if name == 'custom_upload':
        # Return a placeholder dataset for custom upload
        X = np.random.rand(10, 4)
        y = np.zeros(10)
        return X, y, DatasetWrapper(X, y, ['Custom'], [f'Feature {i}' for i in range(X.shape[1])])
    
    # Dataset loader mapping
    loaders = {
        "Digits": lambda: load_sklearn_dataset(datasets.load_digits),
        "Iris": lambda: load_sklearn_dataset(datasets.load_iris),
        "Wine": lambda: load_sklearn_dataset(datasets.load_wine),
        "Breast Cancer": lambda: load_sklearn_dataset(datasets.load_breast_cancer),
        "Fashion MNIST": load_fashion_mnist,
        "MNIST": load_mnist,
        "Elephant": load_elephant,
    }
    
    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}")
    
    dataset = loaders[name]()
    return dataset.data, dataset.target, dataset 