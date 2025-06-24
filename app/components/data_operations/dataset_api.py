import os
import threading
from io import BytesIO

import numpy as np
import glob

import requests
from PIL import Image
from sklearn import datasets

import tensorflow as tf


def load_PACS(domain='photo'):
    """Load PACS dataset."""
    data_path = '../../data/pacs_data'
    if not os.path.exists(data_path):
        raise Exception(f'PACS dataset not found in folder {data_path}.')

    base_path = os.path.join(data_path, domain)
    classes = sorted(os.listdir(base_path))
    images = []
    labels = []

    for idx, class_name in enumerate(classes):
        class_dir = os.path.join(base_path, class_name)
        for ext in ('*.jpg', '*.png'):
            for img_path in glob.glob(os.path.join(class_dir, ext)):
                img = Image.open(img_path).convert("L").resize((28,28))
                images.append(img)
                labels.append(idx)

    X = np.stack(images)
    X = X.reshape(X.shape[0], -1).astype(np.float32)
    y = np.array(labels)

    class Dataset:
        def __init__(self, data, target):
            self.data = data
            self.target = target
            self.target_names = classes
            self.feature_names = [f"pixel_{i}" for i in range(self.data.shape[1])]

    return Dataset(X, y)

def load_fashion_mnist():
    """Load Fashion MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    # Combine train and test sets
    X = np.concatenate([x_train, x_test])
    y = np.concatenate([y_train, y_test])
    # Reshape to 2D array and convert to float
    X = X.reshape(X.shape[0], -1).astype(np.float32)
    # Create a dataset-like object
    class Dataset:
        def __init__(self, data, target):
            self.data = data
            self.target = target
            self.target_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                                 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
            # Add feature names for Fashion MNIST (pixel coordinates)
            self.feature_names = [f'pixel_{i}' for i in range(data.shape[1])]
    return Dataset(X, y)

def load_mnist():
    """Load MNIST digits dataset."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Combine train and test sets
    X = np.concatenate([x_train, x_test])
    y = np.concatenate([y_train, y_test])
    # Reshape to 2D array and convert to float
    X = X.reshape(X.shape[0], -1).astype(np.float32)
    # Create a dataset-like object
    class Dataset:
        def __init__(self, data, target):
            self.data = data
            self.target = target
            self.target_names = [str(i) for i in range(10)]
            # Add feature names for MNIST (pixel coordinates)
            self.feature_names = [f'pixel_{i}' for i in range(data.shape[1])]
    return Dataset(X, y)

def load_elephant():
    """Load elephant dataset from a sample image with variations."""
    # URL of a sample elephant image from a reliable source
    url = "https://upload.wikimedia.org/wikipedia/commons/3/37/African_Bush_Elephant.jpg"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        img = Image.open(BytesIO(response.content))

        # Convert to grayscale and resize
        img = img.convert('L')
        img = img.resize((28, 28))

        # Convert to numpy array
        base_img = np.array(img)

        # Create variations of the image
        n_samples = 10  # Create 10 variations
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

        # Create a dataset-like object
        class Dataset:
            def __init__(self, data, target):
                self.data = data
                self.target = target
                self.target_names = ['Elephant']
                # Add feature names for Elephant (pixel coordinates)
                self.feature_names = [f'pixel_{i}' for i in range(data.shape[1])]
        return Dataset(X, y)
    except Exception as e:
        print(f"Error loading elephant image: {str(e)}")
        # Return a fallback dataset with multiple samples if image loading fails
        n_samples = 10
        X = np.random.rand(n_samples, 784)  # Random data
        y = np.zeros(n_samples)
        class Dataset:
            def __init__(self, data, target):
                self.data = data
                self.target = target
                self.target_names = ['Elephant']
                self.feature_names = [f'pixel_{i}' for i in range(data.shape[1])]
        return Dataset(X, y)

# List of available datasets
DATASET_LOADERS = {
    "Digits": datasets.load_digits,
    "Iris": datasets.load_iris,
    "Wine": datasets.load_wine,
    "Breast Cancer": datasets.load_breast_cancer,
    "Fashion MNIST": load_fashion_mnist,
    "MNIST": load_mnist,
    "Elephant": load_elephant,
    "PACS - Photo": lambda: load_PACS("photo"),
    "PACS - Sketch": lambda: load_PACS("sketch"),
    "PACS - Cartoon": lambda: load_PACS("cartoon"),
    "PACS - Art Painting": lambda: load_PACS("art_painting"),
}

# Single global lock for computations
numba_global_lock = threading.Lock()

def get_dataset(name):
    with numba_global_lock:
        if name == 'custom_upload':
            # Return a placeholder dataset for custom upload
            # This will be handled by the upload functionality later
            X = np.random.rand(10, 4)  # Placeholder data
            y = np.zeros(10)  # Placeholder labels
            class Dataset:
                def __init__(self, data, target):
                    self.data = data
                    self.target = target
                    self.target_names = ['Custom']
                    self.feature_names = [f'Feature {i}' for i in range(data.shape[1])]
            data = Dataset(X, y)
        else:
            loader = DATASET_LOADERS[name]
            data = loader()
            X = data.data
            y = data.target
    return X, y, data


def dataset_shape(dataset_name):
    if dataset_name == "Digits":
        img_shape = (8, 8)
    else:  # MNIST or Fashion MNIST
        img_shape = (28, 28)
    return img_shape

