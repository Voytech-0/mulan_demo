import os
import threading
from io import BytesIO
import json

import numpy as np
import glob

import requests
from PIL import Image
from sklearn import datasets

import tensorflow as tf
from sklearn.datasets import make_s_curve, make_swiss_roll

def load_PACS(domain='photo'):
    """Load PACS dataset."""
    data_path = os.path.join(os.path.dirname(__file__), '../../data/pacs_data')
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

# Testing datasets
# Testing datasets

def load_testing_s_curve(n_samples=1000, noise=0.00):
    """Load testing S-curve dataset."""
    X, y = make_s_curve(n_samples=n_samples, noise=noise, random_state=42)
    class Dataset:
        def __init__(self, data, target):
            self.data = data
            self.target = target
            self.target_names = []
            self.feature_names = [f'feature_{i}' for i in range(data.shape[1])]
    return Dataset(X, y)

def load_testing_swiss_roll(n_samples=1000, noise=0.00):
    """Load testing Swiss Roll dataset."""
    X, y = make_swiss_roll(n_samples=n_samples, noise=noise, random_state=42)
    class Dataset:
        def __init__(self, data, target):
            self.data = data
            self.target = target
            self.target_names = []
            self.feature_names = [f'feature_{i}' for i in range(data.shape[1])]
    return Dataset(X, y)

def load_testing_mammoth():
    # load from data/mammoth_umap.json
    data_path = os.path.join(os.path.dirname(__file__), '../../data/mammoth_umap.json')
    with open(data_path, 'r') as f:
        data = json.load(f)
    X = np.array(data['3d'], dtype=np.float32)
    y = np.array(data['labels'], dtype=np.int32)
    # Sample every 10th point
    X = X[::10]
    y = y[::10]
    class Dataset:
        def __init__(self, data, target):
            self.data = data
            self.target = target
            self.target_names = []
            self.feature_names = [f'feature_{i}' for i in range(data.shape[1])]
    return Dataset(X, y)

# List of available datasets
DATASET_LOADERS = {
    "Digits": datasets.load_digits,
    "Iris": datasets.load_iris,
    "Wine": datasets.load_wine,
    "Breast Cancer": datasets.load_breast_cancer,
    "Fashion MNIST": load_fashion_mnist,
    "MNIST": load_mnist,
    "PACS - Photo": lambda: load_PACS("photo"),
    "PACS - Sketch": lambda: load_PACS("sketch"),
    "PACS - Cartoon": lambda: load_PACS("cartoon"),
    "PACS - Art Painting": lambda: load_PACS("art_painting"),
    # Testing datasets
    "Testing - S-curve": load_testing_s_curve,
    "Testing - Swiss Roll": load_testing_swiss_roll,
    "Testing - Mammoth": load_testing_mammoth,
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
