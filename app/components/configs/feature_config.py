"""
Configuration file for dataset features to display.
Defines which features are shown in the metadata view for each dataset.
"""

# =============================================================================
# DATASET FEATURES
# =============================================================================

# Iris dataset features
IRIS_FEATURES = [
    'sepal length (cm)',
    'sepal width (cm)',
    'petal length (cm)',
    'petal width (cm)'
]

# Wine dataset features
WINE_FEATURES = [
    'alcohol',
    'malic acid',
    'ash',
    'alcalinity of ash',
    'magnesium',
    'total phenols',
    'flavanoids',
    'nonflavanoid phenols',
    'proanthocyanins',
    'color intensity',
    'hue',
    'OD280/OD315 of diluted wines',
    'proline'
]

# Breast Cancer dataset features
BREAST_CANCER_FEATURES = [
    'mean radius',
    'mean texture',
    'mean perimeter',
    'mean area',
    'mean smoothness',
    'mean compactness',
    'mean concavity',
    'mean concave points',
    'mean symmetry',
    'mean fractal dimension',
    'radius error',
    'texture error',
    'perimeter error',
    'area error',
    'smoothness error',
    'compactness error',
    'concavity error',
    'concave points error',
    'symmetry error',
    'fractal dimension error',
    'worst radius',
    'worst texture',
    'worst perimeter',
    'worst area',
    'worst smoothness',
    'worst compactness',
    'worst concavity',
    'worst concave points',
    'worst symmetry',
    'worst fractal dimension'
]

# Image datasets (no meaningful metadata to display)
DIGITS_FEATURES = []
MNIST_FEATURES = []
FASHION_MNIST_FEATURES = []
ELEPHANT_FEATURES = []

# =============================================================================
# DATASET DESCRIPTIONS
# =============================================================================

DATASET_DESCRIPTIONS = {
    "Digits": "The Digits dataset consists of 8x8 images of handwritten digits (0-9) from the scikit-learn library. It is commonly used for classification and image recognition tasks.",
    "Iris": "The Iris dataset is a classic multivariate dataset introduced by Ronald Fisher. It contains measurements of 150 iris flowers from three different species.",
    "Wine": "The Wine dataset contains the results of a chemical analysis of wines grown in the same region in Italy but derived from three different cultivars.",
    "Breast Cancer": "The Breast Cancer Wisconsin dataset contains features computed from digitized images of fine needle aspirate (FNA) of breast masses, used for binary classification (malignant/benign).",
    "MNIST": "The MNIST dataset consists of 28x28 grayscale images of handwritten digits (0-9). It is a benchmark dataset for image classification.",
    "Fashion MNIST": "Fashion MNIST is a dataset of 28x28 grayscale images of 10 fashion categories, such as shoes, shirts, and bags, intended as a drop-in replacement for the original MNIST dataset.",
    "Elephant": "The Elephant dataset is a collection of 28x28 grayscale images of elephants, used for image classification tasks.",
    "PACS - Photo": "PACS Photo is a domain from the PACS dataset, containing real-world photographs of objects from seven categories (dog, elephant, giraffe, guitar, horse, house, person).",
    "PACS - Sketch": "PACS Sketch is a domain from the PACS dataset, containing hand-drawn sketches of objects from seven categories.",
    "PACS - Cartoon": "PACS Cartoon is a domain from the PACS dataset, containing cartoon-style images of objects from seven categories.",
    "PACS - Art Painting": "PACS Art Painting is a domain from the PACS dataset, containing artistic paintings of objects from seven categories.",
    "Testing - S-curve": "A synthetic 3D S-curve dataset, commonly used for testing dimensionality reduction algorithms.",
    "Testing - Swiss Roll": "A synthetic 3D Swiss Roll dataset, commonly used for testing manifold learning and dimensionality reduction.",
    "Testing - Mammoth": "A synthetic 3D dataset shaped like a mammoth, used for visualization and testing.",
    "custom_upload": "A custom dataset uploaded by the user. The structure and content depend on the uploaded data."
}

# =============================================================================
# DATASET CONFIGURATIONS
# =============================================================================

# Dictionary mapping dataset names to their feature lists
DATASET_FEATURES = {
    "Digits": DIGITS_FEATURES,
    "Iris": IRIS_FEATURES,
    "Wine": WINE_FEATURES,
    "Breast Cancer": BREAST_CANCER_FEATURES,
    "MNIST": MNIST_FEATURES,
    "Fashion MNIST": FASHION_MNIST_FEATURES,
    "Elephant": ELEPHANT_FEATURES
}

# List of datasets that are purely image data (no meaningful metadata to display)
IMAGE_ONLY_DATASETS = ["Digits", "MNIST", "Fashion MNIST", "Elephant"] 