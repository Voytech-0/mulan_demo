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