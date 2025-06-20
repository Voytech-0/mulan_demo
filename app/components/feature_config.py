"""
Configuration file for dataset features to display.
Comment out features you don't want to display in the metadata view.
"""

# Features to display for each dataset
DIGITS_FEATURES = [
    # 'pixel_0_0', 'pixel_0_1', 'pixel_0_2', 'pixel_0_3', 'pixel_0_4', 'pixel_0_5', 'pixel_0_6', 'pixel_0_7',
    # 'pixel_1_0', 'pixel_1_1', 'pixel_1_2', 'pixel_1_3', 'pixel_1_4', 'pixel_1_5', 'pixel_1_6', 'pixel_1_7',
    # 'pixel_2_0', 'pixel_2_1', 'pixel_2_2', 'pixel_2_3', 'pixel_2_4', 'pixel_2_5', 'pixel_2_6', 'pixel_2_7',
    # 'pixel_3_0', 'pixel_3_1', 'pixel_3_2', 'pixel_3_3', 'pixel_3_4', 'pixel_3_5', 'pixel_3_6', 'pixel_3_7',
    # 'pixel_4_0', 'pixel_4_1', 'pixel_4_2', 'pixel_4_3', 'pixel_4_4', 'pixel_4_5', 'pixel_4_6', 'pixel_4_7',
    # 'pixel_5_0', 'pixel_5_1', 'pixel_5_2', 'pixel_5_3', 'pixel_5_4', 'pixel_5_5', 'pixel_5_6', 'pixel_5_7',
    # 'pixel_6_0', 'pixel_6_1', 'pixel_6_2', 'pixel_6_3', 'pixel_6_4', 'pixel_6_5', 'pixel_6_6', 'pixel_6_7',
    # 'pixel_7_0', 'pixel_7_1', 'pixel_7_2', 'pixel_7_3', 'pixel_7_4', 'pixel_7_5', 'pixel_7_6', 'pixel_7_7'
]

IRIS_FEATURES = [
    'sepal length (cm)',
    'sepal width (cm)',
    'petal length (cm)',
    'petal width (cm)'
]

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

# MNIST features (784 pixel values) - empty list since it's purely image data
MNIST_FEATURES = [
    # MNIST is purely image data, so we don't display individual pixel features
    # 'pixel_0_0', 'pixel_0_1', ..., 'pixel_27_27' (784 total pixels)
]

# Fashion MNIST features (784 pixel values) - empty list since it's purely image data
FASHION_MNIST_FEATURES = [
    # Fashion MNIST is purely image data, so we don't display individual pixel features
    # 'pixel_0_0', 'pixel_0_1', ..., 'pixel_27_27' (784 total pixels)
]

# Elephant features (pixel values from the elephant image) - empty list since it's purely image data
ELEPHANT_FEATURES = [
    # Elephant dataset is purely image data, so we don't display individual pixel features
    # 'pixel_0_0', 'pixel_0_1', ..., 'pixel_n_m' (depends on image size)
]

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
IMAGE_ONLY_DATASETS = ["Digits", "MNIST", "Fashion MNIST", "Elephant", "PACS - Photo", "PACS - Sketch", "PACS - Cartoon", "PACS - Art Painting"] 