import base64
from io import BytesIO

from components.data_operations.dataset_api import dataset_shape
import matplotlib.pyplot as plt
import numpy as np

import plotly.express as px
import pandas as pd

def match_shape(data_point, dataset_name):
    img_shape = dataset_shape(dataset_name)
    data_point = data_point.reshape(img_shape)
    return data_point

def encode_img_as_str(data_point, fig_size=(4, 4)):
    plt.figure(figsize=fig_size)
    plt.imshow(data_point, cmap='gray')
    plt.axis('off')
    # Convert to base64
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    return img_str

def create_main_fig_dataframe(embedding, X, y, class_names, n_added):
    is_continuous = len(np.unique(y)) > 20
    # Create a list of customdata for each point, including the point index
    point_indices = np.arange(len(y))

    # If class_names is not provided, use unique values in y as strings
    if class_names is None:
        unique_classes = np.unique(y)
        class_names = [str(c) for c in unique_classes]

    # Map y to class names for legend
    y_int = y.astype(int)

    color_seq = px.colors.qualitative.Plotly

    n_original = len(y) - n_added
    sections = [slice(0, n_original)]

    if n_added > 0 and embedding.shape[0] == X.shape[0]:
        sections.append(slice(n_original, None))

    if not is_continuous:
        y_labels = [str(class_names[i]) if 0 <= i < len(class_names) else str(i) for i in y_int]
        unique_labels = pd.Series(y_labels).unique()
        color_map = {label: color_seq[i % len(color_seq)] for i, label in enumerate(sorted(unique_labels))}
    else:
        y_labels = y
        color_map = None

    data_frames = []
    for section in sections:
        df = pd.DataFrame({
            'x': embedding[section, 0],
            'y': embedding[section, 1],
            'point_index': point_indices[section],
            'label': y_labels[section],
            'color': y_labels[section],
        })
        if color_map is not None:
            df['color'] = df['color'].map(color_map)
        data_frames.append(df)
    return data_frames