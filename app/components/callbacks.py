import json

import jax.random as random
from sklearn import datasets
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, callback_context, State, callback
import numpy as np
import threading
import matplotlib
import pandas as pd
import glob

from .feature_config import IMAGE_ONLY_DATASETS, DATASET_FEATURES
from .plot_maker import add_new_data_to_fig
from .settings import get_image_style, get_no_image_message_style, get_generative_placeholder_style, CELL_STYLE, \
    CELL_STYLE_RIGHT, TABLE_STYLE, EMPTY_METADATA_STYLE

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import time
import io
import base64
from PIL import Image
import tensorflow as tf
import os
import requests
from io import BytesIO
from .embedding_storage import save_embedding, load_embedding, embedding_exists
from dash_canvas.utils import parse_jsonstring
from math import ceil, floor
import cv2


from .features import dynamically_add, generate_sample, dataset_shape
from .plot_maker import encode_img_as_str, invisible_interactable_layer
# Import TRIMAP from local package
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'google_research_trimap'))
from google_research_trimap.trimap import trimap

# Try to import UMAP, if available
_image_cache = {}
umap_available = False
umap_lib = None # Use a distinct name for the imported module
try:
    # import umap as umap_lib # Import as umap_lib
    import umap.umap_ as umap_lib
    umap_available = True
except ImportError:
    pass # umap_lib remains None

def load_PACS(domain='photo'):
    """Load PACS dataset."""
    base_path = os.path.join(os.path.dirname(__file__), "pacs_data", domain)
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


def compute_trimap(X, distance):
    key = random.PRNGKey(0)
    start_time = time.time()
    with numba_global_lock:
        print('Starting TRIMAP calculation...')

        # Adjust n_inliers based on dataset size
        n_points = X.shape[0]
        n_inliers = min(5, n_points - 2)  # Use at most 5 inliers, but ensure it's less than n_points-1

        # Time the nearest neighbor search
        nn_start = time.time()
        print('Finding nearest neighbors...')
        emb = trimap.transform(key, X, n_inliers=n_inliers, output_metric=distance, auto_diff=False, export_iters=True)
        nn_time = time.time() - nn_start
        print(f'Nearest neighbor search took: {nn_time:.2f} seconds')

        result = np.array(emb) if hasattr(emb, "shape") else emb
        total_time = time.time() - start_time
    if distance == 'haversine':
        x = np.arctan2(np.sin(result[:, :, 0]) * np.cos(result[:, :, 1]), np.sin(result[:, :, 0]) * np.sin(result[:, :, 1]))
        y = -np.arccos(np.cos(result[:, :, 0]))
        result = np.stack([x, y], axis=-1)
        # x = np.arctan2(np.sin(result[:, 0]) * np.cos(result[:, 1]), np.sin(result[:, 0]) * np.sin(result[:, 1]))
        # y = -np.arccos(np.cos(result[:, 0]))
        # result = np.column_stack((x, y))
    return result, total_time


def compute_tsne(X, distance):
    if distance == 'haversine':
        print("t-SNE is not supported for haversine distance. Returning None.")
        return None, 0
    start_time = time.time()
    with numba_global_lock:
        print('calculating tsne')
        emb = TSNE(n_components=2, random_state=42, metric=distance).fit_transform(X)
        result = np.array(emb)
        print('tsne calculated')
    return result, time.time() - start_time


def compute_umap(X, distance):
    start_time = time.time()
    with numba_global_lock:
        print('calculating umap')
        reducer = umap_lib.UMAP(n_components=2, random_state=42, output_metric=distance)
        emb = reducer.fit_transform(X)
        result = np.array(emb)
        print('umap calculated')
    if distance == 'haversine':
        x = np.arctan2(np.sin(result[:, 0]) * np.cos(result[:, 1]), np.sin(result[:, 0]) * np.sin(result[:, 1]))
        y = -np.arccos(np.cos(result[:, 0]))
        result = np.column_stack((x, y))
    return result, time.time() - start_time

def create_datapoint_image(data_point, size=(20, 20)):
    """Create a small image representation of a datapoint."""
    # Create a cache key based on the data point and size
    cache_key = (hash(data_point.tobytes()), size)

    # Check if we have this image cached
    if cache_key in _image_cache:
        return _image_cache[cache_key]

    # Normalize the data point to 0-1 range
    normalized = (data_point - data_point.min()) / (data_point.max() - data_point.min())

    # Reshape if needed (assuming square image)
    side_length = int(np.sqrt(len(normalized)))
    if side_length * side_length == len(normalized):
        img_data = normalized.reshape(side_length, side_length)
        # Use grayscale colormap for image datasets
        cmap = 'gray'
        # Adjust figure size based on the actual image dimensions
        if side_length == 8:  # Digits dataset
            fig_size = (size[0]/50, size[1]/50)  # Smaller for 8x8
        else:  # MNIST/Fashion MNIST/Elephant (28x28)
            fig_size = (size[0]/25, size[1]/25)  # Larger for 28x28
    else:
        # If not a perfect square, create a rectangular image
        img_data = normalized.reshape(-1, 8)  # Arbitrary width of 8
        # Use viridis for non-image datasets
        cmap = 'viridis'
        fig_size = (size[0]/100, size[1]/100)

    # Create the image with higher DPI for sharper pixels
    plt.figure(figsize=fig_size, dpi=300)  # Increased DPI from 100 to 300
    plt.imshow(img_data, cmap=cmap, interpolation='nearest')  # Use nearest neighbor interpolation
    plt.axis('off')

    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    img_data_url = f"data:image/png;base64,{img_str}"

    # Cache the result
    _image_cache[cache_key] = img_data_url

    # Limit cache size to prevent memory issues
    if len(_image_cache) > 1000:
        # Remove oldest entries (simple FIFO)
        oldest_keys = list(_image_cache.keys())[:100]
        for key in oldest_keys:
            del _image_cache[key]

    return img_data_url

def create_animated_figure(embedding, y, title, label_name):
    n_frames = min(400, len(embedding))
    frames = []
    # Use y for all frames (assume y is static)
    point_indices = np.arange(len(y))
    # Initial frame
    df0 = pd.DataFrame({
        'x': embedding[0][:, 0],
        'y': embedding[0][:, 1],
        'color': y.astype(str),
        'color_num': y.astype(int) if np.issubdtype(y.dtype, np.integer) else pd.factorize(y)[0],
        'point_index': point_indices,
        'label': y.astype(str)
    })
    # Use a consistent color palette for both px.scatter and go.Scatter
    color_palette = px.colors.qualitative.Light24

    # Compute global min/max for all frames for fixed axes
    all_x = np.concatenate([emb[:, 0] for emb in embedding[:n_frames]])
    all_y = np.concatenate([emb[:, 1] for emb in embedding[:n_frames]])
    x_range = [float(all_x.min()), float(all_x.max())]
    y_range = [float(all_y.min()), float(all_y.max())]

    fig = px.scatter(
        df0,
        x='x',
        y='y',
        color='color',
        custom_data=['point_index', 'label'],
        title=title,
        labels={'x': 'Component 1', 'y': 'Component 2', 'color': label_name},
        color_discrete_sequence=color_palette,
        range_x=x_range,
        range_y=y_range
    )
    # Build frames
    for i in range(n_frames):
        dfi = pd.DataFrame({
            'x': embedding[i][:, 0],
            'y': embedding[i][:, 1],
            'color': y.astype(str),
            'color_num': y.astype(int) if np.issubdtype(y.dtype, np.integer) else pd.factorize(y)[0],
            'point_index': point_indices,
            'label': y.astype(str)
        })
        scatter = go.Scatter(
            x=dfi['x'],
            y=dfi['y'],
            mode='markers',
            marker=dict(
                color=dfi['color_num'],
                colorscale=color_palette,
                cmin=0,
                cmax=len(color_palette) - 1
            ),
            customdata=np.stack([dfi['point_index'], dfi['label']], axis=-1),
            showlegend=False,
            hovertemplate="Class: %{customdata[1]}<br>Index: %{customdata[0]}<br>X: %{x}<br>Y: %{y}<extra></extra>"
        )
        frames.append(go.Frame(data=[scatter], name=str(i)))
    fig.frames = frames

    # Add animation slider
    sliders = [{
        "steps": [
            {
                "args": [
                    [str(k)],
                    {
                        "frame": {"duration": 0, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 0}
                    }
                ],
                "label": str(k),
                "method": "animate"
            } for k in range(n_frames)
        ],
        "transition": {"duration": 0},
        "x": 0.1,
        "y": -0.15,
        "currentvalue": {"font": {"size": 14}, "prefix": "Frame: ", "visible": True, "xanchor": "center"},
        "len": 0.9
    }]
    fig.update_layout(
        updatemenus=[{
            "type": "buttons",
            "buttons": [
                {
                    "args": [
                        None,
                        {
                            "frame": {"duration": 50, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 0}
                        }
                    ],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }
                    ],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 70},
            "showactive": False,
            "x": 0.1,
            "y": -0.15,
            "xanchor": "right",
            "yanchor": "top"
        }],
        sliders=sliders,
        margin=dict(l=5, r=5, t=50, b=5)
    )
    return fig


def create_figure(embedding, y, title, label_name, X=None, is_thumbnail=False, show_images=False, class_names=None, n_added=0):
    if embedding is None or len(embedding) == 0 or embedding.shape[-1] < 2:
        return px.scatter(title=f"{title} (no data)")

    if 'trimap' in title.lower():
        if show_images or is_thumbnail or n_added > 0:
            embedding = embedding[-1]
            print("Using last frame of TRIMAP embedding for visualization")

    # Create a list of customdata for each point, including the point index
    point_indices = np.arange(len(y))

    # If class_names is not provided, use unique values in y as strings
    if class_names is None:
        unique_classes = np.unique(y)
        class_names = [str(c) for c in unique_classes]

    # Map y to class names for legend
    y_int = y.astype(int)
    y_labels = [str(class_names[i]) if 0 <= i < len(class_names) else str(i) for i in y_int]

    unique_labels = pd.Series(y_labels).unique()
    color_seq = px.colors.qualitative.Plotly
    color_map = {label: color_seq[i % len(color_seq)] for i, label in enumerate(sorted(unique_labels))}
    n_original = len(y) - n_added
    sections = [slice(0, n_original)]

    visualize_added_samples = n_added > 0 and embedding.shape[0] == X.shape[0]
    if visualize_added_samples:
        sections.append(slice(n_original, None))

    data_frames = []
    for section in sections:
        df = pd.DataFrame({
            'x': embedding[section, 0],
            'y': embedding[section, 1],
            'point_index': point_indices[section],
            'label': y_labels[section]
        })
        df['color'] = df['label'].map(color_map)
        data_frames.append(df)

    # Set category order for consistent color mapping
    category_orders = {'color': class_names}

    # Check if we should show images and if we have image data
    df = data_frames[0]
    if show_images and X is not None and len(X) > 0:
        if len(X[0]) in [64, 784]:
            max_images = 100
            if len(X) > max_images:
                step = len(X) // max_images
                indices_to_show = list(range(0, len(X), step))[:max_images]
                print(f"Showing {len(indices_to_show)} images out of {len(X)} total points ({len(indices_to_show)/len(X)*100:.1f}%)")
            else:
                indices_to_show = list(range(len(X)))
                print(f"Showing all {len(indices_to_show)} images")
            images = []
            for i in indices_to_show:
                img_str = create_datapoint_image(X[i], size=(15, 15))
                images.append(img_str)
            fig = px.scatter(
                df,
                x='x',
                y='y',
                color='label',
                custom_data=['point_index', 'label'],
                title=title,
                labels={'x': 'Component 1', 'y': 'Component 2', 'label': 'Class'},
                category_orders=category_orders
            )
            fig.update_traces(
                marker=dict(
                    size=15,
                    sizeref=1,
                    sizemin=5,
                    sizemode='diameter'
                ),
                selector=dict(type='scatter')
            )
            for i, (idx, img_str) in enumerate(zip(indices_to_show, images)):
                x, y = df.iloc[idx]['x'], df.iloc[idx]['y']
                x_range = df['x'].max() - df['x'].min()
                y_range = df['y'].max() - df['y'].min()
                base_size = max(x_range, y_range) * 0.04
                fig.add_layout_image(
                    dict(
                        source=img_str,
                        xref="x",
                        yref="y",
                        x=x,
                        y=y,
                        sizex=base_size,
                        sizey=base_size,
                        xanchor="center",
                        yanchor="middle"
                    )
                )
        else:
            fig = px.scatter(
                df,
                x='x',
                y='y',
                color='label',
                custom_data=['point_index', 'label'],
                title=title,
                labels={'x': 'Component 1', 'y': 'Component 2', 'label': 'Class'},
                category_orders=category_orders
            )
    else:
        fig = px.scatter(
            df,
            x='x',
            y='y',
            color='label',
            custom_data=['point_index', 'label'],
            title=title,
            labels={'x': 'Component 1', 'y': 'Component 2', 'label': 'Class'},
            category_orders=category_orders
        )


    # Additional points
    if visualize_added_samples:
        add_new_data_to_fig(fig, data_frames[1], color_map)

    if is_thumbnail:
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showticklabels=False, visible=False),
            yaxis=dict(showticklabels=False, visible=False),
            showlegend=False,
            hovermode=False
        )
    else:
        df = data_frames[0]
        fig.add_trace(invisible_interactable_layer(df['x'].min(), df['x'].max(), df['y'].min(), df['y'].max()))
        fig.update_layout(
            margin=dict(l=5, r=5, t=50, b=5)
        )
    return fig

def create_metadata_display(dataset_name, data):
    # This function is now mostly for the dataset-level metadata, not point-level
    return html.Div([
        html.H4(f"Dataset: {dataset_name}"),
        # information abou the dataset
        # html.P(f"{}
        html.P(f"Number of samples: {data.data.shape[0]}"),
        html.P(f"Number of features: {data.data.shape[1]}"),
        html.P(f"Number of classes: {len(np.unique(data.target))}")
    ])

def register_callbacks(app):
    @app.callback(
        Output('main-graph-static', 'figure'),
        Output('main-graph-static', 'style'),
        Output('main-graph-animated', 'figure'),
        Output('main-graph-animated', 'style'),
        Output('trimap-thumbnail', 'figure'),
        Output('tsne-thumbnail', 'figure'),
        Output('umap-thumbnail', 'figure'),
        Output('umap-warning', 'children'),
        Output('metadata-display', 'children'),
        Output('embedding-cache', 'data'),
        Output('trimap-timing', 'children'),
        Output('tsne-timing', 'children'),
        Output('umap-timing', 'children'),
        Output('calculation-status', 'children'),
        Input('dataset-dropdown', 'value'),
        Input('recalculate-switch', 'value'),
        Input('dots-images-switch', 'value'),
        Input('trimap-thumbnail-click', 'n_clicks'),
        Input('tsne-thumbnail-click', 'n_clicks'),
        Input('umap-thumbnail-click', 'n_clicks'),
        State('embedding-cache', 'data'),
        State('added-data-cache', 'data'),
        Input('dist-dropdown', 'value')
    )
    def update_graphs(dataset_name, recalculate_flag, show_images, trimap_n_clicks, tsne_n_clicks, umap_n_clicks, cached_embeddings,
                      added_data_cache, distance):
        if not dataset_name:
            empty_fig = px.scatter(title="No dataset selected")
            return (
                empty_fig, {'display': 'block'},
                {}, {'display': 'none'},
                empty_fig, empty_fig, empty_fig,
                "", "", {}, "", "", ""
            )

        # Handle None value for show_images (default to False)
        if show_images is None:
            show_images = False

        # Get the dataset
        X, y, data = get_dataset(dataset_name)

        # Initialize embeddings dictionary if not exists
        if cached_embeddings is None:
            cached_embeddings = {}

        # Determine which method was clicked
        ctx = callback_context
        if not ctx.triggered:
            method = 'trimap'  # Default method
        else:
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if trigger_id == 'trimap-thumbnail-click':
                method = 'trimap'
            elif trigger_id == 'tsne-thumbnail-click':
                method = 'tsne'
            elif trigger_id == 'umap-thumbnail-click':
                method = 'umap'
            else:
                method = 'trimap'  # Default method

        # Initialize timing variables
        trimap_time = 0
        tsne_time = 0
        umap_time = 0

        # Function to compute or load embeddings
        def get_embedding(method_name, compute_func, *args):
            nonlocal trimap_time, tsne_time, umap_time

            # Check if we should use saved embeddings
            if not recalculate_flag and embedding_exists(dataset_name, method_name, distance):
                embedding, metadata = load_embedding(dataset_name, method_name, distance)
                if embedding is not None:
                    # Update timing from metadata if available
                    if metadata and 'time' in metadata:
                        if method_name == 'trimap':
                            trimap_time = metadata['time']
                        elif method_name == 'tsne':
                            tsne_time = metadata['time']
                        elif method_name == 'umap':
                            umap_time = metadata['time']
                    print(f"Using saved {method_name} embedding")
                    return embedding

            # Only compute new embedding if recalculate is True or no saved embedding exists
            if recalculate_flag or not embedding_exists(dataset_name, method_name, distance):
                print(f"Computing new {method_name} embedding")
                embedding, compute_time = compute_func(*args)

                # Save the embedding if computation was successful
                if embedding is not None:
                    metadata = {'time': compute_time}
                    save_embedding(dataset_name, method_name, embedding, distance, metadata)

                    # Update timing
                    if method_name == 'trimap':
                        trimap_time = compute_time
                    elif method_name == 'tsne':
                        tsne_time = compute_time
                    elif method_name == 'umap':
                        umap_time = compute_time

                return embedding

            return None  # Return None if no embedding is available and we're not computing

        # Get embeddings for all methods
        trimap_emb = get_embedding('trimap', compute_trimap, X, distance)
        tsne_emb = get_embedding('tsne', compute_tsne, X, distance)
        umap_emb = get_embedding('umap', compute_umap, X, distance)

        # Get class names for legend
        class_names = getattr(data, 'target_names', None)

        n_added = 0
        for source in ['user_generated', 'augmented']:
            if source in added_data_cache: # cache not empty
                if source == 'user_generated':
                    X_add = np.array(added_data_cache['user_generated'])
                    y_add = np.zeros(X_add.shape[0]) - 1
                else:
                    X_add, y_add = added_data_cache['augmented']
                n_added += len(X_add)
                X = np.concatenate((X, X_add), axis=0)
                y = np.concatenate((y, y_add), axis=0)
                trimap_emb = dynamically_add(trimap_emb, X)

        is_animated = None
        # return create_animated_figure(embedding, y, title, label_name)
        if method != 'trimap' or n_added!=0 or  show_images:
            is_animated = False
            # Create static figures
            main_fig_static = create_figure(
                trimap_emb if method == 'trimap' else tsne_emb if method == 'tsne' else umap_emb,
                y,
                f"{method.upper()} Embedding of {dataset_name}",
                "Class",
                X,
                show_images=show_images,
                class_names=class_names,
                n_added=n_added
            )
            main_fig_animated = {}
        else:
            is_animated = True
            main_fig_static = {}
            main_fig_animated = create_animated_figure(trimap_emb, y, f"TRIMAP Embedding of {dataset_name}", 'Class')

        trimap_fig = create_figure(trimap_emb, y, "TRIMAP", "Class", X, is_thumbnail=True, show_images=False, class_names=class_names, n_added=n_added)
        tsne_fig = create_figure(tsne_emb, y, "t-SNE", "Class", X, is_thumbnail=True, show_images=False, class_names=class_names, n_added=n_added)
        umap_fig = create_figure(umap_emb, y, "UMAP", "Class", X, is_thumbnail=True, show_images=False, class_names=class_names, n_added=n_added)

        # UMAP warning
        umap_warning = "" if umap_available else "UMAP is not available. Please install it using: pip install umap-learn"

        # Metadata display
        metadata = create_metadata_display(dataset_name, data)

        # Update cache only if we have new embeddings
        if recalculate_flag:
            cached_embeddings[dataset_name] = {
                'trimap': trimap_emb.tolist() if trimap_emb is not None else None,
                'tsne': tsne_emb.tolist() if tsne_emb is not None else None,
                'umap': umap_emb.tolist() if umap_emb is not None else None
            }

        # Show/hide graphs based on is_animated
        if is_animated:
            static_style = {'display': 'none'}
            animated_style = {'display': 'block'}
        else:
            static_style = {'display': 'block'}
            animated_style = {'display': 'none'}

        ctx = callback_context
        calc_status = ""
        if not ctx.triggered:
            calc_status = f"Loaded dataset {dataset_name}"  # Initial state, no message

        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        # For dataset loading
        if trigger_id == 'dataset-dropdown':
            calc_status = f"Loaded dataset {dataset_name}"

        # For recalculation
        elif trigger_id == 'recalculate-switch' and recalculate_flag:
            calc_status = f"Recalculated embeddings for {dataset_name}"

        # For thumbnail clicks
        elif trigger_id in ['trimap-thumbnail-click', 'tsne-thumbnail-click', 'umap-thumbnail-click']:
            method_name = trigger_id.replace('-thumbnail-click', '').upper()
            calc_status = f"Calculated {method_name} embedding"

        return (
            main_fig_static,
            static_style,
            main_fig_animated,
            animated_style,
            trimap_fig,
            tsne_fig,
            umap_fig,
            umap_warning,
            metadata,
            cached_embeddings,
            f"TRIMAP: {trimap_time:.2f}s",
            f"t-SNE: {tsne_time:.2f}s",
            f"UMAP: {umap_time:.2f}s",
            calc_status
        )

    @app.callback(
        Output('selected-image', 'src'),
        Output('selected-image', 'style'),
        Output('no-image-message', 'style'),
        Output('coordinates-display', 'children'),
        Output('point-metadata', 'children'),
        Output('click-message', 'style'),
        Output('generative-mode-placeholder', 'style'),
        Input('main-graph-static', 'clickData'),
        Input('generative-mode-state', 'data'),
        State('dataset-dropdown', 'value'),
        State('main-graph-static', 'figure'),
        State('embedding-cache', 'data')
    )
    def display_clicked_point(clickData, generative_state, dataset_name, figure, embedding_cache):
        enabled = generative_state.get('enabled', False) if generative_state else False

        if not clickData:
            # Return default states when nothing is clicked
            return (
                '',  # selected-image src
                get_image_style('none'),  # selected-image style
                get_no_image_message_style('block'),  # no-image-message style
                "",  # coordinates-display children
                "",  # point-metadata children
                {'display': 'block', 'text-align': 'center', 'padding': '0.5rem', 'margin-bottom': '0.5rem',
                 'color': '#666'},  # click-message style
                get_generative_placeholder_style('none')  # generative-mode-placeholder style
            )

        if enabled:
            x_coord = clickData['points'][0]['x']
            y_coord = clickData['points'][0]['y']
            X, _, _ = get_dataset(dataset_name)
            embedding, _ = load_embedding(dataset_name, 'trimap')
            sample = generate_sample(x_coord, y_coord, X, embedding)
            img_str = encode_img_as_str(sample, dataset_name)
            # In generative mode, show placeholder and hide other image elements
            return (
                f'data:image/png;base64,{img_str}',
                get_image_style('none'), # selected-image style
                get_no_image_message_style('none'), # no-image-message style
                "", # coordinates-display children
                "", # point-metadata children
                {'display': 'none'}, # click-message style
                get_generative_placeholder_style('block') # generative-mode-placeholder style
            )

        # Defensive: check for 'customdata' in clickData['points'][0]
        point_data = clickData['points'][0]
        if 'customdata' in point_data:
            point_index = int(point_data['customdata'][0])
            digit_label = point_data['customdata'][1]
        else:
            # Fallback: try to use pointNumber or index
            point_index = point_data.get('pointIndex', 0)
            # Try to get the class label from the color/category
            digit_label = point_data.get('curveNumber', '')
            # If color/class name is present in 'text' or 'label', use it
            digit_label = point_data.get('label', point_data.get('text', str(point_index)))
        X, y, data = get_dataset(dataset_name)

        # Get features to display from configuration or use default feature names
        features_to_display = DATASET_FEATURES.get(dataset_name, getattr(data, 'feature_names', [f'Feature {i}' for i in range(X.shape[1])]))

        # Get the real class label
        if hasattr(data, 'target_names'):
            # If y is integer index, map to class name
            class_label = data.target_names[y[point_index]] if int(y[point_index]) < len(data.target_names) else str(y[point_index])
        else:
            class_label = f"Class {y[point_index]}"

        # Get the coordinates from the figure
        x_coord = point_data['x']
        y_coord = point_data['y']

        # Create coordinates display with all requested fields
        coordinates_table = html.Table([
            html.Thead(
                html.Tr([
                    html.Th("Property", style=CELL_STYLE),
                    html.Th("Value", style=CELL_STYLE_RIGHT)
                ])
            ),
            html.Tbody([
                html.Tr([
                    html.Td("Sample", style=CELL_STYLE),
                    html.Td(f"#{point_index}", style=CELL_STYLE_RIGHT)
                ]),
                html.Tr([
                    html.Td("Class", style=CELL_STYLE),
                    html.Td(class_label, style=CELL_STYLE_RIGHT)
                ]),
                html.Tr([
                    html.Td("Label", style=CELL_STYLE),
                    html.Td(digit_label, style=CELL_STYLE_RIGHT)
                ]),
                html.Tr([
                    html.Td("X Coordinate", style=CELL_STYLE),
                    html.Td(f"{x_coord:.4f}", style=CELL_STYLE_RIGHT)
                ]),
                html.Tr([
                    html.Td("Y Coordinate", style=CELL_STYLE),
                    html.Td(f"{y_coord:.4f}", style=CELL_STYLE_RIGHT)
                ])
            ])
        ], style=TABLE_STYLE)

        # Create metadata table
        metadata_table = html.Table([
            html.Thead(
                html.Tr([
                    html.Th("Feature", style=CELL_STYLE),
                    html.Th("Value", style=CELL_STYLE_RIGHT)
                ])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(name, style=CELL_STYLE),
                    html.Td(f"{value:.4f}", style=CELL_STYLE_RIGHT)
                ])
                for name, value in zip(features_to_display, X[point_index][:len(features_to_display)])
            ])
        ], style=TABLE_STYLE)

        # For image datasets (Digits, MNIST, Fashion MNIST, Elephant), create and display the image
        if dataset_name in ["Digits", "MNIST", "Fashion MNIST", "Elephant", "PACS - Photo", "PACS - Cartoon", "PACS - Art Painting"]:
            img_str = encode_img_as_str(X[point_index], dataset_name)

            # For image-only datasets, show empty metadata
            if dataset_name in IMAGE_ONLY_DATASETS:
                metadata_content = html.Div("No meaningful metadata to display for image data",
                                          style=EMPTY_METADATA_STYLE)
            else:
                metadata_content = metadata_table

            return (
                f'data:image/png;base64,{img_str}',
                get_image_style('block'),
                get_no_image_message_style('none'),
                coordinates_table,
                metadata_content,
                {'display': 'none'}, # Hide click message
                get_generative_placeholder_style('none') # Hide generative mode placeholder
            )

        # For other datasets, show no image message and metadata
        # For image-only datasets, show empty metadata
        if dataset_name in IMAGE_ONLY_DATASETS:
            metadata_content = html.Div("No meaningful metadata to display for image data",
                                      style=EMPTY_METADATA_STYLE)
        else:
            metadata_content = metadata_table

        return (
            '',
            get_image_style('none'),
            get_no_image_message_style('block'),
            coordinates_table,
            metadata_content,
            {'display': 'none'}, # Hide click message
            get_generative_placeholder_style('none') # Hide generative mode placeholder
        )

    # Callbacks for the new UI elements in the left panel

    # Generative Mode Button Toggle
    @app.callback(
        Output('generative-mode-btn', 'color'),
        Output('generative-mode-btn', 'children'),
        Output('generative-mode-state', 'data'),
        Input('generative-mode-btn', 'n_clicks'),
        State('generative-mode-state', 'data'),
        prevent_initial_call=True
    )
    def toggle_generative_mode(n_clicks, current_state):
        if n_clicks is None:
            return 'secondary', [html.I(className="fas fa-lightbulb me-2"), "Generative Mode"], {'enabled': False}

        # Toggle the state
        new_enabled = not current_state.get('enabled', False)

        if new_enabled:
            return 'success', [html.I(className="fas fa-lightbulb me-2"), "Generative Mode (ON)"], {'enabled': True}
        else:
            return 'secondary', [html.I(className="fas fa-lightbulb me-2"), "Generative Mode"], {'enabled': False}

    # Callback to show/hide iterative slider based on generative mode
    @app.callback(
        Output('iteration-slider', 'className'),
        Output('slider-play-btn', 'style'),
        Input('generative-mode-state', 'data'),
        prevent_initial_call=False
    )
    def toggle_iterative_slider(generative_state):
        enabled = generative_state.get('enabled', False) if generative_state else False

        if enabled:
            # Hide the slider and play button when generative mode is on
            return 'mb-3 d-none', {'display': 'none'}
        else:
            # Show the slider and play button when generative mode is off
            return 'mb-3', {'display': 'block'}


    # Layer Buttons (Mutually Exclusive)
    @app.callback(
        [Output(f'layer-opt{i}-btn', 'className') for i in range(1, 4)],
        [Input(f'layer-opt{i}-btn', 'n_clicks') for i in range(1, 4)],
        prevent_initial_call=True
    )
    def select_layer(*n_clicks):
        ctx = callback_context
        if not ctx.triggered:
            # Default state, make the first one selected
            classes = ['control-button'] * 3
            classes[0] = 'control-button selected'
            return classes

        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        classes = ['control-button'] * 3

        if button_id == 'layer-opt1-btn':
            classes[0] = 'control-button selected'
        elif button_id == 'layer-opt2-btn':
            classes[1] = 'control-button selected'
        elif button_id == 'layer-opt3-btn':
            classes[2] = 'control-button selected'
        return classes

    # Method Buttons (Mutually Exclusive)
    @app.callback(
        Output('tsne-thumbnail-click', 'className'),
        Output('umap-thumbnail-click', 'className'),
        Output('trimap-thumbnail-click', 'className'),
        Input('tsne-thumbnail-click', 'n_clicks'),
        Input('umap-thumbnail-click', 'n_clicks'),
        Input('trimap-thumbnail-click', 'n_clicks'),
        prevent_initial_call=True
    )
    def select_method_button(tsne_n_clicks, umap_n_clicks, trimap_n_clicks):
        ctx = callback_context
        if not ctx.triggered:
            # Default state, TRIMAP is initially selected
            return "method-button-container thumbnail-button mb-3", "method-button-container thumbnail-button mb-3", "method-button-container thumbnail-button mb-3 selected"

        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        tsne_class = "method-button-container thumbnail-button mb-3"
        umap_class = "method-button-container thumbnail-button mb-3"
        trimap_class = "method-button-container thumbnail-button mb-3"

        if button_id == 'tsne-thumbnail-click':
            tsne_class = "method-button-container thumbnail-button mb-3 selected"
        elif button_id == 'umap-thumbnail-click':
            umap_class = "method-button-container thumbnail-button mb-3 selected"
        elif button_id == 'trimap-thumbnail-click':
            trimap_class = "method-button-container thumbnail-button mb-3 selected"

        return tsne_class, umap_class, trimap_class

    # Callback for Upload Custom Dataset dropdown option
    @app.callback(
        Output('umap-warning', 'children', allow_duplicate=True),
        Input('dataset-dropdown', 'value'),
        prevent_initial_call=True
    )
    def handle_custom_dataset_upload_dropdown(dataset_value):
        if dataset_value == 'custom_upload':
            return "Please use the 'Upload new datapoint' button to upload a custom dataset."
        return ""

    # Callback for Upload New Datapoint button
    @app.callback(
        Output('image-display', 'style'),
        Output('image-draw', 'style'),
        Input('upload-new-datapoint-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def handle_upload_new_datapoint_button(n_clicks):
        # Dynamic transforms
        style1 = {'height': '0', 'border': '1px solid #dee2e6', 'padding': '1rem', 'margin-bottom': '0.5rem',
                  'display': 'block', 'visibility': 'hidden'}
        style2 = style1.copy()
        style2['height'] = '56vh'
        style2['visibility'] = 'visible'
        if n_clicks % 2 == 0:
            style1, style2 = style2, style1
        return style1, style2

    @app.callback(
        Output('canvas', 'json_data'),  # Clear the canvas
        Output('added-data-cache', 'data', allow_duplicate=True),  # Store processed data
        Input('submit_drawing', 'n_clicks'),
        State('canvas', 'json_data'),
        State('added-data-cache', 'data'),
        State('dataset-dropdown', 'value'),
        prevent_initial_call=True
    )
    def submit_and_clear(n_clicks, json_data, added_data_cache, dataset_name):
        if not json_data:
            return dash.no_update, dash.no_update

        canvas_shape = (500, 500)
        mask = parse_jsonstring(json_data, canvas_shape)

        # Crop to content
        ys, xs = np.where(mask)
        left, right = xs.min(), xs.max()
        top, bottom = ys.min(), ys.max()
        width, height = right - left, bottom - top
        largest_dim = max(width, height)
        pad = (largest_dim - width) / 2, (largest_dim - height) / 2
        mask = mask[left - ceil(pad[0]):right + floor(pad[0]), top - ceil(pad[1]):bottom + floor(pad[1])]

        processed = mask.astype(float)
        img_shape = dataset_shape(dataset_name)

        processed = cv2.resize(processed, img_shape)
        processed = np.reshape(processed, -1)
        if 'user_generated' not in added_data_cache:
            added_data_cache['user_generated'] = []
        added_data_cache['user_generated'].append(processed)

        return "", added_data_cache

    @app.callback(
        Output('added-data-cache', 'data', allow_duplicate=True),  # Store processed data
        Input('brightness-slider', 'value'),
        Input('contrast-slider', 'value'),
        State('added-data-cache', 'data'),
        State('dataset-dropdown', 'value'),
        prevent_initial_call=True
    )
    def augment_data(brightness, contrast, data_cache, dataset_name, n_samples=100):
        X, y, _ = get_dataset(dataset_name)
        random_indices = np.random.choice(len(X), n_samples, replace=False)
        X, y = X[random_indices], y[random_indices]
        X = contrast * X + (brightness - 1)
        X = np.clip(X, 0, 1).astype(float)
        data_cache['augmented'] = (X, y)
        return data_cache

    # Callback to show/hide metadata row based on dataset type
    @app.callback(
        Output('point-metadata-row', 'style'),
        Input('dataset-dropdown', 'value'),
        prevent_initial_call=False
    )
    def toggle_metadata_row(dataset_name):
        if dataset_name in IMAGE_ONLY_DATASETS:
            # Hide the metadata row for image-only datasets
            return {'display': 'none'}
        else:
            # Show the metadata row for datasets with meaningful features
            return {'display': 'block'}

    @app.callback(
        Output("full-grid-container", "style"),
        [
            Input("full-grid-btn", "n_clicks"),
            Input("main-graph", "clickData")
        ],
        State("full-grid-container", "style"),
        prevent_initial_call=True
    )
    def toggle_full_grid(n_clicks, click_data, current_style):
        triggered = callback_context.triggered_id

        if triggered == "full-grid-btn":
            # Toggle the grid open/close
            if current_style and current_style.get("display") == "block":
                return {"display": "none"}
            return {"display": "block"}

        elif triggered == "main-graph" and click_data is not None:
            # Hide grid after clicking on image
            return {"display": "none"}

        return current_style

    @app.callback(
        Output("full-grid-container", "children"),
        Input("full-grid-btn", "n_clicks"),
        State("dataset-dropdown", "value"),
        prevent_initial_call=True
    )
    def generate_full_grid(n_clicks, dataset_name):
        X, y, data = get_dataset(dataset_name)
        if dataset_name not in IMAGE_ONLY_DATASETS:
            return html.Div("Full grid only available for image datasets.")

        return html.Div([
            html.Img(src=create_datapoint_image(X[i]), style={'width': '40px', 'margin': '2px'})
            for i in range(min(len(X), 300))  # Limit for performance
        ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fill, 40px)', 'gap': '4px'})
    
    @app.callback(
        Output('dataset-dropdown', 'options'),
        Output('dataset-dropdown', 'value'),
        Input('dataset-family-dropdown', 'value'),
    )
    def update_dataset_options(family):
        if family == "classic":
            options = [
                {"label": "Digits", "value": "Digits"},
                {"label": "Iris", "value": "Iris"},
                {"label": "Wine", "value": "Wine"},
                {"label": "Breast Cancer", "value": "Breast Cancer"},
                {"label": "MNIST", "value": "MNIST"},
                {"label": "Fashion MNIST", "value": "Fashion MNIST"},
                {"label": "Elephant", "value": "Elephant"},
            ]
            return options, "Digits"

        elif family == "pacs":
            options = [
                {"label": "Photo", "value": "PACS - Photo"},
                {"label": "Sketch", "value": "PACS - Sketch"},
                {"label": "Cartoon", "value": "PACS - Cartoon"},
                {"label": "Art Painting", "value": "PACS - Art Painting"},
            ]
            return options, "PACS - Photo"

        elif family == "custom_upload":
            return [{"label": "Upload Custom Dataset", "value": "custom_upload"}], "custom_upload"

        return [], None


    # Existing callbacks (ensure they are still present after the edit)

    # Main callback to update graphs and metadata based on dataset-dropdown
    # ... (existing update_graphs and update_calculation_message) ...
    # (No changes to the existing update_graphs and update_calculation_message, but keeping this comment for clarity)

    # Note: The callbacks for top-grid and bottom-grid buttons were removed in the layout update.
    # If new buttons are added that need similar functionality, new callbacks will be needed.
