import threading

import jax.random as random
import matplotlib
import numpy as np
import plotly.express as px
from dash import html, Input, Output, callback_context, State
from sklearn import datasets
from sklearn.manifold import TSNE

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import time
import io
import base64
from PIL import Image
import tensorflow as tf
import requests
from io import BytesIO
from .embedding_storage import save_embedding, load_embedding, embedding_exists

# Import TRIMAP from local package
from google_research_trimap.trimap import trimap
import umap.umap_ as umap



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
}

# Single global lock for computations
numba_global_lock = threading.Lock()

from mulan_demo.app.components.feature_config import DATASET_FEATURES


def get_dataset(name):
    with numba_global_lock:
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
        # n_points = X.shape[0]
        # n_inliers = min(5, n_points - 2)  # Use at most 5 inliers, but ensure it's less than n_points-1

        # Time the nearest neighbor search
        nn_start = time.time()
        print('Finding nearest neighbors...')
        emb = trimap.transform(key, X, distance='euclidean', verbose=False, output_metric=distance, auto_diff=False)
        nn_time = time.time() - nn_start
        print(f'Nearest neighbor search took: {nn_time:.2f} seconds')

        result = np.array(emb) if hasattr(emb, "shape") else emb
        total_time = time.time() - start_time
        print(f'Total TRIMAP calculation took: {total_time:.2f} seconds')
    if distance == 'haversine':
        x = np.arctan2(np.sin(result[:, 0]) * np.cos(result[:, 1]), np.sin(result[:, 0]) * np.sin(result[:, 1]))
        y = -np.arccos(np.cos(result[:, 0]))
        result = np.column_stack((x, y))
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
        reducer = umap.UMAP(n_components=2, random_state=42, output_metric=distance)
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
    # Normalize the data point to 0-1 range
    normalized = (data_point - data_point.min()) / (data_point.max() - data_point.min())

    # Reshape if needed (assuming square image)
    side_length = int(np.sqrt(len(normalized)))
    if side_length * side_length == len(normalized):
        img_data = normalized.reshape(side_length, side_length)
    else:
        # If not a perfect square, create a rectangular image
        img_data = normalized.reshape(-1, 8)  # Arbitrary width of 8

    # Create the image
    plt.figure(figsize=(size[0] / 100, size[1] / 100), dpi=100)
    plt.imshow(img_data, cmap='viridis')
    plt.axis('off')

    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    return f"data:image/png;base64,{img_str}"


def create_figure(embedding, y, title, label_name, X=None, is_thumbnail=False):
    if embedding is None or len(embedding) == 0 or embedding.shape[1] < 2:
        return px.scatter(title=f"{title} (no data)")

    # Create a list of customdata for each point, including the point index
    point_indices = np.arange(len(y))

    # Create a DataFrame with all the data
    import pandas as pd
    df = pd.DataFrame({
        'x': embedding[:, 0],
        'y': embedding[:, 1],
        'color': y.astype(str),
        'point_index': point_indices,
        'label': y.astype(str)
    })

    # Create the figure with the DataFrame
    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='color',
        custom_data=['point_index', 'label'],
        title=title,
        labels={'x': 'Component 1', 'y': 'Component 2', 'color': label_name}
    )

    if is_thumbnail:
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showticklabels=False, visible=False),
            yaxis=dict(showticklabels=False, visible=False),
            showlegend=False,
            hovermode=False
        )
    else:
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
        Output('main-graph', 'figure'),
        Output('trimap-thumbnail', 'figure'),
        Output('tsne-thumbnail', 'figure'),
        Output('umap-thumbnail', 'figure'),
        Output('metadata-display', 'children'),
        Output('embedding-cache', 'data'),
        Output('trimap-timing', 'children'),
        Output('tsne-timing', 'children'),
        Output('umap-timing', 'children'),
        Input('dataset-dropdown', 'value'),
        Input('recalculate-switch', 'value'),
        Input('trimap-thumbnail-click', 'n_clicks'),
        Input('tsne-thumbnail-click', 'n_clicks'),
        Input('umap-thumbnail-click', 'n_clicks'),
        State('embedding-cache', 'data'),
        Input('dist-dropdown', 'value'),
    )
    def update_graphs(dataset_name, recalculate_flag, trimap_n_clicks, tsne_n_clicks, umap_n_clicks, cached_embeddings, distance):
        if not dataset_name:
            return [px.scatter(title="No dataset selected")] * 3 + [""] * 7

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
                    return embedding

            # Compute new embedding
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

        # Get embeddings for all methods
        trimap_emb = get_embedding('trimap', compute_trimap, X, distance)
        if distance == 'haversine':
            tsne_emb = None
        else:
            tsne_emb = get_embedding('tsne', compute_tsne, X, distance)
        umap_emb = get_embedding('umap', compute_umap, X, distance)

        # Create figures
        # If distance is haversine and method is tsne, force fallback to trimap
        if distance == 'haversine' and method == 'tsne':
            main_fig = create_figure(
                trimap_emb,
                y,
                f"TRIMAP Embedding of {dataset_name} (t-SNE disabled for haversine)",
                "Class",
                X
            )
        else:
            main_fig = create_figure(
                trimap_emb if method == 'trimap' else tsne_emb if method == 'tsne' else umap_emb,
                y,
                f"{method.upper()} Embedding of {dataset_name}",
                "Class",
                X
            )

        trimap_fig = create_figure(trimap_emb, y, "TRIMAP", "Class", X, is_thumbnail=True)
        if distance == 'haversine':
            tsne_fig = create_figure(None, y, "t-SNE (disabled for haversine)", "Class", X, is_thumbnail=True)
        else:
            tsne_fig = create_figure(tsne_emb, y, "t-SNE", "Class", X, is_thumbnail=True)
        umap_fig = create_figure(umap_emb, y, "UMAP", "Class", X, is_thumbnail=True)

        # Metadata display
        metadata = create_metadata_display(dataset_name, data)

        # Update cache
        cached_embeddings[dataset_name] = {
            'trimap': trimap_emb.tolist() if trimap_emb is not None else None,
            'tsne': tsne_emb.tolist() if tsne_emb is not None else None if distance != 'haversine' else None,
            'umap': umap_emb.tolist() if umap_emb is not None else None
        }

        # If distance is haversine, show t-SNE timing as "Disabled"
        tsne_time_display = f"t-SNE: Disabled for haversine" if distance == 'haversine' else f"t-SNE: {tsne_time:.2f}s"
        return (
            main_fig,
            trimap_fig,
            tsne_fig,
            umap_fig,
            metadata,
            cached_embeddings,
            f"TRIMAP: {trimap_time:.2f}s",
            tsne_time_display,
            f"UMAP: {umap_time:.2f}s"
        )

    @app.callback(
        Output('calculation-status', 'children'),
        Input('dataset-dropdown', 'value'),
        Input('trimap-thumbnail-click', 'n_clicks'),
        Input('tsne-thumbnail-click', 'n_clicks'),
        Input('umap-thumbnail-click', 'n_clicks'),
        Input('recalculate-switch', 'value'),
        State('embedding-cache', 'data')
    )
    def update_calculation_message(dataset_name, trimap_n_clicks, tsne_n_clicks, umap_n_clicks, recalculate_flag,
                                   cached_embeddings):
        ctx = callback_context
        if not ctx.triggered:
            return f"Loading dataset {dataset_name}..."  # Initial state, no message

        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        # For dataset loading
        if trigger_id == 'dataset-dropdown':
            return f"Loading dataset {dataset_name}..."

        # For recalculation
        elif trigger_id == 'recalculate-switch' and recalculate_flag:
            return f"Recalculating embeddings for {dataset_name}..."

        # For thumbnail clicks
        elif trigger_id in ['trimap-thumbnail-click', 'tsne-thumbnail-click', 'umap-thumbnail-click']:
            method_name = trigger_id.replace('-thumbnail-click', '').upper()
            return f"Calculating {method_name} embedding..."

        return ""

    @app.callback(
        Output('selected-image', 'src'),
        Output('selected-image', 'style'),
        Output('no-image-message', 'style'),
        Output('coordinates-display', 'children'),
        Output('point-metadata', 'children'),
        Output('click-message', 'style'),
        Input('main-graph', 'clickData'),
        State('dataset-dropdown', 'value'),
        State('main-graph', 'figure')
    )
    def display_clicked_point(clickData, dataset_name, figure):
        if not clickData:
            # Return default states when nothing is clicked
            return (
                '',  # selected-image src
                {'display': 'none'},  # selected-image style
                {'display': 'block', 'text-align': 'center', 'padding': '1rem'},  # no-image-message style
                "",  # coordinates-display children
                "",  # point-metadata children
                {'display': 'block', 'text-align': 'center', 'padding': '0.5rem', 'margin-bottom': '0.5rem',
                 'color': '#666'}  # click-message style
            )

        # Get the point index from the custom data
        point_index = int(clickData['points'][0]['customdata'][0])
        X, y, data = get_dataset(dataset_name)

        # Get features to display from configuration or use default feature names
        features_to_display = DATASET_FEATURES.get(dataset_name, getattr(data, 'feature_names',
                                                                         [f'Feature {i}' for i in range(X.shape[1])]))

        # Get the real class label
        if hasattr(data, 'target_names'):
            class_label = data.target_names[y[point_index]]
        else:
            class_label = f"Class {y[point_index]}"

        # Get the coordinates from the figure
        x_coord = clickData['points'][0]['x']
        y_coord = clickData['points'][0]['y']

        # Get the color label from the figure
        digit_label = clickData['points'][0]['customdata'][1]

        # Create coordinates display with all requested fields
        coordinates_table = html.Table([
            html.Thead(
                html.Tr([
                    html.Th("Property", style={'text-align': 'left', 'padding': '8px'}),
                    html.Th("Value", style={'text-align': 'right', 'padding': '8px'})
                ])
            ),
            html.Tbody([
                html.Tr([
                    html.Td("Sample", style={'text-align': 'left', 'padding': '8px'}),
                    html.Td(f"#{point_index}", style={'text-align': 'right', 'padding': '8px'})
                ]),
                html.Tr([
                    html.Td("Class", style={'text-align': 'left', 'padding': '8px'}),
                    html.Td(class_label, style={'text-align': 'right', 'padding': '8px'})
                ]),
                html.Tr([
                    html.Td("Label", style={'text-align': 'left', 'padding': '8px'}),
                    html.Td(digit_label, style={'text-align': 'right', 'padding': '8px'})
                ]),
                html.Tr([
                    html.Td("X Coordinate", style={'text-align': 'left', 'padding': '8px'}),
                    html.Td(f"{x_coord:.4f}", style={'text-align': 'right', 'padding': '8px'})
                ]),
                html.Tr([
                    html.Td("Y Coordinate", style={'text-align': 'left', 'padding': '8px'}),
                    html.Td(f"{y_coord:.4f}", style={'text-align': 'right', 'padding': '8px'})
                ])
            ])
        ], style={'width': '100%', 'border-collapse': 'collapse'})

        # Create metadata table
        metadata_table = html.Table([
            html.Thead(
                html.Tr([
                    html.Th("Feature", style={'text-align': 'left', 'padding': '8px'}),
                    html.Th("Value", style={'text-align': 'right', 'padding': '8px'})
                ])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(name, style={'text-align': 'left', 'padding': '8px'}),
                    html.Td(f"{value:.4f}", style={'text-align': 'right', 'padding': '8px'})
                ])
                for name, value in zip(features_to_display, X[point_index][:len(features_to_display)])
            ])
        ], style={'width': '100%', 'border-collapse': 'collapse'})

        # For image datasets (Digits, MNIST, Fashion MNIST), create and display the image
        if dataset_name in ["Digits", "MNIST", "Fashion MNIST"]:
            import base64
            from io import BytesIO

            # Get the image dimensions based on the dataset
            if dataset_name == "Digits":
                img_shape = (8, 8)
            else:  # MNIST or Fashion MNIST
                img_shape = (28, 28)

            # Create the image with proper aspect ratio
            plt.figure(figsize=(4, 4))
            plt.imshow(X[point_index].reshape(img_shape), cmap='gray')
            plt.axis('off')

            # Convert to base64
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            plt.close()
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode()

            return (
                f'data:image/png;base64,{img_str}',
                {'max-width': '100%', 'height': '22vh', 'object-fit': 'contain', 'display': 'block',
                 'padding': '0.5rem'},
                {'display': 'none'},
                coordinates_table,
                metadata_table,
                {'display': 'none'}  # Hide click message
            )

        # For other datasets, show no image message and metadata
        return (
            '',
            {'display': 'none'},
            {'display': 'block', 'text-align': 'center', 'padding': '0.5rem', 'height': '22vh'},
            coordinates_table,
            metadata_table,
            {'display': 'none'}  # Hide click message
        )

    # Callbacks for the new UI elements in the left panel

    # Generative Mode Button Toggle
    @app.callback(
        Output('generative-mode-btn', 'color'),
        Output('generative-mode-btn', 'children'),
        Input('generative-mode-btn', 'n_clicks'),
        State('generative-mode-btn', 'color'),
        prevent_initial_call=True
    )
    def toggle_generative_mode(n_clicks, current_color):
        if n_clicks is None:
            return 'secondary', [html.I(className="fas fa-lightbulb me-2"), "Generative Mode"]  # Default off

        if current_color == 'secondary':
            return 'success', [html.I(className="fas fa-lightbulb me-2"), "Generative Mode (ON)"]
        else:
            return 'secondary', [html.I(className="fas fa-lightbulb me-2"), "Generative Mode"]

    # Distance Measure Dropdown (Primary selector)
    @app.callback(
        Output('distance-measure-store', 'data'),
        Input('dist-dropdown', 'value'),
        prevent_initial_call=True
    )
    def select_distance_measure_dropdown(value):
        print(f"Distance measure selected from dropdown: {value}")
        return value

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
        Output('tsne-thumbnail-click', 'style'),
        Input('tsne-thumbnail-click', 'n_clicks'),
        Input('umap-thumbnail-click', 'n_clicks'),
        Input('trimap-thumbnail-click', 'n_clicks'),
        Input('dist-dropdown', 'value'),
        prevent_initial_call=True
    )
    def select_method_button(tsne_n_clicks, umap_n_clicks, trimap_n_clicks, distance):
        ctx = callback_context
        if not ctx.triggered:
            # Default state, TRIMAP is initially selected
            tsne_style = {'pointer-events': 'none', 'opacity': 0.5} if distance == 'haversine' else {}
            return "method-button-container thumbnail-button mb-3", "method-button-container thumbnail-button mb-3", "method-button-container thumbnail-button mb-3 selected", tsne_style

        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        tsne_class = "method-button-container thumbnail-button mb-3"
        umap_class = "method-button-container thumbnail-button mb-3"
        trimap_class = "method-button-container thumbnail-button mb-3"

        if button_id == 'tsne-thumbnail-click' and distance != 'haversine':
            tsne_class = "method-button-container thumbnail-button mb-3 selected"
        elif button_id == 'umap-thumbnail-click':
            umap_class = "method-button-container thumbnail-button mb-3 selected"
        elif button_id == 'trimap-thumbnail-click':
            trimap_class = "method-button-container thumbnail-button mb-3 selected"

        tsne_style = {'pointer-events': 'none', 'opacity': 0.5} if distance == 'haversine' else {}
        return tsne_class, umap_class, trimap_class, tsne_style

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
        Output('calculation-status', 'children', allow_duplicate=True),
        Input('upload-new-datapoint-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def handle_upload_new_datapoint_button(n_clicks):
        if n_clicks:
            return ""
        return ""

    # Existing callbacks (ensure they are still present after the edit)

    # Main callback to update graphs and metadata based on dataset-dropdown
    # ... (existing update_graphs and update_calculation_message) ...
    # (No changes to the existing update_graphs and update_calculation_message, but keeping this comment for clarity)

    # Note: The callbacks for top-grid and bottom-grid buttons were removed in the layout update.
    # If new buttons are added that need similar functionality, new callbacks will be needed.
