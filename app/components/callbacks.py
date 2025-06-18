import jax.random as random
from sklearn import datasets
from sklearn.manifold import TSNE
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, callback_context, State, callback
import numpy as np
import threading
import matplotlib
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

# Import TRIMAP from local package
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'google_research_trimap'))
from google_research_trimap.trimap import trimap

# Try to import UMAP, if available
umap_available = False
umap_lib = None # Use a distinct name for the imported module
try:
    import umap as umap_lib # Import as umap_lib
    umap_available = True
except ImportError:
    pass # umap_lib remains None

from mulan_demo.app.components.feature_config import DATASET_FEATURES, IMAGE_ONLY_DATASETS
from .settings import (
    get_image_style, get_generative_placeholder_style, get_no_image_message_style,
    EMPTY_METADATA_STYLE, TABLE_STYLE, CELL_STYLE, CELL_STYLE_RIGHT, IMAGE_FIGURE_SIZE
)

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

def compute_trimap(X, key):
    start_time = time.time()
    with numba_global_lock:
        print('Starting TRIMAP calculation...')
        
        # Adjust n_inliers based on dataset size
        n_points = X.shape[0]
        n_inliers = min(5, n_points - 2)  # Use at most 5 inliers, but ensure it's less than n_points-1
        
        # Time the nearest neighbor search
        nn_start = time.time()
        print('Finding nearest neighbors...')
        emb = trimap.transform(key, X, n_inliers=n_inliers, distance='euclidean', verbose=False)
        nn_time = time.time() - nn_start
        print(f'Nearest neighbor search took: {nn_time:.2f} seconds')
        
        result = np.array(emb) if hasattr(emb, "shape") else emb
        total_time = time.time() - start_time
        print(f'Total TRIMAP calculation took: {total_time:.2f} seconds')
    return result, total_time

def compute_tsne(X):
    start_time = time.time()
    with numba_global_lock:
        print('calculating tsne')
        emb = TSNE(n_components=2, random_state=42).fit_transform(X)
        result = np.array(emb)
        print('tsne calculated')
    return result, time.time() - start_time

def compute_umap(X):
    if not umap_available or umap_lib is None:
        return None, 0
    start_time = time.time()
    with numba_global_lock:
        print('calculating umap')
        reducer = umap_lib.UMAP(n_components=2, random_state=42)
        emb = reducer.fit_transform(X)
        result = np.array(emb)
        print('umap calculated')
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
    plt.figure(figsize=(size[0]/100, size[1]/100), dpi=100)
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
        Output('umap-warning', 'children'),
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
        State('embedding-cache', 'data')
    )
    def update_graphs(dataset_name, recalculate_flag, trimap_n_clicks, tsne_n_clicks, umap_n_clicks, cached_embeddings):
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
            if not recalculate_flag:
                if embedding_exists(dataset_name, method_name):
                    embedding, metadata = load_embedding(dataset_name, method_name)
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
            if recalculate_flag or not embedding_exists(dataset_name, method_name):
                print(f"Computing new {method_name} embedding")
                embedding, compute_time = compute_func(*args)
                
                # Save the embedding if computation was successful
                if embedding is not None:
                    metadata = {'time': compute_time}
                    save_embedding(dataset_name, method_name, embedding, metadata)
                    
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
        key = random.PRNGKey(0)
        trimap_emb = get_embedding('trimap', compute_trimap, X, key)
        tsne_emb = get_embedding('tsne', compute_tsne, X)
        umap_emb = get_embedding('umap', compute_umap, X)
        
        # Create figures
        main_fig = create_figure(
            trimap_emb if method == 'trimap' else tsne_emb if method == 'tsne' else umap_emb,
            y,
            f"{method.upper()} Embedding of {dataset_name}",
            "Class",
            X
        )
        
        trimap_fig = create_figure(trimap_emb, y, "TRIMAP", "Class", X, is_thumbnail=True)
        tsne_fig = create_figure(tsne_emb, y, "t-SNE", "Class", X, is_thumbnail=True)
        umap_fig = create_figure(umap_emb, y, "UMAP", "Class", X, is_thumbnail=True)
        
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
        
        return (
            main_fig,
            trimap_fig,
            tsne_fig,
            umap_fig,
            umap_warning,
            metadata,
            cached_embeddings,
            f"TRIMAP: {trimap_time:.2f}s",
            f"t-SNE: {tsne_time:.2f}s",
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
    def update_calculation_message(dataset_name, trimap_n_clicks, tsne_n_clicks, umap_n_clicks, recalculate_flag, cached_embeddings):
        ctx = callback_context
        if not ctx.triggered:
            return f"Loading dataset {dataset_name}..." # Initial state, no message

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
        Output('generative-mode-placeholder', 'style'),
        Input('main-graph', 'clickData'),
        Input('generative-mode-state', 'data'),
        State('dataset-dropdown', 'value'),
        State('main-graph', 'figure')
    )
    def display_clicked_point(clickData, generative_state, dataset_name, figure):
        enabled = generative_state.get('enabled', False) if generative_state else False
        
        if enabled:
            # In generative mode, show placeholder and hide other image elements
            return (
                '', # selected-image src
                get_image_style('none'), # selected-image style
                get_no_image_message_style('none'), # no-image-message style
                "", # coordinates-display children
                "", # point-metadata children
                {'display': 'none'}, # click-message style
                get_generative_placeholder_style('block') # generative-mode-placeholder style
            )
        
        if not clickData:
            # Return default states when nothing is clicked
            return (
                '', # selected-image src
                get_image_style('none'), # selected-image style
                get_no_image_message_style('block'), # no-image-message style
                "", # coordinates-display children
                "", # point-metadata children
                {'display': 'block', 'text-align': 'center', 'padding': '0.5rem', 'margin-bottom': '0.5rem', 'color': '#666'}, # click-message style
                get_generative_placeholder_style('none') # generative-mode-placeholder style
            )
        
        # Get the point index from the custom data
        point_index = int(clickData['points'][0]['customdata'][0])
        X, y, data = get_dataset(dataset_name)
        
        # Get features to display from configuration or use default feature names
        features_to_display = DATASET_FEATURES.get(dataset_name, getattr(data, 'feature_names', [f'Feature {i}' for i in range(X.shape[1])]))
        
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
        if dataset_name in ["Digits", "MNIST", "Fashion MNIST", "Elephant"]:
            import base64
            from io import BytesIO
            
            # Get the image dimensions based on the dataset
            if dataset_name == "Digits":
                img_shape = (8, 8)
            elif dataset_name == "Elephant":
                img_shape = (28, 28)  # Elephant images are resized to 28x28
            else:  # MNIST or Fashion MNIST
                img_shape = (28, 28)
            
            # Create the image with proper aspect ratio
            plt.figure(figsize=IMAGE_FIGURE_SIZE)
            plt.imshow(X[point_index].reshape(img_shape), cmap='gray')
            plt.axis('off')
            
            # Convert to base64
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            plt.close()
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode()
            
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

    # Callback to show/hide iterative process label based on generative mode
    @app.callback(
        Output('iteration-process-container', 'style'),
        Input('generative-mode-state', 'data'),
        prevent_initial_call=False
    )
    def toggle_iterative_process_label(generative_state):
        enabled = generative_state.get('enabled', False) if generative_state else False
        
        if enabled:
            # Hide the entire iterative process container when generative mode is on
            return {'display': 'none'}
        else:
            # Show the iterative process container when generative mode is off
            return {'display': 'block'}

    # Distance Measure Buttons (Mutually Exclusive)
    @app.callback(
        [Output(f'dist-opt{i}-btn', 'className') for i in range(1, 3)] + [Output('dist-upload-btn', 'className')],
        [Input(f'dist-opt{i}-btn', 'n_clicks') for i in range(1, 3)] + [Input('dist-upload-btn', 'n_clicks')],
        prevent_initial_call=True
    )
    def select_distance_measure(*n_clicks):
        ctx = callback_context
        if not ctx.triggered:
            # Default state, make the first one selected
            classes = ['control-button'] * 3
            classes[0] = 'control-button selected'
            return classes

        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        classes = ['control-button'] * 3

        if button_id == 'dist-opt1-btn':
            classes[0] = 'control-button selected'
        elif button_id == 'dist-opt2-btn':
            classes[1] = 'control-button selected'
        elif button_id == 'dist-upload-btn':
            classes[2] = 'control-button selected'
        return classes

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
        Output('calculation-status', 'children', allow_duplicate=True), 
        Input('upload-new-datapoint-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def handle_upload_new_datapoint_button(n_clicks):
        if n_clicks:
            return "" 
        return ""

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

    # Existing callbacks (ensure they are still present after the edit)

    # Main callback to update graphs and metadata based on dataset-dropdown
    # ... (existing update_graphs and update_calculation_message) ...
    # (No changes to the existing update_graphs and update_calculation_message, but keeping this comment for clarity)

    # Note: The callbacks for top-grid and bottom-grid buttons were removed in the layout update.
    # If new buttons are added that need similar functionality, new callbacks will be needed.
