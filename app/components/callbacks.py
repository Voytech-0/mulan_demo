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

# Import TRIMAP from local package
from google_research_trimap.trimap import trimap

# Try to import UMAP, if available
umap_available = False
umap_lib = None # Use a distinct name for the imported module
try:
    import umap as umap_lib # Import as umap_lib
    umap_available = True
except ImportError:
    pass # umap_lib remains None

# List of available datasets
DATASET_LOADERS = {
    "Digits": datasets.load_digits,
    "Iris": datasets.load_iris,
    "Wine": datasets.load_wine,
    "Breast Cancer": datasets.load_breast_cancer,
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

def compute_trimap(X, key):
    start_time = time.time()
    with numba_global_lock:
        emb = trimap.transform(key, X, distance='euclidean')
        result = np.array(emb) if hasattr(emb, "shape") else emb
    return result, time.time() - start_time

def compute_tsne(X):
    start_time = time.time()
    with numba_global_lock:
        emb = TSNE(n_components=2, random_state=42).fit_transform(X)
        result = np.array(emb)
    return result, time.time() - start_time

def compute_umap(X):
    if not umap_available or umap_lib is None:
        return None, 0
    start_time = time.time()
    with numba_global_lock:
        reducer = umap_lib.UMAP(n_components=2, random_state=42)
        emb = reducer.fit_transform(X)
        result = np.array(emb)
    return result, time.time() - start_time

def create_figure(embedding, y, title, label_name, is_thumbnail=False):
    if embedding is None or len(embedding) == 0 or embedding.shape[1] < 2:
        return px.scatter(title=f"{title} (no data)")
    
    fig = px.scatter(
        x=embedding[:, 0],
        y=embedding[:, 1],
        color=y.astype(str),
        title=title,
        labels={'x': 'Component 1', 'y': 'Component 2', 'color': label_name}
    )

    if is_thumbnail:
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showticklabels=False, visible=False),
            yaxis=dict(showticklabels=False, visible=False),
            showlegend=False,
            # title_text=title.split(' ')[0],
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
        ctx = callback_context
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'dataset-dropdown'

        # Initialize outputs
        main_fig = px.scatter(title="Select a dataset or method")
        trimap_thumb = px.scatter()
        tsne_thumb = px.scatter()
        umap_thumb = px.scatter()
        umap_warning = "" if umap_available else "UMAP is not installed. Run 'pip install umap-learn' to enable UMAP visualizations."
        metadata = html.Div()
        updated_cache = cached_embeddings if cached_embeddings is not None else {}
        trimap_timing = ""
        tsne_timing = ""
        umap_timing = ""

        # If custom_upload is selected, return empty figures and warning
        if dataset_name == 'custom_upload':
            return main_fig, trimap_thumb, tsne_thumb, umap_thumb, "Please use the 'Upload new datapoint' button to upload a custom dataset.", metadata, updated_cache, trimap_timing, tsne_timing, umap_timing

        # Get dataset
        X, y, data = get_dataset(dataset_name)
        label_name = "Label"
        key = random.PRNGKey(42)

        # Determine current method
        current_method = 'TRIMAP'
        if trigger_id == 'trimap-thumbnail-click':
            current_method = 'TRIMAP'
        elif trigger_id == 'tsne-thumbnail-click':
            current_method = 't-SNE'
        elif trigger_id == 'umap-thumbnail-click':
            current_method = 'UMAP'

        # Compute embeddings
        embeddings = {}
        timings = {}
        methods_to_compute = ['TRIMAP', 't-SNE', 'UMAP']

        for method in methods_to_compute:
            embedding_key = f"{dataset_name}_{method}"
            emb = None
            timing = 0
            
            if recalculate_flag or embedding_key not in updated_cache or updated_cache[embedding_key] is None:
                if method == 'TRIMAP':
                    emb, timing = compute_trimap(X, key)
                elif method == 't-SNE':
                    emb, timing = compute_tsne(X)
                elif method == 'UMAP':
                    emb, timing = compute_umap(X)

                if emb is not None:
                    updated_cache[embedding_key] = emb.tolist()
                    timings[method] = timing
                else:
                    updated_cache[embedding_key] = None
            else:
                emb = np.array(updated_cache[embedding_key]) if updated_cache[embedding_key] is not None else None

            embeddings[method] = emb

        # Create figures
        trimap_thumb = create_figure(embeddings['TRIMAP'], y, "TRIMAP", label_name, is_thumbnail=True)
        tsne_thumb = create_figure(embeddings['t-SNE'], y, "t-SNE", label_name, is_thumbnail=True)
        umap_thumb = create_figure(embeddings['UMAP'], y, "UMAP", label_name, is_thumbnail=True) if umap_available else px.scatter(title="UMAP not available")

        # Set main figure based on current method
        if current_method == 'UMAP' and not umap_available:
            main_fig = px.scatter(title=f"UMAP Embedding of {dataset_name} (not available)")
        else:
            main_fig = create_figure(embeddings[current_method], y, f"{current_method} Embedding of {dataset_name}", label_name)
        
        metadata = create_metadata_display(dataset_name, data)

        # Format timing information
        trimap_timing = f"{timings.get('TRIMAP', 0):.1f}s" if 'TRIMAP' in timings else ""
        tsne_timing = f"{timings.get('t-SNE', 0):.1f}s" if 't-SNE' in timings else ""
        umap_timing = f"{timings.get('UMAP', 0):.1f}s" if 'UMAP' in timings else ""

        return main_fig, trimap_thumb, tsne_thumb, umap_thumb, umap_warning, metadata, updated_cache, trimap_timing, tsne_timing, umap_timing

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
        Input('main-graph', 'clickData'),
        State('dataset-dropdown', 'value'),
        State('main-graph', 'figure')
    )
    def display_clicked_point(clickData, dataset_name, figure):
        if not clickData:
            # Return default states when nothing is clicked
            return (
                '', # selected-image src
                {'display': 'none'}, # selected-image style
                {'display': 'block', 'text-align': 'center', 'padding': '1rem'}, # no-image-message style
                "", # coordinates-display children
                "", # point-metadata children
                {'display': 'block', 'text-align': 'center', 'padding': '0.5rem', 'margin-bottom': '0.5rem', 'color': '#666'} # click-message style
            )
        
        point_index = clickData['points'][0]['pointIndex']
        X, y, data = get_dataset(dataset_name)
        
        # Get features to display from configuration
        features_to_display = DATASET_FEATURES.get(dataset_name, data.feature_names)
        
        # Get the real class label
        if hasattr(data, 'target_names'):
            class_label = data.target_names[y[point_index]]
        else:
            class_label = f"Class {y[point_index]}"
            
        # Get the coordinates from the figure
        x_coord = clickData['points'][0]['x']
        y_coord = clickData['points'][0]['y']
        
        # Get the color label from the figure
        color_label = clickData['points'][0].get('customdata', [class_label])[0] 
        
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
                    html.Td(color_label, style={'text-align': 'right', 'padding': '8px'})
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
        
        # For Digits dataset, create and display the image
        if dataset_name == "Digits":
            import base64
            from io import BytesIO
            
            # Create the image with proper aspect ratio
            plt.figure(figsize=(4, 4))
            plt.imshow(X[point_index].reshape(8, 8), cmap='gray')
            plt.axis('off')
            
            # Convert to base64
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            plt.close()
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode()
            
            return (
                f'data:image/png;base64,{img_str}',
                {'max-width': '100%', 'height': '22vh', 'object-fit': 'contain', 'display': 'block', 'padding': '0.5rem'},
                {'display': 'none'},
                coordinates_table,
                metadata_table,
                {'display': 'none'} # Hide click message
            )
        
        # For other datasets, show no image message and metadata
        return (
            '',
            {'display': 'none'},
            {'display': 'block', 'text-align': 'center', 'padding': '0.5rem', 'height': '22vh'},
            coordinates_table,
            metadata_table,
            {'display': 'none'} # Hide click message
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
            return 'secondary', [html.I(className="fas fa-lightbulb me-2"), "Generative Mode"] # Default off

        if current_color == 'secondary':
            return 'success', [html.I(className="fas fa-lightbulb me-2"), "Generative Mode (ON)"]
        else:
            return 'secondary', [html.I(className="fas fa-lightbulb me-2"), "Generative Mode"]

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

    # Existing callbacks (ensure they are still present after the edit)

    # Main callback to update graphs and metadata based on dataset-dropdown
    # ... (existing update_graphs and update_calculation_message) ...
    # (No changes to the existing update_graphs and update_calculation_message, but keeping this comment for clarity)

    # Note: The callbacks for top-grid and bottom-grid buttons were removed in the layout update.
    # If new buttons are added that need similar functionality, new callbacks will be needed.
