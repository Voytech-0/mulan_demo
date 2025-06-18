"""
Callbacks for the MULAN demo app.
Handles all interactive functionality and data processing.
"""

import jax.random as random
import dash
from dash import dcc, html, Input, Output, callback_context, State, callback, ClientsideFunction
import numpy as np
import threading

# Import utility modules
from .dataset_loader import get_dataset
from .embedding_computation import (
    compute_trimap, compute_tsne, compute_umap, 
    get_embedding_computation_function, check_method_availability
)
from .visualization_utils import create_figure, create_metadata_table, create_datapoint_image
from .embedding_storage import save_embedding, load_embedding, embedding_exists
from .feature_config import IMAGE_ONLY_DATASETS
from .settings import (
    get_image_style, get_generative_placeholder_style, get_no_image_message_style
)

# Global lock for computations
computation_lock = threading.Lock()

# Simple cache for generated images
_image_cache = {}


def register_callbacks(app):
    """Register all callbacks for the application."""
    
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
        Input('dots-images-switch', 'value'),
        Input('trimap-thumbnail-click', 'n_clicks'),
        Input('tsne-thumbnail-click', 'n_clicks'),
        Input('umap-thumbnail-click', 'n_clicks'),
        State('embedding-cache', 'data')
    )
    def update_graphs(dataset_name, recalculate_flag, show_images, 
                     trimap_n_clicks, tsne_n_clicks, umap_n_clicks, cached_embeddings):
        """Update all graphs based on dataset and method selection."""
        
        if not dataset_name:
            return [create_figure(None, [], "No dataset selected", "Label")] * 3 + ["", "", {}] + [""] * 3
        
        # Determine which method was clicked
        ctx = callback_context
        if not ctx.triggered:
            selected_method = 'trimap'  # Default
        else:
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if 'trimap' in trigger_id:
                selected_method = 'trimap'
            elif 'tsne' in trigger_id:
                selected_method = 'tsne'
            elif 'umap' in trigger_id:
                selected_method = 'umap'
            else:
                selected_method = 'trimap'
        
        # Load dataset
        try:
            X, y, data = get_dataset(dataset_name)
        except Exception as e:
            error_fig = create_figure(None, [], f"Error loading dataset: {str(e)}", "Label")
            return [error_fig] * 3 + ["", "", {}] + [""] * 3
        
        # Initialize cache if needed
        if cached_embeddings is None:
            cached_embeddings = {}
        
        # Check UMAP availability
        umap_warning = ""
        if not check_method_availability('umap'):
            umap_warning = "UMAP is not available. Please install it with: pip install umap-learn"
        
        # Compute embeddings
        embeddings = {}
        timings = {}
        
        for method in ['trimap', 'tsne', 'umap']:
            if not check_method_availability(method):
                embeddings[method] = None
                timings[method] = "Not available"
                continue
            
            # Check cache first
            cache_key = f"{dataset_name}_{method}"
            if not recalculate_flag and cache_key in cached_embeddings:
                embeddings[method] = cached_embeddings[cache_key]['embedding']
                timings[method] = cached_embeddings[cache_key].get('timing', 'Cached')
            else:
                try:
                    if method == 'trimap':
                        key = random.PRNGKey(42)
                        embedding, timing = compute_trimap(X, key)
                    elif method == 'tsne':
                        embedding, timing = compute_tsne(X)
                    elif method == 'umap':
                        embedding, timing = compute_umap(X)
                    
                    embeddings[method] = embedding
                    timings[method] = f"{timing:.2f}s"
                    
                    # Cache the result
                    cached_embeddings[cache_key] = {
                        'embedding': embedding,
                        'timing': timings[method]
                    }
                    
                except Exception as e:
                    print(f"Error computing {method}: {e}")
                    embeddings[method] = None
                    timings[method] = f"Error: {str(e)}"
        
        # Create figures
        main_fig = create_figure(
            embeddings[selected_method], y[:embeddings[selected_method].shape[0]], 
            f"{selected_method.upper()} - {dataset_name}", 
            "Class", X, False, show_images, class_names=data.target_names
        )
        
        trimap_fig = create_figure(
            embeddings['trimap'], y[:embeddings['trimap'].shape[0]], "TRIMAP", "Class", X, True, False, class_names=data.target_names
        )
        
        tsne_fig = create_figure(
            embeddings['tsne'], y[:embeddings['tsne'].shape[0]], "t-SNE", "Class", X, True, False, class_names=data.target_names
        )
        
        umap_fig = create_figure(
            embeddings['umap'], y[:embeddings['umap'].shape[0]], "UMAP", "Class", X, True, False, class_names=data.target_names
        )
        
        # Create metadata display
        metadata_display = create_metadata_table(dataset_name, data)
        
        return [
            main_fig, trimap_fig, tsne_fig, umap_fig,
            umap_warning, metadata_display, cached_embeddings,
            timings['trimap'], timings['tsne'], timings['umap']
        ]
    
    @app.callback(
        Output('calculation-status', 'children'),
        Input('dataset-dropdown', 'value'),
        Input('trimap-thumbnail-click', 'n_clicks'),
        Input('tsne-thumbnail-click', 'n_clicks'),
        Input('umap-thumbnail-click', 'n_clicks'),
        Input('recalculate-switch', 'value'),
        State('embedding-cache', 'data')
    )
    def update_calculation_message(dataset_name, trimap_n_clicks, tsne_n_clicks, 
                                 umap_n_clicks, recalculate_flag, cached_embeddings):
        """Update calculation status message."""
        
        if not dataset_name:
            return ""
        
        ctx = callback_context
        if not ctx.triggered:
            return ""
        
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if 'dataset-dropdown' in trigger_id:
            return f"Loading dataset: {dataset_name}"
        elif 'recalculate-switch' in trigger_id:
            if recalculate_flag:
                return "Recalculating embeddings..."
            else:
                return "Using cached embeddings"
        elif any(method in trigger_id for method in ['trimap', 'tsne', 'umap']):
            method = next(method for method in ['trimap', 'tsne', 'umap'] if method in trigger_id)
            return f"Switched to {method.upper()}"
        
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
        """Display details of clicked point."""
        
        if not clickData or not dataset_name:
            return "", get_image_style('none'), get_no_image_message_style('block'), "", "", {}, get_generative_placeholder_style('none')
        
        # Check if in generative mode
        if generative_state and generative_state.get('enabled', False):
            return "", get_image_style('none'), get_no_image_message_style('none'), "", "", {}, get_generative_placeholder_style('block')
        
        try:
            point_index = clickData['points'][0]['pointIndex']
            X, y, data = get_dataset(dataset_name)
            
            # Create image if it's an image dataset
            if dataset_name in IMAGE_ONLY_DATASETS:
                img_src = create_datapoint_image(X[point_index])
                img_style = get_image_style('block')
                no_img_style = get_no_image_message_style('none')
            else:
                img_src = ""
                img_style = get_image_style('none')
                no_img_style = get_no_image_message_style('block')
            
            # Create coordinates display
            if 'figure' in figure and 'data' in figure:
                coords = figure['data'][0]
                if 'x' in coords and 'y' in coords:
                    x_val = coords['x'][point_index]
                    y_val = coords['y'][point_index]
                    coords_display = f"X: {x_val:.4f}\nY: {y_val:.4f}"
                else:
                    coords_display = "Coordinates not available"
            else:
                coords_display = "Coordinates not available"
            
            # Create metadata display
            metadata_display = create_metadata_table(dataset_name, data, point_index)
            
            # Update click message
            click_message_style = {'text-align': 'center', 'padding': '0.5rem', 'margin-bottom': '0.5rem', 'color': '#28a745'}
            
            return img_src, img_style, no_img_style, coords_display, metadata_display, click_message_style, get_generative_placeholder_style('none')
            
        except Exception as e:
            print(f"Error displaying clicked point: {e}")
            return "", get_image_style('none'), get_no_image_message_style('block'), "", "", {}, get_generative_placeholder_style('none')
    
    @app.callback(
        Output('generative-mode-btn', 'color'),
        Output('generative-mode-btn', 'children'),
        Output('generative-mode-state', 'data'),
        Input('generative-mode-btn', 'n_clicks'),
        State('generative-mode-state', 'data'),
        prevent_initial_call=True
    )
    def toggle_generative_mode(n_clicks, current_state):
        """Toggle generative mode on/off."""
        
        if current_state is None:
            current_state = {'enabled': False}
        
        new_state = not current_state.get('enabled', False)
        
        if new_state:
            return "success", [html.I(className="fas fa-lightbulb me-2"), "Generative Mode (ON)"], {'enabled': True}
        else:
            return "primary", [html.I(className="fas fa-lightbulb me-2"), "Generative Mode"], {'enabled': False}
    
    @app.callback(
        Output('iteration-slider', 'className'),
        Output('slider-play-btn', 'style'),
        Input('generative-mode-state', 'data'),
        prevent_initial_call=False
    )
    def toggle_iterative_slider(generative_state):
        """Toggle iterative slider visibility based on generative mode."""
        
        if generative_state and generative_state.get('enabled', False):
            return 'd-none', {'display': 'none'}
        else:
            return 'mb-3', {'margin-right': '0.5rem'}
    
    @app.callback(
        Output('iteration-process-container', 'style'),
        Input('generative-mode-state', 'data'),
        prevent_initial_call=False
    )
    def toggle_iterative_process_label(generative_state):
        """Toggle iterative process container visibility."""
        
        if generative_state and generative_state.get('enabled', False):
            return {'display': 'none'}
        else:
            return {'margin-bottom': '1rem'}
    
    @app.callback(
        [Output(f'{method}-thumbnail-click', 'className') for method in ['tsne', 'umap', 'trimap']],
        [Input(f'{method}-thumbnail-click', 'n_clicks') for method in ['tsne', 'umap', 'trimap']],
        prevent_initial_call=True
    )
    def select_method_button(tsne_n_clicks, umap_n_clicks, trimap_n_clicks):
        """Update method button selection states."""
        
        ctx = callback_context
        if not ctx.triggered:
            return ["method-button-container thumbnail-button mb-3"] * 3
        
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        classes = []
        for method in ['tsne', 'umap', 'trimap']:
            if f'{method}-thumbnail-click' in trigger_id:
                classes.append("method-button-container thumbnail-button mb-3 selected")
            else:
                classes.append("method-button-container thumbnail-button mb-3")
        
        return classes
    
    @app.callback(
        Output('umap-warning', 'children', allow_duplicate=True), 
        Input('dataset-dropdown', 'value'),
        prevent_initial_call=True
    )
    def handle_custom_dataset_upload_dropdown(dataset_value):
        """Handle custom dataset upload dropdown selection."""
        
        if dataset_value == 'custom_upload':
            return "Custom dataset upload functionality coming soon!"
        return ""
    
    @app.callback(
        Output('calculation-status', 'children', allow_duplicate=True), 
        Input('upload-new-datapoint-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def handle_upload_new_datapoint_button(n_clicks):
        """Handle new datapoint upload button click."""
        
        return "New datapoint upload functionality coming soon!"
    
    @app.callback(
        Output('point-metadata-row', 'style'),
        Input('dataset-dropdown', 'value'),
        prevent_initial_call=False
    )
    def toggle_metadata_row(dataset_name):
        """Toggle metadata row visibility based on dataset type."""
        
        if dataset_name in IMAGE_ONLY_DATASETS:
            return {'display': 'none'}
        else:
            return {'display': 'block'}
    
    @app.callback(
        Output('main-graph-images', 'data'),
        Input('main-graph', 'figure'),
        State('dots-images-switch', 'value')
    )
    def store_image_data(figure, show_images):
        """Store image data for main graph."""
        
        if not show_images or not figure or 'data' not in figure:
            return {}
        
        return {'figure': figure, 'show_images': show_images}
    
    @app.callback(
        Output('main-graph', 'figure', allow_duplicate=True),
        Input('main-graph', 'relayoutData'),
        State('main-graph', 'figure'),
        State('main-graph-images', 'data'),
        State('dots-images-switch', 'value'),
        prevent_initial_call=True
    )
    def update_image_sizes_on_zoom(relayout_data, figure, stored_images, show_images):
        """Update image sizes when zooming."""
        
        if not show_images or not stored_images or 'figure' not in stored_images:
            return figure
        
        # This is a placeholder for image size adjustment on zoom
        # In a full implementation, you would adjust image sizes based on zoom level
        return figure
