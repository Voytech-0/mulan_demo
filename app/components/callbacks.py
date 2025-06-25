from functools import partial

import dash
import matplotlib
import numpy as np
from dash import html, Input, Output, callback_context, State

from components.configs.feature_config import DATASET_FEATURES
from components.configs.settings import get_image_style, get_no_image_message_style, get_no_metadata_message_style, \
    get_generative_placeholder_style
from components.data_operations.dataset_api import get_dataset, dataset_shape
from components.configs.feature_config import IMAGE_ONLY_DATASETS
from components.slow_backend_operations.added_features_api import dynamically_add, generate_sample, extract_added_data
from components.slow_backend_operations.embedding_calculation import compute_tsne, compute_umap, compute_trimap, \
    compute_all_embeddings
from components.visualization_generators.layout_generators import create_metadata_display, create_coordinate_table, \
    create_metadata_table
from components.visualization_generators.plot_helpers import encode_img_as_str, match_shape
from components.visualization_generators.plot_maker import create_animated_figure, create_figure, empty_fig, \
    create_datapoint_image

matplotlib.use('Agg')  # Use non-interactive backend
from dash_canvas.utils import parse_jsonstring
from math import ceil, floor
import cv2

# Import TRIMAP from local package
trimap_cache_name = 'trimap_cache'
# Try to import UMAP, if available


def register_callbacks(app):
    @app.callback(
        Output('main-graph-static', 'figure'),
        Output('main-graph-static', 'style'),
        Output('main-graph-animated', 'figure'),
        Output('main-graph-animated', 'style'),
        Input('dataset-dropdown', 'value'),
        Input('dots-images-switch', 'value'),
        Input('focused-embedding', 'data'),
        Input('added-data-cache', 'data'),
        Input('dist-dropdown', 'value'),
        Input('parametric-iterative-switch', 'value'),
        Input('is-animated-switch', 'value'),
    )
    def update_main_figure(dataset_name, show_images, method,
                           added_data_cache, distance, parametric, is_animated=False):
        X, y, data = get_dataset(dataset_name)
        fwd_args = (dataset_name, distance, parametric, is_animated)
        (trimap_emb, tsne_emb, umap_emb), _ = compute_all_embeddings(*fwd_args)
        class_names = getattr(data, 'target_names', None)
        X_add, y_add, n_added = extract_added_data(added_data_cache)
        if n_added > 0:
            trimap_emb_add = dynamically_add(added_data_cache, *fwd_args)
            X, y, trimap_emb = (np.concatenate([X, X_add], 0), np.concatenate([y, y_add], 0),
                                np.concatenate([trimap_emb, trimap_emb_add], 0))

        if method != 'trimap' or not is_animated:
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
            main_fig_static = {}
            main_fig_animated = create_animated_figure(trimap_emb, y, f"TRIMAP Embedding of {dataset_name}", 'Class')

        if is_animated:
            static_style = {'display': 'none'}
            animated_style = {'display': 'block', 'height': '60vh'}
        else:
            static_style = {'display': 'block', 'height': '60vh'}
            animated_style = {'display': 'none'}
        return main_fig_static, static_style, main_fig_animated, animated_style,

    @app.callback(
        Output('trimap-thumbnail', 'figure'),
        Output('tsne-thumbnail', 'figure'),
        Output('umap-thumbnail', 'figure'),
        Input('dataset-dropdown', 'value'),
        Input('dist-dropdown', 'value'),
        Input('parametric-iterative-switch', 'value'),
    )
    def update_thumbnails(dataset_name, distance, parametric):
        X, y, data = get_dataset(dataset_name)

        fwd_args = (dataset_name, distance, parametric)
        (trimap_emb, tsne_emb, umap_emb), _ = compute_all_embeddings(*fwd_args)

        trimap_fig = create_figure(trimap_emb, y, "TRIMAP", "Class", X, is_thumbnail=True, show_images=False)
        tsne_fig = create_figure(tsne_emb, y, "t-SNE", "Class", X, is_thumbnail=True, show_images=False)
        umap_fig = create_figure(umap_emb, y, "UMAP", "Class", X, is_thumbnail=True, show_images=False)
        return trimap_fig, tsne_fig, umap_fig

    @app.callback(
        Output('trimap-timing', 'children'),
        Output('tsne-timing', 'children'),
        Output('umap-timing', 'children'),
        Input('dataset-dropdown', 'value'),
        Input('dist-dropdown', 'value'),
        Input('parametric-iterative-switch', 'value'),
    )
    def time_methods(dataset_name, distance, parametric):
        _, (trimap_time, tsne_time, umap_time) = compute_all_embeddings(dataset_name, distance, parametric)
        return (f"TRIMAP: {trimap_time:.2f}s",
                f"t-SNE: {tsne_time:.2f}s",
                f"UMAP: {umap_time:.2f}s")

    @app.callback(
        Output('metadata-display', 'children'),
        Output('calculation-status', 'children'),
        Input('dataset-dropdown', 'value'),
        Input('focused-embedding', 'data'),
    )
    def update_metadata(dataset_name, method):
        X, y, data = get_dataset(dataset_name)
        metadata = create_metadata_display(dataset_name, data)
        ctx = callback_context
        calc_status = ""
        if not ctx.triggered:
            calc_status = f"Loaded dataset {dataset_name}"  # Initial state, no message

        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        # For dataset loading
        if trigger_id == 'dataset-dropdown':
            calc_status = f"Loaded dataset {dataset_name}"

        # For thumbnail clicks
        elif trigger_id == 'focused-embedding':
            calc_status = f"Calculated {method} embedding"

        return metadata, calc_status



    # Mutually exclusive switches callback
    @app.callback(
        Output('is-animated-switch', 'value'),
        Output('parametric-iterative-switch', 'value'),
        Output('dots-images-switch', 'value'),
        Output('upload-new-datapoint-btn', 'disabled'),
        Input('is-animated-switch', 'value'),
        Input('parametric-iterative-switch', 'value'),
        Input('dots-images-switch', 'value'),
        prevent_initial_call=True
    )
    def mutually_exclusive_switches(is_animated, parametric_iterative, dots_images):
        ctx = callback_context
        if not ctx.triggered:
            # No trigger, return current values
            return is_animated, parametric_iterative, dots_images, is_animated
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        # If is-animated-switch turned on, turn off the other two
        if trigger_id == "is-animated-switch" and is_animated:
            return True, False, False, True
        # If parametric-iterative-switch turned on, turn off is-animated
        if trigger_id == "parametric-iterative-switch" and parametric_iterative:
            return False, True, dots_images, False
        # If dots-images-switch turned on, turn off is-animated
        if trigger_id == "dots-images-switch" and dots_images:
            return False, parametric_iterative, True, False
        # Otherwise, return current values
        return is_animated, parametric_iterative, dots_images, is_animated


    @app.callback(
        Output('selected-image', 'src'),
        Output('selected-image', 'style'),
        Output('no-image-message', 'style'),
        Output('no-image-message', 'children'),
        Output('coordinates-display', 'children'),
        Output('point-metadata', 'children'),
        Output('no-metadata-message', 'style'),
        Output('no-metadata-message', 'children'),
        Output('generative-mode-placeholder', 'style'),
        Output('point-metadata-row', 'style'),
        Output('coordinates-col', 'style'),
        Output('full-grid-container', 'style'),
        Output('image-display', 'style'),
        Output('image-draw', 'style'),
        Output('full-grid-visible', 'data'),
        Input('main-graph-static', 'clickData'),
        Input('generative-mode-state', 'data'),
        Input('dataset-dropdown', 'value'),
        Input('full-grid-btn', 'n_clicks'),
        Input('upload-new-datapoint-btn', 'n_clicks'),
        State('dataset-dropdown', 'value'),
        State('main-graph-static', 'figure'),
        State('embedding-cache', 'data'),
        State('full-grid-container', 'style'),
        State('image-display', 'style'),
        State('image-draw', 'style'),
        State('full-grid-visible', 'data'),
        State('parametric-iterative-switch', 'value'),
        State('dist-dropdown', 'value'),
        prevent_initial_call=True
    )
    def display_or_toggle(clickData, generative_state, dataset_name, last_clicked_dataset, figure, embedding_cache, full_grid_clicks, current_grid_style, upload_new_n_clicks, image_display_style, image_draw_style, full_grid_visible,
                          parametric, distance):
        enabled = generative_state.get('enabled', False) if generative_state else False
        triggered = callback_context.triggered_id
        is_image_dataset = dataset_name in IMAGE_ONLY_DATASETS
        image_message = "Click on a point in the graph to display image" if is_image_dataset else "No image to display for this dataset"
        metadata_message = "No metadata to show in this dataset" if is_image_dataset else "Click on a point in the graph to show metadata"

        # Handle full grid toggle
        if triggered == "full-grid-btn":
            is_opening = not full_grid_visible

            # When opening grid: hide all metadata/image blocks
            if is_opening:
                hidden = {'display': 'none'}
                return (
                    "",  # selected-image src
                    hidden,  # selected-image style
                    hidden,  # no-image-message style
                    "",      # no-image-message text
                    "",      # coordinates
                    "",      # metadata
                    hidden,  # no-metadata style
                    "",      # no-metadata message
                    hidden,  # placeholder
                    hidden,  # row
                    hidden,  # col
                    {"display": "block"},
                    hidden,  # unchanged
                    image_draw_style,
                    True
                )

            else:
                # When closing the full grid, reset to default (no click) state
                return (
                    '',  # selected-image src
                    get_image_style('none'),  # selected-image style
                    get_no_image_message_style('block'), # no-image-message style
                    html.Div(image_message, style={'text-align': 'center', 'color': '#999'}),
                    html.Div("Click on a point in the graph to show coordinates", style={'text-align': 'center', 'color': '#999', 'marginTop': '3rem'}),
                    "",  # point-metadata children
                    get_no_metadata_message_style('block'),
                    html.Div(metadata_message, style={'text-align': 'center', 'color': '#999', 'marginTop': '3rem'}),
                    get_generative_placeholder_style('none'),
                    {'display': 'block'},
                    {'marginTop': '2rem'},
                    {'display': 'none'},  # hide full grid
                    {'display': 'block'},
                    image_draw_style,
                    False
                )
        elif triggered == 'upload-new-datapoint-btn':
            # Toggle image-display and image-draw visibility
            display_visible = image_display_style.get("display") != "none"
            new_image_display_style = {'display': 'none'} if display_visible else {'display': 'block'}
            new_image_draw_style = {'display': 'block'} if display_visible else {'display': 'none'}

            return (
                dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                dash.no_update, dash.no_update, dash.no_update,
                dash.no_update,  # full-grid unchanged
                new_image_display_style,
                new_image_draw_style,
                False
            )



        if not clickData:
            is_image_dataset = dataset_name in IMAGE_ONLY_DATASETS
            image_message = "Click on a point in the graph to display image" if is_image_dataset else "No image to display for this dataset"
            metadata_message = "No metadata to show in this dataset" if is_image_dataset else "Click on a point in the graph to show metadata"
            # Return default states when nothing is clicked
            return (
                '',  # selected-image src
                get_image_style('none'),  # selected-image style
                get_no_image_message_style('block'), # no-image-message style
                html.Div(image_message,  style={'text-align': 'center', 'color': '#999'}),
                html.Div("Click on a point in the graph to show coordinates", style={'text-align': 'center', 'color': '#999', 'marginTop': '3rem'}),  # coordinates-display children
                "",  # point-metadata children
                get_no_metadata_message_style('block'),
                html.Div(metadata_message,  style={'text-align': 'center', 'color': '#999', 'marginTop': '3rem'}),
                get_generative_placeholder_style('none'),
                {'display':'block'},  # generative-mode-placeholder style
                {'marginTop': '2rem'},
                {'display': 'none'},
                {'display': 'block'},  # unchanged
                image_draw_style,
                False
            )

        if enabled:
            x_coord = clickData['points'][0]['x']
            y_coord = clickData['points'][0]['y']
            sample = generate_sample(x_coord, y_coord, dataset_name, distance, parametric=parametric)
            sample = match_shape(sample, dataset_name)
            img_str = encode_img_as_str(sample)
            # In generative mode, show placeholder and hide other image elements
            return (
                f'data:image/png;base64,{img_str}',
                get_image_style('block'), # selected-image style
                get_no_image_message_style('block'), # no-image-message style
                html.Div("No image to display for this dataset", style={'text-align': 'center', 'color': '#999'}),
                "",
                "",
                get_no_metadata_message_style('none'),
                "",
                get_generative_placeholder_style('block'), # generative-mode-placeholder style
                {'display': 'none'},
                {'marginTop': '0.5rem'},
                {'display': 'none'},
                image_display_style,  # unchanged
                image_draw_style,
                False
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
        coordinates_table = create_coordinate_table(x_coord, y_coord, point_index, class_label, digit_label)

        # Create metadata table
        metadata_table = create_metadata_table(features_to_display, X, point_index)

        # For image datasets (Digits, MNIST, Fashion MNIST, Elephant), create and display the image
        if dataset_name in ["Digits", "MNIST", "Fashion MNIST", "Elephant", "PACS - Photo", "PACS - Cartoon", "PACS - Art Painting"]:
            sample = match_shape(X[point_index], dataset_name)
            img_str = encode_img_as_str(sample)

            # For image-only datasets, show empty metadata
            if dataset_name in IMAGE_ONLY_DATASETS:
                metadata_content = ""
                no_metadata_style = {'display': 'block', 'text-align': 'center', 'color': '#999', 'padding': '1rem'}
            else:
                metadata_content = metadata_table
                no_metadata_style = {'display': 'none'}

            return (
                f'data:image/png;base64,{img_str}',
                get_image_style('block'),
                get_no_image_message_style('block'),
                "",
                coordinates_table,
                metadata_content,
                no_metadata_style,
                "", # Hide click message
                get_generative_placeholder_style('none'), # Hide generative mode placeholder
                {'display': 'none'},
                {'marginTop': '2rem'},
                {'display': 'none'},
                image_display_style,  # unchanged
                image_draw_style,
                False
            )

        # For other datasets, show no image message and metadata
        # For image-only datasets, show empty metadata
        if dataset_name in IMAGE_ONLY_DATASETS:
                metadata_content = ""
                no_metadata_style = {'display': 'block', 'text-align': 'center', 'color': '#999', 'padding': '1rem'}
        else:
            metadata_content = metadata_table
            no_metadata_style = {'display': 'none'}

        return (
            '',
            get_image_style('none'),
            get_no_image_message_style('block'),
            html.Div("No image to display for this dataset", style={'text-align': 'center', 'color': '#999'}),
            coordinates_table,
            metadata_content,
            no_metadata_style,
            "", # Hide click message
            get_generative_placeholder_style('none'), # Hide generative mode placeholder
            {'display': 'block'},
            {'marginTop': '2rem'},
            {'display': 'none'},
            image_display_style,  # unchanged
            image_draw_style,
            False
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

    @app.callback(
        Output('focused-embedding', 'data'),
        Input('tsne-thumbnail-click', 'n_clicks'),
        Input('umap-thumbnail-click', 'n_clicks'),
        Input('trimap-thumbnail-click', 'n_clicks'),
        prevent_initial_call=True,
    )
    def switch_focused_embedding(tsne_n_clicks, umap_n_clicks, trimap_n_clicks):
        ctx = callback_context
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        button_id_2_focused_embedding = {
            'tsne-thumbnail-click': 'tsne',
            'umap-thumbnail-click': 'umap',
            'trimap-thumbnail-click': 'trimap'
        }
        return button_id_2_focused_embedding[button_id]

    # Method Buttons (Mutually Exclusive)
    @app.callback(
        Output('tsne-thumbnail-click', 'className'),
        Output('umap-thumbnail-click', 'className'),
        Output('trimap-thumbnail-click', 'className'),
        Input('focused-embedding', 'data'),
        prevent_initial_call=True
    )
    def select_method_button(focused_embedding):
        tsne_class = "method-button-container thumbnail-button mb-3"
        umap_class = "method-button-container thumbnail-button mb-3"
        trimap_class = "method-button-container thumbnail-button mb-3"

        if focused_embedding == 'tsne':
            tsne_class = "method-button-container thumbnail-button mb-3 selected"
        elif focused_embedding == 'umap':
            umap_class = "method-button-container thumbnail-button mb-3 selected"
        elif focused_embedding == 'trimap':
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
        Output('image-display', 'style', allow_duplicate=True),
        Output('image-draw', 'style', allow_duplicate=True),
        Input('upload-new-datapoint-btn', 'n_clicks'),
        prevent_initial_call=True,
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
            return dash.no_update, dash.no_updateInput

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

   # @app.callback(
   #     Output("full-grid-container", "style"),
    #    Output('selected-image', 'style'),
     #   Output('no-image-message', 'style'),
      #  Output('coordinates-display', 'style'),
      #  Output('point-metadata', 'style'),
      #  Output('no-metadata-message', 'style'),
      #  Output('generative-mode-placeholder', 'style'),
      #  Output('point-metadata-row', 'style'),
      #  Output('coordinates-col', 'style'),
      #  [
      #      Input("full-grid-btn", "n_clicks"),
      #      Input("main-graph-static", "clickData")
      #  ],
      #  State("full-grid-container", "style"),
      #  prevent_initial_call=True
    #)
    #def toggle_full_grid(n_clicks, click_data, current_style):
    #    triggered = callback_context.triggered_id

       # if triggered == "full-grid-btn":
            # Toggle the grid open/close
      #      if current_style and current_style.get("display") == "block":
       #         return {"display": "none"}
        #    return {"display": "block"}

        #elif triggered == "main-graph" and click_data is not None:
            # Hide grid after clicking on image
         #   return {"display": "none"}

        #return current_style

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

        elif family == "testing":
            options = [
                {"label": "S-curve", "value": "Testing - S-curve"},
                {"label": "Swiss Roll", "value": "Testing - Swiss Roll"},
                {"label": "Mammoth", "value": "Testing - Mammoth"},
            ]
            return options, "Testing - S-curve"

        elif family == "custom_upload":
            return [{"label": "Upload Custom Dataset", "value": "custom_upload"}], "custom_upload"

        return [], None

    # Reset when changing datasets
    @app.callback(
        Output('main-graph-static', 'clickData'),
        Input('dataset-dropdown', 'value')
    )
    def reset_click_on_dataset_change(_):
        return None


    # Existing callbacks (ensure they are still present after the edit)

    # Main callback to update graphs and metadata based on dataset-dropdown
    # ... (existing update_graphs and update_calculation_message) ...
    # (No changes to the existing update_graphs and update_calculation_message, but keeping this comment for clarity)

    # Note: The callbacks for top-grid and bottom-grid buttons were removed in the layout update.
    # If new buttons are added that need similar functionality, new callbacks will be needed.
