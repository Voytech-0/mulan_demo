from math import ceil, floor

import numpy as np
from dash import dash, Output, Input, State, html, callback_context
from dash_canvas.utils import parse_jsonstring

from components.configs.feature_config import IMAGE_ONLY_DATASETS, DATASET_FEATURES
from components.configs.settings import get_image_style, get_no_image_message_style, get_no_metadata_message_style, \
    get_generative_placeholder_style
from components.data_operations.dataset_api import dataset_shape, get_dataset
import cv2

from components.slow_backend_operations.added_features_api import generate_sample
from components.visualization_generators.layout_generators import create_coordinate_table, create_metadata_table
from components.visualization_generators.plot_helpers import match_shape, encode_img_as_str


def register_visualization_callbacks(app):
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
        Input('grid-btn', 'n_clicks'),
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
    def display_or_toggle(clickData, generative_state, dataset_name, last_clicked_dataset, figure, embedding_cache,
                          full_grid_clicks, current_grid_style, upload_new_n_clicks, image_display_style,
                          image_draw_style, full_grid_visible,
                          parametric, distance):
        enabled = generative_state.get('enabled', False) if generative_state else False
        triggered = callback_context.triggered_id
        is_image_dataset = dataset_name in IMAGE_ONLY_DATASETS
        image_message = "Click on a point in the graph to display image" if is_image_dataset else "No image to display for this dataset"
        metadata_message = "No metadata to show in this dataset" if is_image_dataset else "Click on a point in the graph to show metadata"

        # Handle full grid toggle
        if triggered == "grid-btn":
            is_opening = not full_grid_visible

            # When opening grid: hide all metadata/image blocks
            if is_opening:
                hidden = {'display': 'none'}
                return (
                    "",  # selected-image src
                    hidden,  # selected-image style
                    hidden,  # no-image-message style
                    "",  # no-image-message text
                    "",  # coordinates
                    "",  # metadata
                    hidden,  # no-metadata style
                    "",  # no-metadata message
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
                    get_no_image_message_style('block'),  # no-image-message style
                    html.Div(image_message, style={'text-align': 'center', 'color': '#999'}),
                    html.Div("Click on a point in the graph to show coordinates",
                             style={'text-align': 'center', 'color': '#999', 'marginTop': '3rem'}),
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
                get_no_image_message_style('block'),  # no-image-message style
                html.Div(image_message, style={'text-align': 'center', 'color': '#999'}),
                html.Div("Click on a point in the graph to show coordinates",
                         style={'text-align': 'center', 'color': '#999', 'marginTop': '3rem'}),
                # coordinates-display children
                "",  # point-metadata children
                get_no_metadata_message_style('block'),
                html.Div(metadata_message, style={'text-align': 'center', 'color': '#999', 'marginTop': '3rem'}),
                get_generative_placeholder_style('none'),
                {'display': 'block'},  # generative-mode-placeholder style
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
                get_image_style('block'),  # selected-image style
                get_no_image_message_style('block'),  # no-image-message style
                html.Div("No image to display for this dataset", style={'text-align': 'center', 'color': '#999'}),
                "",
                "",
                get_no_metadata_message_style('none'),
                "",
                get_generative_placeholder_style('block'),  # generative-mode-placeholder style
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
        features_to_display = DATASET_FEATURES.get(dataset_name, getattr(data, 'feature_names',
                                                                         [f'Feature {i}' for i in range(X.shape[1])]))

        # Get the real class label
        if hasattr(data, 'target_names'):
            # If y is integer index, map to class name
            class_label = data.target_names[y[point_index]] if int(y[point_index]) < len(data.target_names) else str(
                y[point_index])
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
        if dataset_name in ["Digits", "MNIST", "Fashion MNIST", "Elephant", "PACS - Photo", "PACS - Cartoon",
                            "PACS - Art Painting"]:
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
                "",  # Hide click message
                get_generative_placeholder_style('none'),  # Hide generative mode placeholder
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
            "",  # Hide click message
            get_generative_placeholder_style('none'),  # Hide generative mode placeholder
            {'display': 'block'},
            {'marginTop': '2rem'},
            {'display': 'none'},
            image_display_style,  # unchanged
            image_draw_style,
            False
        )