from math import ceil, floor

import cv2
import numpy as np
from dash import dash, Output, Input, State, html
from dash_canvas.utils import parse_jsonstring

from components.configs.feature_config import IMAGE_ONLY_DATASETS, DATASET_FEATURES
from components.configs.settings import get_image_style, get_no_image_message_style, get_no_metadata_message_style
from components.data_operations.dataset_api import dataset_shape, get_dataset
from components.slow_backend_operations.added_features_api import generate_sample
from components.visualization_generators.layout_generators import create_coordinate_table, create_metadata_table
from components.visualization_generators.plot_helpers import match_shape, encode_img_as_str


def _extract_coords(clickData):
    x_coord = clickData['points'][0]['x']
    y_coord = clickData['points'][0]['y']
    return x_coord, y_coord

def register_visualization_callbacks(app):
    @app.callback(
        Output('selected-image', 'src'),
        Output('selected-image', 'style'),
        Output('no-image-message', 'style'),
        Output('no-image-message', 'children'),

        Input('main-graph-static', 'clickData'),
        Input('generative-mode-state', 'data'),
        Input('dataset-dropdown', 'value'),
        State('parametric-iterative-switch', 'value'),
        State('dist-dropdown', 'value'),
        prevent_initial_call=True
    )
    def display_datum_as_image(clickData, generative_state, dataset_name, parametric, distance):
        invalid = not clickData or not dataset_name in IMAGE_ONLY_DATASETS
        image_message = "Click on a point in the graph to display image" if dataset_name in IMAGE_ONLY_DATASETS else\
            "No image to display for this dataset"

        if invalid:
            return (
                '',  # selected-image src
                get_image_style('none'),  # selected-image style
                get_no_image_message_style('block'),  # no-image-message style
                html.Div(image_message, style={'text-align': 'center', 'color': '#999'}),
            )

        x_coord, y_coord = _extract_coords(clickData)

        enabled = generative_state.get('enabled', False) if generative_state else False
        if enabled:
            sample = generate_sample(x_coord, y_coord, dataset_name, distance, parametric=parametric)
            sample = match_shape(sample, dataset_name)
            img_str = encode_img_as_str(sample)
            # In generative mode, show placeholder and hide other image elements
            return (
                f'data:image/png;base64,{img_str}',
                get_image_style('block'),  # selected-image style
                get_no_image_message_style('block'),  # no-image-message style
                html.Div("No image to display for this dataset", style={'text-align': 'center', 'color': '#999'}),
            )

        # defensive -> check if point data in custom data
        point_data = clickData['points'][0]
        if 'customdata' in point_data:
            point_index = int(point_data['customdata'][0])
        else:
            point_index = point_data.get('pointIndex', 0)

        X, _, _ = get_dataset(dataset_name)

        sample = match_shape(X[point_index], dataset_name)
        img_str = encode_img_as_str(sample)

        return (
            f'data:image/png;base64,{img_str}',
            get_image_style('block'),
            get_no_image_message_style('block'),
            "")

    @app.callback(
        Output('point-metadata', 'children'),
        Output('no-metadata-message', 'style'),
        Output('no-metadata-message', 'children'),
        Output('point-metadata-row', 'style'),
        Input('main-graph-static', 'clickData'),
        Input('generative-mode-state', 'data'),
        Input('dataset-dropdown', 'value'),
        prevent_initial_call=True
    )
    def display_metadata(click_data, generative_state, dataset_name):
        invalid = not click_data or dataset_name in IMAGE_ONLY_DATASETS

        metadata_message = "No metadata to show in this dataset" if dataset_name in IMAGE_ONLY_DATASETS else\
            "Click on a point in the graph to show metadata"

        enabled = generative_state.get('enabled', False) if generative_state else False
        display_style = 'block' if not enabled else 'none'

        if invalid:
            return (
                "",  # point-metadata children
                get_no_metadata_message_style('block'),
                html.Div(metadata_message, style={'text-align': 'center', 'color': '#999', 'marginTop': '3rem'}),
                {'display': display_style},
            )

        # Defensive: check for 'customdata' in clickData['points'][0]
        point_data = click_data['points'][0]
        if 'customdata' in point_data:
            point_index = int(point_data['customdata'][0])
        else:
            point_index = point_data.get('pointIndex', 0)
        X, y, data = get_dataset(dataset_name)

        # Get features to display from configuration or use default feature names
        features_to_display = DATASET_FEATURES.get(dataset_name, getattr(data, 'feature_names',
                                                                         [f'Feature {i}' for i in
                                                                          range(X.shape[1])]))

        metadata_table = create_metadata_table(features_to_display, X, point_index)
        return (
            metadata_table,
            {'display': 'none'},
            "",  # Hide click message
            {'display': 'block'},
        )

    @app.callback(
        Output('coordinates-display', 'children'),
        Output('coordinates-col', 'style'),
        Input('main-graph-static', 'clickData'),
        Input('generative-mode-state', 'data'),
        Input('dataset-dropdown', 'value'),
        prevent_initial_call=True
    )
    def display_coordinates(clickData, generative_state, dataset_name):
        invalid = not clickData or not dataset_name in IMAGE_ONLY_DATASETS

        if invalid:
            return (
                 html.Div("Click on a point in the graph to show coordinates",
                         style={'text-align': 'center', 'color': '#999', 'marginTop': '3rem'}),
                {'marginTop': '2rem'}
            )

        x_coord, y_coord = _extract_coords(clickData)
        point_data = clickData['points'][0]
        if 'customdata' in point_data:
            point_index = int(point_data['customdata'][0])
            digit_label = point_data['customdata'][1]
        else:
            point_index = point_data.get('pointIndex', 0)
            digit_label = point_data.get('label', point_data.get('text', str(point_index)))
        _, y, data = get_dataset(dataset_name)

        if hasattr(data, 'target_names'):
            # If y is integer index, map to class name
            class_label = data.target_names[y[point_index]] if int(y[point_index]) < len(data.target_names) else str(
                y[point_index])
        else:
            class_label = f"Class {y[point_index]}"
        coordinates_table = create_coordinate_table(x_coord, y_coord, point_index, class_label, digit_label)
        return coordinates_table, {'marginTop': '2rem'}

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

