from dash import html
import numpy as np
from components.configs.settings import CELL_STYLE, CELL_STYLE_RIGHT, TABLE_STYLE


def create_metadata_display(dataset_name, data):
    # This function is now mostly for the dataset-level metadata, not point-level
    return html.Div([
        html.Div([
            html.H4(f"Information about dataset '{dataset_name}'", style={'marginBottom': '0.8rem', 'lineHeight': 1.2, 'fontSize': '1.5rem'}),
            # information abou the dataset
            # html.P(f"{}
            # html.P(f"{}
            html.P(f"Number of samples: {data.data.shape[0]}", style={'marginBottom': '0.8rem', 'lineHeight': 1.2}),
            html.P(f"Number of features: {data.data.shape[1]}", style={'marginBottom': '0.8rem', 'lineHeight': 1.2}),
            html.P(f"Number of classes: {len(np.unique(data.target))}", style={'marginBottom': '0.8rem', 'lineHeight': 1.2})
        ], style={'text-align': 'center'})
    ], style={
        'display': 'flex',
        'flexDirection': 'column',
        'justifyContent': 'center',
        'alignItems': 'center',
        'textAlign': 'center'
    })

def create_coordinate_table(x_coord, y_coord, point_index, class_label, digit_label):
    return html.Table([
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

def create_metadata_table(features_to_display, X, point_index):
    return html.Table([
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