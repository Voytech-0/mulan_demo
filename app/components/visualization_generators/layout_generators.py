from dash import html, dcc
import dash_bootstrap_components as dbc
import numpy as np
from dash_canvas import DashCanvas

from components.configs.settings import CELL_STYLE, CELL_STYLE_RIGHT, TABLE_STYLE
from components.visualization_generators.plot_maker import create_data_distribution_plot


def create_metadata_display(dataset_name, data, figure):
    return html.Div([
        dbc.Row([
            dbc.Col(
                html.Div([
                    html.H4(f"Information about dataset '{dataset_name}'", style={'marginBottom': '0.8rem', 'lineHeight': 1.2, 'fontSize': '1.5rem'}),
                    html.P(f"Number of samples: {data.data.shape[0]}", style={'marginBottom': '0.8rem', 'lineHeight': 1.2}),
                    html.P(f"Number of features: {data.data.shape[1]}", style={'marginBottom': '0.8rem', 'lineHeight': 1.2}),
                    html.P(f"Number of classes: {len(np.unique(data.target))}", style={'marginBottom': '0.8rem', 'lineHeight': 1.2})
                ], style={'text-align': 'left'}), width=5),
            dbc.Col(
                dcc.Graph(id='metadata-plot', figure=figure,
                          style={'height': '25vh', 'width': '100%', 'padding': '1rem', 'color': 'white',
                                 'margin-bottom': '2rem'}), width=7)
        ])], style={
        'display': 'flex',
        'flexDirection': 'column',
        'justifyContent': 'center',
        'alignItems': 'center',
        'textAlign': 'center'
    })
#                       html.Div([
#                             # Title
#                             html.H5("Information about Dataset", style={'textAlign': 'center', 'color': 'white'}),
#                             html.Div(
#                                 children=[
#                                 html.Div(
#                                     id='metadata-display',
#                                     children="",
#                                     style={'height': '25vh', 'padding': '1rem', 'color': 'white', 'margin-top':'3rem'}
#                                 ),
#                                 # Middle: Class distribution plot
#                                 dcc.Graph(id='class-distribution-plot', style={'height': '25vh', 'width': '60%', 'padding': '1rem', 'color': 'white', 'margin-bottom': '2rem'}),
#                             ], style={'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'space-between',  'margin-top': '1rem', 'height':'25vh'})
#                         ], style={'border': f'1px solid {BORDER_COLOR}', 'marginTop': '1rem','padding': '1rem', 'backgroundColor': '#23272b', 'height':'25vh'}),
#                         width=7,

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

def new_canvas(n_clicks):
    return DashCanvas(id={'type': 'canvas', 'index': n_clicks},
                      width=500,
                      height=500,
                      lineWidth=50,
                      lineColor='black',
                      hide_buttons=['zoom', 'pan', 'reset', 'save', 'undo',
                                    'redo', 'line', 'select', 'rectangle', 'pencil'],  # remove all except pencil
                      tool='pencil')