"""
Layout components for the MULAN demo app.
Defines the main application layout and UI components.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
from .settings import (
    IMAGE_DISPLAY_CONTAINER_HEIGHT, SELECTED_IMAGE_STYLE, NO_IMAGE_MESSAGE_STYLE,
    GENERATIVE_PLACEHOLDER_STYLE, COORDINATES_DISPLAY_STYLE, METADATA_DISPLAY_STYLE,
    METADATA_DISPLAY_HEIGHT, BACKGROUND_COLOR, BORDER_COLOR
)


def create_dropdown(id_name, options, value, label, width=5):
    """Create a styled dropdown component."""
    return dbc.Col(dcc.Dropdown(
        id=id_name,
        options=options,  # type: ignore
        value=value,
        style={
            'backgroundColor': '#2c3e50',
            'color': 'white',
            'border': '1px solid #dee2e6'
        },
        className="dash-bootstrap-dropdown custom-dropdown",
        optionHeight=35,
        persistence=True,
        persistence_type='session'
    ), width=width)


def create_switch_row(label1, switch_id, label2, value=False):
    """Create a row with a switch between two labels."""
    return dbc.Row([
        dbc.Col(html.P(label1, className="mb-0 me-2 text-white"), width="auto"),
        dbc.Col(dbc.Switch(
            id=switch_id,
            value=value,
            className="me-2"
        ), width="auto"),
        dbc.Col(html.P(label2, className="mb-0 text-white"), width="auto"),
    ], align="center", justify="between", className="mb-3")


def create_thumbnail(method_name, graph_id, timing_id, click_id, is_selected=False):
    """Create a thumbnail component for a method."""
    return html.Div(
        id=click_id,
        className=f"method-button-container thumbnail-button mb-3{' selected' if is_selected else ''}",
        children=[
            html.H4(method_name, className="text-center mb-2"),
            html.Div(id=timing_id, className="timing-display"),
            dcc.Graph(
                id=graph_id,
                style={'height': '13vh'},
                config={'displayModeBar': False, 'staticPlot': True}
            )
        ]
    )


from dash.dependencies import Input, Output

def create_layout(app):
    """Create the main application layout."""

    # Dataset options
    dataset_options = [
        {"label": name, "value": name} for name in [
            "Digits", "Iris", "Wine", "Breast Cancer",
            "MNIST", "Fashion MNIST", "Elephant"
        ]
    ] + [{"label": "Upload Custom Dataset", "value": "custom_upload"}]

    # Distance measure options
    distance_options = [{"label": name, "value": name} for name in ['euclidean', 'manhattan', 'haversine']]

    # Image space options
    image_space_options = [{"label": name, "value": name} for name in ["layer 1", "layer 2", "layer 3", "final layer"]]

    return html.Div([
        dcc.Store(id="distance-measure-store", data="euclidean"),
        html.H1("Manifold Learning Visualizations", className="text-center mb-4"),
        dcc.Store(id='generative-mode-state', data={'enabled': False}),

        dbc.Row([
            # Left Column (4/12 width): Image display and Dataset Info
            dbc.Col([
                # Top-left corner buttons
                dbc.Row([
                    dbc.Col(dbc.Button(
                        [html.I(className="fas fa-plus me-2"), "Add New Datapoint"],
                        id="add-datapoint-btn",
                        color="primary",
                        className="me-2 control-button"
                    ), width="auto"),
                    dbc.Col(dbc.Button(
                        [html.I(className="fas fa-th-large me-2"), "Full Grid"],
                        id="full-grid-btn",
                        color="secondary",
                        className="control-button"
                    ), width="auto")
                ], className="mb-3"),
                # Image display container
                html.Div(
                    id='image-display',
                    children=[
                        html.Div(
                            id='click-message',
                            children="Click on a point in the graph to view its details",
                            style={'text-align': 'center', 'padding': '0.5rem', 'margin-bottom': '0.5rem', 'color': '#666'}
                        ),
                        dbc.Row([
                            dbc.Col([
                                dbc.Row([
                                    html.H5("Image", className="text-center mb-2"),
                                    html.Img(id='selected-image', src='', style=SELECTED_IMAGE_STYLE),
                                    html.Div(id='no-image-message', children="No image to display in this dataset", style=NO_IMAGE_MESSAGE_STYLE),
                                    html.Div(id='generative-mode-placeholder', children="Generative mode content will appear here", style=GENERATIVE_PLACEHOLDER_STYLE)
                                ]),
                                dbc.Row([
                                    html.H5("Coordinates", className="text-center mb-2"),
                                    html.Div(id='coordinates-display', style=COORDINATES_DISPLAY_STYLE)
                                ]),
                                dbc.Row([
                                    html.H5("Point Metadata", className="text-center mb-2"),
                                    html.Div(id='point-metadata', style=METADATA_DISPLAY_STYLE),
                                ], id='point-metadata-row')
                            ], width=12),
                        ])
                    ],
                    style={'height': IMAGE_DISPLAY_CONTAINER_HEIGHT, 'border': f'1px solid {BORDER_COLOR}', 'padding': '1rem', 'margin-bottom': '0.5rem'}
                ),

                # Full Grid container (hidden by default)
                html.Div(
                    id='full-grid-container',
                    children=[],  # Will be filled by callback
                    style={
                        'display': 'none',
                        'border': f'1px solid {BORDER_COLOR}',
                        'padding': '1rem',
                        'margin-bottom': '1rem',
                        'height': '30vh',
                        'overflowY': 'scroll',
                        'backgroundColor': BACKGROUND_COLOR,
                        'color': 'white'
                    }
                ),

                # Dataset selection controls
                dbc.Row([
                    dbc.Col(html.Label("Pick a Dataset :", className="align-self-center", style={'color': 'white', 'white-space': 'nowrap'}), width=3),
                    create_dropdown('dataset-dropdown', dataset_options, 'Digits', "Dataset", 5),
                    dbc.Col(dbc.Button(
                        [html.I(className="fas fa-upload me-2"), "New datapoint"],
                        id="upload-new-datapoint-btn",
                        className="ms-2 control-button"
                    ), width=4)
                ], className="mb-3 align-items-center"),

                # Distance measure controls
                dbc.Row([
                    dbc.Col(html.Label("Distance measure :", className="align-self-center", style={'color': 'white', 'white-space': 'nowrap'}), width=3),
                    dbc.Col(dcc.Dropdown(
                        id='dist-dropdown',
                        options=[{"label": name, "value": name} for name in ['euclidean', 'manhattan', 'haversine']],
                        value='euclidean',
                        style={
                            'backgroundColor': '#2c3e50',
                            'color': 'white',
                            'border': '1px solid #dee2e6'
                        },
                        className="dash-bootstrap-dropdown custom-dropdown",
                        optionHeight=35,
                        persistence=True,
                        persistence_type='session'
                    ), width=5),

                ], className="mb-3 align-items-center"),

                # Image space controls
                dbc.Row([
                    dbc.Col(html.Label("Image space :", className="align-self-center", style={'color': 'white', 'white-space': 'nowrap'}), width=3),
                    create_dropdown('image-space-dropdown', image_space_options, 'Raw', "Image Space", 9)
                ], className="mb-3 align-items-center"),

                # Switches
                dbc.Row([
                    dbc.Col([
                        create_switch_row("Use Saved", "recalculate-switch", "Recalculate"),
                        create_switch_row("Parametric mode", "parametric-iterative-switch", "Iterative mode"),
                        create_switch_row("Dots", "dots-images-switch", "Images")
                    ], width=12),
                ]),

                # Generative mode button
                dbc.Col(dbc.Button(
                    [html.I(className="fas fa-lightbulb me-2"), "Generative Mode"],
                    id="generative-mode-btn",
                    className="mb-3 w-100 control-button"
                ), width=12),

                # Data stores
                dcc.Store(id='embedding-cache', data={}),
                dcc.Store(id='main-graph-images', data={}),

            ], width=4),

            # Right Column (8/12 width): Main Graph and Thumbnails
            dbc.Col([
                # Main graph and controls
                dbc.Row([
                    dbc.Col([
                        dcc.Loading(
                            id="loading-main-graph",
                            type="default",
                            children=[
                                dcc.Graph(
                                    id='main-graph',
                                    style={'height': '60vh', 'margin-bottom': '0.5rem'},
                                    config={'displayModeBar': True}
                                ),
                                html.Div(id='calculation-status', style={'text-align': 'center', 'margin-top': '0.5rem', 'color': '#fd7e14'}),

                            ],
                            fullscreen=False,
                            parent_style={'position': 'relative'},
                            style={'position': 'relative'}
                        )
                    ], width=9),

                    # Thumbnails
                    dbc.Col([
                        html.Div([
                            html.Div(
                                id='tsne-thumbnail-click',
                                className="method-button-container thumbnail-button mb-3",
                                children=[
                                    html.H4("t-SNE", className="text-center mb-2"),
                                    html.Div(id='tsne-timing', className="timing-display"),
                                    dcc.Graph(
                                        id='tsne-thumbnail',
                                        style={'height': '13vh'},
                                        config={'displayModeBar': False, 'staticPlot': True}
                                    )
                                ],
                                n_clicks=0,
                                **{"data-disabled": True}  # This will be toggled by a callback
                            ),
                            html.Div(
                                id='umap-thumbnail-click',
                                className="method-button-container thumbnail-button mb-3",
                                children=[
                                    html.H4("UMAP", className="text-center mb-2"),
                                    html.Div(id='umap-timing', className="timing-display"),
                                    dcc.Graph(
                                        id='umap-thumbnail',
                                        style={'height': '13vh'},
                                        config={'displayModeBar': False, 'staticPlot': True}
                                    )
                                ]
                            ),
                            html.Div(
                                id='trimap-thumbnail-click',
                                className="method-button-container thumbnail-button mb-3 selected",
                                children=[
                                    html.H4("TRIMAP", className="text-center mb-2"),
                                    html.Div(id='trimap-timing', className="timing-display"),
                                    dcc.Graph(
                                        id='trimap-thumbnail',
                                        style={'height': '13vh'},
                                        config={'displayModeBar': False, 'staticPlot': True}
                                    )
                                ]
                            )
                            create_thumbnail("t-SNE", 'tsne-thumbnail', 'tsne-timing', 'tsne-thumbnail-click'),
                            create_thumbnail("UMAP", 'umap-thumbnail', 'umap-timing', 'umap-thumbnail-click'),
                            create_thumbnail("TRIMAP", 'trimap-thumbnail', 'trimap-timing', 'trimap-thumbnail-click', True)
                        ], style={'display': 'flex', 'flexDirection': 'column', 'height': '60vh', 'justifyContent': 'space-between'})
                    ], width=3)
                ]),

                # Warning message
                html.Div(id='umap-warning', style={'color': 'red', 'margin-bottom': '0.5rem'}),

                # Bottom row: Dataset Information and Augmentation
                dbc.Row([
                    dbc.Col(
                        html.Div(
                            id='metadata-display',
                            children="",
                            style={'height': METADATA_DISPLAY_HEIGHT, 'border': f'1px solid {BORDER_COLOR}', 'padding': '1rem', 'margin-top': '1rem', 'backgroundColor': BACKGROUND_COLOR, 'color': 'white'}
                        ),
                        width=7
                    ),
                    dbc.Col(
                        html.Div([
                            html.H5("Dataset Augmentation", className="text-center mb-2", style={'color': 'white'}),
                            html.Label("Brightness", style={'color': 'white'}),
                            dcc.Slider(
                                id='brightness-slider',
                                min=0,
                                max=2,
                                step=0.01,
                                value=1,
                                marks={0: '0', 1: '1', 2: '2'},
                                tooltip={"placement": "bottom", "always_visible": False},
                                className='mb-3'
                            ),
                            html.Label("Contrast", style={'color': 'white'}),
                            dcc.Slider(
                                id='contrast-slider',
                                min=0,
                                max=2,
                                step=0.01,
                                value=1,
                                marks={0: '0', 1: '1', 2: '2'},
                                tooltip={"placement": "bottom", "always_visible": False},
                                className='mb-3'
                            )
                        ], style={'height': '100%', 'border': '1px solid #dee2e6', 'padding': '1rem', 'backgroundColor': '#23272b', 'margin-top': '1rem', 'color': 'white'}),
                        width=5
                    )
                ])
            ], width=8)
        ])
    ], className="p-4")
