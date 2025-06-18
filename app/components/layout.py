from dash import html, dcc
import dash_bootstrap_components as dbc

from dash.dependencies import Input, Output

def create_layout(app):
    return html.Div([
        dcc.Store(id="distance-measure-store", data="euclidean"),
        html.H1("Manifold Learning Visualizations", className="text-center mb-4"),
        dbc.Row([
            # Left Column (4/12 width): Image display and Dataset Info
            dbc.Col([
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
                                    html.Img(id='selected-image', src='', style={'max-width': '100%', 'height': '22vh', 'display': 'none','padding': '0.5rem'}),
                                    html.Div(id='no-image-message', children="No image to display in this dataset", style={'display': 'none', 'text-align': 'center', 'padding': '0.5rem', 'height': '22vh'})
                                ]),
                                dbc.Row([
                                    html.H5("Coordinates", className="text-center mb-2"),
                                    html.Div(id='coordinates-display', style={'height': '24vh', 'overflow-y': 'auto', 'border': '1px solid #dee2e6', 'padding': '0.5rem'})
                                ])
                            ], width=6),

                            dbc.Col([
                                html.H5("Metadata", className="text-center mb-2"),
                                html.Div(id='point-metadata', style={'height': '47vh', 'overflow-y': 'auto', 'border': '1px solid #dee2e6', 'padding': '0.5rem'})
                            ], width=6)
                        ])
                    ],
                    style={'height': '56vh', 'border': '1px solid #dee2e6', 'padding': '1rem', 'margin-bottom': '0.5rem'}
                ),

                # New elements for the left column as per user's image
                dbc.Row([
                    dbc.Col(html.Label("Pick a Dataset :", className="align-self-center", style={'color': 'white', 'white-space': 'nowrap'}), width=3),
                    dbc.Col(dcc.Dropdown(
                        id='dataset-dropdown',
                        options=[
                            {"label": name, "value": name} for name in [
                                "Digits", "Iris", "Wine", "Breast Cancer",
                                "MNIST", "Fashion MNIST", "Elephant"
                            ]
                        ] + [{"label": "Upload Custom Dataset", "value": "custom_upload"}],
                        value='Digits',
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
                    dbc.Col(dbc.Button(
                        [html.I(className="fas fa-upload me-2"), "New datapoint"],
                        id="upload-new-datapoint-btn",
                        className="ms-2 control-button"
                    ), width=4)
                ], className="mb-3 align-items-center"),

                dbc.Row([
                    dbc.Col(html.Label("Distance measure :", className="align-self-center", style={'color': 'white', 'white-space': 'nowrap'}), width=3),
                    dbc.Col(dcc.Dropdown(
                        id='dist-dropdown',
                        options=[{"label": name, "value": name} for name in ['euclidean', 'manhattan', 'haversine']],
                        value='opt1',
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
                dbc.Row([
                    dbc.Col(html.Label("Image space :", className="align-self-center", style={'color': 'white', 'white-space': 'nowrap'}), width=3),
                    dbc.Col(dcc.Dropdown(
                        id='image-space-dropdown',
                        options=[{"label": name, "value": name} for name in ["layer 1", "layer 2", "layer 3", "final layer"]],
                        value='Raw',
                        style={
                            'backgroundColor': '#2c3e50',
                            'color': 'white',
                            'border': '1px solid #dee2e6'
                        },
                        className="dash-bootstrap-dropdown custom-dropdown",
                        optionHeight=35,
                        persistence=True,
                        persistence_type='session'
                    ), width=9)
                ], className="mb-3 align-items-center"),

                dbc.Row([
                    dbc.Col([
                        dbc.Row([
                            dbc.Col(html.P("Recalculate", className="mb-0 me-2 text-white"), width="auto"),
                            dbc.Col(dbc.Switch(
                                id="recalculate-switch",
                                value=False,
                                className="me-2"
                            ), width="auto"),
                            dbc.Col(html.P("Use Saved", className="mb-0 text-white"), width="auto"),
                        ], align="center", justify="between", className="mb-3"),
                        dbc.Row([
                            dbc.Col(html.P("Parametric mode", className="mb-0 me-2 text-white"), width="auto"),
                            dbc.Col(dbc.Switch(
                                id="parametric-iterative-switch",
                                value=False,
                                className="me-2"
                            ), width="auto"),
                            dbc.Col(html.P("Iterative mode", className="mb-0 text-white"), width="auto"),
                        ], align="center", justify="between", className="mb-3"),
                        dbc.Row([
                            dbc.Col(html.P("Dots", className="mb-0 me-2 text-white"), width="auto"),
                            dbc.Col(dbc.Switch(
                                id="dots-images-switch",
                                value=False,
                                className="me-2"
                            ), width="auto"),
                            dbc.Col(html.P("Images", className="mb-0 text-white"), width="auto")
                        ], align="center", justify="between", className="mb-3"),
                    ], width=12),
                ]),
                dbc.Col(dbc.Button(
                            [html.I(className="fas fa-lightbulb me-2"), "Generative Mode"],
                            id="generative-mode-btn",
                            className="mb-3 w-100 control-button"
                        ), width=12),
                dcc.Store(id='embedding-cache', data={}),

            ], width=4),


            # Right Column (8/12 width): Main Graph and Thumbnails
            dbc.Col([
                # Recalculate switch (moved from here, but keeping placeholder comments for clarity)
                # dbc.Row([ ... ]),

                # Top Row: 3 teal buttons with icons (removed as per new design)
                # dbc.Row([ ... ]),

                # Row for Main Graph and Thumbnails
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
                                html.Div([
                                    html.Label("Iterative process:", style={'margin-top': '1rem', 'margin-bottom': '0.5rem', 'font-weight': 'bold'}),
                                    dbc.Row([
                                        dbc.Col(
                                            dbc.Button(
                                                html.I(className="fas fa-play"),
                                                id="slider-play-btn",
                                                color="primary",
                                                outline=True,
                                                style={"margin-right": "0.5rem"}
                                            ),
                                            width="auto"
                                        ),
                                        dbc.Col(
                                            dcc.Slider(
                                                id='iteration-slider',
                                                min=0,
                                                max=10,
                                                step=1,
                                                value=0,
                                                marks=None,
                                                tooltip={"placement": "bottom", "always_visible": True},
                                                updatemode='drag',
                                                included=False,
                                                className='mb-3'
                                            ),
                                            width=True
                                        ),
                                        dcc.Interval(
                                            id='slider-interval',
                                            interval=500,  # ms
                                            n_intervals=0,
                                            disabled=True
                                        )
                                    ], align="center", className="g-0"),
                                ], style={'margin-bottom': '1rem'})
                            ],
                            fullscreen=False,
                            parent_style={'position': 'relative'},
                            style={'position': 'relative'}
                        )
                    ], width=9),
                    # Thumbnails (right, vertical, stretch, now width=3)
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
                        ], style={'display': 'flex', 'flexDirection': 'column', 'height': '60vh', 'justifyContent': 'space-between'})
                    ], width=3)
                ]),
                html.Div(id='umap-warning', style={'color': 'red', 'margin-bottom': '0.5rem'}),
                # Bottom row: Dataset Information (left), Dataset Augmentation (right)
                dbc.Row([
                    dbc.Col(
                        html.Div(
                            id='metadata-display',
                            children="",
                            style={'height': '20vh', 'border': '1px solid #dee2e6', 'padding': '1rem', 'margin-top': '1rem', 'backgroundColor': '#23272b', 'color': 'white'}
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
                        ], style={'height': '100%', 'border': '1px solid #dee2e6', 'padding': '1rem', 'backgroundColor': '#23272b', 'margin-top': '1rem', 'color': 'white'})
                        , width=5
                    )
                ])
            ], width=8)
        ])
    ], className="p-4")
