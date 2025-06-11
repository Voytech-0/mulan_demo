from dash import html, dcc
import dash_bootstrap_components as dbc

def create_layout(app):
    return html.Div([
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
                        options=[{"label": name, "value": name} for name in ["Digits", "Iris", "Wine", "Breast Cancer"]] + [{"label": "Upload Custom Dataset", "value": "custom_upload"}],
                        value='Digits',
                        style={
                            'backgroundColor': '#2c3e50',
                            'color': 'white',
                            'border': '1px solid #dee2e6'
                        },
                        className="dash-bootstrap-dropdown",
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
                        options=[{"label": name, "value": name} for name in ['opt1', 'opt2', 'opt3']],
                        value='Digits',
                        style={
                            'backgroundColor': '#2c3e50',
                            'color': 'white',
                            'border': '1px solid #dee2e6'
                        },
                        className="dash-bootstrap-dropdown",
                        optionHeight=35,
                        persistence=True,
                        persistence_type='session'
                    ), width=5),
                    dbc.Col(dbc.Button(
                        [html.I(className="fas fa-upload me-2"), "Custom distance"],
                        id="custom-distance-btn",
                        className="ms-2 control-button"
                    ), width=4)
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
                            dbc.Col(html.P("Dots", className="mb-0 me-2 text-white"), width="auto"),
                            dbc.Col(dbc.Switch(
                                id="recalculate-switch",
                                value=False,
                                className="me-2"
                            ), width="auto"),
                            dbc.Col(html.P("Images", className="mb-0 text-white"), width="auto")
                        ], align="center", justify="between", className="mb-3"),
                    ], width=6),
                    
                    dbc.Col(dbc.Button(
                            [html.I(className="fas fa-lightbulb me-2"), "Generative Mode"],
                            id="generative-mode-btn",
                            className="mb-3 w-100 control-button"
                        ), width=6)
                ]),


                # html.Label("Distance Measure:", className="mb-1", style={'color': 'white'}),
                # dbc.Row(
                #     [
                #         dbc.Col(dbc.Button([html.I(className="fas fa-pencil-alt me-2"), "opt1"], id="dist-opt1-btn", className="me-2 control-button"), width="auto"),
                #         dbc.Col(dbc.Button([html.I(className="fas fa-pencil-alt me-2"), "opt2"], id="dist-opt2-btn", className="me-2 control-button"), width="auto"),
                #         dbc.Col(dbc.Button([html.I(className="fas fa-pencil-alt me-2"), "Upload Custom"], id="dist-upload-btn", className="control-button"), width="auto"),
                #     ],
                #     className="mb-3"
                # ),

                html.Label("Layer:", className="mb-1", style={'color': 'white'}),
                dbc.Row(
                    [
                        dbc.Col(dbc.Button([html.I(className="fas fa-pencil-alt me-2"), "opt1"], id="layer-opt1-btn", className="me-2 control-button"), width="auto"),
                        dbc.Col(dbc.Button([html.I(className="fas fa-pencil-alt me-2"), "opt2"], id="layer-opt2-btn", className="me-2 control-button"), width="auto"),
                        dbc.Col(dbc.Button([html.I(className="fas fa-pencil-alt me-2"), "opt3"], id="layer-opt3-btn", className="control-button"), width="auto"),
                    ],
                    className="mb-3"
                ),


                dcc.Store(id='embedding-cache', data={})
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
                                html.Div(id='calculation-status', style={'text-align': 'center', 'margin-top': '0.5rem', 'color': '#fd7e14'})
                            ],
                            fullscreen=False,
                            parent_style={'position': 'relative'},
                            style={'position': 'relative'}
                        )
                    ], width=9),
                    dbc.Col([
                        # Thumbnail buttons
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
                                ]
                            )
                        ]),
                        html.Div([
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
                            )
                        ]),
                        html.Div([
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
                        ])
                    ], width=3)
                ]),

                html.Div(id='umap-warning', style={'color': 'red', 'margin-bottom': '0.5rem'}),

                # Dataset information display
                html.Div(
                    id='metadata-display',
                    children="",
                    style={'height': '20vh', 'border': '1px solid #dee2e6', 'padding': '1rem', 'margin-top': '1rem'}
                ),
                # dcc.Store(id='embedding-cache', data={}) # Moved to left column
            ], width=8)
        ])
    ], className="p-4")
