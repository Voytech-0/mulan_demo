from dash import html, Input, Output, callback_context, State

def register_sliders(app):
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

        elif family == "custom_upload":
            return [{"label": "Upload Custom Dataset", "value": "custom_upload"}], "custom_upload"

        return [], None



