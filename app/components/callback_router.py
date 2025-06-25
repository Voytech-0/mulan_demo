import matplotlib
from dash import html, Input, Output, State

from components.callbacks.top_right import register_main_figure_callbacks

from components.callbacks.top_left import register_visualization_callbacks

from components.callbacks.bottom_right import register_data_info_callbacks

from components.callbacks.bottom_left import register_sliders
from components.configs.feature_config import IMAGE_ONLY_DATASETS
from components.data_operations.dataset_api import get_dataset
from components.visualization_generators.plot_maker import create_datapoint_image

matplotlib.use('Agg')  # Use non-interactive backend

# Import TRIMAP from local package
trimap_cache_name = 'trimap_cache'
# Try to import UMAP, if available


def register_callbacks(app):
    register_sliders(app)
    register_data_info_callbacks(app)
    register_visualization_callbacks(app)
    register_main_figure_callbacks(app)

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
