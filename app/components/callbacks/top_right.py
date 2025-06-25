import numpy as np
from dash import Output, Input, callback_context

from components.data_operations.dataset_api import get_dataset
from components.slow_backend_operations.added_features_api import extract_added_data, dynamically_add
from components.slow_backend_operations.embedding_calculation import compute_all_embeddings
from components.visualization_generators.layout_generators import create_metadata_display
from components.visualization_generators.plot_maker import create_figure, create_animated_figure


def register_main_figure_callbacks(app):
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