import numpy as np
from dash import Output, Input, State

from components.data_operations.dataset_api import get_dataset


def register_data_info_callbacks(app):
    @app.callback(
        Output('added-data-cache', 'data', allow_duplicate=True),  # Store processed data
        Input('brightness-slider', 'value'),
        Input('contrast-slider', 'value'),
        State('added-data-cache', 'data'),
        State('dataset-dropdown', 'value'),
        prevent_initial_call=True
    )
    def augment_data(brightness, contrast, data_cache, dataset_name, n_samples=100):
        X, y, _ = get_dataset(dataset_name)
        empirical_min, emirical_max, dtype = X.min(), X.max(), X.dtype
        random_indices = np.random.choice(len(X), n_samples, replace=False)
        X, y = X[random_indices], y[random_indices]
        X = contrast * X + (brightness - 1)
        X = np.clip(X, empirical_min, emirical_max)
        if dtype == int:
            X = np.round(X).astype(int)
        data_cache['augmented'] = (X, y)
        return data_cache