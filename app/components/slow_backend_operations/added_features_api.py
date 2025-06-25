import numpy as np

from components.cache import cache
from components.data_operations.dataset_api import get_dataset
from components.slow_backend_operations.embedding_calculation import compute_trimap_parametric, compute_trimap_iterative
import google_research_trimap.trimap.trimap as trimap
import google_research_trimap.trimap.parametric_trimap as ptrimap
import jax.random as random

def _ensure_batch(data, n_dims=1):
    if len(data.shape) == n_dims:
        data = np.expand_dims(data, 0)
    return data

def extract_added_data(added_data):
    X_add, y_add = None, None
    def extract_as_np(x, y):
        return np.asarray(x), np.asarray(y)

    for source in ['user_generated', 'augmented']:
        if source in added_data:  # cache not empty
            if source == 'user_generated':
                X_add = np.array(added_data['user_generated'])
                y_add = np.zeros(X_add.shape[0]) - 1
            elif X_add is None:
                X_add, y_add = extract_as_np(*added_data['augmented'])
            else:
                X_add2, y_add2 = extract_as_np(*added_data['augmented'])
                X_add = np.concatenate((X_add2, X_add), axis=0)
                y_add = np.concatenate((y_add2, y_add), axis=0)
    n_added = X_add.shape[0] if X_add is not None else 0
    return X_add, y_add, n_added

@cache.memoize()
def dynamically_add(added_data, dataset, distance, parametric=False):
    X_add, y_add, n_added = extract_added_data(added_data)
    if n_added == 0:
        return None

    if parametric:
        _, _, model, params = compute_trimap_parametric(dataset)
        emb_add = ptrimap.transform(X_add, model, params)
    else:
        key = random.PRNGKey(0)
        embedding, _ = compute_trimap_iterative(dataset, distance, False)
        X, _, _ = get_dataset(dataset)
        emb_add = trimap.embed_new_points(key, X_add, X, embedding, n_iters=1, verbose=True)

    return emb_add

def generate_sample(x_coord, y_coord, dataset, distance, parametric=False, export_iters=False):
    key = random.PRNGKey(0)
    data = _ensure_batch(np.array([x_coord, y_coord]))
    if parametric:
        _, _, model, params = compute_trimap_parametric(dataset)
        inverse = ptrimap.inverse_transform(key, model, params)
    else:
        embeddings, _ = compute_trimap_iterative(dataset, distance, export_iters)
        X, _, _ = get_dataset(dataset)
        inverse = trimap.inverse_transform(key, data, X, embeddings, n_iters=1, verbose=True)
    return inverse

