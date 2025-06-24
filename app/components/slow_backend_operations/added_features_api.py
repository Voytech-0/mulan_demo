import numpy as np

from components.cache import cache
from components.data_operations.dataset_api import get_dataset
from components.slow_backend_operations.embedding_calculation import compute_trimap_parametric, compute_trimap_iterative
import google_research_trimap.trimap.trimap as trimap
import google_research_trimap.trimap.parametric_trimap as ptrimap
import jax.random as random

def _ensure_batch(data):
    if len(data.shape) == 2:
        data = np.expand_dims(data, 0)
    return data

def extract_added_data(added_data):
    X_add, y_add = None, None
    for source in ['user_generated', 'augmented']:
        if source in added_data:  # cache not empty
            if source == 'user_generated':
                X_add = np.array(added_data['user_generated'])
                y_add = np.zeros(X_add.shape[0]) - 1
            elif X_add is None:
                X_add, y_add = added_data['augmented']
            else:
                X_add2, y_add2 = added_data['augmented']
                X_add = np.concatenate((X_add, X_add2), axis=0)
                y_add = np.concatenate((y_add, y_add2), axis=0)
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
        embedding = compute_trimap_iterative(dataset, distance)
        X, _, _ = get_dataset(dataset)
        print(_ensure_batch(X_add).shape, X_add.shape)
        emb_add = trimap.embed_multiple_new_points(key, _ensure_batch(X_add), X, embedding)

    return emb_add

def generate_sample(x_coord, y_coord, dataset, distance, parametric=False):
    key = random.PRNGKey(0)
    data = _ensure_batch(np.array([x_coord, y_coord]))

    if parametric:
        _, _, model, params = compute_trimap_parametric(data)
        inverse = ptrimap.inverse_transform(key, model, params)
    else:
        embeddings, _ = compute_trimap_iterative(dataset, distance)
        X, _, _ = get_dataset(dataset)
        inverse = trimap.inverse_transform(key, data, X, embeddings)
    return inverse

