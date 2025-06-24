import numpy as np

from components.slow_backend_operations.embedding_calculation import compute_trimap_parametric, compute_trimap_iterative
import google_research_trimap.trimap.trimap as trimap
import google_research_trimap.trimap.parametric_trimap as ptrimap
import jax.random as random

def _ensure_batch(data):
    if len(data.shape) == 2:
        data = np.expand_dims(data, 0)
    return data

def dynamically_add(X, y, embedding, added_data, parametric=False):
    X_add, y_add, emb_add = None, None, None
    for source in ['user_generated', 'augmented']:
        if source in added_data:  # cache not empty
            if source == 'user_generated':
                X_add = np.array(added_data['user_generated'])
                y_add = np.zeros(X_add.shape[0]) - 1
            elif X_add is None:
                X_add, y_add = added_data['augmented']
            else:
                X_add2, y_add2 = added_data['augmented']
                X_add = np.concatenate((X_add, X_add2))
                y_add = np.concatenate((y_add, y_add2))

    if X_add is None:
        return  X, y, embedding, 0

    n_added = len(X_add)

    if parametric:
        _, _, model, params = compute_trimap_parametric(X)
        emb_add = ptrimap.transform(X_add, model, params)
    else:
        key = random.PRNGKey(0)
        emb_add = trimap.embed_new_points(key, X_add, X, embedding)

    embedding = np.concatenate((embedding, emb_add))
    X = np.concatenate((X, X_add), axis=0)
    y = np.concatenate((y, y_add), axis=0)
    return X, y, embedding, n_added

def generate_sample(x_coord, y_coord, X, distance, parametric=False):
    key = random.PRNGKey(0)
    data = _ensure_batch(np.array([x_coord, y_coord]))

    if parametric:
        _, _, model, params = compute_trimap_parametric(X)
        inverse = ptrimap.inverse_transform(key, model, params)
    else:
        embeddings, _ = compute_trimap_iterative(X, distance)
        inverse = trimap.inverse_transform(key, data, X, embeddings)
    return inverse

