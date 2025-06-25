import threading
import time
import jax.random as random
import umap
from sklearn.manifold import TSNE

from components.cache import cache
from components.data_operations.dataset_api import get_dataset
from components.embedding_storage import load_embedding, embedding_exists, save_embedding

import google_research_trimap.trimap.trimap as trimap
import google_research_trimap.trimap.parametric_trimap as ptrimap
import numpy as np


# Global lock for computations
COMPUTATION_LOCK = threading.Lock()


def post_process(result, distance):
    if result is None:
        return None
    if distance == 'haversine':
        if len(result.shape) == 3:
            x = np.arctan2(np.sin(result[:, :, 0]) * np.cos(result[:, :, 1]),
                           np.sin(result[:, :, 0]) * np.sin(result[:, :, 1]))
            y = -np.arccos(np.cos(result[:, :, 0]))
            result = np.stack([x, y], axis=-1)
        elif len(result.shape) == 2:
            x = np.arctan2(np.sin(result[:, 0]) * np.cos(result[:, 1]), np.sin(result[:, 0]) * np.sin(result[:, 1]))
            y = -np.arccos(np.cos(result[:, 0]))
            result = np.column_stack((x, y))
        else:
            raise ValueError("Unsupported result shape for post-processing")
    return result.astype(np.float32, copy=False)

@cache.memoize()
def _compute_trimap_parametric(dataset_name):
    X, _, _ = get_dataset(dataset_name)
    start_time = time.time()
    print('fitting parametric trimap')
    key = random.PRNGKey(42)  # Deterministic
    emb, model, params = ptrimap.fit_transform(key, X, n_dims=2)
    result = np.array(emb) if hasattr(emb, "shape") else emb
    print('parametric trimap fitted')
    return result, time.time() - start_time, model, params

@cache.memoize()
def _compute_trimap_iterative(dataset_name, distance, export_iters=False):
    if distance == 'euclidean':
        distance = 'squared_euclidean'
    X, _, _ = get_dataset(dataset_name)
    lr = 1 if distance == 'haversine' else 100
    key = random.PRNGKey(42)  # Deterministic
    start_time = time.time()
    print('calculating trimap')
    emb = trimap.transform(key, X, verbose=True, auto_diff=not distance == 'haversine', lr=lr, export_iters=export_iters, output_metric=distance)
    result = np.array(emb) if hasattr(emb, "shape") else emb
    print('trimap calculated')
    result = post_process(result, distance)
    return result, time.time() - start_time

@cache.memoize()
def _compute_tsne(dataset_name, distance):
    if distance == 'haversine':
        print("t-SNE is not supported for haversine distance. Returning euclidean.")
        return None, 0

    X, _, _ = get_dataset(dataset_name)
    start_time = time.time()
    print('calculating tsne')
    emb = TSNE(n_components=2, random_state=42, metric=distance).fit_transform(X)
    result = np.array(emb)
    print('tsne calculated')
    return result, time.time() - start_time

@cache.memoize()
def _compute_umap(dataset_name, distance):
    X, _, _ = get_dataset(dataset_name)
    start_time = time.time()
    print('calculating umap')
    reducer = umap.UMAP(n_components=2, random_state=42, output_metric=distance)
    emb = reducer.fit_transform(X)
    result = np.array(emb)
    print('umap calculated')
    result = post_process(result, distance)
    return result, time.time() - start_time

def _locked_function(fun, *args, **kwargs):
    with COMPUTATION_LOCK:
        return fun(*args, **kwargs)

def compute_trimap(dataset_name, distance, parametric, export_iters=False):
    if parametric:
        result, total_time, _, _ = compute_trimap_parametric(dataset_name)
    else:
        result, total_time = compute_trimap_iterative(dataset_name, distance, export_iters)
    return result, total_time

def compute_umap(dataset_name, distance):
    return _locked_function(_compute_umap, dataset_name, distance)

def compute_tsne(dataset_name, distance):
    return _locked_function(_compute_tsne, dataset_name, distance)

def compute_trimap_parametric(dataset_name):
    return _locked_function(_compute_trimap_parametric, dataset_name)

def compute_trimap_iterative(dataset_name, distance, export_iters):
    return _locked_function(_compute_trimap_iterative, dataset_name, distance, export_iters)

def compute_all_embeddings(dataset_name, distance, parametric=False, is_animated=False):
    # Get embeddings for all methods
    fwd_args = (dataset_name, distance)
    trimap_emb, trimap_time = compute_trimap(*fwd_args, parametric=parametric, export_iters=is_animated)
    tsne_emb, tsne_time = compute_tsne(*fwd_args)
    umap_emb, umap_time = compute_umap(*fwd_args)
    return (trimap_emb, tsne_emb, umap_emb), (trimap_time, tsne_time, umap_time)