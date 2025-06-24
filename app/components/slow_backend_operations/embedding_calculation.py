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
def compute_trimap_parametric(dataset_name):
    X, _, _ = get_dataset(dataset_name)
    start_time = time.time()
    key = random.PRNGKey(42)  # Deterministic
    with COMPUTATION_LOCK:
        emb, model, params = ptrimap.fit_transform(key, X, n_dims=2)
        result = np.array(emb) if hasattr(emb, "shape") else emb
        total_time = time.time() - start_time

    return result, total_time, model, params

@cache.memoize()
def compute_trimap_iterative(dataset_name, distance):
    X, _, _ = get_dataset(dataset_name)
    start_time = time.time()
    key = random.PRNGKey(42)  # Deterministic
    with COMPUTATION_LOCK:
        emb = trimap.transform(key, X, verbose=True)
        result = np.array(emb) if hasattr(emb, "shape") else emb
        total_time = time.time() - start_time

    result = post_process(result, distance)
    return result, total_time

def compute_trimap(dataset_name, distance, parametric):
    print('calculating trimap')
    if parametric:
        result, total_time, _, _ = compute_trimap_parametric(dataset_name)
    else:
        result, total_time = compute_trimap_iterative(dataset_name, distance)
    print('trimap calculated')
    return result, total_time

@cache.memoize()
def compute_tsne(dataset_name, distance):
    if distance == 'haversine':
        print("t-SNE is not supported for haversine distance. Returning None.")
        return None, 0

    X, _, _ = get_dataset(dataset_name)
    start_time = time.time()
    with COMPUTATION_LOCK:
        print('calculating tsne')
        emb = TSNE(n_components=2, random_state=42, metric=distance).fit_transform(X)
        result = np.array(emb)
        print('tsne calculated')
    return result, time.time() - start_time

@cache.memoize()
def compute_umap(dataset_name, distance):
    X, _, _ = get_dataset(dataset_name)
    start_time = time.time()
    with COMPUTATION_LOCK:
        print('calculating umap')
        reducer = umap.UMAP(n_components=2, random_state=42, output_metric=distance)
        emb = reducer.fit_transform(X)
        result = np.array(emb)
        print('umap calculated')
    result = post_process(result, distance)
    return result, time.time() - start_time


def get_embedding(method_name, compute_func, X, distance, recalculate_flag, dataset_name, is_animated=False):
    # Check if we should use saved embeddings
    if not recalculate_flag and embedding_exists(dataset_name, method_name, distance):
        embedding, metadata = load_embedding(dataset_name, method_name, distance)
        compute_time = metadata['time']
        print(f"Using saved {method_name} embedding")
        if method_name == 'trimap' and not is_animated:
            embedding = embedding[-1]
        return embedding, compute_time

    # Only compute new embedding if recalculate is True or no saved embedding exists
    if recalculate_flag or not embedding_exists(dataset_name, method_name, distance):
        print(f"Computing new {method_name} embedding")
        embedding, compute_time = compute_func(X, distance)
        metadata = {'time': compute_time}
        save_embedding(dataset_name, method_name, embedding, distance, metadata)
        if method_name == 'trimap' and not is_animated:
            embedding = embedding[-1]
        return embedding, compute_time

    return None  # Return None if no embedding is available and we're not computing
