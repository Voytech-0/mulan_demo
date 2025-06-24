import threading
import time

import umap
from sklearn.manifold import TSNE

from components.embedding_storage import load_embedding, embedding_exists, save_embedding
from components.projection_wrapper import TrimapWrapper
import numpy as np


# Global lock for computations
COMPUTATION_LOCK = threading.Lock()


def post_process(result, distance):
    if distance == 'haversine':
        x = np.arctan2(np.sin(result[:, :, 0]) * np.cos(result[:, :, 1]),
                       np.sin(result[:, :, 0]) * np.sin(result[:, :, 1]))
        y = -np.arccos(np.cos(result[:, :, 0]))
        result = np.stack([x, y], axis=-1)

        # x = np.arctan2(np.sin(result[:, 0]) * np.cos(result[:, 1]), np.sin(result[:, 0]) * np.sin(result[:, 1]))
        # y = -np.arccos(np.cos(result[:, 0]))
        # result = np.column_stack((x, y))
    return result


def compute_trimap(X, distance, parametric=False):
    start_time = time.time()
    with COMPUTATION_LOCK:
        wrapper = TrimapWrapper()
        emb = wrapper.fit_transform(X, distance_metric=distance)
        wrapper.store('trimap_cache')
        result = np.array(emb) if hasattr(emb, "shape") else emb
        total_time = time.time() - start_time

    result = post_process(result, distance)
    return result, total_time


def compute_tsne(X, distance):
    if distance == 'haversine':
        print("t-SNE is not supported for haversine distance. Returning None.")
        return None, 0
    start_time = time.time()
    with COMPUTATION_LOCK:
        print('calculating tsne')
        emb = TSNE(n_components=2, random_state=42, metric=distance).fit_transform(X)
        result = np.array(emb)
        print('tsne calculated')
    return result, time.time() - start_time


def compute_umap(X, distance):
    start_time = time.time()
    with COMPUTATION_LOCK:
        print('calculating umap')
        reducer = umap.UMAP(n_components=2, random_state=42, output_metric=distance)
        emb = reducer.fit_transform(X)
        result = np.array(emb)
        print('umap calculated')
    result = post_process(result, distance)
    return result, time.time() - start_time


def get_embedding(method_name, compute_func, X, distance, recalculate_flag, dataset_name):
    # Check if we should use saved embeddings
    if not recalculate_flag and embedding_exists(dataset_name, method_name, distance):
        embedding, metadata = load_embedding(dataset_name, method_name, distance)
        compute_time = metadata['time']
        print(f"Using saved {method_name} embedding")
        return embedding, compute_time

    # Only compute new embedding if recalculate is True or no saved embedding exists
    if recalculate_flag or not embedding_exists(dataset_name, method_name, distance):
        print(f"Computing new {method_name} embedding")
        embedding, compute_time = compute_func(X, distance)
        metadata = {'time': compute_time}
        save_embedding(dataset_name, method_name, embedding, distance, metadata)

        return embedding, compute_time

    return None  # Return None if no embedding is available and we're not computing