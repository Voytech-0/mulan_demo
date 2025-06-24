import threading
import time
import os
from flax.training import checkpoints

import umap
from sklearn.manifold import TSNE

from components.embedding_storage import load_embedding, embedding_exists, save_embedding
from components.slow_backend_operations.projection_wrapper import TrimapWrapper
import numpy as np
from components.models.parametric_model import load_pretrained_resnet, EmbeddingExtractor


# Global lock for computations
COMPUTATION_LOCK = threading.Lock()

EMBEDDING_CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '../../saved_embeddings')


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


def save_embedding_dict(dataset_name, embedding_dict):
    """
    Save a dictionary of embeddings (layer_name -> np.ndarray) as a checkpoint.
    """
    ckpt_path = os.path.join(EMBEDDING_CHECKPOINT_DIR, f"{dataset_name}_parametric_embeddings")
    # Convert all arrays to np (in case they're JAX DeviceArrays)
    np_dict = {k: np.array(v) for k, v in embedding_dict.items()}
    checkpoints.save_checkpoint(ckpt_path, np_dict, step=0, overwrite=True)


def load_embedding_dict(dataset_name):
    """
    Load a dictionary of embeddings (layer_name -> np.ndarray) from a checkpoint.
    Returns None if not found.
    """
    ckpt_path = os.path.join(EMBEDDING_CHECKPOINT_DIR, f"{dataset_name}_parametric_embeddings")
    if not os.path.exists(ckpt_path):
        return None
    return checkpoints.restore_checkpoint(ckpt_path, target=None)


def compute_by_layer_embedding(X, dataset_name, layer_name='global_avg_pool', batch_size=256, save_ckpt=False):
    """
    Compute embeddings for all samples in X using the parametric model for the given dataset.
    Returns a dict: {layer_name: embeddings} for all layers, or just the selected layer if save_ckpt=False.
    If save_ckpt=True, saves the full embedding dict to a checkpoint.
    """
    if dataset_name == 'Digits':
        img_shape = (8, 8)
    else:
        img_shape = (28, 28)
    if X.ndim == 2:
        X_img = X.reshape((-1, *img_shape, 1))
    else:
        X_img = X
    model, variables = load_pretrained_resnet(dataset_name)
    extractor = EmbeddingExtractor(model)
    all_embs = {}
    # Get all possible layer names from a single batch
    outputs = extractor.apply(variables, X_img[:min(batch_size, X_img.shape[0])])
    layer_names = list(outputs.keys())
    for lname in layer_names:
        all_embs[lname] = []
    for i in range(0, X_img.shape[0], batch_size):
        batch = X_img[i:i+batch_size]
        outputs = extractor.apply(variables, batch)
        for lname in layer_names:
            all_embs[lname].append(np.array(outputs[lname]))
    for lname in layer_names:
        all_embs[lname] = np.concatenate(all_embs[lname], axis=0)
    if save_ckpt:
        save_embedding_dict(dataset_name, all_embs)
    return all_embs if save_ckpt else {layer_name: all_embs[layer_name]}


def compute_parametric_embedding_to_disk(
    X, dataset_name, layer_names=None, batch_size=64, output_dir=None
):
    """
    Compute and save embeddings for all samples in X using the parametric model for the given dataset.
    Saves each layer's embeddings as a .npy file in output_dir.
    Only keeps one batch in memory at a time.
    """
    if dataset_name == 'Digits':
        img_shape = (8, 8)
    else:
        img_shape = (28, 28)
    if X.ndim == 2:
        X_img = X.reshape((-1, *img_shape, 1))
    else:
        X_img = X

    n_samples = X_img.shape[0]
    model, variables = load_pretrained_resnet(dataset_name)
    extractor = EmbeddingExtractor(model)

    # Get all possible layer names from a single batch if not provided
    if layer_names is None:
        outputs = extractor.apply(variables, X_img[:min(batch_size, X_img.shape[0])])
        layer_names = list(outputs.keys())

    # Prepare output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '../../saved_embeddings')
    os.makedirs(output_dir, exist_ok=True)

    # Prepare memmap files for each layer
    memmaps = {}
    for lname in layer_names:
        # Get embedding shape for this layer
        emb_shape = outputs[lname].shape[1:]
        mmap_path = os.path.join(output_dir, f"{dataset_name}_{lname}_embedding.npy")
        memmaps[lname] = np.lib.format.open_memmap(
            mmap_path, mode='w+', dtype=outputs[lname].dtype, shape=(n_samples, *emb_shape)
        )

    # Write each batch directly to memmap
    for i in range(0, n_samples, batch_size):
        batch = X_img[i:i+batch_size]
        outputs = extractor.apply(variables, batch)
        for lname in layer_names:
            memmaps[lname][i:i+batch.shape[0]] = np.array(outputs[lname])

    # Flush to disk
    for mmap in memmaps.values():
        mmap.flush()

    print(f"Saved embeddings for layers: {layer_names} to {output_dir}")
    return {lname: memmaps[lname].filename for lname in layer_names}