"""
Embedding computation utilities for the MULAN demo app.
Handles computation of various manifold learning algorithms.
"""

import time
import threading
import numpy as np
from sklearn.manifold import TSNE
import jax.random as random

# Try to import UMAP, if available
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# Try to import TRIMAP from local package
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'google_research_trimap'))
    from google_research_trimap.trimap import trimap
    TRIMAP_AVAILABLE = True
except ImportError:
    TRIMAP_AVAILABLE = False

# Global lock for computations
COMPUTATION_LOCK = threading.Lock()


def compute_trimap(X, key):
    """
    Compute TRIMAP embedding.
    
    Args:
        X: Input data matrix
        key: JAX random key
        
    Returns:
        Tuple of (embedding, computation_time)
    """
    if not TRIMAP_AVAILABLE:
        raise ImportError("TRIMAP is not available")
    
    start_time = time.time()
    
    with COMPUTATION_LOCK:
        print('Starting TRIMAP calculation...')
        
        # Adjust n_inliers based on dataset size
        n_points = X.shape[0]
        n_inliers = min(5, n_points - 2)
        
        # Time the nearest neighbor search
        nn_start = time.time()
        print('Finding nearest neighbors...')
        emb = trimap.transform(key, X, n_inliers=n_inliers, distance='euclidean', verbose=False)
        nn_time = time.time() - nn_start
        print(f'Nearest neighbor search took: {nn_time:.2f} seconds')
        
        result = np.array(emb) if hasattr(emb, "shape") else emb
        total_time = time.time() - start_time
        print(f'Total TRIMAP calculation took: {total_time:.2f} seconds')
    
    return result, total_time


def compute_tsne(X):
    """
    Compute t-SNE embedding.
    
    Args:
        X: Input data matrix
        
    Returns:
        Tuple of (embedding, computation_time)
    """
    start_time = time.time()
    
    with COMPUTATION_LOCK:
        print('Starting t-SNE calculation...')
        
        # Use a subset for large datasets to improve performance
        if X.shape[0] > 5000:
            indices = np.random.choice(X.shape[0], 5000, replace=False)
            X_subset = X[indices]
        else:
            X_subset = X
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, X_subset.shape[0] - 1))
        result = tsne.fit_transform(X_subset)
        
        # If we used a subset, we need to handle the full dataset
        if X.shape[0] > 5000:
            # For now, just return the subset result
            # In a full implementation, you might want to use a different approach
            pass
        
        total_time = time.time() - start_time
        print(f't-SNE calculation took: {total_time:.2f} seconds')
    
    return result, total_time


def compute_umap(X):
    """
    Compute UMAP embedding.
    
    Args:
        X: Input data matrix
        
    Returns:
        Tuple of (embedding, computation_time)
    """
    if not UMAP_AVAILABLE:
        raise ImportError("UMAP is not available")
    
    start_time = time.time()
    
    with COMPUTATION_LOCK:
        print('Starting UMAP calculation...')
        
        # Use a subset for large datasets to improve performance
        if X.shape[0] > 5000:
            indices = np.random.choice(X.shape[0], 5000, replace=False)
            X_subset = X[indices]
        else:
            X_subset = X
        
        reducer = umap.UMAP(n_components=2, random_state=42)
        result = reducer.fit_transform(X_subset)
        
        # If we used a subset, we need to handle the full dataset
        if X.shape[0] > 5000:
            # For now, just return the subset result
            # In a full implementation, you might want to use a different approach
            pass
        
        total_time = time.time() - start_time
        print(f'UMAP calculation took: {total_time:.2f} seconds')
    
    return result, total_time


def get_embedding_computation_function(method):
    """
    Get the appropriate computation function for a given method.
    
    Args:
        method: String indicating the method ('trimap', 'tsne', 'umap')
        
    Returns:
        Function that computes the embedding
    """
    computation_functions = {
        'trimap': compute_trimap,
        'tsne': compute_tsne,
        'umap': compute_umap
    }
    
    if method not in computation_functions:
        raise ValueError(f"Unknown method: {method}")
    
    return computation_functions[method]


def check_method_availability(method):
    """
    Check if a method is available.
    
    Args:
        method: String indicating the method
        
    Returns:
        Boolean indicating availability
    """
    if method == 'umap':
        return UMAP_AVAILABLE
    elif method == 'trimap':
        return TRIMAP_AVAILABLE
    elif method == 'tsne':
        return True
    else:
        return False 