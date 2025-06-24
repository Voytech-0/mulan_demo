import json
from pathlib import Path

import numpy as np

# Create a directory for storing embeddings if it doesn't exist
EMBEDDING_DIR = Path(__file__).parent.parent / 'saved_embeddings'
EMBEDDING_DIR.mkdir(exist_ok=True)

def get_embedding_path(dataset_name: str, method: str, output_metric: str) -> Path:
    """Get the path for a specific embedding file."""
    return EMBEDDING_DIR / f"{dataset_name}_{method}_{output_metric}.npy"


def save_embedding(dataset_name: str, method: str, embedding: np.ndarray, output_metric: str,
                   metadata: dict = None):
    """Save an embedding to disk."""
    embedding_path = get_embedding_path(dataset_name, method, output_metric)
    np.save(embedding_path, embedding)

    if metadata:
        metadata_path = embedding_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)


def load_embedding(dataset_name: str, method: str, output_metric: str) -> tuple:
    """Load an embedding from disk. Returns (embedding, metadata) or (None, None) if not found."""
    embedding_path = get_embedding_path(dataset_name, method, output_metric)
    metadata_path = embedding_path.with_suffix('.json')

    if not embedding_path.exists():
        return None, None

    embedding = np.load(embedding_path, allow_pickle=True)
    metadata = None

    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

    return embedding, metadata


def embedding_exists(dataset_name: str, method: str, output_metric: str) -> bool:
    """Check if an embedding exists for the given dataset and method."""
    return get_embedding_path(dataset_name, method, output_metric).exists()
