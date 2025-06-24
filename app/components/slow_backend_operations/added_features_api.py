from components.projection_wrapper import TrimapWrapper
from google_research_trimap.trimap import trimap
import jax.random as random
import jax.numpy as jnp
import numpy as np

def dynamically_add(X, y, embedding, added_data):
    X_add, y_add, emb_add = None, None, None
    n_added = 0

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

    if X_add is not None:
        n_added = len(X_add)
        wrapper = TrimapWrapper.load()
        emb_add = wrapper.transform(X_add)
        embedding = np.concatenate((embedding, emb_add))
        X = np.concatenate((X, X_add), axis=0)
        y = np.concatenate((y, y_add), axis=0)

    return X, y, embedding, n_added

def generate_sample(x_coord, y_coord):
    wrapper = TrimapWrapper.load()
    sample = wrapper.inverse_transform(np.array([x_coord, y_coord]))
    return sample

