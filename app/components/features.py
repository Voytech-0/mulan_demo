from google_research_trimap.trimap import trimap
import jax.random as random
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import base64
from io import BytesIO


def dynamically_add(embedding, X, method='tsne'):
    if embedding.shape[0] == X.shape[0]:
        return embedding

    if method != 'tsne':
        raise NotImplementedError('Dynamic projections are only implemented for t-SNE.')

    X_old, X_new = X[:len(embedding)], X[len(embedding):]
    key = random.PRNGKey(0)
    added_embedding =  trimap.embed_new_points(key, X_new, X_old, embedding)
    return jnp.concatenate((embedding, added_embedding), axis=0)

def generate_sample(x_coord, y_coord, X, embedding):
    key = random.PRNGKey(0)
    sample_embedding =  jnp.expand_dims(jnp.array([x_coord, y_coord]), 0)
    print('generating_sample')
    inversed_data_point = trimap.inverse_transform(key, sample_embedding, X, embedding)
    print('sample_generated')
    return inversed_data_point

def dataset_shape(dataset_name):
    if dataset_name == "Digits":
        img_shape = (8, 8)
    else:  # MNIST or Fashion MNIST
        img_shape = (28, 28)
    return img_shape

def encode_img_as_str(data_point, dataset_name):
    img_shape = dataset_shape(dataset_name)
    plt.figure(figsize=(4, 4))
    plt.imshow(data_point.reshape(img_shape), cmap='gray')
    plt.axis('off')

    # Convert to base64
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    return img_str

def invisible_interactable_layer(x_min, x_max, y_min, y_max):
    x_min, x_max = x_min - 1, x_max + 1
    y_min, y_max = y_min - 1, y_max + 1

    # Create a meshgrid of invisible points
    dummy_x, dummy_y = np.meshgrid(
        np.linspace(x_min, x_max, 50),
        np.linspace(y_min, y_max, 50)
    )

    # Add invisible dummy scatter layer
    layer = go.Scatter(
        x=dummy_x.flatten(),
        y=dummy_y.flatten(),
        mode='markers',
        opacity=1e-6,
        # hoverinfo='skip',
        showlegend=False
    )
    return layer
