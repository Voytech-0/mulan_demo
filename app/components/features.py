from google_research_trimap.trimap import trimap
import jax.random as random
import jax.numpy as jnp



def dynamically_add(embedding, X, method='tsne'):
    if embedding.shape[0] == X.shape[0]:
        return embedding

    if method != 'tsne':
        raise NotImplementedError('Dynamic projections are only implemented for t-SNE.')

    X_old, X_new = X[:len(embedding)], X[len(embedding):]
    key = random.PRNGKey(0)
    added_embedding = trimap.embed_new_points(key, X_new, X_old, embedding)
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

