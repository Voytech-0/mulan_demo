import numpy as np
import time
import trimap_og
import umap.umap_ as umap
from google_research_trimap.trimap import trimap
import jax.random as random
from sklearn.manifold import TSNE

def data_prep(data_path, dataset='MNIST', size=None):
    '''
    This function loads the dataset as numpy array.
    Input:
        data_path: path of the folder you store all the data needed.
        dataset: the name of the dataset. Coil or MNIST
        size: [Optional] the size of the dataset.
    Output:
        X: the dataset in numpy array
        labels: the labels of the dataset.
    '''
    if dataset == 'MNIST':
        X = np.load(data_path + '/mnist_images.npy', allow_pickle=True).reshape(70000, 28 * 28)
        labels = np.load(data_path + '/mnist_labels.npy', allow_pickle=True)
    elif dataset == 'coil_20':
        X = np.load(data_path + '/coil_20.npy', allow_pickle=True).reshape(1440, 128 * 128).astype(np.float32)
        labels = np.load(data_path + '/coil_20_labels.npy', allow_pickle=True)
    else:
        print('Unsupported dataset')
        assert (False)
    return X, labels

if __name__ == '__main__':
    key = random.PRNGKey(42)
    data_path = 'data'
    datasets = ['coil_20']#['MNIST', 'coil_20']
    for dataset in datasets:
        print("Dataset:", dataset)
        X, labels = data_prep(data_path, dataset)
        
        start_time = time.time()
        trimap.transform(key, X, auto_diff=True, output_metric='squared_euclidean', lr=10)
        end_time = time.time()
        print(f"Time taken for Trimap autodiff: {end_time - start_time} seconds")

        start_time = time.time()
        trimap.transform(key, X, auto_diff=False, output_metric='squared_euclidean', lr=10)
        end_time = time.time()
        print(f"Time taken for Trimap manual diff: {end_time - start_time} seconds")

        start_time = time.time()
        trimap_og.transform(key, X)
        end_time = time.time()
        print(f"Time taken for original Trimap: {end_time - start_time} seconds")

        start_time = time.time()
        umap.UMAP().fit_transform(X)
        end_time = time.time()
        print(f"Time taken for  default UMAP: {end_time - start_time} seconds")

        start_time = time.time()
        TSNE().fit_transform(X)
        end_time = time.time()
        print(f"Time taken for default TSNE: {end_time - start_time} seconds")

