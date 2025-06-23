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

import argparse
import csv
import os

def write_to_csv(csv_path, dataset, method, elapsed_time):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['dataset', 'method', 'time'])
        writer.writerow([dataset, method, elapsed_time])

def main():
    parser = argparse.ArgumentParser(description="Measure time for dimensionality reduction methods.")
    parser.add_argument('--method', type=str, required=True, choices=['trimap_manual', 'trimap_auto', 'trimap_og', 'umap', 'tsne'],
                        help='Method to run: trimap_manual, trimap_auto, trimap_og, umap, tsne')
    parser.add_argument('--dataset', type=str, required=True, choices=['MNIST', 'coil_20'],
                        help='Dataset to use: MNIST or coil_20')
    parser.add_argument('--data_path', type=str, default='data', help='Path to the data directory')
    parser.add_argument('--csv_path', type=str, default='timing_results.csv', help='Path to the CSV file for results')
    args = parser.parse_args()

    key = random.PRNGKey(42)
    print("Dataset:", args.dataset)
    X, labels = data_prep(args.data_path, args.dataset)

    if args.method == 'trimap_manual':
        start_time = time.time()
        trimap.transform(key, X, auto_diff=False, output_metric='squared_euclidean', lr=10)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Time taken for Trimap manual diff: {elapsed} seconds")
        write_to_csv(args.csv_path, args.dataset, args.method, elapsed)
    elif args.method == 'trimap_auto':
        start_time = time.time()
        trimap.transform(key, X, auto_diff=True, output_metric='squared_euclidean', lr=10)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Time taken for Trimap autodiff: {elapsed} seconds")
        write_to_csv(args.csv_path, args.dataset, args.method, elapsed)
    elif args.method == 'trimap_og':
        start_time = time.time()
        trimap_og.transform(key, X)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Time taken for original Trimap: {elapsed} seconds")
        write_to_csv(args.csv_path, args.dataset, args.method, elapsed)
    elif args.method == 'umap':
        start_time = time.time()
        umap.UMAP().fit_transform(X)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Time taken for default UMAP: {elapsed} seconds")
        write_to_csv(args.csv_path, args.dataset, args.method, elapsed)
    elif args.method == 'tsne':
        start_time = time.time()
        TSNE().fit_transform(X)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Time taken for default TSNE: {elapsed} seconds")
        write_to_csv(args.csv_path, args.dataset, args.method, elapsed)
    else:
        print("Unknown method.")

if __name__ == '__main__':
    main()
