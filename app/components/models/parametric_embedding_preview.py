import numpy as np
import matplotlib.pyplot as plt
from components.data_operations.dataset_api import get_dataset
from components.models.parametric_model import train_classifier, load_pretrained_resnet, EmbeddingExtractor
from components.slow_backend_operations.projection_wrapper import TrimapWrapper

# List of image datasets to process
image_datasets = [
    "Digits",
    "MNIST",
    "Fashion MNIST",
    "PACS - Photo",
    "PACS - Sketch",
    "PACS - Cartoon",
    "PACS - Art Painting"
]
USE_SAVED_CKPT = False

print(image_datasets)
# Layer to visualize with TRIMAP
selected_layer = 'global_avg_pool'

for dataset_name in image_datasets:
    print(f"\n=== Processing dataset: {dataset_name} ===")
    X, y, data = get_dataset(dataset_name)
    print(f"Loaded {dataset_name}: X shape: {X.shape}, y shape: {y.shape}")
    num_classes = int(np.max(y)) + 1

    # Train classifier if not already trained
    if not USE_SAVED_CKPT:
        print("Training classifier (if needed)...")
        train_classifier(X, y, dataset_name, num_epochs=3, batch_size=4, learning_rate=1e-3)
        
    # Load trained model
    print("Loading trained model...")
    model, variables = load_pretrained_resnet(dataset_name, num_classes=num_classes)
    extractor = EmbeddingExtractor(model)

    # Prepare input images
    if dataset_name == 'Digits':
        img_shape = (8, 8)
    else:
        img_shape = (28, 28)
    X_img = X.reshape((-1, *img_shape, 1))

    # Extract embeddings for all datapoints (in batches for memory efficiency)
    batch_size = 64
    n_samples = X_img.shape[0]
    # Get layer names from a single batch
    outputs = extractor.apply(variables, X_img[:min(batch_size, n_samples)])
    layer_names = list(outputs.keys())
    all_embs = {lname: [] for lname in layer_names}
    print(f"Extracting embeddings for layers: {layer_names}")
    for i in range(0, n_samples, batch_size):
        batch = X_img[i:i+batch_size]
        outputs = extractor.apply(variables, batch)
        for lname in layer_names:
            all_embs[lname].append(np.array(outputs[lname]))
    for lname in layer_names:
        all_embs[lname] = np.concatenate(all_embs[lname], axis=0)
        print(f"Layer: {lname}, Embedding shape: {all_embs[lname].shape}")

    # Visualize TRIMAP for the selected layer
    if selected_layer in all_embs:
        print(f"\nRunning TRIMAP for layer: {selected_layer}")
        layer_emb = all_embs[selected_layer]
        wrapper = TrimapWrapper()
        trimap_emb = wrapper.fit_transform(layer_emb, distance_metric='euclidean')
        trimap_emb = np.array(trimap_emb)
        print(f"TRIMAP embedding shape: {trimap_emb.shape}")
        if trimap_emb.ndim == 3:
            trimap_emb = trimap_emb[-1]  # Use the last frame
        plt.figure(figsize=(8, 6))
        plt.scatter(trimap_emb[:, 0], trimap_emb[:, 1], c=y, cmap='tab10', s=2, alpha=0.7)
        plt.title(f'TRIMAP of {dataset_name} ({selected_layer} embeddings)')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.colorbar(label='Class')
        plt.show()
    else:
        print(f"Layer {selected_layer} not found in extracted embeddings.")

print('---')
print('You can change `selected_layer` in the script to preview TRIMAP for other layers.') 