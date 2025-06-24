# Try to use an importable ResNet backbone from jax-resnet or similar library
# If not installed, instruct user to install: pip install jax-resnet

from typing import Any, Dict, Optional
import jax
import jax.numpy as jnp
import os
import optax
from flax.training import train_state, checkpoints

# Always use the local implementation for full intermediate outputs
RESNET_AVAILABLE = False
from flax import linen as nn


class BasicBlock(nn.Module):
    features: int
    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.Conv(self.features, (3, 3), padding='SAME')(x)
        # x = nn.BatchNorm(use_running_average=True)(x)
        x = nn.relu(x)
        x = nn.Conv(self.features, (3, 3), padding='SAME')(x)
        # x = nn.BatchNorm(use_running_average=True)(x)
        # Project residual if number of channels changes
        if residual.shape[-1] != self.features:
            residual = nn.Conv(self.features, (1, 1), padding='SAME')(residual)
        x += residual
        x = nn.relu(x)
        return x
    
    
class ResNet3(nn.Module):
    num_classes: int = 10
    block_sizes: tuple = (16, 32, 64)
    @nn.compact
    def __call__(self, x, return_intermediates: bool = False):
        intermediates = {}
        x = nn.Conv(self.block_sizes[0], (3, 3), padding='SAME')(x)
        # x = nn.BatchNorm(use_running_average=True)(x)
        x = nn.relu(x)
        for i, features in enumerate(self.block_sizes):
            x = BasicBlock(features)(x)
            x = nn.max_pool(x, (2, 2), (2, 2))
            if return_intermediates:
                intermediates[f'pool{i+1}'] = x
        x = jnp.mean(x, axis=(1, 2))
        if return_intermediates:
            intermediates['global_avg_pool'] = x
        logits = nn.Dense(self.num_classes)(x)
        if return_intermediates:
            # intermediates['logits'] = logits
            return intermediates
        return logits

class EmbeddingExtractor:
    """
    Wraps a ResNet model to return a dictionary of intermediate embeddings.
    """
    def __init__(self, model):
        self.model = model

    def apply(self, variables, x, mutable=False, intermediates=True):
        # For local model, use return_intermediates argument
        outputs = self.model.apply(variables, x, return_intermediates=intermediates, mutable=mutable)
        return outputs

# Pretrained weights loader
PRETRAINED_URLS = {
    'MNIST': None,  # No official ResNet pretrained for MNIST in jax-resnet, but you can train and load your own
    'Fashion MNIST': None,  # Same as above
    # Add URLs or checkpoint paths for custom datasets if available
}

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '../../saved_embeddings')

class TrainState(train_state.TrainState):
    batch_stats: Any = None

def cross_entropy_loss(logits, labels):
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    return -jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1).mean()

def create_train_state(rng, model, input_shape, learning_rate):
    variables = model.init(rng, jnp.ones(input_shape), return_intermediates=False)
    tx = optax.adam(learning_rate)
    return TrainState.create(apply_fn=model.apply, params=variables['params'], tx=tx)

def train_classifier(X, y, dataset_name, num_epochs=3, batch_size=128, learning_rate=1e-3, checkpoint_dir=CHECKPOINT_DIR):
    """
    Train the ResNet3 classifier on (X, y) and save the trained weights as a checkpoint.
    """
    if dataset_name == 'Digits':
        img_shape = (8, 8)
    else:
        img_shape = (28, 28)
    X_img = X.reshape((-1, *img_shape, 1))
    num_classes = int(jnp.max(y)) + 1
    model = ResNet3(num_classes=num_classes)
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, model, (1, *img_shape, 1), learning_rate)

    num_batches = int(jnp.ceil(X_img.shape[0] / batch_size))
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs} started.")
        # Shuffle data
        perm = jax.random.permutation(rng, X_img.shape[0])
        X_img = X_img[perm]
        y = y[perm]
        for i in range(num_batches):
            batch_x = X_img[i*batch_size:(i+1)*batch_size]
            batch_y = y[i*batch_size:(i+1)*batch_size]
            def loss_fn(params):
                logits = model.apply({'params': params}, batch_x, return_intermediates=False)
                loss = cross_entropy_loss(logits, batch_y)
                return loss
            grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)
        print(f"Epoch {epoch+1}/{num_epochs} completed. Loss: {loss}")
    # Save checkpoint
    ckpt_path = os.path.join(checkpoint_dir, f"{dataset_name}_resnet3_classifier")
    checkpoints.save_checkpoint(ckpt_path, state.params, step=0, overwrite=True)
    print(f"Trained weights saved to {ckpt_path}")
    return model, state.params

def load_pretrained_resnet(dataset_name: str, num_classes: int = 10):
    # Always use the local implementation (with num_classes)
    model = ResNet3(num_classes=num_classes)
    if dataset_name == 'Digits':
        input_shape = (1, 8, 8, 1)
    else:
        input_shape = (1, 28, 28, 1)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{dataset_name}_resnet3_classifier")
    if os.path.exists(ckpt_path):
        # Load trained weights
        variables = model.init(jax.random.PRNGKey(0), jnp.ones(input_shape), return_intermediates=False)
        params = checkpoints.restore_checkpoint(ckpt_path, target=None)
        variables = {'params': params}
    else:
        # Random init
        variables = model.init(jax.random.PRNGKey(0), jnp.ones(input_shape), return_intermediates=False)
    return model, variables

# Example usage:
# model, variables = load_pretrained_resnet('MNIST', num_classes=10)
# extractor = EmbeddingExtractor(model)
# outputs = extractor.apply(variables, jnp.ones([1, 28, 28, 1]))
# print(outputs.keys())  # Dict of intermediate embeddings and logits

"""
To add new pretrained weights for a dataset:
- Train a ResNet model on your dataset using JAX/Flax.
- Save the checkpoint (e.g., using flax.training.checkpoints).
- Add the checkpoint path or URL to PRETRAINED_URLS and load it in load_pretrained_resnet.
"""
