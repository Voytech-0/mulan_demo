from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
from .trimap import generate_triplets, trimap_loss
import jax.random as random

from absl import logging

class ParametricTriMap(nn.Module):
    input_dims: int
    latent_dims: int
    hidden_dims: int = 300
    hidden_layers: int = 3
    activation_fn: callable = nn.relu
    kernel_init: callable = nn.initializers.kaiming_normal()
    bias_init: callable = nn.initializers.zeros
    use_residual_connections: bool = False

    def setup(self):
        forwarded_params = (self.hidden_dims, self.hidden_layers, self.activation_fn,
                            self.kernel_init, self.bias_init, self.use_residual_connections)
        self.encoder = MLP(self.latent_dims, *forwarded_params)
        self.decoder = MLP(self.input_dims, *forwarded_params)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, latent):
        return self.decoder(latent)

    @nn.compact
    def __call__(self, x):
        approx_trimap_embedding = self.encode(x)
        approx_x = self.decode(approx_trimap_embedding)
        return approx_x

class MLP(nn.Module):
    out_dims: int
    hidden_dims: int
    hidden_layers: int
    activation_fn: callable
    kernel_init: callable
    bias_init: callable
    use_residual_connections: bool

    @nn.compact
    def __call__(self, x):
        for idx in range(self.hidden_layers):
            skip = x
            x = nn.Dense(self.hidden_dims, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)

            if self.use_residual_connections and idx > 0:
                x = x + skip

            x = self.activation_fn(x)
        latent = nn.Dense(self.out_dims, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        return latent


def initialize_model(input_dims, n_dims, rng_key, lr=1e-4):
    autoencoder = ParametricTriMap(input_dims, n_dims)

    dummy_input = jnp.ones((1, input_dims))
    variables = autoencoder.init(rng_key, dummy_input)
    tx = optax.adam(lr)
    state = train_state.TrainState.create(apply_fn=autoencoder.apply, params=variables['params'], tx=tx)
    return autoencoder, state


def parametric_triplet_loss(params, state, embedding, triplets, weights, alpha=0.3):
    compressed_embedding = state.apply_fn({'params': params}, embedding, method=ParametricTriMap.encode)
    triplet_loss = trimap_loss(compressed_embedding, triplets, weights)
    reconstruction = state.apply_fn({'params': params}, compressed_embedding, method=ParametricTriMap.decode)
    reconstruction_loss = jnp.mean(optax.huber_loss(reconstruction, embedding))
    loss = (1 - alpha) * triplet_loss + alpha * reconstruction_loss
    return loss, {'triplet_loss': triplet_loss, 'reconstruction_loss': reconstruction_loss}


@partial(jax.jit, static_argnames=['alpha'])
def train_step(state, embedding, triplets, weights, alpha):
    grad_fn = jax.value_and_grad(parametric_triplet_loss, has_aux=True)
    (loss, aux), grads = grad_fn(state.params, state, embedding, triplets, weights, alpha)
    state = state.apply_gradients(grads=grads)
    return state, loss, aux


def fit(rng_key, inputs, n_dims,
                  lr=1e-4,
                  n_inliers=10,
                  n_outliers=5,
                  n_random=3,
                  n_epochs=1000,
                  reconstruction_loss_weight=0.05,
                  weight_temp=0.5,
                  distance='euclidean',
                  verbose=False):
    model_init_key, triplet_key = random.split(rng_key)
    model, state = initialize_model(inputs.shape[1], n_dims, model_init_key, lr)
    triplets, weights = generate_triplets(
        triplet_key,
        inputs,
        n_inliers,
        n_outliers,
        n_random,
        weight_temp=weight_temp,
        distance=distance,
        verbose=verbose)

    inputs = jnp.asarray(inputs)
    for epoch in range(n_epochs):
        state, loss, aux = train_step(state, inputs, triplets, weight_temp, reconstruction_loss_weight)
        if verbose:
            logging.info(f'Epoch {epoch} loss: {loss:.3}, '
                         f'trimap_loss {aux["triplet_loss"]:.3}, reconstruction_loss {aux["reconstruction_loss"]:.3}')

    model_description = {'input_dims': inputs.shape[1], 'latent_dims': n_dims}
    return model_description, state.params

def transform(inputs, model_description, params):
    model = ParametricTriMap(**model_description)
    return model.apply({'params': params}, inputs, method=ParametricTriMap.encode)

def fit_transform(rng_key, inputs, n_dims,
                  lr=1e-4,
                  n_inliers=10,
                  n_outliers=5,
                  n_random=3,
                  n_epochs=1000,
                  reconstruction_loss_weight=0.05,
                  weight_temp=0.5,
                  distance='euclidean',
                  verbose=False):
    model_description, params = fit(rng_key, inputs, n_dims, lr, n_inliers, n_outliers, n_random,
                                    n_epochs=n_epochs, reconstruction_loss_weight=reconstruction_loss_weight,
                                    weight_temp=weight_temp, distance=distance, verbose=verbose)
    embedding = transform(inputs, model_description, params)
    return embedding, model_description, params

def inverse_transform(embedding, model_description, params):
    model = ParametricTriMap(**model_description)
    return model.apply({'params': params}, embedding, method=ParametricTriMap.decode)
