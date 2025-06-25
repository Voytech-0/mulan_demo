# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TriMap: Large-scale Dimensionality Reduction Using Triplets.

Source: https://arxiv.org/pdf/1910.00204.pdf
"""

import datetime
import time

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import pynndescent
from absl import logging
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

try:
  from google_research_trimap.trimap import distances
except ImportError:
  import distances

_DIM_PCA = 100
_INIT_SCALE = 0.01
_INIT_MOMENTUM = 0.5
_FINAL_MOMENTUM = 0.8
_SWITCH_ITER = 250
_MIN_GAIN = 0.01
_INCREASE_GAIN = 0.2
_DAMP_GAIN = 0.8
_DISPLAY_ITER = 100


def tempered_log(x, t):
  """Tempered log with temperature t."""
  if jnp.abs(t - 1.0) < 1e-5:
    return jnp.log(x)
  else:
    return 1. / (1. - t) * (jnp.power(x, 1.0 - t) - 1.0)


def get_distance_fn(distance_fn_name):
  """Get the distance function."""
  if distance_fn_name == 'euclidean':
    return euclidean_dist
  elif distance_fn_name == 'squared_euclidean':
    return squared_euclidean_dist
  elif distance_fn_name == 'manhattan':
    return manhattan_dist
  elif distance_fn_name == 'cosine':
    return cosine_dist
  elif distance_fn_name == 'hamming':
    return hamming_dist
  elif distance_fn_name == 'chebyshev':
    return chebyshev_dist
  elif distance_fn_name == "haversine":
    return haversine_dist
  elif distance_fn_name in distances.named_distances:
    print(f'Using UMAP-adapted distance function: {distance_fn_name}')
    return distances.named_distances[distance_fn_name]
  else:
    raise ValueError(f'Distance function {distance_fn_name} not supported.')


def get_output_distance_fn(output_metric_name):
  """Get the output (embedding space) distance function."""
  if callable(output_metric_name):
    return output_metric_name
  return get_distance_fn(output_metric_name)


def sliced_distances(
    indices1,
    indices2,
    inputs,
    distance_fn):
  """Applies distance_fn in smaller slices to avoid memory blow-ups.

  Args:
    indices1: First array of indices.
    indices2: Second array of indices.
    inputs: 2-D array of inputs.
    distance_fn: Distance function that applies row-wise.

  Returns:
    Pairwise distances between the row indices in indices1 and indices2.
  """
  slice_size = inputs.shape[0]
  distances = []
  num_slices = int(np.ceil(len(indices1) / slice_size))
  for slice_id in range(num_slices):
    start = slice_id * slice_size
    end = (slice_id + 1) * slice_size
    distances.append(
      distance_fn(inputs[indices1[start:end]], inputs[indices2[start:end]]))
  return jnp.concatenate(distances)


@jax.jit
def squared_euclidean_dist(x1, x2):
  """Squared Euclidean distance between rows of x1 and x2."""
  return jnp.sum(jnp.power(x1 - x2, 2), axis=-1)


@jax.jit
def euclidean_dist(x1, x2):
  """Euclidean distance between rows of x1 and x2."""
  return jnp.sqrt(jnp.sum(jnp.power(x1 - x2, 2), axis=-1))


@jax.jit
def manhattan_dist(x1, x2):
  """Manhattan distance between rows of x1 and x2."""
  return jnp.sum(jnp.abs(x1 - x2), axis=-1)


@jax.jit
def cosine_dist(x1, x2):
  """Cosine (i.e. cosine) between rows of x1 and x2."""
  x1_norm = jnp.maximum(jnp.linalg.norm(x1, axis=-1), 1e-20)
  x2_norm = jnp.maximum(jnp.linalg.norm(x2, axis=-1), 1e-20)
  return 1. - jnp.sum(x1 * x2, -1) / x1_norm / x2_norm


@jax.jit
def hamming_dist(x1, x2):
  """Hamming distance between two vectors."""
  return jnp.sum(x1 != x2, axis=-1)


@jax.jit
def chebyshev_dist(x1, x2):
  """Chebyshev distance between two vectors."""
  return jnp.max(jnp.abs(x1 - x2), -1)


@jax.custom_jvp
def haversine_dist(x, y):
  """Haversine distance between two points on a sphere."""
  sin_lat = jnp.sin(0.5 * (x[..., 0] - y[..., 0]))
  sin_long = jnp.sin(0.5 * (x[..., 1] - y[..., 1]))
  a = sin_lat ** 2 + jnp.cos(x[..., 0]) * jnp.cos(y[..., 0]) * sin_long ** 2
  return 2.0 * jnp.arcsin(jnp.sqrt(a))


@haversine_dist.defjvp
def haversine_dist_jvp(primals, tangents):
  x, y = primals
  t_x, t_y = tangents

  # Forward pass
  dist = haversine_dist(x, y)

  # Adjust latitude by adding π/2
  adjusted_lat_x = x[..., 0] + jnp.pi / 2
  adjusted_lat_y = y[..., 0] + jnp.pi / 2

  # Backward pass (gradient computation)
  sin_lat = jnp.sin(0.5 * (adjusted_lat_x - adjusted_lat_y))
  sin_long = jnp.sin(0.5 * (x[..., 1] - y[..., 1]))
  cos_lat_x = jnp.cos(adjusted_lat_x)
  cos_lat_y = jnp.cos(adjusted_lat_y)
  a = sin_lat ** 2 + cos_lat_x * cos_lat_y * sin_long ** 2
  sqrt_a = jnp.sqrt(a)
  denom = jnp.sqrt(1 - a) * sqrt_a + 1e-8  # Avoid division by zero

  grad_x = jnp.stack([
      (cos_lat_x * sin_lat) / denom,
      (cos_lat_x * cos_lat_y * sin_long) / denom
  ], axis=-1)
  grad_y = jnp.stack([
      -(cos_lat_y * sin_lat) / denom,
      -(cos_lat_x * cos_lat_y * sin_long) / denom
  ], axis=-1)

  tangent_out = jnp.sum(grad_x * t_x, axis=-1) + jnp.sum(grad_y * t_y, axis=-1)
  return dist, tangent_out


def rejection_sample(key, shape, maxval, rejects):
  """Rejection sample indices.

  Samples integers from a given interval [0, maxval] while rejecting the values
  that are in rejects.

  Args:
    key: Random key.
    shape: Output shape.
    maxval: Maximum allowed index value.
    rejects: Indices to reject.

  Returns:
    samples: Sampled indices.
  """
  in1dvec = jax.vmap(jnp.isin)

  def cond_fun(carry):
    _, _, discard = carry
    return jnp.any(discard)

  def body_fun(carry):
    key, samples, _ = carry
    key, use_key = random.split(key)
    new_samples = random.randint(use_key, shape=shape, minval=0, maxval=maxval)
    discard = jnp.logical_or(
      in1dvec(new_samples, samples), in1dvec(new_samples, rejects))
    samples = jnp.where(discard, samples, new_samples)
    return key, samples, in1dvec(samples, rejects)

  key, use_key = random.split(key)
  samples = random.randint(use_key, shape=shape, minval=0, maxval=maxval)
  discard = in1dvec(samples, rejects)
  _, samples, _ = jax.lax.while_loop(cond_fun, body_fun,
                                     (key, samples, discard))
  return samples


def sample_knn_triplets(key, neighbors, n_inliers, n_outliers):
  """Sample nearest neighbors triplets based on the neighbors.

  Args:
    key: Random key.
    neighbors: Nearest neighbors indices for each point.
    n_inliers: Number of inliers.
    n_outliers: Number of outliers.

Returns:
  triplets: Sampled triplets.
"""

  n_points = neighbors.shape[0]
  anchors = jnp.tile(
    jnp.arange(n_points).reshape([-1, 1]),
    [1, n_inliers * n_outliers]).reshape([-1, 1])
  inliers = jnp.tile(neighbors[:, 1:n_inliers + 1],
                     [1, n_outliers]).reshape([-1, 1])
  outliers = rejection_sample(key, (n_points, n_inliers * n_outliers), n_points,
                              neighbors).reshape([-1, 1])
  triplets = jnp.concatenate((anchors, inliers, outliers), 1)
  return triplets


def sample_random_triplets(key, inputs, n_random, distance_fn, sig):
  """Sample uniformly random triplets.

  Args:
    key: Random key.
    inputs: Input points.
    n_random: Number of random triplets per point.
    distance_fn: Distance function.
    sig: Scaling factor for the distances

Returns:
  triplets: Sampled triplets.
"""

  n_points = inputs.shape[0]
  anchors = jnp.tile(jnp.arange(n_points).reshape([-1, 1]),
                     [1, n_random]).reshape([-1, 1])
  pairs = rejection_sample(key, (n_points * n_random, 2), n_points, anchors)
  triplets = jnp.concatenate((anchors, pairs), 1)
  anc = triplets[:, 0]
  sim = triplets[:, 1]
  out = triplets[:, 2]
  p_sim = -(sliced_distances(anc, sim, inputs, distance_fn) ** 2) / (
      sig[anc] * sig[sim])
  p_out = -(sliced_distances(anc, out, inputs, distance_fn) ** 2) / (
      sig[anc] * sig[out])
  flip = p_sim < p_out
  weights = p_sim - p_out
  pairs = jnp.where(
    jnp.tile(flip.reshape([-1, 1]), [1, 2]), jnp.fliplr(pairs), pairs)
  triplets = jnp.concatenate((anchors, pairs), 1)
  return triplets, weights


def find_scaled_neighbors(inputs, neighbors, distance_fn):
  """Calculates the scaled neighbors and their similarities.

  Args:
    inputs: Input examples.
    neighbors: Nearest neighbors
    distance_fn: Distance function.

Returns:
  Scaled distances and neighbors, and the scale parameter.
"""

  n_points, n_neighbors = neighbors.shape
  anchors = jnp.tile(jnp.arange(n_points).reshape([-1, 1]),
                     [1, n_neighbors]).flatten()
  hits = neighbors.flatten()
  distances = sliced_distances(anchors, hits, inputs, distance_fn) ** 2
  distances = distances.reshape([n_points, -1])
  sig = jnp.maximum(jnp.mean(jnp.sqrt(distances[:, 3:6]), axis=1), 1e-10)
  scaled_distances = distances / (sig.reshape([-1, 1]) * sig[neighbors])
  sort_indices = jnp.argsort(scaled_distances, 1)
  scaled_distances = jnp.take_along_axis(scaled_distances, sort_indices, 1)
  sorted_neighbors = jnp.take_along_axis(neighbors, sort_indices, 1)
  return scaled_distances, sorted_neighbors, sig


def find_triplet_weights(inputs,
                         triplets,
                         neighbors,
                         distance_fn,
                         sig,
                         distances=None):
  """Calculates the weights for the sampled nearest neighbors triplets.

  Args:
    inputs: Input points.
    triplets: Nearest neighbor triplets.
    neighbors: Nearest neighbors.
    distance_fn: Distance function.
    sig: Scaling factor for the distances
    distances: Nearest neighbor distances.

  Returns:
    weights: Triplet weights.
  """
  n_points, n_inliers = neighbors.shape
  if distances is None:
    anchs = jnp.tile(jnp.arange(n_points).reshape([-1, 1]),
                     [1, n_inliers]).flatten()
    inliers = neighbors.flatten()
    distances = sliced_distances(anchs, inliers, inputs, distance_fn) ** 2
    p_sim = -distances / (sig[anchs] * sig[inliers])
  else:
    p_sim = -distances.flatten()
  n_outliers = triplets.shape[0] // (n_points * n_inliers)
  p_sim = jnp.tile(p_sim.reshape([n_points, n_inliers]),
                   [1, n_outliers]).flatten()
  out_distances = sliced_distances(triplets[:, 0], triplets[:, 2], inputs,
                                   distance_fn) ** 2
  p_out = -out_distances / (sig[triplets[:, 0]] * sig[triplets[:, 2]])
  weights = p_sim - p_out
  return weights


def generate_triplets(key,
                      inputs,
                      n_inliers,
                      n_outliers,
                      n_random,
                      weight_temp=0.5,
                      distance='euclidean',
                      verbose=False,
                      return_knn_aux=False):
  """Generate triplets.

  Args:
    key: Random key.
    inputs: Input points.
    n_inliers: Number of inliers.
    n_outliers: Number of outliers.
    n_random: Number of random triplets per point.
    weight_temp: Temperature of the log transformation on the weights.
    distance: Distance type.
    verbose: Whether to print progress.
  precomputed_embeddings: Precomputed embeddings. Used as neighbours, but not as anchors.
  return_knn_aux: Whether to return knn auxiliaries.

  Returns:
    triplets and weights
  """

  n_points = inputs.shape[0]
  n_extra = min(n_inliers + 50, n_points)
  index = pynndescent.NNDescent(inputs, metric=distance)
  index.prepare()
  neighbors = index.query(inputs, n_extra)[0]

  neighbors = np.concatenate((np.arange(inputs.shape[0]).reshape([-1, 1]), neighbors),
                             1)
  if verbose:
    logging.info('found nearest neighbors')
  distance_fn = get_distance_fn(distance)
  # conpute scaled neighbors and the scale parameter
  knn_distances, neighbors, sig = find_scaled_neighbors(inputs, neighbors,
                                                        distance_fn)
  neighbors = neighbors[:, :n_inliers + 1]
  knn_distances = knn_distances[:, :n_inliers + 1]
  key, use_key = random.split(key)
  triplets = sample_knn_triplets(use_key, neighbors, n_inliers, n_outliers)
  weights = find_triplet_weights(
    inputs,
    triplets,
    neighbors[:, 1:n_inliers + 1],
    distance_fn,
    sig,
    distances=knn_distances[:, 1:n_inliers + 1])
  flip = weights < 0
  anchors, pairs = triplets[:, 0].reshape([-1, 1]), triplets[:, 1:]
  pairs = jnp.where(
    jnp.tile(flip.reshape([-1, 1]), [1, 2]), jnp.fliplr(pairs), pairs)
  triplets = jnp.concatenate((anchors, pairs), 1)

  if n_random > 0:
    key, use_key = random.split(key)
    rand_triplets, rand_weights = sample_random_triplets(
      use_key, inputs, n_random, distance_fn, sig)

    triplets = jnp.concatenate((triplets, rand_triplets), 0)
    weights = jnp.concatenate((weights, 0.1 * rand_weights))

  weights -= jnp.min(weights)
  weights = tempered_log(1. + weights, weight_temp)

  if return_knn_aux:
    return triplets, weights, (neighbors, knn_distances)

  return triplets, weights


@jax.jit
def update_embedding_dbd(embedding, grad, vel, gain, lr, iter_num):
  """Update the embedding using delta-bar-delta."""
  gamma = jnp.where(iter_num > _SWITCH_ITER, _FINAL_MOMENTUM, _INIT_MOMENTUM)
  # if grad is nan, set to 0
  grad = jnp.where(jnp.isnan(grad), 0.0, grad)
  gain = jnp.where(
    jnp.sign(vel) != jnp.sign(grad), gain + _INCREASE_GAIN,
    jnp.maximum(gain * _DAMP_GAIN, _MIN_GAIN))

  vel = gamma * vel - lr * gain * grad
  embedding += vel
  return embedding, vel, gain


def metric_grad(x, y, metric):
  if callable(metric):
    single_grad = metric
  else:
    if metric not in distances.named_distances_with_gradients:
      raise ValueError("Not gradient method for", metric)

    single_grad = distances.named_distances_with_gradients[metric]

  d, grad = jax.vmap(single_grad)(x, y)
  return d, grad


def trimap_metrics_grad(embedding, triplets, weights, metric):
  anc_idx = triplets[:, 0]
  sim_idx = triplets[:, 1]
  out_idx = triplets[:, 2]

  anc_points = embedding[anc_idx]
  sim_points = embedding[sim_idx]
  out_points = embedding[out_idx]

  # Get distances and gradients wrt anchor points
  sim_distance, sim_grad = metric_grad(anc_points, sim_points, metric)  # grad wrt anchor
  out_distance, out_grad = metric_grad(anc_points, out_points, metric)  # grad wrt anchor

  sim_distance += 1
  out_distance += 1

  ratio = out_distance / sim_distance
  loss_term = weights / (1.0 + ratio)

  loss = jnp.mean(loss_term)

  # Derivative of loss w.r.t. distances
  dL_dratio = -weights / (1.0 + ratio) ** 2
  dratio_dsim = -out_distance / sim_distance ** 2
  dratio_dout = 1.0 / sim_distance

  dL_dsim = dL_dratio * dratio_dsim  # shape (N,)
  dL_dout = dL_dratio * dratio_dout  # shape (N,)

  dL_dsim = dL_dsim[:, None]
  dL_dout = dL_dout[:, None]

  # Gradient of loss w.r.t. anchor, sim, out
  grad_anc = dL_dsim * sim_grad + dL_dout * out_grad
  grad_sim = -dL_dsim * sim_grad  # sim is "negative side" of the distance
  grad_out = -dL_dout * out_grad  # outlier is also "negative side"

  grad = jnp.zeros_like(embedding)
  grad = grad.at[anc_idx].add(grad_anc)
  grad = grad.at[sim_idx].add(grad_sim)
  grad = grad.at[out_idx].add(grad_out)

  return loss, grad


def trimap_metrics(embedding, triplets, weights, metric='euclidean'):
  """Return trimap loss and number of violated triplets.

  Args:
    embedding: The embedding array.
    triplets: Triplet indices.
    weights: Triplet weights.
    metric: Distance metric to use in the embedding space.
  """

  anc_points = embedding[triplets[:, 0]]
  sim_points = embedding[triplets[:, 1]]
  out_points = embedding[triplets[:, 2]]
  fn = get_output_distance_fn(metric)
  sim_distance = 1. + fn(anc_points, sim_points)
  out_distance = 1. + fn(anc_points, out_points)
  num_violated = jnp.sum(sim_distance > out_distance)
  loss = jnp.mean(weights * 1. / (1. + out_distance / sim_distance))
  return loss, num_violated


@jax.jit
def trimap_loss(embedding, triplets, weights, output_metric='euclidean'):
  """Return trimap loss."""
  loss, _ = trimap_metrics(embedding, triplets, weights, metric=output_metric)
  return loss


def transform(key,
              inputs,
              n_dims=2,
              n_inliers=10,
              n_outliers=5,
              n_random=3,
              weight_temp=0.5,
              distance='euclidean',
              output_metric='squared_euclidean',
              lr=0.1,
              n_iters=400,
              init_embedding='pca',
              apply_pca=True,
              triplets=None,
              weights=None,
              verbose=False,
              export_iters=False,
              auto_diff=True):
  """Transform inputs using TriMap.

  Args:
    key: Random key.
    inputs: Input points.
    n_dims: Number of output dimension.
    n_inliers: Number of inliers.
    n_outliers: Number of outliers.
    n_random: Number of random triplets per point.
    weight_temp: Temperature of the log transformation on the weights.
      distance: Distance type (input space).
      output_metric: Output metric (embedding space).
    lr: Learning rate.
    n_iters: Number of iterations.
    init_embedding: Initial embedding: pca, random, or pass pre-computed.
    apply_pca: Apply PCA to reduce the dimension for knn search.
    triplets: Use pre-sampled triplets.
    weights: Use pre-computed weights.
    verbose: Whether to print progress.
      export_iters: Whether to export the embedding at each iteration.
      auto_diff: Whether to use automatic differentiation for the loss.

    Returns:
      embedding
    """

  if verbose:
    t = time.time()
  n_points, dim = inputs.shape
  assert n_inliers < n_points - 1, (
    'n_inliers must be less than (number of data points - 1).')
  if verbose:
    logging.info('running TriMap on %d points with dimension %d', n_points, dim)
  pca_solution = False
  if triplets is None:
    if verbose:
      logging.info('pre-processing')
    if distance != 'hamming':
      if dim > _DIM_PCA and apply_pca:
        inputs -= np.mean(inputs, axis=0)
        inputs = TruncatedSVD(
          n_components=_DIM_PCA, random_state=0).fit_transform(inputs)
        pca_solution = True
        if verbose:
          logging.info('applied PCA')
        else:
          inputs -= np.min(inputs)
          inputs /= np.max(inputs)
          inputs -= np.mean(inputs, axis=0)
    key, use_key = random.split(key)
    triplets, weights = generate_triplets(
      key,
      inputs,
      n_inliers,
      n_outliers,
      n_random,
      weight_temp=weight_temp,
      distance=distance,
      verbose=verbose)
    if verbose:
      logging.info('sampled triplets')
  else:
    if verbose:
      logging.info('using pre-computed triplets')

  if isinstance(init_embedding, str):
    if init_embedding == 'pca':
      if pca_solution:
        embedding = jnp.array(_INIT_SCALE * inputs[:, :n_dims])
      else:
        embedding = jnp.array(
          _INIT_SCALE *
          PCA(n_components=n_dims).fit_transform(inputs).astype(np.float32))
    elif init_embedding == 'random':
      key, use_key = random.split(key)
      embedding = random.normal(
        use_key, shape=[n_points, n_dims], dtype=jnp.float32) * _INIT_SCALE
  else:
    embedding = jnp.array(init_embedding, dtype=jnp.float32)

  n_triplets = float(triplets.shape[0])

  lr = lr * n_points / n_triplets
  if verbose:
    logging.info('running TriMap using DBD')
  vel = jnp.zeros_like(embedding, dtype=jnp.float32)
  gain = jnp.ones_like(embedding, dtype=jnp.float32)

  if export_iters:
    shape = (n_iters, n_points, n_dims)
    embedings_series = np.zeros(shape, dtype=np.float32)

  def differentiable_loss(embedding, triplets, weights):
    """Wrapper for the loss function to make it differentiable."""
    loss, _ = trimap_metrics(embedding, triplets, weights, metric=output_metric)
    return loss

  if not auto_diff:
    trimap_grad = (
      lambda embedding, triplets, weights: trimap_metrics_grad(embedding, triplets, weights, output_metric)[1])
  else:
    trimap_grad = jax.jit(jax.grad(differentiable_loss))

  for itr in range(n_iters):
    gamma = _FINAL_MOMENTUM if itr > _SWITCH_ITER else _INIT_MOMENTUM
    grad = trimap_grad(embedding + gamma * vel, triplets, weights)

    # update the embedding
    embedding, vel, gain = update_embedding_dbd(embedding, grad, vel, gain, lr,
                                                itr)
    if export_iters:
      embedings_series[itr] = embedding
    if verbose:
      if (itr + 1) % _DISPLAY_ITER == 0:
        loss, n_violated = trimap_metrics(embedding, triplets, weights, metric=output_metric)
        logging.info(
          'Iteration: %4d / %4d, Loss: %3.3f, Violated triplets: %0.4f',
          itr + 1, n_iters, loss, n_violated / n_triplets * 100.0)
  if verbose:
    elapsed = str(datetime.timedelta(seconds=time.time() - t))
    logging.info('Elapsed time: %s', elapsed)
  if export_iters:
    return embedings_series
  return embedding

def _embed_with_base(key,
                     new,
                     base,
                     projected_base,
                     n_inliers=10,
                     n_outliers=5,
                     n_random=3,
                     distance='euclidean',
                     output_metric='euclidean',
                     n_iters=200,
                     lr=0.1,
                     weight_temp=0.5,
                     init_embedding='interpolation',
                     verbose=False,
                     epsilon=1e-7):
  """Embeds a new high-dimensional point into an existing TriMap projection."""
  if verbose:
    t = time.time()
  n_points = new.shape[0]
  projected_dim = projected_base.shape[-1]

  key, use_key = random.split(key)
  full_inputs = np.concatenate((new, base), axis=0)
  triplets, weights, (neig, knn_dist) = generate_triplets(
    use_key,
    full_inputs,
    n_inliers,
    n_outliers,
    n_random,
    weight_temp=weight_temp,
    distance=distance,
    verbose=verbose,
    return_knn_aux=True)
  triplets, weights, neig, knn_dist = triplets[:n_points], weights[:n_points], neig[:n_points], knn_dist[:n_points]
  n_triplets = triplets.shape[0]
  # begin with linear interpolation using weights
  if init_embedding == 'interpolation':
    projected_data = jnp.zeros((n_points, projected_dim))
    projected_base = jnp.concatenate((projected_data, projected_base), axis=0)

    inlier_indices = slice(1, n_inliers + 1)
    neig, knn_dist = neig[:, inlier_indices], knn_dist[:, inlier_indices]
    valid_neig = neig >= n_points

    # triplet weights are all normalized w.r.t same out sample. 0 out weights between 2 new points
    interpolation_weights = jnp.where(valid_neig, jnp.abs(knn_dist), 0)
    interpolation_weights /= (jnp.sum(interpolation_weights, axis=-1, keepdims=True) + epsilon)  # normalize to sum of 1
    interpolation_weights = jnp.expand_dims(interpolation_weights, -1)

    weighted_embeddings = interpolation_weights * projected_base[neig]
    weighted_embeddings = jnp.reshape(weighted_embeddings, (n_points, n_inliers, projected_dim))
    projected_data = jnp.sum(weighted_embeddings, axis=1)

  elif init_embedding == 'random':
    projected_data = random.uniform(key, shape=(n_points, projected_dim))
  elif init_embedding == 'zero':
    projected_data = jnp.zeros((n_points, projected_dim))
  else:
    raise NotImplementedError(f"Invalid init_embedding {init_embedding}")

  lr = lr * n_points / float(n_triplets)

  if verbose:
    logging.info('running TriMap using DBD')

  vel = jnp.zeros_like(projected_data, dtype=jnp.float32)
  gain = jnp.ones_like(projected_data, dtype=jnp.float32)

  modified_trimap_loss = lambda x, y: trimap_loss(jnp.concatenate((x, y), axis=0), triplets, weights)
  trimap_grad = jax.jit(jax.grad(modified_trimap_loss))

  for itr in range(n_iters):
    gamma = _FINAL_MOMENTUM if itr > _SWITCH_ITER else _INIT_MOMENTUM
    grad = trimap_grad(projected_data + gamma * vel, projected_base)
    # update the embedding
    projected_data, vel, gain = update_embedding_dbd(projected_data, grad, vel, gain, lr, itr)
    if verbose:
      if (itr + 1) % _DISPLAY_ITER == 0:
        loss, n_violated = trimap_metrics(jnp.concatenate((projected_data, projected_base)), triplets, weights)
        logging.info(
          'Iteration: %4d / %4d, Loss: %3.3f, Violated triplets: %0.4f',
          itr + 1, n_iters, loss, n_violated / n_triplets * 100.0)

  if verbose:
    elapsed = str(datetime.timedelta(seconds=time.time() - t))
    logging.info('Elapsed time: %s', elapsed)

  return projected_data

def inverse_transform(key,
                      new_embeddings,
                      original_data,
                      embeddings,
                      n_inliers=10,
                      n_outliers=5,
                      n_random=3,
                      weight_temp=0.5,
                      distance='euclidean',
                      lr=0.1,
                      n_iters=20,
                      init_embedding='interpolation',
                      verbose=False):
  inversed_data = _embed_with_base(key, new_embeddings, base=embeddings, projected_base=original_data,
                                   n_inliers=n_inliers, n_outliers=n_outliers, n_random=n_random,
                                   weight_temp=weight_temp, distance=distance,
                                   lr=lr, n_iters=n_iters, init_embedding=init_embedding, verbose=verbose)
  return inversed_data


def embed_new_points(key,
                    new_points,
                    original_data,
                    embeddings,
                    n_inliers=10,
                    n_outliers=5,
                    n_random=3,
                    distance='euclidean',
                    output_metric='euclidean',
                    n_iters=200,
                    lr=0.1,
                    weight_temp=0.5,
                    init_embedding='interpolation',
                    verbose=False):
  """Embeds a new high-dimensional point into an existing TriMap projection."""
  projected_data = _embed_with_base(key, new_points, base=original_data, projected_base=embeddings,
                                    n_inliers=n_inliers, n_outliers=n_outliers, n_random=n_random,
                                    weight_temp=weight_temp, distance=distance,
                                    lr=lr, n_iters=n_iters, init_embedding=init_embedding, verbose=verbose)

  return projected_data
