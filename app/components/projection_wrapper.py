import pickle

from google_research_trimap.trimap import trimap
import numpy as np
import jax.random as random

from google_research_trimap.trimap import parametric_trimap as ptrimap


class TrimapWrapper:
    _data: np.ndarray = None
    _embedding: np.ndarray = None
    _parameters: dict = None
    _distance_metric: str = "euclidean"
    _parametric: bool = False
    _model = None
    _params = None
    _key = random.PRNGKey(0)

    @property
    def key(self):
        self._key, key = random.split(self._key)
        return key

    @staticmethod
    def ensure_batch(data):
        if len(data.shape) == 2:
            data = data.expand_dims(0)
        return data

    def fit_transform(self, data, distance_metric, parametric=False):
        self._data = data
        self._parametric = parametric
        self._distance_metric = distance_metric
        if self._parametric:
            self._embedding, self._model, self._params = ptrimap.fit_transform(self.key, self._data, ndims=2)
        else:
            self._embedding = trimap.transform(self.key, data, output_metric=self._distance_metric, export_iters=True)
        return self._embedding

    def transform(self, data):
        data = self.ensure_batch(data)
        if self._parametric:
            return ptrimap.transform(data, self._model, self._params)
        return trimap.embed_new_points(self.key, data, self._data, self._embedding)

    def inverse_transform(self, data):
        data = self.ensure_batch(data)
        if self._parametric:
            return ptrimap.inverse_transform(data, self._model, self._params)
        return trimap.inverse_transform(self.key, data, self._data, self._embedding)

    def store(self, filename='trimap_cache'):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename='trimap_cache'):
        with open(filename, 'rb') as f:
            return pickle.load(f)

