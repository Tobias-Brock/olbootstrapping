import numpy as np

from bstrapping.interfaces.weights import Weights


class GaussianWeights(Weights):
    def __init__(self, samples: np.ndarray):
        self._samples = samples

    def __call__(self, ):
        return np.array([np.random.normal(loc=1, scale=1) for _, _ in enumerate(self._samples)])
