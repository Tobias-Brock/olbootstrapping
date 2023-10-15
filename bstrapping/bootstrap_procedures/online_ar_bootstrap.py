from typing import Optional

import numpy as np
from tqdm import tqdm

from bstrapping.interfaces.online_bootstrap import OnlineBootstrap
from bstrapping.weights.auto_regressive_weights import generate_recursive_weight


class OnlineARBootstrap(OnlineBootstrap):
    def __init__(self,
                 average_samples: Optional[np.ndarray] = None,
                 index_time: Optional[int] = None,
                 previous_weights: Optional[np.ndarray] = None,
                 bootstrap_averages: Optional[np.ndarray] = None,
                 average_weights: Optional[np.ndarray] = None, ):

        self._bootstrap_averages = bootstrap_averages
        self._average_weights = average_weights
        self._previous_weights = previous_weights
        self._index_time = index_time  # starts at 1
        self._average_samples = average_samples

        if np.any([average_samples is None,
                   index_time is None,
                   previous_weights is None,
                   bootstrap_averages is None,
                   average_weights is None]):
            print('Some parameter(s) were unspecified. Bootstrap procedure starts at time index 0.')
            self._index_time = 0
            self._bootstrap_averages = None
            self._average_samples = None
            self._average_weights = None
            self._previous_weights = None

        elif previous_weights.shape != average_weights.shape:
            raise ValueError(f'previous_weights and sum_old_weights must have the same shape: '
                             f'shape of previous_weights: {previous_weights.shape}, '
                             f'shape of sum_old_weights: {average_weights.shape}')

        elif len(previous_weights) != len(bootstrap_averages):
            raise ValueError('previous_weights and bootstrap_averages must have the same length: '
                             'len(previous_weights): {len(previous_weights)}, '
                             'len(bootstrap_averages): {len(bootstrap_averages)}')

    def __call__(self, new_samples: np.ndarray, number_bootstrap_samples: Optional[int] = None):
        if len(new_samples.shape) == 1:
            new_samples = new_samples.reshape(-1, 1)  # make sure that new_samples is 2 dimensional

        if self._bootstrap_averages is None:
            if number_bootstrap_samples is None:
                raise ValueError(
                    'Either old_bootstrap_samples and sum_old_weights or number_bootstrap_samples must be specified')

            else:
                self._bootstrap_averages = np.zeros((number_bootstrap_samples, new_samples.shape[1]))
                self._average_weights = np.zeros((number_bootstrap_samples, 1))
                self._previous_weights = np.zeros((number_bootstrap_samples, 1))
                self._average_samples = np.zeros(new_samples.shape[1])

        if new_samples.shape[1] != self._bootstrap_averages.shape[1]:
            raise ValueError(f'new_samples and bootstrap_averages must have the same dimension: '
                             f'dimension of new_samples: {new_samples.shape[1]}, '
                             f'dimension of bootstrap_averages: {self._bootstrap_averages.shape[1]}')

        for index, sample in tqdm(enumerate(new_samples), desc='Bootstrapping '):
            self._index_time += 1
            self._previous_weights = self.new_weights_ar()
            self._bootstrap_averages = self.online_update(sample, self._previous_weights, )
            self._average_weights = ((1 - 1 / self._index_time) * self._average_weights + 1 /
                                     self._index_time * self._previous_weights)
            self._average_samples = ((self._index_time - 1) * self._average_samples /
                                     self._index_time + sample / self._index_time)

    def online_update(self,
                      new_sample,
                      new_weights, ):
        return (((self._index_time - 1) * self._average_weights * self._bootstrap_averages + new_weights * new_sample) /
                ((self._index_time - 1) * self._average_weights + new_weights))

    def new_weights_ar(self, ) -> np.ndarray:
        return np.array([generate_recursive_weight(self._index_time, V_i, alpha=2 ** (1 / 2) - 1) for V_i in
                         self._previous_weights])

    def generate_recursive_weight(self, i: int, V_i: float, alpha: float):
        return 1 + (1 - i ** -alpha) * (V_i - 1) + (1 - (1 - i ** -alpha) ** 2) ** (1 / 2) * np.random.normal(loc=0,
                                                                                                              scale=1)

    @property
    def bootstrap_averages(self) -> np.ndarray:
        """Calculate the mean of the bootstrapped samples for each sample and dimension

        Returns
        -------
        np.ndarray
            means of bootstrapped samples
        """
        return self._bootstrap_averages

    @property
    def index_time(self) -> int:
        return self._index_time

    @property
    def average_samples(self) -> np.ndarray:
        return self._average_samples
