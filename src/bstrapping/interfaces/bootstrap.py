from abc import abstractmethod
from typing import Dict

import numpy as np


class Bootstrap:
    def __init__(self,
                 samples: np.ndarray,
                 ):
        if len(np.shape(samples)) > 2:
            raise ValueError('Sample array must have maximal 2 dimensions')

        self._plain_bootstrapped_samples = None
        self._samples = samples

        print(f'{self.number_samples} samples with dimension '
              f'{self.dimension_samples} were obtained. \n')

    @abstractmethod
    def generate_bootstrap_samples(self, *args) -> None:
        pass

    @property
    def bootstrapped_samples(self) -> Dict[str, np.ndarray]:
        return {f'{counter}/{self.number_samples} sample': value for counter, value in
                enumerate(self._plain_bootstrapped_samples)}

    @property
    def samples(self) -> np.ndarray:
        return self._samples

    @property
    def number_samples(self) -> int:
        return np.shape(self.samples)[0]

    @property
    def dimension_samples(self) -> int:
        return np.shape(self.samples)[1]

    @property
    def plain_bootstrapped_samples(self) -> np.ndarray:
        return self._plain_bootstrapped_samples

    @property
    def bootstrapped_means(self) -> float:
        return np.average(self._plain_bootstrapped_samples, axis=1)

# mean_bootstrapped = np.average(sampled_points[0])
