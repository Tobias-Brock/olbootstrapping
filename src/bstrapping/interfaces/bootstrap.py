from abc import abstractmethod
from typing import Dict

import numpy as np


class Bootstrap:

    @property
    @abstractmethod
    def samples(self) -> np.ndarray:
        raise NotImplementedError

    @property
    @abstractmethod
    def plain_bootstrapped_samples(self) -> np.ndarray:
        # 3 dim array: first: number bootstrapped samples,
        # second sample in bootstrapped samples, third dimension of samples
        raise NotImplementedError

    @property
    def bootstrapped_samples(self) -> Dict[str, np.ndarray]:
        return {f'{counter}/{self.number_samples} sample': value for counter, value in
                enumerate(self.plain_bootstrapped_samples)}

    @property
    def number_samples(self) -> int:
        return np.shape(self.samples)[0]

    @property
    def dimension_samples(self) -> int:
        return np.shape(self.samples)[1]

    @property
    def bootstrapped_means(self) -> np.ndarray:
        return np.average(self.plain_bootstrapped_samples, axis=1)

    @property
    def bootstrapped_variance(self) -> np.ndarray:
        return np.var(self.bootstrapped_means)
