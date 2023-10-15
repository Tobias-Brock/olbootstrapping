from abc import abstractmethod
from typing import Optional

import numpy as np


class OnlineBootstrap:
    @abstractmethod
    def __call__(self, new_samples: np.ndarray, number_bootstrap_samples: Optional[int] = None):
        raise NotImplementedError

    @property
    def dimension_samples(self) -> int:
        """

        Returns
        -------
        int
            dimension of each sample
        """
        return np.shape(self.average_samples)[1]

    @property
    @abstractmethod
    def bootstrap_averages(self) -> np.ndarray:
        """Calculate the mean of the bootstrapped samples for each sample and dimension

        Returns
        -------
        np.ndarray
            means of bootstrapped samples
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def index_time(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def average_samples(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def estimated_asymptotic_variance(self) -> np.ndarray:
        """Calculate the variance of the bootstrapped samples in each dimension

        Returns
        -------
        np.ndarray
            variance of the average of the bootstrapped samples
        """
        return self.index_time * np.var(self.bootstrap_averages)

    def bootstrapped_quantile(self, q: float) -> float:
        """Calculate the standardized quantile of the bootstrapped samples
        """
        standardized_samples = (self.bootstrap_averages - np.mean(self.average_samples)) / np.std(
            self.bootstrap_averages)
        return np.quantile(a=standardized_samples, q=q)

    def bootstrapped_confidence_interval(self,
                                         alpha: float = 0.05,
                                         beta: Optional[float] = None) -> np.ndarray:
        if beta is None:
            beta = 1 / 2 * alpha
            alpha = 1 / 2 * alpha

        lower_quantile = self.bootstrapped_quantile(q=1 - beta)
        upper_quantile = self.bootstrapped_quantile(q=alpha)

        return self.average_samples - np.array([lower_quantile * np.std(self.bootstrap_averages),
                                                upper_quantile * np.std(self.bootstrap_averages)])
