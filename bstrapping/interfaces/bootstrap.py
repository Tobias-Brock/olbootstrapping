from abc import abstractmethod
from typing import Dict

import numpy as np


class Bootstrap:
    """Generic interface for bootstrapping
    """

    @property
    @abstractmethod
    def samples(self) -> np.ndarray:
        """

        Returns
        -------
        np.ndarray
            samples on which the bootstrap is based on stored in 2 dimensional array
            with first dimension corresponding to the number of samples
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def plain_bootstrapped_samples(self) -> np.ndarray:
        """

        Returns
        -------
        np.ndarray
            bootstrap samples stored in 3 dimensional array with first dimension
            corresponding to the number of bootstrap samples and second to the number of samples
        """
        raise NotImplementedError

    @property
    def bootstrapped_samples(self) -> Dict[str, np.ndarray]:
        """Human readable form of the bootstrap samples

        Returns
        -------
        Dict[str, np.ndarray]
            keys indicate the number of the respective bootstrap sample whereas the value is the respective bootstrap sample
        """
        return {f'{counter}/{self.number_samples} sample': value for counter, value in
                enumerate(self.plain_bootstrapped_samples)}

    @property
    def number_samples(self) -> int:
        """

        Returns
        -------
        int
            number of samples
        """
        return np.shape(self.samples)[0]

    @property
    def dimension_samples(self) -> int:
        """

        Returns
        -------
        int
            dimension of each sample
        """
        return np.shape(self.samples)[1]

    @property
    def bootstrapped_means(self) -> np.ndarray:
        """Calculate the mean of the bootstrapped samples for each sample and dimension

        Returns
        -------
        np.ndarray
            means of bootstrapped samples
        """
        return np.average(self.plain_bootstrapped_samples, axis=1)

    @property
    def bootstrapped_variance(self) -> np.ndarray:
        """Calculate the variance of the bootstrapped samples in each dimension

        Returns
        -------
        np.ndarray
            variance of the average of the bootstrapped samples
        """
        return np.var(self.bootstrapped_means)
