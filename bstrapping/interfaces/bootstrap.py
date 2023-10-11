import json
from abc import abstractmethod
from typing import Dict, Optional

import numpy as np


class Bootstrap:
    """Generic interface for bootstrapping
    """

    def save_to_csv(self, file_name: str):
        """Save data_old to csv file

        Parameters
        ----------
        file_path : str
            path to csv file
        """
        with open(f'{file_name}.json', 'w') as f:
            json.dump({'samples': self.samples.tolist(),
                       'plain_bootstrapped_samples': self.plain_bootstrapped_samples.tolist()}, f)

    def load_from_csv(self, file_path: str):
        """Load data_old from csv file

        Parameters
        ----------
        file_path : str
            path to csv file
        """
        with open(file_path) as f:
            data = json.load(f)
            self._samples = np.array(data['samples'])
            self._plain_bootstrapped_samples = np.array(data['plain_bootstrapped_samples'])

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
        """Structured form of the bootstrap samples

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

    def bootstrapped_quantile(self, q: float) -> float:
        """Calculate the standardized quantile of the bootstrapped samples
        """
        standardized_samples = (self.bootstrapped_means - np.mean(self.samples)) / np.sqrt(self.bootstrapped_variance)
        return np.quantile(a=standardized_samples, q=q)

    def bootstrapped_confidence_interval(self,
                                         alpha: float = 0.05,
                                         beta: Optional[float] = None) -> np.ndarray:
        """Return bootstrapped confidence intervals
        """
        if beta is None:
            beta = 1/2*alpha
            alpha = 1/2*alpha

        lower_quantile = self.bootstrapped_quantile(q=1-beta)
        upper_quantile = self.bootstrapped_quantile(q=alpha)

        return np.mean(self.samples) - np.array([lower_quantile * np.sqrt(self.bootstrapped_variance),
                                                 upper_quantile * np.sqrt(self.bootstrapped_variance)])
