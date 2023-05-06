import numpy as np

from bstrapping.interfaces.weights import Weights


class GaussianWeights(Weights):
    r"""Generates Gaussian weights

    Generate number of samples many realizations of normally distributed iid random variables
    with mean and variance equal to one, i.e.

    .. math ::

        V_i \sim \mathcal{N}(1,1)

    These weights are only valid for iid data.

    """

    def __init__(self, samples: np.ndarray):
        """

        Parameters
        ----------
        samples :
        np.ndarray
            samples stored in 2 dimensional array with first dimension corresponding to the number of samples

        """

        self._samples = samples

    def __call__(self, ) -> np.ndarray:
        """

        Returns
        -------
        np.ndarray
            number of samples many realizations of normally distributed
            iid random variables with mean and variance equal to one

        """

        return np.array([np.random.normal(loc=1, scale=1) for _, _ in enumerate(self._samples)])
