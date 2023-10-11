import numpy as np

from bstrapping.interfaces.weights import Weights


def triangle_window(m, j):
    return 1 / m * (1 - np.abs(j) / m)


class MovingAverageWeights(Weights):
    r"""Generates moving average weights

    .. epigraph::
        **When to apply:**
        These weights are only valid for iid data_old.

    Generate number-of-samples many realizations of moving average weights.

    **Formula**

    .. math ::

        V_i = \sum_{j\in \mathcal{Z}} b_j \eta_{i-j}

        m = n^{\frac{1}{3}}

    where n denotes the number of samples and

    .. math ::

        b_j= m^{-1}(1-\frac{|j|}{m})

    if

    .. math ::

        |j|\leq m

    and 0 else.

    References
    ----------
    See 6.1 and 6.2 in https://www.research-collection.ethz.ch/handle/20.500.11850/141415

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
        self._gamma_weights = None

    def __call__(self, ) -> np.ndarray:
        """Generate realizations of the sequence of weights

        Returns
        -------
        np.ndarray
            number of samples many realizations of normally distributed
            iid random variables with mean and variance equal to one

        """
        number_sample_points = len(self._samples)
        block_length = int(number_sample_points ** (1 / 3))  # =l_n page 92 Bühlmann

        q = 2 / (3 * block_length) + 1 / (3 * block_length ** 3)
        gamma_weights = [
            np.random.gamma(q, 1 / q)
            for _ in range(number_sample_points + 2 * block_length)
        ]
        self._gamma_weights = gamma_weights

        return np.array([
            np.sum([
                triangle_window(block_length, j) * gamma_weights[t - j]
                for j in range(-block_length, block_length + 1)
            ]) for t in range(number_sample_points)
        ])
