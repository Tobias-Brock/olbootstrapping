import numpy as np

from bstrapping.interfaces.weights import Weights


def rho(i, alpha):
    return 1 - i ** -alpha


def generate_recursive_weight(i: int, V_i: float, alpha: float):
    return 1 + rho(i=i + 1, alpha=alpha) * (V_i - 1) + (1 - rho(i=i + 1, alpha=alpha) ** 2) ** (
            1 / 2) * np.random.normal(loc=0, scale=1)


class AutoRegressiveWeights(Weights):
    r"""Generate recursively defined weights

    .. epigraph::
        **When to apply:**
        These weights are valid for weakly dependent time series

    Generate number-of-samples many realizations of a rescursively defined sequence of bootstrapping weights

    **Formula**

    .. math ::

        \zeta_i\sim \mathcal{N}(0,1)

        V_1=\zeta_1

        V_{i+1}=1+\rho_i(V_{i}-1)+\sqrt{1-\rho_i^2}\zeta_{i+1}

        \rho_i=1-i^{-\alpha}

    """

    def __init__(self,
                 samples: np.ndarray,
                 alpha: float = 2 ** (1 / 2) - 1):
        """

        Parameters
        ----------
        samples :
        np.ndarray
            samples stored in 2 dimensional array with first dimension corresponding to the number of samples

        alpha:
        float
            parameter alpha of the recursive defined weights

        """

        self._samples = samples
        self._alpha = alpha

    def __call__(self, ):
        """Generate realizations of the sequence of weights

        Returns
        -------
        np.ndarray
            number of samples many realizations of normally distributed
            iid random variables with mean and variance equal to one

        """

        weights = [np.random.normal(loc=1, scale=1)]
        for i, _ in enumerate(self._samples):
            weights.append(generate_recursive_weight(i=i, V_i=weights[-1], alpha=self._alpha))
        return np.array(weights[:-1])
