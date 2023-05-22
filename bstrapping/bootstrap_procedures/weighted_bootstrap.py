import numpy as np
from tqdm import tqdm

from bstrapping.interfaces.bootstrap import Bootstrap
from bstrapping.interfaces.weights import Weights


class WeightedBootstrap(Bootstrap):
    """Perform the weighted bootstrap

    The weighted bootstrap generates new samples from a given sample set
    by multiplying each sample with a random weight, i.e.

    .. math::
        \tilde{X_i}=V_iX_i

    where X is the sample and V is the random weight.
    This bootstrap procedure may be used for iid samples but also for weakly dependent samples
    by using suitable weights.
    The weights are specified in a sequence of the same length
    as the samples.
    See the weight module for more information.

    Examples
    --------
    Generate 2-dependent (non-iid) samples X according to

    .. math::
        X_i = Y_i+a*Y_{i+1}

    with Y all iid normally distributed

    >>> import numpy as np
    >>> from bstrapping.bootstrap_procedures.weighted_bootstrap import WeightedBootstrap
    >>> from bstrapping.weights.recursive_defined_sequence import AutoRegressiveWeights
    >>> mean, variance, number_sample_points = 1, 2, 2500 # specify variance, mean and number of the samples
    >>> a = 0.8
    >>> Y = [np.random.normal(loc=mean, scale=variance**(1/2)) for _ in range(number_sample_points+1)]
    >>> samples = np.array(Y[:-1]) + a * np.array(Y[1:])


    The true variance of the empirical mean of the samples is given by

    >>> true_variance = (1 + a) ** 2 / number_sample_points * variance
    0.002592

    Initialize weights for the weighted bootstrap, here, we use the recursive defined weights for weakly dependent data

    >>> weights = AutoRegressiveWeights(samples=samples)

    Perform the weighted bootstrap

    >>> bootstrap = WeightedBootstrap(samples=samples, weights=weights, number_bootstrap_samples=1000)

    Calculate the variance of the empirical mean of bootstrapped samples

    >>> bootstrap.bootstrapped_variance
    0.0025440628841643504 # may vary

    """

    def __init__(self,
                 samples: np.ndarray,
                 weights: Weights,
                 number_bootstrap_samples: int = 100,
                 ):
        """
        Parameters
        ----------
        samples :
        np.ndarray
            samples stored in 2 dimensional array with first dimension corresponding to the number of samples

        weights :
        Weights
            random weights for the bootstrap

        number_bootstrap_samples :
        int
            number of bootstrap samples to be generated

        """
        if len(np.shape(samples)) > 2:
            raise ValueError('Sample array must have maximal 2 dimensions')

        if len(np.shape(samples)) == 1:
            samples = samples.reshape(-1, 1)

        self._samples = samples

        print(f'{self.number_samples} samples with dimension '
              f'{self.dimension_samples} were obtained. \n')

        print('Bootstrapping...')
        resampled_points = []
        for _ in tqdm(range(number_bootstrap_samples)):
            weight = weights().reshape(-1, 1)
            resampled_points.append(1 / np.average(weight) * weight * self.samples)

        self._plain_bootstrapped_samples = np.array(resampled_points)

    @property
    def samples(self) -> np.ndarray:
        """

        Returns
        -------
        np.ndarray
            samples given when class was initialized
        """
        return self._samples

    @property
    def plain_bootstrapped_samples(self) -> np.ndarray:
        """

        Returns
        -------
        np.ndarray
            bootstrap samples stored in 3 dimensional array
            with first dimension corresponding to the number of bootstrap samples and second to the number of samples

        """
        return self._plain_bootstrapped_samples
