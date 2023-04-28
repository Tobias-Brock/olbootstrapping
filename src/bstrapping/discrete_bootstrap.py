# bootstrapped samples
import numpy as np
from tqdm import tqdm

from src.bstrapping.interfaces.bootstrap import Bootstrap

sampled_points_of_distributions_bootstrapped = []
means_bootstrapped = []


class DiscreteBootstrap(Bootstrap):
    """
    Perform the discrete bootstrap

    The discrete bootstrap generates new samples from a given sample set
    with n elements by drawing n times from the sample set with replacements.
    This bootstrap procedure is only valid if the samples are iid.

    See https://en.wikipedia.org/wiki/Bootstrapping_(statistics) in section Case resampling for further details.
    """

    def __init__(self,
                 samples: np.ndarray,
                 number_bootstrap_samples: int = 100
                 ):
        """
        Parameters
        ----------
        samples :
        nd.array
            samples stored in 2 dimensional array with first dimension corresponding to the number of samples

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
            resampled_points.append([self.samples[np.random.choice(self.number_samples)] for _, _ in
                                     enumerate(self.samples)])

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
            bootstrap samples stored in 3 dimensional array with first dimension
            corresponding to the number of bootstrap samples and second to the number of samples
        """
        return self._plain_bootstrapped_samples
