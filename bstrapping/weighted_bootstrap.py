import numpy as np
from tqdm import tqdm

from bstrapping.interfaces.bootstrap import Bootstrap
from bstrapping.interfaces.weights import Weights


class WeightedBootstrap(Bootstrap):
    def __init__(self,
                 samples: np.ndarray,
                 weights: Weights,
                 number_bootstrap_samples: int = 100,
                 ):
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
            weight = weights()
            # TODO: Check
            resampled_points.append(np.average(weight) * weight * self.samples for _, _ in
                                    enumerate(self.samples))

        self._plain_bootstrapped_samples = np.array(resampled_points)

    @property
    def samples(self) -> np.ndarray:
        return self._samples

    @property
    def plain_bootstrapped_samples(self) -> np.ndarray:
        return self._plain_bootstrapped_samples
