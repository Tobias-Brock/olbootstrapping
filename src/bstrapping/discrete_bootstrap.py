# bootstrapped samples
import numpy as np
from tqdm import tqdm

from src.bstrapping.interfaces.bootstrap import Bootstrap

sampled_points_of_distributions_bootstrapped = []
means_bootstrapped = []


class DiscreteBootstrap(Bootstrap):
    def generate_bootstrap_samples(self, number_bootstrap_samples: int = 100) -> None:
        resampled_points = []
        for _ in tqdm(range(number_bootstrap_samples)):
            resampled_points.append([np.random.choice(self.bootstrapped_samples[0]) for _, _ in
                                     enumerate(self.samples)])

        self._plain_bootstrapped_samples = np.array(resampled_points)
