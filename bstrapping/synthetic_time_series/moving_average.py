import numpy as np


class MovingAverage:
    def __init__(self, mean: float, parameters: np.ndarray):
        self.mean = mean
        self.parameters = np.concatenate(([1], parameters))
        self.samples = []

    def generate_samples(self, number_samples: int) -> np.ndarray:
        noise_terms = np.random.normal(loc=0, scale=1, size=number_samples + len(self.parameters))
        q = len(self.parameters)

        self.samples = np.array([self.mean + np.sum([self.parameters[j] * noise_terms[i - j] for j in range(q)])
                                 for i in range(number_samples)])

        return self.samples

    @property
    def asymptotic_variance(self) -> float:
        # limit var(1/n^1/2*sum X_i)
        q = len(self.parameters)

        return np.sum(
            [self.parameters[i] * self.parameters[i + np.abs(j)] for j in range(-q, q) for i in range(q - np.abs(j))])

    @property
    def variance(self) -> float:
        return np.sum(self.parameters ** 2)
