import numpy as np


class MovingAverage:
    def __init__(self, mean: float, parameters: np.ndarray):
        self.mean = mean
        self.parameters = np.append(parameters, 1)
        self.samples = []

    def generate_samples(self, number_samples: int) -> np.ndarray:
        noise_terms = np.random.normal(loc=0, scale=1, size=number_samples + len(self.parameters))

        self.samples = np.array([self.mean + np.sum(
            self.parameters * noise_terms[i:i + len(self.parameters)]) for i in range(number_samples)])

        return self.samples

    @property
    def asymptotic_variance(self) -> float:
        # limit var(1/n^1/2*sum X_i)
        q = len(self.parameters)
        dummy = np.sum(
            [np.sum(self.parameters[:q - i] * self.parameters[i:]) for i in range(1, len(self.parameters))])

        return np.sum(self.parameters ** 2) + 2 * dummy
