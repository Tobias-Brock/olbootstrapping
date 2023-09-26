from bstrapping.weights.auto_regressive_weights import AutoRegressiveWeights

from pybe.benchmark import Benchmark
from pybe.wrappers import timer

import numpy as np
from bstrapping.bootstrap_procedures.weighted_bootstrap import WeightedBootstrap

# specify variance, mean and number of the samples
from bstrapping.weights.gaussian_weights import GaussianWeights

inputs = [100, 200, 500, 1000]

variance = 10
mean = 4

benchmark = Benchmark()


@timer
def gaussian_weights(i: int):
    number_sample_points = i
    true_variance = variance / number_sample_points

    # generate samples from a normal distribution
    samples = np.array([np.random.normal(loc=mean, scale=variance ** (1 / 2)) for _ in range(number_sample_points)])

    # Perform the weighted bootstrap
    weights = GaussianWeights(samples=samples)
    bootstrap = WeightedBootstrap(samples=samples, weights=weights)
    return {'Relative (to true) variance': float(bootstrap.bootstrapped_variance) / true_variance}


@timer
def auto_regressive_weights(i: int):
    number_sample_points = i
    true_variance = variance / number_sample_points

    # generate samples from a normal distribution
    samples = np.array([np.random.normal(loc=mean, scale=variance ** (1 / 2)) for _ in range(number_sample_points)])

    # Perform the weighted bootstrap
    weights = AutoRegressiveWeights(samples=samples)
    bootstrap = WeightedBootstrap(samples=samples, weights=weights)
    return {'Relative (to true) variance': float(bootstrap.bootstrapped_variance) / true_variance}


# benchmark different weights
benchmark(function=gaussian_weights, name='Gaussian iid', inputs=inputs, parallel=False,
          number_runs=100)
benchmark(function=auto_regressive_weights, name='Auto regressive iid', inputs=inputs,
          parallel=False, number_runs=100)
