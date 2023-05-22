"""Example script for applying the weighted bootstrap with Gaussian and recursive defined weights
to independent and identically distributed samples


"""

import numpy as np
from bstrapping.bootstrap_procedures.weighted_bootstrap import WeightedBootstrap

# specify variance, mean and number of the samples
from bstrapping.weights.gaussian_weights import GaussianWeights
from bstrapping.weights.auto_regressive_weights import AutoRegressiveWeights
from bstrapping.weights.moving_average import MovingAverage

variance = 10
mean = 4
number_sample_points = 1000

# generate samples from a normal distribution
samples = np.random.multivariate_normal(
    mean=mean * np.ones(number_sample_points),
    cov=variance * np.identity(number_sample_points))

# Perform the weighted bootstrap
weights = GaussianWeights(samples=samples)
bootstrap = WeightedBootstrap(samples=samples, weights=weights)

# Print bootstrapped variance of the empirical mean along with the true variance
print(f'Bootstrapped variance: \n {bootstrap.bootstrapped_variance}')

print(f'True variance of empirical mean: {variance / number_sample_points}')

weights = AutoRegressiveWeights(samples=samples)
bootstrap = WeightedBootstrap(samples=samples, weights=weights)
# Print bootstrapped variance of the empirical mean along with the true variance
print(f'Bootstrapped variance: \n {bootstrap.bootstrapped_variance}')

print(f'True variance of empirical mean: {variance / number_sample_points}')

weights = MovingAverage(samples=samples)
bootstrap = WeightedBootstrap(samples=samples, weights=weights)
# Print bootstrapped variance of the empirical mean along with the true variance
print(f'Bootstrapped variance: \n {bootstrap.bootstrapped_variance}')

print(f'True variance of empirical mean: {variance / number_sample_points}')
