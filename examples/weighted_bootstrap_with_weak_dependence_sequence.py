"""Example script for applying the discrete and weighted bootstrap with recursive defined weights to 2-dependent samples


"""

import numpy as np

from bstrapping.bootstrap_procedures.discrete_bootstrap import DiscreteBootstrap
from bstrapping.bootstrap_procedures.weighted_bootstrap import WeightedBootstrap
from bstrapping.weights.auto_regressive_weights import AutoRegressiveWeights

# specify variance, mean and number of the samples
from bstrapping.weights.moving_average import MovingAverage

mean = 1
variance = 2
number_sample_points = 2500

# sample number_sample_points-often from distribution (not iid): sample_i = Y_i+a*Y_{i+1}
a = 0.8
Y = [np.random.normal(loc=mean, scale=variance**(1/2)) for _ in range(number_sample_points+1)]
samples = np.array(Y[:-1]) + a * np.array(Y[1:])


# calculated variance
true_variance = (1 + a) ** 2 / number_sample_points * variance
print(f'True variance of empirical mean: {true_variance}')

# Perform the discrete bootstrap
bootstrap = DiscreteBootstrap(samples=samples, number_bootstrap_samples=1000)
# Print bootstrapped variance of the empirical mean along with the true variance
print(f'Bootstrapped variance: \n {bootstrap.bootstrapped_variance}')

# Perform the weighted bootstrap
weights = AutoRegressiveWeights(samples=samples)
bootstrap = WeightedBootstrap(samples=samples, weights=weights, number_bootstrap_samples=1000)
# Print bootstrapped variance of the empirical mean along with the true variance
print(f'Bootstrapped variance: \n {bootstrap.bootstrapped_variance}')

weights = MovingAverage(samples=samples)
bootstrap = WeightedBootstrap(samples=samples, weights=weights)
# Print bootstrapped variance of the empirical mean along with the true variance
print(f'Bootstrapped variance: \n {bootstrap.bootstrapped_variance}')
