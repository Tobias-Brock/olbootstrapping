"""Example script for applying the discrete bootstrap to independent and identically distributed samples


"""

import numpy as np
from src.bstrapping.discrete_bootstrap import DiscreteBootstrap


number_sample_points = 20 ** 2

# iid normal
variance = 10
mean = 4

samples = np.random.multivariate_normal(
    mean=mean * np.ones(number_sample_points),
    cov=variance * np.identity(number_sample_points))

print(f'True variance of empirical mean: {variance / number_sample_points}')

bootstrap = DiscreteBootstrap(samples, number_bootstrap_samples=1000)
