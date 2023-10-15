"""Example script for applying the weighted bootstrap with Gaussian and recursive defined weights
to independent and identically distributed samples


"""

import numpy as np

from bstrapping.bootstrap_procedures.online_ar_bootstrap import OnlineARBootstrap
from bstrapping.synthetic_time_series.moving_average import MovingAverage

mean = 4  # mean of the time series

# Names of the stochastic processes

# Dependence coefficients of the stochastic processes, i.e. of the moving average processes
dependence_coefficients = [
    np.array([0]),
    np.array([0.5]),
    np.array([0.5 ** i for i in range(1, 3)]),
    np.array([0.5 ** i for i in range(1, 11)]),
    np.array([0.5 ** i for i in range(1, 16)]),
    np.array([0.5 ** i for i in range(1, 21)]),
]

names_dependence_coefficients = [
    'iid',
    'MA(1)',
    'MA(2)',
    'MA(10)',
    'MA(15)',
    'MA(20)',
]

list_name_weights = ['AR',
                     'Multiplier',
                     'MA',
                     ]

# generate samples from a normal distribution
sample_size = 5000
index_dependence = 0
bootstrapped_sample_size = 1000
samples = MovingAverage(mean=mean, parameters=dependence_coefficients[index_dependence]).generate_samples(sample_size)

print(
    f'True variance of empirical mean: '
    f'{MovingAverage(mean=mean, parameters=dependence_coefficients[index_dependence]).asymptotic_variance}')

bootstrap = OnlineARBootstrap()
bootstrap(new_samples=samples[:1000], number_bootstrap_samples=bootstrapped_sample_size)
# Print bootstrapped variance of the empirical mean along with the true variance
print(f'Bootstrapped variance: \n {bootstrap.estimated_asymptotic_variance}')
print(bootstrap.bootstrapped_confidence_interval())

bootstrapped_variances = []

for i in range(1, int(sample_size / 1000)):
    bootstrap(samples[1000 * i:1000 * (i + 1)])
    bootstrapped_variances.append(bootstrap.estimated_asymptotic_variance)

print(f'Bootstrapped variances: \n {bootstrapped_variances}')

bootstrap = OnlineARBootstrap()
bootstrap(new_samples=samples, number_bootstrap_samples=bootstrapped_sample_size)

print(bootstrap.estimated_asymptotic_variance)
