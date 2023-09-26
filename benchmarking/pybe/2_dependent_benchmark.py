from typing import Optional

from bstrapping.weights.auto_regressive_weights import AutoRegressiveWeights

from pybe.benchmark import Benchmark
from pybe.wrappers import timer

import numpy as np
from bstrapping.bootstrap_procedures.weighted_bootstrap import WeightedBootstrap

# specify variance, mean and number of the samples
from bstrapping.weights.gaussian_weights import GaussianWeights

inputs = [500, 1000, 2500]

mean = 1
variance = 2

benchmark = Benchmark()


# iid, block and our sequence

# iid, 1 dependent and alpha mixing

# AR 1 process and GARCH process

# schwach, mittel, starke abhängigket

# beta = 0.1,0.5,optimal

# 1000 bootstrap samples, 1000 runs

# over sample sizes

@timer
def gaussian_weights(number_sample_points: int, a: float):
    # sample number_sample_points-often from distribution (not iid): sample_i = Y_i+a*Y_{i+1}
    Y = [np.random.normal(loc=mean, scale=variance ** (1 / 2)) for _ in range(number_sample_points + 1)]
    samples = np.array(Y[:-1]) + a * np.array(Y[1:])
    true_variance = (1 + a) ** 2 / number_sample_points * variance

    # Perform the weighted bootstrap
    weights = GaussianWeights(samples=samples)
    bootstrap = WeightedBootstrap(samples=samples, weights=weights)
    return {f'Relative (to true) variance - a={a}': float(bootstrap.bootstrapped_variance) / true_variance,
            }


@timer
def auto_regressive_weights(number_sample_points: int, a: float, alpha: Optional[float]):
    # sample number_sample_points-often from distribution (not iid): sample_i = Y_i+a*Y_{i+1}
    Y = [np.random.normal(loc=mean, scale=variance ** (1 / 2)) for _ in range(number_sample_points + 1)]
    samples = np.array(Y[:-1]) + a * np.array(Y[1:])
    true_variance = (1 + a) ** 2 / number_sample_points * variance

    # Perform the weighted bootstrap
    weights = AutoRegressiveWeights(samples=samples, alpha=alpha)
    bootstrap = WeightedBootstrap(samples=samples, weights=weights)
    return {'Relative (to true) variance': float(bootstrap.bootstrapped_variance) / true_variance}


# benchmark different weights
a = 0

benchmark(function=lambda x: gaussian_weights(x, a=a),
          name=f'Gaussian 2 dependent - AR(1) a={a}',
          inputs=inputs,
          parallel=False,
          number_runs=100)

benchmark(function=lambda x: auto_regressive_weights(x, a=a, alpha=0.5),
          name=f'Auto regressive 2 dependent - AR(1) a={a}',
          inputs=inputs,
          parallel=False,
          number_runs=100)
