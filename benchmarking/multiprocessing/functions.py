from bstrapping.weights.gaussian_weights import GaussianWeights

from bstrapping.bootstrap_procedures.weighted_bootstrap import WeightedBootstrap
from bstrapping.weights.auto_regressive_weights import AutoRegressiveWeights
from bstrapping.weights.moving_average import MovingAverageWeights


def evaluate_ar(samples, mean=4, alpha=0.05):
    bootstrap = WeightedBootstrap(samples=samples,
                                  weights=AutoRegressiveWeights(samples=samples),
                                  number_bootstrap_samples=100)
    conf_interval = bootstrap.bootstrapped_confidence_interval(alpha=alpha)

    return len(samples) * bootstrap.bootstrapped_variance, conf_interval[0] <= mean <= conf_interval[1]


def evaluate_multiplier_iid(samples, mean=4, alpha=0.05):
    bootstrap = WeightedBootstrap(samples=samples,
                                  weights=GaussianWeights(samples=samples),
                                  number_bootstrap_samples=100)
    conf_interval = bootstrap.bootstrapped_confidence_interval(alpha=alpha)

    return len(samples) * bootstrap.bootstrapped_variance, conf_interval[0] <= mean <= conf_interval[1]


def evaluate_ma(samples, mean=4, alpha=0.05):
    bootstrap = WeightedBootstrap(samples=samples,
                                  weights=MovingAverageWeights(samples=samples),
                                  number_bootstrap_samples=100)
    conf_interval = bootstrap.bootstrapped_confidence_interval(alpha=alpha)

    return len(samples) * bootstrap.bootstrapped_variance, conf_interval[0] <= mean <= conf_interval[1]
