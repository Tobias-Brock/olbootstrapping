# Bstrapping: Bootstrapping in Python

Bstrapping provides several bootstrapping methods implemented in a generic Python interface
and suggestions for when to use which.

## A short summary of bootstrapping

Bootstrapping methods are re-sampling schemes with (asymptotic) theoretical guarantees.
Given some (random) samples according to some distribution, bootstrapping allows you to
generate (computationally cheap) synthetic new samples.
Those samples behave similar in the following sense:
The distribution of the *arithmetic mean* of the synthetic samples is (asymptotically)
the distribution of the *arithmetic mean* of the real samples.
Accordingly, we can use the synthetic samples in order to approximate the distribution of the
arithmetic mean of the real samples.

> **Caution**: bootstrapped samples do **not** approximate the distribution of the real samples, but the arithmetic mean!

However, with some mathematical tricks, one can apply bootstrapping to a multitude of scenarios
where we are interested in some other statistic than the arithmetic mean (see [Advanced Usage](#advanced-usage))

For a more detailled summary of bootstrapping see [this paper](our-paper).

## Installation
The official release is available at PyPi:

```
pip install bstrapping
```

You can clone this repository by running the following command:

```
git clone https://github.com/nicolaipalm/bootstrap
cd bstrapping
pip install
```

## Getting started

Before initializing any bootstrap procedure, the samples need to be
stored in a **2 dimensional numpy array** where the first dimension corresponds to
the number of samples.
Here, we generate 1000 (iid) samples from a normal distribution with mean 1 and variance 10.
```python
import numpy as np
samples = np.random.multivariate_normal(
     mean=np.ones(1),
     cov=10*np.identity(1000)) # generate samples
```

Next, the bootstrap procedure can be executed.
We notice/assume the data to be iid, hence, choose the discrete boostrap.
Say we wish to generate 100 new data sets
(i.e. 100 new data sets with 1000 data points).

```python
from bstrapping.bootstrap_procedures.discrete_bootstrap import DiscreteBootstrap
bootstrap = DiscreteBootstrap(samples=samples,number_bootstrap_samples=100) # Perform discrete bootstrap
```

From the bootstrapped samples, we can for example calculate an estimate of the variance of the
arithmetic mean of the samples.

```python
print(f'Bootstrapped variance: \n {bootstrap.bootstrapped_variance}')

print(f'True variance of arithmetic mean: {variance / number_sample_points}')
```



## When to use which bootstrap?
<table>

<tr>
<th> Name </th> <th> when-to-use </th> <th> Code </th> <th> Reference </th>
</tr>

<tr>
<td>
Discrete bootstrap
</td>
<td>

samples are [iid](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables)

</td>
<td>

    bootstrap = DiscreteBootstrap(samples=samples)

</td>
<td>

</td>
</tr>

<tr>
<td>
Recursively defined bootstrap
</td>
<td>

samples are weakly dependent

</td>
<td>

    weights = RecursiveDefinedWeights(samples=samples)
    bootstrap = WeightedBootstrap(samples=samples, weights=weights)

</td>
<td>

</td>
</tr>

</table>
