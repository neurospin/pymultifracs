"""
.. _benchmark:

===============================
Benchmarking estimation methods
===============================

The pymultifracs toolbox provides a tool to benchmark different methods for
estimating the multifractal exponent, different parameter for those methods
on different synthetic scale free time series, with different parameter for
those time series.

This is done via the :class:`~pymultifracs.robust.benchmark.Benchmark` class
We provide here a simple example for Benchmarking methods.
"""

import numpy as np

from pymultifracs.simul import fbm, mrw
from pymultifracs import wavelet_analysis, mfa
from pymultifracs.robust.benchmark import Benchmark

# %%
# We must first define the functions generating signals

N = 2 ** 12

def fbm_gen(H):
    return fbm(H=H, shape=(N, 40))

def mrw_gen(H):
    return mrw(H=H, shape=(N, 40), L=N, lam=np.sqrt(.05))


# %%
# Then we define the signal generation parameter grids:

signal_param_grid = {
    'fbm':{
        'H': np.array([.7, .8])
    },
    'mrw': {
        'H': np.array([.7, .9])
    }
}

signal_gen_grid = {
    'fbm': fbm_gen,
    'mrw': mrw_gen,
}


# %%
# We then define the functions that will estimate the multifractal analysis
# paramters.

def pleader_est(X, p_exp):

    WT = wavelet_analysis(X).get_leaders(p_exp=p_exp)
    pwt = mfa(WT, [(3, 7)], estimates='c', weighted='Nj')

    return {
        'c1': pwt.cumulants.c1[0, :, 0],
        'c2': pwt.cumulants.c2[0, :, 0],
    }

def coef_est(X):

    WT = wavelet_analysis(X)
    dwt = mfa(WT, [(3, 7)], estimates='c')

    return {
        'c1': dwt.cumulants.c1[0, :, 0],
        'c2': dwt.cumulants.c2[0, :, 0],
    }


# %%
# This is completed by defining the parameter grids, similarly to how it was
# done for the signal generating functions.

estimation_param_grid = {
    'pleader': {
        'p_exp': np.array([0.5, 1, 2, 5]),
        },
    'coef': {}
}
estimation_grid = {
    'pleader': pleader_est,
    'coef': coef_est,
}

# %%
# Finally, we instantiate the :class:`Benchmark` class and compute the
# benchmark.

bench = Benchmark(
    signal_gen_grid, signal_param_grid, estimation_grid, estimation_param_grid)
bench.compute_benchmark()

# %%
# We obtain results in the form of a dataframe, which contains all combinations
# of signal model and signal parameters, and analysis method and method
# parameters.

bench.results.loc[:, :, 'pleader'].head(10)

# %%
# We can then derive statistics from that dataframe.

bench.results.loc['fbm', .7, 'pleader'].groupby('p_exp').mean()

# %%
# Seaborn can effectively use the results dataframe to show group statistics.
# For instance, comparing the performance of the wavelet p-leader estimation
# is a one-line command:

import seaborn as sns

sns.boxplot(data=bench.results.loc['mrw', :, 'pleader'], x='p_exp', y='c2')
