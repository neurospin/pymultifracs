"""
.. _analysis_bivariate:

==========================
Analysis of bivariate data
==========================

The ``pymultifracs`` package can perform bivariate multifractal analysis,
and produces bivariate cumulants and structure functions. The purpose of
bivariate MFA is to determine the relationship between the regularity
fluctuations of a pair of time series.

We will be using pre-generated data to serve as an example of bivariate
multi-fractal time series. Let us first load the data; those example time
series are also found in the ``tests/data/`` folder of the repository on
github.
"""

import pooch
from scipy.io import loadmat

url = ('https://github.com/neurospin/pymultifracs/raw/refs/heads/master/tests/'
       'data/DataSet_ssMF.mat')

fname = pooch.retrieve(url=url, known_hash=None, path=pooch.os_cache("bivar"))

X = loadmat(fname)['data'].T

# %%
# The data is a two-dimensional array, with both self-similarity and
# multifractal correlation.

# sphinx_gallery_start_ignore
# pylint: disable=C0413
# flake8: disable=E402
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 600
# sns.set_theme(style="whitegrid")
# sns.set_context('paper')
# sphinx_gallery_end_ignore

import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(X[:, 0])
ax[1].plot(X[:, 1], c='C1')

# %%
# First we need :func:`wavelet_analysis` as ususal, to obtain p-leaders in this case

from pymultifracs import wavelet_analysis

WTpL = wavelet_analysis(X).integrate(.75).get_leaders(2)

# %%
# Then we use :func:`.bivariate.bimfa` on the wavelet p-leaders. We provide
# the same MRQ twice, in order to compute the estimates for all signal pairs.

import numpy as np
from pymultifracs.bivariate import bimfa

q1 = np.array([0, 1, 2])
q2 = np.array([0, 1, 2])

pwt = bimfa(WTpL, WTpL, scaling_ranges=[(3, 9)], q1=q1, q2=q2)

# %%

# We can obtain the multifractal correlation matrix :math:`\rho_{mf}`:

print(pwt.cumulants.rho_mf.squeeze())

# %%
# We can plot the structure function:
pwt.structure.plot()

# %%
# We can plot the bivariate cumulants:
pwt.cumulants.plot()

# %%
# We can plot the Legendre spectrum reconstituted from the log-cumulants:
pwt.cumulants.plot_legendre(h_support=(.1, .95))
