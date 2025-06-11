"""
.. _analysis_simulation:

========================
Bootstrap-based analysis
========================
"""

# %% [markdown]
# Bootstrapping may be used in `pymultifracs` to determine empirical confidence intervals for the estimates (scaling functions, exponents), as well as an automated determination of the optimal scaling range among multiple choices.
#
# Let us first generate a Multifractal Random Walks with parameters :math:`H = 0.8` and :math:`\lambda=\sqrt{0.05}`

# %% [python]

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
import numpy as np
from pymultifracs.simul import mrw

X = mrw(shape=2**15, H=0.8, lam=np.sqrt(.05), L=2**15)
plt.plot(X)
plt.show()

##############################################################################
#
# Bootstrapped analysis
# ---------------------

# %% [markdown]
# Wavelet transform is performed as usual:

# %%
from pymultifracs import wavelet_analysis

WT = wavelet_analysis(X, wt_name='db3', j2=None, normalization=1)
WTpL = WT.get_leaders(p_exp=2)

# %%

# The simplest way to use bootstrapping is to include the `R` parameter in
# calling :func:`mfa`, which will perform :math:`R > 1` bootstrapping
# repetitions prior to carrying out the multifractal analysis

from pymultifracs import mfa
from pymultifracs.utils import build_q_log

scaling_ranges = [(2, 8), (3, 8)]
pwt = mfa(WTpL, scaling_ranges, weighted=None, q=build_q_log(.1, 4, 15), R=20)

# %%
# Plotting bootstrapped results
# *****************************

# %% [markdown]
# By default, when bootstrapping was applied, the plotting function display
# the empirical confidence intervals:

pwt.cumulants.plot()

# %% [markdown]
# The multifractal spectrum has confidence intervals for both x- and y-axes.

pwt.spectrum.plot()

# %% [markdown]
# Choice of optimal scaling range
# *******************************
#
# In order to determine the best scaling range among multiple choices,
# it is possible to use a bootstrapped-based criterion (see Leonarduzzi et al., 2014):

print('Second mfa')
pwt = mfa(WTpL, [(2, 8), (3, 8), (2, 7), (3, 7)], weighted=None, R=20)

# %% To determine the optimal scale, the :meth:`.ScalingFunction.find_best_range()` method is used:

pwt.cumulants.find_best_range()

# %%
# By default, it selects an overall best range based on the goodness of fit
# of exponents for all cumulants (or moments), but moment-specific optimal
# scales can be obtained by setting the `per_moment` flag:

pwt.cumulants.find_best_range(per_moment=True)
