"""
This script shows how to perform multifractal analysis on two signals and sum
the resulting cumulants.

This is used, for instance, when a MEG session is too long and its results are
split into two files (e.g., posttest and posttest_bis): the MFA analysis is performed
separately on both files and the cumulants are summed at the end.

IMPORTANT:
    log-cumulants (slopes of j x C(j)) are computed AFTER the weighted sum of C(j)

NOTE:
    This example uses the same 1d data as the code demo_pleaders_basic.m written
    by Roberto Leonarduzzi for the PLBMF Matlab toolbox, found at:
    https://www.irit.fr/~Herwig.Wendt/software.html
"""

#-------------------------------------------------------------------------------
# Import pymultifracs package
#-------------------------------------------------------------------------------
import pymultifracs as mf

#-------------------------------------------------------------------------------
# Other imports
#-------------------------------------------------------------------------------
import os
import numpy as np
from scipy.io import loadmat
from copy import deepcopy
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------
# Function to load data
#-------------------------------------------------------------------------------
def get_data_from_mat_file(filename):
    contents = loadmat(filename)
    return contents['data'][0]

#-------------------------------------------------------------------------------
# Load data and split it into two signals
#------------------------------------------------------------------------------- 
# multifractal random walk (c_1=0.75, c_2=-0.05, N=32768)
data_file = 'example_data/mrw07005n32768.mat'  
# Complete path to file
current_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(current_dir, data_file)

data = get_data_from_mat_file(data_file)

signal_1 = data[:2**12]
signal_2 = data[2**12:]


#-------------------------------------------------------------------------------
# Setup analysis
#-------------------------------------------------------------------------------

# Multifractal analysis object
mfa = mf.MFA()

### Set parameters. They can also be set in the constructor of mf.MFA()

# wavelet to be used (see PyWavelets documentation for more options)
mfa.wt_name = 'db3'

# value of p for p-leaders, can be numpy.inf
# NOTE: instead of defining the value of p, we can set the variable mfa.formalism,
#       e.g., mfa.formalism = 'wlmf' (corresponding to p = np.inf) or
#             mfa.formalism = 'wcmf' (which uses wavelet coefficients only, not leaders)
mfa.p = 2

# scaling range
mfa.j1 = 3
mfa.j2 = 9

# range of orders q
mfa.q = np.arange(-8, 9)

# number of cumulants to be computed
mfa.n_cumul = 3

# fractional integration order
mfa.gamint = 0

# verbosity level (0: only important warnings,
#                  1: only text,
#                  2: plots)
mfa.verbose = 1

# regression type (0: ordinary least squares,
#                  1: weighted least squares, weights = nj)
mfa.wtype = False

#-------------------------------------------------------------------------------
# Analyze data and get results
#-------------------------------------------------------------------------------

# Analyze signal 1 and get cumulants
mfa.analyze(signal_1)
cumulants_1 = mfa.cumulants
print("log-cumulants of signal 1: ", cumulants_1.log_cumulants)

# Analyze signal 2 and get cumulants
mfa.analyze(signal_2)
cumulants_2 = mfa.cumulants
print("log-cumulants of signal 2: ", cumulants_2.log_cumulants)

# Weighted sum of cumulants
# The weights for C(j) is nj, the number of coefficients used to compute it
# (!!!) log-cumulants are computed AFTER the weighted sum of C(j)
cumulants_sum = deepcopy(cumulants_1)
cumulants_sum.sum(cumulants_2)
print("log-cumulants of weighted sum: ", cumulants_sum.log_cumulants)


# Compare to the log-cumulants of the whole signal
mfa.analyze(data)
print("log-cumulants of the whole signal: ", mfa.cumulants.log_cumulants)


# Plots
cumulants_1.plot(fignum = 1)
cumulants_2.plot(fignum = 2)
cumulants_sum.plot(fignum = 3)
plt.show()