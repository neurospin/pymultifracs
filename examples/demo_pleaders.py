"""
This script shows how to use the MFA class to compute the structure functions,
cumulants and log-cumulants.

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

#-------------------------------------------------------------------------------
# Function to load data
#-------------------------------------------------------------------------------
def get_data_from_mat_file(filename):
    contents = loadmat(filename)
    return contents['data'][0]

#-------------------------------------------------------------------------------
# Select example data
#-------------------------------------------------------------------------------
mf_process = 2; # 1 or 2

if mf_process == 1:
    # fractional Brownian motion (H=0.8, N=4096)
    data_file = 'example_data/fbm08n4096.mat'
elif mf_process == 2:
    # multifractal random walk (c_1=0.75, c_2=-0.05, N=32768)
    data_file = 'example_data/mrw07005n32768.mat'

# Complete path to file
current_dir = os.getcwd()
data_file = os.path.join(current_dir, data_file)

#-------------------------------------------------------------------------------
# Load data
#-------------------------------------------------------------------------------
data = get_data_from_mat_file(data_file)


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
mfa.p = 2.0


# scaling range
mfa.j1 = 3
mfa.j2 = 12

# range of orders q
mfa.q = np.arange(-8, 9)

# number of cumulants to be computed
mfa.n_cumul = 3

# fractional integration order
mfa.gamint = 0

# verbosity level (0: nothing,
#                  1: only text,
#                  2: plots)
mfa.verbose = 1

# regression type (0: ordinary least squares,
#                  1: weighted least squares, weights = nj)
mfa.wtype = False

#-------------------------------------------------------------------------------
# Analyze data and get results
#-------------------------------------------------------------------------------
mfa.analyze(data)

# get cumulants
# See pymultifracs/cumulants.py for more attributes/methods
cp  = mfa.cumulants.log_cumulants
print("c1 = ", cp[0])
print("c2 = ", cp[1])
print("c3 = ", cp[2])


# wavelet coefficients and wavelet leaders
# - wt_coeffs[j] = wavelet coefficients at scale j
# - leaders[j]   = wavelet (p-) leaders at scale j
# See pymultifracs/multiresquantity.py for more attributes/methods
wt_coeffs = mfa.wavelet_coeffs.values
leaders   = mfa.wavelet_leaders.values

# structure function
# - structure_vals[ind_q, ind_j] = values of S(j, q),
#     with q = structure.q[ind_q] and j = structure.j[ind_j]
# See pymultifracs/structurefunction.py for more attributes/methods
structure      = mfa.structure
structure_vals = mfa.structure.values

# plot cumulants
mfa.plot_cumulants(show = False)

# plot structure function and scaling function
mfa.plot_structure(show = False)

# plot multifractal spectrum
mfa.plot_spectrum(show = False)

# show plots 
mfa.plt.show()