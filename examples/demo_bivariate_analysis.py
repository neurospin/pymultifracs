"""
This script shows how to use the MFA class to compute the structure functions,
cumulants and log-cumulants.

NOTE:
    This example uses the same 1d data as the code demo_pleaders_basic.m written
    by Roberto Leonarduzzi for the PLBMF Matlab toolbox, found at:
    https://www.irit.fr/~Herwig.Wendt/software.html
"""

#-------------------------------------------------------------------------------
# Import mfanalysis package
#-------------------------------------------------------------------------------
import mfanalysis as mf

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
mf_process = 2;

if mf_process == 1:
    # fractional Brownian motion (H=0.8, N=4096)
    data_file_1 = 'example_data/fbm08n4096.mat'
    # multifractal random walk (c_1=0.75, c_2=-0.05, N=32768)
    data_file_2 = 'example_data/mrw07005n32768.mat'
if mf_process == 2:
    # meg sample
    data_file_1 = 'example_data/meg_sample_snr1.mat'
    # multifractal random walk (c_1=0.75, c_2=-0.05, N=32768)
    data_file_2 = 'example_data/mrw07005n32768.mat'

# Complete path to file
current_dir = os.getcwd()
data_file_1 = os.path.join(current_dir, data_file_1)
data_file_2 = os.path.join(current_dir, data_file_2)

#-------------------------------------------------------------------------------
# Load data
#-------------------------------------------------------------------------------
data_1 = get_data_from_mat_file(data_file_1)
data_2 = get_data_from_mat_file(data_file_2)


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
mfa.p = np.inf

# scaling range
mfa.j1 = 8
mfa.j2 = 12

# range of orders q
mfa.q = np.arange(-2, 3)

# number of cumulants to be computed
mfa.n_cumul = 2

# fractional integration order
mfa.gamint = 1

# verbosity level (0: nothing,
#                  1: only text,
#                  2: plots)
mfa.verbose = 1

# regression type (0: ordinary least squares,
#                  1: weighted least squares, weights = nj)
mfa.wtype = 0

#-------------------------------------------------------------------------------
# Analyze data and get results
#-------------------------------------------------------------------------------
mfa.bivariate_analysis(data_1, data_2)

mfa.analyze(data_1)
c20 = mfa.cumulants.log_cumulants[1] # c20

mfa.analyze(data_2)
c02 = mfa.cumulants.log_cumulants[1] # c02


# Print results
print("c10  (c1 of data 1) = ", mfa.bi_cumulants.log_cumulants[1,0])
print("c01  (c1 of data 1) = ", mfa.bi_cumulants.log_cumulants[0,1])
print("c11  = ", mfa.bi_cumulants.log_cumulants[1,1])

c11 =  mfa.bi_cumulants.log_cumulants[1,1]
b = c20*c02 - c11**2
