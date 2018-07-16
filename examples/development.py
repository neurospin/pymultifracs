"""
This code tests features in development.
"""

import mfanalysis as mf
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
current_dir = os.path.dirname(os.path.abspath(__file__))
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
mfa.wt_name = 'db3'
mfa.p = np.inf
mfa.j1 = 3
mfa.j2 = 12
mfa.q = np.arange(-8, 9)
mfa.n_cumul = 3
mfa.gamint = 0
mfa.verbose = 1
mfa.wtype = 0

#-------------------------------------------------------------------------------
# Test
#-------------------------------------------------------------------------------
mfa._test(data)




