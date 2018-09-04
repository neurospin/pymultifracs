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
mf_process = 10; # 1 or 2

if mf_process in [1,2]:
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

else:
	from fbm import FBM
	f = FBM(2**15, 0.75)
	fgn_sample = f.fgn()
	fbm_sample = f.fbm()
	if mf_process == 10:
		data = fbm_sample 
	if mf_process == 11:
		data = fgn_sample


#-------------------------------------------------------------------------------
# Setup analysis
#-------------------------------------------------------------------------------

# Multifractal analysis object
mfa = mf.MFA()
mfa.wt_name = 'db3'
mfa.p = np.inf
mfa.j1 = 3
mfa.j2 = 12
mfa.q = [2] #np.arange(-8, 9)
mfa.n_cumul = 3
mfa.gamint = 0
mfa.verbose = 1
mfa.wtype = 0

#-------------------------------------------------------------------------------
# Test
#-------------------------------------------------------------------------------
# mfa._test(data)
# mfa.analyze(data)
hurst = mfa.compute_hurst(data)
Sj_2 = mfa.hurst_structure

print(Sj_2)

mfa.plot_structure(show=True)
