"""
This script generates a json file containing several parameters for multifractal analysis.
These parameters are then used to compare the results obtained with the mf_analysis python package
and with the PLBMF Matlab toolbox.

* The following parameters are varied across tests:

j1 - smallest scale analysis
j2 - largest scale analysis
p  - p-leaders exponent (can be np.inf, represented by the string 'inf')
wtype - regression type
gamint - fractional integration parameter
nb_vanishing_moments - number of vanishing moments

* The following parameters are fixed for all tests:

q - numpy array containing the exponents for which to compute the structure functions

"""

import numpy as np
import json


# Defining parameters
q = np.arange(-8, 9, 1).tolist()
j1_list      = [1, 3]
j2_list      = [9, 12]
p_list       = [0.5, 2, 8, 'inf', 'none'] # p = 'none' -> formalism = 'wcmf' (wavelet coefficients)
wtype_list   = [0, 1]
gamint_list  = [0.0, 0.5, 1.5]
nb_vanishing_moments_list = [2,3,4]

# Create dictionary of parameters
# params[i] = dic, such that dic['j1'], ..., dic['gamint'] give the parameters of the i-th test
# (starting from 1)
params = {}
test_index = 1
for p in p_list:
    for nb_vanishing_moments in nb_vanishing_moments_list:
        for j1 in j1_list:
            for j2 in j2_list:
                for wtype in wtype_list:
                    for gamint in gamint_list:
                        params[test_index] = {}

                        params[test_index]['q'] = q
                        params[test_index]['nb_vanishing_moments'] = nb_vanishing_moments
                        params[test_index]['p'] = p
                        params[test_index]['j1'] = j1
                        params[test_index]['j2'] = j2
                        params[test_index]['wtype'] = wtype
                        params[test_index]['gamint'] = gamint
                        test_index+=1

# Save file with parameters
with open('params.json', 'w') as outfile:
    json.dump(params, outfile, indent=4, sort_keys=True)
