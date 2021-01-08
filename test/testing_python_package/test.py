"""
This scripts loads the file params.json and runs the multifractal analysis with the parameters defined
in the file.

The following MFA outputs are saved in the output_python.csv file:

mfa.cumulants.log_cumulants[0] (H coefficient)
mfa.cumulants.log_cumulants[1] (M coefficient)
mfa.cumulants.log_cumulants[2] (M coefficient)
mfa.structure.zeta[i], for i = 0, ..., len_q - 1

"""

# Import packages
import os
from os.path import dirname, abspath
import numpy as np
import json
import csv
import pymultifracs
from scipy.io import loadmat


# Function to load data
def get_data_from_mat_file(filename):
    contents = loadmat(filename)
    return contents['data'][0]


#-------------------------------------------------------------------------------
# Path/file settings
#-------------------------------------------------------------------------------
# test folder
test_dir    = dirname(dirname(abspath(__file__)))

# params.json file
params_file = os.path.join(test_dir, 'params.json')

# output folder
out_folder = os.path.join(test_dir, 'test_outputs')

# test data folder
test_data_dir = os.path.join(test_dir, 'test_data')

# list of files containing test data
test_data_files_short = [file  \
                   for file in os.listdir(test_data_dir) \
                        if os.path.splitext(file)[1]=='.mat']
test_data_files_full = [os.path.join(test_data_dir, file)  \
                   for file in os.listdir(test_data_dir) \
                        if os.path.splitext(file)[1]=='.mat']

# output file names
out_filenames = {}
for idx, file in enumerate(test_data_files_full):
    out_filenames[file] = os.path.join(out_folder,
                            'python_output_'+ \
                            os.path.splitext(test_data_files_short[idx])[0]+ \
                            '.csv')

# ------------------------------------------------------------------------------
# Load parameters file
# ------------------------------------------------------------------------------
with open(params_file, 'r') as json_file:
    params = json.load(json_file)

# Length of array q defined in params
len_q = len(params['1']['q'])

# Number of tests
n_tests = len(params)

# ------------------------------------------------------------------------------
# Initialize output files
# ------------------------------------------------------------------------------
for out_filename in list(out_filenames.values()):
    print("Creating file " + out_filename)
    with open(out_filename, 'w') as csvfile:
        writer = csv.writer(csvfile, lineterminator='\n')
        row = ['test_index','hmin','eta_p','c1', 'c2', 'c3']
        for i in range(len_q):
           row.append('zeta_'+str(i))
        for i in range(len_q):
           row.append('hq_'+str(i))
        for i in range(len_q):
           row.append('Dq_'+str(i))
        writer.writerow(row)


# ------------------------------------------------------------------------------
# Run mf_analysis
# ------------------------------------------------------------------------------
for test_filename in test_data_files_full:

    print(" ")
    print("* Analyzing file ", test_filename)

    # Load data
    data = get_data_from_mat_file(test_filename)

    # Get output filename
    output_filename = out_filenames[test_filename]

    # Run mf_analysis and save results
    mfa = pymultifracs.MFA(verbose = 0)
    with open(output_filename, 'a') as csvfile:
        writer = csv.writer(csvfile, lineterminator='\n')
        for test_index in params: # test_index is a string, e.g. '123', and starts from '1'

            if int(test_index)%10==0:
                print(" -- Test " + test_index, " of ", n_tests)

            # Set parameters in MFA object
            mfa.q      = np.array(params[test_index]['q'])
            mfa.j1     = params[test_index]['j1']
            mfa.j2     = params[test_index]['j2']
            mfa.wtype  = params[test_index]['wtype']
            mfa.gamint = params[test_index]['gamint']
            mfa.wt_name  = 'db' + str(params[test_index]['nb_vanishing_moments'])
            p      = params[test_index]['p']
            if p == 'inf':
                mfa.p = np.inf
            elif p == 'none':
                mfa.p = None
                mfa.formalism = 'wcmf'
            else:
                mfa.p = p

            # Analyze data
            mfa.analyze(data)

            #
            index = test_index

            # Get and save results
            cm = mfa.cumulants.log_cumulants
            zeta = mfa.structure.zeta
 
            hq_str   = [str(h) for h in mfa.spectrum.hq]
            Dq_str   = [str(D) for D in mfa.spectrum.Dq]

            zeta_str = [str(z) for z in zeta]
            eta_p_str = str(mfa.eta_p)
            row = [index, str(mfa.hmin), \
                eta_p_str, str(cm[0]), str(cm[1]), str(cm[2])] + zeta_str + hq_str + Dq_str
            writer.writerow(row)
