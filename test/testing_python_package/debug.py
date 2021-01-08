from __future__ import print_function
from __future__ import unicode_literals


# Import packages
import numpy as np
import pymultifracs
import matplotlib.pyplot as plt
from scipy.io import loadmat

import sys, os
from os.path import dirname, abspath


# Function to load data
def get_data_from_mat_file(filename):
    contents = loadmat(filename)
    return contents['data'][0]


def main():
    #---------------------------------------------------------------------------
    # Read MRW from mat file
    #---------------------------------------------------------------------------
    # current folder
    current_dir = dirname(abspath(__file__))
    # folder containing /test_data 
    test_dir = dirname(dirname(abspath(__file__)))

    data_filename = os.path.join(test_dir, 'test_data', 'meg_sample_snr1.mat')
    data = get_data_from_mat_file(data_filename)

    #---------------------------------------------------------------------------
    # Initialize MFA object
    #---------------------------------------------------------------------------
    q = np.arange(-8, 9, 1)
    mfa = pymultifracs.MFA(
        p = 0.5,
        q = q,
        j1 = 1,
        j2 = 9,
        gamint = 0,
        wt_name = 'db3',
        n_cumul = 3,
        verbose = 1,
        wtype = 0)

    #---------------------------------------------------------------------------
    # Analize data
    #---------------------------------------------------------------------------
    c1_list = []
    c2_list = []
    c3_list = []


    mfa.analyze(data)
    cm = mfa.cumulants.log_cumulants
    c1_list.append(cm[0])
    c2_list.append(cm[1])
    c3_list.append(cm[2])


    print("-------------------------------------------------------------")
    print("c1: mean and 0.5*std: ", np.mean(c1_list), 0.5*np.std(c1_list))
    print("c2: mean and 0.5*std: ", np.mean(c2_list), 0.5*np.std(c2_list))
    print("c3: mean and 0.5*std: ", np.mean(c3_list), 0.5*np.std(c3_list))


    mfa.plot_cumulants()
    mfa.plot_structure()

    return mfa, data


if __name__ == '__main__':
    mfa, data = main()
