"""
This script compares the files python_output_*.csv and matlab_output_*.csv in
the folder test_outputs.
"""
from os.path import dirname, abspath
import os
import pandas
import numpy as np
import matplotlib.pyplot as plt

# Variables to plot
what_to_plot = ['hmin', 'c1', 'c2', 'c3', \
                'zeta_0','zeta_16', 'eta_p', \
                'hq_0', 'hq_4', 'hq_10','hq_16', 
                'Dq_0', 'Dq_4', 'Dq_10', 'Dq_16']
n_figures = len(what_to_plot)



#-------------------------------------------------------------------------------
# Subplots params
#-------------------------------------------------------------------------------
plot_dim_1 = int(np.ceil( np.sqrt(n_figures) ))
plot_dim_2 = plot_dim_1



#-------------------------------------------------------------------------------
# Path/file settings
#-------------------------------------------------------------------------------
# current directory
current_dir = dirname(abspath(__file__))

# output folder
out_folder = os.path.join(current_dir, 'test_outputs')

# test data folder
test_data_dir = os.path.join(current_dir, 'test_data')

# list of files containing test data
test_data_files_short = [file  \
                   for file in os.listdir(test_data_dir) \
                        if os.path.splitext(file)[1]=='.mat']
# output file names
out_filenames_python = {}
out_filenames_matlab = {}
for file in test_data_files_short:
    out_filenames_python[file] = os.path.join(out_folder,
                            'python_output_'+ \
                            os.path.splitext(file)[0]+ \
                            '.csv')
    out_filenames_matlab[file] = os.path.join(out_folder,
                            'matlab_output_'+ \
                            os.path.splitext(file)[0]+ \
                            '.csv')

for file in test_data_files_short:
    # Read files
    out_python_file  = out_filenames_python[file]
    out_matlab_file  = out_filenames_matlab[file]
    try:
        print("--------------------------")
        print("Reading ", out_python_file)
        print("Reading ", out_matlab_file)
        out_python = pandas.read_csv(out_python_file)
        out_matlab = pandas.read_csv(out_matlab_file)
    except:
        continue

    # Plot
    fig, axes = plt.subplots(plot_dim_1, plot_dim_2, squeeze = False)
    axes = axes.flatten()
    fig.suptitle(file)

    for idx, label in enumerate(what_to_plot):
        ax = axes[idx]

        x = out_python[label].as_matrix().squeeze()
        y = out_matlab[label].as_matrix().squeeze()
        if label == 'eta_p':
            x = x[np.logical_not(np.isinf(x))]
            y = y[np.logical_not(np.isinf(y))]

        ax.plot(x,y,'r.') # data
        ax.plot( [min(x), max(x)], [min(x), max(x)])         # line x =y
        ax.set_ylabel(label)
        ax.grid()

    plt.show()
