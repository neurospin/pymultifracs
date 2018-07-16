Introduction
============

This package implements wavelet based multifractal analysis of 1D signals.

Implemented features:

* Computation of (1D) multiresolution quantities: wavelet coefficients, wavelet-leaders and p-leaders
* Computation of structure functions, cumulants and log-cumulants.
* Estimation of the multifractal spectrum.


The code in this package is based on the Wavelet p-Leader and Bootstrap based MultiFractal analysis (PLBMF) Matlab toolbox written by Herwig Wendt (https://www.irit.fr/~Herwig.Wendt/software.html) and on the documents provided in his website (his PhD thesis in particular, which can be found at https://www.irit.fr/~Herwig.Wendt/data/ThesisWendt.pdf).


Installation
============


.. code:: shell

    git clone https://github.com/omardrwch/mfanalysis.git
    cd mfanalysis
    python setup.py install


How to use the package
============

See scripts in /examples, mainly /examples/demo_pleaders.py 

The data used for the examples were taken from the example data of the PLBMF Matlab toolbox (https://www.irit.fr/~Herwig.Wendt/software.html, MF_BS_tool_webrelease/demo/WLBMF_example_data).


Testing
============

The scripts in /test allow us to compare the outputs of this package and the PLBMF Matlab toolbox (https://www.irit.fr/~Herwig.Wendt/software.html). 

How to perform a test:

1. Run define_testing_parameters.py to define a range of parameters to be tested and generate params.json.

2. Put .mat files in folder /test_data containing signals to be tested (each file must have a 'data' variable representing a 1d signal).

3. Run /test/testing_python_package/test.py to generate python outputs and run  /test/testing_matlab_toolbox/test.m to generate matlab outputs.

4. Done! Output csv files are stored in /test_outputs. The script compare_outputs.py can be used to compare these output files.