Introduction
============

This package implements wavelet based multifractal analysis of 1D signals.

Implemented features:

* Computation of (1D) multiresolution quantities: wavelet coefficients, wavelet-leaders and p-leaders
* Computation of structure functions, cumulants and log-cumulants.
* Estimation of the multifractal spectrum.


The code in this package is based on the Wavelet p-Leader and Bootstrap based MultiFractal analysis (PLBMF) Matlab toolbox written by Herwig Wendt (https://www.irit.fr/~Herwig.Wendt/software.html) and on the documents provided in his website (his PhD thesis in particular, which can be found at https://www.irit.fr/~Herwig.Wendt/data/ThesisWendt.pdf).


For a brief introduction to multifractal analysis, see the file THEORY.ipynb


Installation
============

.. code:: shell

    git clone https://github.com/neurospin/mfanalysis.git
    pip install -e mfanalysis
    cd mfanalysis
    conda env create -f examples/Maquette.yml

How to use the package
============

See the examples/ folder, mainly examples/Demo.ipynb and examples/Maquette.ipynb
