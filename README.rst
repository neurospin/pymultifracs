Introduction
============

This package implements wavelet based multifractal analysis of 1D signals.

Implemented features:

* Computation of (1D) multiresolution quantities: wavelet coefficients, wavelet-leaders and p-leaders
* Computation of structure functions, cumulants and log-cumulants.
* Estimation of the multifractal spectrum.


The code in this package is based on the Wavelet p-Leader and Bootstrap based MultiFractal analysis (PLBMF) Matlab toolbox written by Herwig Wendt
(https://www.irit.fr/~Herwig.Wendt/software.html) and on the documents provided in his website (his PhD thesis in particular, which can be found at
https://www.irit.fr/~Herwig.Wendt/data/ThesisWendt.pdf).


For a brief introduction to multifractal analysis, see the file THEORY.ipynb


Installing the package only
===========================

Using conda
-----------

You first need to get the environment file for the package

.. code:: shell

    wget https://raw.githubusercontent.com/MerlinDumeur/mfanalysis/master/examples/env.yml

Creating a new environment (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code:: shell

    conda env create -f env.yml

Installing into a pre-existing environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that this package requires a recent version of python (>=3.7)

.. code:: shell

    conda env update -f env.yml

Using pip
---------

.. code:: shell
    
    pip install git+https://github.com/MerlinDumeur/mfanalysis.git@master


Cloning the whole repository (including examples and doc)
=================================================

.. code:: shell

    git clone https://github.com/MerlinDumeur/mfanalysis.git@master

To install the package from the local repo, use the following command.

.. code:: shell

    pip install -e mfanalysis

How to use the package
======================

See the examples/ folder, mainly examples/Demo.ipynb and examples/Maquette.ipynb
