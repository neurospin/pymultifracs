.. -*- mode: rst -*-

|Codecov|_ |Binder|_ |CircleCI|_

.. |Codecov| image:: https://codecov.io/gh/neurospin/pymultifracs/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/neurospin/pymultifracs

.. |Binder| image:: https://mybinder.org/badge_logo.svg
.. _Binder: https://mybinder.org/v2/gh/neurospin/pymultifracs/master

.. |CircleCI| image:: https://circleci.com/gh/neurospin/pymultifracs.svg?style=svg
.. _CircleCI: https://circleci.com/gh/neurospin/pymultifracs



Introduction
============

This package implements wavelet based multifractal analysis of 1D signals.

Implemented features:

* Computation of (1D) multiresolution quantities: wavelet coefficients, wavelet-leaders and p-leaders
* Computation of structure functions, cumulants and log-cumulants.
* Estimation of the multifractal spectrum.


The code in this package is based on the Wavelet p-Leader and Bootstrap based MultiFractal analysis (PLBMF) `Matlab toolbox <http://www.ens-lyon.fr/PHYSIQUE/Equipe3/MultiFracs/software.html>`_ written by Patrice Abry, Herwig Wendt and colleagues. For a thorough introduction to multifractal analysis, you may access H. Wendt's PhD thesis available in `his webiste <https://www.irit.fr/~Herwig.Wendt/data/ThesisWendt.pdf)>`_.


For a brief introduction to multifractal analysis, see the file THEORY.ipynb

There are two ways to install this package: either by using a package manager to install the package only, which will make
the code only usable as an import,
or by cloning the repository first, and then installing the package which will make it editable

Installing the package only
===========================

.. code:: shell

    pip install -U https://github.com/neurospin/pymultifracs/archive/master.zip


Cloning the whole repository (including examples)
=================================================


.. code:: shell

    git clone https://github.com/neurospin/pymultifracs
    pip install -e pymultifracs

To get started, look into the `example/` folder
