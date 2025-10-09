.. -*- mode: rst -*-

|Python|_ |Doc|_ |Codecov|_ |Binder|_

.. |Python| image:: https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-brightgreen
.. _Python: https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-brightgreen

.. |Codecov| image:: https://codecov.io/gh/neurospin/pymultifracs/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/neurospin/pymultifracs

.. |Binder| image:: https://mybinder.org/badge_logo.svg
.. _Binder: https://mybinder.org/v2/gh/neurospin/pymultifracs/master

.. |Doc| image:: https://img.shields.io/badge/docs-online-brightgreen
.. _Doc:  https://neurospin.github.io/pymultifracs/


Introduction
============

The PyMultiFracs package implements wavelet-based multifractal analysis of 1D signals.

Implemented features:

* Computation of (1D) multiresolution quantities: wavelet coefficients, wavelet-leaders and p-leaders.
* Computation of structure functions, cumulants and log-cumulants.
* Estimation of the multifractal spectrum.
* Bivariate multifractal analysis.
* Bootstrap-derived confidence intervals and automated scaling range selection.
* Outlier detection.

The initial implementation of the code in this package was based on the Wavelet p-Leader and Bootstrap based MultiFractal analysis (PLBMF) `Matlab toolbox <http://www.ens-lyon.fr/PHYSIQUE/Equipe3/MultiFracs/software.html>`_ written by Patrice Abry, Herwig Wendt and colleagues. For a thorough introduction to multifractal analysis, you may access H. Wendt's PhD thesis available on `his website <https://www.irit.fr/~Herwig.Wendt/data/ThesisWendt.pdf>`_.

To get started, please look at our `documentation <https://www.neurospin.fr/pymultifracs/>`_.

Installation
------------

PyMultiFracs may be installed using pip:

.. code:: shell

    pip install pymultifracs

.. For a complete installation guide, please check `documentation <user_guide/installation>`.
