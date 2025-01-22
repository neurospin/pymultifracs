=============
API Reference
=============

Multifractal Analysis
=====================

.. currentmodule:: pymultifracs

:py:mod:`pymultifracs`:

.. autosummary::
    :toctree: _autosummary
    :caption: Multifractal analysis

    wavelet_analysis
    mfa

Dataclasses
```````````

Used to compute and store intermediary results. Not meant to be created outside of the analysis functions.

.. autosummary::
    :toctree: _autosummary
    :template: not_instantiated.rst
    :caption: Multi Resolution Quantities

    multiresquantity.WaveletDec
    multiresquantity.WaveletLeader
    multiresquantity.WaveletWSE
    
.. autosummary::
    :toctree: _autosummary
    :caption: Scaling Functions

    scalingfunction.StructureFunction
    scalingfunction.Cumulants
    scalingfunction.MFSpectrum

Storing the multifractal analysis output

.. autosummary::
    :toctree: _autosummary
    :template: namedtuple.rst
    :caption: Analysis output

    utils.MFractalVar
    utils.MFractalBiVar

Visualization
=============

:py:mod:`pymultifracs.viz`:

.. autosummary::
    :toctree: _autosummary
    :caption: Visualization

    viz.plot_psd


Bivariate Analysis
==================

:py:mod:`pymultifracs.bivariate`:

.. autosummary::
    :toctree: _autosummary
    :caption: Bivariate analysis

    bivariate.bimfa
    bivariate.BiStructureFunction
    bivariate.BiCumulants

Simulation
==========

:py:mod:`pymultifracs.simul`:

.. autosummary::
    :toctree: _autosummary
    :caption: Simulation

    simul.mrw
    simul.fbm


Utility functions
=================

:py:mod:`pymultifracs.utils`:

.. autosummary::
    :toctree: _autosummary
    :caption: Utility functions

    utils.build_q_log

Outlier detection
=================

:py:mod:`pymultifracs.robust`:

.. autosummary::
    :toctree: _autosummary
    :caption: Robust estimation

    robust.get_outliers
