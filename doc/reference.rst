=============
API Reference
=============

Multifractal Analysis
---------------------

.. currentmodule:: pymultifracs

:py:mod:`pymultifracs`:

.. autosummary::
    :toctree: _autosummary

    wavelet_analysis
    mfa

Dataclasses
```````````

Used to compute and store intermediary results

.. currentmodule:: pymultifracs

.. autosummary::
    :toctree: _autosummary

    multiresquantity.WaveletDec
    multiresquantity.WaveletLeader
    multiresquantity.Wtwse
    scalingfunction.StructureFunction
    scalingfunction.Cumulants
    scalingfunction.MFSpectrum

Storing the multifractal analysis output

.. autosummary::
    :toctree: _autosummary
    :template: namedtuple.rst

    utils.MFractalVar
    utils.MFractalBiVar

Bivariate Analysis
------------------

.. currentmodule:: pymultifracs.bivariate

:py:mod:`pymultifracs.bivariate`:

.. autosummary::
    :toctree: _autosummary

    bimfa
    BiStructureFunction
    BiCumulants

Simulation
----------

.. currentmodule:: pymultifracs.simul

:py:mod:`pymultifracs.simul`:

.. autosummary::
    :toctree: _autosummary

    mrw
    fbm


Utility functions
-----------------

.. currentmodule:: pymultifracs.utils

:py:mod:`pymultifracs.utils`:

.. autosummary::
    :toctree: _autosummary

    build_q_log
