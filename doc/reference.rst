=============
API Reference
=============

Multifractal Analysis
---------------------

.. currentmodule:: pymultifracs.mfa

:py:mod:`pymultifracs.mfa`:

.. autosummary::
    :toctree: _autosummary

    mf_analysis

Dataclasses
```````````

Used to compute and store intermediary results

.. currentmodule:: pymultifracs

.. autosummary::
    :toctree: _autosummary

    multiresquantity.MultiResolutionQuantity
    cumulants.Cumulants
    structurefunction.StructureFunction
    mfspectrum.MultifractalSpectrum

Storing the multifractal analysis output

.. autosummary::
    :toctree: _autosummary

    utils.MFractalVar

Wavelet Analysis
----------------

.. currentmodule:: pymultifracs.wavelet

:py:mod:`pymultifracs.wavelet`:

.. autosummary::
    :toctree: _autosummary

    decomposition_level
    wavelet_analysis

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

    scale2freq
    freq2scale