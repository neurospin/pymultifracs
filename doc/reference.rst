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
    mf_analysis_full

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

.. autosummary::
    :toctree: _autosummary

    WaveletTransform


Simulation
----------

.. currentmodule:: pymultifracs.simul

:py:mod:`pymultifracs.simul`:

.. autosummary::
    :toctree: _autosummary

    mrw
    fbm


Computing and Plotting PSDs
---------------------------

.. currentmodule:: pymultifracs.psd

:py:mod:`pymultifracs.psd`:

.. autosummary::
    :toctree: _autosummary

    plot_psd
    wavelet_estimation
    welch_estimation

.. autosummary::
    :toctree: _autosummary

    PSD