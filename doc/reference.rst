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
    minimal_mf_analysis

Classes gathering the output of multifractal analysis

.. autosummary::
    :toctree: _autosummary
    :template: namedtuple.rst
    
    MFractalData
    MFractalVar

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
    :template: namedtuple.rst

    WaveletTransform



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
    :template: namedtuple.rst

    PSD