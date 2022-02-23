"""
Authors: Omar D. Domingues <omar.darwiche-domingues@inria.fr>
         Merlin Dumeur <merlin@dumeur.net>
"""

from collections import namedtuple

import numpy as np

from .mfspectrum import MultifractalSpectrum
from .cumulants import Cumulants
from .structurefunction import StructureFunction
from .wavelet import wavelet_analysis
from .estimation import estimate_hmin
from .autorange import sanitize_scaling_ranges

MFractalData = namedtuple('MFractalData', 'dwt lwt')
"""Aggregates wavelet coef-based and wavelet-leader based outputs of mfa

Attributes
----------
dwt : MFractalVar
    Wavelet coef-based estimates
lwt : MFractalVar
    Wavelet leader-based estimates, if applicable (p_exp was not None)
"""

MFractalVar = namedtuple('MFractalVar',
                         'structure cumulants spectrum hmin')
"""Aggregates the output of multifractal analysis

Attributes
----------
strucuture : :class:`~pymultifracs.structurefunction.StructureFunction`
cumulants : :class:`~pymultifracs.cumulants.Cumulants`
spectrum : :class:`~pymultifracs.mfspectrum.MultifractalSpectrum`
hmin : float
    Estimated minimum value of h
"""


def mf_analysis(wt_coefs, wt_leaders, scaling_ranges, weighted,
                n_cumul, q):
    """Perform multifractal analysis, given wavelet coefficients.

    Parameters
    ----------
    wt_coefs : :class:`~pymultifracs.multiresquantity.MultiResolutionQuantity`
        Wavelet coefs
    wt_leaders : \
        :class:`~pymultifracs.multiresquantity.MultiResolutionQuantity` | None
        Wavelet leaders. Set to None if using wavelet coef formalism.
    j2_eff : int
        Effective maximum scale
    j1 : int
        Minimum scale
    weighted : bool
        Whether the linear regressions will be weighted
    n_cumul : int
        Number of cumulants computed
    q : ndarray, shape (n_exponents,)
        List of q values used in the multifractal analysis

    Returns
    -------
    :class:`~pymultifracs.mf_analysis.MFractalData`
        The output of the multifractal analysis
    """

    # In case no value of q is specified, we still include q=2 in order to be
    # able to estimate H
    if q is None:
        q = [2]

    if isinstance(q, list):
        q = np.array(q)

    scaling_ranges = sanitize_scaling_ranges(scaling_ranges, wt_coefs.j2_eff())

    if len(scaling_ranges) == 0:
        raise ValueError("No valid scaling range provided. "
                         f"Effective max scale is {wt_coefs.j2_eff()}")

    j1 = min([sr[0] for sr in scaling_ranges])
    j2 = max([sr[1] for sr in scaling_ranges])

    parameters = {
        'q': q,
        'n_cumul': n_cumul,
        'j1': j1,
        'j2': j2,
        'weighted': weighted,
        'scaling_ranges': scaling_ranges
    }

    param_dwt = {
        'mrq': wt_coefs,
        **parameters
    }

    dwt_struct = StructureFunction.from_dict(param_dwt)
    dwt_cumul = Cumulants.from_dict(param_dwt)
    dwt_spec = MultifractalSpectrum.from_dict(param_dwt)

    # pylint: disable=unbalanced-tuple-unpacking
    dwt_hmin, _ = estimate_hmin(wt_coefs, scaling_ranges, weighted)

    dwt = MFractalVar(dwt_struct, dwt_cumul, dwt_spec, dwt_hmin)

    if wt_leaders is not None:

        param_lwt = {
            'mrq': wt_leaders,
            **parameters
        }

        lwt_struct = StructureFunction.from_dict(param_lwt)
        lwt_cumul = Cumulants.from_dict(param_lwt)
        lwt_spec = MultifractalSpectrum.from_dict(param_lwt)

        # pylint: disable=unbalanced-tuple-unpacking
        lwt_hmin, _ = estimate_hmin(wt_leaders, scaling_ranges, weighted)

        lwt = MFractalVar(lwt_struct, lwt_cumul, lwt_spec, lwt_hmin)

    else:

        lwt = None

    return MFractalData(dwt, lwt)


def bootstrapped_mf_analysis(wt_coefs, wt_leaders, j1, weighted, n_cumul, q, R,
                             wt_name):

    coef_boot = wt_coefs.bootstrap(R, wt_name)
    leader_boot = wt_leaders.bootstrap(R, wt_name)

    j2_eff = min(wt_coefs.j2_eff(), len(coef_boot.values))

    dwt_boot, lwt_boot = mf_analysis(coef_boot, leader_boot, j1, j2_eff,
                                     weighted, n_cumul, q)


def minimal_mf_analysis(wt_coefs, wt_leaders, scaling_ranges, weighted,
                        n_cumul, q):
    """Perform multifractal analysis, returning only what is needed for H and
    M estimation.

    Parameters
    ----------
    wt_coefs : :class:`~pymultifracs.multiresquantity.MultiResolutionQuantity`
        Wavelet coefs
    wt_leaders : \
        :class:`~pymultifracs.multiresquantity.MultiResolutionQuantity`
        Wavelet leaders
    j2_eff : int
        Effective maximum scale
    p_exp : int | None
        Exponent for the wavelet leaders
    j1 : int
        Minimum scale
    weighted : bool
        Perform weighted regression
    n_cumul : int
        Number of cumulants computed
    q : list(float)
        List of q values used in the multifractal analysis

    Returns
    -------
    :class:`~pymultifracs.mf_analysis.MFractalData`
        The output of the multifractal analysis. The only fields that
        are filled are dwt.structure and lwt.cumulants
    """

    if q is None:
        q = [2]

    if isinstance(q, list):
        q = np.array(q)

    scaling_ranges = sanitize_scaling_ranges(scaling_ranges, wt_coefs.j2_eff())

    j1 = min([sr[0] for sr in scaling_ranges])
    j2 = max([sr[1] for sr in scaling_ranges])

    parameters = {
        'q': q,
        'n_cumul': n_cumul,
        'j1': j1,
        'j2': j2,
        'weighted': weighted,
        'scaling_ranges': scaling_ranges
    }

    dwt_struct = StructureFunction.from_dict({'mrq': wt_coefs, **parameters})
    dwt = MFractalVar(dwt_struct, None, None, None)

    lwt_cumul = Cumulants.from_dict({'mrq': wt_leaders, **parameters})
    lwt = MFractalVar(None, lwt_cumul, None, None)

    return MFractalData(dwt, lwt)


def mf_analysis_full(signal, scaling_ranges, normalization=1, gamint=0.0,
                     weighted=True, wt_name='db3', p_exp=None, q=None,
                     n_cumul=3, minimal=False):
    """Perform multifractal analysis on a signal.

    .. note:: This function combines wavelet analysis and multifractal analysis
              for practicality.
              The use of parameters is better described in their
              respective functions

    Parameters
    ----------
    signal : ndarray, shape (n_samples,)
        The signal to perform the analysis on.
    j1 : int
        Minimum scale to perform fit on.
    j2 : int
        Maximum sacle to perform fit on.
    normalization : int
        Norm to use, by default 1.
    gamint : float
        Fractional integration coefficient, by default set to 0.
        To understand how to specify gamint, see ~
    weighted : bool, optional
        Whether to perform a weighted linear regression, by default True.
    wt_name : str, optional
        Name of the wavelet to use, following pywavelet convention,
        by default Daubechies with 3 vanishing moments.
    p_exp : int | np.inf | None
        Value of the p-exponent of the wavelet leaders, by default None.
    q : list (float)
        List of values of q to perform estimates on.
    n_cumul : int, optional
        [description], by default 3
    minimal : bool, optional
        [description], by default False.

    Returns
    -------
    MFractalData
        The output of the multifractal analysis

    See also
    --------
    mf_analysis
    :obj:`~pymultifracs.wavelet.wavelet_analysis`
    """

    j1 = min([sr[0] for sr in scaling_ranges])
    j2 = max([sr[1] for sr in scaling_ranges])

    wt_transform = wavelet_analysis(signal, p_exp=p_exp, wt_name=wt_name,
                                    j1=j1, j2=j2, gamint=gamint,
                                    normalization=normalization,
                                    weighted=weighted)

    fun = minimal_mf_analysis if minimal else mf_analysis

    mf_data = fun(wt_transform.wt_coefs,
                  wt_transform.wt_leaders,
                  scaling_ranges,
                  weighted=weighted,
                  n_cumul=n_cumul,
                  q=q)

    return mf_data
