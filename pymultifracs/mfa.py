"""
Authors: Omar D. Domingues <omar.darwiche-domingues@inria.fr>
         Merlin Dumeur <merlin@dumeur.net>
"""

import numpy as np

from .mfspectrum import MultifractalSpectrum
from .cumulants import Cumulants
from .structurefunction import StructureFunction
from .wavelet import wavelet_analysis
from .estimation import estimate_hmin
from .autorange import sanitize_scaling_ranges
from .utils import MFractalData, MFractalVar


def mf_analysis(wt_coefs, wt_leaders, scaling_ranges, weighted,
                n_cumul, q, bootstrap_weighted):
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
    weighted : str | None
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

    if wt_coefs.bootstrapped_mrq is not None:

        if wt_leaders is not None and wt_leaders.bootstrapped_mrq is not None:
            leader_boot = wt_leaders.bootstrapped_mrq
        else:
            leader_boot = None

        dwt_boot, lwt_boot = mf_analysis(
            wt_coefs.bootstrapped_mrq, leader_boot, scaling_ranges,
            bootstrap_weighted, n_cumul, q, None)

    else:
        dwt_boot, lwt_boot = None, None

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
        'scaling_ranges': scaling_ranges,
    }

    param_dwt = {
        'mrq': wt_coefs,
        'bootstrapped_cm': (dwt_boot.cumulants
                            if dwt_boot is not None else None),
        'bootstrapped_sf': (dwt_boot.structure
                            if dwt_boot is not None else None),
        'bootstrapped_mfs': (dwt_boot.spectrum
                             if dwt_boot is not None else None),
        **parameters
    }

    dwt_struct = StructureFunction.from_dict(param_dwt)
    dwt_cumul = Cumulants.from_dict(param_dwt)
    dwt_spec = MultifractalSpectrum.from_dict(param_dwt)

    # pylint: disable=unbalanced-tuple-unpacking
    dwt_hmin, _ = estimate_hmin(wt_coefs, scaling_ranges, weighted)

    dwt = MFractalVar(dwt_struct, dwt_cumul, dwt_spec, dwt_hmin, dwt_boot)

    if wt_leaders is not None:

        param_lwt = {
            'mrq': wt_leaders,
            'bootstrapped_cm': (lwt_boot.cumulants
                                if lwt_boot is not None else None),
            'bootstrapped_sf': (lwt_boot.structure
                                if lwt_boot is not None else None),
            'bootstrapped_mfs': (lwt_boot.spectrum
                                 if lwt_boot is not None else None),
            **parameters
        }

        lwt_struct = StructureFunction.from_dict(param_lwt)
        lwt_cumul = Cumulants.from_dict(param_lwt)
        lwt_spec = MultifractalSpectrum.from_dict(param_lwt)

        # pylint: disable=unbalanced-tuple-unpacking
        lwt_hmin, _ = estimate_hmin(wt_leaders, scaling_ranges, weighted)

        lwt = MFractalVar(lwt_struct, lwt_cumul, lwt_spec, lwt_hmin, lwt_boot)

    else:

        lwt = None

    return MFractalData(dwt, lwt)


def bootstrapped_mf_analysis(wt_coefs, wt_leaders, scaling_ranges, weighted,
                             weighted_boot, n_cumul, q, R, wt_name):

    scaling_ranges = sanitize_scaling_ranges(scaling_ranges, wt_coefs.j2_eff())

    coef_boot = wt_coefs.bootstrap(R, wt_name)
    leader_boot = wt_leaders.bootstrap(R, wt_name)

    dwt_boot, lwt_boot = mf_analysis(coef_boot, leader_boot, scaling_ranges,
                                     weighted_boot, n_cumul, q)

    dwt, lwt = mf_analysis(wt_coefs, wt_leaders, scaling_ranges, weighted,
                           n_cumul, q, dwt_boot, lwt_boot, coef_boot)


def minimal_mf_analysis(wt_coefs, wt_leaders, scaling_ranges, weighted,
                        n_cumul, q, bootstrap_weighted, robust):
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
    weighted : str | None
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

    if wt_leaders is not None and wt_leaders.bootstrapped_mrq is not None:
        leader_boot = wt_leaders.bootstrapped_mrq
    else:
        leader_boot = None

    if wt_coefs is not None and wt_coefs.bootstrapped_mrq is not None:
        coef_boot = wt_coefs.bootstrapped_mrq
    else:
        coef_boot = None

    if coef_boot is not None or leader_boot is not None:
        dwt_boot, lwt_boot = minimal_mf_analysis(
            coef_boot, leader_boot, scaling_ranges,
            bootstrap_weighted, n_cumul, q, None, robust)
    else:
        dwt_boot, lwt_boot = None, None


    if q is None:
        q = [2]

    if isinstance(q, list):
        q = np.array(q)


    if wt_coefs is not None:
        j2_eff = wt_coefs.j2_eff()
    else:
        j2_eff = wt_leaders.j2_eff()

    scaling_ranges = sanitize_scaling_ranges(scaling_ranges, j2_eff)

    j1 = min([sr[0] for sr in scaling_ranges])
    j2 = max([sr[1] for sr in scaling_ranges])

    parameters = {
        'q': q,
        'n_cumul': n_cumul,
        'j1': j1,
        'j2': j2,
        'weighted': weighted,
        'scaling_ranges': scaling_ranges,
        'bootstrapped_cm': lwt_boot.cumulants if lwt_boot is not None else None,
        'bootstrapped_sf': dwt_boot.structure if dwt_boot is not None else None,
        'robust': robust
    }

    if wt_coefs is not None:
        dwt_struct = StructureFunction.from_dict(
            {'mrq': wt_coefs, **parameters})
        dwt = MFractalVar(dwt_struct, None, None, None,
                          parameters['bootstrapped_sf'])
    else:
        dwt = None

    lwt_cumul = Cumulants.from_dict({'mrq': wt_leaders, **parameters})
    lwt = MFractalVar(None, lwt_cumul, None, None, parameters['bootstrapped_cm'])

    return MFractalData(dwt, lwt)


def mf_analysis_full(signal, scaling_ranges, normalization=1, gamint=0.0,
                     weighted=None, wt_name='db3', p_exp=None, q=None,
                     n_cumul=3, minimal=False, bootstrap_weighted=None):
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
    weighted : str | None
        Whether to perform a weighted linear regression, by default None.
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
                                    j1=j1, j2=j2, gamint=gamint, j2_reg=j2,
                                    normalization=normalization,
                                    weighted=weighted)

    fun = minimal_mf_analysis if minimal else mf_analysis

    mf_data = fun(wt_transform.wt_coefs,
                  wt_transform.wt_leaders,
                  scaling_ranges,
                  weighted=weighted,
                  n_cumul=n_cumul,
                  q=q,
                  bootstrap_weighted=bootstrap_weighted)

    return mf_data
