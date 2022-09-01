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


def mf_analysis(mrq, scaling_ranges, weighted, n_cumul, q,
                bootstrap_weighted=None, R=1, estimates="scm"):
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

    try:
        return ([mf_analysis(m, scaling_ranges, weighted, n_cumul,
                             q, bootstrap_weighted) for m in mrq])
    except TypeError:
        pass

    # if len()

    if R > 1:
        mrq.bootstrap(R)

    if mrq.bootstrapped_mrq is not None:

        mfa_boot = mf_analysis(
            mrq.bootstrapped_mrq, scaling_ranges,
            bootstrap_weighted, n_cumul, q, None, 1)

    else:
        mfa_boot = None

    # In case no value of q is specified, we still include q=2 in order to be
    # able to estimate H
    if q is None:
        q = [2]

    if isinstance(q, list):
        q = np.array(q)

    scaling_ranges = sanitize_scaling_ranges(scaling_ranges, mrq.j2_eff())

    if len(scaling_ranges) == 0:
        raise ValueError("No valid scaling range provided. "
                         f"Effective max scale is {mrq.j2_eff()}")

    j1 = min([sr[0] for sr in scaling_ranges])
    j2 = max([sr[1] for sr in scaling_ranges])

    parameters = {
        'q': q,
        'n_cumul': n_cumul,
        'j1': j1,
        'j2': j2,
        'weighted': weighted,
        'scaling_ranges': scaling_ranges,
        'mrq': mrq,
        'bootstrapped_mfa': mfa_boot,
    }

    struct, cumul, spec = None, None, None

    if 's' in estimates:
        struct = StructureFunction.from_dict(parameters)
    if 'c' in estimates:
        cumul = Cumulants.from_dict(parameters)
    if 'm' in estimates:
        spec = MultifractalSpectrum.from_dict(parameters)

    # pylint: disable=unbalanced-tuple-unpacking
    if mrq.formalism == 'wavelet coef':
        hmin, _ = estimate_hmin(mrq, scaling_ranges, weighted)
    else:
        hmin = None

    return MFractalVar(struct, cumul, spec, hmin)


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
