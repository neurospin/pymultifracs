"""
Authors: Omar D. Domingues <omar.darwiche-domingues@inria.fr>
         Merlin Dumeur <merlin@dumeur.net>
"""

from collections.abc import Iterable
import warnings

import numpy as np

# from .mfspectrum import MultifractalSpectrum
# from .cumulants import Cumulants
# from .structurefunction import StructureFunction
from .scalingfunction import Cumulants, StructureFunction, MFSpectrum
from .wavelet import wavelet_analysis, integrate_wavelet,\
    compute_leaders, compute_wse
from .estimation import estimate_hmin, _estimate_eta_p
from .autorange import sanitize_scaling_ranges
from .utils import MFractalVar


def mfa(mrq, scaling_ranges, weighted=None, n_cumul=2, q=None,
        bootstrap_weighted=None, R=1, estimates="auto", robust=False,
        robust_kwargs=None, idx_reject=None, check_regularity=True):
    """
    Perform multifractal analysis, given wavelet coefficients.

    Parameters
    ----------
    mrq : :class:`.MultiResolutionQuantityBase` | List[:class:`.MultiResolutionQuantityBase`]
        Multi-resolution quantity to analyze, or list of MRQs. If it is a list,
        will return a list of the output of the function applied to each MRQ
        individually.
    scaling_ranges : list[tuple[int, int]]
        List of pairs of :math:`(j_1, j_2)` ranges of scales for the analysis.
    weighted : str | None
        Weighting mode for the linear regressions. Defaults to None, which is
        no weighting. Possible values are 'Nj' which weighs by number of
        coefficients, and 'bootstrap' which weights by bootstrap-derived
        estimates of variance.
    n_cumul : int
        Number of cumulants computed.
    q : ndarray of float, shape (n_exponents,) | None
        List of :math:`q` values used in the multifractal analysis. Defaults to None
        which sets ``q = [2]``.
    bootstrap_weighted : str | None
        Whether the boostrapped mrqs will have weighted regressions.
        See the description of ``weighted``.
    R : int
        Number of bootstrapped repetitions.
    estimates : str
        String containing characters which dictate which quantities to estimate.
        The following characters are available:
            - ``'c'``: cumulants
            - ``'m'``: multifractal spectrum
            - ``'s'``: structure function

        For example, ``"cms"`` would indicate that all should be computed, whereas
        ``"c"`` results in only the cumulants being computed.

        Defaults to ``"auto"`` which determines which quantities to estimate based
        on the ``q`` argument passed: If ``len(q) >= 2`` , then the spectrum is
        estimated, otherwise only the cumulants and structure functions are
        computed.

    robust : bool
        Use robust estimates of cumulants.
    robust_kwargs : dict | None
        Arguments passed for robust estimation. Used for cumulant estimates
        of order >= 3.
    idx_reject : dict[int, ndarray of bool]
        Dictionary associating each scale to a boolean array indicating whether
        certain coefficients should be removed.
    check_regularity : bool
        Whether to check the minimum regularity requirements are met by the
        MRQs.

    Returns
    -------
    :class:`~pymultifracs.utils.MFractalVar`
        The output of the multifractal analysis, is a list if `mrq` was passed
        as an Iterable.
    """

    if isinstance(mrq, Iterable):

        if isinstance(estimates, str):
            estimates = [estimates] * len(mrq)

        elif (n := len(estimates)) != (m := len(mrq)):
            raise ValueError(
                f"Length of `estimates` = {n} does not match `mrq` = {m}"
            )

        return ([mfa(
            m, scaling_ranges, weighted, n_cumul, q,
            bootstrap_weighted, R, estimates[i], robust, robust_kwargs,
            idx_reject, check_regularity=check_regularity)
                 for i, m in enumerate(mrq)])

    # In case no value of q is specified, we still include q=2 in order to be
    # able to estimate H
    if q is None:
        q = np.array([2])
    elif isinstance(q, list):
        q = np.array(q)

    scaling_ranges = sanitize_scaling_ranges(scaling_ranges, mrq.j2_eff())

    if len(scaling_ranges) == 0:
        raise ValueError("No valid scaling range provided. ")

    if (R > 1 and (
            mrq.bootstrapped_obj is None
            or mrq.bootstrapped_obj.n_rep // mrq.bootstrapped_obj.n_sig != R)):
        j1 = min([sr[0] for sr in scaling_ranges])
        mrq.check_regularity(
            scaling_ranges, weighted if weighted != 'bootstrap' else None,
            idx_reject)
        mrq.bootstrap(R, j1)
    else:
        if check_regularity:
            mrq.check_regularity(scaling_ranges, None, idx_reject)
    
    if weighted == 'bootstrap' and mrq.bootstrapped_obj is None:
        raise ValueError(
            'weighted="bootstrap" requires R>1 or prior bootstrap')

    if R > 1 or mrq.bootstrapped_obj is not None:
        mfa_boot = mfa(
            mrq.bootstrapped_obj, scaling_ranges, bootstrap_weighted,
            n_cumul, q, None, 1, estimates, robust,
            robust_kwargs, idx_reject, check_regularity=False)
    else:
        mfa_boot = None

    parameters = {
        'q': q,
        'n_cumul': n_cumul,
        'weighted': weighted,
        'scaling_ranges': scaling_ranges,
        'mrq': mrq,
        'bootstrapped_obj': mfa_boot,
        'robust': robust,
        'idx_reject': idx_reject,
    }

    if robust_kwargs is not None:
        parameters['robust_kwargs'] = robust_kwargs

    struct, cumul, spec = None, None, None

    flag_q = q is not None

    if 's' in estimates or (estimates == 'auto' and flag_q):
        struct = StructureFunction._from_dict(parameters)
    if 'c' in estimates or estimates == 'auto':
        cumul = Cumulants._from_dict(parameters)
    if 'm' in estimates or (estimates == 'auto' and flag_q and len(q) > 1):
        spec = MFSpectrum._from_dict(parameters)
        
    return MFractalVar(struct, cumul, spec)


# def mfa_wse(wt_coef, scaling_ranges, theta=0.5, gamint=0, **kwargs):

#     if wt_coef.formalism != 'wavelet coef' or wt_coef.gamint > 0:
#         raise ValueError(
#             'Input `wt_coef` should be wavelet coefficients with zero gamint')

#     mrq = compute_wse(wt_coef, theta, gamint)

#     return mfa(mrq, scaling_ranges, p_exp=None, gamint=0, **kwargs)


# def mf_analysis_full(signal, scaling_ranges, normalization=1, gamint=0.0,
#                      weighted=None, wt_name='db3', p_exp=None, q=None,
#                      n_cumul=3, bootstrap_weighted=None,
#                      estimates='scm', R=1):
#     """Perform multifractal analysis on a signal.

#     .. note:: This function combines wavelet analysis and multifractal analysis
#               for practicality.
#               The use of parameters is better described in their
#               respective functions

#     Parameters
#     ----------
#     signal : ndarray, shape (n_samples,)
#         The signal to perform the analysis on.
#     j1 : int
#         Minimum scale to perform fit on.
#     j2 : int
#         Maximum sacle to perform fit on.
#     normalization : int
#         Norm to use, by default 1.
#     gamint : float
#         Fractional integration coefficient, by default set to 0.
#         To understand how to specify gamint, see ~
#     weighted : str | None
#         Whether to perform a weighted linear regression, by default None.
#     wt_name : str, optional
#         Name of the wavelet to use, following pywavelet convention,
#         by default Daubechies with 3 vanishing moments.
#     p_exp : int | np.inf | None
#         Value of the p-exponent of the wavelet leaders, by default None.
#     q : list (float)
#         List of values of q to perform estimates on.
#     n_cumul : int, optional
#         [description], by default 3
#     minimal : bool, optional
#         [description], by default False.

#     Returns
#     -------
#     MFractalData
#         The output of the multifractal analysis

#     See also
#     --------
#     mf_analysis
#     :obj:`~pymultifracs.wavelet.wavelet_analysis`
#     """

#     j2 = max([sr[1] for sr in scaling_ranges])

#     wt_transform = wavelet_analysis(signal, p_exp=p_exp, wt_name=wt_name,
#                                     j2=j2, gamint=gamint,
#                                     normalization=normalization)

#     mrq = wt_transform.wt_coefs

#     if wt_transform.wt_leaders is not None:
#         mrq = [mrq, wt_transform.wt_leaders]

#     mf_data = mf_analysis(
#         mrq,
#         scaling_ranges,
#         weighted=weighted,
#         n_cumul=n_cumul,
#         q=q,
#         bootstrap_weighted=bootstrap_weighted,
#         R=R,
#         estimates=estimates,
#     )

#     return mf_data
