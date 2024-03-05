"""
Authors: Omar D. Domingues <omar.darwiche-domingues@inria.fr>
         Merlin Dumeur <merlin@dumeur.net>
"""

from collections.abc import Iterable
import warnings

import numpy as np

from .mfspectrum import MultifractalSpectrum
from .cumulants import Cumulants
from .structurefunction import StructureFunction
from .wavelet import wavelet_analysis, _estimate_eta_p, integrate_wavelet,\
    compute_leaders, compute_wse
from .estimation import estimate_hmin
from .autorange import sanitize_scaling_ranges
from .utils import MFractalVar


def mf_analysis(mrq, scaling_ranges, p_exp=None, gamint=0, weighted=None,
                n_cumul=2, q=None, bootstrap_weighted=None, R=1,
                estimates="auto", robust=False, robust_kwargs=None,
                idx_reject=None, return_mrq=False):
    """Perform multifractal analysis, given wavelet coefficients.

    Parameters
    ----------
    mrq : :class:`.MultiResolutionQuantity` | List[:class:`.MultiResolutionQuantity`]
        Multi-resolution quantity to analyze, or list of MRQs. If it is a list,
        will return a list of the output of the function applied to each MRQ
        individually.
    scaling_ranges : List[Tuple[int]]
        List of pairs of (j1, j2) ranges of scales for the analysis.
    p_exp : float | np.inf | None
        p-exponent for performing (p)-leader based analysis.
    gamint : float | str
        Fractional integration factor. If 'auto' will attempt to identify the
        minimal usable value of gamint.
    weighted : str | None
        Whether the linear regressions will be weighted.
    n_cumul : int
        Number of cumulants computed.
    q : ndarray, shape (n_exponents,)
        List of q values used in the multifractal analysis.
    bootstrap_weighted : str | None
        Whether the boostrapped mrqs will have weighted regressions.
    R : int
        Number of bootstrapped repetitions.
    estimates : str
        Quantities to estimate: string containing a character for each of:
            - "m": multifractal spectrum
            - "s": structure function
            - "c": cumulants

        Defaults to "auto" which determines which quantities to estimate based
        on the value of `q`.
    robust : bool
        Use robust estimates of cumulants.
    robust_kwargs : Dict | None
        Arguments passed for robust estimation. Used for cumulant estimates
        of order >= 3.
    idx_reject : Dict[int, ndarray]
        Dictionary associating each scale to a boolean array indicating whether
        certain coefficients should be removed.
    return_mrq : bool
        If True, will return the MRQ along with the results. Otherwise only the
        results will be returned.

    Returns
    -------
    MultiResolutionQuantity: 
        Optional return, if `return_mrq=True`.
    :class:`~pymultifracs.mf_analysis.MFractalData`
        The output of the multifractal analysis, is a list if `mrq` was passed
        as an Iterable.
    """

    # In case no value of q is specified, we still include q=2 in order to be
    # able to estimate H
    if q is None:
        q = np.array([2])
    elif isinstance(q, list):
        q = np.array(q)

    scaling_ranges = sanitize_scaling_ranges(scaling_ranges, mrq.j2_eff())

    if len(scaling_ranges) == 0:
        raise ValueError("No valid scaling range provided. ")

    if isinstance(mrq, Iterable):

        if isinstance(estimates, str):
            estimates = [estimates] * len(mrq)

        elif (n := len(estimates)) != (m := len(mrq)):
            raise ValueError(
                f"Length of `estimates` = {n} does not match `mrq` = {m}"
            )

        return ([mf_analysis(
            m, scaling_ranges, p_exp, gamint, weighted, n_cumul, q,
            bootstrap_weighted, R, estimates[i], robust, robust_kwargs,
            idx_reject, return_mrq)
                 for i, m in enumerate(mrq)])

    if (mrq.formalism == 'wavelet coef' and p_exp is None
        and mrq.gamint==0 and not isinstance(gamint, str) and gamint!=0):
        
        mrq = integrate_wavelet(mrq, gamint)

        return mf_analysis(
            mrq, scaling_ranges, None, 0, weighted, n_cumul, q,
            bootstrap_weighted, R, estimates, robust, robust_kwargs,
            idx_reject, return_mrq)
    
    elif (mrq.formalism == 'wavelet coef' and p_exp is not None
            and not isinstance(gamint, str)):

        mrq = compute_leaders(mrq, gamint, p_exp)

        return mf_analysis(
            mrq, scaling_ranges, None, 0, weighted, n_cumul, q,
            bootstrap_weighted, R, estimates, robust, robust_kwargs,
            idx_reject, return_mrq)

    j1 = min([sr[0] for sr in scaling_ranges])
    j2 = max([sr[1] for sr in scaling_ranges])

    # Check minimal regularity constraint
    if p_exp is not None:

        eta_p = _estimate_eta_p(
            mrq, p_exp, scaling_ranges, weighted, idx_reject)

        if isinstance(gamint, str) and gamint == 'auto':
            # gamint = -.5 * (eta_p // .5)
            # gamint[eta_p // .5 > 0] = 0
            # gamint[(gamint + eta_p) < 0.25] += .5
            if eta_p // .5 > 0:
                gamint = 0
            else:
                gamint = -.5 * (eta_p.min() // .5)

                if gamint + eta_p < 0.25:
                    gamint += .5

            return mf_analysis(
                mrq, scaling_ranges, p_exp, gamint, weighted, n_cumul, q,
                bootstrap_weighted, R, estimates, robust, robust_kwargs,
                idx_reject, return_mrq)

        if eta_p.max() <= 0:
            # raise ValueError(
            warnings.warn(
                f"Maximum eta(p) = {eta_p.max()} <= 0, no signal can be "
                "analyzed. A smaller value of p (or larger value of gamint) "
                "should be selected.")

        if eta_p.min() <= 0:
            warnings.warn(
                f"Minimum eta(p) = {eta_p.min()} <= 0, p-Leaders correction "
                "cannot be applied. A smaller value of p (or larger value of "
                "gamint) should be selected.")

        mrq.eta_p = eta_p
        mrq.correct_pleaders(min([*mrq.values]), max([*mrq.values]))

    else:

        hmin, _ = estimate_hmin(mrq, scaling_ranges, weighted, idx_reject)

        if isinstance(gamint, str) and gamint == 'auto':
            if hmin // .5 > 0:
                gamint = 0
            else:
                gamint = -.5 * (hmin.min() // .5)

                if gamint + hmin < 0.25:
                    gamint += .5

            return mf_analysis(
                mrq, scaling_ranges, p_exp, gamint, weighted, n_cumul, q,
                bootstrap_weighted, R, estimates, robust, robust_kwargs,
                idx_reject, return_mrq)

        if hmin.max() <= 0:
            raise ValueError(
                f"Maximum hmin = {hmin.max()} <= 0, no signal can be "
                "analyzed. A larger value of gamint or different scaling range"
                " should be selected.")

        if hmin.min() <= 0:
            warnings.warn(
                f"Minimum hmin = {hmin.min()} <= 0, multifractal analysis "
                "cannot be applied. A larger value of gamint) should be "
                "selected.")

    if R > 1:
        mrq.bootstrap(R, j1)
    else:
        mfa_boot = None

    parameters = {
        'q': q,
        'n_cumul': n_cumul,
        'j1': j1,
        'j2': j2,
        'weighted': weighted,
        'scaling_ranges': scaling_ranges,
        'mrq': mrq,
        'bootstrapped_mfa': mfa_boot,
        'robust': robust,
        'idx_reject': idx_reject,
    }

    if robust_kwargs is not None:
        parameters['robust_kwargs'] = robust_kwargs

    struct, cumul, spec = None, None, None

    flag_q = q is not None

    if 's' in estimates or (estimates == 'auto' and flag_q):
        struct = StructureFunction.from_dict(parameters)
    if 'c' in estimates or estimates == 'auto':
        cumul = Cumulants.from_dict(parameters)
    if 'm' in estimates or (estimates == 'auto' and flag_q and len(q) > 1):
        spec = MultifractalSpectrum.from_dict(parameters)

    if return_mrq:
        return mrq,  MFractalVar(struct, cumul, spec)

    return MFractalVar(struct, cumul, spec)


def mf_analysis_wse(wt_coef, scaling_ranges, theta=0.5, gamint=0, **kwargs):

    if wt_coef.formalism != 'wavelet coef' or wt_coef.gamint > 0:
        raise ValueError(
            'Input `wt_coef` should be wavelet coefficients with zero gamint')

    mrq = compute_wse(wt_coef, theta, gamint)

    return mf_analysis(mrq, scaling_ranges, p_exp=None, gamint=0, **kwargs)


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
