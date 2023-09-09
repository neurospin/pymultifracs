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
from .wavelet import wavelet_analysis, _estimate_eta_p
from .estimation import estimate_hmin
from .autorange import sanitize_scaling_ranges
from .utils import MFractalVar


def mf_analysis(mrq, scaling_ranges, weighted=None, n_cumul=2, q=None,
                bootstrap_weighted=None, R=1, estimates="scm", robust=False,
                robust_kwargs=None):
    """Perform multifractal analysis, given wavelet coefficients.

    Parameters
    ----------
    mrq: :class:`MultiResolutionQuantity` | List[MultiResolutionQuantity]
        Multi-resolution quantity to analyze, or list of MRQs.
    scaling_ranges: List[Tuple[int]]
        List of pairs of (j1, j2) ranges of scales for the analysis
    weighted : str | None
        Whether the linear regressions will be weighted
    n_cumul : int
        Number of cumulants computed
    q : ndarray, shape (n_exponents,)
        List of q values used in the multifractal analysis
    bootstrap_weighted: str | None
        Whether the boostrapped mrqs will have weighted regressions
    R: int
        Number of bootstrapped repetitions
    estimates: str
        Quantities to estimate: string containing a character for each of:
            - "m": multifractal spectrum
            - "s": structure function
            - "c": cumulants
    robust: bool
        Use robust estimates of cumulants

    Returns
    -------
    :class:`~pymultifracs.mf_analysis.MFractalData`
        The output of the multifractal analysis, is a list if `mrq` was passed as an Iterable
    """

    if isinstance(mrq, Iterable):

        if isinstance(estimates, str):
            estimates = [estimates] * len(mrq)

        elif (n := len(estimates)) != (m := len(mrq)):
            raise ValueError(
                f"Length of `estimates` = {n} does not match `mrq` = {m}"
            )

        return ([mf_analysis(m, scaling_ranges, weighted, n_cumul,
                             q, bootstrap_weighted, R, estimates[i], robust,
                             robust_kwargs)
                 for i, m in enumerate(mrq)])

    scaling_ranges = sanitize_scaling_ranges(scaling_ranges, mrq.j2_eff())

    if len(scaling_ranges) == 0:
        raise ValueError("No valid scaling range provided. "
                         f"Effective max scale is {mrq.j2_eff()}")

    if mrq.formalism == 'wavelet p-leader':

        eta_p = _estimate_eta_p(
            mrq.origin_mrq, mrq.p_exp, scaling_ranges, weighted)

        if eta_p.max() <= 0:
            raise ValueError(
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

        hmin, _ = estimate_hmin(mrq, scaling_ranges, weighted)

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

    j1 = min([sr[0] for sr in scaling_ranges])
    j2 = max([sr[1] for sr in scaling_ranges])

    if R > 1:
        mrq.bootstrap(R, j1)

    if mrq.bootstrapped_mrq is not None:

        mfa_boot = mf_analysis(
            mrq.bootstrapped_mrq, scaling_ranges,
            bootstrap_weighted, n_cumul, q, None, 1, estimates, robust,
            robust_kwargs)

    else:
        mfa_boot = None

    # In case no value of q is specified, we still include q=2 in order to be
    # able to estimate H
    if q is None:
        q = [2]

    if isinstance(q, list):
        q = np.array(q)

    parameters = {
        'q': q,
        'n_cumul': n_cumul,
        'j1': j1,
        'j2': j2,
        'weighted': weighted,
        'scaling_ranges': scaling_ranges,
        'mrq': mrq,
        'bootstrapped_mfa': mfa_boot,
        'robust': robust
    }

    if robust_kwargs is not None:
        parameters['robust_kwargs'] = robust_kwargs

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


def mf_analysis_full(signal, scaling_ranges, normalization=1, gamint=0.0,
                     weighted=None, wt_name='db3', p_exp=None, q=None,
                     n_cumul=3, bootstrap_weighted=None,
                     estimates='scm', R=1):
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

    mrq = wt_transform.wt_coefs

    if wt_transform.wt_leaders is not None:
        mrq = [mrq, wt_transform.wt_leaders]

    mf_data = mf_analysis(
        mrq,
        scaling_ranges,
        weighted=weighted,
        n_cumul=n_cumul,
        q=q,
        bootstrap_weighted=bootstrap_weighted,
        R=R,
        estimates=estimates,
    )

    return mf_data
