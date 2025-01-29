"""
Authors: Omar D. Domingues <omar.darwiche-domingues@inria.fr>
         Merlin Dumeur <merlin@dumeur.net>
"""

from collections.abc import Iterable

import numpy as np

from .scalingfunction import Cumulants, StructureFunction, MFSpectrum
from .autorange import sanitize_scaling_ranges
from .utils import MFractalVar


def mfa(mrq, scaling_ranges, weighted=None, n_cumul=2, q=None,
        bootstrap_weighted=None, R=1, estimates="auto", robust=False,
        robust_kwargs=None, idx_reject=None, check_regularity=True, min_j=1):
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
        List of :math:`q` values used in the multifractal analysis.
        Defaults to None which sets ``q = [2]``.
    bootstrap_weighted : str | None
        Whether the boostrapped mrqs will have weighted regressions.
        See the description of ``weighted``.
    R : int
        Number of bootstrapped repetitions.
    estimates : str
        String containing characters which dictate which quantities to
        estimate.
        The following characters are available:
            - ``'c'``: cumulants
            - ``'m'``: multifractal spectrum
            - ``'s'``: structure function

        For example, ``"cms"`` would indicate that all should be computed,
        whereas ``"c"`` results in only the cumulants being computed.

        Defaults to ``"auto"`` which determines which quantities to estimate
        based on the ``q`` argument passed: If ``len(q) >= 2`` , then the
        spectrum is estimated, otherwise only the cumulants and structure
        functions are computed.

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

    j1 = min(sr[0] for sr in scaling_ranges)

    if (R > 1 and (
            mrq.bootstrapped_obj is None
            or mrq.bootstrapped_obj.n_rep // mrq.bootstrapped_obj.n_sig != R)):
        mrq.check_regularity(
            scaling_ranges, weighted if weighted != 'bootstrap' else None,
            idx_reject)
        mrq.bootstrap(R, j1)
    else:
        if check_regularity:
            mrq.check_regularity(scaling_ranges, None, idx_reject, min_j=min_j)

    if weighted == 'bootstrap' and mrq.bootstrapped_obj is None:
        raise ValueError(
            'weighted="bootstrap" requires R>1 or prior bootstrap')

    if mrq.bootstrapped_obj is not None:
        mfa_boot = mfa(
            mrq.bootstrapped_obj, scaling_ranges, bootstrap_weighted,
            n_cumul, q, None, 1, estimates, robust,
            robust_kwargs, idx_reject, check_regularity=False)
    else:
        mfa_boot = None

    if min_j == 'auto':
        min_j = j1

    if min_j < (mrq_jmin := min(mrq.values)):
        min_j = mrq_jmin

    if min_j > j1:
        raise ValueError(
            'Minimum j should be lower than the smallest fitting scale')

    parameters = {
        'q': q,
        'n_cumul': n_cumul,
        'weighted': weighted,
        'scaling_ranges': scaling_ranges,
        'mrq': mrq,
        'bootstrapped_obj': mfa_boot,
        'robust': robust,
        'idx_reject': idx_reject,
        'min_j': min_j,
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
