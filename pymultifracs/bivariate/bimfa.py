from collections.abc import Iterable

import numpy as np

from ..autorange import sanitize_scaling_ranges
from ..utils import MFractalBiVar
from ..wavelet import wavelet_analysis
from .biscaling_function import BiStructureFunction, BiCumulants


def bimfa(mrq1, mrq2, scaling_ranges, weighted=None, n_cumul=2, q1=None,
          q2=None, mode='all2all',
          bootstrap_weighted=None, R=1, estimates='auto', robust=False,
          robust_kwargs=None, idx_reject=None, check_regularity=True):
    """
    Bivariate multifractal analysis.

    Parameters
    ----------
    mrq1 : :class:`.MultiResolutionQuantityBase`
        Left-hand multi-resolution quantity to analyze.
    mrq2 : :class:`.MultiResolutionQuantityBase`
        Right-hand multi-resolution quantity to analyze.
    scaling_ranges : list[tuple[int, int]]
        List of pairs of (j1, j2) ranges of scales for the analysis.
    weighted : str | None
        Weighting mode for the linear regressions. Defaults to None, which is
        no weighting. Possible values are 'Nj' which weighs by number of
        coefficients, and 'bootstrap' which weights by bootstrap-derived
        estimates of variance.
    n_cumul : int
        Number of cumulants computed.
    q1 : ndarray of float, shape (n_exponents,)
        List of q values used in the multifractal analysis of the ``mrq1``.
    q2 : ndarray of float, shape (n_exponents,)
        List of q values used in the multifractal analysis of the ``mrq2``.
    mode : str, optional
        Mode of bivariate analysis. Either: 
            - 'all2all': each possible pair of signals between ``mrq1`` and
                ``mrq2`` is analyzed, generating ``mrq1.n_sig x mrq2.n_sig``
                pairs
            - 'pairwise': the signals in ``mrq1`` and ``mrq2`` are paired
                together based on their order of apparition, ``mrq1`` and
                ``mrq2`` need to have the same number of signals.
    bootstrap_weighted : str | None
        Whether the boostrapped mrqs will have weighted regressions.
    R : int
        Number of bootstrapped repetitions, R > 1 not currently tested!
    estimates : str
        Quantities to estimate: string containing a character for each of:
            - "s": structure function
            - "c": cumulants

        Defaults to "auto" which computes both.
    robust : bool
        Use robust estimates of cumulants.
    robust_kwargs : dict | None
        Arguments passed for robust estimation. Used for cumulant estimates
        of order >= 3.
    idx_reject : dict[int, ndarray of bool]
        Dictionary associating each scale to a boolean array indicating whether
        certain coefficients should be removed.
    check_regularity: bool
        Whether to check the minimum regularity requirements are met by the
        MRQs.

    Returns
    -------
    :class:`.MFractalBiVar`
        The output of the bivariate multifractal analysis.
    """

    # if isinstance(mrq1, Iterable):

    #     # for m1, m2 in zip(mrq1, mrq2):
    #     #     if R > 1:
    #     #         m1.bootstrapped_obj, m2.bootstrapped_obj = \
    #     #             m1.bootstrap_multiple(R, j1, [m1, m2])
    #     #     else:
    #     #         m1.bootstrapped_obj, m2.bootstrapped_obj = None, None

    #     if isinstance(estimates, str):
    #         estimates = [estimates] * len(mrq1)

    #     if (n1 := len(mrq1)) != (m := len(estimates)):
    #         raise ValueError(
    #             f"Length of `estimates` = {m} does not match `mrq1` = {n1}"
    #         )

    #     if isinstance(mrq2, Iterable):

    #         if (n1 := len(mrq1)) != (n2 := len(mrq2)):
    #             raise ValueError(
    #                 f"Length of `mrq1` = {n1} does not match `mrq2` = {n2}"
    #             )

    #         return [bimfa(m1, m2, scaling_ranges, weighted,
    #                                    n_cumul, q1, q2, bootstrap_weighted,
    #                                    R=1, estimates=estimates[i])
    #                 for i, (m1, m2) in enumerate(zip(mrq1, mrq2))]

    #     return [bimfa(m1, mrq2, scaling_ranges, weighted, n_cumul,
    #                                q1, q2, bootstrap_weighted, R=1,
    #                                estimates=estimates[i])
    #             for i, m1 in enumerate(mrq1)]

    if R > 1:
        j1 = min([sr[0] for sr in scaling_ranges])
        mrq1.bootstrap_multiple(R, j1, [mrq1, mrq2])

    bimfa_boot = None

    if q1 is None:
        q1 = [2]

    if q2 is None:
        q2 = [2]

    if isinstance(q1, list):
        q1 = np.array(q1)

    if isinstance(q2, list):
        q2 = np.array(q2)

    j2_eff = min(mrq1.j2_eff(), mrq2.j2_eff())

    scaling_ranges = sanitize_scaling_ranges(scaling_ranges, j2_eff)

    if len(scaling_ranges) == 0:
        raise ValueError("No valid scaling range provided. "
                         f"Effective max scale is {j2_eff}")

    if check_regularity:
        mrq1.check_regularity(scaling_ranges, weighted, idx_reject)
        mrq2.check_regularity(scaling_ranges, weighted, idx_reject)

    if mrq1.bootstrapped_obj is not None:

        bimfa_boot = bimfa(
            mrq1.bootstrapped_obj, mrq2.bootsrapped_mrq, scaling_ranges,
            bootstrap_weighted, n_cumul, q1, q2, None, 1, estimates)

    parameters = {
        'q1': q1,
        'q2': q2,
        'mode': mode,
        'weighted': weighted,
        'scaling_ranges': scaling_ranges,
        'mrq1': mrq1,
        'mrq2': mrq2,
        'n_cumul': n_cumul,
        'bootstrapped_obj': bimfa_boot,
        'robust': robust,
        'idx_reject': idx_reject,
    }

    bistruct, bicumul = None, None

    flag_q = q1 is not None or q2 is not None

    if 's' in estimates or (estimates == 'auto' and flag_q):
        bistruct = BiStructureFunction._from_dict(parameters)
    if 'c' in estimates or estimates == 'auto':
        bicumul = BiCumulants._from_dict(parameters)

    return MFractalBiVar(bistruct, bicumul)


# def bivariate_analysis_full(signal1, signal2, scaling_ranges, normalization=1,
#                             gamint=0.0, weighted=None, wt_name='db3',
#                             p_exp=None, q1=None, q2=None, n_cumul=3,
#                             bootstrap_weighted=None, R=1, estimates='sc'):

#     wt_param = {
#         'p_exp': p_exp,
#         'wt_name': wt_name,
#         'gamint': gamint,
#         'normalization': normalization,
#     }

#     WT1 = wavelet_analysis(signal1, **wt_param)
#     WT2 = wavelet_analysis(signal2, **wt_param)

#     mrq1 = [WT1.wt_coefs]
#     mrq2 = [WT2.wt_coefs]

#     if WT1.wt_leaders is not None:
#         mrq1 += [WT1.wt_leaders]
#     if WT2.wt_leaders is not None:
#         mrq2 += [WT2.wt_leaders]

#     return bivariate_analysis(
#         mrq1, mrq2, scaling_ranges, weighted, n_cumul, q1, q2,
#         bootstrap_weighted, R, estimates)
