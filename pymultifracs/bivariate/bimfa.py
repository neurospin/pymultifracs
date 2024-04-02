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

    # if isinstance(mrq1, Iterable):

    #     # for m1, m2 in zip(mrq1, mrq2):
    #     #     if R > 1:
    #     #         m1.bootstrapped_mrq, m2.bootstrapped_mrq = \
    #     #             m1.bootstrap_multiple(R, j1, [m1, m2])
    #     #     else:
    #     #         m1.bootstrapped_mrq, m2.bootstrapped_mrq = None, None

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
        mrq1._check_regularity(scaling_ranges, weighted, idx_reject)
        mrq2._check_regularity(scaling_ranges, weighted, idx_reject)

    if mrq1.bootstrapped_mrq is not None:

        bimfa_boot = bimfa(
            mrq1.bootstrapped_mrq, mrq2.bootsrapped_mrq, scaling_ranges,
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
        'bootstrapped_sf': bimfa_boot,
        'robust': robust,
        'idx_reject': idx_reject,
    }

    bistruct, bicumul = None, None

    flag_q = q1 is not None or q2 is not None

    if 's' in estimates or (estimates == 'auto' and flag_q):
        bistruct = BiStructureFunction.from_dict(parameters)
    if 'c' in estimates or estimates == 'auto':
        bicumul = BiCumulants.from_dict(parameters)

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
