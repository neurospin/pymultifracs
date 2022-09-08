from collections.abc import Iterable

import numpy as np

from ..autorange import sanitize_scaling_ranges
from ..utils import MFractalBiVar
from ..wavelet import wavelet_analysis
from .bivariate_cumulants import BiCumulants
from .bivariate_structurefunction import BiStructureFunction


def bivariate_analysis(mrq1, mrq2, scaling_ranges, weighted, n_cumul, q1, q2,
                       bootstrap_weighted=None, R=1, estimates='sc'):

    if isinstance(mrq1, Iterable):

        if isinstance(estimates, str):
            estimates = [estimates] * len(mrq1)

        if (n1 := len(mrq1)) != (m := len(estimates)):
            raise ValueError(
                f"Length of `estimates` = {m} does not match `mrq1` = {n1}"
            )

        if isinstance(mrq2, Iterable):

            if (n1 := len(mrq1)) != (n2 := len(mrq2)):
                raise ValueError(
                    f"Length of `mrq1` = {n1} does not match `mrq2` = {n2}"
                )

            return [bivariate_analysis(m1, m2, scaling_ranges, weighted,
                                       n_cumul, q1, q2, bootstrap_weighted, R,
                                       estimates[i])
                    for i, m1, m2 in enumerate(zip(mrq1, mrq2))]

        return [bivariate_analysis(m1, mrq2, scaling_ranges, weighted, n_cumul,
                                   q1, q2, bootstrap_weighted, R, estimates[i])
                for i, m1 in enumerate(mrq1)]

    if R > 1:
        mrq1.bootstrap(R)
        mrq2.bootstrap(R)

    mfa_boot = None

    if mrq1.boostrapped_mrq is not None:

        mfa_boot = bivariate_analysis(
            mrq1.bootstrapped_mrq, mrq2.bootsrapped_mrq, scaling_ranges,
            bootstrap_weighted, n_cumul, q1, q2, None, 1, estimates)

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

    j1 = min([sr[0] for sr in scaling_ranges])
    j2 = max([sr[1] for sr in scaling_ranges])

    parameters = {
        'q1': q1,
        'q2': q2,
        'j1': j1,
        'j2': j2,
        'weighted': weighted,
        'scaling_ranges': scaling_ranges,
        'mrq1': mrq1,
        'mrq2': mrq2,
        'n_cumul': n_cumul,
        'bootstrapped_mfa': bimfa_boot
    }

    bistruct, bicumul = None, None

    if 's' in estimates:
        bistruct = BiStructureFunction.from_dict(parameters)
    if 'c' in estimates:
        bicumul = BiCumulants.from_dict(parameters)

    return MFractalBiVar(bistruct, bicumul)


def bivariate_analysis_full(signal1, signal2, scaling_ranges, normalization=1,
                            gamint=0.0, weighted=None, wt_name='db3',
                            p_exp=None, q1=None, q2=None, n_cumul=3,
                            bootstrap_weighted=None, R=1, estimates='sc'):

    j1 = min([sr[0] for sr in scaling_ranges])
    j2 = max([sr[1] for sr in scaling_ranges])

    wt_param = {
        'p_exp': p_exp,
        'wt_name': wt_name,
        'j1': j1,
        'j2': j2,
        'gamint': gamint,
        'normalization': normalization,
        'weighted': weighted
    }

    WT1 = wavelet_analysis(signal1, **wt_param)
    WT2 = wavelet_analysis(signal2, **wt_param)

    mrq1 = [WT1.wt_coefs]
    mrq2 = [WT2.wt_coefs]

    if WT1.wt_leaders is not None:
        mrq1 += [WT1.wt_leaders]
    if WT2.wt_leaders is not None:
        mrq2 += [WT2.wt_leaders]

    if R > 1:
        for m1, m2 in zip(mrq1, mrq2):

            m1.bootstrapped_mrq, m2.bootstrapped_mrq = m1.bootstrap_multiple(
                R, j1, [m1, m2]
            )

    return bivariate_analysis(
        mrq1, mrq2, scaling_ranges, weighted, n_cumul, q1, q2,
        bootstrap_weighted, estimates)
