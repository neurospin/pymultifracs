"""
Authors: Merlin Dumeur <merlin@dumeur.net>

Automated scaling range selection based on bootstrapping.
"""

import numpy as np
from .regression import prepare_regression

# def mf_analysis_ar(wt_coefs, wt_leaders, scaling_rangexs, weighted,
#                    n_cumul, q):

#     if q is None:
#         q = [2]

#     if isinstance(q, list):
#         q = np.array(q)

#     parameters = {
#         'q': q,
#         'n_cumul': n_cumul,
#         'weighted': weighted,
#         'scaling_ranges': scaling_ranges
#         }

#     param_dwt = {
#         'mrq': wt_coefs,
#         **parameters
#     }

#     dwt_struct = StructureFunction._from_dict(param_dwt)
#     dwt_cumul = Cumulants._from_dict(param_dwt)
#     dwt_spec = None  # MultifractalSpectrum._from_dict(param_dwt)

#     # pylint: disable=unbalanced-tuple-unpacking
#     # dwt_hmin, _ = estimate_hmin(wt_coefs, j1, j2, weighted)
#     dwt_hmin = None

#     dwt = MFractalVar(dwt_struct, dwt_cumul, dwt_spec, dwt_hmin)

#     if wt_leaders is not None:

#         param_lwt = {
#             'mrq': wt_leaders,
#             **parameters
#         }

#         lwt_struct = StructureFunction._from_dict(param_lwt)
#         lwt_cumul = Cumulants._from_dict(param_lwt)
#         lwt_spec = None  # MultifractalSpectrum._from_dict(param_lwt)

#         # pylint: disable=unbalanced-tuple-unpacking
#         # lwt_hmin, _ = estimate_hmin(wt_leaders, j1, j2, weighted)
#         lwt_hmin = None

#         lwt = MFractalVar(lwt_struct, lwt_cumul, lwt_spec, lwt_hmin)

#     else:

#         lwt = None

#     return MFractalData(dwt, lwt)


def compute_Lambda(R, R_b):

    return 1 - ((R_b < R).sum(axis=-1) / R_b.shape[-1])


def find_max_lambda(L):

    return np.argwhere(L.mean(axis=0) == np.amax(L.mean(axis=0)))


def compute_R(moment, slope, intercept, weights, j_min_max, j):

    x, _, _, _, j_min_idx, j_max_idx = prepare_regression(j_min_max, j)

    # Shape (n_moments, n_scales, n_scaling_ranges, n_sig, R)
    moment = moment[:, j_min_idx:j_max_idx]
    slope = slope[:, None]
    intercept = intercept[:, None]
    weights = weights[..., None]
    x = x[..., None]

    return np.nansum(
        weights ** 2 * (moment - x * slope - intercept) ** 2, axis=1)


def sanitize_scaling_ranges(scaling_ranges, j2_eff):

    return np.array([(j1, j2) for (j1, j2) in scaling_ranges
                     if j2 <= j2_eff and j1 <= j2 - 2])
