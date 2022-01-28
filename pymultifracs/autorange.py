import numpy as np

# def mf_analysis_ar(wt_coefs, wt_leaders, scaling_ranges, weighted,
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

#     dwt_struct = StructureFunction.from_dict(param_dwt)
#     dwt_cumul = Cumulants.from_dict(param_dwt)
#     dwt_spec = None  # MultifractalSpectrum.from_dict(param_dwt)

#     # pylint: disable=unbalanced-tuple-unpacking
#     # dwt_hmin, _ = estimate_hmin(wt_coefs, j1, j2, weighted)
#     dwt_hmin = None

#     dwt = MFractalVar(dwt_struct, dwt_cumul, dwt_spec, dwt_hmin)

#     if wt_leaders is not None:

#         param_lwt = {
#             'mrq': wt_leaders,
#             **parameters
#         }

#         lwt_struct = StructureFunction.from_dict(param_lwt)
#         lwt_cumul = Cumulants.from_dict(param_lwt)
#         lwt_spec = None  # MultifractalSpectrum.from_dict(param_lwt)

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


def compute_R(mrq, moment, slope, intercept):

    j_min = mrq.j.min()
    j_max = mrq.j.max()

    # Shape (n_moments, n_scales, n_scaling_ranges, n_rep)
    moment = moment[:, j_min-1:j_max, None, :]
    slope = slope[:, None, :]
    intercept = intercept[:, None, :]

    x = np.arange(j_min, j_max + 1)[None, :, None, None]
    j_mask = np.ones((1, x.shape[1], slope.shape[2], 1))

    for i, (j1, j2) in enumerate(mrq.scaling_ranges):
        j_mask[:, j2-j_min+1:, i] = 0
        j_mask[:, :j1-j_min, i] = 0

    return (j_mask * (moment - x * slope - intercept) ** 2).sum(axis=1)


def sanitize_scaling_ranges(scaling_ranges, j2_eff):

    return [(j1, j2) for (j1, j2) in scaling_ranges if j2 <= j2_eff]
