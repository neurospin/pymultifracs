import warnings
from math import ceil

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# from scipy.stats import gennorm
# from scipy import ndimage
from scipy import stats, special
import scipy.spatial.distance as distance
from scipy.optimize import bisect
# from scipy.special import gamma, erf, gammaincc

# from tqdm.auto import tqdm
# import hdbscan
# import umap

from ..utils import fast_power


def _qn(a, c):
    """
    Computes the Qn robust estimator of scale, a more efficient alternative
    to the MAD. The implementation follows the algorithm described in Croux
    and Rousseeuw (1992).
    Parameters
    ----------
    a : array_like
        Input array.
    c : float, optional
        The normalization constant, used to get consistent estimates of the
        standard deviation at the normal distribution.  Defined as
        1/(np.sqrt(2) * scipy.stats.norm.ppf(5/8)), which is 2.219144.
    Returns
    -------
    The Qn robust estimator of scale
    """

    n = a.shape[0]
    h = n/2 + 1
    k = int(h * (h - 1) / 2)
    n_left = int(n * (n + 1) / 2)
    n_right = n * n
    k_new = k + n_left
    i, j, jh, l = [0] * 4
    sump, sumq = 0, 0
    trial, output = 0, 0
    a_sorted = np.sort(a)
    left = np.array([n - i + 1 for i in range(0, n)], dtype=np.intc)
    right = np.array([n if i <= h else n - (i - h) for i in range(0, n)], dtype=np.intc)
    weights = np.zeros((n,), dtype=np.intc)
    work = np.zeros((n,), dtype=np.double)
    p = np.zeros((n,), dtype=np.intc)
    q = np.zeros((n,), dtype=np.intc)

    while n_right - n_left > n:
        j = 0
        for i in range(1, n):
            if left[i] <= right[i]:
                weights[j] = right[i] - left[i] + 1
                jh = left[i] + weights[j] // 2
                work[j] = a_sorted[i] - a_sorted[n - jh]
                j = j + 1
        trial = _high_weighted_median(work[:j], weights[:j])
        j = 0
        for i in range(n - 1, -1, -1):
            while j < n and (a_sorted[i] - a_sorted[n - j - 1]) < trial:
                j = j + 1
            p[i] = j
        j = n + 1
        for i in range(n):
            while (a_sorted[i] - a_sorted[n - j + 1]) > trial:
                j = j - 1
            q[i] = j
        sump = np.sum(p)
        sumq = np.sum(q) - n
        if k_new <= sump:
            right = np.copy(p)
            n_right = sump
        elif k_new > sumq:
            left = np.copy(q)
            n_left = sumq
        else:
            output = c * trial
            return output
    j = 0
    for i in range(1, n):
        for l in range(left[i], right[i] + 1):
            work[j] = a_sorted[i] - a_sorted[n - l]
            j = j + 1
    k_new = k_new - (n_left + 1)
    output = c * np.sort(work[:j])[k_new]
    return output


def compute_robust_cumulants(X, m_array, alpha=1):

    from statsmodels.robust.scale import qn_scale
    from statsmodels.robust.norms import estimate_location, TukeyBiweight
    from statsmodels.tools.validation import array_like, float_like

    # shape X (n_j, n_ranges, n_rep)

    n_j, n_range, n_rep = X.shape
    moments = np.zeros((len(m_array), n_range, n_rep))
    values = np.zeros_like(moments)

    idx_unreliable = (~np.isnan(X)).sum(axis=0) < 3

    # compute robust moments
    for range, rep in np.ndindex(n_range, n_rep):

        if idx_unreliable[range, rep]:
            values[:, range, rep] = np.nan
            continue

        X_norm = X[~np.isinf(X[:, range, rep]) & ~np.isnan(X[:, range, rep]), range, rep]

        if X_norm.shape[0] > 10000:
            values[:, range, rep] = np.nan
            continue

        q_est = qn_scale(X_norm)

        if np.isclose(q_est, 0):
            values[m_array == 1, range, rep] = np.median(X_norm, axis=0)
            continue

        try:
            m_est = estimate_location(X_norm, q_est, norm=TukeyBiweight(),
                                      maxiter=1000)
        except ValueError:

            if X_norm.shape[0] < 20:
                values[:, range, rep] = np.nan
                continue

            m_est = np.median(X_norm)

        X_norm -= m_est
        X_norm /= q_est

        # X_norm -= X_norm.mean()
        # X_norm /= X_norm.std()

        # print(X_norm.mean(), X_norm.std())

        for ind_m, m in enumerate(m_array):

            decaying_factor = (alpha
                               * np.exp(-.5 * (alpha ** 2 - 1) * X_norm ** 2))

            moments[ind_m, range, rep] = np.mean(
                fast_power(alpha * X_norm, m) * decaying_factor, axis=0)

            if m == 1:
                values[ind_m, range, rep] = m_est
            elif m == 2:
                values[ind_m, range, rep] = q_est ** 2
            else:
                aux = 0

                for ind_n, n in enumerate(np.arange(1, m)):

                    if m_array[ind_m - ind_n - 1] > 2:
                        temp_moment = moments[ind_m - ind_n - 1, range, rep]
                    elif m_array[ind_m - ind_n - 1] == 2:
                        temp_moment = X_norm.var()
                    elif m_array[ind_m - ind_n - 1] == 1:
                        temp_moment = X_norm.mean()

                    if m_array[ind_n] > 2:
                        temp_value = values[ind_n, range, rep]
                    elif m_array[ind_n] == 2:
                        temp_value = X_norm.var()
                    elif m_array[ind_n] == 1:
                        temp_value = X_norm.mean()

                    aux += (special.binom(m-1, n-1) * temp_value * temp_moment)

                values[ind_m, :, rep] = moments[ind_m, range, rep] - aux

    return values


def C4_to_m4(C4, C2):
    return C4 + 3 * C2 ** 2


def C2_to_m2(C2):
    return C2


def get_location_scale(cm, fix_c2_slope=False):

    slope_c1 = cm.slope[0][None, :]
    intercept_c1 = cm.intercept[0][None, :]

    slope_c2 = cm.slope[1][None, :]
    intercept_c2 = cm.intercept[1][None, :]

    if fix_c2_slope and slope_c2 > 0:
        slope_c2[:] = 0
        for k, range in enumerate(cm.scaling_ranges):
            j_min = cm.j.min()
            intercept_c2[:, k] = cm.C2[np.s_[range[0]-j_min:range[1]-j_min]].mean()

    j_array = np.arange(1, cm.j.max() + 1)

    C1_array = slope_c1 * j_array[:, None, None] + intercept_c1
    C2_array = slope_c2 * j_array[:, None, None] + intercept_c2

    # Shape N_scales, N_scaling_ranges, N_signals

    return j_array, C1_array, C2_array


def get_location_scale_shape(cm, fix_c2_slope=False):

    slope_c1 = cm.slope[0][None, :]
    intercept_c1 = cm.intercept[0][None, :]

    slope_c2 = cm.slope[1][None, :]
    intercept_c2 = cm.intercept[1][None, :]

    if fix_c2_slope and slope_c2 > 0:
        slope_c2[:] = 0
        for k, range in enumerate(cm.scaling_ranges):
            j_min = cm.j.min()
            intercept_c2[:, k] = cm.C2[np.s_[range[0]-j_min:range[1]-j_min]].mean()

    slope_c4 = cm.slope[3][None, :]
    intercept_c4 = cm.intercept[3][None, :]

    j_array = np.arange(1, cm.j.max() + 1)

    C1_array = slope_c1 * j_array[:, None, None] + intercept_c1
    C2_array = slope_c2 * j_array[:, None, None] + intercept_c2
    C4_array = slope_c4 * j_array[:, None, None] + intercept_c4

    m2 = C2_to_m2(C2_array)
    m4_array = C4_to_m4(C4_array, m2)
    # print(C2_array, m4_array)

    # m2[m2 < 0] = 0
    # m4[m2 < 0] = 0
    # m4[m4 < 0] = 0

    alpha = np.zeros_like(C2_array)
    beta = np.zeros_like(C4_array)

    for i, (C2, C4) in enumerate(zip(C2_array, C4_array)):

        # print(C2, m4)

        for k, l in np.ndindex(beta[i].shape):

            if C2[k, l] <= 0:
                beta[i, k, l] = 1
                continue

            # f_beta = lambda beta: gamma(5/beta) * gamma(1/beta) / gamma(3/beta)**2 - 3 - m4[k, l]
            f_beta = lambda beta: (
                special.gamma(5/beta)
                * special.gamma(1/beta)
                / special.gamma(3/beta)**2
                - 3 - C4[k, l])
            # f_beta = lambda beta: special.gamma(5/beta) * special.gamma(1/beta) / special.gamma(3/beta)**2 - 3 - C4[k, l] / C2[k, l] ** 2

            if f_beta(.1) > 0 and f_beta(10) > 0:

                warnings.warn("Very high value of beta estimated")
                beta[i, k, l] = 10

            elif f_beta(.1) < 0 and f_beta(10) < 0:

                warnings.warn("Very low value of beta estimated")
                beta[i, k, l] = .1

            else:
                beta[i, k, l] = bisect(f_beta, .1, 10)

        alpha[i] = np.sqrt(C2 * special.gamma(1/beta[i]) / special.gamma(3/beta[i]))

    idx_zero = (alpha < 0) | (np.isnan(alpha))
    alpha[idx_zero] = 0

    idx_zero = beta < .1
    beta[idx_zero] = .1

    return j_array, C1_array, alpha, beta


# def sample_p_leaders(p_exp, *gennorm_args):

#     # normal args:
#     # 0 - beta / shape
#     # 1 - location
#     # 2 - alpha / scale
#     # 3 - array size

#     # if gennorm_args[2] == 0:
#     #     return (np.e ** gennorm_args[1] * np.ones(gennorm_args[3])) ** p_exp

#     sim = (np.e ** gennorm.rvs(*gennorm_args)) ** p_exp

#     idx_zero = gennorm_args[2] == 0

#     # print(gennorm_args[1][idx_zero][None, :].shape, sim[:, idx_zero].shape,
#     #       np.ones_like(sim[:, idx_zero]).shape)

#     sim[:, idx_zero] = (np.e ** gennorm_args[1][idx_zero][None, :]
#                         * np.ones_like(sim[:, idx_zero])) ** p_exp

#     return sim


# def sample_reject(k, l, j2, min_scale, p_exp, shape, location, scale, n_samples,
#                   wt_coefs, alpha, previous_reject, max_reject_share, verbose):

#      # if scale[idx, k, l] == 0:
#         #     continue  # Already initialized at zero

#     samples_scale_j = None

#     # N leaders = n_coefs - 2
#     idx_reject = {}

#     for j in range(j2, min_scale-1, -1):

#         idx = j-1
#         idx_below = j-2

#         # if samples_scale_j is None:
#         samples_scale_j = sample_p_leaders(p_exp, shape[idx],
#                                             location[idx], scale[idx],
#                                             (n_samples, *shape[idx].shape))

#         if j == 1:
#             # diff_element = np.zeros((1, *shape[idx].shape))
#             diff_element = 0
#         else:
#             samples_scales_below = [
#                 sample_p_leaders(p_exp, shape[idx_below], location[idx_below], scale[idx_below], (n_samples)),
#                 sample_p_leaders(p_exp, shape[idx_below], location[idx_below], scale[idx_below], (n_samples))
#             ]
#             diff_element = .5 * (samples_scales_below[0]
#                                  + samples_scales_below[1])

#         # N leaders = n_coefs - 2
#         idx_reject[j] = np.zeros((*shape[idx].shape, wt_coefs.values[j].shape[0] - 2), dtype=bool)

#         diff_samples = samples_scale_j - diff_element

#         # print(samples_scale_j.mean(axis=0), samples_scale_j.std(axis=0))

#         # temp_diff = diff_sample

#         if j > 1:
#             diff_samples = diff_samples[diff_samples >= 0]

#         # print(f"{j}, {temp_diff.shape=}")

#         # try:
#         # print(alpha / 2, 100 - alpha / 2)
#         ci = [np.percentile(diff_samples, (alpha / 2 ** (1))),
#                 np.percentile(diff_samples, 100 - (alpha / 2 ** (1)))]

#         vals = np.abs(wt_coefs.values[j][:, l]) ** p_exp

#         v = np.sum(np.stack([vals[:-2], vals[1:-1], vals[2:]], axis=1),
#                     axis=1)

#         check = ((v < ci[0]) | (v > ci[1])) & ~(np.isnan(v))

#         if previous_reject is not None:
#             prev = previous_reject[j][k, l]
#         else:
#             prev = np.zeros_like(check, dtype=bool)

#         prev_kept = check & prev
#         nan = np.isnan(v)
#         N_available = check.shape[0] - (prev_kept|nan).sum()

#         N_new_remove = (check&~prev&~nan).sum()
#         Max_new_reject = int(np.ceil(max_reject_share * N_available))

#         # print(N_new_remove, Max_new_reject)

#         if N_new_remove > Max_new_reject + 1:

#             # pseudo_quantiles = np.argsort(temp_diff) / temp_diff.shape[0]
#             combined_quantiles = np.argsort(np.argsort(np.r_[diff_samples, v])) / (diff_samples.shape[0] + v.shape[0])

#             combined_quantiles = 2 * abs(combined_quantiles - .5)

#             # Same shape as v, associates to every element its position in the sorted set of quantiles of {temp_diff U v}
#             # Small values are associated to more extreme quantiles
#             idx_v = np.arange(diff_samples.shape[0], combined_quantiles.shape[0])

#             # Set quantiles from carried over rejected values to zero so they
#             # Don't appear at the end of the sorted array
#             combined_quantiles[idx_v][prev_kept] = 0

#             order_v = np.argsort(combined_quantiles[idx_v])

#             # Renormalize order_v values from [0, N_v + N_tempdiff[ to [0, N_v]
#             # order_v = np.argsort(np.argsort(order_v))

#             # Nans end up at the end of the argsorted array so need to remove them
#             N_nans = np.isnan(v).sum()

#             # Remove only first [(1-alpha) * N] highest quantile observations
#             check[:] = False
#             check[order_v[-Max_new_reject-N_nans-1:-N_nans]] = True

#             # Add previously rejected values still kept in
#             check |= prev_kept

#         idx_reject[j] = check

#         if j == 12 and verbose:

#             print(scale[idx], shape[idx])

#             # print(v[check], ci[0], ci[1])

#             plt.figure()
#             plt.scatter(diff_element, samples_scale_j)
#             plt.show()

#             # vals = WT.wt_leaders.values[j-1] ** p
#             # lower_vals =  1/2 * np.sum(np.c_[
#             #     vals[:-3:2],
#             #     vals[3::2]
#             # ], axis=1)

#             # plt.scatter(lower_vals, WT.wt_leaders.values[j].squeeze() ** p)

#         if verbose and j == 12:
#             plt.figure()

#             sns.histplot({0: diff_samples[~np.isinf(diff_samples)], 1: v[~np.isnan(v) & (v > 0)]}, stat='percent',
#                             log_scale=True, common_norm=False)
#             ylim = plt.ylim()
#             plt.vlines(ci, *ylim, color='k')
#             plt.ylim(*ylim)
#             plt.show()

#             plt.figure()
#             plt.hist(v[v < .025])
#             plt.show()

#     return {(k, l): idx_reject}


# def reject_coefs(wt_coefs, cm, p_exp, n_samples, alpha, converged, error,
#                  fix_c2_slope=False, previous_reject=None, verbose=False,
#                  max_reject_share=None, min_scale=0, n_jobs=1, ):

#     if max_reject_share is None:
#         max_reject_share = 1 - alpha

#     j_array, location, scale, shape = get_location_scale_shape(cm)

#     idx_reject = {}
#     samples_scale_j = None

#     j2 = j_array.max()

#     for k, l in np.ndindex(location.shape[1:]):
#         if not (converged[k, l] or error[k, l]):

#             try:

#                 idx_reject |= sample_reject(
#                     k, l, j2, min_scale, p_exp, shape[:, k, l], location[:, k, l], scale[:, k, l], n_samples,
#                     wt_coefs, alpha, previous_reject, max_reject_share, verbose)
#             except Exception:
#                 idx_reject[(k, l)] = None
#                 error[k, l] = True

#         else:
#             idx_reject[(k, l)] = None

#     out = {}

#     for scale in range(min_scale, j2+1):

#         out[scale] = np.zeros((*location.shape[1:], wt_coefs.values[scale].shape[0] - 2), dtype=bool)
#         for k, l in np.ndindex(location.shape[1:]):
#             if not (converged[k, l] or error[k, l]):
#                 out[scale][k, l] = idx_reject[(k, l)][scale]

#     return out, error


# def iterate_analysis(WT, n_iter=10):

#     lwt = mf_analysis([WT.wt_leaders], scaling_ranges=[(6, WT.wt_leaders.j2_eff() - 2)], n_cumul=4)[0]

#     c_trace = np.zeros((4, n_iter+1))

#     c_trace[:, 0] = lwt.cumulants.log_cumulants.squeeze()

#     for i in range(n_iter):

#         idx_reject = reject_coefs(WT.wt_coefs, lwt.cumulants, 2, 1000000,
#                                   verbose=False)
#         new_leaders, _ = compute_leaders2(WT.wt_coefs, gamint=1, p_exp=2, j1=6,
#                                           j2_reg=12, idx_reject=idx_reject)
#         lwt = mf_analysis([new_leaders],
#                           scaling_ranges=[(6, WT.wt_leaders.j2_eff() - 2)],
#                           n_cumul=4)[0]

#         c_trace[:, i+1] = lwt.cumulants.log_cumulants.squeeze()

#     return c_trace, new_leaders, idx_reject, lwt


# def compute_aggregate(cdf, j1, j2, leader=False):

#     out = {}

#     for j in range(j1, j2+1):

#         out[j] = cdf[j]

#         for k in range(j1, j):

#             # print(j, k, np.nanmax(out[k]))
#             # out[k] = np.r_['-1, 4, 0', out[k][1::2], out[k][:-1:2]].max(axis=-1)

#             if leader:
#                 # out[k] = (np.r_['-1, 4, 0', out[k][1::2], out[k][:-1:2]] ** 2 / 2).sum(axis=-1)
#                 out[k] = np.sqrt((np.r_['-1, 4, 0', out[k][1::2], out[k][:-1:2]]).prod(axis=-1))
#             else:
#                 out[k] = (np.r_['-1, 4, 0', out[k][1::2], out[k][:-1:2]]).max(axis=-1)

#             # print(k, np.nanmax(out[k]))

#         # if j > j1:
#         #     print(out[j-1][-10:,0, 0])

#     for j in range(j1, j2+1):
#         if not leader:
#             out[j] **= 2 ** ((j2-j))

#     out = np.array([*out.values()]).swapaxes(0, 1)
#     idx_nan = np.isnan(out).any(axis=(1, 2, 3))

#     return out


# def compute_all_aggregate(CDF, j1, j2, leader=False):

#     Agg = {
#         j: compute_aggregate(CDF, j1, j, leader) for j in range(j1, j2+1)
#     }

#     N = int((j2 - (j1) + 1) * (2 + j2 - (j1)) / 2)
#     agg = np.zeros((Agg[j2].shape[0] * 2 ** (j2 - j1), N, *CDF[j1].shape[1:]))

#     # agg shape N_coef, N_aggregates, N_ranges, N_signals
#     # N_coef is determined by the upsampled number of coefficients 

#     end = 0

#     for j in range(min(Agg), j2+1):

#         start = end
#         end = int((j - j1 + 1) * (2 + j - j1) / 2)

#         # cutting extraneous coefficients
#         agg[:, start:end] = np.repeat(Agg[j], 2 ** (j - j1), axis=0)[:agg.shape[0]]

#     return agg


# def inv_log_gamma_cdf(x, window_size):
#     out = gammaincc(window_size, -window_size * np.log(x))

#     if np.isclose(out, 1).any():
#         print('a')

#     return out


# def filter_gmean(input_line, output_line, window_size):

#     if window_size == 1:
#         output_line[:] = input_line

#     else:
#         output_line[:] = np.nan
#         output_line[:] = stats.mstats.gmean(
#             np.lib.stride_tricks.sliding_window_view(input_line, window_size), axis=1)
#         output_line[:] = inv_log_gamma_cdf(output_line, window_size)        


# def compute_all_aggregate2(CDF, j1, j2):

#     N = int((j2 - (j1) + 1) * (2 + j2 - (j1)) / 2)
#     agg = np.zeros((CDF[j2].shape[0] * 2 ** (j2 - j1), N, *CDF[j1].shape[1:]))

#     acc = 0
#     for j in range(j1, j2+1):
        
#         for i in range(j2-j + 1):

#             agg[:, acc] = np.repeat(ndimage.generic_filter1d(
#                 CDF[j], filter_gmean, 2**i, 0, mode='constant', cval=np.nan,
#                 extra_arguments=(2 ** i,), origin=0), 2 ** (j - j1), axis=0)[:agg.shape[0]]

#             acc += 1

#     # agg shape N_coef, N_aggregates, N_ranges, N_signals
#     # N_coef is determined by the upsampled number of coefficients 

#     return agg


# def compute_all_aggregate3(CDF, j1, j2, leader=False):

#     Agg = {
#         j: compute_aggregate(CDF, j, j, leader) for j in range(j1, j2+1)
#     }

#     # N = int((j2 - (j1) + 1) * (2 + j2 - (j1)) / 2)
#     agg = np.zeros((Agg[j2].shape[0] * 2 ** (j2 - j1), j2-j1+1, *CDF[j1].shape[1:]))

#     # agg shape N_coef, N_aggregates, N_ranges, N_signals
#     # N_coef is determined by the upsampled number of coefficients 

#     end = 0

#     for i, j in enumerate(range(min(Agg), j2+1)):

#         # start = end
#         # end = int((j - j1 + 1) * (2 + j - j1) / 2)

#         # cutting extraneous coefficients
#         agg[:, i] = np.repeat(Agg[j][:, 0], 2 ** (j - j1), axis=0)[:agg.shape[0]]

#     return agg


def plot_cdf(cdf, j1, j2, ax=None, vmin=None, vmax=None,
             leader_idx_correction=True, pval=False, cbar=True, figsize=(2.5, 1),
             gamma=.3, nan_idx=None, signal_idx=0, range_idx=0):

    min_all = min([np.nanmin(np.abs(cdf[s])) for s in range(j1, j2+1) if s in cdf])

    if vmax is None:
        vmax = max([np.nanmax(cdf[s]) for s in range(j1, j2+1) if s in cdf])
    if vmin is None:
        vmin = min_all

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, layout='constrained')

    # norm = PowerNorm(vmin=vmin, vmax=vmax, gamma=gamma)

    cmap = mpl.cm.get_cmap('inferno').copy()
    cmap.set_bad('grey')

    for i, scale in enumerate(range(j1, j2 + 1)):

        if scale not in cdf:
            continue

        temp = cdf[scale][:, signal_idx, range_idx]
        # temp = np.exp(abs(temp - .5) ** 2)

        X = ((np.arange(temp.shape[0] + 1))
             * (2 ** (scale - j1 + 1)))
        X = np.tile(X[:, None], (1, 2))

        C = np.copy(temp[:, None])

        if pval:
            C = -np.log(C)

        Y = np.ones(X.shape[0]) * scale
        Y = np.stack([Y - .5, Y + .5]).transpose()

        qm = ax.pcolormesh(X, Y, C, cmap=cmap, rasterized=True)

        if nan_idx is not None:
            idx = np.unique(np.r_[nan_idx[scale], nan_idx[scale] + 1])

            segments = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)

            for seg in segments:

                if len(seg) == 0:
                    continue

                ax.pcolormesh(
                    X[seg[[0, -1]]], Y[seg[[0, -1]]], C[[0]], alpha=1,
                    edgecolor='xkcd:blue')

    ax.set(ylim=(j1-.5, j2+.5), yticks=range(j1, j2+1),
           xlabel='shift $k$', ylabel='scale $j$', facecolor='grey',
           xlim=(0, cdf[j1].shape[0]*2))

    ax.tick_params(which='minor', left=True, right=False, top=False, color='w')
    ax.yaxis.set_minor_locator(mpl.ticker.IndexLocator(base=1, offset=.5))
    ax.tick_params(which='major', right=False, top=False, color='w')

    # ax.set_yticks(range(j1, j2+1))
    # ax.set_xlabel('shift')
    # ax.set_ylabel('scale')
    # ax.set_facecolor('grey')

    if cbar:
        # formatter = mpl.ticker.LogFormatterSciNotation(labelOnlyBase=False, minor_thresholds=(np.inf, np.inf))
        # formatter = mpl.ticker.LogFormatterSciNotation(labelOnlyBase=False, minor_thresholds=(np.inf, np.inf))
        locator = mpl.ticker.MaxNLocator(4, symmetric=False)
        cb = plt.colorbar(qm, ax=ax, ticks=locator, fraction=.1, aspect=8)
        # plt.colorbar(qm, ax=ax    es[0], ticks=locator, aspect=1)
        cb.ax.tick_params(which='major', size=3)
        cb.ax.tick_params(which='minor', right=False)


# def cluster_reject_leaders(j1, j2, p_exp, cm, coefs, leaders, verbose,
#                            generalized=False):

#     ZPJCorr = leaders.correct_pleaders(cm.j.min(), cm.j.max())
#     ZPJCorr = np.log(ZPJCorr).transpose(2, 0, 1)

#     if generalized:

#         j_array, C1_array, scale, shape = get_location_scale_shape(cm)
        
#         CDF = {
#             j: gen_cdf(
#             np.log(leaders.values[j][:, None]), 
#             C1_array[j_array == j] - ZPJCorr[j_array==j],
#             scale[j_array==j], shape[j_array==j])
#             for j in range(j1, j2+1)
#         }

#     else:
#         j_array, C1_array, C2_array = get_location_scale(cm)
    
#         CDF = {
#             j: normal_cdf(
#             np.log(leaders.values[j][:, None]), 
#             C1_array[j_array == j] - ZPJCorr[j_array==j],
#             np.sqrt(C2_array[j_array == j]),
#             p=1)
#             for j in range(j1, j2+1)
#         }

#     if verbose:
#         plt.figure()
#         plot_cdf(CDF, j1, j2, pval=False)

#     idx_reject = {
#         j: np.zeros_like(CDF[j], dtype=bool) for j in range(j1, j2+1)
#     }

#     agg = compute_all_aggregate(CDF, j1, j2, leader=True)

#     # mask_nan = np.isnan(agg[:, :agg.shape[1] // 2, 0, 0]).any(axis=1)
#     mask_nan = np.isnan(agg[:, :, 0, 0]).any(axis=1)

#     clusterer = hdbscan.HDBSCAN(
#         cluster_selection_epsilon=5, metric='euclidean',
#         min_cluster_size=(~mask_nan).sum() // 2 + 1,
#         min_samples=1, cluster_selection_method='eom',
#         allow_single_cluster=True, gen_min_span_tree=verbose,
#         algorithm='boruvka_balltree')

#     # clusterer = hdbscan.HDBSCAN(
#     #     cluster_selection_epsilon=.9, metric='minkowski', min_cluster_size=1920,
#     #     min_samples=1,
#     #     cluster_selection_method='eom', p=2,
#     #     allow_single_cluster=True, gen_min_span_tree=verbose)

#     for idx_signal, idx_range in tqdm(np.ndindex(CDF[j1].shape[1:])):

#         standard_embedding = umap.UMAP(
#             n_components=j2-j1,
#             n_neighbors=20,
#             metric='manhattan',
#             n_epochs=10000,
#             set_op_mix_ratio=.5,
#             min_dist=0,
#         # ).fit_transform(agg[~mask_nan, :agg.shape[1] // 2, idx_signal, idx_range])
#         ).fit_transform(agg[~mask_nan, :, idx_signal, idx_range])

#         if verbose:
#             plt.figure()    
#             N = (~mask_nan).sum()
#             cmap = sns.color_palette('inferno', as_cmap=True)
#             ax = sns.scatterplot(
#                 x=standard_embedding[:, 0], y=standard_embedding[:, 1],
#                 s=30, color=cmap(np.arange(N)/N), legend=False)
#             ax.set(xlabel='1st embedding dimension',
#                    ylabel='2nd embedding dimension',
#                    title='UMAP embedding')

#         p = clusterer.fit_predict(standard_embedding)

#         # p = clusterer.fit_predict(agg[~mask_nan, :28, idx_signal, idx_range])

#         if verbose and idx_signal == 0 and idx_range == 0:
#             plt.figure()
#             sns.histplot(clusterer.minimum_spanning_tree_.to_pandas().distance.values, log=True)

#         # First slice to mask_nan shape, which correspond to the 
#         idx_reject[j1][:mask_nan.shape[0]][~mask_nan, idx_signal, idx_range] = p == -1

#     # if verbose:

#     #     print(agg[~idx_reject[j1][:mask_nan.shape[0], 0, 0]].shape)

#     #     standard_embedding = umap.UMAP(
#     #         n_components=j2-j1, #agg.shape[1],
#     #         n_neighbors=20,
#     #         metric='manhattan',
#     #         n_epochs=100,
#     #         set_op_mix_ratio=.5,
#     #         min_dist=0,
#     #     # ).fit_transform(agg[~mask_nan, :agg.shape[1] // 2, idx_signal, idx_range])
#     #     ).fit_transform(agg[~(idx_reject[j1][:mask_nan.shape[0], idx_signal, idx_range] | mask_nan), :, idx_signal, idx_range])

#     #     plt.figure()
#     #     N = (~mask_nan).sum()
#     #     cmap = sns.color_palette('inferno', as_cmap=True)
#     #     ax = sns.scatterplot(x=standard_embedding[:, 0], y=standard_embedding[:, 1],
#     #                          s=30, legend=False)
#     #     ax.set(xlabel='1st embedding dimension', ylabel='2nd embedding dimension', title='UMAP embedding')

#     #     p = clusterer.fit_predict(standard_embedding)

#     #     # p = clusterer.fit_predict(agg[~mask_nan, :28, idx_signal, idx_range])

#     #     if verbose and idx_signal == 0 and idx_range == 0:
#     #         plt.figure()
#     #         sns.histplot(clusterer.minimum_spanning_tree_.to_pandas().distance.values, log=True)

#     return idx_reject


# def normal_cdf(x, mu, sigma, p):
#     return .5 * (1 + erf((x - p * mu) / (p * sigma * np.sqrt(2))))


from scipy import stats, special


def gen_cdf(x, mu, alpha, beta):
    return (
        .5 + np.sign(x - mu) / 2
        * special.gammainc(1/beta, (np.abs(x - mu) / alpha) ** beta))


def normal_cdf(x, mu, sigma, p):
    return .5 * (1 + special.erf((x - p * mu) / (p * sigma * np.sqrt(2))))


def compute_aggregate(CDF, j1, j2):

    max_index = CDF[j2].shape[0] * 2 ** (j2 - j1)
    agg = np.zeros((max_index, j2-j1+1, *CDF[j1].shape[1:]))

    agg[:, 0] = CDF[j1][:max_index]

    i = 0
    for n in range(1, j2-j1+1):

        xp = np.arange(CDF[j1+n].shape[0]) + .5
        step = 2 ** -n
        x = np.linspace(
            0+step/2, CDF[j1+n].shape[0]-step/2,
            num=CDF[j1+n].shape[0] * 2 ** n)

        for idx_signal, idx_range in np.ndindex(CDF[j1].shape[1:]):

            # x = np.sort(np.r_[*[xp - 2 ** -n + i * 2 ** (-n+1) for k in range(2 ** n)]]
            agg[:, n, idx_signal, idx_range] = np.interp(
                x, xp, CDF[j1+n][:, idx_signal, idx_range])[:max_index]

    return agg


def cluster_reject_leaders(j1, j2, cm, leaders, pelt_beta, verbose=False,
                           generalized=False, pelt_jump=1, threshold=2.5,
                           hilbert_weighted=False):
    
    from .hilbert import HilbertCost, w_hilbert
    import ruptures as rpt
    
    # ZPJCorr = leaders.correct_pleaders(cm.j.min(), cm.j.max())
    idx_j = np.s_[cm.j.min() - min(leaders.values):
                  cm.j.max() - min(leaders.values) + 1]
    ZPJCorr = leaders.correct_pleaders()[..., idx_j]
    ZPJCorr = np.log(ZPJCorr).transpose(2, 0, 1)

    if generalized:

        j_array, C1_array, scale, shape = get_location_scale_shape(cm)

        CDF = {
            j: gen_cdf(
            np.log(leaders.values[j][:, None]),
            C1_array[j_array == j] - ZPJCorr[j_array==j],
            scale[j_array==j], shape[j_array==j])
            for j in range(j1, j2+1)
        }

    else:
        j_array, C1_array, scale = get_location_scale(cm)
    
        CDF = {
            j: normal_cdf(
            np.log(leaders.values[j][:, None]), 
            C1_array[j_array == j] - ZPJCorr[j_array==j],
            np.sqrt(scale[j_array == j]),
            p=1)
            for j in range(j1, j2+1)
        }

    skip_scales = {}

    for idx_range, idx_signal in np.ndindex(CDF[j1].shape[1:]):

        skip_scales[(idx_range, idx_signal)] = [
            j for j in range(j1, j2+1)
            if scale[j_array == j, idx_range, idx_signal] <= 0]

    if verbose:
        plt.figure()
        plot_cdf(CDF, j1, j2, pval=False)

    idx_reject = {
        j: np.zeros_like(CDF[j], dtype=bool) for j in CDF
        # j: np.zeros((CDF[j].shape[0], CDF[j].shape[2]), dtype=bool)
        for j in CDF
    }

    agg = compute_aggregate(CDF, j1, j2)
    # max_index = agg.shape[0]

    # max_index = CDF[j2].shape[0] * 2 ** (j2 - j1)

    for idx_range, idx_signal in np.ndindex(CDF[j1].shape[1:]):

        mask_nan_global = np.isnan(agg[:, :, idx_range, idx_signal]).any(axis=1)

        w = np.r_[
            [-np.sum((pk * np.log(pk))[pk != 0])
            for pk
            in agg[~mask_nan_global, :, idx_range, idx_signal].transpose()]
        ]

        if not hilbert_weighted:
            w = np.ones_like(w)

        if len(skip_scales) > 0:
            for scale in skip_scales[(idx_range, idx_signal)]:
                w[scale-j1] = 0

        w /= w.sum()
        w *= w.shape[0]

        if verbose:
            print(f'{w=}')

        pelt = rpt.Pelt(custom_cost=HilbertCost(w=w), jump=pelt_jump)

        result = [0] + pelt.fit_predict(
            agg[~mask_nan_global, :, idx_range, idx_signal], pelt_beta)
        result[-1] -= 1

        if verbose:
            rpt.display(agg[~mask_nan_global, 0, 0, 0], [], result, figsize=(7, 2))
            kernel_matrix = distance.squareform(distance.pdist(
                agg[~mask_nan_global, :, 0, 0], metric=w_hilbert, w=w))
            plt.show()
            sns.heatmap(kernel_matrix)
            plt.vlines(result, 0, max(result))
            plt.show()
        
        reachable_index = np.arange(agg.shape[0])[~mask_nan_global]
        result_j = [reachable_index[r] for r in result]
        result_j[-1] += 1

        N_bins = ceil(1.5 * agg[~mask_nan_global].shape[0] ** (1/3))

        for j in range(agg.shape[1]):

            # skip this scale because it does not contain relevant information
            if j+j1 in skip_scales[(idx_range, idx_signal)]:
                continue

            stat = []
            median = []

            # mask_nan = np.isnan(agg[:, j, idx_signal, idx_range])

            samples = []

            for i in range(len(result) - 1):
                samples.append(
                    agg[:, j, idx_range, idx_signal][
                        result_j[i]:result_j[i+1]])

            if len(samples) == 1:
                continue

            right_edge = np.nanmax(agg[:, j, idx_range, idx_signal])
            bins = np.sort(
                np.r_[1, 1-np.geomspace(1 - right_edge, 1, N_bins-1)])

            for i in range(len(samples)):

                samp = samples[i]

                # python >= 3.11 
                # other_samples = np.r_[*samples[:i], *samples[i+1:]]
                other_samples = np.concatenate((*samples[:i], *samples[i+1:]))
                
                # bins = np.linspace(0, 1, N_bins)
                # samp_hist, _ = np.histogram(samp, bins=bins)
                # other_hist, _ = np.histogram(other_samples, bins=bins)

                # samp_hist = samp_hist / samp_hist.sum()
                # other_hist = other_hist / other_hist.sum()

                # stat.append(special.kl_div(samp_hist, other_hist).sum())
                stat.append(stats.wasserstein_distance(
                    -np.log(1 - samp),
                    # python >= 3.11
                    # -np.log(1 - np.r_[*samples[:i], *samples[i+1:]])))
                    -np.log(1 - other_samples)))
                # stat.append(spatial.distance.jensenshannon(samp_hist, other_hist))
                median.append(np.median(samp))

            # threshold = 2 ** (j / 4) * 1.25
            outlier_idx = np.arange(len(stat))[
                (np.array(stat) > threshold) & (np.array(median) > .75)]

            # mask = np.zeros(idx_reject[j1+j].shape[0], dtype=bool)

            # mask_nan = np.isnan(CDF[j1+j][:, idx_range, idx_signal])

            # annoyingly, masking returns a view
            # accessible_indices = np.arange(mask.shape[0])[~mask_nan_global]

            for idx in outlier_idx:

                # sl = accessible_indices[result[idx] // (2 ** (j)):result[idx+1] // (2 ** (j))+1]
                # mask[sl] = True

                idx_reject[j1+j][
                    result_j[idx] // (2 ** (j)):
                    result_j[idx+1] // (2 ** (j))+1,
                    idx_range, idx_signal] = True

                for jj in range(j):
                    idx_reject[j1+jj][
                        result_j[idx] // (2 ** (jj)):
                        result_j[idx+1] // (2 ** (jj))+1,
                        idx_range, idx_signal] = True

            if verbose:
                print(stat)

    return idx_reject


def get_outliers(wt_coefs, scaling_ranges, pelt_beta, threshold, pelt_jump=1,
                 robust_cm=False, verbose=False, generalized=False):
    """Detect outliers in a signal.

    Parameters
    ----------
    wt_coefs : WaveletDec
        Input coefficients of the signal with outliers.
    scaling_ranges : list[tuple[int, int]]
        List of pairs of (j1, j2) ranges of scales for the linear regressions.
    pelt_beta : float
        Regularization parameter for the PELT segmentation.
    threshold : float
        Wasserstein distance threshold to indentify a segment as outlier.
    pelt_jump : int
        Optional, PELT algorithm checks segmentations every `pelt_jump` point.
    robust_cm : bool
        Whether to use robust cumulants in the detection.
    generalized : bool
        Whether to use the exponential power distribution model instead of
        the normal distribution for the log 1-leaders in the detection.
    verbose : bool, optional
        Display figures outlining the detection process. If multiple signals
        are being processed, will only show figures for the first signal.

    Returns
    -------
    leaders : WaveletLeader
        Wavelet 1-leaders used in the analysis.
    idx_reject : dict[int, ndarray]
        Dictionary associating to each scale the boolean mask of indices to
        reject.

    See Also
    --------
    mfa : Can be fed the output ``idx_reject``.
    """

    from .. import mfa
    
    p_exp = 1
    n_cumul = 4 if generalized else 2

    leaders = wt_coefs.get_leaders(p_exp, 1)

    lwt = mfa(leaders, scaling_ranges=scaling_ranges, n_cumul=n_cumul,
              robust=robust_cm)
    
    j2 = max(sr[1] for sr in scaling_ranges)
    min_scale = min(sr[0] for sr in scaling_ranges)

    if verbose:
        lwt.cumulants.plot(j1=4, nrow=4, figsize=(3.3, 4), n_cumul=4)
        plt.show()

    idx_reject = cluster_reject_leaders(
        min_scale, j2, lwt.cumulants, leaders, verbose=verbose,
        generalized=generalized, pelt_beta=pelt_beta, pelt_jump=pelt_jump,
        threshold=threshold)

    for j in range(min(idx_reject), max(idx_reject)):

        right_reject = idx_reject[j][1::2]
        left_reject = idx_reject[j][:right_reject.shape[0] * 2:2]

        combined = (left_reject | right_reject)[:idx_reject[j+1].shape[0]]
        idx_reject[j+1][combined] = True

    for j in range(min(idx_reject), max(idx_reject)+1):

        for k in range(3):

            start = k
            end = -3 + k

            idx_reject[j][3:] |= idx_reject[j][start:end]

        for k in range(3):

            start = k+1
            end = -2 + k
            if not end:
                end = None

            idx_reject[j][:-3] |= idx_reject[j][start:end]

    if verbose:
        
        idx_reject_pos = {
            scale: np.arange(idx_reject[scale].shape[0])[idx_reject[scale][:, 0, 0]]
            for scale in idx_reject
        }

        leaders.plot(min_scale, j2, nan_idx=idx_reject_pos)

        plt.figure()
        plt.plot(idx_reject[min_scale][:, 0, 0])

    return leaders, idx_reject
