import warnings
from tqdm.auto import tqdm

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import gennorm
from scipy import stats
from scipy import ndimage
from scipy.optimize import bisect
from scipy.special import gamma, erf, gammaincc

import hdbscan
import umap


def _high_weighted_median(a, weights):
    """
    Computes a weighted high median of a. This is defined as the
    smallest a[j] such that the sum over all a[i]<=a[j] is strictly
    greater than half the total sum of the weights
    """
    a_cp = np.copy(a)
    weights_cp = np.copy(weights)
    n = a_cp.shape[0]
    sorted_a = np.zeros((n,), dtype=np.double)
    a_cand = np.zeros((n,), dtype=np.double)
    weights_cand = np.zeros((n,), dtype=np.intc)
    kcand = 0
    wleft, wright, wmid, wtot, wrest = [0] * 5
    trial = 0
    wtot = np.sum(weights_cp)
    for i in range(100000):
        wleft = 0
        wmid = 0
        wright = 0
        for i in range(n):
            sorted_a[i] = a_cp[i]
        sorted_a = np.partition(sorted_a, kth=n//2)
        trial = sorted_a[n//2]
        for i in range(n):
            if a_cp[i] < trial:
                wleft = wleft + weights_cp[i]
            elif a_cp[i] > trial:
                wright = wright + weights_cp[i]
            else:
                wmid = wmid + weights_cp[i]
        kcand = 0
        if 2 * (wrest + wleft) > wtot:
            for i in range(n):
                if a_cp[i] < trial:
                    a_cand[kcand] = a_cp[i]
                    weights_cand[kcand] = weights_cp[i]
                    kcand = kcand + 1
        elif 2 * (wrest + wleft + wmid) <= wtot:
            for i in range(n):
                if a_cp[i] > trial:
                    a_cand[kcand] = a_cp[i]
                    weights_cand[kcand] = weights_cp[i]
                    kcand = kcand + 1
            wrest = wrest + wleft + wmid
        else:
            return trial
        n = kcand
        for i in range(n):
            a_cp[i] = a_cand[i]
            weights_cp[i] = weights_cand[i]
    print('aargh')
    raise ValueError('Weighted median did not converge')


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
            f_beta = lambda beta: gamma(5/beta) * gamma(1/beta) / gamma(3/beta)**2 - 3 - C4[k, l]
            # f_beta = lambda beta: gamma(5/beta) * gamma(1/beta) / gamma(3/beta)**2 - 3 - C4[k, l] / C2[k, l] ** 2

            if f_beta(.1) > 0 and f_beta(10) > 0:

                warnings.warn("Very high value of beta estimated")
                beta[i, k, l] = 10

            elif f_beta(.1) < 0 and f_beta(10) < 0:

                warnings.warn("Very low value of beta estimated")
                beta[i, k, l] = .1

            else:
                beta[i, k, l] = bisect(f_beta, .1, 10)

        alpha[i] = np.sqrt(C2 * gamma(1/beta[i]) / gamma(3/beta[i]))

        idx_zero = C2 < 0
        alpha[i, idx_zero] = 0

        idx_zero = beta[i] < .1
        beta[i, idx_zero] = .1

    return j_array, C1_array, alpha, beta


def sample_p_leaders(p_exp, *gennorm_args):

    # normal args:
    # 0 - beta / shape
    # 1 - location
    # 2 - alpha / scale
    # 3 - array size

    # if gennorm_args[2] == 0:
    #     return (np.e ** gennorm_args[1] * np.ones(gennorm_args[3])) ** p_exp

    sim = (np.e ** gennorm.rvs(*gennorm_args)) ** p_exp

    idx_zero = gennorm_args[2] == 0

    # print(gennorm_args[1][idx_zero][None, :].shape, sim[:, idx_zero].shape,
    #       np.ones_like(sim[:, idx_zero]).shape)

    sim[:, idx_zero] = (np.e ** gennorm_args[1][idx_zero][None, :]
                        * np.ones_like(sim[:, idx_zero])) ** p_exp

    return sim


def sample_reject(k, l, j2, min_scale, p_exp, shape, location, scale, n_samples,
                  wt_coefs, alpha, previous_reject, max_reject_share, verbose):

     # if scale[idx, k, l] == 0:
        #     continue  # Already initialized at zero

    samples_scale_j = None

    # N leaders = n_coefs - 2
    idx_reject = {}

    for j in range(j2, min_scale-1, -1):

        idx = j-1
        idx_below = j-2

        # if samples_scale_j is None:
        samples_scale_j = sample_p_leaders(p_exp, shape[idx],
                                            location[idx], scale[idx],
                                            (n_samples, *shape[idx].shape))

        if j == 1:
            # diff_element = np.zeros((1, *shape[idx].shape))
            diff_element = 0
        else:
            samples_scales_below = [
                sample_p_leaders(p_exp, shape[idx_below], location[idx_below], scale[idx_below], (n_samples)),
                sample_p_leaders(p_exp, shape[idx_below], location[idx_below], scale[idx_below], (n_samples))
            ]
            diff_element = .5 * (samples_scales_below[0]
                                 + samples_scales_below[1])

        # N leaders = n_coefs - 2
        idx_reject[j] = np.zeros((*shape[idx].shape, wt_coefs.values[j].shape[0] - 2), dtype=bool)

        diff_samples = samples_scale_j - diff_element

        # print(samples_scale_j.mean(axis=0), samples_scale_j.std(axis=0))

        # temp_diff = diff_sample

        if j > 1:
            diff_samples = diff_samples[diff_samples >= 0]

        # print(f"{j}, {temp_diff.shape=}")

        # try:
        # print(alpha / 2, 100 - alpha / 2)
        ci = [np.percentile(diff_samples, (alpha / 2 ** (1))),
                np.percentile(diff_samples, 100 - (alpha / 2 ** (1)))]

        vals = np.abs(wt_coefs.values[j][:, l]) ** p_exp

        v = np.sum(np.stack([vals[:-2], vals[1:-1], vals[2:]], axis=1),
                    axis=1)

        check = ((v < ci[0]) | (v > ci[1])) & ~(np.isnan(v))

        if previous_reject is not None:
            prev = previous_reject[j][k, l]
        else:
            prev = np.zeros_like(check, dtype=bool)

        prev_kept = check & prev
        nan = np.isnan(v)
        N_available = check.shape[0] - (prev_kept|nan).sum()

        N_new_remove = (check&~prev&~nan).sum()
        Max_new_reject = int(np.ceil(max_reject_share * N_available))

        # print(N_new_remove, Max_new_reject)

        if N_new_remove > Max_new_reject + 1:

            # pseudo_quantiles = np.argsort(temp_diff) / temp_diff.shape[0]
            combined_quantiles = np.argsort(np.argsort(np.r_[diff_samples, v])) / (diff_samples.shape[0] + v.shape[0])

            combined_quantiles = 2 * abs(combined_quantiles - .5)

            # Same shape as v, associates to every element its position in the sorted set of quantiles of {temp_diff U v}
            # Small values are associated to more extreme quantiles
            idx_v = np.arange(diff_samples.shape[0], combined_quantiles.shape[0])

            # Set quantiles from carried over rejected values to zero so they
            # Don't appear at the end of the sorted array
            combined_quantiles[idx_v][prev_kept] = 0

            order_v = np.argsort(combined_quantiles[idx_v])

            # Renormalize order_v values from [0, N_v + N_tempdiff[ to [0, N_v]
            # order_v = np.argsort(np.argsort(order_v))

            # Nans end up at the end of the argsorted array so need to remove them
            N_nans = np.isnan(v).sum()

            # Remove only first [(1-alpha) * N] highest quantile observations
            check[:] = False
            check[order_v[-Max_new_reject-N_nans-1:-N_nans]] = True

            # Add previously rejected values still kept in
            check |= prev_kept

        idx_reject[j] = check

        if j == 12 and verbose:

            print(scale[idx], shape[idx])

            # print(v[check], ci[0], ci[1])

            plt.figure()
            plt.scatter(diff_element, samples_scale_j)
            plt.show()

            # vals = WT.wt_leaders.values[j-1] ** p
            # lower_vals =  1/2 * np.sum(np.c_[
            #     vals[:-3:2],
            #     vals[3::2]
            # ], axis=1)

            # plt.scatter(lower_vals, WT.wt_leaders.values[j].squeeze() ** p)

        if verbose and j == 12:
            plt.figure()

            sns.histplot({0: diff_samples[~np.isinf(diff_samples)], 1: v[~np.isnan(v) & (v > 0)]}, stat='percent',
                            log_scale=True, common_norm=False)
            ylim = plt.ylim()
            plt.vlines(ci, *ylim, color='k')
            plt.ylim(*ylim)
            plt.show()

            plt.figure()
            plt.hist(v[v < .025])
            plt.show()

    return {(k, l): idx_reject}


def reject_coefs(wt_coefs, cm, p_exp, n_samples, alpha, converged, error,
                 fix_c2_slope=False, previous_reject=None, verbose=False,
                 max_reject_share=None, min_scale=0, n_jobs=1, ):

    if max_reject_share is None:
        max_reject_share = 1 - alpha

    j_array, location, scale, shape = get_location_scale_shape(cm)

    idx_reject = {}
    samples_scale_j = None

    j2 = j_array.max()

    for k, l in np.ndindex(location.shape[1:]):
        if not (converged[k, l] or error[k, l]):

            try:

                idx_reject |= sample_reject(
                    k, l, j2, min_scale, p_exp, shape[:, k, l], location[:, k, l], scale[:, k, l], n_samples,
                    wt_coefs, alpha, previous_reject, max_reject_share, verbose)
            except Exception:
                idx_reject[(k, l)] = None
                error[k, l] = True

        else:
            idx_reject[(k, l)] = None

    out = {}

    for scale in range(min_scale, j2+1):

        out[scale] = np.zeros((*location.shape[1:], wt_coefs.values[scale].shape[0] - 2), dtype=bool)
        for k, l in np.ndindex(location.shape[1:]):
            if not (converged[k, l] or error[k, l]):
                out[scale][k, l] = idx_reject[(k, l)][scale]

    return out, error


def iterate_analysis(WT, n_iter=10):

    lwt = mf_analysis([WT.wt_leaders], scaling_ranges=[(6, WT.wt_leaders.j2_eff() - 2)], n_cumul=4)[0]

    c_trace = np.zeros((4, n_iter+1))

    c_trace[:, 0] = lwt.cumulants.log_cumulants.squeeze()

    for i in range(n_iter):

        idx_reject = reject_coefs(WT.wt_coefs, lwt.cumulants, 2, 1000000,
                                  verbose=False)
        new_leaders, _ = compute_leaders2(WT.wt_coefs, gamint=1, p_exp=2, j1=6,
                                          j2_reg=12, idx_reject=idx_reject)
        lwt = mf_analysis([new_leaders],
                          scaling_ranges=[(6, WT.wt_leaders.j2_eff() - 2)],
                          n_cumul=4)[0]

        c_trace[:, i+1] = lwt.cumulants.log_cumulants.squeeze()

    return c_trace, new_leaders, idx_reject, lwt


def compute_aggregate(cdf, j1, j2, leader=False):

    out = {}

    for j in range(j1, j2+1):

        out[j] = cdf[j]

        for k in range(j1, j):

            # print(j, k, np.nanmax(out[k]))
            # out[k] = np.r_['-1, 4, 0', out[k][1::2], out[k][:-1:2]].max(axis=-1)

            if leader:
                # out[k] = (np.r_['-1, 4, 0', out[k][1::2], out[k][:-1:2]] ** 2 / 2).sum(axis=-1)
                out[k] = np.sqrt((np.r_['-1, 4, 0', out[k][1::2], out[k][:-1:2]]).prod(axis=-1))
            else:
                out[k] = (np.r_['-1, 4, 0', out[k][1::2], out[k][:-1:2]]).max(axis=-1)

            # print(k, np.nanmax(out[k]))

        # if j > j1:
        #     print(out[j-1][-10:,0, 0])

    for j in range(j1, j2+1):
        if not leader:
            out[j] **= 2 ** ((j2-j))

    out = np.array([*out.values()]).swapaxes(0, 1)
    idx_nan = np.isnan(out).any(axis=(1, 2, 3))

    return out


def compute_all_aggregate(CDF, j1, j2, leader=False):

    Agg = {
        j: compute_aggregate(CDF, j1, j, leader) for j in range(j1, j2+1)
    }

    N = int((j2 - (j1) + 1) * (2 + j2 - (j1)) / 2)
    agg = np.zeros((Agg[j2].shape[0] * 2 ** (j2 - j1), N, *CDF[j1].shape[1:]))

    # agg shape N_coef, N_aggregates, N_ranges, N_signals
    # N_coef is determined by the upsampled number of coefficients 

    end = 0

    for j in range(min(Agg), j2+1):

        start = end
        end = int((j - j1 + 1) * (2 + j - j1) / 2)

        # cutting extraneous coefficients
        agg[:, start:end] = np.repeat(Agg[j], 2 ** (j - j1), axis=0)[:agg.shape[0]]

    return agg


def inv_log_gamma_cdf(x, window_size):
    out = gammaincc(window_size, -window_size * np.log(x))

    if np.isclose(out, 1).any():
        print('a')

    return out


def filter_gmean(input_line, output_line, window_size):

    if window_size == 1:
        output_line[:] = input_line

    else:
        output_line[:] = np.nan
        output_line[:] = stats.mstats.gmean(
            np.lib.stride_tricks.sliding_window_view(input_line, window_size), axis=1)
        output_line[:] = inv_log_gamma_cdf(output_line, window_size)        


def compute_all_aggregate2(CDF, j1, j2):

    N = int((j2 - (j1) + 1) * (2 + j2 - (j1)) / 2)
    agg = np.zeros((CDF[j2].shape[0] * 2 ** (j2 - j1), N, *CDF[j1].shape[1:]))

    acc = 0
    for j in range(j1, j2+1):
        
        for i in range(j2-j + 1):

            agg[:, acc] = np.repeat(ndimage.generic_filter1d(
                CDF[j], filter_gmean, 2**i, 0, mode='constant', cval=np.nan,
                extra_arguments=(2 ** i,), origin=0), 2 ** (j - j1), axis=0)[:agg.shape[0]]

            acc += 1

    # agg shape N_coef, N_aggregates, N_ranges, N_signals
    # N_coef is determined by the upsampled number of coefficients 

    return agg


def compute_all_aggregate3(CDF, j1, j2, leader=False):

    Agg = {
        j: compute_aggregate(CDF, j, j, leader) for j in range(j1, j2+1)
    }

    # N = int((j2 - (j1) + 1) * (2 + j2 - (j1)) / 2)
    agg = np.zeros((Agg[j2].shape[0] * 2 ** (j2 - j1), j2-j1+1, *CDF[j1].shape[1:]))

    # agg shape N_coef, N_aggregates, N_ranges, N_signals
    # N_coef is determined by the upsampled number of coefficients 

    end = 0

    for i, j in enumerate(range(min(Agg), j2+1)):

        # start = end
        # end = int((j - j1 + 1) * (2 + j - j1) / 2)

        # cutting extraneous coefficients
        agg[:, i] = np.repeat(Agg[j][:, 0], 2 ** (j - j1), axis=0)[:agg.shape[0]]

    return agg


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


def cluster_reject_leaders(j1, j2, p_exp, cm, coefs, leaders, verbose,
                           generalized=False):

    ZPJCorr = leaders.correct_pleaders(cm.j.min(), cm.j.max())
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
        j_array, C1_array, C2_array = get_location_scale(cm)
    
        CDF = {
            j: normal_cdf(
            np.log(leaders.values[j][:, None]), 
            C1_array[j_array == j] - ZPJCorr[j_array==j],
            np.sqrt(C2_array[j_array == j]),
            p=1)
            for j in range(j1, j2+1)
        }

    if verbose:
        plt.figure()
        plot_cdf(CDF, j1, j2, pval=False)

    idx_reject = {
        j: np.zeros_like(CDF[j], dtype=bool) for j in range(j1, j2+1)
    }

    agg = compute_all_aggregate(CDF, j1, j2, leader=True)

    # mask_nan = np.isnan(agg[:, :agg.shape[1] // 2, 0, 0]).any(axis=1)
    mask_nan = np.isnan(agg[:, :, 0, 0]).any(axis=1)

    clusterer = hdbscan.HDBSCAN(
        cluster_selection_epsilon=5, metric='euclidean',
        min_cluster_size=(~mask_nan).sum() // 2 + 1,
        min_samples=1, cluster_selection_method='eom',
        allow_single_cluster=True, gen_min_span_tree=verbose,
        algorithm='boruvka_balltree')

    # clusterer = hdbscan.HDBSCAN(
    #     cluster_selection_epsilon=.9, metric='minkowski', min_cluster_size=1920,
    #     min_samples=1,
    #     cluster_selection_method='eom', p=2,
    #     allow_single_cluster=True, gen_min_span_tree=verbose)

    for idx_signal, idx_range in tqdm(np.ndindex(CDF[j1].shape[1:])):

        standard_embedding = umap.UMAP(
            n_components=j2-j1,
            n_neighbors=20,
            metric='manhattan',
            n_epochs=10000,
            set_op_mix_ratio=.5,
            min_dist=0,
        # ).fit_transform(agg[~mask_nan, :agg.shape[1] // 2, idx_signal, idx_range])
        ).fit_transform(agg[~mask_nan, :, idx_signal, idx_range])

        if verbose:
            plt.figure()    
            N = (~mask_nan).sum()
            cmap = sns.color_palette('inferno', as_cmap=True)
            ax = sns.scatterplot(
                x=standard_embedding[:, 0], y=standard_embedding[:, 1],
                s=30, color=cmap(np.arange(N)/N), legend=False)
            ax.set(xlabel='1st embedding dimension',
                   ylabel='2nd embedding dimension',
                   title='UMAP embedding')

        p = clusterer.fit_predict(standard_embedding)

        # p = clusterer.fit_predict(agg[~mask_nan, :28, idx_signal, idx_range])

        if verbose and idx_signal == 0 and idx_range == 0:
            plt.figure()
            sns.histplot(clusterer.minimum_spanning_tree_.to_pandas().distance.values, log=True)

        # First slice to mask_nan shape, which correspond to the 
        idx_reject[j1][:mask_nan.shape[0]][~mask_nan, idx_signal, idx_range] = p == -1

    # if verbose:

    #     print(agg[~idx_reject[j1][:mask_nan.shape[0], 0, 0]].shape)

    #     standard_embedding = umap.UMAP(
    #         n_components=j2-j1, #agg.shape[1],
    #         n_neighbors=20,
    #         metric='manhattan',
    #         n_epochs=100,
    #         set_op_mix_ratio=.5,
    #         min_dist=0,
    #     # ).fit_transform(agg[~mask_nan, :agg.shape[1] // 2, idx_signal, idx_range])
    #     ).fit_transform(agg[~(idx_reject[j1][:mask_nan.shape[0], idx_signal, idx_range] | mask_nan), :, idx_signal, idx_range])

    #     plt.figure()
    #     N = (~mask_nan).sum()
    #     cmap = sns.color_palette('inferno', as_cmap=True)
    #     ax = sns.scatterplot(x=standard_embedding[:, 0], y=standard_embedding[:, 1],
    #                          s=30, legend=False)
    #     ax.set(xlabel='1st embedding dimension', ylabel='2nd embedding dimension', title='UMAP embedding')

    #     p = clusterer.fit_predict(standard_embedding)

    #     # p = clusterer.fit_predict(agg[~mask_nan, :28, idx_signal, idx_range])

    #     if verbose and idx_signal == 0 and idx_range == 0:
    #         plt.figure()
    #         sns.histplot(clusterer.minimum_spanning_tree_.to_pandas().distance.values, log=True)

    return idx_reject


def normal_cdf(x, mu, sigma, p):
    return .5 * (1 + erf((x - p * mu) / (p * sigma * np.sqrt(2))))
