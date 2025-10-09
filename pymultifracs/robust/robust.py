"""
Authors: Merlin Dumeur <merlin@dumeur.net>
"""

import warnings
from math import ceil

import numpy as np
import xarray as xr

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed

from scipy import stats, special
import scipy.spatial.distance as distance
from scipy.optimize import bisect


from ..utils import fast_power, get_edge_reject, Dim


def compute_robust_cumulants(X, dims, m_array, alpha=1):

    from statsmodels.robust.scale import qn_scale
    from statsmodels.robust.norms import estimate_location, TukeyBiweight

    dim = (Dim.m,)
    shape = (len(m_array),)

    mrq_dims, mrq_shapes = [], []

    for d in dims:

        if d in [Dim.k_j, *dim]:
            continue

        mrq_dims.append(d)
        mrq_shapes.append(X.shape[dims.index(d)])

    centered_moment = xr.DataArray(
        np.zeros((*shape, *mrq_shapes)), dims=((*dim, *mrq_dims)),
        coords={Dim.m: m_array})
    values = xr.zeros_like(centered_moment)

    # idx_unreliable = (~np.isnan(X)).sum(axis=dims.index(Dim.k_j)) < 3

    # compute robust moments

    if X.shape[dims.index(Dim.k_j)] > 10000:
        values = np.nan
        return values

    def est_wrapper(X):

        X_norm = X[~np.isinf(X) & ~np.isnan(X)]

        q_est = qn_scale(X_norm)

        try:
            m_est = estimate_location(X_norm, q_est, norm=TukeyBiweight(),
                                      maxiter=1000)
        except ValueError:
            if X_norm.shape[0] < 20:
                m_est = np.nan
            else:
                m_est = np.median(X_norm)

        return np.stack([m_est, q_est ** 2], axis=-1)

    values[{Dim.m: np.s_[:3]}] = xr.DataArray(
        np.apply_along_axis(est_wrapper, axis=dims.index(Dim.k_j), arr=X),
        dims=(*(d for d in dims if d != Dim.k_j), Dim.m))

    if max(m_array) <= 2:
        return values

    X -= values.sel(m=1)
    X /= np.sqrt(values.sel(m=2))

    decaying_factor = (
        alpha
        * np.exp(-.5 * (alpha ** 2 - 1)
                 * np.linalg.norm(X, 2, axis=dims.index(Dim.k_j))
                 )
    )

    for m in m_array:

        if m == 1:
            centered_moment[{Dim.m: m}] = X.mean()

        elif m == 2:
            centered_moment[{Dim.m: m}] = X.var()

        else:
            centered_moment[{Dim.m: m}] = (
                fast_power(alpha * X, m) * decaying_factor).mean(
                axis=dims.index(Dim.k_j))

    for m in m_array:

        if m <= 2:
            continue

        for n in np.arange(1, m):

            aux = xr.zeros_like(centered_moment.sel(m=m))

            temp_moment = centered_moment.sel(m=n)

            if n <= 2:
                temp_value = centered_moment.sel(m=n)
            else:
                temp_value = values.sel(m=n)

            aux += (special.binom(m-1, n-1) * temp_value * temp_moment)

        values[{Dim.m: m}] = centered_moment.sel(m=m) - aux

    return values


def _C4_to_m4(C4, C2):
    return C4 + 3 * C2 ** 2


def _C2_to_m2(C2):
    return C2


def _get_location_scale(cm, fix_c2_slope=False):

    slope_c1 = cm.slope.sel(m=1)
    intercept_c1 = cm.intercept.sel(m=1)

    slope_c2 = cm.slope.sel(m=2)
    intercept_c2 = cm.intercept.sel(m=2)

    if fix_c2_slope and slope_c2 > 0:
        slope_c2.values[slope_c2 > 0] = 0
        for scaling_range in cm.scaling_ranges:
            intercept_c2.iloc[{Dim.scaling_range}] = cm.C2.sel(
                j=slice(scaling_range[0], scaling_range[1])).mean(dim=Dim.j)

    j_array = xr.DataArray(
        np.arange(cm.j.min(), cm.j.max() + 1),
        coords={'j': np.arange(cm.j.min(), cm.j.max() + 1)}
    )

    C1_array = slope_c1 * j_array + intercept_c1
    C2_array = slope_c2 * j_array + intercept_c2

    return j_array, C1_array, C2_array


def _get_location_scale_shape(cm, fix_c2_slope=False):

    slope_c1 = cm.slope.sel(m=1)
    intercept_c1 = cm.intercept.sel(m=1)

    slope_c2 = cm.slope.sel(m=2)
    intercept_c2 = cm.intercept.sel(m=2)

    if fix_c2_slope and slope_c2 > 0:
        slope_c2[:] = 0
        for k, scaling_range in enumerate(cm.scaling_ranges):
            j_min = cm.j.min()
            intercept_c2[:, k] = cm.C2[
                np.s_[scaling_range[0]-j_min:scaling_range[1]-j_min]].mean()

    slope_c4 = cm.slope.sel(m=4)
    intercept_c4 = cm.intercept.sel(m=4)

    j_array = xr.DataArray(
        np.arange(1, cm.j.max() + 1),
        coords={Dim.j: np.arange(1, cm.j.max() + 1)}
    )

    C1_array = slope_c1 * j_array + intercept_c1
    C2_array = slope_c2 * j_array + intercept_c2
    C4_array = slope_c4 * j_array + intercept_c4

    alpha = xr.zeros_like(C2_array)
    beta = xr.zeros_like(C4_array)

    for i, (C2, C4) in enumerate(zip(C2_array, C4_array)):

        for k, l in np.ndindex(beta[i].shape):

            if C2[k, l] <= 0:
                beta[i, k, l] = 1
                continue

            f_beta = lambda beta: (
                special.gamma(5/beta)
                * special.gamma(1/beta)
                / special.gamma(3/beta)**2
                - 3 - C4[k, l])

            if f_beta(.1) > 0 and f_beta(10) > 0:

                warnings.warn("Very high value of beta estimated")
                beta[i, k, l] = 10

            elif f_beta(.1) < 0 and f_beta(10) < 0:

                warnings.warn("Very low value of beta estimated")
                beta[i, k, l] = .1

            else:
                beta[i, k, l] = bisect(f_beta, .1, 10)

        alpha[i] = np.sqrt(
            C2 * special.gamma(1/beta[i]) / special.gamma(3 / beta[i]))

    idx_zero = (alpha < 0) | (np.isnan(alpha))
    alpha.values[idx_zero] = 0

    idx_zero = beta < .1
    beta.values[idx_zero] = .1

    return j_array, C1_array, alpha, beta


def plot_cdf(cdf, j1, j2, ax=None, vmin=None, vmax=None, pval=False, cbar=True,
             figsize=(2.5, 1), nan_idx=None, signal_idx=0,
             range_idx=0):

    min_all = min([np.nanmin(np.abs(cdf[s]))
                   for s in range(j1, j2+1) if s in cdf])

    if vmax is None:
        vmax = max([np.nanmax(cdf[s]) for s in range(j1, j2+1) if s in cdf])
    if vmin is None:
        vmin = min_all

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, layout='constrained')

    cmap = mpl.cm.get_cmap('inferno').copy()
    cmap.set_bad('grey')

    for scale in range(j1, j2 + 1):

        if scale not in cdf:
            continue

        temp = cdf[scale].isel(channel=signal_idx, scaling_range=range_idx)

        X = ((np.arange(temp.shape[0] + 1))
             * (2 ** (scale - j1 + 1)))
        X = np.tile(X[:, None], (1, 2))

        C = np.copy(temp)[:, None]

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

    if cbar:

        locator = mpl.ticker.MaxNLocator(4, symmetric=False)
        cb = plt.colorbar(qm, ax=ax, ticks=locator, fraction=.1, aspect=8)
        cb.ax.tick_params(which='major', size=3)
        cb.ax.tick_params(which='minor', right=False)


def gen_cdf(x, mu, alpha, beta):
    return (
        .5 + np.sign(x - mu) / 2
        * special.gammainc(1/beta, (np.abs(x - mu) / alpha) ** beta))


def normal_cdf(x, mu, sigma, p):
    return .5 * (1 + special.erf((x - p * mu) / (p * sigma * np.sqrt(2))))


def compute_aggregate(CDF, j1, j2):

    max_index = CDF[j2].sizes[Dim.k_j] * 2 ** (j2 - j1)
    agg = xr.DataArray(
        np.zeros((max_index, j2-j1+1,
                  *(s for d, s in CDF[j1].sizes.items() if d != Dim.k_j))),
        dims=(Dim.k_j, Dim.j, *(d for d in CDF[j2].dims if d != Dim.k_j)),
        coords={Dim.j: np.arange(j1, j2+1)},
    )

    agg[{Dim.j: 0}] = CDF[j1].isel({Dim.k_j: np.s_[:max_index]})

    # i = 0

    for n in range(1, j2-j1+1):

        xp = np.arange(CDF[j1+n].sizes[Dim.k_j]) + .5
        step = 2 ** -n
        x = np.linspace(
            0+step/2, CDF[j1+n].sizes[Dim.k_j]-step/2,
            num=CDF[j1+n].sizes[Dim.k_j] * 2 ** n)

        for idx_signal, idx_range in np.ndindex(
               *[CDF[j1].sizes[s]
                 for s in [Dim.channel, Dim.scaling_range]]):

            idx_dict = {Dim.channel: idx_signal, Dim.scaling_range: idx_range}
            agg[{Dim.j: n, **idx_dict}] = np.interp(
                x, xp,
                CDF[j1+n].isel({**idx_dict, Dim.k_j: np.s_[:max_index]})
                )

    return agg


def _segment_reject_signal(
        signal_idx, range_idx, pk, pelt_beta, threshold, hilbert_weighted, j1,
        pelt_jump=1, skip_scales=None, verbose=False, return_stat=False):

    import ruptures as rpt
    from .hilbert import HilbertCost, w_hilbert

    mask_nan_global = np.isnan(pk).any(dim=Dim.j).values

    w = xr.DataArray(
        (pk.values * np.log(pk).where(pk == 0, 0).values).sum(
            axis=pk.dims.index(Dim.k_j)),
        dims=(d for d in pk.dims if d != Dim.k_j)
    )

    if not hilbert_weighted:
        w = np.ones_like(w)

    if skip_scales is not None:
        for scale in skip_scales:
            w[scale-j1] = 0

    w /= w.sum()
    w *= w.shape[0]

    if verbose:
        print(f'{w=}')

    pelt = rpt.Pelt(custom_cost=HilbertCost(w=w), jump=pelt_jump)

    result = [0] + pelt.fit_predict(
        pk.values[~mask_nan_global], pelt_beta)

    result[-1] -= 1

    if verbose:

        rpt.display(
            pk.isel(j=0).values[~mask_nan_global], [], result, figsize=(7, 2))
        plt.show()

        kernel_matrix = distance.squareform(distance.pdist(
            pk.values[~mask_nan_global], metric=w_hilbert, w=w))
        sns.heatmap(kernel_matrix)
        plt.vlines(result, 0, max(result))
        plt.show()

    reachable_index = np.arange(pk.sizes[Dim.k_j])[~mask_nan_global]
    result = {int(j): [reachable_index[r] for r in result]
              for j in pk.j}

    output = {}

    for j in result:

        result[j][-1] += 1

        # skip this scale because it does not contain relevant information
        if skip_scales is not None and j in skip_scales:
            continue

        stat = []
        median = []

        samples = []

        for i in range(len(result[j]) - 1):
            samples.append(
                pk.sel({
                    Dim.j: j,
                    Dim.k_j: np.s_[result[j][i]:result[j][i+1]]
                    })
            )

        if len(samples) == 1:
            continue

        for i, samp in enumerate(samples):

            # python >= 3.11
            other_samples = np.concatenate((*samples[:i], *samples[i+1:]))

            stat.append(stats.wasserstein_distance(
                -np.log(1 - samp),
                # python >= 3.11
                # -np.log(1 - np.r_[*samples[:i], *samples[i+1:]])))
                -np.log(1 - other_samples)))
            median.append(np.median(samp))

        if threshold is None:
            output[j] = np.array(stat)

        else:
            # threshold = 2 ** (j / 4) * 1.25
            output[j] = np.arange(len(stat))[
                (np.array(stat) > threshold) & (np.array(median) > .75)]

        if verbose:
            print(stat)

    return output, result, signal_idx, range_idx


def cluster_reject_leaders(j1, j2, cm, leaders, pelt_beta, verbose=False,
                           generalized=False, pelt_jump=1, threshold=2.5,
                           hilbert_weighted=False, remove_edges=False,
                           n_jobs=1):

    if generalized:

        _, C1_array, scale, shape = _get_location_scale_shape(cm)

        CDF = {
            j: gen_cdf(
                np.log(leaders.get_values(j)),
                C1_array.sel(j=j),
                scale.sel(j=j), shape.sel(j=j))
            for j in range(j1, j2+1)
        }

    else:

        _, C1_array, scale = _get_location_scale(cm)

        CDF = {
            j: normal_cdf(
                np.log(leaders.get_values(j)),
                C1_array.sel(j=j),
                np.sqrt(scale.sel(j=j)),
                p=1)
            for j in range(j1, j2+1)
        }

    skip_scales = {}

    for idx_range in range(CDF[j1].sizes[Dim.scaling_range]):
        for idx_signal in range(CDF[j1].sizes[Dim.channel]):

            skip_scales[(idx_range, idx_signal)] = [
                j for j in range(j1, j2+1)
                if scale.sel(j=j).isel(
                    scaling_range=idx_range, channel=idx_signal) <= 0]

    if verbose:
        plt.figure()
        plot_cdf(CDF, j1, j2, pval=False)
        plt.show()

    if remove_edges:

        idx_reject = get_edge_reject(leaders)

        if threshold is None:

            temp = {
                j: xr.zeros_like(idx_reject[j], dtype=float)
                for j in idx_reject
            }

            for j in temp:
                temp[j].values[idx_reject[j].values] = np.nan

            idx_reject = temp

    else:
        idx_reject = {
            j: xr.zeros_like(
                CDF[j], dtype=bool if threshold is not None else float)
            for j in CDF}

    agg = compute_aggregate(CDF, j1, j2)

    inputs = [
        (idx_signal, idx_range,
         agg.isel(channel=idx_signal, scaling_range=idx_range),
         skip_scales[(idx_range, idx_signal)])
        for idx_range, idx_signal in np.ndindex(
            CDF[j1].sizes[Dim.scaling_range], CDF[j1].sizes[Dim.channel])
    ]

    out = Parallel(n_jobs=n_jobs)(
        delayed(_segment_reject_signal)(
            signal_idx, range_idx, pk, pelt_beta, threshold, hilbert_weighted,
            j1, pelt_jump=pelt_jump, skip_scales=skip, verbose=verbose)
        for signal_idx, range_idx, pk, skip in inputs)

    for outlier_idx, result, signal_idx, range_idx in out:

        for j in result:

            if threshold is None:

                for i, stat in enumerate(outlier_idx[j]):

                    idx_reject[j][
                        {Dim.k_j: np.s_[result[j][i] // (2 ** (j-j1)):
                                        result[j][i+1] // (2 ** (j-j1))+1],
                            Dim.scaling_range: range_idx,
                            Dim.channel: signal_idx}] = stat

                continue

            for idx in outlier_idx[j]:

                idx_reject[j][
                    {Dim.k_j: np.s_[result[j][idx] // (2 ** (j-j1)):
                                    result[j][idx+1] // (2 ** (j-j1))+1],
                        Dim.scaling_range: range_idx,
                        Dim.channel: signal_idx}] = True

                for jj in range(j-j1):
                    idx_reject[j1+jj][
                        {Dim.k_j: np.s_[result[j][idx] // (2 ** (jj)):
                                        result[j][idx+1] // (2 ** (jj))+1],
                            Dim.scaling_range: range_idx,
                            Dim.channel: signal_idx}] = True

    return idx_reject


def get_outliers(wt_coefs, scaling_ranges, pelt_beta, threshold, pelt_jump=1,
                 robust_cm=False, verbose=False, generalized=False,
                 remove_edges=False, n_jobs=1, spread=3):
    """Detect outliers in a signal.

    Parameters
    ----------
    wt_coefs : :class:`.WaveletDec`
        Input coefficients of the signal with outliers.
    scaling_ranges : list[tuple[int, int]]
        List of pairs of (j1, j2) ranges of scales for the linear regressions.
    pelt_beta : float
        Regularization parameter for the PELT segmentation.
    threshold : float | None
        Wasserstein distance threshold to indentify a segment as outlier.
        If None, the function returns an array of floats, corresponding to the
        Wasserstein statistic, which then needs to be transformed into a
        boolean array.
    pelt_jump : int
        Optional, PELT algorithm checks segmentations every `pelt_jump` point.
    robust_cm : bool
        Whether to use robust cumulants in the detection.
    verbose : bool, optional
        Display figures outlining the detection process. If multiple signals
        are being processed, will only show figures for the first signal.
    generalized : bool
        Whether to use the exponential power distribution model instead of
        the normal distribution for the log 1-leaders in the detection.
    remove_edges : bool
        Whether to remove the edge coefficients at finer scales (not covered
        by the detection algorithm).
    n_jobs : int
        Number of joblib parallel threads to use (across channels).

        .. versionadded:: 0.3.2 Multiprocessing support was added
    spread : int
        Number of coefficients neighboring each detected segments that should
        also be removed. Useful to remove the influence of affected
        coefficients at the edge of detection, especially in high noise
        conditions. Should normally be >= 3.

    Returns
    -------
    leaders : :class:`.WaveletLeader`
        Wavelet 1-leaders used in the analysis.
    idx_reject : dict[int, ndarray]
        Dictionary associating to each scale the boolean mask of indices to
        reject.

    See Also
    --------
    :func:`.mfa` : Can be fed the output dictionary: ``idx_reject``.
    """

    from .. import mfa

    p_exp = 1
    n_cumul = 4 if generalized else 2

    leaders = wt_coefs.get_leaders(p_exp, 1, 1)

    j2 = max(sr[1] for sr in scaling_ranges)
    min_scale = min(sr[0] for sr in scaling_ranges)

    lwt = mfa(leaders, scaling_ranges=scaling_ranges, n_cumul=n_cumul,
              robust=robust_cm, min_j=min_scale, estimates='c')

    if verbose:
        lwt.cumulants.plot(j1=min_scale, nrow=4, figsize=(3.3, 4))
        plt.show()

    idx_reject = cluster_reject_leaders(
        min_scale, j2, lwt.cumulants, leaders, verbose=verbose,
        generalized=generalized, pelt_beta=pelt_beta, pelt_jump=pelt_jump,
        threshold=threshold, remove_edges=remove_edges, n_jobs=n_jobs)

    if threshold is None:
        return None, idx_reject

    for j in range(min(idx_reject), max(idx_reject)):

        right_reject = idx_reject[j].isel({Dim.k_j: np.s_[1::2]})
        left_reject = idx_reject[j].isel(
            {Dim.k_j: np.s_[:right_reject.sizes[Dim.k_j] * 2:2]})

        combined = (left_reject | right_reject)[
            {Dim.k_j: np.s_[:idx_reject[j+1].sizes[Dim.k_j]]}]
        idx_reject[j+1].values[combined] = True
        # print(combined.shape, idx_reject[j+1].shape)

    for j in range(min(idx_reject), max(idx_reject)+1):

        for k in range(spread):

            # right-side
            start = k
            end = -spread + k

            idx_reject[j][{Dim.k_j: np.s_[spread:]}] |= \
                idx_reject[j][{Dim.k_j: np.s_[start:end]}]

            # left-side
            start = k + 1
            end = -spread + 1 + k
            if not end:
                end = None

            idx_reject[j][{Dim.k_j: np.s_[:-spread]}] |= \
                idx_reject[j][{Dim.k_j: np.s_[start:end]}]

    if verbose:

        leaders.plot(min_scale, j2, nan_idx=idx_reject)

        plt.figure()
        plt.plot(idx_reject[min_scale].isel(channel=0, scaling_range=0))
        plt.show()

    return leaders, idx_reject
