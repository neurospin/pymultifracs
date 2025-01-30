"""
Authors: Omar D. Domingues <omar.darwiche-domingues@inria.fr>
         Merlin Dumeur <merlin@dumeur.net>
"""

from math import floor

import numpy as np


def prepare_weights(sf_nj_fun, weighted, n_ranges, j_min, j_max,
                    scaling_ranges, y, std=None):
    """
    Calculate regression weights.
    """

    if weighted == 'Nj':

        # w = np.tile(
        #     sf_nj_fun(floor(j_min), floor(j_max)).astype(float)[None, :]
        #     #     None, :, None, :],
        #     # (1, 1, n_ranges, 1)
        # )

        w = sf_nj_fun(floor(j_min), floor(j_max)).astype(float)[None, :]

    elif weighted == 'bootstrap':

        std[std == 0] = std[std != 0].min()
        std = 1/std

        # std shape (n_moments, n_scales, n_scaling_ranges, n_sig) ->
        # (n_moments, n_scales, n_scaling_ranges, n_rep)
        if std.ndim == 2:
            # TODO check this
            w = np.tile(std[:, :, None, None], (1, 1, n_ranges, 1)) ** 2
        # std shape (n_moments, n_scales, n_scaling_ranges, n_sig)
        else:
            w = std ** 2

    else:  # weighted is None
        w = np.ones((1, int(j_max - j_min + 1), n_ranges, 1))

    for i, (j1, j2) in enumerate(scaling_ranges):

        w[:, int(j2-j_min+1):, i, :] = np.nan
        w[:, :int(j1-j_min), i, :] = np.nan

    if np.isnan(y).any():
        mask = np.ones_like(y)
        mask[np.isnan(y)] = np.nan
        w = mask * w

    # shape (n_moments, n_scales, n_scaling_ranges, n_rep)
    return w


def prepare_regression(scaling_ranges, j):
    """
    Prepare range of scales and x support for regression.
    """

    n_ranges = len(scaling_ranges)
    j_min = min(sr[0] for sr in scaling_ranges)
    j_max = max(sr[1] for sr in scaling_ranges)

    # shape (n_moments, n_scales, n_scaling_ranges, n_rep)
    x = np.arange(j_min, j_max + 1)[None, :, None, None]

    return x, n_ranges, j_min, j_max, j_min - j.min(), j_max - j.min() + 1


def linear_regression(x, y, nj, return_variance=False):
    """
    Performs a (weighted or not) linear regression.
    Finds 'a' that minimizes the error:
        sum_j { n[j]*(y[j] - (a*x[j] + b))**2 }

    Args:
        x, y : regression variables
        nj: list containg the weigths
    Returns:
        a, b: angular coefficient and intercept

    (!!!!!!!!!!!!!)
    IMPORTANT:

    return_variance NOT DEBUGGED
    (!!!!!!!!!!!!!)
    """

    # bj = np.array(nj, dtype=np.float)
    assert isinstance(nj, np.ndarray)
    assert nj.shape[1] == x.shape[1]

    # shape (n_moments, n_scales, n_scaling_ranges, n_rep)
    V_0 = np.nansum(nj, axis=1)[:, None, :, :]
    V_1 = np.nansum(nj * x, axis=1)[:, None, :, :]
    V_2 = np.nansum(nj * (x**2), axis=1)[:, None, :, :]

    weights_slope = nj * (V_0*x - V_1)/(V_0*V_2 - V_1*V_1)
    weights_intercept = nj * (V_2 - V_1*x)/(V_0*V_2 - V_1*V_1)

    a = np.nansum(weights_slope*y, axis=1)
    b = np.nansum(weights_intercept*y, axis=1)

    wt = np.zeros_like(nj)
    wt[nj != 0] = 1 / nj[nj != 0]
    var_a = np.nansum(wt*weights_slope*weights_slope, axis=1)

    if not return_variance:
        return a, b
    else:
        return a, b, var_a


def compute_R2(moment, slope, intercept, weights, j_min_max, j):
    """
    Computes :math:`R^2` for linear regression.
    """

    weights = 1 / weights

    x, _, _, _, j_min_idx, j_max_idx = prepare_regression(j_min_max, j)

    # Shape (n_moments, n_scales, n_scaling_ranges, n_rep)
    moment = moment[:, j_min_idx:j_max_idx, :]
    slope = slope[:, None, :]
    intercept = intercept[:, None, :]

    # x = np.arange(j_min, j_max + 1)[None, :, None, None]

    res = (weights ** 2 * (moment - x * slope - intercept) ** 2).sum(axis=1)

    avg = (moment * weights).mean(axis=1)[:, None, :]
    tot = ((moment * weights - avg) ** 2).sum(axis=1)

    return 1 - res / tot


def compute_RMSE(moment, slope, intercept, weights, j_min_max, j):

    weights = 1 / weights

    x, _, _, _, j_min_idx, j_max_idx = prepare_regression(j_min_max, j)

    # Shape (n_moments, n_scales, n_scaling_ranges, n_rep)
    moment = moment[:, j_min_idx:j_max_idx, :]
    slope = slope[:, None, :]
    intercept = intercept[:, None, :]

    # x = np.arange(j_min, j_max + 1)[None, :, None, None]

    res = (weights ** 2 * (moment - x * slope - intercept) ** 2).mean(axis=1)

    return np.sqrt(res)
