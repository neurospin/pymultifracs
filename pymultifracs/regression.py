"""
Authors: Omar D. Domingues <omar.darwiche-domingues@inria.fr>
         Merlin Dumeur <merlin@dumeur.net>
"""

from math import floor
from functools import partial

import numpy as np
import xarray as xr

from .backend import jnp, jit, vmap

from .utils import Dim, scaling_range_to_str


def prepare_weights(sf_nj_fun, weighted, scaling_ranges, j, std=None):
    """
    Calculate regression weights.
    """

    if weighted == 'Nj':
        w = sf_nj_fun().astype(float)  # .copy(deep=True)

    elif weighted == 'bootstrap':

        # Collect the standard deviation of the bootstrapped objects
        std = std()

        if mask := np.isclose(std.values, 0).any():
            std.values[mask] = std.where(std > 0).min()

        std = 1 / std

        # std shape (n_moments, n_scales, n_scaling_ranges, n_channel) ->
        # (n_moments, n_scales, n_scaling_ranges, n_rep)
        if std.ndim == 2:
            # TODO check this
            raise ValueError('')
            n_ranges = len(scaling_ranges)
            w = np.tile(std[:, :, None, None], (1, 1, n_ranges, 1)) ** 2
        # std shape (n_moments, n_scales, n_scaling_ranges, n_channel)
        else:
            w = std ** 2

    else:  # weighted is None
        w = xr.DataArray(jnp.ones(len(j)), coords=j.coords)

    if Dim.scaling_range not in w.dims:
        w = w.expand_dims({Dim.scaling_range: [
                scaling_range_to_str(s) for s in scaling_ranges
            ]}).copy()

    for i, (j1, j2) in enumerate(scaling_ranges):

        # Where j is outside the scaling range, set the weights to zero
        j1 = xr.DataArray(
            jnp.array([sr[0] for sr in scaling_ranges]),
            coords=[w.coords[Dim.scaling_range]], dims=Dim.scaling_range)
        j2 = xr.DataArray(
            jnp.array([sr[1] for sr in scaling_ranges]),
            coords=[w.coords[Dim.scaling_range]], dims=Dim.scaling_range)

        # if (idx_out := (j < j1) | (j > j2)).any():
        w = xr.where((j < j1) | (j > j2), 0, w)

        # w[{Dim.scaling_range: i, Dim.j: (w.j > j2) | (w.j < j1)}] = jnp.nan

    # w.where(np.isnan(y), np.nan)
    # w.values[np.isnan(y)] = np.nan
    # w = w.where(~np.isnan(y), np.nan)

    # if np.isnan(y).any():
    #     mask = np.ones_like(y)
    #     mask[np.isnan(y)] = np.nan
    #     w = mask * w

    # shape (n_moments, n_scales, n_scaling_ranges, n_rep)
    return w


def prepare_regression(scaling_ranges, j, dims):
    """
    Prepare range of scales and x support for regression.
    """

    n_ranges = len(scaling_ranges)
    j_min = min(sr[0] for sr in scaling_ranges)
    j_max = max(sr[1] for sr in scaling_ranges)

    # same shape as scaling function
    # x = xr.DataArray(
    #     np.arange(j_min, j_max + 1).astype(float),
    #     coords={'j': np.arange(j_min, j_max + 1)}
    # )
    # x = x.expand_dims([d for d in dims if d != Dim.j])
    # x = x.transpose(*dims)

    return n_ranges, j_min, j_max, j_min - j.min(), j_max - j.min() + 1


@jit
@partial(jnp.vectorize, signature='(n),(n),(n)->(2)')
def linear_regression_ufunc(x, y, weights):
    """
    Performs a single (weighted) linear regression.
    Meant to be called with :func:`xr.apply_ufunc`.
    """
    return jnp.polyfit(x, y, w=weights, full=False, deg=1)


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
    assert isinstance(nj, xr.DataArray)

    # assert nj.shape[1] == x.shape[1]

    # slope, intercept = np.polyfit(x, y, 1, )

    # # shape (n_moments, n_scales, n_scaling_ranges, n_rep)

    V_0 = nj.sum(dim='j', skipna=True, min_count=3)
    V_1 = (nj * x).sum(dim='j', skipna=True, min_count=3)
    V_2 = (nj * x ** 2).sum(dim='j', skipna=True, min_count=3)

    # V_0 = np.nansum(nj.values, axis=axis_j)
    # V_1 = np.nansum((nj * x).values, axis=axis_j)
    # V_2 = np.nansum((nj * (x**2)), axis=axis_j)

    weights_slope = nj * (V_0*x - V_1)/(V_0*V_2 - V_1*V_1)
    weights_intercept = nj * (V_2 - V_1*x)/(V_0*V_2 - V_1*V_1)

    a = (weights_slope * y).sum(dim='j', skipna=True, min_count=3)
    b = (weights_intercept * y).sum(dim='j', skipna=True, min_count=3)

    # a = np.nansum(weights_slope*y, axis=1)
    # b = np.nansum(weights_intercept*y, axis=1)

    if not return_variance:
        return a, b

    wt = xr.zeros_like(nj)
    # wt.where(nj != 0, 1 / nj.values[nj != 0])
    wt.values[nj != 0] = 1 / nj.values[nj != 0]
    var_a = (wt*weights_slope*weights_slope).sum(dim='j', skipna=True)

    return a, b, var_a


def compute_R2(moment, slope, intercept, weights, j_min_max, j):
    """
    Computes :math:`R^2` for linear regression.
    """

    weights = 1 / weights

    x, _, j_min, j_max, j_min_idx, j_max_idx = prepare_regression(
        j_min_max, j, dims=moment.dims)

    # Shape (n_moments, n_scales, n_scaling_ranges, n_rep)
    moment = moment.sel(j=slice(j_min, j_max))

    # x = np.arange(j_min, j_max + 1)[None, :, None, None]

    res = (weights ** 2 * (moment - x * slope - intercept) ** 2).sum(
        skipna=True, dim=Dim.j)

    avg = (moment * weights).mean(dim=Dim.j, skipna=True)
    tot = ((moment * weights - avg) ** 2).sum(dim=Dim.j, skipna=True)

    return 1 - res / tot


def compute_RMSE(moment, slope, intercept, weights, j_min_max, j):

    weights = 1 / weights

    x, _, j_min, j_max, j_min_idx, j_max_idx = prepare_regression(
        j_min_max, j, dims=moment.dims)

    # Shape (n_moments, n_scales, n_scaling_ranges, n_rep)
    moment = moment.sel(j=slice(j_min,j_max))
    slope = slope
    intercept = intercept

    # x = np.arange(j_min, j_max + 1)[None, :, None, None]

    res = (weights ** 2 * (moment - x * slope - intercept) ** 2).mean(
        dim=Dim.j, skipna=True)

    return np.sqrt(res)
