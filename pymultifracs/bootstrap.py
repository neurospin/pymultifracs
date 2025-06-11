"""
Authors: Merlin Dumeur <merlin@dumeur.net>

Extension of code from the recombinator project:
https://github.com/InvestmentSystems/recombinator/
"""

from typing import Tuple
from collections.abc import Iterable

import numpy as np
import xarray as xr

from recombinator.block_bootstrap import\
     _verify_shape_of_bootstrap_input_data_and_get_dimensions, \
     BlockBootstrapType, _verify_block_bootstrap_arguments, \
     _generate_block_start_indices_and_successive_indices, \
     _general_block_bootstrap_loop, circular_block_bootstrap

from .utils import max_scale_bootstrap, Dim, scaling_range_to_str


def estimate_confidence_interval_from_bootstrap(
        bootstrap_estimates: np.ndarray,
        confidence_level: float = 95.0) \
        -> Tuple[float, float]:
    """
    This function estimates a confidence interval of an estimator given a
    variety of estimates of the same statistic from resampled data.
    Args:
        bootstrap_estimates: a NumPy array of dimension (B, ) containing the
                             statistic computed from resampled data
        confidence_level: the confidence level associated with the confidence
                          interval in percent (i.e. between 0 and 100)
    """

    # bootstrap estimates shape (n_j, n_channel, n_rep)
    percent = 100.0 - confidence_level

    # bootstrap_estimates: shape (..., n_CI)
    # idx_unreliable = (~np.isnan(bootstrap_estimates)).sum(axis=-1) < 3
    idx_unreliable = (~np.isnan(bootstrap_estimates)).sum(dim=Dim.bootstrap) < 3

    bootstrap_confidence_interval = xr.concat([
        bootstrap_estimates.quantile(
            percent / 2.0 / 100, dim=Dim.bootstrap, skipna=True),
        bootstrap_estimates.quantile(
            1.0 - percent / 2.0 / 100, dim=Dim.bootstrap, skipna=True),
        ],
        dim=xr.DataArray(['upper', 'lower'], dims=[Dim.CI])
    )

    bootstrap_confidence_interval.where(idx_unreliable, np.nan)

    # if bootstrap_confidence_interval.ndim > 2:
    #     return np.rollaxis(
    #         bootstrap_confidence_interval, 0,
    #         bootstrap_confidence_interval.ndim)

    # if bootstrap_confidence_interval.ndim > 3:
    #     return bootstrap_confidence_interval.transpose(1, 2, 3, 0)

    # if bootstrap_confidence_interval.ndim > 2:
    #     return bootstrap_confidence_interval.transpose(1, 2, 0)

    return bootstrap_confidence_interval


def get_empirical_variance(mrq, ref_mrq, name):
    """
    Returns empirical variance about a reference value.
    """

    if mrq is None:
        return None

    attribute = getattr(mrq, name)
    ref_attribute = getattr(ref_mrq, name)

    if callable(attribute):

        def wrapper(*args, **kwargs):

            var = np.mean((attribute(*args, **kwargs)
                           - ref_attribute(*args, **kwargs)) ** 2, axis=-1)

            return var

        return wrapper

    # attribute: shape (n_ranges, n_rep) if c_m or s_q
    #            shape (n_scales, n_rep) if C_m or S_q
    var = np.mean((attribute - ref_attribute) ** 2, axis=-1)

    return var


def get_variance(mrq, name):
    """
    Returns the variance of the mrq's attribute.
    """

    if mrq is None:
        return None

    attribute = getattr(mrq, name)

    if callable(attribute):

        def wrapper(*args, **kwargs):

            var = np.var(attribute(*args, **kwargs), ddof=1, axis=-1)

            return var

        return wrapper

    # attribute: shape (n_ranges, n_rep) if c_m or s_q
    #            shape (n_scales, n_rep) if C_m or S_q
    var = np.var((attribute), ddof=1, axis=-1)

    return var


def get_std(mrq, name):
    """
    Returns the standard deviation of an mrq's attribute
    """

    if mrq is None:
        return None

    attribute = getattr(mrq, name)

    if callable(attribute):

        def wrapper(*args, **kwargs):

            var = attribute(*args, **kwargs)
            # var = var.reshape(*var.shape[:-1], mrq.n_channel, -1)
            # unreliable = (~np.isnan(var)).sum(axis=-1) < 3
            # std = np.nanstd(var, ddof=1, axis=-1)
            # std[unreliable] = np.nan
            std = np.nanstd(
                var, axis=var.dims.index(Dim.bootstrap), ddof=1)
            # std.where((~np.isnan(attribute)).sum(dim=Dim.bootstrap))
            std.values[(~np.isnan(var)).sum(dim=Dim.bootstrap) < 3] = np.nan

            return std

        return wrapper

    # n_rep = attribute.shape[-1]

    # # shape (..., n_rep) -> (..., n_channel, n_rep_per_sig)
    # attribute = attribute.reshape(*attribute.shape[:-1], mrq.n_channel, -1)
    # TODO: improve the following section with xarray
    # if (attribute.shape[-2] != mrq.n_channel
    #         or attribute.shape[-1] * attribute.shape[-2] != mrq.n_rep):
    #     attribute = reshape(attribute, mrq.n_channel)

    # unreliable = (~np.isnan(attribute)).sum(axis=-1) < 3
    # std = np.nanstd(
    #     attribute, axis=attribute.dims.index(Dim.bootstrap), ddof=1)

    std = attribute.std(skipna=True, dim=Dim.bootstrap, ddof=1)
    # std.where((~np.isnan(attribute)).sum(dim=Dim.bootstrap), np.nan)
    std.values[(~np.isnan(attribute)).sum(dim=Dim.bootstrap) < 3] = np.nan

    # std[unreliable] = np.nan

    return std


def reshape(attribute, n_channel):
    """
    Automatically reshapes to expected shape.
    """
    if n_channel == 1:
        return attribute
    return attribute.reshape((*attribute.shape[:-1], n_channel, -1))


def get_confidence_interval(mrq, name):
    """
    Computes empirical confidence intervals for an attribute of the MRQ.
    """

    if mrq is None:
        return None

    # shape (n_scale, n_rep)
    attribute = getattr(mrq, name)

    if callable(attribute):

        def wrapper(*args, **kwargs):

            CI = estimate_confidence_interval_from_bootstrap(
                attribute(*args, **kwargs))

            return CI

        return wrapper

    CI = estimate_confidence_interval_from_bootstrap(attribute)

    return CI


def estimate_empirical_bootstrap(bootstrap_estimate,
                                 central_estimate,
                                 confidence_level=95.0):
    """
    Estimates empirical CI using bootstrap.
    """

    return (central_estimate
            - estimate_confidence_interval_from_bootstrap(
                central_estimate - bootstrap_estimate, confidence_level)
            )


def _get_align_slice(attribute, mrq, ref_mrq):
    """
    Align attributes from different multi res quantities in case they have \
        different minimum j values
    """

    # Case where not scaling function (no dependence on j)
    if attribute.shape[0] != mrq.j.shape[0]:
        return np.s_[:], np.s_[:]

    # assert attribute.shape[0] == mrq.j.shape[0]

    mrq_start = 0
    ref_mrq_start = 0

    if (min_diff := mrq.j.min() - ref_mrq.j.min()) > 0:
        # mrq_slice = np.s_[:]
        # ref_mrq_slice = np.s_[mrq.j.min()-ref_mrq.j.min():]
        ref_mrq_start = min_diff
    elif min_diff < 0:
        # mrq_slice = np.s_[ref_mrq.j.min()-mrq.j.min():]
        # ref_mrq_slice = np.s_[:]
        mrq_start = -min_diff

    mrq_end = mrq.j.shape[0]
    ref_mrq_end = ref_mrq.j.shape[0]

    if (max_diff := mrq.j.max() - ref_mrq.j.max()) > 0:
        mrq_end = -max_diff
    elif max_diff < 0:
        ref_mrq_end = max_diff

    mrq_start = int(mrq_start)
    mrq_end = int(mrq_end)
    ref_mrq_start = int(ref_mrq_start)
    ref_mrq_end = int(ref_mrq_end)

    return np.s_[mrq_start:mrq_end], np.s_[ref_mrq_start:ref_mrq_end]


def get_empirical_CI(mrq, ref_mrq, name):
    """
    Returns the empirical CI for an attribue of the MRQ.
    """

    if mrq is None:
        return None

    attribute = getattr(mrq, name)
    ref_attribute = getattr(ref_mrq, name)

    if callable(attribute):

        def wrapper(*args, **kwargs):

            attr = attribute(*args, **kwargs)
            ref = ref_attribute(*args, **kwargs)

            mrq_slice, ref_mrq_slice = _get_align_slice(attr, mrq, ref_mrq)

            CI = estimate_empirical_bootstrap(
                attr[mrq_slice], ref[ref_mrq_slice])

            return CI

        return wrapper

    mrq_slice, ref_mrq_slice = _get_align_slice(attribute, mrq, ref_mrq)

    CI = estimate_empirical_bootstrap(attribute[mrq_slice],
                                      ref_attribute[ref_mrq_slice])

    return CI


def bootstrap(mrq, R, wt_name, min_scale=1):
    """
    Perform regular block bootstraping.
    """

    max_scale = max_scale_bootstrap(mrq)

    values = {
        scale: circular_block_bootstrap(
            data, mrq.filt_len, R).transpose(1, 2, 0)
        for scale, data in mrq.values.items()
        if scale <= max_scale and scale >= min_scale
    }

    for k, v in values.items():
        values[k] = v.reshape((v.shape[0], -1))

    nj = {
        scale: np.array([*array] * R)
        for scale, array in mrq.nj.items()
        if scale <= max_scale and scale >= min_scale
    }

    new_mrq = mrq._from_dict({
        'formalism': mrq.formalism,
        'gamint': mrq.gamint,
        'nj': nj,
        'values': values,
        'wt_name': wt_name,
        # 'n_channel': mrq.n_channel * R,
    })

    return new_mrq


def _general_leader_bootstrap_loop(
        indices, block_length, min_scale, max_scale):

    if block_length / (2 ** max_scale - 1) > 1:
        raise ValueError('block length is too large w/ regards to max scale')

    indices_out = {}

    indices_out[min_scale] = indices

    for scale in range(min_scale+1, max_scale + 1):

        index = indices / (2 ** (scale - min_scale - 1))

        idx_int = (indices % (2 ** (scale - min_scale - 1))) == 0

        idx_int[idx_int] &= (index[idx_int].astype(int) % 2).astype(bool)

        rep_indices = np.zeros_like(idx_int, dtype=int) - 1
        rep_indices[idx_int] = (index[idx_int].astype(int) - 1) / 2

        indices_out[scale] = rep_indices

    return indices_out


def _general_leader_bootstrap(
        x, min_scale, max_scale, block_length, replications,
        sub_sample_length=None, link_rngs=True):

    circular = True
    replace = True

    T, _ = _verify_shape_of_bootstrap_input_data_and_get_dimensions(x)

    if not sub_sample_length:
        sub_sample_length = T

    if circular:
        bootstrap_type = BlockBootstrapType.CIRCULAR_BLOCK
    else:
        bootstrap_type = BlockBootstrapType.MOVING_BLOCK

    _verify_block_bootstrap_arguments(x=x,
                                      block_length=block_length,
                                      replications=replications,
                                      replace=replace,
                                      bootstrap_type=bootstrap_type,
                                      sub_sample_length=sub_sample_length)

    # Looks like it's useless, still keep for now
    block_start_indices, successive_indices \
        = _generate_block_start_indices_and_successive_indices(
                    sample_length=T,
                    block_length=block_length,
                    circular=circular,
                    successive_3d=False)

    block_start_indices = np.arange(0, x.shape[0])

    # Make sure to not have blocks start and end with nans
    # nan_idx = np.isnan(x)
    # idx_start = ~nan_idx & ~np.roll(nan_idx, -block_length)
    # start_indices = np.arange(0, x.shape[0])[idx_start]

    # Samples indices from possible values
    indices \
        = _general_block_bootstrap_loop(
            block_length=block_length, replications=replications,
            block_start_indices=block_start_indices,
            successive_indices=successive_indices,
            sub_sample_length=sub_sample_length,
            replace=replace, link_rngs=link_rngs)

    # Casts the lower-scale indices to higher scales => returns dict
    indices = _general_leader_bootstrap_loop(
        indices, block_length, min_scale, max_scale)

    return indices


def _create_bootstrapped_obj(mrq, indices, min_scale, block_length, double,
                             indices_double, replications):

    if double:

        values_double = {rep: {} for rep in indices_double}
        nj_double = {rep: {} for rep in indices_double}

    values = {}

    min_shape = mrq.get_values(min_scale).transpose(
        Dim.k_j, Dim.channel, ...).shape

    # nj = {}

    for scale, indices_scale in indices.items():

        if scale < min_scale:
            continue

        data = mrq.get_values(scale).transpose(Dim.k_j, Dim.channel, ...)
        dims = data.dims
        shape = data.shape
        data = data.values

        data = np.vstack((data, data[:block_length]))

        idx = indices_scale[indices_scale >= 0]

        if double:

            for rep in indices_double:

                idx_final = np.zeros_like(indices_scale, dtype=int) - 1
                idx_double = indices_double[rep][scale]
                idx_double = idx_double[idx_double >= 0]

                temp_indices = indices_scale[rep]
                temp_indices = temp_indices[temp_indices >= 0]

                if indices_scale[rep].ndim == 1:
                    temp_indices = np.hstack((temp_indices, temp_indices))
                else:
                    temp_indices = np.vstack((temp_indices, temp_indices))

                idx_final[indices_double[rep][scale] >= 0] = \
                    temp_indices[idx_double]

                out_double = np.zeros(
                    (replications, *min_shape), dtype=float) + np.nan
                out_double[idx_final >= 0] = data[idx_final[idx_final >= 0]]

                compact_idx = np.all(np.isnan(out_double), axis=0)

                values_double[rep][scale] = \
                    out_double[:, ~compact_idx].transpose()

                nj_double[rep][scale] = np.array(
                    [idx_final[rep2][idx_final[rep2] >= 0].shape[0]
                     for rep2 in range(replications)])

        out = np.zeros((replications, *min_shape), dtype=float) + np.nan

        out[indices_scale >= 0] = data[idx]
        # out = np.where((indices_scale >= 0).expand_dims(), data[idx], out)

        compact_idx = np.all(
            np.isnan(out), axis=tuple(k for k in range(out.ndim) if k != 1))
        # print(compact_idx.shape, out.shape)
        values[scale] = out[:, ~compact_idx].transpose(
            *[*range(out.ndim)[1:], 0])

        # values[scale] = values[scale].reshape(*values[scale].shape[:-2], -1)

        # nj[scale] = np.array([(~np.isnan(values[scale])).sum(axis=0)])

    new_mrq = mrq._from_dict({
        'values': values,
    })

    new_mrq.eta_p = None

    new_mrq.dims = (*dims, Dim.bootstrap)

    if double:

        double_mrq = {
            rep: mrq._from_dict({
                'values': values_double[rep]
            })
            for rep in values_double}

        for rep in double_mrq:
            double_mrq[rep].eta_p = None

        double_mrq.dims = (*double_mrq.dims, Dim.bootstrap)

        return new_mrq, double_mrq

    return new_mrq


def circular_leader_bootstrap(mrq, min_scale, max_scale, block_length,
                              replications, sub_sample_length=None,
                              link_rngs=True, double=False):
    """
    Returns double-bootstrapped multi-res quantities

    Parameters
    ----------
    mrq: :class:`~pymultifracs.multiresquantity.MultiResQuantity`
        Multi-resolution quantity to bootstrap. Needs to only have 1 repetition

    max_scale: int
        Maximum scale at which to perform bootstrap. In the case where there
        are not enough coefficients, bootstrap will be performed up to the
        maximum scale possible instead.

    block_length: int
        Size of the support of the discrete wavelet function.

    replications: int
        Number of replications to perform.

    sub_sample_length: int | None
        length of the sub-samples to generate

    link_rngs: bool
        whether to synchronize the states of Numba's and Numpy's random number
        generators

    double: bool
        whether to perform double bootstrapping

    Returns
    -------
    bootstrap_obj: :class:`~pymultifracs.multiresquantity.MultiResQuantity`
        A single MRQ that contains all the bootstrapped repetitions

    double_mrq: dict(int,
                     :class:`~pymultifracs.multiresquantity.MultiResQuantity`)
        A dictionary that relates a repetition in the bootstrap_obj to the
        MRQ containing the double-bootstrapped repetitions
        if `double` was passed as True
    """

    if isinstance(mrq, Iterable):
        reference_mrq = mrq[0]
    else:
        reference_mrq = mrq

    max_scale = min(max_scale_bootstrap(reference_mrq), max_scale)

    indices = _general_leader_bootstrap(
        reference_mrq.values[min_scale], min_scale, max_scale,
        block_length, replications, sub_sample_length, link_rngs)

    indices_double = {}

    if double:

        for rep in range(replications):
            indices_double[rep] = _general_leader_bootstrap(
                indices[1][rep][indices[1][rep] >= 0], max_scale, block_length,
                replications, sub_sample_length, link_rngs)

    if isinstance(mrq, Iterable):

        return [_create_bootstrapped_obj(
            m, indices, min_scale, block_length, double, indices_double,
            replications)
            for m in mrq]

    else:
        return _create_bootstrapped_obj(mrq, indices, min_scale, block_length,
                                        double, indices_double, replications)


def _need_redo_bootstrap(mrq, R, scaling_ranges):

    assert mrq.bootstrapped_obj is not None

    if R != mrq.bootstrapped_obj.get_n_bootstrap():
        return True

    if mrq.ZPJCorr is None:
        return False

    if (set(mrq.ZPJCorr.scaling_range.values)
            == set([scaling_range_to_str(s) for s in scaling_ranges[:2]])):
        return True

    return False