"""
Authors: Merlin Dumeur <merlin@dumeur.net>

Extension of code from the recombinator project:
https://github.com/InvestmentSystems/recombinator/
"""

from typing import Tuple

import numpy as np

from recombinator.block_bootstrap import\
     _verify_shape_of_bootstrap_input_data_and_get_dimensions,\
     BlockBootstrapType, _verify_block_bootstrap_arguments,\
     _generate_block_start_indices_and_successive_indices,\
     _general_block_bootstrap_loop, circular_block_bootstrap

from .utils import get_filter_length


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

    percent = 100.0 - confidence_level
    bootstrap_confidence_interval = np.array(
        [np.percentile(bootstrap_estimates, percent / 2.0, axis=-1),
         np.percentile(bootstrap_estimates, 100.0 - percent / 2.0, axis=-1)])

    return bootstrap_confidence_interval.transpose()


def get_confidence_interval(mrq, name):

    if mrq is None:
        return None

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

    return (central_estimate
            - estimate_confidence_interval_from_bootstrap(
                central_estimate - bootstrap_estimate, confidence_level)
            )


def get_empirical_CI(mrq, ref_mrq, name):

    if mrq is None:
        return None

    attribute = getattr(mrq, name)
    ref_attribute = getattr(ref_mrq, name)

    if callable(attribute):

        def wrapper(*args, **kwargs):

            CI = estimate_empirical_bootstrap(
                attribute(*args, **kwargs), ref_attribute(*args, **kwargs))

            return CI

        return wrapper

    CI = estimate_empirical_bootstrap(attribute, ref_attribute)

    return CI


def max_scale_bootstrap(mrq, filt_len):
    """
    Determines maximum scale possible to perform bootstrapping

    Parameters
    ----------
    mrq: :class:`~pymultifracs.multiresquantity.MultiResolutionQuantity`

    """

    for i, nj in mrq.nj.items():
        if nj < filt_len:
            i -= 1
            break

    return i


def bootstrap(mrq, R, wt_name):

    if mrq.nrep > 1:
        raise ValueError("Bootstrap only available for signals with 1 rep")

    filt_len = get_filter_length(wt_name)

    max_scale = max_scale_bootstrap(mrq, filt_len)

    values = {
        scale: circular_block_bootstrap(
            data[~np.isnan(data)], filt_len, R).transpose()
        for scale, data in mrq.values.items() if scale <= max_scale
    }

    nj = {
        scale: np.array([*array] * R)
        for scale, array in mrq.nj.items() if scale <= max_scale
    }

    new_mrq = mrq.from_dict({
        'formalism': mrq.formalism,
        'gamint': mrq.gamint,
        'nj': nj,
        'values': values,
    })

    return new_mrq


def _general_leader_bootstrap_loop(indices, block_length, max_scale):

    if block_length / (2 ** max_scale - 1) > 1:
        raise ValueError('block length is too large w/ regards to max scale')

    indices_out = {}

#     mask_block_end = np.zeros_like(indices[0], dtype=bool)
#     mask_block_end[block_start_indices[1:] - 1] = True

    indices_out[1] = indices

    for scale in range(2, max_scale + 1):

        index = indices / (2 ** (scale - 2))
#             index2 = (indices + 1) / (2 ** (scale - 2))

#             idx_int = index == index.astype(int)

        idx_int = (indices % (2 ** (scale - 2))) == 0

        idx_int[idx_int] &= (index[idx_int].astype(int) % 2).astype(bool)

        rep_indices = np.zeros_like(idx_int, dtype=int) - 1
        rep_indices[idx_int] = (index[idx_int].astype(int) - 1) / 2

        # for rep in range(indices.shape[0]):

        #     idx_int = index[rep] == index[rep].astype(int)
        #     idx_int[idx_int] &= (
        #         index[rep, idx_int].astype(int) % 2).astype(bool)

        #     idx_int2 = index2[rep] == index2[rep].astype(int)
        #     idx_int2 &= mask_block_end
        #     idx_int2[idx_int2] &= (
        #         index2[rep, idx_int2].astype(int) % 2).astype(bool)

        #     if rep == 0:
        #         print(f"{scale=} {idx_int.sum()=} {idx_int2.sum()=}")

        #     rep_indices = np.zeros_like(idx_int, dtype=int) - 1
        #     rep_indices[idx_int] = (
        #         index[rep, idx_int].astype(int) - 1) / 2

        #     rep_indices[idx_int2] = (
        #         index2[rep, idx_int2].astype(int) - 1) / 2

        #     indices_out[scale].append(rep_indices)

        indices_out[scale] = rep_indices

    return indices_out


def _general_leader_bootstrap(x, max_scale, block_length, replications,
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

    block_start_indices, successive_indices \
        = _generate_block_start_indices_and_successive_indices(
                    sample_length=T,
                    block_length=block_length,
                    circular=circular,
                    successive_3d=False)

    indices \
        = _general_block_bootstrap_loop(
            block_length=block_length, replications=replications,
            block_start_indices=np.arange(0, x.shape[0] + block_length),
            successive_indices=successive_indices,
            sub_sample_length=sub_sample_length,
            replace=replace, link_rngs=link_rngs)

    indices = _general_leader_bootstrap_loop(indices, block_start_indices,
                                             block_length, max_scale)

    return indices


def circular_leader_bootstrap(mrq, max_scale, block_length, replications,
                              sub_sample_length=None, link_rngs=True,
                              double=False):
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
    bootstrap_mrq: :class:`~pymultifracs.multiresquantity.MultiResQuantity`
        A single MRQ that contains all the bootstrapped repetitions

    double_mrq: dict(int,
                     :class:`~pymultifracs.multiresquantity.MultiResQuantity`)
        A dictionary that relates a repetition in the bootstrap_mrq to the
        MRQ containing the double-bootstrapped repetitions
        if `double` was passed as True
    """

    x = mrq.values[1]
    x = x[~np.isnan(x)]

    max_scale = min(max_scale_bootstrap(mrq, block_length), max_scale)

    indices = _general_leader_bootstrap(x, max_scale, block_length,
                                        replications, sub_sample_length,
                                        link_rngs)

    values = {}
    nj = {}

    if double:

        indices_double = {}

        for rep in range(replications):
            indices_double[rep] = _general_leader_bootstrap(
                indices[1][rep][indices[1][rep] >= 0], max_scale, block_length,
                replications, sub_sample_length, link_rngs)

        values_double = {rep: {} for rep in indices_double}
        nj_double = {rep: {} for rep in indices_double}

    for scale, indices_scale in indices.items():

        data = mrq.values[scale]
        data = data[~np.isnan(data)]

        if data.ndim == 1:
            data = np.hstack((data, data))
        else:
            data = np.vstack((data, data))

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
                    (replications,
                     mrq.values[1][~np.isnan(mrq.values[1])].shape[0]),
                    dtype=float) + np.nan
                out_double[idx_final >= 0] = data[idx_final[idx_final >= 0]]

                compact_idx = np.all(np.isnan(out_double), axis=0)

                values_double[rep][scale] = \
                    out_double[:, ~compact_idx].transpose()

                nj_double[rep][scale] = np.array(
                    [idx_final[rep2][idx_final[rep2] >= 0].shape[0]
                     for rep2 in range(replications)])

        nj[scale] = np.array(
            [indices_scale[rep, indices_scale[rep] >= 0].shape[0]
             for rep in range(replications)])

        out = np.zeros(
            (replications, mrq.values[1][~np.isnan(mrq.values[1])].shape[0]),
            dtype=float) + np.nan
        out[indices_scale >= 0] = data[idx]

        compact_idx = np.all(np.isnan(out), axis=0)
        values[scale] = out[:, ~compact_idx].transpose()

    new_mrq = mrq.from_dict({
        'formalism': mrq.formalism,
        'gamint': mrq.gamint,
        'nj': nj,
        'values': values,
    })

    if double:

        double_mrq = {
            rep: mrq.from_dict({
                'formalism': mrq.formalism,
                'gamint': mrq.gamint,
                'nj': nj_double[rep],
                'values': values_double[rep],
            })
            for rep in values_double}

        return new_mrq, double_mrq

    return new_mrq
