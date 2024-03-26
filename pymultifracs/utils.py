"""
Authors: Omar D. Domingues <omar.darwiche-domingues@inria.fr>
         Merlin Dumeur <merlin@dumeur.net>
"""

from collections import namedtuple
import warnings

import numpy as np
import pywt


MFractalData = namedtuple('MFractalData', 'dwt lwt')
"""Aggregates wavelet coef-based and wavelet-leader based outputs of mfa

Attributes
----------
dwt : MFractalVar
    Wavelet coef-based estimates
lwt : MFractalVar
    Wavelet leader-based estimates, if applicable (p_exp was not None)
"""

MFractalVar = namedtuple('MFractalVar',
                         'structure cumulants spectrum')
"""Aggregates the output of multifractal analysis

Attributes
----------
strucuture : :class:`~pymultifracs.structurefunction.StructureFunction`
cumulants : :class:`~pymultifracs.cumulants.Cumulants`
spectrum : :class:`~pymultifracs.mfspectrum.MultifractalSpectrum`
hmin : float
    Estimated minimum value of h
"""

MFractalBiVar = namedtuple('MFractalBiVar', 'structure cumulants')


def scale2freq(scale, sfreq):
    return pywt.scale2frequency('db3', 2 ** scale) * sfreq

def freq2scale(freq, sfreq):
    return np.log2(pywt.frequency2scale('db3', freq / sfreq))

def fband2scale(fband, sfreq):
    return (int(np.ceil(freq2scale(fband[1], sfreq))),
            int(np.floor(freq2scale(fband[0], sfreq))))


def fast_power(array, exponent):

    # import warnings
    # warnings.filterwarnings("error")

    if exponent == 1:
        return array

    elif exponent == 2:
        return array * array
        # return np.square(array)

    elif exponent == 0.5:
        return np.sqrt(array)

    elif exponent == 0:
        # np.nan ** 0 = 1.0, adressed here
        ixd_nan = np.isnan(array)
        res = array ** exponent
        res[ixd_nan] = np.nan
        return res

    elif exponent == -1:
        return array ** exponent

    elif isinstance(exponent, int) and exponent > 0 and exponent <= 10:

        array_out = np.ones(array.shape)

        for _ in range(exponent):
            array_out *= array

        return array_out

    return np.power(array, exponent)


def build_q_log(q_min, q_max, n):
    """
    Build a convenient vector of q values for multifractal analysis

    Parameters
    ----------
    q_min : float
        Lower bound of q, needs to be strictly positive
    q_max : float
        Upper value of q, needs to be strictly positive
    n : int
        Number of logspaced values to include

    Returns
    -------
    q : ndarray
        log-spaced values between `q_min` and `q_max`, along with their
        opposites, and accompanied by -2, -1, 0, 1, 2.
    """

    if q_min <= 0 or q_max <= 0:
        raise ValueError('q_min and q_max must be larger than 0')

    q = np.logspace(np.log10(q_min), np.log10(q_max), n)
    q = np.array([*q, 0, 1, 2])
    q = np.unique(np.sort([*q, *(-q)]))

    return q


stat2fun = {
    'mean': np.mean,
    'median': np.median,
    'min': np.nanmin,
    'max': np.nanmax}


def fixednansum(a, **kwargs):
    mx = np.isnan(a).all(**kwargs)
    res = np.nansum(a, **kwargs)
    res[mx] = np.nan
    return res


def fixednanmax(a, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = np.nanmax(a, **kwargs)
    return a


def get_filter_length(wt_name):

    wt = pywt.Wavelet(wt_name)
    return len(wt.dec_hi)


def max_scale_bootstrap(mrq, idx_reject=None):
    """
    Determines maximum scale possible to perform bootstrapping

    Parameters
    ----------
    mrq: :class:`~pymultifracs.multiresquantity.MultiResolutionQuantity`

    """

    filt_len = mrq.filt_len

    for j in mrq.values:
        val = mrq.values[j]

        mask_reject(val[:, None, :], idx_reject, j, mrq.interval_size)

        if ((~np.isnan(val)).sum(axis=0) < filt_len).any():
            j -= 1
            break

    return j


def isclose(a, b, rel_tol=1.98e-03):
    return np.abs(a - b) <= rel_tol * max(np.max(np.abs(a)), np.max(np.abs(b)))


def scale_position(time, scale_min, scale_max, wt_leaders=None):
    """
    Returns indexes for wt coefs and optionally leaders to be set to nan \
    for each scale between scale_min and scale_max
    """

    out_idx = {}
    out_leader = {}

    for scale in range(scale_min, scale_max + 1):

        idx = np.unique(time // 2 ** scale)
        out_idx[scale] = idx

        if wt_leaders is not None:

            n_leaders = wt_leaders.values[scale].shape[0]

            out_leader[scale] = np.unique(np.concatenate([
                idx[(idx - 1 >= 0) & (idx - 1 < n_leaders)] - 1,
                idx[idx < n_leaders],
                idx[(idx - 2 >= 0) & (idx - 2 < n_leaders)] - 2]))

    return out_idx, out_leader


def _correct_pleaders(wt_leaders, p_exp, min_level, max_level):
    """
    Return p-leader correction factor for finite resolution
    """

    JJ = np.arange(min_level, max_level + 1)
    J1LF = 1
    JJ0 = JJ - J1LF + 1

    # eta_p shape (n_ranges, n_rep)
    # JJ0 shape (n_level,)

    JJ0 = JJ0[None, None, :]
    eta_p = wt_leaders.eta_p[:, :, None]

    zqhqcorr = np.log2((1 - np.power(2., -JJ0 * eta_p))
                       / (1 - np.power(2., -eta_p)))
    ZPJCorr = np.power(2, (-1.0 / p_exp) * zqhqcorr)

    # import ipdb; ipdb.set_trace()

    # ZPJCorr shape (n_ranges, n_rep, n_level)
    # wt_leaders shape (n_coef_j, n_rep)
    # for ind_j, j in enumerate(JJ):
    #     wt_leaders.values[j] = \
    #         wt_leaders.values[j][:, None, :]*ZPJCorr[None, :, :, ind_j]

    eta_negative = eta_p <= 0
    ZPJCorr[eta_negative[..., 0], :] = 1

    # ZPJCorr shape (n_ranges, n_rep, n_level)
    return ZPJCorr


def mask_reject(values, idx_reject, j, interval_size):

    if idx_reject is None or j not in idx_reject:
        return values

    mask = np.ones_like(idx_reject[j], dtype=float)

    mask[idx_reject[j]] = np.nan

    delta = (interval_size - 1) // 2

    if delta > 0:
        return values * mask[delta:-delta]
    
    return values * mask
