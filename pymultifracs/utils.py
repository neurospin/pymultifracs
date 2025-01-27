"""
Authors: Omar D. Domingues <omar.darwiche-domingues@inria.fr>
         Merlin Dumeur <merlin@dumeur.net>
"""

from enum import Enum
from dataclasses import dataclass
from typing import Any
import inspect
from collections import namedtuple
import warnings

import numpy as np
import pywt


MFractalVar = namedtuple('MFractalVar', 'structure cumulants spectrum')
"""Aggregates the output of multifractal analysis

Attributes
----------
strucuture : :class:`.StructureFunction`
cumulants : :class:`.Cumulants`
spectrum : :class:`.MFSpectrum`
"""

MFractalBiVar = namedtuple('MFractalBiVar', 'structure cumulants')
"""Aggregates the output of bivariate multifractal analysis

Attributes
----------
strucuture : :class:`.BiStructureFunction`
cumulants : :class:`.BiCumulants`
"""


class Formalism(Enum):
    wavelet_coef = 1
    wavelet_leader = 2
    wavelet_pleader = 3
    weak_scaling_exponent = 4


@dataclass
class AbstractDataclass:
    bootstrapped_obj: Any | None = None

    def _check_enough_rep_bootstrap(self):

        if (ratio := self.n_rep // self.n_sig) < 2:
            raise ValueError(
                f'n_rep = {ratio} per original signal too small to build '
                'confidence intervals'
                )
        
    def _get_bootstrapped_obj(self):

        if self.bootstrapped_obj is None:
            bootstrapped_obj = self
        else:
            bootstrapped_obj = self.bootstrapped_obj

        bootstrapped_obj._check_enough_rep_bootstrap()

        return bootstrapped_obj
    
    def _check_bootstrap_obj(self):

        if self.bootstrapped_obj is None:
            raise ValueError(
                "Bootstrapped mrq needs to be computed prior to estimating "
                "empirical estimators")

        self.bootstrapped_obj._check_enough_rep_bootstrap()

    def std_values(self):

        from .bootstrap import get_std

        self._check_enough_rep_bootstrap()

        return get_std(self, 'values')

    def __getattr__(self, name):

        if name[:3] == 'CI_':
            from .bootstrap import get_confidence_interval

            bootstrapped_obj = self._get_bootstrapped_obj()

            return get_confidence_interval(bootstrapped_obj, name[3:])

        elif name[:4] == 'CIE_':
            from .bootstrap import get_empirical_CI

            self._check_bootstrap_obj()

            return get_empirical_CI(self.bootstrapped_obj, self, name[4:])

        elif name[:3] == 'VE_':
            from .bootstrap import get_empirical_variance

            self._check_bootstrap_obj()

            return get_empirical_variance(self.bootstrapped_obj, self,
                                          name[3:])

        elif name[:3] == 'SE_':

            from .bootstrap import get_empirical_variance

            self._check_bootstrap_obj()

            return np.sqrt(
                get_empirical_variance(self.bootstrapped_obj, self,
                                       name[3:]))

        elif name[:2] == 'V_':

            from .bootstrap import get_variance

            bootstrapped_obj = self._get_bootstrapped_obj()

            return get_variance(bootstrapped_obj, name[2:])

        elif name[:4] == 'STD_':

            from .bootstrap import get_std

            bootstrapped_obj = self._get_bootstrapped_obj()

            return get_std(bootstrapped_obj, name[4:])

        return self.__getattribute__(name)


def scale2freq(scale, sfreq):
    return pywt.scale2frequency('db3', 2 ** scale) * sfreq

def freq2scale(freq, sfreq):
    return np.log2(pywt.frequency2scale('db3', freq / sfreq))

def fband2scale(fband, sfreq):
    return (int(np.ceil(freq2scale(fband[1], sfreq))),
            int(np.floor(freq2scale(fband[0], sfreq))))


def pairing(long, short):
    return long * (long - 1) / 2 + short - 1


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
    q : ndarray of float
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

    # delta = (interval_size - 1) // 2

    # if delta > 0:
    #     return values * mask[delta:-delta]
    
    return values * mask
