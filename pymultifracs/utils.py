"""
Authors: Omar D. Domingues <omar.darwiche-domingues@inria.fr>
         Merlin Dumeur <merlin@dumeur.net>
"""

# pylint: disable=C0415

from enum import Enum
from dataclasses import dataclass
from typing import Any
from collections import namedtuple
import warnings

import numpy as np
import xarray as xr
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


@dataclass
class DimensionNames:
    """
    Standard dimension names
    """
    scaling_range: str = 'scaling_range'
    channel: str = 'channel'
    k_j: str = 'k_j(t)'
    j: str = 'j'
    bootstrap: str = 'bootstrap'
    q: str = 'q'
    m: str = 'm'

    def __getattr__(self, name):

        match name:

            case str() as s if s[-1].isdigit() and s[0].isalpha():

                attribute = s.rstrip('0123456789')

                if (not hasattr(self, attribute)
                        or not s[len(attribute):].isdigit()):
                    raise ValueError(
                        f'No attribute {attribute} or {s[len(attribute):]} '
                        f'are not digits.')

                out = getattr(self, attribute) + s[len(attribute):]
                setattr(self, name, out)
                return out

            case _:
                return self.__getattribute__(name)

Dim = DimensionNames()

# class Dim(Enum):
#     """
#     Standard dimension names
#     """
#     scaling_range = 'scaling_range'


class Formalism(Enum):
    """
    Possible multifractal formalisms.
    """
    wavelet_coef = 1
    wavelet_leader = 2
    wavelet_pleader = 3
    weak_scaling_exponent = 4


@dataclass
class AbstractDataclass:
    """
    Abstract class containing information.
    """
    bootstrapped_obj: Any | None = None

    def _check_enough_rep_bootstrap(self):

        # bootstrap_index = self.dims.index('bootstrap')

        # if self.values[min(self.values)].shape[bootstrap_index] < 2:

        if isinstance(self.values, xr.DataArray):

            if Dim.bootstrap not in self.values.sizes:
                n_bootstrap = 0
            else:
                n_bootstrap = self.values.sizes[Dim.bootstrap]

        else:

            if Dim.bootstrap not in self.dims:
                n_bootstrap = 0
            else:
                n_bootstrap = self.values[max(self.values)].shape[
                    self.dims.index(Dim.bootstrap)]

        if n_bootstrap < 2:
            raise ValueError(
                f'n_bootstrap = {n_bootstrap} per original signal too '
                'small to build confidence intervals'
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

    def std_values(self):  # pylint: disable=C0116

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


def scale2freq(scale, sfreq, wt_name='db3'):
    """
    Returns the frequency associated to a scale.
    """
    return pywt.scale2frequency(wt_name, 2 ** scale) * sfreq


def freq2scale(freq, sfreq, wt_name='db3'):
    """
    Returns the scale associated with a frequency.
    """
    return np.log2(pywt.frequency2scale(wt_name, freq / sfreq))


def fband2scale(fband, sfreq, wt_name='db3'):
    """
    Returns the range of scales associated with a frequency band.
    """
    return (int(np.ceil(freq2scale(fband[1], sfreq, wt_name))),
            int(np.floor(freq2scale(fband[0], sfreq, wt_name))))


# def pairing(long, short):
#     return long * (long - 1) / 2 + short - 1


def fast_power(array, exponent):
    """
    Faster version of the np.pow for often used exponents (1, 2, .5, 0, -1)
    """

    # import warnings
    # warnings.filterwarnings("error")

    if exponent == 1:
        return array

    if exponent == 2:
        return array * array
        # return np.square(array)

    if exponent == 0.5:
        return np.sqrt(array)

    if exponent == 0:
        # np.nan ** 0 = 1.0, adressed here
        ixd_nan = np.isnan(array)
        res = array ** exponent
        res[ixd_nan] = np.nan
        return res

    if exponent == -1:
        return array ** exponent

    if isinstance(exponent, int) and 0 < exponent <= 10:

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
    """
    Fixes nansum.
    """
    mx = np.isnan(a).all(**kwargs)
    res = np.nansum(a, **kwargs)
    res[mx] = np.nan
    return res


def fixednanmax(a, **kwargs):
    """
    Fixes nanmax.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = np.nanmax(a, **kwargs)
    return a


def get_filter_length(wt_name):
    """
    Returns the length of the wavelet filter.
    """
    wt = pywt.Wavelet(wt_name)
    return len(wt.dec_hi)


def max_scale_bootstrap(mrq, idx_reject=None):
    """
    Determines maximum scale possible to perform bootstrapping.

    Parameters
    ----------
    mrq: :class:`~pymultifracs.multiresquantity.MultiResolutionQuantity`

    """

    filt_len = mrq.filt_len

    j = 0  # Avoids error here in the case where mrq.values is empty

    for j in mrq.values:
        val = mrq.values[j]

        mask_reject(val[:, None, :], idx_reject, j, mrq.interval_size)

        if ((~np.isnan(val)).sum(axis=0) < filt_len).any():
            j -= 1
            break

    return j


def isclose(a, b, rel_tol=1.98e-03):
    """
    Custom isclose function with chosen tolerance.
    """
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

def _expand_align(*arrays, reference_order=None):
    """
    Expand xarrays to the covering list of dimensions, and then align their order
    to enable broadcasting.
    """

    out = []

    if reference_order is not None:
        dims = [*reference_order]
    else:
        dims = []

    for arr in arrays:
        for dim in arr.dims:
            if dim not in dims:
                dims.append(dim)

    dims = tuple(dims)

    for a in arrays:
        temp = a.expand_dims(
            [d for d in dims if d not in a.dims],
            create_index_for_new_dim=False)
        temp = temp.transpose(*dims, ...)
        out.append(temp)

    if len(out) == 1:
        return out[0]

    return out

def mask_reject(values, idx_reject, j, interval_size):
    """
    Remove values from an array based on a mask.
    """

    if idx_reject is None or j not in idx_reject:
        return values

    mask = np.ones_like(idx_reject[j], dtype=float)
    mask[idx_reject[j].values] = np.nan

    mask = xr.DataArray(mask, dims=idx_reject[j].dims)

    out = values * mask

    mask_fewcoeff = (~np.isnan(out)).sum(dim=Dim.k_j) < 3

    out = out.where(~mask_fewcoeff, np.nan)

    # delta = (interval_size - 1) // 2

    # if delta > 0:
    #     return values * mask[delta:-delta]

    return out


def get_edge_reject(WT):

    idx_reject = {
        j: np.isnan(WT.values[j], dtype=bool)[:, None] for j in WT.values}

    for j in np.sort(np.arange(7, 12))[::-1]:

        if j not in idx_reject:
            continue

        if j-1 in idx_reject:
            idx_reject[j-1][:idx_reject[j].shape[0] * 2] |= np.repeat(idx_reject[j], 2, axis=0)

    return idx_reject
