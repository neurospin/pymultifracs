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
                         'structure cumulants spectrum hmin')
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
    return (3/4) * sfreq * (2 ** -scale)


def freq2scale(freq, sfreq):
    return - 2 - np.log2(freq / (3 * sfreq))


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


def max_scale_bootstrap(mrq):
    """
    Determines maximum scale possible to perform bootstrapping

    Parameters
    ----------
    mrq: :class:`~pymultifracs.multiresquantity.MultiResolutionQuantity`

    """

    filt_len = mrq.filt_len

    for i, nj in mrq.nj.items():
        if (nj < filt_len).any():
            i -= 1
            break

    return i


def isclose(a, b, rel_tol=1.98e-03):
    return np.abs(a - b) <= rel_tol * max(np.max(np.abs(a)), np.max(np.abs(b)))
