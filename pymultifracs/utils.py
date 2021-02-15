"""
Authors: Omar D. Domingues <omar.darwiche-domingues@inria.fr>
         Merlin Dumeur <merlin@dumeur.net>
"""

from __future__ import print_function
from __future__ import unicode_literals
import warnings

import numpy as np


def scale2freq(scale, sfreq):
    return (3/4) * sfreq * (2 ** -scale)


def freq2scale(freq, sfreq):
    return - 2 - np.log2(freq / (3 * sfreq))


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


stat2fun = {
    'mean': np.mean,
    'median': np.median,
    'min': np.nanmin,
    'max': np.nanmax}


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
    assert len(nj) == len(x)

    V_0 = np.sum(nj, axis=0)
    V_1 = np.sum(nj * x, axis=0)
    V_2 = np.sum(nj * (x**2), axis=0)

    weights_slope = nj * (V_0*x - V_1)/(V_0*V_2 - V_1*V_1)
    weights_intercept = nj * (V_2 - V_1*x)/(V_0*V_2 - V_1*V_1)

    a = np.sum(weights_slope*y, axis=0)
    b = np.sum(weights_intercept*y, axis=0)

    var_a = np.sum((1/nj)*weights_slope*weights_slope, axis=0)

    if not return_variance:
        return a, b
    else:
        return a, b, var_a


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
