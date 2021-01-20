"""
Authors: Omar D. Domingues <omar.darwiche-domingues@inria.fr>
         Merlin Dumeur <merlin@dumeur.net>
"""

from __future__ import print_function
from __future__ import unicode_literals

import numpy as np


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

    elif exponent in [-1, 0]:
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


# class Utils:
#     def __init__(self):
#         pass

#     # TODO:Replace with sklearn import ?

#     @staticmethod

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

    bj = np.array(nj, dtype=np.float)
    assert len(bj) == len(x)

    V_0 = np.sum(bj)
    V_1 = np.sum(bj * x)
    V_2 = np.sum(bj * (x**2))

    weights_slope = bj * (V_0*x - V_1)/(V_0*V_2 - V_1*V_1)
    weights_intercept = bj * (V_2 - V_1*x)/(V_0*V_2 - V_1*V_1)

    a = np.sum(weights_slope*y)
    b = np.sum(weights_intercept*y)

    var_a = np.sum((1/bj)*weights_slope*weights_slope)

    if not return_variance:
        return a, b
    else:
        return a, b, var_a


def build_q_log(q_min, q_max, n):
    """
    Builds a convenient vector of q values for multifractal analysis

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
